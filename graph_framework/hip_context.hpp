//------------------------------------------------------------------------------
///  @file hip_context.hpp
///  @brief HIP context for amd based gpus.
///
///  Defines context for amd gpu.
//------------------------------------------------------------------------------

#ifndef hip_context_h
#define hip_context_h

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include "node.hpp"

namespace gpu {
//------------------------------------------------------------------------------
///  @brief Check results of realtime compile.
///
///  @param[in] result Result code of the operation.
///  @param[in] name   Name of the operation.
//------------------------------------------------------------------------------
    static void check_hiprtc_error(const hiprtcResult result,
                                   const std::string &name) {
#ifndef NDEBUG
        assert(result == HIPRTC_SUCCESS && hiprtcGetErrorString(result));
#endif
    }

//------------------------------------------------------------------------------
///  @brief Check results of hip functions.
///
///  @param[in] result Result code of the operation.
///  @param[in] name   Name of the operation.
//------------------------------------------------------------------------------
    static void check_error(const hipError_t result,
                            const std::string &name) {
#ifndef NDEBUG
        assert(result == hipSuccess && hipGetErrorString(result));
#endif
    }

//------------------------------------------------------------------------------
///  @brief Initalize hip.
//------------------------------------------------------------------------------
     static hipError_t hip_init() {
         const hipError_t result = hipInit(0);
         check_error(result, "hipInit");
         return result;
     }
///  Initalize Hip.
     static const hipError_t result = hip_init();

//------------------------------------------------------------------------------
///  @brief Class representing a hip gpu context.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    class hip_context {
    private:
///  Device id
        int device;
///  Hip stream.
        hipStream_t stream;
///  Hip module.
        hipModule_t module;
///  Result Buffer.
        hipDeviceptr_t result_buffer;
///  Result Buffer.
        hipDeviceptr_t offset_buffer;
///  Argument map.
        std::map<graph::leaf_node<T, SAFE_MATH> *, hipDeviceptr_t> kernel_arguments;

//------------------------------------------------------------------------------
///  @brief  Check results of async hip functions.
///
///  @param[in] result Result code of the operation.
///  @param[in] name   Name of the operation.
//------------------------------------------------------------------------------
        void check_error_async(const hipError_t result,
                               const std::string &name) {
            check_error(result, name);
#ifndef NDEBUG
            std::string async_name = name + "_async";
            check_error(hipStreamSynchronize(stream), async_name);
#endif
        }

    public:
//------------------------------------------------------------------------------
///  @brief Get the maximum number of concurrent instances.
///
///  @returns The maximum available concurrency.
//------------------------------------------------------------------------------
        static size_t max_concurrency() {
            int count;
            check_error(hipGetDeviceCount(&count), "hipGetDeviceCount");
            return count;
        }

//------------------------------------------------------------------------------
///  @brief Device discription.
//------------------------------------------------------------------------------
        static std::string device_type() {
            return "Hip GPU";
        }

//------------------------------------------------------------------------------
///  @brief Hip context constructor.
///
///  @param[in] index Concurrent index.
//------------------------------------------------------------------------------
        hip_context(const size_t index) :
        device(index), result_buffer(0), module(0), offset_buffer(0) {
            check_error(hipSetDevice(device), "hipSetDevice");
            check_error(hipStreamCreate(&stream), "hipStreamCreate");
        }

//------------------------------------------------------------------------------
///  @brief Hip context destructor.
//------------------------------------------------------------------------------
        ~hip_context() {
            if (module) {
                check_error(hipModuleUnload(module), "hipModuleUnload");
                module = 0;
            }

            for (auto &[key, value] : kernel_arguments) {
                check_error(hipFree(value), "hipFree");
            }

            if (result_buffer) {
               check_error(hipFree(result_buffer), "hipFree");
               result_buffer = 0;
            }
            if (offset_buffer) {
                check_error(hipFree(offset_buffer), "hipFree");
                offset_buffer = 0;
            }

            check_error(hipStreamDestroy(stream), "hipStreamDestroy");
        }

//------------------------------------------------------------------------------
///  @brief Compile the kernels.
///
///  @param[in] kernel_source Source code buffer for the kernel.
///  @param[in] names         Names of the kernel functions.
///  @param[in] add_reduction Include the reduction kernel.
//------------------------------------------------------------------------------
        void compile(const std::string kernel_source,
                     std::vector<std::string> names,
                     const bool add_reduction=false) {
            if (add_reduction) {
                names.push_back("max_reduction");
            }

            hiprtcProgram kernel_program;
            check_hiprtc_error(hiprtcCreateProgram(&kernel_program,
                                                   kernel_source.c_str(),
                                                   NULL, 0, NULL, NULL),
                               "hiprtcCreateProgram");

            for (std::string &name : names) {
                check_hiprtc_error(hiprtcAddNameExpression(kernel_program,
                                                           name.c_str()),
                                   "hiprtcAddNameExpression");
            }

            hipDeviceProp_t device_properties;
            check_error(hipGetDeviceProperties(&device_properties, device),
                                               "hipGetDeviceProperties");

            if (jit::verbose) {
                std::cout << "HIP GPU info." << std::endl;
                std::cout << "  Major compute capability        : " << device_properties.major << std::endl;
                std::cout << "  Minor compute capability        : " << device_properties.minor << std::endl;
                std::cout << "  Device name                     : " << device_properties.name << std::endl;
                std::cout << "  Total Global Memory             : " << device_properties.totalGlobalMem << std::endl;
                std::cout << "  Managed Memory                  : " << device_properties.managedMemory << std::endl;
                std::cout << "  Max Threads Per Block           : " << device_properties.maxThreadsPerBlock << std::endl;
                std::cout << "  Max Threads Per Multi Processor : " << device_properties.maxThreadsPerMultiProcessor << std::endl;
                std::cout << "  Warp size                       : " << device_properties.warpSize << std::endl;
            }

            std::array<const char *, 2> options({
                "-std=c++17",
                "-I" HEADER_DIR
            });

            if (hiprtcCompileProgram(kernel_program, options.size(), options.data())) {
                size_t log_size;
                check_hiprtc_error(hiprtcGetProgramLogSize(kernel_program, &log_size),
                                   "hiprtcGetProgramLogSize");

                char *log = static_cast<char *> (malloc(log_size));
                check_hiprtc_error(hiprtcGetProgramLog(kernel_program, log),
                                   "hiprtcGetProgramLog");
                std::cout << log << std::endl;
                free(log);
                std::cout << kernel_source << std::endl;
            }

            size_t code_size;
            check_hiprtc_error(hiprtcGetCodeSize(kernel_program, &code_size),
                               "hiprtcGetCodeSize");

            char *code = static_cast<char *> (malloc(code_size));
            check_hiprtc_error(hiprtcGetCode(kernel_program, code),
                               "hiprtcGetCode");

            check_error(hipModuleLoadData(&module, code),
                        "hipModuleLoadData");

            free(code);
        }

//------------------------------------------------------------------------------
///  @brief Create a kernel calling function.
///
///  @param[in] kernel_name Name of the kernel for later reference.
///  @param[in] inputs      Input nodes of the kernel.
///  @param[in] outputs     Output nodes of the kernel.
///  @param[in] state       Random states.
///  @param[in] num_rays    Number of rays to trace.
///  @param[in] tex1d_list  List of 1D textures.
///  @param[in] tex2d_list  List of 1D textures.
///  @returns A lambda function to run the kernel.
//------------------------------------------------------------------------------
        std::function<void(void)> create_kernel_call(const std::string kernel_name,
                                                     graph::input_nodes<T, SAFE_MATH> inputs,
                                                     graph::output_nodes<T, SAFE_MATH> outputs,
                                                     graph::shared_random_state<T, SAFE_MATH> state,
                                                     const size_t num_rays,
                                                     const jit::texture1d_list &tex1d_list,
                                                     const jit::texture2d_list &tex2d_list) {
            hipFunction_t function;
            check_error(hipModuleGetFunction(&function, module, 
                                             kernel_name.c_str()), 
                        "hipModuleGetFunction");

            std::vector<void *> buffers;
            std::set<graph::leaf_node<T, SAFE_MATH> *> needed_buffers;
            
            for (auto &input : inputs) {
                if (!kernel_arguments.contains(input.get())) {
                    kernel_arguments.try_emplace(input.get());
                    backend::buffer<T> backend = input->evaluate();
                    check_error(hipMallocManaged(&kernel_arguments[input.get()],
                                                 backend.size()*sizeof(T), 
                                                 hipMemAttachGlobal),
                                "hipMallocManaged");
                    check_error(hipMemcpyHtoD(kernel_arguments[input.get()],
                                              &backend[0],
                                              backend.size()*sizeof(T)),
                                "hipMemcpyHtoD");
                    buffers.push_back(reinterpret_cast<void *> (&kernel_arguments[input.get()]));
                    needed_buffers.insert(input.get());
                }
                if (!needed_buffers.contains(input.get())) {
                    buffers.push_back(reinterpret_cast<void *> (&kernel_arguments[input.get()]));
                    needed_buffers.insert(input.get());
                }
            }
            for (auto &output : outputs) {
                if (!kernel_arguments.contains(output.get())) {
       	       	    kernel_arguments.try_emplace(output.get());
       	       	    check_error(hipMallocManaged(&kernel_arguments[output.get()],
                                                 num_rays*sizeof(T), 
                                                 hipMemAttachGlobal), 
                                "hipMallocManaged");
                    buffers.push_back(reinterpret_cast<void *> (&kernel_arguments[output.get()]));
                    needed_buffers.insert(output.get());
                }
                if (!needed_buffers.contains(output.get())) {
                    buffers.push_back(reinterpret_cast<void *> (&kernel_arguments[output.get()]));
                    needed_buffers.insert(output.get());
                }
            }

            const size_t num_buffers = buffers.size();
            if (state.get()) {
                if (!kernel_arguments.contains(state.get())) {
                    kernel_arguments.try_emplace(state.get());
                    check_error(hipMallocManaged(&kernel_arguments[state.get()],
                                                 state->get_size_bytes(),
                                                 hipMemAttachGlobal),
                                "hipMallocManaged");
                    check_error(hipMallocManaged(&offset_buffer, sizeof(uint32_t)), "hipMallocManaged");
                    check_error(hipMemcpyHtoD(kernel_arguments[state.get()],
                                              state->data(),
                                              state->get_size_bytes()),
                                "hipMemcpyHtoD");
                }
                buffers.push_back(reinterpret_cast<void *> (&kernel_arguments[state.get()]));
                buffers.push_back(reinterpret_cast<void *> (&offset_buffer));
            }

            int gridSize;
            int blockSize;
            check_error(hipModuleOccupancyMaxPotentialBlockSize(&gridSize, &blockSize,
                                                                function, 0, 0), 
                        "hipModuleOccupancyMaxPotentialBlockSize");
            const unsigned int numBlocks = num_rays/blockSize + (num_rays%blockSize ? 1 : 0);
            const unsigned int dimBlock = blockSize;

            if (jit::verbose) {
                std::cout << "  Kernel name                     : " << kernel_name << std::endl;
                std::cout << "    Block Size       : " << blockSize << std::endl;
                std::cout << "    Min Grid Size    : " << gridSize << std::endl;
                std::cout << "    Number of Blocks : " << numBlocks << std::endl;
            }

            if (state.get()) {
                return [this, num_rays, function, buffers, dimBlock] () mutable {
                    for (unit32_t i = 0; i < num_rays; i += dimBlock) {
                        for (uint32_t i = 0; i < num_rays; i += dimBlock) {
                            check_error_async(hipMemsetAsync(offset_buffer, i,
                                                             sizeof(uint32_t)),
                                              hipMemcpyDeviceToHost, stream);
                            check_error_async(hipModuleLaunchKernel(function,
                                                                    1, 1, 1,
                                                                    dimBlock, 1, 1,
                                                                    0, stream,
                                                                    buffers.data(), NULL),
                                              "hipModuleLaunchKernel");
                        }
                    }
                };
            } else {
                return [this, function, buffers, numBlocks, dimBlock] () mutable {
                    check_error_async(hipModuleLaunchKernel(function,
                                                            numBlocks, 1, 1,
                                                            dimBlock, 1, 1,
                                                            0, stream,
                                                            buffers.data(), NULL),
                                      "hipModuleLaunchKernel");
                };
            }
        }

//------------------------------------------------------------------------------
///  @brief Create a max compute kernel calling function.
///
///  @param[in] argument Node to reduce.
///  @param[in] run      Function to run before reduction.
///  @returns A lambda function to run the kernel.
//------------------------------------------------------------------------------
        std::function<T(void)> create_max_call(graph::shared_leaf<T, SAFE_MATH> &argument,
                                               std::function<void(void)> run) {
            check_error(hipMallocManaged(&result_buffer, sizeof(T),
                                         hipMemAttachGlobal),
                        "hipMallocManaged");

            std::vector<void *> buffers;

            buffers.push_back(reinterpret_cast<void *> (&kernel_arguments[argument.get()]));
            buffers.push_back(reinterpret_cast<void *> (&result_buffer));

            hipFunction_t function;
            check_error(hipModuleGetFunction(&function, module, "max_reduction"),
                        "hipModuleGetFunction");

            int gridSize;
            int blockSize;
            check_error(hipModuleOccupancyMaxPotentialBlockSize(&gridSize, &blockSize,
                                                                function, 0, 0),
                        "hipModuleOccupancyMaxPotentialBlockSize");

            const unsigned int numBlocks = 1024/blockSize + (1024%blockSize ? 1 : 0);
            const unsigned int dimBlock = blockSize;

            if (jit::verbose) {
                std::cout << "  Kernel name                     : max_reduction" << std::endl;
                std::cout << "    Block Size	   : " << blockSize << std::endl;
                std::cout << "    Min Grid Size    : " << gridSize << std::endl;
                std::cout << "    Number of Blocks : " << numBlocks << std::endl;
            }

            return [this, function, run, buffers, numBlocks, dimBlock] () mutable {
                run();
                check_error_async(hipModuleLaunchKernel(function,
                                                        numBlocks, 1, 1, 
                                                        dimBlock, 1, 1,
                                                        0, stream,
                                                        buffers.data(), NULL),
                                  "hipLaunchKernel");
                wait();

                return reinterpret_cast<T *> (result_buffer)[0];
            };
        }

//------------------------------------------------------------------------------
///  @brief Hold the current thread until the stream has completed.
//------------------------------------------------------------------------------
        void wait() {
            check_error_async(hipStreamSynchronize(stream), 
                              "hipStreamSynchronize");
        }

//------------------------------------------------------------------------------
///  @brief Print out the results.
///
///  @param[in] index Number of times to record.
///  @param[in] nodes Nodes to output.
//------------------------------------------------------------------------------
        void print_results(const size_t index,
                           const graph::output_nodes<T, SAFE_MATH> &nodes) {
            wait();
            for (auto &out : nodes) {
                const T temp = reinterpret_cast<T *> (kernel_arguments[out.get()])[index];
                if constexpr (jit::is_complex<T> ()) {
                    std::cout << std::real(temp) << " " << std::imag(temp) << " ";
                } else {
                    std::cout << temp << " ";
                }
            }
            std::cout << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Check the value.
///
///  @param[in] index Ray index to check value for.
///  @param[in] node  Node to check the value for.
///  @returns The value at the index.
//------------------------------------------------------------------------------
        T check_value(const size_t index,
                      const graph::shared_leaf<T, SAFE_MATH> &node) {
            wait();
            return reinterpret_cast<T *> (kernel_arguments[node.get()])[index];
        }

//------------------------------------------------------------------------------
///  @brief Copy buffer contents to the device.
///
///  @param[in] node   Not to copy buffer to.
///  @param[in] source Host side buffer to copy from.
//------------------------------------------------------------------------------
        void copy_to_device(graph::shared_leaf<T, SAFE_MATH> node,
                            T *source) {
            size_t size;
            check_error(hipMemPtrGetInfo(kernel_arguments[node.get()], &size), 
                                         "hipMemPtrGetInfo");
            check_error_async(hipMemcpyHtoDAsync(kernel_arguments[node.get()], 
                                                 source, size, stream),
                              "hipMemcpyHtoDAsync");
        }

//------------------------------------------------------------------------------
///  @brief Copy buffer contents to host.
///
///  @param[in]     node        Node to copy buffer from.
///  @param[in,out] destination Host side buffer to copy to.
//------------------------------------------------------------------------------
        void copy_to_host(graph::shared_leaf<T, SAFE_MATH> node,
                          T *destination) {
            size_t size;
            check_error(hipMemPtrGetInfo(kernel_arguments[node.get()], &size), 
                        "hipMemPtrGetInfo");
            check_error_async(hipMemcpyDtoHAsync(destination, 
                                                 kernel_arguments[node.get()], 
                                                 size, stream), 
                              "hipMemcpyDtoHAsync");
        }

//------------------------------------------------------------------------------
///  @brief Create the source header.
///
///  @param[in,out] source_buffer Source buffer stream.
//------------------------------------------------------------------------------
        void create_header(std::ostringstream &source_buffer) {
            if constexpr (jit::is_complex<T> ()) {
                source_buffer << "#define HIP_DEVICE_CODE" << std::endl;
                source_buffer << "#define M_PI " << M_PI << std::endl;
                source_buffer << "#include <hip/hip_complex.h>" << std::endl;
                source_buffer << "#include <special_functions.hpp>" << std::endl;
            }
        }

//------------------------------------------------------------------------------
///  @brief Create kernel prefix.
///
///  @param[in,out] source_buffer Source buffer stream.
///  @param[in]     name          Name to call the kernel.
///  @param[in]     inputs        Input variables of the kernel.
///  @param[in]     outputs       Output nodes of the graph to compute.
///  @param[in]     size          Size of the input buffer.
///  @param[in,out] registers     Map of used registers.
//------------------------------------------------------------------------------
        void create_kernel_prefix(std::ostringstream &source_buffer,
                                  const std::string name,
                                  graph::input_nodes<T, SAFE_MATH> &inputs,
                                  graph::output_nodes<T, SAFE_MATH> &outputs,
                                  graph::shared_random_state<T, SAFE_MATH> state,
                                  const size_t size,
                                  const std::vector<bool> &is_constant,
                                  jit::register_map &registers,
                                  jit::texture1d_list &textures1d,
                                  jit::texture2d_list &textures2d) {
            source_buffer << std::endl;
            source_buffer << "extern \"C\" __global__ void " << name << "("
                          << std::endl;

            std::unordered_set<void *> used_args;
            if (inputs.size()) {
                source_buffer << "    ";
                if (is_constant[0]) {
                    source_buffer << "const ";
                }
                jit::add_type<T> (source_buffer);
                source_buffer << " *" << jit::to_string('v', inputs[0].get());
                used_args.insert(inputs[0].get());
            }
            for (size_t i = 1, ie = inputs.size(); i < ie; i++) {
                if (!used_args.contains(inputs[i].get())) {
                    source_buffer << "," << std::endl;
                    source_buffer << "    ";
                    if (is_constant[i]) {
                        source_buffer << "const ";
                    }
                    jit::add_type<T> (source_buffer);
                    source_buffer << " *" << jit::to_string('v', inputs[i].get());
                    used_args.insert(inputs[i].get());
                }
            }
            for (size_t i = 0, ie = outputs.size(); i < ie; i++) {
                if (!used_args.contains(outputs[i].get())) {
                    source_buffer << "," << std::endl;
                    source_buffer << "    ";
                    jit::add_type<T> (source_buffer);
                    source_buffer << " *" << jit::to_string('o', outputs[i].get());
                    used_args.insert(outputs[i].get());
                }
            }
            if (state.get()) {
                source_buffer << "," << std::endl
                              << "    mt_state * __restrict__ "
                              << jit::to_string('s', state.get())
                              << "," << std::endl
                              << "    const uint32_t *offset"
                              << std::endl;
            }
            source_buffer << ") {" << std::endl;

            source_buffer << "    const int index = blockIdx.x*blockDim.x + threadIdx.x;"
                          << std::endl;
            source_buffer << "    if (index < " << size << ") {" << std::endl;

            for (auto &input : inputs) {
                registers[input.get()] = jit::to_string('r', input.get());
                source_buffer << "        const ";
                jit::add_type<T> (source_buffer);
                source_buffer << " " << registers[input.get()] << " = "
                              << jit::to_string('v', input.get()) << "[index];"
                              << std::endl;
            }
        }

//------------------------------------------------------------------------------
///  @brief Create kernel postfix.
///
///  @param[in,out] source_buffer Source buffer stream.
///  @param[in]     outputs       Output nodes of the graph to compute.
///  @param[in]     setters       Map outputs back to input values.
///  @param[in,out] registers     Map of used registers.
//------------------------------------------------------------------------------
        void create_kernel_postfix(std::ostringstream &source_buffer,
                                   graph::output_nodes<T, SAFE_MATH> &outputs,
                                   graph::map_nodes<T, SAFE_MATH> &setters,
                                   jit::register_map &registers) {
            for (auto &[out, in] : setters) {
                graph::shared_leaf<T, SAFE_MATH> a = out->compile(source_buffer, registers);
                source_buffer << "        " << jit::to_string('v',  in.get())
                              << "[index] = ";
                if constexpr (SAFE_MATH) {
                    if constexpr (jit::is_complex<T> ()) {
                        jit::add_type<T> (source_buffer);
                        source_buffer << " (";
                        source_buffer << "isnan(real(" << registers[a.get()]
                                      << ")) ? 0.0 : real(" << registers[a.get()]
                                      << "), ";
                        source_buffer << "isnan(imag(" << registers[a.get()]
                                      << ")) ? 0.0 : imag(" << registers[a.get()]
                                      << "));" << std::endl;
                    } else {
                        source_buffer << "isnan(" << registers[a.get()]
                                      << ") ? 0.0 : " << registers[a.get()]
                                      << ";" << std::endl;
                    }
                } else {
                    source_buffer << registers[a.get()] << ";" << std::endl;
                }
            }

            for (auto &out : outputs) {
                graph::shared_leaf<T, SAFE_MATH> a = out->compile(source_buffer, registers);
                source_buffer << "        " << jit::to_string('o',  out.get())
                              << "[index] = ";
                if constexpr (SAFE_MATH) {
                    if constexpr (jit::is_complex<T> ()) {
                        jit::add_type<T> (source_buffer);
                        source_buffer << " (";
                        source_buffer << "isnan(real(" << registers[a.get()]
                                      << ")) ? 0.0 : real(" << registers[a.get()]
                                      << "), ";
                        source_buffer << "isnan(imag(" << registers[a.get()]
                                      << ")) ? 0.0 : imag(" << registers[a.get()]
                                      << "));" << std::endl;
                    } else {
                        source_buffer << "isnan(" << registers[a.get()]
                                      << ") ? 0.0 : " << registers[a.get()]
                                      << ";" << std::endl;
                    }
                } else {
                    source_buffer << registers[a.get()] << ";" << std::endl;
                }
            }

            source_buffer << "    }" << std::endl << "}" << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Create reduction.
///
///  @param[in,out] source_buffer Source buffer stream.
///  @param[in]     size          Size of the input buffer.
//------------------------------------------------------------------------------
        void create_reduction(std::ostringstream &source_buffer,
                              const size_t size) {
            source_buffer << std::endl;
            source_buffer << "extern \"C\" __global__ void max_reduction(" << std::endl;
            source_buffer << "    ";
            jit::add_type<T> (source_buffer);
            source_buffer << " *input," << std::endl;
            source_buffer << "    ";
            jit::add_type<T> (source_buffer);
            source_buffer << " *result) {" << std::endl;
            source_buffer << "    const unsigned int i = threadIdx.x;" << std::endl;
            source_buffer << "    const unsigned int j = threadIdx.x/64;" << std::endl;
            source_buffer << "    const unsigned int k = threadIdx.x%64;" << std::endl;
            source_buffer << "    if (i < " << size << ") {" << std::endl;
            source_buffer << "        " << jit::type_to_string<T> () << " sub_max = ";
            if constexpr (jit::is_complex<T> ()) {
                source_buffer << "abs<" << jit::type_to_string<T> () << "> (input[i]);" << std::endl;
            } else {
                source_buffer << "input[i];" << std::endl;
            }
            source_buffer << "        for (size_t index = i + 1024; index < " << size <<"; index += 1024) {" << std::endl;
            if constexpr (jit::is_complex<T> ()) {
                source_buffer << "            sub_max = max(sub_max, abs<" << jit::type_to_string<T> () << "> (input[index]));" << std::endl;
            } else {
                source_buffer << "            sub_max = max(sub_max, input[index]);" << std::endl;
            }
            source_buffer << "        }" << std::endl;
            source_buffer << "        __shared__ " << jit::type_to_string<T> () << " thread_max[64];" << std::endl;
            source_buffer << "        for (int index = 32; index > 0; index /= 2) {" << std::endl;
            source_buffer << "            sub_max = max(sub_max, __shfl_down(sub_max, index));" << std::endl;
            source_buffer << "        }" << std::endl;
            source_buffer << "        thread_max[j] = sub_max;" << std::endl;
            source_buffer << "        __syncthreads();" << std::endl;
            source_buffer << "        if (j == 0) {"  << std::endl;
            source_buffer << "            for (int index = 32; index > 0; index /= 2) {" << std::endl;
            source_buffer << "                thread_max[k] = max(thread_max[k], __shfl_down(thread_max[k], index));" << std::endl;
            source_buffer << "            }" << std::endl;
            source_buffer << "            *result = ";
            if constexpr (jit::is_complex<T> ()) {
                source_buffer << "make_complex<" << jit::type_to_string<T> () << "> (thread_max[0], thread_max[0]);" << std::endl;
            } else {
                source_buffer << "thread_max[0];" << std::endl;
            }
            source_buffer << "        }"  << std::endl;
            source_buffer << "    }"  << std::endl;
            source_buffer << "}" << std::endl << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Create a preamble.
///
///  @param[in,out] source_buffer Source buffer stream.
//------------------------------------------------------------------------------
        void create_preamble(std::ostringstream &source_buffer) {
            source_buffer << "extern \"C\" __global__ ";
        }

//------------------------------------------------------------------------------
///  @brief Create arg prefix.
///
///  @param[in,out] source_buffer Source buffer stream.
//------------------------------------------------------------------------------
        void create_argument_prefix(std::ostringstream &source_buffer) {}

//------------------------------------------------------------------------------
///  @brief Create arg postfix.
///
///  @param[in,out] source_buffer Source buffer stream.
///  @param[in]     index         Argument index.
//------------------------------------------------------------------------------
        void create_argument_postfix(std::ostringstream &source_buffer,
                                     const size_t index) {}

//------------------------------------------------------------------------------
///  @brief Create index argument.
///
///  @param[in,out] source_buffer Source buffer stream.
//------------------------------------------------------------------------------
        void create_index_argument(std::ostringstream &source_buffer) {}

//------------------------------------------------------------------------------
///  @brief Create index.
///
///  @param[in,out] source_buffer Source buffer stream.
//------------------------------------------------------------------------------
        void create_index(std::ostringstream &source_buffer) {
            source_buffer << "blockIdx.x*blockDim.x + threadIdx.x;";
        }

//------------------------------------------------------------------------------
///  @brief Get the buffer for a node.
///
///  @param[in] node Node to get the buffer for.
//------------------------------------------------------------------------------
        T *get_buffer(graph::shared_leaf<T, SAFE_MATH> &node) {
            return reinterpret_cast<T *> (kernel_arguments[node.get()]);
        }
    };
}

#endif /* hip_context_h */

