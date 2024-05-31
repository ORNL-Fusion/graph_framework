//------------------------------------------------------------------------------
///  @file cuda_context.hpp
///  @brief Cuda context for metal based gpus.
///
///  Defines context for cuda gpu.
//------------------------------------------------------------------------------

#ifndef cuda_context_h
#define cuda_context_h

#include <array>

#include <cuda.h>
#include <nvrtc.h>

#include "node.hpp"

namespace gpu {
//------------------------------------------------------------------------------
///  @brief  Check results of realtime compile.
///
///  @params[in] result Result code of the operation.
///  @params[in] name   Name of the operation.
//------------------------------------------------------------------------------
    static void check_nvrtc_error(nvrtcResult result,
                                  const std::string &name) {
#ifndef NDEBUG
        assert(result == NVRTC_SUCCESS && nvrtcGetErrorString(result));
#endif
    }

//------------------------------------------------------------------------------
///  @brief  Check results of cuda functions.
///
///  @params[in] result Result code of the operation.
///  @params[in] name   Name of the operation.
//------------------------------------------------------------------------------
    static void check_error(CUresult result,
                            const std::string &name) {
#ifndef NDEBUG
        const char *error;
        cuGetErrorString(result, &error);
        assert(result == CUDA_SUCCESS && error);
#endif
    }

//------------------------------------------------------------------------------
///   @brief Initalize cuda.
//------------------------------------------------------------------------------
    static CUresult cuda_init() {
        const CUresult result = cuInit(0);
        check_error(result, "cuInit");
        return result;
    }
///  Initalize Cuda.
    static const CUresult result = cuda_init();

//------------------------------------------------------------------------------
///  @brief Class representing a cuda gpu context.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class cuda_context {
    private:
///  The cuda device.
        CUdevice device;
///  The cuda context.
        CUcontext context;
///  The cuda code library.
        CUmodule module;
///  Argument map.
        std::map<graph::leaf_node<T, SAFE_MATH> *, CUdeviceptr> kernel_arguments;
///  Result buffer.
        CUdeviceptr result_buffer;
///  Cuda stream.
        CUstream stream;

//------------------------------------------------------------------------------
///  @brief  Check results of async cuda functions.
///
///  @params[in] result Result code of the operation.
///  @params[in] name   Name of the operation.
//------------------------------------------------------------------------------
        void check_error_async(CUresult result,
                               const std::string &name) {
            check_error(result, name);
#ifndef NDEBUG
            std::string async_name = name + "_async";
            check_error(cuStreamSynchronize(stream), async_name);
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
            check_error(cuDeviceGetCount(&count), "cuDeviceGetCount");
            return count;
        }

//------------------------------------------------------------------------------
///  @brief Device discription.
//------------------------------------------------------------------------------
        static std::string device_type() {
            return "Cuda GPU";
        }

//------------------------------------------------------------------------------
///  @brief Cuda context constructor.
///
///  @params[in] index Concurrent index.
//------------------------------------------------------------------------------
        cuda_context(const size_t index) : result_buffer(0), module(0) {
            check_error(cuDeviceGet(&device, index), "cuDeviceGet");
            check_error(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
            check_error(cuCtxSetCurrent(context), "cuCtxSetCurrent");
            check_error(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate");
        }

//------------------------------------------------------------------------------
///  @brief Cuda context destructor.
//------------------------------------------------------------------------------
        ~cuda_context() {
            if (module) {
                 check_error(cuModuleUnload(module), "cuModuleUnload");
                 module = 0;
            }

            for (auto &[key, value] : kernel_arguments) {
                check_error(cuMemFree(value), "cuMemFree");
            }

            if (result_buffer) {
                check_error(cuMemFree(result_buffer), "cuMemFree");
                result_buffer = 0;
            }

            check_error(cuStreamDestroy(stream), "cuStreamDestroy");
            check_error(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
        }

//------------------------------------------------------------------------------
///  @brief Compile the kernels.
///
///  @params[in] kernel_source Source code buffer for the kernel.
///  @params[in] names         Names of the kernel functions.
///  @params[in] add_reduction Include the reduction kernel.
//------------------------------------------------------------------------------
        void compile(const std::string kernel_source,
                     std::vector<std::string> names,
                     const bool add_reduction=false) {
            if (add_reduction) {
                names.push_back("max_reduction");
            }

            nvrtcProgram kernel_program;
            check_nvrtc_error(nvrtcCreateProgram(&kernel_program,
                                                 kernel_source.c_str(),
                                                 NULL, 0, NULL, NULL),
                              "nvrtcCreateProgram");

            for (std::string &name : names) {
                check_nvrtc_error(nvrtcAddNameExpression(kernel_program,
                                                         name.c_str()),
                                  "nvrtcAddNameExpression");
            }
            if (add_reduction) {
                check_nvrtc_error(nvrtcAddNameExpression(kernel_program,
                                                         "max_reduction"),
                                  "nvrtcAddNameExpression");
            }

            std::ostringstream arch;
            int compute_version;
            check_error(cuDeviceGetAttribute(&compute_version,
                                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                             device), "cuDeviceGetAttribute");
            arch << "--gpu-architecture=compute_";
            arch << compute_version;
            if (jit::verbose) {
                std::cout << "CUDA GPU info." << std::endl;
                std::cout << "  Major compute capability : " << compute_version << std::endl;
            }

            check_error(cuDeviceGetAttribute(&compute_version,
                                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                             device), "cuDeviceGetAttribute");
            arch << compute_version;
            if (jit::verbose) {
                std::cout << "  Minor compute capability : " << compute_version << std::endl;
            }

            char device_name[100];
            check_error(cuDeviceGetName(device_name, 100, device), "cuDeviceGetName");
            if (jit::verbose) {
                std::cout << "  Device name              : " << device_name << std::endl;
            }

            const std::string temp = arch.str();
            std::array<const char *, 6> options({
                temp.c_str(),
                "--std=c++17",
                "--include-path=" CUDA_INCLUDE,
                "--include-path=" HEADER_DIR,
                "--extra-device-vectorization",
		        "--device-as-default-execution-space"
            });

            if (nvrtcCompileProgram(kernel_program, options.size(), options.data())) {
                size_t log_size;
                check_nvrtc_error(nvrtcGetProgramLogSize(kernel_program, &log_size),
                                  "nvrtcGetProgramLogSize");

                char *log = static_cast<char *> (malloc(log_size));
                check_nvrtc_error(nvrtcGetProgramLog(kernel_program, log),
                                  "nvrtcGetProgramLog");
                std::cout << log << std::endl;
                free(log);
                std::cout << kernel_source << std::endl;
            }

            check_error(cuDeviceGetAttribute(&compute_version,
                                             CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
                                             device), "cuDeviceGetAttribute");
            if (jit::verbose) {
                std::cout << "  Managed Memory           : " << compute_version << std::endl;
            }

            size_t ptx_size;
            check_nvrtc_error(nvrtcGetPTXSize(kernel_program, &ptx_size),
                              "nvrtcGetPTXSize");

            char *ptx = static_cast<char *> (malloc(ptx_size));
            check_nvrtc_error(nvrtcGetPTX(kernel_program, ptx), "nvrtcGetPTX");

            check_nvrtc_error(nvrtcDestroyProgram(&kernel_program),
                              "nvrtcDestroyProgram");

            check_error(cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL), "cuModuleLoadDataEx");

            free(ptx);
        }

//------------------------------------------------------------------------------
///  @brief Create a kernel calling function.
///
///  @params[in] kernel_name   Name of the kernel for later reference.
///  @params[in] inputs        Input nodes of the kernel.
///  @params[in] outputs       Output nodes of the kernel.
///  @params[in] num_rays      Number of rays to trace.
///  @returns A lambda function to run the kernel.
//------------------------------------------------------------------------------
        std::function<void(void)> create_kernel_call(const std::string kernel_name,
                                                     graph::input_nodes<T, SAFE_MATH> inputs,
                                                     graph::output_nodes<T, SAFE_MATH> outputs,
                                                     const size_t num_rays) {
            CUfunction function;
            check_error(cuModuleGetFunction(&function, module, kernel_name.c_str()), "cuModuleGetFunction");

            std::vector<void *> buffers;

            const size_t buffer_element_size = sizeof(T);
            for (auto &input : inputs) {
                if (!kernel_arguments.contains(input.get())) {
                    kernel_arguments.try_emplace(input.get());
                    const backend::buffer<T> backend = input->evaluate();
                    check_error(cuMemAllocManaged(&kernel_arguments[input.get()],
                                                  backend.size()*sizeof(T),
                                                  CU_MEM_ATTACH_GLOBAL),
                                "cuMemAllocManaged");
                    check_error(cuMemcpyHtoD(kernel_arguments[input.get()],
                                             &backend[0],
                                             backend.size()*sizeof(T)),
                                "cuMemcpyHtoD");
                }
                buffers.push_back(reinterpret_cast<void *> (&kernel_arguments[input.get()]));
            }
            for (auto &output : outputs) {
                if (!kernel_arguments.contains(output.get())) {
                    kernel_arguments.try_emplace(output.get());
                    check_error(cuMemAllocManaged(&kernel_arguments[output.get()],
                                                  num_rays*sizeof(T),
                                                  CU_MEM_ATTACH_GLOBAL),
                                "cuMemAllocManaged");
                }
                buffers.push_back(reinterpret_cast<void *> (&kernel_arguments[output.get()]));
            }

            int value;
            check_error(cuFuncGetAttribute(&value, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                           function), "cuFuncGetAttribute");
            unsigned int threads_per_group = value;
            unsigned int thread_groups = num_rays/threads_per_group + (num_rays%threads_per_group ? 1 : 0);
            if (jit::verbose) {
                std::cout << "  Kernel name              : " << kernel_name << std::endl;
                std::cout << "    Threads per group  : " << threads_per_group << std::endl;
                std::cout << "    Number of groups   : " << thread_groups << std::endl;
                std::cout << "    Total problem size : " << threads_per_group*thread_groups << std::endl;
            }

            return [this, function, thread_groups, threads_per_group, buffers] () mutable {
                check_error_async(cuLaunchKernel(function, thread_groups, 1, 1,
                                                 threads_per_group, 1, 1, 0, stream,
                                                 buffers.data(), NULL),
                                  "cuLaunchKernel");
            };
        }

//------------------------------------------------------------------------------
///  @brief Create a max compute kernel calling function.
///
///  @params[in] argument Node to reduce.
///  @params[in] run      Function to run before reduction.
///  @returns A lambda function to run the kernel.
//------------------------------------------------------------------------------
        std::function<T(void)> create_max_call(graph::shared_leaf<T, SAFE_MATH> &argument,
                                               std::function<void(void)> run) {
            check_error(cuMemAllocManaged(&result_buffer, sizeof(T),
                                          CU_MEM_ATTACH_GLOBAL),
                        "cuMemAllocManaged");

            std::vector<void *> buffers;

            buffers.push_back(reinterpret_cast<void *> (&kernel_arguments[argument.get()]));
            buffers.push_back(reinterpret_cast<void *> (&result_buffer));

            CUfunction function;
            check_error(cuModuleGetFunction(&function, module, "max_reduction"),
                        "cuModuleGetFunction");

            if (jit::verbose) {
                std::cout << "  Kernel name              : max_reduction" << std::endl;
            }

            return [this, function, run, buffers] () mutable {
                run();
                check_error_async(cuLaunchKernel(function, 1, 1, 1,
                                                 1024, 1, 1, 0, stream,
                                                 buffers.data(), NULL),
                                  "cuLaunchKernel");
                wait();

                return reinterpret_cast<T *> (result_buffer)[0];
            };
        }

//------------------------------------------------------------------------------
///  @brief Hold the current thread until the stream has completed.
//------------------------------------------------------------------------------
        void wait() {
            check_error_async(cuStreamSynchronize(stream), "cuStreamSynchronize");
            check_error(cuCtxSynchronize(), "cuCtxSynchronize");
        }

//------------------------------------------------------------------------------
///  @brief Print out the results.
///
///  @params[in] index Number of times to record.
///  @params[in] nodes Nodes to output.
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
///  @params[in] index Ray index to check value for.
///  @params[in] node  Node to check the value for.
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
///  @params[in] node   Not to copy buffer to.
///  @params[in] source Host side buffer to copy from.
//------------------------------------------------------------------------------
        void copy_to_device(graph::shared_leaf<T, SAFE_MATH> node,
                            T *source) {
            size_t size;
            check_error(cuMemGetAddressRange(NULL, &size, kernel_arguments[node.get()]), "cuMemGetAddressRange");
            check_error_async(cuMemcpyHtoDAsync(kernel_arguments[node.get()], source, size, stream), "cuMemcpyHtoDAsync");
        }

//------------------------------------------------------------------------------
///  @brief Copy buffer contents to host.
///
///  @params[in]     node        Node to copy buffer from.
///  @params[in,out] destination Host side buffer to copy to.
//------------------------------------------------------------------------------
        void copy_to_host(graph::shared_leaf<T, SAFE_MATH> node,
                          T *destination) {
            size_t size;
            check_error(cuMemGetAddressRange(NULL, &size, kernel_arguments[node.get()]), "cuMemGetAddressRange");
            check_error_async(cuMemcpyDtoHAsync(destination, kernel_arguments[node.get()], size, stream), "cuMemcpyDtoHAsync");
        }

//------------------------------------------------------------------------------
///  @brief Create the source header.
///
///  @params[in,out] source_buffer Source buffer stream.
//------------------------------------------------------------------------------
        void create_header(std::ostringstream &source_buffer) {
            if constexpr (jit::is_complex<T> ()) {
                source_buffer << "#define CUDA_DEVICE_CODE" << std::endl;
                source_buffer << "#define M_PI " << M_PI << std::endl;
                source_buffer << "#include <cuda/std/complex>" << std::endl;
                source_buffer << "#include <special_functions.hpp>" << std::endl;
            }
        }

//------------------------------------------------------------------------------
///  @brief Create kernel prefix.
///
///  @params[in,out] source_buffer Source buffer stream.
///  @params[in]     name          Name to call the kernel.
///  @params[in]     inputs        Input variables of the kernel.
///  @params[in]     outputs       Output nodes of the graph to compute.
///  @params[in]     size          Size of the input buffer.
///  @params[in]     is_constant   Flags if the input is read only.
///  @params[in,out] registers     Map of used registers.
//------------------------------------------------------------------------------
        void create_kernel_prefix(std::ostringstream &source_buffer,
                                  const std::string name,
                                  graph::input_nodes<T, SAFE_MATH> &inputs,
                                  graph::output_nodes<T, SAFE_MATH> &outputs,
                                  const size_t size, 
                                  const std::vector<bool> &is_constant,
                                  jit::register_map &registers) {
            source_buffer << std::endl;
            source_buffer << "extern \"C\" __global__ __launch_bounds__(1024) void "
                          << name << "(" << std::endl;

            source_buffer << "    ";
            if (is_constant[0]) {
                source_buffer << "const ";
            }
            jit::add_type<T> (source_buffer);
            source_buffer << " *" << jit::to_string('v', inputs[0].get());
            for (size_t i = 1, ie = inputs.size(); i < ie; i++) {
                source_buffer << "," << std::endl;
                source_buffer << "    ";
                if (is_constant[i]) {
                    source_buffer << "const ";
                }
                jit::add_type<T> (source_buffer);
                source_buffer << " *" << jit::to_string('v', inputs[i].get());
            }

            for (size_t i = 0, ie = outputs.size(); i < ie; i++) {
                source_buffer << "," << std::endl;
                source_buffer << "    ";
                jit::add_type<T> (source_buffer);
                source_buffer << " *" << jit::to_string('o', outputs[i].get());
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
///  @params[in,out] source_buffer Source buffer stream.
///  @params[in]     outputs       Output nodes of the graph to compute.
///  @params[in]     setters       Map outputs back to input values.
///  @params[in,out] registers     Map of used registers.

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
///  @params[in,out] source_buffer Source buffer stream.
///  @params[in]     size          Size of the input buffer.
//------------------------------------------------------------------------------
        void create_reduction(std::ostringstream &source_buffer,
                              const size_t size) {
            source_buffer << std::endl;
            source_buffer << "extern \"C\" __global__ __launch_bounds__(1024) void max_reduction(" << std::endl;
            source_buffer << "    const ";
            jit::add_type<T> (source_buffer);
            source_buffer << " *input," << std::endl;
            source_buffer << "    ";
            jit::add_type<T> (source_buffer);
            source_buffer << " *result) {" << std::endl;
            source_buffer << "    const unsigned int i = threadIdx.x;" << std::endl;
            source_buffer << "    const unsigned int j = threadIdx.x/32;" << std::endl;
            source_buffer << "    const unsigned int k = threadIdx.x%32;" << std::endl;
            source_buffer << "    if (i < " << size << ") {" << std::endl;
            source_buffer << "        " << jit::type_to_string<T> () << " sub_max = ";
            if constexpr (jit::is_complex<T> ()) {
                source_buffer << "abs(input[i]);" << std::endl;
            } else {
                source_buffer << "input[i];" << std::endl;
            }
            source_buffer << "        for (size_t index = i + 1024; index < " << size <<"; index += 1024) {" << std::endl;
            if constexpr (jit::is_complex<T> ()) {
                source_buffer << "            sub_max = max(sub_max, abs(input[index]));" << std::endl;
            } else {
                source_buffer << "            sub_max = max(sub_max, input[index]);" << std::endl;
            }
            source_buffer << "        }" << std::endl;
            source_buffer << "        __shared__ " << jit::type_to_string<T> () << " thread_max[32];" << std::endl;
            source_buffer << "        for (int index = 16; index > 0; index /= 2) {" << std::endl;
            source_buffer << "            sub_max = max(sub_max, __shfl_down_sync(__activemask(), sub_max, index));" << std::endl;
            source_buffer << "        }" << std::endl;
            source_buffer << "        thread_max[j] = sub_max;" << std::endl;
            source_buffer << "        __syncthreads();" << std::endl;
            source_buffer << "        if (j == 0) {"  << std::endl;
            source_buffer << "            for (int index = 16; index > 0; index /= 2) {" << std::endl;
            source_buffer << "                thread_max[k] = max(thread_max[k], __shfl_down_sync(__activemask(), thread_max[k], index));" << std::endl;
            source_buffer << "            }" << std::endl;
            source_buffer << "            *result = thread_max[0];" << std::endl;
            source_buffer << "        }"  << std::endl;
            source_buffer << "    }"  << std::endl;
            source_buffer << "}" << std::endl << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Create a preamble.
///
///  @params[in,out] source_buffer Source buffer stream.
//------------------------------------------------------------------------------
        void create_preamble(std::ostringstream &source_buffer) {
            source_buffer << "extern \"C\" __global__ ";
        }

//------------------------------------------------------------------------------
///  @brief Create arg prefix.
///
///  @params[in,out] source_buffer Source buffer stream.
//------------------------------------------------------------------------------
        void create_argument_prefix(std::ostringstream &source_buffer) {}

//------------------------------------------------------------------------------
///  @brief Create arg postfix.
///
///  @params[in,out] source_buffer Source buffer stream.
///  @params[in]     index         Argument index.
//------------------------------------------------------------------------------
        void create_argument_postfix(std::ostringstream &source_buffer,
                                     const size_t index) {}

//------------------------------------------------------------------------------
///  @brief Create index argument.
///
///  @params[in,out] source_buffer Source buffer stream.
//------------------------------------------------------------------------------
        void create_index_argument(std::ostringstream &source_buffer) {}

//------------------------------------------------------------------------------
///  @brief Create index.
///
///  @params[in,out] source_buffer Source buffer stream.
//------------------------------------------------------------------------------
        void create_index(std::ostringstream &source_buffer) {
            source_buffer << "blockIdx.x*blockDim.x + threadIdx.x;";
        }

//------------------------------------------------------------------------------
///  @brief Get the buffer for a node.
///
///  @params[in] node Node to get the buffer for.
//------------------------------------------------------------------------------
        T *get_buffer(graph::shared_leaf<T, SAFE_MATH> &node) {
            return reinterpret_cast<T *> (kernel_arguments[node.get()]);
        }
    };
}

#endif /* cuda_context_h */
