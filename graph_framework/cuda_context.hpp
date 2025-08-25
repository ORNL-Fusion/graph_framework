//------------------------------------------------------------------------------
///  @file cuda_context.hpp
///  @brief Cuda context for metal based gpus.
///
///  Defines context for cuda gpu.
//------------------------------------------------------------------------------

#ifndef cuda_context_h
#define cuda_context_h

#include <unordered_set>
#include <array>
#include <cstring>

#include <cuda.h>
#include <nvrtc.h>

#include "random.hpp"

///  Maximum number of registers to use.
#define MAX_REG 128

namespace gpu {
//------------------------------------------------------------------------------
///  @brief  Check results of realtime compile.
///
///  @param[in] result Result code of the operation.
///  @param[in] name   Name of the operation.
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
///  @param[in] result Result code of the operation.
///  @param[in] name   Name of the operation.
//------------------------------------------------------------------------------
    static void check_error(CUresult result,
                            const std::string &name) {
#ifndef NDEBUG
        const char *error;
        cuGetErrorString(result, &error);
        if (result != CUDA_SUCCESS) {
            std::cerr << name << " " << std::string(error) << std::endl;
        }
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
#ifdef USE_CUDA_TEXTURES
///  Textures.
        std::map<void *, CUtexObject> texture_arguments;
#endif
///  Result buffer.
        CUdeviceptr result_buffer;
///  Offset buffer.
        CUdeviceptr offset_buffer;
///  Cuda stream.
        CUstream stream;

//------------------------------------------------------------------------------
///  @brief  Check results of async cuda functions.
///
///  @param[in] result Result code of the operation.
///  @param[in] name   Name of the operation.
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
///  Size of random state needed.
        constexpr static size_t random_state_size = 1024;

///  Remaining constant memory in bytes.
        int remaining_const_memory;

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
///  @param[in] index Concurrent index.
//------------------------------------------------------------------------------
        cuda_context(const size_t index) : result_buffer(0), module(0), offset_buffer(0) {
            check_error(cuDeviceGet(&device, index), "cuDeviceGet");
            check_error(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
            check_error(cuCtxSetCurrent(context), "cuCtxSetCurrent");
            check_error(cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1), "cuCtxSetCacheConfig");
            check_error(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate");
            check_error(cuDeviceGetAttribute(&remaining_const_memory,
                                             CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
                                             device), "cuDeviceGetAttribute");
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

#ifdef USE_CUDA_TEXTURES
            for (auto &[key, value] : texture_arguments) {
                CUDA_RESOURCE_DESC resource;
                check_error(cuTexObjectGetResourceDesc(&resource, value),
                            "cuTexObjectGetResourceDesc");

                check_error(cuArrayDestroy(resource.res.array.hArray), "cuArrayDestroy");
                check_error(cuTexObjectDestroy(value), "cuTexObjectDestroy");
            }
#endif

            if (result_buffer) {
                check_error(cuMemFree(result_buffer), "cuMemFree");
                result_buffer = 0;
            }
            if (offset_buffer) {
                check_error(cuMemFree(offset_buffer), "cuMemFree");
                offset_buffer = 0;
            }

            check_error(cuStreamDestroy(stream), "cuStreamDestroy");
            check_error(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
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
            std::array<const char *, 8> options({
                temp.c_str(),
                "--std=c++17",
                "--relocatable-device-code=false",
                "--include-path=" CUDA_INCLUDE,
                "--include-path=" HEADER_DIR,
                "--extra-device-vectorization",
                "--device-as-default-execution-space",
                "--use_fast_math"
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

            std::array<CUjit_option, 3> module_options = {
                CU_JIT_MAX_REGISTERS,
                CU_JIT_LTO,
                CU_JIT_POSITION_INDEPENDENT_CODE
            };
            std::array<void *, 3> module_values = {
                reinterpret_cast<void *> (MAX_REG),
                reinterpret_cast<void *> (1),
                reinterpret_cast<void *> (0)
            };

            check_error(cuModuleLoadDataEx(&module, ptx, module_options.size(),
                                           module_options.data(),
                                           module_values.data()), "cuModuleLoadDataEx");

            free(ptx);
        }

//------------------------------------------------------------------------------
///  @brief Create a kernel calling function.
///
///  @param[in] kernel_name Name of the kernel for later reference.
///  @param[in] inputs      Input nodes of the kernel.
///  @param[in] outputs     Output nodes of the kernel.
///  @param[in] state       Random states.
///  @param[in] num_rays    Number of rays to trace.'
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
                    buffers.push_back(reinterpret_cast<void *> (&kernel_arguments[input.get()]));
                }
            }
            for (auto &output : outputs) {
                if (!kernel_arguments.contains(output.get())) {
                    kernel_arguments.try_emplace(output.get());
                    check_error(cuMemAllocManaged(&kernel_arguments[output.get()],
                                                  num_rays*sizeof(T),
                                                  CU_MEM_ATTACH_GLOBAL),
                                "cuMemAllocManaged");
                    buffers.push_back(reinterpret_cast<void *> (&kernel_arguments[output.get()]));
                }
            }

            const size_t num_buffers = buffers.size();
            if (state.get()) {
                if (!kernel_arguments.contains(state.get())) {
                    kernel_arguments.try_emplace(state.get());
                    check_error(cuMemAllocManaged(&kernel_arguments[state.get()],
                                                  state->get_size_bytes(),
                                                  CU_MEM_ATTACH_GLOBAL),
                                "cuMemAllocManaged");
                    check_error(cuMemAlloc(&offset_buffer, sizeof(uint32_t)), "cuMemAlloc");
                    check_error(cuMemcpyHtoD(kernel_arguments[state.get()],
                                             state->data(),
                                             state->get_size_bytes()),
                                "cuMemcpyHtoD");
                }
                buffers.push_back(reinterpret_cast<void *> (&kernel_arguments[state.get()]));
                buffers.push_back(reinterpret_cast<void *> (&offset_buffer));
            }

#ifdef USE_CUDA_TEXTURES
            for (auto &[data, size] : tex1d_list) {
                if (!texture_arguments.contains(data)) {
                    texture_arguments.try_emplace(data);
                    CUDA_RESOURCE_DESC resource_desc;
                    CUDA_TEXTURE_DESC texture_desc;
                    CUDA_ARRAY_DESCRIPTOR array_desc;

                    array_desc.Width = size;
                    array_desc.Height = 1;

                    memset(&resource_desc, 0, sizeof(CUDA_RESOURCE_DESC));
                    memset(&texture_desc, 0, sizeof(CUDA_TEXTURE_DESC));

                    resource_desc.resType = CU_RESOURCE_TYPE_ARRAY;
                    texture_desc.addressMode[0] = CU_TR_ADDRESS_MODE_BORDER;
                    texture_desc.addressMode[1] = CU_TR_ADDRESS_MODE_BORDER;
                    texture_desc.addressMode[2] = CU_TR_ADDRESS_MODE_BORDER;
                    if constexpr (jit::float_base<T>) {
                        array_desc.Format = CU_AD_FORMAT_FLOAT;
                        if constexpr (jit::complex_scalar<T>) {
                            array_desc.NumChannels = 2;
                        } else {
                            array_desc.NumChannels = 1;
                        }
                    } else {
                        array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
                        if constexpr (jit::complex_scalar<T>) {
                            array_desc.NumChannels = 4;
                        } else {
                            array_desc.NumChannels = 2;
                        }
                    }
                    check_error(cuArrayCreate(&resource_desc.res.array.hArray, &array_desc),
                                "cuArrayCreate");
                    check_error(cuMemcpyHtoA(resource_desc.res.array.hArray, 0, data,
                                             size*sizeof(float)*array_desc.NumChannels),
                                "cuMemcpyHtoA");

                    check_error(cuTexObjectCreate(&texture_arguments[data],
                                                  &resource_desc, &texture_desc,
                                                  NULL),
                                "cuTexObjectCreate");
                }
                buffers.push_back(reinterpret_cast<void *> (&texture_arguments[data]));
            }
            for (auto &[data, size] : tex2d_list) {
                if (!texture_arguments.contains(data)) {
                    texture_arguments.try_emplace(data);
                    CUDA_RESOURCE_DESC resource_desc;
                    CUDA_TEXTURE_DESC texture_desc;
                    CUDA_ARRAY_DESCRIPTOR array_desc;

                    array_desc.Width = size[0];
                    array_desc.Height = size[1];

                    memset(&resource_desc, 0, sizeof(CUDA_RESOURCE_DESC));
                    memset(&texture_desc, 0, sizeof(CUDA_TEXTURE_DESC));

                    resource_desc.resType = CU_RESOURCE_TYPE_ARRAY;
                    texture_desc.addressMode[0] = CU_TR_ADDRESS_MODE_BORDER;
                    texture_desc.addressMode[1] = CU_TR_ADDRESS_MODE_BORDER;
                    texture_desc.addressMode[2] = CU_TR_ADDRESS_MODE_BORDER;
                    const size_t total = size[0]*size[1];
                    if constexpr (jit::float_base<T>) {
                        array_desc.Format = CU_AD_FORMAT_FLOAT;
                        if constexpr (jit::complex_scalar<T>) {
                            array_desc.NumChannels = 2;
                        } else {
                            array_desc.NumChannels = 1;
                        }
                    } else {
                        array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
                        if constexpr (jit::complex_scalar<T>) {
                            array_desc.NumChannels = 4;
                        } else {
                            array_desc.NumChannels = 2;
                        }
                    }
                    check_error(cuArrayCreate(&resource_desc.res.array.hArray, &array_desc),
                                "cuArrayCreate");

                    CUDA_MEMCPY2D copy_desc;
                    memset(&copy_desc, 0, sizeof(copy_desc));

                    copy_desc.srcPitch = size[0]*sizeof(float)*array_desc.NumChannels;
                    copy_desc.srcMemoryType = CU_MEMORYTYPE_HOST;
                    copy_desc.srcHost = data;

                    copy_desc.dstMemoryType = CU_MEMORYTYPE_ARRAY;
                    copy_desc.dstArray = resource_desc.res.array.hArray;

                    copy_desc.WidthInBytes = copy_desc.srcPitch;
                    copy_desc.Height = size[0];

                    check_error(cuMemcpy2D(&copy_desc), "cuMemcpy2D");

                    check_error(cuTexObjectCreate(&texture_arguments[data],
                                                  &resource_desc, &texture_desc,
                                                  NULL),
                                "cuTexObjectCreate");
                }
                buffers.push_back(reinterpret_cast<void *> (&texture_arguments[data]));
            }
#endif

            int value;
            check_error(cuFuncGetAttribute(&value, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                           function), "cuFuncGetAttribute");
            unsigned int threads_per_group = value;
            unsigned int thread_groups = num_rays/threads_per_group + (num_rays%threads_per_group ? 1 : 0);

            int min_grid;
            check_error(cuOccupancyMaxPotentialBlockSize(&min_grid, &value, function, 0, 0, 0),
                        "cuOccupancyMaxPotentialBlockSize");

            if (jit::verbose) {
                std::cout << "  Kernel name              : " << kernel_name << std::endl;
                std::cout << "    Threads per group    : " << threads_per_group << std::endl;
                std::cout << "    Number of groups     : " << thread_groups << std::endl;
                std::cout << "    Total problem size   : " << threads_per_group*thread_groups << std::endl;
                std::cout << "    Min grid size        : " << min_grid << std::endl;
                std::cout << "    Suggested Block size : " << value << std::endl;
            }

            if (state.get()) {
                return [this, num_rays, function, threads_per_group, buffers] () mutable {
                    for (uint32_t i = 0; i < num_rays; i += threads_per_group) {
                        check_error_async(cuStreamWriteValue32(stream, offset_buffer, i,
                                                               CU_STREAM_WRITE_VALUE_DEFAULT),
                                          "cuStreamWriteValue32");
                        check_error_async(cuLaunchKernel(function,
                                                         1, 1, 1,
                                                         threads_per_group, 1, 1,
                                                         0, stream,
                                                         buffers.data(), NULL),
                                          "cuLaunchKernel");
                    }
                };
            } else {
                return [this, function, thread_groups, threads_per_group, buffers] () mutable {
                    check_error_async(cuLaunchKernel(function, thread_groups, 1, 1,
                                                     threads_per_group, 1, 1, 0, stream,
                                                     buffers.data(), NULL),
                                      "cuLaunchKernel");
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
            check_error(cuMemAllocManaged(&result_buffer, sizeof(T),
                                          CU_MEM_ATTACH_GLOBAL),
                        "cuMemAllocManaged");

            std::vector<void *> buffers;

            buffers.push_back(reinterpret_cast<void *> (&kernel_arguments[argument.get()]));
            buffers.push_back(reinterpret_cast<void *> (&result_buffer));

            CUfunction function;
            check_error(cuModuleGetFunction(&function, module, "max_reduction"),
                        "cuModuleGetFunction");

            int value;
            int min_grid;
            check_error(cuOccupancyMaxPotentialBlockSize(&min_grid, &value, function, 0, 0, 0),
                        "cuOccupancyMaxPotentialBlockSize");

            if (jit::verbose) {
                std::cout << "  Kernel name              : max_reduction" << std::endl;
                std::cout << "    Min grid size        : " << min_grid << std::endl;
                std::cout << "    Suggested Block size : " << value << std::endl;
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
///  @param[in] index Number of times to record.
///  @param[in] nodes Nodes to output.
//------------------------------------------------------------------------------
        void print_results(const size_t index,
                           const graph::output_nodes<T, SAFE_MATH> &nodes) {
            wait();
            for (auto &out : nodes) {
                const T temp = reinterpret_cast<T *> (kernel_arguments[out.get()])[index];
                if constexpr (jit::complex_scalar<T>) {
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
            check_error(cuMemGetAddressRange(NULL, &size, kernel_arguments[node.get()]), "cuMemGetAddressRange");
            check_error_async(cuMemcpyHtoDAsync(kernel_arguments[node.get()], source, size, stream), "cuMemcpyHtoDAsync");
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
            check_error(cuMemGetAddressRange(NULL, &size, kernel_arguments[node.get()]), "cuMemGetAddressRange");
            check_error_async(cuMemcpyDtoHAsync(destination, kernel_arguments[node.get()], size, stream), "cuMemcpyDtoHAsync");
        }

//------------------------------------------------------------------------------
///  @brief Create the source header.
///
///  @param[in,out] source_buffer Source buffer stream.
//------------------------------------------------------------------------------
        void create_header(std::ostringstream &source_buffer) {
            source_buffer << "typedef unsigned int uint32_t;"                << std::endl
                          << "typedef unsigned short uint16_t;"              << std::endl
                          << "typedef short int16_t;"                        << std::endl
                          << "template<typename T, size_t S>"                << std::endl
                          << "class array {"                                 << std::endl
                          << "private:"                                      << std::endl
                          << "    T _buffer[S];"                             << std::endl
                          << "public:"                                       << std::endl
                          << "    T operator[] (const size_t index) const {" << std::endl
                          << "        return _buffer[index];"                << std::endl
                          << "    }"                                         << std::endl
                          << "    T &operator[] (const size_t index) {"      << std::endl
                          << "        return _buffer[index];"                << std::endl
                          << "    }"                                         << std::endl
                          << "};"                                            << std::endl;
            if constexpr (jit::complex_scalar<T>) {
                source_buffer << "#define CUDA_DEVICE_CODE"         << std::endl
                              << "#define M_PI " << M_PI            << std::endl
                              << "#include <cuda/std/complex>"      << std::endl
                              << "#include <special_functions.hpp>" << std::endl;
#ifdef USE_CUDA_TEXTURES
                if constexpr (jit::float_base<T>) {
                    source_buffer << "static __inline__ __device__ complex<float> to_cmp_float(float2 p) {"
                                  << std::endl
                                  << "    return ";
                    jit::add_type<T> (source_buffer);
                    source_buffer << " (p.x, p.y);" << std::endl
                                  << "}" << std::endl;
                } else {
                    source_buffer << "static __inline__ __device__ complex<double> to_cmp_double(uint4 p) {"
                                  << std::endl
                                  << "    return ";
                    jit::add_type<T> (source_buffer);
                    source_buffer << " (__hiloint2double(p.y, p.x), __hiloint2double(p.w, p.z));"
                                  << std::endl
                                  << "}" << std::endl;
                }
            } else if constexpr (jit::double_base<T>) {
                source_buffer << "static __inline__ __device__ double to_double(uint2 p) {"
                              << std::endl
                              << "    return __hiloint2double(p.y, p.x);"
                              << std::endl
                              << "}" << std::endl;
#endif
            }
        }

//------------------------------------------------------------------------------
///  @brief Create kernel prefix.
///
///  @param[in,out] source_buffer Source buffer stream.
///  @param[in]     name          Name to call the kernel.
///  @param[in]     inputs        Input variables of the kernel.
///  @param[in]     outputs       Output nodes of the graph to compute.
///  @param[in]     state         Random states.
///  @param[in]     size          Size of the input buffer.
///  @param[in]     is_constant   Flags if the input is read only.
///  @param[in,out] registers     Map of used registers.
///  @param[in]     usage         List of register usage count.
///  @param[in]     textures1d    List of 1D kernel textures.
///  @param[in]     textures2d    List of 2D kernel textures.
//------------------------------------------------------------------------------
        void create_kernel_prefix(std::ostringstream &source_buffer,
                                  const std::string name,
                                  graph::input_nodes<T, SAFE_MATH> &inputs,
                                  graph::output_nodes<T, SAFE_MATH> &outputs,
                                  graph::shared_random_state<T, SAFE_MATH> state,
                                  const size_t size,
                                  const std::vector<bool> &is_constant,
                                  jit::register_map &registers,
                                  const jit::register_usage &usage,
                                  jit::texture1d_list &textures1d,
                                  jit::texture2d_list &textures2d) {
            source_buffer << std::endl;
            source_buffer << "extern \"C\" __global__ void "
                          << name << "(" << std::endl;

            std::unordered_set<void *> used_args;
            if (inputs.size()) {
                source_buffer << "    ";
                if (is_constant[0]) {
                    source_buffer << "const ";
                }
                jit::add_type<T> (source_buffer);
                source_buffer << " * __restrict__ "
                              << jit::to_string('v', inputs[0].get());
                used_args.insert(inputs[0].get());
            }
            for (size_t i = 1, ie = inputs.size(); i < ie; i++) {
                if (!used_args.contains(inputs[i].get())) {
                    source_buffer << ", // " << inputs[i - 1]->get_symbol()
#ifndef USE_INPUT_CACHE
#ifdef SHOW_USE_COUNT
                                  << " used " << usage.at(inputs[i - 1].get())
#endif
#endif
                                  << std::endl;
                    source_buffer << "    ";
                    if (is_constant[i]) {
                        source_buffer << "const ";
                    }
                    jit::add_type<T> (source_buffer);
                    source_buffer << " * __restrict__ "
                                  << jit::to_string('v', inputs[i].get());
                    used_args.insert(inputs[i].get());
                }
            }
            for (size_t i = 0, ie = outputs.size(); i < ie; i++) {
                if (i == 0) {
                    if (inputs.size()) {
                        source_buffer << ", // "
                                      << inputs[inputs.size() - 1]->get_symbol();
#ifndef USE_INPUT_CACHE
#ifdef SHOW_USE_COUNT
                        source_buffer << " used "
                                      << usage.at(inputs[inputs.size() - 1].get());
#endif
#endif
                        source_buffer << std::endl;
                    }
                } else {
                    source_buffer << "," << std::endl;
                }

                if (!used_args.contains(outputs[i].get())) {
                    source_buffer << "    ";
                    jit::add_type<T> (source_buffer);
                    source_buffer << " *  __restrict__ "
                                  << jit::to_string('o', outputs[i].get());
                    used_args.insert(outputs[i].get());
                }
            }
            if (state.get()) {
                source_buffer << "," << std::endl
                              << "    mt_state * __restrict__ "
                              << jit::to_string('s', state.get())
                              << "," << std::endl
                              << "    const uint32_t * __restrict__ offset"
                              << std::endl;
            }
#ifdef USE_CUDA_TEXTURES
            for (auto &[key, value] : textures1d) {
                source_buffer << "," << std::endl;
                source_buffer << "    cudaTextureObject_t "
                              << jit::to_string('a', key);
            }
            for (auto &[key, value] : textures2d) {
                source_buffer << "," << std::endl;
                source_buffer << "    cudaTextureObject_t "
                              << jit::to_string('a', key);
            }
#endif
            source_buffer << ") {" << std::endl
                          << "    const int index = blockIdx.x*blockDim.x + threadIdx.x;"
                          << std::endl;
            if (state.get()) {
#ifdef USE_INPUT_CACHE
                registers[state.get()] = jit::to_string('r', state.get());
                source_buffer << "    mt_state &" << registers[state.get()] << " = "
                              << jit::to_string('s', state.get())
                              << "[threadIdx.x];"
#ifdef SHOW_USE_COUNT
                              << " // used " << usage.at(state.get())
#endif
                              << std::endl;
#else
                registers[state.get()] = jit::to_string('s', state.get()) + "[threadIdx.x]";
#endif
            }
            source_buffer << "    if (";
            if (state.get()) {
                source_buffer << "offset[0] + ";
            }
            source_buffer << "index < " << size << ") {" << std::endl;


            for (auto &input : inputs) {
#ifdef USE_INPUT_CACHE
                if (usage.at(input.get())) {
                    registers[input.get()] = jit::to_string('r', input.get());
                    source_buffer << "        const ";
                    jit::add_type<T> (source_buffer);
                    source_buffer << " " << registers[input.get()] << " = "
                                  << jit::to_string('v', input.get())
                                  << "[";
                    if (state.get()) {
                        source_buffer << "offset[0] + ";
                    }
                    source_buffer << "index]; // " << input->get_symbol()
#ifdef SHOW_USE_COUNT
                                  << " used " << usage.at(input.get())
#endif
                                  << std::endl;
                }
#else
                registers[input.get()] = jit::to_string('v', input.get()) + "[index]";
#endif
            }
        }

//------------------------------------------------------------------------------
///  @brief Create kernel postfix.
///
///  @param[in,out] source_buffer Source buffer stream.
///  @param[in]     outputs       Output nodes of the graph to compute.
///  @param[in]     setters       Map outputs back to input values.
///  @param[in]     state         Random states.
///  @param[in,out] registers     Map of used registers.
///  @param[in,out] indices       Map of used indices.
///  @param[in]     usage         List of register usage count.
//------------------------------------------------------------------------------
        void create_kernel_postfix(std::ostringstream &source_buffer,
                                   graph::output_nodes<T, SAFE_MATH> &outputs,
                                   graph::map_nodes<T, SAFE_MATH> &setters,
                                   graph::shared_random_state<T, SAFE_MATH> state,
                                   jit::register_map &registers,
                                   jit::register_map &indices,
                                   const jit::register_usage &usage) {
            std::unordered_set<void *> out_registers;
            for (auto &[out, in] : setters) {
                if (!out->is_match(in) &&
                    !out_registers.contains(out.get())) {
                    graph::shared_leaf<T, SAFE_MATH> a = out->compile(source_buffer,
                                                                      registers,
                                                                      indices,
                                                                      usage);
                    source_buffer << "        "
                                  << jit::to_string('v',  in.get())
                                  << "[";
                    if (state.get()) {
                        source_buffer << "offset[0] + ";
                    }
                    source_buffer << "index] = ";
                    if constexpr (SAFE_MATH) {
                        if constexpr (jit::complex_scalar<T>) {
                            jit::add_type<T> (source_buffer);
                            source_buffer << " (";
                            source_buffer << "isnan(real(" << registers[a.get()]
                                          << ")) ? 0.0 : real("
                                          << registers[a.get()]
                                          << "), ";
                            source_buffer << "isnan(imag(" << registers[a.get()]
                                          << ")) ? 0.0 : imag("
                                          << registers[a.get()]
                                          << "));" << std::endl;
                        } else {
                            source_buffer << "isnan(" << registers[a.get()]
                                          << ") ? 0.0 : " << registers[a.get()]
                                          << ";" << std::endl;
                        }
                    } else {
                        source_buffer << registers[a.get()] << ";" << std::endl;
                    }
                    out_registers.insert(out.get());
                }
            }

            for (auto &out : outputs) {
                if (!graph::variable_cast(out).get() &&
                    !out_registers.contains(out.get())) {
                    graph::shared_leaf<T, SAFE_MATH> a = out->compile(source_buffer,
                                                                      registers,
                                                                      indices,
                                                                      usage);
                    source_buffer << "        "
                                  << jit::to_string('o',  out.get())
                                  << "[";
                    if (state.get()) {
                        source_buffer << "offset[0] + ";
                    }
                    source_buffer << "index] = ";
                    if constexpr (SAFE_MATH) {
                        if constexpr (jit::complex_scalar<T>) {
                            jit::add_type<T> (source_buffer);
                            source_buffer << " (";
                            source_buffer << "isnan(real(" << registers[a.get()]
                                          << ")) ? 0.0 : real("
                                          << registers[a.get()]
                                          << "), ";
                            source_buffer << "isnan(imag(" << registers[a.get()]
                                          << ")) ? 0.0 : imag("
                                          << registers[a.get()]
                                          << "));" << std::endl;
                        } else {
                            source_buffer << "isnan(" << registers[a.get()]
                                          << ") ? 0.0 : " << registers[a.get()]
                                          << ";" << std::endl;
                        }
                    } else {
                        source_buffer << registers[a.get()] << ";" << std::endl;
                    }
                    out_registers.insert(out.get());
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
            source_buffer << "    const ";
            jit::add_type<T> (source_buffer);
            source_buffer << " * __restrict__ input," << std::endl;
            source_buffer << "    ";
            jit::add_type<T> (source_buffer);
            source_buffer << " * __restrict__ result) {" << std::endl;
            source_buffer << "    const unsigned int i = threadIdx.x;" << std::endl;
            source_buffer << "    const unsigned int j = threadIdx.x/32;" << std::endl;
            source_buffer << "    const unsigned int k = threadIdx.x%32;" << std::endl;
            source_buffer << "    if (i < " << size << ") {" << std::endl;
            source_buffer << "        " << jit::type_to_string<T> () << " sub_max = ";
            if constexpr (jit::complex_scalar<T>) {
                source_buffer << "abs(input[i]);" << std::endl;
            } else {
                source_buffer << "input[i];" << std::endl;
            }
            source_buffer << "        for (size_t index = i + 1024; index < " << size <<"; index += 1024) {" << std::endl;
            if constexpr (jit::complex_scalar<T>) {
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
///  @brief Get the buffer for a node.
///
///  @param[in] node Node to get the buffer for.
//------------------------------------------------------------------------------
        T *get_buffer(graph::shared_leaf<T, SAFE_MATH> &node) {
            return reinterpret_cast<T *> (kernel_arguments[node.get()]);
        }
    };
}

#endif /* cuda_context_h */
