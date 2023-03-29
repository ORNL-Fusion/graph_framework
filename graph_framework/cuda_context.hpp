//------------------------------------------------------------------------------
///  @file cuda_context.hpp
///  @brief Base nodes of graph computation framework.
///
///  Defines context for cuda gpu.
//------------------------------------------------------------------------------

#ifndef cuda_context_h
#define cuda_context_h

#import <vector>
#import <array>

#import <cuda.h>
#import <nvrtc.h>

#include "node.hpp"

namespace gpu {
//  Initialize the cuda driver.
    static const CUresult init = cuInit(0);

//------------------------------------------------------------------------------
///  @brief Class representing a cuda gpu context.
//------------------------------------------------------------------------------
    class cuda_context {
    private:
///  The cuda device.
        CUdevice device;
///  The cuda context.
        CUcontext context;
///  The cuda code library.
        CUmodule module;
///  The cuda kernel.
        CUfunction function;
///  The cuda max reduction kernel.
        CUfunction max_function;
///  The cuda library.
        nvrtcProgram kernel_program;
///  Buffer objects.
        std::vector<CUdeviceptr> buffers;
///  Result buffer.
        CUdeviceptr result_buffer;
///  Cuda stream.
        CUstream stream;
///  Number of thread groups.
        unsigned int thread_groups;
///  Number of threads in a group.
        unsigned int threads_per_group;
///  Kernel arguments.
        std::vector<void *> kernel_arguments;
///  Max kernel arguments.
        std::vector<void *> max_kernel_arguments;

//------------------------------------------------------------------------------
///  @brief  Check results of realtime compile.
///  @param[in] name   Name of the operation.
//------------------------------------------------------------------------------
        void check_nvrtc_error(nvrtcResult result,
                               const std::string &name) {
#ifndef NDEBUG
            std::cout << name << " " << result << " "
                      << nvrtcGetErrorString(result) << std::endl;
            assert(result == NVRTC_SUCCESS && "NVTRC Error");
#endif
        }

//------------------------------------------------------------------------------
///  @brief  Check results of cuda functions.
///
///  @param[in] result Result code of the operation.
///  @param[in] name   Name of the operation.
//------------------------------------------------------------------------------
        void check_error(CUresult result,
                         const std::string &name) {
#ifndef NDEBUG
            const char *error;
            cuGetErrorString(result, &error);
            std::cout << name << " "
                      << result << " " << error << std::endl;
            assert(result == CUDA_SUCCESS && "Cuda Error");
#endif
        }

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
//------------------------------------------------------------------------------
///  @brief Cuda context constructor.
//------------------------------------------------------------------------------
        cuda_context() : result_buffer(0) {
            check_error(cuDeviceGet(&device, 0), "cuDeviceGet");
            check_error(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
            check_error(cuCtxSetCurrent(context), "cuCtxSetCurrent");
            check_error(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate");
        }

//------------------------------------------------------------------------------
///  @brief Cuda context destructor.
//------------------------------------------------------------------------------
        ~cuda_context() {
            check_error(cuModuleUnload(module), "cuModuleUnload");

            for (CUdeviceptr &ptr : buffers) {
                check_error(cuMemFree(ptr), "cuMemFree");
            }

            check_nvrtc_error(nvrtcDestroyProgram(&kernel_program),
                              "nvrtcDestroyProgram");
            check_error(cuMemFree(result_buffer), "cuMemFree");

            check_error(cuStreamDestroy(stream), "cuStreamDestroy");
            check_error(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
        }

//------------------------------------------------------------------------------
///  @brief Create a compute pipeline.
///
///  @param[in] kernel_source Source code buffer for the kernel.
///  @param[in] kernel_name   Name of the kernel for later reference.
///  @param[in] inputs        Input nodes of the kernel.
///  @param[in] outputs       Output nodes of the kernel.
///  @param[in] num_rays      Number of rays to trace.
///  @param[in] add_reduction Optional argument to generate the reduction
///                           kernel.
//------------------------------------------------------------------------------
        template<typename T>
        void create_pipeline(const std::string kernel_source,
                             const std::string kernel_name,
                             graph::input_nodes<T> inputs,
                             graph::output_nodes<T> outputs,
                             const size_t num_rays,
                             const bool add_reduction=false) {
            check_nvrtc_error(nvrtcCreateProgram(&kernel_program,
                                                 kernel_source.c_str(),
                                                 NULL, 0, NULL, NULL),
                              "nvrtcCreateProgram");

            check_nvrtc_error(nvrtcAddNameExpression(kernel_program,
                                                     kernel_name.c_str()),
                              "nvrtcAddNameExpression");

            if (add_reduction) {
                check_nvrtc_error(nvrtcAddNameExpression(kernel_program,
                                                         "max_reduction"),
                                  "nvrtcAddNameExpression");
            }

            std::stringstream arch;
            int compute_version;
            check_error(cuDeviceGetAttribute(&compute_version,
                                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                             device), "cuDeviceGetAttribute");
            arch << "--gpu-architecture=compute_";
            arch << compute_version;
            std::cout << "CUDA GPU info." << std::endl;
            std::cout << "  Major compute capability : " << compute_version << std::endl;

            check_error(cuDeviceGetAttribute(&compute_version,
                                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                             device), "cuDeviceGetAttribute");
            arch << compute_version;
            std::cout << "  Minor compute capability : " << compute_version << std::endl;

            char device_name[100];
            check_error(cuDeviceGetName(device_name, 100, device), "cuDeviceGetName");
            std::cout << "  Device name              : " << device_name << std::endl;

            std::array<const char *, 3> options({
                arch.str().c_str(),
                "--std=c++17",
                "--include-path=" CUDA_INCLUDE
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
            }

            const char *mangled_kernel_name;
            check_nvrtc_error(nvrtcGetLoweredName(kernel_program,
                                                  kernel_name.c_str(),
                                                  &mangled_kernel_name),
                              "nvrtcGetLoweredName");

            std::cout << "  Mangled Kernel Name      : " << mangled_kernel_name << std::endl;

            check_error(cuDeviceGetAttribute(&compute_version,
                                             CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
                                             device), "cuDeviceGetAttribute");
            std::cout << "  Managed Memory           : " << compute_version << std::endl;

            size_t ptx_size;
            check_nvrtc_error(nvrtcGetPTXSize(kernel_program, &ptx_size),
                              "nvrtcGetPTXSize");

            char *ptx = static_cast<char *> (malloc(ptx_size));
            check_nvrtc_error(nvrtcGetPTX(kernel_program, ptx), "nvrtcGetPTX");

            check_error(cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL), "cuModuleLoadDataEx");
            check_error(cuModuleGetFunction(&function, module, mangled_kernel_name), "cuModuleGetFunction");

            free(ptx);

            buffers.resize(inputs.size() + outputs.size());

            const size_t buffer_element_size = sizeof(T);
            for (size_t i = 0, ie = inputs.size(); i < ie; i++) {
                const backend::cpu<T> backend = inputs[i]->evaluate();

                check_error(cuMemAllocManaged(&buffers[i], backend.size()*buffer_element_size,
                                              CU_MEM_ATTACH_GLOBAL),
                            "cuMemAllocManaged");
                check_error(cuMemcpyHtoD(buffers[i], &backend[0], backend.size()*buffer_element_size), 
                            "cuMemcpyHtoD");
                kernel_arguments.push_back(reinterpret_cast<void *> (&buffers[i]));
            }
            for (size_t i = inputs.size(), ie = buffers.size(), j = 0; i < ie; i++, j++) {
                const backend::cpu<T> backend = outputs[j]->evaluate();

                check_error(cuMemAllocManaged(&buffers[i], backend.size()*buffer_element_size,
                                              CU_MEM_ATTACH_GLOBAL), 
                            "cuMemAllocManaged");
                check_error(cuMemcpyHtoD(buffers[i], &backend[0], backend.size()*buffer_element_size), 
                                         "cuMemcpyHtoD");
                kernel_arguments.push_back(reinterpret_cast<void *> (&buffers[i]));
            }

            int value;
            check_error(cuFuncGetAttribute(&value, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                           function), "cuFuncGetAttribute");
            threads_per_group = value;
            thread_groups = num_rays/threads_per_group + (num_rays%threads_per_group ? 1 : 0);
            std::cout << "  Threads per group        : " << threads_per_group << std::endl;
            std::cout << "  Number of groups         : " << thread_groups << std::endl;
            std::cout << "  Total problem size       : " << threads_per_group*thread_groups << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Create a max compute pipeline.
//------------------------------------------------------------------------------
        template<typename T>
        void create_max_pipeline() {
            const char *mangled_kernel_name;
            check_nvrtc_error(nvrtcGetLoweredName(kernel_program,
                                                  "max_reduction",
                                                  &mangled_kernel_name),
                              "nvrtcGetLoweredName");

            std::cout << "  Mangled Kernel Name      : " << mangled_kernel_name << std::endl;

            check_error(cuMemAllocManaged(&result_buffer, sizeof(T),
                                          CU_MEM_ATTACH_GLOBAL),
                        "cuMemAllocManaged");

            max_kernel_arguments.push_back(reinterpret_cast<void *> (&buffers.back()));
            max_kernel_arguments.push_back(reinterpret_cast<void *> (&result_buffer));

            check_error(cuModuleGetFunction(&max_function, module, mangled_kernel_name),
                        "cuModuleGetFunction");
        }

//------------------------------------------------------------------------------
///  @brief Perform a time step.
///
///  This calls dispatches a kernel instance to the command buffer and the
///  commits the job. This method is asynchronous.
//------------------------------------------------------------------------------
        void run() {
            check_error_async(cuLaunchKernel(function, thread_groups, 1, 1,
                                             threads_per_group, 1, 1, 0, stream,
                                             kernel_arguments.data(), NULL),
                              "cuLaunchKernel");
        }

//------------------------------------------------------------------------------
///  @brief Compute the max reduction.
///
///  @returns The maximum value from the input buffer.
//------------------------------------------------------------------------------
        template<typename T>
        T max_reduction() {
            run();
            check_error_async(cuLaunchKernel(max_function, 1, 1, 1,
                                             threads_per_group, 1, 1, 0, stream,
                                             max_kernel_arguments.data(), NULL),
                              "cuLaunchKernel");
            wait();

            return reinterpret_cast<T *> (result_buffer)[0];
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
//------------------------------------------------------------------------------
        template<typename T>
        void print_results(const size_t index) {
            wait();
            for (CUdeviceptr &buffer : buffers) {
                std::cout << reinterpret_cast<T *> (buffer)[index] << " ";
            }
            std::cout << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Copy buffer contents.
///
///  @param[in]     source_index Index of the GPU buffer.
///  @param[in,out] destination  Host side buffer to copy to.
//------------------------------------------------------------------------------
        template<typename T>
        void copy_buffer(const size_t source_index,
                         T *destination) {
	    size_t size;
	    check_error(cuMemGetAddressRange(NULL, &size, buffers[source_index]), "cuMemGetAddressRange");
            check_error_async(cuMemcpyDtoHAsync(destination, buffers[source_index], size, stream), "cuMemcpyDtoHAsync");
        }
    };
}

#endif /* cuda_context_h */
