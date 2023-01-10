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
///  The cuda kernel;
        CUfunction function;
///  Buffer objects.
        std::vector<CUdeviceptr> buffers;
///  Cuda stream.
        CUstream stream;
///  Number of thread groups.
        unsigned int thread_groups;
///  Number of threads in a group.
        unsigned int threads_per_group;
///  Result buffers.
        std::vector<CUdeviceptr> result_buffers;
///  Index offset.
        size_t buffer_offset;
///  Buffer element size.
        size_t buffer_element_size;
///  Time offset.
        size_t time_offset;
///  Result buffer size;
        size_t result_size;
///  Kernel arguments.
        std::vector<void *> kernel_arguments;

//------------------------------------------------------------------------------
///  @brief  Check Results of cuda functions.
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
        cuda_context() {
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
            for (CUdeviceptr &ptr : result_buffers) {
                check_error(cuMemFree(ptr), "cuMemFree");
            }

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
///  @param[in] num_times     Number of times to record.
///  @param[in] ray_index     Index of the ray to save.
//------------------------------------------------------------------------------
        template<class BACKEND>
        void create_pipeline(const std::string kernel_source,
                             const std::string kernel_name,
                             graph::input_nodes<BACKEND> inputs,
                             graph::output_nodes<BACKEND> outputs,
                             const size_t num_rays,
                             const size_t num_times,
                             const size_t ray_index) {
            nvrtcProgram kernel_program;
            nvrtcCreateProgram(&kernel_program,
                               kernel_source.c_str(),
                               NULL, 0, NULL, NULL);

            nvrtcAddNameExpression(kernel_program, kernel_name.c_str());

            int compute_version;
            check_error(cuDeviceGetAttribute(&compute_version,
                                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                             device), "cuDeviceGetAttribute");
            std::cout << "CUDA GPU info." << std::endl;
            std::cout << "  Major compute capability : " << compute_version << std::endl;

            check_error(cuDeviceGetAttribute(&compute_version,
                                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                             device), "cuDeviceGetAttribute");

            std::cout << "  Minor compute capability : " << compute_version << std::endl;

            char device_name[100];
            check_error(cuDeviceGetName(device_name, 100, device), "cuDeviceGetName");
            std::cout << "  Device name              : " << device_name << std::endl;

//  FIXME: Hardcoded for ada gpus for now.
            std::array<const char *, 2> options({
                "--gpu-architecture=compute_80",
                "--std=c++17"
            });

            if (nvrtcCompileProgram(kernel_program, 2, options.data())) {
                size_t log_size;
                nvrtcGetProgramLogSize(kernel_program, &log_size);

                char *log = static_cast<char *> (malloc(log_size));
                nvrtcGetProgramLog(kernel_program, log);
                std::cout << log << std::endl;
                free(log);
            }

            const char *mangled_kernel_name;
            nvrtcGetLoweredName(kernel_program,
                                kernel_name.c_str(),
                                &mangled_kernel_name);

            std::cout << "  Mangled Kernel Name      : " << mangled_kernel_name << std::endl;

            check_error(cuDeviceGetAttribute(&compute_version,
                                             CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
                                             device), "cuDeviceGetAttribute");
            std::cout << "  Managed Memory           : " << compute_version << std::endl;

            size_t ptx_size;
            nvrtcGetPTXSize(kernel_program, &ptx_size);

            char *ptx = static_cast<char *> (malloc(ptx_size));
            nvrtcGetPTX(kernel_program, ptx);

            check_error(cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL), "cuModuleLoadDataEx");
            check_error(cuModuleGetFunction(&function, module, mangled_kernel_name), "cuModuleGetFunction");

            free(ptx);

            buffers.resize(inputs.size() + outputs.size());
            result_buffers.resize(inputs.size() + outputs.size());

            buffer_element_size = sizeof(typename BACKEND::base);
            buffer_offset = ray_index;
            time_offset = 0;
            result_size = num_times*buffer_element_size;
	    for (size_t i = 0, ie = inputs.size(); i < ie; i++) {
                const BACKEND backend = inputs[i]->evaluate();

                check_error(cuMemAlloc(&buffers[i], backend.size()*buffer_element_size), "cuMemAlloc");
                check_error(cuMemcpyHtoD(buffers[i], &backend[0], backend.size()*buffer_element_size), "cuMemcpyHtoD");
                kernel_arguments.push_back(reinterpret_cast<void *> (&buffers[i]));

                check_error(cuMemAllocManaged(&result_buffers[i], result_size, CU_MEM_ATTACH_GLOBAL), "cuMemAllocManaged");
            }
	    for	(size_t i = inputs.size(), ie = buffers.size(), j = 0; i < ie; i++, j++)	{
                const BACKEND backend = outputs[j]->evaluate();

                check_error(cuMemAlloc(&buffers[i], backend.size()*buffer_element_size), "cuMemAlloc");
                check_error(cuMemcpyHtoD(buffers[i], &backend[0], backend.size()*buffer_element_size), "cuMemcpyHtoD");
                kernel_arguments.push_back(reinterpret_cast<void *> (&buffers[i]));

                check_error(cuMemAllocManaged(&result_buffers[i], result_size, CU_MEM_ATTACH_GLOBAL), "cuMemAllocManaged");
            }

            int value;
            check_error(cuFuncGetAttribute(&value, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                           function), "cuFuncGetAttribute");
            threads_per_group = value;
            thread_groups = num_rays/threads_per_group + (num_rays%threads_per_group ? 1 : 0);
            std::cout << "  Threads per group        : " << threads_per_group << std::endl;
            std::cout << "  Number of groups         : " << thread_groups << std::endl;
            std::cout << "  Total problem size       : " << threads_per_group*thread_groups << std::endl;

            encode_blit();
        }

//------------------------------------------------------------------------------
///  @brief  Encode a blit command to the stream.
///
///  blit is the metal terminology for a memcopy operation added to the command
///  stream. Don't know what the cuda term is.
//------------------------------------------------------------------------------
        void encode_blit() {
            for (size_t i = 0, ie = buffers.size(); i < ie; i++) {
                check_error_async(cuMemcpyDtoDAsync(result_buffers[i] + time_offset,
                                                    buffers[i] + buffer_offset,
                                                    buffer_element_size, stream),
                                  "check_error_async");
            }

            time_offset += buffer_element_size;
        }

//------------------------------------------------------------------------------
///  @brief Perform a time step.
///
///  This calls dispatches a kernel instance to the command buffer and the 
///  commits the job. This method is asynchronous.
//------------------------------------------------------------------------------
        void step() {
            check_error_async(cuLaunchKernel(function, thread_groups, 1, 1,
                                             threads_per_group, 1, 1, 0, stream,
                                             kernel_arguments.data(), NULL),
                              "cuLaunchKernel");
            encode_blit();
        }

//------------------------------------------------------------------------------
///  @brief Hold the current thread until the stream has completed.
//------------------------------------------------------------------------------
        void wait() {
            check_error_async(cuStreamSynchronize(stream), "cuStreamSynchronize");
        }

//------------------------------------------------------------------------------
///  @brief Print out the results.
///
///  @param[in] num_times Number of times to record.
//------------------------------------------------------------------------------
        template<class BACKEND>
        void print_results(const size_t num_times) {
            check_error(cuCtxSynchronize(), "cuCtxSynchronize");
            for (size_t i = 0, ie = num_times + 1; i < ie; i++) {
                std::cout << i << " ";
                for (CUdeviceptr &buffer : result_buffers) {
                    std::cout << reinterpret_cast<typename BACKEND::base *> (buffer)[i] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    };
}

#endif /* cuda_context_h */
