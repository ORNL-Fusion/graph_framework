//------------------------------------------------------------------------------
///  @file cuda_context.hpp
///  @brief Base nodes of graph computation framework.
///
///  Defines context for cuda gpu.
//------------------------------------------------------------------------------

#ifndef cuda_context_h
#define cuda_context_h

#import <vector>

#import <cuda.h>
#import <nvrtc.h>

#include "node.hpp"

namespace gpu {
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
        CUlibrary library;
///  The cuda kernel;
        CUkernel kernel;
///  Buffer objects.
        std::vector<CUdeviceptr> buffers;
///  Cuda stream.
        CUstream stream;
///  Number of thread groups.
        unsigned int thread_groups;
///  Number of threads in a group.
        unsigned int threads_per_group;
        
    public:
//------------------------------------------------------------------------------
///  @brief Cuda context constructor.
//------------------------------------------------------------------------------
        cuda_context() {
            cuDeviceGet(&device, 0);
            cuDevicePrimaryCtxRetain(&context, device);
            cuStreamCreate(&stream, CU_STREAM_DEFAULT);
        }

//------------------------------------------------------------------------------
///  @brief Cuda context destructor.
//------------------------------------------------------------------------------
        ~cuda_context() {
            cuLibraryUnload(library);
            
            for (CUdeviceptr &ptr : buffers) {
                cuMemFree(ptr);
            }
            
            cuStreamDestroy(stream);
            cuDevicePrimaryCtxRelease(device);
        }

//------------------------------------------------------------------------------
///  @brief Create a compute pipeline.
///
///  @param[in] kernel_source Source code buffer for the kernel.
///  @param[in] kernel_name   Name of the kernel for later reference.
///  @param[in] inputs        Input nodes of the kernel.
///  @param[in] num_rays      Number of rays to trace.
//------------------------------------------------------------------------------
        template<class BACKEND>
        void create_pipeline(const std::string kernel_source,
                             const std::string kernel_name,
                             graph::input_nodes<BACKEND> inputs,
                             const size_t num_rays) {
            nvrtcProgram kernel_program;
            nvrtcCreateProgram(&kernel_program,
                               kernel_source.c_str(),
                               NULL, 0, NULL, NULL);

            int compute_version;
            cuDeviceGetAttribute(&compute_version,
                                 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                 device);
            std::cout << "CUDA GPU info." << std::endl;
            std::cout << "  Major compute capability : " << compute_version << std::endl;
            
//  FIXME: Hardcoded for ada gpus for now.
            const char **options = {
                "--gpu-architecture=compute_90",
                "--std=c++20"
            };
            nvrtcCompileProgram(kernel_program, 2, options);
            
            char *mangled_kernel_name;
            nvrtcGetLoweredName(kernel_program,
                                kernel_name.c_str(),
                                &mangled_kernel_name);
            
            std::cout << "  Mangled Kernel Name      : " << mangled_kernel_name << std::endl;
            
            size_t ptx_size;
            nvrtcGetPTXSize(kernel_program, &ptx_size);
            
            char *ptx = malloc(ptx_size);
            nvrtcGetPTX(kernel_program, ptx);

            cuLibraryLoadData(&library, ptx, NULL, NULL, 0, NULL, NULL, 0);
            cuLibraryGetKernel(&kernel, library, mangled_kernel_name);
            
            for (graph::shared_variable<BACKEND> &input : inputs) {
                const BACKEND backend = input->evaluate();
                
                CUdeviceptr ptr;
                const size_t bytes = backend.size()*sizeof(typename BACKEND::base);
                cuMemAlloc(&ptr, bytes);
                cuMemcpuHtoD(ptr, &backend[0], bytes);
                buffers.push_back(ptr);
            }
            
            int value;
            cuKernelGetAttribute(&value, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                 kernel, device);
            threads_per_group = value;
            thread_groups = num_rays/threads_per_group + (num_rays%threads_per_group ? 1 : 0);
            std::cout << "  Threads per group  : " << threads_per_group << std::endl;
            std::cout << "  Number of groups   : " << thread_groups << std::endl;
            std::cout << "  Total problem size : " << threads_per_group*thread_groups << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Perform a time step.
///
///  This calls dispatches a kernel instance to the command buffer and the commits
///  the job. This method is asyncronus.
//------------------------------------------------------------------------------
        void step() {
            cuLaunchKernel(kernel, threads_per_group, 0, 0, thread_groups, 0, 0,
                           NULL, stream, buffers.data(), NULL);
        }

//------------------------------------------------------------------------------
///  @brief Hold the current thread until the current command buffer has complete.
//------------------------------------------------------------------------------
        void wait() {
            cuStreamSynchronize(stream);
        }
    };
}

#endif /* cuda_context_h */
