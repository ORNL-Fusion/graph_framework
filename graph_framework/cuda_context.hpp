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
        std::vector<void *> result_buffers;
///  Index offset.
        size_t buffer_offset;
///  Buffer element size.
        size_t buffer_element_size;
///  Time offset.
        size_t time_offset;
///  Result buffer size;
        size_t result_size;

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
            cuModuleUnload(module);

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

            int compute_version;
            cuDeviceGetAttribute(&compute_version,
                                 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                 device);
            std::cout << "CUDA GPU info." << std::endl;
            std::cout << "  Major compute capability : " << compute_version << std::endl;

//  FIXME: Hardcoded for ada gpus for now.
            std::array<char *, 2> options({
                "--gpu-architecture=compute_90",
                "--std=c++20"
            });
            nvrtcCompileProgram(kernel_program, 2, options.data());

            char *mangled_kernel_name;
            nvrtcGetLoweredName(kernel_program,
                                kernel_name.c_str(),
                                const_cast<const char **> (&mangled_kernel_name));

            std::cout << "  Mangled Kernel Name      : " << mangled_kernel_name << std::endl;

            size_t ptx_size;
            nvrtcGetPTXSize(kernel_program, &ptx_size);

            char *ptx = static_cast<char *> (malloc(ptx_size));
            nvrtcGetPTX(kernel_program, ptx);

            cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL);
            cuModuleGetFunction(&function, module, mangled_kernel_name);

            free(ptx);

            buffer_element_size = sizeof(typename BACKEND::base);
            buffer_offset = ray_index*buffer_element_size;
            time_offset = 0;
            result_size = num_times*buffer_element_size;
            for (graph::shared_variable<BACKEND> &input : inputs) {
                const BACKEND backend = input->evaluate();

                CUdeviceptr ptr;
                cuMemAlloc(&ptr, backend.size()*buffer_element_size);
                cuMemcpyHtoD(ptr, &backend[0], backend.size()*buffer_element_size);
                buffers.push_back(ptr);

                void *hptr;
                cuMemHostAlloc(&hptr, result_size, 0);
                result_buffers.push_back(hptr);
            }
            for (graph::shared_leaf<BACKEND> &output : outputs) {
                const BACKEND backend = output->evaluate();

                CUdeviceptr ptr;
                cuMemAlloc(&ptr, backend.size()*buffer_element_size);
                cuMemcpyHtoD(ptr, &backend[0], backend.size()*buffer_element_size);
                buffers.push_back(ptr);

                void *hptr;
                cuMemHostAlloc(&hptr, result_size, 0);
                result_buffers.push_back(hptr);
            }

            int value;
            cuFuncGetAttribute(&value, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                               function);
            threads_per_group = value;
            thread_groups = num_rays/threads_per_group + (num_rays%threads_per_group ? 1 : 0);
            std::cout << "  Threads per group  : " << threads_per_group << std::endl;
            std::cout << "  Number of groups   : " << thread_groups << std::endl;
            std::cout << "  Total problem size : " << threads_per_group*thread_groups << std::endl;

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
                CUdeviceptr hdptr;
                cuMemHostGetDevicePointer(&hdptr, result_buffers[i], 0);
                cuMemcpyDtoDAsync(hdptr + time_offset,
                                  buffers[i] + buffer_offset,
                                  buffer_element_size, stream);
            }
        }

//------------------------------------------------------------------------------
///  @brief Perform a time step.
///
///  This calls dispatches a kernel instance to the command buffer and the 
///  commits the job. This method is asynchronous.
//------------------------------------------------------------------------------
        void step() {
            cuLaunchKernel(function, threads_per_group, 0, 0, thread_groups, 0, 0,
                           NULL, stream, reinterpret_cast<void**> (buffers.data()), NULL);
            encode_blit();
        }

//------------------------------------------------------------------------------
///  @brief Hold the current thread until the stream has completed.
//------------------------------------------------------------------------------
        void wait() {

            for (void *hptr : result_buffers) {
                CUdeviceptr hdptr;
                cuMemHostGetDevicePointer(&hdptr, hptr, 0);
                cuMemcpyDtoHAsync(hptr, hdptr,
                                  result_size, stream);
            }
            cuStreamSynchronize(stream);
        }

//------------------------------------------------------------------------------
///  @brief Print out the results.
///
///  @param[in] num_times Number of times to record.
//------------------------------------------------------------------------------
        template<class BACKEND>
        void print_results(const size_t num_times) {
            for (size_t i = 0, ie = num_times + 1; i < ie; i++) {
                std::cout << i << " ";
                for (void *buffer : result_buffers) {
                    std::cout << *(static_cast<typename BACKEND::base *> (buffer) + i) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    };
}

#endif /* cuda_context_h */
