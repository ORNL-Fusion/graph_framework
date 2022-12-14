//------------------------------------------------------------------------------
///  @file metal_context.hpp
///  @brief Base nodes of graph computation framework.
///
///  Defines context for metal gpu.
//------------------------------------------------------------------------------

#ifndef metal_context_h
#define metal_context_h

#import <vector>

#import <Metal/Metal.h>

#include "node.hpp"

namespace gpu {
//------------------------------------------------------------------------------
///  @brief Class representing a metal gpu context.
//------------------------------------------------------------------------------
    class metal_context {
    private:
///  The metal device.
        id<MTLDevice> device;
///  The metal command queue.
        id<MTLCommandQueue> queue;
///  Buffer objects.
        std::vector<id<MTLBuffer>> buffers;
///  Compute pipeline discriptor.
        id<MTLComputePipelineState> state;
///  Metal command buffer.
        id<MTLCommandBuffer> command_buffer;
///  Number of thread groups.
        NSUInteger thread_groups;
///  Number of threads in a group.
        NSUInteger threads_per_group;
        
    public:
//------------------------------------------------------------------------------
///  @brief Construct a metal context.
//------------------------------------------------------------------------------
        metal_context() :
        device(MTLCopyAllDevices().firstObject),
        queue([device newCommandQueue]) {}
        
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
            @autoreleasepool {
                MTLCompileOptions *options = [MTLCompileOptions new];
                options.fastMathEnabled = NO;
                
                NSError *error;
                id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithCString:kernel_source.c_str()
                                                                                         encoding:NSUTF8StringEncoding]
                                                              options:options
                                                                error:&error];
                
                if (error) {
                    NSLog(@"%@", error);
                }
                
                id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithCString:kernel_name.c_str()
                                                                                           encoding:NSUTF8StringEncoding]];
                
                MTLComputePipelineDescriptor *compute = [MTLComputePipelineDescriptor new];
                compute.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
                compute.computeFunction = function;
                
                state = [device newComputePipelineStateWithDescriptor:compute
                                                              options:MTLPipelineOptionNone
                                                           reflection:NULL
                                                                error:&error];
                
                if (error) {
                    NSLog(@"%@", error);
                }
                
                for (graph::shared_variable<BACKEND> &input : inputs) {
                    const BACKEND backend = input->evaluate();
                    buffers.push_back([device newBufferWithBytes:&backend[0]
                                                          length:backend.size()*sizeof(typename BACKEND::base)
                                                         options:MTLResourceStorageModeManaged]);
                }
                
                threads_per_group = state.maxTotalThreadsPerThreadgroup;
                thread_groups = num_rays/threads_per_group + (num_rays%threads_per_group ? 1 : 0);
                std::cout << "Metal GPU info." << std::endl;
                std::cout << "  Threads per group  : " << threads_per_group << std::endl;
                std::cout << "  Number of groups   : " << thread_groups << std::endl;
                std::cout << "  Total problem size : " << threads_per_group*thread_groups << std::endl;
            }
        }

//------------------------------------------------------------------------------
///  @brief Perform a time step.
///
///  This calls dispatches a kernel instance to the command buffer and the commits
///  the job. This method is asyncronus.
//------------------------------------------------------------------------------
        void step() {
            @autoreleasepool {
                command_buffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                
                [encoder setComputePipelineState:state];
                for (size_t i = 0, ie = buffers.size(); i < ie; i++) {
                    [encoder setBuffer:buffers[i]
                                offset:0
                               atIndex:i];
                }
                [encoder dispatchThreadgroups:MTLSizeMake(thread_groups, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
                [encoder endEncoding];
                [command_buffer commit];
            }
        }

//------------------------------------------------------------------------------
///  @brief Hold the current thread until the current command buffer has complete.
//------------------------------------------------------------------------------
        void wait() {
            [command_buffer waitUntilCompleted];
        }
    };
}

#endif /* metal_context_h */
