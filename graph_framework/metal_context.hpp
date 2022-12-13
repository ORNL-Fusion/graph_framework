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
        id<MTLCommandBuffer> command_buffer;
        NSUInteger thread_groups;
        NSUInteger threads_per_group;
        
    public:
//------------------------------------------------------------------------------
///  @brief Construct a metal context.
//------------------------------------------------------------------------------
        metal_context() :
        device(MTLCopyAllDevices().firstObject),
        queue([device newCommandQueue]) {}
        
//------------------------------------------------------------------------------
///  @brief Create a compute pipline.
//------------------------------------------------------------------------------
        template<class BACKEND>
        void create_pipline(const std::string kernel_source,
                            const std::string kernel_name,
                            std::vector<std::shared_ptr<graph::variable_node<BACKEND>>> inputs,
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
                
                for (std::shared_ptr<graph::variable_node<BACKEND>> &input : inputs) {
                    const BACKEND backend = input->evaluate();
                    buffers.push_back([device newBufferWithBytes:&backend[0]
                                                          length:backend.size()*sizeof(typename BACKEND::base)
                                                         options:MTLResourceStorageModeManaged]);
                }
                
                threads_per_group = state.maxTotalThreadsPerThreadgroup;
                thread_groups = num_rays/threads_per_group + (num_rays%threads_per_group ? 1 : 0);
                std::cout << threads_per_group << " " << thread_groups << std::endl;
            }
        }
        
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
        
        void wait() {
            [command_buffer waitUntilCompleted];
        }
    };
}

#endif /* metal_context_h */
