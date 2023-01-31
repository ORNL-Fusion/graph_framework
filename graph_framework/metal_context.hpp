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
///  Buffer offsets.
        std::vector<NSUInteger> offsets;
///  Range
        NSRange range;
///  Max Buffer.
        id<MTLBuffer> result;
///  Compute pipeline discriptor.
        id<MTLComputePipelineState> state;
///  Compute pipeline discriptor.
        id<MTLComputePipelineState> max_state;
///  Metal command buffer.
        id<MTLCommandBuffer> command_buffer;
///  Metal library.
        id<MTLLibrary> library;
///  Number of thread groups.
        NSUInteger thread_groups;
///  Number of threads in a group.
        NSUInteger threads_per_group;
///  Buffer element size.
        size_t buffer_element_size;
///  Time offset.
        size_t time_offset;

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
///  @param[in] outputs       Output nodes of the kernel.
///  @param[in] num_rays      Number of rays to trace.
//------------------------------------------------------------------------------
        template<class BACKEND>
        void create_pipeline(const std::string kernel_source,
                             const std::string kernel_name,
                             graph::input_nodes<BACKEND> inputs,
                             graph::output_nodes<BACKEND> outputs,
                             const size_t num_rays) {
            @autoreleasepool {
                NSError *error;
                library = [device newLibraryWithSource:[NSString stringWithCString:kernel_source.c_str()
                                                                          encoding:NSUTF8StringEncoding]
                                               options:compile_options()
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

                buffer_element_size = sizeof(typename BACKEND::base);
                time_offset = 0;
                for (graph::shared_variable<BACKEND> &input : inputs) {
                    BACKEND buffer = input->evaluate();
                    buffers.push_back([device newBufferWithBytes:buffer.data()
                                                          length:buffer.size()*buffer_element_size
                                                         options:MTLResourceStorageModeManaged]);
                }
                for (graph::shared_leaf<BACKEND> &output : outputs) {
                    BACKEND buffer = output->evaluate();
                    buffers.push_back([device newBufferWithBytes:&buffer[0]
                                                          length:buffer.size()*buffer_element_size
                                                         options:MTLResourceStorageModeManaged]);
                }

                offsets.assign(buffers.size(), 0);
                range = NSMakeRange(0, buffers.size());

                threads_per_group = state.maxTotalThreadsPerThreadgroup;
                thread_groups = num_rays/threads_per_group + (num_rays%threads_per_group ? 1 : 0);
                std::cout << "Metal GPU info." << std::endl;
                std::cout << "  Threads per group  : " << threads_per_group << std::endl;
                std::cout << "  Number of groups   : " << thread_groups << std::endl;
                std::cout << "  Total problem size : " << threads_per_group*thread_groups << std::endl;
            }
        }

//------------------------------------------------------------------------------
///  @brief Create a max compute pipeline.
//------------------------------------------------------------------------------
        template<class BACKEND>
        void create_max_pipeline() {
            MTLComputePipelineDescriptor *compute = [MTLComputePipelineDescriptor new];
            compute.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
            compute.computeFunction = [library newFunctionWithName:@"max_reduction"];
            
            NSError *error;
            max_state = [device newComputePipelineStateWithDescriptor:compute
                                                              options:MTLPipelineOptionNone
                                                           reflection:NULL
                                                                error:&error];
            
            if (error) {
                NSLog(@"%@", error);
            }
            
            result = [device newBufferWithLength:sizeof(typename BACKEND::base)
                                         options:MTLResourceStorageModeManaged];
        }

//------------------------------------------------------------------------------
///  @brief Get the compile options.
//------------------------------------------------------------------------------
        MTLCompileOptions *compile_options() {
            MTLCompileOptions *options = [MTLCompileOptions new];
            options.fastMathEnabled = NO;
            return options;
        }

//------------------------------------------------------------------------------
///  @brief Perform a time step.
///
///  This calls dispatches a kernel instance to the command buffer and the commits
///  the job. This method is asyncronus.
//------------------------------------------------------------------------------
        void run() {
            @autoreleasepool {
                command_buffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                
                [encoder setComputePipelineState:state];
                [encoder setBuffers:buffers.data()
                            offsets:offsets.data()
                          withRange:range];

                [encoder dispatchThreadgroups:MTLSizeMake(thread_groups, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
                [encoder endEncoding];
                
                [command_buffer commit];
            }
        }

//------------------------------------------------------------------------------
///  @brief Compute the max reduction.
///
///  @returns The maximum value from the input buffer.
//------------------------------------------------------------------------------
        template<class BACKEND>
        typename BACKEND::base max_reduction() {
            run();
            command_buffer = [queue commandBuffer];
            
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
            
            [encoder setComputePipelineState:max_state];
            [encoder setBuffer:buffers.back() offset:0 atIndex:0];
            [encoder setBuffer:result offset:0 atIndex:1];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
            [encoder endEncoding];

            id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
            [blit synchronizeResource:result];
            [blit endEncoding];

            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            
            return static_cast<typename BACKEND::base *> ([result contents])[0];
        }
        
//------------------------------------------------------------------------------
///  @brief Hold the current thread until the current command buffer has complete.
//------------------------------------------------------------------------------
        void wait() {
            command_buffer = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
            for (id<MTLBuffer> buffer : buffers) {
                [blit synchronizeResource:buffer];
            }
            [blit endEncoding];
            
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
        }

//------------------------------------------------------------------------------
///  @brief Print out the results.
///
///  @param[in] index Particle index to print.
//------------------------------------------------------------------------------
        template<class BACKEND>
        void print_results(const size_t index) {
            wait();
            for (id<MTLBuffer> buffer : buffers) {
                const typename BACKEND::base *contents = static_cast<typename BACKEND::base *> ([buffer contents]);
                std::cout << contents[index] << " ";
            }
            std::cout << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Copy buffer contents.
///
///  @param[in]     source_index Index of the GPU buffer.
///  @param[in,out] destination  Host side buffer to copy to.
//------------------------------------------------------------------------------
        template<typename BASE>
        void copy_buffer(const size_t source_index,
                         BASE *destination) {
            command_buffer = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
            [blit synchronizeResource:buffers[source_index]];
            [blit endEncoding];
            
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            
            memcpy(destination,
                   [buffers[source_index] contents],
                   [buffers[source_index] length]);
        }
    };
}

#endif /* metal_context_h */
