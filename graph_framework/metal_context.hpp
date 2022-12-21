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
///  Result buffers.
        std::vector<id<MTLBuffer>> result_buffers;
///  Index offset.
        size_t buffer_offset;
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
            @autoreleasepool {
                NSError *error;
                id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithCString:kernel_source.c_str()
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
                buffer_offset = ray_index*buffer_element_size;
                time_offset = 0;
                for (graph::shared_variable<BACKEND> &input : inputs) {
                    const BACKEND backend = input->evaluate();
                    buffers.push_back([device newBufferWithBytes:&backend[0]
                                                          length:backend.size()*buffer_element_size
                                                         options:MTLResourceStorageModeManaged]);
                    result_buffers.push_back([device newBufferWithLength:num_times*buffer_element_size
                                                                 options:MTLResourceStorageModeManaged]);
                }
                for (graph::shared_leaf<BACKEND> &output : outputs) {
                    const BACKEND backend = output->evaluate();
                    buffers.push_back([device newBufferWithBytes:&backend[0]
                                                          length:backend.size()*buffer_element_size
                                                         options:MTLResourceStorageModeManaged]);
                    result_buffers.push_back([device newBufferWithLength:num_times*buffer_element_size
                                                                 options:MTLResourceStorageModeManaged]);
                }
                
                threads_per_group = state.maxTotalThreadsPerThreadgroup;
                thread_groups = num_rays/threads_per_group + (num_rays%threads_per_group ? 1 : 0);
                std::cout << "Metal GPU info." << std::endl;
                std::cout << "  Threads per group  : " << threads_per_group << std::endl;
                std::cout << "  Number of groups   : " << thread_groups << std::endl;
                std::cout << "  Total problem size : " << threads_per_group*thread_groups << std::endl;
                
                command_buffer = [queue commandBuffer];
                encode_blit();
                [command_buffer commit];
            }
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
///  @brief Encode a blit command.
//------------------------------------------------------------------------------
        void encode_blit() {
            id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
            for (size_t i = 0, ie = buffers.size(); i < ie; i++) {
                [blit copyFromBuffer:buffers[i]
                        sourceOffset:buffer_offset
                            toBuffer:result_buffers[i]
                   destinationOffset:time_offset
                                size:buffer_element_size];
            }
            [blit endEncoding];

            time_offset += buffer_element_size;
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
                
                encode_blit();
                
                [command_buffer commit];
            }
        }

//------------------------------------------------------------------------------
///  @brief Hold the current thread until the current command buffer has complete.
//------------------------------------------------------------------------------
        void wait() {
            command_buffer = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
            for (size_t i = 0, ie = buffers.size(); i < ie; i++) {
                [blit synchronizeResource:result_buffers[i]];
            }
            [blit endEncoding];
            
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
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
                for (id<MTLBuffer> buffer : result_buffers) {
                    const typename BACKEND::base *contents = static_cast<typename BACKEND::base *> ([buffer contents]);
                    std::cout << contents[i] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    };
}

#endif /* metal_context_h */
