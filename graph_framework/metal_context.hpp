//------------------------------------------------------------------------------
///  @file metal_context.hpp
///  @brief Metal context for metal based gpus.
///
///  Defines context for metal gpu.
//------------------------------------------------------------------------------

#ifndef metal_context_h
#define metal_context_h

#include <vector>
#include <map>

#import <Metal/Metal.h>

#include "node.hpp"

namespace gpu {
//------------------------------------------------------------------------------
///  @brief Class representing a metal gpu context.
//------------------------------------------------------------------------------
    template<typename T>
    class metal_context {
    private:
///  The metal device.
        id<MTLDevice> device;
///  The metal command queue.
        id<MTLCommandQueue> queue;
///  Argument map.
        std::map<graph::leaf_node<T> *, id<MTLBuffer>> kernel_arguments;
///  Max Buffer.
        id<MTLBuffer> result;
///  Metal command buffer.
        id<MTLCommandBuffer> command_buffer;
///  Metal library.
        id<MTLLibrary> library;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a metal context.
//------------------------------------------------------------------------------
        metal_context() :
        device(MTLCopyAllDevices().firstObject),
        queue([device newCommandQueue]) {}

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
            NSError *error;
            library = [device newLibraryWithSource:[NSString stringWithCString:kernel_source.c_str()
                                                                      encoding:NSUTF8StringEncoding]
                                           options:compile_options()
                                             error:&error];
            
            if (error) {
                NSLog(@"%@", error);
            }
            
            std::cout << "Metal GPU info." << std::endl;
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
                                                     graph::input_nodes<T> inputs,
                                                     graph::output_nodes<T> outputs,
                                                     const size_t num_rays) {
            NSError *error;

            id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithCString:kernel_name.c_str()
                                                                                       encoding:NSUTF8StringEncoding]];

            MTLComputePipelineDescriptor *compute = [MTLComputePipelineDescriptor new];
            compute.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
            compute.computeFunction = function;

            id<MTLComputePipelineState> state = [device newComputePipelineStateWithDescriptor:compute
                                                                                      options:MTLPipelineOptionNone
                                                                                   reflection:NULL
                                                                                        error:&error];

            if (error) {
                NSLog(@"%@", error);
            }

            std::vector<id<MTLBuffer>> buffers;

            const size_t buffer_element_size = sizeof(T);
            for (graph::shared_variable<T> &input : inputs) {
                if (!kernel_arguments.contains(input.get())) {
                    backend::buffer<T> buffer = input->evaluate();
                    kernel_arguments[input.get()] = [device newBufferWithBytes:buffer.data()
                                                                        length:buffer.size()*buffer_element_size
                                                                       options:MTLResourceStorageModeManaged];
                }
                buffers.push_back(kernel_arguments[input.get()]);
            }
            for (graph::shared_leaf<T> &output : outputs) {
                if (!kernel_arguments.contains(output.get())) {
                    kernel_arguments[output.get()] = [device newBufferWithLength:[buffers.back() length]
                                                                         options:MTLResourceStorageModeManaged];
                }
                buffers.push_back(kernel_arguments[output.get()]);
            }

            std::vector<NSUInteger> offsets(buffers.size(), 0);
            NSRange range = NSMakeRange(0, buffers.size());

            NSUInteger threads_per_group = state.maxTotalThreadsPerThreadgroup;
            NSUInteger thread_groups = num_rays/threads_per_group + (num_rays%threads_per_group ? 1 : 0);

            std::cout << "  Kernel name : " << kernel_name << std::endl;
            std::cout << "    Threads per group  : " << threads_per_group << std::endl;
            std::cout << "    Number of groups   : " << thread_groups << std::endl;
            std::cout << "    Total problem size : " << threads_per_group*thread_groups << std::endl;
            
            return [this, state, buffers, offsets, range, thread_groups, threads_per_group] () mutable {
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
            };
        }

//------------------------------------------------------------------------------
///  @brief Create a max compute kernel calling function.
///
///  @params[in] argument Node to reduce.
///  @params[in] run      Function to run before reduction.
///  @returns A lambda function to run the kernel.
//------------------------------------------------------------------------------
        std::function<T(void)> create_max_call(graph::shared_leaf<T> &argument,
                                               std::function<void(void)> run) {
            MTLComputePipelineDescriptor *compute = [MTLComputePipelineDescriptor new];
            compute.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
            compute.computeFunction = [library newFunctionWithName:@"max_reduction"];

            NSError *error;
            id<MTLComputePipelineState> max_state = [device newComputePipelineStateWithDescriptor:compute
                                                                                          options:MTLPipelineOptionNone
                                                                                       reflection:NULL
                                                                                            error:&error];

            if (error) {
                NSLog(@"%@", error);
            }

            id<MTLBuffer> result = [device newBufferWithLength:sizeof(T)
                                                       options:MTLResourceStorageModeManaged];
            
            id<MTLBuffer> buffer = kernel_arguments[argument.get()];
            
            return [this, run, buffer, result, max_state] () mutable {
                run();
                command_buffer = [queue commandBuffer];

                id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];

                [encoder setComputePipelineState:max_state];
                [encoder setBuffer:buffer offset:0 atIndex:0];
                [encoder setBuffer:result offset:0 atIndex:1];
                [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
                [encoder endEncoding];

                id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
                [blit synchronizeResource:result];
                [blit endEncoding];

                [command_buffer commit];
                [command_buffer waitUntilCompleted];

                return static_cast<T *> ([result contents])[0];
            };
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
///  @brief Hold the current thread until the current command buffer has complete.
//------------------------------------------------------------------------------
        void wait() {
            command_buffer = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
            for (const auto &[key, value] : kernel_arguments) {
                [blit synchronizeResource:value];
            }
            [blit endEncoding];

            [command_buffer commit];
            [command_buffer waitUntilCompleted];
        }

//------------------------------------------------------------------------------
///  @brief Print out the results.
///
///  @params[in] index Particle index to print.
//------------------------------------------------------------------------------
        void print_results(const size_t index) {
            wait();
            for (const auto &[key, value] : kernel_arguments) {
                const T *contents = static_cast<T *> ([value contents]);
                std::cout << contents[index] << " ";
            }
            std::cout << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Copy buffer contents to the device.
///
///  @params[in] node   Not to copy buffer to.
///  @params[in] source Host side buffer to copy from.
//------------------------------------------------------------------------------
        void copy_to_device(graph::shared_leaf<T> node,
                            T *source) {
            const size_t size = [kernel_arguments[node.get()] length];
            memcpy([kernel_arguments[node.get()] contents],
                   source, size);
            [kernel_arguments[node.get()] didModifyRange:NSMakeRange(0, size)];
        }

//------------------------------------------------------------------------------
///  @brief Copy buffer contents to host.
///
///  @params[in]     node        Node to copy buffer from.
///  @params[in,out] destination Host side buffer to copy to.
//------------------------------------------------------------------------------
        void copy_to_host(graph::shared_leaf<T> node,
                          T *destination) {
            command_buffer = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
            [blit synchronizeResource:kernel_arguments[node.get()]];
            [blit endEncoding];

            [command_buffer commit];
            [command_buffer waitUntilCompleted];

            memcpy(destination,
                   [kernel_arguments[node.get()] contents],
                   [kernel_arguments[node.get()] length]);
        }

//------------------------------------------------------------------------------
///  @brief Create the source header.
///
///  @params[in,out] source_buffer Source buffer stream.
//------------------------------------------------------------------------------
        void create_header(std::stringstream &source_buffer) {
            source_buffer << "#include <metal_stdlib>" << std::endl;
            source_buffer << "#include <metal_simdgroup>" << std::endl;
            source_buffer << "using namespace metal;" << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Create kernel prefix.
///
///  @params[in,out] source_buffer Source buffer stream.
///  @params[in]     name          Name to call the kernel.
///  @params[in]     inputs        Input variables of the kernel.
///  @params[in]     outputs       Output nodes of the graph to compute.
///  @params[in]     size          Size of the input buffer.
///  @params[in,out] registers     Map of used registers.
//------------------------------------------------------------------------------
        void create_kernel_prefix(std::stringstream &source_buffer,
                                  const std::string name,
                                  graph::input_nodes<T> &inputs,
                                  graph::output_nodes<T> &outputs,
                                  const size_t size,
                                  jit::register_map &registers) {
            source_buffer << std::endl;
            source_buffer << "kernel void " << name << "(" << std::endl;
            
            for (size_t i = 0, ie = inputs.size(); i < ie; i++) {
                source_buffer << "    device float *"
                              << jit::to_string('v', inputs[i].get())
                              << " [[buffer(" << i << ")]]," << std::endl;
            }
            
            for (size_t i = 0, ie = outputs.size(); i < ie; i++) {
                source_buffer << "    device float *"
                              << jit::to_string('o', outputs[i].get())
                              << " [[buffer(" << i + inputs.size() << ")]],"
                              << std::endl;
            }
            
            source_buffer << "    uint index [[thread_position_in_grid]]) {" << std::endl;
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
        void create_kernel_postfix(std::stringstream &source_buffer,
                                   graph::output_nodes<T> &outputs,
                                   graph::map_nodes<T> &setters,
                                   jit::register_map &registers) {
            for (auto &[out, in] : setters) {
                graph::shared_leaf<T> a = out->compile(source_buffer, registers);
                source_buffer << "        " << jit::to_string('v',  in.get())
                              << "[index] = " << registers[a.get()] << ";"
                              << std::endl;
            }
            
            for (auto &out : outputs) {
                graph::shared_leaf<T> a = out->compile(source_buffer, registers);
                source_buffer << "        " << jit::to_string('o',  out.get())
                              << "[index] = " << registers[a.get()] << ";"
                              << std::endl;
            }
    
            source_buffer << "    }" << std::endl << "}" << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Create reduction.
///
///  @params[in,out] source_buffer Source buffer stream.
///  @params[in]     size          Size of the input buffer.
//------------------------------------------------------------------------------
        void create_reduction(std::stringstream &source_buffer,
                              const size_t size) {
            source_buffer << std::endl;
            source_buffer << "kernel void max_reduction(" << std::endl;
            source_buffer << "    device float *input [[buffer(0)]]," << std::endl;
            source_buffer << "    device float *result [[buffer(1)]]," << std::endl;
            source_buffer << "    uint i [[thread_position_in_grid]]," << std::endl;
            source_buffer << "    uint j [[simdgroup_index_in_threadgroup]]," << std::endl;
            source_buffer << "    uint k [[thread_index_in_simdgroup]]) {" << std::endl;
            source_buffer << "    if (i < " << size << ") {" << std::endl;
            source_buffer << "        float sub_max = input[i];" << std::endl;
            source_buffer << "        for (size_t index = i + 1024; index < " << size <<"; index += 1024) {" << std::endl;
            source_buffer << "            sub_max = max(sub_max, input[index]);" << std::endl;
            source_buffer << "        }" << std::endl;
            source_buffer << "        threadgroup float thread_max[32];" << std::endl;
            source_buffer << "        thread_max[j] = simd_max(sub_max);" << std::endl;
            source_buffer << "        threadgroup_barrier(mem_flags::mem_threadgroup);" << std::endl;
            source_buffer << "        if (j == 0) {"  << std::endl;
            source_buffer << "            *result = simd_max(thread_max[k]);"  << std::endl;
            source_buffer << "        }"  << std::endl;
            source_buffer << "    }"  << std::endl;
            source_buffer << "}" << std::endl << std::endl;
        }
    };
}

#endif /* metal_context_h */
