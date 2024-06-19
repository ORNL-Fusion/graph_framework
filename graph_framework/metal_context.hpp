//------------------------------------------------------------------------------
///  @file metal_context.hpp
///  @brief Metal context for metal based gpus.
///
///  Defines context for metal gpu.
//------------------------------------------------------------------------------

#ifndef metal_context_h
#define metal_context_h

#import <Metal/Metal.h>

#include "node.hpp"

namespace gpu {
//------------------------------------------------------------------------------
///  @brief Class representing a metal gpu context.
///
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<bool SAFE_MATH=false>
    class metal_context {
    private:
///  The metal device.
        id<MTLDevice> device;
///  The metal command queue.
        id<MTLCommandQueue> queue;
///  Argument map.
        std::map<graph::leaf_node<float, SAFE_MATH> *, id<MTLBuffer>> kernel_arguments;
///  Textures.
        std::map<void *, id<MTLTexture>> texture_arguments;
///  Max Buffer.
        id<MTLBuffer> result;
///  Metal command buffer.
        id<MTLCommandBuffer> command_buffer;
///  Metal library.
        id<MTLLibrary> library;
///  Buffer mutability discriptor.
        std::map<std::string, std::vector<MTLMutability>> bufferMutability;

    public:
///  Remaining constant memory in bytes. NOT USED.
        int remaining_const_memory;

//------------------------------------------------------------------------------
///  @brief Get the maximum number of concurrent instances.
///
///  @returns The maximum available concurrency.
//------------------------------------------------------------------------------
        static size_t max_concurrency() {
            return MTLCopyAllDevices().count;
        }

//------------------------------------------------------------------------------
///  @brief Device discription.
//------------------------------------------------------------------------------
        static std::string device_type() {
            return "Metal GPU";
        }

//------------------------------------------------------------------------------
///  @brief Construct a metal context.
///
///  @params[in] index Concurrent index.
//------------------------------------------------------------------------------
        metal_context(const size_t index) :
        device([MTLCopyAllDevices() objectAtIndex:index]),
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

            if (jit::verbose) {
                std::cout << "Metal GPU info." << std::endl;
            }
        }

//------------------------------------------------------------------------------
///  @brief Create a kernel calling function.
///
///  @params[in] kernel_name Name of the kernel for later reference.
///  @params[in] inputs      Input nodes of the kernel.
///  @params[in] outputs     Output nodes of the kernel.
///  @params[in] num_rays    Number of rays to trace.
///  @params[in] tex1d_list  List of 1D textures.
///  @params[in] tex2d_list  List of 1D textures.
///  @returns A lambda function to run the kernel.
//------------------------------------------------------------------------------
        std::function<void(void)> create_kernel_call(const std::string kernel_name,
                                                     graph::input_nodes<float, SAFE_MATH> inputs,
                                                     graph::output_nodes<float, SAFE_MATH> outputs,
                                                     const size_t num_rays,
                                                     const jit::texture1d_list &tex1d_list,
                                                     const jit::texture2d_list &tex2d_list) {
            NSError *error;

            id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithCString:kernel_name.c_str()
                                                                                       encoding:NSUTF8StringEncoding]];

            MTLComputePipelineDescriptor *compute = [MTLComputePipelineDescriptor new];
            compute.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
            compute.computeFunction = function;
            compute.maxTotalThreadsPerThreadgroup = 1024;
            for (size_t i = 0, ie = bufferMutability[kernel_name].size(); i < ie; i++) {
                compute.buffers[i].mutability = bufferMutability[kernel_name][i];
            }

            id<MTLComputePipelineState> state = [device newComputePipelineStateWithDescriptor:compute
                                                                                      options:MTLPipelineOptionNone
                                                                                   reflection:NULL
                                                                                        error:&error];

            if (error) {
                NSLog(@"%@", error);
            }

            std::vector<id<MTLBuffer>> buffers;

            const size_t buffer_element_size = sizeof(float);
            for (graph::shared_variable<float, SAFE_MATH> &input : inputs) {
                if (!kernel_arguments.contains(input.get())) {
                    backend::buffer<float> buffer = input->evaluate();
                    kernel_arguments[input.get()] = [device newBufferWithBytes:buffer.data()
                                                                        length:buffer.size()*buffer_element_size
                                                                       options:MTLResourceStorageModeManaged];
                }
                buffers.push_back(kernel_arguments[input.get()]);
            }
            for (graph::shared_leaf<float, SAFE_MATH> &output : outputs) {
                if (!kernel_arguments.contains(output.get())) {
                    kernel_arguments[output.get()] = [device newBufferWithLength:[buffers.back() length]
                                                                         options:MTLResourceStorageModeManaged];
                }
                buffers.push_back(kernel_arguments[output.get()]);
            }

            std::vector<id<MTLTexture>> textures;
            command_buffer = [queue commandBuffer];
            id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
            for (auto &[data, size] : tex1d_list) {
                if (!texture_arguments.contains(data)) {
                    MTLTextureDescriptor *discriptor = [MTLTextureDescriptor new];
                    discriptor.textureType = MTLTextureType1D;
                    discriptor.pixelFormat = MTLPixelFormatR32Float;
                    discriptor.width = size;
                    discriptor.storageMode = MTLStorageModeManaged;
                    discriptor.cpuCacheMode = MTLCPUCacheModeWriteCombined;
                    discriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
                    discriptor.usage = MTLTextureUsageShaderRead;
                    texture_arguments[data] = [device newTextureWithDescriptor:discriptor];
                    [texture_arguments[data] replaceRegion:MTLRegionMake1D(0, size)
                                               mipmapLevel:0
                                                 withBytes:reinterpret_cast<float *> (data)
                                               bytesPerRow:4*size];

                    [encoder optimizeContentsForGPUAccess:texture_arguments[data]];
                }
                textures.push_back(texture_arguments[data]);
            }
            for (auto &[data, size] : tex2d_list) {
                if (!texture_arguments.contains(data)) {
                    MTLTextureDescriptor *discriptor = [MTLTextureDescriptor new];
                    discriptor.textureType = MTLTextureType2D;
                    discriptor.pixelFormat = MTLPixelFormatR32Float;
                    discriptor.width = size[0];
                    discriptor.height = size[1];
                    discriptor.storageMode = MTLStorageModeManaged;
                    discriptor.cpuCacheMode = MTLCPUCacheModeWriteCombined;
                    discriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
                    discriptor.usage = MTLTextureUsageShaderRead;
                    texture_arguments[data] = [device newTextureWithDescriptor:discriptor];
                    [texture_arguments[data] replaceRegion:MTLRegionMake2D(0, 0, size[0], size[1])
                                               mipmapLevel:0
                                                 withBytes:reinterpret_cast<float *> (data)
                                               bytesPerRow:4*size[0]];

                    [encoder optimizeContentsForGPUAccess:texture_arguments[data]];
                }
                textures.push_back(texture_arguments[data]);
            }
            [encoder endEncoding];
            [command_buffer commit];

            std::vector<NSUInteger> offsets(buffers.size(), 0);
            NSRange range = NSMakeRange(0, buffers.size());
            NSRange tex_range = NSMakeRange(0, textures.size());

            NSUInteger threads_per_group = state.maxTotalThreadsPerThreadgroup;
            NSUInteger thread_width = state.threadExecutionWidth;
            NSUInteger thread_groups = num_rays/threads_per_group + (num_rays%threads_per_group ? 1 : 0);

            if (jit::verbose) {
                std::cout << "  Kernel name : " << kernel_name << std::endl;
                std::cout << "    Thread execution width : " << thread_width << std::endl;
                std::cout << "    Threads per group      : " << threads_per_group << std::endl;
                std::cout << "    Number of groups       : " << thread_groups << std::endl;
                std::cout << "    Total problem size     : " << threads_per_group*thread_groups << std::endl;
            }

            return [this, state, buffers, offsets, range, tex_range, thread_groups, threads_per_group, textures] () mutable {
                command_buffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];

                [encoder setComputePipelineState:state];
                [encoder setBuffers:buffers.data()
                            offsets:offsets.data()
                          withRange:range];
                [encoder setTextures:textures.data()
                           withRange:tex_range];

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
        std::function<float(void)> create_max_call(graph::shared_leaf<float, SAFE_MATH> &argument,
                                                   std::function<void(void)> run) {
            MTLComputePipelineDescriptor *compute = [MTLComputePipelineDescriptor new];
            compute.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
            compute.computeFunction = [library newFunctionWithName:@"max_reduction"];
            compute.maxTotalThreadsPerThreadgroup = 1024;
            compute.buffers[0].mutability = MTLMutabilityImmutable;

            NSError *error;
            id<MTLComputePipelineState> max_state = [device newComputePipelineStateWithDescriptor:compute
                                                                                          options:MTLPipelineOptionNone
                                                                                       reflection:NULL
                                                                                            error:&error];
            if (error) {
                NSLog(@"%@", error);
            }

            id<MTLBuffer> result = [device newBufferWithLength:sizeof(float)
                                                       options:MTLResourceStorageModeManaged];
            
            id<MTLBuffer> buffer = kernel_arguments[argument.get()];
            
            NSUInteger threads_per_group = max_state.maxTotalThreadsPerThreadgroup;
            NSUInteger thread_width = max_state.threadExecutionWidth;
            if (jit::verbose) {
                std::cout << "  Kernel name : max_reduction" << std::endl;
                std::cout << "    Thread execution width : " << thread_width << std::endl;
                std::cout << "    Threads per group      : " << threads_per_group << std::endl;
                std::cout << "    Number of groups       : " << 1 << std::endl;
                std::cout << "    Total problem size     : " << threads_per_group*1 << std::endl;
            }

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

                return static_cast<float *> ([result contents])[0];
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
///  @brief Hold the current thread until the command buffer has completed.
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
///  @params[in] nodes Nodes to output.
//------------------------------------------------------------------------------
        void print_results(const size_t index,
                           const graph::output_nodes<float, SAFE_MATH> &nodes) {
            wait();
            for (auto &out : nodes) {
                std::cout << static_cast<float *> ([kernel_arguments[out.get()] contents])[index] << " ";
            }
            std::cout << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Check the value.
///
///  @params[in] index Ray index to check value for.
///  @params[in] node  Node to check the value for.
///  @returns The value at the index.
//------------------------------------------------------------------------------
        float check_value(const size_t index,
                          const graph::shared_leaf<float, SAFE_MATH> &node) {
            wait();
            return static_cast<float *> ([kernel_arguments[node.get()] contents])[index];
        }

//------------------------------------------------------------------------------
///  @brief Copy buffer contents to the device.
///
///  @params[in] node   Not to copy buffer to.
///  @params[in] source Host side buffer to copy from.
//------------------------------------------------------------------------------
        void copy_to_device(graph::shared_leaf<float, SAFE_MATH> node,
                            float *source) {
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
        void copy_to_host(graph::shared_leaf<float, SAFE_MATH> node,
                          float *destination) {
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
        void create_header(std::ostringstream &source_buffer) {
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
///  @params[in]     is_constant   Flags if the input is read only.
///  @params[in,out] registers     Map of used registers.
///  @params[in]     usage         List of register usage count.
///  @params[in]     textures1d    List of 1D kernel textures.
///  @params[in]     textures2d    List of 2D kernel textures.
//------------------------------------------------------------------------------
        void create_kernel_prefix(std::ostringstream &source_buffer,
                                  const std::string name,
                                  graph::input_nodes<float, SAFE_MATH> &inputs,
                                  graph::output_nodes<float, SAFE_MATH> &outputs,
                                  const size_t size, 
                                  const std::vector<bool> &is_constant,
                                  jit::register_map &registers,
                                  const jit::register_usage &usage,
                                  jit::texture1d_list &textures1d,
                                  jit::texture2d_list &textures2d) {
            source_buffer << std::endl;
            source_buffer << "kernel void " << name << "(" << std::endl;

            bufferMutability[name] = std::vector<MTLMutability> ();

            for (size_t i = 0, ie = inputs.size(); i < ie; i++) {
                bufferMutability[name].push_back(is_constant[i] ? MTLMutabilityMutable : MTLMutabilityImmutable);
                source_buffer << "    " << (is_constant[i] ? "constant" : "device")
                              << " float *"
                              << jit::to_string('v', inputs[i].get())
                              << " [[buffer(" << i << ")]], // "
                              << inputs[i]->get_symbol()
#ifndef USE_INPUT_CACHE
                              << " used " << usage.at(inputs[i].get())
#endif
                              << std::endl;
            }
            for (size_t i = 0, ie = outputs.size(); i < ie; i++) {
                bufferMutability[name].push_back(MTLMutabilityMutable);
                source_buffer << "    device float *"
                              << jit::to_string('o', outputs[i].get())
                              << " [[buffer(" << i + inputs.size() << ")]],"
                              << std::endl;
            }
            for (size_t i = 0, ie = textures1d.size(); i < ie; i++) {
                source_buffer << "    const texture1d<float, access::read> "
                              << jit::to_string('a', textures1d[i].first)
                              << " [[texture(" << i << ")]],"
                              << std::endl;
            }
            for (size_t i = 0, ie = textures2d.size(); i < ie; i++) {
                source_buffer << "    const texture2d<float, access::read> "
                              << jit::to_string('a', textures2d[i].first)
                              << " [[texture(" << i + textures1d.size() << ")]],"
                              << std::endl;
            }

            source_buffer << "    uint index [[thread_position_in_grid]]) {" << std::endl;
            source_buffer << "    if (index < " << size << ") {" << std::endl;
            
            for (auto &input : inputs) {
#ifdef USE_INPUT_CACHE
                if (usage.at(input.get())) {
                    registers[input.get()] = jit::to_string('r', input.get());
                    source_buffer << "        const ";
                    jit::add_type<float> (source_buffer);
                    source_buffer << " " << registers[input.get()] << " = "
                                  << jit::to_string('v', input.get())
                                  << "[index]; // " << input->get_symbol()
                                  << " used " << usage.at(input.get()) << std::endl;
                }
#else
                registers[input.get()] = jit::to_string('v', input.get()) + "[index]";
#endif
            }
        }

//------------------------------------------------------------------------------
///  @brief Create kernel postfix.
///
///  @params[in,out] source_buffer Source buffer stream.
///  @params[in]     outputs       Output nodes of the graph to compute.
///  @params[in]     setters       Map outputs back to input values.
///  @params[in,out] registers     Map of used registers.
///  @params[in]     usage         List of register usage count.
//------------------------------------------------------------------------------
        void create_kernel_postfix(std::ostringstream &source_buffer,
                                   graph::output_nodes<float, SAFE_MATH> &outputs,
                                   graph::map_nodes<float, SAFE_MATH> &setters,
                                   jit::register_map &registers,
                                   const jit::register_usage &usage) {
            for (auto &[out, in] : setters) {
                graph::shared_leaf<float, SAFE_MATH> a = out->compile(source_buffer, 
                                                                      registers,
                                                                      usage);
                source_buffer << "        " << jit::to_string('v',  in.get())
                              << "[index] = ";
                if constexpr (SAFE_MATH) {
                    source_buffer << "isnan(" << registers[a.get()]
                                  << ") ? 0.0 : ";
                }
                source_buffer << registers[a.get()] << ";" << std::endl;
            }
            
            for (auto &out : outputs) {
                graph::shared_leaf<float, SAFE_MATH> a = out->compile(source_buffer,
                                                                      registers,
                                                                      usage);
                source_buffer << "        " << jit::to_string('o',  out.get())
                              << "[index] = "; 
                if constexpr (SAFE_MATH) {
                    source_buffer << "isnan(" << registers[a.get()]
                                  << ") ? 0.0 : ";
                }
                source_buffer << registers[a.get()] << ";" << std::endl;
            }
    
            source_buffer << "    }" << std::endl << "}" << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Create reduction.
///
///  @params[in,out] source_buffer Source buffer stream.
///  @params[in]     size          Size of the input buffer.
//------------------------------------------------------------------------------
        void create_reduction(std::ostringstream &source_buffer,
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

//------------------------------------------------------------------------------
///  @brief Get the buffer for a node.
///
///  @params[in] node Node to get the buffer for.
//------------------------------------------------------------------------------
        float *get_buffer(graph::shared_leaf<float, SAFE_MATH> &node) {
            return static_cast<float *> ([kernel_arguments[node.get()] contents]);
        }
    };
}

#endif /* metal_context_h */
