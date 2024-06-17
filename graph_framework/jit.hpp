//------------------------------------------------------------------------------
///  @file jit.hpp
///  @brief Class to just in time compile a kernel.
///
///  Just in time compiles kernels.
//------------------------------------------------------------------------------

#ifndef jit_h
#define jit_h

#include <algorithm>
#include <iterator>

#ifdef USE_METAL
#include "metal_context.hpp"
#elif defined(USE_CUDA)
#include "cuda_context.hpp"
#endif
#include "cpu_context.hpp"

#ifdef USE_METAL
#define START_GPU @autoreleasepool {
#define END_GPU }
#else
#define START_GPU
#define END_GPU
#endif

namespace jit {
//------------------------------------------------------------------------------
///  @brief Class for JIT compile of the GPU kernels.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<float_scalar T, bool SAFE_MATH=false>
    class context {
    private:
///  String stream to build the kernel source.
        std::ostringstream source_buffer;
///  Nodes that have been jitted.
        register_map registers;
///  Kernel names.
        std::vector<std::string> kernel_names;
///  Kernel textures.
        std::map<std::string, texture1d_list> kernel_1dtextures;
///  Kernel textures.
        std::map<std::string, texture2d_list> kernel_2dtextures;

///  Type for the GPU context.
        using gpu_context_type = typename std::conditional<use_gpu<T> (),
#ifdef USE_CUDA
                                                           gpu::cuda_context<T, SAFE_MATH>,
#elif defined(USE_METAL)
                                                           gpu::metal_context<SAFE_MATH>,
#else
                                                           gpu::cpu_context<T, SAFE_MATH>,
#endif
                                                           gpu::cpu_context<T, SAFE_MATH>>::type;

///  GPU Context.
        gpu_context_type gpu_context;

    public:
//------------------------------------------------------------------------------
///  @brief Get the maximum number of concurrent instances.
///
///  @returns The maximum available concurrency.
//------------------------------------------------------------------------------
        static size_t max_concurrency() {
            const size_t num = gpu_context_type::max_concurrency();
            std::cout << "Located " << num << " "
                      << gpu_context_type::device_type() << " device"
                      << (num == 1 ? "." : "s.")
                      << std::endl;
            return num;
        }

//------------------------------------------------------------------------------
///  @brief Construct a jit context object.
///
///  @params[in] index Concurrent index. Not used.
//------------------------------------------------------------------------------
        context(const size_t index) : gpu_context(index) {
            source_buffer << std::setprecision(max_digits10<T> ());
            gpu_context.create_header(source_buffer);
        }

//------------------------------------------------------------------------------
///  @brief Add a kernel.
///
///  Build the source code for a kernel graph.
///
///  @params[in] name    Name to call the kernel.
///  @params[in] inputs  Input variables of the kernel.
///  @params[in] outputs Output nodes of the graph to compute.
///  @params[in] setters Map outputs back to input values.
//------------------------------------------------------------------------------
        void add_kernel(const std::string name,
                        graph::input_nodes<T, SAFE_MATH> inputs,
                        graph::output_nodes<T, SAFE_MATH> outputs,
                        graph::map_nodes<T, SAFE_MATH> setters) {
            kernel_names.push_back(name);

            const size_t size = inputs[0]->size();

            std::vector<bool> is_constant(inputs.size(), true);
            visiter_map visited;
            register_usage usage;
            kernel_1dtextures[name] = texture1d_list();
            kernel_2dtextures[name] = texture2d_list();
            for (auto &[out, in] : setters) {
                auto found = std::distance(inputs.begin(),
                                           std::find(inputs.begin(),
                                                     inputs.end(), in));
                if (found < is_constant.size()) {
                    is_constant[found] = false;
                }
                out->compile_preamble(source_buffer, registers,
                                      visited, usage,
                                      kernel_1dtextures[name],
                                      kernel_2dtextures[name]);
            }
            for (auto &out : outputs) {
                out->compile_preamble(source_buffer, registers,
                                      visited, usage,
                                      kernel_1dtextures[name],
                                      kernel_2dtextures[name]);
            }

            for (auto &in : inputs) {
                if (usage.find(in.get()) == usage.end()) {
                    usage[in.get()] == 0;
                }
            }

            gpu_context.create_kernel_prefix(source_buffer,
                                             name, inputs, outputs, 
                                             size, is_constant,
                                             registers, usage,
                                             kernel_1dtextures[name],
                                             kernel_2dtextures[name]);

            for (auto &[out, in] : setters) {
                out->compile(source_buffer, registers, usage);
            }
            for (auto &out : outputs) {
                out->compile(source_buffer, registers, usage);
            }

            gpu_context.create_kernel_postfix(source_buffer, outputs,
                                              setters, registers, usage);

//  Delete the registers so that they can be used again in other kernels.
            std::vector<void *> removed_elements;
            for (auto &[key, value] : registers) {
                if (value[0] == 'r') {
                    removed_elements.push_back(key);
                }
            }

            for (auto &key : removed_elements) {
                registers.erase(key);
            }
        }

//------------------------------------------------------------------------------
///  @brief Add max reduction kernel.
///
///  @params[in] size Size of the input buffer.
//------------------------------------------------------------------------------
        void add_max_reduction(const size_t size) {
            gpu_context.create_reduction(source_buffer, size);
        }

//------------------------------------------------------------------------------
///  @brief Print the kernel source.
//------------------------------------------------------------------------------
        void print_source() {
            std::cout << std::endl << source_buffer.str() << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Compile the kernel.
///
///  @params[in] add_reduction Optional argument to generate the reduction
///                            kernel.
//------------------------------------------------------------------------------
        void compile(const bool add_reduction=false) {
            gpu_context.compile(source_buffer.str(),
                                kernel_names,
                                add_reduction);
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
                                                     graph::input_nodes<T, SAFE_MATH> inputs,
                                                     graph::output_nodes<T, SAFE_MATH> outputs,
                                                     const size_t num_rays) {
            return gpu_context.create_kernel_call(kernel_name, inputs, outputs, num_rays,
                                                  kernel_1dtextures[kernel_name],
                                                  kernel_2dtextures[kernel_name]);
        }

//------------------------------------------------------------------------------
///  @brief Create a max compute kernel calling function.
///
///  @params[in] argument Node to reduce.
///  @params[in] run      Function to run before reduction.
///  @returns A lambda function to run the kernel.
//------------------------------------------------------------------------------
        std::function<T(void)> create_max_call(graph::shared_leaf<T, SAFE_MATH> &argument,
                                               std::function<void(void)> run) {
            return gpu_context.create_max_call(argument, run);
        }

//------------------------------------------------------------------------------
///  @brief Print output.
///
///  @params[in] index Particle index to print.
///  @params[in] nodes Nodes to output.
//------------------------------------------------------------------------------
        void print(const size_t index,
                   const graph::output_nodes<T, SAFE_MATH> &nodes) {
            gpu_context.print_results(index, nodes);
        }

//------------------------------------------------------------------------------
///  @brief Check the value.
///
///  @params[in] index Ray index to check value for.
///  @params[in] node  Node to check the value for.
///  @returns The value at the index.
//------------------------------------------------------------------------------
        T check_value(const size_t index,
                      const graph::shared_leaf<T, SAFE_MATH> &node) {
            return gpu_context.check_value(index, node);
        }

//------------------------------------------------------------------------------
///  @brief Wait for kernel to finish.
//------------------------------------------------------------------------------
        void wait() {
            gpu_context.wait();
        }

//------------------------------------------------------------------------------
///  @brief Copy contexts of buffer to device.
///
///  @params[in] node   Not to copy buffer to.
///  @params[in] source Host side buffer to copy from.
//------------------------------------------------------------------------------
        void copy_to_device(graph::shared_leaf<T, SAFE_MATH> &node,
                            T *source) {
            gpu_context.copy_to_device(node, source);
        }

//------------------------------------------------------------------------------
///  @brief Copy contexts of buffer to host.
///
///  @params[in]     node        Node to copy buffer from.
///  @params[in,out] destination Host side buffer to copy to.
//------------------------------------------------------------------------------
        void copy_to_host(graph::shared_leaf<T, SAFE_MATH> &node,
                          T *destination) {
            gpu_context.copy_to_host(node, destination);
        }

//------------------------------------------------------------------------------
///  @brief Get buffer frim the gpu\_context.
///
///  @params[in] node Node to get the gpu buffer for.
//------------------------------------------------------------------------------
        T *get_buffer(graph::shared_leaf<T, SAFE_MATH> &node) {
            return gpu_context.get_buffer(node);
        }
    };
}

#endif /* jit_h */
