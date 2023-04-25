//------------------------------------------------------------------------------
///  @file jit.hpp
///  @brief Class to just in time compile a kernel.
///
///  Just in time compiles kernels.
//------------------------------------------------------------------------------

#ifndef jit_h
#define jit_h

#include <chrono>

#ifdef USE_METAL
#include "metal_context.hpp"
#elif defined(USE_CUDA)
#include "cuda_context.hpp"
#endif
#include "cpu_context.hpp"

#include "timing.hpp"

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
//------------------------------------------------------------------------------
    template<typename T>
    class context {
    private:
///  String stream to build the kernel source.
        std::stringstream source_buffer;
///  Nodes that have been jitted.
        register_map registers;
///  Kernel names.
        std::vector<std::string> kernel_names;

///  Type for the GPU context.
        using gpu_context_type = typename std::conditional<use_gpu<T> (),
#ifdef USE_CUDA
                                                           gpu::cuda_context<T>,
#elif defined(USE_METAL)
                                                           gpu::metal_context<T>,
#else
                                                           gpu::cpu_context<T>,
#endif
                                                           gpu::cpu_context<T>>::type;

///  GPU Context.
        gpu_context_type gpu_context;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a jit context object.
//------------------------------------------------------------------------------
        context() {
            source_buffer << std::setprecision(jit::max_digits10<T> ());
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
                        graph::input_nodes<T> inputs,
                        graph::output_nodes<T> outputs,
                        graph::map_nodes<T> setters) {
            kernel_names.push_back(name);
            
            const size_t size = inputs[0]->size();

            gpu_context.create_kernel_prefix(source_buffer,
                                             name, inputs, outputs, size,
                                             registers);

            for (auto &[out, in] : setters) {
                out->compile(source_buffer, registers);
            }
            for (auto &out : outputs) {
                out->compile(source_buffer, registers);
            }

            gpu_context.create_kernel_postfix(source_buffer, outputs,
                                              setters, registers);
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
                                                     graph::input_nodes<T> inputs,
                                                     graph::output_nodes<T> outputs,
                                                     const size_t num_rays) {
            return gpu_context.create_kernel_call(kernel_name, inputs, outputs,
                                                  num_rays);
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
            return gpu_context.create_max_call(argument, run);
        }

//------------------------------------------------------------------------------
///  @brief Print output.
///
///  @params[in] index Particle index to print.
//------------------------------------------------------------------------------
        void print(const size_t index) {
            gpu_context.print_results(index);
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
        void copy_to_device(graph::shared_leaf<T> &node,
                            T *source) {
            gpu_context.copy_to_device(node, source);
        }

//------------------------------------------------------------------------------
///  @brief Copy contexts of buffer to host.
///
///  @params[in]     node        Node to copy buffer from.
///  @params[in,out] destination Host side buffer to copy to.
//------------------------------------------------------------------------------
        void copy_to_host(graph::shared_leaf<T> &node,
                          T *destination) {
            gpu_context.copy_to_host(node, destination);
        }
    };
}

#endif /* jit_h */
