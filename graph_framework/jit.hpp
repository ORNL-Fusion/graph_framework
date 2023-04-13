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
        register_map<graph::leaf_node<T>> registers;

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
///  @params[in] name  Name to call the kernel.
///  @params[in] input Graph node to reduce.
//------------------------------------------------------------------------------
        void add_max_reduction(const std::string name,
                               graph::shared_variable<T> input) {
            gpu_context.create_reduction(source_buffer, input->size());
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
///  @params[in] name          Name of the kernel for reference.
///  @params[in] inputs        Input variables of the kernel.
///  @params[in] outputs       Output nodes to calculate results of.
///  @params[in] num_rays      Number of rays.
///  @params[in] add_reduction Optional argument to generate the reduction
///                           kernel.
//------------------------------------------------------------------------------
        void compile(const std::string name,
                     graph::input_nodes<T> inputs,
                     graph::output_nodes<T> outputs,
                     const size_t num_rays,
                     const bool add_reduction=false) {
            gpu_context.create_pipeline(source_buffer.str(), name,
                                        inputs, outputs, num_rays,
                                        add_reduction);
        }

//------------------------------------------------------------------------------
///  @brief Compile the max kernel.
//------------------------------------------------------------------------------
        void compile_max() {
            gpu_context.create_max_pipeline();
        }

//------------------------------------------------------------------------------
///  @brief Run the kernel.
//------------------------------------------------------------------------------
        void run() {
            gpu_context.run();
        }

//------------------------------------------------------------------------------
///  @brief Max reduction.
///
///  @returns The maximum value from the input buffer.
//------------------------------------------------------------------------------
        T max_reduction() {
            return gpu_context.max_reduction();
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
///  @brief Copy contexts of buffer.
///
///  @params[in]     source_index Index of the GPU buffer.
///  @params[in,out] destination  Host side buffer to copy to.
//------------------------------------------------------------------------------
        void copy_buffer(const size_t source_index,
                         T *destination) {
            gpu_context.copy_buffer(source_index, destination);
        }
    };
}

#endif /* jit_h */
