//------------------------------------------------------------------------------
///  @file jit.hpp
///  @brief Class to just in time compile a kernel.
///
///  Defines a tree of operations that allows automatic differentiation.
//------------------------------------------------------------------------------

#ifndef jit_h
#define jit_h

#include <chrono>

#ifdef USE_METAL
#include "metal_context.hpp"
#elif defined(USE_CUDA)
#include "cuda_context.hpp"
#endif

#include "node.hpp"
#include "timing.hpp"

#ifdef USE_METAL
#define GPU_CONTEXT gpu::metal_context
#elif defined(USE_CUDA)
#define GPU_CONTEXT gpu::cuda_context
#elif defined(USE_HIP)
#define GPU_CONTEXT gpu::hip_context
#endif

namespace jit {
//------------------------------------------------------------------------------
///  @brief Class for JIT compile of the GPU kernels.
//------------------------------------------------------------------------------
    template<class BACKEND>
    class kernel {
    private:
///  String stream to build the kernel source.
        std::stringstream source_buffer;
///  Nodes that have been jitted.
        register_map<graph::leaf_node<BACKEND>> registers;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a kernel object.
///
///  Build the source code for a kernel graph.
///
///  @param[in] name    Name to call the kernel.
///  @param[in] inputs  Input variables of the kernel.
///  @param[in] setters Map outputs back to input values.
//------------------------------------------------------------------------------
        kernel(const std::string name,
               graph::input_nodes<BACKEND> inputs,
               graph::map_nodes<BACKEND> setters) {
            const size_t test_size = inputs[0]->size();
            
            create_preamble(name);
            add_kernel_argument(to_string('v', inputs[0].get()), 0);

            for (size_t i = 1, ie = inputs.size(); i < ie; i++) {
                assert(test_size == inputs[i]->size() &&
                       "Kernel input variables all need to be the same size.");
                add_kernel_argument(to_string('v', inputs[i].get()), i);
            }
            
            add_argument_index(test_size);
            
            for (graph::shared_variable<BACKEND> &input : inputs) {
                load_variable(input.get());
            }
            
            for (auto &[out, in] : setters) {
                out->compile(source_buffer, registers);
            }
            
            for (auto &[out, in] : setters) {
                graph::shared_leaf<BACKEND> a = out->compile(source_buffer, registers);
                store_variable(in.get(), registers[a.get()]);
            }
            
            source_buffer << "}" << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Create the kernel preamble.
///
///  Defines the kernel signature for what ever GPU type we are using.
///
///  Metal:
///
///    kernel void <name>(
///
///  Cuda, ROCM/HIP:
///
///    __global__ void <name>(
///
///  @param[in] name Name of the kernel to generate.
//------------------------------------------------------------------------------
        void create_preamble(const std::string name) {
#ifdef USE_METAL
            source_buffer << "using namespace metal;" << std::endl
                          << "kernel ";
#else
            source_buffer << "__global__ ";
#endif
            source_buffer << "void " << name << "(";
        }

//------------------------------------------------------------------------------
///  @brief Create the kernel argument.
///
///  Add a kernel argument to the source.
///
///  Metal:
///
///    device <type> *<name> [[buffer(<index>)]]
///
///  Cuda, ROCM/HIP:
///
///    <name> *<name>
///
///  @param[in] name  Name of the kernel to generate.
///  @param[in] index Index of the kernel.
//------------------------------------------------------------------------------
        void add_kernel_argument(std::string name,
                                 const size_t index) {
            if (index > 0) {
                source_buffer << "," << std::endl;
            } else {
                source_buffer << std::endl;
            }
            
            source_buffer << "    ";
#ifdef USE_METAL
            source_buffer << "device ";
#endif
            add_type<graph::leaf_node<BACKEND>> (source_buffer);
            source_buffer << " *" << name;
#ifdef USE_METAL
            source_buffer << " [[buffer("<< index <<")]]";
#endif
            
        }

//------------------------------------------------------------------------------
///  @brief Add buffer argument indexing.
///
///  @param[in] size Size of the array.
//------------------------------------------------------------------------------
        void add_argument_index(const size_t size) {
#ifdef USE_METAL
            source_buffer << "," << std::endl;
            source_buffer << "    uint i [[thread_position_in_grid]]";
#endif
            source_buffer << ") {" << std::endl;
            source_buffer << "    const size_t index = min(";
#ifdef USE_METAL
            source_buffer << "i, uint(";
#elif defined (USE_CUDA)
            source_buffer << "blockIdx.x*blockDim.x + threadIdx.x, (";
#elif defined (USE_HIP)
            source_buffer << "hipBlockIdx_x*hipBlockDim_x + hipThreadIdx_x, ";
#endif
            source_buffer << size - 1 << "));" << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Add buffer argument indexing.
///
///  @param[in] pointer Pointer to the variable node.
//------------------------------------------------------------------------------
        void load_variable(graph::variable_node<BACKEND> *pointer) {
            registers[pointer] = to_string('r', pointer);
            source_buffer << "    const ";
            add_type<graph::leaf_node<BACKEND>> (source_buffer);
            source_buffer << " " << registers[pointer] << " = "
                          << to_string('v', pointer) << "[index];"
                          << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Store the final buffer.
///
///  @param[in] pointer Pointer to the variable node.
///  @param[in] result  Name of the result reguster.
//------------------------------------------------------------------------------
        void store_variable(graph::variable_node<BACKEND> *pointer,
                            const std::string result) {
            source_buffer << "    " << to_string('v',  pointer)
                          << "[index] = " << result << ";" << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Print the kernel source.
//------------------------------------------------------------------------------
        void print() {
            std::cout << std::endl << source_buffer.str() << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Compile the kernel.
///
///  @param[in] inputs Input variables of the kernel.
//------------------------------------------------------------------------------
        void compile(const std::string name,
                     graph::input_nodes<BACKEND> inputs,
                     const size_t num_steps,
                     const size_t num_rays) {
            GPU_CONTEXT context;
            context.create_pipeline(source_buffer.str(), name, inputs, num_rays,
                                    num_steps, 0);
            
            const timeing::measure_diagnostic gpu_time("GPU Time");

            for (size_t i = 0; i < num_steps; i++) {
                context.step();
            }
            context.wait();
            gpu_time.stop();
            
            context.print_results<BACKEND> (num_steps);
        }
    };
}

#endif /* jit_h */
