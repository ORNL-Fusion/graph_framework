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
///  @brief Tests if the backend can be jit'ed.
///
///  Metal contexts only support float. Cuda contexts support everthing.
///  FIXME: HIP support is missing.
///
///  @returns True if the backend is compatable with JIT.
//------------------------------------------------------------------------------
    template<class BACKEND>
    constexpr bool can_jit() {
#ifdef USE_GPU
#ifdef USE_METAL
        return std::is_same<typename BACKEND::base, float>::value;
#endif
#ifdef USE_CUDA
        return true;
#endif
#else
        return false;
#endif
    }

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
#ifdef USE_GPU
///  GPU Context;
        GPU_CONTEXT context;
#endif

    public:
//------------------------------------------------------------------------------
///  @brief Construct a kernel object.
///
///  Build the source code for a kernel graph.
///
///  @param[in] name    Name to call the kernel.
///  @param[in] inputs  Input variables of the kernel.
///  @param[in] outputs Output nodes of the graph to compute.
///  @param[in] setters Map outputs back to input values.
//------------------------------------------------------------------------------
        kernel(const std::string name,
               graph::input_nodes<BACKEND> inputs,
               graph::output_nodes<BACKEND> outputs,
               graph::map_nodes<BACKEND> setters) {
            const size_t test_size = inputs[0]->size();
            
            create_preamble(name);
            
            add_kernel_argument(to_string('v', inputs[0].get()), 0);
            for (size_t i = 1, ie = inputs.size(); i < ie; i++) {
                assert(test_size == inputs[i]->size() &&
                       "Kernel input variables all need to be the same size.");
                add_kernel_argument(to_string('v', inputs[i].get()), i);
            }
            
            for (size_t i = 0, ie = outputs.size(); i < ie; i++) {
                add_kernel_argument(to_string('o', outputs[i].get()),
                                    i + inputs.size());
            }

            add_argument_index(test_size);
            
            for (graph::shared_variable<BACKEND> &input : inputs) {
                load_variable(input.get());
            }
            
            for (auto &[out, in] : setters) {
                out->compile(source_buffer, registers);
            }
            for (auto &out : outputs) {
                out->compile(source_buffer, registers);
            }

            for (auto &[out, in] : setters) {
                graph::shared_leaf<BACKEND> a = out->compile(source_buffer, registers);
                store_variable(in.get(), registers[a.get()]);
            }
            
            for (auto &out : outputs) {
                graph::shared_leaf<BACKEND> a = out->compile(source_buffer, registers);
                store_node(out.get(), registers[a.get()]);
            }

            source_buffer << "    }" << std::endl << "}" << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Add max reduction.
//------------------------------------------------------------------------------
        void add_max_reduction(graph::shared_variable<BACKEND> input) {
            source_buffer << std::endl;
#ifdef USE_METAL
            source_buffer << "kernel ";
#else
            source_buffer << "extern \"C\" __global__ ";
#endif
            source_buffer << "void max_reduction(";
            
            add_kernel_argument("input", 0);
            add_kernel_argument("result", 1);
            
#ifdef USE_METAL
            source_buffer << "," << std::endl;
            source_buffer << "    uint i [[thread_position_in_grid]]," << std::endl;
            source_buffer << "    uint j [[simdgroup_index_in_threadgroup]]," << std::endl;
            source_buffer << "    uint k [[thread_index_in_simdgroup]]) {" << std::endl;
            
            source_buffer << "    if (i < " << input->size() << ") {" << std::endl;
            source_buffer << "        float sub_max = input[i];" << std::endl;
            source_buffer << "        for (size_t index = i + 1024; index < " << input->size() << "; index += 1024) {" << std::endl;
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
#endif
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
            source_buffer << "#include <metal_stdlib>" << std::endl;
            source_buffer << "#include <metal_simdgroup>" << std::endl;
            source_buffer << "using namespace metal;" << std::endl
                          << "kernel ";
#else
            source_buffer << "extern \"C\" __global__ ";
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
            source_buffer << "    const size_t index = ";//min(";
#ifdef USE_METAL
            source_buffer << "i;";//, uint(";
#elif defined (USE_CUDA)
            source_buffer << "blockIdx.x*blockDim.x + threadIdx.x;";//, (";
#elif defined (USE_HIP)
            source_buffer << "hipBlockIdx_x*hipBlockDim_x + hipThreadIdx_x;";//, (";
#endif
            source_buffer << std::endl << "    if (index < " << size << ") {" << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Add buffer argument indexing.
///
///  @param[in] pointer Pointer to the variable node.
//------------------------------------------------------------------------------
        void load_variable(graph::variable_node<BACKEND> *pointer) {
            registers[pointer] = to_string('r', pointer);
            source_buffer << "        const ";
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
            source_buffer << "        " << to_string('v',  pointer)
                          << "[index] = " << result << ";" << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Store the final buffer.
///
///  @param[in] pointer Pointer to the result node.
///  @param[in] result  Name of the result reguster.
//------------------------------------------------------------------------------
        void store_node(graph::leaf_node<BACKEND> *pointer,
                        const std::string result) {
            source_buffer << "        " << to_string('o',  pointer)
                          << "[index] = " << result << ";" << std::endl;
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
///  @param[in] name      Name of the kernel for reference.
///  @param[in] inputs    Input variables of the kernel.
///  @param[in] outputs   Output nodes to calculate results of.
///  @param[in] num_rays  Number of rays.
//------------------------------------------------------------------------------
        void compile(const std::string name,
                     graph::input_nodes<BACKEND> inputs,
                     graph::output_nodes<BACKEND> outputs,
                     const size_t num_rays) {
#ifdef USE_GPU
            context.create_pipeline(source_buffer.str(), name,
                                    inputs, outputs, num_rays);
#endif
        }

//------------------------------------------------------------------------------
///  @brief Compile the max kernel.
//------------------------------------------------------------------------------
        void compile_max() {
#ifdef USE_GPU
            context.create_max_pipeline<BACKEND> ();
#endif
        }

//------------------------------------------------------------------------------
///  @brief Run the kernel.
//------------------------------------------------------------------------------
        void run() {
#ifdef USE_GPU
            context.run();
#endif
        }

//------------------------------------------------------------------------------
///  @brief Max reduction.
///
///  @returns The maximum value from the input buffer.
//------------------------------------------------------------------------------
        typename BACKEND::base max_reduction() {
#ifdef USE_GPU
            return context.max_reduction<BACKEND> ();
#endif
        }

//------------------------------------------------------------------------------
///  @brief Print output.
///
///  @param[in] index Particle index to print.
//------------------------------------------------------------------------------
        void print(const size_t index) {
#ifdef USE_GPU
            context.print_results<BACKEND> (index);
#endif
        }

//------------------------------------------------------------------------------
///  @brief Wait for kernel to finish.
//------------------------------------------------------------------------------
        void wait() {
#ifdef USE_GPU
            context.wait();
#endif
        }

//------------------------------------------------------------------------------
///  @brief Copy contexts of buffer.
///
///  @param[in]     source_index Index of the GPU buffer.
///  @param[in,out] destination  Host side buffer to copy to.
//------------------------------------------------------------------------------
        void copy_buffer(const size_t source_index,
                         typename BACKEND::base *destination) {
#ifdef USE_GPU
            context.copy_buffer(source_index, destination);
#endif
        }
    };
}

#endif /* jit_h */
