//------------------------------------------------------------------------------
///  @file cpu_context.hpp
///  @brief Cpu context for cpus.
///
///  Defines context for cpu.
//------------------------------------------------------------------------------

#ifndef cpu_context_h
#define cpu_context_h

#include <iostream>
#include <fstream>
#include <cstdlib>

#include <dlfcn.h>

#include "node.hpp"

namespace gpu {
//------------------------------------------------------------------------------
///  @brief Class representing a cpu context.
//------------------------------------------------------------------------------
    template<typename T>
    class cpu_context {
    private:
///  Library name.
        std::string library_name;
///  Handle for the dynamic library.
        void *lib_handle;
///  Dynamic header
        void *kernel;
///  Kernel arguments.
        std::vector<std::vector<T>> kernel_args;
///  Argument index map.
        std::map<graph::leaf_node<T> *, size_t> arg_index;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a cpu context.
//------------------------------------------------------------------------------
        cpu_context() {}

//------------------------------------------------------------------------------
///  @brief Destruct a cpu context.
//------------------------------------------------------------------------------
        ~cpu_context() {
            dlclose(lib_handle);
            
            std::stringstream temp_stream;
            temp_stream << "rm " << library_name;
            system(temp_stream.str().c_str());
        }

//------------------------------------------------------------------------------
///  @brief Create a compute pipeline.
///
///  @params[in] kernel_source Source code buffer for the kernel.
///  @params[in] kernel_name   Name of the kernel for later reference.
///  @params[in] inputs        Input nodes of the kernel.
///  @params[in] outputs       Output nodes of the kernel.
///  @params[in] num_rays      Number of rays to trace.
///  @params[in] add_reduction Optional argument to generate the reduction
///                           kernel.
//------------------------------------------------------------------------------
        void create_pipeline(const std::string kernel_source,
                             const std::string kernel_name,
                             graph::input_nodes<T> inputs,
                             graph::output_nodes<T> outputs,
                             const size_t num_rays,
                             const bool add_reduction=false) {
            std::stringstream temp_stream;
            temp_stream << reinterpret_cast<size_t> (this);
            const std::string thread_id = temp_stream.str();

            temp_stream.str(std::string());
            temp_stream.clear();

            temp_stream << "temp_" << thread_id << ".cpp";

            const std::string filename = temp_stream.str();
            std::ofstream out(filename);

            out << kernel_source;
            out.close();
            
            temp_stream.str(std::string());
            temp_stream.clear();

            temp_stream << filename << ".so";
            library_name = temp_stream.str();

            temp_stream.str(std::string());
            temp_stream.clear();
#ifdef __APPLE__
            temp_stream << CXX << " -O3 -dynamiclib -flat_namespace ";
#else
            temp_stream << CXX << " -O3 -fPIC -shared ";
#endif
            temp_stream << filename << " -o " << library_name;

            std::cout << "CPU info." << std::endl;
            std::cout << "  Command Line    : " << temp_stream.str() << std::endl;
            int error = system(temp_stream.str().c_str());
            if (error) {
                std::cout << "Failed to compile cpu kernel. Check source code in "
                          << filename << std::endl;
                exit(error);
            }
            
            temp_stream.str(std::string());
            temp_stream.clear();
            temp_stream << "rm " << filename;
            system(temp_stream.str().c_str());

            lib_handle = dlopen(library_name.c_str(), RTLD_LAZY);
            if (!lib_handle) {
                std::cout << "Failed to load library. " << library_name
                          << std::endl;
                exit(1);
            }
            kernel = dlsym(lib_handle, kernel_name.c_str());
            if (!kernel) {
                std::cout << "Failed to load function. " << kernel_name
                          << std::endl;
                exit(1);
            }

            for (auto &input : inputs) {
                backend::buffer<T> buffer = input->evaluate();
                std::vector<T> arg(buffer.size());
                memcpy(arg.data(), buffer.data(), buffer.size()*sizeof(T));
                kernel_args.push_back(arg);
            }
            for (auto &output : outputs) {
                backend::buffer<T> buffer = output->evaluate();
                std::vector<T> arg(buffer.size());
                kernel_args.push_back(arg);
            }
            
            std::cout << "  Library name    : " << library_name << std::endl;
            std::cout << "  Library handle  : " << reinterpret_cast<size_t> (lib_handle) << std::endl;
            std::cout << "  Function pointer: " << reinterpret_cast<size_t> (kernel) << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Create a max compute pipeline.
//------------------------------------------------------------------------------
        void create_max_pipeline() {}

//------------------------------------------------------------------------------
///  @brief Perform a time step.
///
///  This calls dispatches a kernel instance to the command buffer and the commits
///  the job. This method is asyncronus.
//------------------------------------------------------------------------------
        void run() {
            ((void (*)(std::vector<std::vector<T>> &))kernel)(kernel_args);
        }

//------------------------------------------------------------------------------
///  @brief Hold the current thread until the current command buffer has complete.
//------------------------------------------------------------------------------
        void wait() {}

//------------------------------------------------------------------------------
///  @brief Print out the results.
///
///  @params[in] index Particle index to print.
//------------------------------------------------------------------------------
        void print_results(const size_t index) {
            for (auto &buffer : kernel_args) {
                std::cout << buffer[index] << " ";
            }
            std::cout << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Copy buffer contents.
///
///  @params[in]     source_index Index of the GPU buffer.
///  @params[in,out] destination  Host side buffer to copy to.
//------------------------------------------------------------------------------
        void copy_buffer(const size_t source_index,
                         T *destination) {
            memcpy(destination,
                   kernel_args[source_index].data(),
                   sizeof(T)*kernel_args[source_index].size());
        }

//------------------------------------------------------------------------------
///  @brief Compute the max reduction.
///
///  @returns The maximum value from the input buffer.
//------------------------------------------------------------------------------
        T max_reduction() {
            run();
            if constexpr (jit::is_complex<T> ()) {
                return *std::max_element(kernel_args.back().cbegin(),
                                         kernel_args.back().cend(),
                                         [] (const T a, const T b) {
                    return std::abs(a) < std::abs(b);
                });
            } else {
                return *std::max_element(kernel_args.back().cbegin(),
                                         kernel_args.back().cend());
            }
        }

//------------------------------------------------------------------------------
///  @brief Create the source header.
///
///  @params[in,out] source_buffer Source buffer stream.
//------------------------------------------------------------------------------
        void create_header(std::stringstream &source_buffer) {
            source_buffer << "#include <vector>" << std::endl;
            if (jit::is_complex<T> ()) {
                source_buffer << "#include <complex>" << std::endl;
            } else {
                source_buffer << "#include <cmath>" << std::endl;
            }
            source_buffer << "using namespace std;" << std::endl;
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
                                  jit::register_map<graph::leaf_node<T>> &registers) {
            source_buffer << std::endl;
            source_buffer << "extern \"C\" void " << name << "(" << std::endl;
            
            source_buffer << "    vector<vector<";
            jit::add_type<T> (source_buffer);
            source_buffer << " > > &args) {" << std::endl;
            
            source_buffer << "    for (size_t i = 0; i < " << size << "; i++) {" << std::endl;
            for (size_t i = 0, ie = inputs.size(); i < ie; i++) {
                registers[inputs[i].get()] = jit::to_string('r', inputs[i].get());
                source_buffer << "        const ";
                jit::add_type<T> (source_buffer);
                source_buffer << " " << registers[inputs[i].get()]
                              << " = args[" << i << "][i];" << std::endl;
                arg_index[inputs[i].get()] = i;
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
                                   jit::register_map<graph::leaf_node<T>> &registers) {
            for (auto &[out, in] : setters) {
                graph::shared_leaf<T> a = out->compile(source_buffer, registers);
                source_buffer << "        args[" << arg_index[in.get()];
                source_buffer << "][i] = " << registers[a.get()] << ";" << std::endl;
            }

            for (size_t i = 0, ie = outputs.size(); i < ie; i++) {
                graph::shared_leaf<T> a = outputs[i]->compile(source_buffer, registers);
                source_buffer << "        args[" << arg_index.size() + i
                              << "][i] = " << registers[a.get()] << ";"
                              << std::endl;
            }
                    
            source_buffer << "    }" << std::endl;
            source_buffer << "}" << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Create a reduction kernel.
///
///  @params[in,out] source_buffer Source buffer stream.
///  @params[in]     size          Size of the input buffer.
//------------------------------------------------------------------------------
        void create_reduction(std::stringstream &source_buffer,
                              const size_t size) {}
    };
}

#endif /* cpu_context_h */
