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
///  Argument map.
        std::map<graph::leaf_node<T> *, std::vector<T>> kernel_arguments;
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
///  @brief Compile the kernels.
///
///  @params[in] kernel_source Source code buffer for the kernel.
///  @params[in] names         Names of the kernel functions.
///  @params[in] add_reduction Include the reduction kernel.
//------------------------------------------------------------------------------
        void compile(const std::string kernel_source,
                     std::vector<std::string> names,
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
            temp_stream << CXX << " -dynamiclib -flat_namespace ";
#else
            temp_stream << CXX << " -fPIC -shared ";
#endif
#ifndef NDEBUG
            temp_stream << "-g ";
#else
            temp_stream << "-O3 ";
#endif
            temp_stream << filename << " -o " << library_name;

            std::cout << "CPU info." << std::endl;
            std::cout << "  Command Line    : " << temp_stream.str() << std::endl;
            int error = system(temp_stream.str().c_str());
            if (error) {
                std::cerr << "Failed to compile cpu kernel. Check source code in "
                          << filename << std::endl;
                exit(error);
            }

#ifdef NDEBUG
            temp_stream.str(std::string());
            temp_stream.clear();
            temp_stream << "rm " << filename;
            system(temp_stream.str().c_str());
#endif

            lib_handle = dlopen(library_name.c_str(), RTLD_LAZY);
            if (!lib_handle) {
                std::cout << "Failed to load library. " << library_name
                          << std::endl;
                exit(1);
            }

            std::cout << "  Library name    : " << library_name << std::endl;
            std::cout << "  Library handle  : " << reinterpret_cast<size_t> (lib_handle) << std::endl;
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
        std::function<void(void)>  create_kernel_call(const std::string kernel_name,
                                                      graph::input_nodes<T> inputs,
                                                      graph::output_nodes<T> outputs,
                                                      const size_t num_rays) {
            void *kernel = dlsym(lib_handle, kernel_name.c_str());
            if (!kernel) {
                std::cerr << "Failed to load function. " << kernel_name
                          << std::endl;
                exit(1);
            }

            std::vector<T *> buffers;

            for (auto &input : inputs) {
                if (!kernel_arguments.contains(input.get())) {
                    backend::buffer<T> buffer = input->evaluate();
                    std::vector<T> arg(buffer.size());
                    memcpy(arg.data(), buffer.data(), buffer.size()*sizeof(T));
                    kernel_arguments[input.get()] = arg;
                }
                buffers.push_back(kernel_arguments[input.get()].data());
            }
            for (auto &output : outputs) {
                if (!kernel_arguments.contains(output.get())) {
                    std::vector<T> arg(num_rays);
                    kernel_arguments[output.get()] = arg;
                }
                buffers.push_back(kernel_arguments[output.get()].data());
            }

            std::cout << "  Function pointer: " << reinterpret_cast<size_t> (kernel) << std::endl;

            return [kernel, buffers] () mutable {
                ((void (*)(std::vector<T *> &))kernel)(buffers);
            };
        }

//------------------------------------------------------------------------------
///  @brief Create a max compute pipeline.
///
///  @params[in] argument Node to reduce.
///  @params[in] run      Function to run before reduction.
//------------------------------------------------------------------------------
        std::function<T(void)> create_max_call(graph::shared_leaf<T> &argument,
                                               std::function<void(void)> run) {
            auto begin = kernel_arguments[argument.get()].cbegin();
            auto end = kernel_arguments[argument.get()].cend();
            
            return [run, begin, end] () mutable {
                run();
                if constexpr (jit::is_complex<T> ()) {
                    return *std::max_element(begin, end,
                                             [] (const T a, const T b) {
                        return std::abs(a) < std::abs(b);
                    });
                } else {
                    return *std::max_element(begin, end);
                }
            };
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
            for (const auto &[key, value] : kernel_arguments) {
                std::cout << value[index] << " ";
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
            memcpy(kernel_arguments[node.get()].data(),
                   source,
                   sizeof(T)*kernel_arguments[node.get()].size());
        }

//------------------------------------------------------------------------------
///  @brief Copy buffer contents to host.
///
///  @params[in]     node        Node to copy buffer from.
///  @params[in,out] destination Host side buffer to copy to.
//------------------------------------------------------------------------------
        void copy_to_host(const graph::shared_leaf<T> node,
                          T *destination) {
            memcpy(destination,
                   kernel_arguments[node.get()].data(),
                   sizeof(T)*kernel_arguments[node.get()].size());
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
            
            source_buffer << "    vector<";
            jit::add_type<T> (source_buffer);
            source_buffer << " *> &args) {" << std::endl;
            
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
