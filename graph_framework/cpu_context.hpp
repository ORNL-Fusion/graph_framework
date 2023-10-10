//------------------------------------------------------------------------------
///  @file cpu_context.hpp
///  @brief Cpu context for cpus.
///
///  Defines context for cpu.
//------------------------------------------------------------------------------

#ifndef cpu_context_h
#define cpu_context_h

#include <fstream>
#include <cstdlib>
#include <cstring>
#include <thread>

#include <dlfcn.h>

#include "node.hpp"

namespace gpu {
//------------------------------------------------------------------------------
///  @brief Class representing a cpu context.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class cpu_context {
    private:
///  Library name.
        std::string library_name;
///  Handle for the dynamic library.
        void *lib_handle;
///  Argument map.
        std::map<graph::leaf_node<T, SAFE_MATH> *, std::vector<T>> kernel_arguments;
///  Argument index map.
        std::map<graph::leaf_node<T, SAFE_MATH> *, size_t> arg_index;

    public:
//------------------------------------------------------------------------------
///  @brief Get the maximum number of concurrent instances.
///
///  @returns The maximum available concurrency.
//------------------------------------------------------------------------------
        static size_t max_concurrency() {
            return std::thread::hardware_concurrency();
        }

//------------------------------------------------------------------------------
///  @brief Device discription.
//------------------------------------------------------------------------------
        static std::string device_type() {
            return "CPU";
        }

//------------------------------------------------------------------------------
///  @brief Construct a cpu context.
///
///  @params[in] index Concurrent index. Not used.
//------------------------------------------------------------------------------
        cpu_context(const size_t index) {}

//------------------------------------------------------------------------------
///  @brief Destruct a cpu context.
//------------------------------------------------------------------------------
        ~cpu_context() {
            dlclose(lib_handle);

            if (!library_name.empty()) {
                std::ostringstream temp_stream;
                temp_stream << "rm " << library_name;
                system(temp_stream.str().c_str());
            }
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
            std::ostringstream temp_stream;
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

            temp_stream << "./" << filename << ".so";
            library_name = temp_stream.str();

            temp_stream.str(std::string());
            temp_stream.clear();
#ifdef __APPLE__
            temp_stream << CXX << " -dynamiclib -flat_namespace ";
#else
            temp_stream << CXX << " -fPIC -shared ";
#endif
#ifndef NDEBUG
            temp_stream << CXX_FLAGS << " ";
#else
            temp_stream << "-O3 ";
#endif
            temp_stream << filename << " -o " << library_name;

            if (jit::verbose) {
                std::cout << "CPU info." << std::endl;
                std::cout << "  Command Line    : " << temp_stream.str() << std::endl;
            }
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
                std::cout << dlerror() << std::endl;
                exit(1);
            }

            if (jit::verbose) {
                std::cout << "  Library name    : " << library_name << std::endl;
                std::cout << "  Library handle  : " << reinterpret_cast<size_t> (lib_handle) << std::endl;
            }
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
                                                      graph::input_nodes<T, SAFE_MATH> inputs,
                                                      graph::output_nodes<T, SAFE_MATH> outputs,
                                                      const size_t num_rays) {
            void *kernel = dlsym(lib_handle, kernel_name.c_str());
            if (!kernel) {
                std::cerr << "Failed to load function. " << kernel_name
                          << std::endl;
                exit(1);
            }

            std::map<std::string, T *> buffers;

            for (auto &input : inputs) {
                if (!kernel_arguments.contains(input.get())) {
                    backend::buffer<T> buffer = input->evaluate();
                    std::vector<T> arg(buffer.size());
                    memcpy(arg.data(), buffer.data(), buffer.size()*sizeof(T));
                    kernel_arguments[input.get()] = arg;
                }
                buffers[jit::to_string('v', input.get())] = kernel_arguments[input.get()].data();
            }
            for (auto &output : outputs) {
                if (!kernel_arguments.contains(output.get())) {
                    std::vector<T> arg(num_rays);
                    kernel_arguments[output.get()] = arg;
                }
                buffers[jit::to_string('o', output.get())] = kernel_arguments[output.get()].data();
            }

            if (jit::verbose) {
                std::cout << "  Function pointer: " << reinterpret_cast<size_t> (kernel) << std::endl;
            }

            return [kernel, buffers] () mutable {
                ((void (*)(std::map<std::string, T *> &))kernel)(buffers);
            };
        }

//------------------------------------------------------------------------------
///  @brief Create a max compute pipeline.
///
///  @params[in] argument Node to reduce.
///  @params[in] run      Function to run before reduction.
//------------------------------------------------------------------------------
        std::function<T(void)> create_max_call(graph::shared_leaf<T, SAFE_MATH> &argument,
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
///  @brief Hold the current thread until the command buffer has completed.
//------------------------------------------------------------------------------
        void wait() {}

//------------------------------------------------------------------------------
///  @brief Print out the results.
///
///  @params[in] index Particle index to print.
///  @params[in] nodes Nodes to output.
//------------------------------------------------------------------------------
        void print_results(const size_t index,
                           const graph::output_nodes<T, SAFE_MATH> &nodes) {
            for (auto &out : nodes) {
                const T temp = kernel_arguments[out.get()][index];
                if constexpr (jit::is_complex<T> ()) {
                    std::cout << std::real(temp) << " " << std::imag(temp) << " ";
                } else {
                    std::cout << temp << " ";
                }
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
        T check_value(const size_t index,
                      const graph::shared_leaf<T, SAFE_MATH> &node) {
            return kernel_arguments[node.get()][index];
        }

//------------------------------------------------------------------------------
///  @brief Copy buffer contents to the device.
///
///  @params[in] node   Not to copy buffer to.
///  @params[in] source Host side buffer to copy from.
//------------------------------------------------------------------------------
        void copy_to_device(graph::shared_leaf<T, SAFE_MATH> node,
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
        void copy_to_host(const graph::shared_leaf<T, SAFE_MATH> node,
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
        void create_header(std::ostringstream &source_buffer) {
            source_buffer << "#include <map>" << std::endl;
            source_buffer << "#include <string>" << std::endl;
            if (jit::is_complex<T> ()) {
                source_buffer << "#include <complex>" << std::endl;
                source_buffer << "#include <special_functions.hpp>" << std::endl;
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
        void create_kernel_prefix(std::ostringstream &source_buffer,
                                  const std::string name,
                                  graph::input_nodes<T, SAFE_MATH> &inputs,
                                  graph::output_nodes<T, SAFE_MATH> &outputs,
                                  const size_t size,
                                  jit::register_map &registers) {
            source_buffer << std::endl;
            source_buffer << "extern \"C\" void " << name << "(" << std::endl;
            
            source_buffer << "    std::map<std::string, ";
            jit::add_type<T> (source_buffer);
            source_buffer << " *> &args) {" << std::endl;
            
            source_buffer << "    for (size_t i = 0; i < " << size << "; i++) {" << std::endl;

            for (auto &input : inputs) {
                registers[input.get()] = jit::to_string('r', input.get());
                source_buffer << "        const ";
                jit::add_type<T> (source_buffer);
                source_buffer << " " << registers[input.get()]
                              << " = args[std::string(\"" << jit::to_string('v', input.get())
                              << "\")][i]; //" << input->get_symbol() << std::endl;
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
        void create_kernel_postfix(std::ostringstream &source_buffer,
                                   graph::output_nodes<T, SAFE_MATH> &outputs,
                                   graph::map_nodes<T, SAFE_MATH> &setters,
                                   jit::register_map &registers) {
            for (auto &[out, in] : setters) {
                graph::shared_leaf<T, SAFE_MATH> a = out->compile(source_buffer, registers);
                source_buffer << "        args[std::string(\"" << jit::to_string('v', in.get());
                source_buffer << "\")][i] = " << registers[a.get()] << ";" << std::endl;
            }
            for (auto &out : outputs) {
                graph::shared_leaf<T, SAFE_MATH> a = out->compile(source_buffer, registers);
                source_buffer << "        args[std::string(\"" << jit::to_string('o', out.get());
                source_buffer << "\")][i] = " << registers[a.get()] << ";" << std::endl;
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
        void create_reduction(std::ostringstream &source_buffer,
                              const size_t size) {}

//------------------------------------------------------------------------------
///  @brief Get the buffer for a node.
///
///  @params[in] node Node to get the buffer for.
//------------------------------------------------------------------------------
        T *get_buffer(graph::shared_leaf<T, SAFE_MATH> &node) {
            return kernel_arguments[node.get()].data();
        }
    };
}

#endif /* cpu_context_h */
