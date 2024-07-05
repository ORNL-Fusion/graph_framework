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

//  Clang headers will define IBAction and IBOutlet these so undefine them here.
#undef IBAction
#undef IBOutlet
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#ifndef NDEBUG
#include "llvm/ExecutionEngine/Orc/Debugging/DebuggerSupport.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"
#endif
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"

#include "node.hpp"

#ifndef NDEBUG
//------------------------------------------------------------------------------
///  @brief This just exposes the functions so the debugger links.
//------------------------------------------------------------------------------
LLVM_ATTRIBUTE_USED void linkComponents() {
    llvm::errs() << (void *)&llvm_orc_registerJITLoaderGDBWrapper
                 << (void *)&llvm_orc_registerJITLoaderGDBAllocAction;
}
#endif

namespace gpu {
//------------------------------------------------------------------------------
///  @brief Split a string by the space delimiter.
///
///  The exec functions need the arguments split into individual calls. So this
///  splits the strings that come from cmake into a char \* vector. Note the
///  first token will be duplacted in the first two elements.
///
///  @param[in] string Input string.
///  @returns The string split into an array of arguments.
//------------------------------------------------------------------------------
    llvm::SmallVector<const char *, 8> split_string(char *string) {
        llvm::SmallVector<const char *, 8> args = {string};

        while (*(++string) != '\0') {
            if (*string == ' ') {
                *string = '\0';
                args.push_back(++string);
            }
        }

        return args;
    }

//------------------------------------------------------------------------------
///  @brief Class representing a cpu context.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class cpu_context {
    private:
///  Handle for the dynamic library.
        std::unique_ptr<llvm::orc::LLJIT> jit;
///  Argument map.
        std::map<graph::leaf_node<T, SAFE_MATH> *, std::vector<T>> kernel_arguments;
///  Host buffer map.
        std::map<graph::leaf_node<T, SAFE_MATH> *, std::vector<T>> host_buffers;
///  Argument index map.
        std::map<graph::leaf_node<T, SAFE_MATH> *, size_t> arg_index;

    public:
///  Remaining constant memory in bytes. NOT USED.
        int remaining_const_memory;

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
        cpu_context(const size_t index) {
            llvm::InitializeNativeTarget();
            llvm::InitializeNativeTargetAsmPrinter();
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

            if (jit::verbose) {
                std::cout << "CPU info." << std::endl;
                std::cout << "  Command Line    : " << std::endl;
            }

            char arg_string[] = CXX_ARGS;
            llvm::SmallVector<const char *, 8> args = split_string(arg_string);
            args.push_back(filename.c_str());
#ifdef NDEBUG
            args.push_back("-O3");
#else
            args.push_back("-debug-info-kind=standalone");
#endif
            if (jit::verbose) {
                for (auto &arg : args) {
                    std::cout << "    " << arg << std::endl;
                }
            }

            auto diagnostic_options = llvm::makeIntrusiveRefCnt<clang::DiagnosticOptions> ();
            auto diagnostic_printer = std::make_unique<clang::TextDiagnosticPrinter> (llvm::errs(),
                                                                                      diagnostic_options.get());

            auto diagnostic_ids = llvm::makeIntrusiveRefCnt<clang::DiagnosticIDs> ();
            clang::DiagnosticsEngine diagnostic_engine(diagnostic_ids,
                                                       diagnostic_options,
                                                       diagnostic_printer.release());

            auto invocation = std::make_shared<clang::CompilerInvocation> ();
            clang::CompilerInvocation::CreateFromArgs(*(invocation.get()), args,
                                                      diagnostic_engine);

            llvm::StringRef source_code_data(kernel_source);
            auto buffer = llvm::MemoryBuffer::getMemBuffer(source_code_data);
            invocation->getPreprocessorOpts().addRemappedFile(filename.c_str(),
                                                              buffer.release());

            clang::CompilerInstance clang;
            clang.setInvocation(invocation);
            clang.createDiagnostics();

            const auto target_options = std::make_shared<clang::TargetOptions> ();
            target_options->Triple = llvm::sys::getProcessTriple();
            auto *target_info = clang::TargetInfo::CreateTargetInfo(diagnostic_engine,
                                                                    target_options);
            clang.setTarget(target_info);

            clang::EmitLLVMOnlyAction action;
            clang.ExecuteAction(action);

            auto ir_module = action.takeModule();
            auto context = std::unique_ptr<llvm::LLVMContext> (action.takeLLVMContext());

            auto jit_try = llvm::orc::LLJITBuilder()
#ifndef NDEBUG
                               .setPrePlatformSetup([](llvm::orc::LLJIT &J) {
                                   return llvm::orc::enableDebuggerSupport(J);
                               })
#endif
                               .create();
            if (auto jiterror = jit_try.takeError()) {
                std::cerr << "Failed to build JIT : " << toString(std::move(jiterror)) << std::endl;
                exit(-1);
            }
            jit = std::move(jit_try.get());

            auto error = jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(ir_module),
                                                                      llvm::orc::ThreadSafeContext(std::move(context))));
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
                                                     graph::input_nodes<T, SAFE_MATH> inputs,
                                                     graph::output_nodes<T, SAFE_MATH> outputs,
                                                     const size_t num_rays,
                                                     const jit::texture1d_list &tex1d_list,
                                                     const jit::texture2d_list &tex2d_list) {
            auto entry = std::move(jit->lookup(kernel_name)).get();
            auto kernel = entry.toPtr<void(*)(std::map<size_t, T *> &)> ();

            if (!kernel) {
                std::cerr << "Failed to load function. " << kernel_name
                          << std::endl;
                exit(-1);
            }

            std::map<size_t, T *> buffers;

            for (auto &input : inputs) {
                if (!kernel_arguments.contains(input.get())) {
                    backend::buffer<T> buffer = input->evaluate();
                    std::vector<T> arg(buffer.size());
                    memcpy(arg.data(), buffer.data(), buffer.size()*sizeof(T));
                    kernel_arguments[input.get()] = arg;
                }
                buffers[reinterpret_cast<size_t> (input.get())] = kernel_arguments[input.get()].data();
            }
            for (auto &output : outputs) {
                if (!kernel_arguments.contains(output.get())) {
                    std::vector<T> arg(num_rays);
                    kernel_arguments[output.get()] = arg;
                }
                buffers[reinterpret_cast<size_t> (output.get())] = kernel_arguments[output.get()].data();
            }

            if (jit::verbose) {
                std::cout << "  Function pointer: " << reinterpret_cast<size_t> (kernel) << std::endl;
            }

            return [kernel, buffers] () mutable {
                kernel(buffers);
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
///
///  This syncs the host buffers with the kernel arguments so a kernel can run
///  while another thread reads the results.
//------------------------------------------------------------------------------
        void wait() {
            for (auto &item : host_buffers) {
                memcpy(item.second.data(),
                       kernel_arguments[item.first].data(),
                       sizeof(T)*kernel_arguments[item.first].size());
            }
        }

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
///  @params[in]     is_constant   Flags if the input is read only.
///  @params[in,out] registers     Map of used registers.
///  @params[in]     usage         List of register usage count.
///  @params[in]     textures1d    List of 1D kernel textures.
///  @params[in]     textures2d    List of 2D kernel textures.
//------------------------------------------------------------------------------
        void create_kernel_prefix(std::ostringstream &source_buffer,
                                  const std::string name,
                                  graph::input_nodes<T, SAFE_MATH> &inputs,
                                  graph::output_nodes<T, SAFE_MATH> &outputs,
                                  const size_t size, 
                                  const std::vector<bool> &is_constant,
                                  jit::register_map &registers,
                                  const jit::register_usage &usage,
                                  jit::texture1d_list &textures1d,
                                  jit::texture2d_list &textures2d) {
            source_buffer << std::endl;
            source_buffer << "extern \"C\" void " << name << "(" << std::endl;

            source_buffer << "    map<size_t, ";
            jit::add_type<T> (source_buffer);
            source_buffer << " *> &args) {" << std::endl;

            for (size_t i = 0, ie = inputs.size(); i < ie; i++) {
                source_buffer << "    ";
                if (is_constant[i]) {
                    source_buffer << "const ";
                }
                jit::add_type<T> (source_buffer);
                source_buffer << " *" << jit::to_string('v', inputs[i].get())
                              << " = args[" << reinterpret_cast<size_t> (inputs[i].get())
                              << "];" << std::endl;
            }
            for (auto &output : outputs) {
                source_buffer << "    ";
                jit::add_type<T> (source_buffer);
                source_buffer << " *" << jit::to_string('o', output.get())
                              << " = args[" << reinterpret_cast<size_t> (output.get()) 
                              << "];" << std::endl;
            }

            source_buffer << "    for (size_t i = 0; i < " << size << "; i++) {" << std::endl;

            for (auto &input : inputs) {
                registers[input.get()] = jit::to_string('r', input.get());
                source_buffer << "        const ";
                jit::add_type<T> (source_buffer);
                source_buffer << " " << registers[input.get()]
                              << " = " << jit::to_string('v', input.get())
                              << "[i]; // " << input->get_symbol()
                              << " used " << usage.at(input.get()) << std::endl;
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
                                   graph::output_nodes<T, SAFE_MATH> &outputs,
                                   graph::map_nodes<T, SAFE_MATH> &setters,
                                   jit::register_map &registers,
                                   const jit::register_usage &usage) {
            for (auto &[out, in] : setters) {
                graph::shared_leaf<T, SAFE_MATH> a = out->compile(source_buffer,
                                                                  registers,
                                                                  usage);
                source_buffer << "        " << jit::to_string('v', in.get());
                source_buffer << "[i] = ";
                if constexpr (SAFE_MATH) {
                    if constexpr (jit::is_complex<T> ()) {
                        jit::add_type<T> (source_buffer);
                        source_buffer << " (";
                        source_buffer << "isnan(real(" << registers[a.get()]
                                      << ")) ? 0.0 : real(" << registers[a.get()]
                                      << "), ";
                        source_buffer << "isnan(imag(" << registers[a.get()]
                                      << ")) ? 0.0 : imag(" << registers[a.get()]
                                      << "));" << std::endl;
                    } else {
                        source_buffer << "isnan(" << registers[a.get()]
                                      << ") ? 0.0 : " << registers[a.get()]
                                      << ";" << std::endl;
                    }
                } else {
                    source_buffer << registers[a.get()] << ";" << std::endl;
                }
            }
            for (auto &out : outputs) {
                graph::shared_leaf<T, SAFE_MATH> a = out->compile(source_buffer,
                                                                  registers,
                                                                  usage);
                source_buffer << "        " << jit::to_string('o', out.get());
                source_buffer << "[i] = ";
                if constexpr (SAFE_MATH) {
                    if constexpr (jit::is_complex<T> ()) {
                        jit::add_type<T> (source_buffer);
                        source_buffer << " (";
                        source_buffer << "isnan(real(" << registers[a.get()]
                                      << ")) ? 0.0 : real(" << registers[a.get()]
                                      << "), ";
                        source_buffer << "isnan(imag(" << registers[a.get()]
                                      << ")) ? 0.0 : imag(" << registers[a.get()]
                                      << "));" << std::endl;
                    } else {
                        source_buffer << "isnan(" << registers[a.get()]
                                      << ") ? 0.0 : " << registers[a.get()]
                                      << ";" << std::endl;
                    }
                } else {
                    source_buffer << registers[a.get()] << ";" << std::endl;
                }
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
///  GPU contexts have the concept of a host side and device side buffer which
///  the CPU doesn't. Create a second map of host buffers and reference the
///  memory pointer from that. This allows one thread to run the kernel while a
///  different thread can use the results.
///
///  @params[in] node Node to get the buffer for.
//------------------------------------------------------------------------------
        T *get_buffer(graph::shared_leaf<T, SAFE_MATH> &node) {
            if (!host_buffers.contains(node.get())) {
                host_buffers[node.get()] = kernel_arguments[node.get()];
            }
            return host_buffers[node.get()].data();
        }
    };
}

#endif /* cpu_context_h */
