//------------------------------------------------------------------------------
///  @file hip_context.hpp
///  @brief HIP context for amd based gpus.
///
///  Defines context for amd gpu.
//------------------------------------------------------------------------------

#ifndef hip_context_h
#define hip_context_h

#include <hip/hip.h>
#include <hip/hiprtc.h>

#include "node.hpp"

namespace gpu {
//------------------------------------------------------------------------------
///  @brief Check results of realtime compile.
///
///  @params[in] result Result code of the operation.
///  @params[in] name   Name of the operation.
//------------------------------------------------------------------------------
    static void check_hiprtc_error(const hiprtcResult result,
                                   const std::string &name) {
#ifndef NDEBUG
        assert(result == HIPRTC_SUCCESS && hiprtcGetErrorString(result));
#endif
    }

//------------------------------------------------------------------------------
///  @brief Check results of hip functions.
///
///  @params[in] result Result code of the operation.
///  @params[in] name   Name of the operation.
//------------------------------------------------------------------------------
    static void check_error(const hipError_t result,
                            const std::string &name) {
#ifndef NDEBUG
        assert(result == hipSuccess && hipGetErrorString(result));
#endif
    }

//------------------------------------------------------------------------------
///  @brief Initalize cuda.
//------------------------------------------------------------------------------
     static hipError_t hip_init() {
         const hipError_t result hipInit(0);
         check_error(result, "hipInit");
         return result;
     }
///  Initalize Hip.
     static const hipError_t result = hip_init();

//------------------------------------------------------------------------------
///  @brief Class representing a hip gpu context.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    class hip_context {
    private:
///  Device id
        int device;
///  Hip stream.
        hipStream_t stream;
///  Hip module.
        hipModule_t module;

    public:
//------------------------------------------------------------------------------
///  @brief Get the maximum number of concurrent instances.
///
///  @returns The maximum available concurrency.
//------------------------------------------------------------------------------
        static size_t max_concurrency() {
            int count;
            check_error(hipGetDeviceCount(&count), "hipGetDeviceCount");
            return count;
        }

//------------------------------------------------------------------------------
///  @brief Device discription.
//------------------------------------------------------------------------------
        static std::string device_type() {
            return "Hip GPU";
        }

//------------------------------------------------------------------------------
///  @brief Hip context constructor.
///
///  @params[in] index Concurrent index.
//------------------------------------------------------------------------------
        hip_context(const size_t index) :
        device(index), result_buffer(0) {
            check_error(hipSetDevice(device), "hipSetDevice");
            check_error(hipStreamCreate(&stream), "hipStreamCreate");
        }

//------------------------------------------------------------------------------
///  @brief Hip context destructor.
//------------------------------------------------------------------------------
        ~hip_context() {
            check_error(hipStreamDestroy(stream), "hipStreamDestroy");
            check_error(hipModuleUnload(module), "hipModuleUnload");
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
            if (add_reduction) {
                name.push_back("max_reduction");
            }

            hiprtcProgram kernel_program;
            check_hiprtc_error(hiprtcCreateProgram(&kernel_program,
                                                   kernel_source.c_str(),
                                                   NULL, 0, NULL, NULL),
                               "hiprtcCreateProgram");

            for (std::string &name : names) {
                check_hiprtc_error(hiprtcAddNameExpression(kernel_program,
                                                           name.c_str()));
            }

            hipDeviceProp_t device_properties;
            check_error(hipGetDeviceProperties(&device_properties, device),
                                               "hipGetDeviceProperties");

            if (jit::verbose) {
                std::cout << "HIP GPU info." << std::endl;
                std::cout << "  Major compute capability : " << device_properties.major << std::endl;
                std::cout << "  Minor compute capability : " << device_properties.minor << std::endl;
                std::cout << "  Device name              : " << device_properties.name << std::endl;
                std::cout << "  Total Global Memory      : " << device_properties.totalGlobalMem << std::endl;
                std::cout << "  Managed Memory           : " << device_properties.managedMemory << std::endl;
            }

            const std::array<const char *, 2> options({
                "-std=c++17",
                "-I " HEADER_DIR
            });

            if (hiprtcCompileProgram(kernel_program, options.size(), options.data())) {
                size_t log_size;
                check_hiprtc_error(hiprtcGetProgramLogSize(kernel_program, &log_size),
                                   "hiprtcGetProgramLogSize");

                char *log = static_cast<char *> (malloc(log_size));
                check_hiprtc_error(hiprtcGetProgramLog(kernel_program, log),
                                   "hiprtcGetProgramLog");
                std::cout << log << std::endl;
                free(log);
                std::cout << kernel_source << std::endl;
            }

            size_t code_size;
            check_hiprtc_error(hiprtcGetCodeSize(kernel_program, &code_size),
                               "hiprtcGetCodeSize");

            char *code = static_cast<char *> (malloc(code_size));
            check_hiprtc_error(hiprtcGetCode(kernel_program, code),
                               "hiprtcGetCode");

            check_error(hipModuleLoadData(&module, code),
                        "hipModuleLoadData");

            free(code);
        }
    };
}

#endif /* hip_context_h */

