find_package (NetCDF MODULE REQUIRED)

add_library (rays INTERFACE)
target_compile_features (rays
                         INTERFACE
                         cxx_std_20
)

#  Capture the system include search paths from the compiler used.
find_package (Python REQUIRED COMPONENTS Interpreter)
execute_process (COMMAND ${Python_EXECUTABLE} get_includes.py --compiler=${CMAKE_CXX_COMPILER}
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../utilities
                 OUTPUT_VARIABLE jit_include_paths
                 OUTPUT_STRIP_TRAILING_WHITESPACE
)

target_compile_definitions (rays
                            INTERFACE
                            $<$<PLATFORM_ID:Darwin>:CXX_ARGS="-I${CMAKE_CURRENT_SOURCE_DIR}${jit_include_paths} -fgnuc-version=4.2.1 -std=gnu++2a">
                            $<$<PLATFORM_ID:Linux>:CXX_ARGS="-I${CMAKE_CURRENT_SOURCE_DIR}${jit_include_paths} -std=gnu++2a -fno-use-cxa-atexit">
                            EFIT_FILE="${CMAKE_CURRENT_SOURCE_DIR}/../graph_tests/efit.nc"
                            VMEC_FILE="${CMAKE_CURRENT_SOURCE_DIR}/../graph_tests/vmec.nc"
                            $<$<BOOL:${USE_CUDA}>:HEADER_DIR="$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>">
                            $<$<BOOL:${USE_STATIC}>:STATIC>
                            $<$<BOOL:${SAVE_KERNEL_SOURCE}>:SAVE_KERNEL_SOURCE>
)

target_include_directories (rays
                            INTERFACE
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries (rays
                       INTERFACE
                       sanitizer
                       gpu_lib
                       NetCDF::NetCDF
                       $<$<PLATFORM_ID:Linux>:${CMAKE_DL_LIBS}>
                       $<$<PLATFORM_ID:Linux>:pthread>
                       llvm_dep
)
target_precompile_headers (rays
                           INTERFACE
                           $<$<BOOL:${USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/arithmetic.hpp>>
                           $<$<BOOL:${USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/backend.hpp>>
                           $<$<BOOL:${USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/dispersion.hpp>>
                           $<$<BOOL:${USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/math.hpp>>
                           $<$<BOOL:${USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/node.hpp>>
                           $<$<BOOL:${USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/solver.hpp>>
                           $<$<BOOL:${USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/trigonometry.hpp>>
                           $<$<BOOL:${USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vector.hpp>>
                           $<$<BOOL:${USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/equilibrium.hpp>>
                           $<$<BOOL:${USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/jit.hpp>>
                           $<$<BOOL:${USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/workflow.hpp>>
                           $<$<BOOL:${USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/register.hpp>>
                           $<$<BOOL:${USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/timing.hpp>>
                           $<$<BOOL:${USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpu_context.hpp>>
                           $<$<BOOL:${USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/piecewise.hpp>>
                           $<$<BOOL:${USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/newton.hpp>>
                           $<$<BOOL:${USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/special_functions.hpp>>
                           $<$<BOOL:${USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/output.hpp>>
                           $<$<BOOL:${USE_PCH}>:$<$<BOOL:${USE_METAL}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/metal_context.hpp>>>
                           $<$<BOOL:${USE_PCH}>:$<$<BOOL:${USE_CUDA}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cuda_context.hpp>>>
)
