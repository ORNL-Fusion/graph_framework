find_package (NetCDF MODULE REQUIRED)

add_library (rays INTERFACE)
target_compile_features (rays
                         INTERFACE
                         cxx_std_20
)

target_compile_definitions (rays
                            INTERFACE
                            $<$<CONFIG:Release>:NDEBUG>
                            $<$<PLATFORM_ID:Darwin>:CXX="${CMAKE_CXX_COMPILER} -isysroot ${CMAKE_OSX_SYSROOT}">
                            $<$<PLATFORM_ID:Linux>:CXX="${CMAKE_CXX_COMPILER}">
                            NC_FILE="/Users/m4c/Projects/graph_framework/graph_tests/efit.nc"
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
)
target_precompile_headers (rays
                           INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/arithmetic.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/backend.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/dispersion.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/math.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/node.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/solver.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/trigonometry.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vector.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/equilibrium.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/jit.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/workflow.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/register.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/timing.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpu_context.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/piecewise.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/newton.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/coordinates.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cublic_splines.hpp>
                           $<$<BOOL:${USE_METAL}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/metal_context.hpp>>
                           $<$<BOOL:${USE_CUDA}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cuda_context.hpp>>
)
