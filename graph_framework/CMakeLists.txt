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
                            NC_FILE="${CMAKE_CURRENT_SOURCE_DIR}/../graph_tests/efit.nc"
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
                           $<$<BOOL:${USE_PCH}>:$<$<BOOL:${USE_METAL}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/metal_context.hpp>>>
                           $<$<BOOL:${USE_PCH}>:$<$<BOOL:${USE_CUDA}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cuda_context.hpp>>>
)
