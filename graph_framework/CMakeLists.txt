add_library (rays INTERFACE)
target_compile_features (rays
                         INTERFACE
                         cxx_std_17
)
target_compile_definitions (rays
                            INTERFACE
                            $<$<CONFIG:Release>:NDEBUG>
)
target_include_directories (rays
                            INTERFACE
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries (rays
                       INTERFACE
                       options
                       sanitizer
)
target_precompile_headers (rays
                           INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/arithmetic.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/backend_protocall.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpu_backend.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/dispersion.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/math.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/node.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/solver.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/trigonometry.hpp>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vector.hpp>
)
