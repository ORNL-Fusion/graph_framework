cmake_minimum_required (VERSION 3.21)

project (rays CXX)

#-------------------------------------------------------------------------------
#  Setup build types
#-------------------------------------------------------------------------------
set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
              Debug
              Release
              Sanitized
)

option (USE_REDUCE "Enable the graph reduction" ON)

#-------------------------------------------------------------------------------
#  GPU config
#-------------------------------------------------------------------------------

if (${APPLE})
    option (USE_METAL "Enable the metal backend" OFF)

    if (${USE_METAL})
        enable_language (OBJCXX)

        add_library (metal_lib INTERFACE)
        target_link_libraries (metal_lib
                               INTERFACE
                               "-framework Metal"
                               "-framework Foundation"
        )

        target_compile_definitions (metal_lib
                                    INTERFACE
                                    USE_METAL
        )
        target_compile_options (metal_lib
                                INTERFACE
                                -fobjc-arc
        )
    endif ()
else ()
    option (USE_CUDA "Enable the cuda backend" OFF)

    if (${USE_CUDA})
        add_library (cuda_lib INTERFACE)

        find_package (CUDAToolkit REQUIRED)

        target_compile_definitions (cuda_lib
                                    INTERFACE
                                    USE_CUDA
                                    CUDA_INCLUDE="${CUDAToolkit_INCLUDE_DIRS}"
        )
        target_link_libraries (cuda_lib
                               INTERFACE
                               $<$<BOOL:${CUDAToolkit_FOUND}>:CUDA::cuda_driver>
                               $<$<BOOL:${CUDAToolkit_FOUND}>:CUDA::nvrtc>
        )
    endif ()
endif ()

add_library (gpu_lib INTERFACE)
target_link_libraries (gpu_lib
                       INTERFACE
                       $<$<BOOL:${USE_METAL}>:metal_lib>
                       $<$<BOOL:${USE_CUDA}>:cuda_lib>
)
target_compile_definitions (gpu_lib
                            INTERFACE
                            $<$<BOOL:${USE_METAL}>:USE_GPU>
                            $<$<BOOL:${USE_CUDA}>:USE_GPU>
)

#-------------------------------------------------------------------------------
#  Sanitizer options
#-------------------------------------------------------------------------------
add_library (sanitizer INTERFACE)
target_compile_options (sanitizer
                        INTERFACE
                        $<$<CONFIG:Sanitized>:-g>
)

macro (register_sanitizer_option name default)
    string (TOUPPER ${name} upper_name)

    option (SANITIZE_${upper_name} "Enable the ${name} sanitizer" ${default})

    target_compile_options (sanitizer
                            INTERFACE
                            $<$<CONFIG:Sanitized>:$<$<BOOL:${SANITIZE_${upper_name}}>:-fsanitize=${name}>>
    )
    target_link_options (sanitizer
                         INTERFACE
                         $<$<CONFIG:Sanitized>:$<$<BOOL:${SANITIZE_${upper_name}}>:-fsanitize=${name}>>
    )
endmacro ()

register_sanitizer_option (address ON)
register_sanitizer_option (leak OFF)
register_sanitizer_option (memory OFF)
register_sanitizer_option (thread OFF)
register_sanitizer_option (undefined ON)
register_sanitizer_option (float-divide-by-zero ON)

#-------------------------------------------------------------------------------
#  Setup targets
#-------------------------------------------------------------------------------

add_subdirectory (graph_framework)

#-------------------------------------------------------------------------------
#  Tool setup
#-------------------------------------------------------------------------------
macro (add_tool_target target)
    add_executable (${target})
    target_sources (${target}
                    PRIVATE
                    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${target}.cpp>
    )
    
    if (${USE_METAL})
        set_source_files_properties (${CMAKE_CURRENT_SOURCE_DIR}/${target}.cpp
                                     PROPERTIES
                                     LANGUAGE OBJCXX
        )
    endif ()
    
    target_link_libraries (${target}
                           PUBLIC
                           rays
    )
endmacro ()

add_subdirectory (graph_driver)

#-------------------------------------------------------------------------------
#  Setup testing
#-------------------------------------------------------------------------------
enable_testing ()

#-------------------------------------------------------------------------------
#  Define macro function to register tests.
#-------------------------------------------------------------------------------
macro (add_test_target target)
    add_tool_target (${target})

    add_test (NAME ${target}
              COMMAND ${target}
    )

    target_precompile_headers (${target} REUSE_FROM xrays)
endmacro ()

add_subdirectory (graph_tests)
