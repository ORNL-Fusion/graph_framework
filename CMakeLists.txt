cmake_minimum_required (VERSION 3.21)

project (rays CXX)

#-------------------------------------------------------------------------------
#  Build Options
#-------------------------------------------------------------------------------
option (USE_FMA "Enable Fused multiply add nodes." OFF)

add_library (options INTERFACE)
target_compile_definitions (options
                            INTERFACE
                            $<$<BOOL:${USE_FMA}>:USE_FMA>
)

#-------------------------------------------------------------------------------
#  Setup build types
#-------------------------------------------------------------------------------
set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
              Debug
              Release
              Sanitized
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
