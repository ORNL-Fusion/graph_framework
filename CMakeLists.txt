cmake_minimum_required (VERSION 3.21)

project (rays CXX)

#-------------------------------------------------------------------------------
#  Build Options
#-------------------------------------------------------------------------------
option (USE_PCH "Enable the use of precompiled headers" ON)
option (USE_STATIC "Limits the dyamics for testing." OFF)

#-------------------------------------------------------------------------------
#  Set the cmake module path.
#-------------------------------------------------------------------------------
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake)

#-------------------------------------------------------------------------------
#  Setup build types
#-------------------------------------------------------------------------------
if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif ()
set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
              Debug
              Release
              MinSizeRel
              RelWithDebInfo
)

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

#-------------------------------------------------------------------------------
#  Sanitizer options
#-------------------------------------------------------------------------------
add_library (sanitizer INTERFACE)
target_compile_options (sanitizer
                        INTERFACE
                        $<$<CONFIG:Sanitized>:-g>
)

macro (register_sanitizer_option name)
    string (TOUPPER ${name} upper_name)

    option (SANITIZE_${upper_name} "Enable the ${name} sanitizer" OFF)

    target_compile_options (sanitizer
                            INTERFACE
                            $<$<BOOL:${SANITIZE_${upper_name}}>:-fsanitize=${name}>
    )
    target_link_options (sanitizer
                         INTERFACE
                         $<$<BOOL:${SANITIZE_${upper_name}}>:-fsanitize=${name}>
    )
endmacro ()

register_sanitizer_option (address)
register_sanitizer_option (leak)
register_sanitizer_option (memory)
register_sanitizer_option (thread)
register_sanitizer_option (undefined)
register_sanitizer_option (float-divide-by-zero)

#-------------------------------------------------------------------------------
#  Setup access method.
#-------------------------------------------------------------------------------
option (USE_SSH "Use ssh to access git repos" OFF)
if (${USE_SSH})
    set (URL_PROTO "git@")
    set (URL_SEP ":")
else ()
    set (URL_PROTO "https://")
    set (URL_SEP "/")
endif ()

#-------------------------------------------------------------------------------
#  Define a macro to register new projects.
#-------------------------------------------------------------------------------
include (FetchContent)
find_package (Git)

function (register_project reg_name dir url default_tag sub_dir)
    set (BUILD_TAG_${dir} ${default_tag} CACHE STRING "Name of the tag to checkout.")

# Anything after the tag is assumed to be a project setting.
    foreach (i RANGE 6 ${ARGC} 3)
        math (EXPR key "${i} - 1" OUTPUT_FORMAT DECIMAL)
        math (EXPR type "${i} + 1" OUTPUT_FORMAT DECIMAL)
        set (${ARGV${key}} ${ARGV${i}} CACHE ${ARGV${type}} "" FORCE)
    endforeach ()

    FetchContent_Declare (
        ${reg_name}
        GIT_REPOSITORY ${url}
        GIT_TAG origin/${BUILD_TAG_${dir}}
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${dir}
        SOURCE_SUBDIR ${sub_dir}
    )

    FetchContent_MakeAvailable (${reg_name})

    if (${GIT_FOUND})
#  By default cmake clones projects in a headless state. After the repo is
#  cloned checkout the requested tag so repo is in a working state.
        execute_process (
            COMMAND ${GIT_EXECUTABLE} checkout ${BUILD_TAG_${dir}}
            WORKING_DIRECTORY ${${reg_name}_SOURCE_DIR}
        )

#  Add a taraget to pull the latest version before building. Note dependency is
#  registered in the sub project CMakeList.txt. Not sure how this should handle
#  multiple targets in a project yet. Name must match the subproject pull_
#  dependency.
        add_custom_target (
            pull_${reg_name}
            ALL
            COMMAND ${GIT_EXECUTABLE} pull
            WORKING_DIRECTORY ${${reg_name}_SOURCE_DIR}
        )
    endif ()
endfunction ()

register_project (llvm
                  LLVM
                  ${URL_PROTO}github.com${URL_SEP}llvm/llvm-project.git
                  main
                  llvm
                  LLVM_TARGETS_TO_BUILD Native STRING
                  LLVM_ENABLE_PROJECTS clang STRING
)

#-------------------------------------------------------------------------------
#  Define an llvm dependency target
#-------------------------------------------------------------------------------
FetchContent_GetProperties (
    llvm
    BINARY_DIR llvm_binary_dir
    SOURCE_DIR llvm_source_dir
)

#  Do not build llvm until pull command is finished.
add_dependencies (LLVMHello pull_llvm)
add_dependencies (LLVMHello_exports pull_llvm)
add_dependencies (LLVMDemangle pull_llvm)
add_dependencies (Remarks_exports pull_llvm)
add_dependencies (LLVMSupportBlake3 pull_llvm)
add_dependencies (BugpointPasses_exports pull_llvm)
add_dependencies (LTO_exports pull_llvm)
add_dependencies (llvm-PerfectShuffle pull_llvm)
add_dependencies (SecondLib pull_llvm)
add_dependencies (DynamicLibraryLib pull_llvm)
add_dependencies (PipSqueak pull_llvm)
add_dependencies (obj.LLVMTableGenCommon pull_llvm)
add_dependencies (llvm_vcsrevision_h pull_llvm)
add_dependencies (benchmark pull_llvm)
add_dependencies (benchmark_main pull_llvm)
add_dependencies (obj.LLVMTableGenBasic pull_llvm)
add_dependencies (core-resource-headers pull_llvm)
add_dependencies (mips-resource-headers pull_llvm)
add_dependencies (utility-resource-headers pull_llvm)
add_dependencies (ppc-htm-resource-headers pull_llvm)
add_dependencies (loongarch-resource-headers pull_llvm)
add_dependencies (systemz-resource-headers pull_llvm)
add_dependencies (hexagon-resource-headers pull_llvm)
add_dependencies (webassembly-resource-headers pull_llvm)
add_dependencies (riscv-resource-headers pull_llvm)
add_dependencies (hlsl-resource-headers pull_llvm)
add_dependencies (ve-resource-headers pull_llvm)
add_dependencies (opencl-resource-headers pull_llvm)
add_dependencies (llvm-libc-resource-headers pull_llvm)
add_dependencies (openmp-resource-headers pull_llvm)
add_dependencies (windows-resource-headers pull_llvm)
add_dependencies (SampleAnalyzerPlugin_exports pull_llvm)
add_dependencies (CheckerDependencyHandlingAnalyzerPlugin_exports pull_llvm)
add_dependencies (CheckerOptionHandlingAnalyzerPlugin_exports pull_llvm)
add_dependencies (clangAnalysisFlowSensitiveResources pull_llvm)
add_dependencies (ppc-resource-headers pull_llvm)
add_dependencies (hmaptool pull_llvm)
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_dependencies (libclang_exports pull_llvm)
endif ()
add_dependencies (hip-resource-headers pull_llvm)
add_dependencies (scan-view pull_llvm)
add_dependencies (scan-build pull_llvm)
add_dependencies (cuda-resource-headers pull_llvm)
add_dependencies (scan-build-py pull_llvm)
add_dependencies (x86-resource-headers pull_llvm)
add_dependencies (obj.clangSupport pull_llvm)

add_library (llvm_dep INTERFACE)
target_include_directories (llvm_dep
                            INTERFACE
                            $<BUILD_INTERFACE:${llvm_source_dir}/llvm/include>
                            $<BUILD_INTERFACE:${llvm_source_dir}/clang/include>
                            $<BUILD_INTERFACE:${llvm_binary_dir}/include>
                            $<BUILD_INTERFACE:${llvm_binary_dir}/tools/clang/include>
)

#  Need to manually figure out the native arch.
string(REGEX MATCH "^[^-]*" LLVM_NATIVE_ARCH ${LLVM_HOST_TRIPLE})
if (LLVM_NATIVE_ARCH MATCHES "arm64")
    set (LLVM_NATIVE_ARCH AArch64)
elseif (LLVM_NATIVE_ARCH MATCHES "x86_64")
    set (LLVM_NATIVE_ARCH X86)
else ()
   message (FATAL_ERROR "Unknown native arch ${LLVM_NATIVE_ARCH}")
endif ()

target_link_libraries (llvm_dep
                       INTERFACE
                       clangCodeGen
                       LLVM${LLVM_NATIVE_ARCH}CodeGen
                       LLVMOrcJIT
)

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
add_subdirectory (graph_benchmark)

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

    if (${USE_PCH})
        target_precompile_headers (${target} REUSE_FROM xrays)
    endif ()
endmacro ()

add_subdirectory (graph_tests)
