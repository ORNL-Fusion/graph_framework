cmake_minimum_required (VERSION 3.21)

project (rays CXX)

#-------------------------------------------------------------------------------
#  Build Options
#-------------------------------------------------------------------------------
option (USE_PCH "Enable the use of precompiled headers" ON)
option (SAVE_KERNEL_SOURCE "Writes the kernel source code to a file." OFF)
option (USE_INPUT_CACHE "Cache the values kernel input values." OFF)
option (USE_CONSTANT_CACHE "Cache the value of constantants in kernel registers." OFF)
option (SHOW_USE_COUNT "Add a comment showing the use count in kernel sources." OFF)
option (USE_INDEX_CACHE "Cache index values instead of computing them every time." OFF)

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
    option (USE_CUDA "Enable the cuda backend" ON)

    if (${USE_CUDA})
        add_library (cuda_lib INTERFACE)

        find_package (CUDAToolkit REQUIRED)

        option (USE_CUDA_TEXTURES "Enable the use of cuda textures" OFF)

        target_compile_definitions (cuda_lib
                                    INTERFACE
                                    USE_CUDA
                                    $<$<BOOL:${USE_CUDA_TEXTURES}>:USE_CUDA_TEXTURES>
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
                            $<$<BOOL:${USE_INPUT_CACHE}>:USE_INPUT_CACHE>
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

# Modified function to handle hard-coded LLVM commit
function (register_project reg_name dir url default_tag sub_dir)
    # Special handling for LLVM with hard-coded commit
    if (${reg_name} STREQUAL "llvm")
        # Hard-code the specific commit for LLVM
        set (LLVM_COMMIT "a6494a3bbf0620ec472d44db4e79c4c508618a61")
        
        FetchContent_Declare (
            ${reg_name}
            GIT_REPOSITORY ${url}
            GIT_TAG ${LLVM_COMMIT}
            SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${dir}
            SOURCE_SUBDIR ${sub_dir}
        )
        
        FetchContent_MakeAvailable (${reg_name})
        
        if (${GIT_FOUND})
            # Checkout the specific commit to ensure we're on it
            execute_process (
                COMMAND ${GIT_EXECUTABLE} checkout ${LLVM_COMMIT}
                WORKING_DIRECTORY ${${reg_name}_SOURCE_DIR}
            )
            
            # Add a target to pull, but for LLVM it will just verify we're on the right commit
            add_custom_target (
                pull_${reg_name}
                ALL
                COMMAND ${GIT_EXECUTABLE} fetch
                COMMAND ${GIT_EXECUTABLE} checkout ${LLVM_COMMIT}
                WORKING_DIRECTORY ${${reg_name}_SOURCE_DIR}
            )
        endif ()
        
        # Handle LLVM-specific project settings
        set (LLVM_TARGETS_TO_BUILD Native CACHE STRING "" FORCE)
        set (LLVM_ENABLE_PROJECTS clang CACHE STRING "" FORCE)
    else ()
        # Original behavior for non-LLVM projects
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
            # By default cmake clones projects in a headless state. After the repo is
            # cloned checkout the requested tag so repo is in a working state.
            execute_process (
                COMMAND ${GIT_EXECUTABLE} checkout ${BUILD_TAG_${dir}}
                WORKING_DIRECTORY ${${reg_name}_SOURCE_DIR}
            )
            
            # Add a target to pull the latest version before building. Note dependency is
            # registered in the sub project CMakeList.txt. Not sure how this should handle
            # multiple targets in a project yet. Name must match the subproject pull_
            # dependency.
            add_custom_target (
                pull_${reg_name}
                ALL
                COMMAND ${GIT_EXECUTABLE} pull
                WORKING_DIRECTORY ${${reg_name}_SOURCE_DIR}
            )
        endif ()
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
#  Conditionally add dependencies only if targets exist
macro (add_llvm_dependency target)
    if (TARGET ${target})
        add_dependencies (${target} pull_llvm)
    endif ()
endmacro ()

add_llvm_dependency (gpu-resource-headers)
add_llvm_dependency (llvm-offload-resource-headers)
add_llvm_dependency (LLVMDemangle)
add_llvm_dependency (Remarks_exports)
add_llvm_dependency (LLVMSupportBlake3)
add_llvm_dependency (BugpointPasses_exports)
add_llvm_dependency (LTO_exports)
add_llvm_dependency (llvm-PerfectShuffle)
add_llvm_dependency (SecondLib)
add_llvm_dependency (DynamicLibraryLib)
add_llvm_dependency (PipSqueak)
add_llvm_dependency (obj.LLVMTableGenCommon)
add_llvm_dependency (llvm_vcsrevision_h)
add_llvm_dependency (benchmark)
add_llvm_dependency (benchmark_main)
add_llvm_dependency (obj.LLVMTableGenBasic)
add_llvm_dependency (core-resource-headers)
add_llvm_dependency (mips-resource-headers)
add_llvm_dependency (utility-resource-headers)
add_llvm_dependency (ppc-htm-resource-headers)
add_llvm_dependency (loongarch-resource-headers)
add_llvm_dependency (systemz-resource-headers)
add_llvm_dependency (hexagon-resource-headers)
add_llvm_dependency (webassembly-resource-headers)
add_llvm_dependency (riscv-resource-headers)
add_llvm_dependency (hlsl-resource-headers)
add_llvm_dependency (ve-resource-headers)
add_llvm_dependency (opencl-resource-headers)
add_llvm_dependency (llvm-libc-resource-headers)
add_llvm_dependency (openmp-resource-headers)
add_llvm_dependency (windows-resource-headers)
add_llvm_dependency (SampleAnalyzerPlugin_exports)
add_llvm_dependency (CheckerDependencyHandlingAnalyzerPlugin_exports)
add_llvm_dependency (CheckerOptionHandlingAnalyzerPlugin_exports)
add_llvm_dependency (clangAnalysisFlowSensitiveResources)
add_llvm_dependency (ppc-resource-headers)
add_llvm_dependency (hmaptool)
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_llvm_dependency (libclang_exports)
endif ()
add_llvm_dependency (hip-resource-headers)
add_llvm_dependency (scan-view)
add_llvm_dependency (scan-build)
add_llvm_dependency (cuda-resource-headers)
add_llvm_dependency (scan-build-py)
add_llvm_dependency (x86-resource-headers)
add_llvm_dependency (obj.clangSupport)
add_llvm_dependency (arm-common-resource-headers)
add_llvm_dependency (arm-resource-headers)
add_llvm_dependency (aarch64-resource-headers)

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
                       LLVMOrcDebugging
                       LLVMOrcTargetProcess
)

#-------------------------------------------------------------------------------
#  Build Documentation
#-------------------------------------------------------------------------------
find_package(Doxygen)

if (DOXYGEN_FOUND)
    set (DOXYGEN_PROJECT_NAME "Graph Framework")
    set (DOXYGEN_EXCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/LLVM ${CMAKE_CURRENT_SOURCE_DIR}/build)
    set (DOXYGEN_GENERATE_TREEVIEW YES)

    doxygen_add_docs (doc)
endif ()

#-------------------------------------------------------------------------------
#  Setup targets
#-------------------------------------------------------------------------------

add_subdirectory (graph_framework)

#-------------------------------------------------------------------------------
#  Setup testing
#-------------------------------------------------------------------------------
enable_testing ()

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
add_subdirectory (graph_playground)
add_subdirectory (graph_korc)



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