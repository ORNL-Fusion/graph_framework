add_library (graph_f)

target_include_directories (graph_f
                            PUBLIC
                            $<TARGET_PROPERTY:graph_f,BINARY_DIR>
)
target_link_libraries (graph_f
                       PUBLIC
                       graph_c
)
target_sources (graph_f
                PRIVATE
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/graph_fortran_binding.f90>
)

