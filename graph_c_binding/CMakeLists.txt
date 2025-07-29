add_library (graph_c)

target_include_directories (graph_c
                            PUBLIC
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries (graph_c
                       PUBLIC
                       graph_framework
)
target_sources (graph_c
                PRIVATE
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/graph_c_binding.cpp>
)
set_source_files_properties (${CMAKE_CURRENT_SOURCE_DIR}/graph_c_binding.cpp
                             PROPERTIES
                             LANGUAGE OBJCXX
)

