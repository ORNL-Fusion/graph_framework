add_tool_target (xrays)

target_link_libraries (xrays

                       PUBLIC
                       $<$<PLATFORM_ID:Linux>:pthread>
)
