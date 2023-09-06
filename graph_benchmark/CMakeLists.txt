add_tool_target (xrays_bench)

target_link_libraries (xrays_bench

                       PUBLIC
                       $<$<PLATFORM_ID:Linux>:pthread>
)

if (${USE_PCH})
    target_precompile_headers (xrays_bench REUSE_FROM xrays)
endif ()
