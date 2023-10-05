add_tool_target (xrays_bench)

if (${USE_PCH})
    target_precompile_headers (xrays_bench REUSE_FROM xrays)
endif ()
