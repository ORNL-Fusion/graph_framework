add_tool_target (xkorc)

if (${USE_PCH})
    target_precompile_headers (xrays_bench REUSE_FROM xrays)
endif ()

