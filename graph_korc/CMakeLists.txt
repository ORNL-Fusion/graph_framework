add_tool_target (xkorc)

if (${USE_PCH})
    target_precompile_headers (xkorc REUSE_FROM xrays)
endif ()

