add_tool_target (xplayground cpp)

if (${USE_PCH})
    target_precompile_headers (xplayground REUSE_FROM xrays)
endif ()
