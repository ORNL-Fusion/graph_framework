add_tool_target (xrays_bench  cpp)

if (${USE_PCH} AND NOT ${BUILD_C_BINDING})
    target_precompile_headers (xrays_bench REUSE_FROM xrays)
endif ()
