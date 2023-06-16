add_test_target (node_test)
add_test_target (arithmetic_test)
add_test_target (math_test)
add_test_target (dispersion_test)
add_test_target (solver_test)
add_test_target (backend_test)
add_test_target (vector_test)
add_test_target (physics_test)
add_test_target (jit_test)
add_test_target (trigonometry_test)
add_test_target (piecewise_test)
add_test_target (erfi_test)

target_compile_definitions (erfi_test
                            PRIVATE
                            ERFI_FILE="${CMAKE_CURRENT_SOURCE_DIR}/test_erfi.nc"
)
