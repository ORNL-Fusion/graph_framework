add_test_target (node_test cpp)
add_test_target (arithmetic_test cpp)
add_test_target (math_test cpp)
add_test_target (dispersion_test cpp)
add_test_target (solver_test cpp)
add_test_target (backend_test cpp)
add_test_target (vector_test cpp)
add_test_target (physics_test cpp)
add_test_target (jit_test cpp)
add_test_target (trigonometry_test cpp)
add_test_target (piecewise_test cpp)
add_test_target (erfi_test cpp)
add_test_target (efit_test cpp)
add_test_target (random_test cpp)

target_compile_definitions (erfi_test
                            PRIVATE
                            ERFI_FILE="${CMAKE_CURRENT_SOURCE_DIR}/test_erfi.nc"
)

target_compile_definitions (efit_test
                            PRIVATE
                            EFIT_GOLD_FILE="${CMAKE_CURRENT_SOURCE_DIR}/efit_gold.nc"
)

if (${BUILD_C_BINDING})
    add_test_target (c_binding_test c)
    target_link_libraries (c_binding_test
                           PRIVATE
                           graph_c
    )
endif ()

if (${BUILD_Fortran_BINDING})
    add_test_target (f_binding_test f90)
    target_link_libraries (f_binding_test
                           PRIVATE
                           graph_f
    )
endif ()
