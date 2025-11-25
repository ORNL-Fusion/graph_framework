//------------------------------------------------------------------------------
///  @file c_binding_test.c
///  @brief Tests for c bindings.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <complex.h>
#include <assert.h>
#include <stdio.h>

#include "../graph_c_binding/graph_c_binding.h"

//------------------------------------------------------------------------------
///  @brief Run tests
///
///  @param[in] type          Type to run tests on.
///  @param[in] use_safe_math Use safe math.
//------------------------------------------------------------------------------
void run_tests(const enum graph_type type,
               const bool use_safe_math) {
    struct graph_c_context *c_context = graph_construct_context(type, use_safe_math);

    graph_node x = graph_variable(c_context, 1, "x");
    graph_node m;
    graph_node b;
    if (type == FLOAT || type == DOUBLE) {
        m = graph_constant(c_context, 0.5);
        b = graph_constant(c_context, 0.2);
    } else {
        m = graph_constant_c(c_context, 0.5, 0.0);
        b = graph_constant_c(c_context, 0.2, 0.0);
    }
    graph_node y = graph_add(c_context, graph_mul(c_context, m, x), b);

    graph_node px = graph_pseudo_variable(c_context, x);
    assert(graph_remove_pseudo(c_context, px) == x &&
           "Expected to receive x.");

    graph_node one = graph_constant(c_context, 1.0);
    graph_node zero = graph_constant(c_context, 0.0);
    assert(graph_sub(c_context, one, one) == zero &&
           "Expected to receive zero.");
    assert(graph_div(c_context, one, one) == one &&
           "Expected to receive one.");
    assert(graph_sqrt(c_context, one) == one &&
           "Expected to receive one.");
    assert(graph_exp(c_context, zero) == one &&
           "Expected to receive one.");
    assert(graph_log(c_context, one) == zero &&
           "Expected to receive zero.");
    assert(graph_pow(c_context, one, one) == one &&
           "Expected to receive one.");

    if (type == COMPLEX_FLOAT || type == COMPLEX_DOUBLE) {
        assert(graph_erfi(c_context, zero) == zero &&
               "Expected to receive zero.");
    }

    assert(graph_sin(c_context, zero) == zero &&
           "Expected to receive zero.");
    assert(graph_cos(c_context, zero) == one &&
           "Expected to receive one.");
    assert(graph_atan(c_context, one, zero) == zero &&
           "Expected to receive zero.");

    graph_node dydx = graph_df(c_context, y, x);
    graph_node dydm = graph_df(c_context, y, m);
    graph_node dydb = graph_df(c_context, y, b);
    graph_node dydy = graph_df(c_context, y, y);

    switch (c_context->type) {
        case FLOAT: {
            float value = 2.0;
            graph_set_variable(c_context, x, &value);
            break;
        }

        case DOUBLE: {
            double value = 2.0;
            graph_set_variable(c_context, x, &value);
            break;
        }

        case COMPLEX_FLOAT: {
            float complex value = CMPLXF(2.0, 0.0);
            graph_set_variable(c_context, x, &value);
            break;
        }

        case COMPLEX_DOUBLE: {
            double complex value = CMPLX(2.0, 0.0);
            graph_set_variable(c_context, x, &value);
            break;
        }
    }

    graph_node state = graph_random_state(c_context, 0);
    graph_node rand = graph_random(c_context, state);

    const size_t max_device = graph_get_max_concurrency(c_context) - 1;
    graph_set_device_number(c_context, max_device);

    graph_node inputs[1] = {x};
    graph_node outputs[5] = {y, dydx, dydm, dydb, dydy};
    graph_node *map_inputs = NULL;
    graph_node *map_outputs = NULL;

    graph_node z = graph_variable(c_context, 1, "z");
    graph_node root = graph_sub(c_context,
                                graph_pow(c_context, z,
                                          graph_constant(c_context, 3.0)),
                                graph_pow(c_context, z,
                                          graph_constant(c_context, 2.0)));
    graph_node root2 = graph_mul(c_context, root, root);
    graph_node dz = graph_sub(c_context, z,
                              graph_div(c_context, root,
                                        graph_df(c_context, root, z)));

    graph_node p1;
    graph_node p2;
    graph_node i = graph_variable(c_context, 1, "i");
    graph_node j = graph_variable(c_context, 1, "j");
    switch (c_context->type) {
        case FLOAT: {
            float value1[3] = {2.0, 4.0, 6.0};
            float value2[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
            float value3 = 1.5;
            float value4 = 2.5;
            graph_set_variable(c_context, i, &value3);
            graph_set_variable(c_context, j, &value4);
            p1 = graph_piecewise_1D(c_context, i, 1.0, 0.0, value1, 3);
            p2 = graph_piecewise_2D(c_context, 3, j, 1.0, 0.0, i, 1.0, 0.0, value2, 9);
            break;
        }

        case DOUBLE: {
            double value1[3] = {2.0, 4.0, 6.0};
            double value2[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
            double value3 = 1.5;
            double value4 = 2.5;
            graph_set_variable(c_context, i, &value3);
            graph_set_variable(c_context, j, &value4);
            p1 = graph_piecewise_1D(c_context, i, 1.0, 0.0, value1, 3);
            p2 = graph_piecewise_2D(c_context, 3, j, 1.0, 0.0, i, 1.0, 0.0, value2, 9);
            break;
        }

        case COMPLEX_FLOAT: {
            float complex value1[3] = {CMPLXF(2.0, 0.0), CMPLXF(4.0, 0.0), CMPLXF(6.0, 0.0)};
            float complex value2[9] = {CMPLXF(1.0, 0.0), CMPLXF(2.0, 0.0), CMPLXF(3.0, 0.0),
                                       CMPLXF(4.0, 0.0), CMPLXF(5.0, 0.0), CMPLXF(6.0, 0.0),
                                       CMPLXF(7.0, 0.0), CMPLXF(8.0, 0.0), CMPLXF(9.0, 0.0)};
            float complex value3 = CMPLXF(1.5, 0.0);
            float complex value4 = CMPLXF(2.5, 0.0);
            graph_set_variable(c_context, i, &value3);
            graph_set_variable(c_context, j, &value4);
            p1 = graph_piecewise_1D(c_context, i, 1.0, 0.0, value1, 3);
            p2 = graph_piecewise_2D(c_context, 3, j, 1.0, 0.0, i, 1.0, 0.0, value2, 9);
            break;
        }

        case COMPLEX_DOUBLE: {
            double complex value1[3] = {CMPLX(2.0, 0.0), CMPLX(4.0, 0.0), CMPLX(6.0, 0.0)};
            double complex value2[9] = {CMPLX(1.0, 0.0), CMPLX(2.0, 0.0), CMPLX(3.0, 0.0),
                                        CMPLX(4.0, 0.0), CMPLX(5.0, 0.0), CMPLX(6.0, 0.0),
                                        CMPLX(7.0, 0.0), CMPLX(8.0, 0.0), CMPLX(9.0, 0.0)};
            double complex value3 = CMPLXF(1.5, 0.0);
            double complex value4 = CMPLXF(2.5, 0.0);
            graph_set_variable(c_context, i, &value3);
            graph_set_variable(c_context, j, &value4);
            p1 = graph_piecewise_1D(c_context, i, 1.0, 0.0, value1, 3);
            p2 = graph_piecewise_2D(c_context, 3, j, 1.0, 0.0, i, 1.0, 0.0, value2, 9);
            break;
        }
    }

    graph_node inputs2[2] = {i, j};
    graph_node outputs2[2] = {p1, p2};
    graph_node *map_inputs2 = NULL;
    graph_node *map_outputs2 = NULL;

    graph_add_pre_item(c_context,
                       NULL, 0,
                       &rand, 1,
                       NULL, NULL, 0,
                       state,
                       "c_binding_pre_kernel", 1);
    graph_add_item(c_context,
                   inputs, 1,
                   outputs, 5,
                   map_inputs, map_outputs, 0,
                   NULL, "c_binding", 1);
    graph_add_item(c_context,
                   inputs2, 2,
                   outputs2, 2,
                   map_inputs2, map_outputs2, 0,
                   NULL, "c_binding_piecewise", 1);
    graph_add_converge_item(c_context, &z, 1,
                            &root2, 1,
                            &z, &dz, 1,
                            NULL, "c_binding_converge", 1,
                            1.0E-30, 1000);
    graph_compile(c_context);
    switch (c_context->type) {
        case FLOAT: {
            float value = 10.0;
            graph_copy_to_device(c_context, z, &value);
            break;
        }

        case DOUBLE: {
            double value = 10.0;
            graph_copy_to_device(c_context, z, &value);
            break;
        }

        case COMPLEX_FLOAT: {
            float complex value = CMPLXF(10.0, 0.0);
            graph_copy_to_device(c_context, z, &value);
            break;
        }

        case COMPLEX_DOUBLE: {
            double complex value = CMPLX(10.0, 0.0);
            graph_copy_to_device(c_context, z, &value);
            break;
        }
    }
    graph_pre_run(c_context);
    graph_run(c_context);
    graph_wait(c_context);
    inputs2[0] = z;
    inputs2[1] = y;
    graph_print(c_context, 0, inputs2, 2);

    switch (c_context->type) {
        case FLOAT: {
            float value[9];
            graph_copy_to_host(c_context, y, value);
            graph_copy_to_host(c_context, dydx, value + 1);
            graph_copy_to_host(c_context, dydm, value + 2);
            graph_copy_to_host(c_context, dydb, value + 3);
            graph_copy_to_host(c_context, dydy, value + 4);
            graph_copy_to_host(c_context, rand, value + 5);
            graph_copy_to_host(c_context, z, value + 6);
            graph_copy_to_host(c_context, p1, value + 7);
            graph_copy_to_host(c_context, p2, value + 8);
            assert(value[0] == 0.5f*2.0f + 0.2f && "Value of y does not match.");
            assert(value[1] == 0.5f && "Value of dydx does not match.");
            assert(value[2] == 2.0f && "Value of dydm does not match.");
            assert(value[3] == 1.0f && "Value of dydb does not match.");
            assert(value[4] == 1.0f && "Value of dydy does not match.");
            if (c_context->safe_math) {
                assert(value[5] == 2546248192.0f && "Value of rand does not match.");
            } else {
                assert(value[5] == 2357136128.0f && "Value of rand does not match.");
            }
            assert(value[6] == 1.0f && "Value of root does not match.");
            assert(value[7] == 4.0f && "Value of p1 does not match.");
            assert(value[8] == 8.0f && "Value of p2 does not match.");
            break;
        }

        case DOUBLE: {
            double value[9];
            graph_copy_to_host(c_context, y, value);
            graph_copy_to_host(c_context, dydx, value + 1);
            graph_copy_to_host(c_context, dydm, value + 2);
            graph_copy_to_host(c_context, dydb, value + 3);
            graph_copy_to_host(c_context, dydy, value + 4);
            graph_copy_to_host(c_context, rand, value + 5);
            graph_copy_to_host(c_context, z, value + 6);
            graph_copy_to_host(c_context, p1, value + 7);
            graph_copy_to_host(c_context, p2, value + 8);
            assert(value[0] == 0.5*2.0 + 0.2 && "Value of y does not match.");
            assert(value[1] == 0.5 && "Value of dydx does not match.");
            assert(value[2] == 2.0 && "Value of dydm does not match.");
            assert(value[3] == 1.0 && "Value of dydb does not match.");
            assert(value[4] == 1.0 && "Value of dydy does not match.");
            if (c_context->safe_math) {
                assert(value[5] == 2546248239.0 && "Value of rand does not match.");
            } else {
                assert(value[5] == 2357136044.0 && "Value of rand does not match.");
            }
            assert(value[6] == 1.0 && "Value of root does not match.");
            assert(value[7] == 4.0 && "Value of p1 does not match.");
            assert(value[8] == 8.0 && "Value of p2 does not match.");
            break;
        }

        case COMPLEX_FLOAT: {
            float complex value[9];
            graph_copy_to_host(c_context, y, value);
            graph_copy_to_host(c_context, dydx, value + 1);
            graph_copy_to_host(c_context, dydm, value + 2);
            graph_copy_to_host(c_context, dydb, value + 3);
            graph_copy_to_host(c_context, dydy, value + 4);
            graph_copy_to_host(c_context, rand, value + 5);
            graph_copy_to_host(c_context, z, value + 6);
            graph_copy_to_host(c_context, p1, value + 7);
            graph_copy_to_host(c_context, p2, value + 8);
            assert(crealf(value[0]) == 0.5f*2.0f + 0.2f && "Value of y does not match.");
            assert(crealf(value[1]) == 0.5f && "Value of dydx does not match.");
            assert(crealf(value[2]) == 2.0f && "Value of dydm does not match.");
            assert(crealf(value[3]) == 1.0f && "Value of dydb does not match.");
            assert(crealf(value[4]) == 1.0f && "Value of dydy does not match.");
            if (c_context->safe_math) {
                assert(crealf(value[5]) == 2546248192.0f && "Value of rand does not match.");
            } else {
                assert(crealf(value[5]) == 2357136128.0f && "Value of rand does not match.");
            }
            assert(crealf(value[6]) == 1.0f && "Value of root does not match.");
            assert(crealf(value[7]) == 4.0f && "Value of p1 does not match.");
            assert(crealf(value[8]) == 8.0f && "Value of p2 does not match.");
            break;
        }

        case COMPLEX_DOUBLE: {
            double complex value[9];
            graph_copy_to_host(c_context, y, value);
            graph_copy_to_host(c_context, dydx, value + 1);
            graph_copy_to_host(c_context, dydm, value + 2);
            graph_copy_to_host(c_context, dydb, value + 3);
            graph_copy_to_host(c_context, dydy, value + 4);
            graph_copy_to_host(c_context, rand, value + 5);
            graph_copy_to_host(c_context, z, value + 6);
            graph_copy_to_host(c_context, p1, value + 7);
            graph_copy_to_host(c_context, p2, value + 8);
            assert(creal(value[0]) == 0.5*2.0 + 0.2 && "Value of y does not match.");
            assert(creal(value[1]) == 0.5 && "Value of dydx does not match.");
            assert(creal(value[2]) == 2.0 && "Value of dydm does not match.");
            assert(creal(value[3]) == 1.0 && "Value of dydb does not match.");
            assert(creal(value[4]) == 1.0 && "Value of dydy does not match.");
            if (c_context->safe_math) {
                assert(creal(value[5]) == 2546248239.0 && "Value of rand does not match.");
            } else {
                assert(creal(value[5]) == 2357136044.0 && "Value of rand does not match.");
            }
            assert(creal(value[6]) == 1.0 && "Value of root does not match.");
            assert(creal(value[7]) == 4.0 && "Value of p1 does not match.");
            assert(creal(value[8]) == 8.0 && "Value of p2 does not match.");
            break;
        }
    }

    graph_destroy_context(c_context);
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    START_GPU
    (void)argc;
    (void)argv;

    run_tests(FLOAT, false);
    run_tests(FLOAT, true);
    run_tests(DOUBLE, false);
    run_tests(DOUBLE, true);
    run_tests(COMPLEX_FLOAT, false);
    run_tests(COMPLEX_FLOAT, true);
    run_tests(COMPLEX_DOUBLE, false);
    run_tests(COMPLEX_DOUBLE, true);

    END_GPU
}
