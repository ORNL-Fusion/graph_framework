//------------------------------------------------------------------------------
///  @file c_binding_test.cpp
///  @brief Tests for c bindings.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <complex.h>
#include <assert.h>

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
    graph_node m = graph_constant(c_context, 0.5);
    graph_node b = graph_constant(c_context, 0.2);
    graph_node y = graph_add(c_context, graph_mul(c_context, m, x), b);

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

    graph_node inputs[1] = {x};
    graph_node outputs[5] = {y, dydx, dydm, dydb, dydy};
    graph_node *map_inputs = NULL;
    graph_node *map_outputs = NULL;

    graph_add_item(c_context,
                   inputs, 1,
                   outputs, 5,
                   map_inputs, map_outputs, 0,
                   NULL, "c_binding", 1);
    graph_compile(c_context);
    graph_run(c_context);
    graph_wait(c_context);

    switch (c_context->type) {
        case FLOAT: {
            float value[5];
            graph_copy_to_host(c_context, y, value);
            graph_copy_to_host(c_context, dydx, value + 1);
            graph_copy_to_host(c_context, dydm, value + 2);
            graph_copy_to_host(c_context, dydb, value + 3);
            graph_copy_to_host(c_context, dydy, value + 4);
            assert(value[0] == 0.5f*2.0f + 0.2f && "Value of y does not match.");
            assert(value[1] == 0.5f && "Value of dydx does not match.");
            assert(value[2] == 2.0f && "Value of dydm does not match.");
            assert(value[3] == 1.0f && "Value of dydb does not match.");
            assert(value[4] == 1.0f && "Value of dydy does not match.");
            break;
        }

        case DOUBLE: {
            double value[5];
            graph_copy_to_host(c_context, y, value);
            graph_copy_to_host(c_context, dydx, value + 1);
            graph_copy_to_host(c_context, dydm, value + 2);
            graph_copy_to_host(c_context, dydb, value + 3);
            graph_copy_to_host(c_context, dydy, value + 4);
            assert(value[0] == 0.5*2.0 + 0.2 && "Value of y does not match.");
            assert(value[1] == 0.5 && "Value of dydx does not match.");
            assert(value[2] == 2.0 && "Value of dydm does not match.");
            assert(value[3] == 1.0 && "Value of dydb does not match.");
            assert(value[4] == 1.0 && "Value of dydy does not match.");
            break;
        }

        case COMPLEX_FLOAT: {
            float complex value[5];
            graph_copy_to_host(c_context, y, value);
            graph_copy_to_host(c_context, dydx, value + 1);
            graph_copy_to_host(c_context, dydm, value + 2);
            graph_copy_to_host(c_context, dydb, value + 3);
            graph_copy_to_host(c_context, dydy, value + 4);
            assert(crealf(value[0]) == 0.5f*2.0f + 0.2f && "Value of y does not match.");
            assert(crealf(value[1]) == 0.5f && "Value of dydx does not match.");
            assert(crealf(value[2]) == 2.0f && "Value of dydm does not match.");
            assert(crealf(value[3]) == 1.0f && "Value of dydb does not match.");
            assert(crealf(value[4]) == 1.0f && "Value of dydy does not match.");
            break;
        }

        case COMPLEX_DOUBLE: {
            double complex value[5];
            graph_copy_to_host(c_context, y, value);
            graph_copy_to_host(c_context, dydx, value + 1);
            graph_copy_to_host(c_context, dydm, value + 2);
            graph_copy_to_host(c_context, dydb, value + 3);
            graph_copy_to_host(c_context, dydy, value + 4);
            assert(creal(value[0]) == 0.5*2.0 + 0.2 && "Value of y does not match.");
            assert(creal(value[1]) == 0.5 && "Value of dydx does not match.");
            assert(creal(value[2]) == 2.0 && "Value of dydm does not match.");
            assert(creal(value[3]) == 1.0 && "Value of dydb does not match.");
            assert(creal(value[4]) == 1.0 && "Value of dydy does not match.");
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
