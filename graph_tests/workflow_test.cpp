//------------------------------------------------------------------------------
///  @file workflow_test.cpp
///  @brief Tests for workflows.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/graph_framework.hpp"

//------------------------------------------------------------------------------
///  @brief Test setting multiple variables with the same map.
///
///  @tparam T Base type of the calculation.
///  @tparam O The @ref workflow::order
//------------------------------------------------------------------------------
template<jit::float_scalar T, workflow::order O> void test_zeros() {
    auto a = graph::variable<T> (1, "");
    auto b = graph::variable<T> (1, "");
    backend::buffer<T> buffer(1, static_cast<T> (1));
    a->set(buffer);
    b->set(buffer);

    workflow::manager<T> work(0);
    work.template add_zero_item<O> ({
        graph::variable_cast(a),
        graph::variable_cast(b)
    });

    work.compile();

    assert(work.check_value(0, a) == static_cast<T> (1) && "Expected one.");
    assert(work.check_value(0, b) == static_cast<T> (1) && "Expected one.");
    work.template run<O>();
    assert(work.check_value(0, a) == static_cast<T> (0) && "Expected zero.");
    assert(work.check_value(0, b) == static_cast<T> (0) && "Expected zero.");
}

//------------------------------------------------------------------------------
///  @brief Test setting multiple variables with the same map.
///
///  @tparam T Base type of the calculation.
///  @tparam O The @ref workflow::order
//------------------------------------------------------------------------------
template<jit::float_scalar T, workflow::order O> void test_copy() {
    auto a = graph::variable<T> (1, "");
    auto b = graph::variable<T> (1, "");
    backend::buffer<T> buffer1(1, static_cast<T> (1));
    backend::buffer<T> buffer2(1, static_cast<T> (2));
    a->set(buffer1);
    b->set(buffer2);

    workflow::manager<T> work(0);
    work.template add_copy_item<O> ({
        {graph::variable_cast(a), graph::variable_cast(b)}
    });

    work.compile();

    assert(work.check_value(0, a) == static_cast<T> (1) && "Expected one.");
    assert(work.check_value(0, b) == static_cast<T> (2) && "Expected two.");
    work.template run<O> ();
    assert(work.check_value(0, a) == static_cast<T> (1) && "Expected one.");
    assert(work.check_value(0, b) == static_cast<T> (1) && "Expected one.");
}

//------------------------------------------------------------------------------
///  @brief Test callback functions.
///
///  @tparam T Base type of the calculation.
///  @tparam O The @ref workflow::order
//------------------------------------------------------------------------------
template<jit::float_scalar T, workflow::order O> void test_callbacks() {
    int i = 1;

    workflow::manager<T> work(0);
    work.template add_callback_item<O> ([&i]() {
        i = 2;
    });

    work.compile();

    assert(i == 1 && "Expected 1");
    work.template run<O> ();
    work.wait();
    assert(i == 2 && "Expected 2");
}

//------------------------------------------------------------------------------
///  @brief Test setting multiple variables with the same map.
///
///  @tparam T Base type of the calculation.
///  @tparam O The @ref workflow::order
//------------------------------------------------------------------------------
template<jit::float_scalar T, workflow::order O> void test_maps() {
    auto a = graph::variable<T> (1, "");
    auto b = graph::variable<T> (1, "");
    backend::buffer<T> buffer(1, static_cast<T> (1));
    a->set(buffer);
    b->set(buffer);

    auto zero = graph::zero<T> ();

    workflow::manager<T> work(0);
    work.template add_item<O> ({
        graph::variable_cast(a),
        graph::variable_cast(b)
    }, {}, {
        {zero, graph::variable_cast(a)},
        {zero, graph::variable_cast(b)}
    }, NULL, "test_maps", 1);

    work.compile();

    assert(work.check_value(0, a) == static_cast<T> (1) && "Expected one.");
    assert(work.check_value(0, b) == static_cast<T> (1) && "Expected one.");
    work.template run<O> ();
    assert(work.check_value(0, a) == static_cast<T> (0) && "Expected zero.");
    assert(work.check_value(0, b) == static_cast<T> (0) && "Expected zero.");
}

//------------------------------------------------------------------------------
///  @brief Test loop items.
///
///  @tparam T Base type of the calculation.
///  @tparam O The @ref workflow::order
//------------------------------------------------------------------------------
template<jit::float_scalar T, workflow::order O> void test_loops() {
    auto a = graph::variable<T> (1, "");
    backend::buffer<T> buffer(1, static_cast<T> (0));
    a->set(buffer);

    auto a_next = a + static_cast<T> (1);
    
    workflow::manager<T> work(0);
    work.template add_loop_item<O> ({
        graph::variable_cast(a)
    }, {}, {
        {a_next, graph::variable_cast(a)}
    }, NULL, "test_maps", 1, 10);

    work.compile();

    assert(work.check_value(0, a) == static_cast<T> (0) && "Expected zero.");
    work.template run<O> ();
    assert(work.check_value(0, a) == static_cast<T> (10) && "Expected ten.");
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
///
///  @tparam T Base type of the calculation.
///  @tparam O The @ref workflow::order
//------------------------------------------------------------------------------
template<jit::float_scalar T, workflow::order O> void run_tests_order() {
    test_zeros<T, O> ();
    test_copy<T, O> ();
    test_callbacks<T, O> ();
    test_maps<T, O> ();
    test_loops<T, O> ();
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void run_tests() {
    run_tests_order<T, workflow::order::pre_run_item> ();
    run_tests_order<T, workflow::order::run_item> ();
    run_tests_order<T, workflow::order::post_run_item> ();
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    (void)argc;
    (void)argv;
    run_tests<float> ();
    run_tests<double> ();
    run_tests<std::complex<float>> ();
    run_tests<std::complex<double>> ();
}
