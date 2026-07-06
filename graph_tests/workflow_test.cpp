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
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_prezeros() {
    auto a = graph::variable<T> (1, "");
    auto b = graph::variable<T> (1, "");
    backend::buffer<T> buffer(1, static_cast<T> (1));
    a->set(buffer);
    b->set(buffer);

    workflow::manager<T> work(0);
    work.add_prezero_item({
        graph::variable_cast(a),
        graph::variable_cast(b)
    });

    work.compile();

    assert(work.check_value(0, a) == static_cast<T> (1) && "Expected one.");
    assert(work.check_value(0, b) == static_cast<T> (1) && "Expected one.");
    work.pre_run();
    assert(work.check_value(0, a) == static_cast<T> (0) && "Expected zero.");
    assert(work.check_value(0, b) == static_cast<T> (0) && "Expected zero.");
}

//------------------------------------------------------------------------------
///  @brief Test setting multiple variables with the same map.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_precopy() {
    auto a = graph::variable<T> (1, "");
    auto b = graph::variable<T> (1, "");
    backend::buffer<T> buffer1(1, static_cast<T> (1));
    backend::buffer<T> buffer2(1, static_cast<T> (2));
    a->set(buffer1);
    b->set(buffer2);

    workflow::manager<T> work(0);
    work.add_precopy_item({
        {graph::variable_cast(a), graph::variable_cast(b)}
    });

    work.compile();

    assert(work.check_value(0, a) == static_cast<T> (1) && "Expected one.");
    assert(work.check_value(0, b) == static_cast<T> (2) && "Expected two.");
    work.pre_run();
    assert(work.check_value(0, a) == static_cast<T> (1) && "Expected one.");
    assert(work.check_value(0, b) == static_cast<T> (1) && "Expected one.");
}

//------------------------------------------------------------------------------
///  @brief Test setting multiple variables with the same map.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_premaps() {
    auto a = graph::variable<T> (1, "");
    auto b = graph::variable<T> (1, "");
    backend::buffer<T> buffer(1, static_cast<T> (1));
    a->set(buffer);
    b->set(buffer);

    auto zero = graph::zero<T> ();

    workflow::manager<T> work(0);
    work.add_preitem({
        graph::variable_cast(a),
        graph::variable_cast(b)
    }, {}, {
        {zero, graph::variable_cast(a)},
        {zero, graph::variable_cast(b)}
    }, NULL, "test_maps", 1);

    work.compile();

    assert(work.check_value(0, a) == static_cast<T> (1) && "Expected one.");
    assert(work.check_value(0, b) == static_cast<T> (1) && "Expected one.");
    work.pre_run();
    assert(work.check_value(0, a) == static_cast<T> (0) && "Expected zero.");
    assert(work.check_value(0, b) == static_cast<T> (0) && "Expected zero.");
}

//------------------------------------------------------------------------------
///  @brief Test loop items.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_preloops() {
    auto a = graph::variable<T> (1, "");
    backend::buffer<T> buffer(1, static_cast<T> (0));
    a->set(buffer);

    auto a_next = a + static_cast<T> (1);
    
    workflow::manager<T> work(0);
    work.add_preloop_item({
        graph::variable_cast(a)
    }, {}, {
        {a_next, graph::variable_cast(a)}
    }, NULL, "test_maps", 1, 10);

    work.compile();

    assert(work.check_value(0, a) == static_cast<T> (0) && "Expected zero.");
    work.pre_run();
    assert(work.check_value(0, a) == static_cast<T> (10) && "Expected ten.");
}

//------------------------------------------------------------------------------
///  @brief Test setting multiple variables with the same map.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_zeros() {
    auto a = graph::variable<T> (1, "");
    auto b = graph::variable<T> (1, "");
    backend::buffer<T> buffer(1, static_cast<T> (1));
    a->set(buffer);
    b->set(buffer);

    workflow::manager<T> work(0);
    work.add_zero_item({
        graph::variable_cast(a),
        graph::variable_cast(b)
    });

    work.compile();

    assert(work.check_value(0, a) == static_cast<T> (1) && "Expected one.");
    assert(work.check_value(0, b) == static_cast<T> (1) && "Expected one.");
    work.run();
    assert(work.check_value(0, a) == static_cast<T> (0) && "Expected zero.");
    assert(work.check_value(0, b) == static_cast<T> (0) && "Expected zero.");
}

//------------------------------------------------------------------------------
///  @brief Test setting multiple variables with the same map.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_copy() {
    auto a = graph::variable<T> (1, "");
    auto b = graph::variable<T> (1, "");
    backend::buffer<T> buffer1(1, static_cast<T> (1));
    backend::buffer<T> buffer2(1, static_cast<T> (2));
    a->set(buffer1);
    b->set(buffer2);

    workflow::manager<T> work(0);
    work.add_copy_item({
        {graph::variable_cast(a), graph::variable_cast(b)}
    });

    work.compile();

    assert(work.check_value(0, a) == static_cast<T> (1) && "Expected one.");
    assert(work.check_value(0, b) == static_cast<T> (2) && "Expected two.");
    work.run();
    assert(work.check_value(0, a) == static_cast<T> (1) && "Expected one.");
    assert(work.check_value(0, b) == static_cast<T> (1) && "Expected one.");
}

//------------------------------------------------------------------------------
///  @brief Test setting multiple variables with the same map.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_maps() {
    auto a = graph::variable<T> (1, "");
    auto b = graph::variable<T> (1, "");
    backend::buffer<T> buffer(1, static_cast<T> (1));
    a->set(buffer);
    b->set(buffer);

    auto zero = graph::zero<T> ();

    workflow::manager<T> work(0);
    work.add_item({
        graph::variable_cast(a),
        graph::variable_cast(b)
    }, {}, {
        {zero, graph::variable_cast(a)},
        {zero, graph::variable_cast(b)}
    }, NULL, "test_maps", 1);

    work.compile();

    assert(work.check_value(0, a) == static_cast<T> (1) && "Expected one.");
    assert(work.check_value(0, b) == static_cast<T> (1) && "Expected one.");
    work.run();
    assert(work.check_value(0, a) == static_cast<T> (0) && "Expected zero.");
    assert(work.check_value(0, b) == static_cast<T> (0) && "Expected zero.");
}

//------------------------------------------------------------------------------
///  @brief Test loop items.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_loops() {
    auto a = graph::variable<T> (1, "");
    backend::buffer<T> buffer(1, static_cast<T> (0));
    a->set(buffer);

    auto a_next = a + static_cast<T> (1);
    
    workflow::manager<T> work(0);
    work.add_loop_item({
        graph::variable_cast(a)
    }, {}, {
        {a_next, graph::variable_cast(a)}
    }, NULL, "test_maps", 1, 10);

    work.compile();

    assert(work.check_value(0, a) == static_cast<T> (0) && "Expected zero.");
    work.run();
    assert(work.check_value(0, a) == static_cast<T> (10) && "Expected ten.");
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void run_tests() {
    test_prezeros<T> ();
    test_precopy<T> ();
    test_premaps<T> ();
    test_preloops<T> ();
    test_zeros<T> ();
    test_copy<T> ();
    test_maps<T> ();
    test_loops<T> ();
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
