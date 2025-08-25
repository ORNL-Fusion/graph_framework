//------------------------------------------------------------------------------
///  @file trigonometry_test.cpp
///  @brief Tests for trig nodes.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/math.hpp"
#include "../graph_framework/trigonometry.hpp"
#include "../graph_framework/arithmetic.hpp"
#include "../graph_framework/piecewise.hpp"

//------------------------------------------------------------------------------
///  @brief Tests for sine nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_sin() {
    assert(graph::constant_cast(graph::sin(graph::constant(static_cast<T> (10.0)))).get() &&
           "Expected constant");

    auto y = graph::variable<T> (1, "");
    auto siny = graph::sin(y);
    assert(graph::sin_cast(siny).get() && "Expected a sine node");

//  Test derivatives.
    auto dsiny = siny->df(y);
    assert(graph::cos_cast(dsiny).get() && "Expected cosine node.");

//  Sin(Atan(x,y)) -> y/Sqrt(x^2 + y^2)
    auto x = graph::variable<T> (1, "");
    auto sinatan = graph::sin(graph::atan(x, y));
    assert(graph::divide_cast(sinatan).get() && "Expected a divide node.");
}

//------------------------------------------------------------------------------
///  @brief Tests for cosine nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_cos() {
    assert(graph::constant_cast(graph::cos(graph::constant(static_cast<T> (10.0)))).get() &&
           "Expected constant");

    auto y = graph::variable<T> (1, "");
    auto cosy = graph::cos(y);
    assert(graph::cos_cast(cosy).get() && "Expected a cosine node");

//  Test derivatives.
    auto dcosy = cosy->df(y);
    assert(graph::multiply_cast(dcosy).get() && "Expected multiply node.");

//  Cos(Atan(x,y)) -> x/Sqrt(x^2 + y^2)
    auto x = graph::variable<T> (1, "");
    auto cosatan = graph::cos(graph::atan(x, y));
    assert(graph::divide_cast(cosatan).get() && "Expected a divide node.");
}

//------------------------------------------------------------------------------
///  @brief Tests for tan nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_tan() {
    assert(graph::constant_cast(graph::tan(graph::constant(static_cast<T> (10.0)))).get() &&
           "Expected constant");

    auto y = graph::variable<T> (1, "");
    auto tany = graph::tan(y);
    assert(graph::divide_cast(tany).get() && "Expected divide node");

//  Test derivatives.
    auto dtany = tany->df(y);
    assert(graph::add_cast(dtany).get() && "Expected add node.");
}

//------------------------------------------------------------------------------
///  @brief Tests for tan nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_atan() {
    assert(graph::constant_cast(graph::atan(10.0,
                                            graph::constant(static_cast<T> (11.0)))).get() &&
           "Expected constant");
    assert(graph::constant_cast(graph::atan(graph::zero<T> (),
                                            11.0)).get() &&
           "Expected constant");

    auto x = graph::variable<T> (1, "");
    auto y = graph::variable<T> (1, "");
    auto atanxy = graph::atan(x, y);
    assert(graph::atan_cast(atanxy).get() && "Expected an atan node");

//  Test derivatives.
    assert(graph::divide_cast(atanxy->df(x)).get() &&
           "Expected multiply node.");
    assert(graph::divide_cast(atanxy->df(y)).get() &&
           "Expected divide node.");
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void run_tests() {
    test_sin<T> ();
    test_cos<T> ();
    test_tan<T> ();
    test_atan<T> ();
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
