//------------------------------------------------------------------------------
///  @file trigonometry\_test.cpp
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
///  @brief Tests for cosine nodes.
//------------------------------------------------------------------------------
template<typename T> void test_sin() {
    assert(graph::constant_cast(graph::sin(graph::constant(static_cast<T> (10.0)))).get() &&
           "Expected constant");

    auto y = graph::variable<T> (1, "");
    auto siny = graph::sin(y);
    assert(graph::sin_cast(siny) && "Expected a sine node");

//  Test derivatives.
    auto dsiny = siny->df(y);
    assert(graph::cos_cast(dsiny) && "Expected cosine node.");
}

//------------------------------------------------------------------------------
///  @brief Tests for cosine nodes.
//------------------------------------------------------------------------------
template<typename T> void test_cos() {
    assert(graph::constant_cast(graph::cos(graph::constant(static_cast<T> (10.0)))).get() &&
           "Expected constant");

    auto y = graph::variable<T> (1, "");
    auto cosy = graph::cos(y);
    assert(graph::cos_cast(cosy).get() && "Expected a cosine node");

//  Test derivatives.
    auto dcosy = cosy->df(y);
    assert(graph::multiply_cast(dcosy).get() && "Expected multiply node.");
}

//------------------------------------------------------------------------------
///  @brief Tests for tan nodes.
//------------------------------------------------------------------------------
template<typename T> void test_tan() {
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
//------------------------------------------------------------------------------
template<typename T> void test_atan() {
    assert(graph::constant_cast(graph::atan(graph::constant(static_cast<T> (10.0)),
                                            graph::constant(static_cast<T> (11.0)))).get() &&
           "Expected constant");
    assert(graph::constant_cast(graph::atan(graph::zero<T> (),
                                            graph::constant(static_cast<T> (11.0)))).get() &&
           "Expected constant");

    auto x = graph::variable<T> (1, "");
    auto y = graph::variable<T> (1, "");
    auto atanxy = graph::atan(x, y);
    assert(graph::atan_cast(atanxy).get() && "Expected an atan node");

//  Test derivatives.
    assert(graph::multiply_cast(atanxy->df(x)).get() &&
           "Expected multiply node.");
    assert(graph::divide_cast(atanxy->df(y)).get() &&
           "Expected divide node.");
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
//------------------------------------------------------------------------------
template<typename T> void run_tests() {
    test_sin<T> ();
    test_cos<T> ();
    test_tan<T> ();
    test_atan<T> ();
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @params[in] argc Number of commandline arguments.
///  @params[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    run_tests<float> ();
    run_tests<double> ();
    run_tests<std::complex<float>> ();
    run_tests<std::complex<double>> ();
}
