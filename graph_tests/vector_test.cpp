//------------------------------------------------------------------------------
///  @file vector\_test.cpp
///  @brief Tests for the vector interface.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/vector.hpp"
#include "../graph_framework/math.hpp"
#include "../graph_framework/trigonometry.hpp"
#include "../graph_framework/arithmetic.hpp"
#include "../graph_framework/piecewise.hpp"

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void run_tests() {
    auto one = graph::one<T> ();
    auto zero = graph::zero<T> ();

//  test a zero vector length.
    auto v0 = graph::vector(zero, zero, zero);
    assert(zero.get() == v0->length().get() && "Expected zero.");
    assert(v0->length()->evaluate()[0] == static_cast<T> (0.0) &&
           "Expected zero.");

    auto v1 = graph::vector(one, zero, zero);
    auto length = v1->length();
    assert(one.get() == length.get() && "Expected one.");
    assert(length->evaluate()[0] == static_cast<T> (1.0) &&
           "Expected one.");
    auto dot = v1->dot(v1);
    assert(dot.get() == one.get() && "Expected one.");
    assert(dot->evaluate()[0] == static_cast<T> (1.0) &&
           "Expected one.");
    auto cross = v1->cross(v1);
    assert(cross->get_x().get() == zero.get() && "Expected zero.");
    assert(cross->get_y().get() == zero.get() && "Expected zero.");
    assert(cross->get_z().get() == zero.get() && "Expected zero.");
    assert(cross->get_x()->evaluate()[0] == static_cast<T> (0.0) &&
           "Expected zero.");
    assert(cross->get_y()->evaluate()[0] == static_cast<T> (0.0) &&
           "Expected zero.");
    assert(cross->get_z()->evaluate()[0] == static_cast<T> (0.0) &&
           "Expected zero.");
    auto unit = v1->unit();
    assert(unit->get_x().get() == one.get() && "Expected one.");
    assert(unit->get_y().get() == zero.get() && "Expected zero.");
    assert(unit->get_z().get() == zero.get() && "Expected zero.");
    assert(unit->get_x()->evaluate()[0] == static_cast<T> (1.0) &&
           "Expected zero.");
    assert(unit->get_y()->evaluate()[0] == static_cast<T> (0.0) &&
           "Expected zero.");
    assert(unit->get_z()->evaluate()[0] == static_cast<T> (0.0) &&
           "Expected zero.");

    auto v2 = graph::vector(zero, one, zero);
    dot = v1->dot(v2);
    assert(dot.get() == zero.get() && "Expected zero.");
    assert(dot->evaluate()[0] == static_cast<T> (0.0) &&
           "Expected zero.");
    cross = v1->cross(v2);
    assert(cross->get_x().get() == zero.get() && "Expected zero.");
    assert(cross->get_y().get() == zero.get() && "Expected zero.");
    assert(cross->get_z().get() == one.get() && "Expected one.");
    assert(cross->get_x()->evaluate()[0] == static_cast<T> (0.0) &&
           "Expected zero.");
    assert(cross->get_y()->evaluate()[0] == static_cast<T> (0.0) &&
           "Expected zero.");
    assert(cross->get_z()->evaluate()[0] == static_cast<T> (1.0) &&
           "Expected one.");
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
