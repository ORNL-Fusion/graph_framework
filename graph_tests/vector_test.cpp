//------------------------------------------------------------------------------
///  @file node_test.cpp
///  @brief Tests for the node interface.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "vector.hpp"

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
//------------------------------------------------------------------------------
template<typename BACKEND> void run_tests() {
    auto one = graph::constant<BACKEND> (1.0);
    auto zero = graph::constant<BACKEND> (0.0);

    auto v1 = graph::vector(one, zero, zero);
    auto length = v1->length();
    assert(one.get() == length.get() && "Expected one.");
    auto dot = v1->dot(v1);
    assert(dot.get() == one.get() && "Expected one.");
    auto cross = v1->cross(v1);
    assert(cross->get_x().get() == zero.get() && "Expected zero.");
    assert(cross->get_y().get() == zero.get() && "Expected zero.");
    assert(cross->get_z().get() == zero.get() && "Expected zero.");
    auto unit = v1->unit();
    assert(unit->get_x().get() == one.get() && "Expected one.");
    assert(unit->get_y().get() == zero.get() && "Expected zero.");
    assert(unit->get_z().get() == zero.get() && "Expected zero.");

    auto v2 = graph::vector(zero, one, zero);
    dot = v1->dot(v2);
    assert(dot.get() == zero.get() && "Expected zero.");
    cross = v1->cross(v2);
    assert(cross->get_x().get() == zero.get() && "Expected zero.");
    assert(cross->get_y().get() == zero.get() && "Expected zero.");
    assert(cross->get_z().get() == one.get() && "Expected one.");
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    run_tests<backend::cpu<float>> ();
    run_tests<backend::cpu<double>> ();
    run_tests<backend::cpu<std::complex<float>>> ();
    run_tests<backend::cpu<std::complex<double>>> ();
}
