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

//  test a zero vector length.
    auto v0 = graph::vector(zero, zero, zero);
#ifdef USE_REDUCE
    assert(zero.get() == v0->length().get() && "Expected zero.");
#endif
    assert(v0->length()->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected zero.");

    auto v1 = graph::vector(one, zero, zero);
    auto length = v1->length();
#ifdef USE_REDUCE
    assert(one.get() == length.get() && "Expected one.");
#endif
    assert(length->evaluate()[0] == backend::base_cast<BACKEND> (1.0) &&
           "Expected one.");
    auto dot = v1->dot(v1);
#ifdef USE_REDUCE
    assert(dot.get() == one.get() && "Expected one.");
#endif
    assert(dot->evaluate()[0] == backend::base_cast<BACKEND> (1.0) &&
           "Expected one.");
    auto cross = v1->cross(v1);
#ifdef USE_REDUCE
    assert(cross->get_x().get() == zero.get() && "Expected zero.");
    assert(cross->get_y().get() == zero.get() && "Expected zero.");
    assert(cross->get_z().get() == zero.get() && "Expected zero.");
#endif
    assert(cross->get_x()->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected zero.");
    assert(cross->get_y()->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected zero.");
    assert(cross->get_z()->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected zero.");
    auto unit = v1->unit();
#ifdef USE_REDUCE
    assert(unit->get_x().get() == one.get() && "Expected one.");
    assert(unit->get_y().get() == zero.get() && "Expected zero.");
    assert(unit->get_z().get() == zero.get() && "Expected zero.");
#endif
    assert(unit->get_x()->evaluate()[0] == backend::base_cast<BACKEND> (1.0) &&
           "Expected zero.");
    assert(unit->get_y()->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected zero.");
    assert(unit->get_z()->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected zero.");

    auto v2 = graph::vector(zero, one, zero);
    dot = v1->dot(v2);
#ifdef USE_REDUCE
    assert(dot.get() == zero.get() && "Expected zero.");
#endif
    assert(dot->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected zero.");
    cross = v1->cross(v2);
#ifdef USE_REDUCE
    assert(cross->get_x().get() == zero.get() && "Expected zero.");
    assert(cross->get_y().get() == zero.get() && "Expected zero.");
    assert(cross->get_z().get() == one.get() && "Expected one.");
#endif
    assert(cross->get_x()->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected zero.");
    assert(cross->get_y()->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected zero.");
    assert(cross->get_z()->evaluate()[0] == backend::base_cast<BACKEND> (1.0) &&
           "Expected one.");
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
