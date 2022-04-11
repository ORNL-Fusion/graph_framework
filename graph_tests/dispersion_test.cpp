//------------------------------------------------------------------------------
///  @file dispersion_test.cpp
///  @brief Tests for math nodes.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/cpu_backend.hpp"
#include "../graph_framework/dispersion.hpp"

//------------------------------------------------------------------------------
///  @brief The newton solve for dispersion relation.
//------------------------------------------------------------------------------
template<typename DISPERSION>
void test_solve() {
    auto w = graph::variable<typename DISPERSION::backend> (1, 0.5);
    auto kx = graph::variable<typename DISPERSION::backend> (1, 0.25);
    auto ky = graph::variable<typename DISPERSION::backend> (1, 0.25);
    auto kz = graph::variable<typename DISPERSION::backend> (1, 0.15);
    auto x = graph::variable<typename DISPERSION::backend> (1, 0.0);
    auto y = graph::variable<typename DISPERSION::backend> (1, 0.0);
    auto z = graph::variable<typename DISPERSION::backend> (1, 0.0);

    dispersion::dispersion_interface<DISPERSION> D(w, kx, ky, kz, x, y, z);

    auto loss = D.get_d()*D.get_d();

    D.solve(kx);
    assert(loss->evaluate().at(0) < 1.0E-30 &&
           "Solve failed to meet expected result for kx.");

    kx->set(0.2);
    D.solve(ky);
    assert(loss->evaluate().at(0) < 1.0E-30 &&
           "Solve failed to meet expected result for ky.");

    ky->set(0.25);
    D.solve(kz);
    assert(loss->evaluate().at(0) < 1.0E-30 &&
           "Solve failed to meet expected result for kz.");

    kz->set(0.15);
    D.solve(w);
    assert(loss->evaluate().at(0) < 1.0E-30 &&
           "Solve failed to meet expected result for w.");
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    test_solve<dispersion::simple<backend::cpu>> ();
    test_solve<dispersion::guassian_well<backend::cpu>> ();
}
