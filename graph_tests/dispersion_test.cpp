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
///
///  @param[in] omega   Ray frequency.
///  @param[in] k_guess Inital guess for the wave number.
//------------------------------------------------------------------------------
template<typename DISPERSION>
void test_solve(const double omega,
                const double k_guess) {
    auto w = graph::variable<typename DISPERSION::backend> (1, omega);
    auto kx = graph::variable<typename DISPERSION::backend> (1, 0.25);
    auto ky = graph::variable<typename DISPERSION::backend> (1, 0.25);
    auto kz = graph::variable<typename DISPERSION::backend> (1, 0.15);
    auto x = graph::variable<typename DISPERSION::backend> (1, 0.0);
    auto y = graph::variable<typename DISPERSION::backend> (1, 0.0);
    auto z = graph::variable<typename DISPERSION::backend> (1, 0.0);

    auto eq = equilibrium::make_guassian_density<typename DISPERSION::backend> ();

    dispersion::dispersion_interface<DISPERSION> D(w, kx, ky, kz, x, y, z, eq);

    auto loss = D.get_d()*D.get_d();

    kx->set(k_guess);
    D.solve(kx);
    assert(loss->evaluate().at(0) < 1.0E-30 &&
           "Solve failed to meet expected result for kx.");

    kx->set(0.2);
    ky->set(k_guess);
    D.solve(ky);
    assert(loss->evaluate().at(0) < 1.0E-30 &&
           "Solve failed to meet expected result for ky.");

    ky->set(0.25);
    kz->set(k_guess);
    D.solve(kz);
    assert(loss->evaluate().at(0) < 1.0E-30 &&
           "Solve failed to meet expected result for kz.");

    kz->set(0.15);
    kx->set(k_guess);
    D.solve(w);
    assert(loss->evaluate().at(0) < 1.0E-30 &&
           "Solve failed to meet expected result for w.");
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
//------------------------------------------------------------------------------
template<typename BACKEND> void run_tests() {
    test_solve<dispersion::simple<BACKEND>> (0.5, 1.0);
    test_solve<dispersion::guassian_well<BACKEND>> (0.5, 1.0);
    test_solve<dispersion::cold_plasma<BACKEND>> (900.0, 1000.0);
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    run_tests<backend::cpu> ();
}
