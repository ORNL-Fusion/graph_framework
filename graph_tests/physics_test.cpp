//------------------------------------------------------------------------------
///  @file physics_test.cpp
///  @brief Tests for math nodes.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/cpu_backend.hpp"
#include "../graph_framework/solver.hpp"

//------------------------------------------------------------------------------
///  @brief Reflection test.
///
///  Given a wave frequency, a wave with zero k will not propagate.
//------------------------------------------------------------------------------
template<typename BACKEND>
void test_reflection() {
    const double q = 1.602176634E-19;
    const double me = 9.1093837015E-31;
    const double mu0 = M_PI*4.0E-7;
    const double epsilon0 = 8.8541878138E-12;
    const double c = 1.0/sqrt(mu0*epsilon0);
    const double OmegaCE = -q*1/(me*c);
    
    auto w = graph::variable<BACKEND> (1, OmegaCE, "\\omega");
    auto kx = graph::variable<BACKEND> (1, 0.0, "k_{x}");
    auto ky = graph::variable<BACKEND> (1, 0.0, "k_{y}");
    auto kz = graph::variable<BACKEND> (1, 0.7*OmegaCE, "k_{z}");
    auto x = graph::variable<BACKEND> (1, 0.1, "x");
    auto y = graph::variable<BACKEND> (1, 0.0, "y");
    auto z = graph::variable<BACKEND> (1, 0.0, "z");
    
    auto eq = equilibrium::make_slab<BACKEND> ();
    solver::rk4<dispersion::cold_plasma<BACKEND>> solve(w, kx, ky, kz, x, y, z, 0.0001, eq);

// Solve for a location where the wave is cut off.
    solve.init(x);
    const double cuttoff_location = x->evaluate().at(0);

//  Set the ray starting point close to the cut off to reduce the number of
//  times steps that need to be taken.
    x->set(cuttoff_location - 0.00001*cuttoff_location);

//  Set an inital guess for kx and solve for the wave number at the new
//  location.
    kx->set(22.0);
    solve.init(kx);
    
    double max_x = solve.state.back().x.at(0);
    double new_x = max_x;
    do {
        solve.step();
        new_x = solve.state.back().x.at(0);
        max_x = std::max(new_x, max_x);
        assert(max_x < cuttoff_location && "Ray exceeded cutoff.");
    } while (max_x == new_x);
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
//------------------------------------------------------------------------------
template<typename BACKEND> void run_tests() {
    test_reflection<BACKEND> ();
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
