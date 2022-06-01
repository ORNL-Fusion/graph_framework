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
///
///  @param[in] tolarance Tolarance to solver the dispersion function to.
///  @param[in] n0        Starting nz value.
///  @param[in] x0        Starting x guess.
///  @param[in] kx0       Starting kx guess.
//------------------------------------------------------------------------------
template<typename BACKEND>
void test_reflection(const typename BACKEND::base tolarance,
                     const typename BACKEND::base n0,
                     const typename BACKEND::base x0,
                     const typename BACKEND::base kx0) {
    const typename BACKEND::base q = 1.602176634E-19;
    const typename BACKEND::base me = 9.1093837015E-31;
    const typename BACKEND::base mu0 = M_PI*4.0E-7;
    const typename BACKEND::base epsilon0 = 8.8541878138E-12;
    const typename BACKEND::base c = 1.0/sqrt(mu0*epsilon0);
    const typename BACKEND::base OmegaCE = -q*1/(me*c);
    
    auto w = graph::variable<BACKEND> (1, OmegaCE, "\\omega");
    auto kx = graph::variable<BACKEND> (1, 0.0, "k_{x}");
    auto ky = graph::variable<BACKEND> (1, 0.0, "k_{y}");
    auto kz = graph::variable<BACKEND> (1, n0*OmegaCE, "k_{z}");
    auto x = graph::variable<BACKEND> (1, x0, "x");
    auto y = graph::variable<BACKEND> (1, 0.0, "y");
    auto z = graph::variable<BACKEND> (1, 0.0, "z");
    
    auto eq = equilibrium::make_slab<BACKEND> ();
    solver::rk4<dispersion::cold_plasma<BACKEND>> solve(w, kx, ky, kz, x, y, z, 0.0001, eq);

// Solve for a location where the wave is cut off.
    solve.init(x, tolarance);
    const typename BACKEND::base cuttoff_location = x->evaluate().at(0);

//  Set the ray starting point close to the cut off to reduce the number of
//  times steps that need to be taken.
    x->set(cuttoff_location - backend::base_cast<BACKEND> (0.00001)*cuttoff_location);

//  Set an inital guess for kx and solve for the wave number at the new
//  location.
    kx->set(kx0);
    solve.init(kx, tolarance);
    
    typename BACKEND::base max_x = solve.state.back().x.at(0);
    typename BACKEND::base new_x = max_x;
    do {
        solve.step();
        new_x = solve.state.back().x.at(0);
        max_x = std::max(new_x, max_x);
        assert(max_x < cuttoff_location && "Ray exceeded cutoff.");
    } while (max_x == new_x);
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
///
///  @param[in] tolarance Tolarance to solver the dispersion function to.
//------------------------------------------------------------------------------
template<typename BACKEND> void run_tests(const typename BACKEND::base tolarance) {
    test_reflection<BACKEND> (tolarance, 0.7, 0.1, 22.0);
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
//  No there is not enough precision in float to pass the test.
    run_tests<backend::cpu<double>> (1.0E-30);
}
