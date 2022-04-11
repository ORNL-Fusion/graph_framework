//------------------------------------------------------------------------------
///  @file solver_test.cpp
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
///  @brief The second order runga kutta ode solve.
///
///  @param[in] dt Timestep for the solver.
//------------------------------------------------------------------------------
template<typename DISPERSION>
void test_rk2(const double dt) {
    auto w = graph::variable<typename DISPERSION::backend> (1, 0.5);
    auto kx = graph::variable<typename DISPERSION::backend> (1, 0.25);
    auto ky = graph::variable<typename DISPERSION::backend> (1, 0.25);
    auto kz = graph::variable<typename DISPERSION::backend> (1, 0.15);
    auto x = graph::variable<typename DISPERSION::backend> (1, 0.0);
    auto y = graph::variable<typename DISPERSION::backend> (1, 0.0);
    auto z = graph::variable<typename DISPERSION::backend> (1, 0.0);

    solver::rk2<DISPERSION> solve(w, kx, ky, kz, x, y, z, dt);
    solve.init(kx);
    auto residule = solve.residule();

    for(size_t i = 0; i < 5; i++) {
        solve.step();
        assert(residule->evaluate().at(0) < 1.0E-30 &&
               "Solver failed to retain initial acuracy");
    }
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    test_rk2<dispersion::simple<backend::cpu>> (1.0);
    test_rk2<dispersion::guassian_well<backend::cpu>> (0.00001);
}
