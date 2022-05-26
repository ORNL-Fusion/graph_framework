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
///  @brief Test the solver.
///
///  @param[in] omega0 Ray frequency.
///  @param[in] kx0    Wave number guess.
///  @param[in] dt     Timestep for the solver.
//------------------------------------------------------------------------------
template<typename SOLVER>
void test_solver(const double omega0,
                 const double kx0,
                 const double dt) {
    auto w = graph::variable<typename SOLVER::backend> (1, omega0, "\\omega");
    auto kx = graph::variable<typename SOLVER::backend> (1, kx0, "k_{x}");
    auto ky = graph::variable<typename SOLVER::backend> (1, 0.25, "k_{y}");
    auto kz = graph::variable<typename SOLVER::backend> (1, 0.15, "k_{z}");
    auto x = graph::variable<typename SOLVER::backend> (1, 0.0, "x");
    auto y = graph::variable<typename SOLVER::backend> (1, 0.0, "y");
    auto z = graph::variable<typename SOLVER::backend> (1, 0.0, "z");

    auto eq = equilibrium::make_guassian_density<typename SOLVER::backend> ();

    SOLVER solve(w, kx, ky, kz, x, y, z, dt, eq);
    solve.init(kx);
    auto residule = solve.residule();

    for (size_t i = 0; i < 5; i++) {
        solve.step();
        assert(residule->evaluate().at(0) < 1.0E-30 &&
               "Solver failed to retain initial acuracy");
    }
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified disperions Relation.
///
///  @param[in] omega0 Ray frequency.
///  @param[in] kx0    Wave number guess.
///  @param[in] dt     Timestep for the solver.
//------------------------------------------------------------------------------
template<typename DISPERSION> void run_disperions_tests(const double omega0,
                                                        const double kx0,
                                                        const double dt) {
    test_solver<solver::rk2<DISPERSION>> (omega0, kx0, dt);
    test_solver<solver::rk4<DISPERSION>> (omega0, kx0, dt);
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
//------------------------------------------------------------------------------
template<typename BACKEND> void run_tests() {
    run_disperions_tests<dispersion::simple<BACKEND>> (0.5, 0.25, 1.0);
    run_disperions_tests<dispersion::guassian_well<BACKEND>> (0.5, 0.25, 0.00001);
    run_disperions_tests<dispersion::cold_plasma<BACKEND>> (900.0, 1000.0, 0.5/10000.0);
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
