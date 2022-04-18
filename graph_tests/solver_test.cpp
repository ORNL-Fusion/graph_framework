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
///  @param[in] dt Timestep for the solver.
//------------------------------------------------------------------------------
template<typename SOLVER>
void test_solver(const double dt) {
    auto w = graph::variable<typename SOLVER::backend> (1, 0.5);
    auto kx = graph::variable<typename SOLVER::backend> (1, 0.25);
    auto ky = graph::variable<typename SOLVER::backend> (1, 0.25);
    auto kz = graph::variable<typename SOLVER::backend> (1, 0.15);
    auto x = graph::variable<typename SOLVER::backend> (1, 0.0);
    auto y = graph::variable<typename SOLVER::backend> (1, 0.0);
    auto z = graph::variable<typename SOLVER::backend> (1, 0.0);

    SOLVER solve(w, kx, ky, kz, x, y, z, dt);
    solve.init(kx);
    auto residule = solve.residule();

    for(size_t i = 0; i < 5; i++) {
        solve.step();
        assert(residule->evaluate().at(0) < 1.0E-30 &&
               "Solver failed to retain initial acuracy");
    }
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified disperions Relation.
///
///  @param[in] dt Timestep for the solver.
//------------------------------------------------------------------------------
template<typename DISPERSION> void run_disperions_tests(const double dt) {
    test_solver<solver::rk2<DISPERSION>> (dt);
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
//------------------------------------------------------------------------------
template<typename BACKEND> void run_tests() {
    run_disperions_tests<dispersion::simple<BACKEND>> (1.0);
    run_disperions_tests<dispersion::guassian_well<BACKEND>> (0.00001);
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
