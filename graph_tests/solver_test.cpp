//------------------------------------------------------------------------------
///  @file dispersion_test.cpp
///  @brief Tests for math nodes.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/dispersion.hpp"

//------------------------------------------------------------------------------
///  @brief The second order runga kutta ode solve.
//------------------------------------------------------------------------------
template<typename DISPERSION>
void test_rk2() {
    auto w = graph::variable(1, 0.5);
    auto kx = graph::variable(1, 0.25);
    auto ky = graph::variable(1, 0.25);
    auto kz = graph::variable(1, 0.15);
    auto x = graph::variable(1, 0.0);
    auto y = graph::variable(1, 0.0);
    auto z = graph::variable(1, 0.0);

    solver::rk2<DISPERSION> solve(w, kx, ky, kz, x, y, z, 1.0);
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
    test_rk2<dispersion::simple> ();
}
