//------------------------------------------------------------------------------
///  @file solver\_test.cpp
///  @brief Tests for solvers.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/backend.hpp"
#include "../graph_framework/solver.hpp"
#include "../graph_framework/timing.hpp"

//------------------------------------------------------------------------------
///  @brief Test the solver.
///
///  @tparam SOLVER Class of solver to use.
///
///  @params[in] tolarance Tolarance to solver the dispersion function to.
///  @params[in] omega0    Ray frequency.
///  @params[in] kx0       Wave number guess.
///  @params[in] dt        Timestep for the solver.
//------------------------------------------------------------------------------
template<typename SOLVER>
void test_solver(const typename SOLVER::base tolarance,
                 const typename SOLVER::base omega0,
                 const typename SOLVER::base kx0,
                 const typename SOLVER::base dt) {
    auto w = graph::variable<typename SOLVER::base> (1, omega0, "\\omega");
    auto kx = graph::variable<typename SOLVER::base> (1, kx0, "k_{x}");
    auto ky = graph::variable<typename SOLVER::base> (1, 0.25, "k_{y}");
    auto kz = graph::variable<typename SOLVER::base> (1, 0.15, "k_{z}");
    auto x = graph::variable<typename SOLVER::base> (1, 0.0, "x");
    auto y = graph::variable<typename SOLVER::base> (1, 0.0, "y");
    auto z = graph::variable<typename SOLVER::base> (1, 0.0, "z");
    auto t = graph::variable<typename SOLVER::base> (1, 0.0, "t");

    auto eq = equilibrium::make_guassian_density<typename SOLVER::base> ();

    SOLVER solve(w, kx, ky, kz, x, y, z, t, dt, eq);
    const timeing::measure_diagnostic solver("init");
    auto residule = solve.init(kx, tolarance);
    solver.print();

    const timeing::measure_diagnostic compile("compile");
    solve.compile();
    compile.print();

    const timeing::measure_diagnostic step("step");
    for (size_t i = 0; i < 5; i++) {
        solve.step();
        assert(std::abs(solve.check_residule(0)) < std::abs(tolarance) &&
               "Solver failed to retain initial acuracy");
    }
    step.print();
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified disperions Relation.
///
///  @tparam DISPERSION Class of dispersion function to use.
///
///  @params[in] tolarance Tolarance to solver the dispersion function to.
///  @params[in] omega0    Ray frequency.
///  @params[in] kx0       Wave number guess.
///  @params[in] dt        Timestep for the solver.
//------------------------------------------------------------------------------
template<typename DISPERSION>
void run_disperions_tests(const typename DISPERSION::base tolarance,
                          const typename DISPERSION::base omega0,
                          const typename DISPERSION::base kx0,
                          const typename DISPERSION::base dt) {
    test_solver<solver::rk2<DISPERSION>> (tolarance, omega0, kx0, dt);
    std::cout << "Test completed for rk2 solver." << std::endl;

    test_solver<solver::rk4<DISPERSION>> (tolarance, omega0, kx0, dt);
    std::cout << "Test completed for rk4 solver." << std::endl;
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
///
///  @tparam T Base type of the calculation.
///
///  @params[in] tolarance Tolarance to solver the dispersion function to.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
void run_tests(const T tolarance) {
    run_disperions_tests<dispersion::simple<T>> (tolarance, 0.5, 0.25, 1.0);
    run_disperions_tests<dispersion::guassian_well<T>> (tolarance, 0.5, 0.25, 0.00001);
    run_disperions_tests<dispersion::cold_plasma<T>> (tolarance, 900.0, 1000.0, 0.5/10000.0);
    std::cout << "Tests completed for ";
    jit::add_type<T> (std::cout);
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @params[in] argc Number of commandline arguments.
///  @params[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    START_GPU
    run_tests<float> (4.0E-15);
    run_tests<double> (1.0E-30);
    run_tests<std::complex<float>> (3.0E-15);
    run_tests<std::complex<double>> (1.0E-30);
    END_GPU
}
