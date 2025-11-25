//------------------------------------------------------------------------------
///  @file solver_test.cpp
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
///  @param[in] tolerance Tolerance to solver the dispersion function to.
///  @param[in] omega0    Ray frequency.
///  @param[in] kx0       Wave number guess.
///  @param[in] dt        Time step for the solver.
//------------------------------------------------------------------------------
template<solver::method SOLVER>
void test_solver(const typename SOLVER::base tolerance,
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

    auto eq = equilibrium::make_gaussian_density<typename SOLVER::base> ();

    auto dt_const = graph::constant(static_cast<typename SOLVER::base> (dt));
    SOLVER solve(w, kx, ky, kz, x, y, z, t, dt_const, eq);
    const timing::measure_diagnostic solver("init");
    auto residual = solve.init(kx, tolerance);
    solver.print();

    const timing::measure_diagnostic compile("compile");
    solve.compile();
    compile.print();

    const timing::measure_diagnostic step("step");
    for (size_t i = 0; i < 5; i++) {
        solve.step();
        assert(std::abs(solve.check_residual(0)) < std::abs(tolerance) &&
               "Solver failed to retain initial accuracy");
    }
    step.print();
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified dispersions Relation.
///
///  @tparam DISPERSION Class of dispersion function to use.
///
///  @param[in] tolerance Tolerance to solver the dispersion function to.
///  @param[in] omega0    Ray frequency.
///  @param[in] kx0       Wave number guess.
///  @param[in] dt        Timestep for the solver.
//------------------------------------------------------------------------------
template<typename DISPERSION>
void run_dispersions_tests(const typename DISPERSION::base tolerance,
                           const typename DISPERSION::base omega0,
                           const typename DISPERSION::base kx0,
                           const typename DISPERSION::base dt) {
    test_solver<solver::rk2<DISPERSION>> (tolerance, omega0, kx0, dt);
    std::cout << "Test completed for rk2 solver." << std::endl;

    test_solver<solver::rk4<DISPERSION>> (tolerance, omega0, kx0, dt);
    std::cout << "Test completed for rk4 solver." << std::endl;
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] tolerance Tolerance to solver the dispersion function to.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
void run_tests(const T tolerance) {
    run_dispersions_tests<dispersion::simple<T>> (tolerance, 0.5, 0.25, 1.0);
    run_dispersions_tests<dispersion::gaussian_well<T>> (tolerance, 0.5, 0.25, 0.00001);
    run_dispersions_tests<dispersion::cold_plasma<T>> (tolerance, 900.0, 1000.0, 0.5/10000.0);
    std::cout << "Tests completed for ";
    jit::add_type<T> (std::cout);
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    START_GPU
    (void)argc;
    (void)argv;
    run_tests<float> (4.0E-15);
    run_tests<double> (1.0E-30);
    if constexpr (jit::use_cuda()) {
        run_tests<std::complex<float>> (5.6E-15);
    } else {
        run_tests<std::complex<float>> (4.0E-15);
    }
    run_tests<std::complex<double>> (1.0E-30);
    END_GPU
}
