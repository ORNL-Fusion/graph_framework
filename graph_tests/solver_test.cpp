//------------------------------------------------------------------------------
///  @file solver_test.cpp
///  @brief Tests for solvers.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../graph_framework/backend.hpp"
#include "../graph_framework/solver.hpp"

//------------------------------------------------------------------------------
///  @brief Test the solver.
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
    timeing::measure_diagnostic solver("init");
    auto residule = solve.init(kx, tolarance);
    solver.stop();

    timeing::measure_diagnostic compile("compile");
    solve.compile();
    compile.stop();

    timeing::measure_diagnostic step("step");
    for (size_t i = 0; i < 5; i++) {
        solve.step();
        assert(std::abs(residule->evaluate().at(0)) < std::abs(tolarance) &&
               "Solver failed to retain initial acuracy");
    }
    step.stop();
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified disperions Relation.
///
///  @params[in] tolarance Tolarance to solver the dispersion function to.
///  @params[in] omega0    Ray frequency.
///  @params[in] kx0       Wave number guess.
///  @params[in] dt        Timestep for the solver.
//------------------------------------------------------------------------------
template<typename DISPERSION> void run_disperions_tests(const typename DISPERSION::base tolarance,
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
///  @params[in] tolarance Tolarance to solver the dispersion function to.
//------------------------------------------------------------------------------
template<typename T>
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
#if defined(USE_METAL) || defined(USE_CUDA)
    run_tests<float> (2.0E-14);
#else
    run_tests<float> (1.0E-14);
#endif
    run_tests<double> (1.0E-30);
    run_tests<std::complex<float>> (2.0E-14);
    run_tests<std::complex<double>> (1.0E-30);
    END_GPU
}
