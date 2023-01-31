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
///  @param[in] tolarance Tolarance to solver the dispersion function to.
///  @param[in] omega   Ray frequency.
///  @param[in] k_guess Inital guess for the wave number.
//------------------------------------------------------------------------------
template<typename DISPERSION>
void test_solve(const typename DISPERSION::base tolarance,
                const typename DISPERSION::base omega,
                const typename DISPERSION::base k_guess,
                equilibrium::unique_equilibrium<typename DISPERSION::backend> &eq) {
    auto w = graph::variable<typename DISPERSION::backend> (1, omega, "\\omega");
    auto kx = graph::variable<typename DISPERSION::backend> (1, 0.25, "k_{x}");
    auto ky = graph::variable<typename DISPERSION::backend> (1, 0.25, "k_{y}");
    auto kz = graph::variable<typename DISPERSION::backend> (1, 0.15, "k_{z}");
    auto x = graph::variable<typename DISPERSION::backend> (1, 0.0, "x");
    auto y = graph::variable<typename DISPERSION::backend> (1, 0.0, "y");
    auto z = graph::variable<typename DISPERSION::backend> (1, 0.0, "z");
    auto t = graph::variable<typename DISPERSION::backend> (1, 0.0, "t");
    
    dispersion::dispersion_interface<DISPERSION> D(w, kx, ky, kz, x, y, z, t, eq);

    kx->set(k_guess);
    graph::input_nodes<typename DISPERSION::backend> inputs({
        graph::variable_cast(w),
        graph::variable_cast(x),
        graph::variable_cast(y),
        graph::variable_cast(z),
        graph::variable_cast(ky),
        graph::variable_cast(kz),
        graph::variable_cast(t),
    });
    auto loss = D.solve(kx, inputs, tolarance);
    assert(std::abs(loss->evaluate().at(0)) < std::abs(tolarance) &&
           "Solve failed to meet expected result for kx.");

    kx->set(backend::base_cast<typename DISPERSION::backend> (0.2));
    ky->set(k_guess);
    inputs = {
        graph::variable_cast(w),
        graph::variable_cast(x),
        graph::variable_cast(y),
        graph::variable_cast(z),
        graph::variable_cast(kx),
        graph::variable_cast(kz),
        graph::variable_cast(t),
    };
    loss = D.solve(ky, inputs, tolarance);
    assert(std::abs(loss->evaluate().at(0)) < std::abs(tolarance) &&
           "Solve failed to meet expected result for ky.");

    ky->set(backend::base_cast<typename DISPERSION::backend> (0.25));
    kz->set(k_guess);
    inputs = {
        graph::variable_cast(w),
        graph::variable_cast(x),
        graph::variable_cast(y),
        graph::variable_cast(z),
        graph::variable_cast(kx),
        graph::variable_cast(ky),
        graph::variable_cast(t),
    };
    loss = D.solve(kz, inputs, tolarance);
    assert(std::abs(loss->evaluate().at(0)) < std::abs(tolarance) &&
           "Solve failed to meet expected result for kz.");

    kz->set(backend::base_cast<typename DISPERSION::backend> (0.15));
    kx->set(k_guess);
    inputs = {
        graph::variable_cast(x),
        graph::variable_cast(y),
        graph::variable_cast(z),
        graph::variable_cast(kx),
        graph::variable_cast(ky),
        graph::variable_cast(kz),
        graph::variable_cast(t),
    };
    loss = D.solve(w, inputs, tolarance);
    assert(std::abs(loss->evaluate().at(0)) < std::abs(tolarance) &&
           "Solve failed to meet expected result for w.");
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
///
///  @param[in] tolarance Tolarance to solver the dispersion function to.
//------------------------------------------------------------------------------
template<typename BACKEND> void run_tests(const typename BACKEND::base tolarance) {
    auto eq_den = equilibrium::make_guassian_density<BACKEND> ();
    auto eq_no = equilibrium::make_no_magnetic_field<BACKEND> ();
    
    test_solve<dispersion::simple<BACKEND>> (tolarance, 0.5, 1.0, eq_den);
    test_solve<dispersion::acoustic_wave<BACKEND>> (tolarance, 1.0, 600.0, eq_no);
    test_solve<dispersion::guassian_well<BACKEND>> (tolarance, 0.5, 1.0, eq_den);
    test_solve<dispersion::cold_plasma<BACKEND>> (tolarance, 900.0, 1000.0, eq_den);
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
#ifdef USE_REDUCE
    run_tests<backend::cpu<float>> (2.0E-14);
#else
    run_tests<backend::cpu<float>> (1.5E-14);
#endif
    run_tests<backend::cpu<double>> (1.0E-30);
    run_tests<backend::cpu<std::complex<float>>> (2.0E-14);
    run_tests<backend::cpu<std::complex<double>>> (1.0E-30);
}
