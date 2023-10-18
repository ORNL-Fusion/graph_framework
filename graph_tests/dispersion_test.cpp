//------------------------------------------------------------------------------
///  @file dispersion\_test.cpp
///  @brief Tests for math nodes.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/dispersion.hpp"

//------------------------------------------------------------------------------
///  @brief The newton solve for dispersion relation.
///
///  @tparam DISPERSION Class of dispersion function to use.
///
///  @params[in] tolarance Tolarance to solver the dispersion function to.
///  @params[in] omega   Ray frequency.
///  @params[in] k_guess Inital guess for the wave number.
//------------------------------------------------------------------------------
template<dispersion::function DISPERSION>
void test_solve(const typename DISPERSION::base tolarance,
                const typename DISPERSION::base omega,
                const typename DISPERSION::base k_guess,
                equilibrium::shared<typename DISPERSION::base> &eq) {
    auto w = graph::variable<typename DISPERSION::base> (1, omega, "\\omega");
    auto kx = graph::variable<typename DISPERSION::base> (1, 0.25, "k_{x}");
    auto ky = graph::variable<typename DISPERSION::base> (1, 0.25, "k_{y}");
    auto kz = graph::variable<typename DISPERSION::base> (1, 0.15, "k_{z}");
    auto x = graph::variable<typename DISPERSION::base> (1, 0.0, "x");
    auto y = graph::variable<typename DISPERSION::base> (1, 0.0, "y");
    auto z = graph::variable<typename DISPERSION::base> (1, 0.0, "z");
    auto t = graph::variable<typename DISPERSION::base> (1, 0.0, "t");

    dispersion::dispersion_interface<DISPERSION> D(w, kx, ky, kz, x, y, z, t, eq);

    kx->set(k_guess);
    graph::input_nodes<typename DISPERSION::base> inputs({
        graph::variable_cast(w),
        graph::variable_cast(x),
        graph::variable_cast(y),
        graph::variable_cast(z),
        graph::variable_cast(kx),
        graph::variable_cast(ky),
        graph::variable_cast(kz),
        graph::variable_cast(t),
    });
    D.solve(kx, inputs, 0, tolarance);

    kx->set(static_cast<typename DISPERSION::base> (0.2));
    D.solve(ky, inputs, 0, tolarance);

    ky->set(static_cast<typename DISPERSION::base> (0.25));
    kz->set(k_guess);
    D.solve(kz, inputs, 0, tolarance);

    kz->set(static_cast<typename DISPERSION::base> (0.15));
    kx->set(k_guess);
    D.solve(w, inputs, 0, tolarance);
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
    auto eq_den = equilibrium::make_guassian_density<T> ();
    auto eq_no = equilibrium::make_no_magnetic_field<T> ();

    test_solve<dispersion::simple<T>> (tolarance, 0.5, 1.0, eq_den);
    test_solve<dispersion::acoustic_wave<T>> (tolarance, 1.0, 600.0, eq_no);
    test_solve<dispersion::guassian_well<T>> (tolarance, 0.5, 1.0, eq_den);
    test_solve<dispersion::cold_plasma<T>> (tolarance, 900.0, 1000.0, eq_den);
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @params[in] argc Number of commandline arguments.
///  @params[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    START_GPU
    if constexpr (jit::use_cuda()) {
        run_tests<float> (3.2E-14);
    } else {
        run_tests<float> (4.0E-15);
    }
    run_tests<double> (1.0E-30);
    if constexpr (jit::use_cuda()) {
        run_tests<std::complex<float>> (5.7E-14);
    } else {
        run_tests<std::complex<float>> (2.0E-14);
    }
    run_tests<std::complex<double>> (1.0E-30);
    END_GPU
}
