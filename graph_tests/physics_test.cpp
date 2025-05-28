//------------------------------------------------------------------------------
///  @file physics\_test.cpp
///  @brief Tests for math nodes.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <chrono>
#include <random>

#include "../graph_framework/solver.hpp"

//------------------------------------------------------------------------------
///  @brief Constant Test
///
///  A wave in no medium with a constant phase velocity should propagate such
///  that
///
///  k.x - wt = Constant
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
void test_constant() {
    std::mt19937_64 engine(static_cast<uint64_t> (std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())));
    std::uniform_real_distribution<double> real_dist(0.1, 1.0);

    auto omega = graph::variable<T> (1, "\\omega");
    auto kx = graph::variable<T> (1, "k_{x}");
    auto ky = graph::variable<T> (1, "k_{y}");
    auto kz = graph::variable<T> (1, "k_{z}");
    auto x = graph::variable<T> (1, "x");
    auto y = graph::variable<T> (1, "y");
    auto z = graph::variable<T> (1, "z");

    auto t = graph::variable<T> (1, "t");

    omega->set(static_cast<T> (real_dist(engine)));
    kx->set(static_cast<T> (real_dist(engine)));
    ky->set(static_cast<T> (0.0));
    kz->set(static_cast<T> (0.0));
    x->set(static_cast<T> (real_dist(engine)));
    y->set(static_cast<T> (real_dist(engine)));
    z->set(static_cast<T> (real_dist(engine)));
    t->set(static_cast<T> (0.0));

//  The equilibrum isn't used;
    auto eq = equilibrium::make_slab<T> ();
    auto dt_const = graph::constant(static_cast<T> (1.0));
    solver::rk2<dispersion::simple<T>>
        solve(omega, kx, ky, kz, x, y, z, t, dt_const, eq);
    solve.init(kx);
    solve.compile();

    auto constant = kx*x + ky*y + kz*z - omega*t;
    const auto c0 = constant->evaluate().at(0);
    for (size_t i = 0; i < 10; i++) {
        solve.step();
        solve.sync_host();
    }

    assert(std::abs(c0 - constant->evaluate().at(0)) < 5.0E-15 &&
           "Constant expression not preserved.");
}

//------------------------------------------------------------------------------
///  @brief Bohm-Gross Test
///
///  In the bohm-gross dispersion relation, the group velocity should be.
///
///  vg = 3/2*vth^2*k/⍵                                                      (1)
///
///  Where vth is the thermal velocity.
///
///  vth = sqrt(2*kb*T/m)                                                    (2)
///
///  The wave number varies with time.
///
///  k(t) = -⍵pe'(x)/(2⍵)*t + k0                                             (3)
///
///  Where ⍵pe is the plasma frequency.
///
///  ⍵pe2 = q^2*n(x))/(ϵ0*m)                                                 (4)
///
///  For a linear gradient in the density ⍵pe2'(x) is a constant.
///
///  ⍵pe2' = ne0*q^2*0.1/(ϵ0*m)                                              (5)
///
///  k0 must be a solution of the dispersion relation.
///
///  k0 = sqrt(3/2(⍵^2 - ⍵pe^2)/vth^2)                                       (6)
///
///  Putting equation 3 into 1 yields the group velocity as a function of time.
///
///  vg(t) = -3/2*vth^2/⍵*⍵pe'(x)/(2⍵)*t + 3/2*vth^2/⍵*k0                    (7)
///
///  This expression can be integrated to find a parabolic ray trajectory.
///
///  x(t) = -3/8*vth^2/⍵*⍵pe'(x)/⍵*t^2 + 3/2*vth^2/⍵*k0*t + x0               (8)
///
///  B = 0 or k || B
///
///  @tparam SOLVER Class of solver to use.
///
///  @param[in] tolarance Tolarance to solver the dispersion function to.
//------------------------------------------------------------------------------
template<solver::method SOLVER>
void test_bohm_gross(const typename SOLVER::base tolarance) {
    auto omega = graph::variable<typename SOLVER::base> (1, "\\omega");
    auto kx = graph::variable<typename SOLVER::base> (1, "k_{x}");
    auto ky = graph::variable<typename SOLVER::base> (1, "k_{y}");
    auto kz = graph::variable<typename SOLVER::base> (1, "k_{z}");
    auto x = graph::variable<typename SOLVER::base> (1, "x");
    auto y = graph::variable<typename SOLVER::base> (1, "y");
    auto z = graph::variable<typename SOLVER::base> (1, "z");
    auto t = graph::variable<typename SOLVER::base> (1, "t");

//  Constants
    const typename SOLVER::base q = 1.602176634E-19;
    const typename SOLVER::base me = 9.1093837015E-31;
    const typename SOLVER::base mu0 = M_PI*4.0E-7;
    const typename SOLVER::base epsilon0 = 8.8541878138E-12;
    const typename SOLVER::base c = 1.0/sqrt(mu0*epsilon0);
    const typename SOLVER::base omega0 = 600.0;
    const typename SOLVER::base ne0 = 1.0E19;
    const typename SOLVER::base te = 1000.0;

    const typename SOLVER::base omega2 = (ne0*0.9*q*q)/(epsilon0*me*c*c);
    const typename SOLVER::base omega2p = (ne0*0.1*q*q)/(epsilon0*me*c*c);
    const typename SOLVER::base vth2 = 2*1.602176634E-19*te/(me*c*c);

    const typename SOLVER::base k0 = std::sqrt(2.0/3.0*(omega0*omega0 - omega2)/vth2);

//  Omega must be greater than plasma frequency for the wave to propagate.
    omega->set(static_cast<typename SOLVER::base> (600.0));
    kx->set(static_cast<typename SOLVER::base> (1000.0));
    ky->set(static_cast<typename SOLVER::base> (0.0));
    kz->set(static_cast<typename SOLVER::base> (0.0));
    x->set(static_cast<typename SOLVER::base> (-1.0));
    y->set(static_cast<typename SOLVER::base> (0.0));
    z->set(static_cast<typename SOLVER::base> (0.0));
    t->set(static_cast<typename SOLVER::base> (0.0));

    auto eq = equilibrium::make_no_magnetic_field<typename SOLVER::base> ();
    auto dt_const = graph::constant(static_cast<typename SOLVER::base> (0.1));
    SOLVER solve(omega, kx, ky, kz, x, y, z, t, dt_const, eq);
    if constexpr (jit::use_cuda()) {
        solve.init(kx, tolarance);
    } else {
        solve.init(kx);
    }
    solve.compile();

    for (size_t i = 0; i < 20; i++) {
        solve.step();
        solve.sync_host();
    }
    const typename SOLVER::base time = t->evaluate().at(0);
    const typename SOLVER::base expected_x = -3.0/8.0*vth2*omega2p/(omega0*omega0)*time*time
                                           + 3.0/2.0*vth2/omega0*k0*time - 1.0;

    const auto diff_x = x->evaluate().at(0) - expected_x;
    assert(std::abs(diff_x*diff_x) < std::abs(tolarance) &&
           "Failed to reach expected x.");
}

//------------------------------------------------------------------------------
///  @brief Light wave Test
///
///  In the bohm-gross dispersion relation, the group velocity should be.
///
///  vg = c^2*k/⍵                                                            (1)
///
///  Where c is the speed of light. The wave number varies with time.
///
///  k(t) = -⍵pe'(x)/(2⍵)*t + k0                                             (3)
///
///  Where ⍵pe is the plasma frequency.
///
///  ⍵pe2 = q^2*n(x))/(ϵ0*m)                                                 (4)
///
///  For a linear gradient in the density ⍵pe2'(x) is a constant.
///
///  ⍵pe2' = ne0*q^2*0.1/(ϵ0*m)                                              (5)
///
///  k0 must be a solution of the dispersion relation.
///
///  k0 = sqrt((⍵^2 - ⍵pe^2)/c^2)                                            (6)
///
///  Putting equation 3 into 1 yields the group velocity as a function of time.
///
///  vg(t) = -c^2/⍵*⍵pe'(x)/(2⍵)*t + c^2/⍵*k0                                (7)
///
///  This expression can be integrated to find a parabolic ray trajectory.
///
///  x(t) = -1/4*c^2/⍵*⍵pe'(x)/⍵*t^2 + c^2/⍵*k0*t + x0                       (8)
///
///  B = 0
///
///  @tparam SOLVER Class of solver to use.
///
///  @param[in] tolarance Tolarance to solver the dispersion function to.
//------------------------------------------------------------------------------
template<solver::method SOLVER>
void test_light_wave(const typename SOLVER::base tolarance) {
    auto omega = graph::variable<typename SOLVER::base> (1, "\\omega");
    auto kx = graph::variable<typename SOLVER::base> (1, "k_{x}");
    auto ky = graph::variable<typename SOLVER::base> (1, "k_{y}");
    auto kz = graph::variable<typename SOLVER::base> (1, "k_{z}");
    auto x = graph::variable<typename SOLVER::base> (1, "x");
    auto y = graph::variable<typename SOLVER::base> (1, "y");
    auto z = graph::variable<typename SOLVER::base> (1, "z");
    auto t = graph::variable<typename SOLVER::base> (1, "t");

//  Constants
    const typename SOLVER::base q = 1.602176634E-19;
    const typename SOLVER::base me = 9.1093837015E-31;
    const typename SOLVER::base mu0 = M_PI*4.0E-7;
    const typename SOLVER::base epsilon0 = 8.8541878138E-12;
    const typename SOLVER::base c = 1.0/sqrt(mu0*epsilon0);
    const typename SOLVER::base omega0 = 600.0;
    const typename SOLVER::base ne0 = 1.0E19;

    const typename SOLVER::base omega2 = (ne0*0.9*q*q)/(epsilon0*me*c*c);
    const typename SOLVER::base omega2p = (ne0*0.1*q*q)/(epsilon0*me*c*c);

    const typename SOLVER::base k0 = std::sqrt(omega0*omega0 - omega2);

//  Omega must be greater than plasma frequency for the wave to propagate.
    omega->set(static_cast<typename SOLVER::base> (600.0));
    kx->set(static_cast<typename SOLVER::base> (100.0));
    ky->set(static_cast<typename SOLVER::base> (0.0));
    kz->set(static_cast<typename SOLVER::base> (0.0));
    x->set(static_cast<typename SOLVER::base> (-1.0));
    y->set(static_cast<typename SOLVER::base> (0.0));
    z->set(static_cast<typename SOLVER::base> (0.0));
    t->set(static_cast<typename SOLVER::base> (0.0));

    auto eq = equilibrium::make_no_magnetic_field<typename SOLVER::base> ();
    auto dt_const = graph::constant(static_cast<typename SOLVER::base> (0.1));
    SOLVER solve(omega, kx, ky, kz, x, y, z, t, dt_const, eq);
    solve.init(kx, tolarance);
    solve.compile();

    for (size_t i = 0; i < 20; i++) {
        solve.step();
        solve.sync_host();
    }
    const typename SOLVER::base time = t->evaluate().at(0);
    const typename SOLVER::base expected_x = -omega2p/(4.0*omega0*omega0)*time*time
                                           + k0/omega0*time - 1.0;

    const auto diff_x = x->evaluate().at(0) - expected_x;
    assert(std::abs(diff_x*diff_x) < std::abs(tolarance) &&
           "Failed to reach expected x.");
}

//------------------------------------------------------------------------------
///  @brief Ion acoustic wave Test
///
///  In the ion-wave dispersion relation, the group velocity should be.
///
///  vg = vs^2*k/⍵                                                           (1)
///
///  Where vs is the sound speed.
///
///  vs = sqrt((kb*Te - ɣ*kb*ti)/mi)                                         (2)
///
///  The wave number is constant in time.
///
///  k(t) = 0                                                                (3)
///
///  The slope of the wave trajectory is vs
///
///  dx/dt = vs^2                                                            (4)
///
///  @tparam T Base type of the calculation.
///
///  @param[in] tolarance Tolarance to solver the dispersion function to.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
void test_acoustic_wave(const T tolarance) {
    auto omega = graph::variable<T> (1, "\\omega");
    auto kx = graph::variable<T> (1, "k_{x}");
    auto ky = graph::variable<T> (1, "k_{y}");
    auto kz = graph::variable<T> (1, "k_{z}");
    auto x = graph::variable<T> (1, "x");
    auto y = graph::variable<T> (1, "y");
    auto z = graph::variable<T> (1, "z");
    auto t = graph::variable<T> (1, "t");

//  Constants
    const T q = 1.602176634E-19;
    const T mi = 3.34449469E-27;
    const T mu0 = M_PI*4.0E-7;
    const T epsilon0 = 8.8541878138E-12;
    const T c = 1.0/sqrt(mu0*epsilon0);
    const T omega0 = 1.0;
    const T te = 1000.0;
    const T ti = te;
    const T gamma = 3;

    const T vs = std::sqrt((q*te+gamma*q*ti)/mi)/c;

//  Omega must be greater than plasma frequency for the wave to propagate.
    omega->set(static_cast<T> (omega0));
    kx->set(static_cast<T> (600.0));
    ky->set(static_cast<T> (0.0));
    kz->set(static_cast<T> (0.0));
    x->set(static_cast<T> (0.0));
    y->set(static_cast<T> (0.0));
    z->set(static_cast<T> (0.0));
    t->set(static_cast<T> (0.0));

    auto eq = equilibrium::make_no_magnetic_field<T> ();
    auto dt_const = graph::constant(static_cast<T> (0.0001));
    solver::rk4<dispersion::acoustic_wave<T>>
        solve(omega, kx, ky, kz, x, y, z, t, dt_const, eq);
    solve.init(kx, tolarance);
    solve.compile();

    for (size_t i = 0; i < 20; i++) {
        solve.step();
        solve.sync_host();
    }

    const auto diff_x = x->evaluate().at(0)/t->evaluate().at(0) - vs;
    assert(std::abs(diff_x*diff_x) < std::abs(tolarance) &&
           "Ray progated at different speed.");
}

//------------------------------------------------------------------------------
///  @brief O Mode Test.
///
///  For a linear density gradient, the O-Mode cut off should be located at
///
///  1 - ⍵pe^2(x)/⍵^2 = 0                                                    (1)
///
///  ⍵^2 - 1 = ⍵pe^2                                                         (2)
///
///  The plasma frequency is defined as
///
///  ⍵pe^2 = ne0*q^2/(ϵ0*m)*(0.1*x + 1)                                      (3)
///
///  Putting equation 3 into 2 yields
///
///  ⍵^2 - 1 = ne0*q^2/(ϵ0*m)*(0.1*x + 1)                                    (4)
///
///  Solving for x
///
///  (⍵^2 - 1 - ne0*q^2/(ϵ0*m))/(ne0*q^2/(ϵ0*m)*0.1) = x                     (5)
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
void test_o_mode_wave() {
    auto omega = graph::variable<T> (1, "\\omega");
    auto kx = graph::variable<T> (1, "k_{x}");
    auto ky = graph::variable<T> (1, "k_{y}");
    auto kz = graph::variable<T> (1, "k_{z}");
    auto x = graph::variable<T> (1, "x");
    auto y = graph::variable<T> (1, "y");
    auto z = graph::variable<T> (1, "z");
    auto t = graph::variable<T> (1, "t");

    const T q = 1.602176634E-19;
    const T me = 9.1093837015E-31;
    const T mu0 = M_PI*4.0E-7;
    const T epsilon0 = 8.8541878138E-12;
    const T c = 1.0/sqrt(mu0*epsilon0);
    const T ne0 = 1.0E19;
    const T omega2 = (ne0*q*q)/(epsilon0*me*c*c);
    const T omega0 = 1000.0;

    const T x_cut = (omega0*omega0 - 1.0 - omega2)/(omega2*0.1);

//  Omega must be greater than plasma frequency for the wave to propagate.
    omega->set(static_cast<T> (omega0));
    kx->set(static_cast<T> (0.0));
    ky->set(static_cast<T> (0.0));
    kz->set(static_cast<T> (0.0));
    x->set(static_cast<T> (0.0));
    y->set(static_cast<T> (0.0));
    z->set(static_cast<T> (0.0));
    t->set(static_cast<T> (0.0));

    auto eq = equilibrium::make_slab_density<T> ();
    auto dt_const = graph::constant(static_cast<T> (0.0001));
    solver::rk4<dispersion::ordinary_wave<T>>
        solve(omega, kx, ky, kz, x, y, z, t, dt_const, eq);

    solve.init(x);

    const auto diff = x->evaluate().at(0) - x_cut;
    assert(std::abs(diff*diff) < 8.0E-10 &&
           "Failed to reach expected tolarance.");
}

//------------------------------------------------------------------------------
///  @brief Cold Plasma Dispersion Relation Right Cutoff Frequency.
///
///  There are two branches on the dispersion relation. The O-Mode branch can
///  propagate past the right cuttoff and the upper hybrid resonance but is cut
///  off at the Plasma frequency. The x-mode is cut off by the right cutoff for
///  frequencies above and trapped between the left and cutoff and the upper
///  hybird resonance.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
void test_cold_plasma_cutoffs() {
    const T omega0 = 1100.0;

    auto w = graph::variable<T> (2, omega0, "\\omega");
    auto kx = graph::variable<T> (2, 0.0, "k_{x}");
    auto ky = graph::variable<T> (2, 0.0, "k_{y}");
    auto kz = graph::variable<T> (2, 0.0, "k_{z}");
    auto x = graph::variable<T> (2, 0.0, "x");
    auto y = graph::variable<T> (2, 0.0, "y");
    auto z = graph::variable<T> (2, 0.0, "z");
    auto t = graph::variable<T> (2, 0.0, "t");

    auto eq = equilibrium::make_slab_density<T> ();
    auto dt_const = graph::constant(static_cast<T> (0.1));
    solver::rk4<dispersion::cold_plasma<T>>
        solve(w, kx, ky, kz, x, y, z, t, dt_const, eq);

//  Solve for plasma frequency and right cutoff..
    x->set(0, static_cast<T> (25.0));
    x->set(1, static_cast<T> (5.0));
    solve.init(x);

    T wpecut_pos = x->evaluate().at(0);
    const T wrcut_pos = x->evaluate().at(1);

//  Set wave back to zero.
    x->set(0, static_cast<T> (0.0));
    x->set(1, static_cast<T> (0.0));

//  Solve for X-Mode and O-Mode wave numbers.
    kx->set(0, static_cast<T> (1000.0)); // O-Mode
    kx->set(1, static_cast<T> (500.0));  // X-Mode
    solve.init(kx);
    solve.compile();

    while (std::abs(t->evaluate().at(0)) < 30.0) {
        solve.step();
        solve.sync_host();
    }

    backend::buffer<T> result = x->evaluate();
    assert(std::real(result.at(0)) > std::real(wrcut_pos) &&
           std::real(result.at(0)) < std::real(wpecut_pos) &&
           "Expected O-Mode to cross right cuttoff but not plasma cutoff.");
    assert(std::real(result.at(1)) < std::real(wrcut_pos) &&
           "Expected X-Mode to stay above right cuttoff.");

//  Setup problem for trapped modes.
    w->set(0, static_cast<T> (800.0));
    w->set(1, static_cast<T> (800.0));

//  Solve for plasma frequency and left cutoff..
    x->set(0, static_cast<T> (25.0));
    x->set(1, static_cast<T> (5.0));
    kx->set(0, static_cast<T> (0.0));
    kx->set(1, static_cast<T> (0.0));
    t->set(0, static_cast<T> (0.0));
    t->set(1, static_cast<T> (0.0));
    if constexpr (jit::complex_scalar<T>) {
        solve.init(x, 2.8E-29);
    } else {
        solve.init(x, 5.0E-30);
    }
    solve.sync_device();

    wpecut_pos = x->evaluate().at(1);

//  Set wave back to zero.
    x->set(0, static_cast<T> (0.0));
    x->set(1, static_cast<T> (0.0));

//  Solve for X-Mode and O-Mode wave numbers.
    kx->set(0, static_cast<T> (500.0));  // O-Mode
    kx->set(1, static_cast<T> (1500.0)); // X-Mode
    if constexpr (jit::complex_scalar<T>) {
        solve.init(kx, 2.2E-30);
    } else {
        solve.init(kx);
    }
    solve.sync_device();

    while (std::abs(t->evaluate().at(0)) < 60.0) {
        solve.step();
        solve.sync_host();
    }

    result = x->evaluate();
    assert(std::real(result.at(0)) < std::real(wpecut_pos) &&
           "Expected O-Mode to stay above plasma cuttoff.");
    assert(std::real(result.at(1)) > std::real(wpecut_pos) &&
           "Expected X-Mode to cross plasma cutoff.");
}

//------------------------------------------------------------------------------
///  @brief Reflection test.
///
///  Given a wave frequency, a wave with zero k will not propagate.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] tolarance Tolarance to solver the dispersion function to.
///  @param[in] n0        Starting nz value.
///  @param[in] x0        Starting x guess.
///  @param[in] kx0       Starting kx guess.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
void test_reflection(const T tolarance,
                     const T n0,
                     const T x0,
                     const T kx0) {
    const T q = static_cast<T> (1.602176634E-19);
    const T me = static_cast<T> (9.1093837015E-31);
    const T mu0 = static_cast<T> (M_PI*4.0E-7);
    const T epsilon0 = static_cast<T> (8.8541878138E-12);
    const T c = static_cast<T> (1.0)/sqrt(mu0*epsilon0);
    const T OmegaCE = -q/(me*c);

    auto w = graph::variable<T> (1, OmegaCE, "\\omega");
    auto kx = graph::variable<T> (1, 0.0, "k_{x}");
    auto ky = graph::variable<T> (1, 0.0, "k_{y}");
    auto kz = graph::variable<T> (1, n0*OmegaCE, "k_{z}");
    auto x = graph::variable<T> (1, x0, "x");
    auto y = graph::variable<T> (1, 0.0, "y");
    auto z = graph::variable<T> (1, 0.0, "z");
    auto t = graph::variable<T> (1, 0.0, "t");

    auto eq = equilibrium::make_slab<T> ();
    auto dt_const = graph::constant(static_cast<T> (0.0001));
    solver::rk4<dispersion::cold_plasma<T>>
        solve(w, kx, ky, kz, x, y, z, t, dt_const, eq);

// Solve for a location where the wave is cut off.
    solve.init(x, tolarance);
    const T cuttoff_location = x->evaluate().at(0);

//  Set the ray starting point close to the cut off to reduce the number of
//  times steps that need to be taken.
    x->set(cuttoff_location - static_cast<T> (0.00001)*cuttoff_location);

//  Set an inital guess for kx and solve for the wave number at the new
//  location.
    kx->set(kx0);
    solve.init(kx, tolarance);
    solve.compile();
    solve.sync_host();

    auto max_x = std::real(x->evaluate().at(0));
    auto new_x = max_x;
    do {
        solve.step();
        solve.sync_host();
        new_x = std::real(x->evaluate().at(0));
        max_x = std::max(new_x, max_x);
        if constexpr (jit::use_cuda() || !jit::use_gpu<T> ()) {
            assert(std::abs(max_x - cuttoff_location) < 1.9E-6 &&
                   "Ray exceeded cutoff.");
        } else {
            assert(max_x < std::abs(cuttoff_location) &&
                   "Ray exceeded cutoff.");
        }
    } while (max_x == new_x);
}

//------------------------------------------------------------------------------
///  @brief Efit equilibrium test.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_efit() {
    auto omega = graph::variable<T> (1, "\\omega");
    auto kx = graph::variable<T> (1, "k_{x}");
    auto ky = graph::variable<T> (1, "k_{y}");
    auto kz = graph::variable<T> (1, "k_{z}");
    auto x = graph::variable<T> (1, "x");
    auto y = graph::variable<T> (1, "y");
    auto z = graph::variable<T> (1, "z");
    auto t = graph::variable<T> (1, "t");

    const T omega0 = 590.0;

//  Omega must be greater than plasma frequency for the wave to propagate.
    omega->set(static_cast<T> (omega0));
    kx->set(static_cast<T> (-600.0));
    ky->set(static_cast<T> (0.0));
    kz->set(static_cast<T> (0.0));
    x->set(static_cast<T> (2.5));
    y->set(static_cast<T> (0.0));
    z->set(static_cast<T> (0.0));
    t->set(static_cast<T> (0.0));
    
    auto eq = equilibrium::make_efit<T> (EFIT_FILE);
    auto dt_const = graph::constant(static_cast<T> (0.0001));
    solver::rk4<dispersion::ordinary_wave<T>>
        solve(omega, kx, ky, kz, x, y, z, t, dt_const, eq);
    solve.init(kx);
    solve.compile();
    
    for (size_t i = 0; i < 10000; i++) {
        solve.step();
    }

    solve.sync_host();
    assert(std::abs(kx->evaluate().at(0)) > 0 && "Expected wave to reflect.");
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] tolarance Tolarance to solver the dispersion function to.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void run_tests(const T tolarance) {
    test_constant<T> ();
    test_bohm_gross<solver::rk4<dispersion::bohm_gross<T>>> (tolarance);
    test_bohm_gross<solver::split_simplextic<dispersion::bohm_gross<T>>> (tolarance);
    test_light_wave<solver::rk4<dispersion::light_wave<T>>> (tolarance);
    test_light_wave<solver::split_simplextic<dispersion::light_wave<T>>> (tolarance);
    test_acoustic_wave<T> (tolarance);
    test_o_mode_wave<T> ();
    test_reflection<T> (tolarance, 0.7, 0.1, 22.0);
    test_cold_plasma_cutoffs<T> ();
    test_efit<T> ();
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
//  There is not enough precision in float to pass the test.
    if constexpr (jit::use_cuda()) {
        run_tests<double> (1.6E-21);
    } else {
        run_tests<double> (2.0E-29);
    }
    run_tests<std::complex<double>> (3.2E-27);
    END_GPU
}
