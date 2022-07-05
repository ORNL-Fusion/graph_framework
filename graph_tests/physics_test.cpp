//------------------------------------------------------------------------------
///  @file physics_test.cpp
///  @brief Tests for math nodes.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <chrono>
#include <random>
#include <cassert>

#include "../graph_framework/cpu_backend.hpp"
#include "../graph_framework/solver.hpp"

//------------------------------------------------------------------------------
///  @brief Constant Test
///
///  A wave in no medium with a constant phase velocity should propagate such that
///
///  k.x - wt = Constant
//------------------------------------------------------------------------------
template<typename BACKEND>
void test_constant() {
    std::mt19937_64 engine(static_cast<uint64_t> (std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())));
    std::uniform_real_distribution<double> real_dist(0.1, 1.0);

    auto omega = graph::variable<BACKEND> (1, "\\omega");
    auto kx = graph::variable<BACKEND> (1, "k_{x}");
    auto ky = graph::variable<BACKEND> (1, "k_{y}");
    auto kz = graph::variable<BACKEND> (1, "k_{z}");
    auto x = graph::variable<BACKEND> (1, "x");
    auto y = graph::variable<BACKEND> (1, "y");
    auto z = graph::variable<BACKEND> (1, "z");
    
    auto dt = 1.0;
    auto t = graph::variable<BACKEND> (1, "t");
    
    omega->set(backend::base_cast<BACKEND> (real_dist(engine)));
    kx->set(backend::base_cast<BACKEND> (real_dist(engine)));
    ky->set(backend::base_cast<BACKEND> (0.0));
    kz->set(backend::base_cast<BACKEND> (0.0));
    x->set(backend::base_cast<BACKEND> (real_dist(engine)));
    y->set(backend::base_cast<BACKEND> (real_dist(engine)));
    z->set(backend::base_cast<BACKEND> (real_dist(engine)));
    
//  The equilibrum isn't used;
    auto eq = equilibrium::make_slab<BACKEND> ();
    solver::rk2<dispersion::simple<BACKEND>> solve(omega, kx, ky, kz, x, y, z, dt, eq);
    solve.init(kx);
    
    auto constant = kx*x + ky*y + kz*z - omega*t;
    t->set(backend::base_cast<BACKEND> (0.0));
    const auto c0 = constant->evaluate().at(0);
    for (size_t i = 0; i < 10; i++) {
        solve.step();
        t->set(backend::base_cast<BACKEND> ((i + 1)*dt));
    }

    assert(std::abs(c0 - constant->evaluate().at(0)) < 5.0E-15 &&
           "Constant expression not preserved.");
}

//------------------------------------------------------------------------------
///  @brief Bohm-Gross Test
///
///  In the bohm-gross dispersion relation, the group velocity should be.
///
///  vg = 3/2*vth^2*k/⍵                                                        (1)
///
///  Where vth is the thermal velocity.
///
///  vth = sqrt(2*kb*T/m)                                                      (2)
///
///  The wave number varies with time.
///
///  k(t) = -⍵pe'(x)/(2⍵)*t + k0                                               (3)
///
///  Where ⍵pe is the plasma frequency.
///
///  ⍵pe2 = q^2*n(x))/(ϵ0*m)                                                   (4)
///
///  For a linear gradient in the density ⍵pe2'(x) is a constant.
///
///  ⍵pe2' = ne0*q^2*0.1/(ϵ0*m)                                                (5)
///
///  k0 must be a solution of the dispersion relation.
///
///  k0 = sqrt(3/2(⍵^2 - ⍵pe^2)/vth^2)                                         (6)
///
///  Putting equation 3 into 1 yields the group velocity as a function of time.
///
///  vg(t) = -3/2*vth^2/⍵*⍵pe'(x)/(2⍵)*t + 3/2*vth^2/⍵*k0                      (7)
///
///  This expression can be integrated to find a parabolic ray trajectory.
///
///  x(t) = -3/8*vth^2/⍵*⍵pe'(x)/⍵*t^2 + 3/2*vth^2/⍵*k0*t + x0                 (8)
///
///  B = 0 or k || B
///
///  @param[in] tolarance Tolarance to solver the dispersion function to.
//------------------------------------------------------------------------------
template<typename BACKEND>
void test_bohm_gross(const typename BACKEND::base tolarance) {
    std::mt19937_64 engine(static_cast<uint64_t> (std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())));
    std::uniform_real_distribution<double> real_dist(0.1, 1.0);

    auto omega = graph::variable<BACKEND> (1, "\\omega");
    auto kx = graph::variable<BACKEND> (1, "k_{x}");
    auto ky = graph::variable<BACKEND> (1, "k_{y}");
    auto kz = graph::variable<BACKEND> (1, "k_{z}");
    auto x = graph::variable<BACKEND> (1, "x");
    auto y = graph::variable<BACKEND> (1, "y");
    auto z = graph::variable<BACKEND> (1, "z");

//  Constants
    const typename BACKEND::base q = 1.602176634E-19;
    const typename BACKEND::base me = 9.1093837015E-31;
    const typename BACKEND::base mu0 = M_PI*4.0E-7;
    const typename BACKEND::base epsilon0 = 8.8541878138E-12;
    const typename BACKEND::base c = 1.0/sqrt(mu0*epsilon0);
    const typename BACKEND::base omega0 = 600.0;
    const typename BACKEND::base ne0 = 1.0E19;
    const typename BACKEND::base te = 1000.0;
    
    const typename BACKEND::base omega2 = (ne0*0.9*q*q)/(epsilon0*me*c*c);
    const typename BACKEND::base omega2p = (ne0*0.1*q*q)/(epsilon0*me*c*c);
    const typename BACKEND::base vth2 = 2*1.602176634E-19*te/(me*c*c);
    
    const typename BACKEND::base k0 = std::sqrt(2.0/3.0*(omega0*omega0 - omega2)/vth2);
    
//  Omega must be greater than plasma frequency for the wave to propagate.
    omega->set(backend::base_cast<BACKEND> (600.0));
    kx->set(backend::base_cast<BACKEND> (1000.0));
    ky->set(backend::base_cast<BACKEND> (0.0));
    kz->set(backend::base_cast<BACKEND> (0.0));
    x->set(backend::base_cast<BACKEND> (-1.0));
    y->set(backend::base_cast<BACKEND> (0.0));
    z->set(backend::base_cast<BACKEND> (0.0));

    const typename BACKEND::base dt = 0.1;
    
    auto eq = equilibrium::make_no_magnetic_field<BACKEND> ();
    solver::rk4<dispersion::bohm_gross<BACKEND>> solve(omega, kx, ky, kz, x, y, z, dt, eq);
    solve.init(kx, tolarance);
    
    const auto diff = kx->evaluate().at(0) - k0;
    assert(std::abs(diff*diff) < 3.0E-25 &&
           "Failed to reach expected k0.");
    
    for (size_t i = 0; i < 20; i++) {
        solve.step();
    }
    const typename BACKEND::base t = 20.0*dt;
    const typename BACKEND::base expected_x = -3.0/8.0*vth2*omega2p/(omega0*omega0)*t*t
                                            + 3.0/2.0*vth2/omega0*k0*t - 1.0;
    
    const auto diff_x = x->evaluate().at(0) - expected_x;
    assert(std::abs(diff_x*diff_x) < std::abs(tolarance) &&
           "Failed to reach expected x.");
}

//------------------------------------------------------------------------------
///  @brief Light wave Test
///
///  In the bohm-gross dispersion relation, the group velocity should be.
///
///  vg = c^2*k/⍵                                                              (1)
///
///  Where c is the speed of light. The wave number varies with time.
///
///  k(t) = -⍵pe'(x)/(2⍵)*t + k0                                               (3)
///
///  Where ⍵pe is the plasma frequency.
///
///  ⍵pe2 = q^2*n(x))/(ϵ0*m)                                                   (4)
///
///  For a linear gradient in the density ⍵pe2'(x) is a constant.
///
///  ⍵pe2' = ne0*q^2*0.1/(ϵ0*m)                                                (5)
///
///  k0 must be a solution of the dispersion relation.
///
///  k0 = sqrt((⍵^2 - ⍵pe^2)/c^2)                                              (6)
///
///  Putting equation 3 into 1 yields the group velocity as a function of time.
///
///  vg(t) = -c^2/⍵*⍵pe'(x)/(2⍵)*t + c^2/⍵*k0                                  (7)
///
///  This expression can be integrated to find a parabolic ray trajectory.
///
///  x(t) = -1/4*c^2/⍵*⍵pe'(x)/⍵*t^2 + c^2/⍵*k0*t + x0                         (8)
///
///  B = 0
///
///  @param[in] tolarance Tolarance to solver the dispersion function to.
//------------------------------------------------------------------------------
template<typename BACKEND>
void test_light_wave(const typename BACKEND::base tolarance) {
    std::mt19937_64 engine(static_cast<uint64_t> (std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())));
    std::uniform_real_distribution<double> real_dist(0.1, 1.0);

    auto omega = graph::variable<BACKEND> (1, "\\omega");
    auto kx = graph::variable<BACKEND> (1, "k_{x}");
    auto ky = graph::variable<BACKEND> (1, "k_{y}");
    auto kz = graph::variable<BACKEND> (1, "k_{z}");
    auto x = graph::variable<BACKEND> (1, "x");
    auto y = graph::variable<BACKEND> (1, "y");
    auto z = graph::variable<BACKEND> (1, "z");

//  Constants
    const typename BACKEND::base q = 1.602176634E-19;
    const typename BACKEND::base me = 9.1093837015E-31;
    const typename BACKEND::base mu0 = M_PI*4.0E-7;
    const typename BACKEND::base epsilon0 = 8.8541878138E-12;
    const typename BACKEND::base c = 1.0/sqrt(mu0*epsilon0);
    const typename BACKEND::base omega0 = 600.0;
    const typename BACKEND::base ne0 = 1.0E19;
    
    const typename BACKEND::base omega2 = (ne0*0.9*q*q)/(epsilon0*me*c*c);
    const typename BACKEND::base omega2p = (ne0*0.1*q*q)/(epsilon0*me*c*c);
    
    const typename BACKEND::base k0 = std::sqrt(omega0*omega0 - omega2);
    
//  Omega must be greater than plasma frequency for the wave to propagate.
    omega->set(backend::base_cast<BACKEND> (600.0));
    kx->set(backend::base_cast<BACKEND> (100.0));
    ky->set(backend::base_cast<BACKEND> (0.0));
    kz->set(backend::base_cast<BACKEND> (0.0));
    x->set(backend::base_cast<BACKEND> (-1.0));
    y->set(backend::base_cast<BACKEND> (0.0));
    z->set(backend::base_cast<BACKEND> (0.0));

    const typename BACKEND::base dt = 0.1;
    
    auto eq = equilibrium::make_no_magnetic_field<BACKEND> ();
    solver::rk4<dispersion::light_wave<BACKEND>> solve(omega, kx, ky, kz, x, y, z, dt, eq);
    solve.init(kx, tolarance);
    
    const auto diff = kx->evaluate().at(0) - k0;
    assert(std::abs(diff*diff) < 3.0E-25 &&
           "Failed to reach expected k0.");
    
    for (size_t i = 0; i < 20; i++) {
        solve.step();
    }
    const typename BACKEND::base t = 20.0*dt;
    const typename BACKEND::base expected_x = -omega2p/(4.0*omega0*omega0)*t*t
                                            + k0/omega0*t - 1.0;
    
    const auto diff_x = x->evaluate().at(0) - expected_x;
    assert(std::abs(diff_x*diff_x) < std::abs(tolarance) &&
           "Failed to reach expected x.");
}

//------------------------------------------------------------------------------
///  @brief Ion acoustic wave Test
///
///  In the ion-wave dispersion relation, the group velocity should be.
///
///  vg = vs^2*k/⍵                                                             (1)
///
///  Where vs is the sound speed.
///
///  vs = sqrt((kb*Te - ɣ*kb*ti)/mi)                                           (2)
///
///  The wave number is constant in time.
///
///  k(t) = 0                                                                  (3)
///
///  The slope of the wave trajectory is vs
///
///  dx/dt = vs^2                                                              (4)
///
///  @param[in] tolarance Tolarance to solver the dispersion function to.
//------------------------------------------------------------------------------
template<typename BACKEND>
void test_acoustic_wave(const typename BACKEND::base tolarance) {
    std::mt19937_64 engine(static_cast<uint64_t> (std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())));
    std::uniform_real_distribution<double> real_dist(0.1, 1.0);

    auto omega = graph::variable<BACKEND> (1, "\\omega");
    auto kx = graph::variable<BACKEND> (1, "k_{x}");
    auto ky = graph::variable<BACKEND> (1, "k_{y}");
    auto kz = graph::variable<BACKEND> (1, "k_{z}");
    auto x = graph::variable<BACKEND> (1, "x");
    auto y = graph::variable<BACKEND> (1, "y");
    auto z = graph::variable<BACKEND> (1, "z");

//  Constants
    const typename BACKEND::base q = 1.602176634E-19;
    const typename BACKEND::base mi = 3.34449469E-27;
    const typename BACKEND::base mu0 = M_PI*4.0E-7;
    const typename BACKEND::base epsilon0 = 8.8541878138E-12;
    const typename BACKEND::base c = 1.0/sqrt(mu0*epsilon0);
    const typename BACKEND::base omega0 = 1.0;
    const typename BACKEND::base te = 1000.0;
    const typename BACKEND::base ti = te;
    const typename BACKEND::base gamma = 3;
    
    const typename BACKEND::base vs = std::sqrt((q*te+gamma*q*ti)/mi)/c;
    
    const typename BACKEND::base k0 = omega0/vs;
    
//  Omega must be greater than plasma frequency for the wave to propagate.
    omega->set(backend::base_cast<BACKEND> (omega0));
    kx->set(backend::base_cast<BACKEND> (600.0));
    ky->set(backend::base_cast<BACKEND> (0.0));
    kz->set(backend::base_cast<BACKEND> (0.0));
    x->set(backend::base_cast<BACKEND> (0.0));
    y->set(backend::base_cast<BACKEND> (0.0));
    z->set(backend::base_cast<BACKEND> (0.0));

    auto eq = equilibrium::make_no_magnetic_field<BACKEND> ();
    solver::rk4<dispersion::acoustic_wave<BACKEND>> solve(omega, kx, ky, kz, x, y, z, 0.0001, eq);
    solve.init(kx, tolarance);
    
    const auto diff = kx->evaluate().at(0) - k0;
    assert(std::abs(diff*diff) < 5.0E-24 &&
           "Failed to reach expected k0.");
    
    for (size_t i = 0; i < 20; i++) {
        solve.step();
    }
    const typename BACKEND::base t = 20.0*0.0001;
    
    const auto diff_x = x->evaluate().at(0)/t - vs;
    assert(std::abs(diff_x*diff_x) < std::abs(tolarance) &&
           "Ray progated at different speed.");
}

//------------------------------------------------------------------------------
///  @brief O Mode Test.
///
///  For a linear density gradient, the O-Mode cut off should be located at
///
///  1 - ⍵pe^2(x)/⍵^2 = 0                                                      (1)
///
///  ⍵^2 - 1 = ⍵pe^2                                                           (2)
///
///  The plasma frequency is defined as
///
///  ⍵pe^2 = ne0*q^2/(ϵ0*m)*(0.1*x + 1)                                        (3)
///
///  Putting equation 3 into 2 yields
///
///  ⍵^2 - 1 = ne0*q^2/(ϵ0*m)*(0.1*x + 1)                                      (4)
///
///  Solving for x
///
///  (⍵^2 - 1 - ne0*q^2/(ϵ0*m))/(ne0*q^2/(ϵ0*m)*0.1) = x                       (5)
//------------------------------------------------------------------------------
template<typename BACKEND>
void test_o_mode_wave() {
    auto omega = graph::variable<BACKEND> (1, "\\omega");
    auto kx = graph::variable<BACKEND> (1, "k_{x}");
    auto ky = graph::variable<BACKEND> (1, "k_{y}");
    auto kz = graph::variable<BACKEND> (1, "k_{z}");
    auto x = graph::variable<BACKEND> (1, "x");
    auto y = graph::variable<BACKEND> (1, "y");
    auto z = graph::variable<BACKEND> (1, "z");
    
    auto eq = equilibrium::make_slab_density<BACKEND> ();
    solver::rk4<dispersion::ordinary_wave<BACKEND>> solve(omega, kx, ky, kz, x, y, z, 0.0001, eq);
    
    const typename BACKEND::base q = 1.602176634E-19;
    const typename BACKEND::base me = 9.1093837015E-31;
    const typename BACKEND::base mu0 = M_PI*4.0E-7;
    const typename BACKEND::base epsilon0 = 8.8541878138E-12;
    const typename BACKEND::base c = 1.0/sqrt(mu0*epsilon0);
    const typename BACKEND::base ne0 = 1.0E19;
    const typename BACKEND::base omega2 = (ne0*q*q)/(epsilon0*me*c*c);
    const typename BACKEND::base omega0 = 1000.0;
    
    const typename BACKEND::base x_cut = (omega0*omega0 - 1.0 - omega2)/(omega2*0.1);
    
//  Omega must be greater than plasma frequency for the wave to propagate.
    omega->set(backend::base_cast<BACKEND> (omega0));
    kx->set(backend::base_cast<BACKEND> (0.0));
    ky->set(backend::base_cast<BACKEND> (0.0));
    kz->set(backend::base_cast<BACKEND> (0.0));
    x->set(backend::base_cast<BACKEND> (0.0));
    y->set(backend::base_cast<BACKEND> (0.0));
    z->set(backend::base_cast<BACKEND> (0.0));
    
    solve.init(x);
    
    const auto diff = x->evaluate().at(0) - x_cut;
    assert(std::abs(diff*diff) < 8.0E-10 &&
           "Failed to reach expected tolarance.");
}

//------------------------------------------------------------------------------
///  @brief Cold Plasma Dispersion Relation Right Cutoff Frequency.
///
///  Above the right cut off frequncy, there are two branches on the dispersion
///  relation. The O-Mode branch can propagate past the right cuttoff and the upper
///  hybrid resonance but is cut off at the Plasma frequency.
//------------------------------------------------------------------------------
template<typename BACKEND>
void test_reflection(const typename BACKEND::base tolarance) {
    const typename BACKEND::base q = 1.602176634E-19;
    const typename BACKEND::base me = 9.1093837015E-31;
    const typename BACKEND::base mu0 = M_PI*4.0E-7;
    const typename BACKEND::base epsilon0 = 8.8541878138E-12;
    const typename BACKEND::base c = 1.0/sqrt(mu0*epsilon0);
    const typename BACKEND::base OmegaCE = -q/(me*c);
    
    const typename BACKEND::base ne0 = 1.0E19;
    
    const typename BACKEND::base wce = -q/(me*c);
    
    const typename BACKEND::base wpe = std::sqrt(ne0*q*q/(epsilon0*me*c*c));
    const typename BACKEND::base wh = wpe + wce;
    const typename BACKEND::base wr = 0.5*(wce + std::sqrt(wce*wce + 4*wpe*wpe));
    
    const typename BACKEND::base omega0 = 1100.0;

    auto w = graph::variable<BACKEND> (1, omega0, "\\omega");
    auto kx = graph::variable<BACKEND> (1, 0.0, "k_{x}");
    auto ky = graph::variable<BACKEND> (1, 0.0, "k_{y}");
    auto kz = graph::variable<BACKEND> (1, 0.0, "k_{z}");
    auto x = graph::variable<BACKEND> (1, 0.0, "x");
    auto y = graph::variable<BACKEND> (1, 0.0, "y");
    auto z = graph::variable<BACKEND> (1, 0.0, "z");
    
    auto eq = equilibrium::make_slab_density<BACKEND> ();
    solver::rk4<dispersion::cold_plasma<BACKEND>> solve(w, kx, ky, kz, x, y, z, 0.001, eq);

//  Solve for plasma cut off.
    x->set(x->set(backend::base_cast<BACKEND> (25.0)));
    solve.init(x);
    
    const typename BACKEND::base wpecut = x->evaluate(0);

//  Solve for right cutoff.
}

//------------------------------------------------------------------------------
///  @brief Reflection test.
///
///  Given a wave frequency, a wave with zero k will not propagate.
///
///  @param[in] tolarance Tolarance to solver the dispersion function to.
///  @param[in] n0        Starting nz value.
///  @param[in] x0        Starting x guess.
///  @param[in] kx0       Starting kx guess.
//------------------------------------------------------------------------------
template<typename BACKEND>
void test_reflection(const typename BACKEND::base tolarance,
                     const typename BACKEND::base n0,
                     const typename BACKEND::base x0,
                     const typename BACKEND::base kx0) {
    const typename BACKEND::base q = 1.602176634E-19;
    const typename BACKEND::base me = 9.1093837015E-31;
    const typename BACKEND::base mu0 = M_PI*4.0E-7;
    const typename BACKEND::base epsilon0 = 8.8541878138E-12;
    const typename BACKEND::base c = 1.0/sqrt(mu0*epsilon0);
    const typename BACKEND::base OmegaCE = -q/(me*c);
    
    auto w = graph::variable<BACKEND> (1, OmegaCE, "\\omega");
    auto kx = graph::variable<BACKEND> (1, 0.0, "k_{x}");
    auto ky = graph::variable<BACKEND> (1, 0.0, "k_{y}");
    auto kz = graph::variable<BACKEND> (1, n0*OmegaCE, "k_{z}");
    auto x = graph::variable<BACKEND> (1, x0, "x");
    auto y = graph::variable<BACKEND> (1, 0.0, "y");
    auto z = graph::variable<BACKEND> (1, 0.0, "z");
    
    auto eq = equilibrium::make_slab<BACKEND> ();
    solver::rk4<dispersion::cold_plasma<BACKEND>> solve(w, kx, ky, kz, x, y, z, 0.0001, eq);

// Solve for a location where the wave is cut off.
    solve.init(x, tolarance);
    const typename BACKEND::base cuttoff_location = x->evaluate().at(0);

//  Set the ray starting point close to the cut off to reduce the number of
//  times steps that need to be taken.
    x->set(cuttoff_location - backend::base_cast<BACKEND> (0.00001)*cuttoff_location);

//  Set an inital guess for kx and solve for the wave number at the new
//  location.
    kx->set(kx0);
    solve.init(kx, tolarance);
    
    auto max_x = std::real(solve.state.back().x.at(0));
    auto new_x = max_x;
    do {
        solve.step();
        new_x = std::real(solve.state.back().x.at(0));
        max_x = std::max(new_x, max_x);
        assert(max_x < std::abs(cuttoff_location) && "Ray exceeded cutoff.");
    } while (max_x == new_x);
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
///
///  @param[in] tolarance Tolarance to solver the dispersion function to.
//------------------------------------------------------------------------------
template<typename BACKEND> void run_tests(const typename BACKEND::base tolarance) {
    test_constant<BACKEND> ();
    test_bohm_gross<BACKEND> (tolarance);
    test_light_wave<BACKEND> (tolarance);
    test_acoustic_wave<BACKEND> (tolarance);
    test_o_mode_wave<BACKEND> ();
    test_reflection<BACKEND> (tolarance, 0.7, 0.1, 22.0);
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
//  No there is not enough precision in float to pass the test.
    run_tests<backend::cpu<double>> (1.0E-30);
    run_tests<backend::cpu<std::complex<double>>> (5.0E-29);
}
