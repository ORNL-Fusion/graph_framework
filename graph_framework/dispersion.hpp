//------------------------------------------------------------------------------
///  @file dispersion.hpp
///  @brief Base class for a dispersion relation.
///
///  Defines a dispersion function.
//------------------------------------------------------------------------------

#ifndef dispersion_h
#define dispersion_h

#include "newton.hpp"
#include "equilibrium.hpp"

namespace dispersion {
//******************************************************************************
//  Z Function interface.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class interface to build dispersion relation functions.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class z_function {
    public:
//------------------------------------------------------------------------------
///  @brief Method to build the Z function.
///
///  @params[in] zeta The zeta argument.
///  @returns The constructed Z function.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        Z(graph::shared_leaf<T, SAFE_MATH> zeta) = 0;

///  Type def to retrieve the backend base type.
        typedef T base;
///  Retrieve template parameter of safe math.
        static constexpr bool safe_math = SAFE_MATH;
    };

//------------------------------------------------------------------------------
///  @brief Class interface to build dispersion relation functions.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::complex_scalar T, bool SAFE_MATH=false>
    class z_power_series final : public z_function<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Method to build the Z function.
///
///  @params[in] zeta The zeta argument.
///  @returns The constructed Z function.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        Z(graph::shared_leaf<T, SAFE_MATH> zeta) {
            auto zeta2 = zeta*zeta;
            auto zeta4 = zeta2*zeta2;
            auto zeta6 = zeta4*zeta2;
            return graph::i<T>*std::sqrt(M_PI)/graph::exp(zeta2) -
                   2.0*(1.0 - 2.0/3.0*zeta2
                            + 4.0/15.0*zeta4
                            - 8.0/105.0*zeta6)*zeta;
        }
    };

//------------------------------------------------------------------------------
///  @brief Class interface to build dispersion relation functions.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::complex_scalar T, bool SAFE_MATH=false>
    class z_erfi final : public z_function<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Method to build the Z function.
///
///  @params[in] zeta The zeta argument.
///  @returns The constructed Z function.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        Z(graph::shared_leaf<T, SAFE_MATH> zeta) {
            return -std::sqrt(M_PI)*graph::exp(-zeta*zeta)*(graph::erfi(zeta) -
                                                            graph::i<T>);
        }
    };

///  Dispersion concept.
    template<class Z>
    concept z_func = std::is_base_of<z_function<typename Z::base, Z::safe_math>, Z>::value;

//******************************************************************************
//  Common physics expressions.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Build plasma fequency expression.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] n       Density.
///  @param[in] q       Species charge.
///  @param[in] m       Species mass.
///  @param[in] c       Speed of light
///  @param[in] epsion0 Vacuum permitixity.
///  @returns The plasma frequency.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    static constexpr graph::shared_leaf<T, SAFE_MATH>
    build_plasma_fequency(graph::shared_leaf<T, SAFE_MATH> n,
                          const T q,
                          const T m,
                          const T c,
                          const T epsion0) {
        return n*q*q/(epsion0*m*c*c);
    }

//------------------------------------------------------------------------------
///  @brief Build cyclotron fequency expression.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] q Species charge.
///  @param[in] b Magnetic field.
///  @param[in] m Species mass.
///  @param[in] c Speed of light
///  @returns The cyclotron frequency.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    static constexpr graph::shared_leaf<T, SAFE_MATH>
    build_cyclotron_fequency(const T q,
                             graph::shared_leaf<T, SAFE_MATH> b,
                             const T m,
                             const T c) {
        return q*b/(m*c);
    }

//******************************************************************************
//  Dispersion function.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Interface for dispersion functions.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class dispersion_function {
    public:
//------------------------------------------------------------------------------
///  @brief Interface for a dispersion function.
///
///  @params[in] w  Omega variable.
///  @params[in] kx Kx variable.
///  @params[in] ky Ky variable.
///  @params[in] kz Kz variable.
///  @params[in] x  x variable.
///  @params[in] y  y variable.
///  @params[in] z  z variable.
///  @params[in] t  Current time.
///  @params[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        D(graph::shared_leaf<T, SAFE_MATH> w,
          graph::shared_leaf<T, SAFE_MATH> kx,
          graph::shared_leaf<T, SAFE_MATH> ky,
          graph::shared_leaf<T, SAFE_MATH> kz,
          graph::shared_leaf<T, SAFE_MATH> x,
          graph::shared_leaf<T, SAFE_MATH> y,
          graph::shared_leaf<T, SAFE_MATH> z,
          graph::shared_leaf<T, SAFE_MATH> t,
          equilibrium::shared<T, SAFE_MATH> &eq) = 0;

///  Type def to retrieve the backend base type.
        typedef T base;
///  Retrieve template parameter of safe math.
        static constexpr bool safe_math = SAFE_MATH;
    };

//------------------------------------------------------------------------------
///  @brief Stiff dispersion function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class stiff final : public dispersion_function<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Stiff function.
///
///  This is not really a dispersion function but is an example of a stiff
///  system.
///
///  dx/dt = -1.0E3*(x - Exp(-t)) - Exp(-t)                                  (1)
///
///  We need to figure out a disperison function D(w,k,x) such that
///
///  dx/dt = -(dD/dk)/(dD/dw) = -1.0E3*(x - Exp(-t)) - Exp(-t).              (2)
///
///  If we assume,
///
///  D = (1.0E3*(x - Exp(-t)) - Exp(-t))*kx + w                              (3)
///
///  dD/dw = 1                                                               (4)
///
///  dD/dkx = (1.0E3*(x - Exp(-t)) - Exp(-t))                                (5)
///
///  This satisfies equations 1.
///
///  @params[in] w  Omega variable.
///  @params[in] kx Kx variable.
///  @params[in] ky Ky variable.
///  @params[in] kz Kz variable.
///  @params[in] x  x variable.
///  @params[in] y  y variable.
///  @params[in] z  z variable.
///  @params[in] t  Current time.
///  @params[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        D(graph::shared_leaf<T, SAFE_MATH> w,
          graph::shared_leaf<T, SAFE_MATH> kx,
          graph::shared_leaf<T, SAFE_MATH> ky,
          graph::shared_leaf<T, SAFE_MATH> kz,
          graph::shared_leaf<T, SAFE_MATH> x,
          graph::shared_leaf<T, SAFE_MATH> y,
          graph::shared_leaf<T, SAFE_MATH> z,
          graph::shared_leaf<T, SAFE_MATH> t,
          equilibrium::shared<T, SAFE_MATH> &eq) {
            return (1.0E3*(x - graph::exp(-t)) - graph::exp(-t))*kx + w;
        }
    };

//------------------------------------------------------------------------------
///  @brief Simple dispersion function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class simple final : public dispersion_function<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Simple dispersion function.
///
///  D = npar^2 + nperp^2 - 1
///
///  @params[in] w  Omega variable.
///  @params[in] kx Kx variable.
///  @params[in] ky Ky variable.
///  @params[in] kz Kz variable.
///  @params[in] x  x variable.
///  @params[in] y  y variable.
///  @params[in] z  z variable.
///  @params[in] t  Current time.
///  @params[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        D(graph::shared_leaf<T, SAFE_MATH> w,
          graph::shared_leaf<T, SAFE_MATH> kx,
          graph::shared_leaf<T, SAFE_MATH> ky,
          graph::shared_leaf<T, SAFE_MATH> kz,
          graph::shared_leaf<T, SAFE_MATH> x,
          graph::shared_leaf<T, SAFE_MATH> y,
          graph::shared_leaf<T, SAFE_MATH> z,
          graph::shared_leaf<T, SAFE_MATH> t,
          equilibrium::shared<T, SAFE_MATH> &eq) {
            const T c = 1.0;

            auto npar2 = kz*kz*c*c/(w*w);
            auto nperp2 = (kx*kx + ky*ky)*c*c/(w*w);
            return npar2 + nperp2 - c;
        }
    };

//------------------------------------------------------------------------------
///  @brief Physics
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class physics : public dispersion_function<T, SAFE_MATH> {
    protected:
//  Define some common constants.
///  Vacuum permitivity.
        const T epsion0 = 8.8541878138E-12;
///  Vacuum permeability
        const T mu0 = M_PI*4.0E-7;
///  Fundamental charge.
        const T q = 1.602176634E-19;
///  Electron mass.
        const T me = 9.1093837015E-31;
/// Speed of light.
        const T c = static_cast<T> (1.0)/std::sqrt(epsion0*mu0);
    };

//------------------------------------------------------------------------------
///  @brief Bohm-Gross dispersion function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class bohm_gross final : public physics<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Bohm-Gross function.
///
///  D = ⍵_p^2 + 3/2(kx^2 + ky^2 + kz^2)vth^2 - ⍵^2                          (1)
///
///  vth = Sqrt(2*ne*te/me)                                                  (2)
///
///  @params[in] w  Omega variable.
///  @params[in] kx Kx variable.
///  @params[in] ky Ky variable.
///  @params[in] kz Kz variable.
///  @params[in] x  x variable.
///  @params[in] y  y variable.
///  @params[in] z  z variable.
///  @params[in] t  Current time.
///  @params[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        D(graph::shared_leaf<T, SAFE_MATH> w,
          graph::shared_leaf<T, SAFE_MATH> kx,
          graph::shared_leaf<T, SAFE_MATH> ky,
          graph::shared_leaf<T, SAFE_MATH> kz,
          graph::shared_leaf<T, SAFE_MATH> x,
          graph::shared_leaf<T, SAFE_MATH> y,
          graph::shared_leaf<T, SAFE_MATH> z,
          graph::shared_leaf<T, SAFE_MATH> t,
          equilibrium::shared<T, SAFE_MATH> &eq) {

//  Equilibrium quantities.
            auto ne = eq->get_electron_density(x, y, z);
            auto wpe2 = build_plasma_fequency(ne, physics<T, SAFE_MATH>::q,
                                              physics<T, SAFE_MATH>::me,
                                              physics<T, SAFE_MATH>::c,
                                              physics<T, SAFE_MATH>::epsion0);
            auto te = eq->get_electron_temperature(x, y, z);
//  2*1.602176634E-19 to convert eV to J.

            auto vterm2 = static_cast<T> (2.0)*physics<T, SAFE_MATH>::q*te
                        / (physics<T, SAFE_MATH>::me *
                           physics<T, SAFE_MATH>::c *
                           physics<T, SAFE_MATH>::c);

//  Wave numbers should be parallel to B if there is a magnetic field. Otherwise
//  B should be zero.
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto k = kx*eq->get_esup1(x, y, z)
                   + ky*eq->get_esup2(x, y, z)
                   + kz*eq->get_esup3(x, y, z);
            graph::shared_leaf<T, SAFE_MATH> kpara2;
            if (b_vec->length()->is_match(graph::zero<T, SAFE_MATH> ())) {
                kpara2 = k->dot(k);
            } else {
                auto b_hat = b_vec->unit();
                auto kpara = b_hat->dot(k);
                kpara2 = kpara*kpara;
            }
            
            return wpe2 + 3.0/2.0*kpara2*vterm2 - w*w;
        }
    };

//------------------------------------------------------------------------------
///  @brief Light Wave dispersion function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class light_wave final : public physics<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Light-wave function.
///
///  D = ⍵_p^2 + 3/2(kx^2 + ky^2 + kz^2)c^2 - ⍵^2                            (1)
///
///  B = 0.
///
///  @params[in] w  Omega variable.
///  @params[in] kx Kx variable.
///  @params[in] ky Ky variable.
///  @params[in] kz Kz variable.
///  @params[in] x  x variable.
///  @params[in] y  y variable.
///  @params[in] z  z variable.
///  @params[in] t  Current time.
///  @params[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        D(graph::shared_leaf<T, SAFE_MATH> w,
          graph::shared_leaf<T, SAFE_MATH> kx,
          graph::shared_leaf<T, SAFE_MATH> ky,
          graph::shared_leaf<T, SAFE_MATH> kz,
          graph::shared_leaf<T, SAFE_MATH> x,
          graph::shared_leaf<T, SAFE_MATH> y,
          graph::shared_leaf<T, SAFE_MATH> z,
          graph::shared_leaf<T, SAFE_MATH> t,
          equilibrium::shared<T, SAFE_MATH> &eq) {

//  Equilibrium quantities.
            auto ne = eq->get_electron_density(x, y, z);
            auto wpe2 = build_plasma_fequency(ne,
                                              physics<T, SAFE_MATH>::q,
                                              physics<T, SAFE_MATH>::me,
                                              physics<T, SAFE_MATH>::c,
                                              physics<T, SAFE_MATH>::epsion0);

//  Wave numbers should be parallel to B if there is a magnetic field. Otherwise
//  B should be zero.
            assert(eq->get_magnetic_field(x, y, z)->length()->is_match(graph::zero<T, SAFE_MATH> ()) &&
                   "Expected equilibrium with no magnetic field.");

            auto k = kx*eq->get_esup1(x, y, z)
                   + ky*eq->get_esup2(x, y, z)
                   + kz*eq->get_esup3(x, y, z);
            auto k2 = k->dot(k);
            
            return wpe2 + k2 - w*w;
        }
    };

//------------------------------------------------------------------------------
///  @brief Ion wave dispersion function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class acoustic_wave final : public physics<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Ion acoustic wave function.
///
///  D = (kx^2 + ky^2 + kz^2)vs^2 - ⍵^2                                      (1)
///
///  vs = Sqrt(kb*Te/M + ɣ*kb*Ti/M)                                          (2)
///
///  @params[in] w  Omega variable.
///  @params[in] kx Kx variable.
///  @params[in] ky Ky variable.
///  @params[in] kz Kz variable.
///  @params[in] x  x variable.
///  @params[in] y  y variable.
///  @params[in] z  z variable.
///  @params[in] t  Current time.
///  @params[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        D(graph::shared_leaf<T, SAFE_MATH> w,
          graph::shared_leaf<T, SAFE_MATH> kx,
          graph::shared_leaf<T, SAFE_MATH> ky,
          graph::shared_leaf<T, SAFE_MATH> kz,
          graph::shared_leaf<T, SAFE_MATH> x,
          graph::shared_leaf<T, SAFE_MATH> y,
          graph::shared_leaf<T, SAFE_MATH> z,
          graph::shared_leaf<T, SAFE_MATH> t,
          equilibrium::shared<T, SAFE_MATH> &eq) {

//  Equilibrium quantities.
            const T mi = eq->get_ion_mass(0);
            auto te = eq->get_electron_temperature(x, y, z);
            auto ti = eq->get_ion_temperature(0, x, y, z);
            const T gamma = 3.0;
            auto vs2 = (physics<T, SAFE_MATH>::q*te + gamma*physics<T, SAFE_MATH>::q*ti)
                     / (mi*physics<T, SAFE_MATH>::c*physics<T, SAFE_MATH>::c);

//  Wave numbers should be parallel to B if there is a magnetic field. Otherwise
//  B should be zero.
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto k = kx*eq->get_esup1(x, y, z)
                   + ky*eq->get_esup2(x, y, z)
                   + kz*eq->get_esup3(x, y, z);
            graph::shared_leaf<T, SAFE_MATH> kpara2;
            if (b_vec->length()->is_match(graph::zero<T, SAFE_MATH> ())) {
                kpara2 = k->dot(k);
            } else {
                auto b_hat = b_vec->unit();
                auto kpara = b_hat->dot(k);
                kpara2 = kpara*kpara;
            }
            
            return kpara2*vs2 - w*w;
        }
    };

//------------------------------------------------------------------------------
///  @brief Guassian Well dispersion function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class guassian_well final : public dispersion_function<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Disperison relation with a non uniform well.
///
///  D = npar^2 + nperp^2 - (1 - 0.5*Exp(-x^2/0.1)
///
///  @params[in] w  Omega variable.
///  @params[in] kx Kx variable.
///  @params[in] ky Ky variable.
///  @params[in] kz Kz variable.
///  @params[in] x  x variable.
///  @params[in] y  y variable.
///  @params[in] z  z variable.
///  @params[in] t  Current time.
///  @params[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        D(graph::shared_leaf<T, SAFE_MATH> w,
          graph::shared_leaf<T, SAFE_MATH> kx,
          graph::shared_leaf<T, SAFE_MATH> ky,
          graph::shared_leaf<T, SAFE_MATH> kz,
          graph::shared_leaf<T, SAFE_MATH> x,
          graph::shared_leaf<T, SAFE_MATH> y,
          graph::shared_leaf<T, SAFE_MATH> z,
          graph::shared_leaf<T, SAFE_MATH> t,
          equilibrium::shared<T, SAFE_MATH> &eq) {
            const T c = 1.0;
            auto well = c - 0.5*exp(-(x*x + y*y)/0.1);
            auto npar2 = kz*kz*c*c/(w*w);
            auto nperp2 = (kx*kx + ky*ky)*c*c/(w*w);
            return npar2 + nperp2 - well;
        }
    };

//------------------------------------------------------------------------------
///  @brief Electrostatic ion cyclotron wave dispersion function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class ion_cyclotron final : public physics<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Disperison relation for the O mode.
///
///  D = ⍵ce^2 + k^2*vs^2 - ⍵^2                                              (1)
///
///  ⍵ce is the electron cyclotron frequency and vs
///
///  vs = Sqrt(kb*Te/M + ɣ*kb*Ti/M)                                          (2)
///
///  @params[in] w  Omega variable.
///  @params[in] kx Kx variable.
///  @params[in] ky Ky variable.
///  @params[in] kz Kz variable.
///  @params[in] x  x variable.
///  @params[in] y  y variable.
///  @params[in] z  z variable.
///  @params[in] t  Current time.
///  @params[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        D(graph::shared_leaf<T, SAFE_MATH> w,
          graph::shared_leaf<T, SAFE_MATH> kx,
          graph::shared_leaf<T, SAFE_MATH> ky,
          graph::shared_leaf<T, SAFE_MATH> kz,
          graph::shared_leaf<T, SAFE_MATH> x,
          graph::shared_leaf<T, SAFE_MATH> y,
          graph::shared_leaf<T, SAFE_MATH> z,
          graph::shared_leaf<T, SAFE_MATH> t,
          equilibrium::shared<T, SAFE_MATH> &eq) {
//  Equilibrium quantities.
            const T mi = eq->get_ion_mass(0);

            auto te = eq->get_electron_temperature(x, y, z);
            auto ti = eq->get_ion_temperature(0, x, y, z);
            const T gamma = 3.0;
            auto vs2 = (physics<T, SAFE_MATH>::q*te +
                        gamma*physics<T, SAFE_MATH>::q*ti)
                     / (mi*physics<T, SAFE_MATH>::c *
                        physics<T, SAFE_MATH>::c);
            
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto wce = build_cyclotron_fequency(-physics<T, SAFE_MATH>::q,
                                                b_vec->length(),
                                                physics<T, SAFE_MATH>::me,
                                                physics<T, SAFE_MATH>::c);

//  Wave numbers.
            auto k = kx*eq->get_esup1(x, y, z)
                   + ky*eq->get_esup2(x, y, z)
                   + kz*eq->get_esup3(x, y, z);
            auto b_hat = b_vec->unit();
            auto kperp = b_hat->cross(k)->length();
            auto kperp2 = kperp*kperp;

            auto w2 = w*w;
                        
            return wce - kperp2*vs2 - w*w;
        }
    };

//------------------------------------------------------------------------------
///  @brief Ordinary wave dispersion function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class ordinary_wave final : public physics<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Disperison relation for the O mode.
///
///  D = 1 - ⍵pe^2/⍵^2 - c^2/⍵^2*(kx^2 + ky^2 + kz^2)                        (1)
///
///  ⍵pe is the plasma frequency.
///
///  @params[in] w  Omega variable.
///  @params[in] kx Kx variable.
///  @params[in] ky Ky variable.
///  @params[in] kz Kz variable.
///  @params[in] x  x variable.
///  @params[in] y  y variable.
///  @params[in] z  z variable.
///  @params[in] t  Current time.
///  @params[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        D(graph::shared_leaf<T, SAFE_MATH> w,
          graph::shared_leaf<T, SAFE_MATH> kx,
          graph::shared_leaf<T, SAFE_MATH> ky,
          graph::shared_leaf<T, SAFE_MATH> kz,
          graph::shared_leaf<T, SAFE_MATH> x,
          graph::shared_leaf<T, SAFE_MATH> y,
          graph::shared_leaf<T, SAFE_MATH> z,
          graph::shared_leaf<T, SAFE_MATH> t,
          equilibrium::shared<T, SAFE_MATH> &eq) {
//  Equilibrium quantities.
            auto ne = eq->get_electron_density(x, y, z);
            auto wpe2 = build_plasma_fequency(ne,
                                              physics<T, SAFE_MATH>::q,
                                              physics<T, SAFE_MATH>::me,
                                              physics<T, SAFE_MATH>::c,
                                              physics<T, SAFE_MATH>::epsion0);

//  Wave numbers.
            auto n = (kx*eq->get_esup1(x, y, z) +
                      ky*eq->get_esup2(x, y, z) +
                      kz*eq->get_esup3(x, y, z))/w;
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto b_hat = b_vec->unit();
            auto nperp = b_hat->cross(n)->length();
            auto nperp2 = nperp*nperp;

            auto w2 = w*w;

            return 1.0 - wpe2/w2 - nperp2;
        }
    };

//------------------------------------------------------------------------------
///  @brief Extra ordinary wave dispersion function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class extra_ordinary_wave final : public physics<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Disperison relation for the X-Mode.
///
///  D = 1 - ⍵pe^2/⍵^2(⍵^2 - ⍵pe^2)/(⍵^2 - ⍵h^2)
///    - c^2/⍵^2*(kx^2 + ky^2 + kz^2)                                        (1)
///
///  Where ⍵h is the upper hybrid frequency and defined by
///
///  ⍵h = ⍵pe^2 + ⍵ce^2
///
///  ⍵pe is the plasma frequency while ⍵ce is the cyclotron frequency.
///
///  @params[in] w  Omega variable.
///  @params[in] kx Kx variable.
///  @params[in] ky Ky variable.
///  @params[in] kz Kz variable.
///  @params[in] x  x variable.
///  @params[in] y  y variable.
///  @params[in] z  z variable.
///  @params[in] t  Current time.
///  @params[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        D(graph::shared_leaf<T, SAFE_MATH> w,
          graph::shared_leaf<T, SAFE_MATH> kx,
          graph::shared_leaf<T, SAFE_MATH> ky,
          graph::shared_leaf<T, SAFE_MATH> kz,
          graph::shared_leaf<T, SAFE_MATH> x,
          graph::shared_leaf<T, SAFE_MATH> y,
          graph::shared_leaf<T, SAFE_MATH> z,
          graph::shared_leaf<T, SAFE_MATH> t,
          equilibrium::shared<T, SAFE_MATH> &eq) {
//  Equilibrium quantities.
            auto ne = eq->get_electron_density(x, y, z);
            auto wpe2 = build_plasma_fequency(ne,
                                              physics<T, SAFE_MATH>::q,
                                              physics<T, SAFE_MATH>::me,
                                              physics<T, SAFE_MATH>::c,
                                              physics<T, SAFE_MATH>::epsion0);
            
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto b_len = b_vec->length();
            auto wec = build_cyclotron_fequency(-physics<T, SAFE_MATH>::q,
                                                b_len,
                                                physics<T, SAFE_MATH>::me,
                                                physics<T, SAFE_MATH>::c);
            
//  Wave numbers.
            auto n = (kx*eq->get_esup1(x, y, z) +
                      ky*eq->get_esup2(x, y, z) +
                      kz*eq->get_esup3(x, y, z))/w;
            auto b_hat = b_vec->unit();
            auto nperp = b_hat->cross(n)->length();
            auto nperp2 = nperp*nperp;
        
            auto wh = wpe2 + wec*wec;
            
            auto w2 = w*w;
            
            return 1.0 - wpe2/(w2)*(w2 - wpe2)/(w2 - wh) - nperp;
        }
    };

//------------------------------------------------------------------------------
///  @brief Cold Plasma Disperison function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class cold_plasma : public physics<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Cold Plasma Disperison function.
///
///  D = Det[ϵ + nn - n^2I]                                                  (1)
///
///      [  ϵ_11 ϵ_12 0    ]
///  ϵ = [ -ϵ_12 ϵ_11 0    ]                                                 (2)
///      [  0    0    ϵ_33 ]
///
///  s represents each plasma species.
///
///  ϵ_11 = 1 - Σ_s ɑ(s)^2/(1 - ɣ(s)^2)                                      (3)
///  ϵ_12 = -iΣ_s ɣ(s)ɑ(s)^2/(1 - ɣ(s)^2)                                    (4)
///  ϵ_33 = 1 - Σ_s ɑ(s)^2
///
///  ɑ(s) is the normalized plasma frequency for a plasma species and ɣ(s) is
///  the normalized cyclotron frequency for each species. Note that electrons
///  have a negative charge.
///
///  ɑ = ⍵_p/⍵                                                               (5)
///  ɣ = Ω_c/⍵                                                               (6)
///
///  The plasma frequency is defined as
///
///  ⍵_p^2 = n*q^2/ϵ0m                                                       (7)
///  Ω_c = qB/m                                                              (8)
///
///  @params[in] w  Omega variable.
///  @params[in] kx Kx variable.
///  @params[in] ky Ky variable.
///  @params[in] kz Kz variable.
///  @params[in] x  x variable.
///  @params[in] y  y variable.
///  @params[in] z  z variable.
///  @params[in] t  Current time.
///  @params[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        D(graph::shared_leaf<T, SAFE_MATH> w,
          graph::shared_leaf<T, SAFE_MATH> kx,
          graph::shared_leaf<T, SAFE_MATH> ky,
          graph::shared_leaf<T, SAFE_MATH> kz,
          graph::shared_leaf<T, SAFE_MATH> x,
          graph::shared_leaf<T, SAFE_MATH> y,
          graph::shared_leaf<T, SAFE_MATH> z,
          graph::shared_leaf<T, SAFE_MATH> t,
          equilibrium::shared<T, SAFE_MATH> &eq) {
//  Dielectric terms.
//  Frequencies
            auto ne = eq->get_electron_density(x, y, z);
            auto wpe2 = build_plasma_fequency(ne,
                                              physics<T, SAFE_MATH>::q,
                                              physics<T, SAFE_MATH>::me,
                                              physics<T, SAFE_MATH>::c,
                                              physics<T, SAFE_MATH>::epsion0);
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto b_len = b_vec->length();
            auto ec = build_cyclotron_fequency(-physics<T, SAFE_MATH>::q,
                                               b_len,
                                               physics<T, SAFE_MATH>::me,
                                               physics<T, SAFE_MATH>::c);

            auto w2 = w*w;
            auto denome = 1.0 - ec*ec/w2;
            auto e11 = 1.0 - (wpe2/w2)/denome;
            auto e12 = ((ec/w)*(wpe2/w2))/denome;
            auto e33 = wpe2;

            for (size_t i = 0, ie = eq->get_num_ion_species(); i < ie; i++) {
                const T mi = eq->get_ion_mass(i);
                const T charge = static_cast<T> (eq->get_ion_charge(i))
                               * physics<T, SAFE_MATH>::q;

                auto ni = eq->get_ion_density(i, x, y, z);
                auto wpi2 = build_plasma_fequency(ni, charge, mi,
                                                  physics<T, SAFE_MATH>::c,
                                                  physics<T, SAFE_MATH>::epsion0);
                auto ic = build_cyclotron_fequency(charge, b_len, mi,
                                                   physics<T, SAFE_MATH>::c);

                auto denomi = 1.0 - ic*ic/w2;
                e11 = e11 - (wpi2/w2)/denomi;
                e12 = e12 + ((ic/w)*(wpi2/w2))/denomi;
                e33 = e33 + wpi2;
            }

            e12 = -1.0*e12;
            e33 = 1.0 - e33/w2;

//  Wave numbers.
            auto n = (kx*eq->get_esup1(x, y, z) +
                      ky*eq->get_esup2(x, y, z) +
                      kz*eq->get_esup3(x, y, z))/w;
            auto b_hat = b_vec->unit();

            auto npara = b_hat->dot(n);
            auto npara2 = npara*npara;
            auto nperp = b_hat->cross(n)->length();
            auto nperp2 = nperp*nperp;
            
//  Determinate matrix elements
            auto m11 = e11 - npara2;
            auto m12 = e12;
            auto m13 = npara*nperp;
            auto m22 = e11 - npara2 - nperp2;
            auto m33 = e33 - nperp2;

            return (m11*m22 - m12*m12)*m33 - m22*(m13*m13);
        }
    };

//------------------------------------------------------------------------------
///  @brief Cold Plasma expansion disperison function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class cold_plasma_expansion : public physics<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Cold Plasma expansion Disperison function.
///
///  Dc = -P/2(1 + Ωe/⍵)Γ0 + (1 - Ωe^2/⍵^2)Γ1                                (1)
///
///  Γ0 = n⟂^2(n^2 - 2(1 - 2q)) + (1 - P)(2(1 - 2q) - (n^2 + n||^2))         (2)
///
///  Γ1 = n⟂^2((1 - q)n^2 - (1 - 2q))
///     + (1 - P)(n^2n||^2 - (1 - q)(n^2 + n||^2) + (1 - 2q))                (3)
///
///  P = ⍵pe^2/⍵^2                                                           (4)
///
///  q = P/(2(1 + Ωe/⍵))                                                     (5)
///
///  @params[in] w  Omega variable.
///  @params[in] kx Kx variable.
///  @params[in] ky Ky variable.
///  @params[in] kz Kz variable.
///  @params[in] x  x variable.
///  @params[in] y  y variable.
///  @params[in] z  z variable.
///  @params[in] t  Current time.
///  @params[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        D(graph::shared_leaf<T, SAFE_MATH> w,
          graph::shared_leaf<T, SAFE_MATH> kx,
          graph::shared_leaf<T, SAFE_MATH> ky,
          graph::shared_leaf<T, SAFE_MATH> kz,
          graph::shared_leaf<T, SAFE_MATH> x,
          graph::shared_leaf<T, SAFE_MATH> y,
          graph::shared_leaf<T, SAFE_MATH> z,
          graph::shared_leaf<T, SAFE_MATH> t,
          equilibrium::shared<T, SAFE_MATH> &eq) {
//  Setup plasma parameters.
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto b_len = b_vec->length();
            auto b_hat = b_vec/b_len;
            auto ne = eq->get_electron_density(x, y, z);
            auto te = eq->get_electron_temperature(x, y, z);

            auto ve = graph::sqrt(2.0*physics<T, SAFE_MATH>::q*te /
                                  physics<T, SAFE_MATH>::me)
                    / physics<T, SAFE_MATH>::c;

//  Setup characteristic frequencies.
            auto ec = build_cyclotron_fequency(physics<T, SAFE_MATH>::q,
                                               b_len,
                                               physics<T, SAFE_MATH>::me,
                                               physics<T, SAFE_MATH>::c);
            auto wpe2 = build_plasma_fequency(ne, physics<T, SAFE_MATH>::q,
                                              physics<T, SAFE_MATH>::me,
                                              physics<T, SAFE_MATH>::c,
                                              physics<T, SAFE_MATH>::epsion0);

//  Disperison quantities.
            auto P = wpe2/(w*w);
            auto q = P/(2.0*(1.0 + ec/w));

            auto n = (kx*eq->get_esup1(x, y, z) +
                      ky*eq->get_esup2(x, y, z) +
                      kz*eq->get_esup3(x, y, z))/w;
            auto n2 = n->dot(n);
            auto npara = n->dot(b_hat);
            auto npara2 = npara*npara;
            auto nperp = b_hat->cross(n)->length();
            auto nperp2 = nperp*nperp;
            auto n2nperp2 = n2*nperp2;

            auto q_func = 1.0 - 2.0*q;
            auto n_func = n2 + npara2;
            auto p_func = 1.0 - P;

            auto gamma1 = (1.0 - q)*n2nperp2
                        + p_func*(n2*npara2 - (1.0 - q)*n_func)
                        + q_func*(p_func - nperp2);
            auto gamma0 = nperp2*(n2 - 2.0*q_func) + p_func*(2.0*q_func - n_func);

            return -P/2.0*(1.0 + ec/w)*gamma0 + (1.0 - ec*ec/(w*w))*gamma1;
        }
    };

//------------------------------------------------------------------------------
///  @brief Hot Plasma Disperison function.
//------------------------------------------------------------------------------
    template<jit::complex_scalar T, z_func Z, bool SAFE_MATH=false>
    class hot_plasma final : public physics<T, SAFE_MATH> {
    private:
///  Z function.
        Z z;

    public:
//------------------------------------------------------------------------------
///  @brief Hot Plasma Disperison function.
///
///  D = iσΓ0 + Γ1 + n⟂^2P⍵/Ωe(1 + ζZ(ζ))(Γ2 + Γ5F)                          (1)
///
///  Γ0 = n⟂^2(n^2 - 2(1 - 2q)) + (1 - P)(2(1 - 2q) - (n^2 + n||^2))         (2)
///
///  Γ1 = n⟂^2((1 - q)n^2 - (1 - 2q))
///     + (1 - P)(n^2n||^2 - (1 - q)(n^2 + n||^2) + (1 - 2q))                (3)
///
///  Γ2 = (n^2 - (1 - 2q)) + P⍵/(4Ωen||^2)((n^2 + n||^2) - 2(1 - 2q))        (4)
///
///  Γ5 = n^2n||^2 - (1 - q)(n^2 + n||^2) + (1 - 2q)                         (5)
///
///  iσ = PZ(ζ)/(2n||ve)                                                     (6)
///
///  ζ = (1 - Ωe/⍵)/(n||ve/c)                                                (7)
///
///  F = ve(1 + ζZ(ζ))⍵/(2n||Ωe)                                             (8)
///
///  P = ⍵pe^2/⍵^2                                                           (9)
///
///  q = P/(2(1 + Ωe/⍵))                                                    (10)
///
///  ve = Sqrt(2*ne*te/me)                                                  (11)
///
///  @params[in] w  Omega variable.
///  @params[in] kx Kx variable.
///  @params[in] ky Ky variable.
///  @params[in] kz Kz variable.
///  @params[in] x  x variable.
///  @params[in] y  y variable.
///  @params[in] z  z variable.
///  @params[in] t  Current time.
///  @params[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        D(graph::shared_leaf<T, SAFE_MATH> w,
          graph::shared_leaf<T, SAFE_MATH> kx,
          graph::shared_leaf<T, SAFE_MATH> ky,
          graph::shared_leaf<T, SAFE_MATH> kz,
          graph::shared_leaf<T, SAFE_MATH> x,
          graph::shared_leaf<T, SAFE_MATH> y,
          graph::shared_leaf<T, SAFE_MATH> z,
          graph::shared_leaf<T, SAFE_MATH> t,
          equilibrium::shared<T, SAFE_MATH> &eq) {
//  Setup plasma parameters.
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto b_len = b_vec->length();
            auto b_hat = b_vec/b_len;
            auto ne = eq->get_electron_density(x, y, z);
            auto te = eq->get_electron_temperature(x, y, z);
            
            auto ve = graph::sqrt(2.0*physics<T, SAFE_MATH>::q*te /
                                  physics<T, SAFE_MATH>::me)
                    / physics<T, SAFE_MATH>::c;

//  Setup characteristic frequencies.
            auto ec = build_cyclotron_fequency(physics<T, SAFE_MATH>::q,
                                               b_len,
                                               physics<T, SAFE_MATH>::me,
                                               physics<T, SAFE_MATH>::c);
            auto wpe2 = build_plasma_fequency(ne, physics<T, SAFE_MATH>::q,
                                              physics<T, SAFE_MATH>::me,
                                              physics<T, SAFE_MATH>::c,
                                              physics<T, SAFE_MATH>::epsion0);

//  Disperison quantities.
            auto P = wpe2/(w*w);
            auto q = P/(2.0*(1.0 + ec/w));

            auto n = (kx*eq->get_esup1(x, y, z) +
                      ky*eq->get_esup2(x, y, z) +
                      kz*eq->get_esup3(x, y, z))/w;
            auto n2 = n->dot(n);
            auto npara = n->dot(b_hat);
            auto npara2 = npara*npara;
            auto nperp = b_hat->cross(n)->length();
            auto nperp2 = nperp*nperp;

            auto zeta = (1.0 - ec/w)/(npara*ve);
            auto Z_func = this->z.Z(zeta);
            auto zeta_func = 1.0 + zeta*Z_func;
            auto F = ve*zeta*w/(2.0*npara*ec);
            auto isigma = P*Z_func/(2.0*npara*ve);

            auto q_func = 1.0 - 2.0*q;
            auto n_func = n2 + npara2;
            auto p_func = 1.0 - P;

            auto gamma5 = n2*npara2 - (1.0 - q)*n_func + q_func;
            auto gamma2 = (n2 - q_func)
                        + P*w/(4.0*ec*npara2)*(n_func - 2.0*q_func);
            auto gamma1 = nperp2*((1.0 - q)*n2 - q_func)
                        + p_func*(n2*npara2 - (1.0 - q)*n_func + q_func);
            auto gamma0 = nperp2*(n2 - 2.0*q_func) + p_func*(2.0*q_func - n_func);

            return isigma*gamma0 + gamma1 + nperp2*P*w/ec*zeta_func*(gamma2 + gamma5*F);
        }
    };

//------------------------------------------------------------------------------
///  @brief Hot Plasma Expansion Disperison function.
///
///  @tparam T         Base type of the calculation.
///  @tparam Z         Z function class.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, class Z, bool SAFE_MATH=false>
    class hot_plasma_expansion final : public physics<T, SAFE_MATH> {
    private:
///  Z function.
        Z z;

    public:
//------------------------------------------------------------------------------
///  @brief Hot plasma expansion dispersion function.
///
///  Dw = -(1 + Ωe/⍵)n||ve/c(Γ1 + Γ2 +
///                          n⟂^2/(2n||)⍵^2/Ωe^2ve/cζΓ5)(1/Z(ζ) + ζ)         (1)
///
///  Where:
///
///  Γ1 = (1 - q)n^2n⟂^2 + (1 - P)(n^2n||^2 - (1 - q)(n^2 + n||^2))
///     + (1 - 2q)((1 - P) - n⟂^2)                                           (2)
///
///  Γ2 = P⍵/Ωen⟂^2(n^2 - (1 - 2q))
///     + P^2⍵^2/(4Ωe^2)((n^2 + n||^2) - 2(1 - 2q))n⟂^2/n||^2                (3)
///
///  Γ5 = P(n^2n||^2 - (1 - q)(n^2 + n||^2) + (1 - 2q))                      (4)
///
///  ζ = (1 - Ωe/⍵)/(n||ve/c)                                                (5)
///
///  P = ⍵pe^2/⍵^2                                                           (6)
///
///  q = P/(2(1 + Ωe/⍵))                                                     (7)
///
///  ve = Sqrt(2*ne*te/me)                                                   (8)
///
///  @params[in] w  Omega variable.
///  @params[in] kx Kx variable.
///  @params[in] ky Ky variable.
///  @params[in] kz Kz variable.
///  @params[in] x  x variable.
///  @params[in] y  y variable.
///  @params[in] z  z variable.
///  @params[in] t  Current time.
///  @params[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        D(graph::shared_leaf<T, SAFE_MATH> w,
          graph::shared_leaf<T, SAFE_MATH> kx,
          graph::shared_leaf<T, SAFE_MATH> ky,
          graph::shared_leaf<T, SAFE_MATH> kz,
          graph::shared_leaf<T, SAFE_MATH> x,
          graph::shared_leaf<T, SAFE_MATH> y,
          graph::shared_leaf<T, SAFE_MATH> z,
          graph::shared_leaf<T, SAFE_MATH> t,
          equilibrium::shared<T, SAFE_MATH> &eq) {
//  Setup plasma parameters.
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto b_hat = b_vec->unit();
            auto b_len = b_vec->length();
            auto ne = eq->get_electron_density(x, y, z);
            auto te = eq->get_electron_temperature(x, y, z);

            auto ve = graph::sqrt(2.0*physics<T, SAFE_MATH>::q*te /
                                  physics<T, SAFE_MATH>::me);

//  Setup characteristic frequencies.
            auto ec = build_cyclotron_fequency(physics<T, SAFE_MATH>::q, b_len,
                                               physics<T, SAFE_MATH>::me,
                                               physics<T, SAFE_MATH>::c);
            auto wpe2 = build_plasma_fequency(ne, physics<T, SAFE_MATH>::q,
                                              physics<T, SAFE_MATH>::me,
                                              physics<T, SAFE_MATH>::c,
                                              physics<T, SAFE_MATH>::epsion0);
    
//  Disperison quantities.
            auto P = wpe2/(w*w);
            auto q = P/(2.0*(1.0 + ec/w));

            auto n = (kx*eq->get_esup1(x, y, z) +
                      ky*eq->get_esup2(x, y, z) +
                      kz*eq->get_esup3(x, y, z))/w;
            auto n2 = n->dot(n);
            auto npara = b_hat->dot(n);
            auto npara2 = npara*npara;
            auto nperp = b_hat->cross(n)->length();
            auto nperp2 = nperp*nperp;
    
            auto vtnorm = ve/physics<T, SAFE_MATH>::c;

            auto zeta = (1.0 - ec/w)/(npara*vtnorm);
            auto Z_func = this->z.Z(zeta);

            auto q_func = 1.0 - 2.0*q;
            auto n_func = n2 + npara2;
            auto n2nperp2 = n2*nperp2;
            auto p_func = 1.0 - P;

            auto gamma5 = P*(n2*npara2 - (1.0 - q)*n_func + q_func);
            auto gamma2 = P*w/ec*nperp2*(n2 - q_func)
                        + P*P*w*w/(4.0*ec*ec)*(n_func - 2.0*q_func)*nperp2/npara2;
            auto gamma1 = (1.0 - q)*n2nperp2
                        + p_func*(n2*npara2 - (1.0 - q)*n_func)
                        + q_func*(p_func - nperp2);

            return -(1.0 + ec/w)*npara*vtnorm *
                   (gamma1 + gamma2 + nperp2/(2.0*npara)*(w*w/(ec*ec))*vtnorm*zeta*gamma5)*(1.0/Z_func + zeta);
        }
    };

//******************************************************************************
//  Dispersion interface.
//******************************************************************************
///  Dispersion concept.
    template<class D>
    concept function = std::is_base_of<dispersion_function<typename D::base, D::safe_math>, D>::value;

//------------------------------------------------------------------------------
///  @brief Class interface to build dispersion relation functions.
///
///  @tparam DISPERSION_FUNCTION Class of dispersion function to use.
//------------------------------------------------------------------------------
    template<function DISPERSION_FUNCTION>
    class dispersion_interface {
    protected:
///  Disperison function.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> D;

///  Derivative with respect to kx.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> dxdt;
///  Derivative with respect to ky.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> dydt;
///  Derivative with respect to kz.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> dzdt;
///  Derivative with respect to kx.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> dkxdt;
///  Derivative with respect to ky.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> dkydt;
///  Derivative with respect to kz.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> dkzdt;
///  Derivative with respect to omega.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> dsdt;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a new dispersion_interface.
///
///  @params[in] w  Wave frequency.
///  @params[in] kx Wave number in x.
///  @params[in] ky Wave number in y.
///  @params[in] kz Wave number in z.
///  @params[in] x  Position in x.
///  @params[in] y  Position in y.
///  @params[in] z  Position in z.
///  @params[in] t  Current time.
///  @params[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        dispersion_interface(graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                                DISPERSION_FUNCTION::safe_math> w,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                                DISPERSION_FUNCTION::safe_math> kx,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                                DISPERSION_FUNCTION::safe_math> ky,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                                DISPERSION_FUNCTION::safe_math> kz,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                                DISPERSION_FUNCTION::safe_math> x,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                                DISPERSION_FUNCTION::safe_math> y,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                                DISPERSION_FUNCTION::safe_math> z,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                                DISPERSION_FUNCTION::safe_math> t,
                             equilibrium::shared<typename DISPERSION_FUNCTION::base,
                                                 DISPERSION_FUNCTION::safe_math> &eq) :
        D(DISPERSION_FUNCTION().D(w, kx, ky, kz, x, y, z, t, eq)) {
            auto dDdw = this->D->df(w);
            auto dDdkx = this->D->df(kx);
            auto dDdky = this->D->df(ky);
            auto dDdkz = this->D->df(kz);
            auto dDdx = this->D->df(x);
            auto dDdy = this->D->df(y);
            auto dDdz = this->D->df(z);

            if (graph::pseudo_variable_cast(x).get()) {
                dDdw = dDdw->remove_pseudo();
                dDdkx = dDdkx->remove_pseudo();
                dDdky = dDdky->remove_pseudo();
                dDdkz = dDdkz->remove_pseudo();
                dDdx = dDdx->remove_pseudo();
                dDdy = dDdy->remove_pseudo();
                dDdz = dDdz->remove_pseudo();
            }

            dxdt = -dDdkx/dDdw;
            dydt = -dDdky/dDdw;
            dzdt = -dDdkz/dDdw;
            dkxdt = dDdx/dDdw;
            dkydt = dDdy/dDdw;
            dkzdt = dDdz/dDdw;

            dsdt = graph::vector(dxdt, dydt, dzdt)->length();
        }

//------------------------------------------------------------------------------
///  @brief Solve the dispersion relation for x.
///
///  This uses newtons methods to solver for D(x) = 0.
///
///  @params[in,out] x              The unknown to solver for.
///  @params[in]     inputs         Inputs for jit compile.
///  @params[in]     index          Concurrent index.
///  @params[in]     tolarance      Tolarance to solve the dispersion function
///                                 to.
///  @params[in]     max_iterations Maximum number of iterations before giving
///                                 up.
///  @returns The residule graph.
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math>
        solve(graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                 DISPERSION_FUNCTION::safe_math> x,
              graph::input_nodes<typename DISPERSION_FUNCTION::base,
                                 DISPERSION_FUNCTION::safe_math> inputs,
              const size_t index=0,
              const typename DISPERSION_FUNCTION::base tolarance = 1.0E-30,
              const size_t max_iterations = 1000) {
            auto x_var = graph::variable_cast(x);

            workflow::manager<typename DISPERSION_FUNCTION::base,
                              DISPERSION_FUNCTION::safe_math> work(index);

            solver::newton(work, {x}, inputs, this->D, tolarance, max_iterations);

            work.compile();
            work.run();

            work.copy_to_host(x, x_var->data());

            return this->D*this->D;
        }

//------------------------------------------------------------------------------
///  @brief Get the disperison residule.
///
///  @return D*D
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math>
        get_residule() {
            return this->D*this->D;
        }

//------------------------------------------------------------------------------
///  @brief Get the disperison function.
///
///  @return D(x,y,z,kx,ky,kz,w)
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math>
        get_d() {
            return this->D;
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for s update.
///
///  @return dx/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math>
        get_dsdt() {
            return this->dsdt;
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for x update.
///
///  @return dx/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math>
        get_dxdt() {
            return this->dxdt;
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for y update.
///
///  @return dy/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math>
        get_dydt() {
            return this->dydt;
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dz/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math>
        get_dzdt() {
            return this->dzdt;
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dkx/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math>
        get_dkxdt() {
            return this->dkxdt;
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dky/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math>
        get_dkydt() {
            return this->dkydt;
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dkz/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math>
        get_dkzdt() {
            return this->dkzdt;
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the dispersion relation.
//------------------------------------------------------------------------------
        void print_dispersion() {
            D->to_latex();
            std::cout << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the dkxdt.
//------------------------------------------------------------------------------
        void print_dkxdt() {
            get_dkxdt()->to_latex();
            std::cout << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the dkydt.
//------------------------------------------------------------------------------
        void print_dkydt() {
            get_dkydt()->to_latex();
            std::cout << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the dkzdt.
//------------------------------------------------------------------------------
        void print_dkzdt() {
            get_dkzdt()->to_latex();
            std::cout << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the dxdt.
//------------------------------------------------------------------------------
        void print_dxdt() {
            get_dxdt()->to_latex();
            std::cout << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the dydt.
//------------------------------------------------------------------------------
        void print_dydt() {
            get_dydt()->to_latex();
            std::cout << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the dzdt.
//------------------------------------------------------------------------------
        void print_dzdt() {
            get_dzdt()->to_latex();
            std::cout << std::endl;
        }
    };
}

#endif /* dispersion_h */
