//------------------------------------------------------------------------------
///  @file dispersion.hpp
///  @brief Base class for a dispersion relation.
///
///  Defines a dispersion function.
//------------------------------------------------------------------------------

#ifndef dispersion_h
#define dispersion_h

#include <iostream>
#include <cassert>

#include "vector.hpp"
#include "equilibrium.hpp"

namespace dispersion {
//******************************************************************************
//  Common physics expressions.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Build plasma fequency expression.
//------------------------------------------------------------------------------
    template<class BACKEND>
    static graph::shared_leaf<BACKEND> build_plasma_fequency(graph::shared_leaf<BACKEND> n,
                                                             graph::shared_leaf<BACKEND> q,
                                                             graph::shared_leaf<BACKEND> m,
                                                             graph::shared_leaf<BACKEND> c,
                                                             graph::shared_leaf<BACKEND> epsion0) {
        return n*q*q/(epsion0*m*c*c);
    }

//------------------------------------------------------------------------------
///  @brief Build cyclotron fequency expression.
//------------------------------------------------------------------------------
    template<class BACKEND>
    static graph::shared_leaf<BACKEND> build_cyclotron_fequency(graph::shared_leaf<BACKEND> q,
                                                                graph::shared_leaf<BACKEND> b,
                                                                graph::shared_leaf<BACKEND> m,
                                                                graph::shared_leaf<BACKEND> c) {
        return q*b/(m*c);
    }

//******************************************************************************
//  Dispersion interface.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class interface to build dispersion relation functions.
//------------------------------------------------------------------------------
    template<class DISPERSION_FUNCTION>
    class dispersion_interface {
    protected:
///  Disperison function.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> D;

///  Derivative with respect to kx.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> dxdt;
///  Derivative with respect to ky.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> dydt;
///  Derivative with respect to kz.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> dzdt;
///  Derivative with respect to kx.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> dkxdt;
///  Derivative with respect to ky.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> dkydt;
///  Derivative with respect to kz.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> dkzdt;
///  Derivative with respect to omega.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> dsdt;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a new dispersion_interface.
///
///  @param[in] w  Wave frequency.
///  @param[in] kx Wave number in x.
///  @param[in] ky Wave number in y.
///  @param[in] kz Wave number in z.
///  @param[in] x  Position in x.
///  @param[in] y  Position in y.
///  @param[in] z  Position in z.
///  @param[in] t  Current time.
///  @param[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        dispersion_interface(graph::shared_leaf<typename DISPERSION_FUNCTION::backend> w,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kx,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::backend> ky,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kz,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::backend> x,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::backend> y,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::backend> z,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::backend> t,
                             equilibrium::unique_equilibrium<typename DISPERSION_FUNCTION::backend> &eq) :
        D(DISPERSION_FUNCTION().D(w, kx, ky, kz, x, y, z, t, eq)) {
            auto dDdw = this->D->df(w)->reduce();
            auto dDdkx = this->D->df(kx)->reduce();
            auto dDdky = this->D->df(ky)->reduce();
            auto dDdkz = this->D->df(kz)->reduce();
            auto dDdx = this->D->df(x)->reduce();
            auto dDdy = this->D->df(y)->reduce();
            auto dDdz = this->D->df(z)->reduce();

            auto neg_one = graph::constant<typename DISPERSION_FUNCTION::backend> (-1);
            dxdt = neg_one*dDdkx/dDdw;
            dydt = neg_one*dDdky/dDdw;
            dzdt = neg_one*dDdkz/dDdw;
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
///  @param[in] x              The unknown to solver for.
///  @param[in] tolarance      Tolarance to solver the dispersion function to.
///  @param[in] max_iterations Maximum number of iterations before giving up.
//------------------------------------------------------------------------------
        void solve(graph::shared_leaf<typename DISPERSION_FUNCTION::backend> x,
                   const typename DISPERSION_FUNCTION::base tolarance=1.0E-30,
                   const size_t max_iterations = 1000) {
            auto loss = D*D;
            auto x_next = x
                        - loss/(loss->df(x) +
                                graph::constant<typename DISPERSION_FUNCTION::backend> (tolarance));
            
            typename DISPERSION_FUNCTION::base max_residule =
                loss->evaluate().max();
            size_t iterations = 0;
            
            while (std::abs(max_residule) > std::abs(tolarance) &&
                   iterations++ < max_iterations) {
                x->set(x_next->evaluate());
                max_residule = loss->evaluate().max();
            }

//  In release mode asserts are diaables so write error to standard err. Need to
//  flip the comparison operator because we want to assert to trip if false.
            assert(iterations < max_iterations &&
                   "Newton solve failed to converge with in given iterations.");
            if (iterations > max_iterations) {
                std::cerr << "Newton solve failed to converge with in given iterations."
                          << std::endl;
                std::cerr << "Minimum residule reached: " << max_residule
                          << std::endl;
            }
        }

//------------------------------------------------------------------------------
///  @brief Get the disperison function.
///
///  @return dx/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend>
        get_d() {
            return this->D->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for s update.
///
///  @return dx/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend>
        get_dsdt() {
            return this->dsdt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for x update.
///
///  @return dx/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend>
        get_dxdt() {
            return this->dxdt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for y update.
///
///  @return dy/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend>
        get_dydt() {
            return this->dydt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dz/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend>
        get_dzdt() {
            return this->dzdt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dkx/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend>
        get_dkxdt() {
            return this->dkxdt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dky/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend>
        get_dkydt() {
            return this->dkydt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dkz/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend>
        get_dkzdt() {
            return this->dkzdt->reduce();
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

//******************************************************************************
//  Dispersion function.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Interface for dispersion functions.
//------------------------------------------------------------------------------
    template<class BACKEND>
    class dispersion_function {
    public:
//------------------------------------------------------------------------------
///  @brief Interface for a dispersion function.
///
///  @param[in] w  Omega variable.
///  @param[in] kx Kx variable.
///  @param[in] ky Ky variable.
///  @param[in] kz Kz variable.
///  @param[in] x  x variable.
///  @param[in] y  y variable.
///  @param[in] z  z variable.
///  @param[in] t  Current time.
///  @param[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> D(graph::shared_leaf<BACKEND> w,
                                              graph::shared_leaf<BACKEND> kx,
                                              graph::shared_leaf<BACKEND> ky,
                                              graph::shared_leaf<BACKEND> kz,
                                              graph::shared_leaf<BACKEND> x,
                                              graph::shared_leaf<BACKEND> y,
                                              graph::shared_leaf<BACKEND> z,
                                              graph::shared_leaf<BACKEND> t,
                                              equilibrium::unique_equilibrium<BACKEND> &eq) = 0;

///  Type def to retrieve the backend type.
        typedef BACKEND backend;
///  Type def to retrieve the backend base type.
        typedef typename BACKEND::base base;
    };

//------------------------------------------------------------------------------
///  @brief Stiff dispersion function.
//------------------------------------------------------------------------------
    template<class BACKEND>
    class stiff final : public dispersion_function<BACKEND> {
    public:
//------------------------------------------------------------------------------
///  @brief Stiff function.
///
///  This is not really a dispersion function but is an example of a stiff system.
///
///  dx/dt = -1.0E3*(x - Exp(-t)) - Exp(-t)                                    (1)
///
///  We need to figure out a disperison function D(w,k,x) such that
///
///  dx/dt = -(dD/dk)/(dD/dw) = -1.0E3*(x - Exp(-t)) - Exp(-t).                (2)
///
///  If we assume,
///
///  D = (1.0E3*(x - Exp(-t)) - Exp(-t))*kx + w                                (3)
///
///  dD/dw = 1                                                                 (4)
///
///  dD/dkx = (1.0E3*(x - Exp(-t)) - Exp(-t))                                  (5)
///
///  This satisfies equations 1.
///
///  @param[in] w  Omega variable.
///  @param[in] kx Kx variable.
///  @param[in] ky Ky variable.
///  @param[in] kz Kz variable.
///  @param[in] x  x variable.
///  @param[in] y  y variable.
///  @param[in] z  z variable.
///  @param[in] t  Current time.
///  @param[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> D(graph::shared_leaf<BACKEND> w,
                                              graph::shared_leaf<BACKEND> kx,
                                              graph::shared_leaf<BACKEND> ky,
                                              graph::shared_leaf<BACKEND> kz,
                                              graph::shared_leaf<BACKEND> x,
                                              graph::shared_leaf<BACKEND> y,
                                              graph::shared_leaf<BACKEND> z,
                                              graph::shared_leaf<BACKEND> t,
                                              equilibrium::unique_equilibrium<BACKEND> &eq) final {
            auto none = graph::constant<BACKEND> (-1);
            auto c = graph::constant<BACKEND> (1.0E3);
            return (c*(x - graph::exp(none*t)) - graph::exp(none*t))*kx + w;
        }
    };

//------------------------------------------------------------------------------
///  @brief Simple dispersion function.
//------------------------------------------------------------------------------
    template<class BACKEND>
    class simple final : public dispersion_function<BACKEND> {
    public:
//------------------------------------------------------------------------------
///  @brief Simple dispersion function.
///
///  D = npar^2 + nperp^2 - 1
///
///  @param[in] w  Omega variable.
///  @param[in] kx Kx variable.
///  @param[in] ky Ky variable.
///  @param[in] kz Kz variable.
///  @param[in] x  x variable.
///  @param[in] y  y variable.
///  @param[in] z  z variable.
///  @param[in] t  Current time.
///  @param[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> D(graph::shared_leaf<BACKEND> w,
                                              graph::shared_leaf<BACKEND> kx,
                                              graph::shared_leaf<BACKEND> ky,
                                              graph::shared_leaf<BACKEND> kz,
                                              graph::shared_leaf<BACKEND> x,
                                              graph::shared_leaf<BACKEND> y,
                                              graph::shared_leaf<BACKEND> z,
                                              graph::shared_leaf<BACKEND> t,
                                              equilibrium::unique_equilibrium<BACKEND> &eq) final {
            auto c = graph::constant<BACKEND> (1);

            auto npar2 = kz*kz*c*c/(w*w);
            auto nperp2 = (kx*kx + ky*ky)*c*c/(w*w);
            return npar2 + nperp2 - c;
        }
    };

//------------------------------------------------------------------------------
///  @brief Bohm-Gross dispersion function.
//------------------------------------------------------------------------------
    template<class BACKEND>
    class bohm_gross final : public dispersion_function<BACKEND> {
    public:
//------------------------------------------------------------------------------
///  @brief Bohm-Gross function.
///
///  D = ⍵_p^2 + 3/2(kx^2 + ky^2 + kz^2)vth^2 - ⍵^2                            (1)
///
///  vth = Sqrt(2*ne*te/me)                                                    (2)
///
///  @param[in] w  Omega variable.
///  @param[in] kx Kx variable.
///  @param[in] ky Ky variable.
///  @param[in] kz Kz variable.
///  @param[in] x  x variable.
///  @param[in] y  y variable.
///  @param[in] z  z variable.
///  @param[in] t  Current time.
///  @param[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> D(graph::shared_leaf<BACKEND> w,
                                              graph::shared_leaf<BACKEND> kx,
                                              graph::shared_leaf<BACKEND> ky,
                                              graph::shared_leaf<BACKEND> kz,
                                              graph::shared_leaf<BACKEND> x,
                                              graph::shared_leaf<BACKEND> y,
                                              graph::shared_leaf<BACKEND> z,
                                              graph::shared_leaf<BACKEND> t,
                                              equilibrium::unique_equilibrium<BACKEND> &eq) final {
//  Constants
            auto epsion0 = graph::constant<BACKEND> (8.8541878138E-12);
            auto mu0 = graph::constant<BACKEND> (M_PI*4.0E-7);
            auto c = graph::constant<BACKEND> (1)/graph::sqrt(epsion0*mu0);

//  Equilibrium quantities.
            auto me = graph::constant<BACKEND> (9.1093837015E-31);
            auto q = graph::constant<BACKEND> (1.602176634E-19);

            auto ne = eq->get_electron_density(x, y, z);
            auto wpe2 = build_plasma_fequency(ne, q, me, c, epsion0);
            auto te = eq->get_electron_temperature(x, y, z);
//  2*1.602176634E-19 to convert eV to J.
            auto temp = graph::constant<BACKEND> (2.602176634E-19)*te;
            auto vterm2 = graph::constant<BACKEND> (2*1.602176634E-19)*te/(me*c*c);

//  Wave numbers should be parallel to B if there is a magnetic field. Otherwise
//  B should be zero.
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto k = graph::vector(kx, ky, kz);
            graph::shared_leaf<BACKEND> kpara2;
            auto zero = graph::constant<BACKEND> (0.0);
            if (b_vec->length()->is_match(zero)) {
                kpara2 = k->dot(k);
            } else {
                auto b_hat = b_vec->unit();
                auto kpara = b_hat->dot(k);
                kpara2 = kpara*kpara;
            }
            
            return wpe2 +
                   graph::constant<BACKEND> (3.0/2.0)*kpara2*vterm2 -
                   w*w;
        }
    };

//------------------------------------------------------------------------------
///  @brief Light Wave dispersion function.
//------------------------------------------------------------------------------
    template<class BACKEND>
    class light_wave final : public dispersion_function<BACKEND> {
    public:
//------------------------------------------------------------------------------
///  @brief Bohm-Gross function.
///
///  D = ⍵_p^2 + 3/2(kx^2 + ky^2 + kz^2)c^2 - ⍵^2                              (1)
///
///  B = 0.
///
///  @param[in] w  Omega variable.
///  @param[in] kx Kx variable.
///  @param[in] ky Ky variable.
///  @param[in] kz Kz variable.
///  @param[in] x  x variable.
///  @param[in] y  y variable.
///  @param[in] z  z variable.
///  @param[in] t  Current time.
///  @param[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> D(graph::shared_leaf<BACKEND> w,
                                              graph::shared_leaf<BACKEND> kx,
                                              graph::shared_leaf<BACKEND> ky,
                                              graph::shared_leaf<BACKEND> kz,
                                              graph::shared_leaf<BACKEND> x,
                                              graph::shared_leaf<BACKEND> y,
                                              graph::shared_leaf<BACKEND> z,
                                              graph::shared_leaf<BACKEND> t,
                                              equilibrium::unique_equilibrium<BACKEND> &eq) final {
//  Constants
            auto epsion0 = graph::constant<BACKEND> (8.8541878138E-12);
            auto mu0 = graph::constant<BACKEND> (M_PI*4.0E-7);
            auto c = graph::constant<BACKEND> (1)/graph::sqrt(epsion0*mu0);

//  Equilibrium quantities.
            auto me = graph::constant<BACKEND> (9.1093837015E-31);
            auto q = graph::constant<BACKEND> (1.602176634E-19);

            auto ne = eq->get_electron_density(x, y, z);
            auto wpe2 = build_plasma_fequency(ne, q, me, c, epsion0);

//  Wave numbers should be parallel to B if there is a magnetic field. Otherwise
//  B should be zero.
            auto zero = graph::constant<BACKEND> (0.0);
            assert(eq->get_magnetic_field(x, y, z)->length()->is_match(zero) &&
                   "Expected equilibrium with no magnetic field.");
                   
            auto k = graph::vector(kx, ky, kz);
            auto k2 = k->dot(k);
            
            return wpe2 + k2 - w*w;
        }
    };

//------------------------------------------------------------------------------
///  @brief Ion wave dispersion function.
//------------------------------------------------------------------------------
    template<class BACKEND>
    class acoustic_wave final : public dispersion_function<BACKEND> {
    public:
//------------------------------------------------------------------------------
///  @brief Ion acoustic wave function.
///
///  D = (kx^2 + ky^2 + kz^2)vs^2 - ⍵^2                                        (1)
///
///  vs = Sqrt(kb*Te/M + ɣ*kb*Ti/M)                                            (2)
///
///  @param[in] w  Omega variable.
///  @param[in] kx Kx variable.
///  @param[in] ky Ky variable.
///  @param[in] kz Kz variable.
///  @param[in] x  x variable.
///  @param[in] y  y variable.
///  @param[in] z  z variable.
///  @param[in] t  Current time.
///  @param[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> D(graph::shared_leaf<BACKEND> w,
                                              graph::shared_leaf<BACKEND> kx,
                                              graph::shared_leaf<BACKEND> ky,
                                              graph::shared_leaf<BACKEND> kz,
                                              graph::shared_leaf<BACKEND> x,
                                              graph::shared_leaf<BACKEND> y,
                                              graph::shared_leaf<BACKEND> z,
                                              graph::shared_leaf<BACKEND> t,
                                              equilibrium::unique_equilibrium<BACKEND> &eq) final {
//  Constants
            auto epsion0 = graph::constant<BACKEND> (8.8541878138E-12);
            auto mu0 = graph::constant<BACKEND> (M_PI*4.0E-7);
            auto c = graph::constant<BACKEND> (1)/graph::sqrt(epsion0*mu0);

//  Equilibrium quantities.
            auto mi = graph::constant<BACKEND> (eq->get_ion_mass(0));
            auto q = graph::constant<BACKEND> (1.602176634E-19);

            auto te = eq->get_electron_temperature(x, y, z);
            auto ti = eq->get_ion_temperature(0, x, y, z);
            auto gamma = graph::constant<BACKEND> (3.0);
            auto vs2 = (q*te + gamma*q*ti)/(mi*c*c);

//  Wave numbers should be parallel to B if there is a magnetic field. Otherwise
//  B should be zero.
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto k = graph::vector(kx, ky, kz);
            graph::shared_leaf<BACKEND> kpara2;
            auto zero = graph::constant<BACKEND> (0.0);
            if (b_vec->length()->is_match(zero)) {
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
//------------------------------------------------------------------------------
    template<class BACKEND>
    class guassian_well final : public dispersion_function<BACKEND> {
    public:
//------------------------------------------------------------------------------
///  @brief Disperison relation with a non uniform well.
///
///  D = npar^2 + nperp^2 - (1 - 0.5*Exp(-x^2/0.1)
///
///  @param[in] w  Omega variable.
///  @param[in] kx Kx variable.
///  @param[in] ky Ky variable.
///  @param[in] kz Kz variable.
///  @param[in] x  x variable.
///  @param[in] y  y variable.
///  @param[in] z  z variable.
///  @param[in] t  Current time.
///  @param[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> D(graph::shared_leaf<BACKEND> w,
                                              graph::shared_leaf<BACKEND> kx,
                                              graph::shared_leaf<BACKEND> ky,
                                              graph::shared_leaf<BACKEND> kz,
                                              graph::shared_leaf<BACKEND> x,
                                              graph::shared_leaf<BACKEND> y,
                                              graph::shared_leaf<BACKEND> z,
                                              graph::shared_leaf<BACKEND> t,
                                              equilibrium::unique_equilibrium<BACKEND> &eq) final {
            auto c = graph::constant<BACKEND> (1);
            auto well = c - graph::constant<BACKEND>(0.5)*exp(graph::constant<BACKEND> (-1)*(x*x + y*y)/graph::constant<BACKEND> (0.1));
            auto npar2 = kz*kz*c*c/(w*w);
            auto nperp2 = (kx*kx + ky*ky)*c*c/(w*w);
            return npar2 + nperp2 - well;
        }
    };

//------------------------------------------------------------------------------
///  @brief Electrostatic ion cyclotron wave dispersion function.
//------------------------------------------------------------------------------
    template<class BACKEND>
    class ion_cyclotron final : public dispersion_function<BACKEND> {
    public:
//------------------------------------------------------------------------------
///  @brief Disperison relation for the O mode.
///
///  D = ⍵ce^2 + k^2*vs^2 - ⍵^2                                                (1)
///
///  ⍵ce is the electron cyclotron frequency and vs
///
///  vs = Sqrt(kb*Te/M + ɣ*kb*Ti/M)                                            (2)
///
///  @param[in] w  Omega variable.
///  @param[in] kx Kx variable.
///  @param[in] ky Ky variable.
///  @param[in] kz Kz variable.
///  @param[in] x  x variable.
///  @param[in] y  y variable.
///  @param[in] z  z variable.
///  @param[in] t  Current time.
///  @param[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> D(graph::shared_leaf<BACKEND> w,
                                              graph::shared_leaf<BACKEND> kx,
                                              graph::shared_leaf<BACKEND> ky,
                                              graph::shared_leaf<BACKEND> kz,
                                              graph::shared_leaf<BACKEND> x,
                                              graph::shared_leaf<BACKEND> y,
                                              graph::shared_leaf<BACKEND> z,
                                              graph::shared_leaf<BACKEND> t,
                                              equilibrium::unique_equilibrium<BACKEND> &eq) final {
//  Constants
            auto epsion0 = graph::constant<BACKEND> (8.8541878138E-12);
            auto mu0 = graph::constant<BACKEND> (M_PI*4.0E-7);
            auto c = graph::constant<BACKEND> (1.0)/graph::sqrt(epsion0*mu0);
            auto one = graph::constant<BACKEND> (1);
            auto none = graph::constant<BACKEND> (-1);
                        
//  Equilibrium quantities.
            auto me = graph::constant<BACKEND> (eq->get_electron_mass(0));
            auto mi = graph::constant<BACKEND> (eq->get_ion_mass(0));
            auto q = graph::constant<BACKEND> (1.602176634E-19);

            auto te = eq->get_electron_temperature(x, y, z);
            auto ti = eq->get_ion_temperature(0, x, y, z);
            auto gamma = graph::constant<BACKEND> (3.0);
            auto vs2 = (q*te + gamma*q*ti)/(mi*c*c);
            
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto wce = build_cyclotron_fequency(none*q, b_vec->length(), me, c);

//  Wave numbers.
            auto k = graph::vector(kx, ky, kz);
            auto b_hat = b_vec->unit();
            auto kperp = b_hat->cross(k)->length();
            auto kperp2 = kperp*kperp;

            auto w2 = w*w;
                        
            return wce - kperp2*vs2 - w*w;
        }
    };

//------------------------------------------------------------------------------
///  @brief Ordinary wave dispersion function.
//------------------------------------------------------------------------------
    template<class BACKEND>
    class ordinary_wave final : public dispersion_function<BACKEND> {
    public:
//------------------------------------------------------------------------------
///  @brief Disperison relation for the O mode.
///
///  D = 1 - ⍵pe^2/⍵^2 - c^2/⍵^2*(kx^2 + ky^2 + kz^2)                          (1)
///
///  ⍵pe is the plasma frequency.
///
///  @param[in] w  Omega variable.
///  @param[in] kx Kx variable.
///  @param[in] ky Ky variable.
///  @param[in] kz Kz variable.
///  @param[in] x  x variable.
///  @param[in] y  y variable.
///  @param[in] z  z variable.
///  @param[in] t  Current time.
///  @param[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> D(graph::shared_leaf<BACKEND> w,
                                              graph::shared_leaf<BACKEND> kx,
                                              graph::shared_leaf<BACKEND> ky,
                                              graph::shared_leaf<BACKEND> kz,
                                              graph::shared_leaf<BACKEND> x,
                                              graph::shared_leaf<BACKEND> y,
                                              graph::shared_leaf<BACKEND> z,
                                              graph::shared_leaf<BACKEND> t,
                                              equilibrium::unique_equilibrium<BACKEND> &eq) final {
//  Constants
            auto epsion0 = graph::constant<BACKEND> (8.8541878138E-12);
            auto mu0 = graph::constant<BACKEND> (M_PI*4.0E-7);
            auto c = graph::constant<BACKEND> (1.0)/graph::sqrt(epsion0*mu0);
            auto one = graph::constant<BACKEND> (1);
            auto none = graph::constant<BACKEND> (-1);
                        
//  Equilibrium quantities.
            auto me = graph::constant<BACKEND> (9.1093837015E-31);
            auto q = graph::constant<BACKEND> (1.602176634E-19);

            auto ne = eq->get_electron_density(x, y, z);
            auto wpe2 = build_plasma_fequency(ne, q, me, c, epsion0);

//  Wave numbers.
            auto n = graph::vector(kx/w, ky/w, kz/w);
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto b_hat = b_vec->unit();
            auto nperp = b_hat->cross(n)->length();
            auto nperp2 = nperp*nperp;

            auto w2 = w*w;
                        
            return one - wpe2/w2 - nperp2;
        }
    };

//------------------------------------------------------------------------------
///  @brief Extra ordinary wave dispersion function.
//------------------------------------------------------------------------------
    template<class BACKEND>
    class extra_ordinary_wave final : public dispersion_function<BACKEND> {
    public:
//------------------------------------------------------------------------------
///  @brief Disperison relation for the X-Mode.
///
///  D = 1 - ⍵pe^2/⍵^2(⍵^2 - ⍵pe^2)/(⍵^2 - ⍵h^2)
///    - c^2/⍵^2*(kx^2 + ky^2 + kz^2)                                          (1)
///
///  Where ⍵h is the upper hybrid frequency and defined by
///
///  ⍵h = ⍵pe^2 + ⍵ce^2
///
///  ⍵pe is the plasma frequency while ⍵ce is the cyclotron frequency.
///
///  @param[in] w  Omega variable.
///  @param[in] kx Kx variable.
///  @param[in] ky Ky variable.
///  @param[in] kz Kz variable.
///  @param[in] x  x variable.
///  @param[in] y  y variable.
///  @param[in] z  z variable.
///  @param[in] t  Current time.
///  @param[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> D(graph::shared_leaf<BACKEND> w,
                                              graph::shared_leaf<BACKEND> kx,
                                              graph::shared_leaf<BACKEND> ky,
                                              graph::shared_leaf<BACKEND> kz,
                                              graph::shared_leaf<BACKEND> x,
                                              graph::shared_leaf<BACKEND> y,
                                              graph::shared_leaf<BACKEND> z,
                                              graph::shared_leaf<BACKEND> t,
                                              equilibrium::unique_equilibrium<BACKEND> &eq) final {
//  Constants
            auto epsion0 = graph::constant<BACKEND> (8.8541878138E-12);
            auto mu0 = graph::constant<BACKEND> (M_PI*4.0E-7);
            auto c = graph::constant<BACKEND> (1.0)/graph::sqrt(epsion0*mu0);
            auto one = graph::constant<BACKEND> (1);
            auto none = graph::constant<BACKEND> (-1);
            
//  Equilibrium quantities.
            auto me = graph::constant<BACKEND> (9.1093837015E-31);
            auto q = graph::constant<BACKEND> (1.602176634E-19);

            auto ne = eq->get_electron_density(x, y, z);
            auto wpe2 = build_plasma_fequency(ne, q, me, c, epsion0);
            
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto b_len = b_vec->length();
            auto wec = build_cyclotron_fequency(none*q, b_len, me, c);
            
//  Wave numbers.
            auto n = graph::vector(kx/w, ky/w, kz/w);
            auto b_hat = b_vec->unit();
            auto nperp = b_hat->cross(n)->length();
            auto nperp2 = nperp*nperp;
        
            auto wh = wpe2 + wec*wec;
            
            auto w2 = w*w;
            
            return one - wpe2/(w2)*(w2 - wpe2)/(w2 - wh) - nperp;
        }
    };

//------------------------------------------------------------------------------
///  @brief Cold Plasma Disperison function.
//------------------------------------------------------------------------------
    template<class BACKEND>
    class cold_plasma final : public dispersion_function<BACKEND> {
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
///  @param[in] w  Omega variable.
///  @param[in] kx Kx variable.
///  @param[in] ky Ky variable.
///  @param[in] kz Kz variable.
///  @param[in] x  x variable.
///  @param[in] y  y variable.
///  @param[in] z  z variable.
///  @param[in] t  Current time.
///  @param[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> D(graph::shared_leaf<BACKEND> w,
                                              graph::shared_leaf<BACKEND> kx,
                                              graph::shared_leaf<BACKEND> ky,
                                              graph::shared_leaf<BACKEND> kz,
                                              graph::shared_leaf<BACKEND> x,
                                              graph::shared_leaf<BACKEND> y,
                                              graph::shared_leaf<BACKEND> z,
                                              graph::shared_leaf<BACKEND> t,
                                              equilibrium::unique_equilibrium<BACKEND> &eq) final {
//  Constants
            auto epsion0 = graph::constant<BACKEND> (8.8541878138E-12);
            auto mu0 = graph::constant<BACKEND> (M_PI*4.0E-7);
            auto c = graph::constant<BACKEND> (1)/graph::sqrt(epsion0*mu0);
            auto one = graph::constant<BACKEND> (1);
            auto none = graph::constant<BACKEND> (-1);

//  Equilibrium quantities.
            auto me = graph::constant<BACKEND> (9.1093837015E-31);
            auto q = graph::constant<BACKEND> (1.602176634E-19);

//  Dielectric terms.
//  Frequencies
            auto ne = eq->get_electron_density(x, y, z);
            auto wpe2 = build_plasma_fequency(ne, q, me, c, epsion0);
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto b_len = b_vec->length();
            auto ec = build_cyclotron_fequency(none*q, b_len, me, c);

            auto w2 = w*w;
            auto denome = one - ec*ec/w2;
            auto e11 = one - (wpe2/w2)/denome;
            auto e12 = ((ec/w)*(wpe2/w2))/denome;
            auto e33 = wpe2;

            for (size_t i = 0, ie = eq->get_num_ion_species(); i < ie; i++) {
                auto mi = graph::constant<BACKEND> (eq->get_ion_mass(i));
                auto charge = graph::constant<BACKEND> (eq->get_ion_charge(i))*q;

                auto ni = eq->get_ion_density(i, x, y, z);
                auto wpi2 = build_plasma_fequency(ni, charge, mi, c, epsion0);
                auto ic = build_cyclotron_fequency(charge, b_len, mi, c);

                auto denomi = one - ic*ic/w2;
                e11 = e11 - (wpi2/w2)/denomi;
                e12 = e12 + ((ic/w)*(wpi2/w2))/denomi;
                e33 = e33 + wpi2;
            }

            e12 = none*e12;
            e33 = one - e33/w2;

//  Wave numbers.
            auto n = graph::vector(kx/w, ky/w, kz/w);
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
}

#endif /* dispersion_h */
