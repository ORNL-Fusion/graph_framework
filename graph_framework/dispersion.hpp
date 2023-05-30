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
//  Common physics expressions.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Build plasma fequency expression.
//------------------------------------------------------------------------------
    template<typename T>
    static graph::shared_leaf<T> build_plasma_fequency(graph::shared_leaf<T> n,
                                                       graph::shared_leaf<T> q,
                                                       graph::shared_leaf<T> m,
                                                       graph::shared_leaf<T> c,
                                                       graph::shared_leaf<T> epsion0) {
        return n*q*q/(epsion0*m*c*c);
    }

//------------------------------------------------------------------------------
///  @brief Build cyclotron fequency expression.
//------------------------------------------------------------------------------
    template<typename T>
    static graph::shared_leaf<T> build_cyclotron_fequency(graph::shared_leaf<T> q,
                                                          graph::shared_leaf<T> b,
                                                          graph::shared_leaf<T> m,
                                                          graph::shared_leaf<T> c) {
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
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> D;

///  Derivative with respect to kx.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> dxdt;
///  Derivative with respect to ky.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> dydt;
///  Derivative with respect to kz.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> dzdt;
///  Derivative with respect to kx.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> dkxdt;
///  Derivative with respect to ky.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> dkydt;
///  Derivative with respect to kz.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> dkzdt;
///  Derivative with respect to omega.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> dsdt;

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
        dispersion_interface(graph::shared_leaf<typename DISPERSION_FUNCTION::base> w,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::base> kx,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::base> ky,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::base> kz,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::base> x,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::base> y,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::base> z,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::base> t,
                             equilibrium::shared<typename DISPERSION_FUNCTION::base> &eq) :
        D(DISPERSION_FUNCTION().D(w, kx, ky, kz, x, y, z, t, eq)) {
            auto dDdw = this->D->df(w)->reduce();
            auto dDdkx = this->D->df(kx)->reduce();
            auto dDdky = this->D->df(ky)->reduce();
            auto dDdkz = this->D->df(kz)->reduce();
            auto dDdx = this->D->df(x)->reduce();
            auto dDdy = this->D->df(y)->reduce();
            auto dDdz = this->D->df(z)->reduce();

            auto neg_one = graph::none<typename DISPERSION_FUNCTION::base> ();
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
///  @params[in,out] x              The unknown to solver for.
///  @params[in]     inputs         Inputs for jit compile.
///  @params[in]     tolarance      Tolarance to solve the dispersion function
///                                 to.
///  @params[in]     max_iterations Maximum number of iterations before giving
///                                 up.
///  @returns The residule graph.
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::base>
        solve(graph::shared_leaf<typename DISPERSION_FUNCTION::base> x,
              graph::input_nodes<typename DISPERSION_FUNCTION::base> inputs,
              const typename DISPERSION_FUNCTION::base tolarance = 1.0E-30,
              const size_t max_iterations = 1000) {
            auto loss = D*D;
            auto x_var = graph::variable_cast(x);

            workflow::manager<typename DISPERSION_FUNCTION::base> work;

            solver::newton(work, {x}, inputs, loss, tolarance, max_iterations);

            work.compile();
            work.run();

            work.copy_to_host(x, x_var->data());

            return loss;
        }

//------------------------------------------------------------------------------
///  @brief Get the disperison function.
///
///  @return dx/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::base>
        get_d() {
            return this->D->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for s update.
///
///  @return dx/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::base>
        get_dsdt() {
            return this->dsdt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for x update.
///
///  @return dx/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::base>
        get_dxdt() {
            return this->dxdt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for y update.
///
///  @return dy/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::base>
        get_dydt() {
            return this->dydt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dz/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::base>
        get_dzdt() {
            return this->dzdt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dkx/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::base>
        get_dkxdt() {
            return this->dkxdt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dky/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::base>
        get_dkydt() {
            return this->dkydt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dkz/dt
//------------------------------------------------------------------------------
        graph::shared_leaf<typename DISPERSION_FUNCTION::base>
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
    template<typename T>
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
        virtual graph::shared_leaf<T> D(graph::shared_leaf<T> w,
                                        graph::shared_leaf<T> kx,
                                        graph::shared_leaf<T> ky,
                                        graph::shared_leaf<T> kz,
                                        graph::shared_leaf<T> x,
                                        graph::shared_leaf<T> y,
                                        graph::shared_leaf<T> z,
                                        graph::shared_leaf<T> t,
                                        equilibrium::shared<T> &eq) = 0;

///  Type def to retrieve the backend base type.
        typedef T base;
    };

//------------------------------------------------------------------------------
///  @brief Stiff dispersion function.
//------------------------------------------------------------------------------
    template<typename T>
    class stiff final : public dispersion_function<T> {
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
        virtual graph::shared_leaf<T> D(graph::shared_leaf<T> w,
                                        graph::shared_leaf<T> kx,
                                        graph::shared_leaf<T> ky,
                                        graph::shared_leaf<T> kz,
                                        graph::shared_leaf<T> x,
                                        graph::shared_leaf<T> y,
                                        graph::shared_leaf<T> z,
                                        graph::shared_leaf<T> t,
                                        equilibrium::shared<T> &eq) {
            auto none = graph::none<T> ();
            auto c = graph::constant(static_cast<T> (1.0E3));
            return (c*(x - graph::exp(none*t)) - graph::exp(none*t))*kx + w;
        }
    };

//------------------------------------------------------------------------------
///  @brief Simple dispersion function.
//------------------------------------------------------------------------------
    template<typename T>
    class simple final : public dispersion_function<T> {
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
        virtual graph::shared_leaf<T> D(graph::shared_leaf<T> w,
                                        graph::shared_leaf<T> kx,
                                        graph::shared_leaf<T> ky,
                                        graph::shared_leaf<T> kz,
                                        graph::shared_leaf<T> x,
                                        graph::shared_leaf<T> y,
                                        graph::shared_leaf<T> z,
                                        graph::shared_leaf<T> t,
                                        equilibrium::shared<T> &eq) {
            auto c = graph::one<T> ();

            auto npar2 = kz*kz*c*c/(w*w);
            auto nperp2 = (kx*kx + ky*ky)*c*c/(w*w);
            return npar2 + nperp2 - c;
        }
    };

//------------------------------------------------------------------------------
///  @brief Physics
//------------------------------------------------------------------------------
    template<typename T>
    class physics : public dispersion_function<T> {
    protected:
//  Define some common constants.
///  Vacuum permitivity.
        graph::shared_leaf<T> epsion0 = graph::constant(static_cast<T> (8.8541878138E-12));
///  Vacuum permeability
        graph::shared_leaf<T> mu0 = graph::constant(static_cast<T> (M_PI*4.0E-7));
///  Fundamental charge.
        graph::shared_leaf<T> q = graph::constant(static_cast<T> (1.602176634E-19));
///  Electron mass.
        graph::shared_leaf<T> me = graph::constant(static_cast<T> (9.1093837015E-31));
/// Speed of light.
        graph::shared_leaf<T> c = graph::one<T> ()/graph::sqrt(epsion0*mu0);
    };

//------------------------------------------------------------------------------
///  @brief Bohm-Gross dispersion function.
//------------------------------------------------------------------------------
    template<typename T>
    class bohm_gross final : public physics<T> {
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
        virtual graph::shared_leaf<T> D(graph::shared_leaf<T> w,
                                        graph::shared_leaf<T> kx,
                                        graph::shared_leaf<T> ky,
                                        graph::shared_leaf<T> kz,
                                        graph::shared_leaf<T> x,
                                        graph::shared_leaf<T> y,
                                        graph::shared_leaf<T> z,
                                        graph::shared_leaf<T> t,
                                        equilibrium::shared<T> &eq) {

//  Equilibrium quantities.
            auto ne = eq->get_electron_density(x, y, z);
            auto wpe2 = build_plasma_fequency(ne, physics<T>::q, physics<T>::me,
                                              physics<T>::c, physics<T>::epsion0);
            auto te = eq->get_electron_temperature(x, y, z);
//  2*1.602176634E-19 to convert eV to J.
            
            auto temp = graph::two<T> ()*physics<T>::q*te;
            auto vterm2 = temp/(physics<T>::me*physics<T>::c*physics<T>::c);

//  Wave numbers should be parallel to B if there is a magnetic field. Otherwise
//  B should be zero.
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto k = graph::vector(kx, ky, kz);
            graph::shared_leaf<T> kpara2;
            if (b_vec->length()->is_match(graph::zero<T> ())) {
                kpara2 = k->dot(k);
            } else {
                auto b_hat = b_vec->unit();
                auto kpara = b_hat->dot(k);
                kpara2 = kpara*kpara;
            }
            
            return wpe2 +
                   graph::constant(static_cast<T> (3.0/2.0))*kpara2*vterm2 -
                   w*w;
        }
    };

//------------------------------------------------------------------------------
///  @brief Light Wave dispersion function.
//------------------------------------------------------------------------------
    template<typename T>
    class light_wave final : public physics<T> {
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
        virtual graph::shared_leaf<T> D(graph::shared_leaf<T> w,
                                        graph::shared_leaf<T> kx,
                                        graph::shared_leaf<T> ky,
                                        graph::shared_leaf<T> kz,
                                        graph::shared_leaf<T> x,
                                        graph::shared_leaf<T> y,
                                        graph::shared_leaf<T> z,
                                        graph::shared_leaf<T> t,
                                        equilibrium::shared<T> &eq) {

//  Equilibrium quantities.
            auto ne = eq->get_electron_density(x, y, z);
            auto wpe2 = build_plasma_fequency(ne, physics<T>::q, physics<T>::me,
                                              physics<T>::c, physics<T>::epsion0);

//  Wave numbers should be parallel to B if there is a magnetic field. Otherwise
//  B should be zero.
            assert(eq->get_magnetic_field(x, y, z)->length()->is_match(graph::zero<T> ()) &&
                   "Expected equilibrium with no magnetic field.");

            auto k = graph::vector(kx, ky, kz);
            auto k2 = k->dot(k);
            
            return wpe2 + k2 - w*w;
        }
    };

//------------------------------------------------------------------------------
///  @brief Ion wave dispersion function.
//------------------------------------------------------------------------------
    template<typename T>
    class acoustic_wave final : public physics<T> {
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
        virtual graph::shared_leaf<T> D(graph::shared_leaf<T> w,
                                        graph::shared_leaf<T> kx,
                                        graph::shared_leaf<T> ky,
                                        graph::shared_leaf<T> kz,
                                        graph::shared_leaf<T> x,
                                        graph::shared_leaf<T> y,
                                        graph::shared_leaf<T> z,
                                        graph::shared_leaf<T> t,
                                        equilibrium::shared<T> &eq) {

//  Equilibrium quantities.
            auto mi = graph::constant(eq->get_ion_mass(0));
            auto te = eq->get_electron_temperature(x, y, z);
            auto ti = eq->get_ion_temperature(0, x, y, z);
            auto gamma = graph::constant(static_cast<T> (3.0));
            auto vs2 = (physics<T>::q*te + gamma*physics<T>::q*ti)
                     / (mi*physics<T>::c*physics<T>::c);

//  Wave numbers should be parallel to B if there is a magnetic field. Otherwise
//  B should be zero.
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto k = graph::vector(kx, ky, kz);
            graph::shared_leaf<T> kpara2;
            if (b_vec->length()->is_match(graph::zero<T> ())) {
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
    template<typename T>
    class guassian_well final : public dispersion_function<T> {
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
        virtual graph::shared_leaf<T> D(graph::shared_leaf<T> w,
                                        graph::shared_leaf<T> kx,
                                        graph::shared_leaf<T> ky,
                                        graph::shared_leaf<T> kz,
                                        graph::shared_leaf<T> x,
                                        graph::shared_leaf<T> y,
                                        graph::shared_leaf<T> z,
                                        graph::shared_leaf<T> t,
                                        equilibrium::shared<T> &eq) {
            auto c = graph::one<T> ();
            auto well = c - graph::constant(static_cast<T> (0.5))*exp(graph::constant(static_cast<T> (-1.0))*(x*x + y*y)/graph::constant(static_cast<T> (0.1)));
            auto npar2 = kz*kz*c*c/(w*w);
            auto nperp2 = (kx*kx + ky*ky)*c*c/(w*w);
            return npar2 + nperp2 - well;
        }
    };

//------------------------------------------------------------------------------
///  @brief Electrostatic ion cyclotron wave dispersion function.
//------------------------------------------------------------------------------
    template<typename T>
    class ion_cyclotron final : public physics<T> {
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
        virtual graph::shared_leaf<T> D(graph::shared_leaf<T> w,
                                        graph::shared_leaf<T> kx,
                                        graph::shared_leaf<T> ky,
                                        graph::shared_leaf<T> kz,
                                        graph::shared_leaf<T> x,
                                        graph::shared_leaf<T> y,
                                        graph::shared_leaf<T> z,
                                        graph::shared_leaf<T> t,
                                        equilibrium::shared<T> &eq) {
//  Constants
            auto none = graph::constant(static_cast<T> (-1.0));
                        
//  Equilibrium quantities.
            auto mi = graph::constant(eq->get_ion_mass(0));

            auto te = eq->get_electron_temperature(x, y, z);
            auto ti = eq->get_ion_temperature(0, x, y, z);
            auto gamma = graph::constant(static_cast<T> (3.0));
            auto vs2 = (physics<T>::q*te + gamma*physics<T>::q*ti)
                     / (mi*physics<T>::c*physics<T>::c);
            
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto wce = build_cyclotron_fequency(none*physics<T>::q,
                                                b_vec->length(), physics<T>::me,
                                                physics<T>::c);

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
    template<typename T>
    class ordinary_wave final : public physics<T> {
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
        virtual graph::shared_leaf<T> D(graph::shared_leaf<T> w,
                                        graph::shared_leaf<T> kx,
                                        graph::shared_leaf<T> ky,
                                        graph::shared_leaf<T> kz,
                                        graph::shared_leaf<T> x,
                                        graph::shared_leaf<T> y,
                                        graph::shared_leaf<T> z,
                                        graph::shared_leaf<T> t,
                                        equilibrium::shared<T> &eq) {
//  Constants
            auto one = graph::one<T> ();

//  Equilibrium quantities.
            auto ne = eq->get_electron_density(x, y, z);
            auto wpe2 = build_plasma_fequency(ne, physics<T>::q, physics<T>::me,
                                              physics<T>::c, physics<T>::epsion0);

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
    template<typename T>
    class extra_ordinary_wave final : public physics<T> {
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
        virtual graph::shared_leaf<T> D(graph::shared_leaf<T> w,
                                        graph::shared_leaf<T> kx,
                                        graph::shared_leaf<T> ky,
                                        graph::shared_leaf<T> kz,
                                        graph::shared_leaf<T> x,
                                        graph::shared_leaf<T> y,
                                        graph::shared_leaf<T> z,
                                        graph::shared_leaf<T> t,
                                        equilibrium::shared<T> &eq) {
//  Constants
            auto one = graph::one<T> ();
            auto none = graph::constant(static_cast<T> (-1.0));
            
//  Equilibrium quantities.
            auto ne = eq->get_electron_density(x, y, z);
            auto wpe2 = build_plasma_fequency(ne, physics<T>::q, physics<T>::me,
                                              physics<T>::c,
                                              physics<T>::epsion0);
            
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto b_len = b_vec->length();
            auto wec = build_cyclotron_fequency(none*physics<T>::q, b_len,
                                                physics<T>::me, physics<T>::c);
            
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
    template<typename T>
    class cold_plasma final : public physics<T> {
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
        virtual graph::shared_leaf<T> D(graph::shared_leaf<T> w,
                                        graph::shared_leaf<T> kx,
                                        graph::shared_leaf<T> ky,
                                        graph::shared_leaf<T> kz,
                                        graph::shared_leaf<T> x,
                                        graph::shared_leaf<T> y,
                                        graph::shared_leaf<T> z,
                                        graph::shared_leaf<T> t,
                                        equilibrium::shared<T> &eq) {
//  Constants
            auto one = graph::one<T> ();
            auto none = graph::none<T> ();

//  Dielectric terms.
//  Frequencies
            auto ne = eq->get_electron_density(x, y, z);
            auto wpe2 = build_plasma_fequency(ne, physics<T>::q, physics<T>::me,
                                              physics<T>::c,
                                              physics<T>::epsion0);
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto b_len = b_vec->length();
            auto ec = build_cyclotron_fequency(none*physics<T>::q, b_len,
                                               physics<T>::me, physics<T>::c);

            auto w2 = w*w;
            auto denome = one - ec*ec/w2;
            auto e11 = one - (wpe2/w2)/denome;
            auto e12 = ((ec/w)*(wpe2/w2))/denome;
            auto e33 = wpe2;

            for (size_t i = 0, ie = eq->get_num_ion_species(); i < ie; i++) {
                auto mi = graph::constant(eq->get_ion_mass(i));
                auto charge = graph::constant(static_cast<T> (eq->get_ion_charge(i)))
                            * physics<T>::q;

                auto ni = eq->get_ion_density(i, x, y, z);
                auto wpi2 = build_plasma_fequency(ni, charge, mi, physics<T>::c,
                                                  physics<T>::epsion0);
                auto ic = build_cyclotron_fequency(charge, b_len, mi,
                                                   physics<T>::c);

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

//------------------------------------------------------------------------------
///  @brief Hot Plasma Disperison function.
//------------------------------------------------------------------------------
    template<typename T, class Z>
    class hot_plasma final : public physics<T> {
    private:
///  Z function.
        Z z;

    public:
//------------------------------------------------------------------------------
///  @brief Cold Plasma Disperison function.
///
///  D = iσΓ0 + Γ1 + Γ2(1 + ζZ(ζ)) + Γ3ζZ(ζ)(1 + ζZ(ζ)) + Γ4iσF + Γ5F        (1)
///
///  Γ0 = n^2n⟂^2 - (1 - P)(n^2 + n||^2) - 2(1 - 2q)n⟂^2 + 2(1 - P)(1 - 2q)  (2)
///
///  Γ1 = (1 - q)n^2n⟂^2 + (1 - P)n^2n||^2 - (1 - q)(1 - P)(n^2 + n||^2)
///     - (1 - 2q)n⟂^2 + (1 - 2q)(1 - P)                                     (3)
///
///  Γ2 = P⍵/Ωe(n^2n⟂^2 - (1 - 2q)n⟂^2)
///     + P^2⍵^2/(4Ωe^2)(n⟂^2/n||^2(n^2 + n||^2) - 2(1 - 2q)n⟂^2/n||^2)      (4)
///
///  Γ3 = P⍵^2/(4Ωe^2)n⟂^2/n||^2((n^2 + n||^2) - 2(1 - 2q))                  (5)
///
///  Γ4 = P(-(n^2 + n||^2) + 2(1 - 2q))                                      (6)
///
///  Γ5 = P(n^2n||^2 - (1 - q)(n^2 + n||^2) + (1 - 2q))                      (7)
///
///  iσ = ⍵pe^2/(2⍵)1/(k||ve)Z(ζ)                                            (8)
///
///  ζ = (⍵ - Ωe)/(k||ve)                                                    (9)
///
///  F = n⟂^2/(2n||^2)⍵^2/Ωe^2ve/cζ(1 + ζZ(ζ))                              (10)
///
///  P = ⍵pe^2/⍵^2                                                          (11)
///
///  q = ⍵pe^2/(2⍵(⍵ + Ωe))                                                 (12)
///
///  ve = Sqrt(2*ne*te/me)                                                  (13)
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
        virtual graph::shared_leaf<T> D(graph::shared_leaf<T> w,
                                        graph::shared_leaf<T> kx,
                                        graph::shared_leaf<T> ky,
                                        graph::shared_leaf<T> kz,
                                        graph::shared_leaf<T> x,
                                        graph::shared_leaf<T> y,
                                        graph::shared_leaf<T> z,
                                        graph::shared_leaf<T> t,
                                        equilibrium::shared<T> &eq) {
//  Constants
            auto one = graph::one<T> ();
            auto none = graph::none<T> ();
            auto two = graph::two<T> ();
            auto four = two*two;

//  Setup plasma parameters.
            auto b_vec = eq->get_magnetic_field(x, y, z);
            auto b_hat = b_vec->unit();
            auto b_len = b_vec->length();
            auto ne = eq->get_electron_density(x, y, z);
            auto te = eq->get_electron_temperature(x, y, z);
            
            auto ve = graph::sqrt(two*ne*te/physics<T>::me);

//  Setup characteristic frequencies.
            auto ec = build_cyclotron_fequency(none*physics<T>::q, b_len,
                                               physics<T>::me, physics<T>::c);
            auto wpe2 = build_plasma_fequency(ne, physics<T>::q, physics<T>::me,
                                              physics<T>::c,
                                              physics<T>::epsion0);

//  Disperison quantities.
            auto q = wpe2/(two*w*(w + ec));
            auto P = wpe2/(w*w);

            auto kvec = graph::vector(kx, ky, kz);
            auto n = graph::vector(kx/w, ky/w, kz/w);
            auto n2 = n->dot(n);
            auto kpara = b_hat->dot(kvec);
            auto npara = b_hat->dot(n);
            auto npara2 = npara*npara;
            auto nperp = b_hat->cross(n)->length();
            auto nperp2 = nperp*nperp;
            
            auto zeta = (w - ec)/(kpara*ve);
            auto Z_func = this->z.Z(zeta);
            auto F = nperp2/(two*npara)*w*w/(ec*ec)*ve/physics<T>::c*zeta*(one + zeta*Z_func);
            auto isigma = wpe2/(two*w)*Z_func/(kpara*ve);

            auto q_func = one - two*q;
            auto n_func = n2 + npara2;
            auto n2nperp2 = n2*nperp2;
            auto p_func = one - P;

            auto gamma5 = P*(n2*npara2 - (one - q)*n_func + q_func);
            auto gamma4 = P*(two*q_func - n_func);
            auto gamma3 = P*w*w/(four*ec*ec)*nperp2/npara2*(n_func - two*q_func);
            auto gamma2 = P*w/ec*(n2nperp2 - q_func*nperp2)
                        + P*P*w*w/(four*ec*ec)*(n_func - two*q_func)*nperp2/npara2;
            auto gamma1 = (one - q)*n2nperp2 + p_func*n2*npara2
                        - (one - q)*p_func*n_func - q_func*nperp2 + q_func*p_func;
            auto gamma0 = n2nperp2 - p_func*n_func - two*q_func*nperp2 + two*p_func*q_func;

            auto zeta_func = one + zeta*Z_func;

            return isigma*gamma0 + gamma1 + gamma2*zeta_func +
                   gamma2*zeta*Z_func*zeta_func + gamma4*isigma*F + gamma5*F;
        }
    };

//******************************************************************************
//  Z Function interface.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class interface to build dispersion relation functions.
//------------------------------------------------------------------------------
    template<typename T>
    class z_function {
    public:
//------------------------------------------------------------------------------
///  @brief Method to build the Z function.
///
///  @params[in] zeta The zeta argument.
///  @returns The constructed Z function.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> Z(graph::shared_leaf<T> zeta) = 0;
    };

//------------------------------------------------------------------------------
///  @brief Class interface to build dispersion relation functions.
//------------------------------------------------------------------------------
    template<typename T>
    class z_power_series final : public z_function<T> {

    static_assert(jit::is_complex<T> (), "Only supported for complex base types.");

    public:
//------------------------------------------------------------------------------
///  @brief Method to build the Z function.
///
///  @params[in] zeta The zeta argument.
///  @returns The constructed Z function.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> Z(graph::shared_leaf<T> zeta) {
            auto one = graph::one<T> ();
            auto two = graph::two<T> ();
            auto four = two*two;
            auto eight = four*two;
            auto zeta2 = zeta*zeta;
            auto zeta4 = zeta2*zeta2;
            auto zeta6 = zeta4*zeta2;
            T i(0.0, 1.0);
            return graph::constant(i)*graph::sqrt(graph::pi<T> ())/graph::exp(zeta2) -
                   two*(one - two/graph::constant(static_cast<T> (3.0))*zeta2
                            + four/graph::constant(static_cast<T> (15.0))*zeta4
                            - eight/graph::constant(static_cast<T> (105.0))*zeta6)*zeta;
        }
    };
}

#endif /* dispersion_h */
