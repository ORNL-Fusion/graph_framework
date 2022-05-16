//------------------------------------------------------------------------------
///  @file dispersion.hpp
///  @brief Base class for a dispersion relation.
///
///  Defines a dispersion function.
//------------------------------------------------------------------------------

#ifndef dispersion_h
#define dispersion_h

#include <iostream>

#include "vector.hpp"

namespace dispersion {
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
//------------------------------------------------------------------------------
        dispersion_interface(graph::shared_leaf<typename DISPERSION_FUNCTION::backend> w,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kx,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::backend> ky,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kz,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::backend> x,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::backend> y,
                             graph::shared_leaf<typename DISPERSION_FUNCTION::backend> z) :
        D(DISPERSION_FUNCTION().D(w, kx, ky, kz, x, y, z)) {
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
            dsdt = graph::length(dxdt, dydt, dzdt);
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
        virtual void solve(graph::shared_leaf<typename DISPERSION_FUNCTION::backend> x,
                           const double tolarance=1.0E-30,
                           const size_t max_iterations = 1000) {
            auto loss = D*D;
            auto x_next = x
                        - loss/(loss->df(x) +
                                graph::constant<typename DISPERSION_FUNCTION::backend> (tolarance));

            double max_residule = loss->evaluate().max();
            size_t iterations = 0;

            while (max_residule > tolarance && iterations++ < max_iterations) {
                x->set(x_next->evaluate());
                max_residule = loss->evaluate().max();
            }

            if (iterations > max_iterations) {
                std::cerr << "Newton solve failed to converge with in given iterations."
                          << std::endl;
            }
        }

//------------------------------------------------------------------------------
///  @brief Get the disperison function.
///
///  @return dx/dt
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<typename DISPERSION_FUNCTION::backend>
        get_d() final {
            return this->D->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for s update.
///
///  @return dx/dt
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<typename DISPERSION_FUNCTION::backend>
        get_dsdt() final {
            return this->dsdt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for x update.
///
///  @return dx/dt
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<typename DISPERSION_FUNCTION::backend>
        get_dxdt() final {
            return this->dxdt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for y update.
///
///  @return dy/dt
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<typename DISPERSION_FUNCTION::backend>
        get_dydt() final {
            return this->dydt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dz/dt
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<typename DISPERSION_FUNCTION::backend>
        get_dzdt() final {
            return this->dzdt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dkx/dt
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<typename DISPERSION_FUNCTION::backend>
        get_dkxdt() final {
            return this->dkxdt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dky/dt
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<typename DISPERSION_FUNCTION::backend>
        get_dkydt() final {
            return this->dkydt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dkz/dt
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<typename DISPERSION_FUNCTION::backend>
        get_dkzdt() final {
            return this->dkzdt->reduce();
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
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> D(graph::shared_leaf<BACKEND> w,
                                              graph::shared_leaf<BACKEND> kx,
                                              graph::shared_leaf<BACKEND> ky,
                                              graph::shared_leaf<BACKEND> kz,
                                              graph::shared_leaf<BACKEND> x,
                                              graph::shared_leaf<BACKEND> y,
                                              graph::shared_leaf<BACKEND> z) = 0;

///  Type def to retrieve the backend type.
        typedef BACKEND backend;
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
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> D(graph::shared_leaf<BACKEND> w,
                                              graph::shared_leaf<BACKEND> kx,
                                              graph::shared_leaf<BACKEND> ky,
                                              graph::shared_leaf<BACKEND> kz,
                                              graph::shared_leaf<BACKEND> x,
                                              graph::shared_leaf<BACKEND> y,
                                              graph::shared_leaf<BACKEND> z) final {
            auto c = graph::constant<BACKEND> (1);

            auto npar2 = kz*kz*c*c/(w*w);
            auto nperp2 = (kx*kx + ky*ky)*c*c/(w*w);
            return npar2 + nperp2 - c;
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
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> D(graph::shared_leaf<BACKEND> w,
                                              graph::shared_leaf<BACKEND> kx,
                                              graph::shared_leaf<BACKEND> ky,
                                              graph::shared_leaf<BACKEND> kz,
                                              graph::shared_leaf<BACKEND> x,
                                              graph::shared_leaf<BACKEND> y,
                                              graph::shared_leaf<BACKEND> z) final {
            auto c = graph::constant<BACKEND> (1);
            auto well = c - graph::constant<BACKEND>(0.5)*exp(graph::constant<BACKEND> (-1)*(x*x + y*y)/graph::constant<BACKEND> (0.1));
            auto npar2 = kz*kz*c*c/(w*w);
            auto nperp2 = (kx*kx + ky*ky)*c*c/(w*w);
            return npar2 + nperp2 - well;
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
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> D(graph::shared_leaf<BACKEND> w,
                                              graph::shared_leaf<BACKEND> kx,
                                              graph::shared_leaf<BACKEND> ky,
                                              graph::shared_leaf<BACKEND> kz,
                                              graph::shared_leaf<BACKEND> x,
                                              graph::shared_leaf<BACKEND> y,
                                              graph::shared_leaf<BACKEND> z) final {
//  Constants
            auto epsion0 = graph::constant<BACKEND> (8.8541878138E-12);
            auto mu0 = graph::constant<BACKEND> (M_PI*4.0E-7);
            auto c = graph::constant<BACKEND> (1)/graph::sqrt(epsion0*mu0);
            auto one = graph::constant<BACKEND> (1);
            auto none = graph::constant<BACKEND> (-1);

//  Equilibrium quantities.
            auto B = one;
            auto me = graph::constant<BACKEND> (9.1093837015E-31);
            auto mi = graph::constant<BACKEND> (3.34449469E-27);
            auto q = graph::constant<BACKEND> (1.602176634E-19);
            auto ne = graph::constant<BACKEND> (1.0E19)*graph::exp((x*x + y*y)/graph::constant<BACKEND> (-0.2));
            auto ni = ne;

//  Frequencies
            auto wpe2 = ne*q*q/(epsion0*me*c*c);
            auto wpi2 = ni*q*q/(epsion0*mi*c*c);
            auto ec = none*q*B/(me*c);
            auto ic = q*B/(mi*c);
            auto w2 = w*w;

//  Dielectric terms.
            auto denome = one - ec*ec/w2;
            auto denomi = one - ic*ic/w2;

            auto e11 = one - (wpe2/w2)/denome - (wpi2/w2)/denomi;
            auto e12 = none*(((ec/w)*(wpe2/w2))/denome + ((ic/w)*(wpi2/w2))/denomi);
            auto e33 = one - (wpe2 + wpi2)/w2;

//  Wave numbers.
            auto nx = kx/w;
            auto ny = ky/w;
            auto nz = kz/w;

            auto npara = nx;
            auto npara2 = npara*npara;
            auto nperp = sqrt(ny*ny + nz*nz);
            auto nperp2 = nperp*nperp;

//  Determinate matrix elements
            auto m11 = e11 - npara2;
            auto m12 = e12;
            auto m13 = npara*nperp;
            auto m22 = e11 - npara2 - nperp2;
            auto m33 = e33 - nperp2;

            return (m11*m22 - m12*m12)*m33 - m22*m13*m13;
        }
    };
}

#endif /* dispersion_h */
