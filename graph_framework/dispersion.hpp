//------------------------------------------------------------------------------
///  @file dispersion.hpp
///  @brief Base class for a dispersion relation.
///
///  Defines a dispersion function.
//------------------------------------------------------------------------------

#ifndef dispersion_h
#define dispersion_h

#include <algorithm>

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
        std::shared_ptr<graph::leaf_node> D;

///  Derivative with respect to kx.
        std::shared_ptr<graph::leaf_node> dxdt;
///  Derivative with respect to ky.
        std::shared_ptr<graph::leaf_node> dydt;
///  Derivative with respect to kz.
        std::shared_ptr<graph::leaf_node> dzdt;
///  Derivative with respect to kx.
        std::shared_ptr<graph::leaf_node> dkxdt;
///  Derivative with respect to ky.
        std::shared_ptr<graph::leaf_node> dkydt;
///  Derivative with respect to kz.
        std::shared_ptr<graph::leaf_node> dkzdt;
///  Derivative with respect to omega.
        std::shared_ptr<graph::leaf_node> dsdt;

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
        dispersion_interface(std::shared_ptr<graph::leaf_node> w,
                             std::shared_ptr<graph::leaf_node> kx,
                             std::shared_ptr<graph::leaf_node> ky,
                             std::shared_ptr<graph::leaf_node> kz,
                             std::shared_ptr<graph::leaf_node> x,
                             std::shared_ptr<graph::leaf_node> y,
                             std::shared_ptr<graph::leaf_node> z) :
        D(DISPERSION_FUNCTION().D(w, kx, ky, kz, x, y, z)) {
            auto dDdw = this->D->df(w)->reduce();
            auto dDdkx = this->D->df(kx)->reduce();
            auto dDdky = this->D->df(ky)->reduce();
            auto dDdkz = this->D->df(kz)->reduce();
            auto dDdx = this->D->df(x)->reduce();
            auto dDdy = this->D->df(y)->reduce();
            auto dDdz = this->D->df(z)->reduce();

            auto neg_one = graph::constant(-1);
            dxdt = neg_one*dDdkx/dDdw;
            dydt = neg_one*dDdky/dDdw;
            dzdt = neg_one*dDdkz/dDdw;
            dkxdt = dDdx/dDdw;
            dkydt = dDdx/dDdw;
            dkzdt = dDdx/dDdw;
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
        virtual void solve(std::shared_ptr<graph::leaf_node> x,
                           const double tolarance=1.0E-30,
                           const size_t max_iterations = 1000) {
            auto loss = D*D;
            auto x_next = x - loss/(loss->df(x));

            std::vector<double> residule_vector = loss->evaluate();
            double max_residule = *std::max_element(residule_vector.cbegin(),
                                                    residule_vector.cend());
            size_t iterations = 0;

            while (max_residule > tolarance && iterations++ < max_iterations) {
                x->set(x_next->evaluate());

                residule_vector = loss->evaluate();
                max_residule = *std::max_element(residule_vector.cbegin(),
                                                 residule_vector.cend());
            }

            if (iterations > max_iterations) {
                std::cerr << "Newton solve failed to converge with in given iterations."
                          << std::endl;
            }
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for s update.
///
///  @return dx/dt
//------------------------------------------------------------------------------
        virtual std::shared_ptr<graph::leaf_node> get_dsdt() final {
            return this->dsdt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for x update.
///
///  @return dx/dt
//------------------------------------------------------------------------------
        virtual std::shared_ptr<graph::leaf_node> get_dxdt() final {
            return this->dxdt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for y update.
///
///  @return dy/dt
//------------------------------------------------------------------------------
        virtual std::shared_ptr<graph::leaf_node> get_dydt() final {
            return this->dydt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dz/dt
//------------------------------------------------------------------------------
        virtual std::shared_ptr<graph::leaf_node> get_dzdt() final {
            return this->dzdt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dkx/dt
//------------------------------------------------------------------------------
        virtual std::shared_ptr<graph::leaf_node> get_dkxdt() final {
            return this->dkxdt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dky/dt
//------------------------------------------------------------------------------
        virtual std::shared_ptr<graph::leaf_node> get_dkydt() final {
            return this->dkydt->reduce();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dkz/dt
//------------------------------------------------------------------------------
        virtual std::shared_ptr<graph::leaf_node> get_dkzdt() final {
            return this->dkzdt->reduce();
        }
    };

//******************************************************************************
//  Dispersion function.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Interface for dispersion functions.
//------------------------------------------------------------------------------
    class dispersion_function {
    public:
        virtual std::shared_ptr<graph::leaf_node> D(std::shared_ptr<graph::leaf_node> w,
                                                    std::shared_ptr<graph::leaf_node> kx,
                                                    std::shared_ptr<graph::leaf_node> ky,
                                                    std::shared_ptr<graph::leaf_node> kz,
                                                    std::shared_ptr<graph::leaf_node> x,
                                                    std::shared_ptr<graph::leaf_node> y,
                                                    std::shared_ptr<graph::leaf_node> z) = 0;
    };

//------------------------------------------------------------------------------
///  @brief Simple dispersion function.
//------------------------------------------------------------------------------
    class simple final : public dispersion_function {
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
        virtual std::shared_ptr<graph::leaf_node> D(std::shared_ptr<graph::leaf_node> w,
                                                    std::shared_ptr<graph::leaf_node> kx,
                                                    std::shared_ptr<graph::leaf_node> ky,
                                                    std::shared_ptr<graph::leaf_node> kz,
                                                    std::shared_ptr<graph::leaf_node> x,
                                                    std::shared_ptr<graph::leaf_node> y,
                                                    std::shared_ptr<graph::leaf_node> z) final {
            auto c = graph::constant(1);

            auto npar2 = kz*kz*c*c/(w*w);
            auto nperp2 = (kx*kx + ky*ky)*c*c/(w*w);
            return npar2 + nperp2 - c;
        }
    };
}

#endif /* dispersion_h */
