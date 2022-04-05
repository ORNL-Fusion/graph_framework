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
///  @param[in] D  Dispersion function.
///  @param[in] w  Wave frequency.
///  @param[in] kx Wave number in x.
///  @param[in] ky Wave number in y.
///  @param[in] kz Wave number in z.
///  @param[in] x  Position in x.
///  @param[in] y  Position in y.
///  @param[in] z  Position in z.
//------------------------------------------------------------------------------
        dispersion_interface(std::shared_ptr<graph::leaf_node> D,
                             std::shared_ptr<graph::leaf_node> w,
                             std::shared_ptr<graph::leaf_node> kx,
                             std::shared_ptr<graph::leaf_node> ky,
                             std::shared_ptr<graph::leaf_node> kz,
                             std::shared_ptr<graph::leaf_node> x,
                             std::shared_ptr<graph::leaf_node> y,
                             std::shared_ptr<graph::leaf_node> z) :
        D(D) {
            auto dDdw = D->df(w)->reduce();
            auto dDdkx = D->df(kx)->reduce();
            auto dDdky = D->df(ky)->reduce();
            auto dDdkz = D->df(kz)->reduce();
            auto dDdx = D->df(x)->reduce();
            auto dDdy = D->df(y)->reduce();
            auto dDdz = D->df(z)->reduce();

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
            auto x_next = x - loss/loss->df(x);

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
        virtual std::vector<double> get_dsdt() final {
            return this->dsdt->evaluate();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for x update.
///
///  @return dx/dt
//------------------------------------------------------------------------------
        virtual std::vector<double> get_dxdt() final {
            return this->dxdt->evaluate();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for y update.
///
///  @return dy/dt
//------------------------------------------------------------------------------
        virtual std::vector<double> get_dydt() final {
            return this->dydt->evaluate();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dz/dt
//------------------------------------------------------------------------------
        virtual std::vector<double> get_dzdt() final {
            return this->dzdt->evaluate();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dkx/dt
//------------------------------------------------------------------------------
        virtual std::vector<double> get_dkxdt() final {
            return this->dkxdt->evaluate();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dky/dt
//------------------------------------------------------------------------------
        virtual std::vector<double> get_dkydt() final {
            return this->dkydt->evaluate();
        }

//------------------------------------------------------------------------------
///  @brief Provide right hand side for z update.
///
///  @return dkz/dt
//------------------------------------------------------------------------------
        virtual std::vector<double> get_dkzdt() final {
            return this->dkzdt->evaluate();
        }
    };

//******************************************************************************
//  Dispersion interface.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Simple dispersion function.
//------------------------------------------------------------------------------
    class simple final : public dispersion_interface {
    private:
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
        static std::shared_ptr<graph::leaf_node> D(std::shared_ptr<graph::leaf_node> w,
                                                   std::shared_ptr<graph::leaf_node> kx,
                                                   std::shared_ptr<graph::leaf_node> ky,
                                                   std::shared_ptr<graph::leaf_node> kz,
                                                   std::shared_ptr<graph::leaf_node> x,
                                                   std::shared_ptr<graph::leaf_node> y,
                                                   std::shared_ptr<graph::leaf_node> z) {
            auto c = graph::constant(1);

            auto npar2 = kz*kz*c*c/(w*w);
            auto nperp2 = (kx*kx + ky*ky)*c*c/(w*w);
            return npar2 + nperp2 - c;
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a simple dispersion function.
///
///  @param[in] w  Omega variable.
///  @param[in] kx Kx variable.
///  @param[in] ky Ky variable.
///  @param[in] kz Kz variable.
///  @param[in] x  x variable.
///  @param[in] y  y variable.
///  @param[in] z  z variable.
//------------------------------------------------------------------------------
        simple(std::shared_ptr<graph::leaf_node> w,
               std::shared_ptr<graph::leaf_node> kx,
               std::shared_ptr<graph::leaf_node> ky,
               std::shared_ptr<graph::leaf_node> kz,
               std::shared_ptr<graph::leaf_node> x,
               std::shared_ptr<graph::leaf_node> y,
               std::shared_ptr<graph::leaf_node> z) :
        dispersion_interface(D(w, kx, ky, kz, x, y, z)->reduce(),
                             w, kx, ky, kz, x, y, z) {}
    };
}

#endif /* dispersion_h */
