//------------------------------------------------------------------------------
///  @file dispersion.hpp
///  @brief Base class for a dispersion relation.
///
///  Defines a dispersion function.
//------------------------------------------------------------------------------

#ifndef dispersion_h
#define dispersion_h

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
        dispersion_interface(std::shared_ptr<graph::leaf_node> dDdw,
                             std::shared_ptr<graph::leaf_node> dDdkx,
                             std::shared_ptr<graph::leaf_node> dDdky,
                             std::shared_ptr<graph::leaf_node> dDdkz,
                             std::shared_ptr<graph::leaf_node> dDdx,
                             std::shared_ptr<graph::leaf_node> dDdy,
                             std::shared_ptr<graph::leaf_node> dDdz) :
        dxdt(graph::constant(-1)*dDdkx/dDdw),
        dydt(graph::constant(-1)*dDdky/dDdw),
        dzdt(graph::constant(-1)*dDdkz/dDdw),
        dkxdt(dDdx/dDdw), dkydt(dDdy/dDdw), dkzdt(dDdz/dDdw),
        dsdt(graph::length(dxdt, dydt, dzdt)) {}

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
///  D = npar^2 + nperp^2 - ⍵^2/c(x)^2
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

            auto npar2 = kz*kz*c/w;
            auto nperp2 = (kx*kx + ky*ky)*c/w;
            auto D = npar2 + nperp2 - w*w/(c*c);
            return D;
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
        dispersion_interface(D(w, kx, ky, kz, x, y, z)->df(w)->reduce(),
                             D(w, kx, ky, kz, x, y, z)->df(kx)->reduce(),
                             D(w, kx, ky, kz, x, y, z)->df(ky)->reduce(),
                             D(w, kx, ky, kz, x, y, z)->df(kz)->reduce(),
                             D(w, kx, ky, kz, x, y, z)->df(x)->reduce(),
                             D(w, kx, ky, kz, x, y, z)->df(y)->reduce(),
                             D(w, kx, ky, kz, x, y, z)->df(z)->reduce()) {}
    };
}

#endif /* dispersion_h */
