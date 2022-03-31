//------------------------------------------------------------------------------
///  @file dispersion.hpp
///  @brief Base class for a dispersion relation.
///
///  Defines a dispersion function.
//------------------------------------------------------------------------------

#ifndef dispersion_h
#define dispersion_h

#include "arithmetic.hpp"

namespace dispersion {
//******************************************************************************
//  Dispersion interface.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class interface to build dispersion relation functions.
//------------------------------------------------------------------------------
    class dispersion_interface {
    protected:
///  Derivative with respect to omega.
        std::shared_ptr<graph::leaf_node> dDdw;
///  Derivative with respect to kx.
        std::shared_ptr<graph::leaf_node> dDdkx;
///  Derivative with respect to ky.
        std::shared_ptr<graph::leaf_node> dDdky;
///  Derivative with respect to kz.
        std::shared_ptr<graph::leaf_node> dDdkz;
///  Derivative with respect to kx.
        std::shared_ptr<graph::leaf_node> dDdx;
///  Derivative with respect to ky.
        std::shared_ptr<graph::leaf_node> dDdy;
///  Derivative with respect to kz.
        std::shared_ptr<graph::leaf_node> dDdz;

    public:
        dispersion_interface(std::shared_ptr<graph::leaf_node> dw,
                             std::shared_ptr<graph::leaf_node> dkx,
                             std::shared_ptr<graph::leaf_node> dky,
                             std::shared_ptr<graph::leaf_node> dkz,
                             std::shared_ptr<graph::leaf_node> dx,
                             std::shared_ptr<graph::leaf_node> dy,
                             std::shared_ptr<graph::leaf_node> dz) :
        dDdw(dw), dDdkx(dkx), dDdky(dky), dDdkz(dkz),
        dDdx(dx), dDdy(dy), dDdz(dz) {}

//------------------------------------------------------------------------------
///  @brief Evaluate derivative with respect to omega.
///
///  @return dD/dw
//------------------------------------------------------------------------------
        virtual std::vector<double> dw() final {
            return dDdw->evaluate();
        }

//------------------------------------------------------------------------------
///  @brief Evaluate derivative with respect to omega.
///
///  @return dD/dkx
//------------------------------------------------------------------------------
        virtual std::vector<double> dkx() final {
            return dDdkx->evaluate();
        }

//------------------------------------------------------------------------------
///  @brief Evaluate derivative with respect to omega.
///
///  @return dD/dky
//------------------------------------------------------------------------------
        virtual std::vector<double> dky() final {
            return dDdky->evaluate();
        }

//------------------------------------------------------------------------------
///  @brief Evaluate derivative with respect to omega.
///
///  @return dD/dkz
//------------------------------------------------------------------------------
        virtual std::vector<double> dkz() final {
            return dDdkz->evaluate();
        }

//------------------------------------------------------------------------------
///  @brief Evaluate derivative with respect to omega.
///
///  @return dD/dx
//------------------------------------------------------------------------------
        virtual std::vector<double> dx() final {
            return dDdx->evaluate();
        }

//------------------------------------------------------------------------------
///  @brief Evaluate derivative with respect to omega.
///
///  @return dD/dy
//------------------------------------------------------------------------------
        virtual std::vector<double> dy() final {
            return dDdy->evaluate();
        }

//------------------------------------------------------------------------------
///  @brief Evaluate derivative with respect to omega.
///
///  @return dD/dz
//------------------------------------------------------------------------------
        virtual std::vector<double> dz() final {
            return dDdz->evaluate();
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
