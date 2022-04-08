//------------------------------------------------------------------------------
///  @file solver.hpp
///  @brief Base class for a ode solvers.
///
///  Defines a ode solver.
//------------------------------------------------------------------------------

#ifndef solver_h
#define solver_h

#include <list>

#include "dispersion.hpp"

namespace solver {
//******************************************************************************
//  Solve State
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Solve state contains the variables.
//------------------------------------------------------------------------------
    struct solve_state {
///  Current state of the wave number in the x direction.
        std::vector<double> kx;
///  Current state of the wave number in the y direction.
        std::vector<double> ky;
///  Current state of the wave number in the z direction.
        std::vector<double> kz;
///  Current state x position.
        std::vector<double> x;
///  Current state y position.
        std::vector<double> y;
///  Current state z position.
        std::vector<double> z;

//------------------------------------------------------------------------------
///  @brief Construct a new solve_state with inital conditions.
///
///  @param[in] kx0 Inital kx.
///  @param[in] ky0 Inital ky.
///  @param[in] kz0 Inital kz.
///  @param[in] x0  Inital x.
///  @param[in] y0  Inital y.
///  @param[in] z0  Inital z.
//------------------------------------------------------------------------------
        solve_state(const std::vector<double> &kx0,
                    const std::vector<double> &ky0,
                    const std::vector<double> &kz0,
                    const std::vector<double> &x0,
                    const std::vector<double> &y0,
                    const std::vector<double> &z0) :
        kx(kx0), ky(ky0), kz(kz0), x(x0), y(y0), z(z0) {}

//------------------------------------------------------------------------------
///  @brief Construct a new solve_state.
///
///  @param[in] size Number of rays to solve in this context.
//------------------------------------------------------------------------------
        solve_state(const size_t size) :
        kx(size), ky(size), kz(size),
        x(size), y(size), z(size) {}

//------------------------------------------------------------------------------
///  @brief Get the size of the state buffers.
///
///  Assumes all buffers are equal size.
//------------------------------------------------------------------------------
        size_t size() const {
            return x.size();
        }
    };

//******************************************************************************
//  Solver interface.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class interface the solver.
//------------------------------------------------------------------------------
    template<class DISPERSION_FUNCTION>
    class solver_interface {
    protected:
///  w variable.
        std::shared_ptr<graph::leaf_node> w;
///  kx variable.
        std::shared_ptr<graph::leaf_node> kx;
///  ky variable.
        std::shared_ptr<graph::leaf_node> ky;
///  kz variable.
        std::shared_ptr<graph::leaf_node> kz;
///  x variable.
        std::shared_ptr<graph::leaf_node> x;
///  y variable.
        std::shared_ptr<graph::leaf_node> y;
///  z variable.
        std::shared_ptr<graph::leaf_node> z;

///  Dispersion function interface.
       dispersion::dispersion_interface<DISPERSION_FUNCTION> D;

    public:
///  Ray solution.
        std::list<solve_state> state;

//------------------------------------------------------------------------------
///  @brief Construct a new solver_interface with inital conditions.
///
///  @param[in] w  Inital w.
///  @param[in] kx Inital kx.
///  @param[in] ky Inital ky.
///  @param[in] kz Inital kz.
///  @param[in] x  Inital x.
///  @param[in] y  Inital y.
///  @param[in] z  Inital z.
//------------------------------------------------------------------------------
        solver_interface(std::shared_ptr<graph::leaf_node> w,
                         std::shared_ptr<graph::leaf_node> kx,
                         std::shared_ptr<graph::leaf_node> ky,
                         std::shared_ptr<graph::leaf_node> kz,
                         std::shared_ptr<graph::leaf_node> x,
                         std::shared_ptr<graph::leaf_node> y,
                         std::shared_ptr<graph::leaf_node> z) :
        D(w, kx, ky, kz, x, y, z), w(w),
        kx(kx), ky(ky), kz(kz),
        x(x), y(y), z(z) {}

//------------------------------------------------------------------------------
///  @brief Method to initalize the rays.
//------------------------------------------------------------------------------
        virtual void init(std::shared_ptr<graph::leaf_node> x,
                          const double tolarance=1.0E-30,
                          const size_t max_iterations = 1000) final {
            this->D.solve(x, tolarance, max_iterations);

            this->state.push_back(solve_state(this->kx->evaluate(),
                                              this->ky->evaluate(),
                                              this->kz->evaluate(),
                                              this->x->evaluate(),
                                              this->y->evaluate(),
                                              this->z->evaluate()));
        }

//------------------------------------------------------------------------------
///  @brief Evaluate the dispersion relation residule.
//------------------------------------------------------------------------------
        std::shared_ptr<graph::leaf_node> residule() {
            return D.get_d()*D.get_d();
        }

//------------------------------------------------------------------------------
///  @brief Method to step the rays.
//------------------------------------------------------------------------------
        virtual void step() = 0;
    };

//******************************************************************************
//  Second Order Runge Kutta.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Second Order Runge Kutta class.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
///  @brief Class interface the solver.
//------------------------------------------------------------------------------
    template<class DISPERSION_FUNCTION>
    class rk2 : public solver_interface<DISPERSION_FUNCTION> {
///  Next kx value.
        std::shared_ptr<graph::leaf_node> kx_next;
///  Next ky value.
        std::shared_ptr<graph::leaf_node> ky_next;
///  Next kz value.
        std::shared_ptr<graph::leaf_node> kz_next;
///  Next kx value.
        std::shared_ptr<graph::leaf_node> x_next;
///  Next ky value.
        std::shared_ptr<graph::leaf_node> y_next;
///  Next kz value.
        std::shared_ptr<graph::leaf_node> z_next;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a new second order runge kutta solver.
///
///  @param[in] w  Inital omega.
///  @param[in] kx Inital kx.
///  @param[in] ky Inital ky.
///  @param[in] kz Inital kz.
///  @param[in] x  Inital x.
///  @param[in] y  Inital y.
///  @param[in] z  Inital z.
///  @param[in] dt Inital dt.
//------------------------------------------------------------------------------
        rk2(std::shared_ptr<graph::leaf_node> w,
            std::shared_ptr<graph::leaf_node> kx,
            std::shared_ptr<graph::leaf_node> ky,
            std::shared_ptr<graph::leaf_node> kz,
            std::shared_ptr<graph::leaf_node> x,
            std::shared_ptr<graph::leaf_node> y,
            std::shared_ptr<graph::leaf_node> z,
            const double dt) :
        solver_interface<DISPERSION_FUNCTION> (w, kx, ky, kz, x, y, z) {
            auto dt_const = graph::constant(dt);

            auto kx1 = this->D.get_dkxdt();
            auto ky1 = this->D.get_dkydt();
            auto kz1 = this->D.get_dkzdt();
            auto x1 = this->D.get_dxdt();
            auto y1 = this->D.get_dydt();
            auto z1 = this->D.get_dzdt();

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D2(w,
                                                                     kx + dt_const*kx1,
                                                                     ky + dt_const*ky1,
                                                                     kz + dt_const*kz1,
                                                                     x + dt_const*x1,
                                                                     y + dt_const*y1,
                                                                     z + dt_const*z1);

            auto two = graph::constant(2);

            kx_next = kx + dt_const*(kx1 + D2.get_dkxdt())/two;
            ky_next = ky + dt_const*(ky1 + D2.get_dkydt())/two;
            kz_next = kz + dt_const*(kz1 + D2.get_dkzdt())/two;
            x_next  = x + dt_const*(x1 + D2.get_dxdt())/two;
            y_next  = y + dt_const*(y1 + D2.get_dydt())/two;
            z_next  = z + dt_const*(z1 + D2.get_dzdt())/two;
        }

//------------------------------------------------------------------------------
///  @brief Method to step the rays.
//------------------------------------------------------------------------------
        virtual void step() final {
//  First intermedate steps.
            this->kx->set(this->kx_next->evaluate());
            this->ky->set(this->ky_next->evaluate());
            this->kz->set(this->kz_next->evaluate());
            this->x->set(this->x_next->evaluate());
            this->y->set(this->y_next->evaluate());
            this->z->set(this->z_next->evaluate());

            this->state.push_back(solve_state(this->kx->evaluate(),
                                              this->ky->evaluate(),
                                              this->kz->evaluate(),
                                              this->x->evaluate(),
                                              this->y->evaluate(),
                                              this->z->evaluate()));
            this->state.pop_front();
        }
    };
}

#endif /* solver_h */
