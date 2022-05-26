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
#include "equilibrium.hpp"

namespace solver {
//******************************************************************************
//  Solve State
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Solve state contains the variables.
//------------------------------------------------------------------------------
    template<class BACKEND>
    struct solve_state {
///  Current state of the wave number in the x direction.
        const BACKEND kx;
///  Current state of the wave number in the y direction.
        const BACKEND ky;
///  Current state of the wave number in the z direction.
        const BACKEND kz;
///  Current state x position.
        const BACKEND x;
///  Current state y position.
        const BACKEND y;
///  Current state z position.
        const BACKEND z;

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
        solve_state(const BACKEND &kx0,
                    const BACKEND &ky0,
                    const BACKEND &kz0,
                    const BACKEND &x0,
                    const BACKEND &y0,
                    const BACKEND &z0) :
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
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> w;
///  kx variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kx;
///  ky variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> ky;
///  kz variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kz;
///  x variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> x;
///  y variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> y;
///  z variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> z;

///  Dispersion function interface.
       dispersion::dispersion_interface<DISPERSION_FUNCTION> D;

///  Next kx value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kx_next;
///  Next ky value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> ky_next;
///  Next kz value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kz_next;
///  Next kx value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> x_next;
///  Next ky value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> y_next;
///  Next kz value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> z_next;

    public:
///  Ray solution.
        std::list<solve_state<typename DISPERSION_FUNCTION::backend>> state;

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
///  @param[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        solver_interface(graph::shared_leaf<typename DISPERSION_FUNCTION::backend> w,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kx,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::backend> ky,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kz,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::backend> x,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::backend> y,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::backend> z,
                         equilibrium::unique_equilibrium<typename DISPERSION_FUNCTION::backend> &eq) :
        D(w, kx, ky, kz, x, y, z, eq), w(w),
        kx(kx), ky(ky), kz(kz),
        x(x), y(y), z(z) {}

//------------------------------------------------------------------------------
///  @brief Method to initalize the rays.
//------------------------------------------------------------------------------
        virtual void init(graph::shared_leaf<typename DISPERSION_FUNCTION::backend> x,
                          const double tolarance = 1.0E-30,
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
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> residule() {
            return D.get_d()*D.get_d();
        }

//------------------------------------------------------------------------------
///  @brief Method to step the rays.
//------------------------------------------------------------------------------
        void step() {
            this->reset_cache();

//  Need to evaluate all the steps before setting them otherwise later values
//  will have the wrong conditions for when earlier values are set.
            const typename DISPERSION_FUNCTION::backend kx_result = this->kx_next->evaluate();
            const typename DISPERSION_FUNCTION::backend ky_result = this->ky_next->evaluate();
            const typename DISPERSION_FUNCTION::backend kz_result = this->kz_next->evaluate();
            const typename DISPERSION_FUNCTION::backend x_result = this->x_next->evaluate();
            const typename DISPERSION_FUNCTION::backend y_result = this->y_next->evaluate();
            const typename DISPERSION_FUNCTION::backend z_result = this->z_next->evaluate();

            this->kx->set(kx_result);
            this->ky->set(ky_result);
            this->kz->set(kz_result);
            this->x->set(x_result);
            this->y->set(y_result);
            this->z->set(z_result);

            this->state.push_back(solve_state(kx_result,
                                              ky_result,
                                              kz_result,
                                              x_result,
                                              y_result,
                                              z_result));
            this->state.pop_front();
        }

//------------------------------------------------------------------------------
///  @brief Reset Cache.
//------------------------------------------------------------------------------
        virtual void reset_cache() = 0;

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the dispersion relation.
//------------------------------------------------------------------------------
        void print_dispersion() {
            D.print_dispersion();
        }

///  Type def to retrieve the backend type.
        typedef typename DISPERSION_FUNCTION::backend backend;
    };

//******************************************************************************
//  Second Order Runge Kutta.
//******************************************************************************
//    template<class DISPERSION_FUNCTION>
//    class leap_frog : public

//******************************************************************************
//  Second Order Runge Kutta.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Second Order Runge Kutta class.
//------------------------------------------------------------------------------
    template<class DISPERSION_FUNCTION>
    class rk2 : public solver_interface<DISPERSION_FUNCTION> {
    protected:
///  kx1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kx1;
///  ky1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> ky1;
///  kz1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kz1;
///  x1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> x1;
///  y1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> y1;
///  z1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> z1;

///  kx2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kx2;
///  ky2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> ky2;
///  kz2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kz2;
///  x2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> x2;
///  y2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> y2;
///  z2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> z2;

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
        rk2(graph::shared_leaf<typename DISPERSION_FUNCTION::backend> w,
            graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kx,
            graph::shared_leaf<typename DISPERSION_FUNCTION::backend> ky,
            graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kz,
            graph::shared_leaf<typename DISPERSION_FUNCTION::backend> x,
            graph::shared_leaf<typename DISPERSION_FUNCTION::backend> y,
            graph::shared_leaf<typename DISPERSION_FUNCTION::backend> z,
            const double dt,
            equilibrium::unique_equilibrium<typename DISPERSION_FUNCTION::backend> &eq) :
        solver_interface<DISPERSION_FUNCTION> (w, kx, ky, kz, x, y, z, eq) {
            auto dt_const = graph::constant<typename DISPERSION_FUNCTION::backend> (dt);

            this->kx1 = graph::cache(dt_const*this->D.get_dkxdt());
            this->ky1 = graph::cache(dt_const*this->D.get_dkydt());
            this->kz1 = graph::cache(dt_const*this->D.get_dkzdt());
            this->x1  = graph::cache(dt_const*this->D.get_dxdt());
            this->y1  = graph::cache(dt_const*this->D.get_dydt());
            this->z1  = graph::cache(dt_const*this->D.get_dzdt());

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D2(this->w,
                                                                     graph::pseudo_variable(this->kx + kx1),
                                                                     graph::pseudo_variable(this->ky + ky1),
                                                                     graph::pseudo_variable(this->kz + kz1),
                                                                     graph::pseudo_variable(this->x  + x1),
                                                                     graph::pseudo_variable(this->y  + y1),
                                                                     graph::pseudo_variable(this->z  + z1),
                                                                     eq);

            this->kx2 = graph::cache(dt_const*D2.get_dkxdt());
            this->ky2 = graph::cache(dt_const*D2.get_dkydt());
            this->kz2 = graph::cache(dt_const*D2.get_dkzdt());
            this->x2  = graph::cache(dt_const*D2.get_dxdt());
            this->y2  = graph::cache(dt_const*D2.get_dydt());
            this->z2  = graph::cache(dt_const*D2.get_dzdt());

            auto two = graph::constant<typename DISPERSION_FUNCTION::backend> (2);

            this->kx_next = this->kx + (this->kx1 + this->kx2)/two;
            this->ky_next = this->ky + (this->ky1 + this->ky2)/two;
            this->kz_next = this->kz + (this->kz1 + this->kz2)/two;
            this->x_next  = this->x  + (this->x1  + this->x2 )/two;
            this->y_next  = this->y  + (this->y1  + this->y2 )/two;
            this->z_next  = this->z  + (this->z1  + this->z2 )/two;
        }

//------------------------------------------------------------------------------
///  @brief Reset Cache.
//------------------------------------------------------------------------------
        virtual void reset_cache() final {
            this->kx1->reset_cache();
            this->ky1->reset_cache();
            this->kz1->reset_cache();
            this->x1->reset_cache();
            this->y1->reset_cache();
            this->z1->reset_cache();

            this->kx2->reset_cache();
            this->ky2->reset_cache();
            this->kz2->reset_cache();
            this->x2->reset_cache();
            this->y2->reset_cache();
            this->z2->reset_cache();
        }
    };

//******************************************************************************
//  Fourth Order Runge Kutta.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Fourth Order Runge Kutta class.
//------------------------------------------------------------------------------
    template<class DISPERSION_FUNCTION>
    class rk4 : public solver_interface<DISPERSION_FUNCTION> {
    protected:
///  kx1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kx1;
///  ky1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> ky1;
///  kz1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kz1;
///  x1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> x1;
///  y1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> y1;
///  z1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> z1;

///  kx2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kx2;
///  ky2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> ky2;
///  kz2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kz2;
///  x2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> x2;
///  y2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> y2;
///  z2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> z2;

///  kx3 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kx3;
///  ky3 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> ky3;
///  kz3 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kz3;
///  x3 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> x3;
///  y3 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> y3;
///  z3 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> z3;

///  kx4 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kx4;
///  ky4 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> ky4;
///  kz4 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kz4;
///  x4 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> x4;
///  y4 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> y4;
///  z4 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::backend> z4;

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
        rk4(graph::shared_leaf<typename DISPERSION_FUNCTION::backend> w,
            graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kx,
            graph::shared_leaf<typename DISPERSION_FUNCTION::backend> ky,
            graph::shared_leaf<typename DISPERSION_FUNCTION::backend> kz,
            graph::shared_leaf<typename DISPERSION_FUNCTION::backend> x,
            graph::shared_leaf<typename DISPERSION_FUNCTION::backend> y,
            graph::shared_leaf<typename DISPERSION_FUNCTION::backend> z,
            const double dt,
            equilibrium::unique_equilibrium<typename DISPERSION_FUNCTION::backend> &eq) :
        solver_interface<DISPERSION_FUNCTION> (w, kx, ky, kz, x, y, z, eq) {
            auto dt_const = graph::constant<typename DISPERSION_FUNCTION::backend> (dt);

            this->kx1 = graph::cache(dt_const*this->D.get_dkxdt());
            this->ky1 = graph::cache(dt_const*this->D.get_dkydt());
            this->kz1 = graph::cache(dt_const*this->D.get_dkzdt());
            this->x1  = graph::cache(dt_const*this->D.get_dxdt());
            this->y1  = graph::cache(dt_const*this->D.get_dydt());
            this->z1  = graph::cache(dt_const*this->D.get_dzdt());

            auto two = graph::constant<typename DISPERSION_FUNCTION::backend> (2);

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D2(this->w,
                                                                     graph::pseudo_variable(this->kx + kx1/two),
                                                                     graph::pseudo_variable(this->ky + ky1/two),
                                                                     graph::pseudo_variable(this->kz + kz1/two),
                                                                     graph::pseudo_variable(this->x  + x1 /two),
                                                                     graph::pseudo_variable(this->y  + y1 /two),
                                                                     graph::pseudo_variable(this->z  + z1 /two),
                                                                     eq);

            this->kx2 = graph::cache(dt_const*D2.get_dkxdt());
            this->ky2 = graph::cache(dt_const*D2.get_dkydt());
            this->kz2 = graph::cache(dt_const*D2.get_dkzdt());
            this->x2  = graph::cache(dt_const*D2.get_dxdt());
            this->y2  = graph::cache(dt_const*D2.get_dydt());
            this->z2  = graph::cache(dt_const*D2.get_dzdt());

            auto result = this->kx2->evaluate();

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D3(this->w,
                                                                     graph::pseudo_variable(this->kx + kx2/two),
                                                                     graph::pseudo_variable(this->ky + ky2/two),
                                                                     graph::pseudo_variable(this->kz + kz2/two),
                                                                     graph::pseudo_variable(this->x  + x2 /two),
                                                                     graph::pseudo_variable(this->y  + y2 /two),
                                                                     graph::pseudo_variable(this->z  + z2 /two),
                                                                     eq);

            this->kx3 = graph::cache(dt_const*D3.get_dkxdt());
            this->ky3 = graph::cache(dt_const*D3.get_dkydt());
            this->kz3 = graph::cache(dt_const*D3.get_dkzdt());
            this->x3  = graph::cache(dt_const*D3.get_dxdt());
            this->y3  = graph::cache(dt_const*D3.get_dydt());
            this->z3  = graph::cache(dt_const*D3.get_dzdt());

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D4(this->w,
                                                                     graph::pseudo_variable(this->kx + kx3),
                                                                     graph::pseudo_variable(this->ky + ky3),
                                                                     graph::pseudo_variable(this->kz + kz3),
                                                                     graph::pseudo_variable(this->x  + x3),
                                                                     graph::pseudo_variable(this->y  + y3),
                                                                     graph::pseudo_variable(this->z  + z3),
                                                                     eq);

            this->kx4 = graph::cache(dt_const*D4.get_dkxdt());
            this->ky4 = graph::cache(dt_const*D4.get_dkydt());
            this->kz4 = graph::cache(dt_const*D4.get_dkzdt());
            this->x4  = graph::cache(dt_const*D4.get_dxdt());
            this->y4  = graph::cache(dt_const*D4.get_dydt());
            this->z4  = graph::cache(dt_const*D4.get_dzdt());

            auto six = graph::constant<typename DISPERSION_FUNCTION::backend> (6);

            this->kx_next = this->kx + (this->kx1 + two*(this->kx2 + this->kx3) + this->kx4)/six;
            this->ky_next = this->ky + (this->ky1 + two*(this->ky2 + this->ky3) + this->ky4)/six;
            this->kz_next = this->kz + (this->kz1 + two*(this->kz2 + this->kz3) + this->kz4)/six;
            this->x_next  = this->x  + (this->x1  + two*(this->x2  + this->x3 ) + this->x4 )/six;
            this->y_next  = this->y  + (this->y1  + two*(this->y2  + this->y3 ) + this->y4 )/six;
            this->z_next  = this->z  + (this->z1  + two*(this->z2  + this->z3 ) + this->z4 )/six;
        }

//------------------------------------------------------------------------------
///  @brief Reset Cache.
//------------------------------------------------------------------------------
        virtual void reset_cache() final {
            this->kx1->reset_cache();
            this->ky1->reset_cache();
            this->kz1->reset_cache();
            this->x1->reset_cache();
            this->y1->reset_cache();
            this->z1->reset_cache();

            this->kx2->reset_cache();
            this->ky2->reset_cache();
            this->kz2->reset_cache();
            this->x2->reset_cache();
            this->y2->reset_cache();
            this->z2->reset_cache();

            this->kx3->reset_cache();
            this->ky3->reset_cache();
            this->kz3->reset_cache();
            this->x3->reset_cache();
            this->y3->reset_cache();
            this->z3->reset_cache();

            this->kx4->reset_cache();
            this->ky4->reset_cache();
            this->kz4->reset_cache();
            this->x4->reset_cache();
            this->y4->reset_cache();
            this->z4->reset_cache();
        }
    };
}

#endif /* solver_h */
