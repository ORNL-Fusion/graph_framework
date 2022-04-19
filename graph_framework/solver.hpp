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
        std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> w;
///  kx variable.
        std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> kx;
///  ky variable.
        std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> ky;
///  kz variable.
        std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> kz;
///  x variable.
        std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> x;
///  y variable.
        std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> y;
///  z variable.
        std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> z;

///  Dispersion function interface.
       dispersion::dispersion_interface<DISPERSION_FUNCTION> D;

///  Next kx value.
        std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> kx_next;
///  Next ky value.
        std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> ky_next;
///  Next kz value.
        std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> kz_next;
///  Next kx value.
        std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> x_next;
///  Next ky value.
        std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> y_next;
///  Next kz value.
        std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> z_next;

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
//------------------------------------------------------------------------------
        solver_interface(std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> w,
                         std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> kx,
                         std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> ky,
                         std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> kz,
                         std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> x,
                         std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> y,
                         std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> z) :
        D(w, kx, ky, kz, x, y, z), w(w),
        kx(kx), ky(ky), kz(kz),
        x(x), y(y), z(z) {}

//------------------------------------------------------------------------------
///  @brief Method to initalize the rays.
//------------------------------------------------------------------------------
        virtual void init(std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> x,
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
        std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> residule() {
            return D.get_d()*D.get_d();
        }

//------------------------------------------------------------------------------
///  @brief Method to step the rays.
//------------------------------------------------------------------------------
        void step() {
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

///  Type def to retrieve the backend type.
        typedef typename DISPERSION_FUNCTION::backend backend;
    };

//******************************************************************************
//  Second Order Runge Kutta.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Second Order Runge Kutta class.
//------------------------------------------------------------------------------
    template<class DISPERSION_FUNCTION>
    class rk2 : public solver_interface<DISPERSION_FUNCTION> {
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
        rk2(std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> w,
            std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> kx,
            std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> ky,
            std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> kz,
            std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> x,
            std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> y,
            std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> z,
            const double dt) :
        solver_interface<DISPERSION_FUNCTION> (w, kx, ky, kz, x, y, z) {
            auto dt_const = graph::constant<typename DISPERSION_FUNCTION::backend> (dt);

            auto kx1 = this->D.get_dkxdt();
            auto ky1 = this->D.get_dkydt();
            auto kz1 = this->D.get_dkzdt();
            auto x1  = this->D.get_dxdt();
            auto y1  = this->D.get_dydt();
            auto z1  = this->D.get_dzdt();

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D2(this->w,
                                                                     this->kx + dt_const*kx1,
                                                                     this->ky + dt_const*ky1,
                                                                     this->kz + dt_const*kz1,
                                                                     this->x  + dt_const*x1,
                                                                     this->y  + dt_const*y1,
                                                                     this->z  + dt_const*z1);

            auto two = graph::constant<typename DISPERSION_FUNCTION::backend> (2);

            this->kx_next = this->kx + dt_const*(kx1 + D2.get_dkxdt())/two;
            this->ky_next = this->ky + dt_const*(ky1 + D2.get_dkydt())/two;
            this->kz_next = this->kz + dt_const*(kz1 + D2.get_dkzdt())/two;
            this->x_next  = this->x  + dt_const*(x1  + D2.get_dxdt() )/two;
            this->y_next  = this->y  + dt_const*(y1  + D2.get_dydt() )/two;
            this->z_next  = this->z  + dt_const*(z1  + D2.get_dzdt() )/two;
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
        rk4(std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> w,
            std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> kx,
            std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> ky,
            std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> kz,
            std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> x,
            std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> y,
            std::shared_ptr<graph::leaf_node<typename DISPERSION_FUNCTION::backend>> z,
            const double dt) :
        solver_interface<DISPERSION_FUNCTION> (w, kx, ky, kz, x, y, z) {
            auto dt_const = graph::constant<typename DISPERSION_FUNCTION::backend> (dt);

            auto kx1 = this->D.get_dkxdt();
            auto ky1 = this->D.get_dkydt();
            auto kz1 = this->D.get_dkzdt();
            auto x1  = this->D.get_dxdt();
            auto y1  = this->D.get_dydt();
            auto z1  = this->D.get_dzdt();

            auto two = graph::constant<typename DISPERSION_FUNCTION::backend> (2);

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D2(this->w,
                                                                     this->kx + dt_const*kx1/two,
                                                                     this->ky + dt_const*ky1/two,
                                                                     this->kz + dt_const*kz1/two,
                                                                     this->x  + dt_const*x1 /two,
                                                                     this->y  + dt_const*y1 /two,
                                                                     this->z  + dt_const*z1 /two);

            auto kx2 = D2.get_dkxdt();
            auto ky2 = D2.get_dkydt();
            auto kz2 = D2.get_dkzdt();
            auto x2  = D2.get_dxdt();
            auto y2  = D2.get_dydt();
            auto z2  = D2.get_dzdt();

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D3(this->w,
                                                                     this->kx + dt_const*kx2/two,
                                                                     this->ky + dt_const*ky2/two,
                                                                     this->kz + dt_const*kz2/two,
                                                                     this->x  + dt_const*x2 /two,
                                                                     this->y  + dt_const*y2 /two,
                                                                     this->z  + dt_const*z2 /two);

            auto kx3 = D3.get_dkxdt();
            auto ky3 = D3.get_dkydt();
            auto kz3 = D3.get_dkzdt();
            auto x3  = D3.get_dxdt();
            auto y3  = D3.get_dydt();
            auto z3  = D3.get_dzdt();

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D4(this->w,
                                                                     this->kx + dt_const*kx3,
                                                                     this->ky + dt_const*ky3,
                                                                     this->kz + dt_const*kz3,
                                                                     this->x  + dt_const*x3,
                                                                     this->y  + dt_const*y3,
                                                                     this->z  + dt_const*z3);

            auto six = graph::constant<typename DISPERSION_FUNCTION::backend> (6);

            this->kx_next = this->kx + dt_const*(kx1 + two*kx2 + two*kx3 + D4.get_dkxdt())/six;
            this->ky_next = this->ky + dt_const*(ky1 + two*ky2 + two*ky3 + D4.get_dkydt())/six;
            this->kz_next = this->kz + dt_const*(kz1 + two*kz2 + two*kz3 + D4.get_dkzdt())/six;
            this->x_next  = this->x  + dt_const*(x1  + two*x2  + two*x3  + D4.get_dxdt() )/six;
            this->y_next  = this->y  + dt_const*(y1  + two*y2  + two*y3  + D4.get_dydt() )/six;
            this->z_next  = this->z  + dt_const*(z1  + two*z2  + two*z3  + D4.get_dzdt() )/six;
        }
    };
}

#endif /* solver_h */
