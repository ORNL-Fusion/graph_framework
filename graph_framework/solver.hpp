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
    class solver_interface {
    protected:
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

    public:
///  Ray solution.
        std::list<solve_state> state;

//------------------------------------------------------------------------------
///  @brief Construct a new solver_interface with inital conditions.
///
///  @param[in] kx    Inital kx.
///  @param[in] ky    Inital ky.
///  @param[in] kz    Inital kz.
///  @param[in] x     Inital x.
///  @param[in] y     Inital y.
///  @param[in] z     Inital z.
//------------------------------------------------------------------------------
        solver_interface(std::shared_ptr<graph::leaf_node> kx,
                         std::shared_ptr<graph::leaf_node> ky,
                         std::shared_ptr<graph::leaf_node> kz,
                         std::shared_ptr<graph::leaf_node> x,
                         std::shared_ptr<graph::leaf_node> y,
                         std::shared_ptr<graph::leaf_node> z) :
        state(1, solve_state(kx->evaluate(),
                             ky->evaluate(),
                             kz->evaluate(),
                             x->evaluate(),
                             y->evaluate(),
                             z->evaluate())),
        kx(kx), ky(ky), kz(kz),
        x(x), y(y), z(z) {}

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
    template<class DISPERSION>
    class rk2 : public solver_interface {
///  Time step
        const double dt;

///  Second order rk 1st intermediate for kx.
        std::vector<double> kx1;
///  Second order rk 2nd intermediate for kx.
        std::vector<double> kx2;

///  Second order rk 1st intermediate for ky.
        std::vector<double> ky1;
///  Second order rk 2nd intermediate for ky.
        std::vector<double> ky2;

///  Second order rk 1st intermediate for kz.
        std::vector<double> kz1;
///  Second order rk 2nd intermediate for kz.
        std::vector<double> kz2;

///  Second order rk 1st intermediate for x.
        std::vector<double> x1;
///  Second order rk 2nd intermediate for x.
        std::vector<double> x2;

///  Second order rk 1st intermediate for y.
        std::vector<double> y1;
///  Second order rk 2nd intermediate for y.
        std::vector<double> y2;

///  Second order rk 1st intermediate for z.
        std::vector<double> z1;
///  Second order rk 2nd intermediate for z.
        std::vector<double> z2;

///  Dispersion function.
        DISPERSION D;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a new second order runge kutta solver.
///
///  @param[in] D   Dispersion function.
///  @param[in] kx Inital kx.
///  @param[in] ky Inital ky.
///  @param[in] kz Inital kz.
///  @param[in] x  Inital x.
///  @param[in] y  Inital y.
///  @param[in] z  Inital z.
///  @param[in] dt  Inital dt.
//------------------------------------------------------------------------------
        rk2(DISPERSION &D,
            std::shared_ptr<graph::leaf_node> kx,
            std::shared_ptr<graph::leaf_node> ky,
            std::shared_ptr<graph::leaf_node> kz,
            std::shared_ptr<graph::leaf_node> x,
            std::shared_ptr<graph::leaf_node> y,
            std::shared_ptr<graph::leaf_node> z,
            const double dt) :
        solver_interface(kx, ky, kz, x, y, z),
        D(D), dt(dt),
        kx1(state.back().size()), kx2(state.back().size()),
        ky1(state.back().size()), ky2(state.back().size()),
        kz1(state.back().size()), kz2(state.back().size()),
        x1(state.back().size()), x2(state.back().size()),
        y1(state.back().size()), y2(state.back().size()),
        z1(state.back().size()), z2(state.back().size()) {}

//------------------------------------------------------------------------------
///  @brief Method to step the rays.
//------------------------------------------------------------------------------
        virtual void step() final {
//  First intermedate steps.
            kx1 = D.get_dkxdt();
            kz1 = D.get_dkydt();
            ky1 = D.get_dkzdt();
            x1 = D.get_dxdt();;
            y1 = D.get_dydt();;
            z1 = D.get_dzdt();;

            for (size_t i = 0, ie = state.front().size(); i < ie; i++) {
                this->kx->set(i, state.back().kx.at(i) + dt*kx1.at(i));
                this->ky->set(i, state.back().ky.at(i) + dt*ky1.at(i));
                this->kz->set(i, state.back().kz.at(i) + dt*kz1.at(i));
                this->x->set(i, state.back().x.at(i) + dt*kx1.at(i));
                this->y->set(i, state.back().y.at(i) + dt*ky1.at(i));
                this->z->set(i, state.back().z.at(i) + dt*kz1.at(i));
            }

            kx2 = D.get_dkxdt();
            ky2 = D.get_dkydt();
            kz2 = D.get_dkzdt();
            x2 = D.get_dxdt();;
            y2 = D.get_dydt();;
            z2 = D.get_dzdt();;

            for (size_t i = 0, ie = state.front().size(); i < ie; i++) {
                this->kx->set(i, state.back().kx.at(i) + dt*(kx1.at(i) + kx2.at(i))/2);
                this->ky->set(i, state.back().ky.at(i) + dt*(ky1.at(i) + ky2.at(i))/2);
                this->kz->set(i, state.back().kz.at(i) + dt*(kz1.at(i) + kz2.at(i))/2);
                this->x->set(i, state.back().x.at(i) + dt*(x1.at(i) + x2.at(i))/2);
                this->y->set(i, state.back().y.at(i) +  dt*(y1.at(i) + y2.at(i))/2);
                this->z->set(i, state.back().z.at(i) + dt*(z1.at(i) + z2.at(i))/2);
            }

            state.push_back(solve_state(this->kx->evaluate(),
                                        this->ky->evaluate(),
                                        this->kz->evaluate(),
                                        this->x->evaluate(),
                                        this->y->evaluate(),
                                        this->z->evaluate()));
        }
    };
}

#endif /* solver_h */
