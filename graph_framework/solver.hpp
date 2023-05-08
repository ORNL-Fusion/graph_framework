//------------------------------------------------------------------------------
///  @file solver.hpp
///  @brief Base class for a ode solvers.
///
///  Defines a ode solver.
//------------------------------------------------------------------------------

#ifndef solver_h
#define solver_h

#include <list>
#include <array>

#include "dispersion.hpp"

namespace solver {
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
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> w;
///  kx variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kx;
///  ky variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> ky;
///  kz variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kz;
///  x variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> x;
///  y variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> y;
///  z variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> z;
///  t variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> t;

///  Dispersion function interface.
        dispersion::dispersion_interface<DISPERSION_FUNCTION> D;

///  Next kx value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kx_next;
///  Next ky value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> ky_next;
///  Next kz value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kz_next;
///  Next kx value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> x_next;
///  Next ky value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> y_next;
///  Next kz value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> z_next;
///  Next t value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> t_next;

///  Residule.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> residule;

///  Workflow manager.
        workflow::manager<typename DISPERSION_FUNCTION::base> work;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a new solver_interface with inital conditions.
///
///  @params[in] w  Inital w.
///  @params[in] kx Inital kx.
///  @params[in] ky Inital ky.
///  @params[in] kz Inital kz.
///  @params[in] x  Inital x.
///  @params[in] y  Inital y.
///  @params[in] z  Inital z.
///  @params[in] t  Inital t.
///  @params[in] eq The plasma equilibrium.
//------------------------------------------------------------------------------
        solver_interface(graph::shared_leaf<typename DISPERSION_FUNCTION::base> w,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base> kx,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base> ky,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base> kz,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base> x,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base> y,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base> z,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base> t,
                         equilibrium::unique_equilibrium<typename DISPERSION_FUNCTION::base> &eq) :
        D(w, kx, ky, kz, x, y, z, t, eq), w(w),
        kx(kx), ky(ky), kz(kz),
        x(x), y(y), z(z), t(t) {}

//------------------------------------------------------------------------------
///  @brief Method to initalize the rays.
///
///  @params[in,out] x              Variable reference to update.
///  @params[in]     tolarance      Tolarance to solve to dispersion function to.
///  @params[in]     max_iterations Maximum number of iterations to run.
///  @returns The residule graph.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<typename DISPERSION_FUNCTION::base>
        init(graph::shared_leaf<typename DISPERSION_FUNCTION::base> x,
             const typename DISPERSION_FUNCTION::base tolarance = 1.0E-30,
             const size_t max_iterations = 1000) final {
            graph::input_nodes<typename DISPERSION_FUNCTION::base> inputs {
                graph::variable_cast(this->t),
                graph::variable_cast(this->w),
                graph::variable_cast(this->x),
                graph::variable_cast(this->y),
                graph::variable_cast(this->z),
                graph::variable_cast(this->kx),
                graph::variable_cast(this->ky),
                graph::variable_cast(this->kz)
            };

            residule = this->D.solve(x, inputs, tolarance, max_iterations);

            return residule;
        }

//------------------------------------------------------------------------------
///  @brief Compile the solver function.
//------------------------------------------------------------------------------
        void compile() {
            graph::input_nodes<typename DISPERSION_FUNCTION::base> inputs = {
                graph::variable_cast(this->t),
                graph::variable_cast(this->w),
                graph::variable_cast(this->x),
                graph::variable_cast(this->y),
                graph::variable_cast(this->z),
                graph::variable_cast(this->kx),
                graph::variable_cast(this->ky),
                graph::variable_cast(this->kz)
            };

            graph::output_nodes<typename DISPERSION_FUNCTION::base> outputs = {
                this->residule
            };

            graph::map_nodes<typename DISPERSION_FUNCTION::base> setters = {
                {this->kx_next, graph::variable_cast(this->kx)},
                {this->ky_next, graph::variable_cast(this->ky)},
                {this->kz_next, graph::variable_cast(this->kz)},
                {this->x_next, graph::variable_cast(this->x)},
                {this->y_next, graph::variable_cast(this->y)},
                {this->z_next, graph::variable_cast(this->z)},
                {this->t_next, graph::variable_cast(this->t)}
            };

            work.add_item(inputs, outputs, setters, "solver_kernel");
            work.compile();
        }

//------------------------------------------------------------------------------
///  @brief Syncronize results from host to gpu.
//------------------------------------------------------------------------------
        void sync_device() {
            work.copy_to_device(this->t, graph::variable_cast(this->t)->data());
            work.copy_to_device(this->w, graph::variable_cast(this->w)->data());
            work.copy_to_device(this->x, graph::variable_cast(this->x)->data());
            work.copy_to_device(this->y, graph::variable_cast(this->y)->data());
            work.copy_to_device(this->z, graph::variable_cast(this->z)->data());
            work.copy_to_device(this->kx, graph::variable_cast(this->kx)->data());
            work.copy_to_device(this->ky, graph::variable_cast(this->ky)->data());
            work.copy_to_device(this->kz, graph::variable_cast(this->kz)->data());
        }

//------------------------------------------------------------------------------
///  @brief Syncronize results from gpu to host.
//------------------------------------------------------------------------------
        void sync_host() {
            work.copy_to_host(this->t, graph::variable_cast(this->t)->data());
            work.copy_to_host(this->w, graph::variable_cast(this->w)->data());
            work.copy_to_host(this->x, graph::variable_cast(this->x)->data());
            work.copy_to_host(this->y, graph::variable_cast(this->y)->data());
            work.copy_to_host(this->z, graph::variable_cast(this->z)->data());
            work.copy_to_host(this->kx, graph::variable_cast(this->kx)->data());
            work.copy_to_host(this->ky, graph::variable_cast(this->ky)->data());
            work.copy_to_host(this->kz, graph::variable_cast(this->kz)->data());
        }

//------------------------------------------------------------------------------
///  @brief Method to step the rays.
//------------------------------------------------------------------------------
        void step() {
            work.run();
        }

//------------------------------------------------------------------------------
///  @brief Print out the results.
///
///  @params[in] index Ray index to print results of.
//------------------------------------------------------------------------------
        void print(const size_t index) {
            work.print(index);
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the dispersion relation.
//------------------------------------------------------------------------------
        void print_dispersion() {
            D.print_dispersion();
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the dkxdt.
//------------------------------------------------------------------------------
        void print_dkxdt() {
            D.print_dkxdt();
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the dkydt.
//------------------------------------------------------------------------------
        void print_dkydt() {
            D.print_dkydt();
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the dkzdt.
//------------------------------------------------------------------------------
        void print_dkzdt() {
            D.print_dkzdt();
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the dxdt.
//------------------------------------------------------------------------------
        void print_dxdt() {
            D.print_dxdt();
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the dydt.
//------------------------------------------------------------------------------
        void print_dydt() {
            D.print_dydt();
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the dzdt.
//------------------------------------------------------------------------------
        void print_dzdt() {
            D.print_dzdt();
        }

///  Type def to retrieve the backend base type.
        typedef typename DISPERSION_FUNCTION::base base;
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
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kx1;
///  ky1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> ky1;
///  kz1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kz1;
///  x1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> x1;
///  y1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> y1;
///  z1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> z1;

///  kx2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kx2;
///  ky2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> ky2;
///  kz2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kz2;
///  x2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> x2;
///  y2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> y2;
///  z2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> z2;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a new second order runge kutta solver.
///
///  @params[in] w  Inital omega.
///  @params[in] kx Inital kx.
///  @params[in] ky Inital ky.
///  @params[in] kz Inital kz.
///  @params[in] x  Inital x.
///  @params[in] y  Inital y.
///  @params[in] z  Inital z.
///  @params[in] t  Inital t.
///  @params[in] dt Inital dt.
//------------------------------------------------------------------------------
        rk2(graph::shared_leaf<typename DISPERSION_FUNCTION::base> w,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base> kx,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base> ky,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base> kz,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base> x,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base> y,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base> z,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base> t,
            const typename DISPERSION_FUNCTION::base dt,
            equilibrium::unique_equilibrium<typename DISPERSION_FUNCTION::base> &eq) :
        solver_interface<DISPERSION_FUNCTION> (w, kx, ky, kz, x, y, z, t, eq) {
            auto dt_const = graph::constant(static_cast<typename DISPERSION_FUNCTION::base> (dt));

            this->kx1 = dt_const*this->D.get_dkxdt();
            this->ky1 = dt_const*this->D.get_dkydt();
            this->kz1 = dt_const*this->D.get_dkzdt();
            this->x1  = dt_const*this->D.get_dxdt();
            this->y1  = dt_const*this->D.get_dydt();
            this->z1  = dt_const*this->D.get_dzdt();

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D2(this->w,
                                                                     graph::pseudo_variable(this->kx + kx1),
                                                                     graph::pseudo_variable(this->ky + ky1),
                                                                     graph::pseudo_variable(this->kz + kz1),
                                                                     graph::pseudo_variable(this->x  + x1),
                                                                     graph::pseudo_variable(this->y  + y1),
                                                                     graph::pseudo_variable(this->z  + z1),
                                                                     graph::pseudo_variable(this->t  + dt_const),
                                                                     eq);

            this->kx2 = dt_const*D2.get_dkxdt();
            this->ky2 = dt_const*D2.get_dkydt();
            this->kz2 = dt_const*D2.get_dkzdt();
            this->x2  = dt_const*D2.get_dxdt();
            this->y2  = dt_const*D2.get_dydt();
            this->z2  = dt_const*D2.get_dzdt();

            auto two = graph::two<typename DISPERSION_FUNCTION::base> ();

            this->kx_next = this->kx + (this->kx1 + this->kx2)/two;
            this->ky_next = this->ky + (this->ky1 + this->ky2)/two;
            this->kz_next = this->kz + (this->kz1 + this->kz2)/two;
            this->x_next  = this->x  + (this->x1  + this->x2 )/two;
            this->y_next  = this->y  + (this->y1  + this->y2 )/two;
            this->z_next  = this->z  + (this->z1  + this->z2 )/two;
            this->t_next  = this->t  + dt_const;
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
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kx1;
///  ky1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> ky1;
///  kz1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kz1;
///  x1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> x1;
///  y1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> y1;
///  z1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> z1;

///  kx2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kx2;
///  ky2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> ky2;
///  kz2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kz2;
///  x2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> x2;
///  y2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> y2;
///  z2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> z2;

///  kx3 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kx3;
///  ky3 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> ky3;
///  kz3 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kz3;
///  x3 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> x3;
///  y3 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> y3;
///  z3 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> z3;

///  kx4 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kx4;
///  ky4 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> ky4;
///  kz4 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kz4;
///  x4 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> x4;
///  y4 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> y4;
///  z4 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> z4;

///  t  subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> t_sub;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a new second order runge kutta solver.
///
///  @params[in] w  Inital omega.
///  @params[in] kx Inital kx.
///  @params[in] ky Inital ky.
///  @params[in] kz Inital kz.
///  @params[in] x  Inital x.
///  @params[in] y  Inital y.
///  @params[in] z  Inital z.
///  @params[in] t  Inital t.
///  @params[in] dt Inital dt.
//------------------------------------------------------------------------------
        rk4(graph::shared_leaf<typename DISPERSION_FUNCTION::base> w,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base> kx,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base> ky,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base> kz,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base> x,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base> y,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base> z,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base> t,
            const typename DISPERSION_FUNCTION::base dt,
            equilibrium::unique_equilibrium<typename DISPERSION_FUNCTION::base> &eq) :
        solver_interface<DISPERSION_FUNCTION> (w, kx, ky, kz, x, y, z, t, eq) {
            auto dt_const = graph::constant(static_cast<typename DISPERSION_FUNCTION::base> (dt));

            this->kx1 = dt_const*this->D.get_dkxdt();
            this->ky1 = dt_const*this->D.get_dkydt();
            this->kz1 = dt_const*this->D.get_dkzdt();
            this->x1  = dt_const*this->D.get_dxdt();
            this->y1  = dt_const*this->D.get_dydt();
            this->z1  = dt_const*this->D.get_dzdt();

            auto two = graph::two<typename DISPERSION_FUNCTION::base> ();

            this->t_sub = this->t + dt_const/two;

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D2(this->w,
                                                                     graph::pseudo_variable(this->kx + kx1/two),
                                                                     graph::pseudo_variable(this->ky + ky1/two),
                                                                     graph::pseudo_variable(this->kz + kz1/two),
                                                                     graph::pseudo_variable(this->x  +  x1/two),
                                                                     graph::pseudo_variable(this->y  +  y1/two),
                                                                     graph::pseudo_variable(this->z  +  z1/two),
                                                                     graph::pseudo_variable(this->t_sub),
                                                                     eq);

            this->kx2 = dt_const*D2.get_dkxdt();
            this->ky2 = dt_const*D2.get_dkydt();
            this->kz2 = dt_const*D2.get_dkzdt();
            this->x2  = dt_const*D2.get_dxdt();
            this->y2  = dt_const*D2.get_dydt();
            this->z2  = dt_const*D2.get_dzdt();

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D3(this->w,
                                                                     graph::pseudo_variable(this->kx + kx2/two),
                                                                     graph::pseudo_variable(this->ky + ky2/two),
                                                                     graph::pseudo_variable(this->kz + kz2/two),
                                                                     graph::pseudo_variable(this->x  +  x2/two),
                                                                     graph::pseudo_variable(this->y  +  y2/two),
                                                                     graph::pseudo_variable(this->z  +  z2/two),
                                                                     graph::pseudo_variable(this->t_sub),
                                                                     eq);

            this->kx3 = dt_const*D3.get_dkxdt();
            this->ky3 = dt_const*D3.get_dkydt();
            this->kz3 = dt_const*D3.get_dkzdt();
            this->x3  = dt_const*D3.get_dxdt();
            this->y3  = dt_const*D3.get_dydt();
            this->z3  = dt_const*D3.get_dzdt();

            this->t_next = this->t + dt_const;

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D4(this->w,
                                                                     graph::pseudo_variable(this->kx + kx3),
                                                                     graph::pseudo_variable(this->ky + ky3),
                                                                     graph::pseudo_variable(this->kz + kz3),
                                                                     graph::pseudo_variable(this->x  + x3),
                                                                     graph::pseudo_variable(this->y  + y3),
                                                                     graph::pseudo_variable(this->z  + z3),
                                                                     graph::pseudo_variable(this->t_next),
                                                                     eq);

            this->kx4 = dt_const*D4.get_dkxdt();
            this->ky4 = dt_const*D4.get_dkydt();
            this->kz4 = dt_const*D4.get_dkzdt();
            this->x4  = dt_const*D4.get_dxdt();
            this->y4  = dt_const*D4.get_dydt();
            this->z4  = dt_const*D4.get_dzdt();

            auto six = graph::constant(static_cast<typename DISPERSION_FUNCTION::base> (6.0));

            this->kx_next = this->kx + (this->kx1 + two*(this->kx2 + this->kx3) + this->kx4)/six;
            this->ky_next = this->ky + (this->ky1 + two*(this->ky2 + this->ky3) + this->ky4)/six;
            this->kz_next = this->kz + (this->kz1 + two*(this->kz2 + this->kz3) + this->kz4)/six;
            this->x_next  = this->x  + (this->x1  + two*(this->x2  + this->x3 ) + this->x4 )/six;
            this->y_next  = this->y  + (this->y1  + two*(this->y2  + this->y3 ) + this->y4 )/six;
            this->z_next  = this->z  + (this->z1  + two*(this->z2  + this->z3 ) + this->z4 )/six;
        }
    };

//******************************************************************************
//  Split simplextic integrator
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Predictor corrector that trys to minimize the disperison residule.
//------------------------------------------------------------------------------
    template<class DISPERSION_FUNCTION>
    class split_simplextic : public solver_interface<DISPERSION_FUNCTION> {
    protected:
///  Half step x
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> x1;
///  Half step y
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> y1;
///  Half step z
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> z1;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a split simplextic integrator.
///
///  @params[in] w         Inital omega.
///  @params[in] kx        Inital kx.
///  @params[in] ky        Inital ky.
///  @params[in] kz        Inital kz.
///  @params[in] x         Inital x.
///  @params[in] y         Inital y.
///  @params[in] z         Inital z.
///  @params[in] t         Inital t.
///  @params[in] dt        Inital dt.
//------------------------------------------------------------------------------
        split_simplextic(graph::shared_leaf<typename DISPERSION_FUNCTION::base> w,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base> kx,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base> ky,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base> kz,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base> x,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base> y,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base> z,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base> t,
                         const typename DISPERSION_FUNCTION::base dt,
                         equilibrium::unique_equilibrium<typename DISPERSION_FUNCTION::base> &eq) :
        solver_interface<DISPERSION_FUNCTION> (w, kx, ky, kz, x, y, z, t, eq) {

//  Test if the function is separatable.
#ifdef USE_REDUCE
            auto zero = graph::zero<typename DISPERSION_FUNCTION::base> ();
            
            assert(zero->is_match(this->D.get_dkxdt()->df(kx)) &&
                   zero->is_match(this->D.get_dkxdt()->df(ky)) &&
                   zero->is_match(this->D.get_dkxdt()->df(kz)) &&
                   zero->is_match(this->D.get_dkydt()->df(kx)) &&
                   zero->is_match(this->D.get_dkydt()->df(ky)) &&
                   zero->is_match(this->D.get_dkydt()->df(kz)) &&
                   zero->is_match(this->D.get_dkzdt()->df(kx)) &&
                   zero->is_match(this->D.get_dkzdt()->df(ky)) &&
                   zero->is_match(this->D.get_dkzdt()->df(kz)) &&
                   zero->is_match(this->D.get_dxdt()->df(x))   &&
                   zero->is_match(this->D.get_dxdt()->df(y))   &&
                   zero->is_match(this->D.get_dxdt()->df(z))   &&
                   zero->is_match(this->D.get_dydt()->df(x))   &&
                   zero->is_match(this->D.get_dydt()->df(y))   &&
                   zero->is_match(this->D.get_dydt()->df(z))   &&
                   zero->is_match(this->D.get_dzdt()->df(x))   &&
                   zero->is_match(this->D.get_dzdt()->df(y))   &&
                   zero->is_match(this->D.get_dzdt()->df(z))   &&
                   "Hamiltonian is not separable.");
#else
            auto check_zero = static_cast<typename DISPERSION_FUNCTION::base> (0.0);
            assert(this->D.get_dkxdt()->df(kx)->evaluate()[0] == check_zero &&
                   this->D.get_dkxdt()->df(ky)->evaluate()[0] == check_zero &&
                   this->D.get_dkxdt()->df(kz)->evaluate()[0] == check_zero &&
                   this->D.get_dkydt()->df(kx)->evaluate()[0] == check_zero &&
                   this->D.get_dkydt()->df(ky)->evaluate()[0] == check_zero &&
                   this->D.get_dkydt()->df(kz)->evaluate()[0] == check_zero &&
                   this->D.get_dkzdt()->df(kx)->evaluate()[0] == check_zero &&
                   this->D.get_dkzdt()->df(ky)->evaluate()[0] == check_zero &&
                   this->D.get_dkzdt()->df(kz)->evaluate()[0] == check_zero &&
                   this->D.get_dxdt()->df(x)->evaluate()[0] == check_zero   &&
                   this->D.get_dxdt()->df(y)->evaluate()[0] == check_zero   &&
                   this->D.get_dxdt()->df(z)->evaluate()[0] == check_zero   &&
                   this->D.get_dydt()->df(x)->evaluate()[0] == check_zero   &&
                   this->D.get_dydt()->df(y)->evaluate()[0] == check_zero   &&
                   this->D.get_dydt()->df(z)->evaluate()[0] == check_zero   &&
                   this->D.get_dzdt()->df(x)->evaluate()[0] == check_zero   &&
                   this->D.get_dzdt()->df(y)->evaluate()[0] == check_zero   &&
                   this->D.get_dzdt()->df(z)->evaluate()[0] == check_zero   &&
                   "Hamiltonian is not separable.");
#endif

            auto dt_const = graph::constant(static_cast<typename DISPERSION_FUNCTION::base> (dt));
            auto two = graph::two<typename DISPERSION_FUNCTION::base> ();

            this->t_next = this->t + dt_const;

            this->x1 = this->x + dt_const*this->D.get_dxdt()/two;
            this->y1 = this->y + dt_const*this->D.get_dydt()/two;
            this->z1 = this->z + dt_const*this->D.get_dzdt()/two;

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D2(this->w,
                                                                     graph::pseudo_variable(this->kx),
                                                                     graph::pseudo_variable(this->ky),
                                                                     graph::pseudo_variable(this->kz),
                                                                     graph::pseudo_variable(this->x1),
                                                                     graph::pseudo_variable(this->y1),
                                                                     graph::pseudo_variable(this->z1),
                                                                     graph::pseudo_variable(this->t),
                                                                     eq);

            this->kx_next = this->kx + dt_const*D2.get_dkxdt();
            this->ky_next = this->ky + dt_const*D2.get_dkydt();
            this->kz_next = this->kz + dt_const*D2.get_dkzdt();

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D3(this->w,
                                                                     graph::pseudo_variable(this->kx_next),
                                                                     graph::pseudo_variable(this->ky_next),
                                                                     graph::pseudo_variable(this->kz_next),
                                                                     graph::pseudo_variable(this->x1),
                                                                     graph::pseudo_variable(this->y1),
                                                                     graph::pseudo_variable(this->z1),
                                                                     graph::pseudo_variable(this->t),
                                                                     eq);

            this->x_next  = this->x1 + dt_const*D3.get_dxdt()/two;
            this->y_next  = this->y1 + dt_const*D3.get_dydt()/two;
            this->z_next  = this->z1 + dt_const*D3.get_dzdt()/two;
        }
    };
}

#endif /* solver_h */
