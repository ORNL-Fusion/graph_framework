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
#include "equilibrium.hpp"
#include "jit.hpp"

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

///  Jit context for the kernels.
        std::unique_ptr<jit::context<typename DISPERSION_FUNCTION::base>> source;
///  Kernel function call.
        std::function<void(void)> run;

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
            graph::input_nodes<typename DISPERSION_FUNCTION::base> inputs;
            if (x->is_match(this->w)) {
                inputs.push_back(graph::variable_cast(this->x));
                inputs.push_back(graph::variable_cast(this->y));
                inputs.push_back(graph::variable_cast(this->z));
                inputs.push_back(graph::variable_cast(this->kx));
                inputs.push_back(graph::variable_cast(this->ky));
                inputs.push_back(graph::variable_cast(this->kz));
                inputs.push_back(graph::variable_cast(this->t));
            } else if (x->is_match(this->x)) {
                inputs.push_back(graph::variable_cast(this->w));
                inputs.push_back(graph::variable_cast(this->y));
                inputs.push_back(graph::variable_cast(this->z));
                inputs.push_back(graph::variable_cast(this->kx));
                inputs.push_back(graph::variable_cast(this->ky));
                inputs.push_back(graph::variable_cast(this->kz));
                inputs.push_back(graph::variable_cast(this->t));
            } else if (x->is_match(this->y)) {
                inputs.push_back(graph::variable_cast(this->w));
                inputs.push_back(graph::variable_cast(this->x));
                inputs.push_back(graph::variable_cast(this->z));
                inputs.push_back(graph::variable_cast(this->kx));
                inputs.push_back(graph::variable_cast(this->ky));
                inputs.push_back(graph::variable_cast(this->kz));
                inputs.push_back(graph::variable_cast(this->t));
            } else if (x->is_match(this->z)) {
                inputs.push_back(graph::variable_cast(this->w));
                inputs.push_back(graph::variable_cast(this->x));
                inputs.push_back(graph::variable_cast(this->y));
                inputs.push_back(graph::variable_cast(this->kx));
                inputs.push_back(graph::variable_cast(this->ky));
                inputs.push_back(graph::variable_cast(this->kz));
                inputs.push_back(graph::variable_cast(this->t));
            } else if (x->is_match(this->kx)) {
                inputs.push_back(graph::variable_cast(this->w));
                inputs.push_back(graph::variable_cast(this->x));
                inputs.push_back(graph::variable_cast(this->y));
                inputs.push_back(graph::variable_cast(this->z));
                inputs.push_back(graph::variable_cast(this->ky));
                inputs.push_back(graph::variable_cast(this->kz));
                inputs.push_back(graph::variable_cast(this->t));
            } else if (x->is_match(this->ky)) {
                inputs.push_back(graph::variable_cast(this->w));
                inputs.push_back(graph::variable_cast(this->x));
                inputs.push_back(graph::variable_cast(this->y));
                inputs.push_back(graph::variable_cast(this->z));
                inputs.push_back(graph::variable_cast(this->kx));
                inputs.push_back(graph::variable_cast(this->kz));
                inputs.push_back(graph::variable_cast(this->t));
            } else if (x->is_match(this->kz)) {
                inputs.push_back(graph::variable_cast(this->w));
                inputs.push_back(graph::variable_cast(this->x));
                inputs.push_back(graph::variable_cast(this->y));
                inputs.push_back(graph::variable_cast(this->z));
                inputs.push_back(graph::variable_cast(this->kx));
                inputs.push_back(graph::variable_cast(this->ky));
                inputs.push_back(graph::variable_cast(this->t));
            } else if (x->is_match(this->t)) {
                inputs.push_back(graph::variable_cast(this->w));
                inputs.push_back(graph::variable_cast(this->x));
                inputs.push_back(graph::variable_cast(this->y));
                inputs.push_back(graph::variable_cast(this->z));
                inputs.push_back(graph::variable_cast(this->kx));
                inputs.push_back(graph::variable_cast(this->ky));
                inputs.push_back(graph::variable_cast(this->kz));
            }
            residule = this->D.solve(x, inputs, tolarance, max_iterations);

            return residule;
        }

//------------------------------------------------------------------------------
///  @brief Compile the solver function.
///
///  FIXME: For now this compiles and run the kernel for all time steps.
///
///  @params[in] num_rays  Number of rays in the solution.
//------------------------------------------------------------------------------
        void compile(const size_t num_rays) {
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

            this->source = std::make_unique<jit::context<typename DISPERSION_FUNCTION::base>> ();
            this->source->add_kernel("solver_kernel",
                                     inputs, outputs, setters);
            
            this->source->compile();
            this->run = source->create_kernel_call("solver_kernel",
                                                   inputs, outputs, num_rays);
        }

//------------------------------------------------------------------------------
///  @brief Syncronize results between host and gpu.
//------------------------------------------------------------------------------
        void sync() {
            this->source->copy_buffer(this->t, graph::variable_cast(this->t)->data());
            this->source->copy_buffer(this->w, graph::variable_cast(this->w)->data());
            this->source->copy_buffer(this->x, graph::variable_cast(this->x)->data());
            this->source->copy_buffer(this->y, graph::variable_cast(this->y)->data());
            this->source->copy_buffer(this->z, graph::variable_cast(this->z)->data());
            this->source->copy_buffer(this->kx, graph::variable_cast(this->kx)->data());
            this->source->copy_buffer(this->ky, graph::variable_cast(this->ky)->data());
            this->source->copy_buffer(this->kz, graph::variable_cast(this->kz)->data());
        }

//------------------------------------------------------------------------------
///  @brief Method to step the rays.
//------------------------------------------------------------------------------
        void step() {
            this->run();
        }

//------------------------------------------------------------------------------
///  @brief Print out the results.
///
///  @params[in] index Ray index to print results of.
//------------------------------------------------------------------------------
        void print(const size_t index) {
            this->source->print(index);
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
                                                                     graph::pseudo_variable(this->t  + dt_const),
                                                                     eq);

            this->kx2 = graph::cache(dt_const*D2.get_dkxdt());
            this->ky2 = graph::cache(dt_const*D2.get_dkydt());
            this->kz2 = graph::cache(dt_const*D2.get_dkzdt());
            this->x2  = graph::cache(dt_const*D2.get_dxdt());
            this->y2  = graph::cache(dt_const*D2.get_dydt());
            this->z2  = graph::cache(dt_const*D2.get_dzdt());

            auto two = graph::two<typename DISPERSION_FUNCTION::base> ();

            this->kx_next = this->kx + (this->kx1 + this->kx2)/two;
            this->ky_next = this->ky + (this->ky1 + this->ky2)/two;
            this->kz_next = this->kz + (this->kz1 + this->kz2)/two;
            this->x_next  = this->x  + (this->x1  + this->x2 )/two;
            this->y_next  = this->y  + (this->y1  + this->y2 )/two;
            this->z_next  = this->z  + (this->z1  + this->z2 )/two;
            this->t_next  = this->t  + dt_const;
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

            this->kx1 = graph::cache(dt_const*this->D.get_dkxdt());
            this->ky1 = graph::cache(dt_const*this->D.get_dkydt());
            this->kz1 = graph::cache(dt_const*this->D.get_dkzdt());
            this->x1  = graph::cache(dt_const*this->D.get_dxdt());
            this->y1  = graph::cache(dt_const*this->D.get_dydt());
            this->z1  = graph::cache(dt_const*this->D.get_dzdt());

            auto two = graph::two<typename DISPERSION_FUNCTION::base> ();

            this->t_sub = graph::cache(this->t + dt_const/two);

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D2(this->w,
                                                                     graph::pseudo_variable(this->kx + kx1/two),
                                                                     graph::pseudo_variable(this->ky + ky1/two),
                                                                     graph::pseudo_variable(this->kz + kz1/two),
                                                                     graph::pseudo_variable(this->x  +  x1/two),
                                                                     graph::pseudo_variable(this->y  +  y1/two),
                                                                     graph::pseudo_variable(this->z  +  z1/two),
                                                                     graph::pseudo_variable(this->t_sub),
                                                                     eq);

            this->kx2 = graph::cache(dt_const*D2.get_dkxdt());
            this->ky2 = graph::cache(dt_const*D2.get_dkydt());
            this->kz2 = graph::cache(dt_const*D2.get_dkzdt());
            this->x2  = graph::cache(dt_const*D2.get_dxdt());
            this->y2  = graph::cache(dt_const*D2.get_dydt());
            this->z2  = graph::cache(dt_const*D2.get_dzdt());

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D3(this->w,
                                                                     graph::pseudo_variable(this->kx + kx2/two),
                                                                     graph::pseudo_variable(this->ky + ky2/two),
                                                                     graph::pseudo_variable(this->kz + kz2/two),
                                                                     graph::pseudo_variable(this->x  +  x2/two),
                                                                     graph::pseudo_variable(this->y  +  y2/two),
                                                                     graph::pseudo_variable(this->z  +  z2/two),
                                                                     graph::pseudo_variable(this->t_sub),
                                                                     eq);

            this->kx3 = graph::cache(dt_const*D3.get_dkxdt());
            this->ky3 = graph::cache(dt_const*D3.get_dkydt());
            this->kz3 = graph::cache(dt_const*D3.get_dkzdt());
            this->x3  = graph::cache(dt_const*D3.get_dxdt());
            this->y3  = graph::cache(dt_const*D3.get_dydt());
            this->z3  = graph::cache(dt_const*D3.get_dzdt());

            this->t_next = graph::cache(this->t + dt_const);

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D4(this->w,
                                                                     graph::pseudo_variable(this->kx + kx3),
                                                                     graph::pseudo_variable(this->ky + ky3),
                                                                     graph::pseudo_variable(this->kz + kz3),
                                                                     graph::pseudo_variable(this->x  + x3),
                                                                     graph::pseudo_variable(this->y  + y3),
                                                                     graph::pseudo_variable(this->z  + z3),
                                                                     graph::pseudo_variable(this->t_next),
                                                                     eq);

            this->kx4 = graph::cache(dt_const*D4.get_dkxdt());
            this->ky4 = graph::cache(dt_const*D4.get_dkydt());
            this->kz4 = graph::cache(dt_const*D4.get_dkzdt());
            this->x4  = graph::cache(dt_const*D4.get_dxdt());
            this->y4  = graph::cache(dt_const*D4.get_dydt());
            this->z4  = graph::cache(dt_const*D4.get_dzdt());

            auto six = graph::constant(static_cast<typename DISPERSION_FUNCTION::base> (6.0));

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
            this->t_sub->reset_cache();
            this->t_next->reset_cache();

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

//******************************************************************************
//  Predictor Corrector
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Predictor corrector that trys to minimize the disperison residule.
//------------------------------------------------------------------------------
    template<class DISPERSION_FUNCTION>
    class predictor_corrector : public solver_interface<DISPERSION_FUNCTION> {
    protected:
///  First kx Predictor.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> dkxdt;
///  First ky Predictor.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> dkydt;
///  First kz Predictor.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> dkzdt;
///  First x Predictor.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> dxdt;
///  First y Predictor.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> dydt;
///  First z Predictor.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> dzdt;

///  First kx Predictor.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kx0_pred;
///  First ky Predictor.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> ky0_pred;
///  First kz Predictor.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kz0_pred;
///  First x Predictor.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> x0_pred;
///  First y Predictor.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> y0_pred;
///  First z Predictor.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> z0_pred;

//  Temp variable for predictor corrector iteration.
///  First kx Predictor.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kx1_var;
///  First ky Predictor.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> ky1_var;
///  First kz Predictor.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> kz1_var;
///  First x Predictor.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> x1_var;
///  First y Predictor.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> y1_var;
///  First z Predictor.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> z1_var;

///  Dispersion residule of the predicted corrected step.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base> residule_pred;

///  Convergence tolarance.
        const typename DISPERSION_FUNCTION::base tolarance;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a predictor corrector solver.
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
///  @params[in] tolarance Tolarance to solver the dispersion function to.
//------------------------------------------------------------------------------
        predictor_corrector(graph::shared_leaf<typename DISPERSION_FUNCTION::base> w,
                            graph::shared_leaf<typename DISPERSION_FUNCTION::base> kx,
                            graph::shared_leaf<typename DISPERSION_FUNCTION::base> ky,
                            graph::shared_leaf<typename DISPERSION_FUNCTION::base> kz,
                            graph::shared_leaf<typename DISPERSION_FUNCTION::base> x,
                            graph::shared_leaf<typename DISPERSION_FUNCTION::base> y,
                            graph::shared_leaf<typename DISPERSION_FUNCTION::base> z,
                            graph::shared_leaf<typename DISPERSION_FUNCTION::base> t,
                            const typename DISPERSION_FUNCTION::base dt,
                            equilibrium::unique_equilibrium<typename DISPERSION_FUNCTION::base> &eq,
                            const typename DISPERSION_FUNCTION::base tolarance = 1.0E-30) :
        solver_interface<DISPERSION_FUNCTION> (w, kx, ky, kz, x, y, z, t, eq),
        tolarance(tolarance) {
            auto dt_const = graph::constant(static_cast<typename DISPERSION_FUNCTION::base> (dt));

            this->dkxdt = graph::cache(dt_const*this->D.get_dkxdt());
            this->dkydt = graph::cache(dt_const*this->D.get_dkydt());
            this->dkzdt = graph::cache(dt_const*this->D.get_dkzdt());
            this->dxdt  = graph::cache(dt_const*this->D.get_dxdt());
            this->dydt  = graph::cache(dt_const*this->D.get_dydt());
            this->dzdt  = graph::cache(dt_const*this->D.get_dzdt());

            this->kx0_pred = kx + this->dkxdt;
            this->ky0_pred = ky + this->dkydt;
            this->kz0_pred = kz + this->dkzdt;
            this->x0_pred  = x  + this->dxdt;
            this->y0_pred  = y  + this->dydt;
            this->z0_pred  = z  + this->dzdt;

            const size_t size = x->evaluate().size();

            this->kx1_var = graph::variable<typename DISPERSION_FUNCTION::base> (size, "\tilde{k_{x}}");
            this->ky1_var = graph::variable<typename DISPERSION_FUNCTION::base> (size, "\tilde{k_{y}}");
            this->kz1_var = graph::variable<typename DISPERSION_FUNCTION::base> (size, "\tilde{k_{z}}");
            this->x1_var  = graph::variable<typename DISPERSION_FUNCTION::base> (size, "\tilde{x}");
            this->y1_var  = graph::variable<typename DISPERSION_FUNCTION::base> (size, "\tilde{y}");
            this->z1_var  = graph::variable<typename DISPERSION_FUNCTION::base> (size, "\tilde{z}");

            this->t_next = graph::cache(this->t + dt_const);

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D2(this->w,
                                                                     this->kx1_var,
                                                                     this->ky1_var,
                                                                     this->kz1_var,
                                                                     this->x1_var,
                                                                     this->y1_var,
                                                                     this->z1_var,
                                                                     graph::pseudo_variable(this->t_next),
                                                                     eq);

            this->residule_pred = D2.get_d()*D2.get_d();

            auto two = graph::two<typename DISPERSION_FUNCTION::base> ();

            this->kx_next = graph::cache(kx + (this->dkxdt + dt_const*D2.get_dkxdt())/two);
            this->ky_next = graph::cache(ky + (this->dkydt + dt_const*D2.get_dkydt())/two);
            this->kz_next = graph::cache(kz + (this->dkzdt + dt_const*D2.get_dkzdt())/two);
            this->x_next  = graph::cache(x  + (this->dxdt  + dt_const*D2.get_dxdt() )/two);
            this->y_next  = graph::cache(y  + (this->dydt  + dt_const*D2.get_dydt() )/two);
            this->z_next  = graph::cache(z  + (this->dzdt  + dt_const*D2.get_dzdt() )/two);
        }

//------------------------------------------------------------------------------
///  @brief Reset Cache.
//------------------------------------------------------------------------------
        virtual void reset_cache() final {
            this->t_next->reset_cache();

            this->dkxdt->reset_cache();
            this->dkydt->reset_cache();
            this->dkzdt->reset_cache();
            this->dxdt->reset_cache();
            this->dydt->reset_cache();
            this->dzdt->reset_cache();

            typename DISPERSION_FUNCTION::base kx_result = this->kx0_pred->evaluate();
            typename DISPERSION_FUNCTION::base ky_result = this->ky0_pred->evaluate();
            typename DISPERSION_FUNCTION::base kz_result = this->kz0_pred->evaluate();
            typename DISPERSION_FUNCTION::base x_result  = this->x0_pred->evaluate();
            typename DISPERSION_FUNCTION::base y_result  = this->y0_pred->evaluate();
            typename DISPERSION_FUNCTION::base z_result  = this->z0_pred->evaluate();

            this->kx1_var->set(kx_result);
            this->ky1_var->set(ky_result);
            this->kz1_var->set(kz_result);
            this->x1_var->set(x_result);
            this->y1_var->set(y_result);
            this->z1_var->set(z_result);

            this->kx_next->reset_cache();
            this->ky_next->reset_cache();
            this->kz_next->reset_cache();
            this->x_next->reset_cache();
            this->y_next->reset_cache();
            this->z_next->reset_cache();

            typename DISPERSION_FUNCTION::base max_residule =
                this->residule_pred->evaluate().max();
            typename DISPERSION_FUNCTION::base d_residule = 1000.0;

            while (std::abs(max_residule) > std::abs(this->tolarance) &&
                   std::real(d_residule) > 0) {
                kx_result = this->kx_next->evaluate();
                ky_result = this->ky_next->evaluate();
                kz_result = this->kz_next->evaluate();
                x_result  = this->x_next->evaluate();
                y_result  = this->y_next->evaluate();
                z_result  = this->z_next->evaluate();

                this->kx1_var->set(kx_result);
                this->ky1_var->set(ky_result);
                this->kz1_var->set(kz_result);
                this->x1_var->set(x_result);
                this->y1_var->set(y_result);
                this->z1_var->set(z_result);

                this->kx_next->reset_cache();
                this->ky_next->reset_cache();
                this->kz_next->reset_cache();
                this->x_next->reset_cache();
                this->y_next->reset_cache();
                this->z_next->reset_cache();

                typename DISPERSION_FUNCTION::base temp_residule =
                    this->residule_pred->evaluate().max();
                d_residule = std::abs(max_residule) - std::abs(temp_residule);
                max_residule = temp_residule;
            }
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

            this->t_next = graph::cache(this->t + dt_const);

            this->x1 = graph::cache(this->x + dt_const*this->D.get_dxdt()/two);
            this->y1 = graph::cache(this->y + dt_const*this->D.get_dydt()/two);
            this->z1 = graph::cache(this->z + dt_const*this->D.get_dzdt()/two);

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D2(this->w,
                                                                     graph::pseudo_variable(this->kx),
                                                                     graph::pseudo_variable(this->ky),
                                                                     graph::pseudo_variable(this->kz),
                                                                     graph::pseudo_variable(this->x1),
                                                                     graph::pseudo_variable(this->y1),
                                                                     graph::pseudo_variable(this->z1),
                                                                     graph::pseudo_variable(this->t),
                                                                     eq);

            this->kx_next = graph::cache(this->kx + dt_const*D2.get_dkxdt());
            this->ky_next = graph::cache(this->ky + dt_const*D2.get_dkydt());
            this->kz_next = graph::cache(this->kz + dt_const*D2.get_dkzdt());

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D3(this->w,
                                                                     graph::pseudo_variable(this->kx_next),
                                                                     graph::pseudo_variable(this->ky_next),
                                                                     graph::pseudo_variable(this->kz_next),
                                                                     graph::pseudo_variable(this->x1),
                                                                     graph::pseudo_variable(this->y1),
                                                                     graph::pseudo_variable(this->z1),
                                                                     graph::pseudo_variable(this->t),
                                                                     eq);

            this->x_next  = graph::cache(this->x1 + dt_const*D3.get_dxdt()/two);
            this->y_next  = graph::cache(this->y1 + dt_const*D3.get_dydt()/two);
            this->z_next  = graph::cache(this->z1 + dt_const*D3.get_dzdt()/two);
        }

//------------------------------------------------------------------------------
///  @brief Reset Cache.
//------------------------------------------------------------------------------
        virtual void reset_cache() final {
            this->t_next->reset_cache();

            this->x1->reset_cache();
            this->y1->reset_cache();
            this->z1->reset_cache();

            this->kx_next->reset_cache();
            this->ky_next->reset_cache();
            this->kz_next->reset_cache();

            this->kx->set(this->kx_next->evaluate());
            this->ky->set(this->ky_next->evaluate());
            this->kz->set(this->kz_next->evaluate());

            this->x_next->reset_cache();
            this->y_next->reset_cache();
            this->z_next->reset_cache();
        }
    };
}

#endif /* solver_h */
