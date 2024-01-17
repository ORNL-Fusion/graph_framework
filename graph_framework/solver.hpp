//------------------------------------------------------------------------------
///  @file solver.hpp
///  @brief Base class for a ode solvers.
///
///  Defines a ode solver.
//------------------------------------------------------------------------------

#ifndef solver_h
#define solver_h

#include <thread>

#include "dispersion.hpp"
#include "output.hpp"

namespace solver {
//******************************************************************************
//  Solver interface.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class interface the solver.
///
///  @tparam DISPERSION_FUNCTION Class of dispersion function to use.
//------------------------------------------------------------------------------
    template<dispersion::function DISPERSION_FUNCTION>
    class solver_interface {
    protected:
///  w variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> w;
///  kx variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> kx;
///  ky variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> ky;
///  kz variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> kz;
///  x variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> x;
///  y variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> y;
///  z variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> z;
///  t variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> t;

///  Dispersion function interface.
        dispersion::dispersion_interface<DISPERSION_FUNCTION> D;

///  Next kx value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> kx_next;
///  Next ky value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> ky_next;
///  Next kz value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> kz_next;
///  Next x value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> x_next;
///  Next y value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> y_next;
///  Next z value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> z_next;
///  Next t value.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> t_next;

///  Residule.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> residule;

///  Workflow manager.
        workflow::manager<typename DISPERSION_FUNCTION::base,
                          DISPERSION_FUNCTION::safe_math> work;
///  Concurrent index.
        const size_t index;

///  Output file.
        output::result_file file;
///  Output dataset.
        output::data_set<typename DISPERSION_FUNCTION::base> dataset;

///  Async thread to write data files.
        std::thread sync;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a new solver_interface with inital conditions.
///
///  @params[in] w        Inital w.
///  @params[in] kx       Inital kx.
///  @params[in] ky       Inital ky.
///  @params[in] kz       Inital kz.
///  @params[in] x        Inital x.
///  @params[in] y        Inital y.
///  @params[in] z        Inital z.
///  @params[in] t        Inital t.
///  @params[in] eq       The plasma equilibrium.
///  @params[in] filename Result filename, empty names will be blank.
///  @params[in] num_rays Number of rays to write.
///  @params[in] index    Concurrent index.
//------------------------------------------------------------------------------
        solver_interface(graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                            DISPERSION_FUNCTION::safe_math> w,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                            DISPERSION_FUNCTION::safe_math> kx,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                            DISPERSION_FUNCTION::safe_math> ky,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                            DISPERSION_FUNCTION::safe_math> kz,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                            DISPERSION_FUNCTION::safe_math> x,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                            DISPERSION_FUNCTION::safe_math> y,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                            DISPERSION_FUNCTION::safe_math> z,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                            DISPERSION_FUNCTION::safe_math> t,
                         equilibrium::shared<typename DISPERSION_FUNCTION::base,
                                             DISPERSION_FUNCTION::safe_math> &eq,
                         const std::string &filename="",
                         const size_t num_rays=0,
                         const size_t index=0) :
        D(w, kx, ky, kz, x, y, z, t, eq), w(w),
        kx(kx), ky(ky), kz(kz), x(x), y(y), z(z), t(t),
        file(filename, num_rays), dataset(file),
        index(index), work(index), sync([]{}) {}

//------------------------------------------------------------------------------
///  @brief Destructor.
//------------------------------------------------------------------------------
        ~solver_interface() {
            sync.join();
        }
        
//------------------------------------------------------------------------------
///  @brief Method to initalize the rays.
///
///  @params[in,out] x              Variable reference to update.
///  @params[in]     tolarance      Tolarance to solve to dispersion function
///                                 to.
///  @params[in]     max_iterations Maximum number of iterations to run.
///  @returns The residule graph.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                   DISPERSION_FUNCTION::safe_math>
        init(graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                DISPERSION_FUNCTION::safe_math> x,
             const typename DISPERSION_FUNCTION::base tolarance = 1.0E-30,
             const size_t max_iterations = 1000) final {
            graph::input_nodes<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> inputs {
                graph::variable_cast(this->t),
                graph::variable_cast(this->w),
                graph::variable_cast(this->x),
                graph::variable_cast(this->y),
                graph::variable_cast(this->z),
                graph::variable_cast(this->kx),
                graph::variable_cast(this->ky),
                graph::variable_cast(this->kz)
            };

            residule = this->D.solve(x, inputs, index,
                                     tolarance, max_iterations);

            return residule;
        }

//------------------------------------------------------------------------------
///  @brief Compile the solver function.
//------------------------------------------------------------------------------
        virtual void compile() {
            graph::input_nodes<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> inputs = {
                graph::variable_cast(this->t),
                graph::variable_cast(this->w),
                graph::variable_cast(this->x),
                graph::variable_cast(this->y),
                graph::variable_cast(this->z),
                graph::variable_cast(this->kx),
                graph::variable_cast(this->ky),
                graph::variable_cast(this->kz)
            };

            graph::output_nodes<typename DISPERSION_FUNCTION::base,
                                DISPERSION_FUNCTION::safe_math> outputs = {
                this->residule
            };

            graph::map_nodes<typename DISPERSION_FUNCTION::base,
                             DISPERSION_FUNCTION::safe_math> setters = {
                {this->kx_next, graph::variable_cast(this->kx)},
                {this->ky_next, graph::variable_cast(this->ky)},
                {this->kz_next, graph::variable_cast(this->kz)},
                {this->x_next,  graph::variable_cast(this->x)},
                {this->y_next,  graph::variable_cast(this->y)},
                {this->z_next,  graph::variable_cast(this->z)},
                {this->t_next,  graph::variable_cast(this->t)}
            };

            work.add_item(inputs, outputs, setters, "solver_kernel");
            work.compile();

            dataset.create_variable(file, "time",     this->t, work.get_context());
            dataset.create_variable(file, "residule", residule, work.get_context());
            dataset.create_variable(file, "w",        this->w, work.get_context());
            dataset.create_variable(file, "x",        this->x, work.get_context());
            dataset.create_variable(file, "y",        this->y, work.get_context());
            dataset.create_variable(file, "z",        this->z, work.get_context());
            dataset.create_variable(file, "kx",       this->kx, work.get_context());
            dataset.create_variable(file, "ky",       this->ky, work.get_context());
            dataset.create_variable(file, "kz",       this->kz, work.get_context());

            file.end_define_mode();
        }

//------------------------------------------------------------------------------
///  @brief Syncronize results from host to gpu.
//------------------------------------------------------------------------------
        void sync_device() {
            work.copy_to_device(this->t,  graph::variable_cast(this->t)->data());
            work.copy_to_device(this->w,  graph::variable_cast(this->w)->data());
            work.copy_to_device(this->x,  graph::variable_cast(this->x)->data());
            work.copy_to_device(this->y,  graph::variable_cast(this->y)->data());
            work.copy_to_device(this->z,  graph::variable_cast(this->z)->data());
            work.copy_to_device(this->kx, graph::variable_cast(this->kx)->data());
            work.copy_to_device(this->ky, graph::variable_cast(this->ky)->data());
            work.copy_to_device(this->kz, graph::variable_cast(this->kz)->data());
        }

//------------------------------------------------------------------------------
///  @brief Syncronize results from gpu to host.
//------------------------------------------------------------------------------
        void sync_host() {
            work.copy_to_host(this->t,  graph::variable_cast(this->t)->data());
            work.copy_to_host(this->w,  graph::variable_cast(this->w)->data());
            work.copy_to_host(this->x,  graph::variable_cast(this->x)->data());
            work.copy_to_host(this->y,  graph::variable_cast(this->y)->data());
            work.copy_to_host(this->z,  graph::variable_cast(this->z)->data());
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
///  @brief Check the residule.
///
///  @params[in] index Ray index to check residule for.
///  @returns The value of the residule at the index.
//------------------------------------------------------------------------------
        typename DISPERSION_FUNCTION::base check_residule(const size_t index) {
            return work.check_value(index, this->residule);
        }

//------------------------------------------------------------------------------
///  @brief Print out the results.
///
///  @params[in] index Ray index to print results of.
//------------------------------------------------------------------------------
        void print(const size_t index) {
            work.print(index, {
                this->t,
                this->residule,
                this->w,
                this->x,
                this->y,
                this->z,
                this->kx,
                this->ky,
                this->kz
            });
        }

//------------------------------------------------------------------------------
///  @brief Write result step.
//------------------------------------------------------------------------------
        void write_step() {
            sync.join();
            work.wait();
            sync = std::thread([this] {
                dataset.write(file);
            });
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

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the residule.
//------------------------------------------------------------------------------
        void print_residule() {
            residule->to_latex();
            std::cout << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the x\_next.
//------------------------------------------------------------------------------
        void print_x_next() {
            x_next->to_latex();
            std::cout << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the y\_next.
//------------------------------------------------------------------------------
        void print_y_next() {
            y_next->to_latex();
            std::cout << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the z\_next.
//------------------------------------------------------------------------------
        void print_z_next() {
            z_next->to_latex();
            std::cout << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the kx\_next.
//------------------------------------------------------------------------------
        void print_kx_next() {
            kx_next->to_latex();
            std::cout << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the ky\_next.
//------------------------------------------------------------------------------
        void print_ky_next() {
            ky_next->to_latex();
            std::cout << std::endl;
        }

//------------------------------------------------------------------------------
///  @brief Print out the latex expression for the kz\_next.
//------------------------------------------------------------------------------
        void print_kz_next() {
            kz_next->to_latex();
            std::cout << std::endl;
        }
///  Type def to retrieve the dispersion function.
        typedef DISPERSION_FUNCTION dispersion_function;
///  Type def to retrieve the backend base type.
        typedef typename DISPERSION_FUNCTION::base base;
    };

///  Solver method concept.
    template<class S>
    concept method = std::is_base_of<solver_interface<typename S::dispersion_function>, S>::value;

//******************************************************************************
//  Second Order Runge Kutta.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Second Order Runge Kutta class.
///
///  @tparam DISPERSION_FUNCTION Class of dispersion function to use.
//------------------------------------------------------------------------------
    template<dispersion::function DISPERSION_FUNCTION>
    class rk2 : public solver_interface<DISPERSION_FUNCTION> {
    protected:
///  kx1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> kx1;
///  ky1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> ky1;
///  kz1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> kz1;
///  x1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> x1;
///  y1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> y1;
///  z1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> z1;

///  kx2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> kx2;
///  ky2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> ky2;
///  kz2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> kz2;
///  x2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> x2;
///  y2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> y2;
///  z2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> z2;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a new second order runge kutta solver.
///
///  @params[in] w        Inital omega.
///  @params[in] kx       Inital kx.
///  @params[in] ky       Inital ky.
///  @params[in] kz       Inital kz.
///  @params[in] x        Inital x.
///  @params[in] y        Inital y.
///  @params[in] z        Inital z.
///  @params[in] t        Inital t.
///  @params[in] dt       Inital dt.
///  @params[in] filename Result filename, empty names will be blank.
///  @params[in] num_rays Number of rays to write.
///  @params[in] index    Concurrent index.
//------------------------------------------------------------------------------
        rk2(graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> w,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> kx,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> ky,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> kz,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> x,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> y,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> z,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> t,
            const typename DISPERSION_FUNCTION::base dt,
            equilibrium::shared<typename DISPERSION_FUNCTION::base,
                                DISPERSION_FUNCTION::safe_math> &eq,
            const std::string &filename="",
            const size_t num_rays=0,
            const size_t index=0) :
        solver_interface<DISPERSION_FUNCTION> (w, kx, ky, kz, x, y, z, t, eq,
                                               filename, num_rays, index) {
            auto dt_const = graph::constant<typename DISPERSION_FUNCTION::base,
                                            DISPERSION_FUNCTION::safe_math> (static_cast<typename DISPERSION_FUNCTION::base> (dt));

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

            auto two = graph::two<typename DISPERSION_FUNCTION::base,
                                  DISPERSION_FUNCTION::safe_math> ();

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
///
///  @tparam DISPERSION_FUNCTION Class of dispersion function to use.
//------------------------------------------------------------------------------
    template<dispersion::function DISPERSION_FUNCTION>
    class rk4 : public solver_interface<DISPERSION_FUNCTION> {
    protected:
///  kx1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> kx1;
///  ky1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> ky1;
///  kz1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> kz1;
///  x1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> x1;
///  y1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> y1;
///  z1 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> z1;

///  kx2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                        DISPERSION_FUNCTION::safe_math> kx2;
///  ky2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> ky2;
///  kz2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> kz2;
///  x2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> x2;
///  y2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> y2;
///  z2 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> z2;

///  kx3 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> kx3;
///  ky3 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> ky3;
///  kz3 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> kz3;
///  x3 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> x3;
///  y3 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> y3;
///  z3 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> z3;

///  kx4 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> kx4;
///  ky4 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> ky4;
///  kz4 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> kz4;
///  x4 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> x4;
///  y4 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> y4;
///  z4 subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> z4;

///  t  subexpression.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> t_sub;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a new second order runge kutta solver.
///
///  @params[in] w        Inital omega.
///  @params[in] kx       Inital kx.
///  @params[in] ky       Inital ky.
///  @params[in] kz       Inital kz.
///  @params[in] x        Inital x.
///  @params[in] y        Inital y.
///  @params[in] z        Inital z.
///  @params[in] t        Inital t.
///  @params[in] dt       Inital dt.
///  @params[in] filename Result filename, empty names will be blank.
///  @params[in] num_rays Number of rays to write.
///  @params[in] index    Concurrent index.
//------------------------------------------------------------------------------
        rk4(graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> w,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> kx,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> ky,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> kz,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> x,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> y,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> z,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> t,
            const typename DISPERSION_FUNCTION::base dt,
            equilibrium::shared<typename DISPERSION_FUNCTION::base,
                                DISPERSION_FUNCTION::safe_math> &eq,
            const std::string &filename="",
            const size_t num_rays=0,
            const size_t index=0) :
        rk4(w, kx, ky, kz, x, y, z, t,
            graph::constant<typename DISPERSION_FUNCTION::base,
                            DISPERSION_FUNCTION::safe_math> (static_cast<typename DISPERSION_FUNCTION::base> (dt)), eq,
            filename, num_rays, index) {}

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
///  @params[in] filename Result filename, empty names will be blank.
///  @params[in] num_rays Number of rays to write.
///  @params[in] index    Concurrent index.
//------------------------------------------------------------------------------
        rk4(graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> w,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> kx,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> ky,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> kz,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> x,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> y,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> z,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> t,
            graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> dt,
            equilibrium::shared<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> &eq,
            const std::string &filename="",
            const size_t num_rays=0,
            const size_t index=0) :
        solver_interface<DISPERSION_FUNCTION> (w, kx, ky, kz, x, y, z, t, eq,
                                               filename, num_rays, index) {
            this->kx1 = dt*this->D.get_dkxdt();
            this->ky1 = dt*this->D.get_dkydt();
            this->kz1 = dt*this->D.get_dkzdt();
            this->x1  = dt*this->D.get_dxdt();
            this->y1  = dt*this->D.get_dydt();
            this->z1  = dt*this->D.get_dzdt();

            auto two = graph::two<typename DISPERSION_FUNCTION::base,
                                  DISPERSION_FUNCTION::safe_math> ();

            this->t_sub = this->t + dt/two;

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D2(this->w,
                                                                     graph::pseudo_variable(this->kx + kx1/two),
                                                                     graph::pseudo_variable(this->ky + ky1/two),
                                                                     graph::pseudo_variable(this->kz + kz1/two),
                                                                     graph::pseudo_variable(this->x  +  x1/two),
                                                                     graph::pseudo_variable(this->y  +  y1/two),
                                                                     graph::pseudo_variable(this->z  +  z1/two),
                                                                     graph::pseudo_variable(this->t_sub),
                                                                     eq);

            this->kx2 = dt*D2.get_dkxdt();
            this->ky2 = dt*D2.get_dkydt();
            this->kz2 = dt*D2.get_dkzdt();
            this->x2  = dt*D2.get_dxdt();
            this->y2  = dt*D2.get_dydt();
            this->z2  = dt*D2.get_dzdt();

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D3(this->w,
                                                                     graph::pseudo_variable(this->kx + kx2/two),
                                                                     graph::pseudo_variable(this->ky + ky2/two),
                                                                     graph::pseudo_variable(this->kz + kz2/two),
                                                                     graph::pseudo_variable(this->x  +  x2/two),
                                                                     graph::pseudo_variable(this->y  +  y2/two),
                                                                     graph::pseudo_variable(this->z  +  z2/two),
                                                                     graph::pseudo_variable(this->t_sub),
                                                                     eq);

            this->kx3 = dt*D3.get_dkxdt();
            this->ky3 = dt*D3.get_dkydt();
            this->kz3 = dt*D3.get_dkzdt();
            this->x3  = dt*D3.get_dxdt();
            this->y3  = dt*D3.get_dydt();
            this->z3  = dt*D3.get_dzdt();

            this->t_next = this->t + dt;

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D4(this->w,
                                                                     graph::pseudo_variable(this->kx + kx3),
                                                                     graph::pseudo_variable(this->ky + ky3),
                                                                     graph::pseudo_variable(this->kz + kz3),
                                                                     graph::pseudo_variable(this->x  + x3),
                                                                     graph::pseudo_variable(this->y  + y3),
                                                                     graph::pseudo_variable(this->z  + z3),
                                                                     graph::pseudo_variable(this->t_next),
                                                                     eq);

            this->kx4 = dt*D4.get_dkxdt();
            this->ky4 = dt*D4.get_dkydt();
            this->kz4 = dt*D4.get_dkzdt();
            this->x4  = dt*D4.get_dxdt();
            this->y4  = dt*D4.get_dydt();
            this->z4  = dt*D4.get_dzdt();

            auto six = graph::constant<typename DISPERSION_FUNCTION::base,
                                       DISPERSION_FUNCTION::safe_math>(static_cast<typename DISPERSION_FUNCTION::base> (6.0));

            this->kx_next = this->kx + (this->kx1 + two*(this->kx2 + this->kx3) + this->kx4)/six;
            this->ky_next = this->ky + (this->ky1 + two*(this->ky2 + this->ky3) + this->ky4)/six;
            this->kz_next = this->kz + (this->kz1 + two*(this->kz2 + this->kz3) + this->kz4)/six;
            this->x_next  = this->x  + (this->x1  + two*(this->x2  + this->x3 ) + this->x4 )/six;
            this->y_next  = this->y  + (this->y1  + two*(this->y2  + this->y3 ) + this->y4 )/six;
            this->z_next  = this->z  + (this->z1  + two*(this->z2  + this->z3 ) + this->z4 )/six;
        }
    };

//******************************************************************************
//  Adaptive timestep Fourth Order Runge Kutta.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Adaptive Fourth Order Runge Kutta class.
///
///  @tparam DISPERSION_FUNCTION Class of dispersion function to use.
//------------------------------------------------------------------------------
    template<dispersion::function DISPERSION_FUNCTION>
    class adaptive_rk4 : public rk4<DISPERSION_FUNCTION> {
    protected:
///  Dispersion residule.
        dispersion::dispersion_interface<DISPERSION_FUNCTION> D;
///  Time step variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> dt_var;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a new second order runge kutta solver.
///
///  @params[in] w        Inital omega.
///  @params[in] kx       Inital kx.
///  @params[in] ky       Inital ky.
///  @params[in] kz       Inital kz.
///  @params[in] x        Inital x.
///  @params[in] y        Inital y.
///  @params[in] z        Inital z.
///  @params[in] t        Inital t.
///  @params[in] dt       Inital dt.
///  @params[in] filename Result filename, empty names will be blank.
///  @params[in] num_rays Number of rays to write.
///  @params[in] index    Concurrent index.
//------------------------------------------------------------------------------
        adaptive_rk4(graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                        DISPERSION_FUNCTION::safe_math> w,
                     graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                        DISPERSION_FUNCTION::safe_math> kx,
                     graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                        DISPERSION_FUNCTION::safe_math> ky,
                     graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                        DISPERSION_FUNCTION::safe_math> kz,
                     graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                        DISPERSION_FUNCTION::safe_math> x,
                     graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                        DISPERSION_FUNCTION::safe_math> y,
                     graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                        DISPERSION_FUNCTION::safe_math> z,
                     graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                        DISPERSION_FUNCTION::safe_math> t,
                     graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                        DISPERSION_FUNCTION::safe_math> dt,
                     equilibrium::shared<typename DISPERSION_FUNCTION::base,
                                         DISPERSION_FUNCTION::safe_math> &eq,
                     const std::string &filename="",
                     const size_t num_rays=0,
                     const size_t index=0) :
        rk4<DISPERSION_FUNCTION> (w, kx, ky, kz, x, y, z, t, dt, eq,
                                  filename, num_rays, index),
        D(w,
          graph::pseudo_variable(this->kx_next),
          graph::pseudo_variable(this->ky_next),
          graph::pseudo_variable(this->kz_next),
          graph::pseudo_variable(this->x_next),
          graph::pseudo_variable(this->y_next),
          graph::pseudo_variable(this->z_next),
          graph::pseudo_variable(this->t_next),
          eq),
        dt_var(dt) {}

//------------------------------------------------------------------------------
///  @brief Compile the solver function.
//------------------------------------------------------------------------------
        virtual void compile() final {
            auto var = graph::variable_cast(dt_var);
            auto lambda = graph::variable(var->size(), static_cast<typename DISPERSION_FUNCTION::base> (1.0), "\\lambda");
            auto loss = graph::one<typename DISPERSION_FUNCTION::base,
                                   DISPERSION_FUNCTION::safe_math> ()/dt_var + lambda*D.get_d()*D.get_d();

            graph::input_nodes<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> inputs = {
                graph::variable_cast(this->t),
                graph::variable_cast(this->w),
                graph::variable_cast(this->x),
                graph::variable_cast(this->y),
                graph::variable_cast(this->z),
                graph::variable_cast(this->kx),
                graph::variable_cast(this->ky),
                graph::variable_cast(this->kz),
                var,
                graph::variable_cast(lambda)
            };

            solver::newton(this->work, {var, graph::variable_cast(lambda)},
                           inputs, loss);

            inputs = {
                graph::variable_cast(this->t),
                graph::variable_cast(this->w),
                graph::variable_cast(this->x),
                graph::variable_cast(this->y),
                graph::variable_cast(this->z),
                graph::variable_cast(this->kx),
                graph::variable_cast(this->ky),
                graph::variable_cast(this->kz),
                var
            };

            graph::output_nodes<typename DISPERSION_FUNCTION::base,
                                DISPERSION_FUNCTION::safe_math> outputs = {
                this->residule
            };

            graph::map_nodes<typename DISPERSION_FUNCTION::base,
                             DISPERSION_FUNCTION::safe_math> setters = {
                {this->kx_next, graph::variable_cast(this->kx)},
                {this->ky_next, graph::variable_cast(this->ky)},
                {this->kz_next, graph::variable_cast(this->kz)},
                {this->x_next, graph::variable_cast(this->x)},
                {this->y_next, graph::variable_cast(this->y)},
                {this->z_next, graph::variable_cast(this->z)},
                {this->t_next, graph::variable_cast(this->t)}
            };

            this->work.add_item(inputs, outputs, setters, "solver_kernel");
            this->work.compile();
        }
    };

//******************************************************************************
//  Split simplextic integrator
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Predictor corrector that trys to minimize the disperison residule.
///
///  @tparam DISPERSION_FUNCTION Class of dispersion function to use.
//------------------------------------------------------------------------------
    template<dispersion::function DISPERSION_FUNCTION>
    class split_simplextic : public solver_interface<DISPERSION_FUNCTION> {
    protected:
///  Half step x
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> x1;
///  Half step y
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> y1;
///  Half step z
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> z1;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a split simplextic integrator.
///
///  @params[in] w        Inital omega.
///  @params[in] kx       Inital kx.
///  @params[in] ky       Inital ky.
///  @params[in] kz       Inital kz.
///  @params[in] x        Inital x.
///  @params[in] y        Inital y.
///  @params[in] z        Inital z.
///  @params[in] t        Inital t.
///  @params[in] dt       Inital dt.
///  @params[in] filename Result filename, empty names will be blank.
///  @params[in] num_rays Number of rays to write.
///  @params[in] index    Concurrent index.
//------------------------------------------------------------------------------
        split_simplextic(graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                            DISPERSION_FUNCTION::safe_math> w,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                            DISPERSION_FUNCTION::safe_math> kx,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                            DISPERSION_FUNCTION::safe_math> ky,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                            DISPERSION_FUNCTION::safe_math> kz,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                            DISPERSION_FUNCTION::safe_math> x,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                            DISPERSION_FUNCTION::safe_math> y,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                            DISPERSION_FUNCTION::safe_math> z,
                         graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                            DISPERSION_FUNCTION::safe_math> t,
                         const typename DISPERSION_FUNCTION::base dt,
                         equilibrium::shared<typename DISPERSION_FUNCTION::base,
                                             DISPERSION_FUNCTION::safe_math> &eq,
                         const std::string &filename="",
                         const size_t num_rays=0,
                         const size_t index=0) :
        solver_interface<DISPERSION_FUNCTION> (w, kx, ky, kz, x, y, z, t, eq,
                                               filename, num_rays, index) {

//  Test if the function is separatable.
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

            auto dt_const = graph::constant<typename DISPERSION_FUNCTION::base,
                                            DISPERSION_FUNCTION::safe_math> (static_cast<typename DISPERSION_FUNCTION::base> (dt));
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
