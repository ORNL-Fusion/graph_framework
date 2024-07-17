//------------------------------------------------------------------------------
///  @file xrays.cpp
///  @brief Driver program for the rays library.
//------------------------------------------------------------------------------

#include <iostream>
#include <thread>
#include <random>

#include "../graph_framework/solver.hpp"
#include "../graph_framework/timing.hpp"
#include "../graph_framework/output.hpp"
#include "../graph_framework/absorption.hpp"
#include "../graph_framework/commandline_parser.hpp"

//------------------------------------------------------------------------------
///  @brief Set the normal distribution.
///
///  @tparam T Base type of the calculation.
///
///  @params[in] mean  Mean value.
///  @params[in] sigma Sigma value.
//------------------------------------------------------------------------------
template<typename T>
std::normal_distribution<T> set_distribution(const T mean,
                                             const T sigma) {
    return std::normal_distribution<T> (mean, sigma);
}

//------------------------------------------------------------------------------
///  @brief Set the normal distribution.
///
///  @tparam T Base type of the calculation.
///
///  @params[in] mean  Mean value.
///  @params[in] sigma Sigma value.
//------------------------------------------------------------------------------
template<typename T>
std::normal_distribution<std::complex<T>> set_distribution(const std::complex<T> mean,
                                                           const std::complex<T> sigma) {
    return std::normal_distribution<T> (std::real(mean), std::real(sigma));
}

//------------------------------------------------------------------------------
///  @brief Initalize value.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @params[in]     cl       Parsed commandline.
///  @params[in,out] var      Variable to set.
///  @params[in]     name     Variable name.
///  @params[in,out] engine   Random engine.
///  @params[in]     num_rays Numbers of rays.
//------------------------------------------------------------------------------
template<typename T, bool SAFE_MATH>
void set_variable(const commandline::parser &cl,
                  graph::shared_leaf<T, SAFE_MATH> var,
                  const std::string &name,
                  std::mt19937_64 &engine,
                  const size_t num_rays) {
    const T mean = cl.get_option_value<T> ("init_" + name + "_mean");
    const std::string dist_option = "init_" + name + "_dist";
    if (cl.is_option_set(dist_option) &&
        cl.get_option_value<std::string> (dist_option) == "normal") {
        const T sigma = cl.get_option_value<T> ("init_" + name + "_sigma");
        auto normal_dist = set_distribution(mean, sigma);
        for (size_t i = 0; i < num_rays; i++) {
            var->set(i, static_cast<T> (normal_dist(engine)));
        }
    } else {
        var->set(mean);
    }
}

//------------------------------------------------------------------------------
///  @brief Initialize the x and y direction.
///
///  @params[in]     cl       Parsed commandline.
///  @params[in,out] x        X variable to set.
///  @params[in,out] y        Y variable to set.
///  @params[in,out] engine   Random engine.
///  @params[in]     num_rays Numbers of rays.
//------------------------------------------------------------------------------
template<typename T, bool SAFE_MATH>
void set_xy_variables(const commandline::parser &cl,
                      graph::shared_leaf<T, SAFE_MATH> x,
                      graph::shared_leaf<T, SAFE_MATH> y,
                      std::mt19937_64 &engine,
                      const size_t num_rays) {
    if (cl.is_option_set("use_cyl_xy")) {
        const T radius_mean = cl.get_option_value<T> ("init_x_mean");
        const T phi_mean = cl.get_option_value<T> ("init_y_mean");

        if (cl.is_option_set("init_x_dist") &&
            cl.get_option_value<std::string> ("init_x_dist") == "normal") {
            const T radius_sigma = cl.get_option_value<T> ("init_x_sigma");
            auto radius_dist = set_distribution(radius_mean, radius_sigma);
            if (cl.is_option_set("init_y_dist") &&
                cl.get_option_value<std::string> ("init_y_dist") == "normal") {
                const T phi_sigma = cl.get_option_value<T> ("init_y_sigma");
                auto phi_dist = set_distribution(phi_mean, phi_sigma);
                for (size_t i = 0; i < num_rays; i++) {
                    x->set(i, static_cast<T> (radius_dist(engine))*cos(static_cast<T> (phi_dist(engine))));
                    y->set(i, static_cast<T> (radius_dist(engine))*sin(static_cast<T> (phi_dist(engine))));
                }
            } else {
                for (size_t i = 0; i < num_rays; i++) {
                    x->set(i, static_cast<T> (radius_dist(engine))*cos(phi_mean));
                    y->set(i, static_cast<T> (radius_dist(engine))*sin(phi_mean));
                }
            }
        } else {
            if (cl.is_option_set("init_y_dist") &&
                cl.get_option_value<std::string> ("init_y_dist") == "normal") {
                const T phi_sigma = cl.get_option_value<T> ("init_y_sigma");
                auto phi_dist = set_distribution(phi_mean, phi_sigma);
                for (size_t i = 0; i < num_rays; i++) {
                    x->set(i, radius_mean*cos(static_cast<T> (phi_dist(engine))));
                    y->set(i, radius_mean*sin(static_cast<T> (phi_dist(engine))));
                }
            } else {
                for (size_t i = 0; i < num_rays; i++) {
                    x->set(i, radius_mean*cos(phi_mean));
                    y->set(i, radius_mean*sin(phi_mean));
                }
            }
        }
    } else {
        set_variable(cl, x, "x", engine, num_rays);
        set_variable(cl, y, "y", engine, num_rays);
    }
}

//------------------------------------------------------------------------------
///  @brief Run Solver
///
///  @tparam SOLVER_METHOD The solver method.
///
///  @params[in] cl        Parsed commandline.
///  @params[in] omega     Wave frequency.
///  @params[in] kx        Wave number in x direction.
///  @params[in] ky        Wave number in y direction.
///  @params[in] kz        Wave number in z direction.
///  @params[in] x         Initial position in x direction.
///  @params[in] y         Initial position in y direction.
///  @params[in] z         Initial position in z direction.
///  @params[in] t         Initial position in t direction.
///  @params[in] dt        Inital dt.
///  @params[in] eq        Equilibrium object.
///  @params[in] num_steps Equilibrium object.
///  @params[in] sub_steps Equilibrium object.
///  @params[in] filename  Result filename, empty names will be blank.
///  @params[in] num_rays  Number of rays to write.
///  @params[in] index     Concurrent index.
//------------------------------------------------------------------------------
template<solver::method SOLVER_METHOD>
void run_solver(const commandline::parser &cl,
                graph::shared_leaf<typename SOLVER_METHOD::base,
                                   SOLVER_METHOD::safe_math> omega,
                graph::shared_leaf<typename SOLVER_METHOD::base,
                                   SOLVER_METHOD::safe_math> kx,
                graph::shared_leaf<typename SOLVER_METHOD::base,
                                   SOLVER_METHOD::safe_math> ky,
                graph::shared_leaf<typename SOLVER_METHOD::base,
                                   SOLVER_METHOD::safe_math> kz,
                graph::shared_leaf<typename SOLVER_METHOD::base,
                                   SOLVER_METHOD::safe_math> x,
                graph::shared_leaf<typename SOLVER_METHOD::base,
                                   SOLVER_METHOD::safe_math> y,
                graph::shared_leaf<typename SOLVER_METHOD::base,
                                   SOLVER_METHOD::safe_math> z,
                graph::shared_leaf<typename SOLVER_METHOD::base,
                                   SOLVER_METHOD::safe_math> t,
                graph::shared_leaf<typename SOLVER_METHOD::base,
                                   SOLVER_METHOD::safe_math> dt,
                equilibrium::shared<typename SOLVER_METHOD::base,
                                    SOLVER_METHOD::safe_math> &eq,
                const size_t num_steps,
                const size_t sub_steps,
                const std::string &filename="",
                const size_t num_rays=0,
                const size_t index=0) {
    SOLVER_METHOD solve(omega, kx, ky, kz, x, y, z, t, dt, eq,
                        filename, num_rays, index);

    if (!cl.is_option_set("init_kx_dist")) {
        solve.init(kx);
    } else if (!cl.is_option_set("init_ky_dist")) {
        solve.init(ky);
    } else {
        solve.init(kz);
    }
    solve.compile();

    if (index == 0 && cl.is_option_set("print_expressions")) {
        solve.print_dispersion();
        std::cout << std::endl;
        solve.print_dkxdt();
        std::cout << std::endl;
        solve.print_dkydt();
        std::cout << std::endl;
        solve.print_dkzdt();
        std::cout << std::endl;
        solve.print_dxdt();
        std::cout << std::endl;
        solve.print_dydt();
        std::cout << std::endl;
        solve.print_dzdt();
        std::cout << std::endl;
        solve.print_residule();
        std::cout << std::endl;
        solve.print_x_next();
        std::cout << std::endl;
        solve.print_y_next();
        std::cout << std::endl;
        solve.print_z_next();
        std::cout << std::endl;
        solve.print_kx_next();
        std::cout << std::endl;
        solve.print_ky_next();
        std::cout << std::endl;
        solve.print_kz_next();
        std::cout << std::endl;
    }

#ifndef STATIC
    std::mt19937_64 engine((index + 1)*static_cast<uint64_t> (std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())));
#else
    std::mt19937_64 engine(index + 1);
#endif
    std::uniform_int_distribution<size_t> int_dist(0, num_rays - 1);
    
    const size_t sample = int_dist(engine);

    if (index == 0) {
        std::cout << "Omega " << omega->evaluate().at(sample) << std::endl;
    }

    const bool print = cl.is_option_set("print");
    for (size_t j = 0; j < num_steps; j++) {
        if (index == 0 && print) {
            solve.print(sample);
        }
        solve.write_step();
        for (size_t k = 0; k < sub_steps; k++) {
            solve.step();
        }
    }

    if (index == 0 && print) {
        solve.print(sample);
    } else {
        solve.write_step();
    }
}

//------------------------------------------------------------------------------
///  @brief Run Disperison
///
///  @tparam DISPERSION_FUNCTION The dispersion method.
///
///  @params[in] cl        Parsed commandline.
///  @params[in] omega     Wave frequency.
///  @params[in] kx        Wave number in x direction.
///  @params[in] ky        Wave number in y direction.
///  @params[in] kz        Wave number in z direction.
///  @params[in] x         Initial position in x direction.
///  @params[in] y         Initial position in y direction.
///  @params[in] z         Initial position in z direction.
///  @params[in] t         Initial position in t direction.
///  @params[in] dt        Inital dt.
///  @params[in] eq        Equilibrium object.
///  @params[in] num_steps Equilibrium object.
///  @params[in] sub_steps Equilibrium object.
///  @params[in] filename  Result filename, empty names will be blank.
///  @params[in] num_rays  Number of rays to write.
///  @params[in] index     Concurrent index.
//------------------------------------------------------------------------------
template<dispersion::function DISPERSION_FUNCTION>
void run_dispersion(const commandline::parser &cl,
                    graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                       DISPERSION_FUNCTION::safe_math> omega,
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
                    const size_t num_steps,
                    const size_t sub_steps,
                    const std::string &filename="",
                    const size_t num_rays=0,
                    const size_t index=0) {
    const std::string solver_method = cl.get_option_value<std::string> ("solver");
    if (solver_method == "split_simplextic") {
        auto dt_const = graph::constant(static_cast<typename DISPERSION_FUNCTION::base> (dt));
        run_solver<solver::split_simplextic<DISPERSION_FUNCTION>> (cl, omega,
                                                                   kx, ky, kz,
                                                                   x, y, z,
                                                                   t, dt_const, eq,
                                                                   num_steps,
                                                                   sub_steps,
                                                                   filename,
                                                                   num_rays,
                                                                   index);
    } else if (solver_method == "rk2") {
        auto dt_const = graph::constant(static_cast<typename DISPERSION_FUNCTION::base> (dt));
        run_solver<solver::rk2<DISPERSION_FUNCTION>> (cl, omega,
                                                      kx, ky, kz,
                                                      x, y, z,
                                                      t, dt_const, eq,
                                                      num_steps,
                                                      sub_steps,
                                                      filename,
                                                      num_rays,
                                                      index);
    } else if (solver_method == "rk4") {
        auto dt_const = graph::constant(static_cast<typename DISPERSION_FUNCTION::base> (dt));
        run_solver<solver::rk4<DISPERSION_FUNCTION>> (cl, omega,
                                                      kx, ky, kz,
                                                      x, y, z,
                                                      t, dt_const, eq,
                                                      num_steps,
                                                      sub_steps,
                                                      filename,
                                                      num_rays,
                                                      index);
    } else {
        auto dt_var = graph::variable(num_rays,
                                      static_cast<typename DISPERSION_FUNCTION::base> (dt),
                                      "dt");
        run_solver<solver::adaptive_rk4<DISPERSION_FUNCTION>> (cl, omega,
                                                               kx, ky, kz,
                                                               x, y, z,
                                                               t, dt_var, eq,
                                                               num_steps,
                                                               sub_steps,
                                                               filename,
                                                               num_rays,
                                                               index);
    }
}

//------------------------------------------------------------------------------
///  @brief  Make an equilibrum.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @params[in] cl Parsed commandline.
//------------------------------------------------------------------------------
template<jit::float_scalar T, bool SAFE_MATH=false>
equilibrium::shared<T, SAFE_MATH> make_equilibrium(const commandline::parser &cl) {
    const std::string eq = cl.get_option_value<std::string> ("equilibrium");
    const std::string file_name = cl.get_option_value<std::string> ("equilibrium_file");

    if (eq == "efit") {
        return equilibrium::make_efit<T, SAFE_MATH> (file_name);
    } else {
        return equilibrium::make_vmec<T, SAFE_MATH> (file_name);
    }
}

//------------------------------------------------------------------------------
///  @brief Trace the rays.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @params[in] cl        Parsed commandline.
///  @params[in] num_times Total number of time steps.
///  @params[in] sub_steps Number of substeps to push the rays.
///  @params[in] num_rays  Number of rays to trace.
//------------------------------------------------------------------------------
template<jit::float_scalar T, bool SAFE_MATH=false>
void trace_ray(const commandline::parser &cl,
               const size_t num_times,
               const size_t sub_steps,
               const size_t num_rays) {
    const timeing::measure_diagnostic total("Total Ray Time");

    std::vector<std::thread> threads(std::max(std::min(static_cast<unsigned int> (jit::context<T, SAFE_MATH>::max_concurrency()),
                                                       static_cast<unsigned int> (num_rays)),
                                              static_cast<unsigned int> (1)));

    const size_t batch = num_rays/threads.size();
    const size_t extra = num_rays%threads.size();

    for (size_t i = 0, ie = threads.size(); i < ie; i++) {
        threads[i] = std::thread([&cl, num_times, sub_steps, 
                                  batch, extra] (const size_t thread_number) -> void {

            const size_t num_steps = num_times/sub_steps;
            const size_t local_num_rays = batch
                                        + (extra > thread_number ? 1 : 0);

#ifndef STATIC
            std::mt19937_64 engine((thread_number + 1)*static_cast<uint64_t> (std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())));
#else
            std::mt19937_64 engine(thread_number + 1);
#endif
            std::uniform_int_distribution<size_t> int_dist(0, local_num_rays - 1);

            auto omega = graph::variable<T, SAFE_MATH> (local_num_rays, "\\omega");
            auto kx    = graph::variable<T, SAFE_MATH> (local_num_rays, "k_{x}");
            auto ky    = graph::variable<T, SAFE_MATH> (local_num_rays, "k_{y}");
            auto kz    = graph::variable<T, SAFE_MATH> (local_num_rays, "k_{z}");
            auto x     = graph::variable<T, SAFE_MATH> (local_num_rays, "x");
            auto y     = graph::variable<T, SAFE_MATH> (local_num_rays, "y");
            auto z     = graph::variable<T, SAFE_MATH> (local_num_rays, "z");
            auto t     = graph::variable<T, SAFE_MATH> (local_num_rays, "t");

            t->set(static_cast<T> (0.0));

//  Inital conditions.
            set_variable(cl, omega, "w", engine, local_num_rays);
            set_variable(cl, kx, "kx", engine, local_num_rays);
            set_variable(cl, ky, "ky", engine, local_num_rays);
            set_variable(cl, kz, "kz", engine, local_num_rays);
            set_variable(cl, z, "z", engine, local_num_rays);
            set_xy_variables(cl, x, y, engine, local_num_rays);

            auto eq = make_equilibrium<T, SAFE_MATH> (cl);

            const T endtime = cl.get_option_value<T> ("endtime");
            const T dt = endtime/static_cast<T> (num_times);

            std::ostringstream stream;
            stream << "result" << thread_number << ".nc";

            const std::string dispersion = 
                cl.get_option_value<std::string> ("dispersion");
            if (dispersion == "simple") {
                run_dispersion<dispersion::simple<T, SAFE_MATH>> (cl, omega,
                                                                  kx, ky, kz,
                                                                  x, y, z,
                                                                  t, dt, eq,
                                                                  num_steps,
                                                                  sub_steps,
                                                                  stream.str(),
                                                                  local_num_rays,
                                                                  thread_number);
            } else if (dispersion == "bohm_gross") {
                run_dispersion<dispersion::bohm_gross<T, SAFE_MATH>> (cl, omega,
                                                                      kx, ky, kz,
                                                                      x, y, z,
                                                                      t, dt, eq,
                                                                      num_steps,
                                                                      sub_steps,
                                                                      stream.str(),
                                                                      local_num_rays,
                                                                      thread_number);
            } else if (dispersion == "ordinary_wave") {
                run_dispersion<dispersion::ordinary_wave<T, SAFE_MATH>> (cl, omega,
                                                                         kx, ky, kz,
                                                                         x, y, z,
                                                                         t, dt, eq,
                                                                         num_steps,
                                                                         sub_steps,
                                                                         stream.str(),
                                                                         local_num_rays,
                                                                         thread_number);
            } else if (dispersion == "extra_ordinary_wave") {
                run_dispersion<dispersion::extra_ordinary_wave<T, SAFE_MATH>> (cl, omega,
                                                                               kx, ky, kz,
                                                                               x, y, z,
                                                                               t, dt, eq,
                                                                               num_steps,
                                                                               sub_steps,
                                                                               stream.str(),
                                                                               local_num_rays,
                                                                               thread_number);
            } else {
                run_dispersion<dispersion::cold_plasma<T, SAFE_MATH>> (cl, omega,
                                                                       kx, ky, kz,
                                                                       x, y, z,
                                                                       t, dt, eq,
                                                                       num_steps,
                                                                       sub_steps,
                                                                       stream.str(),
                                                                       local_num_rays,
                                                                       thread_number);
            }
        }, i);
    }

    for (std::thread &t : threads) {
        t.join();
    }

    total.print();
}

//------------------------------------------------------------------------------
///  @brief Run absorption model.
///
///  @tparam ABSORPTION_MODEL Absoption model to use.
///
///  @params[in] kamp      Wave number amplitude.
///  @params[in] kx        Wave number in x direction.
///  @params[in] ky        Wave number in y direction.
///  @params[in] kz        Wave number in z direction.
///  @params[in] x         Initial position in x direction.
///  @params[in] y         Initial position in y direction.
///  @params[in] z         Initial position in z direction.
///  @params[in] t         Initial position in t direction.
///  @params[in] eq        Equilibrium object.
///  @params[in] num_steps Number of time steps.
///  @params[in] filename  Result filename, empty names will be blank.
///  @params[in] index     Concurrent index.
//------------------------------------------------------------------------------
template<absorption::model ABSORPTION_MODEL>
void run_absorption(const commandline::parser &cl,
                    graph::shared_leaf<typename ABSORPTION_MODEL::base,
                                       ABSORPTION_MODEL::safe_math> kamp,
                    graph::shared_leaf<typename ABSORPTION_MODEL::base,
                                       ABSORPTION_MODEL::safe_math> omega,
                    graph::shared_leaf<typename ABSORPTION_MODEL::base,
                                       ABSORPTION_MODEL::safe_math> kx,
                    graph::shared_leaf<typename ABSORPTION_MODEL::base,
                                       ABSORPTION_MODEL::safe_math> ky,
                    graph::shared_leaf<typename ABSORPTION_MODEL::base,
                                       ABSORPTION_MODEL::safe_math> kz,
                    graph::shared_leaf<typename ABSORPTION_MODEL::base,
                                       ABSORPTION_MODEL::safe_math> x,
                    graph::shared_leaf<typename ABSORPTION_MODEL::base,
                                       ABSORPTION_MODEL::safe_math> y,
                    graph::shared_leaf<typename ABSORPTION_MODEL::base,
                                       ABSORPTION_MODEL::safe_math> z,
                    graph::shared_leaf<typename ABSORPTION_MODEL::base,
                                       ABSORPTION_MODEL::safe_math> t,
                    equilibrium::shared<typename ABSORPTION_MODEL::base,
                                        ABSORPTION_MODEL::safe_math> &eq,
                    const size_t num_steps,
                    const std::string &filename="",
                    const size_t index=0) {
    ABSORPTION_MODEL power(kamp, omega,
                           kx, ky, kz,
                           x, y, z, t,
                           eq, filename, index);
    power.compile();

    for (size_t j = 0, je = num_steps + 1; j < je; j++) {
        power.run(j);
    }
}

//------------------------------------------------------------------------------
///  @brief Calculate absorption.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @params[in] cl        Parsed commandline.
///  @params[in] num_times Total number of time steps.
///  @params[in] sub_steps Number of substeps to push the rays.
///  @params[in] num_rays  Number of rays to trace.
//------------------------------------------------------------------------------
template<jit::float_scalar T, bool SAFE_MATH=false>
void calculate_power(const commandline::parser &cl,
                     const size_t num_times,
                     const size_t sub_steps,
                     const size_t num_rays) {
    const timeing::measure_diagnostic total("Power Time");

    std::vector<std::thread> threads(std::max(std::min(static_cast<unsigned int> (jit::context<T, SAFE_MATH>::max_concurrency()),
                                                       static_cast<unsigned int> (num_rays)),
                                              static_cast<unsigned int> (1)));

    const size_t batch = num_rays/threads.size();
    const size_t extra = num_rays%threads.size();

    for (size_t i = 0, ie = threads.size(); i < ie; i++) {
        threads[i] = std::thread([&cl, num_times, sub_steps, 
                                  batch, extra] (const size_t thread_number) -> void {
            std::ostringstream stream;
            stream << "result" << thread_number << ".nc";

            const size_t num_steps = num_times/sub_steps;
            const size_t local_num_rays = batch
                                        + (extra > thread_number ? 1 : 0);

            auto omega = graph::variable<T, SAFE_MATH> (local_num_rays, "\\omega");
            auto kx    = graph::variable<T, SAFE_MATH> (local_num_rays, "k_{x}");
            auto ky    = graph::variable<T, SAFE_MATH> (local_num_rays, "k_{y}");
            auto kz    = graph::variable<T, SAFE_MATH> (local_num_rays, "k_{z}");
            auto x     = graph::variable<T, SAFE_MATH> (local_num_rays, "x");
            auto y     = graph::variable<T, SAFE_MATH> (local_num_rays, "y");
            auto z     = graph::variable<T, SAFE_MATH> (local_num_rays, "z");
            auto t     = graph::variable<T, SAFE_MATH> (local_num_rays, "t");
            auto kamp  = graph::variable<T, SAFE_MATH> (local_num_rays, "kamp");

            auto eq = make_equilibrium<T, SAFE_MATH> (cl);

            const std::string absorption_model = cl.get_option_value<std::string> ("absorption_model");
            if (absorption_model == "root_find") {
                run_absorption<absorption::root_finder<T, SAFE_MATH>> (cl, kamp,
                                                                       omega,
                                                                       kx, ky, kz,
                                                                       x, y, z, t,
                                                                       eq, num_steps,
                                                                       stream.str(),
                                                                       thread_number);
            } else {
                run_absorption<absorption::weak_damping<T, SAFE_MATH>> (cl, kamp,
                                                                        omega,
                                                                        kx, ky, kz,
                                                                        x, y, z, t,
                                                                        eq, num_steps,
                                                                        stream.str(),
                                                                        thread_number);
            }
        }, i);
    }

    for (std::thread &t : threads) {
        t.join();
    }

    total.print();
}

//------------------------------------------------------------------------------
///  @brief Bin power.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @params[in] cl        Parsed commandline.
///  @params[in] num_times Total number of time steps.
///  @params[in] sub_steps Number of substeps to push the rays.
///  @params[in] num_rays  Number of rays to trace.
//------------------------------------------------------------------------------
template<jit::float_scalar T, bool SAFE_MATH=false>
void bin_power(const commandline::parser &cl,
               const size_t num_times,
               const size_t sub_steps,
               const size_t num_rays) {
    const timeing::measure_diagnostic total("Power Time");

    std::vector<std::thread> threads(std::max(std::min(static_cast<unsigned int> (jit::context<T, SAFE_MATH>::max_concurrency()),
                                                       static_cast<unsigned int> (num_rays)),
                                              static_cast<unsigned int> (1)));

    const size_t batch = num_rays/threads.size();
    const size_t extra = num_rays%threads.size();

    for (size_t i = 0, ie = threads.size(); i < ie; i++) {
        threads[i] = std::thread([&cl, num_times, sub_steps,
                                  batch, extra] (const size_t thread_number) -> void {
            std::ostringstream stream;
            stream << "result" << thread_number << ".nc";

            const size_t num_steps = num_times/sub_steps;
            const size_t local_num_rays = batch
                                        + (extra > thread_number ? 1 : 0);

            auto x          = graph::variable<T, SAFE_MATH> (local_num_rays, "x");
            auto y          = graph::variable<T, SAFE_MATH> (local_num_rays, "y");
            auto z          = graph::variable<T, SAFE_MATH> (local_num_rays, "z");
            auto x_last     = graph::variable<T, SAFE_MATH> (local_num_rays, "x_last");
            auto y_last     = graph::variable<T, SAFE_MATH> (local_num_rays, "y_last");
            auto z_last     = graph::variable<T, SAFE_MATH> (local_num_rays, "z_last");
            auto kamp       = graph::variable<T, SAFE_MATH> (local_num_rays, "kamp");
            auto power      = graph::variable<T, SAFE_MATH> (local_num_rays, static_cast<T> (1.0), "power");
            auto k_sum      = graph::variable<T, SAFE_MATH> (local_num_rays, static_cast<T> (0.0), "k_sum");

            auto eq = make_equilibrium<T, SAFE_MATH> (cl);

            auto x_real = eq->get_x(x, y, z);
            auto y_real = eq->get_y(x, y, z);
            auto z_real = eq->get_z(x, y, z);
            auto x_real_last = eq->get_x(x_last, y_last, z_last);
            auto y_real_last = eq->get_y(x_last, y_last, z_last);
            auto z_real_last = eq->get_z(x_last, y_last, z_last);

            auto dlvec = graph::vector(x_real - x_real_last,
                                       y_real - y_real_last,
                                       z_real - z_real_last);
            auto dl = dlvec->length();
            auto kdl = kamp*dl;
            auto k_next = kdl + k_sum;
            auto p_next = graph::exp(graph::none<T, SAFE_MATH> ()*graph::two<T, SAFE_MATH> ()*k_sum);
            auto d_power = p_next - power;
            d_power = graph::sqrt(d_power*d_power);

            workflow::manager<T, SAFE_MATH> work(thread_number);
            work.add_item({
                graph::variable_cast(x),
                graph::variable_cast(y),
                graph::variable_cast(z),
                graph::variable_cast(x_last),
                graph::variable_cast(y_last),
                graph::variable_cast(z_last),
                graph::variable_cast(kamp),
                graph::variable_cast(power),
                graph::variable_cast(k_sum),
            }, {d_power}, {
                {x, graph::variable_cast(x_last)},
                {y, graph::variable_cast(y_last)},
                {z, graph::variable_cast(z_last)},
                {p_next, graph::variable_cast(power)},
                {k_next, graph::variable_cast(k_sum)}
            }, "power");
            work.compile();

            output::result_file file(stream.str());
            output::data_set<T> dataset(file);

            dataset.create_variable(file, "power", power, work.get_context());
            dataset.create_variable(file, "d_power", d_power, work.get_context());

            dataset.reference_variable(file, "x", graph::variable_cast(x));
            dataset.reference_variable(file, "y", graph::variable_cast(y));
            dataset.reference_variable(file, "z", graph::variable_cast(z));
            dataset.reference_imag_variable(file, "kamp", graph::variable_cast(kamp));
            file.end_define_mode();

            dataset.read(file, 0);
            x_last->set(x->evaluate());
            y_last->set(y->evaluate());
            z_last->set(z->evaluate());
            dataset.write(file, 0);

            std::thread sync([]{});
            work.copy_to_device(x_last, graph::variable_cast(x_last)->data());
            work.copy_to_device(y_last, graph::variable_cast(y_last)->data());
            work.copy_to_device(z_last, graph::variable_cast(z_last)->data());
            for (size_t j = 1, je = num_steps + 1; j < je; j++) {
                dataset.read(file, j);
                work.copy_to_device(x,    graph::variable_cast(x)->data());
                work.copy_to_device(y,    graph::variable_cast(y)->data());
                work.copy_to_device(z,    graph::variable_cast(z)->data());
                work.copy_to_device(kamp, graph::variable_cast(kamp)->data());

                work.run();

                sync.join();
                work.wait();
                sync = std::thread([&file, &dataset] (const size_t k) -> void {
                    dataset.write(file, k);
                }, j);
            }

            sync.join();
        }, i);
    }

    for (std::thread &t : threads) {
        t.join();
    }

    total.print();
}

//------------------------------------------------------------------------------
///  @brief Setup and parse commandline options.
///
///  @params[in] argc Number of commandline arguments.
///  @params[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
commandline::parser parse_commandline(int argc, const char * argv[]) {
    commandline::parser cl(argv[0]);
    cl.add_option("verbose",           false, "Show verbose output.");
    cl.add_option("num_times",         true,  "Number of times.");
    cl.add_option("sub_steps",         true,  "Number of substeps.");
    cl.add_option("num_rays",          true,  "Number of rays.");
    cl.add_option("endtime",           true,  "End time.");
    cl.add_option("print_expressions", false, "Print out rays expressions.");
    cl.add_option("print",             false, "Print sample rays to screen.");
    cl.add_option("solver",            true,  "Solver method.");
    cl.add_option_values("solver", {
        "split_simplextic",
        "rk2",
        "rk4",
        "adaptive_rk4"
    });
    cl.add_option("dispersion",        true,  "Disperison method.");
    cl.add_option_values("dispersion", {
        "simple",
        "bohm_gross",
        "ordinary_wave",
        "extra_ordinary_wave",
        "cold_plasma"
    });
    cl.add_option("equilibrium",       true,  "Equilibrium to use.");
    cl.add_option_values("equilibrium", {
        "efit",
        "vmec"
    });
    cl.add_option("equilibrium_file",  true,  "File to read the equilibrum from.");
    cl.add_option("init_w_dist",       true,  "Inital omega distribution.");
    cl.add_option_values("init_w_dist", {
        "uniform",
        "normal"
    });
    cl.add_option("init_w_mean",       true,  "Inital omega mean");
    cl.add_option("init_w_sigma",      true,  "Inital omega sigma");
    cl.add_option("init_kx_dist",      true,  "Inital kx distribution.");
    cl.add_option_values("init_kx_dist", {
        "uniform",
        "normal"
    });
    cl.add_option("init_kx_mean",      true,  "Inital kx mean");
    cl.add_option("init_kx_sigma",     true,  "Inital kx sigma");
    cl.add_option("init_ky_dist",      true,  "Inital ky distribution.");
    cl.add_option_values("init_ky_dist", {
        "uniform",
        "normal"
    });
    cl.add_option("init_ky_mean",      true,  "Inital ky mean");
    cl.add_option("init_ky_sigma",     true,  "Inital ky sigma");
    cl.add_option("init_kz_dist",      true,  "Inital kz distribution.");
    cl.add_option_values("init_kz_dist", {
        "uniform",
        "normal"
    });
    cl.add_option("init_kz_mean",      true,  "Inital kz mean");
    cl.add_option("init_kz_sigma",     true,  "Inital kz sigma");
    cl.add_option("init_x_dist",       true,  "Inital x distribution.");
    cl.add_option_values("init_x_dist", {
        "uniform",
        "normal"
    });
    cl.add_option("init_x_mean",       true,  "Inital x mean");
    cl.add_option("init_x_sigma",      true,  "Inital x sigma");
    cl.add_option("init_y_dist",       true,  "Inital y distribution.");
    cl.add_option_values("init_y_dist", {
        "uniform",
        "normal"
    });
    cl.add_option("init_y_mean",       true,  "Inital y mean");
    cl.add_option("init_y_sigma",      true,  "Inital y sigma");
    cl.add_option("init_z_dist",       true,  "Inital z distribution.");
    cl.add_option_values("init_z_dist", {
        "uniform",
        "normal"
    });
    cl.add_option("init_z_mean",       true,  "Inital z mean");
    cl.add_option("init_z_sigma",      true,  "Inital z sigma");
    cl.add_option("use_cyl_xy",        false, "Use cylindical coordinates for x and y.");
    cl.add_option("absorption_model",  true,  "Power absoption model to use.");
    cl.add_option_values("absorption_model", {
        "root_find",
        "weak_damping"
    });

    cl.parse(argc, argv);

    return cl;
}

//------------------------------------------------------------------------------
///  @brief Main program of the driver.
///
///  @params[in] argc Number of commandline arguments.
///  @params[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    START_GPU
    (void)argc;
    (void)argv;
    const timeing::measure_diagnostic total("Total Time");
    const commandline::parser cl = parse_commandline(argc, argv);

    jit::verbose = cl.is_option_set("verbose");

    const size_t num_times = cl.get_option_value<size_t> ("num_times");
    const size_t sub_steps = cl.get_option_value<size_t> ("sub_steps");
#ifndef STATIC
    const size_t num_rays = cl.get_option_value<size_t> ("num_rays");
#else
    const size_t num_rays = 1;
#endif

    const bool use_safe_math = true;

    typedef double base;

    trace_ray<base> (cl, num_times, sub_steps, num_rays);
    calculate_power<std::complex<base>, use_safe_math> (cl,
                                                        num_times,
                                                        sub_steps,
                                                        num_rays);
    bin_power<base> (cl, num_times, sub_steps, num_rays);

    std::cout << std::endl << "Timing:" << std::endl;
    total.print();

    END_GPU
}
