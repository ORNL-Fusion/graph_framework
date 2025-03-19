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
///  @param[in] mean  Mean value.
///  @param[in] sigma Sigma value.
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
///  @param[in] mean  Mean value.
///  @param[in] sigma Sigma value.
//------------------------------------------------------------------------------
template<typename T>
std::normal_distribution<std::complex<T>> set_distribution(const std::complex<T> mean,
                                                           const std::complex<T> sigma) {
    return std::normal_distribution<T> (std::real(mean), std::real(sigma));
}

//------------------------------------------------------------------------------
///  @brief Initialize value.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in]     cl       Parsed commandline.
///  @param[in,out] var      Variable to set.
///  @param[in]     name     Variable name.
///  @param[in,out] engine   Random engine.
///  @param[in]     num_rays Numbers of rays.
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
///  @param[in]     cl       Parsed commandline.
///  @param[in,out] x        X variable to set.
///  @param[in,out] y        Y variable to set.
///  @param[in,out] engine   Random engine.
///  @param[in]     num_rays Numbers of rays.
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
                    const T r = static_cast<T> (phi_dist(engine));
                    const T phi = static_cast<T> (phi_dist(engine));
                    x->set(i, r*cos(phi));
                    y->set(i, r*sin(phi));
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
                    const T phi = static_cast<T> (phi_dist(engine));
                    x->set(i, radius_mean*cos(phi));
                    y->set(i, radius_mean*sin(phi));
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
///  @param[in]     cl        Parsed commandline.
///  @param[in]     omega     Wave frequency.
///  @param[in]     kx        Wave number in x direction.
///  @param[in]     ky        Wave number in y direction.
///  @param[in]     kz        Wave number in z direction.
///  @param[in]     x         Initial position in x direction.
///  @param[in]     y         Initial position in y direction.
///  @param[in]     z         Initial position in z direction.
///  @param[in]     t         Initial position in t direction.
///  @param[in]     dt        Initial dt.
///  @param[in]     eq        Equilibrium object.
///  @param[in]     num_steps Equilibrium object.
///  @param[in]     sub_steps Equilibrium object.
///  @param[in,out] engine    Random engine.
///  @param[in]     filename  Result filename, empty names will be blank.
///  @param[in]     num_rays  Number of rays to write.
///  @param[in]     index     Concurrent index.
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
                std::mt19937_64 &engine,
                const std::string &filename="",
                const size_t num_rays=0,
                const size_t index=0) {
    SOLVER_METHOD solve(omega, kx, ky, kz, x, y, z, t, dt, eq,
                        filename, num_rays, index);

    if (cl.is_option_set("init_kx") &&
        !cl.is_option_set("init_kx_dist")) {
        solve.init(kx);
    } else if (cl.is_option_set("init_ky") &&
               !cl.is_option_set("init_ky_dist")) {
        solve.init(ky);
    } else if (cl.is_option_set("init_kz") &&
               !cl.is_option_set("init_kz_dist")) {
        solve.init(kz);
    } else {
        solve.init();
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
        solve.print_residual();
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
    }
    solve.write_step();
}

//------------------------------------------------------------------------------
///  @brief Run Dispersion
///
///  @tparam DISPERSION_FUNCTION The dispersion method.
///
///  @param[in]     cl        Parsed commandline.
///  @param[in]     omega     Wave frequency.
///  @param[in]     kx        Wave number in x direction.
///  @param[in]     ky        Wave number in y direction.
///  @param[in]     kz        Wave number in z direction.
///  @param[in]     x         Initial position in x direction.
///  @param[in]     y         Initial position in y direction.
///  @param[in]     z         Initial position in z direction.
///  @param[in]     t         Initial position in t direction.
///  @param[in]     dt        Initial dt.
///  @param[in]     eq        Equilibrium object.
///  @param[in]     num_steps Equilibrium object.
///  @param[in]     sub_steps Equilibrium object.
///  @param[in,out] engine    Random engine.
///  @param[in]     filename  Result filename, empty names will be blank.
///  @param[in]     num_rays  Number of rays to write.
///  @param[in]     index     Concurrent index.
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
                    std::mt19937_64 &engine,
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
                                                                   engine,
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
                                                      engine,
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
                                                      engine,
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
                                                               engine,
                                                               filename,
                                                               num_rays,
                                                               index);
    }
}

//------------------------------------------------------------------------------
///  @brief  Make an equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] cl Parsed commandline.
//------------------------------------------------------------------------------
template<jit::float_scalar T, bool SAFE_MATH=false>
equilibrium::shared<T, SAFE_MATH> make_equilibrium(const commandline::parser &cl) {
    const std::string eq = cl.get_option_value<std::string> ("equilibrium");
    const std::string file_name = cl.get_option_value<std::string> ("equilibrium_file");

    if (eq == "efit") {
        return equilibrium::make_efit<T, SAFE_MATH> (file_name);
    } else if (eq == "mpex") {
        return equilibrium::make_mpex<T, SAFE_MATH> (file_name,
                                                     cl.get_option_value<T> ("mpex_ne_scale"),
                                                     cl.get_option_value<T> ("mpex_te_scale"),
                                                     cl.get_option_value<T> ("mpex_ps1_current"),
                                                     cl.get_option_value<T> ("mpex_ps2_current"),
                                                     cl.get_option_value<T> ("mpex_tr1_current"),
                                                     cl.get_option_value<T> ("mpex_tr2_current"));
    } else {
        return equilibrium::make_vmec<T, SAFE_MATH> (file_name);
    }
}

//------------------------------------------------------------------------------
///  @brief Generate the engine.
///
///  @param[in] cl    Parsed commandline.
///  @param[in] index Thread index.
//------------------------------------------------------------------------------
std::mt19937_64 make_engine(const commandline::parser &cl,
                            const size_t index) {
    if (cl.is_option_set("seed")) {
        return std::mt19937_64(index);
    }

    return std::mt19937_64((index + 1)*static_cast<uint64_t> (std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())));
}

//------------------------------------------------------------------------------
///  @brief Trace the rays.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] cl        Parsed commandline.
///  @param[in] num_times Total number of time steps.
///  @param[in] sub_steps Number of sub-steps to push the rays.
///  @param[in] num_rays  Number of rays to trace.
//------------------------------------------------------------------------------
template<jit::float_scalar T, bool SAFE_MATH=false>
void trace_ray(const commandline::parser &cl,
               const size_t num_times,
               const size_t sub_steps,
               const size_t num_rays) {
    const timing::measure_diagnostic total("Total Ray Time");

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

            std::mt19937_64 engine = make_engine(cl, thread_number);

            auto omega = graph::variable<T, SAFE_MATH> (local_num_rays, "\\omega");
            auto kx    = graph::variable<T, SAFE_MATH> (local_num_rays, "k_{x}");
            auto ky    = graph::variable<T, SAFE_MATH> (local_num_rays, "k_{y}");
            auto kz    = graph::variable<T, SAFE_MATH> (local_num_rays, "k_{z}");
            auto x     = graph::variable<T, SAFE_MATH> (local_num_rays, "x");
            auto y     = graph::variable<T, SAFE_MATH> (local_num_rays, "y");
            auto z     = graph::variable<T, SAFE_MATH> (local_num_rays, "z");
            auto t     = graph::variable<T, SAFE_MATH> (local_num_rays, "t");

            t->set(static_cast<T> (0.0));

//  Initial conditions.
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
                                                                  engine,
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
                                                                      engine,
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
                                                                         engine,
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
                                                                               engine,
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
                                                                       engine,
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
///  @tparam ABSORPTION_MODEL Absorption model to use.
///
///  @param[in] cl        Parsed commandline.
///  @param[in] kamp      Wave number amplitude.
///  @param[in] omega     Wave frequency.
///  @param[in] kx        Wave number in x direction.
///  @param[in] ky        Wave number in y direction.
///  @param[in] kz        Wave number in z direction.
///  @param[in] x         Initial position in x direction.
///  @param[in] y         Initial position in y direction.
///  @param[in] z         Initial position in z direction.
///  @param[in] t         Initial position in t direction.
///  @param[in] eq        Equilibrium object.
///  @param[in] num_steps Number of time steps.
///  @param[in] filename  Result filename, empty names will be blank.
///  @param[in] index     Concurrent index.
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
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] cl        Parsed commandline.
///  @param[in] num_times Total number of time steps.
///  @param[in] sub_steps Number of sub-steps to push the rays.
///  @param[in] num_rays  Number of rays to trace.
//------------------------------------------------------------------------------
template<jit::float_scalar T, bool SAFE_MATH=false>
void calculate_power(const commandline::parser &cl,
                     const size_t num_times,
                     const size_t sub_steps,
                     const size_t num_rays) {
    const timing::measure_diagnostic total("Power Time");

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
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] cl        Parsed commandline.
///  @param[in] num_times Total number of time steps.
///  @param[in] sub_steps Number of sub-steps to push the rays.
///  @param[in] num_rays  Number of rays to trace.
//------------------------------------------------------------------------------
template<jit::float_scalar T, bool SAFE_MATH=false>
void bin_power(const commandline::parser &cl,
               const size_t num_times,
               const size_t sub_steps,
               const size_t num_rays) {
    const timing::measure_diagnostic total("Power Time");

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
            auto p_next = graph::exp(-2.0*k_sum);
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
            }, graph::shared_random_state<T, SAFE_MATH> (), "power", local_num_rays);
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
///  @page xrays_commandline xrays Command Line Arguments
///  @brief Command Line Arguments for the xrays RF Ray tracing code.
///  @tableofcontents
///
///  @section xrays_commandline_intro Introduction
///  This page documents the commandline arguments or the RF ray tracing code
///  xrays. All arguments take the form of
///  @code
///  xrays [--options] [--options=with_value]
///  @endcode
///
///  @section xrays_commandline_args Command Options
///  <table>
///  <tr><th>Command                           <th>Values                                                  <th>Description
///  <tr><th colspan="3">General Options
///  <tr><td>@code --help @endcode             <td>                                                        <td>Display help text
///  <tr><td>@code --verbose @endcode          <td>                                                        <td>Show verbose output about kernel information.
///  <tr><td>@code --print_expressions @endcode<td>                                                        <td>Render ray equations as @f$\LaTeX@f$ expressions.
///  <tr><td>@code --print @endcode            <td>                                                        <td>Display a sample of ray progress to the screen.
///  <tr><td>@code --seed @endcode             <td>                                                        <td>Use a fixed random seed.
///  <tr><th colspan="3">Control Options
///  <tr><td>@code --num_times @endcode        <td>Positive Integer                                        <td>Total number of time steps to run.
///  <tr><td>@code --sub_steps @endcode        <td>Positive Integer                                        <td>Number of steps to run between outputs.
///  <tr><td>@code --num_rays @endcode         <td>Positive Integer                                        <td>Total number rays to run.
///  <tr><td>@code --endtime @endcode          <td>Positive Number                                         <td>Total time to trace the ray to.
///  <tr><th colspan="3">Ray Initialization Options
///  <tr><td>@code --init_w_dist @endcode      <td>
///                                                 * uniform
///                                                 * normal                                               <td>Distribution function for wave frequency.
///  <tr><td>@code --init_w_mean @endcode      <td>Positive Number                                         <td>Mean value for the wave frequency distribution function.
///  <tr><td>@code --init_w_sigma @endcode     <td>Positive Number                                         <td>Standard deviation of for the wave frequency distribution function.
///  <tr><td>@code --init_kx_dist @endcode     <td>
///                                                 * uniform
///                                                 * normal                                               <td>Distribution function for wave number in the x direction.
///  <tr><td>@code --init_kx @endcode          <td>                                                        <td>Solve for initial wave number in the x direction position.
///  <tr><td>@code --init_kx_mean @endcode     <td>Positive Number                                         <td>Mean value for the wave number in the x direction distribution function.
///  <tr><td>@code --init_kx_sigma @endcode    <td>Positive Number                                         <td>Standard deviation of for the wave number in the y direction distribution function.
///  <tr><td>@code --init_ky_dist @endcode     <td>
///                                                 * uniform
///                                                 * normal                                               <td>Distribution function for wave number in the y direction.
///  <tr><td>@code --init_ky @endcode          <td>                                                        <td>Solve for initial wave number in the y direction position.
///  <tr><td>@code --init_ky_mean @endcode     <td>Positive Number                                         <td>Mean value for the wave number in the y direction distribution function.
///  <tr><td>@code --init_ky_sigma @endcode    <td>Positive Number                                         <td>Standard deviation of for the wave number in the y direction distribution function.
///  <tr><td>@code --init_kz_dist @endcode     <td>
///                                                 * uniform
///                                                 * normal                                               <td>Distribution function for wave number in the z direction.
///  <tr><td>@code --init_kz @endcode          <td>                                                        <td>Solve for initial wave number in the z direction position.
///  <tr><td>@code --init_kz_mean @endcode     <td>Positive Number                                         <td>Mean value for the wave number in the z direction distribution function.
///  <tr><td>@code --init_kz_sigma @endcode    <td>Positive Number                                         <td>Standard deviation of for the wave number in the z direction distribution function.
///  <tr><td>@code --init_x_dist @endcode      <td>
///                                                 * uniform
///                                                 * normal                                               <td>Distribution function for ray x position.
///  <tr><td>@code --init_x_mean @endcode      <td>Positive Number                                         <td>Mean value for the ray x position distribution function.
///  <tr><td>@code --init_x_sigma @endcode     <td>Positive Number                                         <td>Standard deviation of for the ray x position distribution function.
///  <tr><td>@code --init_y_dist @endcode      <td>
///                                                 * uniform
///                                                 * normal                                               <td>Distribution function for ray y position.
///  <tr><td>@code --init_y_mean @endcode      <td>Positive Number                                         <td>Mean value for the ray y position distribution function.
///  <tr><td>@code --init_y_sigma @endcode     <td>Positive Number                                         <td>Standard deviation of for the ray y position distribution function.
///  <tr><td>@code --init_z_dist @endcode      <td>
///                                                 * uniform
///                                                 * normal                                               <td>Distribution function for ray z position.
///  <tr><td>@code --init_z_mean @endcode      <td>Positive Number                                         <td>Mean value for the ray z position distribution function.
///  <tr><td>@code --init_z_sigma @endcode     <td>Positive Number                                         <td>Standard deviation of for the ray z position distribution function.
///  <tr><td>@code --use_cyl_xy @endcode       <td>                                                        <td>Use cylindrical coordinates for x and y.
///  <tr><th colspan="3">Ray Tracing Physics Options
///  <tr><td>@code --equilibrium @endcode      <td>
///                                                 * @ref equilibrium_efit "efit"
///                                                 * @ref equilibrium_vmec "vmec"                         <td>Equilibrium to use.
///  <tr><td>@code --equilibrium_file @endcode <td>Path to @ref equilibrium_models "equilibrium file"      <td>Equilibrium file path.
///  <tr><td>@code --dispersion @endcode       <td>
///                                                 * @ref dispersion_function_simple "simple"
///                                                 * @ref dispersion_function_bohm_gross "bohm_gross"
///                                                 * @ref dispersion_function_o_wave "ordinary_wave"
///                                                 * @ref dispersion_function_x_wave "extra_ordinary_wave"
///                                                 * @ref dispersion_function_cold_plasma "code_plasma"   <td>Wave dispersion function to trace rays from.
///  <tr><td>@code --absorption_model @endcode <td>
///                                                 * @ref absorption_model_root "root_find"
///                                                 * @ref absorption_model_root "weak_damping"            <td>Power absorption model to use.
///  <tr><td>@code --solver @endcode           <td>
///                                                 * @ref solvers_split_simplextic "split_simplextic"
///                                                 * @ref solvers_rk2 "rk2"
///                                                 * @ref solvers_rk4 "rk4"
///                                                 * @ref solvers_adaptive_rk4 "adaptive_rk4"             <td>Method used to solve the equation.
///  </table>
///
///  <hr>
///  @section xrays_commandline_example Example commandline
///  Take the example command line
///  @code
///  ./graph_driver/xrays --absorption_model=weak_damping --dispersion=ordinary_wave --endtime=2.0 --equilibrium=efit --equilibrium_file=../graph_tests/efit.nc --init_kx --init_kx_mean=-700.0 --init_ky_dist=normal --init_ky_mean=-100.0 --init_ky_sigma=10.0 --init_kz_dist=normal --init_kz_mean=0.0 --init_kz_sigma=10.0 --init_w_dist=normal --init_w_mean=700 --init_w_sigma=10.0 --init_x_mean=2.5 --init_y_dist=normal --init_y_mean=0.0 --init_y_sigma=0.05 --init_z_dist=normal --init_z_mean=0.0 --init_z_sigma=0.05 --num_rays=100000 --num_times=100000 --solver=rk4 --sub_steps=100 --use_cyl_xy --verbose
///  @endcode
///  This example should be run from the build directory.
///
///  The options
///  @code
///  --num_rays=100000 --num_times=100000 --sub_steps=100 --verbose
///  @endcode
///  In this example, we will run a 100000 rays for 100000 steps and output
///  every 100th step. This is also the provides verbose output of the kernel
///  information.
///
///  @code
///  --endtime=2.0 --init_w_mean=700
///  @endcode
///  Time and frequency are input in
///  @ref dispersion_function_normal "modified units".
///  <table>
///  <caption id="xrays_commandline_example_normal_units">Conversion between input and real units.</caption>
///  <tr><th>Value        <th>Input Value<th>Real Unit
///  <tr><td>@f$\omega @f$<td>700        <td>@f$\frac{700}{c}@f$ @f$\frac{rad}{s}@f$
///  <tr><td>@f$t @f$     <td>2.0        <td>@f$2.0c @f$ @f$s @f$
///  </table>
///
///  @subsection xrays_commandline_example_dist Ray initialization.
///  @code
///  --init_x_mean=2.5 --init_y_dist=normal --init_y_mean=0.0 --init_y_sigma=0.05 --init_z_dist=normal --init_z_mean=0.0 --init_z_sigma=0.05 --use_cyl_xy
///  @endcode
///  Initial values for the position for @f$y @f$ and @f$z @f$ will be sampled
///  from a normal distribution function. Both values have a mean of zero and as
///  standard deviation of 0.05. @f$x @f$ is initialized with a uniform value of
///  2.5. We also set the command to use cylindrical coordinates. So
///  @f$xyz\rightarrow r\phi z @f$.
///
///  @code
///  --init_w_dist=normal --init_w_mean=700 --init_w_sigma=10.0
///  @endcode
///  Frequency uses a normal distribution with a mean of 700 and a standard
///  deviation of 10.
///
///  @code
///  --init_kx --init_kx_mean=-700.0 --init_ky_dist=normal --init_ky_mean=-100.0 --init_ky_sigma=10.0 --init_kz_dist=normal --init_kz_mean=0.0 --init_kz_sigma=10.0
///  @endcode
///  Initial values for @f$k_{y}@f$ and @f$k_{z}@f$ will be sampled from a normal
///  distribution function. Both of them have a standard deviation of 10.
///  @f$k_{y}@f$ has a mean of -100 and @f$k_{z}@f$ has zero mean.
///  @f$k_{x}@f$ uses the default uniform value of -700.0. However, it is
///  configured to solve for @f$k_{x}@f$ which satisfies the dispersion function
///  given the values of @f$\omega,k_{y},k_{z},\vec{x}@f$.
///
///  @subsection xrays_commandline_example_model Ray Models.
///  @code
///  --equilibrium=efit --equilibrium_file=../graph_tests/efit.nc
///  @endcode
///  We are using a @ref equilibrium_efit "EFIT" equilibrium with values
///  initialized from <tt>../graph_tests/efit.nc</tt>.
///
///  @code
///  --absorption_model=weak_damping --dispersion=ordinary_wave --solver=rk4
///  @endcode
///  It uses the @ref dispersion_function_o_wave "o-mode" dispersion function.
///  Rays are integrated using a @ref solvers_rk4 integrator. Power absorption
///  uses the @ref absorption_model_damping Model.
//------------------------------------------------------------------------------
///  @brief Setup and parse commandline options.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
commandline::parser parse_commandline(int argc, const char * argv[]) {
    commandline::parser cl(argv[0]);
    cl.add_option("verbose",           false, "Show verbose output about kernel information.");
    cl.add_option("num_times",         true,  "Total number of time steps to run.");
    cl.add_option("sub_steps",         true,  "Number of steps to run between outputs.");
    cl.add_option("num_rays",          true,  "Number of rays.");
    cl.add_option("endtime",           true,  "End time.");
    cl.add_option("print_expressions", false, "Render ray equations as LaTeX expressions.");
    cl.add_option("print",             false, "Print sample rays to screen.");
    cl.add_option("solver",            true,  "Method used to solve the equation.", {
        "split_simplextic",
        "rk2",
        "rk4",
        "adaptive_rk4"
    });
    cl.add_option("dispersion",        true,  "Wave dispersion function to trace rays from.", {
        "simple",
        "bohm_gross",
        "ordinary_wave",
        "extra_ordinary_wave",
        "cold_plasma"
    });
    cl.add_option("equilibrium",       true,  "Equilibrium to use.", {
        "efit",
        "vmec",
        "mpex"
    });
    cl.add_option("equilibrium_file",  true,  "File to read the equilibrium from.");
    cl.add_option("init_w_dist",       true,  "Distribution function for wave frequency.", {
        "uniform",
        "normal"
    });
    cl.add_option("mpex_ne_scale",     true,  "Scale factor for electron density profiles.");
    cl.add_option("mpex_te_scale",     true,  "Scale factor for electron temperature profiles.");
    cl.add_option("mpex_ps2_current",  true,  "Current for the ps2 coil set.");
    cl.add_option("mpex_ps1_current",  true,  "Current for the ps1 coil set.");
    cl.add_option("mpex_ps2_current",  true,  "Current for the ps2 coil set.");
    cl.add_option("mpex_tr1_current",  true,  "Current for the tr1 coil set.");
    cl.add_option("mpex_tr2_current",  true,  "Current for the tr2 coil set.");
    cl.add_option("init_w_mean",       true,  "Mean value for the wave frequency distribution function.");
    cl.add_option("init_w_sigma",      true,  "Standard deviation of for the wave frequency distribution function.");
    cl.add_option("init_kx_dist",      true,  "Distribution function for wave number in the x direction.", {
        "uniform",
        "normal"
    });
    cl.add_option("init_kx",           false, "Solve for initial wave number in the x direction position.");
    cl.add_option("init_kx_mean",      true,  "Mean value for the wave number in the x direction distribution function.");
    cl.add_option("init_kx_sigma",     true,  "Standard deviation of for the wave number in the x direction distribution function.");
    cl.add_option("init_ky_dist",      true,  "Distribution function for wave number in the y direction.", {
        "uniform",
        "normal"
    });
    cl.add_option("init_ky",           false, "Solve for initial wave number in the y direction position.");
    cl.add_option("init_ky_mean",      true,  "Mean value for the wave number in they direction distribution function.");
    cl.add_option("init_ky_sigma",     true,  "Standard deviation of for the wave number in the y direction distribution function.");
    cl.add_option("init_kz_dist",      true,  "Initial kz distribution.", {
        "uniform",
        "normal"
    });
    cl.add_option("init_kz",           false, "Distribution function for wave number in the z direction.");
    cl.add_option("init_kz_mean",      true,  "Solve for initial wave number in the z direction position.");
    cl.add_option("init_kz_sigma",     true,  "Standard deviation of for the wave number in the z direction distribution function.");
    cl.add_option("init_x_dist",       true,  "Distribution function for ray x position.", {
        "uniform",
        "normal"
    });
    cl.add_option("init_x_mean",       true,  "Mean value for the ray x position distribution function.");
    cl.add_option("init_x_sigma",      true,  "Standard deviation of for the ray x position distribution function.");
    cl.add_option("init_y_dist",       true,  "Distribution function for ray y position.", {
        "uniform",
        "normal"
    });
    cl.add_option("init_y_mean",       true,  "Mean value for the ray y position distribution function.");
    cl.add_option("init_y_sigma",      true,  "Standard deviation of for the ray y position distribution function.");
    cl.add_option("init_z_dist",       true,  "Distribution function for ray z position.", {
        "uniform",
        "normal"
    });
    cl.add_option("init_z_mean",       true,  "Mean value for the ray z position distribution function.");
    cl.add_option("init_z_sigma",      true,  "Standard deviation of for the ray z position distribution function.");
    cl.add_option("use_cyl_xy",        false, "Use cylindrical coordinates for x and y.");
    cl.add_option("absorption_model",  true,  "Power absorption model to use.", {
        "root_find",
        "weak_damping"
    });
    cl.add_option("seed",              false, "Fix the random seed.");

    cl.parse(argc, argv);

    return cl;
}

//------------------------------------------------------------------------------
///  @page xrays_output xrays Output File
///  @brief Result file format for the traced rays.
///  @tableofcontents
///
///  @section xrays_output_intro Introduction
///  The results of ray tracing are saved in several NetCDF files depending on
///  how many devices were found. The files have the name format of
///  <tt>result<i>n</i>.nc</tt>.
///
///  @section xrays_output_format File Format
///  The result file contains the following information. Note that to allow for
///  complex values, we add an extra dimension the end of 2D arrays. For real
///  values this dimension has size one while complex values have space for the
///  real and imaginary part.
///  <table>
///  <caption id="xrays_output_format_data">Result file quantities</caption>
///  <tr><th colspan="3">Dimensions
///  <tr><th colspan="2">Name                                          <th>Description
///  <tr><td colspan="2"><tt>time</tt>                                 <td>Size of time dimension.
///  <tr><td colspan="2"><tt>num_rays</tt>                             <td>Local number of rays for the device.
///  <tr><td colspan="2"><tt>ray_dim</tt>                              <td>Size of the dimension for quantities. (1 Real, 2 Complex)
///  <tr><td colspan="2"><tt>ray_dim_cplx</tt>                         <td>Size of the dimension for complex quantities (Real, Imagine).
///  <tr><th colspan="3">2D Quantities
///  <tr><th>Name             <th>Dimensions                           <th>Description
///  <tr><td><tt>d_power</tt> <td><tt>(time,num_rays,ray_dim)</tt>     <td>Change in power.
///  <tr><td><tt>kamp</tt>    <td><tt>(time,num_rays,ray_dim_cplx)</tt><td>@f$\sqrt{\vec{k}\cdot\vec{k}}@f$
///  <tr><td><tt>kx</tt>      <td><tt>(time,num_rays,ray_dim)</tt>     <td>Wave number in @f$\hat{x}@f$ direction.
///  <tr><td><tt>ky</tt>      <td><tt>(time,num_rays,ray_dim)</tt>     <td>Wave number in @f$\hat{y}@f$ direction.
///  <tr><td><tt>kz</tt>      <td><tt>(time,num_rays,ray_dim)</tt>     <td>Wave number in @f$\hat{z}@f$ direction.
///  <tr><td><tt>power</tt>   <td><tt>(time,num_rays,ray_dim)</tt>     <td>Wave power.
///  <tr><td><tt>residual</tt><td><tt>(time,num_rays,ray_dim)</tt>     <td>Dispersion function residual.
///  <tr><td><tt>time</tt>    <td><tt>(time,num_rays,ray_dim)</tt>     <td>Time
///  <tr><td><tt>w</tt>       <td><tt>(time,num_rays,ray_dim)</tt>     <td>Wave frequency.
///  <tr><td><tt>x</tt>       <td><tt>(time,num_rays,ray_dim)</tt>     <td>Position in @f$\hat{x}@f$ direction.
///  <tr><td><tt>y</tt>       <td><tt>(time,num_rays,ray_dim)</tt>     <td>Position in @f$\hat{y}@f$ direction.
///  <tr><td><tt>z</tt>       <td><tt>(time,num_rays,ray_dim)</tt>     <td>Position in @f$\hat{z}@f$ direction.
///  </table>
//------------------------------------------------------------------------------
///  @brief Main program of the driver.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    START_GPU
    (void)argc;
    (void)argv;
    const timing::measure_diagnostic total("Total Time");
    const commandline::parser cl = parse_commandline(argc, argv);

    jit::verbose = cl.is_option_set("verbose");

    const size_t num_times = cl.get_option_value<size_t> ("num_times");
    const size_t sub_steps = cl.get_option_value<size_t> ("sub_steps");
    const size_t num_rays = cl.get_option_value<size_t> ("num_rays");

    const bool use_safe_math = true;

    typedef double base;

    std::cout << "Using " << cl.get_option_value<std::string> ("equilibrium") << " equilibrium from " << cl.get_option_value<std::string> ("equilibrium_file") << std::endl;
    std::cout << "Using " << cl.get_option_value<std::string> ("dispersion") << " dispersion relation" << std::endl;
    std::cout << "Using " << cl.get_option_value<std::string> ("solver") <<  " solver methd" << std::endl;
    std::cout << "Using " << cl.get_option_value<std::string> ("absorption_model") << " absorption model" << std::endl << std::endl;

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
