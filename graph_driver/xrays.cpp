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

const bool print = false;
const bool write_step = true;
const bool print_expressions = false;
const bool verbose = true;

//------------------------------------------------------------------------------
///  @brief Trace the rays.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @params[in] num_times Total number of time steps.
///  @params[in] sub_steps Number of substeps to push the rays.
///  @params[in] num_rays  Number of rays to trace.
//------------------------------------------------------------------------------
template<typename T, bool SAFE_MATH=false>
void trace_ray(const size_t num_times,
               const size_t sub_steps,
               const size_t num_rays) {

    std::vector<std::thread> threads(std::max(std::min(static_cast<unsigned int> (jit::context<T, SAFE_MATH>::max_concurrency()),
                                                       static_cast<unsigned int> (num_rays)),
                                              static_cast<unsigned int> (1)));

    const size_t batch = num_rays/threads.size();
    const size_t extra = num_rays%threads.size();

    for (size_t i = 0, ie = threads.size(); i < ie; i++) {
        threads[i] = std::thread([num_times, sub_steps, num_rays, batch, extra] (const size_t thread_number,
                                                                                 const size_t num_threads) -> void {

            const size_t num_steps = num_times/sub_steps;
            const size_t local_num_rays = batch
                                        + (extra > thread_number ? 1 : 0);

            std::mt19937_64 engine((thread_number + 1)*static_cast<uint64_t> (std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())));
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
            if constexpr (jit::is_float<T> ()) {
                std::normal_distribution<float> norm_dist1(static_cast<float> (700.0),
                                                           static_cast<float> (10.0));
                std::normal_distribution<float> norm_dist2(static_cast<float> (0.0),
                                                           static_cast<float> (0.05));
                std::normal_distribution<float> norm_dist3(static_cast<float> (-100.0),
                                                           static_cast<float> (10.0));
                std::normal_distribution<float> norm_dist4(static_cast<float> (0.0),
                                                           static_cast<float> (10.0));

                for (size_t j = 0; j < local_num_rays; j++) {
                    omega->set(j, static_cast<T> (norm_dist1(engine)));
                    x->set(j, static_cast<T> (2.5*cos(norm_dist2(engine)/2.5)));
                    y->set(j, static_cast<T> (2.5*sin(norm_dist2(engine)/2.5)));
                    z->set(j, static_cast<T> (norm_dist2(engine)));
                    ky->set(j, static_cast<T> (norm_dist3(engine)));
                    kz->set(j, static_cast<T> (norm_dist4(engine)));
                }
            } else {
                std::normal_distribution<double> norm_dist1(static_cast<double> (700.0),
                                                            static_cast<double> (10.0));
                std::normal_distribution<double> norm_dist2(static_cast<double> (0.0),
                                                            static_cast<double> (0.05));
                std::normal_distribution<double> norm_dist3(static_cast<double> (-100.0),
                                                            static_cast<double> (10.0));
                std::normal_distribution<double> norm_dist4(static_cast<double> (0.0),
                                                            static_cast<double> (10.0));

                for (size_t j = 0; j < local_num_rays; j++) {
                    omega->set(j, static_cast<T> (norm_dist1(engine)));
                    x->set(j, static_cast<T> (2.5*cos(norm_dist2(engine)/2.5)));
                    y->set(j, static_cast<T> (2.5*sin(norm_dist2(engine)/2.5)));
                    z->set(j, static_cast<T> (norm_dist2(engine)));
                    ky->set(j, static_cast<T> (norm_dist3(engine)));
                    kz->set(j, static_cast<T> (norm_dist4(engine)));
                }
            }
            kx->set(static_cast<T> (-700));

            auto eq = equilibrium::make_efit<T, SAFE_MATH> (NC_FILE);
            //auto eq = equilibrium::make_slab_density<T, SAFE_MATH> ();
            //auto eq = equilibrium::make_slab_field<T, SAFE_MATH> ();
            //auto eq = equilibrium::make_no_magnetic_field<T, SAFE_MATH> ();

            const T endtime = static_cast<T> (2.0);
            const T dt = endtime/static_cast<T> (num_times);

            //auto dt_var = graph::variable(num_rays, static_cast<T> (dt), "dt");

            std::ostringstream stream;
            stream << "result" << thread_number << ".nc";

            //solver::split_simplextic<dispersion::bohm_gross<T, SAFE_MATH>>
            //solver::rk4<dispersion::bohm_gross<T, SAFE_MATH>>
            //solver::adaptive_rk4<dispersion::bohm_gross<T, SAFE_MATH>>
            //solver::rk4<dispersion::simple<T, SAFE_MATH>>
            solver::rk4<dispersion::ordinary_wave<T, SAFE_MATH>>
            //solver::rk4<dispersion::extra_ordinary_wave<T, SAFE_MATH>>
            //solver::rk4<dispersion::cold_plasma<T, SAFE_MATH>>
            //solver::adaptive_rk4<dispersion::ordinary_wave<T, SAFE_MATH>>
            //solver::rk4<dispersion::hot_plasma<T, dispersion::z_erfi<T, SAFE_MATH>, use_safe_math>>
            //solver::rk4<dispersion::hot_plasma_expandion<T, dispersion::z_erfi<T, SAFE_MATH>, use_safe_math>>
                solve(omega, kx, ky, kz, x, y, z, t, dt, eq,
                      stream.str(), local_num_rays, thread_number);
            solve.init(kx);
            solve.compile();
            if (thread_number == 0 && print_expressions) {
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

            const size_t sample = int_dist(engine);

            if (thread_number == 0) {
                std::cout << "Omega " << omega->evaluate().at(sample) << std::endl;
            }

            for (size_t j = 0; j < num_steps; j++) {
                if (thread_number == 0 && print) {
                    solve.print(sample);
                }
                if (write_step) {
                    solve.write_step();
                }
                for (size_t k = 0; k < sub_steps; k++) {
                    solve.step();
                }
            }

            if (thread_number == 0 && print) {
                solve.print(sample);
            } else if (write_step) {
                solve.write_step();
            } else {
                solve.sync_host();
            }

        }, i, threads.size());
    }

    for (std::thread &t : threads) {
        t.join();
    }
}

//------------------------------------------------------------------------------
///  @brief Calculate absorption.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
template<typename T, bool SAFE_MATH=false>
void calculate_power(const size_t num_times,
                     const size_t sub_steps,
                     const size_t num_rays) {
    std::vector<std::thread> threads(std::max(std::min(static_cast<unsigned int> (jit::context<T, SAFE_MATH>::max_concurrency()),
                                                       static_cast<unsigned int> (num_rays)),
                                              static_cast<unsigned int> (1)));

    const size_t batch = num_rays/threads.size();
    const size_t extra = num_rays%threads.size();

    for (size_t i = 0, ie = threads.size(); i < ie; i++) {
        threads[i] = std::thread([num_times, sub_steps, num_rays, batch, extra] (const size_t thread_number,
                                                                                 const size_t num_threads) -> void {
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

            omega->set(static_cast<T> (0.0));
            graph::shared_variable<T, SAFE_MATH> omega_var = graph::variable_cast(omega);
            graph::shared_variable<T, SAFE_MATH> kx_var = graph::variable_cast(kx);
            graph::shared_variable<T, SAFE_MATH> ky_var = graph::variable_cast(ky);
            graph::shared_variable<T, SAFE_MATH> kz_var = graph::variable_cast(kz);
            graph::shared_variable<T, SAFE_MATH> x_var = graph::variable_cast(x);
            graph::shared_variable<T, SAFE_MATH> y_var = graph::variable_cast(y);
            graph::shared_variable<T, SAFE_MATH> z_var = graph::variable_cast(z);
            graph::shared_variable<T, SAFE_MATH> t_var = graph::variable_cast(t);

            auto eq = equilibrium::make_efit<T, SAFE_MATH> (NC_FILE);
            //auto eq = equilibrium::make_slab_density<T, SAFE_MATH> ();
            //auto eq = equilibrium::make_slab_field<T, SAFE_MATH> ();
            //auto eq = equilibrium::make_no_magnetic_field<T, SAFE_MATH> ();

            absorption::root_finder<dispersion::hot_plasma<T, dispersion::z_erfi<T, SAFE_MATH>, SAFE_MATH>>
                root(kamp, omega, kx, ky, kz, x, y, z, t, eq,
                     stream.str(), local_num_rays, thread_number);
            root.compile();
            
            for (size_t j = 0, je = num_steps + 1; j < je; j++) {
                root.run(j);
            }
        }, i, threads.size());
    }

    for (std::thread &t : threads) {
        t.join();
    }
}

//------------------------------------------------------------------------------
///  @brief Main program of the driver.
///
///  @params[in] argc Number of commandline arguments.
///  @params[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    START_GPU
    const timeing::measure_diagnostic total("Total Time");

    jit::verbose = verbose;

    const size_t num_times = 100000;
    const size_t sub_steps = 100;
    const size_t num_rays = 100000;

    const bool use_safe_math = true;

    typedef double base;

    trace_ray<base> (num_times, sub_steps, num_rays);
    calculate_power<std::complex<base>, use_safe_math> (num_times,
                                                        sub_steps,
                                                        num_rays);

    std::cout << std::endl << "Timing:" << std::endl;
    total.print();

    END_GPU
}
