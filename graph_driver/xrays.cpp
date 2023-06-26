//------------------------------------------------------------------------------
///  @file xrays.cpp
///  @brief Driver program for the rays library.
//------------------------------------------------------------------------------

#include <iostream>
#include <chrono>
#include <thread>
#include <random>

#include "../graph_framework/solver.hpp"
#include "../graph_framework/timing.hpp"

const bool print = false;
const bool write_step = true;

//void write_time(const std::string &name, const std::chrono::nanoseconds time);

//------------------------------------------------------------------------------
///  @brief Main program of the driver.
///
///  @params[in] argc Number of commandline arguments.
///  @params[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    START_GPU

    std::mutex sync;

    //typedef float base;
    //typedef double base;
    //typedef std::complex<float> base;
    typedef std::complex<double> base;

    const timeing::measure_diagnostic total("Total Time");

    const size_t num_times = 100000;
    const size_t sub_steps = 10;
    const size_t num_steps = num_times/sub_steps;
    const size_t num_rays = 10000;//000;

    std::vector<std::thread> threads(0);
    if constexpr (jit::use_gpu<base> ()) {
        threads.resize(1);
    } else {
        threads.resize(std::max(std::min(std::thread::hardware_concurrency(),
                                         static_cast<unsigned int> (num_rays)),
                                static_cast<unsigned int> (1)));
    }

    for (size_t i = 0, ie = threads.size(); i < ie; i++) {
        threads[i] = std::thread([num_times, num_rays, &sync] (const size_t thread_number,
                                                               const size_t num_threads) -> void {
            const size_t local_num_rays = num_rays/num_threads
                                        + std::min(thread_number, num_rays%num_threads);

            std::mt19937_64 engine((thread_number + 1)*static_cast<uint64_t> (std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())));
            std::uniform_int_distribution<size_t> int_dist(0, local_num_rays - 1);
            
            auto omega = graph::variable<base> (local_num_rays, "\\omega");
            auto kx = graph::variable<base> (local_num_rays, "k_{x}");
            auto ky = graph::variable<base> (local_num_rays, "k_{y}");
            auto kz = graph::variable<base> (local_num_rays, "k_{z}");
            auto x = graph::variable<base> (local_num_rays, "x");
            auto y = graph::variable<base> (local_num_rays, "y");
            auto z = graph::variable<base> (local_num_rays, "z");
            auto t = graph::variable<base> (local_num_rays, "t");

            t->set(static_cast<base> (0.0));

//  Inital conditions.
            if constexpr (jit::is_complex<base, float> ()) {
                std::normal_distribution<float> norm_dist(static_cast<float> (600.0), static_cast<float> (10.0));
                for (size_t j = 0; j < local_num_rays; j++) {
                    omega->set(j, static_cast<base> (norm_dist(engine)));
                }
            } else if constexpr (jit::is_complex<base, double> ()) {
                std::normal_distribution<double> norm_dist(static_cast<double> (600.0), static_cast<double> (10.0));
                for (size_t j = 0; j < local_num_rays; j++) {
                    omega->set(j, static_cast<base> (norm_dist(engine)));
                }
            } else {
                std::normal_distribution<base> norm_dist(static_cast<base> (600.0), static_cast<base> (10.0));
                for (size_t j = 0; j < local_num_rays; j++) {
                    omega->set(j, static_cast<base> (norm_dist(engine)));
                }
            }

            x->set(static_cast<base> (2.5));
            //x->set(static_cast<base> (0.0));
            y->set(static_cast<base> (0.0));
            z->set(static_cast<base> (0.0));
            kx->set(static_cast<base> (-600.0));
            //kx->set(static_cast<base> (600.0));
            ky->set(static_cast<base> (0.0));
            kz->set(static_cast<base> (0.0));


            auto eq = equilibrium::make_efit<base> (NC_FILE, sync);
            //auto eq = equilibrium::make_slab_density<base> ();
            //auto eq = equilibrium::make_no_magnetic_field<base> ();

            //const base endtime = static_cast<base> (4.0);
            const base endtime = static_cast<base> (10.0);
            const base dt = endtime/static_cast<base> (num_times);

            //auto dt_var = graph::variable(num_rays, static_cast<base> (dt), "dt");

            std::ostringstream stream;
            stream << "result" << thread_number << ".nc";

            //solver::split_simplextic<dispersion::bohm_gross<base>>
            //solver::adaptive_rk4<dispersion::bohm_gross<base>>
            //solver::rk4<dispersion::simple<base>>
            //solver::rk4<dispersion::ordinary_wave<base>>
            //solver::rk4<dispersion::extra_ordinary_wave<base>>
            //solver::rk4<dispersion::cold_plasma<base>>
            //solver::adaptive_rk4<dispersion::ordinary_wave<base>>
            solver::rk4<dispersion::hot_plasma<base, dispersion::z_erfi<base>>>
                solve(omega, kx, ky, kz, x, y, z, t, dt, eq,
                      stream.str(), local_num_rays);
                //solve(omega, kx, ky, kz, x, y, z, t, dt_var, eq,
                //      stream.str(), num_rays);
            solve.init(ky);
            solve.compile();
            if (thread_number == 0 && false) {
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
                for(size_t k = 0; k < sub_steps; k++) {
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

    std::cout << std::endl << "Timing:" << std::endl;
    total.stop();

    END_GPU
}
