//------------------------------------------------------------------------------
///  @file xrays.cpp
///  @brief Driver program for the rays library.
//------------------------------------------------------------------------------

#include <iostream>
#include <chrono>
#include <thread>
#include <random>

#include "../graph_framework/backend.hpp"
#include "../graph_framework/solver.hpp"
#include "../graph_framework/timing.hpp"

void write_time(const std::string &name, const std::chrono::nanoseconds time);

//------------------------------------------------------------------------------
///  @brief Main program of the driver.
///
///  @params[in] t Current Time.
//------------------------------------------------------------------------------
template<typename base>
static base solution(const base t) {
    return std::exp(-t) - std::exp(-1000.0*t);
}

//------------------------------------------------------------------------------
///  @brief Main program of the driver.
///
///  @params[in] argc Number of commandline arguments.
///  @params[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    START_GPU

    //typedef std::complex<double> base;
    //typedef double base;
    typedef float base;
    //typedef std::complex<float> base;
    
    const timeing::measure_diagnostic total("Total Time");

    const size_t num_times = 10000;
    //const size_t num_rays = 1;
    const size_t num_rays = 1000000;

    std::vector<std::thread> threads(0);
    if constexpr (jit::use_gpu<base> ()) {
        threads.resize(1);
    } else {
        threads.resize(std::max(std::min(std::thread::hardware_concurrency(),
                                         static_cast<unsigned int> (num_rays)),
                                static_cast<unsigned int> (1)));
    }

    for (size_t i = 0, ie = threads.size(); i < ie; i++) {
        threads[i] = std::thread([num_times, num_rays] (const size_t thread_number,
                                                        const size_t num_threads) -> void {
            const size_t local_num_rays = num_rays/num_threads
                                        + std::min(thread_number, num_rays%num_threads);

            std::mt19937_64 engine((thread_number + 1)*static_cast<uint64_t> (std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())));
            std::uniform_real_distribution<base> real_dist(0.6, 1.0);
            std::normal_distribution<base> norm_dist(600.0, 10.0);
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
            for (size_t j = 0; j < local_num_rays; j++) {
                omega->set(j, norm_dist(engine));
            }

            x->set(static_cast<base> (0.0));
            y->set(static_cast<base> (0.0));
            z->set(static_cast<base> (0.0));
            kx->set(static_cast<base> (600.0));
            ky->set(static_cast<base> (0.0));
            kz->set(static_cast<base> (0.0));

            auto eq = equilibrium::make_slab_density<base> ();
            //auto eq = equilibrium::make_no_magnetic_field<base> ();

            //solver::split_simplextic<dispersion::bohm_gross<base>>
            //solver::rk4<dispersion::bohm_gross<base>>
            //solver::rk4<dispersion::simple<base>>
            //solver::rk4<dispersion::ordinary_wave<base>>
            //solver::rk4<dispersion::extra_ordinary_wave<base>>
            solver::rk4<dispersion::cold_plasma<base>>
                solve(omega, kx, ky, kz, x, y, z, t, 60.0/num_times, eq);
            solve.init(kx);
            solve.compile(num_rays);
            if (thread_number == 0) {
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

            for (size_t j = 0; j < num_times; j++) {
                if (thread_number == 0) {
                    solve.print(sample);
                }
                solve.step();
            }

            if (thread_number == 0) {
                solve.print(sample);
            } else {
                solve.sync();
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
