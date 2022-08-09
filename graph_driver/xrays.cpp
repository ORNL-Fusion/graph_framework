//------------------------------------------------------------------------------
///  @file xrays.cpp
///  @brief Driver program for the rays library.
//------------------------------------------------------------------------------

#include <iostream>
#include <chrono>
#include <thread>
#include <random>

#include "../graph_framework/cpu_backend.hpp"
#include "../graph_framework/solver.hpp"

void write_time(const std::string &name, const std::chrono::nanoseconds time);

//------------------------------------------------------------------------------
///  @brief Main program of the driver.
///
///  @param[in] t Current Time.
//------------------------------------------------------------------------------
template<typename base>
static base solution(const base t) {
    return std::exp(-t) - std::exp(-1000.0*t);
}

//------------------------------------------------------------------------------
///  @brief Main program of the driver.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    //typedef std::complex<double> base;
    typedef double base;
    //typedef float base;
    //typedef std::complex<float> base;
    typedef backend::cpu<base> cpu;
    
    const std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    const size_t num_times = 10000;
    const size_t num_rays = 1;
    //const size_t num_rays = 10000;
    
    std::vector<std::thread> threads(std::max(std::min(std::thread::hardware_concurrency(),
                                                       static_cast<unsigned int> (num_rays)),
                                              static_cast<unsigned int> (1)));

    for (size_t i = 0, ie = threads.size(); i < ie; i++) {
        threads[i] = std::thread([num_times, num_rays] (const size_t thread_number,
                                                        const size_t num_threads) -> void {
            const size_t local_num_rays = num_rays/num_threads
                                        + std::min(thread_number, num_rays%num_threads);

            std::mt19937_64 engine((thread_number + 1)*static_cast<uint64_t> (std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())));
            std::uniform_real_distribution<double> real_dist(0.6, 1.0);
            std::uniform_int_distribution<size_t> int_dist(0, local_num_rays - 1);

            auto omega = graph::variable<cpu> (local_num_rays, "\\omega");
            auto kx = graph::variable<cpu> (local_num_rays, "k_{x}");
            auto ky = graph::variable<cpu> (local_num_rays, "k_{y}");
            auto kz = graph::variable<cpu> (local_num_rays, "k_{z}");
            auto x = graph::variable<cpu> (local_num_rays, "x");
            auto y = graph::variable<cpu> (local_num_rays, "y");
            auto z = graph::variable<cpu> (local_num_rays, "z");
            auto t = graph::variable<cpu> (local_num_rays, "t");

            t->set(backend::base_cast<cpu> (0.0));

//  Inital conditions.
            for (size_t j = 0; j < local_num_rays; j++) {
                omega->set(j, 600.0);
            }

            x->set(backend::base_cast<cpu> (0.0));
            y->set(backend::base_cast<cpu> (0.0));
            z->set(backend::base_cast<cpu> (0.0));
            kx->set(backend::base_cast<cpu> (600.0));
            ky->set(backend::base_cast<cpu> (0.0));
            kz->set(backend::base_cast<cpu> (0.0));
            
            //auto eq = equilibrium::make_slab_density<cpu> ();
            auto eq = equilibrium::make_no_magnetic_field<cpu> ();

            //solver::split_simplextic<dispersion::bohm_gross<cpu>>
            //solver::rk4<dispersion::bohm_gross<cpu>>
            solver::rk4<dispersion::simple<cpu>>
                solve(omega, kx, ky, kz, x, y, z, t, 30.0/num_times, eq);
            solve.init(kx);
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
                
            auto residule = solve.residule();

            const size_t sample = int_dist(engine);

            if (thread_number == 0) {
                std::cout << "Omega " << omega->evaluate().at(sample) << std::endl;
                std::cout << "t = " << 0.0 << " ";
                std::cout << solve.state.back().x.at(sample) << std::endl;
            }

            for (size_t j = 0; j < num_times; j++) {
                if (thread_number == 0) {
                    std::cout << "Time Step " << j << " Sample " << sample << " "
                              << solve.state.back().t.at(sample) << " "
                              << solve.state.back().x.at(sample) << " "
                              << solve.state.back().y.at(sample) << " "
                              << solve.state.back().z.at(sample) << " "
                              << solve.state.back().kx.at(sample) << " "
                              << solve.state.back().ky.at(sample) << " "
                              << solve.state.back().kz.at(sample) << " "
                              << residule->evaluate().at(sample)
                              << std::endl;
                }
                solve.step();
            }
            if (thread_number == 0) {
                std::cout << "Time Step " << num_times << " Sample " << sample << " "
                          << solve.state.back().t.at(sample) << " "
                          << solve.state.back().x.at(sample) << " "
                          << solve.state.back().y.at(sample) << " "
                          << solve.state.back().z.at(sample) << " "
                          << solve.state.back().kx.at(sample) << " "
                          << solve.state.back().ky.at(sample) << " "
                          << solve.state.back().kz.at(sample) << " "
                          << residule->evaluate().at(sample)
                          << std::endl;
            }
        }, i, threads.size());
    }

    for (std::thread &t : threads) {
        t.join();
    }

    const std::chrono::high_resolution_clock::time_point evaluate = std::chrono::high_resolution_clock::now();

    const auto total_time = evaluate - start;

    const std::chrono::nanoseconds total_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds> (total_time);

    std::cout << std::endl << "Timing:" << std::endl;
    std::cout << std::endl;
    write_time("  Total time : ", total_time_ns);
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
///  @brief Print out timings.
///
///  @param[in] name Discription of the times.
///  @param[in] time Elapsed time in nanoseconds.
//------------------------------------------------------------------------------
void write_time(const std::string &name, const std::chrono::nanoseconds time) {
    if (time.count() < 1000) {
        std::cout << name << time.count()               << " ns" << std::endl;
    } else if (time.count() < 1000000) {
        std::cout << name << time.count()/1000.0        << " μs" << std::endl;
    } else if (time.count() < 1000000000) {
        std::cout << name << time.count()/1000000.0     << " ms" << std::endl;
    } else if (time.count() < 60000000000) {
        std::cout << name << time.count()/1000000000.0  << " s" << std::endl;
    } else if (time.count() < 3600000000000) {
        std::cout << name << time.count()/60000000000.0 << " min" << std::endl;
    } else {
        std::cout << name << time.count()/3600000000000 << " h" << std::endl;
    }
}

