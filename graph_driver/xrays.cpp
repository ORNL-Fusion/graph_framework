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
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    typedef std::complex<double> base;
    //typedef double base;
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

            //const double q = 1.602176634E-19;
            //const double me = 9.1093837015E-31;
            //const double mu0 = M_PI*4.0E-7;
            //const double epsilon0 = 8.8541878138E-12;
            //const double c = 1.0/std::sqrt(mu0*epsilon0);
            //const double OmegaCE = -q*1/(me*c);

//  Inital conditions.
            for (size_t j = 0; j < local_num_rays; j++) {
                //omega->set(j, 1000.0*real_dist(engine));
                //omega->set(j, OmegaCE);
                //omega->set(j, real_dist(engine));
                //omega->set(j, 600.0);
                //omega->set(j, 1000.0);
                //omega->set(j, 1100.0);
                omega->set(j, 800.0);
                //omega->set(j, 1.0);
            }

            //x->set(backend::base_cast<cpu> (8.58));
            //y->set(backend::base_cast<cpu> (0.0));
            //z->set(backend::base_cast<cpu> (-0.25));
            //kx->set(backend::base_cast<cpu> (16.0));
            //ky->set(backend::base_cast<cpu> (0.0));
            //kz->set(backend::base_cast<cpu> (0.8*OmegaCE));

            //x->set(backend::base_cast<cpu> (-11.0));
            //y->set(backend::base_cast<cpu> (0.0));
            //z->set(backend::base_cast<cpu> (-0.25));
            //kx->set(backend::base_cast<cpu> (std::complex<double> (149.0, 450.0)));
            //ky->set(backend::base_cast<cpu> (0.0));
            //kz->set(backend::base_cast<cpu> (0.8*OmegaCE));
            
            //x->set(backend::base_cast<cpu> (0.2));
            //y->set(backend::base_cast<cpu> (0.0));
            //z->set(backend::base_cast<cpu> (-0.25));
            //kx->set(backend::base_cast<cpu> (std::complex<double> (22.0, 1.0)));
            //ky->set(backend::base_cast<cpu> (0.0));
            //kz->set(backend::base_cast<cpu> (0.7*OmegaCE));

            //x->set(backend::base_cast<cpu> (0.1));
            //y->set(backend::base_cast<cpu> (0.0));
            //z->set(backend::base_cast<cpu> (-0.25));
            //kx->set(backend::base_cast<cpu> (22.0));
            //ky->set(backend::base_cast<cpu> (0.0));
            //kz->set(backend::base_cast<cpu> (0.7*OmegaCE));
            
            //x->set(backend::base_cast<cpu> (-3.95));
            //y->set(backend::base_cast<cpu> (0.0));
            //z->set(backend::base_cast<cpu> (-0.25));
            //kx->set(backend::base_cast<cpu> (14.0));
            //ky->set(backend::base_cast<cpu> (0.0));
            //kz->set(backend::base_cast<cpu> (0.6*OmegaCE));
            
            //x->set(backend::base_cast<cpu> (-7.75));
            //y->set(backend::base_cast<cpu> (0.0));
            //z->set(backend::base_cast<cpu> (-0.25));
            //kx->set(backend::base_cast<cpu> (25.0));
            //ky->set(backend::base_cast<cpu> (0.0));
            //kz->set(backend::base_cast<cpu> (0.4*OmegaCE));
            
            //x->set(backend::base_cast<cpu> (-1.0));
            //y->set(backend::base_cast<cpu> (-0.2));
            //z->set(backend::base_cast<cpu> (0.0));
            //kx->set(backend::base_cast<cpu> (1000.0));
            //ky->set(backend::base_cast<cpu> (0.0));
            //kz->set(backend::base_cast<cpu> (0.0));

            //x->set(backend::base_cast<cpu> (real_dist(engine)));
            //y->set(backend::base_cast<cpu> (real_dist(engine)));
            //z->set(backend::base_cast<cpu> (real_dist(engine)));
            //kx->set(backend::base_cast<cpu> (base(real_dist(engine),
            //                                      real_dist(engine))));
            //ky->set(backend::base_cast<cpu> (real_dist(engine)));
            //kz->set(backend::base_cast<cpu> (real_dist(engine)));

            //x->set(backend::base_cast<cpu> (-1.0));
            //y->set(backend::base_cast<cpu> (0.0));
            //z->set(backend::base_cast<cpu> (0.0));
            //kx->set(backend::base_cast<cpu> (1000.0));
            //ky->set(backend::base_cast<cpu> (0.0));
            //kz->set(backend::base_cast<cpu> (0.0));

            //x->set(backend::base_cast<cpu> (10.0));
            //x->set(backend::base_cast<cpu> (-10.0));
            //x->set(backend::base_cast<cpu> (-35.0));
            //y->set(backend::base_cast<cpu> (0.0));
            //z->set(backend::base_cast<cpu> (0.0));
            //kx->set(backend::base_cast<cpu> (-600.0));
            //kx->set(backend::base_cast<cpu> (600.0));
            //ky->set(backend::base_cast<cpu> (0.0));
            //kz->set(backend::base_cast<cpu> (0.0));

            //x->set(backend::base_cast<cpu> (0.0));
            x->set(backend::base_cast<cpu> (10.0));
            y->set(backend::base_cast<cpu> (0.0));
            z->set(backend::base_cast<cpu> (0.0));
            //kx->set(backend::base_cast<cpu> (600.0));
            //kx->set(backend::base_cast<cpu> (1000.0));
            //kx->set(backend::base_cast<cpu> (500.0));
            kx->set(backend::base_cast<cpu> (1500.0));
            ky->set(backend::base_cast<cpu> (0.0));
            kz->set(backend::base_cast<cpu> (0.0));
            
            //auto eq = equilibrium::make_guassian_density<cpu> ();
            //auto eq = equilibrium::make_slab<cpu> ();
            auto eq = equilibrium::make_slab_density<cpu> ();

            //solver::rk4<dispersion::cold_plasma<cpu>> solve(omega, kx, ky, kz, x, y, z, 60.0/num_times, eq);
            solver::rk4<dispersion::cold_plasma<cpu>> solve(omega, kx, ky, kz, x, y, z, 200.0/num_times, eq);
            //solver::rk4<dispersion::cold_plasma<cpu>> solve(omega, kx, ky, kz, x, y, z, 70.0/num_times, eq);
            //solver::rk4<dispersion::ordinary_wave<cpu>> solve(omega, kx, ky, kz, x, y, z, 100.0/num_times, eq);
            //solver::rk4<dispersion::extra_ordinary_wave<cpu>> solve(omega, kx, ky, kz, x, y, z, 300.0/num_times, eq);
            //solver::rk4<dispersion::extra_ordinary_wave<cpu>> solve(omega, kx, ky, kz, x, y, z, 80.0/num_times, eq);
            //solver::rk4<dispersion::guassian_well<cpu>> solve(omega, kx, ky, kz, x, y, z, 2.0/num_times, eq);
            //solver::rk4<dispersion::bohm_gross<cpu>> solve(omega, kx, ky, kz, x, y, z, 70.0/num_times, eq);
            //solver::rk4<dispersion::acoustic_wave<cpu>> solve(omega, kx, ky, kz, x, y, z, 1.0/num_times, eq);
            //solver::rk4<dispersion::simple<cpu>> solve(omega, kx, ky, kz, x, y, z, 1.0/num_times, eq);
            solve.init(kx);
            if (thread_number == 0) {
                solve.print_dispersion();
            }
                
            auto residule = solve.residule();

            const size_t sample = int_dist(engine);

            if (thread_number == 0) {
                std::cout << "Omega " << omega->evaluate().at(sample) << std::endl;
            }

            for (size_t j = 0; j < num_times; j++) {
                if (thread_number == 0) {
                    std::cout << "Time Step " << j << " Sample " << sample << " "
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

