//------------------------------------------------------------------------------
///  @file xrays.cpp
///  @brief Driver program for the rays library.
//------------------------------------------------------------------------------

#include <iostream>
#include <chrono>
#include <thread>

#include "../graph_framework/solver.hpp"

void write_time(const std::string &name, const std::chrono::nanoseconds time);

//------------------------------------------------------------------------------
///  @brief Main program of the driver.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    const std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    const size_t num_times = 10000;
    const size_t num_rays = 10000;

    std::vector<std::thread> threads(std::max(std::thread::hardware_concurrency(),
                                              static_cast<unsigned int> (1)));

    for (size_t i = 0, ie = threads.size(); i < ie; i++) {
        threads[i] = std::thread([num_times, num_rays] (const size_t thread_number,
                                                        const size_t num_threads) -> void {
            const size_t local_num_rays = num_rays/num_threads
                                        + std::min(thread_number, num_rays%num_threads);

            auto omega = graph::variable(local_num_rays);
            auto kx = graph::variable(local_num_rays);
            auto ky = graph::variable(local_num_rays);
            auto kz = graph::variable(local_num_rays);
            auto x = graph::variable(local_num_rays);
            auto y = graph::variable(local_num_rays);
            auto z = graph::variable(local_num_rays);

//  Inital conditions.
            omega->set(1.0);
            x->set(1.0);
            y->set(0.0);
            z->set(0.0);
            kx->set(1.0);
            ky->set(0.0);
            kz->set(0.0);

            solver::rk2<dispersion::simple> solve(omega, kx, ky, kz, x, y, z, 1.0);
            solve.init(kx);

            for (size_t i = 0; i < num_times; i++) {
                if (thread_number == 1) {
                    std::cout << "Time Step " << i << " ";
                    std::cout << solve.state.back().x.at(0) << " "
                              << solve.state.back().y.at(0) << " "
                              << solve.state.back().z.at(0) << " "
                              << solve.state.back().kx.at(0) << " "
                              << solve.state.back().ky.at(0) << " "
                              << solve.state.back().kz.at(0) << " "
                              << solve.state.size() << std::endl;
                }

                solve.step();
            }
            if (thread_number == 1) {
                std::cout << "Time Step " << num_times << " ";
                std::cout << solve.state.back().x.at(0) << " "
                          << solve.state.back().y.at(0) << " "
                          << solve.state.back().z.at(0) << " "
                          << solve.state.back().kx.at(0) << " "
                          << solve.state.back().ky.at(0) << " "
                          << solve.state.back().kz.at(0) << " "
                          << solve.state.size() << std::endl;
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

