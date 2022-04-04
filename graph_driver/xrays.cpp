//
//  main.cpp
//  graph_driver
//
//  Created by Cianciosa, Mark R. on 7/13/19.
//  Copyright © 2019 Cianciosa, Mark R. All rights reserved.
//

#include <iostream>
#include <chrono>

#include "../graph_framework/solver.hpp"

void write_time(const std::string &name, const std::chrono::nanoseconds time);

int main(int argc, const char * argv[]) {
    const std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    auto omega = graph::variable(1);
    auto kx = graph::variable(1);
    auto ky = graph::variable(1);
    auto kz = graph::variable(1);
    auto x = graph::variable(1);
    auto y = graph::variable(1);
    auto z = graph::variable(1);

    dispersion::simple D(omega, kx, ky, kz, x, y, z);

//  Inital conditions.
    omega->set(1.0);
    x->set(1.0);
    y->set(0.0);
    z->set(0.0);
    kx->set(1.0);
    ky->set(0.0);
    kz->set(0.0);

    solver::rk2<dispersion::simple> solve(D, kx, ky, kz, x, y, z, 1.0);

    const size_t num_times = 100;

    for (size_t i = 0; i < num_times; i++) {
        std::cout << "Time Step " << i << " ";
        std::cout << solve.state.back().x.at(0) << " "
                  << solve.state.back().y.at(0) << " "
                  << solve.state.back().z.at(0) << " "
                  << solve.state.back().kx.at(0) << " "
                  << solve.state.back().ky.at(0) << " "
                  << solve.state.back().kz.at(0) << " "
                  << solve.state.size() << std::endl;

        solve.step();
    }
    std::cout << "Time Step " << num_times << " ";
    std::cout << solve.state.back().x.at(0) << " "
              << solve.state.back().y.at(0) << " "
              << solve.state.back().z.at(0) << " "
              << solve.state.back().kx.at(0) << " "
              << solve.state.back().ky.at(0) << " "
              << solve.state.back().kz.at(0) << " "
              << solve.state.size() << std::endl;

    const std::chrono::high_resolution_clock::time_point evaluate = std::chrono::high_resolution_clock::now();

    const auto total_time = evaluate - start;

    const std::chrono::nanoseconds total_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds> (total_time);

    std::cout << std::endl << "Timing:" << std::endl;
    std::cout << std::endl;
    write_time("  Total time : ", total_time_ns);
    std::cout << std::endl;
}

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

