//
//  main.cpp
//  graph_driver
//
//  Created by Cianciosa, Mark R. on 7/13/19.
//  Copyright © 2019 Cianciosa, Mark R. All rights reserved.
//

#include <iostream>
#include <chrono>

#include "../graph_framework/trigonometry.hpp"
#include "../graph_framework/dispersion.hpp"

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

    std::cout << D.dx().at(0) << std::endl;
    std::cout << D.dy().at(0) << std::endl;
    std::cout << D.dz().at(0) << std::endl;

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

