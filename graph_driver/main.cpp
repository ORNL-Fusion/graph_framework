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

void write_time(const std::string &name, const std::chrono::nanoseconds time);

int main(int argc, const char * argv[]) {
    const std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    auto zero = graph::zero;
    auto one = graph::one;

    auto variable = graph::variable(1, 5);
    auto constant = graph::variable(1.3);

    auto variable_vector = graph::variable(1000);

    std::vector<double> temp(1000);
    for (size_t i = 0; i < 1000; i++) {
        temp[i] = i;
    }
    variable_vector->set(temp);
    auto div = variable_vector/(one + one);
    temp = div->evaluate();

    auto add_constant = constant + constant;

    auto tangent = tan(variable_vector);

    std::cout << zero->evaluate().at(0) << std::endl;
    std::cout << one->evaluate().at(0) << std::endl;
    std::cout << (one + one)->evaluate().at(0) << std::endl;
    std::cout << (one - one)->evaluate().at(0) << std::endl;
    std::cout << (one*one)->evaluate().at(0) << std::endl;
    std::cout << (one/(one + one))->evaluate().at(0) << std::endl;
    std::cout << variable->evaluate().at(0) << std::endl;
    std::cout << add_constant->evaluate().at(0) << std::endl;
    std::cout << tangent->df(variable_vector)->evaluate().at(0) << std::endl;

    variable->set(20);

    std::cout << variable->evaluate().at(0) << std::endl;
    std::cout << (variable->df(variable))->evaluate().at(0) << std::endl;

    auto trig_sin = sin(variable);

    std::cout << trig_sin->evaluate().at(0) << std::endl;
    std::cout << (trig_sin->df(variable))->evaluate().at(0) << std::endl;

    std::cout << std::endl;
    for (double e: temp) {
        std::cout << e << std::endl;
    }
    std::cout << std::endl;

    temp = div->df(variable_vector)->evaluate();
    std::cout << std::endl;
    for (double e: temp) {
        std::cout << e << std::endl;
    }
    std::cout << std::endl;

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

