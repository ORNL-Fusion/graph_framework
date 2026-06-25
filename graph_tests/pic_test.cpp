//------------------------------------------------------------------------------
///  @file pic_test.cpp
///  @brief Tests for the particle in cell functions interface.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <random>

#include "../graph_framework/graph_framework.hpp"

//------------------------------------------------------------------------------
///  @brief Run interpolation test.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<std::floating_point  T> void run_interpolation_test() {
    const size_t num_mesh = 100;
    const size_t num_particles = 10000;

//  Characteristic factors
    const std::vector<T> ion_masses{pic::m_hydrogen<T>};
    const std::vector<uint8_t> ion_zs{1};

    const pic::characteristics norms(ion_masses, ion_zs, static_cast<T> (2.5E19));
    std::vector<pic::ion<T>> ions{pic::ion<T> (ion_masses[0], ion_zs[0], num_particles, norms)};
    pic::mesh<T> mesh(-3.0*norms.l, 3.0*norms.l, num_mesh, norms);

    std::function<T(T)> func([&norms](const T x) -> T {
        return std::sin(std::exp(x));
    });

    for (size_t i = 0; i < num_mesh; i++) {
        graph::variable_cast(mesh.y)->data()[i] = func(mesh.dx*i + mesh.xmin);
    }

    const T dxp = (mesh.xmax - mesh.xmin)/(num_particles - 1);
    for (size_t i = 0; i < num_particles; i++) {
        graph::variable_cast(ions[0].x)->data()[i] = dxp*i + mesh.xmin;
    }

    auto weights = pic::build_weights<T> (mesh, ions[0]);
    auto field = pic::build_interpolation<T> (mesh, ions[0]);
    auto weight = weights[0] + weights[1] + weights[2];

    workflow::manager<T> work(0);
    work.add_item({
        graph::variable_cast(mesh.y),
        graph::variable_cast(ions[0].x)
    }, {
        weight,
        field
    }, {}, NULL, "Mesh_Interpolation", num_particles);
    work.compile();
    work.run();
    work.wait();

//  The weights should sum to 1.
    for (size_t i = 0; i < num_particles; i++) {
        const T recieved = work.check_value(i, weight);
        const T diff = static_cast<T> (1) - recieved;
        if constexpr (std::same_as<T, float>) {
            assert(diff*diff < static_cast<T> (4.7E-12) &&
                   "Weight not equal to 1±4.7E-12");
        } else {
            assert(diff*diff < static_cast<T> (7.1E-30) &&
                   "Weight not equal to 1±7.1E-30");
        }
    }

    for (size_t i = 0, ie = num_particles/10; i < ie; i++) {
        const T x = work.check_value(i, ions[0].x);
        const T received = work.check_value(i, field);
        const T diff = func(x) - received;
        if constexpr (std::same_as<T, float>) {
            assert(diff*diff < static_cast<T> (4.0E-7) &&
                   "Profile not equal ±4.0E-7");
        } else {
            assert(diff*diff < static_cast<T> (4.0E-7) &&
                   "Profile not equal ±4.0E-7");
        }
    }
    for (size_t i = num_particles/10, ie = 2*num_particles/10; i < ie; i++) {
        const T x = work.check_value(i, ions[0].x);
        const T received = work.check_value(i, field);
        const T diff = func(x) - received;
        if constexpr (jit::use_cuda) {
            if constexpr (std::same_as<T, float>) {
                assert(diff*diff < static_cast<T> (1.22E-4) &&
                       "Profile not equal ±1.22E-4");
            } else {
                assert(diff*diff < static_cast<T> (1.04E-6) &&
                       "Profile not equal ±1.04E-6");
            }
        } else {
            if constexpr (std::same_as<T, float>) {
                assert(diff*diff < static_cast<T> (1.4E-6) &&
                       "Profile not equal ±1.4E-6");
            } else {
                assert(diff*diff < static_cast<T> (1.1E-6) &&
                       "Profile not equal ±1.1E-7");
            }
        }
    }
    std::cout << std::endl;
    for (size_t i = 2*num_particles/10, ie = 3*num_particles/10; i < ie; i++) {
        const T x = work.check_value(i, ions[0].x);
        const T received = work.check_value(i, field);
        const T diff = func(x) - received;
        if constexpr (jit::use_cuda) {
            if constexpr (std::same_as<T, float>) {
                assert(diff*diff < static_cast<T> (3.1E-4) &&
                       "Profile not equal ±3.1E-4");
            } else {
                assert(diff*diff < static_cast<T> (1.42E-8) &&
                       "Profile not equal ±1.42E-8");
            }
        } else {
            if constexpr (std::same_as<T, float>) {
                assert(diff*diff < static_cast<T> (3.8E-6) &&
                       "Profile not equal ±3.8E-6");
            } else {
                assert(diff*diff < static_cast<T> (1.5E-8) &&
                       "Profile not equal ±1.5E-8");
            }
        }
    }
    for (size_t i = 3*num_particles/10, ie = 4*num_particles/10; i < ie; i++) {
        const T x = work.check_value(i, ions[0].x);
        const T received = work.check_value(i, field);
        const T diff = func(x) - received;
        if constexpr (std::same_as<T, float>) {
            assert(diff*diff < static_cast<T> (5.9E-4) &&
                   "Profile not equal ±5.9E-4");
        } else {
            assert(diff*diff < static_cast<T> (7.9E-6) &&
                   "Profile not equal ±7.9E-6");
        }
    }
    for (size_t i = 4*num_particles/10, ie = 5*num_particles/10; i < ie; i++) {
        const T x = work.check_value(i, ions[0].x);
        const T received = work.check_value(i, field);
        const T diff = func(x) - received;
        if constexpr (std::same_as<T, float>) {
            assert(diff*diff < static_cast<T> (1.9E-5) &&
                   "Profile not equal ±1.9E-5");
        } else {
            assert(diff*diff < static_cast<T> (2.1E-8) &&
                   "Profile not equal ±2.1E-8");
        }
    }
    for (size_t i = 5*num_particles/10, ie = 6*num_particles/10; i < ie; i++) {
        const T x = work.check_value(i, ions[0].x);
        const T received = work.check_value(i, field);
        const T diff = func(x) - received;
        if constexpr (jit::use_metal<T> ()) {
            assert(diff*diff < static_cast<T> (1.6E-5) &&
                   "Profile not equal ±1.6E-5");
        } else {
            if constexpr (std::same_as<T, float>) {
                assert(diff*diff < static_cast<T> (1.7E-5) &&
                       "Profile not equal ±1.7E-5");
            } else {
                assert(diff*diff < static_cast<T> (2.9E-6) &&
                       "Profile not equal ±2.9E-6");
            }
        }
    }
    for (size_t i = 6*num_particles/10, ie = 7*num_particles/10; i < ie; i++) {
        const T x = work.check_value(i, ions[0].x);
        const T received = work.check_value(i, field);
        const T diff = func(x) - received;
        if constexpr (std::same_as<T, float>) {
            assert(diff*diff < static_cast<T> (4.0E-2) &&
                   "Profile not equal ±4.0E-2");
        } else {
            assert(diff*diff < static_cast<T> (7.0E-6) &&
                   "Profile not equal ±7.0E-6");
        }
    }
    for (size_t i = 7*num_particles/10, ie = 8*num_particles/10; i < ie; i++) {
        const T x = work.check_value(i, ions[0].x);
        const T received = work.check_value(i, field);
        const T diff = func(x) - received;
        if constexpr (std::same_as<T, float>) {
            assert(diff*diff < static_cast<T> (1.5E-3) &&
                   "Profile not equal ±1.5E-3");
        } else {
            assert(diff*diff < static_cast<T> (1.5E-3) &&
                   "Profile not equal ±1.5E-3");
        }
    }
    for (size_t i = 8*num_particles/10, ie = 9*num_particles/10; i < ie; i++) {
        const T x = work.check_value(i, ions[0].x);
        const T received = work.check_value(i, field);
        const T diff = func(x) - received;
        if constexpr (std::same_as<T, float>) {
            assert(diff*diff < static_cast<T> (5.0E-3) &&
                   "Profile not equal ±5.0E-3");
        } else {
            assert(diff*diff < static_cast<T> (3.0E-3) &&
                   "Profile not equal ±3.0E-3");
        }
    }
    for (size_t i = 9*num_particles/10, ie = 10*num_particles/10; i < ie; i++) {
        const T x = work.check_value(i, ions[0].x);
        const T received = work.check_value(i, field);
        const T diff = func(x) - received;
        if constexpr (std::same_as<T, float>) {
            assert(diff*diff < static_cast<T> (1.8E-2) &&
                   "Profile not equal ±1.8E-2");
        } else {
            assert(diff*diff < static_cast<T> (1.8E-2) &&
                   "Profile not equal ±1.8E-2");
        }
    }
}

//------------------------------------------------------------------------------
///  @brief Field solve test.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<std::floating_point T> void run_field_solve_test() {
    const size_t num_mesh = 100;
    const size_t num_particles = 1000000;

//  Characteristic factors
    const std::vector<T> ion_masses{pic::m_hydrogen<T>};
    const std::vector<uint8_t> ion_zs{1};

    const pic::characteristics norms(ion_masses, ion_zs, static_cast<T> (2.5E19));
    std::vector<pic::ion<T>> ions{pic::ion<T> (ion_masses[0], ion_zs[0], num_particles, norms)};
    pic::mesh<T> mesh(-3.0*norms.l, 3.0*norms.l, num_mesh, norms);

//  Initialize particle positions.
    backend::buffer<T> buffer(num_particles);

    std::mt19937 gen(0);
    std::normal_distribution<T> dist(0.0, 1.0);

    for (size_t i = 0; i < num_particles; i++) {
        do {
            buffer[i] = dist(gen);
        } while(buffer[i] < -3.0 || buffer[i] > 3.0);
    }
    ions[0].x->set(buffer);

//  Count particles in mesh bins. This builds a histogram of particle counts.
    std::vector<size_t> counts(num_mesh, 0);
    for (size_t i = 0; i < num_mesh; i++) {
        const T bin_low = i*mesh.dx + mesh.xmin - mesh.dx/2;
        const T bin_high = i*mesh.dx + mesh.xmin + mesh.dx/2;
        for (size_t j = 0; j < num_particles; j++) {
            if (buffer[j] >= bin_low && buffer[j] < bin_high) {
                counts[i]++;
            }
        }
    }

    auto weights = pic::build_weights<T> (mesh, ions[0]);
    auto mesh_i = mesh.build_i_index(ions[0]);
    auto mesh_solve = mesh.build_mesh_solve(ions[0]);

    workflow::manager<T> work(0);
    work.add_zero_item({
        graph::variable_cast(mesh.index),
        graph::variable_cast(mesh.y)
    });
    work.add_item({
        graph::variable_cast(ions[0].x),
        graph::variable_cast(ions[0].weights[0]),
        graph::variable_cast(ions[0].weights[1]),
        graph::variable_cast(ions[0].weights[2]),
        graph::variable_cast(ions[0].indices)
    }, {}, {
        {weights[0], graph::variable_cast(ions[0].weights[0])},
        {weights[1], graph::variable_cast(ions[0].weights[1])},
        {weights[2], graph::variable_cast(ions[0].weights[2])},
        {mesh_i, graph::variable_cast(ions[0].indices)}
    }, NULL, "compute_weights", num_particles);
    work.add_loop_item({
        graph::variable_cast(ions[0].indices),
        graph::variable_cast(ions[0].weights[0]),
        graph::variable_cast(ions[0].weights[1]),
        graph::variable_cast(ions[0].weights[2]),
        graph::variable_cast(mesh.index),
        graph::variable_cast(mesh.y)
    }, {}, {
        {mesh_solve[0], graph::variable_cast(mesh.index)},
        {mesh_solve[1], graph::variable_cast(mesh.y)}
    }, NULL, "sum_weights", num_mesh, num_particles);

    work.compile();

    const timing::measure_diagnostic t_run("Run Time");
    work.run();
    work.wait();
    t_run.print();

    for (size_t i = 0; i < num_mesh; i++) {
        const T recieved = work.check_value(i, mesh.y);
        const T error = std::abs((counts[i] - recieved)/counts[i]);
        if constexpr (std::is_same_v<float, T>) {
            assert(error < 0.155 && "Error outside tolarance range.");
        } else {
            assert(error < 0.192 && "Error outside tolarance range.");
        }
    }
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified precision.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<std::floating_point T> void run_tests() {
    run_interpolation_test<T> ();
    run_field_solve_test<T> ();
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    START_GPU

    (void)argc;
    (void)argv;
    run_tests<float> ();
    run_tests<double> ();

    END_GPU
}
