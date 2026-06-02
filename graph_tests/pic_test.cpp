//------------------------------------------------------------------------------
///  @file pic_test.cpp
///  @brief Tests for the particle in cell functions interface.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/graph_framework.hpp"

//------------------------------------------------------------------------------
///  @brief Run interpolation test.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<std::floating_point  T> void run_interpolation_test() {
    const size_t num_mesh = 100;
    const size_t num_particles = 10000;

    auto xmesh = graph::variable<T> (num_mesh, "x_mesh");
    auto ymesh = graph::variable<T> (num_mesh, "y_mesh");

    const T xmin = static_cast<T> (-3);
    const T xmax = static_cast<T> (3);
    const T dx = (xmax - xmin)/(num_mesh - 1);

    std::function<T(T)> func([](const T x) -> T {
        return std::sin(std::exp(x));
    });

    for (size_t i = 0; i < num_mesh; i++) {
        graph::variable_cast(xmesh)->data()[i] = dx*i + xmin;
        graph::variable_cast(ymesh)->data()[i] = func(graph::variable_cast(xmesh)->data()[i]);
    }

    auto xp = graph::variable<T> (num_particles, "xp");
    const T dxp = (xmax - xmin)/(num_particles - 1);
    for (size_t i = 0; i < num_particles; i++) {
        graph::variable_cast(xp)->data()[i] = dxp*i + xmin;
    }

    auto field = pic::build_interpolation<T, true> (xmesh, ymesh, xp, xmin, dx);

    workflow::manager<T> work(0);
    work.add_item({
        graph::variable_cast(xmesh),
        graph::variable_cast(ymesh),
        graph::variable_cast(xp)
    }, {
        field
    }, {}, NULL, "Mesh_Interpolation", num_particles);
    work.compile();

    work.run();
    work.wait();

    auto xp_cast = graph::variable_cast(xp);
    for (size_t i = 0, ie = xp_cast->size()/10; i < ie; i++) {
        const T x = work.check_value(i, xp);
        const T recieved = work.check_value(i, field);
        const T diff = func(x) - recieved;
        if constexpr (std::same_as<T, float>) {
            assert(diff*diff < static_cast<T> (4.0E-7) &&
                   "Profile not equal to 1±4.0E-7");
        } else {
            assert(diff*diff < static_cast<T> (4.0E-7) &&
                   "Profile not equal to 1±4.0E-7");
        }
    }
    for (size_t i = xp_cast->size()/10, ie = 2*xp_cast->size()/10; i < ie; i++) {
        const T x = work.check_value(i, xp);
        const T recieved = work.check_value(i, field);
        const T diff = func(x) - recieved;
        if constexpr (std::same_as<T, float>) {
            assert(diff*diff < static_cast<T> (1.4E-6) &&
                   "Profile not equal to 1±1.4E-6");
        } else {
            assert(diff*diff < static_cast<T> (1.1E-6) &&
                   "Profile not equal to 1±1.1E-7");
        }
    }
    for (size_t i = 2*xp_cast->size()/10, ie = 3*xp_cast->size()/10; i < ie; i++) {
        const T x = work.check_value(i, xp);
        const T recieved = work.check_value(i, field);
        const T diff = func(x) - recieved;
        if constexpr (std::same_as<T, float>) {
            assert(diff*diff < static_cast<T> (3.8E-6) &&
                   "Profile not equal to 1±3.8E-6");
        } else {
            assert(diff*diff < static_cast<T> (1.5E-8) &&
                   "Profile not equal to 1±1.5E-8");
        }
    }
    for (size_t i = 3*xp_cast->size()/10, ie = 4*xp_cast->size()/10; i < ie; i++) {
        const T x = work.check_value(i, xp);
        const T recieved = work.check_value(i, field);
        const T diff = func(x) - recieved;
        if constexpr (std::same_as<T, float>) {
            assert(diff*diff < static_cast<T> (5.9E-4) &&
                   "Profile not equal to 1±5.9E-4");
        } else {
            assert(diff*diff < static_cast<T> (7.9E-6) &&
                   "Profile not equal to 1±7.9E-6");
        }
    }
    for (size_t i = 4*xp_cast->size()/10, ie = 5*xp_cast->size()/10; i < ie; i++) {
        const T x = work.check_value(i, xp);
        const T recieved = work.check_value(i, field);
        const T diff = func(x) - recieved;
        if constexpr (std::same_as<T, float>) {
            assert(diff*diff < static_cast<T> (1.9E-5) &&
                   "Profile not equal to 1±1.9E-5");
        } else {
            assert(diff*diff < static_cast<T> (2.1E-8) &&
                   "Profile not equal to 1±2.1E-8");
        }
    }
    for (size_t i = 5*xp_cast->size()/10, ie = 6*xp_cast->size()/10; i < ie; i++) {
        const T x = work.check_value(i, xp);
        const T recieved = work.check_value(i, field);
        const T diff = func(x) - recieved;
        if constexpr (std::same_as<T, float>) {
            assert(diff*diff < static_cast<T> (1.6E-5) &&
                   "Profile not equal to 1±1.6E-5");
        } else {
            assert(diff*diff < static_cast<T> (2.9E-6) &&
                   "Profile not equal to 1±2.9E-6");
        }
    }
    for (size_t i = 6*xp_cast->size()/10, ie = 7*xp_cast->size()/10; i < ie; i++) {
        const T x = work.check_value(i, xp);
        const T recieved = work.check_value(i, field);
        const T diff = func(x) - recieved;
        if constexpr (std::same_as<T, float>) {
            assert(diff*diff < static_cast<T> (4.0E-2) &&
                   "Profile not equal to 1±4.0E-2");
        } else {
            assert(diff*diff < static_cast<T> (7.0E-6) &&
                   "Profile not equal to 1±7.0E-6");
        }
    }
    for (size_t i = 7*xp_cast->size()/10, ie = 8*xp_cast->size()/10; i < ie; i++) {
        const T x = work.check_value(i, xp);
        const T recieved = work.check_value(i, field);
        const T diff = func(x) - recieved;
        if constexpr (std::same_as<T, float>) {
            assert(diff*diff < static_cast<T> (1.5E-3) &&
                   "Profile not equal to 1±1.5E-3");
        } else {
            assert(diff*diff < static_cast<T> (1.5E-3) &&
                   "Profile not equal to 1±1.5E-3");
        }
    }
    for (size_t i = 8*xp_cast->size()/10, ie = 9*xp_cast->size()/10; i < ie; i++) {
        const T x = work.check_value(i, xp);
        const T recieved = work.check_value(i, field);
        const T diff = func(x) - recieved;
        if constexpr (std::same_as<T, float>) {
            assert(diff*diff < static_cast<T> (5.0E-3) &&
                   "Profile not equal to 1±5.0E-3");
        } else {
            assert(diff*diff < static_cast<T> (3.0E-3) &&
                   "Profile not equal to 1±3.0E-3");
        }
    }
    for (size_t i = 9*xp_cast->size()/10, ie = 10*xp_cast->size()/10; i < ie; i++) {
        const T x = work.check_value(i, xp);
        const T recieved = work.check_value(i, field);
        const T diff = func(x) - recieved;
        if constexpr (std::same_as<T, float>) {
            assert(diff*diff < static_cast<T> (1.8E-2) &&
                   "Profile not equal to 1±1.8E-2");
        } else {
            assert(diff*diff < static_cast<T> (1.8E-2) &&
                   "Profile not equal to 1±1.8E-2");
        }
    }
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<std::floating_point T> void run_tests() {
    run_interpolation_test<T> ();
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
