//------------------------------------------------------------------------------
///  @file xrays.cpp
///  @brief Benchmark Case for the Rays code.
//------------------------------------------------------------------------------

#include <iostream>
#include <thread>

#include "../graph_framework/solver.hpp"
#include "../graph_framework/timing.hpp"

//------------------------------------------------------------------------------
///  @brief Bench runner.
///
///  @tparam T         Base type of the calculation.
///  @tparam NUM_TIMES Total number of times steps.
///  @tparam SUB_STEPS Number of substeps.
///  @tparam NUM_RAYS  Number of rays.
//------------------------------------------------------------------------------
template<jit::float_scalar T, size_t NUM_TIMES, size_t SUB_STEPS, size_t NUM_RAYS>
void bench_runner() {
    if constexpr (std::is_same<T, float>::value) {
        std::cout << "Float --------------------------------------------------------------------------" << std::endl;
    } else if constexpr (std::is_same<T, double>::value) {
        std::cout << "Double -------------------------------------------------------------------------" << std::endl;
    } else if constexpr (std::is_same<T, std::complex<float>>::value) {
        std::cout << "Complex Float ------------------------------------------------------------------" << std::endl;
    } else {
        std::cout << "Complex Double -----------------------------------------------------------------" << std::endl;
    }

    const size_t num_steps = NUM_TIMES/SUB_STEPS;

    std::vector<std::thread> threads(std::max(std::min(static_cast<unsigned int> (jit::context<T>::max_concurrency()),
                                                       static_cast<unsigned int> (NUM_RAYS)),
                                              static_cast<unsigned int> (1)));

    const size_t batch = NUM_RAYS/threads.size();
    const size_t extra = NUM_RAYS%threads.size();

    timeing::measure_diagnostic_threaded timing;

    for (size_t i = 0, ie = threads.size(); i < ie; i++) {
        threads[i] = std::thread([&timing, batch, extra] (const size_t thread_number) -> void {

            const size_t local_num_rays = batch
                                        + (extra > thread_number ? 1 : 0);

            auto omega = graph::variable<T> (local_num_rays, "\\omega");
            auto kx    = graph::variable<T> (local_num_rays, "k_{x}");
            auto ky    = graph::variable<T> (local_num_rays, "k_{y}");
            auto kz    = graph::variable<T> (local_num_rays, "k_{z}");
            auto x     = graph::variable<T> (local_num_rays, "x");
            auto y     = graph::variable<T> (local_num_rays, "y");
            auto z     = graph::variable<T> (local_num_rays, "z");
            auto t     = graph::variable<T> (local_num_rays, "t");

            t->set(static_cast<T> (0.0));

//  Inital conditions.
            omega->set(static_cast<T> (500.0));
            x->set(static_cast<T> (2.5));
            y->set(static_cast<T> (0.0));
            z->set(static_cast<T> (0.0));
            kx->set(static_cast<T> (-600));
            ky->set(static_cast<T> (0.0));
            kz->set(static_cast<T> (0.0));

            auto eq = equilibrium::make_efit<T> (EFIT_FILE);

            const T endtime = static_cast<T> (1.0);
            const T dt = endtime/static_cast<T> (NUM_TIMES);

            solver::rk4<dispersion::cold_plasma<T>> solve(omega,
                                                          kx, ky, kz,
                                                          x, y, z,
                                                          t, dt,
                                                          eq, "",
                                                          local_num_rays,
                                                          thread_number);

            solve.init(kx);
            solve.compile();

            timing.start_time(thread_number);
            for (size_t j = 0; j < num_steps; j++) {
                for (size_t k = 0; k < SUB_STEPS; k++) {
                    solve.step();
                }
            }
            solve.sync_host();
            timing.end_time(thread_number);
        }, i);
    }

    for (std::thread &t : threads) {
        t.join();
    }
    timing.print();

    std::cout << "--------------------------------------------------------------------------------"
              << std::endl << std::endl;
}

//------------------------------------------------------------------------------
///  @brief Main program of the benchmark.
///
///  @params[in] argc Number of commandline arguments.
///  @params[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    START_GPU
    (void)argc;
    (void)argv;

    bench_runner<float,                1000, 10, 100000> ();
    bench_runner<double,               1000, 10, 100000> ();
    bench_runner<std::complex<float>,  1000, 10, 100000> ();
    bench_runner<std::complex<double>, 1000, 10, 100000> ();

    END_GPU
}
