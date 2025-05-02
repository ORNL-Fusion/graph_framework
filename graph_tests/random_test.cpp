//------------------------------------------------------------------------------
///  @file random\_test.cpp
///  @brief Tests for random nodes.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../graph_framework/jit.hpp"
#include "../graph_framework/workflow.hpp"
#include "../graph_framework/piecewise.hpp"
#include "../graph_framework/arithmetic.hpp"
#include "../graph_framework/math.hpp"
#include "../graph_framework/trigonometry.hpp"
#include "../graph_framework/timing.hpp"

//------------------------------------------------------------------------------
///  @brief Compute the auto correlation for a specific offset.
///
///  @tparam T Base type.
///
///  @param[in] sequence Random sequence.
///  @param[in] offset   Offset of the correlation.
///  @returns The autocorrelation for a given offset.
//------------------------------------------------------------------------------
template<typename T>
T autocorrelation(const std::vector<T> &sequence,
                  const size_t offset) {
    T result = 0.0;
    for (size_t i = 0, ie = sequence.size() - offset; i < ie; i++) {
        result += sequence[i]*sequence[offset + i];
    }
    return result/static_cast<T> (sequence.size() - offset);
}

//------------------------------------------------------------------------------
///  @brief Run test with a specified backend.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T, size_t N> void run_test() {
    auto state = graph::random_state<T> (jit::context<T>::random_state_size, 0);
    auto random = graph::random<T> (graph::random_state_cast(state));
    const T max = 1.0;
    const T min = -1.0;
    auto random_real = (max - min)/graph::random_scale<T> ()*random + min;

    workflow::manager<T> work(0);
    work.add_item({}, {random_real}, {}, graph::random_state_cast(state),
                  "step", N);
    work.compile();
    work.run();

    std::vector<T> result(N);
    work.copy_to_host(random_real, result.data());

    const T base = autocorrelation(result, 0)*static_cast<T> (0.015);
    for (size_t i = 1, ie = N/10; i < ie; i++) {
        const T test = autocorrelation(result, i);
        assert(std::abs(test) < std::abs(base) && "Auto correlation failure.");
    }
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

    run_test<float, 1000000> ();
    run_test<double, 1000000> ();
    run_test<std::complex<float>, 1000000> ();
    run_test<std::complex<double>, 1000000> ();

    END_GPU
}
