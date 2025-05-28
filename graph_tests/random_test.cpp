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
///  @tparam N Number of random numbers to use.
//------------------------------------------------------------------------------
template<jit::float_scalar T, size_t N> void test_dist() {
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
///  @brief Test graph properties of random numbers.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_graph() {
    auto state = graph::random_state<T> (jit::context<T>::random_state_size, 0);
    auto random = graph::random<T> (graph::random_state_cast(state));

//  r + r -> r + r
    assert(graph::add_cast(random + random).get() && "Expected add node.");
//  r + 0.0 -> r
    assert(graph::random_cast(random + 0.0).get() && "Expected random node.");
//  r - r -> r - r
    assert(graph::subtract_cast(random - random).get() &&
           "Expected subtract node.");
//  r - 0.0 -> r
    assert(graph::random_cast(random - 0.0).get() && "Expected random node.");
//  r*r -> r*r
    assert(graph::multiply_cast(random*random).get() &&
           "Expected multiply node.");
//  1*r -> r
    assert(graph::random_cast(1.0*random).get() && "Expected random node.");
//  r/r -> r/r
    assert(graph::divide_cast(random/random).get() && "Expected divide node.");
//  r/1 -> r
    assert(graph::random_cast(random/1.0).get() && "Expected random node.");
//  fma(r,r,1) -> fma(r,r,1)
    assert(graph::fma_cast(graph::fma(random, random, 1.0)).get() &&
           "Expected fma node.");
//  fma(r,2,r) -> fma(2,r,r)
    assert(graph::fma_cast(graph::fma(random, 2.0, random)).get() &&
           "Expected fma node.");
//  fma(2.0,r,r) -> fma(2.0,r,r)
    assert(graph::fma_cast(graph::fma(2.0, random, random)).get() &&
           "Expected fma node.");
//  fma(r,r,0.0) -> r*r
    assert(graph::multiply_cast(graph::fma(random, random, 0.0)).get() &&
           "Expected multiply node.");
//  fma(r,1.0,r) -> r + r
    assert(graph::add_cast(graph::fma(random, 1.0, random)).get() &&
           "Expected add node.");
//  sqrt(r) -> sqrt(r)
    assert(graph::sqrt_cast(graph::sqrt(random)).get() &&
           "Expected sqrt node.");
//  exp(r) -> exp(r)
    assert(graph::exp_cast(graph::exp(random)).get() &&
           "Expected exp node.");
//  ln(r) -> ln(r)
    assert(graph::log_cast(graph::log(random)).get() &&
           "Expected log node.");
//  pow(r,r) -> pow(r,r)
    assert(graph::pow_cast(graph::pow(random, random)).get() &&
           "Expected pow node.");
//  pow(r,1) -> r
    assert(graph::random_cast(graph::pow(random, 1.0)).get() &&
           "Expected random node.");

    if constexpr(jit::complex_scalar<T>) {
//  efi(r) -> efi(r)
        assert(graph::erfi_cast(graph::erfi(random)).get() &&
               "Expected erfi node.");
    }
//  sin(r) -> sin(r)
    assert(graph::sin_cast(graph::sin(random)).get() &&
           "Expected sin node.");
//  cos(r) -> cos(r)
    assert(graph::cos_cast(graph::cos(random)).get() &&
           "Expected cos node.");
//  atan(r,r) -> atan(r,r)
    assert(graph::atan_cast(graph::atan(random, random)).get() &&
           "Expected atan node.");
}

//------------------------------------------------------------------------------
///  @brief Run tests.
///
///  @tparam T Base type of the calculation.
///  @tparam N Number of random numbers to use.
//------------------------------------------------------------------------------
template<jit::float_scalar T, size_t N> void run_tests() {
    test_dist<T, N> ();
    test_graph<T> ();
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

    run_tests<float, 1000000> ();
    run_tests<double, 1000000> ();
    run_tests<std::complex<float>, 1000000> ();
    run_tests<std::complex<double>, 1000000> ();

    END_GPU
}
