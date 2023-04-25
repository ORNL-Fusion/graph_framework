//------------------------------------------------------------------------------
///  @file piecewise_test.cpp
///  @brief Tests for piecewise constants nodes.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../graph_framework/math.hpp"
#include "../graph_framework/trigonometry.hpp"
#include "../graph_framework/jit.hpp"

//------------------------------------------------------------------------------
///  @brief Assert when difference is greater than the tolarance.
///
///  Specialize to check for complex numbers since complex has not <= operator.
///
///  @params[in] test      Test value.
///  @params[in] tolarance Test tolarance.
//------------------------------------------------------------------------------
template<typename T> void check(const T test,
                                const T tolarance) {
    if constexpr (jit::is_complex<T> ()) {
        assert(std::real(test) <= std::real(tolarance) &&
               "Real GPU and CPU values differ.");
        assert(std::imag(test) <= std::imag(tolarance) &&
               "Imaginary GPU and CPU values differ.");
    } else {
        assert(test <= tolarance && "GPU and CPU values differ.");
    }
}

//------------------------------------------------------------------------------
///  @brief Compile kernal and check the result of the output.
///
///  @params[in] inputs    Kernel input nodes.
///  @params[in] outputs   Kernel output nodes.
///  @params[in] setters   Kernel set nodes.
///  @params[in] expected  Expected result.
///  @params[in] tolarance Check tolarances.
//------------------------------------------------------------------------------
template<typename T> void compile(graph::input_nodes<T> inputs,
                                  graph::output_nodes<T> outputs,
                                  graph::map_nodes<T> setters,
                                  const T expected,
                                  const T tolarance) {
    jit::context<T> source;
    source.add_kernel("test_kernel", inputs, outputs, setters);

    source.compile();
    
    auto run = source.create_kernel_call("test_kernel", inputs, outputs, 1);
    run();

    T result;
    source.copy_to_host(outputs.back(), &result);

    const T diff = std::abs(result - expected);
    check(diff, tolarance);
}

//------------------------------------------------------------------------------
///  @brief Tests for cosine nodes.
//------------------------------------------------------------------------------
template<typename T> void piecewise_1D() {
    auto a = graph::variable<T> (1, "");
    auto p = graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (1.0),
                                                      static_cast<T> (2.0),
                                                      static_cast<T> (3.0)}), a);
    auto zero = graph::zero<T> ();
    
#ifdef USE_REDUCE
    assert(graph::constant_cast(p*zero).get() &&
           "Expected a zero constant.");
#else
    assert(graph::multiply_cast(p*zero).get() &&
           "Expected a multiply node.")
#endif
    
    auto two = graph::two<T> ();

#ifdef USE_REDUCE
    assert(graph::piecewise_1D_cast(p*two).get() &&
           "Expected a piecewise constant.");
#else
    assert(graph::multiply_cast(p*two).get() &&
           "Expected a multiply node.")
#endif
    
#ifdef USE_REDUCE
    assert(graph::piecewise_1D_cast(p + two).get() &&
           "Expected a piecewise constant.");
#else
    assert(graph::add_cast(p + two).get() &&
           "Expected a add node.")
#endif

#ifdef USE_REDUCE
    assert(graph::piecewise_1D_cast(p - two).get() &&
           "Expected a piecewise constant.");
#else
    assert(graph::subtract_cast(p - two).get() &&
           "Expected a subtract node.")
#endif

#ifdef USE_REDUCE
    assert(graph::piecewise_1D_cast(p/two).get() &&
           "Expected a piecewise constant.");
#else
    assert(graph::divide_cast(p/two).get() &&
           "Expected a divide node.")
#endif
    
#ifdef USE_REDUCE
    assert(graph::piecewise_1D_cast(graph::fma(p, two, zero)).get() &&
           "Expected a piecewise constant.");
#else
    assert(graph::fma_cast(graph::fma(p, two, zero)).get() &&
           "Expected a fma node.")
#endif

#ifdef USE_REDUCE
    assert(graph::piecewise_1D_cast(graph::sqrt(p)).get() &&
           "Expected a piecewise constant.");
#else
    assert(graph::sqrt_cast(graph::sqrt(p)).get() &&
           "Expected a sqrt node.")
#endif

#ifdef USE_REDUCE
    assert(graph::piecewise_1D_cast(graph::exp(p)).get() &&
           "Expected a piecewise constant.");
#else
    assert(graph::exp_cast(graph::exp(p)).get() &&
           "Expected a exp node.")
#endif

#ifdef USE_REDUCE
    assert(graph::piecewise_1D_cast(graph::log(p)).get() &&
           "Expected a piecewise constant.");
#else
    assert(graph::log_cast(graph::log(p)).get() &&
           "Expected a log node.")
#endif

#ifdef USE_REDUCE
    assert(graph::piecewise_1D_cast(graph::pow(p, two)).get() &&
           "Expected a piecewise constant.");
#else
    assert(graph::pow_cast(graph::pow(p, two)).get() &&
           "Expected a pow node.")
#endif
    
#ifdef USE_REDUCE
    assert(graph::piecewise_1D_cast(graph::sin(p)).get() &&
           "Expected a piecewise constant.");
#else
    assert(graph::sin_cast(graph::sin(p)).get() &&
           "Expected a sin node.")
#endif

#ifdef USE_REDUCE
    assert(graph::piecewise_1D_cast(graph::cos(p)).get() &&
           "Expected a piecewise constant.");
#else
    assert(graph::cos_cast(graph::cos(p)).get() &&
           "Expected a cos node.")
#endif
    
    a->set(static_cast<T> (1.5));
    compile<T> ({graph::variable_cast(a)},
                {p}, {},
                static_cast<T> (2.0), 0.0);
    
    a->set(static_cast<T> (0.5));
    compile<T> ({graph::variable_cast(a)},
                {p}, {},
                static_cast<T> (1.0), 0.0);

    a->set(static_cast<T> (2.5));
    compile<T> ({graph::variable_cast(a)},
                {p}, {},
                static_cast<T> (3.0), 0.0);
    
    auto pc = graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (10.0),
                                                       static_cast<T> (10.0),
                                                       static_cast<T> (10.0)}), a);
#ifdef USE_REDUCE
    assert(graph::constant_cast(pc).get() &&
           "Expected a constant node.");
#else
    assert(graph::piecewise_1D_cast(pc).get() &&
           "Expected a piecewise constant.");
#endif
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
//------------------------------------------------------------------------------
template<typename T> void run_tests() {
    piecewise_1D<T> ();
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @params[in] argc Number of commandline arguments.
///  @params[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    run_tests<float> ();
    run_tests<double> ();
    run_tests<std::complex<float>> ();
    run_tests<std::complex<double>> ();
}
