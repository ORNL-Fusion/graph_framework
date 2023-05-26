//------------------------------------------------------------------------------
///  @file piecewise_test.cpp
///  @brief Tests for piecewise constants nodes.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../graph_framework/arithmetic.hpp"
#include "../graph_framework/piecewise.hpp"
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
    source.print_source();
    
    auto run = source.create_kernel_call("test_kernel", inputs, outputs, 1);
    run();

    T result;
    source.copy_to_host(outputs.back(), &result);

    const T diff = std::abs(result - expected);
    check(diff, tolarance);
}

//------------------------------------------------------------------------------
///  @brief Tests for 1D piecewise nodes.
//------------------------------------------------------------------------------
template<typename T> void piecewise_1D() {
    auto a = graph::variable<T> (1, "");
    auto b = graph::variable<T> (1, "");
    auto p1 = graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (1.0),
                                                       static_cast<T> (2.0),
                                                       static_cast<T> (3.0)}), a);
    auto p2 = graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (2.0),
                                                       static_cast<T> (4.0),
                                                       static_cast<T> (6.0)}), b);
    auto zero = graph::zero<T> ();

    assert(graph::constant_cast(p1*zero).get() &&
           "Expected a constant node.");

    auto two = graph::two<T> ();

    assert(graph::multiply_cast(p1*two).get() &&
           "Expected a multiply node.");
    assert(graph::multiply_cast(p1*p2).get() &&
           "Expected a multiply node.");

    assert(graph::piecewise_1D_cast(p1 + zero).get() &&
           "Expected a piecewise_1D node.");
    assert(graph::add_cast(p1 + two).get() &&
           "Expected an add node.");
    assert(graph::add_cast(p1 + p2).get() &&
           "Expected an add node.");

    assert(graph::piecewise_1D_cast(p1 - zero).get() &&
           "Expected a piecewise_1D node.");
    assert(graph::subtract_cast(p1 - two).get() &&
           "Expected a subtract node.");
    assert(graph::subtract_cast(p1 - p2).get() &&
           "Expected a subtract node.");

    assert(graph::constant_cast(zero/p1).get() &&
           "Expected a constant node.");
    assert(graph::multiply_cast(p1/two).get() &&
           "Expected a multiply node.");
    assert(graph::divide_cast(p1/p2).get() &&
           "Expected a divide node.");

    assert(graph::multiply_cast(graph::fma(p1, two, zero)).get() &&
           "Expected a multiply node.");
    assert(graph::fma_cast(graph::fma(p1, two, p2)).get() &&
           "Expected a fma constant.");
    assert(graph::fma_cast(graph::fma(p1, p2, two)).get() &&
           "Expected a fma node.");

    assert(graph::sqrt_cast(graph::sqrt(p1)).get() &&
           "Expected a sqrt node.");

    assert(graph::exp_cast(graph::exp(p1)).get() &&
           "Expected a exp node.");

    assert(graph::log_cast(graph::log(p1)).get() &&
           "Expected a log node.");

    assert(graph::pow_cast(graph::pow(p1, two)).get() &&
           "Expected a pow node.");
    assert(graph::pow_cast(graph::pow(p1, p2)).get() &&
           "Expected a pow constant.");

    assert(graph::sin_cast(graph::sin(p1)).get() &&
           "Expected a sin node.");

    assert(graph::cos_cast(graph::cos(p1)).get() &&
           "Expected a cos node.");

    assert(graph::divide_cast(graph::tan(p1)).get() &&
           "Expected a divide node.");

    assert(graph::atan_cast(graph::atan(p1, two)).get() &&
           "Expected an atan node.");
    assert(graph::atan_cast(graph::atan(p1, p2)).get() &&
           "Expected a atan constant.");

    a->set(static_cast<T> (1.5));
    compile<T> ({graph::variable_cast(a)},
                {p1}, {},
                static_cast<T> (2.0), 0.0);
    
    a->set(static_cast<T> (0.5));
    compile<T> ({graph::variable_cast(a)},
                {p1}, {},
                static_cast<T> (1.0), 0.0);

    a->set(static_cast<T> (2.5));
    compile<T> ({graph::variable_cast(a)},
                {p1}, {},
                static_cast<T> (3.0), 0.0);
    
    auto pc = graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (10.0),
                                                       static_cast<T> (10.0),
                                                       static_cast<T> (10.0)}), a);
    assert(graph::constant_cast(pc).get() &&
           "Expected a constant.");
}

//------------------------------------------------------------------------------
///  @brief Tests for 2D piecewise nodes.
//------------------------------------------------------------------------------
template<typename T> void piecewise_2D() {
    auto ax = graph::variable<T> (1, "");
    auto ay = graph::variable<T> (1, "");
    auto bx = graph::variable<T> (1, "");
    auto by = graph::variable<T> (1, "");
    auto p1 = graph::piecewise_2D<T> (std::vector<T> ({static_cast<T> (1.0),
                                                       static_cast<T> (2.0),
                                                       static_cast<T> (3.0),
                                                       static_cast<T> (4.0)}),
                                      2, ax, ay);
    auto p2 = graph::piecewise_2D<T> (std::vector<T> ({static_cast<T> (2.0),
                                                       static_cast<T> (4.0),
                                                       static_cast<T> (6.0),
                                                       static_cast<T> (10.0)}),
                                      2, bx, by);

    auto zero = graph::zero<T> ();

    assert(graph::constant_cast(p1*zero).get() &&
           "Expected a constant node.");

    auto two = graph::two<T> ();

    assert(graph::multiply_cast(p1*two).get() &&
           "Expected a multiply node.");
    assert(graph::multiply_cast(p1*p2).get() &&
           "Expected a multiply node.");

    assert(graph::piecewise_2D_cast(p1 + zero).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::add_cast(p1 + two).get() &&
           "Expected an add node.");
    assert(graph::add_cast(p1 + p2).get() &&
           "Expected an add node.");

    assert(graph::piecewise_2D_cast(p1 - zero).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::subtract_cast(p1 - two).get() &&
           "Expected a subtract node.");
    assert(graph::subtract_cast(p1 - p2).get() &&
           "Expected a subtract node.");

    assert(graph::constant_cast(zero/p1).get() &&
           "Expected a constant node.");
    assert(graph::multiply_cast(p1/two).get() &&
           "Expected a multiply node.");
    assert(graph::divide_cast(p1/p2).get() &&
           "Expected a divide node.");

    assert(graph::multiply_cast(graph::fma(p1, two, zero)).get() &&
           "Expected a multiply node.");
    assert(graph::fma_cast(graph::fma(p1, two, p2)).get() &&
           "Expected a fma constant.");
    assert(graph::fma_cast(graph::fma(p1, p2, two)).get() &&
           "Expected a fma node.");

    assert(graph::sqrt_cast(graph::sqrt(p1)).get() &&
           "Expected a sqrt node.");

    assert(graph::exp_cast(graph::exp(p1)).get() &&
           "Expected a exp node.");

    assert(graph::log_cast(graph::log(p1)).get() &&
           "Expected a log node.");

    assert(graph::pow_cast(graph::pow(p1, two)).get() &&
           "Expected a pow node.");
    assert(graph::pow_cast(graph::pow(p1, p2)).get() &&
           "Expected a pow constant.");

    assert(graph::sin_cast(graph::sin(p1)).get() &&
           "Expected a sin node.");

    assert(graph::cos_cast(graph::cos(p1)).get() &&
           "Expected a cos node.");

    assert(graph::divide_cast(graph::tan(p1)).get() &&
           "Expected a divide node.");

    assert(graph::atan_cast(graph::atan(p1, two)).get() &&
           "Expected an atan node.");
    assert(graph::atan_cast(graph::atan(p1, p2)).get() &&
           "Expected a atan constant.");

    ax->set(static_cast<T> (1.5));
    ay->set(static_cast<T> (1.5));
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {p1}, {},
                static_cast<T> (4.0), 0.0);
    
    ax->set(static_cast<T> (0.5));
    ay->set(static_cast<T> (0.5));
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {p1}, {},
                static_cast<T> (1.0), 0.0);
    
    ax->set(static_cast<T> (0.5));
    ay->set(static_cast<T> (1.5));
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {p1}, {},
                static_cast<T> (2.0), 0.0);

    ax->set(static_cast<T> (1.5));
    ay->set(static_cast<T> (0.5));
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {p1}, {},
                static_cast<T> (3.0), 0.0);
    
    auto pc = graph::piecewise_2D<T> (std::vector<T> ({static_cast<T> (10.0),
                                                       static_cast<T> (10.0),
                                                       static_cast<T> (10.0),
                                                       static_cast<T> (10.0)}),
                                      2, ax, bx);
    assert(graph::constant_cast(pc).get() &&
           "Expected a constant.");
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
//------------------------------------------------------------------------------
template<typename T> void run_tests() {
    piecewise_1D<T> ();
    piecewise_2D<T> ();
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @params[in] argc Number of commandline arguments.
///  @params[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    START_GPU
    run_tests<float> ();
    run_tests<double> ();
    run_tests<std::complex<float>> ();
    run_tests<std::complex<double>> ();
    END_GPU
}
