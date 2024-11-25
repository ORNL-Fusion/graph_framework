//------------------------------------------------------------------------------
///  @file piecewise\_test.cpp
///  @brief Tests for piecewise constants nodes.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cmath>
#include <cassert>

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
///  @tparam T Base type of the calculation.
///
///  @param[in] test      Test value.
///  @param[in] tolarance Test tolarance.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void check(const T test,
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
///  @tparam T Base type of the calculation.
///
///  @param[in] inputs    Kernel input nodes.
///  @param[in] outputs   Kernel output nodes.
///  @param[in] setters   Kernel set nodes.
///  @param[in] expected  Expected result.
///  @param[in] tolarance Check tolarances.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void compile(graph::input_nodes<T> inputs,
                                           graph::output_nodes<T> outputs,
                                           graph::map_nodes<T> setters,
                                           const T expected,
                                           const T tolarance) {
    jit::context<T> source(0);
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
///  @brief Tests for 1D piecewise nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void piecewise_1D() {
    auto a = graph::variable<T> (1, "");
    auto b = graph::variable<T> (1, "");
    auto p1 = graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (1.0),
                                                       static_cast<T> (2.0),
                                                       static_cast<T> (3.0)}), a);
    auto p2 = graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (2.0),
                                                       static_cast<T> (4.0),
                                                       static_cast<T> (6.0)}), b);
    auto p3 = graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (2.0),
                                                       static_cast<T> (4.0),
                                                       static_cast<T> (6.0)}), a);

    assert(graph::constant_cast(p1*0.0).get() &&
           "Expected a constant node.");

    assert(graph::piecewise_1D_cast(p1*2.0).get() &&
           "Expected a piecewise_1D node.");
    assert(graph::multiply_cast(p1*p2).get() &&
           "Expected a multiply node.");
    assert(graph::piecewise_1D_cast(p1*p3).get() &&
           "Expected a piecewise_1D node.");

    assert(graph::piecewise_1D_cast(p1 + 0.0).get() &&
           "Expected a piecewise_1D node.");
    assert(graph::piecewise_1D_cast(p1 + 2.0).get() &&
           "Expected a piecewise_1D node.");
    assert(graph::add_cast(p1 + p2).get() &&
           "Expected an add node.");
    assert(graph::piecewise_1D_cast(p1 + p3).get() &&
           "Expected a piecewise_1D node.");

    assert(graph::piecewise_1D_cast(p1 - 0.0).get() &&
           "Expected a piecewise_1D node.");
    assert(graph::piecewise_1D_cast(p1 - 2.0).get() &&
           "Expected a piecewise_1D node.");
    assert(graph::subtract_cast(p1 - p2).get() &&
           "Expected a subtract node.");
    assert(graph::piecewise_1D_cast(p1 - p3).get() &&
           "Expected a piecewise_1D node.");

    assert(graph::constant_cast(0.0/p1).get() &&
           "Expected a constant node.");
    assert(graph::piecewise_1D_cast(p1/2.0).get() &&
           "Expected a piecewise_1D node.");
    assert(graph::multiply_cast(p1/p2).get() &&
           "Expected a multiply node.");
    assert(graph::constant_cast(p1/p3).get() &&
           "Expected a constant node.");

    assert(graph::piecewise_1D_cast(graph::fma(p1, 2.0, 0.0)).get() &&
           "Expected a piecewise_1D node.");
    assert(graph::add_cast(graph::fma(p1, 2.0, p2)).get() &&
           "Expected an add node.");
    auto temp = graph::fma(p1, p2, 2.0);
    assert(graph::multiply_cast(graph::fma(p1, p2, 2.0)).get() &&
           "Expected a multiply node.");
    assert(graph::add_cast(graph::fma(p1, p3, p2)).get() &&
           "Expected an add node.");
    assert(graph::piecewise_1D_cast(graph::fma(p1, p3, 2.0)).get() &&
           "Expected a piecewise_1D node.");
    assert(graph::piecewise_1D_cast(graph::fma(p1, p3, p1)).get() &&
           "Expected a piecewise_1D node.");

    assert(graph::piecewise_1D_cast(graph::sqrt(p1)).get() &&
           "Expected a piecewise_1D node.");

    assert(graph::piecewise_1D_cast(graph::exp(p1)).get() &&
           "Expected a piecewise_1D node.");

    assert(graph::piecewise_1D_cast(graph::log(p1)).get() &&
           "Expected a piecewise_1D node.");

    assert(graph::piecewise_1D_cast(graph::pow(p1, 2.0)).get() &&
           "Expected a piecewise_1D node.");
    assert(graph::pow_cast(graph::pow(p1, p2)).get() &&
           "Expected a pow constant.");
    assert(graph::piecewise_1D_cast(graph::pow(p1, p3)).get() &&
           "Expected a piecewise_1D node.");

    assert(graph::piecewise_1D_cast(graph::sin(p1)).get() &&
           "Expected a piecewise_1D node.");

    assert(graph::piecewise_1D_cast(graph::cos(p1)).get() &&
           "Expected a piecewise_1D node.");

    assert(graph::piecewise_1D_cast(graph::tan(p1)).get() &&
           "Expected a piecewise_1D node.");

    assert(graph::piecewise_1D_cast(graph::atan(p1, 2.0)).get() &&
           "Expected a piecewise_1D node.");
    assert(graph::atan_cast(graph::atan(p1, p2)).get() &&
           "Expected an atan node.");
    assert(graph::constant_cast(graph::atan(p1, p3)).get() &&
           "Expected a constant node.");

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

    a->set(static_cast<T> (1.5));
    compile<T> ({graph::variable_cast(a)},
                {p1 + p3}, {},
                static_cast<T> (6.0), 0.0);
    compile<T> ({graph::variable_cast(a)},
                {p1 - p3}, {},
                static_cast<T> (-2.0), 0.0);
    compile<T> ({graph::variable_cast(a)},
                {p1*p3}, {},
                static_cast<T> (8.0), 0.0);
    compile<T> ({graph::variable_cast(a)},
                {p1/p3}, {},
                static_cast<T> (0.5), 0.0);
    compile<T> ({graph::variable_cast(a),
                 graph::variable_cast(b)},
                {graph::fma(p1, p3, p2)}, {},
                static_cast<T> (10.0), 0.0);
    if constexpr (jit::is_complex<T> ()) {
        compile<T> ({graph::variable_cast(a)},
                    {graph::pow(p1, p3)}, {},
                    static_cast<T> (16.0), 2.0E-15);
    } else {
        compile<T> ({graph::variable_cast(a)},
                    {graph::pow(p1, p3)}, {},
                    static_cast<T> (16.0), 0.0);
    }
    if constexpr (jit::is_complex<T> ()) {
        compile<T> ({graph::variable_cast(a)},
                    {graph::atan(p1, p3)}, {},
                    static_cast<T> (std::atan(static_cast<T> (4.0) /
                                              static_cast<T> (2.0))),
                    0.0);
    } else {
        compile<T> ({graph::variable_cast(a)},
                    {graph::atan(p1, p3)}, {},
                    static_cast<T> (std::atan2(static_cast<T> (4.0),
                                               static_cast<T> (2.0))),
                    0.0);
    }

    auto pc = graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (10.0),
                                                       static_cast<T> (10.0),
                                                       static_cast<T> (10.0)}), a);
    assert(graph::constant_cast(pc).get() &&
           "Expected a constant.");
}

//------------------------------------------------------------------------------
///  @brief Tests for 2D piecewise nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void piecewise_2D() {
    auto ax = graph::variable<T> (1, "");
    auto ay = graph::variable<T> (1, "");
    auto bx = graph::variable<T> (1, "");
    auto by = graph::variable<T> (1, "");
    auto p1 = graph::piecewise_2D<T> (std::vector<T> ({
        static_cast<T> (1.0), static_cast<T> (2.0),
        static_cast<T> (3.0), static_cast<T> (4.0)
    }), 2, ax, ay);
    auto p2 = graph::piecewise_2D<T> (std::vector<T> ({
        static_cast<T> (2.0), static_cast<T> (4.0),
        static_cast<T> (6.0), static_cast<T> (10.0)
    }), 2, bx, by);
    auto p3 = graph::piecewise_2D<T> (std::vector<T> ({
        static_cast<T> (2.0), static_cast<T> (4.0),
        static_cast<T> (6.0), static_cast<T> (10.0)
    }), 2, ax, ay);
    auto p4 = graph::piecewise_1D<T> (std::vector<T> ({
        static_cast<T> (2.0), static_cast<T> (4.0)
    }),  ax);
    auto p5 = graph::piecewise_1D<T> (std::vector<T> ({
        static_cast<T> (2.0), static_cast<T> (4.0)
    }), ay);

    assert(graph::constant_cast(p1*0.0).get() &&
           "Expected a constant node.");

    assert(graph::piecewise_2D_cast(p1*2.0).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::multiply_cast(p1*p2).get() &&
           "Expected a multiply node.");
    assert(graph::piecewise_2D_cast(p1*p3).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::piecewise_2D_cast(p1*p4).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::piecewise_2D_cast(p1*p5).get() &&
           "Expected a piecewise_2D node.");

    assert(graph::piecewise_2D_cast(p1 + 0.0).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::piecewise_2D_cast(p1 + 2.0).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::add_cast(p1 + p2).get() &&
           "Expected an add node.");
    assert(graph::piecewise_2D_cast(p1 + p3).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::piecewise_2D_cast(p1 + p4).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::piecewise_2D_cast(p1 + p5).get() &&
           "Expected a piecewise_2D node.");

    assert(graph::piecewise_2D_cast(p1 - 0.0).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::piecewise_2D_cast(p1 - 2.0).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::subtract_cast(p1 - p2).get() &&
           "Expected a subtract node.");
    assert(graph::piecewise_2D_cast(p1 - p3).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::piecewise_2D_cast(p1 - p4).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::piecewise_2D_cast(p1 - p5).get() &&
           "Expected a piecewise_2D node.");

    assert(graph::constant_cast(0.0/p1).get() &&
           "Expected a constant node.");
    assert(graph::piecewise_2D_cast(p1/2.0).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::multiply_cast(p1/p2).get() &&
           "Expected a multiply node.");
    assert(graph::piecewise_2D_cast(p1/p3).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::piecewise_2D_cast(p1/p4).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::piecewise_2D_cast(p1/p5).get() &&
           "Expected a piecewise_2D node.");

    assert(graph::piecewise_2D_cast(graph::fma(p1, 2.0, 0.0)).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::add_cast(graph::fma(p1, 2.0, p2)).get() &&
           "Expected an add node.");
    assert(graph::multiply_cast(graph::fma(p1, p2, 2.0)).get() &&
           "Expected a multiply node.");
    assert(graph::add_cast(graph::fma(p1, p3, p2)).get() &&
           "Expected an add node.");
    assert(graph::piecewise_2D_cast(graph::fma(p1, p3, 2.0)).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::piecewise_2D_cast(graph::fma(p1, p3, p1)).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::add_cast(graph::fma(p1, p4, p2)).get() &&
           "Expected an add node.");
    assert(graph::add_cast(graph::fma(p1, p5, p2)).get() &&
           "Expected an add node.");

    assert(graph::piecewise_2D_cast(graph::sqrt(p1)).get() &&
           "Expected a piecewise_2D node.");

    assert(graph::piecewise_2D_cast(graph::exp(p1)).get() &&
           "Expected a piecewise_2D node.");

    assert(graph::piecewise_2D_cast(graph::log(p1)).get() &&
           "Expected a piecewise_2D node.");

    assert(graph::piecewise_2D_cast(graph::pow(p1, 2.0)).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::pow_cast(graph::pow(p1, p2)).get() &&
           "Expected a pow node.");
    assert(graph::piecewise_2D_cast(graph::pow(p1, p3)).get() &&
           "Expected a pow node.");
    assert(graph::piecewise_2D_cast(graph::pow(p1, p4)).get() &&
           "Expected a piecewise_2D node.");
    assert(graph::piecewise_2D_cast(graph::pow(p1, p5)).get() &&
           "Expected a piecewise_2D node.");

    assert(graph::piecewise_2D_cast(graph::sin(p1)).get() &&
           "Expected a piecewise_2D node.");

    assert(graph::piecewise_2D_cast(graph::cos(p1)).get() &&
           "Expected a piecewise_2D node.");

    assert(graph::piecewise_2D_cast(graph::tan(p1)).get() &&
           "Expected a piecewise_2D node.");

    assert(graph::piecewise_2D_cast(graph::atan(p1, 2.0)).get() &&
           "Expected a piecewise_2d node.");
    assert(graph::atan_cast(graph::atan(p1, p2)).get() &&
           "Expected an atan node.");
    assert(graph::piecewise_2D_cast(graph::atan(p1, p3)).get() &&
           "Expected a piecewise_2d node.");
    assert(graph::piecewise_2D_cast(graph::atan(p1, p4)).get() &&
           "Expected a piecewise_2d node.");
    assert(graph::piecewise_2D_cast(graph::atan(p1, p5)).get() &&
           "Expected a piecewise_2d node.");

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

    ax->set(static_cast<T> (0.5));
    ay->set(static_cast<T> (1.5));
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {p1 + p3}, {},
                static_cast<T> (6.0), 0.0);
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {p1 - p3}, {},
                static_cast<T> (-2.0), 0.0);
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {p1*p3}, {},
                static_cast<T> (8.0), 0.0);
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {p1/p3}, {},
                static_cast<T> (0.5), 0.0);
    bx->set(static_cast<T> (1.5));
    by->set(static_cast<T> (0.5));
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay),
                 graph::variable_cast(bx),
                 graph::variable_cast(by)},
                {graph::fma(p1, p3, p2)}, {},
                static_cast<T> (14.0), 0.0);
    if constexpr (jit::is_complex<T> ()) {
        compile<T> ({graph::variable_cast(ax),
                     graph::variable_cast(ay)},
                    {graph::pow(p1, p3)}, {},
                     static_cast<T> (16.0), 2.0E-15);
    } else {
        compile<T> ({graph::variable_cast(ax),
                     graph::variable_cast(ay)},
                    {graph::pow(p1, p3)}, {},
                     static_cast<T> (16.0), 0.0);
    }
    if constexpr (jit::is_complex<T> ()) {
        compile<T> ({graph::variable_cast(ax),
                     graph::variable_cast(ay)},
                    {graph::atan(p1, p3)}, {},
                    static_cast<T> (std::atan(static_cast<T> (4.0) /
                                              static_cast<T> (2.0))),
                    0.0);
    } else {
        compile<T> ({graph::variable_cast(ax),
                     graph::variable_cast(ay)},
                    {graph::atan(p1, p3)}, {},
                    static_cast<T> (std::atan2(static_cast<T> (4.0),
                                               static_cast<T> (2.0))),
                    0.0);
    }

//  Test row combines.
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {p1}, {},
                static_cast<T> (2.0), 0.0);
    compile<T> ({graph::variable_cast(ax)},
                {p4}, {},
                static_cast<T> (2.0), 0.0);
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {p1 + p4}, {},
                static_cast<T> (4.0), 0.0);
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {p1 - p4}, {},
                static_cast<T> (0.0), 0.0);
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {p1*p4}, {},
                static_cast<T> (4.0), 0.0);
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {p1/p4}, {},
                static_cast<T> (1.0), 0.0);
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay),
                 graph::variable_cast(bx),
                 graph::variable_cast(by)},
                {graph::fma(p1, p4, p2)}, {},
                static_cast<T> (10.0), 0.0);
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {graph::pow(p1, p4)}, {},
                static_cast<T> (std::pow(static_cast<T> (2.0),
                                         static_cast<T> (2.0))), 0.0);
    if constexpr (jit::is_complex<T> ()) {
        compile<T> ({graph::variable_cast(ax),
                     graph::variable_cast(ay)},
                    {graph::atan(p1, p4)}, {},
                    static_cast<T> (std::atan(static_cast<T> (2.0) /
                                              static_cast<T> (2.0))),
                    0.0);
    } else {
        compile<T> ({graph::variable_cast(ax),
                     graph::variable_cast(ay)},
                    {graph::atan(p1, p4)}, {},
                    static_cast<T> (std::atan2(static_cast<T> (2.0),
                                               static_cast<T> (2.0))),
                    0.0);
    }

//  Test column combines.
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {p1 + p5}, {},
                static_cast<T> (6.0), 0.0);
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {p1 - p5}, {},
                static_cast<T> (-2.0), 0.0);
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {p1*p5}, {},
                static_cast<T> (8.0), 0.0);
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {p1/p5}, {},
                static_cast<T> (0.5), 0.0);
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay),
                 graph::variable_cast(bx),
                 graph::variable_cast(by)},
                {graph::fma(p1, p5, p2)}, {},
                static_cast<T> (14.0), 0.0);
    if constexpr (jit::is_complex<T> ()) {
        compile<T> ({graph::variable_cast(ax),
                     graph::variable_cast(ay)},
                    {graph::pow(p1, p5)}, {},
                    static_cast<T> (16.0), 2.0E-15);
    } else {
        compile<T> ({graph::variable_cast(ax),
                     graph::variable_cast(ay)},
                    {graph::pow(p1, p5)}, {},
                    static_cast<T> (16.0), 0.0);
    }
    if constexpr (jit::is_complex<T> ()) {
        compile<T> ({graph::variable_cast(ax),
                     graph::variable_cast(ay)},
                    {graph::atan(p1, p5)}, {},
                    static_cast<T> (std::atan(static_cast<T> (4.0) /
                                              static_cast<T> (2.0))),
                    0.0);
    } else {
        compile<T> ({graph::variable_cast(ax),
                     graph::variable_cast(ay)},
                    {graph::atan(p1, p5)}, {},
                    static_cast<T> (std::atan2(static_cast<T> (4.0),
                                               static_cast<T> (2.0))),
                    0.0);
    }

    auto pc = graph::piecewise_2D<T> (std::vector<T> ({static_cast<T> (10.0),
                                                       static_cast<T> (10.0),
                                                       static_cast<T> (10.0),
                                                       static_cast<T> (10.0)}),
                                      2, ax, bx);
    assert(graph::constant_cast(pc).get() &&
           "Expected a constant.");

    auto prc = graph::piecewise_1D<T> (std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0),
        static_cast<T> (3.0)
    }), ax);
    auto pcc = graph::piecewise_1D<T> (std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0),
        static_cast<T> (3.0)
    }), ay);
    auto p2Dc = graph::piecewise_2D<T> (std::vector<T> ({
        static_cast<T> (1.0), static_cast<T> (2.0), 
        static_cast<T> (3.0), static_cast<T> (4.0),
        static_cast<T> (5.0), static_cast<T> (6.0)
    }), 2, ax, ay);

    auto row_test = prc + p2Dc;
    auto row_test_cast = graph::piecewise_2D_cast(row_test);
    assert(row_test_cast.get() && "Expected a 2D piecewise node..");

    auto col_test = pcc + p2Dc;
    auto col_test_cast = graph::add_cast(col_test);
    assert(col_test_cast.get() && "Expected an add node.");

    ax->set(static_cast<T> (2.5));
    ay->set(static_cast<T> (1.5));
    compile<T> ({graph::variable_cast(ax)},
                {prc}, {},
                static_cast<T> (3.0), 0.0);
    compile<T> ({graph::variable_cast(ay)},
                {pcc}, {},
                static_cast<T> (2.0), 0.0);
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {p2Dc}, {},
                static_cast<T> (6.0), 0.0);
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {row_test}, {},
                static_cast<T> (9.0), 0.0);
    compile<T> ({graph::variable_cast(ax),
                 graph::variable_cast(ay)},
                {col_test}, {},
                static_cast<T> (8.0), 0.0);
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void run_tests() {
    piecewise_1D<T> ();
    piecewise_2D<T> ();
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
    run_tests<std::complex<float>> ();
    run_tests<std::complex<double>> ();
    END_GPU
}
