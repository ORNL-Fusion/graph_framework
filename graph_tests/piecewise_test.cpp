//------------------------------------------------------------------------------
///  @file piecewise_test.cpp
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
///  @brief Assert when difference is greater than the tolerance.
///
///  Specialize to check for complex numbers since complex has not <= operator.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] test      Test value.
///  @param[in] tolerance Test tolerance.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void check(const T test,
                                         const T tolerance) {
    if constexpr (jit::complex_scalar<T>) {
        assert(std::real(test) <= std::real(tolerance) &&
               "Real GPU and CPU values differ.");
        assert(std::imag(test) <= std::imag(tolerance) &&
               "Imaginary GPU and CPU values differ.");
    } else {
        assert(test <= tolerance && "GPU and CPU values differ.");
    }
}

//------------------------------------------------------------------------------
///  @brief Compile kernel and check the result of the output.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] inputs    Kernel input nodes.
///  @param[in] outputs   Kernel output nodes.
///  @param[in] setters   Kernel set nodes.
///  @param[in] expected  Expected result.
///  @param[in] tolerance Check tolerances.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void compile(graph::input_nodes<T> inputs,
                                           graph::output_nodes<T> outputs,
                                           graph::map_nodes<T> setters,
                                           const T expected,
                                           const T tolerance) {
    jit::context<T> source(0);
    source.add_kernel("test_kernel", inputs, outputs, setters,
                      graph::shared_random_state<T> (), inputs.back()->size());

    source.compile();

    auto run = source.create_kernel_call("test_kernel", inputs, outputs,
                                         graph::shared_random_state<T> (), 1);
    run();

    T result;
    source.copy_to_host(outputs.back(), &result);

    const T diff = std::abs(result - expected);
    check(diff, tolerance);
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
                                                       static_cast<T> (3.0)}),
                                      a, 1.0, 0.0);
    auto p2 = graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (2.0),
                                                       static_cast<T> (4.0),
                                                       static_cast<T> (6.0)}),
                                      b, 1.0, 0.0);
    auto p3 = graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (2.0),
                                                       static_cast<T> (4.0),
                                                       static_cast<T> (6.0)}),
                                      a, 1.0, 0.0);

    auto c = graph::constant<T> (static_cast<T> (2.5));
    auto pconst = graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (2.0),
                                                           static_cast<T> (4.0),
                                                           static_cast<T> (6.0)}),
                                          c, 1.0, 0.0);
    auto pc_cast = constant_cast(pconst);
    assert(pc_cast.get() && "Expected a constant node.");
    assert(pc_cast->is(6.0) && "Expected a value of 6");

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
    if constexpr (jit::complex_scalar<T>) {
        compile<T> ({graph::variable_cast(a)},
                    {graph::pow(p1, p3)}, {},
                    static_cast<T> (16.0), 2.0E-15);
    } else {
        compile<T> ({graph::variable_cast(a)},
                    {graph::pow(p1, p3)}, {},
                    static_cast<T> (16.0), 0.0);
    }
    if constexpr (jit::complex_scalar<T>) {
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
                                                       static_cast<T> (10.0)}),
                                      a, 1.0, 0.0);
    assert(graph::constant_cast(pc).get() &&
           "Expected a constant.");

//  fma(p1,c1 + a,p2) -> fma(p1,a,p3)
    auto fma_combine = fma(p1,1.0 + a,p3);
    auto fma_combine_cast = graph::fma_cast(fma_combine);
    assert(fma_combine_cast.get() && "Expected an fma node.");
    assert(fma_combine_cast->get_middle()->is_match(a) &&
           "Expected a in the middle.");
    assert(fma_combine_cast->get_left()->is_match(p1) &&
           "Expected p1 on the left.");
    assert(fma_combine_cast->get_right()->is_match(p1 + p3) &&
           "Expected p1 + p3 on the right.");
//  fma(p1,c1 - a,p2) -> p3 - p1*a
    auto fma_combine2 = fma(p1,1.0 - a,p3);
    auto fma_combine2_cast = graph::fma_cast(fma_combine2);
    assert(fma_combine2_cast.get() && "Expected a fma node.");
    assert(fma_combine2_cast->get_left()->is_match(-p1) &&
           "Expected -p1 on the left.");
    assert(fma_combine2_cast->get_middle()->is_match(a) &&
           "Expected a in the middle.");
    assert(fma_combine2_cast->get_right()->is_match(p1 + p3) &&
           "Expected p1 + p3 on the right.");
//  p1*(c1 + a) - p2 -> fma(p1,a,p3)
    auto fma_combine3 = p1*(1.0 + a) - p3;
    auto fma_combine3_cast = graph::fma_cast(fma_combine3);
    assert(fma_combine3_cast.get() && "Expected a fma node.");
    assert(fma_combine3_cast->get_middle()->is_match(a) &&
           "Expected a in the middle.");
    assert(fma_combine3_cast->get_left()->is_match(p1) &&
           "Expected p1 on the left.");
    assert(fma_combine3_cast->get_right()->is_match(p1 - p3) &&
           "Expected p1 - p3 on the right.");
//  p1*(c1 - a) - p2 -> p3 - p1*a
    auto fma_combine4 = p1*(1.0 - a) - p3;
    auto fma_combine4_cast = graph::subtract_cast(fma_combine4);
    assert(fma_combine4_cast.get() && "Expected an subtract node.");
    assert(fma_combine4_cast->get_right()->is_match(p1*a) &&
           "Expected p1*a on the right.");
    assert(fma_combine4_cast->get_left()->is_match(p1 - p3) &&
           "Expected p1 - p3 on the left.");
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
    }), 2, ax, 1.0, 0.0, ay, 1.0, 0.0);
    auto p2 = graph::piecewise_2D<T> (std::vector<T> ({
        static_cast<T> (2.0), static_cast<T> (4.0),
        static_cast<T> (6.0), static_cast<T> (10.0)
    }), 2, bx, 1.0, 0.0, by, 1.0, 0.0);
    auto p3 = graph::piecewise_2D<T> (std::vector<T> ({
        static_cast<T> (2.0), static_cast<T> (4.0),
        static_cast<T> (6.0), static_cast<T> (10.0)
    }), 2, ax, 1.0, 0.0, ay, 1.0, 0.0);
    auto p4 = graph::piecewise_1D<T> (std::vector<T> ({
        static_cast<T> (2.0), static_cast<T> (4.0)
    }),  ax, 1.0, 0.0);
    auto p5 = graph::piecewise_1D<T> (std::vector<T> ({
        static_cast<T> (2.0), static_cast<T> (4.0)
    }), ay, 1.0, 0.0);

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
    if constexpr (jit::complex_scalar<T>) {
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
    if constexpr (jit::complex_scalar<T>) {
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
    if constexpr (jit::complex_scalar<T>) {
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
    if constexpr (jit::complex_scalar<T>) {
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
    if constexpr (jit::complex_scalar<T>) {
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
                                      2, ax, 1.0, 0.0, bx, 1.0, 0.0);
    assert(graph::constant_cast(pc).get() &&
           "Expected a constant.");

    auto prc = graph::piecewise_1D<T> (std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0),
        static_cast<T> (3.0)
    }), ax, 1.0, 0.0);
    auto pcc = graph::piecewise_1D<T> (std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0),
        static_cast<T> (3.0)
    }), ay, 1.0, 0.0);
    auto p2Dc = graph::piecewise_2D<T> (std::vector<T> ({
        static_cast<T> (1.0), static_cast<T> (2.0), 
        static_cast<T> (3.0), static_cast<T> (4.0),
        static_cast<T> (5.0), static_cast<T> (6.0)
    }), 2, ax, 1.0, 0.0, ay, 1.0, 0.0);

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

//  fma(p1,c1 + a,p2) -> fma(p1,a,p3)
    auto fma_combine = fma(p1,1.0 + ax,p3);
    auto fma_combine_cast = graph::fma_cast(fma_combine);
    assert(fma_combine_cast.get() && "Expected an fma node.");
    assert(fma_combine_cast->get_middle()->is_match(ax) &&
           "Expected a in the middle.");
    assert(fma_combine_cast->get_left()->is_match(p1) &&
           "Expected p1 on the left.");
    assert(fma_combine_cast->get_right()->is_match(p1 + p3) &&
           "Expected p1 + p3 on the right.");
//  fma(p1,c1 - a,p2) -> p3 - p1*a
    auto fma_combine2 = fma(p1,1.0 - ax,p3);
    auto fma_combine2_cast = graph::fma_cast(fma_combine2);
    assert(fma_combine2_cast.get() && "Expected a fma node.");
    assert(fma_combine2_cast->get_left()->is_match(-p1) &&
           "Expected -p1 on the right.");
    assert(fma_combine2_cast->get_middle()->is_match(ax) &&
           "Expected a in the middle.");
    assert(fma_combine2_cast->get_right()->is_match(p1 + p3) &&
           "Expected p1 + p3 on the right.");
//  p1*(c1 + a) - p2 -> fma(p1,a,p3)
    auto fma_combine3 = p1*(1.0 + ax) - p3;
    auto fma_combine3_cast = graph::fma_cast(fma_combine3);
    assert(fma_combine3_cast.get() && "Expected a fma node.");
    assert(fma_combine3_cast->get_middle()->is_match(ax) &&
           "Expected a in the middle.");
    assert(fma_combine3_cast->get_left()->is_match(p1) &&
           "Expected p1 on the left.");
    assert(fma_combine3_cast->get_right()->is_match(p1 - p3) &&
           "Expected p1 - p3 on the right.");
//  p1*(c1 - a) - p2 -> p3 - p1*a
    auto fma_combine4 = p1*(1.0 - ax) - p3;
    auto fma_combine4_cast = graph::subtract_cast(fma_combine4);
    assert(fma_combine4_cast.get() && "Expected an subtract node.");
    assert(fma_combine4_cast->get_right()->is_match(p1*ax) &&
           "Expected p1*a on the right.");
    assert(fma_combine4_cast->get_left()->is_match(p1 - p3) &&
           "Expected p1 - p3 on the left.");
}

//------------------------------------------------------------------------------
///  @brief Tests for 1D index nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void index_1D() {
    auto variable = graph::variable<T> (11, "");
    auto arg = graph::variable<T> (1, "");

    auto index = graph::index_1D<T> (variable, arg, 1.0, 0.0);

    variable->set({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    arg->set(static_cast<T> (3.5));

    compile<T> ({graph::variable_cast(variable),
                 graph::variable_cast(arg)},
                {index}, {},
                static_cast<T> (3.0), 0.0);
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void run_tests() {
    piecewise_1D<T> ();
    piecewise_2D<T> ();
    index_1D<T> ();
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
