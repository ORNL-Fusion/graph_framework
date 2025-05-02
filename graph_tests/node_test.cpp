//------------------------------------------------------------------------------
///  @file node\_test.cpp
///  @brief Tests for the node interface.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/piecewise.hpp"
#include "../graph_framework/math.hpp"
#include "../graph_framework/trigonometry.hpp"
#include "../graph_framework/arithmetic.hpp"

//------------------------------------------------------------------------------
///  @brief Tests for constant nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
void test_constant() {
    auto zero = graph::zero<T> ();
    auto zero_cast = graph::constant_cast(zero);
    assert(zero_cast.get() && "Expected a constant type.");
    assert(graph::variable_cast(zero).get() == nullptr &&
           "Expected a constant type.");
    assert(zero_cast->is(0) && "Constant value expeced zero.");
    assert(!zero_cast->is(1) && "Constant value not expeced one.");
    const backend::buffer<T> zero_result = zero->evaluate();
    assert(zero_result.size() == 1 && "Expected single value.");
    assert(zero_result.at(0) == static_cast<T> (0.0) &&
           "Constant value evalute expeced zero.");
    auto dzero = zero->df(zero);
    auto dzero_cast = graph::constant_cast(dzero);
    assert(dzero_cast.get() && "Expected a constant type for derivative.");
    assert(dzero_cast->is(0.0) && "Constant value expeced zero.");
    zero->set(static_cast<T> (1.0));
    assert(zero_cast->is(0.0) && "Constant value expeced zero.");

    auto one = graph::one<T> ();
    auto one_cast = graph::constant_cast(one);
    assert(one_cast.get() && "Expected a constant type.");
    assert(one_cast->is(1.0) && "Constant value expeced zero.");
    const backend::buffer<T> one_result = one->evaluate();
    assert(one_result.size() == 1 && "Expected single value.");
    assert(one_result.at(0) == static_cast<T> (1.0) &&
           "Constant value evalute expeced one.");

    auto done = one->df(zero);
    auto done_cast = graph::constant_cast(done);
    assert(done_cast.get() && "Expected a constant type for derivative.");
    assert(done_cast->is(0) && "Constant value expeced zero.");

//  Test is_match
    auto c1 = graph::constant(static_cast<T> (5.0));
    auto c2 = graph::constant(static_cast<T> (5.0));
    assert(c1.get() == c2.get() && "Expected same pointers");
    assert(c1->is_match(c2) && "Expected match.");

//  Test node properties.
    assert(c1->is_constant() && "Expected a constant.");
    assert(!c1->is_all_variables() && "Did not expect a variable.");
    assert(c1->is_power_like() && "Expected a power like.");
}

//------------------------------------------------------------------------------
///  @brief Tests for variable nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
void test_variable() {
    auto zero = graph::variable<T> (1, "");
    zero->set(static_cast<T> (0.0));
    assert(graph::variable_cast(zero).get() && "Expected a variable type.");
    assert(graph::constant_cast(zero).get() == nullptr &&
           "Expected a variable type.");
    const backend::buffer<T> zero_result = zero->evaluate();
    assert(zero_result.size() == 1 && "Expected single value.");
    assert(zero_result.at(0) == static_cast<T> (0.0) &&
           "Variable value evalute expeced zero.");
    zero->set(static_cast<T> (1.0));
    const backend::buffer<T> zero_result2 = zero->evaluate();
    assert(zero_result2.size() == 1 && "Expected single value.");
    assert(zero_result2.at(0) == static_cast<T> (1.0) &&
           "Variable value evalute expeced zero.");
    auto dzero = zero->df(zero);
    assert(graph::constant_cast(dzero).get() && "Expected a constant type.");
    const backend::buffer<T> dzero_result = dzero->evaluate();
    assert(dzero_result.size() == 1 && "Expected single value.");
    assert(dzero_result.at(0) == static_cast<T> (1.0) &&
           "Constant value evalute expeced one.");

    auto ones = graph::variable<T> (2, 1, "");
    auto dzerodone = zero->df(ones);
    assert(graph::constant_cast(dzerodone).get() &&
           "Expected a constant type.");
    const backend::buffer<T> dzerodone_result = dzerodone->evaluate();
    assert(dzerodone_result.size() == 1 && "Expected single value.");
    assert(dzerodone_result.at(0) == static_cast<T> (0.0) &&
           "Constant value evalute expeced zero.");

    auto one_two = graph::variable<T> (std::vector<T> ({1.0, 2.0}), "");
    const backend::buffer<T> one_two_result = one_two->evaluate();
    assert(one_two_result.size() == 2 && "Expected two elements in constant");
    assert(one_two_result.at(0) == static_cast<T> (1.0) &&
           "Expected one for first elememt");
    assert(one_two_result.at(1) == static_cast<T> (2.0) &&
           "Expected two for second elememt");
    one_two->set(std::vector<T> ({3.0, 4.0}));
    const backend::buffer<T> one_two_result2 = one_two->evaluate();
    assert(one_two_result2.size() == 2 && "Expected two elements in constant");
    assert(one_two_result2.at(0) == static_cast<T> (3.0) &&
           "Expected three for first elememt");
    assert(one_two_result2.at(1) == static_cast<T> (4.0) &&
           "Expected four for second elememt");

//  Test is_match
    auto v1 = graph::variable<T> (1, "");
    auto v2 = graph::variable<T> (1, "");
    assert(v1.get() != v2.get() && "Expected different pointers");
    assert(!v1->is_match(v2) && "Expected no match.");

//  Test node properties.
    assert(!v1->is_constant() && "Did not expect a constant.");
    assert(v1->is_all_variables() && "Expected a variable.");
    assert(v1->is_power_like() && "Expected a power like.");
}

//------------------------------------------------------------------------------
///  @brief Tests for pseudo variable nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
void test_pseudo_variable() {
    auto a = graph::variable<T> (1, "");
    auto b = graph::variable<T> (1, "");
    auto c = graph::pseudo_variable(a + b);
    assert(graph::constant_cast(c->df(a))->is(0) && "Expected zero.");
    assert(graph::constant_cast(c->df(c))->is(1) && "Expected one.");

    a->set(static_cast<T> (1.0));
    b->set(static_cast<T> (2.0));
    assert(c->evaluate().at(0) == static_cast<T> (3.0) &&
           "Expected three.");

    auto v2 = graph::pseudo_variable(a + b);
    assert(c.get() != v2.get() && "Expected different pointers");
    assert(!c->is_match(v2) && "Expected match.");
    
    auto remove = 2.0 + graph::pseudo_variable(graph::one<T> ());
    assert(add_cast(remove).get() && "Expected add node.");
    assert(constant_cast(remove->remove_pseudo()).get() &&
           "Expected constant node.");

//  Test node properties.
    assert(!c->is_constant() && "Did not expect a constant.");
    assert(c->is_all_variables() && "Expected a variable.");
    assert(c->is_power_like() && "Expected a power like.");
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void run_tests() {
    test_constant<T> ();
    test_variable<T> ();
    test_pseudo_variable<T> ();
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    (void)argc;
    (void)argv;
    run_tests<float> ();
    run_tests<double> ();
    run_tests<std::complex<float>> ();
    run_tests<std::complex<double>> ();
}
