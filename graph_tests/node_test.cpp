//------------------------------------------------------------------------------
///  @file node_test.cpp
///  @brief Tests for the node interface.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/cpu_backend.hpp"
#include "../graph_framework/node.hpp"

//------------------------------------------------------------------------------
///  @brief Tests for constant nodes.
//------------------------------------------------------------------------------
template<typename BACKEND>
void test_constant() {
    auto zero = graph::constant<BACKEND> (0);
    auto zero_cast = graph::constant_cast(zero);
    assert(zero_cast.get() && "Expected a constant type.");
    assert(graph::variable_cast(zero).get() == nullptr &&
           "Expected a constant type.");
    assert(zero_cast->is(0) && "Constant value expeced zero.");
    assert(!zero_cast->is(1) && "Constant value not expeced one.");
    const BACKEND zero_result = zero->evaluate();
    assert(zero_result.size() == 1 && "Expected single value.");
    assert(zero_result.at(0) == backend::base_cast<BACKEND> (0.0) &&
           "Constant value evalute expeced zero.");
    auto dzero = zero->df(zero);
    auto dzero_cast = graph::constant_cast(dzero);
    assert(dzero_cast.get() && "Expected a constant type for derivative.");
    assert(dzero_cast->is(0) && "Constant value expeced zero.");
    zero->set(backend::base_cast<BACKEND> (1.0));
    assert(zero_cast->is(0) && "Constant value expeced zero.");

    auto one = graph::constant<BACKEND> (std::vector<typename BACKEND::base> ({1.0, 1.0}));
    auto one_cast = graph::constant_cast(one);
    assert(one_cast.get() && "Expected a constant type.");
#ifdef USE_REDUCE
    assert(one_cast->is(1) && "Constant value expeced zero.");
#endif
    const BACKEND one_result = one->evaluate();
#ifdef USE_REDUCE
    assert(one_result.size() == 1 && "Expected single value.");
    assert(one_result.at(0) == backend::base_cast<BACKEND> (1.0) &&
           "Constant value evalute expeced one.");
#else
    assert(one_result.size() == 2 && "Expected two values.");
    assert(one_result.at(0) == backend::base_cast<BACKEND> (1.0) &&
           "Constant value evalute expeced one.");
    assert(one_result.at(1) == backend::base_cast<BACKEND> (1.0) &&
           "Constant value evalute expeced one.");
#endif
    auto done = one->df(zero);
    auto done_cast = graph::constant_cast(done);
    assert(done_cast.get() && "Expected a constant type for derivative.");
    assert(done_cast->is(0) && "Constant value expeced zero.");

    auto one_two = graph::constant<BACKEND> (std::vector<typename BACKEND::base> ({1.0, 2.0}));
    auto one_two_cast = graph::constant_cast(one_two);
    assert(one_two_cast.get() && "Expected a constant type.");
    assert(!one_two_cast->is(1) && "Constant expected to not be one.");
    const BACKEND one_two_result = one_two->evaluate();
    assert(one_two_result.size() == 2 && "Expected two elements in constant");
    assert(one_two_result.at(0) == backend::base_cast<BACKEND> (1.0) &&
           "Expected one for first elememt");
    assert(one_two_result.at(1) == backend::base_cast<BACKEND> (2.0) &&
           "Expected two for second elememt");

//  Test is_match
    auto c1 = graph::constant<BACKEND> (5);
    auto c2 = graph::constant<BACKEND> (5);
    assert(c1.get() != c2.get() && "Expected different pointers");
    assert(c1->is_match(c2) && "Expected match.");
}

//------------------------------------------------------------------------------
///  @brief Tests for variable nodes.
//------------------------------------------------------------------------------
template<typename BACKEND>
void test_variable() {
    auto zero = graph::variable<BACKEND> (1, "");
    zero->set(backend::base_cast<BACKEND> (0.0));
    assert(graph::variable_cast(zero).get() && "Expected a variable type.");
    assert(graph::constant_cast(zero).get() == nullptr &&
           "Expected a variable type.");
    const BACKEND zero_result = zero->evaluate();
    assert(zero_result.size() == 1 && "Expected single value.");
    assert(zero_result.at(0) == backend::base_cast<BACKEND> (0.0) &&
           "Variable value evalute expeced zero.");
    zero->set(backend::base_cast<BACKEND> (1.0));
    const BACKEND zero_result2 = zero->evaluate();
    assert(zero_result2.size() == 1 && "Expected single value.");
    assert(zero_result2.at(0) == backend::base_cast<BACKEND> (1.0) &&
           "Variable value evalute expeced zero.");
    auto dzero = zero->df(zero);
    assert(graph::constant_cast(dzero).get() && "Expected a constant type.");
    const BACKEND dzero_result = dzero->evaluate();
    assert(dzero_result.size() == 1 && "Expected single value.");
    assert(dzero_result.at(0) == backend::base_cast<BACKEND> (1.0) &&
           "Constant value evalute expeced one.");

    auto ones = graph::variable<BACKEND> (2, 1, "");
    auto dzerodone = zero->df(ones);
    assert(graph::constant_cast(dzerodone).get() &&
           "Expected a constant type.");
    const BACKEND dzerodone_result = dzerodone->evaluate();
    assert(dzerodone_result.size() == 1 && "Expected single value.");
    assert(dzerodone_result.at(0) == backend::base_cast<BACKEND> (0.0) &&
           "Constant value evalute expeced zero.");

    auto one_two = graph::variable<BACKEND> (std::vector<typename BACKEND::base> ({1.0, 2.0}), "");
    const BACKEND one_two_result = one_two->evaluate();
    assert(one_two_result.size() == 2 && "Expected two elements in constant");
    assert(one_two_result.at(0) == backend::base_cast<BACKEND> (1.0) &&
           "Expected one for first elememt");
    assert(one_two_result.at(1) == backend::base_cast<BACKEND> (2.0) &&
           "Expected two for second elememt");
    one_two->set(std::vector<typename BACKEND::base> ({3.0, 4.0}));
    const BACKEND one_two_result2 = one_two->evaluate();
    assert(one_two_result2.size() == 2 && "Expected two elements in constant");
    assert(one_two_result2.at(0) == backend::base_cast<BACKEND> (3.0) &&
           "Expected three for first elememt");
    assert(one_two_result2.at(1) == backend::base_cast<BACKEND> (4.0) &&
           "Expected four for second elememt");

//  Test is_match
    auto v1 = graph::variable<BACKEND> (1, "");
    auto v2 = graph::variable<BACKEND> (1, "");
    assert(v1.get() != v2.get() && "Expected different pointers");
    assert(!v1->is_match(v2) && "Expected no match.");
}

//------------------------------------------------------------------------------
///  @brief Tests for cache nodes.
//------------------------------------------------------------------------------
template<typename BACKEND>
void test_cache() {
    auto five = graph::variable<BACKEND> (1, "");
    five->set(backend::base_cast<BACKEND> (5.0));
    auto cache_five = graph::cache(five);
    cache_five->evaluate();
    five->set(backend::base_cast<BACKEND> (6.0));
    assert(cache_five->evaluate().at(0) == backend::base_cast<BACKEND> (5.0) &&
           "Expected a value of five.");
    cache_five->reset_cache();
    assert(cache_five->evaluate().at(0) == backend::base_cast<BACKEND> (6.0) &&
           "Expected a value of six.");
    assert(graph::cache_cast(cache_five).get() && "Expected cache node.");

    auto three = graph::constant<BACKEND> (3.0);
    auto cache_three = graph::cache(three);
#ifdef USE_REDUCE
    assert(graph::constant_cast(cache_three).get() &&
           "Expected a constant node.");
#else
    assert(graph::cache_cast(cache_three).get() && "Expected a constant node.");
#endif

    assert(graph::constant_cast(cache_five->df(cache_five))->is(1) &&
           "Expected the constant 1");
    assert(graph::constant_cast(cache_five->df(five))->is(1) &&
           "Expected the constant 1");
    assert(graph::constant_cast(cache_five->df(three))->is(0) &&
           "Expected the constant 0");

//  Test is_match
    auto c1 = graph::cache(five);
    auto c2 = graph::cache(five);
    assert(c1.get() != c2.get() && "Expected different pointers");
    assert(c1->is_match(c2) && "Expected match.");
}

//------------------------------------------------------------------------------
///  @brief Tests for pseudo variable nodes.
//------------------------------------------------------------------------------
template<typename BACKEND>
void test_pseudo_variable() {
    auto a = graph::variable<BACKEND> (1, "");
    auto b = graph::variable<BACKEND> (1, "");
    auto c = graph::pseudo_variable(a + b);
    assert(graph::constant_cast(c->df(a))->is(0) && "Expected zero.");
    assert(graph::constant_cast(c->df(c))->is(1) && "Expected one.");

    a->set(backend::base_cast<BACKEND> (1.0));
    b->set(backend::base_cast<BACKEND> (2.0));
    assert(c->evaluate().at(0) == backend::base_cast<BACKEND> (3.0) &&
           "Expected three.");

    auto v2 = graph::pseudo_variable(a + b);
    assert(c.get() != v2.get() && "Expected different pointers");
    assert(!c->is_match(v2) && "Expected no match.");
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
//------------------------------------------------------------------------------
template<typename BACKEND> void run_tests() {
    test_constant<BACKEND> ();
    test_variable<BACKEND> ();
    test_cache<BACKEND> ();
    test_pseudo_variable<BACKEND> ();
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    run_tests<backend::cpu<float>> ();
    run_tests<backend::cpu<double>> ();
    run_tests<backend::cpu<std::complex<float>>> ();
    run_tests<backend::cpu<std::complex<double>>> ();
}
