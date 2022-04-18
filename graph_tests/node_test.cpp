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
    assert(zero_cast.get() != nullptr && "Expected a constant type.");
    assert(graph::variable_cast(zero).get() == nullptr &&
           "Expected a constant type.");
    assert(zero_cast->is(0) && "Constant value expeced zero.");
    assert(!zero_cast->is(1) && "Constant value not expeced one.");
    const BACKEND zero_result = zero->evaluate();
    assert(zero_result.size() == 1 && "Expected single value.");
    assert(zero_result.at(0) == 0 && "Constant value evalute expeced zero.");
    auto dzero = zero->df(zero);
    auto dzero_cast = graph::constant_cast(dzero);
    assert(dzero_cast.get() != nullptr  &&
           "Expected a constant type for derivative.");
    assert(dzero_cast->is(0) && "Constant value expeced zero.");
    zero->set(1);
    assert(zero_cast->is(0) && "Constant value expeced zero.");

    auto one = graph::constant<BACKEND> (std::vector<double> ({1.0, 1.0}));
    auto one_cast = graph::constant_cast(one);
    assert(one_cast.get() != nullptr && "Expected a constant type.");
    assert(one_cast->is(1) && "Constant value expeced zero.");
    const BACKEND one_result = one->evaluate();
    assert(one_result.size() == 1 && "Expected single value.");
    assert(one_result.at(0) == 1 && "Constant value evalute expeced one.");
    auto done = one->df(zero);
    auto done_cast = graph::constant_cast(done);
    assert(done_cast.get() != nullptr &&
           "Expected a constant type for derivative.");
    assert(done_cast->is(0) && "Constant value expeced zero.");

    auto one_two = graph::constant<BACKEND> (std::vector<double> ({1.0, 2.0}));
    auto one_two_cast = graph::constant_cast(one_two);
    assert(one_two_cast.get() != nullptr && "Expected a constant type.");
    assert(!one_two_cast->is(1) && "Constant expected to not be one.");
    const BACKEND one_two_result = one_two->evaluate();
    assert(one_two_result.size() == 2 && "Expected two elements in constant");
    assert(one_two_result.at(0) == 1 && "Expected one for first elememt");
    assert(one_two_result.at(1) == 2 && "Expected two for second elememt");
}

//------------------------------------------------------------------------------
///  @brief Tests for variable nodes.
//------------------------------------------------------------------------------
template<typename BACKEND>
void test_variable() {
    auto zero = graph::variable<BACKEND> (1);
    zero->set(0);
    auto zero_cast = graph::variable_cast(zero);
    assert(zero_cast.get() != nullptr && "Expected a variable type.");
    assert(graph::constant_cast(zero).get() == nullptr &&
           "Expected a variable type.");
    const BACKEND zero_result = zero->evaluate();
    assert(zero_result.size() == 1 && "Expected single value.");
    assert(zero_result.at(0) == 0 && "Variable value evalute expeced zero.");
    zero->set(1);
    const BACKEND zero_result2 = zero->evaluate();
    assert(zero_result2.size() == 1 && "Expected single value.");
    assert(zero_result2.at(0) == 1 && "Variable value evalute expeced zero.");
    auto dzero = zero->df(zero);
    auto dzero_cast = graph::constant_cast(dzero);
    assert(zero_cast.get() != nullptr && "Expected a constant type.");
    const BACKEND dzero_result = dzero->evaluate();
    assert(dzero_result.size() == 1 && "Expected single value.");
    assert(dzero_result.at(0) == 1 && "Constant value evalute expeced one.");

    auto ones = graph::variable<BACKEND> (2, 1);
    auto dzerodone = zero->df(ones);
    auto dzerodone_cast = graph::constant_cast(dzerodone);
    assert(dzerodone.get() != nullptr && "Expected a constant type.");
    const BACKEND dzerodone_result = dzerodone->evaluate();
    assert(dzerodone_result.size() == 1 && "Expected single value.");
    assert(dzerodone_result.at(0) == 0 && "Constant value evalute expeced zero.");

    auto one_two = graph::variable<BACKEND> (std::vector<double> ({1.0, 2.0}));
    const BACKEND one_two_result = one_two->evaluate();
    assert(one_two_result.size() == 2 && "Expected two elements in constant");
    assert(one_two_result.at(0) == 1 && "Expected one for first elememt");
    assert(one_two_result.at(1) == 2 && "Expected two for second elememt");
    one_two->set(std::vector<double> ({3.0, 4.0}));
    const BACKEND one_two_result2 = one_two->evaluate();
    assert(one_two_result2.size() == 2 && "Expected two elements in constant");
    assert(one_two_result2.at(0) == 3 && "Expected three for first elememt");
    assert(one_two_result2.at(1) == 4 && "Expected four for second elememt");
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
//------------------------------------------------------------------------------
template<typename BACKEND> void run_tests() {
    test_constant<BACKEND> ();
    test_variable<BACKEND> ();
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    run_tests<backend::cpu> ();
}
