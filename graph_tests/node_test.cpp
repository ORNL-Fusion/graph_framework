//------------------------------------------------------------------------------
///  @file node_test.cpp
///  @brief Tests for the node interface.
//------------------------------------------------------------------------------

#include <cassert>

#include "../graph_framework/node.hpp"

//  Turn on asserts even in release builds.
#ifndef NDEBUG
#define NDEBUG
#endif

//------------------------------------------------------------------------------
///  @brief Tests for constant nodes.
//------------------------------------------------------------------------------
void test_constant() {
    auto zero = graph::constant(0);
    auto zero_cast = std::dynamic_pointer_cast<graph::constant_node> (zero);
    assert(zero_cast.get() != nullptr && "Expected a constant type.");
    assert(std::dynamic_pointer_cast<graph::variable_node> (zero).get() == nullptr &&
           "Expected a constant type.");
    assert(zero_cast->is(0) && "Constant value expeced zero.");
    assert(!zero_cast->is(1) && "Constant value not expeced one.");
    const std::vector<double> zero_result = zero->evaluate();
    assert(zero_result.size() == 1 && "Expected single value.");
    assert(zero_result.at(0) == 0 && "Constant value evalute expeced zero.");
    auto dzero = zero->df(zero);
    auto dzero_cast = std::dynamic_pointer_cast<graph::constant_node> (dzero);
    assert(dzero_cast.get() != nullptr  &&
           "Expected a constant type for derivative.");
    assert(dzero_cast->is(0) && "Constant value expeced zero.");
    zero->set(1);
    assert(zero_cast->is(0) && "Constant value expeced zero.");

    auto one = graph::constant({1,1});
    auto one_cast = std::dynamic_pointer_cast<graph::constant_node> (one);
    assert(one_cast.get() != nullptr && "Expected a constant type.");
    assert(one_cast->is(1) && "Constant value expeced zero.");
    const std::vector<double> one_result = one->evaluate();
    assert(one_result.size() == 1 && "Expected single value.");
    assert(one_result.at(0) == 1 && "Constant value evalute expeced one.");
    auto done = one->df(zero);
    auto done_cast = std::dynamic_pointer_cast<graph::constant_node> (done);
    assert(done_cast.get() != nullptr &&
           "Expected a constant type for derivative.");
    assert(done_cast->is(0) && "Constant value expeced zero.");

    auto one_two = graph::constant({1,2});
    auto one_two_cast = std::dynamic_pointer_cast<graph::constant_node> (one_two);
    assert(one_two_cast.get() != nullptr && "Expected a constant type.");
    assert(!one_two_cast->is(1) && "Constant expected to not be one.");
    const std::vector<double> one_two_result = one_two->evaluate();
    assert(one_two_result.size() == 2 && "Expected two elements in constant");
    assert(one_two_result.at(0) == 1 && "Expected one for first elememt");
    assert(one_two_result.at(1) == 2 && "Expected two for second elememt");
}

//------------------------------------------------------------------------------
///  @brief Tests for variable nodes.
//------------------------------------------------------------------------------
void test_variable() {
    auto zero = graph::variable(1);
    zero->set(0);
    auto zero_cast = std::dynamic_pointer_cast<graph::variable_node> (zero);
    assert(zero_cast.get() != nullptr && "Expected a variable type.");
    assert(std::dynamic_pointer_cast<graph::constant_node> (zero).get() == nullptr &&
           "Expected a variable type.");
    const std::vector<double> zero_result = zero->evaluate();
    assert(zero_result.size() == 1 && "Expected single value.");
    assert(zero_result.at(0) == 0 && "Variable value evalute expeced zero.");
    zero->set(1);
    const std::vector<double> zero_result2 = zero->evaluate();
    assert(zero_result2.size() == 1 && "Expected single value.");
    assert(zero_result2.at(0) == 1 && "Variable value evalute expeced zero.");
    auto dzero = zero->df(zero);
    auto dzero_cast = std::dynamic_pointer_cast<graph::constant_node> (dzero);
    assert(zero_cast.get() != nullptr && "Expected a constant type.");
    const std::vector<double> dzero_result = dzero->evaluate();
    assert(dzero_result.size() == 1 && "Expected single value.");
    assert(dzero_result.at(0) == 1 && "Constant value evalute expeced one.");

    auto ones = graph::variable(2, 1);
    auto dzerodone = zero->df(ones);
    auto dzerodone_cast = std::dynamic_pointer_cast<graph::constant_node> (dzerodone);
    assert(dzerodone.get() != nullptr && "Expected a constant type.");
    const std::vector<double> dzerodone_result = dzerodone->evaluate();
    assert(dzerodone_result.size() == 1 && "Expected single value.");
    assert(dzerodone_result.at(0) == 0 && "Constant value evalute expeced zero.");

    auto one_two = graph::variable({1,2});
    const std::vector<double> one_two_result = one_two->evaluate();
    assert(one_two_result.size() == 2 && "Expected two elements in constant");
    assert(one_two_result.at(0) == 1 && "Expected one for first elememt");
    assert(one_two_result.at(1) == 2 && "Expected two for second elememt");
    one_two->set({3,4});
    const std::vector<double> one_two_result2 = one_two->evaluate();
    assert(one_two_result2.size() == 2 && "Expected two elements in constant");
    assert(one_two_result2.at(0) == 3 && "Expected three for first elememt");
    assert(one_two_result2.at(1) == 4 && "Expected four for second elememt");
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    test_constant();
    test_variable();
}
