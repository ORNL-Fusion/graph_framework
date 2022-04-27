//------------------------------------------------------------------------------
///  @file arithmetic_test.cpp
///  @brief Tests for arithmetic nodes.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/cpu_backend.hpp"
#include "../graph_framework/arithmetic.hpp"

//------------------------------------------------------------------------------
///  @brief Tests for addition nodes.
//------------------------------------------------------------------------------
template<typename BACKEND> void test_add() {
//  Three constant nodes should reduce to a single constant node with added
//  operands.
    auto one = graph::constant<BACKEND> (1);
    auto three = one + one + one;
    assert(graph::constant_cast(three).get() != nullptr &&
           "Expected a constant type.");

    const BACKEND result_three = three->evaluate();
    assert(result_three.size() == 1 && "Expected single value.");
    assert(result_three.at(0) == 3 && "Expected three for result");

//  Any zero nodes should reduce to the other operand.
    auto zero = graph::constant<BACKEND> (0);
    auto one_plus_zero = one + zero;
    assert(one_plus_zero.get() == one.get() &&
           "Expected to retrive the left side.");
    auto zero_plus_one = zero + one;
    assert(zero_plus_one.get() == one.get() &&
           "Expected to retrive the right side.");

//  Test vector scalar quanties.
    auto vec_constant = graph::constant<BACKEND> (std::vector<double> ({3.0, 4.0}));
    auto vec_plus_zero = vec_constant + zero;
    assert(vec_plus_zero.get() == vec_constant.get() &&
           "Expected to retrive the left side.");
    auto zero_plus_vec = zero + vec_constant;
    assert(zero_plus_vec.get() == vec_constant.get() &&
           "Expected to retrive the right side.");

    auto vec_plus_one = vec_constant + one;
    assert(graph::constant_cast(vec_plus_one).get() != nullptr &&
           "Expected a constant type.");
    auto vec_plus_one_result = vec_plus_one->evaluate();
    assert(vec_plus_one_result.size() == 2 && "Size mismatch in result.");
    assert(vec_plus_one_result.at(0) == 4 && "Expected 3 + 1.");
    assert(vec_plus_one_result.at(1) == 5 && "Expected 4 + 1.");

    auto one_plus_vec = one + vec_constant;
    assert(graph::constant_cast(one_plus_vec).get() != nullptr &&
           "Expected a constant type.");
    auto one_plus_vec_result = one_plus_vec->evaluate();
    assert(one_plus_vec_result.size() == 2 && "Size mismatch in result.");
    assert(one_plus_vec_result.at(0) == 4 && "Expected 1 + 3.");
    assert(one_plus_vec_result.at(1) == 5 && "Expected 1 + 4.");

// Test vector vector quaties.
    auto vec_plus_vec = vec_constant + vec_constant;
    const BACKEND vec_plus_vec_result = vec_plus_vec->evaluate();
    assert(vec_plus_vec_result.size() == 2 &&
           "Size mismatch in result.");
    assert(vec_plus_vec_result.at(0) == 6 && "Expected 3 + 3.");
    assert(vec_plus_vec_result.at(1) == 8 && "Expected 4 + 4.");

//  Test variable quanities.
//  Any zero nodes should reduce to the other operand.
    auto variable = graph::variable<BACKEND> (1);
    auto var_plus_zero = variable + zero;
    assert(var_plus_zero.get() == variable.get() &&
           "Expected to retrive the left side.");
    auto zero_plus_var = zero + variable;
    assert(zero_plus_var.get() == variable.get() &&
           "Expected to retrive the right side.");

//  Variable plus a variable should return an add node.
    auto var_plus_var = variable + variable;
    assert(graph::multiply_cast(var_plus_var).get() != nullptr &&
           "Expected an multiply node.");
    variable->set(10);
    const BACKEND var_plus_var_result = var_plus_var->evaluate();
    assert(var_plus_var_result.size() == 1 && "Expected single value.");
    assert(var_plus_var_result.at(0) == 20 && "Expected 10 + 10 for result");

//  Variable plus a variable should return an add node.
    auto variable_b = graph::variable<BACKEND> (1);
    auto var_plus_varb = variable + variable_b;
    assert(graph::add_cast(var_plus_varb).get() != nullptr &&
           "Expected an add node.");
    variable_b->set(5);
    const BACKEND var_plus_varb_result = var_plus_varb->evaluate();
    assert(var_plus_varb_result.size() == 1 && "Expected single value.");
    assert(var_plus_varb_result.at(0) == 15 && "Expected 10 + 5 for result");

//  Test variable vectors.
    auto varvec = graph::variable<BACKEND> (std::vector<double> ({10.0, 20.0}));
    auto varvec_plus_varvec = varvec + varvec;
    const BACKEND varvec_plus_varvec_result =
        varvec_plus_varvec->evaluate();
    assert(varvec_plus_varvec_result.size() == 2 &&
           "Size mismatch in result.");
    assert(varvec_plus_varvec_result.at(0) == 20 && "Expected 10 + 10.");
    assert(varvec_plus_varvec_result.at(1) == 40 && "Expected 40 + 40.");

//  Test derivatives
//  d (1 + x) / dx = d1/dx + dx/dx = 0 + 1 = 1
    auto one_plus_var = one + variable;
    auto done_plus_var = one_plus_var->df(variable);
    auto done_plus_var_cast = graph::constant_cast(done_plus_var);
    assert(done_plus_var_cast.get() != nullptr && "Expected a constant type.");
    assert(done_plus_var_cast->is(1) &&
           "Expected to reduce to a constant one.");

//  Test common factors.
    auto var_a = graph::variable<BACKEND> (1);
    auto var_b = graph::variable<BACKEND> (1);
    auto var_c = graph::variable<BACKEND> (1);
    auto common_a = var_a*var_b + var_a*var_c;
    assert(graph::add_cast(common_a) == nullptr && "Did not expect add node.");
    assert(graph::multiply_cast(common_a) != nullptr &&
           "Expected multiply node.");

    auto common_b = var_a*var_b + var_b*var_c;
    assert(graph::add_cast(common_b) == nullptr && "Did not expect add node.");
    assert(graph::multiply_cast(common_b) != nullptr &&
           "Expected multiply node.");

    auto common_c = var_a*var_c + var_b*var_c;
    assert(graph::add_cast(common_c) == nullptr && "Did not expect add node.");
    assert(graph::multiply_cast(common_c) != nullptr &&
           "Expected multiply node.");

//  Test common denominator.
    auto common_d = var_a/var_b + var_c/var_b;
    assert(graph::add_cast(common_d) == nullptr && "Did not expect add node.");
    assert(graph::divide_cast(common_d) != nullptr && "Expected divide node.");
}

//------------------------------------------------------------------------------
///  @brief Tests for subtract nodes.
//------------------------------------------------------------------------------
template<typename BACKEND> void test_subtract() {
//  Three constant nodes should reduce to a single constant node with added
//  operands.
    auto one = graph::constant<BACKEND> (1);
    auto zero = one - one;
    auto zero_cast = graph::constant_cast(zero);
    assert(zero_cast.get() != nullptr &&
           "Expected a constant type.");
    assert(zero_cast->is(0));

    auto neg_one = one - one - one;
    auto neg_one_cast = graph::constant_cast(neg_one);
    assert(neg_one_cast.get() != nullptr &&
           "Expected a constant type.");
    assert(neg_one_cast->is(-1));

//  A right side zero node should reduce to left side.
    auto one_minus_zero = one - zero;
    assert(one_minus_zero.get() == one.get() &&
           "Expected to retrive the left side.");

//  A left side zero node should reduce to a negative right side.
    auto zero_minus_one = zero - one;
    auto zero_minus_one_cast = graph::constant_cast(zero_minus_one);
    assert(zero_minus_one_cast.get() != nullptr &&
           "Expected a constant type.");
    assert(zero_minus_one_cast->is(-1) && "Expected -1 for result");

//  Test vector scalar quantities.
    auto vec_constant_a = graph::constant<BACKEND> (std::vector<double> ({3.0, 4.0}));
    auto vec_minus_one = vec_constant_a - one;
    assert(graph::constant_cast(vec_minus_one).get() != nullptr &&
           "Expected a constant type.");
    auto vec_minus_one_result = vec_minus_one->evaluate();
    assert(vec_minus_one_result.size() == 2 && "Size mismatch in result.");
    assert(vec_minus_one_result.at(0) == 2 && "Expected 3 - 1.");
    assert(vec_minus_one_result.at(1) == 3 && "Expected 4 - 1.");

    auto vec_constant_b = graph::constant<BACKEND> (std::vector<double> ({2.0, 5.0}));
    auto one_minus_vec = one - vec_constant_b;
    assert(graph::constant_cast(one_minus_vec).get() != nullptr &&
           "Expected a constant type.");
    auto one_minus_vec_result = one_minus_vec->evaluate();
    assert(one_minus_vec_result.size() == 2 && "Size mismatch in result.");
    assert(one_minus_vec_result.at(0) == -1 && "Expected 1 - 2.");
    assert(one_minus_vec_result.at(1) == -4 && "Expected 1 - 5.");

//  Test vector vector quanties.
    auto vec_minus_vec = vec_constant_a - vec_constant_b;
    assert(graph::constant_cast(vec_minus_vec).get() != nullptr &&
           "Expected a constant type.");
    const BACKEND vec_minus_vec_result = vec_minus_vec->evaluate();
    assert(vec_minus_vec_result.size() == 2 && "Expected a constant type.");
    assert(vec_minus_vec_result.at(0) == 1 && "Expected 3 - 1 for result.");
    assert(vec_minus_vec_result.at(1) == -1 && "Expected 4 - 5 for result.");

//  Test variable quanities.
//  Any right side zero nodes should reduce to the other operand.
    auto variable = graph::variable<BACKEND> (1);
    auto var_minus_zero = variable - zero;
    assert(var_minus_zero.get() == variable.get() &&
           "Expected to retrive the left side.");

//  Any right side zero should reduce to a the a multiply node.
    auto zero_minus_var = zero - variable;
    assert(graph::multiply_cast(zero_minus_var).get() != nullptr &&
           "Expected multiply node.");
    variable->set(3);
    const BACKEND zero_minus_var_result = zero_minus_var->evaluate();
    assert(zero_minus_var_result.size() == 1 && "Expected single value.");
    assert(zero_minus_var_result.at(0) == -3 && "Expected 0 - 3 for result.");

//  Variable minus a variable should return an minus node.
    auto variable_b = graph::variable<BACKEND> (1);
    auto var_minus_var = variable - variable_b;
    assert(graph::subtract_cast(var_minus_var).get() != nullptr &&
           "Expected a subtraction node.");
    variable_b->set(10);
    const BACKEND var_minus_var_result = var_minus_var->evaluate();
    assert(var_minus_var_result.size() == 1 && "Expected single value.");
    assert(var_minus_var_result.at(0) == -7 && "Expected 3 - 10 for result");

//  Test variable vectors.
    auto varvec_a = graph::variable<BACKEND> (std::vector<double> ({10.0, 20.0}));
    auto varvec_b = graph::variable<BACKEND> (std::vector<double> ({-3.0, 5.0}));
    auto varvec_minus_varvec = varvec_a - varvec_b;
    const BACKEND varvec_minus_varvec_result =
        varvec_minus_varvec->evaluate();
    assert(varvec_minus_varvec_result.size() == 2 &&
           "Size mismatch in result.");
    assert(varvec_minus_varvec_result.at(0) == 13 && "Expected 10 - -3.");
    assert(varvec_minus_varvec_result.at(1) == 15 && "Expected 20 - 5.");

//  Test derivatives.
//  d (1 - x) / dx = d1/dx - dx/dx = 0 - 1 = -1
    auto one_minus_var = one - variable;
    auto done_minus_var = one_minus_var->df(variable);
    auto done_minus_var_cast = graph::constant_cast(done_minus_var);
    assert(done_minus_var_cast.get() != nullptr && "Expected a constant type.");
    assert(done_minus_var_cast->is(-1) &&
           "Expected to reduce to a constant minus one.");

//  Test common factors.
    auto var_a = graph::variable<BACKEND> (1);
    auto var_b = graph::variable<BACKEND> (1);
    auto var_c = graph::variable<BACKEND> (1);
    auto common_a = var_a*var_b - var_a*var_c;
    assert(graph::add_cast(common_a) == nullptr && "Did not expect add node.");
    assert(graph::multiply_cast(common_a) != nullptr &&
           "Expected multiply node.");

    auto common_b = var_a*var_b - var_b*var_c;
    assert(graph::add_cast(common_b) == nullptr && "Did not expect add node.");
    assert(graph::multiply_cast(common_b) != nullptr &&
           "Expected multiply node.");

    auto common_c = var_a*var_c - var_b*var_c;
    assert(graph::add_cast(common_c) == nullptr && "Did not expect add node.");
    assert(graph::multiply_cast(common_c) != nullptr &&
           "Expected multiply node.");

//  Test common denominator.
    auto common_d = var_a/var_b - var_c/var_b;
    assert(graph::subtract_cast(common_d) == nullptr &&
           "Did not expect subtract node.");
    assert(graph::divide_cast(common_d) != nullptr && "Expected divide node.");
}

//------------------------------------------------------------------------------
///  @brief Tests for multiply nodes.
//------------------------------------------------------------------------------
template<typename BACKEND> void test_multiply() {
//  Three constant nodes should reduce to a single constant node with multiplied
//  operands.
    auto one = graph::constant<BACKEND> (1);
    auto one_cubed = one*one*one;
    assert(one_cubed.get() == one.get() && "Expected to reduce back to one");

//  Any zero nodes should reduce zero.
    auto zero = graph::constant<BACKEND> (0);
    assert((zero*one).get() == zero.get() && "Expected to reduce back to zero");
    assert((one*zero).get() == zero.get() && "Expected to reduce back to zero");

//  Test constant times constant.
    auto two = graph::constant<BACKEND> (2);
    auto three = graph::constant<BACKEND> (3);
    auto two_times_three = two*three;
    assert(graph::constant_cast(two_times_three).get() != nullptr &&
           "Expected a constant type.");
    auto three_times_two = three*two;
    assert(graph::constant_cast(three_times_two).get() != nullptr &&
           "Expected a constant type.");
    const BACKEND two_times_three_result =
        two_times_three->evaluate();
    const BACKEND three_times_two_result =
        three_times_two->evaluate();
    assert(two_times_three_result.size() == 1 && "Expected single value.");
    assert(three_times_two_result.size() == 1 && "Expected single value.");
    assert(three_times_two_result.at(0) == 6 && "Expected 3*2 for result.");
    assert(three_times_two_result.at(0) == two_times_three_result.at(0) &&
           "Expected 3*2 == 2*3.");

//  Test vec times constant.
    auto vec = graph::constant<BACKEND> (std::vector<double> ({4.0, 5.0}));
    assert((zero*vec).get() == zero.get() && "Expected left side zero.");
    assert((vec*zero).get() == zero.get() && "Expected right side zero.");
    assert((one*vec).get() == vec.get() && "Expected right side vec.");
    assert((vec*one).get() == vec.get() && "Expected left side vec.");

    auto two_times_vec = two*vec;
    assert(graph::constant_cast(two_times_vec).get() != nullptr &&
           "Expected a constant type.");
    auto vec_times_two = vec*two;
    const BACKEND two_times_vec_result = two_times_vec->evaluate();
    const BACKEND vec_times_two_result = vec_times_two->evaluate();
    assert(two_times_vec_result.size() == 2 && "Expected two values.");
    assert(vec_times_two_result.size() == 2 && "Expected two values.");
    assert(two_times_vec_result.at(0) == 8 && "Expected 2*4 for result.");
    assert(two_times_vec_result.at(1) == 10 && "Expected 2*5 for result.");
    assert(two_times_vec_result.at(0) == vec_times_two_result.at(0) &&
           "Expected 2*4 == 4*2.");
    assert(two_times_vec_result.at(1) == vec_times_two_result.at(1) &&
           "Expected 2*5 == 5*2.");

//  Test vec times vec.
    auto vec_times_vec = vec*vec;
    assert(graph::constant_cast(vec_times_vec).get() != nullptr &&
           "Expected a constant type.");
    const BACKEND vec_times_vec_result = vec_times_vec->evaluate();
    assert(vec_times_vec_result.size() == 2 && "Expected two values.");
    assert(vec_times_vec_result.at(0) == 16 && "Expected 4*4 for result.");
    assert(vec_times_vec_result.at(1) == 25 && "Expected 5*5 for result.");

//  Test reduction short cut. If all the elements in the numerator are zero, an
//  denominator does not need to be evaluated. This test makes sure that a sum
//  or product is not used to avoid cases like {-1, 0, 1} which sum and product
//  are zero.
    auto vec_sum_prod = graph::constant<BACKEND> (std::vector<double> ({-1.0, 0.0, 1.0}));
    auto vec_sum_prod_multiply_two = vec_sum_prod*two;
    const BACKEND vec_sum_prod_multiply_two_result =
        vec_sum_prod_multiply_two->evaluate();
    assert(vec_sum_prod_multiply_two_result.at(0) == -1.0*2.0 &&
           "Expected -1/2 for result.");
    assert(vec_sum_prod_multiply_two_result.at(1) == 0 &&
           "Expected 0/2 for result.");
    assert(vec_sum_prod_multiply_two_result.at(2) == 1.0*2.0 &&
           "Expected 1/2 for result.");

//  Test variable quanities.
//  Any zero should reduce back to zero.
    auto variable = graph::variable<BACKEND> (1);
    assert((variable*zero).get() == zero.get() &&
           "Expected to retrive the right side.");
    assert((zero*variable).get() == zero.get() &&
           "Expected to retrive the left side.");
//  Any one should reduce to the opposite side.
    assert((variable*one).get() == variable.get() &&
           "Expected to retrive the left side.");
    assert((one*variable).get() == variable.get() &&
           "Expected to retrive the right side.");

//  Varibale times a non 0 or 1 constant should reduce to a multiply node.
    auto two_times_var = two*variable;
    assert(graph::multiply_cast(two_times_var).get() != nullptr &&
           "Expected multiply node.");
    variable->set(6);
    const BACKEND two_times_var_result = two_times_var->evaluate();
    assert(two_times_var_result.size() == 1 && "Expected single value.");
    assert(two_times_var_result.at(0) == 12 && "Expected 2*6 for result.");

    auto var_times_var = variable*variable;
    assert(graph::multiply_cast(var_times_var).get() != nullptr &&
           "Expected multiply node.");
    const BACKEND var_times_var_result = var_times_var->evaluate();
    assert(var_times_var_result.size() == 1 && "Expected single value.");
    assert(var_times_var_result.at(0) == 36 && "Expected 6*6 for result.");

//  Test variable vectors.
    auto varvec_a = graph::variable<BACKEND> (std::vector<double> ({4.0, -2.0}));
    auto varvec_b = graph::variable<BACKEND> (std::vector<double> ({-4.0, -2.0}));
    auto varvec_times_varvec = varvec_a*varvec_b;
    const BACKEND varvec_times_varvec_result =
        varvec_times_varvec->evaluate();
    assert(varvec_times_varvec_result.size() == 2 &&
           "Size mismatch in result.");
    assert(varvec_times_varvec_result.at(0) == -16 && "Expected 4*-4.");
    assert(varvec_times_varvec_result.at(1) == 4 && "Expected -2*-2.");

//  Test reduction short cut. If all the elements in the numerator are zero, an
//  denominator does not need to be evaluated. This test makes sure that a sum
//  or product is not used to avoid cases like {-1, 0, 1} which sum and product
//  are zero.
    auto var_sum_prod = graph::variable<BACKEND> (std::vector<double> ({-2.0, 2.0, 0.0}));
    auto var_sum_prod_multiply_two = var_sum_prod*two;
    const BACKEND var_sum_prod_multiply_two_result =
        var_sum_prod_multiply_two->evaluate();
    assert(var_sum_prod_multiply_two_result.at(0) == -2.0*2.0 &&
           "Expected -2/2 for result.");
    assert(var_sum_prod_multiply_two_result.at(1) == 2.0*2.0 &&
           "Expected 2/2 for result.");
    assert(var_sum_prod_multiply_two_result.at(2) == 0 &&
           "Expected 0/2 for result.");

//  Test derivatives.
//  d (c*x) / dx = dc/dx*x + c*dx/dx = c*1 = c;
    assert(two_times_var->df(variable).get() == two.get() &&
           "Expect to reduce back to the constant.");
//  d (x*x) / dx = dx/dx*x + x*dx/dx = x + x = 2*x;
    auto varvec_sqrd = varvec_a*varvec_a;
    auto dvarvec_sqrd = varvec_sqrd->df(varvec_a);
    assert(graph::multiply_cast(dvarvec_sqrd).get() != nullptr &&
           "Expected multiply node.");
    const BACKEND dvarvec_sqrd_result = dvarvec_sqrd->evaluate();
    assert(dvarvec_sqrd_result.size() == 2 && "Size mismatch in result.");
    assert(dvarvec_sqrd_result.at(0) == 8 && "Expected 2*4 for result.");
    assert(dvarvec_sqrd_result.at(1) == -4 && "Expected 2*-2 for result.");
}

//------------------------------------------------------------------------------
///  @brief Tests for divide nodes.
//------------------------------------------------------------------------------
template<typename BACKEND> void test_divide() {
// A zero in the numerator should result in zero.
    auto zero = graph::constant<BACKEND> (0);
    auto one = graph::constant<BACKEND> (1);
    assert((zero/one).get() == zero.get() && "Expected to recover zero.");

// A one in the denominator should result in numerator.
    assert((one/one).get() == one.get() && "Expected to recover one.");
    auto two = graph::constant<BACKEND> (2);
    assert((two/one).get() == two.get() && "Expected to recover two.");

//  A value divided by it self should be a constant one.
    auto two_divided_two = two/two;
    auto two_divided_two_cast = graph::constant_cast(two_divided_two);
    assert(graph::constant_cast(two_divided_two).get() != nullptr &&
           "Expected a constant type.");
    assert(two_divided_two_cast->is(1) && "Expected 1 for result");

//  A constant a divided by constant b should be a constant with value of a/b.
    auto three = graph::constant<BACKEND> (3);
    auto two_divided_three = two/three;
    auto two_divided_three_cast = graph::constant_cast(two_divided_three);
    assert(two_divided_three_cast.get() != nullptr &&
           "Expected a constant type.");
    assert(two_divided_three_cast->is(2.0/3.0) && "Expected 2/3 for result");

//  Test vector constants.
    auto vec = graph::constant<BACKEND>(std::vector<double> ({4.0, 3.0}));
    assert((zero/vec).get() == zero.get() && "Expected to recover zero.");
    assert((vec/one).get() == vec.get() && "Expected to recover numerator.");
    auto vec_divided_vec = vec/vec;
    auto vec_divided_vec_cast = graph::constant_cast(vec_divided_vec);
    assert(vec_divided_vec_cast.get() != nullptr &&
           "Expected a constant type.");
    assert(vec_divided_vec_cast->is(1) && "Expected 1 for result");
    auto two_divided_vec = two/vec;
    assert(graph::constant_cast(two_divided_vec).get() != nullptr &&
           "Expected a constant type.");
    const BACKEND two_divided_vec_result =
        two_divided_vec->evaluate();
    assert(two_divided_vec_result.size() == 2 && "Size mismatch in result.");
    assert(two_divided_vec_result.at(0) == 2.0/4.0 &&
           "Expected 2/4 for result.");
    assert(two_divided_vec_result.at(1) == 2.0/3.0 &&
           "Expected 2/3 for result.");

//  Test reduction short cut. If all the elements in the numerator are zero, an
//  denominator does not need to be evaluated. This test makes sure that a sum
//  or product is not used to avoid cases like {-1, 0, 1} which sum and product
//  are zero.
    auto vec_sum_prod = graph::constant<BACKEND> (std::vector<double> ({-1.0, 0.0, 1.0}));
    auto vec_sum_prod_divided_two = vec_sum_prod/two;
    const BACKEND vec_sum_prod_divided_two_result =
        vec_sum_prod_divided_two->evaluate();
    assert(vec_sum_prod_divided_two_result.at(0) == -1.0/2.0 &&
           "Expected -1/2 for result.");
    assert(vec_sum_prod_divided_two_result.at(1) == 0 &&
           "Expected 0/2 for result.");
    assert(vec_sum_prod_divided_two_result.at(2) == 1.0/2.0 &&
           "Expected 1/2 for result.");

//  Test variables.
    auto variable = graph::variable<BACKEND> (1);
    assert((zero/variable).get() == zero.get() && "Expected to recover zero.");
    assert((variable/one).get() == variable.get() &&
           "Expected to recover numerator.");

    auto two_divided_var = two/variable;
    assert(graph::divide_cast(two_divided_var).get() != nullptr &&
           "Expected divide node.");
    variable->set(3);
    const BACKEND two_divided_var_result = two_divided_var->evaluate();
    assert(two_divided_var_result.size() == 1 && "Expected single value.");
    assert(two_divided_var_result.at(0) == 2.0/3.0 &&
           "Expected 2/3 for result.");

    auto var_divided_two = variable/two;
    assert(graph::divide_cast(var_divided_two).get() != nullptr &&
           "Expected divide node.");
    const BACKEND var_divided_two_result = var_divided_two->evaluate();
    assert(var_divided_two_result.size() == 1 && "Expected single value.");
    assert(var_divided_two_result.at(0) == 3.0/2.0 &&
           "Expected 3/2 for result.");

    auto var_divided_var = variable/variable;
    auto var_divided_var_cast = graph::constant_cast(var_divided_var);
    assert(var_divided_var_cast.get() != nullptr && "Expeced constant node.");
    assert(var_divided_var_cast->is(1) && "Expeced one.");

    auto variable_b = graph::variable<BACKEND> (1, 4);
    auto var_divided_varb = variable/variable_b;
    assert(graph::divide_cast(var_divided_varb).get() != nullptr &&
           "Expected divide node.");
    const BACKEND var_divided_varb_result = var_divided_varb->evaluate();
    assert(var_divided_varb_result.size() == 1 && "Expected single value.");
    assert(var_divided_varb_result.at(0) == 3.0/4.0 &&
           "Expected 3/4 for result.");

//  Test vector variables.
    auto varvec = graph::variable<BACKEND> (std::vector<double> ({2.0, 6.0}));
    assert((zero/varvec).get() == zero.get() && "Expected to recover zero.");
    assert((varvec/one).get() == varvec.get() &&
           "Expected to recover numerator.");

    auto varvec_divided_two = varvec/two;
    assert(graph::divide_cast(varvec_divided_two).get() != nullptr &&
           "Expect divide node.");
    const BACKEND varvec_divided_two_result = varvec_divided_two->evaluate();
    assert(varvec_divided_two_result.size() == 2 && "Size mismatch in result.");
    assert(varvec_divided_two_result.at(0) == 1 && "Expected 2/2 for result.");
    assert(varvec_divided_two_result.at(1) == 3 && "Expected 6/2 for result.");

    auto two_divided_varvec = two/varvec;
    assert(graph::divide_cast(two_divided_varvec).get() != nullptr &&
           "Expect divide node.");
    const BACKEND two_divided_varvec_result = two_divided_varvec->evaluate();
    assert(two_divided_varvec_result.size() == 2 && "Size mismatch in result.");
    assert(two_divided_varvec_result.at(0) == 1 && "Expected 2/2 for result.");
    assert(two_divided_varvec_result.at(1) == 2.0/6.0 &&
           "Expected 2/6 for result.");

    auto varvec_b = graph::variable<BACKEND> (std::vector<double> ({-3.0, 6.0}));
    auto varvec_divided_varvecb = varvec/varvec_b;
    assert(graph::divide_cast(varvec_divided_varvecb).get() != nullptr &&
           "Expect divide node.");
    const BACKEND varvec_divided_varvecb_result =
        varvec_divided_varvecb->evaluate();
    assert(varvec_divided_varvecb_result.size() == 2 &&
           "Size mismatch in result.");
    assert(varvec_divided_varvecb_result.at(0) == 2.0/-3.0 &&
           "Expected 2/-3 for result.");
    assert(varvec_divided_varvecb_result.at(1) == 1 &&
           "Expected 6/6 for result.");

    auto varvecb_divided_varvec = varvec_b/varvec;
    assert(graph::divide_cast(varvecb_divided_varvec).get() != nullptr &&
           "Expect divide node.");
    const BACKEND varvecb_divided_varvec_result =
        varvecb_divided_varvec->evaluate();
    assert(varvecb_divided_varvec_result.size() == 2 &&
           "Size mismatch in result.");
    assert(varvecb_divided_varvec_result.at(0) == -3.0/2.0 &&
           "Expected -3/2 for result.");
    assert(varvecb_divided_varvec_result.at(1) == 1 &&
           "Expected 6/6 for result.");

//  Test reduction short cut. If all the elements in the numerator are zero, an
//  denominator does not need to be evaluated. This test makes sure that a sum
//  or product is not used to avoid cases like {-1, 0, 1} which sum and product
//  are zero.
    auto var_sum_prod = graph::variable<BACKEND> (std::vector<double> ({-2.0, 2.0, 0.0}));
    auto var_sum_prod_divided_two = var_sum_prod/two;
    const BACKEND var_sum_prod_divided_two_result =
        var_sum_prod_divided_two->evaluate();
    assert(var_sum_prod_divided_two_result.at(0) == -2.0/2.0 &&
           "Expected -2/2 for result.");
    assert(var_sum_prod_divided_two_result.at(1) == 2.0/2.0 &&
           "Expected 2/2 for result.");
    assert(var_sum_prod_divided_two_result.at(2) == 0 &&
           "Expected 0/2 for result.");

//  Test derivatives.
//  d (x/c) / dx = dxdx/c + x d 1/c /dx = 1/c
    auto dvar_divided_two = var_divided_two->df(variable);
    const BACKEND dvar_divided_two_result = dvar_divided_two->evaluate();
    assert(dvar_divided_two_result.at(0) == 1.0/2.0 &&
           "Expected 1/2 for result.");

//  d (c/x) / dx = dc/dx x - c/x^2 dx/dx = -c/x^2
    auto dtwo_divided_var = two_divided_var->df(variable);
    const BACKEND dtwo_divided_var_result = dtwo_divided_var->evaluate();
    assert(dtwo_divided_var_result.at(0) == -2.0/(3.0*3.0) &&
           "Expected 2/3^2 for result.");
}

//------------------------------------------------------------------------------
///  @brief Tests for fma nodes.
//------------------------------------------------------------------------------
template<typename BACKEND> void test_fma() {
//  Three constant nodes should reduce to a single constant node with a*b + c.
    auto zero = graph::constant<BACKEND> (0);
    auto one = graph::constant<BACKEND> (1);
    auto two = graph::constant<BACKEND> (2);

    auto zero_times_one_plus_two = graph::fma(zero, one, two);
    auto zero_times_one_plus_two_cast =
        graph::constant_cast(zero_times_one_plus_two);
    assert(zero_times_one_plus_two_cast.get() != nullptr &&
           "Expected a constant type.");
    assert(zero_times_one_plus_two_cast.get() == two.get() &&
           "Expected two.");

    auto one_times_zero_plus_two = graph::fma(one, zero, two);
    auto one_times_zero_plus_two_cast =
        graph::constant_cast(one_times_zero_plus_two);
    assert(one_times_zero_plus_two_cast.get() != nullptr &&
           "Expected a constant type.");
    assert(one_times_zero_plus_two_cast.get() == two.get() &&
           "Expected two.");

    auto one_times_two_plus_zero = graph::fma(one, two, zero);
    auto one_times_two_plus_zero_cast =
        graph::constant_cast(one_times_two_plus_zero);
    assert(one_times_two_plus_zero_cast.get() != nullptr &&
           "Expected a constant type.");
    assert(one_times_two_plus_zero_cast.get() == two.get() &&
           "Expected two.");

    auto three = graph::constant<BACKEND> (3);
    auto one_two_three = graph::fma(one, two, three);
    const BACKEND one_two_three_result = one_two_three->evaluate();
    assert(one_two_three_result.size() == 1 && "Expected single value.");
    assert(one_two_three_result.at(0) == 5 && "Expected five for result");

    auto two_three_one = graph::fma(two, three, one);
    const BACKEND two_three_one_result = two_three_one->evaluate();
    assert(two_three_one_result.size() == 1 && "Expected single value.");
    assert(two_three_one_result.at(0) == 7 && "Expected seven for result");

//  Test a variable.
    auto var = graph::variable<BACKEND> (1);
    auto zero_times_var_plus_two = graph::fma(zero, var, two);
    auto zero_times_var_plus_two_cast =
        graph::constant_cast(zero_times_var_plus_two);
    assert(zero_times_var_plus_two_cast.get() != nullptr &&
           "Expected a constant type.");
    assert(zero_times_var_plus_two_cast.get() == two.get() &&
           "Expected two.");

    auto var_times_zero_plus_two = graph::fma(var, zero, two);
    auto var_times_zero_plus_two_cast =
        graph::constant_cast(var_times_zero_plus_two);
    assert(var_times_zero_plus_two_cast.get() != nullptr &&
           "Expected a constant type.");
    assert(var_times_zero_plus_two_cast.get() == two.get() &&
           "Expected two.");

    auto zero_times_two_plus_var = graph::fma(zero, two, var);
    auto zero_times_two_plus_var_cast =
        graph::variable_cast(zero_times_two_plus_var);
    assert(zero_times_two_plus_var_cast.get() != nullptr &&
           "Expected a variable type.");
    assert(zero_times_two_plus_var_cast.get() == var.get() &&
           "Expected var.");

//  Test derivative.
    auto constant_df = one_times_two_plus_zero->df(var);
    auto constant_df_cast = graph::constant_cast(constant_df);
    assert(constant_df_cast.get() != nullptr &&
           "Expected a constant node.");
    assert(constant_df_cast->is(0) &&
           "Expected zero.");

    auto zero_times_var_plus_two_df = zero_times_var_plus_two->df(var);
    auto zero_times_var_plus_two_df_cast =
        graph::constant_cast(zero_times_var_plus_two_df);
    assert(zero_times_var_plus_two_df_cast.get() != nullptr &&
           "Expected a constant node.");
    assert(zero_times_var_plus_two_df_cast->is(0) &&
           "Expected zero.");

    auto var_times_zero_plus_two_df = zero_times_var_plus_two->df(var);
    auto var_times_zero_plus_two_df_cast =
        graph::constant_cast(var_times_zero_plus_two_df);
    assert(var_times_zero_plus_two_df_cast.get() != nullptr &&
           "Expected a constant node.");
    assert(var_times_zero_plus_two_df_cast->is(0) &&
           "Expected zero.");

    auto zero_times_two_plus_var_df = zero_times_two_plus_var->df(var);
    auto zero_times_two_plus_var_df_cast =
        graph::constant_cast(zero_times_two_plus_var_df);
    assert(zero_times_two_plus_var_df_cast.get() != nullptr &&
           "Expected a constant node.");
    assert(zero_times_two_plus_var_df_cast->is(1) &&
           "Expected one.");
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
//------------------------------------------------------------------------------
template<typename BACKEND> void run_tests() {
    test_add<BACKEND> ();
    test_subtract<BACKEND> ();
    test_multiply<BACKEND> ();
    test_divide<BACKEND> ();
    test_fma<BACKEND> ();
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
