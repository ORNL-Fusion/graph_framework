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
#include "../graph_framework/math.hpp"

//------------------------------------------------------------------------------
///  @brief Tests for addition nodes.
//------------------------------------------------------------------------------
template<typename BACKEND> void test_add() {
//  Three constant nodes should reduce to a single constant node with added
//  operands.
    auto one = graph::constant<BACKEND> (1);
    auto three = one + one + one;
#ifdef USE_REDUCE
    assert(graph::constant_cast(three).get() && "Expected a constant type.");
#else
    assert(graph::add_cast(three).get() && "Expected a add node.");
#endif
    assert(three->evaluate()[0] == backend::base_cast<BACKEND> (3.0) &&
           "Expected the evaluation of one.");

    const BACKEND result_three = three->evaluate();
    assert(result_three.size() == 1 && "Expected single value.");
    assert(result_three.at(0) == backend::base_cast<BACKEND> (3.0) &&
           "Expected three for result");

//  Any zero nodes should reduce to the other operand.
    auto zero = graph::constant<BACKEND> (0);
    auto one_plus_zero = one + zero;
#ifdef USE_REDUCE
    assert(one_plus_zero.get() == one.get() &&
           "Expected to retrive the left side.");
#else
    assert(graph::add_cast(three).get() && "Expected a add node.");
#endif
    assert(one_plus_zero->evaluate()[0] == backend::base_cast<BACKEND> (1.0) &&
           "Expected the evaluation of one.");
    auto zero_plus_one = zero + one;
#ifdef USE_REDUCE
    assert(zero_plus_one.get() == one.get() &&
           "Expected to retrive the right side.");
#else
    assert(zero_plus_one->evaluate()[0] == backend::base_cast<BACKEND> (1.0) &&
           "Expected the evaluation of one.");
#endif

//  Test vector scalar quanties.
    auto vec_constant = graph::constant<BACKEND> (std::vector<typename BACKEND::base> ({3.0, 4.0}));
    auto vec_plus_zero = vec_constant + zero;
#ifdef USE_REDUCE
    assert(vec_plus_zero.get() == vec_constant.get() &&
           "Expected to retrive the left side.");
#else
    assert(graph::add_cast(vec_plus_zero).get() && "Expected an add node.");
    assert(vec_plus_zero->evaluate()[0] == backend::base_cast<BACKEND> (3.0) &&
           "Expected three for the first index");
    assert(vec_plus_zero->evaluate()[1] == backend::base_cast<BACKEND> (4.0) &&
           "Expected four for the second index");
#endif
    auto zero_plus_vec = zero + vec_constant;
#ifdef USE_REDUCE
    assert(zero_plus_vec.get() == vec_constant.get() &&
           "Expected to retrive the right side.");
#else
    assert(graph::add_cast(zero_plus_vec).get() && "Expected an add node.");
    assert(zero_plus_vec->evaluate()[0] == backend::base_cast<BACKEND> (3.0) &&
           "Expected three for the first index");
    assert(zero_plus_vec->evaluate()[1] == backend::base_cast<BACKEND> (4.0) &&
           "Expected four for the second index");
#endif

    auto vec_plus_one = vec_constant + one;
#ifdef USE_REDUCE
    assert(graph::constant_cast(vec_plus_one).get() &&
           "Expected a constant type.");
#else
    assert(graph::add_cast(vec_plus_one).get() && "Expected an add node.");
#endif
    auto vec_plus_one_result = vec_plus_one->evaluate();
    assert(vec_plus_one_result.size() == 2 && "Size mismatch in result.");
    assert(vec_plus_one_result.at(0) == backend::base_cast<BACKEND> (4.0) &&
           "Expected 3 + 1.");
    assert(vec_plus_one_result.at(1) == backend::base_cast<BACKEND> (5.0) &&
           "Expected 4 + 1.");

    auto one_plus_vec = one + vec_constant;
#ifdef USE_REDUCE
    assert(graph::constant_cast(one_plus_vec).get() &&
           "Expected a constant type.");
#else
    assert(graph::add_cast(one_plus_vec).get() && "Expected an add node.");
#endif
    auto one_plus_vec_result = one_plus_vec->evaluate();
    assert(one_plus_vec_result.size() == 2 && "Size mismatch in result.");
    assert(one_plus_vec_result.at(0) == backend::base_cast<BACKEND> (4.0) &&
           "Expected 1 + 3.");
    assert(one_plus_vec_result.at(1) == backend::base_cast<BACKEND> (5.0) &&
           "Expected 1 + 4.");

// Test vector vector quaties.
    auto vec_plus_vec = vec_constant + vec_constant;
    const BACKEND vec_plus_vec_result = vec_plus_vec->evaluate();
    assert(vec_plus_vec_result.size() == 2 &&
           "Size mismatch in result.");
    assert(vec_plus_vec_result.at(0) == backend::base_cast<BACKEND> (6.0) &&
           "Expected 3 + 3.");
    assert(vec_plus_vec_result.at(1) == backend::base_cast<BACKEND> (8.0) &&
           "Expected 4 + 4.");

//  Test variable quanities.
//  Any zero nodes should reduce to the other operand.
    auto variable = graph::variable<BACKEND> (1, "");
    auto var_plus_zero = variable + zero;
#ifdef USE_REDUCE
    assert(var_plus_zero.get() == variable.get() &&
           "Expected to retrive the left side.");
#else
    assert(graph::add_cast(var_plus_zero) && "Expected an add node.");
#endif
    auto zero_plus_var = zero + variable;
#ifdef USE_REDUCE
    assert(zero_plus_var.get() == variable.get() &&
           "Expected to retrive the right side.");
#else
    assert(graph::add_cast(zero_plus_var) && "Expected an add node.");
#endif

//  Variable plus a variable should return an add node.
    auto var_plus_var = variable + variable;
#ifdef USE_REDUCE
    assert(graph::multiply_cast(var_plus_var).get() &&
           "Expected an multiply node.");
#else
    assert(graph::add_cast(var_plus_var).get() &&
           "Expected an add node.");
#endif
    variable->set(backend::base_cast<BACKEND> (10.0));
    const BACKEND var_plus_var_result = var_plus_var->evaluate();
    assert(var_plus_var_result.size() == 1 && "Expected single value.");
    assert(var_plus_var_result.at(0) == backend::base_cast<BACKEND> (20.0) &&
           "Expected 10 + 10 for result");

//  Variable plus a variable should return an add node.
    auto variable_b = graph::variable<BACKEND> (1, "");
    auto var_plus_varb = variable + variable_b;
    assert(graph::add_cast(var_plus_varb).get() && "Expected an add node.");
    variable_b->set(backend::base_cast<BACKEND> (5.0));
    const BACKEND var_plus_varb_result = var_plus_varb->evaluate();
    assert(var_plus_varb_result.size() == 1 && "Expected single value.");
    assert(var_plus_varb_result.at(0) == backend::base_cast<BACKEND> (15.0) &&
           "Expected 10 + 5 for result");

//  Test variable vectors.
    auto varvec = graph::variable<BACKEND> (std::vector<typename BACKEND::base> ({10.0, 20.0}), "");
    auto varvec_plus_varvec = varvec + varvec;
    const BACKEND varvec_plus_varvec_result =
        varvec_plus_varvec->evaluate();
    assert(varvec_plus_varvec_result.size() == 2 &&
           "Size mismatch in result.");
    assert(varvec_plus_varvec_result.at(0) == backend::base_cast<BACKEND> (20.0) &&
           "Expected 10 + 10.");
    assert(varvec_plus_varvec_result.at(1) == backend::base_cast<BACKEND> (40.0) &&
           "Expected 20 + 20.");

//  Test derivatives
//  d (1 + x) / dx = d1/dx + dx/dx = 0 + 1 = 1
    auto one_plus_var = one + variable;
    auto done_plus_var = one_plus_var->df(variable);
#ifdef USE_REDUCE
    auto done_plus_constant_cast = graph::constant_cast(done_plus_var);
    assert(done_plus_constant_cast.get() && "Expected a constant type.");
    assert(done_plus_constant_cast->is(1) &&
           "Expected to reduce to a constant one.");
#else
    auto done_plus_add_cast = graph::add_cast(done_plus_var);
    assert(done_plus_add_cast.get() && "Expected an add node.");
    auto done_plus_constant_cast_left = graph::constant_cast(done_plus_add_cast->get_left());
    assert(done_plus_constant_cast_left.get() &&
           "Expected constant node for left.");
    assert(done_plus_constant_cast_left->is(0) && "Expected a value of zero.");
    auto done_plus_constant_cast_right = graph::constant_cast(done_plus_add_cast->get_right());
    assert(done_plus_constant_cast_right.get() &&
           "Expected constant node for right.");
    assert(done_plus_constant_cast_right->is(1) && "Expected a value of one.");
#endif
    assert(done_plus_var->evaluate()[0] == backend::base_cast<BACKEND> (1.0) &&
           "Expected value of one.");
    
//  Test common factors.
    auto var_a = graph::variable<BACKEND> (1, "");
    auto var_b = graph::variable<BACKEND> (1, "");
    auto var_c = graph::variable<BACKEND> (1, "");
    auto common_a = var_a*var_b + var_a*var_c;
#ifdef USE_REDUCE
    assert(graph::add_cast(common_a).get() == nullptr &&
           "Did not expect add node.");
    assert(graph::multiply_cast(common_a).get() &&
           "Expected multiply node.");
#else
    assert(graph::add_cast(common_a).get() &&
           "Expected add node.");
    assert(graph::multiply_cast(common_a).get() == nullptr &&
           "Did not expect multiply node.");
#endif

    auto common_b = var_a*var_b + var_b*var_c;
#ifdef USE_REDUCE
    assert(graph::add_cast(common_b).get() == nullptr &&
           "Did not expect add node.");
    assert(graph::multiply_cast(common_b).get() && "Expected multiply node.");
#else
    assert(graph::add_cast(common_b).get() && "Expected add node.");
    assert(graph::multiply_cast(common_b).get()  == nullptr &&
           "Did not expect multiply node.");
#endif

    auto common_c = var_a*var_c + var_b*var_c;
#ifdef USE_REDUCE
    assert(graph::add_cast(common_c).get() == nullptr &&
           "Did not expect add node.");
    assert(graph::multiply_cast(common_c).get() && "Expected multiply node.");
#else
    assert(graph::add_cast(common_c).get() && "Expected add node.");
    assert(graph::multiply_cast(common_c).get() == nullptr &&
           "Did not expect multiply node.");
#endif

//  Test common denominator.
    auto common_d = var_a/var_b + var_c/var_b;
#ifdef USE_REDUCE
    assert(graph::add_cast(common_d).get() == nullptr &&
           "Did not expect add node.");
    assert(graph::divide_cast(common_d).get() && "Expected divide node.");
#else
    assert(graph::add_cast(common_d).get() && "Expected add node.");
    assert(graph::divide_cast(common_d).get() == nullptr  &&
           "Did not expect divide node.");
#endif

//  Test is_match
    auto match = graph::constant<BACKEND> (1)*var_a
               + graph::constant<BACKEND> (1)*var_a;
#ifdef USE_REDUCE
    assert(graph::multiply_cast(match).get() && "Expected multiply node.");
#else
    assert(graph::add_cast(match).get() && "Expected add node.");
#endif

//  Reduce (a/y)^e + (b/y)^e -> (a^2 + b^2)/(y^e).
    auto var_d = graph::variable<BACKEND> (1, "");
    auto common_power1 = graph::pow(var_a/var_b,var_c) +
                         graph::pow(var_d/var_b,var_c);
#ifdef USE_REDUCE
    assert(graph::divide_cast(common_power1) && "Expected Divide node.");
#else
    assert(graph::add_cast(common_power1).get() && "Expected add node.");
#endif
//  Reduce (a/y)^e + b/y^e -> (a^2 + b)/(y^e).
    auto common_power2 = graph::pow(var_a/var_b,var_c) +
                         var_d/graph::pow(var_b,var_c);
#ifdef USE_REDUCE
    assert(graph::divide_cast(common_power2) && "Expected Divide node.");
#else
    assert(graph::add_cast(common_power2).get() && "Expected add node.");
#endif
    //  Reduce a/y^e + (b/y)^e -> (a + b^2)/(y^e).
    auto common_power3 = var_a/graph::pow(var_b,var_c) +
                         graph::pow(var_d/var_b,var_c);
#ifdef USE_REDUCE
    assert(graph::divide_cast(common_power3) && "Expected Divide node.");
#else
    assert(graph::add_cast(common_power3).get() && "Expected add node.");
#endif

//  v1 + -c*v2 -> v1 - c*v2
    auto negate = var_a + graph::constant<BACKEND> (-2)*var_b;
#ifdef USE_REDUCE
    assert(graph::subtract_cast(negate).get() && "Expected subtract node.");
#else
    assert(graph::add_cast(negate).get() && "Expected add node.");
#endif

//  -c1*v1 + v2 -> v2 - c*v1
    auto negate2 = graph::constant<BACKEND> (-2)*var_a + var_b;
#ifdef USE_REDUCE
    auto negate2_cast = graph::subtract_cast(negate2);
    assert(negate2_cast.get() && "Expected subtract node.");
    assert(negate2_cast->get_left()->is_match(var_b) && "Expected var_b.");
#else
    assert(graph::add_cast(negate2).get() && "Expected add node.");
#endif

//  (c1*v1 + c2) + (c3*v1 + c4) -> c5*v1 + c6
    auto addfma = graph::fma(var_b, var_a, var_d)
                + graph::fma(var_c, var_a, var_d);
#ifdef USE_REDUCE
    assert(graph::fma_cast(addfma).get() &&
           "Expected fused multiply add node.");
#else
    assert(graph::add_cast(addfma).get() && "Expected add node.");
#endif

//  Test cases like
//  (c1 + c2/x) + c3/x -> c1 + c4/x
//  (c1 - c2/x) + c3/x -> c1 + c4/x
    common_d = (one + three/var_a) + (one/var_a);
    auto common_d_acast = graph::add_cast(common_d);
    assert(common_d_acast.get() && "Expected add node.");
#ifdef USE_REDUCE
    assert(graph::constant_cast(common_d_acast->get_left()).get() &&
           "Expected constant on the left.");
#endif

    common_d = (one - three/var_a) + (one/var_a);
    common_d_acast = graph::add_cast(common_d);
    assert(common_d_acast.get() && "Expected add node.");
#ifdef USE_REDUCE
    assert(graph::constant_cast(common_d_acast->get_left()).get() &&
           "Expected constant on the left.");
#endif

//  c1*a + c2*b -> c1*(a + c3*b)
    auto constant_factor = three*variable + (one + one)*var_b;
#ifdef USE_REDUCE
    assert(graph::multiply_cast(constant_factor).get() &&
           "Expected multilpy node.");
#else
    assert(graph::add_cast(constant_factor).get() &&
           "Expected add node.");
#endif
    
//  Test is_match
    auto match1 = graph::constant<BACKEND> (1) + variable;
    auto match2 = graph::constant<BACKEND> (1) + variable;
    assert(match1->is_match(match2) && "Expected match");
}

//------------------------------------------------------------------------------
///  @brief Tests for subtract nodes.
//------------------------------------------------------------------------------
template<typename BACKEND> void test_subtract() {
//  Three constant nodes should reduce to a single constant node with added
//  operands.
    auto one = graph::constant<BACKEND> (1);
    auto zero = one - one;
#ifdef USE_REDUCE
    auto zero_cast = graph::constant_cast(zero);
    assert(zero_cast.get() && "Expected a constant type.");
    assert(zero_cast->is(0) && "Expected a value of zero.");
#else
    assert(graph::subtract_cast(zero).get() && "Expected an subtract node.");
#endif
    assert(zero->evaluate()[0] == backend::base_cast<BACKEND> (0) &&
           "Expected a value of zero.");

    auto neg_one = one - one - one;
#ifdef USE_REDUCE
    auto neg_one_cast = graph::constant_cast(neg_one);
    assert(neg_one_cast.get() && "Expected a constant type.");
    assert(neg_one_cast->is(-1) && "Expected a value of -1.");
#else
    assert(graph::subtract_cast(neg_one).get() && "Expected an subtract node.");
#endif
    assert(neg_one->evaluate()[0] == backend::base_cast<BACKEND> (-1.0) &&
           "Expected a value of -1.");

//  A right side zero node should reduce to left side.
    auto one_minus_zero = one - zero;
#ifdef USE_REDUCE
    assert(one_minus_zero.get() == one.get() &&
           "Expected to retrive the left side.");
#else
    assert(graph::subtract_cast(one_minus_zero).get() &&
           "Expected an subtract node.");
#endif
    assert(one_minus_zero->evaluate()[0] == backend::base_cast<BACKEND> (1.0) &&
           "Expected a value of 1.");

//  A left side zero node should reduce to a negative right side.
    auto zero_minus_one = zero - one;
#ifdef USE_REDUCE
    auto zero_minus_one_cast = graph::constant_cast(zero_minus_one);
    assert(zero_minus_one_cast.get() && "Expected a constant type.");
    assert(zero_minus_one_cast->is(-1) && "Expected -1 for result");
#else
    assert(graph::subtract_cast(zero_minus_one).get() && "Expected an subtract node.");
#endif
    assert(zero_minus_one->evaluate()[0] == backend::base_cast<BACKEND> (-1.0) &&
           "Expected a value of -1.");

//  Test vector scalar quantities.
    auto vec_constant_a = graph::constant<BACKEND> (std::vector<typename BACKEND::base> ({3.0, 4.0}));
    auto vec_minus_one = vec_constant_a - one;
#ifdef USE_REDUCE
    assert(graph::constant_cast(vec_minus_one).get() &&
           "Expected a constant type.");
#else
    assert(graph::subtract_cast(vec_minus_one).get() &&
           "Expected an subtract node.");
#endif
    auto vec_minus_one_result = vec_minus_one->evaluate();
    assert(vec_minus_one_result.size() == 2 && "Size mismatch in result.");
    assert(vec_minus_one_result.at(0) == backend::base_cast<BACKEND> (2.0) &&
           "Expected 3 - 1.");
    assert(vec_minus_one_result.at(1) == backend::base_cast<BACKEND> (3.0) &&
           "Expected 4 - 1.");

    auto vec_constant_b = graph::constant<BACKEND> (std::vector<typename BACKEND::base> ({2.0, 5.0}));
    auto one_minus_vec = one - vec_constant_b;
#ifdef USE_REDUCE
    assert(graph::constant_cast(one_minus_vec).get() &&
           "Expected a constant type.");
#else
    assert(graph::subtract_cast(one_minus_vec).get() &&
           "Expected an subtract node.");
#endif
    auto one_minus_vec_result = one_minus_vec->evaluate();
    assert(one_minus_vec_result.size() == 2 && "Size mismatch in result.");
    assert(one_minus_vec_result.at(0) == backend::base_cast<BACKEND> (-1.0) &&
           "Expected 1 - 2.");
    assert(one_minus_vec_result.at(1) == backend::base_cast<BACKEND> (-4.0) &&
           "Expected 1 - 5.");

//  Test vector vector quanties.
    auto vec_minus_vec = vec_constant_a - vec_constant_b;
#ifdef USE_REDUCE
    assert(graph::constant_cast(vec_minus_vec).get() &&
           "Expected a constant type.");
#else
    assert(graph::subtract_cast(vec_minus_vec).get() &&
           "Expected an subtract node.");
#endif
    const BACKEND vec_minus_vec_result = vec_minus_vec->evaluate();
    assert(vec_minus_vec_result.size() == 2 && "Expected a constant type.");
    assert(vec_minus_vec_result.at(0) == backend::base_cast<BACKEND> (1.0) &&
           "Expected 3 - 1 for result.");
    assert(vec_minus_vec_result.at(1) == backend::base_cast<BACKEND> (-1.0) &&
           "Expected 4 - 5 for result.");

//  Test variable quanities.
//  Any right side zero nodes should reduce to the other operand.
    auto variable = graph::variable<BACKEND> (1, "");
    auto var_minus_zero = variable - zero;
#ifdef USE_REDUCE
    assert(var_minus_zero.get() == variable.get() &&
           "Expected to retrive the left side.");
#else
    assert(graph::subtract_cast(var_minus_zero).get() &&
           "Expected an subtract node.");
#endif

//  Any right side zero should reduce to a the a multiply node.
    auto zero_minus_var = zero - variable;
#ifdef USE_REDUCE
    assert(graph::multiply_cast(zero_minus_var).get() &&
           "Expected multiply node.");
#else
    assert(graph::subtract_cast(zero_minus_var).get() &&
           "Expected an subtract node.");
#endif
    variable->set(backend::base_cast<BACKEND> (3.0));
    const BACKEND zero_minus_var_result = zero_minus_var->evaluate();
    assert(zero_minus_var_result.size() == 1 && "Expected single value.");
    assert(zero_minus_var_result.at(0) == backend::base_cast<BACKEND> (-3.0) &&
           "Expected 0 - 3 for result.");

//  Variable minus a variable should return an minus node.
    auto variable_b = graph::variable<BACKEND> (1, "");
    auto var_minus_var = variable - variable_b;
    assert(graph::subtract_cast(var_minus_var).get() &&
           "Expected a subtraction node.");
    variable_b->set(backend::base_cast<BACKEND> (10.0));
    const BACKEND var_minus_var_result = var_minus_var->evaluate();
    assert(var_minus_var_result.size() == 1 && "Expected single value.");
    assert(var_minus_var_result.at(0) == backend::base_cast<BACKEND> (-7) &&
           "Expected 3 - 10 for result");

//  Test variable vectors.
    auto varvec_a = graph::variable<BACKEND> (std::vector<typename BACKEND::base> ({10.0, 20.0}), "");
    auto varvec_b = graph::variable<BACKEND> (std::vector<typename BACKEND::base> ({-3.0, 5.0}), "");
    auto varvec_minus_varvec = varvec_a - varvec_b;
    const BACKEND varvec_minus_varvec_result =
        varvec_minus_varvec->evaluate();
    assert(varvec_minus_varvec_result.size() == 2 &&
           "Size mismatch in result.");
    assert(varvec_minus_varvec_result.at(0) == backend::base_cast<BACKEND> (13.0) &&
           "Expected 10 - -3.");
    assert(varvec_minus_varvec_result.at(1) == backend::base_cast<BACKEND> (15.0) &&
           "Expected 20 - 5.");

//  Test derivatives.
//  d (1 - x) / dx = d1/dx - dx/dx = 0 - 1 = -1
    auto one_minus_var = one - variable;
    auto done_minus_var = one_minus_var->df(variable);
#ifdef USE_REDUCE
    auto done_minus_var_cast = graph::constant_cast(done_minus_var);
    assert(done_minus_var_cast.get() && "Expected a constant type.");
    assert(done_minus_var_cast->is(-1) &&
           "Expected to reduce to a constant minus one.");
#else
    auto done_minus_var_cast = graph::subtract_cast(done_minus_var);
    assert(done_minus_var_cast.get() && "Expected an subtract node.");
    auto done_minus_var_cast_left = graph::constant_cast(done_minus_var_cast->get_left());
    assert(done_minus_var_cast_left.get() && "Expected a constant type.");
    assert(done_minus_var_cast_left->is(0) &&
           "Expected to reduce to a constant zero.");
    auto done_minus_var_cast_right = graph::constant_cast(done_minus_var_cast->get_right());
    assert(done_minus_var_cast_right.get() && "Expected a constant type.");
    assert(done_minus_var_cast_right->is(1) &&
           "Expected to reduce to a constant one.");
#endif

//  Test common factors.
    auto var_a = graph::variable<BACKEND> (1, "");
    auto var_b = graph::variable<BACKEND> (1, "");
    auto var_c = graph::variable<BACKEND> (1, "");
    auto common_a = var_a*var_b - var_a*var_c;
    assert(graph::add_cast(common_a).get() == nullptr &&
           "Did not expect add node.");
#ifdef USE_REDUCE
    assert(graph::multiply_cast(common_a).get() && "Expected multiply node.");
#else
    assert(graph::subtract_cast(common_a).get() && "Expected subtract node.");
#endif

    auto common_b = var_a*var_b - var_b*var_c;
    assert(graph::add_cast(common_b).get() == nullptr &&
           "Did not expect add node.");
#ifdef USE_REDUCE
    assert(graph::multiply_cast(common_b).get() && "Expected multiply node.");
#else
    assert(graph::subtract_cast(common_b).get() && "Expected subtract node.");
#endif

    auto common_c = var_a*var_c - var_b*var_c;
#ifdef USE_REDUCE
    assert(graph::add_cast(common_c).get() == nullptr &&
           "Did not expect add node.");
    assert(graph::multiply_cast(common_c).get() && "Expected multiply node.");
#else
    assert(graph::subtract_cast(common_c).get() && "Expected subtract node.");
#endif

//  Test common denominator.
    auto common_d = var_a/var_b - var_c/var_b;
#ifdef USE_REDUCE
    assert(graph::subtract_cast(common_d).get() == nullptr &&
           "Did not expect subtract node.");
    assert(graph::divide_cast(common_d).get() && "Expected divide node.");
#else
    assert(graph::subtract_cast(common_d).get() && "Expected subtract node.");
#endif

//  Test is_match
    auto match = graph::constant<BACKEND> (1)*var_a
               - graph::constant<BACKEND> (1)*var_a;
#ifdef USE_REDUCE
    auto match_cast = graph::constant_cast(match);
    assert(match_cast.get() && "Expected a constant type.");
    assert(match_cast->is(0) && "Expected zero node.");
#else
    assert(graph::subtract_cast(match).get() && "Expected subtract node.");
#endif

//  Reduce (a/y)^e - (b/y)^e -> (a^2 - b^2)/(y^e).
    auto var_d = graph::variable<BACKEND> (1, "");
    auto common_power1 = graph::pow(var_a/var_b,var_c) -
                         graph::pow(var_d/var_b,var_c);
#ifdef USE_REDUCE
    assert(graph::divide_cast(common_power1).get() && "Expected Divide node.");
#else
    assert(graph::subtract_cast(common_power1).get() &&
           "Expected subtract node.");
#endif
//  Reduce a/y^e - (b/y)^e -> (a - b^2)/(y^e).
    auto common_power2 = graph::pow(var_a/var_b,var_c) -
                         var_d/graph::pow(var_b,var_c);
#ifdef USE_REDUCE
    assert(graph::divide_cast(common_power2) && "Expected Divide node.");
#else
    assert(graph::subtract_cast(common_power2).get() &&
           "Expected subtract node.");
#endif
    auto common_power3 = var_d/graph::pow(var_b,var_c) -
                         graph::pow(var_a/var_b,var_c);
#ifdef USE_REDUCE
    assert(graph::divide_cast(common_power3) && "Expected Divide node.");
#else
    assert(graph::subtract_cast(common_power3).get() &&
           "Expected subtract node.");
#endif

//  v1 - -c*v2 -> v1 + c*v2
    auto negate = var_a - graph::constant<BACKEND> (-2)*var_b;
#ifdef USE_REDUCE
    assert(graph::add_cast(negate).get() && "Expected addition node.");
#else
    assert(graph::subtract_cast(negate).get() && "Expected subtract node.");
#endif

//  (c1*v1 + c2) - (c3*v1 + c4) -> c5*(v1 - c6)
    auto two = graph::constant<BACKEND> (2);
    auto three = graph::constant<BACKEND> (3);
    auto subfma = graph::fma(three, var_a, two)
                - graph::fma(two, var_a, three);
#ifdef USE_REDUCE
    assert(graph::multiply_cast(subfma).get() && "Expected a multiply node.");
#else
    assert(graph::subtract_cast(subfma).get() && "Expected a subtract node.");
#endif

//  Test cases like
//  (c1 + c2/x) - c3/x -> c1 + c4/x
//  (c1 - c2/x) - c3/x -> c1 - c4/x
    common_d = (one + three/var_a) - (one/var_a);
#ifdef USE_REDUCE
    auto common_d_acast = graph::add_cast(common_d);
    assert(common_d_acast.get() && "Expected add node.");
    assert(graph::constant_cast(common_d_acast->get_left()).get() &&
           "Expected constant on the left.");
#else
    assert(graph::subtract_cast(common_d).get() && "Expected a subtract node.");
#endif
    common_d = (one - three/var_a) - (one/var_a);
    auto common_d_scast = graph::subtract_cast(common_d);
    assert(common_d_scast.get() && "Expected subtract node.");
#ifdef USE_REDUCE
    assert(graph::constant_cast(common_d_scast->get_left()).get() &&
           "Expected constant on the left.");
#endif

//  c1*a - c2*b -> c1*(a - c2*b)
    auto common_factor = three*var_a - (one + one)*var_b;
#ifdef USE_REDUCE
    assert(graph::multiply_cast(common_factor).get() &&
           "Expected multilpy node.");
#else
    assert(graph::subtract_cast(common_factor).get() &&
           "Expected a subtract node.");
#endif
}

//------------------------------------------------------------------------------
///  @brief Tests for multiply nodes.
//------------------------------------------------------------------------------
template<typename BACKEND> void test_multiply() {
//  Three constant nodes should reduce to a single constant node with multiplied
//  operands.
    auto one = graph::constant<BACKEND> (1);
    auto one_cubed = one*one*one;
#ifdef USE_REDUCE
    assert(one_cubed.get() == one.get() && "Expected to reduce back to one");
#else
    assert(graph::multiply_cast(one_cubed) && "Expected a multiply node.");
#endif
    assert(one_cubed->evaluate()[0] == backend::base_cast<BACKEND> (1) &&
           "Expected one.");

//  Any zero nodes should reduce zero.
    auto zero = graph::constant<BACKEND> (0);
#ifdef USE_REDUCE
    assert((zero*one).get() == zero.get() && "Expected to reduce back to zero");
    assert((one*zero).get() == zero.get() && "Expected to reduce back to zero");
#else
    assert(graph::multiply_cast(zero*one).get() && "Multiply node.");
    assert(graph::multiply_cast(one*zero).get() && "Multiply node.");
#endif
    assert((zero*one)->evaluate()[0] == backend::base_cast<BACKEND> (0) &&
           "Expected zero.");
    assert((one*zero)->evaluate()[0] == backend::base_cast<BACKEND> (0) &&
           "Expected zero.");
    
//  Test constant times constant.
    auto two = graph::constant<BACKEND> (2);
    auto three = graph::constant<BACKEND> (3);
    auto two_times_three = two*three;
#ifdef USE_REDUCE
    assert(graph::constant_cast(two_times_three).get() &&
           "Expected a constant type.");
#else
    assert(graph::multiply_cast(two_times_three).get() &&
           "Expected a multiply node.");
#endif
    auto three_times_two = three*two;
#ifdef USE_REDUCE
    assert(graph::constant_cast(three_times_two).get() &&
           "Expected a constant type.");
#else
    assert(graph::multiply_cast(three_times_two).get() &&
           "Expected a multiply node.");
#endif
    const BACKEND two_times_three_result =
        two_times_three->evaluate();
    const BACKEND three_times_two_result =
        three_times_two->evaluate();
    assert(two_times_three_result.size() == 1 && "Expected single value.");
    assert(three_times_two_result.size() == 1 && "Expected single value.");
    assert(three_times_two_result.at(0) == backend::base_cast<BACKEND> (6.0) &&
           "Expected 3*2 for result.");
    assert(three_times_two_result.at(0) == two_times_three_result.at(0) &&
           "Expected 3*2 == 2*3.");

//  Test vec times constant.
    auto vec = graph::constant<BACKEND> (std::vector<typename BACKEND::base> ({4.0, 5.0}));
#ifdef USE_REDUCE
    assert((zero*vec).get() == zero.get() && "Expected left side zero.");
    assert((vec*zero).get() == zero.get() && "Expected right side zero.");
    assert((one*vec).get() == vec.get() && "Expected right side vec.");
    assert((vec*one).get() == vec.get() && "Expected left side vec.");
    assert((vec*zero)->evaluate().size() == 1 && "Expected size of zero.");
#else
    assert(graph::multiply_cast(zero*vec).get() && "Expected a multiply node.");
    assert(graph::multiply_cast(vec*zero).get() && "Expected a multiply node.");
    assert(graph::multiply_cast(one*vec).get() && "Expected a multiply node.");
    assert(graph::multiply_cast(vec*one).get() && "Expected a multiply node.");
    assert((vec*zero)->evaluate().size() == 2 && "Expected size of two.");
#endif
    assert((zero*vec)->evaluate().size() == 1 && "Expected size of zero.");
    assert((zero*vec)->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected value of zero.");
    assert((vec*zero)->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected value of zero.");
#ifndef USE_REDUCE
    assert((vec*zero)->evaluate()[1] == backend::base_cast<BACKEND> (0.0) &&
           "Expected value of zero.");
#endif
    assert((one*vec)->evaluate()[0] == backend::base_cast<BACKEND> (4.0) &&
           "Expected value of four.");
    assert((one*vec)->evaluate()[1] == backend::base_cast<BACKEND> (5.0) &&
           "Expected value of five.");
    assert((vec*one)->evaluate()[0] == backend::base_cast<BACKEND> (4.0) &&
           "Expected value of four.");
    assert((vec*one)->evaluate()[1] == backend::base_cast<BACKEND> (5.0) &&
           "Expected value of five.");

    auto two_times_vec = two*vec;
#ifdef USE_REDUCE
    assert(graph::constant_cast(two_times_vec).get() &&
           "Expected a constant type.");
#else
    assert(graph::multiply_cast(two_times_vec).get() &&
           "Expected a multiply mode.");
#endif

    auto vec_times_two = vec*two;
    const BACKEND two_times_vec_result = two_times_vec->evaluate();
    const BACKEND vec_times_two_result = vec_times_two->evaluate();
    assert(two_times_vec_result.size() == 2 && "Expected two values.");
    assert(vec_times_two_result.size() == 2 && "Expected two values.");
    assert(two_times_vec_result.at(0) == backend::base_cast<BACKEND> (8.0) &&
           "Expected 2*4 for result.");
    assert(two_times_vec_result.at(1) == backend::base_cast<BACKEND> (10.0) &&
           "Expected 2*5 for result.");
    assert(two_times_vec_result.at(0) == vec_times_two_result.at(0) &&
           "Expected 2*4 == 4*2.");
    assert(two_times_vec_result.at(1) == vec_times_two_result.at(1) &&
           "Expected 2*5 == 5*2.");

//  Test vec times vec.
    auto vec_times_vec = vec*vec;
#ifdef USE_REDUCE
    assert(graph::constant_cast(vec_times_vec).get() &&
           "Expected a constant type.");
#else
    assert(graph::multiply_cast(vec_times_vec).get() &&
           "Expected a multiply mode.");
#endif
    const BACKEND vec_times_vec_result = vec_times_vec->evaluate();
    assert(vec_times_vec_result.size() == 2 && "Expected two values.");
    assert(vec_times_vec_result.at(0) == backend::base_cast<BACKEND> (16.0) &&
           "Expected 4*4 for result.");
    assert(vec_times_vec_result.at(1) == backend::base_cast<BACKEND> (25.0) &&
           "Expected 5*5 for result.");

//  Test reduction short cut. If all the elements in the numerator are zero, an
//  denominator does not need to be evaluated. This test makes sure that a sum
//  or product is not used to avoid cases like {-1, 0, 1} which sum and product
//  are zero.
    auto vec_sum_prod = graph::constant<BACKEND> (std::vector<typename BACKEND::base> ({-1.0, 0.0, 1.0}));
    auto vec_sum_prod_multiply_two = vec_sum_prod*two;
    const BACKEND vec_sum_prod_multiply_two_result =
        vec_sum_prod_multiply_two->evaluate();
    assert(vec_sum_prod_multiply_two_result.at(0) == backend::base_cast<BACKEND> (-1.0) *
                                                     backend::base_cast<BACKEND> (2.0) &&
           "Expected -1/2 for result.");
    assert(vec_sum_prod_multiply_two_result.at(1) == backend::base_cast<BACKEND> (0.0) &&
           "Expected 0/2 for result.");
    assert(vec_sum_prod_multiply_two_result.at(2) == backend::base_cast<BACKEND> (1.0) *
                                                     backend::base_cast<BACKEND> (2.0) &&
           "Expected 1/2 for result.");

//  Test variable quanities.
//  Any zero should reduce back to zero.
    auto variable = graph::variable<BACKEND> (1, "");
#ifdef USE_REDUCE
    assert((variable*zero).get() == zero.get() &&
           "Expected to retrive the right side.");
    assert((zero*variable).get() == zero.get() &&
           "Expected to retrive the left side.");
#else
    assert(graph::multiply_cast(variable*zero).get() &&
           "Expected a multiply mode.");
    assert(graph::multiply_cast(zero*variable).get() &&
           "Expected a multiply mode.");
#endif
#ifdef USE_REDUCE
//  Any one should reduce to the opposite side.
    assert((variable*one).get() == variable.get() &&
           "Expected to retrive the left side.");
    assert((one*variable).get() == variable.get() &&
           "Expected to retrive the right side.");
#else
    assert(graph::multiply_cast(variable*one).get() &&
           "Expected a multiply mode.");
    assert(graph::multiply_cast(one*variable).get() &&
           "Expected a multiply mode.");
#endif
    assert((variable*zero)->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected value of zero.");
    assert((variable*zero)->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected value of zero.");

//  Varibale times a non 0 or 1 constant should reduce to a multiply node.
    auto two_times_var = two*variable;
    assert(graph::multiply_cast(two_times_var).get() &&
           "Expected multiply node.");
    variable->set(backend::base_cast<BACKEND> (6.0));
    const BACKEND two_times_var_result = two_times_var->evaluate();
    assert(two_times_var_result.size() == 1 && "Expected single value.");
    assert(two_times_var_result.at(0) == backend::base_cast<BACKEND> (12.0) &&
           "Expected 2*6 for result.");

//  Test variable vectors.
    auto varvec_a = graph::variable<BACKEND> (std::vector<typename BACKEND::base> ({4.0, -2.0}), "a");
    auto varvec_b = graph::variable<BACKEND> (std::vector<typename BACKEND::base> ({-4.0, -2.0}), "b");
    auto varvec_times_varvec = varvec_a*varvec_b;
    const BACKEND varvec_times_varvec_result =
        varvec_times_varvec->evaluate();
    assert(varvec_times_varvec_result.size() == 2 &&
           "Size mismatch in result.");
    assert(varvec_times_varvec_result.at(0) == backend::base_cast<BACKEND> (-16.0) &&
           "Expected 4*-4.");
    assert(varvec_times_varvec_result.at(1) == backend::base_cast<BACKEND> (4.0) &&
           "Expected -2*-2.");

//  Test reduction short cut. If all the elements in the numerator are zero, an
//  denominator does not need to be evaluated. This test makes sure that a sum
//  or product is not used to avoid cases like {-1, 0, 1} which sum and product
//  are zero.
    auto var_sum_prod = graph::variable<BACKEND> (std::vector<typename BACKEND::base> ({-2.0, 2.0, 0.0}), "");
    auto var_sum_prod_multiply_two = var_sum_prod*two;
    const BACKEND var_sum_prod_multiply_two_result =
        var_sum_prod_multiply_two->evaluate();
    assert(var_sum_prod_multiply_two_result.at(0) == backend::base_cast<BACKEND> (-2.0) *
                                                     backend::base_cast<BACKEND> (2.0) &&
           "Expected -2/2 for result.");
    assert(var_sum_prod_multiply_two_result.at(1) == backend::base_cast<BACKEND> (2.0) *
                                                     backend::base_cast<BACKEND> (2.0) &&
           "Expected 2/2 for result.");
    assert(var_sum_prod_multiply_two_result.at(2) == backend::base_cast<BACKEND> (0.0) &&
           "Expected 0/2 for result.");

//  Test derivatives.
//  d (c*x) / dx = dc/dx*x + c*dx/dx = c*1 = c;
#ifdef USE_REDUCE
    assert(two_times_var->df(variable).get() == two.get() &&
           "Expect to reduce back to the constant.");
#else
    assert(graph::add_cast(two_times_var->df(variable)) &&
           "Expected an add node.");
#endif
//  d (x*x) / dx = dx/dx*x + x*dx/dx = x + x = 2*x;
    auto varvec_sqrd = varvec_a*varvec_a;
    auto dvarvec_sqrd = varvec_sqrd->df(varvec_a);
#ifdef USE_REDUCE
    assert(graph::multiply_cast(dvarvec_sqrd).get() &&
           "Expected multiply node.");
#else
    assert(graph::add_cast(dvarvec_sqrd) &&
           "Expected an add node.");
#endif
    const BACKEND dvarvec_sqrd_result = dvarvec_sqrd->evaluate();
    assert(dvarvec_sqrd_result.size() == 2 && "Size mismatch in result.");
    assert(dvarvec_sqrd_result.at(0) == backend::base_cast<BACKEND> (8.0) &&
           "Expected 2*4 for result.");
    assert(dvarvec_sqrd_result.at(1) == backend::base_cast<BACKEND> (-4.0) &&
           "Expected 2*-2 for result.");

#ifdef USE_REDUCE
//  Variables should always go to the right and constant to he left.
    auto swap = multiply_cast(variable*two);
    assert(graph::constant_cast(swap->get_left()).get() &&
           "Expected a constant on he left");
    assert(graph::variable_cast(swap->get_right()).get() &&
           "Expected a variable on he right");
#endif
    
//  Test reduction of common constants c1*x*c2*y = c3*x*y.
    auto x1 = graph::constant<BACKEND> (2)*graph::variable<BACKEND> (1, "");
    auto x2 = graph::constant<BACKEND> (5)*graph::variable<BACKEND> (1, "");
    auto x3 = x1*x2;
    auto x3_cast = graph::multiply_cast(x3);
    assert(x3_cast.get() && "Expected a multiply node.");
#ifdef USE_REDUCE
    assert(graph::constant_cast(x3_cast->get_left()).get() &&
           "Expected a constant coefficent.");
#else
    assert(graph::multiply_cast(x3_cast->get_left()).get() &&
           "Expected a multipy node.");
#endif
    assert(graph::multiply_cast(x3_cast->get_right()).get() &&
           "Expected a multipy node.");

//  Test reduction of common constants x*c1*c2*y = c3*x*y.
    auto x4 = graph::variable<BACKEND> (1, "")*graph::constant<BACKEND> (2);
    auto x5 = graph::constant<BACKEND> (5)*graph::variable<BACKEND> (1, "");
    auto x6 = x4*x5;
    auto x6_cast = graph::multiply_cast(x6);
    assert(x6_cast.get() && "Expected a multiply node.");
#ifdef USE_REDUCE
    assert(graph::constant_cast(x6_cast->get_left()).get() &&
           "Expected a constant coefficent.");
#else
    assert(graph::multiply_cast(x6_cast->get_left()).get() &&
           "Expected a multipy node.");
#endif
    assert(graph::multiply_cast(x6_cast->get_right()).get() &&
           "Expected multipy node.");

//  Test reduction of common constants c1*x*y*c2 = c3*x*y.
    auto x7 = graph::constant<BACKEND> (2)*graph::variable<BACKEND> (1, "");
    auto x8 = graph::variable<BACKEND> (1, "")*graph::constant<BACKEND> (5);
    auto x9 = x7*x8;
    auto x9_cast = graph::multiply_cast(x9);
    assert(x9_cast.get() && "Expected a multiply node.");
#ifdef USE_REDUCE
    assert(graph::constant_cast(x9_cast->get_left()).get() &&
           "Expected a constant coefficent.");
#else
    assert(graph::multiply_cast(x9_cast->get_left()).get() &&
           "Expected a multipy node.");
#endif
    assert(graph::multiply_cast(x9_cast->get_right()).get() &&
           "Expected multipy node.");

//  Test reduction of common constants x*c1*y*c2 = c3*x*y.
    auto x10 = graph::variable<BACKEND> (1, "")*graph::constant<BACKEND> (2);
    auto x11 = graph::constant<BACKEND> (5)*graph::variable<BACKEND> (1, "");
    auto x12 = x10*x11;
    auto x12_cast = graph::multiply_cast(x12);
    assert(x12_cast.get() && "Expected a multiply node.");
#ifdef USE_REDUCE
    assert(graph::constant_cast(x12_cast->get_left()).get() &&
           "Expected a constant coefficent.");
#else
    assert(graph::multiply_cast(x12_cast->get_left()).get() &&
           "Expected a multipy node.");
#endif
    assert(graph::multiply_cast(x12_cast->get_right()).get() &&
           "Expected a multipy node.");

//  Test gathering of terms.
    auto v1 = graph::variable<BACKEND> (1, "v1");
    auto v2 = graph::variable<BACKEND> (1, "v2");
    auto gather_v1 = (v1*v2)*v1;
#ifdef USE_REDUCE
    assert(pow_cast(multiply_cast(gather_v1)->get_left()).get() &&
           "Expected power node.");
#else
    assert(multiply_cast(multiply_cast(gather_v1)->get_left()).get() &&
           "Expected a multipy node.");
#endif
    auto gather_v2 = (v2*v1)*v1;
#ifdef USE_REDUCE
    assert(pow_cast(multiply_cast(gather_v2)->get_left()).get() &&
           "Expected power node.");
#else
    assert(multiply_cast(multiply_cast(gather_v2)->get_left()).get() &&
           "Expected a multipy node.");
#endif
    auto gather_v3 = v1*(v1*v2);
#ifdef USE_REDUCE
    assert(pow_cast(multiply_cast(gather_v3)->get_left()).get() &&
           "Expected power node.");
#else
    assert(multiply_cast(multiply_cast(gather_v3)->get_right()).get() &&
           "Expected a multipy node.");
#endif
    auto gather_v4 = v1*(v2*v1);
#ifdef USE_REDUCE
    assert(pow_cast(multiply_cast(gather_v4)->get_left()).get() &&
           "Expected power node.");
#else
    assert(multiply_cast(multiply_cast(gather_v4)->get_right()).get() &&
           "Expected a multipy node.");
#endif

//  Test double multiply cases.
    auto gather_v5 = (v1*v2)*(v1*v2);
#ifdef USE_REDUCE
    auto gather_v5_cast = graph::pow_cast(gather_v5);
    assert(gather_v5_cast.get() && "Expected power node.");
    assert(graph::multiply_cast(gather_v5_cast->get_left()).get() &&
           "Expected multiply inside power.");
    assert(graph::constant_cast(gather_v5_cast->get_right())->is(2) &&
           "Expected power of 2.");
#else
    auto gather_v5_cast = graph::multiply_cast(gather_v5);
    assert(gather_v5_cast.get() && "Expected power node.");
    assert(graph::multiply_cast(gather_v5_cast->get_left()).get() &&
           "Expected multiply inside power.");
    assert(graph::multiply_cast(gather_v5_cast->get_right()).get() &&
           "Expected power of 2.");
#endif

//  Test gather of terms. This test is setup to trigger an infinite recursive
//  loop if a critical check is not in place no need to check the values.
    auto a = graph::variable<BACKEND> (1, "");
    auto aaa = (a*sqrt(a))*(a*sqrt(a));

//  Test power reduction.
    auto var_times_var = variable*variable;
#ifdef USE_REDUCE
    assert(graph::pow_cast(var_times_var).get() &&
           "Expected a power node.");
#else
    assert(graph::multiply_cast(var_times_var).get() &&
           "Expected a multiply node.");
#endif
    const BACKEND var_times_var_result = var_times_var->evaluate();
    assert(var_times_var_result.size() == 1 && "Expected single value.");
    assert(var_times_var_result.at(0) == backend::base_cast<BACKEND> (36) &&
           "Expected 6*6 for result.");
    
//  Test c1*(c2*v) -> c3*v
    auto c3 = two*(three*a);
    auto c3_cast = graph::multiply_cast(c3);
    assert(c3_cast.get() && "Expected multiply node.");
    assert(graph::constant_cast(c3_cast->get_left()) &&
           "Expected constant on the left.");
#ifdef USE_REDUCE
    assert(graph::variable_cast(c3_cast->get_right()) &&
           "Expected variable on the right.");
#else
    assert(graph::multiply_cast(c3_cast->get_right()) &&
           "Expected a multiply node on the right.");
#endif

//  Test (c1*v)*c2 -> c4*v
    auto c4 = (three*a)*two;
    auto c4_cast = graph::multiply_cast(c4);
    assert(c4_cast.get() && "Expected multiply node.");
#ifdef USE_REDUCE
    assert(graph::constant_cast(c4_cast->get_left()) &&
           "Expected constant on the left.");
    assert(graph::variable_cast(c4_cast->get_right()) &&
           "Expected variable on the right.");
#else
    assert(graph::multiply_cast(c4_cast->get_left()) &&
           "Expected a multiply node on the right.");
    assert(graph::constant_cast(c4_cast->get_right()) &&
           "Expected constant on the right.");
#endif

//  Test c1*(c2/v) -> c5/v
    auto c5 = two*(three/a);
#ifdef USE_REDUCE
    auto c5_cast = graph::divide_cast(c5);
    assert(c5_cast.get() && "Expected a divide node.");
#else
    auto c5_cast = graph::multiply_cast(c5);
    assert(c5_cast.get() && "Expected a mutliply node.");
#endif
    assert(graph::constant_cast(c5_cast->get_left()).get() &&
           "Expected constant in the numerator.");
#ifdef USE_REDUCE
    assert(graph::variable_cast(c5_cast->get_right()).get() &&
           "Expected variable in the denominator.");
#else
    assert(graph::divide_cast(c5_cast->get_right()).get() &&
           "Expected a divide node on the right.");
#endif

//  Test c1*(v/c2) -> c6*v
    auto c6 = two*(a/three);
    auto c6_cast = graph::multiply_cast(c6);
    assert(c6_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(c6_cast->get_left()).get() &&
           "Expected constant for the left.");
#ifdef USE_REDUCE
    assert(graph::variable_cast(c6_cast->get_right()).get() &&
           "Expected variable for the right.");
#else
    assert(graph::divide_cast(c6_cast->get_right()).get() &&
           "Expected a divide node on the right.");
#endif

//  Test (c2/v)*c1 -> c7/v
    auto c7 = (three/a)*two;
#ifdef USE_REDUCE
    auto c7_cast = graph::divide_cast(c7);
    assert(c7_cast.get() && "Expected a divide node.");
    assert(graph::constant_cast(c7_cast->get_left()).get() &&
           "Expected constant for the numerator.");
    assert(graph::variable_cast(c7_cast->get_right()).get() &&
           "Expected variable for the denominator.");
#else
    auto c7_cast = graph::multiply_cast(c7);
    assert(c7_cast.get() && "Expected a multiply node.");
    assert(graph::divide_cast(c7_cast->get_left()).get() &&
           "Expected a divide node.");
    assert(graph::constant_cast(c7_cast->get_right()).get() &&
           "Expected constant for the right.");
#endif

//  Test c1*(v/c2) -> c8*v
    auto c8 = two*(a/three);
    auto c8_cast = graph::multiply_cast(c8);
    assert(c8_cast.get() && "Expected divide node.");
    assert(graph::constant_cast(c8_cast->get_left()).get() &&
           "Expected constant for the left.");
#ifdef USE_REDUCE
    assert(graph::variable_cast(c8_cast->get_right()).get() &&
           "Expected variable for the right.");
#else
    assert(graph::divide_cast(c8_cast->get_right()).get() &&
           "Expected a divide node on the right.");
#endif

//  Test v1*(c*v2) -> c*(v1*v2)
    auto c9 = a*(three*variable);
    auto c9_cast = graph::multiply_cast(c9);
    assert(c9_cast.get() && "Expected multiply node.");
#ifdef USE_REDUCE
    assert(graph::constant_cast(c9_cast->get_left()).get() &&
           "Expected a constant node first.");
#else
    assert(graph::variable_cast(c9_cast->get_left()).get() &&
           "Expected a variable node first.");
#endif

//  Test v1*(v2*c) -> c*(v1*v2)
    auto c10 = a*(variable*three);
    auto c10_cast = graph::multiply_cast(c10);
    assert(c10_cast.get() && "Expected multiply node.");
#ifdef USE_REDUCE
    assert(graph::constant_cast(c10_cast->get_left()).get() &&
           "Expected a constant node first.");
#else
    assert(graph::variable_cast(c10_cast->get_left()).get() &&
           "Expected a variable node first.");
#endif

//  Test (c*v1)*v2) -> c*(v1*v2)
    auto c11 = (three*variable)*a;
    auto c11_cast = graph::multiply_cast(c11);
    assert(c11_cast.get() && "Expected multiply node.");
#ifdef USE_REDUCE
    assert(graph::constant_cast(c11_cast->get_left()).get() &&
           "Expected a constant node first.");
#else
    assert(graph::multiply_cast(c11_cast->get_left()).get() &&
           "Expected a multiply node first.");
#endif

//  Test (v1*c)*v2 -> c*(v1*v2)
    auto c12 = (variable*three)*a;
    auto c12_cast = graph::multiply_cast(c12);
    assert(c12_cast.get() && "Expected multiply node.");
#ifdef USE_REDUCE
    assert(graph::constant_cast(c12_cast->get_left()).get() &&
           "Expected constant node first.");
#else
    assert(graph::multiply_cast(c12_cast->get_left()).get() &&
           "Expected a multiply node first.");
#endif

//  Test a^b*a^c -> a^(b + c) -> a^d
    auto pow_bc = graph::pow(a, two)*graph::pow(a, three);
#ifdef USE_REDUCE
    auto pow_bc_cast = graph::pow_cast(pow_bc);
    assert(pow_bc_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_bc_cast->get_right()).get() &&
           "Expected constant exponent.");
#else
    assert(graph::multiply_cast(pow_bc).get() && "Expected a multiply node.");
#endif

//  Test a*a^c -> a^(1 + c) -> a^c2
    auto pow_c = a*graph::pow(a, three);
#ifdef USE_REDUCE
    auto pow_c_cast = graph::pow_cast(pow_c);
    assert(pow_c_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_c_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_c_cast->get_right())->is(4) &&
           "Expected constant exponent equal to 4.");
#else
    assert(graph::multiply_cast(pow_c).get() && "Expected a multiply node.");
#endif

//  Test a^b*a -> a^(b + 1) -> a^b2
    auto pow_b = graph::pow(a, two)*a;
#ifdef USE_REDUCE
    auto pow_b_cast = graph::pow_cast(pow_b);
    assert(pow_b_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_b_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_b_cast->get_right())->is(3) &&
           "Expected constant exponent equal to 3.");
#else
    assert(graph::multiply_cast(pow_b).get() && "Expected a multiply node.");
#endif

//  Test a^b*sqrt(a) -> a^(b + 0.5) -> a^b2
    auto pow_sqb = graph::pow(a, two)*graph::sqrt(a);
#ifdef USE_REDUCE
    auto pow_sqb_cast = graph::pow_cast(pow_sqb);
    assert(pow_sqb_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_sqb_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_sqb_cast->get_right())->is(2.5) &&
           "Expected constant exponent equal to 2.5.");
#else
    assert(graph::multiply_cast(pow_sqb).get() && "Expected a multiply node.");
#endif

//  Test sqrt(a)*a^c -> a^(0.5 + c) -> a^c2
    auto pow_sqc = graph::sqrt(a)*graph::pow(a, three);
#ifdef USE_REDUCE
    auto pow_sqc_cast = graph::pow_cast(pow_sqc);
    assert(pow_sqc_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_sqc_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_sqc_cast->get_right())->is(3.5) &&
           "Expected constant exponent equal to 3.5.");
#else
    assert(graph::multiply_cast(pow_sqc).get() && "Expected a multiply node.");
#endif

//  Test a*sqrt(a) -> a^(1.5)
    auto pow_asqa = a*graph::sqrt(a);
#ifdef USE_REDUCE
    auto pow_asqa_cast = graph::pow_cast(pow_asqa);
    assert(pow_asqa_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_asqa_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_asqa_cast->get_right())->is(1.5) &&
           "Expected constant exponent equal to 1.5.");
#else
    assert(graph::multiply_cast(pow_asqa).get() && "Expected a multiply node.");
#endif

//  Test sqrt(a)*a -> a^(1.5)
    auto pow_sqaa = graph::sqrt(a)*a;
#ifdef USE_REDUCE
    assert(pow_sqaa->is_match(pow_asqa) && "Expected to match.");
#else
    assert(graph::multiply_cast(pow_sqaa).get() && "Expected a multiply node.");
#endif

    //  (c*v)*v -> c*v^2
#ifdef USE_REDUCE
    auto test_var_move = [two](graph::shared_leaf<BACKEND> x) {
        auto var_move = (two*x)*x;
        auto var_move_cast = graph::multiply_cast(var_move);
        assert(var_move_cast.get() && "Expected multiply.");
        assert(!graph::is_variable_like(var_move_cast->get_left()) &&
               "Expected Non variable like in the left side.");
        assert(graph::is_variable_like(var_move_cast->get_right()) &&
               "Expected variable like in the right side.");
    };

    test_var_move(a);
    test_var_move(pow_sqaa);
    test_var_move(graph::sqrt(a));
#endif
}

//------------------------------------------------------------------------------
///  @brief Tests for divide nodes.
//------------------------------------------------------------------------------
template<typename BACKEND> void test_divide() {
// Check for potential divide by zero.
    auto zero = graph::constant<BACKEND> (0);
#ifdef USE_REDUCE
    assert((zero/zero).get() == zero.get() && "Expected to recover zero.");
#endif
    assert((zero/zero)->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected to recover zero.");

// A zero in the numerator should result in zero.
    auto one = graph::constant<BACKEND> (1);
#ifdef USE_REDUCE
    assert((zero/one).get() == zero.get() && "Expected to recover zero.");
#else
    assert(graph::divide_cast(zero/one).get() && "Expected a divide node.");
#endif
    assert((zero/one)->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected a value of zero.");

// A one in the denominator should result in numerator.
#ifdef USE_REDUCE
    assert((one/one).get() == one.get() && "Expected to recover one.");
#else
    assert(graph::divide_cast(one/one).get() && "Expected a divide node.");
#endif
    assert((one/one)->evaluate()[0] == backend::base_cast<BACKEND> (1.0) &&
           "Expected a value of one.");
    auto two = graph::constant<BACKEND> (2);
#ifdef USE_REDUCE
    assert((two/one).get() == two.get() && "Expected to recover two.");
#else
    assert(graph::divide_cast(two/one).get() && "Expected a divide node.");
#endif
    assert((two/one)->evaluate()[0] == backend::base_cast<BACKEND> (2.0) &&
           "Expected a value of zero.");

//  A value divided by it self should be a constant one.
    auto two_divided_two = two/two;
#ifdef USE_REDUCE
    auto two_divided_two_cast = graph::constant_cast(two_divided_two);
    assert(two_divided_two_cast.get() && "Expected a constant type.");
    assert(two_divided_two_cast->is(1) && "Expected 1 for result");
#else
    auto two_divided_two_cast = graph::divide_cast(two_divided_two);
    assert(two_divided_two_cast.get() && "Expected a divide node.");
#endif
    assert(two_divided_two->evaluate()[0] == backend::base_cast<BACKEND> (1.0) &&
           "Expected 1 for result");

//  A constant a divided by constant b should be a constant with value of a/b.
    auto three = graph::constant<BACKEND> (3);
    auto two_divided_three = two/three;
#ifdef USE_REDUCE
    auto two_divided_three_cast = graph::constant_cast(two_divided_three);
    assert(two_divided_three_cast.get() && "Expected a constant type.");
    assert(two_divided_three_cast->is(2.0/3.0) && "Expected 2/3 for result");
#else
    auto two_divided_three_cast = graph::divide_cast(two_divided_three);
    assert(two_divided_three.get() && "Expected a divide node.");
#endif
    assert(two_divided_three->evaluate()[0] == backend::base_cast<BACKEND> (2.0/3.0) &&
           "Expected 2/3 for result");

//  Test vector constants.
    auto vec = graph::constant<BACKEND>(std::vector<typename BACKEND::base> ({4.0, 3.0}));
#ifdef USE_REDUCE
    assert((zero/vec).get() == zero.get() && "Expected to recover zero.");
    assert((vec/one).get() == vec.get() && "Expected to recover numerator.");
#else
    assert(graph::divide_cast(zero/vec).get() && "Expected a divide node.");
    assert(graph::divide_cast(vec/one).get() && "Expected a divide node.");
#endif
    assert((zero/vec)->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected a value of zero.");
    assert((vec/one)->evaluate()[0] == backend::base_cast<BACKEND> (4.0) &&
           "Expected a value of zero.");
    assert((vec/one)->evaluate()[1] == backend::base_cast<BACKEND> (3.0) &&
           "Expected a value of zero.");
    auto vec_divided_vec = vec/vec;
#ifdef USE_REDUCE
    auto vec_divided_vec_cast = graph::constant_cast(vec_divided_vec);
    assert(vec_divided_vec_cast.get() && "Expected a constant type.");
    assert(vec_divided_vec_cast->is(1) && "Expected 1 for result");
#else
    assert(graph::divide_cast(vec_divided_vec).get() &&
           "Expected a constant type.");
    assert(vec_divided_vec->evaluate()[1] == backend::base_cast<BACKEND> (1.0) &&
           "Expected a value of zero.");
#endif
    assert(vec_divided_vec->evaluate()[0] == backend::base_cast<BACKEND> (1.0) &&
           "Expected a value of zero.");
    auto two_divided_vec = two/vec;
#ifdef USE_REDUCE
    assert(graph::constant_cast(two_divided_vec).get() &&
           "Expected a constant type.");
#else
    assert(graph::divide_cast(two_divided_vec).get() &&
           "Expected a divide node.");
#endif
    const BACKEND two_divided_vec_result =
        two_divided_vec->evaluate();
    assert(two_divided_vec_result.size() == 2 && "Size mismatch in result.");
    assert(two_divided_vec_result.at(0) == backend::base_cast<BACKEND> (2.0) /
                                           backend::base_cast<BACKEND> (4.0) &&
           "Expected 2/4 for result.");
    assert(two_divided_vec_result.at(1) == backend::base_cast<BACKEND> (2.0) /
                                           backend::base_cast<BACKEND> (3.0) &&
           "Expected 2/3 for result.");

//  Test reduction short cut. If all the elements in the numerator are zero, an
//  denominator does not need to be evaluated. This test makes sure that a sum
//  or product is not used to avoid cases like {-1, 0, 1} which sum and product
//  are zero.
    auto vec_sum_prod = graph::constant<BACKEND> (std::vector<typename BACKEND::base> ({-1.0, 0.0, 1.0}));
    auto vec_sum_prod_divided_two = vec_sum_prod/two;
    const BACKEND vec_sum_prod_divided_two_result =
        vec_sum_prod_divided_two->evaluate();
    assert(vec_sum_prod_divided_two_result.at(0) == backend::base_cast<BACKEND> (-1.0) /
                                                    backend::base_cast<BACKEND> (2.0) &&
           "Expected -1/2 for result.");
    assert(vec_sum_prod_divided_two_result.at(1) == backend::base_cast<BACKEND> (0.0) &&
           "Expected 0/2 for result.");
    assert(vec_sum_prod_divided_two_result.at(2) == backend::base_cast<BACKEND> (1.0) /
                                                    backend::base_cast<BACKEND> (2.0) &&
           "Expected 1/2 for result.");

//  Test variables.
    auto variable = graph::variable<BACKEND> (1, "");
#ifdef USE_REDUCE
    assert((zero/variable).get() == zero.get() && "Expected to recover zero.");
    assert((variable/one).get() == variable.get() &&
           "Expected to recover numerator.");
#else
    assert(graph::divide_cast(zero/variable).get() &&
           "Expected a divide node.");
    assert(graph::divide_cast(variable/one).get() &&
           "Expected a divide node.");
#endif
    assert((zero/variable)->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected a value of zero.");
    
    auto two_divided_var = two/variable;
    assert(graph::divide_cast(two_divided_var).get() &&
           "Expected divide node.");
    variable->set(backend::base_cast<BACKEND> (3.0));
    const BACKEND two_divided_var_result = two_divided_var->evaluate();
    assert(two_divided_var_result.size() == 1 && "Expected single value.");
    assert(two_divided_var_result.at(0) == backend::base_cast<BACKEND> (2.0) /
                                           backend::base_cast<BACKEND> (3.0) &&
           "Expected 2/3 for result.");

//  v/c1 -> (1/c1)*v -> c2*v
    auto var_divided_two = variable/two;
#ifdef USE_REDUCE
    assert(graph::multiply_cast(var_divided_two).get() &&
           "Expected a multiply node.");
#else
    assert(graph::divide_cast(var_divided_two).get() &&
           "Expected a divide node.");
#endif
    const BACKEND var_divided_two_result = var_divided_two->evaluate();
    assert(var_divided_two_result.size() == 1 && "Expected single value.");
    assert(var_divided_two_result.at(0) == backend::base_cast<BACKEND> (3.0) /
                                           backend::base_cast<BACKEND> (2.0) &&
           "Expected 3/2 for result.");

    auto var_divided_var = variable/variable;
#ifdef USE_REDUCE
    auto var_divided_var_cast = graph::constant_cast(var_divided_var);
    assert(var_divided_var_cast.get() && "Expeced constant node.");
    assert(var_divided_var_cast->is(1) && "Expeced one.");
#endif

    auto variable_b = graph::variable<BACKEND> (1, 4, "");
    auto var_divided_varb = variable/variable_b;
    assert(graph::divide_cast(var_divided_varb).get() &&
           "Expected divide node.");
    const BACKEND var_divided_varb_result = var_divided_varb->evaluate();
    assert(var_divided_varb_result.size() == 1 && "Expected single value.");
    assert(var_divided_varb_result.at(0) == backend::base_cast<BACKEND> (3.0) /
                                            backend::base_cast<BACKEND> (4.0) &&
           "Expected 3/4 for result.");

//  Test vector variables.
    auto varvec = graph::variable<BACKEND> (std::vector<typename BACKEND::base> ({2.0, 6.0}), "");
#ifdef USE_REDUCE
    assert((zero/varvec).get() == zero.get() && "Expected to recover zero.");
    assert((varvec/one).get() == varvec.get() &&
           "Expected to recover numerator.");
#else
    assert(graph::divide_cast(zero/varvec).get() && "Expected a divide node.");
    assert(graph::divide_cast(varvec/one).get() && "Expected a divide node.");
#endif
    assert((zero/varvec)->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected a value of zero.");
    assert((varvec/one)->evaluate()[0] == backend::base_cast<BACKEND> (2.0) &&
           "Expected a value of two.");
    assert((varvec/one)->evaluate()[1] == backend::base_cast<BACKEND> (6.0) &&
           "Expected a value of six.");

    auto varvec_divided_two = varvec/two;
#ifdef USE_REDUCE
    assert(graph::multiply_cast(varvec_divided_two).get() &&
           "Expect a mutliply node.");
#else
    assert(graph::divide_cast(varvec_divided_two).get() &&
           "Expect a divide node.");
#endif
    const BACKEND varvec_divided_two_result = varvec_divided_two->evaluate();
    assert(varvec_divided_two_result.size() == 2 && "Size mismatch in result.");
    assert(varvec_divided_two_result.at(0) == backend::base_cast<BACKEND> (1.0) &&
           "Expected 2/2 for result.");
    assert(varvec_divided_two_result.at(1) == backend::base_cast<BACKEND> (3.0) &&
           "Expected 6/2 for result.");

    auto two_divided_varvec = two/varvec;
    assert(graph::divide_cast(two_divided_varvec).get() &&
           "Expect divide node.");
    const BACKEND two_divided_varvec_result = two_divided_varvec->evaluate();
    assert(two_divided_varvec_result.size() == 2 && "Size mismatch in result.");
    assert(two_divided_varvec_result.at(0) == backend::base_cast<BACKEND> (1.0) &&
           "Expected 2/2 for result.");
    assert(two_divided_varvec_result.at(1) == backend::base_cast<BACKEND> (2.0) /
                                              backend::base_cast<BACKEND> (6.0) &&
           "Expected 2/6 for result.");

    auto varvec_b = graph::variable<BACKEND> (std::vector<typename BACKEND::base> ({-3.0, 6.0}), "");
    auto varvec_divided_varvecb = varvec/varvec_b;
    assert(graph::divide_cast(varvec_divided_varvecb).get() &&
           "Expect divide node.");
    const BACKEND varvec_divided_varvecb_result =
        varvec_divided_varvecb->evaluate();
    assert(varvec_divided_varvecb_result.size() == 2 &&
           "Size mismatch in result.");
    assert(varvec_divided_varvecb_result.at(0) == backend::base_cast<BACKEND> (2.0) /
                                                  backend::base_cast<BACKEND> (-3.0) &&
           "Expected 2/-3 for result.");
    assert(varvec_divided_varvecb_result.at(1) == backend::base_cast<BACKEND> (1.0) &&
           "Expected 6/6 for result.");

    auto varvecb_divided_varvec = varvec_b/varvec;
    assert(graph::divide_cast(varvecb_divided_varvec).get() &&
           "Expect divide node.");
    const BACKEND varvecb_divided_varvec_result =
        varvecb_divided_varvec->evaluate();
    assert(varvecb_divided_varvec_result.size() == 2 &&
           "Size mismatch in result.");
    assert(varvecb_divided_varvec_result.at(0) == backend::base_cast<BACKEND> (-3.0) /
                                                  backend::base_cast<BACKEND> (2.0) &&
           "Expected -3/2 for result.");
    assert(varvecb_divided_varvec_result.at(1) == backend::base_cast<BACKEND> (1.0) &&
           "Expected 6/6 for result.");

//  Test reduction short cut. If all the elements in the numerator are zero, an
//  denominator does not need to be evaluated. This test makes sure that a sum
//  or product is not used to avoid cases like {-1, 0, 1} which sum and product
//  are zero.
    auto var_sum_prod = graph::variable<BACKEND> (std::vector<typename BACKEND::base> ({-2.0, 2.0, 0.0}), "");
    auto var_sum_prod_divided_two = var_sum_prod/two;
    const BACKEND var_sum_prod_divided_two_result =
        var_sum_prod_divided_two->evaluate();
    assert(var_sum_prod_divided_two_result.at(0) == backend::base_cast<BACKEND> (-2.0) /
                                                    backend::base_cast<BACKEND> (2.0) &&
           "Expected -2/2 for result.");
    assert(var_sum_prod_divided_two_result.at(1) == backend::base_cast<BACKEND> (2.0) /
                                                    backend::base_cast<BACKEND> (2.0) &&
           "Expected 2/2 for result.");
    assert(var_sum_prod_divided_two_result.at(2) == backend::base_cast<BACKEND> (0.0) &&
           "Expected 0/2 for result.");

//  Test derivatives.
//  d (x/c) / dx = dxdx/c + x d 1/c /dx = 1/c
    auto dvar_divided_two = var_divided_two->df(variable);
    const BACKEND dvar_divided_two_result = dvar_divided_two->evaluate();
    assert(dvar_divided_two_result.at(0) == backend::base_cast<BACKEND> (1.0) /
                                            backend::base_cast<BACKEND> (2.0) &&
           "Expected 1/2 for result.");

//  d (c/x) / dx = dc/dx x - c/x^2 dx/dx = -c/x^2
    auto dtwo_divided_var = two_divided_var->df(variable);
    const BACKEND dtwo_divided_var_result = dtwo_divided_var->evaluate();
    assert(dtwo_divided_var_result.at(0) == backend::base_cast<BACKEND> (-2.0) /
                                            (backend::base_cast<BACKEND> (3.0) *
                                             backend::base_cast<BACKEND> (3.0)) &&
           "Expected 2/3^2 for result.");

//  Test is_match
    auto match = (graph::constant<BACKEND> (1) + variable)
               / (graph::constant<BACKEND> (1) + variable);
#ifdef USE_REDUCE
    auto match_cast = graph::constant_cast(match);
    assert(match_cast->is(1) &&
           "Expected one constant for result.");
#else
    assert(graph::divide_cast(match).get() &&
           "Expected a divide node.");
#endif

//  Test reduction of common constants (c1*x)/(c2*y) = c3*x/y.
    auto x1 = graph::constant<BACKEND> (2)*graph::variable<BACKEND> (1, "");
    auto x2 = graph::constant<BACKEND> (5)*graph::variable<BACKEND> (1, "");
    auto x3 = x1/x2;
#ifdef USE_REDUCE
    auto x3_cast = graph::multiply_cast(x3);
    assert(x3_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(x3_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::divide_cast(x3_cast->get_right()).get() &&
           "Expected multipy node.");
#else
    auto x3_cast = graph::divide_cast(x3);
    assert(x3_cast.get() && "Expected a divide node.");
    assert(graph::multiply_cast(x3_cast->get_left()).get() &&
           "Expected a multipy node.");
    assert(graph::multiply_cast(x3_cast->get_right()).get() &&
           "Expected a multipy node.");
#endif

//  Test reduction of common constants (c1*x)/(y*c2) = c3*x/y.
    auto x4 = graph::variable<BACKEND> (1, "")*graph::constant<BACKEND> (2);
    auto x5 = graph::constant<BACKEND> (5)*graph::variable<BACKEND> (1, "");
    auto x6 = x4/x5;
#ifdef USE_REDUCE
    auto x6_cast = graph::multiply_cast(x6);
    assert(x6_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(x6_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::divide_cast(x6_cast->get_right()).get() &&
           "Expected multipy node.");
#else
    auto x6_cast = graph::divide_cast(x6);
    assert(x6_cast.get() && "Expected a divide node.");
    assert(graph::multiply_cast(x6_cast->get_left()).get() &&
           "Expected a multipy node.");
    assert(graph::multiply_cast(x6_cast->get_right()).get() &&
           "Expected a multipy node.");
#endif

//  Test reduction of common constants (x*c1)/(c2*y) = c3*x/y.
    auto x7 = graph::constant<BACKEND> (2)*graph::variable<BACKEND> (1, "");
    auto x8 = graph::variable<BACKEND> (1, "")*graph::constant<BACKEND> (5);
    auto x9 = x7/x8;
#ifdef USE_REDUCE
    auto x9_cast = graph::multiply_cast(x9);
    assert(x9_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(x9_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::divide_cast(x9_cast->get_right()).get() &&
           "Expected multipy node.");
#else
    auto x9_cast = graph::divide_cast(x9);
    assert(x9_cast.get() && "Expected a divide node.");
    assert(graph::multiply_cast(x9_cast->get_left()).get() &&
           "Expected a multipy node.");
    assert(graph::multiply_cast(x9_cast->get_right()).get() &&
           "Expected a multipy node.");
#endif

//  Test reduction of common constants (x*c1)/(y*c2) = c3*x/y.
    auto x10 = graph::variable<BACKEND> (1, "")*graph::constant<BACKEND> (2);
    auto x11 = graph::constant<BACKEND> (5)*graph::variable<BACKEND> (1, "");
    auto x12 = x10/x11;
#ifdef USE_REDUCE
    auto x12_cast = graph::multiply_cast(x12);
    assert(x12_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(x12_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::divide_cast(x12_cast->get_right()).get() &&
           "Expected multipy node.");
#else
    auto x12_cast = graph::divide_cast(x12);
    assert(x12_cast.get() && "Expected a divide node.");
    assert(graph::multiply_cast(x12_cast->get_left()).get() &&
           "Expected a multipy node.");
    assert(graph::multiply_cast(x12_cast->get_right()).get() &&
           "Expected a multipy node.");
#endif

//  c1/(c2*v) -> c3/v
    auto c3 = two/(three*variable);
    auto c3_cast = graph::divide_cast(c3);
    assert(c3_cast.get() && "Expected divide node");
    assert(graph::constant_cast(c3_cast->get_left()).get() &&
           "Expected a constant in numerator.");
#ifdef USE_REDUCE
    assert(graph::variable_cast(c3_cast->get_right()).get() &&
           "Expected a variable in the denominator");
#endif

//  c1/(v*c2) -> c4/v
    auto c4 = two/(three*variable);
    auto c4_cast = graph::divide_cast(c4);
    assert(c4_cast.get() && "Expected divide node");
    assert(graph::constant_cast(c4_cast->get_left()).get() &&
           "Expected a constant in numerator.");
#ifdef USE_REDUCE
    assert(graph::variable_cast(c4_cast->get_right()).get() &&
           "Expected a variable in the denominator");
#endif

//  (c1*v)/c2 -> c5*v
    auto c5 = (two*variable)/three;
#ifdef USE_REDUCE
    auto c5_cast = graph::multiply_cast(c5);
    assert(c5_cast.get() && "Expected a multiply node");
    assert(graph::constant_cast(c5_cast->get_left()).get() &&
           "Expected a constant in the numerator");
    assert(graph::variable_cast(c5_cast->get_right()).get() &&
           "Expected a variable in the denominator.");
#else
    assert(graph::divide_cast(c5).get() && "Expected a divide node");
#endif

//  (v*c1)/c2 -> c5*v
    auto c6 = (variable*two)/three;
#ifdef USE_REDUCE
    auto c6_cast = graph::multiply_cast(c6);
    assert(c6_cast.get() && "Expected multiply node");
    assert(graph::constant_cast(c6_cast->get_left()).get() &&
           "Expected a constant in the numerator");
    assert(graph::variable_cast(c6_cast->get_right()).get() &&
           "Expected a variable in the denominator.");
#else
    assert(graph::divide_cast(c6).get() && "Expected a divide node");
#endif

//  (c*v1)/v2 -> c*(v1/v2)
    auto a = graph::variable<BACKEND> (1, "");
    auto c7 = (two*variable)/a;
#ifdef USE_REDUCE
    auto c7_cast = graph::multiply_cast(c7);
    assert(c7_cast.get() && "Expected multiply node");
    assert(graph::constant_cast(c7_cast->get_left()).get() &&
           "Expected a constant");
#else
    assert(graph::divide_cast(c7).get() && "Expected a divide node");
#endif

//  (v1*c)/v2 -> c*(v1/v2)
    auto c8 = (two*variable)/a;
#ifdef USE_REDUCE
    auto c8_cast = graph::multiply_cast(c8);
    assert(c8_cast.get() && "Expected multiply node");
    assert(graph::constant_cast(c8_cast->get_left()).get() &&
           "Expected a constant");
#else
    assert(graph::divide_cast(c8).get() && "Expected a divide node");
#endif

//  Test a^b/a^c -> a^(b - c)
    auto pow_bc = graph::pow(a, two)/graph::pow(a, three);
#ifdef USE_REDUCE
    auto pow_bc_cast = graph::pow_cast(pow_bc);
    assert(pow_bc_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_bc_cast->get_right()).get() &&
           "Expected constant exponent.");
#else
    assert(graph::divide_cast(pow_bc).get() && "Expected a divide node");
#endif

//  Test a/a^c -> a^(1 - c)
    auto pow_c = a/graph::pow(a, three);
#ifdef USE_REDUCE
    auto pow_c_cast = graph::pow_cast(pow_c);
    assert(pow_c_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_c_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_c_cast->get_right())->is(-2) &&
           "Expected constant exponent equal to -2.");
#else
    assert(graph::divide_cast(pow_c).get() && "Expected a divide node");
#endif

//  Test a^b/a -> a^(b - 1)
    auto pow_b = graph::pow(a, two)/a;
#ifdef USE_REDUCE
    assert(pow_b->is_match(a) && "Expected to recover a.");
#else
    assert(graph::divide_cast(pow_b).get() && "Expected a divide node");
#endif

//  Test a^b/sqrt(a) -> a^(b - 0.5)
    auto pow_sqb = graph::pow(a, two)/graph::sqrt(a);
#ifdef USE_REDUCE
    auto pow_sqb_cast = graph::pow_cast(pow_sqb);
    assert(pow_sqb_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_sqb_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_sqb_cast->get_right())->is(1.5) &&
           "Expected constant exponent equal to 1.5.");
#else
    assert(graph::divide_cast(pow_sqb).get() && "Expected a divide node");
#endif

//  Test sqrt(a)/a^c -> a^(0.5 - c)
    auto pow_sqc = graph::sqrt(a)/graph::pow(a, three);
#ifdef USE_REDUCE
    auto pow_sqc_cast = graph::pow_cast(pow_sqc);
    assert(pow_sqc_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_sqc_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_sqc_cast->get_right())->is(-2.5) &&
           "Expected constant exponent equal to -2.5.");
#else
    assert(graph::divide_cast(pow_sqc).get() && "Expected a divide node");
#endif

//  Test a/sqrt(a) -> sqrt(a)
    auto pow_asqa = a/graph::sqrt(a);
#ifdef USE_REDUCE
    auto pow_asqa_cast = graph::sqrt_cast(pow_asqa);
    assert(pow_asqa_cast.get() && "Expected sqrt node.");
#else
    assert(graph::divide_cast(pow_asqa).get() && "Expected a divide node");
#endif

//  Test sqrt(a)/a -> 1.0/sqrt(a)
    auto pow_sqaa = graph::sqrt(a)/a;
#ifdef USE_REDUCE
    auto pow_sqaa_cast = graph::divide_cast(pow_sqaa);
    assert(pow_sqaa_cast.get() && "Expected divide node.");
    assert(graph::sqrt_cast(pow_sqaa_cast->get_right()).get() &&
           "Expected sqrt in denominator.");
#else
    assert(graph::divide_cast(pow_sqaa).get() && "Expected a divide node");
#endif

//  (c*v)/v -> c*v
//  (c/v)/v -> c/v
#ifdef USE_REDUCE
    auto test_var_move = [two](graph::shared_leaf<BACKEND> x,
                               graph::shared_leaf<BACKEND> y) {
        auto var_move = (two*x)/y;
        auto var_move_cast = graph::multiply_cast(var_move);
        assert(var_move_cast.get() && "Expected multiply.");
        assert(!graph::is_variable_like(var_move_cast->get_left()) &&
               "Expected Non variable like in the left side.");
        assert(graph::is_variable_like(var_move_cast->get_right()) &&
               "Expected variable like in the right side.");
        
        auto var_move2 = (two/x)/y;
        auto var_move2_cast = graph::divide_cast(var_move2);
        assert(var_move2_cast.get() && "Expected multiply.");
        assert(!graph::is_variable_like(var_move2_cast->get_left()) &&
               "Expected Non variable like in the left side.");
        assert(graph::is_variable_like(var_move2_cast->get_right()) &&
               "Expected variable like in the right side.");
    };

    test_var_move(a, pow_sqc);
    test_var_move(pow_sqc, a);
    test_var_move(pow_asqa, pow_sqc);
#endif
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
#ifdef USE_REDUCE
    auto zero_times_one_plus_two_cast =
        graph::constant_cast(zero_times_one_plus_two);
    assert(zero_times_one_plus_two_cast.get() && "Expected a constant type.");
    assert(zero_times_one_plus_two_cast.get() == two.get() &&
           "Expected two.");
#else
    assert(graph::fma_cast(zero_times_one_plus_two).get() &&
           "Expected a fma node.");
#endif
    assert(zero_times_one_plus_two->evaluate()[0] == backend::base_cast<BACKEND> (2.0) &&
           "Expected a value of two.");

    auto one_times_zero_plus_two = graph::fma(one, zero, two);
#ifdef USE_REDUCE
    auto one_times_zero_plus_two_cast =
        graph::constant_cast(one_times_zero_plus_two);
    assert(one_times_zero_plus_two_cast.get() && "Expected a constant type.");
    assert(one_times_zero_plus_two_cast.get() == two.get() &&
           "Expected two.");
#else
    assert(graph::fma_cast(one_times_zero_plus_two).get() &&
           "Expected a fma node.");
#endif
    assert(one_times_zero_plus_two->evaluate()[0] == backend::base_cast<BACKEND> (2.0) &&
           "Expected a value of two.");
    
    auto one_times_two_plus_zero = graph::fma(one, two, zero);
#ifdef USE_REDUCE
    auto one_times_two_plus_zero_cast =
        graph::constant_cast(one_times_two_plus_zero);
    assert(one_times_two_plus_zero_cast.get() && "Expected a constant type.");
    assert(one_times_two_plus_zero_cast.get() == two.get() &&
           "Expected two.");
#else
    assert(graph::fma_cast(one_times_two_plus_zero).get() &&
           "Expected a fma node.");
#endif
    assert(one_times_two_plus_zero->evaluate()[0] == backend::base_cast<BACKEND> (2.0) &&
           "Expected a value of two.");

    auto three = graph::constant<BACKEND> (3);
    auto one_two_three = graph::fma(one, two, three);
    const BACKEND one_two_three_result = one_two_three->evaluate();
    assert(one_two_three_result.size() == 1 && "Expected single value.");
    assert(one_two_three_result.at(0) == backend::base_cast<BACKEND> (5.0) &&
           "Expected five for result");

    auto two_three_one = graph::fma(two, three, one);
    const BACKEND two_three_one_result = two_three_one->evaluate();
    assert(two_three_one_result.size() == 1 && "Expected single value.");
    assert(two_three_one_result.at(0) == backend::base_cast<BACKEND> (7) &&
           "Expected seven for result");

//  Test a variable.
    auto var = graph::variable<BACKEND> (1, "");
    auto zero_times_var_plus_two = graph::fma(zero, var, two);
#ifdef USE_REDUCE
    auto zero_times_var_plus_two_cast =
        graph::constant_cast(zero_times_var_plus_two);
    assert(zero_times_var_plus_two_cast.get() && "Expected a constant type.");
    assert(zero_times_var_plus_two_cast.get() == two.get() &&
           "Expected two.");
#else
    assert(graph::fma_cast(zero_times_var_plus_two).get() &&
           "Expected a fma node.");
#endif
    assert(zero_times_var_plus_two->evaluate()[0] == backend::base_cast<BACKEND> (2.0) &&
           "Expected a value of two.");

    auto var_times_zero_plus_two = graph::fma(var, zero, two);
#ifdef USE_REDUCE
    auto var_times_zero_plus_two_cast =
        graph::constant_cast(var_times_zero_plus_two);
    assert(var_times_zero_plus_two_cast.get() && "Expected a constant type.");
    assert(var_times_zero_plus_two_cast.get() == two.get() &&
           "Expected two.");
#else
    assert(graph::fma_cast(var_times_zero_plus_two).get() &&
           "Expected a fma node.");
#endif

    auto zero_times_two_plus_var = graph::fma(zero, two, var);
#ifdef USE_REDUCE
    auto zero_times_two_plus_var_cast =
        graph::variable_cast(zero_times_two_plus_var);
    assert(zero_times_two_plus_var_cast.get() && "Expected a variable type.");
    assert(zero_times_two_plus_var_cast.get() == var.get() &&
           "Expected var.");
#else
    assert(graph::fma_cast(zero_times_two_plus_var).get() &&
           "Expected a fma node.");
#endif

//  Test derivative.
    auto constant_df = one_times_two_plus_zero->df(var);
#ifdef USE_REDUCE
    auto constant_df_cast = graph::constant_cast(constant_df);
    assert(constant_df_cast.get() && "Expected a constant node.");
    assert(constant_df_cast->is(0) && "Expected zero.");
#endif
    assert(constant_df->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected a value of zero.");
    
    auto zero_times_var_plus_two_df = zero_times_var_plus_two->df(var);
#ifdef USE_REDUCE
    auto zero_times_var_plus_two_df_cast =
        graph::constant_cast(zero_times_var_plus_two_df);
    assert(zero_times_var_plus_two_df_cast.get() &&
           "Expected a constant node.");
    assert(zero_times_var_plus_two_df_cast->is(0) && "Expected zero.");
#endif
    assert(zero_times_var_plus_two_df->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected a value of zero.");

    auto var_times_zero_plus_two_df = zero_times_var_plus_two->df(var);
#ifdef USE_REDUCE
    auto var_times_zero_plus_two_df_cast =
        graph::constant_cast(var_times_zero_plus_two_df);
    assert(var_times_zero_plus_two_df_cast.get() &&
           "Expected a constant node.");
    assert(var_times_zero_plus_two_df_cast->is(0) && "Expected zero.");
#endif
    assert(var_times_zero_plus_two_df->evaluate()[0] == backend::base_cast<BACKEND> (0.0) &&
           "Expected a value of zero.");

    auto zero_times_two_plus_var_df = zero_times_two_plus_var->df(var);
#ifdef USE_REDUCE
    auto zero_times_two_plus_var_df_cast =
        graph::constant_cast(zero_times_two_plus_var_df);
    assert(zero_times_two_plus_var_df_cast.get() &&
           "Expected a constant node.");
    assert(zero_times_two_plus_var_df_cast->is(1) && "Expected one.");
#endif
    assert(zero_times_two_plus_var_df->evaluate()[0] == backend::base_cast<BACKEND> (1.0) &&
           "Expected a value of one.");

//  Test reduction.
    auto var_a = graph::variable<BACKEND> (1, "");
    auto var_b = graph::variable<BACKEND> (1, "");
    auto var_c = graph::variable<BACKEND> (1, "");

    auto reduce1 = graph::fma(var_a, var_b, var_a*var_c);
#ifdef USE_REDUCE
    auto reduce1_cast = graph::multiply_cast(reduce1);
    assert(reduce1_cast.get() && "Expected multiply node.");
    assert(reduce1_cast->get_right()->is_match(var_a) &&
           "Expected common var_a");
#else
    assert(graph::fma_cast(reduce1).get() && "Expected a fma node.");
#endif

    auto reduce2 = graph::fma(var_a, var_b, var_b*var_c);
#ifdef USE_REDUCE
    auto reduce2_cast = graph::multiply_cast(reduce2);
    assert(reduce2_cast.get() && "Expected multiply node.");
    assert(reduce2_cast->get_right()->is_match(var_b) &&
           "Expected common var_b");
#else
    assert(graph::fma_cast(reduce2).get() && "Expected a fma node.");
#endif

    auto reduce3 = graph::fma(var_a, var_b, var_c*var_a);
#ifdef USE_REDUCE
    auto reduce3_cast = graph::multiply_cast(reduce3);
    assert(reduce3_cast.get() && "Expected multiply node.");
    assert(reduce3_cast->get_right()->is_match(var_a) &&
           "Expected common var_a");
#else
    assert(graph::fma_cast(reduce3).get() && "Expected a fma node.");
#endif

    auto reduce4 = graph::fma(var_a, var_b, var_c*var_b);
#ifdef USE_REDUCE
    auto reduce4_cast = graph::multiply_cast(reduce4);
    assert(reduce4_cast.get() && "Expected multiply node.");
    assert(reduce4_cast->get_right()->is_match(var_b) &&
           "Expected common var_b");
#else
    assert(graph::fma_cast(reduce4).get() && "Expected a fma node.");
#endif

#ifdef USE_REDUCE
    assert(graph::multiply_cast(graph::fma(two, var_a, one)).get() &&
           "Expected multiply node.");
#else
    assert(graph::fma_cast(graph::fma(two, var_a, one)).get() &&
           "Expected a fma node.");
#endif
    
//  fma(c1*a,b,c2*d) -> c1*(a*b + c2/c1*d)
#ifdef USE_REDUCE
    assert(graph::multiply_cast(graph::fma(two*var_b,
                                           var_a,
                                           two*two*var_b)).get() &&
           "Expected multiply node.");
#else
    assert(graph::fma_cast(graph::fma(two*var_b,
                                      var_a,
                                      two*two*var_b)).get() &&
           "Expected a fma node.");
#endif

//  fma(c1*a,b,c2/d) -> c1*(a*b + c1/(c2*d))
//  fma(c1*a,b,d/c2) -> c1*(a*b + d/(c1*c2))
#ifdef USE_REDUCE
    assert(graph::multiply_cast(graph::fma(two*var_b,
                                           var_a,
                                           two*two/var_b)).get() &&
           "Expected multiply node.");
    assert(graph::multiply_cast(graph::fma(two*var_b,
                                           var_a,
                                           var_b/(two*two))).get() &&
           "Expected multiply node.");
#else
    assert(graph::fma_cast(graph::fma(two*var_b,
                                      var_a,
                                      two*two/var_b)).get() &&
           "Expected a fma node.");
    assert(graph::fma_cast(graph::fma(two*var_b,
                                      var_a,
                                      var_b/(two*two))).get() &&
           "Expected a fma node.");
#endif

//  fma(a,v1,b*v2) -> (a + b*v1/v2)*v1
//  fma(a,v1,c*b*v2) -> (a + c*b*v1/v2)*v1
#ifdef USE_REDUCE
    assert(graph::multiply_cast(graph::fma(two,
                                           var_a,
                                           two*sqrt(var_a))).get() &&
           "Expected multiply node.");
    assert(graph::multiply_cast(graph::fma(two,
                                           var_a,
                                           two*(var_b*sqrt(var_a)))).get() &&
           "Expected multiply node.");
#else
    assert(graph::fma_cast(graph::fma(two,
                                      var_a,
                                      two*sqrt(var_a))).get() &&
           "Expected a fma node.");
    assert(graph::fma_cast(graph::fma(two,
                                      var_a,
                                      two*(var_b*sqrt(var_a)))).get() &&
           "Expected a fma node.");
#endif
}

//------------------------------------------------------------------------------
///  @brief Tests function for variable like expressions.
//------------------------------------------------------------------------------
template<typename BACKEND> void test_variable_like() {
    auto a = graph::variable<BACKEND> (1, "");
    auto c = graph::constant<BACKEND> (1);
    
    assert(graph::is_variable_like(a) && "Expected a to be variable like.");
    assert(graph::is_variable_like(graph::sqrt(a)) &&
           "Expected sqrt(a) to be variable like.");
    assert(graph::is_variable_like(graph::pow(a, c)) &&
           "Expected a^c to be variable like.");
    
    assert(!graph::is_variable_like(c) &&
           "Expected c to not be variable like.");
    assert(!graph::is_variable_like(graph::sqrt(c)) &&
           "Expected sqrt(c) to not be variable like.");
    assert(!graph::is_variable_like(graph::pow(c, a)) &&
           "Expected c^a to not be variable like.");
    
    assert(graph::get_argument(a)->is_match(a) &&
           "Expected argument of a.");
    assert(graph::get_argument(graph::sqrt(a))->is_match(a) &&
           "Expected argument of a.");
    assert(graph::get_argument(graph::pow(a, c))->is_match(a) &&
           "Expected argument of a.");
    
    assert(graph::is_same_variable_like(a, graph::sqrt(a)) &&
           "Expected same.");
    assert(graph::is_same_variable_like(graph::sqrt(a), a) &&
           "Expected same.");
    assert(graph::is_same_variable_like(a, graph::pow(a, c)) &&
           "Expected same.");
    assert(graph::is_same_variable_like(graph::pow(a, c), a) &&
           "Expected same.");
    assert(graph::is_same_variable_like(graph::sqrt(a),
                                        graph::pow(a, c)) &&
           "Expected same.");
    assert(graph::is_same_variable_like(graph::pow(a, c),
                                        graph::sqrt(a)) &&
           "Expected same.");
    assert(!graph::is_same_variable_like(graph::pow(c, a),
                                         graph::sqrt(a)) &&
           "Expected different.");
    
    auto b = graph::variable<BACKEND> (1, "");
    assert(!graph::is_same_variable_like(a, graph::sqrt(b)) &&
           "Expected different.");
    assert(!graph::is_same_variable_like(graph::sqrt(a), b) &&
           "Expected different.");
    assert(!graph::is_same_variable_like(a, graph::pow(b, c)) &&
           "Expected different.");
    assert(!graph::is_same_variable_like(graph::pow(a, c), b) &&
           "Expected different.");
    assert(!graph::is_same_variable_like(graph::sqrt(a),
                                         graph::pow(b, c)) &&
           "Expected different.");
    assert(!graph::is_same_variable_like(graph::pow(a, c),
                                         graph::sqrt(b)) &&
           "Expected different.");
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
//------------------------------------------------------------------------------
template<typename BACKEND> void run_tests() {
    test_variable_like<BACKEND> ();
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
    run_tests<backend::cpu<float>> ();
    run_tests<backend::cpu<double>> ();
    run_tests<backend::cpu<std::complex<float>>> ();
    run_tests<backend::cpu<std::complex<double>>> ();
}
