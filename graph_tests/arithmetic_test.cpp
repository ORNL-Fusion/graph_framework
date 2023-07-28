//------------------------------------------------------------------------------
///  @file arithmetic\_test.cpp
///  @brief Tests for arithmetic nodes.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/math.hpp"
#include "../graph_framework/arithmetic.hpp"
#include "../graph_framework/piecewise.hpp"

//------------------------------------------------------------------------------
///  @brief Tests for addition nodes.
//------------------------------------------------------------------------------
template<typename T> void test_add() {
//  Three constant nodes should reduce to a single constant node with added
//  operands.
    auto one = graph::one<T> ();
    auto three = one + one + one;
    assert(graph::constant_cast(three).get() && "Expected a constant type.");
    assert(three->evaluate()[0] == static_cast<T> (3.0) &&
           "Expected the evaluation of one.");

    const backend::buffer<T> result_three = three->evaluate();
    assert(result_three.size() == 1 && "Expected single value.");
    assert(result_three.at(0) == static_cast<T> (3.0) &&
           "Expected three for result");

//  Any zero nodes should reduce to the other operand.
    auto zero = graph::zero<T> ();
    auto one_plus_zero = one + zero;
    assert(one_plus_zero.get() == one.get() &&
           "Expected to retrive the left side.");
    assert(one_plus_zero->evaluate()[0] == static_cast<T> (1.0) &&
           "Expected the evaluation of one.");
    auto zero_plus_one = zero + one;
    assert(zero_plus_one.get() == one.get() &&
           "Expected to retrive the right side.");

//  Test variable quanities.
//  Any zero nodes should reduce to the other operand.
    auto variable = graph::variable<T> (1, "");
    auto var_plus_zero = variable + zero;
    assert(var_plus_zero.get() == variable.get() &&
           "Expected to retrive the left side.");
    auto zero_plus_var = zero + variable;
    assert(zero_plus_var.get() == variable.get() &&
           "Expected to retrive the right side.");

//  Variable plus a variable should return an add node.
    auto var_plus_var = variable + variable;
    assert(graph::multiply_cast(var_plus_var).get() &&
           "Expected an multiply node.");
    variable->set(static_cast<T> (10.0));
    const backend::buffer<T> var_plus_var_result = var_plus_var->evaluate();
    assert(var_plus_var_result.size() == 1 && "Expected single value.");
    assert(var_plus_var_result.at(0) == static_cast<T> (20.0) &&
           "Expected 10 + 10 for result");

//  Variable plus a variable should return an add node.
    auto variable_b = graph::variable<T> (1, "");
    auto var_plus_varb = variable + variable_b;
    assert(graph::add_cast(var_plus_varb).get() && "Expected an add node.");
    variable_b->set(static_cast<T> (5.0));
    const backend::buffer<T> var_plus_varb_result = var_plus_varb->evaluate();
    assert(var_plus_varb_result.size() == 1 && "Expected single value.");
    assert(var_plus_varb_result.at(0) == static_cast<T> (15.0) &&
           "Expected 10 + 5 for result");

//  Test variable vectors.
    auto varvec = graph::variable<T> (std::vector<T> ({10.0, 20.0}), "");
    auto varvec_plus_varvec = varvec + varvec;
    const backend::buffer<T> varvec_plus_varvec_result =
        varvec_plus_varvec->evaluate();
    assert(varvec_plus_varvec_result.size() == 2 &&
           "Size mismatch in result.");
    assert(varvec_plus_varvec_result.at(0) == static_cast<T> (20.0) &&
           "Expected 10 + 10.");
    assert(varvec_plus_varvec_result.at(1) == static_cast<T> (40.0) &&
           "Expected 20 + 20.");

//  Test derivatives
//  d (1 + x) / dx = d1/dx + dx/dx = 0 + 1 = 1
    auto one_plus_var = one + variable;
    auto done_plus_var = one_plus_var->df(variable);
    auto done_plus_constant_cast = graph::constant_cast(done_plus_var);
    assert(done_plus_constant_cast.get() && "Expected a constant type.");
    assert(done_plus_constant_cast->is(1) &&
           "Expected to reduce to a constant one.");
    assert(done_plus_var->evaluate()[0] == static_cast<T> (1.0) &&
           "Expected value of one.");
    
//  Test common factors.
    auto var_a = graph::variable<T> (1, "");
    auto var_b = graph::variable<T> (1, "");
    auto var_c = graph::variable<T> (1, "");
    auto common_a = var_a*var_b + var_a*var_c;
    assert(graph::add_cast(common_a).get() == nullptr &&
           "Did not expect add node.");
    assert(graph::multiply_cast(common_a).get() &&
           "Expected multiply node.");

    auto common_b = var_a*var_b + var_b*var_c;
    assert(graph::add_cast(common_b).get() == nullptr &&
           "Did not expect add node.");
    assert(graph::multiply_cast(common_b).get() && "Expected multiply node.");

    auto common_c = var_a*var_c + var_b*var_c;
    assert(graph::add_cast(common_c).get() == nullptr &&
           "Did not expect add node.");
    assert(graph::multiply_cast(common_c).get() && "Expected multiply node.");

//  Test common denominator.
    auto common_d = var_a/var_b + var_c/var_b;
    assert(graph::add_cast(common_d).get() == nullptr &&
           "Did not expect add node.");
    assert(graph::divide_cast(common_d).get() && "Expected divide node.");

//  Test is_match
    auto match = graph::one<T> ()*var_a
               + graph::one<T> ()*var_a;
    assert(graph::multiply_cast(match).get() && "Expected multiply node.");

//  Reduce (a/y)^e + (b/y)^e -> (a^2 + b^2)/(y^e).
    auto var_d = graph::variable<T> (1, "");
    auto common_power1 = graph::pow(var_a/var_b,var_c) +
                         graph::pow(var_d/var_b,var_c);
    assert(graph::divide_cast(common_power1) && "Expected Divide node.");
//  Reduce (a/y)^e + b/y^e -> (a^2 + b)/(y^e).
    auto common_power2 = graph::pow(var_a/var_b,var_c) +
                         var_d/graph::pow(var_b,var_c);
    assert(graph::divide_cast(common_power2) && "Expected Divide node.");
    //  Reduce a/y^e + (b/y)^e -> (a + b^2)/(y^e).
    auto common_power3 = var_a/graph::pow(var_b,var_c) +
                         graph::pow(var_d/var_b,var_c);
    assert(graph::divide_cast(common_power3) && "Expected Divide node.");

//  v1 + -c*v2 -> v1 - c*v2
//    auto negate = var_a + graph::constant(static_cast<T> (-2.0))*var_b;
//    assert(graph::add_cast(negate).get() && "Expected add node.");

//  v1 + -1*v2 -> v1 - v2
    auto add_neg = var_a + graph::none<T> ()*var_b;
    assert(graph::subtract_cast(add_neg).get() && "Expected subtract node.");

//  -c1*v1 + v2 -> v2 - c*v1
//    auto negate2 = graph::constant(static_cast<T> (-2.0))*var_a + var_b;
//    auto negate2_cast = graph::subtract_cast(negate2);
//    assert(negate2_cast.get() && "Expected subtract node.");
//    assert(negate2_cast->get_left()->is_match(var_b) && "Expected var_b.");

//  (c1*v1 + c2) + (c3*v1 + c4) -> c5*v1 + c6
    auto var_e = graph::variable<T> (1, "");
    auto addfma1 = graph::fma(var_b, var_a, var_d)
                + graph::fma(var_c, var_a, var_e);
    assert(graph::fma_cast(addfma1).get() &&
           "Expected fused multiply add node.");
//  (v1*c1 + c2) + (v1*c3 + c4) -> c5*v1 + c6
    auto addfma2 = graph::fma(var_a, var_b, var_d)
                 + graph::fma(var_a, var_c, var_e);
    assert(graph::fma_cast(addfma2).get() &&
           "Expected fused multiply add node.");
//  (c1*v1 + c2) + (v1*c3 + c4) -> c5*v1 + c6
    auto addfma3 = graph::fma(var_b, var_a, var_d)
                 + graph::fma(var_a, var_c, var_e);
    assert(graph::fma_cast(addfma3).get() &&
           "Expected fused multiply add node.");
//  (v1*c1 + c2) + (c3*v1 + c4) -> c5*v1 + c6
    auto addfma4 = graph::fma(var_a, var_b, var_d)
                 + graph::fma(var_c, var_a, var_e);
    assert(graph::fma_cast(addfma4).get() &&
           "Expected fused multiply add node.");

//  Test cases like
//  (c1 + c2/x) + c3/x -> c1 + c4/x
//  (c1 - c2/x) + c3/x -> c1 + c4/x
    common_d = (one + three/var_a) + (one/var_a);
    auto common_d_acast = graph::add_cast(common_d);
    assert(common_d_acast.get() && "Expected add node.");
    assert(graph::constant_cast(common_d_acast->get_left()).get() &&
           "Expected constant on the left.");

    common_d = (one - three/var_a) + (one/var_a);
    common_d_acast = graph::add_cast(common_d);
    assert(common_d_acast.get() && "Expected add node.");
    assert(graph::constant_cast(common_d_acast->get_left()).get() &&
           "Expected constant on the left.");

//  c1*a + c2*b -> c1*(a + c3*b)
    auto constant_factor = three*variable + (one + one)*var_b;
    assert(graph::multiply_cast(constant_factor).get() &&
           "Expected multilpy node.");
    
//  Test is_match
    auto match1 = graph::one<T> () + variable;
    auto match2 = graph::one<T> () + variable;
    assert(match1->is_match(match2) && "Expected match");

//  Chained addition reductions.
//  a + (a + b) = fma(2,a,b)
//  a + (b + a) = fma(2,a,b)
//  (a + b) + a = fma(2,a,b)
//  (b + a) + a = fma(2,a,b)
    assert(fma_cast(var_a + (var_a + var_b)).get() && "Expected fma node.");
    assert(fma_cast(var_a + (var_b + var_a)).get() && "Expected fma node.");
    assert(fma_cast((var_a + var_b) + var_a).get() && "Expected fma node.");
    assert(fma_cast((var_b + var_a) + var_a).get() && "Expected fma node.");

//  fma(a,b,c) + d -> fma(a,b,c + d)
    assert(fma_cast(fma(var_a, var_b, three) + one).get() &&
           "Expected fma node.");
//  a + fma(b,c + d) -> fma(b,c,a + d)
    assert(fma_cast(one + fma(var_a, var_b, three)).get() &&
           "Expected fma node.");

//  (a/c)^d + (b/c)^d -> (a^d + b^d)/c^d
    auto common_pow_denom = graph::pow(var_a/var_d, var_b)
                          + graph::pow(var_c/var_d, var_b);
    auto common_pow_denom_cast = graph::divide_cast(common_pow_denom);
    assert(common_pow_denom_cast.get() && "Expected a divide node.");
    assert(graph::add_cast(common_pow_denom_cast->get_left()).get() &&
           "Expected an add node");
    assert(graph::pow_cast(common_pow_denom_cast->get_right()).get() &&
           "Expected a power node");

//  a + fma(b,c,d) -> fma(b,c,a + d)
    auto add_fma = var_a + graph::fma(var_b, var_c, var_d);
    auto add_fma_cast = graph::fma_cast(add_fma);
    assert(add_fma_cast.get() && "Expected fused multiply add node.");
    assert(add_fma_cast->get_left()->is_match(var_b) &&
           "Expected var_b in the first slot.");
    assert(add_fma_cast->get_middle()->is_match(var_c) &&
           "Expected var_c in the second slot.");
    assert(graph::add_cast(add_fma_cast->get_right()) &&
           "Expected add_node in the third slot.");
}

//------------------------------------------------------------------------------
///  @brief Tests for subtract nodes.
//------------------------------------------------------------------------------
template<typename T> void test_subtract() {
//  Three constant nodes should reduce to a single constant node with added
//  operands.
    auto one = graph::one<T> ();
    auto zero = one - one;
    auto zero_cast = graph::constant_cast(zero);
    assert(zero_cast.get() && "Expected a constant type.");
    assert(zero_cast->is(0) && "Expected a value of zero.");
    assert(zero->evaluate()[0] == static_cast<T> (0) &&
           "Expected a value of zero.");

    auto neg_one = one - one - one;
    auto neg_one_cast = graph::constant_cast(neg_one);
    assert(neg_one_cast.get() && "Expected a constant type.");
    assert(neg_one_cast->is(-1) && "Expected a value of -1.");
    assert(neg_one->evaluate()[0] == static_cast<T> (-1.0) &&
           "Expected a value of -1.");

//  A right side zero node should reduce to left side.
    auto one_minus_zero = one - zero;
    assert(one_minus_zero.get() == one.get() &&
           "Expected to retrive the left side.");
    assert(one_minus_zero->evaluate()[0] == static_cast<T> (1.0) &&
           "Expected a value of 1.");

//  A left side zero node should reduce to a negative right side.
    auto zero_minus_one = zero - one;
    auto zero_minus_one_cast = graph::constant_cast(zero_minus_one);
    assert(zero_minus_one_cast.get() && "Expected a constant type.");
    assert(zero_minus_one_cast->is(-1) && "Expected -1 for result");
    assert(zero_minus_one->evaluate()[0] == static_cast<T> (-1.0) &&
           "Expected a value of -1.");

//  Test variable quanities.
//  Any right side zero nodes should reduce to the other operand.
    auto variable = graph::variable<T> (1, "");
    auto var_minus_zero = variable - zero;
    assert(var_minus_zero.get() == variable.get() &&
           "Expected to retrive the left side.");

//  Any right side zero should reduce to a the a multiply node.
    auto zero_minus_var = zero - variable;
    assert(graph::multiply_cast(zero_minus_var).get() &&
           "Expected multiply node.");
    variable->set(static_cast<T> (3.0));
    const backend::buffer<T> zero_minus_var_result = zero_minus_var->evaluate();
    assert(zero_minus_var_result.size() == 1 && "Expected single value.");
    assert(zero_minus_var_result.at(0) == static_cast<T> (-3.0) &&
           "Expected 0 - 3 for result.");

//  Variable minus a variable should return an minus node.
    auto variable_b = graph::variable<T> (1, "");
    auto var_minus_var = variable - variable_b;
    assert(graph::subtract_cast(var_minus_var).get() &&
           "Expected a subtraction node.");
    variable_b->set(static_cast<T> (10.0));
    const backend::buffer<T> var_minus_var_result = var_minus_var->evaluate();
    assert(var_minus_var_result.size() == 1 && "Expected single value.");
    assert(var_minus_var_result.at(0) == static_cast<T> (-7) &&
           "Expected 3 - 10 for result");

//  Test variable vectors.
    auto varvec_a = graph::variable<T> (std::vector<T> ({10.0, 20.0}), "");
    auto varvec_b = graph::variable<T> (std::vector<T> ({-3.0, 5.0}), "");
    auto varvec_minus_varvec = varvec_a - varvec_b;
    const backend::buffer<T> varvec_minus_varvec_result =
        varvec_minus_varvec->evaluate();
    assert(varvec_minus_varvec_result.size() == 2 &&
           "Size mismatch in result.");
    assert(varvec_minus_varvec_result.at(0) == static_cast<T> (13.0) &&
           "Expected 10 - -3.");
    assert(varvec_minus_varvec_result.at(1) == static_cast<T> (15.0) &&
           "Expected 20 - 5.");

//  Test derivatives.
//  d (1 - x) / dx = d1/dx - dx/dx = 0 - 1 = -1
    auto one_minus_var = one - variable;
    auto done_minus_var = one_minus_var->df(variable);
    auto done_minus_var_cast = graph::constant_cast(done_minus_var);
    assert(done_minus_var_cast.get() && "Expected a constant type.");
    assert(done_minus_var_cast->is(-1) &&
           "Expected to reduce to a constant minus one.");

//  Test common factors.
    auto var_a = graph::variable<T> (1, "");
    auto var_b = graph::variable<T> (1, "");
    auto var_c = graph::variable<T> (1, "");
    auto common_a = var_a*var_b - var_a*var_c;
    assert(graph::add_cast(common_a).get() == nullptr &&
           "Did not expect add node.");
    assert(graph::multiply_cast(common_a).get() && "Expected multiply node.");

    auto common_b = var_a*var_b - var_b*var_c;
    assert(graph::add_cast(common_b).get() == nullptr &&
           "Did not expect add node.");
    assert(graph::multiply_cast(common_b).get() && "Expected multiply node.");

    auto common_c = var_a*var_c - var_b*var_c;
    assert(graph::add_cast(common_c).get() == nullptr &&
           "Did not expect add node.");
    assert(graph::multiply_cast(common_c).get() && "Expected multiply node.");

//  Test common denominator.
    auto common_d = var_a/var_b - var_c/var_b;
    assert(graph::subtract_cast(common_d).get() == nullptr &&
           "Did not expect subtract node.");
    assert(graph::divide_cast(common_d).get() && "Expected divide node.");

//  Test is_match
    auto match = graph::one<T> ()*var_a
               - graph::one<T> ()*var_a;
    auto match_cast = graph::constant_cast(match);
    assert(match_cast.get() && "Expected a constant type.");
    assert(match_cast->is(0) && "Expected zero node.");

//  Reduce (a/y)^e - (b/y)^e -> (a^2 - b^2)/(y^e).
    auto var_d = graph::variable<T> (1, "");
    auto common_power1 = graph::pow(var_a/var_b,var_c) -
                         graph::pow(var_d/var_b,var_c);
    assert(graph::divide_cast(common_power1).get() && "Expected Divide node.");
//  Reduce a/y^e - (b/y)^e -> (a - b^2)/(y^e).
    auto common_power2 = graph::pow(var_a/var_b,var_c) -
                         var_d/graph::pow(var_b,var_c);
    assert(graph::divide_cast(common_power2) && "Expected Divide node.");
    auto common_power3 = var_d/graph::pow(var_b,var_c) -
                         graph::pow(var_a/var_b,var_c);
    assert(graph::divide_cast(common_power3) && "Expected Divide node.");

//  v1 - -c*v2 -> v1 + c*v2
    auto negate = var_a - graph::constant(static_cast<T> (-2.0))*var_b;
    assert(graph::subtract_cast(negate).get() && "Expected subtraction node.");

//  v1 - -1*v2 -> v1 + v2
    neg_one = var_a - graph::none<T> ()*var_b;
    assert(graph::add_cast(neg_one).get() && "Expected addition node.");

//  (c1*v1 + c2) - (c3*v1 + c4) -> c5*(v1 - c6)
    auto two = graph::constant(static_cast<T> (2.0));
    auto three = graph::constant(static_cast<T> (3.0));
    auto subfma = graph::fma(three, var_a, two)
                - graph::fma(two, var_a, three);
    assert(graph::multiply_cast(subfma).get() && "Expected a multiply node.");

//  Test cases like
//  (c1 + c2/x) - c3/x -> c1 + c4/x
//  (c1 - c2/x) - c3/x -> c1 - c4/x
    common_d = (one + three/var_a) - (one/var_a);
    auto common_d_acast = graph::add_cast(common_d);
    assert(common_d_acast.get() && "Expected add node.");
    assert(graph::constant_cast(common_d_acast->get_left()).get() &&
           "Expected constant on the left.");
    common_d = (one - three/var_a) - (one/var_a);
    auto common_d_scast = graph::subtract_cast(common_d);
    assert(common_d_scast.get() && "Expected subtract node.");
    assert(graph::constant_cast(common_d_scast->get_left()).get() &&
           "Expected constant on the left.");

//  c1*a - c2*b -> c1*(a - c2*b)
    auto common_factor = three*var_a - (one + one)*var_b;
    assert(graph::multiply_cast(common_factor).get() &&
           "Expected multilpy node.");

//  (c1 - c2*v) - c3*v -> c1 - c4*v (1 - 3v) - 2v = 1 - 5*v
    auto chained_subtract = (one - three*var_a) - two*var_a;
    auto chained_subtract_cast = graph::subtract_cast(chained_subtract);
    assert(chained_subtract_cast.get() &&
           "Expected subtract node.");
    assert(graph::constant_cast(chained_subtract_cast->get_left()).get() &&
           "Expected a constant node on the left.");
    assert(graph::multiply_cast(chained_subtract_cast->get_right()).get() &&
           "Expected a multiply node on the right.");

//  (a - b*c) - d*c -> a - (b + d)*c
    auto chained_subtract2 = (var_b - two*var_a) - var_c*var_a;
    auto chained_subtract2_cast = graph::subtract_cast(chained_subtract2);
    assert(chained_subtract2_cast.get() &&
           "Expected subtract node.");
    assert(graph::variable_cast(chained_subtract2_cast->get_left()).get() &&
           "Expected a constant node on the left.");
    auto multiply_term = graph::multiply_cast(chained_subtract2_cast->get_right());
    assert(multiply_term.get() &&
           "Expected a multiply node on the right.");
    assert(add_cast(multiply_term->get_left()).get() &&
           "Expected an add node on the right.");
//  (a - b*c) - c*d -> a - (b + d)*c
    auto chained_subtract3 = (var_b - two*var_a) - var_a*var_c;
    auto chained_subtract3_cast = graph::subtract_cast(chained_subtract3);
    assert(chained_subtract3_cast.get() &&
           "Expected subtract node.");
    assert(graph::variable_cast(chained_subtract3_cast->get_left()).get() &&
           "Expected a constant node on the left.");
    auto multiply_term2 = graph::multiply_cast(chained_subtract3_cast->get_right());
    assert(multiply_term2.get() &&
           "Expected a multiply node on the right.");
    assert(add_cast(multiply_term2->get_left()).get() &&
           "Expected an add node on the right.");
//  (a - c*b) - d*c -> a - (b + d)*c
        auto chained_subtract4 = (var_b - var_a*two) - var_c*var_a;
        auto chained_subtract4_cast = graph::subtract_cast(chained_subtract3);
        assert(chained_subtract4_cast.get() &&
               "Expected subtract node.");
        assert(graph::variable_cast(chained_subtract4_cast->get_left()).get() &&
               "Expected a constant node on the left.");
        auto multiply_term3 = graph::multiply_cast(chained_subtract4_cast->get_right());
        assert(multiply_term.get() &&
               "Expected a multiply node on the right.");
        assert(add_cast(multiply_term3->get_left()).get() &&
               "Expected an add node on the right.");
//  (a - b*c) - c*d -> a - (b + d)*c
        auto chained_subtract5 = (var_b - var_a*two) - var_a*var_c;
        auto chained_subtract5_cast = graph::subtract_cast(chained_subtract3);
        assert(chained_subtract5_cast.get() &&
               "Expected subtract node.");
        assert(graph::variable_cast(chained_subtract5_cast->get_left()).get() &&
               "Expected a constant node on the left.");
        auto multiply_term4 = graph::multiply_cast(chained_subtract5_cast->get_right());
        assert(multiply_term4.get() &&
               "Expected a multiply node on the right.");
        assert(add_cast(multiply_term4->get_left()).get() &&
               "Expected an add node on the right.");
    
//  a*b - c*(d*b) -> (a - c*d)*b
//  a*b - c*(b*d) -> (a - c*d)*b
//  b*a - c*(d*b) -> (a - c*d)*b
//  b*a - c*(b*d) -> (a - c*d)*b
    auto common_factor2 = var_a*var_b - two*(var_c*var_b);
    assert(graph::multiply_cast(common_factor2).get() &&
           "Expected multiply node.");
    auto common_factor3 = var_a*var_b - two*(var_b*var_c);
    assert(graph::multiply_cast(common_factor3).get() &&
           "Expected multiply node.");
    auto common_factor4 = var_b*var_a - two*(var_c*var_b);
    assert(graph::multiply_cast(common_factor4).get() &&
           "Expected multiply node.");
    auto common_factor5 = var_b*var_a - two*(var_b*var_c);
    assert(graph::multiply_cast(common_factor5).get() &&
           "Expected multiply node.");
//  c*(d*b) - a*b -> (c*d - a)*b
//  c*(b*d) - a*b -> (c*d - a)*b
//  c*(d*b) - b*a -> (c*d - a)*b
//  c*(b*d) - b*a -> (c*d - a)*b
    auto common_factor6 = two*(var_c*var_b) - var_a*var_b;
    assert(graph::multiply_cast(common_factor6).get() &&
           "Expected multiply node.");
    auto common_factor7 = two*(var_b*var_c) - var_a*var_b;
    assert(graph::multiply_cast(common_factor7).get() &&
           "Expected multiply node.");
    auto common_factor8 = two*(var_c*var_b) - var_b*var_a;
    assert(graph::multiply_cast(common_factor8).get() &&
           "Expected multiply node.");
    auto common_factor9 = two*(var_b*var_c) - var_b*var_a;
    assert(graph::multiply_cast(common_factor9).get() &&
           "Expected multiply node.");

//  (a - b*c) - d*c -> a - (b + d)*c
    auto chained_subtract_multiply = (var_a - var_b*var_c) - var_d*var_c;
    auto chained_subtract_multiply_cast = graph::subtract_cast(chained_subtract_multiply);
    assert(chained_subtract_multiply_cast.get() && "Expected subtract node.");
    assert(graph::multiply_cast(chained_subtract_multiply_cast->get_right()).get() &&
           "Expected a multiply node on the left.");
//  (a - b*c) - c/d -> a - (b*c + c/d)
    auto chained_subtract_multiply2 = (var_a - var_b*var_c) - var_c/var_d;
    auto chained_subtract_multiply_cast2 = graph::subtract_cast(chained_subtract_multiply2);
    assert(chained_subtract_multiply_cast2.get() && "Expected subtract node.");
    assert(graph::fma_cast(chained_subtract_multiply_cast2->get_right()).get() &&
           "Expected a fused multiply add node on the left.");
//  (a - b/c) - d/c -> a - (b + d)*c
    auto chained_subtract_divide = (var_a - var_b/var_c) - var_d/var_c;
    auto chained_subtract_divide_cast = graph::subtract_cast(chained_subtract_divide);
    assert(chained_subtract_divide_cast.get() && "Expected subtract node.");
    assert(graph::divide_cast(chained_subtract_divide_cast->get_right()).get() &&
           "Expected a divide node on the left.");
//  (a - b/c) - d*c -> a - (d*c + b/c)
    auto chained_subtract_divide2 = (var_a - var_b/var_c) - var_d*var_c;
    auto chained_subtract_divide_cast2 = graph::subtract_cast(chained_subtract_divide2);
    assert(chained_subtract_divide_cast2.get() && "Expected subtract node.");
    assert(graph::fma_cast(chained_subtract_divide_cast2->get_right()).get() &&
           "Expected a fused multiply add node on the left.");
}

//------------------------------------------------------------------------------
///  @brief Tests for multiply nodes.
//------------------------------------------------------------------------------
template<typename T> void test_multiply() {
//  Three constant nodes should reduce to a single constant node with multiplied
//  operands.
    auto one = graph::one<T> ();
    auto one_cubed = one*one*one;
    assert(one_cubed.get() == one.get() && "Expected to reduce back to one");
    assert(one_cubed->evaluate()[0] == static_cast<T> (1) &&
           "Expected one.");

//  Any zero nodes should reduce zero.
    auto zero = graph::zero<T> ();
    assert((zero*one).get() == zero.get() && "Expected to reduce back to zero");
    assert((one*zero).get() == zero.get() && "Expected to reduce back to zero");
    assert((zero*one)->evaluate()[0] == static_cast<T> (0) &&
           "Expected zero.");
    assert((one*zero)->evaluate()[0] == static_cast<T> (0) &&
           "Expected zero.");
    
//  Test constant times constant.
    auto two = graph::constant(static_cast<T> (2));
    auto three = graph::constant(static_cast<T> (3));
    auto two_times_three = two*three;
    assert(graph::constant_cast(two_times_three).get() &&
           "Expected a constant type.");
    auto three_times_two = three*two;
    assert(graph::constant_cast(three_times_two).get() &&
           "Expected a constant type.");
    const backend::buffer<T> two_times_three_result =
        two_times_three->evaluate();
    const backend::buffer<T> three_times_two_result =
        three_times_two->evaluate();
    assert(two_times_three_result.size() == 1 && "Expected single value.");
    assert(three_times_two_result.size() == 1 && "Expected single value.");
    assert(three_times_two_result.at(0) == static_cast<T> (6.0) &&
           "Expected 3*2 for result.");
    assert(three_times_two_result.at(0) == two_times_three_result.at(0) &&
           "Expected 3*2 == 2*3.");

//  Test variable quanities.
//  Any zero should reduce back to zero.
    auto variable = graph::variable<T> (1, "");
    assert((variable*zero).get() == zero.get() &&
           "Expected to retrive the right side.");
    assert((zero*variable).get() == zero.get() &&
           "Expected to retrive the left side.");
//  Any one should reduce to the opposite side.
    assert((variable*one).get() == variable.get() &&
           "Expected to retrive the left side.");
    assert((one*variable).get() == variable.get() &&
           "Expected to retrive the right side.");
    assert((variable*zero)->evaluate()[0] == static_cast<T> (0.0) &&
           "Expected value of zero.");
    assert((variable*zero)->evaluate()[0] == static_cast<T> (0.0) &&
           "Expected value of zero.");

//  Varibale times a non 0 or 1 constant should reduce to a multiply node.
    auto two_times_var = two*variable;
    assert(graph::multiply_cast(two_times_var).get() &&
           "Expected multiply node.");
    variable->set(static_cast<T> (6.0));
    const backend::buffer<T> two_times_var_result = two_times_var->evaluate();
    assert(two_times_var_result.size() == 1 && "Expected single value.");
    assert(two_times_var_result.at(0) == static_cast<T> (12.0) &&
           "Expected 2*6 for result.");

//  Test variable vectors.
    auto varvec_a = graph::variable<T> (std::vector<T> ({4.0, -2.0}), "a");
    auto varvec_b = graph::variable<T> (std::vector<T> ({-4.0, -2.0}), "b");
    auto varvec_times_varvec = varvec_a*varvec_b;
    const backend::buffer<T> varvec_times_varvec_result =
        varvec_times_varvec->evaluate();
    assert(varvec_times_varvec_result.size() == 2 &&
           "Size mismatch in result.");
    assert(varvec_times_varvec_result.at(0) == static_cast<T> (-16.0) &&
           "Expected 4*-4.");
    assert(varvec_times_varvec_result.at(1) == static_cast<T> (4.0) &&
           "Expected -2*-2.");

    assert((varvec_a*varvec_b)->is_match(varvec_b*varvec_a) &&
           "Expected these to match.");

//  Test reduction short cut. If all the elements in the numerator are zero, an
//  denominator does not need to be evaluated. This test makes sure that a sum
//  or product is not used to avoid cases like {-1, 0, 1} which sum and product
//  are zero.
    auto var_sum_prod = graph::variable<T> (std::vector<T> ({-2.0, 2.0, 0.0}), "");
    auto var_sum_prod_multiply_two = var_sum_prod*two;
    const backend::buffer<T> var_sum_prod_multiply_two_result =
        var_sum_prod_multiply_two->evaluate();
    assert(var_sum_prod_multiply_two_result.at(0) == static_cast<T> (-2.0) *
                                                     static_cast<T> (2.0) &&
           "Expected -2/2 for result.");
    assert(var_sum_prod_multiply_two_result.at(1) == static_cast<T> (2.0) *
                                                     static_cast<T> (2.0) &&
           "Expected 2/2 for result.");
    assert(var_sum_prod_multiply_two_result.at(2) == static_cast<T> (0.0) &&
           "Expected 0/2 for result.");

//  Test derivatives.
//  d (c*x) / dx = dc/dx*x + c*dx/dx = c*1 = c;
    assert(two_times_var->df(variable).get() == two.get() &&
           "Expect to reduce back to the constant.");
//  d (x*x) / dx = dx/dx*x + x*dx/dx = x + x = 2*x;
    auto varvec_sqrd = varvec_a*varvec_a;
    auto dvarvec_sqrd = varvec_sqrd->df(varvec_a);
    assert(graph::multiply_cast(dvarvec_sqrd).get() &&
           "Expected multiply node.");
    const backend::buffer<T> dvarvec_sqrd_result = dvarvec_sqrd->evaluate();
    assert(dvarvec_sqrd_result.size() == 2 && "Size mismatch in result.");
    assert(dvarvec_sqrd_result.at(0) == static_cast<T> (8.0) &&
           "Expected 2*4 for result.");
    assert(dvarvec_sqrd_result.at(1) == static_cast<T> (-4.0) &&
           "Expected 2*-2 for result.");

//  Variables should always go to the right and constant to he left.
    auto swap = multiply_cast(variable*two);
    assert(graph::constant_cast(swap->get_left()).get() &&
           "Expected a constant on he left");
    assert(graph::variable_cast(swap->get_right()).get() &&
           "Expected a variable on he right");
    
//  Test reduction of common constants c1*x*c2*y = c3*x*y.
    auto x1 = graph::constant(static_cast<T> (2.0))*graph::variable<T> (1, "");
    auto x2 = graph::constant(static_cast<T> (5.0))*graph::variable<T> (1, "");
    auto x3 = x1*x2;
    auto x3_cast = graph::multiply_cast(x3);
    assert(x3_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(x3_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::multiply_cast(x3_cast->get_right()).get() &&
           "Expected a multipy node.");

//  Test reduction of common constants x*c1*c2*y = c3*x*y.
    auto x4 = graph::variable<T> (1, "")*graph::constant(static_cast<T> (2.0));
    auto x5 = graph::constant(static_cast<T> (5.0))*graph::variable<T> (1, "");
    auto x6 = x4*x5;
    auto x6_cast = graph::multiply_cast(x6);
    assert(x6_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(x6_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::multiply_cast(x6_cast->get_right()).get() &&
           "Expected multipy node.");

//  Test reduction of common constants c1*x*y*c2 = c3*x*y.
    auto x7 = graph::constant(static_cast<T> (2.0))*graph::variable<T> (1, "");
    auto x8 = graph::variable<T> (1, "")*graph::constant(static_cast<T> (5.0));
    auto x9 = x7*x8;
    auto x9_cast = graph::multiply_cast(x9);
    assert(x9_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(x9_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::multiply_cast(x9_cast->get_right()).get() &&
           "Expected multipy node.");

//  Test reduction of common constants x*c1*y*c2 = c3*x*y.
    auto x10 = graph::variable<T> (1, "")*graph::constant(static_cast<T> (2.0));
    auto x11 = graph::constant(static_cast<T> (5.0))*graph::variable<T> (1, "");
    auto x12 = x10*x11;
    auto x12_cast = graph::multiply_cast(x12);
    assert(x12_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(x12_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::multiply_cast(x12_cast->get_right()).get() &&
           "Expected a multipy node.");

//  Test gathering of terms.
    auto v1 = graph::variable<T> (1, "v1");
    auto v2 = graph::variable<T> (1, "v2");
    auto gather_v1 = (v1*v2)*v1;
    assert(pow_cast(multiply_cast(gather_v1)->get_left()).get() &&
           "Expected power node.");
    auto gather_v2 = (v2*v1)*v1;
    assert(pow_cast(multiply_cast(gather_v2)->get_left()).get() &&
           "Expected power node.");
    auto gather_v3 = v1*(v1*v2);
    assert(pow_cast(multiply_cast(gather_v3)->get_left()).get() &&
           "Expected power node.");
    auto gather_v4 = v1*(v2*v1);
    assert(pow_cast(multiply_cast(gather_v4)->get_left()).get() &&
           "Expected power node.");

//  Test double multiply cases.
    auto gather_v5 = (v1*v2)*(v1*v2);
    auto gather_v5_cast = graph::pow_cast(gather_v5);
    assert(gather_v5_cast.get() && "Expected power node.");
    assert(graph::multiply_cast(gather_v5_cast->get_left()).get() &&
           "Expected multiply inside power.");
    assert(graph::constant_cast(gather_v5_cast->get_right())->is(2) &&
           "Expected power of 2.");

//  Test gather of terms. This test is setup to trigger an infinite recursive
//  loop if a critical check is not in place no need to check the values.
    auto a = graph::variable<T> (1, "");
    auto aaa = (a*sqrt(a))*(a*sqrt(a));

//  Test power reduction.
    auto var_times_var = variable*variable;
    assert(graph::pow_cast(var_times_var).get() &&
           "Expected a power node.");
    const backend::buffer<T> var_times_var_result = var_times_var->evaluate();
    assert(var_times_var_result.size() == 1 && "Expected single value.");
    assert(var_times_var_result.at(0) == static_cast<T> (36) &&
           "Expected 6*6 for result.");
    
//  Test c1*(c2*v) -> c3*v
    auto c3 = two*(three*a);
    auto c3_cast = graph::multiply_cast(c3);
    assert(c3_cast.get() && "Expected multiply node.");
    assert(graph::constant_cast(c3_cast->get_left()) &&
           "Expected constant on the left.");
    assert(graph::variable_cast(c3_cast->get_right()) &&
           "Expected variable on the right.");

//  Test (c1*v)*c2 -> c4*v
    auto c4 = (three*a)*two;
    auto c4_cast = graph::multiply_cast(c4);
    assert(c4_cast.get() && "Expected multiply node.");
    assert(graph::constant_cast(c4_cast->get_left()) &&
           "Expected constant on the left.");
    assert(graph::variable_cast(c4_cast->get_right()) &&
           "Expected variable on the right.");

//  Test c1*(c2/v) -> c5/v
    auto c5 = two*(three/a);
    auto c5_cast = graph::divide_cast(c5);
    assert(c5_cast.get() && "Expected a divide node.");
    assert(graph::constant_cast(c5_cast->get_left()).get() &&
           "Expected constant in the numerator.");
    assert(graph::variable_cast(c5_cast->get_right()).get() &&
           "Expected variable in the denominator.");

//  Test c1*(v/c2) -> c6*v
    auto c6 = two*(a/three);
    auto c6_cast = graph::multiply_cast(c6);
    assert(c6_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(c6_cast->get_left()).get() &&
           "Expected constant for the left.");
    assert(graph::variable_cast(c6_cast->get_right()).get() &&
           "Expected variable for the right.");

//  Test (c2/v)*c1 -> c7/v
    auto c7 = (three/a)*two;
    auto c7_cast = graph::divide_cast(c7);
    assert(c7_cast.get() && "Expected a divide node.");
    assert(graph::constant_cast(c7_cast->get_left()).get() &&
           "Expected constant for the numerator.");
    assert(graph::variable_cast(c7_cast->get_right()).get() &&
           "Expected variable for the denominator.");

//  Test c1*(v/c2) -> c8*v
    auto c8 = two*(a/three);
    auto c8_cast = graph::multiply_cast(c8);
    assert(c8_cast.get() && "Expected divide node.");
    assert(graph::constant_cast(c8_cast->get_left()).get() &&
           "Expected constant for the left.");
    assert(graph::variable_cast(c8_cast->get_right()).get() &&
           "Expected variable for the right.");

//  Test v1*(c*v2) -> c*(v1*v2)
    auto c9 = a*(three*variable);
    auto c9_cast = graph::multiply_cast(c9);
    assert(c9_cast.get() && "Expected multiply node.");
    assert(graph::constant_cast(c9_cast->get_left()).get() &&
           "Expected a constant node first.");

//  Test v1*(v2*c) -> c*(v1*v2)
    auto c10 = a*(variable*three);
    auto c10_cast = graph::multiply_cast(c10);
    assert(c10_cast.get() && "Expected multiply node.");
    assert(graph::constant_cast(c10_cast->get_left()).get() &&
           "Expected a constant node first.");

//  Test (c*v1)*v2) -> c*(v1*v2)
    auto c11 = (three*variable)*a;
    auto c11_cast = graph::multiply_cast(c11);
    assert(c11_cast.get() && "Expected multiply node.");
    assert(graph::constant_cast(c11_cast->get_left()).get() &&
           "Expected a constant node first.");

//  Test (v1*c)*v2 -> c*(v1*v2)
    auto c12 = (variable*three)*a;
    auto c12_cast = graph::multiply_cast(c12);
    assert(c12_cast.get() && "Expected multiply node.");
    assert(graph::constant_cast(c12_cast->get_left()).get() &&
           "Expected constant node first.");

//  Test (c/v1)*v2 -> c*(v2/v1)
    auto c13 = (three/variable)*a;
    auto c13_cast = graph::multiply_cast(c13);
    assert(c13_cast.get() && "Expected multiply node.");
    assert(graph::constant_cast(c13_cast->get_left()).get() &&
           "Expected constant node first.");
    assert(graph::divide_cast(c13_cast->get_right()).get() &&
           "Expected divide node second.");

//  Test a^b*a^c -> a^(b + c) -> a^d
    auto pow_bc = graph::pow(a, two)*graph::pow(a, three);
    auto pow_bc_cast = graph::pow_cast(pow_bc);
    assert(pow_bc_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_bc_cast->get_right()).get() &&
           "Expected constant exponent.");

//  Test a*a^c -> a^(1 + c) -> a^c2
    auto pow_c = a*graph::pow(a, three);
    auto pow_c_cast = graph::pow_cast(pow_c);
    assert(pow_c_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_c_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_c_cast->get_right())->is(4) &&
           "Expected constant exponent equal to 4.");

//  Test a^b*a -> a^(b + 1) -> a^b2
    auto pow_b = graph::pow(a, two)*a;
    auto pow_b_cast = graph::pow_cast(pow_b);
    assert(pow_b_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_b_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_b_cast->get_right())->is(3) &&
           "Expected constant exponent equal to 3.");

//  Test a^b*sqrt(a) -> a^(b + 0.5) -> a^b2
    auto pow_sqb = graph::pow(a, two)*graph::sqrt(a);
    auto pow_sqb_cast = graph::pow_cast(pow_sqb);
    assert(pow_sqb_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_sqb_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_sqb_cast->get_right())->is(2.5) &&
           "Expected constant exponent equal to 2.5.");

//  Test sqrt(a)*a^c -> a^(0.5 + c) -> a^c2
    auto pow_sqc = graph::sqrt(a)*graph::pow(a, three);
    auto pow_sqc_cast = graph::pow_cast(pow_sqc);
    assert(pow_sqc_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_sqc_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_sqc_cast->get_right())->is(3.5) &&
           "Expected constant exponent equal to 3.5.");

//  Test a*sqrt(a) -> a^(1.5)
    auto pow_asqa = a*graph::sqrt(a);
    auto pow_asqa_cast = graph::pow_cast(pow_asqa);
    assert(pow_asqa_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_asqa_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_asqa_cast->get_right())->is(1.5) &&
           "Expected constant exponent equal to 1.5.");

//  Test sqrt(a)*a -> a^(1.5)
    auto pow_sqaa = graph::sqrt(a)*a;
    assert(pow_sqaa->is_match(pow_asqa) && "Expected to match.");

//  Test a^b*a^c -> a^(b + c)
    auto pow_mul = graph::pow(v1, v2)*graph::pow(v1, variable);
    auto pow_mul_cast = graph::pow_cast(pow_mul);
    assert(pow_mul_cast.get() && "Expected power node.");
    assert(add_cast(pow_mul_cast->get_right()) &&
           "Expected add node in expoent.");

//  (c*v)*v -> c*v^2
    auto test_var_move = [two](graph::shared_leaf<T> x) {
        auto var_move = (two*x)*x;
        auto var_move_cast = graph::multiply_cast(var_move);
        assert(var_move_cast.get() && "Expected multiply.");
        assert(!var_move_cast->get_left()->is_all_variables() &&
               "Expected Non variable like in the left side.");
        assert(var_move_cast->get_right()->is_all_variables() &&
               "Expected variable like in the right side.");
    };

    test_var_move(a);
    test_var_move(pow_sqaa);
    test_var_move(graph::sqrt(a));

//  ((c + d)*v^a)*v^b -> (c + d)*v^(a + b)
    auto common_base = ((two + varvec_a)*graph::pow(variable, three))*graph::pow(variable, two);
    auto common_base_cast = graph::multiply_cast(common_base);
    assert(common_base_cast.get() && "Expected multiply node.");
    assert(graph::add_cast(common_base_cast->get_left()).get() &&
           "Expected add cast on the left.");
    assert(graph::pow_cast(common_base_cast->get_right()).get() &&
           "Expected power cast on the right.");
//  (v^a*(c + d))*v^b -> (c + d)*v^(a + b)
    auto common_base2 = (graph::pow(variable, three)*(two + varvec_a))*graph::pow(variable, two);
    auto common_base_cast2 = graph::multiply_cast(common_base2);
    assert(common_base_cast2.get() && "Expected multiply node.");
    assert(graph::add_cast(common_base_cast2->get_left()).get() &&
           "Expected add cast on the left.");
    assert(graph::pow_cast(common_base_cast2->get_right()).get() &&
           "Expected power cast on the right.");
//  v^b*((c + d)*v^a) -> (c + d)*v^(a + b)
    auto common_base3 = graph::pow(variable, two)*((two + varvec_a)*graph::pow(variable, three));
    auto common_base_cast3 = graph::multiply_cast(common_base3);
    assert(common_base_cast3.get() && "Expected multiply node.");
    assert(graph::add_cast(common_base_cast3->get_left()).get() &&
           "Expected add cast on the left.");
    assert(graph::pow_cast(common_base_cast3->get_right()).get() &&
           "Expected power cast on the right.");
//  v^b*(v^a*(c + d)) -> (c + d)*v^(a + b)
    auto common_base4 = graph::pow(variable, two)*(graph::pow(variable, three)*(two + varvec_a));
    auto common_base_cast4 = graph::multiply_cast(common_base4);
    assert(common_base_cast4.get() && "Expected multiply node.");
    assert(graph::add_cast(common_base_cast4->get_left()).get() &&
           "Expected add cast on the left.");
    assert(graph::pow_cast(common_base_cast4->get_right()).get() &&
           "Expected power cast on the right.");
}

//------------------------------------------------------------------------------
///  @brief Tests for divide nodes.
//------------------------------------------------------------------------------
template<typename T> void test_divide() {
// Check for potential divide by zero.
    auto zero = graph::zero<T> ();
    assert((zero/zero).get() == zero.get() && "Expected to recover zero.");
    assert((zero/zero)->evaluate()[0] == static_cast<T> (0.0) &&
           "Expected to recover zero.");

// A zero in the numerator should result in zero.
    auto one = graph::one<T> ();
    assert((zero/one).get() == zero.get() && "Expected to recover zero.");
    assert((zero/one)->evaluate()[0] == static_cast<T> (0.0) &&
           "Expected a value of zero.");

// A one in the denominator should result in numerator.
    assert((one/one).get() == one.get() && "Expected to recover one.");
    assert((one/one)->evaluate()[0] == static_cast<T> (1.0) &&
           "Expected a value of one.");
    auto two = graph::constant(static_cast<T> (2.0));
    assert((two/one).get() == two.get() && "Expected to recover two.");
    assert((two/one)->evaluate()[0] == static_cast<T> (2.0) &&
           "Expected a value of zero.");

//  A value divided by it self should be a constant one.
    auto two_divided_two = two/two;
    auto two_divided_two_cast = graph::constant_cast(two_divided_two);
    assert(two_divided_two_cast.get() && "Expected a constant type.");
    assert(two_divided_two_cast->is(1) && "Expected 1 for result");
    assert(two_divided_two->evaluate()[0] == static_cast<T> (1.0) &&
           "Expected 1 for result");

//  A constant a divided by constant b should be a constant with value of a/b.
    auto three = graph::constant(static_cast<T> (3.0));
    auto two_divided_three = two/three;
    auto two_divided_three_cast = graph::constant_cast(two_divided_three);
    assert(two_divided_three_cast.get() && "Expected a constant type.");
    assert(two_divided_three_cast->is(2.0/3.0) && "Expected 2/3 for result");
    assert(two_divided_three->evaluate()[0] == static_cast<T> (2.0/3.0) &&
           "Expected 2/3 for result");

//  Test variables.
    auto variable = graph::variable<T> (1, "");
    assert((zero/variable).get() == zero.get() && "Expected to recover zero.");
    assert((variable/one).get() == variable.get() &&
           "Expected to recover numerator.");
    assert((zero/variable)->evaluate()[0] == static_cast<T> (0.0) &&
           "Expected a value of zero.");
    
    auto two_divided_var = two/variable;
    assert(graph::divide_cast(two_divided_var).get() &&
           "Expected divide node.");
    variable->set(static_cast<T> (3.0));
    const backend::buffer<T> two_divided_var_result = two_divided_var->evaluate();
    assert(two_divided_var_result.size() == 1 && "Expected single value.");
    assert(two_divided_var_result.at(0) == static_cast<T> (2.0) /
                                           static_cast<T> (3.0) &&
           "Expected 2/3 for result.");

//  v/c1 -> (1/c1)*v -> c2*v
    auto var_divided_two = variable/two;
    assert(graph::multiply_cast(var_divided_two).get() &&
           "Expected a multiply node.");
    const backend::buffer<T> var_divided_two_result = var_divided_two->evaluate();
    assert(var_divided_two_result.size() == 1 && "Expected single value.");
    assert(var_divided_two_result.at(0) == static_cast<T> (3.0) /
                                           static_cast<T> (2.0) &&
           "Expected 3/2 for result.");

    auto var_divided_var = variable/variable;
    auto var_divided_var_cast = graph::constant_cast(var_divided_var);
    assert(var_divided_var_cast.get() && "Expeced constant node.");
    assert(var_divided_var_cast->is(1) && "Expeced one.");

    auto variable_b = graph::variable<T> (1, 4, "");
    auto var_divided_varb = variable/variable_b;
    assert(graph::divide_cast(var_divided_varb).get() &&
           "Expected divide node.");
    const backend::buffer<T> var_divided_varb_result = var_divided_varb->evaluate();
    assert(var_divided_varb_result.size() == 1 && "Expected single value.");
    assert(var_divided_varb_result.at(0) == static_cast<T> (3.0) /
                                            static_cast<T> (4.0) &&
           "Expected 3/4 for result.");

//  Test vector variables.
    auto varvec = graph::variable<T> (std::vector<T> ({2.0, 6.0}), "");
    assert((zero/varvec).get() == zero.get() && "Expected to recover zero.");
    assert((varvec/one).get() == varvec.get() &&
           "Expected to recover numerator.");
    assert((zero/varvec)->evaluate()[0] == static_cast<T> (0.0) &&
           "Expected a value of zero.");
    assert((varvec/one)->evaluate()[0] == static_cast<T> (2.0) &&
           "Expected a value of two.");
    assert((varvec/one)->evaluate()[1] == static_cast<T> (6.0) &&
           "Expected a value of six.");

    auto varvec_divided_two = varvec/two;
    assert(graph::multiply_cast(varvec_divided_two).get() &&
           "Expect a mutliply node.");
    const backend::buffer<T> varvec_divided_two_result = varvec_divided_two->evaluate();
    assert(varvec_divided_two_result.size() == 2 && "Size mismatch in result.");
    assert(varvec_divided_two_result.at(0) == static_cast<T> (1.0) &&
           "Expected 2/2 for result.");
    assert(varvec_divided_two_result.at(1) == static_cast<T> (3.0) &&
           "Expected 6/2 for result.");

    auto two_divided_varvec = two/varvec;
    assert(graph::divide_cast(two_divided_varvec).get() &&
           "Expect divide node.");
    const backend::buffer<T> two_divided_varvec_result = two_divided_varvec->evaluate();
    assert(two_divided_varvec_result.size() == 2 && "Size mismatch in result.");
    assert(two_divided_varvec_result.at(0) == static_cast<T> (1.0) &&
           "Expected 2/2 for result.");
    assert(two_divided_varvec_result.at(1) == static_cast<T> (2.0) /
                                              static_cast<T> (6.0) &&
           "Expected 2/6 for result.");

    auto varvec_b = graph::variable<T> (std::vector<T> ({-3.0, 6.0}), "");
    auto varvec_divided_varvecb = varvec/varvec_b;
    assert(graph::divide_cast(varvec_divided_varvecb).get() &&
           "Expect divide node.");
    const backend::buffer<T> varvec_divided_varvecb_result =
        varvec_divided_varvecb->evaluate();
    assert(varvec_divided_varvecb_result.size() == 2 &&
           "Size mismatch in result.");
    assert(varvec_divided_varvecb_result.at(0) == static_cast<T> (2.0) /
                                                  static_cast<T> (-3.0) &&
           "Expected 2/-3 for result.");
    assert(varvec_divided_varvecb_result.at(1) == static_cast<T> (1.0) &&
           "Expected 6/6 for result.");

    auto varvecb_divided_varvec = varvec_b/varvec;
    assert(graph::divide_cast(varvecb_divided_varvec).get() &&
           "Expect divide node.");
    const backend::buffer<T> varvecb_divided_varvec_result =
        varvecb_divided_varvec->evaluate();
    assert(varvecb_divided_varvec_result.size() == 2 &&
           "Size mismatch in result.");
    assert(varvecb_divided_varvec_result.at(0) == static_cast<T> (-3.0) /
                                                  static_cast<T> (2.0) &&
           "Expected -3/2 for result.");
    assert(varvecb_divided_varvec_result.at(1) == static_cast<T> (1.0) &&
           "Expected 6/6 for result.");

//  Test reduction short cut. If all the elements in the numerator are zero, an
//  denominator does not need to be evaluated. This test makes sure that a sum
//  or product is not used to avoid cases like {-1, 0, 1} which sum and product
//  are zero.
    auto var_sum_prod = graph::variable<T> (std::vector<T> ({-2.0, 2.0, 0.0}), "");
    auto var_sum_prod_divided_two = var_sum_prod/two;
    const backend::buffer<T> var_sum_prod_divided_two_result =
        var_sum_prod_divided_two->evaluate();
    assert(var_sum_prod_divided_two_result.at(0) == static_cast<T> (-2.0) /
                                                    static_cast<T> (2.0) &&
           "Expected -2/2 for result.");
    assert(var_sum_prod_divided_two_result.at(1) == static_cast<T> (2.0) /
                                                    static_cast<T> (2.0) &&
           "Expected 2/2 for result.");
    assert(var_sum_prod_divided_two_result.at(2) == static_cast<T> (0.0) &&
           "Expected 0/2 for result.");

//  Test derivatives.
//  d (x/c) / dx = dxdx/c + x d 1/c /dx = 1/c
    auto dvar_divided_two = var_divided_two->df(variable);
    const backend::buffer<T> dvar_divided_two_result = dvar_divided_two->evaluate();
    assert(dvar_divided_two_result.at(0) == static_cast<T> (1.0) /
                                            static_cast<T> (2.0) &&
           "Expected 1/2 for result.");

//  d (c/x) / dx = dc/dx x - c/x^2 dx/dx = -c/x^2
    auto dtwo_divided_var = two_divided_var->df(variable);
    const backend::buffer<T> dtwo_divided_var_result = dtwo_divided_var->evaluate();
    assert(dtwo_divided_var_result.at(0) == static_cast<T> (-2.0) /
                                            (static_cast<T> (3.0) *
                                             static_cast<T> (3.0)) &&
           "Expected 2/3^2 for result.");

//  Test is_match
    auto match = (graph::one<T> () + variable)
               / (graph::one<T> () + variable);
    auto match_cast = graph::constant_cast(match);
    assert(match_cast->is(1) &&
           "Expected one constant for result.");

//  Test reduction of common constants (c1*x)/(c2*y) = c3*x/y.
    auto x1 = graph::constant(static_cast<T> (2.0))*graph::variable<T> (1, "");
    auto x2 = graph::constant(static_cast<T> (5.0))*graph::variable<T> (1, "");
    auto x3 = x1/x2;
    auto x3_cast = graph::multiply_cast(x3);
    assert(x3_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(x3_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::divide_cast(x3_cast->get_right()).get() &&
           "Expected multipy node.");

//  Test reduction of common constants (c1*x)/(y*c2) = c3*x/y.
    auto x4 = graph::variable<T> (1, "")*graph::constant(static_cast<T> (2.0));
    auto x5 = graph::constant(static_cast<T> (5.0))*graph::variable<T> (1, "");
    auto x6 = x4/x5;
    auto x6_cast = graph::multiply_cast(x6);
    assert(x6_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(x6_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::divide_cast(x6_cast->get_right()).get() &&
           "Expected multipy node.");

//  Test reduction of common constants (x*c1)/(c2*y) = c3*x/y.
    auto x7 = graph::constant(static_cast<T> (2.0))*graph::variable<T> (1, "");
    auto x8 = graph::variable<T> (1, "")*graph::constant(static_cast<T> (5.0));
    auto x9 = x7/x8;
    auto x9_cast = graph::multiply_cast(x9);
    assert(x9_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(x9_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::divide_cast(x9_cast->get_right()).get() &&
           "Expected multipy node.");

//  Test reduction of common constants (x*c1)/(y*c2) = c3*x/y.
    auto x10 = graph::variable<T> (1, "")*graph::constant(static_cast<T> (2.0));
    auto x11 = graph::constant(static_cast<T> (5.0))*graph::variable<T> (1, "");
    auto x12 = x10/x11;
    auto x12_cast = graph::multiply_cast(x12);
    assert(x12_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(x12_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::divide_cast(x12_cast->get_right()).get() &&
           "Expected multipy node.");

//  c1/(c2*v) -> c3/v
    auto c3 = two/(three*variable);
    auto c3_cast = graph::divide_cast(c3);
    assert(c3_cast.get() && "Expected divide node");
    assert(graph::constant_cast(c3_cast->get_left()).get() &&
           "Expected a constant in numerator.");
    assert(graph::variable_cast(c3_cast->get_right()).get() &&
           "Expected a variable in the denominator");

//  c1/(v*c2) -> c4/v
    auto c4 = two/(three*variable);
    auto c4_cast = graph::divide_cast(c4);
    assert(c4_cast.get() && "Expected divide node");
    assert(graph::constant_cast(c4_cast->get_left()).get() &&
           "Expected a constant in numerator.");
    assert(graph::variable_cast(c4_cast->get_right()).get() &&
           "Expected a variable in the denominator");

//  (c1*v)/c2 -> c5*v
    auto c5 = (two*variable)/three;
    auto c5_cast = graph::multiply_cast(c5);
    assert(c5_cast.get() && "Expected a multiply node");
    assert(graph::constant_cast(c5_cast->get_left()).get() &&
           "Expected a constant in the numerator");
    assert(graph::variable_cast(c5_cast->get_right()).get() &&
           "Expected a variable in the denominator.");

//  (v*c1)/c2 -> c5*v
    auto c6 = (variable*two)/three;
    auto c6_cast = graph::multiply_cast(c6);
    assert(c6_cast.get() && "Expected multiply node");
    assert(graph::constant_cast(c6_cast->get_left()).get() &&
           "Expected a constant in the numerator");
    assert(graph::variable_cast(c6_cast->get_right()).get() &&
           "Expected a variable in the denominator.");

//  (c*v1)/v2 -> c*(v1/v2)
    auto a = graph::variable<T> (1, "");
    auto c7 = (two*variable)/a;
    auto c7_cast = graph::multiply_cast(c7);
    assert(c7_cast.get() && "Expected multiply node");
    assert(graph::constant_cast(c7_cast->get_left()).get() &&
           "Expected a constant");

//  (v1*c)/v2 -> c*(v1/v2)
    auto c8 = (two*variable)/a;
    auto c8_cast = graph::multiply_cast(c8);
    assert(c8_cast.get() && "Expected multiply node");
    assert(graph::constant_cast(c8_cast->get_left()).get() &&
           "Expected a constant");

//  (v1*v2)/v1 -> v2
//  (v2*v1)/v1 -> v2
    auto v1 = (variable*a)/variable;
    auto v2 = (a*variable)/variable;
    assert(v1->is_match(a) && "Expected to reduce to a");
    assert(v2->is_match(a) && "Expected to reduce to a");

//  Test a^b/a^c -> a^(b - c)
    auto pow_bc = graph::pow(a, two)/graph::pow(a, three);
    auto pow_bc_cast = graph::pow_cast(pow_bc);
    assert(pow_bc_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_bc_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_bc_cast->get_right())->is(-1) &&
           "Expected negative 1");

//  Test a/a^c -> a^(1 - c)
    auto pow_c = a/graph::pow(a, three);
    auto pow_c_cast = graph::pow_cast(pow_c);
    assert(pow_c_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_c_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_c_cast->get_right())->is(-2) &&
           "Expected constant exponent equal to -2.");

//  Test a^b/a -> a^(b - 1)
    auto pow_b = graph::pow(a, two)/a;
    assert(pow_b->is_match(a) && "Expected to recover a.");

//  Test a^b/sqrt(a) -> a^(b - 0.5)
    auto pow_sqb = graph::pow(a, two)/graph::sqrt(a);
    auto pow_sqb_cast = graph::pow_cast(pow_sqb);
    assert(pow_sqb_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_sqb_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_sqb_cast->get_right())->is(1.5) &&
           "Expected constant exponent equal to 1.5.");

//  Test sqrt(a)/a^c -> a^(0.5 - c)
    auto pow_sqc = graph::sqrt(a)/graph::pow(a, three);
    auto pow_sqc_cast = graph::pow_cast(pow_sqc);
    assert(pow_sqc_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_sqc_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_sqc_cast->get_right())->is(-2.5) &&
           "Expected constant exponent equal to -2.5.");

//  Test a/sqrt(a) -> sqrt(a)
    auto pow_asqa = a/graph::sqrt(a);
    auto pow_asqa_cast = graph::sqrt_cast(pow_asqa);
    assert(pow_asqa_cast.get() && "Expected sqrt node.");

//  Test sqrt(a)/a -> 1.0/sqrt(a)
    auto pow_sqaa = graph::sqrt(a)/a;
    auto pow_sqaa_cast = graph::pow_cast(pow_sqaa);
    assert(pow_sqaa_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_sqaa_cast->get_right()).get() &&
           "Expected const in exponent.");

//  (c*v)/v -> c*v
//  (c/v)/v -> c/v
    auto test_var_move = [two](graph::shared_leaf<T> x,
                               graph::shared_leaf<T> y) {
        auto var_move = (two*x)/y;
        auto var_move_cast = graph::multiply_cast(var_move);
        assert(var_move_cast.get() && "Expected multiply.");
        assert(!var_move_cast->get_left()->is_all_variables() &&
               "Expected Non variable like in the left side.");
        assert(var_move_cast->get_right()->is_all_variables() &&
               "Expected variable like in the right side.");
        
        auto var_move2 = (two/x)/y;
        auto var_move2_cast = graph::divide_cast(var_move2);
        assert(var_move2_cast.get() && "Expected divide.");
        assert(!var_move2_cast->get_left()->is_all_variables() &&
               "Expected Non variable like in the left side.");
        assert(var_move2_cast->get_right()->is_all_variables() &&
               "Expected variable like in the right side.");
    };

    test_var_move(a, graph::sqrt(a));
    test_var_move(graph::pow(a, three), a);
    test_var_move(graph::pow(a, three), graph::pow(a, two));

//  fma(a,d,c*d)/d -> a + c
    auto fma_divide = graph::fma(graph::variable<T> (1, ""),
                                 a,
                                 graph::variable<T> (1, "")*a)/a;
    auto fma_divide_cast = graph::add_cast(fma_divide);
    assert(fma_divide_cast.get() && "Expected an add node.");
//  fma(a,d,c*d)/d -> a + c
    auto fma_divide2 = graph::fma(graph::variable<T> (1, ""),
                                  a,
                                  a*graph::variable<T> (1, ""))/a;
    auto fma_divide_cast2 = graph::add_cast(fma_divide2);
    assert(fma_divide_cast2.get() && "Expected an fma node.");
//  fma(d,a,c*d)/d -> a + c
    auto fma_divide3 = graph::fma(a,
                                 graph::variable<T> (1, ""),
                                 graph::variable<T> (1, "")*a)/a;
    auto fma_divide_cast3 = graph::add_cast(fma_divide3);
    assert(fma_divide_cast3.get() && "Expected an add node.");
//  fma(d,a,c*d)/d -> a + c
    auto fma_divide4 = graph::fma(a,
                                  graph::variable<T> (1, ""),
                                  a*graph::variable<T> (1, ""))/a;
    auto fma_divide_cast4 = graph::add_cast(fma_divide4);
    assert(fma_divide_cast4.get() && "Expected an add node.");

//  (a*b^c)/b^d -> a*b^(c - d)
    auto common_power = (variable*graph::pow(a, three))/graph::pow(a, two);
    assert(graph::multiply_cast(common_power).get() &&
           "Expected a multiply node.");
//  (b^c*a)/b^d -> a*b^(c - d)
    auto common_power2 = (graph::pow(a, three)*variable)/graph::pow(a, two);
    assert(graph::multiply_cast(common_power2).get() &&
           "Expected a multiply node.");
}

//------------------------------------------------------------------------------
///  @brief Tests for fma nodes.
//------------------------------------------------------------------------------
template<typename T> void test_fma() {
//  Three constant nodes should reduce to a single constant node with a*b + c.
    auto zero = graph::zero<T> ();
    auto one = graph::one<T> ();
    auto two = graph::two<T> ();

    auto zero_times_one_plus_two = graph::fma(zero, one, two);
    auto zero_times_one_plus_two_cast =
        graph::constant_cast(zero_times_one_plus_two);
    assert(zero_times_one_plus_two_cast.get() && "Expected a constant type.");
    assert(zero_times_one_plus_two_cast.get() == two.get() &&
           "Expected two.");
    assert(zero_times_one_plus_two->evaluate()[0] == static_cast<T> (2.0) &&
           "Expected a value of two.");

    auto one_times_zero_plus_two = graph::fma(one, zero, two);
    auto one_times_zero_plus_two_cast =
        graph::constant_cast(one_times_zero_plus_two);
    assert(one_times_zero_plus_two_cast.get() && "Expected a constant type.");
    assert(one_times_zero_plus_two_cast.get() == two.get() &&
           "Expected two.");
    assert(one_times_zero_plus_two->evaluate()[0] == static_cast<T> (2.0) &&
           "Expected a value of two.");
    
    auto one_times_two_plus_zero = graph::fma(one, two, zero);
    auto one_times_two_plus_zero_cast =
        graph::constant_cast(one_times_two_plus_zero);
    assert(one_times_two_plus_zero_cast.get() && "Expected a constant type.");
    assert(one_times_two_plus_zero_cast.get() == two.get() &&
           "Expected two.");
    assert(one_times_two_plus_zero->evaluate()[0] == static_cast<T> (2.0) &&
           "Expected a value of two.");

    auto three = graph::constant(static_cast<T> (3.0));
    auto one_two_three = graph::fma(one, two, three);
    const backend::buffer<T> one_two_three_result = one_two_three->evaluate();
    assert(one_two_three_result.size() == 1 && "Expected single value.");
    assert(one_two_three_result.at(0) == static_cast<T> (5.0) &&
           "Expected five for result");

    auto two_three_one = graph::fma(two, three, one);
    const backend::buffer<T> two_three_one_result = two_three_one->evaluate();
    assert(two_three_one_result.size() == 1 && "Expected single value.");
    assert(two_three_one_result.at(0) == static_cast<T> (7) &&
           "Expected seven for result");

//  Test a variable.
    auto var = graph::variable<T> (1, "");
    auto zero_times_var_plus_two = graph::fma(zero, var, two);
    auto zero_times_var_plus_two_cast =
        graph::constant_cast(zero_times_var_plus_two);
    assert(zero_times_var_plus_two_cast.get() && "Expected a constant type.");
    assert(zero_times_var_plus_two_cast.get() == two.get() &&
           "Expected two.");
    assert(zero_times_var_plus_two->evaluate()[0] == static_cast<T> (2.0) &&
           "Expected a value of two.");

    auto var_times_zero_plus_two = graph::fma(var, zero, two);
    auto var_times_zero_plus_two_cast =
        graph::constant_cast(var_times_zero_plus_two);
    assert(var_times_zero_plus_two_cast.get() && "Expected a constant type.");
    assert(var_times_zero_plus_two_cast.get() == two.get() &&
           "Expected two.");

    auto zero_times_two_plus_var = graph::fma(zero, two, var);
    auto zero_times_two_plus_var_cast =
        graph::variable_cast(zero_times_two_plus_var);
    assert(zero_times_two_plus_var_cast.get() && "Expected a variable type.");
    assert(zero_times_two_plus_var_cast.get() == var.get() &&
           "Expected var.");

//  Test derivative.
    auto constant_df = one_times_two_plus_zero->df(var);
    auto constant_df_cast = graph::constant_cast(constant_df);
    assert(constant_df_cast.get() && "Expected a constant node.");
    assert(constant_df_cast->is(0) && "Expected zero.");
    assert(constant_df->evaluate()[0] == static_cast<T> (0.0) &&
           "Expected a value of zero.");
    
    auto zero_times_var_plus_two_df = zero_times_var_plus_two->df(var);
    auto zero_times_var_plus_two_df_cast =
        graph::constant_cast(zero_times_var_plus_two_df);
    assert(zero_times_var_plus_two_df_cast.get() &&
           "Expected a constant node.");
    assert(zero_times_var_plus_two_df_cast->is(0) && "Expected zero.");
    assert(zero_times_var_plus_two_df->evaluate()[0] == static_cast<T> (0.0) &&
           "Expected a value of zero.");

    auto var_times_zero_plus_two_df = zero_times_var_plus_two->df(var);
    auto var_times_zero_plus_two_df_cast =
        graph::constant_cast(var_times_zero_plus_two_df);
    assert(var_times_zero_plus_two_df_cast.get() &&
           "Expected a constant node.");
    assert(var_times_zero_plus_two_df_cast->is(0) && "Expected zero.");
    assert(var_times_zero_plus_two_df->evaluate()[0] == static_cast<T> (0.0) &&
           "Expected a value of zero.");

    auto zero_times_two_plus_var_df = zero_times_two_plus_var->df(var);
    auto zero_times_two_plus_var_df_cast =
        graph::constant_cast(zero_times_two_plus_var_df);
    assert(zero_times_two_plus_var_df_cast.get() &&
           "Expected a constant node.");
    assert(zero_times_two_plus_var_df_cast->is(1) && "Expected one.");
    assert(zero_times_two_plus_var_df->evaluate()[0] == static_cast<T> (1.0) &&
           "Expected a value of one.");

//  Test reduction.
    auto var_a = graph::variable<T> (1, "");
    auto var_b = graph::variable<T> (1, "");
    auto var_c = graph::variable<T> (1, "");

    auto reduce1 = graph::fma(var_a, var_b, var_a*var_c);
    auto reduce1_cast = graph::multiply_cast(reduce1);
    assert(reduce1_cast.get() && "Expected multiply node.");
    assert(reduce1_cast->get_right()->is_match(var_a) &&
           "Expected common var_a");

    auto reduce2 = graph::fma(var_a, var_b, var_b*var_c);
    auto reduce2_cast = graph::multiply_cast(reduce2);
    assert(reduce2_cast.get() && "Expected multiply node.");
    assert(reduce2_cast->get_right()->is_match(var_b) &&
           "Expected common var_b");

    auto reduce3 = graph::fma(var_a, var_b, var_c*var_a);
    auto reduce3_cast = graph::multiply_cast(reduce3);
    assert(reduce3_cast.get() && "Expected multiply node.");
    assert(reduce3_cast->get_right()->is_match(var_a) &&
           "Expected common var_a");

    auto reduce4 = graph::fma(var_a, var_b, var_c*var_b);
    auto reduce4_cast = graph::multiply_cast(reduce4);
    assert(reduce4_cast.get() && "Expected multiply node.");
    assert(reduce4_cast->get_right()->is_match(var_b) &&
           "Expected common var_b");

    assert(graph::multiply_cast(graph::fma(two, var_a, one)).get() &&
           "Expected multiply node.");

//  fma(a, b, a*b) -> 2*a*b
//  fma(b, a, a*b) -> 2*a*b
//  fma(a, b, b*a) -> 2*a*b
    assert(graph::fma(var_a, var_b, var_a*var_b)->is_match(two*var_a*var_b) &&
           "Expected to match 2*a*b");
    assert(graph::fma(var_b, var_a, var_a*var_b)->is_match(two*var_a*var_b) &&
           "Expected to match 2*a*b");
    assert(graph::fma(var_a, var_b, var_b*var_a)->is_match(two*var_a*var_b) &&
           "Expected to match 2*a*b");

//  fma(c1*a,b,c2*d) -> c1*(a*b + c2/c1*d)
    assert(graph::multiply_cast(graph::fma(two*var_b,
                                           var_a,
                                           two*two*var_b)).get() &&
           "Expected multiply node.");

//  fma(c1*a,b,c2/d) -> c1*(a*b + c1/(c2*d))
//  fma(c1*a,b,d/c2) -> c1*(a*b + d/(c1*c2))
    assert(graph::multiply_cast(graph::fma(two*var_b,
                                           var_a,
                                           two*two/var_b)).get() &&
           "Expected multiply node.");
    assert(graph::multiply_cast(graph::fma(two*var_b,
                                           var_a,
                                           var_b/(two*two))).get() &&
           "Expected multiply node.");

//  fma(a,v1,b*v2) -> (a + b*v1/v2)*v1
//  fma(a,v1,c*b*v2) -> (a + c*b*v1/v2)*v1
    assert(graph::multiply_cast(graph::fma(two,
                                           var_a,
                                           two*sqrt(var_a))).get() &&
           "Expected multiply node.");
    assert(graph::multiply_cast(graph::fma(two,
                                           var_a,
                                           two*(var_b*sqrt(var_a)))).get() &&
           "Expected multiply node.");

//  fma(a,b,fma(a,b,c)) -> fma(2a,b,c)
    auto chained_fma = fma(var_a, var_b, fma(var_a, var_b, two));
    auto chained_fma_cast = fma_cast(chained_fma);
    assert(chained_fma_cast.get() && "Expected fma node.");
    assert(constant_cast(chained_fma_cast->get_right()) &&
           "Expected constant node.");
//  fma(a,b,fma(b,a,c)) -> fma(2a,b,c)
    auto chained_fma2 = fma(var_a, var_b, fma(var_b, var_a, two));
    auto chained_fma_cast2 = fma_cast(chained_fma2);
    assert(chained_fma_cast2.get() && "Expected fma node.");
    assert(constant_cast(chained_fma_cast2->get_right()) &&
           "Expected constant node.");
    
//  fma(a,b/c,fma(d,e/c,g)) -> (a*b + d*e)/c + g
    auto var_d = graph::variable<T> (1, "");
    auto var_e = graph::variable<T> (1, "");
    auto chained_fma3 = fma(var_a, var_b/var_c, fma(var_d, var_e/var_c, var));
    assert(add_cast(chained_fma3).get() && "expected add node.");
//  fma(a,b/c,fma(e/c,f,g)) -> (a*b + e*f)/c + g
    auto chained_fma4 = fma(var_a, var_b/var_c, fma(var_d/var_c, var_e, var));
    assert(add_cast(chained_fma3).get() && "expected add node.");
//  fma(a/c,b,fma(e,f/c,g)) -> (a*b + e*f)/c + g
    auto chained_fma5 = fma(var_a/var_c, var_b, fma(var_d, var_e/var_c, var));
    assert(add_cast(chained_fma5).get() && "expected add node.");
//  fma(a/c,b,fma(e/c,f,g)) -> (a*b + e*f)/c + g
    auto chained_fma6 = fma(var_a/var_c, var_b, fma(var_d/var_c, var_e, var));
    assert(add_cast(chained_fma6).get() && "expected add node.");

//  fma(a,b^-c,d/b^c) -> (a + d)/b^c
    auto none = graph::none<T> ();
    auto power_factor = graph::fma(var_a, graph::pow(var_b, none*two),
                                   var_c/graph::pow(var_b, two));
    auto power_factor_cast = divide_cast(power_factor);
    assert(power_factor_cast.get() && "Expected a divide node.");
//  fma(b^-c,a,d/b^c) -> (a + d)/b^c
    auto power_factor2 = graph::fma(graph::pow(var_b, none*two), var_a,
                                    var_c/graph::pow(var_b, two));
    auto power_factor_cast2 = divide_cast(power_factor2);
    assert(power_factor_cast2.get() && "Expected a divide node.");

//  fma(a,b/c,b/d) -> b*(a/c + 1/d)
    auto divide_factor = graph::fma(var_a, var_b/var_c, var_b/var_d);
    assert(graph::multiply_cast(divide_factor).get() &&
           "Expetced a multiply node.");
//  fma(a,c/b,d/b) -> (a*c + d)/b
    auto divide_factor2 = graph::fma(var_a, var_c/var_b, var_d/var_b);
    assert(graph::divide_cast(divide_factor2).get() &&
           "Expetced a divide node.");
//  fma(b/c,a,b/d) -> b*(a/c + 1/d)
    auto divide_factor3 = graph::fma(var_b/var_c, var_a, var_b/var_d);
    assert(graph::multiply_cast(divide_factor3).get() &&
           "Expetced a multiply node.");
//  fma(c/b,a,d/b) -> (a*c + d)/b
    auto divide_factor4 = graph::fma(var_c/var_b, var_a, var_d/var_b);
    assert(graph::divide_cast(divide_factor4).get() &&
           "Expetced a divide node.");
}

//------------------------------------------------------------------------------
///  @brief Tests function for variable like expressions.
//------------------------------------------------------------------------------
template<typename T> void test_variable_like() {
    auto a = graph::variable<T> (1, "");
    auto c = graph::one<T> ();
    
    assert(a->is_all_variables() && "Expected a to be variable like.");
    assert(graph::sqrt(a)->is_all_variables() &&
           "Expected sqrt(a) to be variable like.");
    assert(graph::pow(a, c)->is_all_variables() &&
           "Expected a^c to be variable like.");
    
    assert(!c->is_all_variables() &&
           "Expected c to not be variable like.");
    assert(!graph::sqrt(c)->is_all_variables() &&
           "Expected sqrt(c) to not be variable like.");
    assert(!graph::pow(c, a)->is_all_variables() &&
           "Expected c^a to not be variable like.");
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
//------------------------------------------------------------------------------
template<typename T> void run_tests() {
    test_variable_like<T> ();
    test_add<T> ();
    test_subtract<T> ();
    test_multiply<T> ();
    test_divide<T> ();
    test_fma<T> ();
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
