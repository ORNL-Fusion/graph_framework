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
#include "../graph_framework/trigonometry.hpp"

//------------------------------------------------------------------------------
///  @brief Tests for addition nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_add() {
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

//  v1 + -1*v2 -> v1 - v2
    auto add_neg = var_a + -var_b;
    assert(graph::subtract_cast(add_neg).get() && "Expected subtract node.");

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
    auto common_d_acast2 = graph::subtract_cast(common_d);
    assert(common_d_acast2.get() && "Expected add node.");
    assert(graph::constant_cast(common_d_acast2->get_left()).get() &&
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
    
//  (a/(b*c) + d/(e*c)) -> (a/b + d/e)/c
    auto muliply_divide_factor = var_a/(var_b*var_c) + var_d/(var_e*var_c);
    auto muliply_divide_factor_cast = divide_cast(muliply_divide_factor);
    assert(muliply_divide_factor_cast.get() && "Expected divide node.");
    assert(muliply_divide_factor_cast->get_right()->is_match(var_c) &&
           "Expected var_c to be factored out.");
//  (a/(b*c) + d/(c*e)) -> (a/b + d/e)/c
    auto muliply_divide_factor2 = var_a/(var_b*var_c) + var_d/(var_c*var_e);
    auto muliply_divide_factor_cast2 = divide_cast(muliply_divide_factor2);
    assert(muliply_divide_factor_cast2.get() && "Expected divide node.");
    assert(muliply_divide_factor_cast2->get_right()->is_match(var_c) &&
           "Expected var_c to be factored out.");
//  (a/(c*b) + d/(e*c)) -> (a/b + d/e)/c
    auto muliply_divide_factor3 = var_a/(var_c*var_b) + var_d/(var_e*var_c);
    auto muliply_divide_factor_cast3 = divide_cast(muliply_divide_factor3);
    assert(muliply_divide_factor_cast3.get() && "Expected divide node.");
    assert(muliply_divide_factor_cast3->get_right()->is_match(var_c) &&
           "Expected var_c to be factored out.");
//  (a/(c*b) + d/(c*e)) -> (a/b + d/e)/c
    auto muliply_divide_factor4 = var_a/(var_c*var_b) + var_d/(var_c*var_e);
    auto muliply_divide_factor_cast4 = divide_cast(muliply_divide_factor4);
    assert(muliply_divide_factor_cast4.get() && "Expected divide node.");
    assert(muliply_divide_factor_cast4->get_right()->is_match(var_c) &&
           "Expected var_c to be factored out.");

//  Test node properties.
    assert(three->is_constant() && "Expected a constant.");
    assert(!three->is_all_variables() && "Did not expect a variable.");
    assert(three->is_power_like() && "Expected a power like.");
    auto constant_add = three + graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (1.0),
                                                                         static_cast<T> (2.0)}), var_a);
    assert(constant_add->is_constant() && "Expected a constant.");
    assert(!constant_add->is_all_variables() && "Did not expect a variable.");
    assert(constant_add->is_power_like() && "Expected a power like.");
    auto constant_var_add = three + var_a;
    assert(!constant_var_add->is_constant() && "Did not expect a constant.");
    assert(!constant_var_add->is_all_variables() && "Did not expect a variable.");
    assert(!constant_var_add->is_power_like() && "Did not expect a power like.");
    auto var_var_add = var_a + variable;
    assert(!var_var_add->is_constant() && "Did not expect a constant.");
    assert(var_var_add->is_all_variables() && "Expected a variable.");
    assert(!var_var_add->is_power_like() && "Did not expect a power like.");

//  Test common denominators.
//  a/b + c/(b*d) -> (a*d + c)/(b*d)
    auto common_denom1 = var_a/var_b + var_c/(var_b*var_d);
    auto common_denom1_cast = graph::divide_cast(common_denom1);
    assert(common_denom1_cast.get() && "Expected a divide node.");
    assert(common_denom1_cast->get_right()->is_match(var_b*var_d) &&
           "Expected var_b*var_d as common denominator.");
    assert(common_denom1_cast->get_left()->is_match(graph::fma(var_a,
                                                               var_d,
                                                               var_c)) &&
           "Expected fma(a,d,c) as numerator.");
//  a/b + c/(d*b) -> (a*d + c)/(d*b)
    auto common_denom2 = var_a/var_b + var_c/(var_d*var_b);
    auto common_denom2_cast = graph::divide_cast(common_denom2);
    assert(common_denom2_cast.get() && "Expected a divide node.");
    assert(common_denom2_cast->get_right()->is_match(var_d*var_b) &&
           "Expected var_b*var_d as common denominator.");
    assert(common_denom2_cast->get_left()->is_match(graph::fma(var_a,
                                                               var_d,
                                                               var_c)) &&
           "Expected fma(a,d,c) as numerator.");
//  a/(b*d) + c/b -> (c*d + a)/(b*d)
    auto common_denom3 = var_a/(var_b*var_d) + var_c/var_b;
    auto common_denom3_cast = graph::divide_cast(common_denom3);
    assert(common_denom3_cast.get() && "Expected a divide node.");
    assert(common_denom3_cast->get_right()->is_match(var_b*var_d) &&
           "Expected var_b*var_d as common denominator.");
    assert(common_denom3_cast->get_left()->is_match(graph::fma(var_c,
                                                               var_d,
                                                               var_a)) &&
           "Expected fma(c,d,a) as numerator.");
//  a/(d*b) + c/b -> (c*d + a)/(d*b)
    auto common_denom4 = var_a/(var_d*var_b) + var_c/var_b;
    auto common_denom4_cast = graph::divide_cast(common_denom4);
    assert(common_denom4_cast.get() && "Expected a divide node.");
    assert(common_denom4_cast->get_right()->is_match(var_d*var_b) &&
           "Expected var_b*var_d as common denominator.");
    assert(common_denom4_cast->get_left()->is_match(graph::fma(var_c,
                                                               var_d,
                                                               var_a)) &&
           "Expected fma(c,d,a) as numerator.");

//  a*b/c + d*b/e -> (a/c + d/e)*b
    auto factor = var_a*var_b/var_c + var_d*var_b/var_e;
    auto factor_cast = graph::multiply_cast(factor);
    assert(factor_cast.get() && "Expected a multiply node.");
    assert(factor->is_match((var_a/var_c + var_d/var_e)*var_b) &&
           "Expected (a/c + d/e)*b.");
//  a*b/c + b*d/e -> (a/c + d/e)*b
    auto factor2 = var_a*var_b/var_c + var_b*var_d/var_e;
    auto factor2_cast = graph::multiply_cast(factor2);
    assert(factor2_cast.get() && "Expected a multiply node.");
    assert(factor2->is_match((var_a/var_c + var_d/var_e)*var_b) &&
           "Expected (a/c + d/e)*b.");
//  b*a/c + d*b/e -> (a/c + d/e)*b
    auto factor3 = var_b*var_a/var_c + var_d*var_b/var_e;
    auto factor3_cast = graph::multiply_cast(factor3);
    assert(factor3_cast.get() && "Expected a multiply node.");
    assert(factor3->is_match((var_a/var_c + var_d/var_e)*var_b) &&
           "Expected (a/c + d/e)*b.");
//  b*a/c + b*d/e -> (a/c + d/e)*b
    auto factor4 = var_b*var_a/var_c + var_b*var_d/var_e;
    auto factor4_cast = graph::multiply_cast(factor4);
    assert(factor4_cast.get() && "Expected a multiply node.");
    assert(factor4->is_match((var_a/var_c + var_d/var_e)*var_b) &&
           "Expected (a/c + d/e)*b.");

//  c1*a/b + c2*a/d = c3*(a/b + c4*a/d)
    auto two = graph::constant(static_cast<T> (2.0));
    auto common_const = two*var_a/var_b + three*var_c/var_d;
    auto common_const_cast = graph::multiply_cast(common_const);
    assert(common_const_cast.get() && "Expected a multiply node.");
    assert(common_const_cast->get_left()->is_match(two) &&
           "Expected a constant of 2.0");
    assert(common_const_cast->get_right()->is_match(var_a/var_b + 3.0/2.0*var_c/var_d) &&
           "Expected a/b + 3/2*c/d");

//  a/b - c*a/d -> (1/b + c/d)*a
    auto common_var = var_a/var_b + var_c*var_a/var_d;
    auto common_var_cast = graph::multiply_cast(common_var);
    assert(common_var_cast.get() && "Expected a multiply node.");
    assert(common_var_cast->get_right()->is_match(var_a) &&
           "Expected var_a");
    assert(common_var_cast->get_left()->is_match(1.0/var_b + var_c/var_d) &&
           "Expected 1/b + c/d");
//  a/b - a*c/d -> (1/b + c/d)*a
    auto common_var2 = var_a/var_b + var_a*var_c/var_d;
    auto common_var2_cast = graph::multiply_cast(common_var2);
    assert(common_var2_cast.get() && "Expected a multiply node.");
    assert(common_var2_cast->get_right()->is_match(var_a) &&
           "Expected var_a");
    assert(common_var2_cast->get_left()->is_match(1.0/var_b + var_c/var_d) &&
           "Expected 1/b + c/d");
//  c*a/b - a/d -> (c/b + 1/d)*a
    auto common_var3 = var_c*var_a/var_b + var_a/var_d;
    auto common_var3_cast = graph::multiply_cast(common_var3);
    assert(common_var3_cast.get() && "Expected a multiply node.");
    assert(common_var3_cast->get_right()->is_match(var_a) &&
           "Expected var_a");
    assert(common_var3_cast->get_left()->is_match(var_c/var_b + 1.0/var_d) &&
           "Expected c/b + 1/d");
//  a*c/b - a/d -> (c/b + 1/d)*a
    auto common_var4 = var_a*var_c/var_b + var_a/var_d;
    auto common_var4_cast = graph::multiply_cast(common_var4);
    assert(common_var4_cast.get() && "Expected a multiply node.");
    assert(common_var4_cast->get_right()->is_match(var_a) &&
           "Expected var_a");
    assert(common_var4_cast->get_left()->is_match(var_c/var_b + 1.0/var_d) &&
           "Expected c/b + 1/d");

    auto common_var5 = 2.0*var_a/var_b + 3.0*var_a/var_c;
    auto common_var5_cast = graph::multiply_cast(common_var5);
    assert(common_var5_cast.get() && "Expected a multiply node.");
    assert(common_var5_cast->get_right()->is_match(var_a) &&
           "Expected var_a");
    assert(common_var5_cast->get_left()->is_match(2.0/var_b + 3.0/var_c) &&
           "Expected 2/b + 3/c");

//  (a*b)^c + (a*d)^c -> a^c*(b^c + d^c)
    auto common_power_factor = graph::pow(var_a*var_b, 2.0)
                             + graph::pow(var_a*var_c, 2.0);
    auto common_power_factor_cast = multiply_cast(common_power_factor);
    assert(common_power_factor_cast.get() && "Expected a multiply node.");
    assert(common_power_factor_cast->get_right()->is_match(var_a*var_a) &&
           "Expected a^2 on the right.");
    assert(common_power_factor_cast->get_left()->is_match(var_b*var_b +
                                                          var_c*var_c) &&
           "Expected b^2 + c^2 on the left.");
//  (a*b)^c + (d*a)^c -> a^c*(b^c + d^c)
    auto common_power_factor2 = graph::pow(var_a*var_b, 2.0)
                              + graph::pow(var_c*var_a, 2.0);
    auto common_power_factor2_cast = multiply_cast(common_power_factor2);
    assert(common_power_factor2_cast.get() && "Expected a multiply node.");
    assert(common_power_factor2_cast->get_right()->is_match(var_a*var_a) &&
           "Expected a^2 on the right.");
    assert(common_power_factor2_cast->get_left()->is_match(var_b*var_b +
                                                           var_c*var_c) &&
           "Expected b^2 + c^2 on the left.");
//  (b*a)^c + (a*d)^c -> a^c*(b^c + d^c)
    auto common_power_factor3 = graph::pow(var_b*var_a, 2.0)
                              + graph::pow(var_a*var_c, 2.0);
    auto common_power_factor3_cast = multiply_cast(common_power_factor3);
    assert(common_power_factor3_cast.get() && "Expected a multiply node.");
    assert(common_power_factor3_cast->get_right()->is_match(var_a*var_a) &&
           "Expected a^2 on the right.");
    assert(common_power_factor3_cast->get_left()->is_match(var_b*var_b +
                                                           var_c*var_c) &&
           "Expected b^2 + c^2 on the left.");
//  (b*a)^c + (d*a)^c -> a^c*(b^c + d^c)
    auto common_power_factor4 = graph::pow(var_b*var_a, 2.0)
                              + graph::pow(var_c*var_a, 2.0);
    auto common_power_factor4_cast = multiply_cast(common_power_factor4);
    assert(common_power_factor4_cast.get() && "Expected a multiply node.");
    assert(common_power_factor4_cast->get_right()->is_match(var_a*var_a) &&
           "Expected a^2 on the right.");
    assert(common_power_factor4_cast->get_left()->is_match(var_b*var_b +
                                                           var_c*var_c) &&
           "Expected b^2 + c^2 on the left.");

//  cos(x)^2 + sin(x)^2 -> 1
    auto trig = graph::cos(var_a)*graph::cos(var_a)
              + graph::sin(var_a)*graph::sin(var_a);
    auto trig_cast = graph::constant_cast(trig);
    assert(trig_cast.get() && "Expected a constant node.");
    assert(trig_cast->is(static_cast<T> (1.0)) && "Expected 1.");
//  sin(x)^2 + cos(x)^2 -> 1
    auto trig2 = graph::sin(var_a)*graph::sin(var_a)
               + graph::cos(var_a)*graph::cos(var_a);
    auto trig2_cast = graph::constant_cast(trig2);
    assert(trig2_cast.get() && "Expected a constant node.");
    assert(trig2_cast->is(static_cast<T> (1.0)) && "Expected 1.");
}

//------------------------------------------------------------------------------
///  @brief Tests for subtract nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_subtract() {
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
    auto negate = var_a - (-2.0*var_b);
    assert(graph::fma_cast(negate).get() && "Expected addition node.");

//  v1 - -1*v2 -> v1 + v2
    neg_one = var_a - -var_b;
    assert(graph::add_cast(neg_one).get() && "Expected addition node.");

//  (c1*v1 + c2) - (c3*v1 + c4) -> c5*(v1 - c6)
    auto subfma = graph::fma(3.0, var_a, 2.0)
                - graph::fma(2.0, var_a, 3.0);
//  -1 + a
    assert(graph::add_cast(subfma).get() && "Expected an add node.");

//  Test cases like
//  (c1 + c2/x) - c3/x -> c1 + c4/x
//  (c1 - c2/x) - c3/x -> c1 - c4/x
    common_d = (1.0 + 3.0/var_a) - (1.0/var_a);
    auto common_d_acast = graph::add_cast(common_d);
    assert(common_d_acast.get() && "Expected add node.");
    assert(graph::constant_cast(common_d_acast->get_left()).get() &&
           "Expected constant on the left.");
    common_d = (1.0 - 3.0/var_a) - (1.0/var_a);
    auto common_d_scast = graph::subtract_cast(common_d);
    assert(common_d_scast.get() && "Expected subtract node.");
    assert(graph::constant_cast(common_d_scast->get_left()).get() &&
           "Expected constant on the left.");

//  c1*a - c2*b -> c1*(a - c3*b)
    auto common_factor = 3.0*var_a - 2.0*var_b;
    assert(graph::multiply_cast(common_factor).get() &&
           "Expected multilpy node.");

//  (c1 - c2*v) - c3*v -> c1 - c4*v (1 - 3v) - 2v = 1 - 5*v
    auto chained_subtract = (1.0 - 3.0*var_a) - 2.0*var_a;
    auto chained_subtract_cast = graph::subtract_cast(chained_subtract);
    assert(chained_subtract_cast.get() &&
           "Expected subtract node.");
    assert(graph::constant_cast(chained_subtract_cast->get_left()).get() &&
           "Expected a constant node on the left.");
    assert(graph::multiply_cast(chained_subtract_cast->get_right()).get() &&
           "Expected a multiply node on the right.");

//  (a - b*c) - d*c -> a - (b + d)*c
    auto chained_subtract2 = (var_b - 2.0*var_a) - var_c*var_a;
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
    auto chained_subtract3 = (var_b - 2.0*var_a) - var_a*var_c;
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
    auto chained_subtract4 = (var_b - var_a*2.0) - var_c*var_a;
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
    auto chained_subtract5 = (var_b - var_a*2.0) - var_a*var_c;
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
    auto common_factor2 = var_a*var_b - 2.0*(var_c*var_b);
    assert(graph::multiply_cast(common_factor2).get() &&
           "Expected multiply node.");
    auto common_factor3 = var_a*var_b - 2.0*(var_b*var_c);
    assert(graph::multiply_cast(common_factor3).get() &&
           "Expected multiply node.");
    auto common_factor4 = var_b*var_a - 2.0*(var_c*var_b);
    assert(graph::multiply_cast(common_factor4).get() &&
           "Expected multiply node.");
    auto common_factor5 = var_b*var_a - 2.0*(var_b*var_c);
    assert(graph::multiply_cast(common_factor5).get() &&
           "Expected multiply node.");
//  c*(d*b) - a*b -> (c*d - a)*b
//  c*(b*d) - a*b -> (c*d - a)*b
//  c*(d*b) - b*a -> (c*d - a)*b
//  c*(b*d) - b*a -> (c*d - a)*b
    auto common_factor6 = 2.0*(var_c*var_b) - var_a*var_b;
    assert(graph::multiply_cast(common_factor6).get() &&
           "Expected multiply node.");
    auto common_factor7 = 2.0*(var_b*var_c) - var_a*var_b;
    assert(graph::multiply_cast(common_factor7).get() &&
           "Expected multiply node.");
    auto common_factor8 = 2.0*(var_c*var_b) - var_b*var_a;
    assert(graph::multiply_cast(common_factor8).get() &&
           "Expected multiply node.");
    auto common_factor9 = 2.0*(var_b*var_c) - var_b*var_a;
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

//  Test node properties.
    assert(zero->is_constant() && "Expected a constant.");
    assert(!zero->is_all_variables() && "Did not expect a variable.");
    assert(zero->is_power_like() && "Expected a power like.");
    auto constant_sub = one - graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (1.0),
                                                                       static_cast<T> (2.0)}), var_a);
    assert(constant_sub->is_constant() && "Expected a constant.");
    assert(!constant_sub->is_all_variables() && "Did not expect a variable.");
    assert(constant_sub->is_power_like() && "Expected a power like.");
    auto constant_var_sub = one - var_a;
    assert(!constant_var_sub->is_constant() && "Did not expect a constant.");
    assert(!constant_var_sub->is_all_variables() && "Did not expect a variable.");
    assert(!constant_var_sub->is_power_like() && "Did not expect a power like.");
    auto var_var_sub = var_a - var_b;
    assert(!var_var_sub->is_constant() && "Did not expect a constant.");
    assert(var_var_sub->is_all_variables() && "Expected a variable.");
    assert(!var_var_sub->is_power_like() && "Did not expect a power like.");

//  a/b*c - d/b*e -> (a*b - d*e)/b
    auto var_e = graph::variable<T> (1, "");
    assert(graph::divide_cast((var_a/var_b)*var_c - (var_d/var_b)*var_e).get() &&
           "Expected a divide node.");
//  a/b*c - d*e/b -> (a*b - d*e)/b
    assert(graph::divide_cast((var_a/var_b)*var_c - var_d*(var_e/var_b)).get() &&
           "Expected a divide node.");
//  a*c/b - d/b*e -> (a*b - d*e)/b
    assert(graph::divide_cast(var_a*(var_c/var_b) - (var_d/var_b)*var_e).get() &&
           "Expected a divide node.");
//  a*c/b - d*e/b -> (a*b - d*e)/b
    assert(graph::divide_cast(var_a*(var_c/var_b) - var_d*(var_e/var_b)).get() &&
           "Expected a divide node.");

//  (a*v) - v -> (a - 1)*v
    auto factor1 = (var_a*var_b) - var_b;
    assert(graph::multiply_cast(factor1).get() &&
           "Expected a multiply node.");
//  (v*a) - v -> (a - 1)*v
    auto factor2 = (var_b*var_a) - var_b;
    assert(graph::multiply_cast(factor2).get() &&
           "Expected a multiply node.");
//  v - (a*v) -> (1 - a)*v
    auto factor3 = var_b - (var_a*var_b);
    assert(graph::multiply_cast(factor3).get() &&
           "Expected a multiply node.");
//  v - (v*a) -> (1 - a)*v
    auto factor4 = var_b - (var_b*var_a);
    assert(graph::multiply_cast(factor4).get() &&
           "Expected a multiply node.");

//  -1*a - b -> -1*(a + b)
    auto neg_vara_minus_varb = (-var_a) - var_b;
    assert(graph::multiply_cast(neg_vara_minus_varb).get() &&
           "Expected a multiply node.");

//  (a/(b*c) - d/(e*c)) -> (a/b + d/e)/c
    auto muliply_divide_factor = var_a/(var_b*var_c) - var_d/(var_e*var_c);
    auto muliply_divide_factor_cast = divide_cast(muliply_divide_factor);
    assert(muliply_divide_factor_cast.get() && "Expected divide node.");
    assert(muliply_divide_factor_cast->get_right()->is_match(var_c) &&
           "Expected var_c to be factored out.");
//  (a/(b*c) - d/(c*e)) -> (a/b - d/e)/c
    auto muliply_divide_factor2 = var_a/(var_b*var_c) - var_d/(var_c*var_e);
    auto muliply_divide_factor_cast2 = divide_cast(muliply_divide_factor2);
    assert(muliply_divide_factor_cast2.get() && "Expected divide node.");
    assert(muliply_divide_factor_cast2->get_right()->is_match(var_c) &&
           "Expected var_c to be factored out.");
//  (a/(c*b) - d/(e*c)) -> (a/b - d/e)/c
    auto muliply_divide_factor3 = var_a/(var_c*var_b) - var_d/(var_e*var_c);
    auto muliply_divide_factor_cast3 = divide_cast(muliply_divide_factor3);
    assert(muliply_divide_factor_cast3.get() && "Expected divide node.");
    assert(muliply_divide_factor_cast3->get_right()->is_match(var_c) &&
           "Expected var_c to be factored out.");
//  (a/(c*b) - d/(c*e)) -> (a/b - d/e)/c
    auto muliply_divide_factor4 = var_a/(var_c*var_b) - var_d/(var_c*var_e);
    auto muliply_divide_factor_cast4 = divide_cast(muliply_divide_factor4);
    assert(muliply_divide_factor_cast4.get() && "Expected divide node.");
    assert(muliply_divide_factor_cast4->get_right()->is_match(var_c) &&
           "Expected var_c to be factored out.");

//  Test common denominators.
//  a/b - c/(b*d) -> (a*d - c)/(b*d)
    auto common_denom1 = var_a/var_b - var_c/(var_b*var_d);
    auto common_denom1_cast = graph::divide_cast(common_denom1);
    assert(common_denom1_cast.get() && "Expected a divide node.");
    assert(common_denom1_cast->get_right()->is_match(var_b*var_d) &&
           "Expected var_b*var_d as common denominator.");
    assert(common_denom1_cast->get_left()->is_match(var_a*var_d - var_c) &&
           "Expected a*d - c as numerator.");
//  a/b - c/(d*b) -> (a*d - c)/(d*b)
    auto common_denom2 = var_a/var_b - var_c/(var_d*var_b);
    auto common_denom2_cast = graph::divide_cast(common_denom2);
    assert(common_denom2_cast.get() && "Expected a divide node.");
    assert(common_denom2_cast->get_right()->is_match(var_d*var_b) &&
           "Expected var_b*var_d as common denominator.");
    assert(common_denom2_cast->get_left()->is_match(var_a*var_d - var_c) &&
           "Expected a*d - c as numerator.");
//  a/(b*d) - c/b -> (a - c*d)/(b*d)
    auto common_denom3 = var_a/(var_b*var_d) - var_c/var_b;
    auto common_denom3_cast = graph::divide_cast(common_denom3);
    assert(common_denom3_cast.get() && "Expected a divide node.");
    assert(common_denom3_cast->get_right()->is_match(var_b*var_d) &&
           "Expected var_b*var_d as common denominator.");
    assert(common_denom3_cast->get_left()->is_match(var_a - var_c*var_d) &&
           "Expected a - c*d as numerator.");
//  a/(d*b) - c/b -> (a - c*d)/(d*b)
    auto common_denom4 = var_a/(var_d*var_b) - var_c/var_b;
    auto common_denom4_cast = graph::divide_cast(common_denom4);
    assert(common_denom4_cast.get() && "Expected a divide node.");
    assert(common_denom4_cast->get_right()->is_match(var_d*var_b) &&
           "Expected var_b*var_d as common denominator.");
    assert(common_denom4_cast->get_left()->is_match(var_a - var_c*var_d) &&
           "Expected a - c*d as numerator.");

//  -a/b - d -> -(a/b + d)
    auto common_neg = -var_a/var_b - var_c;
    auto common_neg_cast = graph::multiply_cast(common_neg);
    assert(common_neg_cast.get() && "Expected a multiply node.");
    assert(common_neg->is_match(-(var_a/var_b + var_c)) &&
           "Expected -(a/b + d)");

//  a*b/c - d*b/e -> (a/c - d/e)*b
    auto factor5 = var_a*var_b/var_c - var_d*var_b/var_e;
    auto factor5_cast = graph::multiply_cast(factor5);
    assert(factor5_cast.get() && "Expected a multiply node.");
    assert(factor5->is_match((var_a/var_c - var_d/var_e)*var_b) &&
           "Expected (a/c - d/e)*b.");
//  a*b/c - b*d/e -> (a/c - d/e)*b
    auto factor6 = var_a*var_b/var_c - var_b*var_d/var_e;
    auto factor6_cast = graph::multiply_cast(factor6);
    assert(factor6_cast.get() && "Expected a multiply node.");
    assert(factor6->is_match((var_a/var_c - var_d/var_e)*var_b) &&
           "Expected (a/c - d/e)*b.");
//  b*a/c - d*b/e -> (a/c - d/e)*b
    auto factor7 = var_b*var_a/var_c - var_d*var_b/var_e;
    auto factor7_cast = graph::multiply_cast(factor7);
    assert(factor7_cast.get() && "Expected a multiply node.");
    assert(factor7->is_match((var_a/var_c - var_d/var_e)*var_b) &&
           "Expected (a/c - d/e)*b.");
//  b*a/c - b*d/e -> (a/c - d/e)*b
    auto factor8 = var_b*var_a/var_c - var_b*var_d/var_e;
    auto factor8_cast = graph::multiply_cast(factor8);
    assert(factor8_cast.get() && "Expected a multiply node.");
    assert(factor8->is_match((var_a/var_c - var_d/var_e)*var_b) &&
           "Expected (a/c - d/e)*b.");
    
//  c1*a/b - c2*a/d = c3*(a/b - c4*a/d)
    auto two = graph::constant(static_cast<T> (2.0));
    auto common_const = two*var_a/var_b - 3.0*var_c/var_d;
    auto common_const_cast = graph::multiply_cast(common_const);
    assert(common_const_cast.get() && "Expected a multiply node.");
    assert(common_const_cast->get_left()->is_match(two) &&
           "Expected a constant of 2.0");
    assert(common_const_cast->get_right()->is_match(var_a/var_b - 3.0/2.0*var_c/var_d) &&
           "Expected a/b - 3/2*c/d");

//  a/b - c*a/d -> (1/b - c/d)*a
    auto common_var = var_a/var_b - var_c*var_a/var_d;
    auto common_var_cast = graph::multiply_cast(common_var);
    assert(common_var_cast.get() && "Expected a multiply node.");
    assert(common_var_cast->get_right()->is_match(var_a) &&
           "Expected var_a");
    assert(common_var_cast->get_left()->is_match(1.0/var_b - var_c/var_d) &&
           "Expected 1/b - c/d");
//  a/b - a*c/d -> (1/b - c/d)*a
    auto common_var2 = var_a/var_b - var_a*var_c/var_d;
    auto common_var2_cast = graph::multiply_cast(common_var2);
    assert(common_var2_cast.get() && "Expected a multiply node.");
    assert(common_var2_cast->get_right()->is_match(var_a) &&
           "Expected var_a");
    assert(common_var2_cast->get_left()->is_match(1.0/var_b - var_c/var_d) &&
           "Expected 1/b - c/d");
//  c*a/b - a/d -> (c/b - 1/d)*a
    auto common_var3 = var_c*var_a/var_b - var_a/var_d;
    auto common_var3_cast = graph::multiply_cast(common_var3);
    assert(common_var3_cast.get() && "Expected a multiply node.");
    assert(common_var3_cast->get_right()->is_match(var_a) &&
           "Expected var_a");
    assert(common_var3_cast->get_left()->is_match(var_c/var_b - 1.0/var_d) &&
           "Expected c/b - 1/d");
//  a*c/b - a/d -> (c/b - 1/d)*a
    auto common_var4 = var_a*var_c/var_b - var_a/var_d;
    auto common_var4_cast = graph::multiply_cast(common_var4);
    assert(common_var4_cast.get() && "Expected a multiply node.");
    assert(common_var4_cast->get_right()->is_match(var_a) &&
           "Expected var_a");
    assert(common_var4_cast->get_left()->is_match(var_c/var_b - 1.0/var_d) &&
           "Expected c/b - 1/d");

    auto common_var5 = 2.0*var_c/var_a - 3.0*var_c/var_b;
    auto common_var5_cast = graph::multiply_cast(common_var5);
    assert(common_var5_cast.get() && "Expected a multiply node.");
    assert(common_var5_cast->get_right()->is_match(var_c) &&
           "Expected var_a");
    assert(common_var5_cast->get_left()->is_match(2.0/var_a - 3.0/var_b) &&
           "Expected 2/a - 3/b");

    auto constant_combine = (1.0 - var_a) - 2.0;
    auto constant_combine_cast = graph::subtract_cast(constant_combine);
    assert(constant_combine_cast.get() && "Expected a subtract node.");
    assert(constant_combine_cast->get_left()->evaluate().at(0) == static_cast<T> (-1.0) &&
           "Expected -1 on the left.");
    assert(constant_combine_cast->get_right()->is_match(var_a) &&
           "Expected a on the right.");
    auto constant_combine2 = (1.0 + var_a) - 2.0;
    auto constant_combine_cast2 = graph::add_cast(constant_combine2);
    assert(constant_combine_cast2.get() && "Expected a add node.");
    assert(constant_combine_cast2->get_left()->evaluate().at(0) == static_cast<T> (-1.0) &&
           "Expected -1 on the left.");
    assert(constant_combine_cast2->get_right()->is_match(var_a) &&
           "Expected a on the right.");
    auto constant_combine3 = (var_a - 1.0) - 2.0;
    auto constant_combine3_cast = graph::subtract_cast(constant_combine3);
    assert(constant_combine3_cast.get() && "Expected a subtract node.");
    assert(constant_combine3_cast->get_left()->evaluate().at(0) == static_cast<T> (-3.0) &&
           "Expected -1 on the left.");
    assert(constant_combine3_cast->get_right()->is_match(var_a) &&
           "Expected a on the right.");
    auto constant_combine4 = 2.0 - (1.0 - var_a);
    auto constant_combine4_cast = graph::subtract_cast(constant_combine4);
    assert(constant_combine4_cast.get() && "Expected a subtract node.");
    assert(constant_combine4_cast->get_left()->evaluate().at(0) == static_cast<T> (1.0) &&
           "Expected 1 on the left.");
    assert(constant_combine4_cast->get_right()->is_match(var_a) &&
           "Expected a on the right.");
    auto constant_combine5 = 2.0 - (1.0 + var_a);
    auto constant_combine5_cast = graph::add_cast(constant_combine5);
    assert(constant_combine5_cast.get() && "Expected an add node.");
    assert(constant_combine5_cast->get_left()->evaluate().at(0) == static_cast<T> (1.0) &&
           "Expected 1 on the left.");
    assert(constant_combine5_cast->get_right()->is_match(var_a) &&
           "Expected a on the right.");
    auto constant_combine6 = 2.0 - (var_a - 1.0);
    auto constant_combine6_cast = graph::subtract_cast(constant_combine6);
    assert(constant_combine6_cast.get() && "Expected a subtract node.");
    assert(constant_combine6_cast->get_left()->evaluate().at(0) == static_cast<T> (3.0) &&
           "Expected 3 on the left.");
    assert(constant_combine6_cast->get_right()->is_match(var_a) &&
           "Expected a on the right.");
}

//------------------------------------------------------------------------------
///  @brief Tests for multiply nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_multiply() {
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
    auto two = graph::constant(static_cast<T> (2.0));
    auto three = graph::constant(static_cast<T> (3.0));
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
    auto two_times_var = 2.0*variable;
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
    auto swap = multiply_cast(variable*2.0);
    assert(graph::constant_cast(swap->get_left()).get() &&
           "Expected a constant on he left");
    assert(graph::variable_cast(swap->get_right()).get() &&
           "Expected a variable on he right");
    
//  Test reduction of common constants c1*x*c2*y = c3*x*y.
    auto x1 = 2.0*graph::variable<T> (1, "");
    auto x2 = 5.0*graph::variable<T> (1, "");
    auto x3 = x1*x2;
    auto x3_cast = graph::multiply_cast(x3);
    assert(x3_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(x3_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::multiply_cast(x3_cast->get_right()).get() &&
           "Expected a multipy node.");

//  Test reduction of common constants x*c1*c2*y = c3*x*y.
    auto x4 = graph::variable<T> (1, "")*2.0;
    auto x5 = 5.0*graph::variable<T> (1, "");
    auto x6 = x4*x5;
    auto x6_cast = graph::multiply_cast(x6);
    assert(x6_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(x6_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::multiply_cast(x6_cast->get_right()).get() &&
           "Expected multipy node.");

//  Test reduction of common constants c1*x*y*c2 = c3*x*y.
    auto x7 = 2.0*graph::variable<T> (1, "");
    auto x8 = graph::variable<T> (1, "")*5.0;
    auto x9 = x7*x8;
    auto x9_cast = graph::multiply_cast(x9);
    assert(x9_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(x9_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::multiply_cast(x9_cast->get_right()).get() &&
           "Expected multipy node.");

//  Test reduction of common constants x*c1*y*c2 = c3*x*y.
    auto x10 = graph::variable<T> (1, "")*2.0;
    auto x11 = 5.0*graph::variable<T> (1, "");
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
    auto a = graph::variable<T> (1, "a");
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
    auto c3 = 2.0*(3.0*a);
    auto c3_cast = graph::multiply_cast(c3);
    assert(c3_cast.get() && "Expected multiply node.");
    assert(graph::constant_cast(c3_cast->get_left()) &&
           "Expected constant on the left.");
    assert(graph::variable_cast(c3_cast->get_right()) &&
           "Expected variable on the right.");

//  Test (c1*v)*c2 -> c4*v
    auto c4 = (3.0*a)*2.0;
    auto c4_cast = graph::multiply_cast(c4);
    assert(c4_cast.get() && "Expected multiply node.");
    assert(graph::constant_cast(c4_cast->get_left()) &&
           "Expected constant on the left.");
    assert(graph::variable_cast(c4_cast->get_right()) &&
           "Expected variable on the right.");

//  Test c1*(c2/v) -> c5/v
    auto c5 = 2.0*(3.0/a);
    auto c5_cast = graph::divide_cast(c5);
    assert(c5_cast.get() && "Expected a divide node.");
    assert(graph::constant_cast(c5_cast->get_left()).get() &&
           "Expected constant in the numerator.");
    assert(graph::variable_cast(c5_cast->get_right()).get() &&
           "Expected variable in the denominator.");

//  Test c1*(v/c2) -> c6*v
    auto c6 = 2.0*(a/3.0);
    auto c6_cast = graph::multiply_cast(c6);
    assert(c6_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(c6_cast->get_left()).get() &&
           "Expected constant for the left.");
    assert(graph::variable_cast(c6_cast->get_right()).get() &&
           "Expected variable for the right.");

//  Test (c2/v)*c1 -> c7/v
    auto c7 = (3.0/a)*2.0;
    auto c7_cast = graph::divide_cast(c7);
    assert(c7_cast.get() && "Expected a divide node.");
    assert(graph::constant_cast(c7_cast->get_left()).get() &&
           "Expected constant for the numerator.");
    assert(graph::variable_cast(c7_cast->get_right()).get() &&
           "Expected variable for the denominator.");

//  Test c1*(v/c2) -> c8*v
    auto c8 = 2.0*(a/3.0);
    auto c8_cast = graph::multiply_cast(c8);
    assert(c8_cast.get() && "Expected divide node.");
    assert(graph::constant_cast(c8_cast->get_left()).get() &&
           "Expected constant for the left.");
    assert(graph::variable_cast(c8_cast->get_right()).get() &&
           "Expected variable for the right.");

//  Test v1*(c*v2) -> c*(v1*v2)
    auto c9 = a*(3.0*variable);
    auto c9_cast = graph::multiply_cast(c9);
    assert(c9_cast.get() && "Expected multiply node.");
    assert(graph::constant_cast(c9_cast->get_left()).get() &&
           "Expected a constant node first.");

//  Test v1*(v2*c) -> c*(v1*v2)
    auto c10 = a*(variable*3.0);
    auto c10_cast = graph::multiply_cast(c10);
    assert(c10_cast.get() && "Expected multiply node.");
    assert(graph::constant_cast(c10_cast->get_left()).get() &&
           "Expected a constant node first.");

//  Test (c*v1)*v2) -> c*(v1*v2)
    auto c11 = (3.0*variable)*a;
    auto c11_cast = graph::multiply_cast(c11);
    assert(c11_cast.get() && "Expected multiply node.");
    assert(graph::constant_cast(c11_cast->get_left()).get() &&
           "Expected a constant node first.");

//  Test (v1*c)*v2 -> c*(v1*v2)
    auto c12 = (variable*3.0)*a;
    auto c12_cast = graph::multiply_cast(c12);
    assert(c12_cast.get() && "Expected multiply node.");
    assert(graph::constant_cast(c12_cast->get_left()).get() &&
           "Expected constant node first.");

//  Test (c/v1)*v2 -> (c*v2)/v1
    auto c13 = (3.0/variable)*a;
    auto c13_cast = graph::divide_cast(c13);
    assert(c13_cast.get() && "Expected divide node.");
    assert(c13_cast->get_left()->is_match(3.0*a) &&
           "Expected 3*a in the numerator.");
    assert(c13_cast->get_right()->is_match(variable) &&
           "Expected variable in the denominator.");

//  Test a^b*a^c -> a^(b + c) -> a^d
    auto pow_bc = graph::pow(a, 2.0)*graph::pow(a, 3.0);
    auto pow_bc_cast = graph::pow_cast(pow_bc);
    assert(pow_bc_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_bc_cast->get_right()).get() &&
           "Expected constant exponent.");

//  Test a*a^c -> a^(1 + c) -> a^c2
    auto pow_c = a*graph::pow(a, 3.0);
    auto pow_c_cast = graph::pow_cast(pow_c);
    assert(pow_c_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_c_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_c_cast->get_right())->is(4) &&
           "Expected constant exponent equal to 4.");

//  Test a^b*a -> a^(b + 1) -> a^b2
    auto pow_b = graph::pow(a, 2.0)*a;
    auto pow_b_cast = graph::pow_cast(pow_b);
    assert(pow_b_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_b_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_b_cast->get_right())->is(3) &&
           "Expected constant exponent equal to 3.");

//  Test a^b*sqrt(a) -> a^(b + 0.5) -> a^b2
    auto pow_sqb = graph::pow(a, 2.0)*graph::sqrt(a);
    auto pow_sqb_cast = graph::pow_cast(pow_sqb);
    assert(pow_sqb_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_sqb_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_sqb_cast->get_right())->is(2.5) &&
           "Expected constant exponent equal to 2.5.");

//  Test sqrt(a)*a^c -> a^(0.5 + c) -> a^c2
    auto pow_sqc = graph::sqrt(a)*graph::pow(a, 3.0);
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
    auto test_var_move = [](graph::shared_leaf<T> x) {
        auto var_move = (2.0*x)*x;
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
    auto common_base = ((2.0 + varvec_a)*graph::pow(variable, 3.0))*graph::pow(variable, 2.0);
    auto common_base_cast = graph::multiply_cast(common_base);
    assert(common_base_cast.get() && "Expected multiply node.");
    assert(graph::add_cast(common_base_cast->get_left()).get() &&
           "Expected add cast on the left.");
    assert(graph::pow_cast(common_base_cast->get_right()).get() &&
           "Expected power cast on the right.");
//  (v^a*(c + d))*v^b -> (c + d)*v^(a + b)
    auto common_base2 = (graph::pow(variable, 3.0)*(2.0 + varvec_a))
                      * graph::pow(variable, 2.0);
    auto common_base_cast2 = graph::multiply_cast(common_base2);
    assert(common_base_cast2.get() && "Expected multiply node.");
    assert(graph::add_cast(common_base_cast2->get_left()).get() &&
           "Expected add cast on the left.");
    assert(graph::pow_cast(common_base_cast2->get_right()).get() &&
           "Expected power cast on the right.");
//  v^b*((c + d)*v^a) -> (c + d)*v^(a + b)
    auto common_base3 = graph::pow(variable, 2.0)
                      *((2.0 + varvec_a)*graph::pow(variable, 3.0));
    auto common_base_cast3 = graph::multiply_cast(common_base3);
    assert(common_base_cast3.get() && "Expected multiply node.");
    assert(graph::add_cast(common_base_cast3->get_left()).get() &&
           "Expected add cast on the left.");
    assert(graph::pow_cast(common_base_cast3->get_right()).get() &&
           "Expected power cast on the right.");
//  v^b*(v^a*(c + d)) -> (c + d)*v^(a + b)
    auto common_base4 = graph::pow(variable, 2.0)
                      * (graph::pow(variable, 3.0)*(2.0 + varvec_a));
    auto common_base_cast4 = graph::multiply_cast(common_base4);
    assert(common_base_cast4.get() && "Expected multiply node.");
    assert(graph::add_cast(common_base_cast4->get_left()).get() &&
           "Expected add cast on the left.");
    assert(graph::pow_cast(common_base_cast4->get_right()).get() &&
           "Expected power cast on the right.");

//  (a/b)^c*b^d -> a^c*b^(c-d)
    auto divide_pow = graph::pow(v2/v1, 2.0)*graph::pow(v1, 2.0);
    assert(divide_pow->is_power_base_match(v2) && "Expected the v2 variable.");
//  (b/a)^c*b^d -> b^(c+d)/a^c
    auto divide_pow2 = graph::pow(v1/v2, 2.0)*graph::pow(v1, 2.0);
    assert(graph::divide_cast(divide_pow2).get() &&
           "Expected a divide node.");
//  b^d*(a/b)^c -> a^c*b^(c-d)
    auto divide_pow3 = graph::pow(v1, 2.0)*graph::pow(v2/v1, 2.0);
    assert(divide_pow3->is_power_base_match(v2) && "Expected the v2 variable.");
//  b^d*(b/a)^c -> b^(c+d)/a^c
    auto divide_pow4 = graph::pow(v1, 2.0)*graph::pow(v1/v2, 2.0);
    assert(graph::divide_cast(divide_pow4).get() &&
           "Expected a divide node.");

//  Test node properties.
    assert(two_times_three->is_constant() && "Expected a constant.");
    assert(!two_times_three->is_all_variables() && "Did not expect a variable.");
    assert(two_times_three->is_power_like() && "Expected a power like.");
    auto constant_mul = three*graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (1.0),
                                                                       static_cast<T> (2.0)}), variable);
    assert(constant_mul->is_constant() && "Expected a constant.");
    assert(!constant_mul->is_all_variables() && "Did not expect a variable.");
    assert(constant_mul->is_power_like() && "Expected a power like.");
    auto constant_var_mul = three*variable;
    assert(!constant_var_mul->is_constant() && "Did not expect a constant.");
    assert(!constant_var_mul->is_all_variables() && "Did not expect a variable.");
    assert(!constant_var_mul->is_power_like() && "Did not expect a power like.");
    auto var_var_mul = variable*a;
    assert(!var_var_mul->is_constant() && "Did not expect a constant.");
    assert(var_var_mul->is_all_variables() && "Expected a variable.");
    assert(!var_var_mul->is_power_like() && "Did not expect a power like.");

//  exp(-v)*exp(v) -> 1
    auto expnexp = graph::exp(-variable)*graph::exp(variable);
    auto expnexp_cast = constant_cast(expnexp);
    assert(expnexp_cast.get() && "Expected a constant.");
    assert(expnexp_cast->is(1) && "Expected one.");
//  exp(v)*exp(-v) -> 1
    auto expnexp2 = graph::exp(variable)*graph::exp(-variable);
    auto expnexp2_cast = constant_cast(expnexp2);
    assert(expnexp2_cast.get() && "Expected a constant.");
    assert(expnexp2_cast->is(1) && "Expected one.");
//  exp(-cv)*exp(cv) -> 1
    auto expnexp3 = graph::exp(-3.0*variable)*graph::exp(3.0*variable);
    auto expnexp3_cast = constant_cast(expnexp);
    assert(expnexp3_cast.get() && "Expected a constant.");
    assert(expnexp3_cast->is(1) && "Expected one.");
//  exp(cv)*exp(-cv) -> 1
    auto expnexp4 = graph::exp(3.0*variable)*graph::exp(-3.0*variable);
    auto expnexp4_cast = constant_cast(expnexp2);
    assert(expnexp4_cast.get() && "Expected a constant.");
    assert(expnexp4_cast->is(1) && "Expected one.");

//  c*exp(-v)*exp(v) -> c
    auto cexpnexp = (3.0*graph::exp(-variable))*graph::exp(variable);
    auto cexpnexp_cast = constant_cast(cexpnexp);
    assert(cexpnexp_cast.get() && "Expected a constant.");
    assert(cexpnexp_cast->is(3) && "Expected one.");
//  exp(-v)*c*exp(v) -> c
    auto cexpnexp2 = graph::exp(-variable)*(3.0*graph::exp(variable));
    auto cexpnexp2_cast = constant_cast(cexpnexp2);
    assert(cexpnexp2_cast.get() && "Expected a constant.");
    assert(cexpnexp2_cast->is(3) && "Expected one.");
//  c*exp(v)*exp(-v) -> c
    auto cexpnexp3 = (3.0*graph::exp(variable))*graph::exp(-variable);
    auto cexpnexp3_cast = constant_cast(cexpnexp3);
    assert(cexpnexp3_cast.get() && "Expected a constant.");
    assert(cexpnexp3_cast->is(3) && "Expected one.");
//  exp(v)*c*exp(-v) -> c
    auto cexpnexp4 = graph::exp(variable)*(3.0*graph::exp(-variable));
    auto cexpnexp4_cast = constant_cast(cexpnexp4);
    assert(cexpnexp4_cast.get() && "Expected a constant.");
    assert(cexpnexp4_cast->is(3) && "Expected one.");
//  c*exp(-c*v)*exp(c*v) -> c
    auto cexpnexp5 = (3.0*graph::exp(-3.0*variable))*graph::exp(3.0*variable);
    auto cexpnexp5_cast = constant_cast(cexpnexp5);
    assert(cexpnexp5_cast.get() && "Expected a constant.");
    assert(cexpnexp5_cast->is(3) && "Expected one.");
//  exp(-v)*c*exp(v) -> c
    auto cexpnexp6 = graph::exp(-3.0*variable)*(3.0*graph::exp(3.0*variable));
    auto cexpnexp6_cast = constant_cast(cexpnexp6);
    assert(cexpnexp6_cast.get() && "Expected a constant.");
    assert(cexpnexp6_cast->is(3) && "Expected one.");
//  c*exp(v)*exp(-v) -> c
    auto cexpnexp7 = (3.0*graph::exp(3.0*variable))*graph::exp(-3.0*variable);
    auto cexpnexp7_cast = constant_cast(cexpnexp7);
    assert(cexpnexp7_cast.get() && "Expected a constant.");
    assert(cexpnexp7_cast->is(3) && "Expected one.");
//  exp(v)*c*exp(-v) -> c
    auto cexpnexp8 = graph::exp(3.0*variable)*(3.0*graph::exp(-3.0*variable));
    auto cexpnexp8_cast = constant_cast(cexpnexp8);
    assert(cexpnexp8_cast.get() && "Expected a constant.");
    assert(cexpnexp8_cast->is(3) && "Expected one.");

//  exp(-v)*(exp(v)*a) -> a
    auto regroup_exp = graph::exp(-variable)*(graph::exp(variable)*v1);
    assert(regroup_exp->is_match(v1) && "Expected the v1 variable node.");
//  exp(-v)*(a*exp(v)) -> a
    auto regroup_exp2 = graph::exp(-variable)*(v1*graph::exp(variable));
    assert(regroup_exp2->is_match(v1) && "Expected the v1 variable node.");
//  (exp(-v)*a)*exp(v) -> a
    auto regroup_exp3 = (graph::exp(-variable)*v1)*graph::exp(variable);
    assert(regroup_exp3->is_match(v1) && "Expected the v1 variable node.");
//  (a*exp(-v))*exp(v) -> a
    auto regroup_exp4 = (graph::exp(-variable)*v1)*graph::exp(variable);
    assert(regroup_exp4->is_match(v1) && "Expected the v1 variable node.");

//  exp(a)*exp(b) -> exp(a + b)
    auto exp_a = graph::exp(2.0*v1);
    auto exp_b = graph::exp(2.0*v2);
    auto exp_mul = exp_a*exp_b;
    assert(graph::exp_cast(exp_mul) &&
           "Expected a exp node.");

//  exp(a)*(exp(b)*c) -> c*exp(a + b)
    auto expression_c = 2.0 + variable;
    auto exp_mul2 = exp_a*(exp_b*expression_c);
    auto exp_mul2_cast = graph::multiply_cast(exp_mul2);
    assert(exp_mul2_cast.get() && "Expected a multiply node.");
    assert(graph::exp_cast(exp_mul2_cast->get_right()) &&
           "Expected a exp node on the right.");
//  exp(a)*(c*exp(b)) -> c*exp(a + b)
    auto exp_mul3 = exp_a*(expression_c*exp_b);
    auto exp_mul3_cast = graph::multiply_cast(exp_mul3);
    assert(exp_mul3_cast.get() && "Expected a multiply node.");
    assert(graph::exp_cast(exp_mul3_cast->get_right()) &&
           "Expected a exp node on the right.");

//  (exp(a)*c)*exp(b) -> c*exp(a + b)
    auto exp_mul4 = (exp_a*expression_c)*exp_b;
    auto exp_mul4_cast = graph::multiply_cast(exp_mul4);
    assert(exp_mul4_cast.get() && "Expected a multiply node.");
    assert(graph::exp_cast(exp_mul4_cast->get_right()) &&
           "Expected a exp node on the right.");
//  (c*exp(a))*exp(b) -> c*exp(a + b)
    auto exp_mul5 = (expression_c*exp_a)*exp_b;
    auto exp_mul5_cast = graph::multiply_cast(exp_mul5);
    assert(exp_mul5_cast.get() && "Expected a multiply node.");
    assert(graph::exp_cast(exp_mul5_cast->get_right()) &&
           "Expected a exp node on the right.");

//  (exp(a)*c)*(exp(b)*d) -> (c*d)*exp(a + b)
    auto expression_d = 3.0 + variable;
    auto exp_mul6 = (exp_a*expression_c)*(exp_b*expression_d);
    auto exp_mul6_cast = graph::multiply_cast(exp_mul6);
    assert(exp_mul6_cast.get() && "Expected a multiply node.");
    assert(graph::exp_cast(exp_mul6_cast->get_right()) &&
           "Expected a exp node on the right.");
//  (exp(a)*c)*(d*exp(b)) -> (c*d)*exp(a + b)
    auto exp_mul7 = (exp_a*expression_c)*(expression_d*exp_b);
    auto exp_mul7_cast = graph::multiply_cast(exp_mul7);
    assert(exp_mul7_cast.get() && "Expected a multiply node.");
    assert(graph::exp_cast(exp_mul7_cast->get_right()) &&
           "Expected a exp node on the right.");
//  (c*exp(a))*(exp(b)*d) -> (c*d)*exp(a + b)
    auto exp_mul8 = (expression_c*exp_a)*(exp_b*expression_d);
    auto exp_mul8_cast = graph::multiply_cast(exp_mul8);
    assert(exp_mul8_cast.get() && "Expected a multiply node.");
    assert(graph::exp_cast(exp_mul8_cast->get_right()) &&
           "Expected a exp node on the right.");
//  (c*exp(a))*(d*exp(b)) -> (c*d)*exp(a + b)
    auto exp_mul9 = (expression_c*exp_a)*(expression_d*exp_b);
    auto exp_mul9_cast = graph::multiply_cast(exp_mul9);
    assert(exp_mul9_cast.get() && "Expected a multiply node.");
    assert(graph::exp_cast(exp_mul9_cast->get_right()) &&
           "Expected a exp node on the right.");

//  (c/exp(a))*exp(b) -> c*exp(b - a)
    auto regroup_exp5 = (expression_c/exp_a)*exp_b;
    auto regroup_exp5_cast = graph::multiply_cast(regroup_exp5);
    assert(regroup_exp5_cast.get() && "Expected multiply node.");
    assert(graph::exp_cast(regroup_exp5_cast->get_right()).get() &&
           "Expected a exp node on the right.");
//  (exp(a)/c)*exp(b) -> exp(a + b)/c
    auto regroup_exp6 = (exp_a/expression_c)*exp_b;
    auto regroup_exp6_cast = graph::divide_cast(regroup_exp6);
    assert(regroup_exp6_cast.get() && "Expected divide node.");
    assert(graph::exp_cast(regroup_exp6_cast->get_left()).get() &&
           "Expected a exp node on the left.");
//  exp(a)*(c/exp(a)) -> c*exp(a - b)
    auto regroup_exp7 = exp_a*(expression_c/exp_b);
    auto regroup_exp7_cast = graph::multiply_cast(regroup_exp7);
    assert(regroup_exp7_cast.get() && "Expected multiply node.");
    assert(graph::exp_cast(regroup_exp7_cast->get_right()).get() &&
           "Expected a exp node on the right.");
//  exp(a)*(exp(b)/c) -> exp(a + b)/c
    auto regroup_exp8 = exp_a*(exp_b/expression_c);
    auto regroup_exp8_cast = graph::divide_cast(regroup_exp8);
    assert(regroup_exp8_cast.get() && "Expected divide node.");
    assert(graph::exp_cast(regroup_exp8_cast->get_left()).get() &&
           "Expected a exp node on the left.");

//  (c/exp(a))*(exp(b)*d) -> (c*d)*exp(b - a)
    auto regroup_exp9 = (expression_c/exp_a)*(exp_b*expression_d);
    auto regroup_exp9_cast = graph::multiply_cast(regroup_exp9);
    assert(regroup_exp9_cast.get() && "Expected multiply node.");
    assert(graph::exp_cast(regroup_exp9_cast->get_right()).get() &&
           "Expected a exp node on the right.");
//  (exp(a)/c)*(exp(b)*d) -> (d*exp(a + b))/c
    auto regroup_exp10 = (exp_a/expression_c)*(exp_b*expression_d);
    auto regroup_exp10_cast = graph::divide_cast(regroup_exp10);
    assert(regroup_exp10_cast.get() && "Expected divide node.");
    assert(regroup_exp10_cast->get_right()->is_match(expression_c));
    assert(regroup_exp10_cast->get_left()->is_match(expression_d*exp_a*exp_b));
//  (c/exp(a))*(d*exp(b)) -> (c*d)*exp(b - a)
    auto regroup_exp11 = (expression_c/exp_a)*(expression_d*exp_b);
    auto regroup_exp11_cast = graph::multiply_cast(regroup_exp11);
    assert(regroup_exp11_cast.get() && "Expected multiply node.");
    assert(graph::exp_cast(regroup_exp11_cast->get_right()).get() &&
           "Expected a exp node on the right.");
//  (exp(a)/c)*(d*exp(b)) -> (d*exp(a + b)/c
    auto regroup_exp12 = (exp_a/expression_c)*(expression_d*exp_b);
    auto regroup_exp12_cast = graph::divide_cast(regroup_exp10);
    assert(regroup_exp12_cast.get() && "Expected divide node.");
    assert(regroup_exp12_cast->get_right()->is_match(expression_c));
    assert(regroup_exp12_cast->get_left()->is_match(expression_d*exp_a*exp_b));

//  (c*exp(a))*(exp(b)/d) -> (c*exp(a + b))/d
    auto regroup_exp13 = (expression_c*exp_a)*(exp_b/expression_d);
    auto regroup_exp13_cast = graph::divide_cast(regroup_exp13);
    assert(regroup_exp13_cast.get() && "Expected divide node.");
    assert(regroup_exp13_cast->get_right()->is_match(expression_d) &&
           "Expected expression d in the denominator.");
    assert(regroup_exp13_cast->get_left()->is_match(expression_c*exp_a*exp_b) &&
           "Expected c*exp(a + b) in the numerator.");
//  (c*exp(a))*(d/exp(b)) -> (c*e)*exp(a - b)
    auto regroup_exp14 = (expression_c*exp_a)*(expression_d/exp_b);
    auto regroup_exp14_cast = graph::multiply_cast(regroup_exp14);
    assert(regroup_exp14_cast.get() && "Expected multiply node.");
    assert(graph::exp_cast(regroup_exp14_cast->get_right()).get() &&
           "Expected a exp node on the right.");
//  (exp(a)*c)*(d/exp(b)) -> (c*d)*exp(a - b)
    auto regroup_exp15 = (exp_a*expression_c)*(expression_d/exp_b);
    auto regroup_exp15_cast = graph::multiply_cast(regroup_exp15);
    assert(regroup_exp15_cast.get() && "Expected multiply node.");
    assert(graph::exp_cast(regroup_exp15_cast->get_right()).get() &&
           "Expected a exp node on the right.");
//  (exp(a)*c)*(exp(b)/d) -> (c*exp(a + b))/d
    auto regroup_exp16 = (exp_a*expression_c)*(exp_b/expression_d);
    auto regroup_exp16_cast = graph::divide_cast(regroup_exp16);
    assert(regroup_exp16_cast.get() && "Expected divide node.");
    assert(regroup_exp16_cast->get_right()->is_match(expression_d) &&
           "Expected expression d in the denominator.");
    assert(regroup_exp16_cast->get_left()->is_match(expression_c*exp_a*exp_b) &&
           "Expected c*exp(a + b) in the numerator.");

//  (c/exp(a))*(exp(b)/d) -> (c*exp(b - a))/d
    auto exp_mul10 = (expression_c/exp_a)*(exp_b/expression_d);
    auto exp_mul10_cast = graph::divide_cast(exp_mul10);
    assert(exp_mul10_cast.get() && "Expected divide node.");
    assert(exp_mul10_cast->get_right()->is_match(expression_d) &&
           "Expected expression d in the denominator.");
    assert(exp_mul10_cast->get_left()->is_match(expression_c*exp_b/exp_a) &&
           "Expected c*exp(b - a) in the numerator.");
//  (c/exp(a))*(d/exp(b)) -> (c*e)/exp(a + b)
    auto exp_mul11 = (expression_c/exp_a)*(expression_d/exp_b);
    auto exp_mul11_cast = graph::divide_cast(exp_mul11);
    assert(exp_mul11_cast.get() && "Expected divide node.");
    assert(graph::exp_cast(exp_mul11_cast->get_right()).get() &&
           "Expected a exp node on the right.");
//  (exp(a)/c)*(d/exp(b)) -> (d*exp(a - b))/c
    auto exp_mul12 = (exp_a/expression_c)*(expression_d/exp_b);
    auto exp_mul12_cast = graph::divide_cast(exp_mul12);
    assert(exp_mul12_cast.get() && "Expected divide node.");
    assert(exp_mul12_cast->get_right()->is_match(expression_c) &&
           "Expected expression c in the denominator.");
    assert(exp_mul12_cast->get_left()->is_match(expression_d*exp_a/exp_b) &&
           "Expected d*exp(b - a) in the numerator.");
//  (exp(a)/c)*(exp(b)/d) -> exp(a + b)/(c*d)
    auto exp_mul13 = (exp_a/expression_c)*(exp_b/expression_d);
    auto exp_mul13_cast = graph::divide_cast(exp_mul13);
    assert(exp_mul13_cast.get() && "Expected divide node.");
    assert(graph::exp_cast(exp_mul13_cast->get_left()).get() &&
           "Expected a exp node on the left.");

//  cos(v)*a -> a*cos(v)
    auto cosine = graph::cos(variable);
    auto sine = graph::sin(variable);
    auto move_cos1 = cosine*(1.0 + variable);
    auto move_cos1_cast = graph::multiply_cast(move_cos1);
    assert(move_cos1_cast.get() &&
           "Expected a multiply node.");
    assert(graph::cos_cast(move_cos1_cast->get_right()) &&
           "Expected a cosine node on the right.");
//  cos(v)*v -> cos(v)*v
    auto move_cos2 = cosine*variable;
    auto move_cos2_cast = graph::multiply_cast(move_cos2);
    assert(move_cos2_cast.get() &&
           "Expected a multiply node.");
    assert(graph::cos_cast(move_cos2_cast->get_left()) &&
           "Expected a cosine node on the left.");
//  sin(v)*a -> a*sin(v)
    auto move_sin1 = sine*(1.0 + variable);
    auto move_sin1_cast = graph::multiply_cast(move_sin1);
    assert(move_sin1_cast.get() &&
           "Expected a multiply node.");
    assert(graph::sin_cast(move_sin1_cast->get_right()) &&
           "Expected a sine node on the right.");
//  sin(v)*v -> sin(v)*v
    auto move_sin2 = sine*variable;
    auto move_sin2_cast = graph::multiply_cast(move_sin2);
    assert(move_sin2_cast.get() &&
           "Expected a multiply node.");
    assert(graph::sin_cast(move_sin2_cast->get_left()) &&
           "Expected a sine node on the left.");
//  sin(v)*cos(v) -> cos(v)*sin(v)
    auto move_sin3 = sine*cosine;
    auto move_sin3_cast = graph::multiply_cast(move_sin3);
    assert(move_sin3_cast.get() &&
           "Expected a multiply node.");
    assert(graph::sin_cast(move_sin3_cast->get_right()) &&
           "Expected a sine node on the right.");
//  cos(v)*sin(v) -> cos(v)*sin(v)
    auto move_sin4 = cosine*sine;
    auto move_sin4_cast = graph::multiply_cast(move_sin4);
    assert(move_sin4_cast.get() &&
           "Expected a multiply node.");
    assert(graph::sin_cast(move_sin4_cast->get_right()) &&
           "Expected a sine node on the right.");

//  a*(b*sin) -> (a*b)*sin
    auto move_sin5 = (1.0 + variable)*((2.0 + variable)*sine);
    auto move_sin5_cast = graph::multiply_cast(move_sin5);
    assert(move_sin5_cast.get() &&
           "Expected a multiply node.");
    assert(graph::sin_cast(move_sin5_cast->get_right()).get() &&
           "Expected a sine node on the right.");
//  (a*sin)*b -> (a*b)*sin
    auto move_sin6 = ((1.0 + variable)*sine)*(2.0 + variable);
    auto move_sin6_cast = graph::multiply_cast(move_sin6);
    assert(move_sin6_cast.get() &&
           "Expected a multiply node.");
    assert(graph::sin_cast(move_sin6_cast->get_right()).get() &&
           "Expected a sine node on the right.");
//  a*(b*cos) -> (a*b)*cos
    auto move_cos5 = (1.0 + variable)*((2.0 + variable)*cosine);
    auto move_cos5_cast = graph::multiply_cast(move_cos5);
    assert(move_cos5_cast.get() &&
           "Expected a multiply node.");
    assert(graph::cos_cast(move_cos5_cast->get_right()).get() &&
           "Expected a sine node on the right.");
//  (a*cos)*b -> (a*b)*cos
    auto move_cos6 = ((1.0 + variable)*cosine)*(2.0 + variable);
    auto move_cos6_cast = graph::multiply_cast(move_cos6);
    assert(move_cos6_cast.get() &&
           "Expected a multiply node.");
    assert(graph::cos_cast(move_cos6_cast->get_right()).get() &&
           "Expected a sine node on the right.");

//  (a*sin)*cos -> (a*cos)*sin
    auto move_sin7 = ((1.0 + variable)*sine)*cosine;
    auto move_sin7_cast = graph::multiply_cast(move_sin7);
    assert(move_sin7_cast.get() &&
           "Expected a multiply node.");
    assert(graph::sin_cast(move_sin7_cast->get_right()).get() &&
           "Expected a sine node on the right.");
//  (a*cos)*sin -> (a*cos)*sin
    auto move_cos7 = ((1.0 + variable)*sine)*cosine;
    auto move_cos7_cast = graph::multiply_cast(move_cos7);
    assert(move_cos7_cast.get() &&
           "Expected a multiply node.");
    assert(graph::sin_cast(move_sin7_cast->get_right()).get() &&
           "Expected a sine node on the right.");
//  (a*sin)*v -> (a*sin)*v
    auto move_sin8 = ((1.0 + variable)*sine)*variable;
    auto move_sin8_cast = graph::multiply_cast(move_sin8);
    assert(move_sin8_cast.get() &&
           "Expected a multiply node.");
    assert(graph::variable_cast(move_sin8_cast->get_right()).get() &&
           "Expected a variable node on the right.");
//  (a*cos)*v -> (a*cos)*v
    auto move_cos8 = ((1.0 + variable)*cosine)*variable;
    auto move_cos8_cast = graph::multiply_cast(move_cos8);
    assert(move_cos8_cast.get() &&
           "Expected a multiply node.");
    assert(graph::variable_cast(move_cos8_cast->get_right()).get() &&
           "Expected a variable node on the right.");

//  c*(a*sin) -> c*(a*sin)
    auto move_sin9 = 2.0*((1.0 + variable)*sine);
    auto move_sin9_cast = graph::multiply_cast(move_sin9);
    assert(move_sin9_cast.get() &&
           "Expected a multiply node.");
    assert(graph::constant_cast(move_sin9_cast->get_left()).get() &&
           "Expected a constant node on the left.");
//  c*(a*cos) -> c*(a*cos)
    auto move_cos9 = 2.0*((1.0 + variable)*cosine);
    auto move_cos9_cast = graph::multiply_cast(move_cos9);
    assert(move_cos9_cast.get() &&
           "Expected a multiply node.");
    assert(graph::constant_cast(move_cos9_cast->get_left()).get() &&
           "Expected a constant node on the left.");

    auto var_a = (1.0 + graph::variable<T> (1, ""));
    auto var_b = (2.0 + graph::variable<T> (1, ""));
    auto var_c = (3.0 + graph::variable<T> (1, ""));
//  a*(b/c) -> (a*b)/c
    auto todivide1 = var_a*(var_b/var_c);
    assert(graph::divide_cast(todivide1).get() &&
           "Expected a divide node.");
    assert(todivide1->is_match((var_a*var_b)/var_c) &&
           "Expected a (a*b)/c");
//  (a/c)*b -> (a*b)/c
    auto todivide2 = (var_a/var_c)*var_b;
    assert(graph::divide_cast(todivide2).get() &&
           "Expected a divide node.");
    assert(todivide1->is_match((var_a*var_b)/var_c) &&
           "Expected a (a*b)/c");

//  e1*(e2*v) -> (e1*e2)*v
    auto promote_var = var_b*(var_c*a);
    auto promote_var_cast = graph::multiply_cast(promote_var);
    assert(promote_var_cast.get() && "Expected a multiply node.");
    assert(promote_var_cast->get_right()->is_match(a) && "Expected a");
    assert(promote_var_cast->get_left()->is_match(var_b*var_c) &&
           "Expected (2 + b)*(3 + c)");
//  e1*(e2*v^2) -> (e1*e2)*v^2
    auto promote_var2 = var_b*(var_c*(a*a));
    auto promote_var2_cast = graph::multiply_cast(promote_var2);
    assert(promote_var2_cast.get() && "Expected a multiply node.");
    assert(promote_var2_cast->get_right()->is_match(a*a) && "Expected a^2");
    assert(promote_var2_cast->get_left()->is_match(var_b*var_c) &&
           "Expected (2 + b)*(3 + c)");
//  (e1*v)*e2 -> (e1*e2)*v
    auto promote_var3 = (var_b*a)*var_c;
    auto promote_var3_cast = graph::multiply_cast(promote_var3);
    assert(promote_var3_cast.get() && "Expected a multiply node.");
    assert(promote_var3_cast->get_right()->is_match(a) && "Expected a");
    assert(promote_var3_cast->get_left()->is_match(var_b*var_c) &&
           "Expected (2 + b)*(3 + c)");
//  (e1*v^2)*e2 -> (e1*e2)*v^2
    auto promote_var4 = (var_b*(a*a))*var_c;
    auto promote_var4_cast = graph::multiply_cast(promote_var4);
    assert(promote_var4_cast.get() && "Expected a multiply node.");
    assert(promote_var4_cast->get_right()->is_match(a*a) && "Expected a^2");
    assert(promote_var4_cast->get_left()->is_match(var_b*var_c) &&
           "Expected (2 + b)*(3 + c)");

//  (a*b)*a -> a^2*b
    auto gather = (var_a*var_b)*var_a;
    auto gather_cast = graph::multiply_cast(gather);
    assert(gather_cast.get() && "Expected a multiply node.");
    assert(gather_cast->get_right()->is_match(var_b) && "Expected b");
    assert(gather_cast->get_left()->is_match(var_a*var_a) && "Expected a^2");
//  (b*a)*a -> a^2*b
    auto gather2 = (var_b*var_a)*var_a;
    auto gather2_cast = graph::multiply_cast(gather2);
    assert(gather2_cast.get() && "Expected a multiply node.");
    assert(gather2_cast->get_right()->is_match(var_b) && "Expected b");
    assert(gather2_cast->get_left()->is_match(var_a*var_a) && "Expected a^2");

//  (a*(b*c)^2)*c^2 -> a*b^2*c^4
    auto common_pow = (var_a*graph::pow(var_b*var_c, 2.0))*graph::pow(var_c, 2.0);
    auto common_pow_cast = graph::multiply_cast(common_pow);
    assert(common_pow_cast.get() && "Expected a multiply node.");
    assert(common_pow_cast->get_left()->is_match(var_a*graph::pow(var_b,2.0)) &&
           "Expected a*b^2.");
    assert(common_pow_cast->get_right()->is_match(graph::pow(var_c,4.0)) &&
           "Expected c^4.");

//  (b*a)^2*a^2 -> b^2*a^4
    auto common_pow2 = graph::pow(var_b*var_a, 2.0)*graph::pow(var_a, 2.0);
    auto common_pow2_cast = graph::multiply_cast(common_pow2);
    assert(common_pow2_cast.get() && "Expected a multiply node.");
    assert(common_pow2_cast->get_left()->is_match(graph::pow(var_b, 2.0)) &&
           "Expected b^2.");
    assert(common_pow2_cast->get_right()->is_match(graph::pow(var_a, 4.0)) &&
           "Expected a^4.");
//  (a*b)^2*a^2 -> b^2*a^4
    auto common_pow3 = graph::pow(var_a*var_b, 2.0)*graph::pow(var_a, 2.0);
    auto common_pow3_cast = graph::multiply_cast(common_pow3);
    assert(common_pow3_cast.get() && "Expected a multiply node.");
    assert(common_pow3_cast->get_left()->is_match(graph::pow(var_b, 2.0)) &&
           "Expected b^2.");
    assert(common_pow3_cast->get_right()->is_match(graph::pow(var_a, 4.0)) &&
           "Expected a^4.");
//  a^2*(b*a)^2 -> b^2*a^4
    auto common_pow4 = graph::pow(var_a, 2.0)*graph::pow(var_b*var_a, 2.0);
    auto common_pow4_cast = graph::multiply_cast(common_pow4);
    assert(common_pow4_cast.get() && "Expected a multiply node.");
    assert(common_pow4_cast->get_left()->is_match(graph::pow(var_b, 2.0)) &&
           "Expected b^2.");
    assert(common_pow4_cast->get_right()->is_match(graph::pow(var_a, 4.0)) &&
           "Expected a^4.");
//  a^2*(b*a)^2 -> b^2*a^4
    auto common_pow5 = graph::pow(var_a, 2.0)*graph::pow(var_a*var_b, 2.0);
    auto common_pow5_cast = graph::multiply_cast(common_pow5);
    assert(common_pow5_cast.get() && "Expected a multiply node.");
    assert(common_pow5_cast->get_left()->is_match(graph::pow(var_b, 2.0)) &&
           "Expected b^2.");
    assert(common_pow5_cast->get_right()->is_match(graph::pow(var_a, 4.0)) &&
           "Expected a^4.");
}

//------------------------------------------------------------------------------
///  @brief Tests for divide nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_divide() {
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
    auto two = graph::constant<T> (static_cast<T> (2.0));
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

    auto two_divided_var = 2.0/variable;
    assert(graph::divide_cast(two_divided_var).get() &&
           "Expected divide node.");
    variable->set(static_cast<T> (3.0));
    const backend::buffer<T> two_divided_var_result = two_divided_var->evaluate();
    assert(two_divided_var_result.size() == 1 && "Expected single value.");
    assert(two_divided_var_result.at(0) == static_cast<T> (2.0) /
                                           static_cast<T> (3.0) &&
           "Expected 2/3 for result.");

//  v/c1 -> (1/c1)*v -> c2*v
    auto var_divided_two = variable/2.0;
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
    assert((0.0/varvec).get() == zero.get() && "Expected to recover zero.");
    assert((varvec/1.0).get() == varvec.get() &&
           "Expected to recover numerator.");
    assert((zero/varvec)->evaluate()[0] == static_cast<T> (0.0) &&
           "Expected a value of zero.");
    assert((varvec/one)->evaluate()[0] == static_cast<T> (2.0) &&
           "Expected a value of two.");
    assert((varvec/one)->evaluate()[1] == static_cast<T> (6.0) &&
           "Expected a value of six.");

    auto varvec_divided_two = varvec/2.0;
    assert(graph::multiply_cast(varvec_divided_two).get() &&
           "Expect a mutliply node.");
    const backend::buffer<T> varvec_divided_two_result = varvec_divided_two->evaluate();
    assert(varvec_divided_two_result.size() == 2 && "Size mismatch in result.");
    assert(varvec_divided_two_result.at(0) == static_cast<T> (1.0) &&
           "Expected 2/2 for result.");
    assert(varvec_divided_two_result.at(1) == static_cast<T> (3.0) &&
           "Expected 6/2 for result.");

    auto two_divided_varvec = 2.0/varvec;
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
    auto var_sum_prod_divided_two = var_sum_prod/2.0;
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
    auto match = (1.0 + variable)/(1.0 + variable);
    auto match_cast = graph::constant_cast(match);
    assert(match_cast->is(1) &&
           "Expected one constant for result.");

//  Test reduction of common constants (c1*x)/(c2*y) = (c3*x)/y.
    auto var_x = graph::variable<T> (1, "");
    auto var_y = graph::variable<T> (1, "");
    auto x3 = (2.0*var_x)/(5.0*var_y);
    auto x3_cast = graph::divide_cast(x3);
    assert(x3_cast.get() && "Expected a divide node.");
    assert(x3_cast->get_right()->is_match(var_y) &&
           "Expected y in the denominator.");
    assert(x3_cast->get_left()->is_match(2.0/5.0*var_x) &&
           "Expected 2/5*x in the numerator.");

//  Test reduction of common constants (c1*x)/(y*c2) = c3*x/y.
    auto x6 = (2.0*var_x)/(var_y*5.0);
    auto x6_cast = graph::divide_cast(x6);
    assert(x6_cast.get() && "Expected a divide node.");
    assert(x6_cast->get_right()->is_match(var_y) &&
           "Expected y in the denominator.");
    assert(x6_cast->get_left()->is_match(2.0/5.0*var_x) &&
           "Expected 2/5*x in the numerator.");

//  Test reduction of common constants (x*c1)/(c2*y) = c3*x/y.
    auto x9 = (var_x*2.0)/(var_y*5.0);
    auto x9_cast = graph::divide_cast(x9);
    assert(x9_cast.get() && "Expected a divide node.");
    assert(x9_cast->get_right()->is_match(var_y) &&
           "Expected y in the denominator.");
    assert(x9_cast->get_left()->is_match(2.0/5.0*var_x) &&
           "Expected 2/5*x in the numerator.");

//  Test reduction of common constants (x*c1)/(y*c2) = c3*x/y.
    auto x12 = (var_x*2.0)/(var_y*5.0);
    auto x12_cast = graph::divide_cast(x12);
    assert(x12_cast.get() && "Expected a divide node.");
    assert(x12_cast->get_right()->is_match(var_y) &&
           "Expected y in the denominator.");
    assert(x12_cast->get_left()->is_match(2.0/5.0*var_x) &&
           "Expected 2/5*x in the numerator.");

//  c1/(c2*v) -> c3/v
    auto c3 = 2.0/(3.0*variable);
    auto c3_cast = graph::divide_cast(c3);
    assert(c3_cast.get() && "Expected divide node");
    assert(graph::constant_cast(c3_cast->get_left()).get() &&
           "Expected a constant in numerator.");
    assert(graph::variable_cast(c3_cast->get_right()).get() &&
           "Expected a variable in the denominator");

//  c1/(v*c2) -> c4/v
    auto c4 = 2.0/(3.0*variable);
    auto c4_cast = graph::divide_cast(c4);
    assert(c4_cast.get() && "Expected divide node");
    assert(graph::constant_cast(c4_cast->get_left()).get() &&
           "Expected a constant in numerator.");
    assert(graph::variable_cast(c4_cast->get_right()).get() &&
           "Expected a variable in the denominator");

//  (c1*v)/c2 -> c5*v
    auto c5 = (2.0*variable)/3.0;
    auto c5_cast = graph::multiply_cast(c5);
    assert(c5_cast.get() && "Expected a multiply node");
    assert(graph::constant_cast(c5_cast->get_left()).get() &&
           "Expected a constant in the numerator");
    assert(graph::variable_cast(c5_cast->get_right()).get() &&
           "Expected a variable in the denominator.");

//  (v*c1)/c2 -> c5*v
    auto c6 = (variable*2.0)/3.0;
    auto c6_cast = graph::multiply_cast(c6);
    assert(c6_cast.get() && "Expected multiply node");
    assert(graph::constant_cast(c6_cast->get_left()).get() &&
           "Expected a constant in the numerator");
    assert(graph::variable_cast(c6_cast->get_right()).get() &&
           "Expected a variable in the denominator.");

//  (v1*c)/v2 -> (c*v1/v2)
    auto a = graph::variable<T> (1, "");
    auto c8 = (variable*2.0)/a;
    auto c8_cast = graph::divide_cast(c8);
    assert(c8_cast.get() && "Expected divide node");
    auto c8_cast2 = graph::multiply_cast(c8_cast->get_left());
    assert(c8_cast2.get() && "Expected multiply node in numerator.");
    assert(graph::constant_cast(c8_cast2->get_left()).get() &&
           "Expected a constant on the left in the numerator.");

//  (v1*v2)/v1 -> v2
//  (v2*v1)/v1 -> v2
    auto v1 = (variable*a)/variable;
    auto v2 = (a*variable)/variable;
    assert(v1->is_match(a) && "Expected to reduce to a");
    assert(v2->is_match(a) && "Expected to reduce to a");

//  Test a^b/a^c -> a^(b - c)
    auto pow_bc = graph::pow(a, 2.0)/graph::pow(a, 3.0);
    auto pow_bc_cast = graph::pow_cast(pow_bc);
    assert(pow_bc_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_bc_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_bc_cast->get_right())->is(-1) &&
           "Expected negative 1");

//  Test a/a^c -> a^(1 - c)
    auto pow_c = a/graph::pow(a, 3.0);
    auto pow_c_cast = graph::pow_cast(pow_c);
    assert(pow_c_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_c_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_c_cast->get_right())->is(-2) &&
           "Expected constant exponent equal to -2.");

//  Test a^b/a -> a^(b - 1)
    auto pow_b = graph::pow(a, 2.0)/a;
    assert(pow_b->is_match(a) && "Expected to recover a.");

//  Test a^b/sqrt(a) -> a^(b - 0.5)
    auto pow_sqb = graph::pow(a, 2.0)/graph::sqrt(a);
    auto pow_sqb_cast = graph::pow_cast(pow_sqb);
    assert(pow_sqb_cast.get() && "Expected power node.");
    assert(graph::constant_cast(pow_sqb_cast->get_right()).get() &&
           "Expected constant exponent.");
    assert(graph::constant_cast(pow_sqb_cast->get_right())->is(1.5) &&
           "Expected constant exponent equal to 1.5.");

//  Test sqrt(a)/a^c -> a^(0.5 - c)
    auto pow_sqc = graph::sqrt(a)/graph::pow(a, 3.0);
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
    auto test_var_move = [](graph::shared_leaf<T> x,
                            graph::shared_leaf<T> y) {
        auto var_move = (2.0*x)/y;
        auto var_move_cast = graph::multiply_cast(var_move);
        assert(var_move_cast.get() && "Expected multiply.");
        assert(!var_move_cast->get_left()->is_all_variables() &&
               "Expected Non variable like in the left side.");
        assert(var_move_cast->get_right()->is_all_variables() &&
               "Expected variable like in the right side.");
        
        auto var_move2 = (2.0/x)/y;
        auto var_move2_cast = graph::divide_cast(var_move2);
        assert(var_move2_cast.get() && "Expected divide.");
        assert(!var_move2_cast->get_left()->is_all_variables() &&
               "Expected Non variable like in the left side.");
        assert(var_move2_cast->get_right()->is_all_variables() &&
               "Expected variable like in the right side.");
    };

    test_var_move(a, graph::sqrt(a));
    test_var_move(graph::pow(a, 3.0), a);
    test_var_move(graph::pow(a, 3.0), graph::pow(a, 2.0));

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

//  fma(a,b,a)/a -> 1 + b
    auto fma_divide5 = graph::fma(a, graph::variable<T> (1, ""), a)/a;
    auto fma_divide5_cast = graph::add_cast(fma_divide5);
    assert(fma_divide5_cast.get() && "Expected an add node.");
//  fma(b,a,a)/a -> 1 + b
    auto fma_divide6 = graph::fma(graph::variable<T> (1, ""), a, a)/a;
    auto fma_divide6_cast = graph::add_cast(fma_divide6);
    assert(fma_divide6_cast.get() && "Expected an add node.");

//  (a*b^c)/b^d -> a*b^(c - d)
    auto common_power = (variable*graph::pow(a, 3.0))/graph::pow(a, 2.0);
    assert(graph::multiply_cast(common_power).get() &&
           "Expected a multiply node.");
//  (b^c*a)/b^d -> a*b^(c - d)
    auto common_power2 = (graph::pow(a, 3.0)*variable)/graph::pow(a, 2.0);
    assert(graph::multiply_cast(common_power2).get() &&
           "Expected a multiply node.");

//  Test node properties.
    assert(two_divided_three->is_constant() && "Expected a constant.");
    assert(!two_divided_three->is_all_variables() && "Did not expect a variable.");
    assert(two_divided_three->is_power_like() && "Expected a power like.");
    auto constant_div = two_divided_three/graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (1.0),
                                                                                   static_cast<T> (2.0)}), variable);
    assert(constant_div->is_constant() && "Expected a constant.");
    assert(!constant_div->is_all_variables() && "Did not expect a variable.");
    assert(constant_div->is_power_like() && "Expected a power like.");
    auto constant_var_div = two_divided_three/variable;
    assert(!constant_var_div->is_constant() && "Did not expect a constant.");
    assert(!constant_var_div->is_all_variables() && "Did not expect a variable.");
    assert(!constant_var_div->is_power_like() && "Did not expect a power like.");
    auto var_var_div = variable/a;
    assert(!var_var_div->is_constant() && "Did not expect a constant.");
    assert(var_var_div->is_all_variables() && "Expected a variable.");
    assert(!var_var_div->is_power_like() && "Did not expect a power like.");

//  exp(a)/exp(b) -> exp(a - b)
    auto exp_a = graph::exp(2.0*graph::variable<T> (1, ""));
    auto exp_b = graph::exp(3.0*graph::variable<T> (1, ""));
    auto exp_over_exp = exp_a/exp_b;
    auto exp_over_exp_cast = graph::exp_cast(exp_over_exp);
    assert(exp_over_exp_cast.get() &&
           "Expected an exp node.");

//  (c*exp(a))/exp(b) -> c*exp(a - b)
    auto expression_c = (2.0 - variable);
    auto exp_over_exp2 = (expression_c*exp_a)/exp_b;
    auto exp_over_exp2_cast = graph::multiply_cast(exp_over_exp2);
    assert(exp_over_exp2_cast.get() &&
           "Expected a multiply node.");
    assert(graph::exp_cast(exp_over_exp2_cast->get_right()).get() &&
           "Expected a exp node on the right.");
//  (exp(a)*c)/exp(b) -> c*exp(a - b)
    auto exp_over_exp3 = (exp_a*expression_c)/exp_b;
    auto exp_over_exp3_cast = graph::multiply_cast(exp_over_exp3);
    assert(exp_over_exp3_cast.get() &&
           "Expected a multiply node.");
    assert(graph::exp_cast(exp_over_exp3_cast->get_right()).get() &&
           "Expected a exp node on the right.");
//  exp(a)/(c*exp(b)) -> exp(a - b)/c
    auto exp_over_exp4 = exp_a/(expression_c*exp_b);
    auto exp_over_exp4_cast = graph::divide_cast(exp_over_exp4);
    assert(exp_over_exp4_cast.get() &&
           "Expected a divide node.");
    assert(graph::exp_cast(exp_over_exp4_cast->get_left()).get() &&
           "Expected a exp node on the left.");
//  exp(a)/(exp(b)*c) -> exp(a - b)/c
    auto exp_over_exp5 = exp_a/(exp_b*expression_c);
    auto exp_over_exp5_cast = graph::divide_cast(exp_over_exp5);
    assert(exp_over_exp5_cast.get() &&
           "Expected a divide node.");
    assert(graph::exp_cast(exp_over_exp5_cast->get_left()).get() &&
           "Expected a exp node on the left.");

//  (c*exp(a))/(d*exp(b)) -> (c*exp(a - b))/d
    auto expression_d = (3.0 - variable);
    auto exp_over_exp6 = (expression_c*exp_a)/(expression_d*exp_b);
    auto exp_over_exp6_cast = graph::divide_cast(exp_over_exp6);
    assert(exp_over_exp6_cast.get() &&
           "Expected a multiply node.");
    assert(exp_over_exp6_cast->get_right()->is_match(expression_d) &&
           "Expected expression d in the denominator.");
    assert(exp_over_exp6_cast->get_left()->is_match(expression_c*exp_a/exp_b) &&
           "Expected c*exp(a - b) in the numerator.");
//  (c*exp(a))/(exp(b)*d) -> (c*exp(a - b))/d
    auto exp_over_exp7 = (expression_c*exp_a)/(exp_b*expression_d);
    auto exp_over_exp7_cast = graph::divide_cast(exp_over_exp7);
    assert(exp_over_exp7_cast.get() &&
           "Expected a multiply node.");
    assert(exp_over_exp7_cast->get_right()->is_match(expression_d) &&
           "Expected expression d in the denominator.");
    assert(exp_over_exp7_cast->get_left()->is_match(expression_c*exp_a/exp_b) &&
           "Expected c*exp(a - b) in the numerator.");
//  (exp(a)*c)/(d*exp(b)) -> (c*exp(a - b))/d
    auto exp_over_exp8 = (exp_a*expression_c)/(expression_d*exp_b);
    auto exp_over_exp8_cast = graph::divide_cast(exp_over_exp8);
    assert(exp_over_exp8_cast.get() &&
           "Expected a multiply node.");
    assert(exp_over_exp8_cast->get_right()->is_match(expression_d) &&
           "Expected expression d in the denominator.");
    assert(exp_over_exp8_cast->get_left()->is_match(expression_c*exp_a/exp_b) &&
           "Expected c*exp(a - b) in the numerator.");
//  (exp(a)*c)/(exp(b)*d) -> (c*exp(a - b))/d
    auto exp_over_exp9 = (exp_a*expression_c)/(exp_b*expression_d);
    auto exp_over_exp9_cast = graph::divide_cast(exp_over_exp9);
    assert(exp_over_exp9_cast.get() &&
           "Expected a multiply node.");
    assert(exp_over_exp9_cast->get_right()->is_match(expression_d) &&
           "Expected expression d in the denominator.");
    assert(exp_over_exp9_cast->get_left()->is_match(expression_c*exp_a/exp_b) &&
           "Expected c*exp(a - b) in the numerator.");

//  exp(a)/(c/exp(b)) -> exp(a + b)/c
    auto exp_over_exp10 = exp_a/(expression_c/exp_b);
    auto exp_over_exp10_cast = graph::divide_cast(exp_over_exp10);
    assert(exp_over_exp10_cast.get() && "Expected a divide node.");
    assert(graph::exp_cast(exp_over_exp10_cast->get_left()).get() &&
           "Expected a exp node on the left.");
//  exp(a)/(exp(b)/c) -> c*exp(a - b)
    auto exp_over_exp11 = exp_a/(exp_b/expression_c);
    auto exp_over_exp11_cast = graph::multiply_cast(exp_over_exp11);
    assert(exp_over_exp11_cast.get() && "Expected a multiply node.");
    assert(graph::exp_cast(exp_over_exp11_cast->get_right()).get() &&
           "Expected a exp node on the right.");

//  (c/exp(a))/exp(b) -> c/exp(a - b)
    auto exp_over_exp12 = (expression_c/exp_a)/exp_b;
    auto exp_over_exp12_cast = graph::divide_cast(exp_over_exp12);
    assert(exp_over_exp12_cast.get() && "Expected a divide node.");
    assert(graph::exp_cast(exp_over_exp12_cast->get_right()).get() &&
           "Expected a exp node on the right.");
//  (exp(a)/c)/exp(b) -> exp(a - b)/c
    auto exp_over_exp13 = (exp_a/expression_c)/exp_b;
    auto exp_over_exp13_cast = graph::divide_cast(exp_over_exp13);
    assert(exp_over_exp13_cast.get() && "Expected a divide node.");
    assert(graph::exp_cast(exp_over_exp13_cast->get_left()).get() &&
           "Expected a exp node on the left.");

//  ((c*exp(a))*d)/exp(b)
    auto exp_over_exp14 = ((expression_c*exp_a)*expression_d)/exp_b;
    auto exp_over_exp14_cast = graph::multiply_cast(exp_over_exp14);
    assert(exp_over_exp14_cast.get() && "Expected a multiply node.");
    assert(exp_over_exp14_cast->get_right()->is_match(exp_a/exp_b) &&
           "Expected exp(a - b).");
    assert(exp_over_exp14_cast->get_left()->is_match(expression_c*expression_d) &&
           "Expected c*d.");
//  ((exp(a)*c)*d)/exp(b)
    auto exp_over_exp15 = ((exp_a*expression_c)*expression_d)/exp_b;
    auto exp_over_exp15_cast = graph::multiply_cast(exp_over_exp15);
    assert(exp_over_exp15_cast.get() && "Expected a multiply node.");
    assert(exp_over_exp15_cast->get_right()->is_match(exp_a/exp_b) &&
           "Expected exp(a - b).");
    assert(exp_over_exp15_cast->get_left()->is_match(expression_c*expression_d) &&
           "Expected c*d.");
//  (c*(exp(a)*d))/exp(b)
    auto exp_over_exp16 = (expression_c*(exp_a*expression_d))/exp_b;
    auto exp_over_exp16_cast = graph::multiply_cast(exp_over_exp16);
    assert(exp_over_exp16_cast.get() && "Expected a multiply node.");
    assert(exp_over_exp16_cast->get_right()->is_match(exp_a/exp_b) &&
           "Expected exp(a - b).");
    assert(exp_over_exp16_cast->get_left()->is_match(expression_c*expression_d) &&
           "Expected c*d.");
//  (c*(d*exp(a)))/exp(b)
    auto exp_over_exp17 = (expression_c*(expression_d*exp_a))/exp_b;
    auto exp_over_exp17_cast = graph::multiply_cast(exp_over_exp17);
    assert(exp_over_exp17_cast.get() && "Expected a multiply node.");
    assert(exp_over_exp17_cast->get_right()->is_match(exp_a/exp_b) &&
           "Expected exp(a - b).");
    assert(exp_over_exp17_cast->get_left()->is_match(expression_c*expression_d) &&
           "Expected c*d.");

//  a/(b/c + d) -> a*c/(c*d + b)
    auto b = graph::variable<T> (1, "");
    auto c = graph::variable<T> (1, "");
    auto d = graph::variable<T> (1, "");
    auto nest_div1 = a/(b/c + d);
    auto nest_div1_cast = graph::divide_cast(nest_div1);
    assert(nest_div1_cast.get() && "Expected divide node.");
    assert(nest_div1_cast->get_left()->is_match(a*c) &&
           "Expected a*c in the numerator.");
    assert(nest_div1_cast->get_right()->is_match(c*d + b) &&
           "Expected c*d + b in the numerator.");
//  a/(b + b/c) -> a*c/(c*d + b)
    auto nest_div2 = a/(d + b/c);
    auto nest_div2_cast = graph::divide_cast(nest_div2);
    assert(nest_div2_cast.get() && "Expected divide node.");
    assert(nest_div2_cast->get_left()->is_match(a*c) &&
           "Expected a*c in the numerator.");
    assert(nest_div2_cast->get_right()->is_match(c*d + b) &&
           "Expected c*d + b in the numerator.");
//  a/(b/c - d) -> a*c/(b - c*d)
    auto nest_div3 = a/(b/c - d);
    auto nest_div3_cast = graph::divide_cast(nest_div3);
    assert(nest_div3_cast.get() && "Expected divide node.");
    assert(nest_div3_cast->get_left()->is_match(a*c) &&
           "Expected a*c in the numerator.");
    assert(nest_div3_cast->get_right()->is_match(b - c*d) &&
           "Expected b - c*d in the numerator.");
//  a/(d - b/c) -> a*c/(c*d - b)
    auto nest_div4 = a/(d - b/c );
    auto nest_div4_cast = graph::divide_cast(nest_div4);
    assert(nest_div4_cast.get() && "Expected divide node.");
    assert(nest_div4_cast->get_left()->is_match(a*c) &&
           "Expected a*c in the numerator.");
    assert(nest_div4_cast->get_right()->is_match(c*d - b) &&
           "Expected c*d - b in the numerator.");

//  a/((b/c + d)*e) -> a*c/((c*d + b)*e)
    auto e = graph::variable<T> (1, "");
    auto nest_div5 = a/((b/c + d)*e);
    auto nest_div5_cast = graph::divide_cast(nest_div5);
    assert(nest_div5_cast.get() && "Expected divide node.");
    assert(nest_div5_cast->get_left()->is_match(a*c) &&
           "Expected a*c in the numerator.");
    assert(nest_div5_cast->get_right()->is_match((c*d + b)*e) &&
           "Expected (c*d + b)*e in the numerator.");
//  a/(e*(b/c + d)) -> a*c/((c*d + b)*e)
    auto nest_div6 = a/((a + e)*(b/c + d));
    auto nest_div6_cast = graph::divide_cast(nest_div6);
    assert(nest_div6_cast.get() && "Expected divide node.");
    assert(nest_div6_cast->get_left()->is_match(a*c) &&
           "Expected a*c in the numerator.");
    assert(nest_div6_cast->get_right()->is_match((c*d + b)*(a + e)) &&
           "Expected (c*d + b)*e in the numerator.");
//  a/((d + b/c)*e) -> a*c/((c*d + b)*e)
    auto nest_div7 = a/((d + b/c)*e);
    auto nest_div7_cast = graph::divide_cast(nest_div7);
    assert(nest_div7_cast.get() && "Expected divide node.");
    assert(nest_div7_cast->get_left()->is_match(a*c) &&
           "Expected a*c in the numerator.");
    assert(nest_div7_cast->get_right()->is_match((c*d + b)*e) &&
           "Expected (c*d + b)*e in the numerator.");
//  a/(e*(d + b/c)) -> a*c/((c*d + b)*e)
    auto nest_div8 = a/((a + e)*(d + b/c));
    auto nest_div8_cast = graph::divide_cast(nest_div8);
    assert(nest_div8_cast.get() && "Expected divide node.");
    assert(nest_div8_cast->get_left()->is_match(a*c) &&
           "Expected a*c in the numerator.");
    assert(nest_div8_cast->get_right()->is_match((c*d + b)*(a + e)) &&
           "Expected (c*d + b)*e in the numerator.");
//  a/((b/c - d)*e) -> a*c/((b - c*d)*e)
    auto nest_div9 = a/((b/c - d)*e);
    auto nest_div9_cast = graph::divide_cast(nest_div9);
    assert(nest_div9_cast.get() && "Expected divide node.");
    assert(nest_div9_cast->get_left()->is_match(a*c) &&
           "Expected a*c in the numerator.");
    assert(nest_div9_cast->get_right()->is_match((b - c*d)*e) &&
           "Expected (b - c*d)*e in the numerator.");
//  a/(e*(b/c - d)) -> a*c/((b - c*d)*e)
    auto nest_div10 = a/((a + e)*(b/c - d));
    auto nest_div10_cast = graph::divide_cast(nest_div10);
    assert(nest_div10_cast.get() && "Expected divide node.");
    assert(nest_div10_cast->get_left()->is_match(a*c) &&
           "Expected a*c in the numerator.");
    assert(nest_div10_cast->get_right()->is_match((b - c*d)*(a + e)) &&
           "Expected (b - c*d)*e in the numerator.");
//  a/((d - b/c)*e) -> a*c/((c*d - b)*e)
    auto nest_div11 = a/((d - b/c)*e);
    auto nest_div11_cast = graph::divide_cast(nest_div11);
    assert(nest_div11_cast.get() && "Expected divide node.");
    assert(nest_div11_cast->get_left()->is_match(a*c) &&
           "Expected a*c in the numerator.");
    assert(nest_div11_cast->get_right()->is_match((c*d - b)*e) &&
           "Expected (c*d - b)*e in the numerator.");
//  a/(e*(d - b/c)) -> a*c/((c*d - b)*e)
    auto nest_div12 = a/((a + e)*(d - b/c));
    auto nest_div12_cast = graph::divide_cast(nest_div12);
    assert(nest_div12_cast.get() && "Expected divide node.");
    assert(nest_div12_cast->get_left()->is_match(a*c) &&
           "Expected a*c in the numerator.");
    assert(nest_div12_cast->get_right()->is_match((c*d - b)*(a + e)) &&
           "Expected (c*d - b)*e in the numerator.");

//  (a*b)^2/(a^2) = b^2
    auto powdiv = graph::pow(a*b, 2.0)/graph::pow(a, 2.0);
    assert(powdiv->is_match(b*b));
//  (b*a)^2/(a^2) = b^2
    auto powdiv2 = graph::pow(b*a, 2.0)/graph::pow(a, 2.0);
    assert(powdiv2->is_match(b*b));
//  (a*b)^2/((a^2)*c) = b^2/c
    auto powdiv3 = graph::pow(a*b, 2.0)/(graph::pow(a, 2.0)*c);
    assert(powdiv3->is_match((b*b)/c));
//  (b*a)^2/((a^2)*c) = b^2/c
    auto powdiv4 = graph::pow(b*a, 2.0)/(graph::pow(a, 2.0)*c);
    assert(powdiv4->is_match((b*b)/c));
//  (a*b)^2/(c*(a^2)) = b^2/c
    auto powdiv5 = graph::pow(a*b, 2.0)/(expression_c*graph::pow(a, 2.0));
    assert(powdiv5->is_match((b*b)/expression_c));
//  (b*a)^2/(c*(a^2)) = b^2/c
    auto powdiv6 = graph::pow(b*a, 2.0)/(expression_c*graph::pow(a, 2.0));
    assert(powdiv6->is_match((b*b)/expression_c));
//  (e*(a*b)^2)/(a^2) = e*b^2
    auto expression_e = 1.0 + e;
    auto powdiv7 = expression_e*graph::pow(a*b, 2.0)/graph::pow(a, 2.0);
    assert(powdiv7->is_match(expression_e*b*b));
//  ((a*b)^2*e)/(a^2) = e*b^2
    auto powdiv8 = graph::pow(a*b, 2.0)*expression_e/graph::pow(a, 2.0);
    assert(powdiv8->is_match(expression_e*b*b));
//  (e*(b*a)^2)/(a^2) = e*b^2
    auto powdiv9 = expression_e*graph::pow(b*a, 2.0)/graph::pow(a, 2.0);
    assert(powdiv9->is_match(expression_e*b*b));
//  ((b*a)^2*e)/(a^2) = e*b^2
    auto powdiv10 = graph::pow(b*a, 2.0)*expression_e/graph::pow(a, 2.0);
    assert(powdiv10->is_match(expression_e*b*b));
//  e*(a*b)^2/((a^2)*c) = e*b^2/c
    auto powdiv11 = expression_e*graph::pow(a*b, 2.0)/(graph::pow(a, 2.0)*expression_c);
    assert(powdiv11->is_match((expression_e*b*b)/expression_c));
//  e*(b*a)^2/((a^2)*c) = e*b^2/c
    auto powdiv12 = expression_e*graph::pow(b*a, 2.0)/(graph::pow(a, 2.0)*expression_c);
    assert(powdiv12->is_match((expression_e*b*b)/expression_c));
//  (a*b)^2*e/((a^2)*c) = e*b^2/c
    auto powdiv13 = graph::pow(a*b, 2.0)*expression_e/(graph::pow(a, 2.0)*expression_c);
    assert(powdiv13->is_match((expression_e*b*b)/expression_c));
//  (b*a)^2*e/((a^2)*c) = e*b^2/c
    auto powdiv14 = graph::pow(b*a, 2.0)*expression_e/(graph::pow(a, 2.0)*expression_c);
    assert(powdiv14->is_match((expression_e*b*b)/expression_c));
//  e*(a*b)^2/(c*(a^2)) = e*b^2/c
    auto powdiv15 = expression_e*graph::pow(a*b, 2.0)/(expression_c*graph::pow(a, 2.0));
    assert(powdiv15->is_match((expression_e*b*b)/expression_c));
//  e*(b*a)^2/(c*(a^2)) = e*b^2/c
    auto powdiv16 = expression_e*graph::pow(b*a, 2.0)/(expression_c*graph::pow(a, 2.0));
    assert(powdiv16->is_match((expression_e*b*b)/expression_c));
//  (a*b)^2*e/(c*(a^2)) = e*b^2/c
    auto powdiv17 = graph::pow(a*b, 2.0)*expression_e/(expression_c*graph::pow(a, 2.0));
    assert(powdiv17->is_match((expression_e*b*b)/expression_c));
//  (b*a)^2*e/(c*(a^2)) = e*b^2/c
    auto powdiv18 = graph::pow(a*b, 2.0)*expression_e/(expression_c*graph::pow(a, 2.0));
    assert(powdiv18->is_match((expression_e*b*b)/expression_c));

//  a*(b*c)/c -> a*b
    assert(((a*(b*c))/c)->is_match(a*b) && "Expected a*b");
//  a*(c*b)/c -> a*b
    assert(((a*(c*b))/c)->is_match(a*b) && "Expected a*b");
//  (a*c)*b/c -> a*b
    assert((((a*c)*b)/c)->is_match(a*b) && "Expected a*b");
//  (c*a)*b/c -> a*b
    assert((((c*a)*b)/c)->is_match(a*b) && "Expected a*b");

//  a/(b/c) -> a*c/b
    assert((a/(b/c))->is_match(a*c/b) && "Expected a*b/c");

//  (a*b*c)^2/a^2 -> (b*c)^2
//  (a*b*c)^2/(a^2*d) -> (b*c)^2/d
//  (e*(a*b*c)^2)/(a^2*d) -> e*(b*c)^2/d
}

//------------------------------------------------------------------------------
///  @brief Tests for fma nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_fma() {
//  Three constant nodes should reduce to a single constant node with a*b + c.
    auto zero = graph::zero<T> ();
    auto one = graph::one<T> ();
    auto two = graph::constant<T> (static_cast<T> (2.0));

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

    auto one_two_three = graph::fma(one, two, 3.0);
    const backend::buffer<T> one_two_three_result = one_two_three->evaluate();
    assert(one_two_three_result.size() == 1 && "Expected single value.");
    assert(one_two_three_result.at(0) == static_cast<T> (5.0) &&
           "Expected five for result");

    auto two_three_one = graph::fma(two, 3.0, one);
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

//  fma(1,a,b) = a + b
    auto one_times_vara_plus_varb = graph::fma(one, var_a, var_b);
    auto one_times_vara_plus_varb_cast =
        graph::add_cast(one_times_vara_plus_varb);
    assert(one_times_vara_plus_varb_cast.get() && "Expected an add node.");

//  fma(a,1,b) = a + b
    auto vara_times_one_plus_varb = graph::fma(var_a, one, var_b);
    auto vara_times_one_plus_varb_cast =
        graph::add_cast(vara_times_one_plus_varb);
    assert(vara_times_one_plus_varb_cast.get() && "Expected an add node.");

//  fma(b,a,a) = a*(1 + b)
    auto common1 = graph::fma(var_a, var_b, var_a);
    auto common1_cast = graph::multiply_cast(common1);
    assert(common1_cast.get() && "Expected multiply node.");
//  fma(b,a,a) = a*(1 + b)
    auto common2 = graph::fma(var_b, var_a, var_a);
    auto common2_cast = graph::multiply_cast(common2);
    assert(common2_cast.get() && "Expected multiply node.");
    assert(common1->is_match(common2) && "Expected same graph");

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

//  fma(a, b, fma(c, b, d)) -> fma(b, a + c, d)
    auto var_d = graph::variable<T> (1, "");
    auto match1 = graph::fma(var_b, var_a + var_c, var_d);
    auto nested_fma1 = graph::fma(var_a, var_b, 
                                  graph::fma(var_c, var_b, var_d));
    assert(nested_fma1->is_match(match1) && "Expected match.");
//  fma(b, a, fma(c, b, d)) -> fma(b, a + c, d)
    auto nested_fma2 = graph::fma(var_b, var_a, 
                                  graph::fma(var_c, var_b, var_d));
    assert(nested_fma2->is_match(match1) && "Expected match.");
//  fma(a, b, fma(b, c, d)) -> fma(b, a + c, d)
    auto nested_fma3 = graph::fma(var_a, var_b, 
                                  graph::fma(var_b, var_c, var_d));
    assert(nested_fma3->is_match(match1) && "Expected match.");
//  fma(b, a, fma(b, c, d)) -> fma(b, a + c, d)
    auto nested_fma4 = graph::fma(var_b, var_a, 
                                  graph::fma(var_b, var_c, var_d));
    assert(nested_fma4->is_match(match1) && "Expected match.");

//  fma(a, e*b, fma(c, b, d)) -> fma(b, fma(a, e, c), d)
    auto var_e = graph::variable<T> (1, "");
    auto match2 = graph::fma(var_b, graph::fma(var_a, var_e, var_c), var_d);
    auto nested_fma5 = graph::fma(var_a,
                                  var_e*var_b,
                                  graph::fma(var_c, var_b, var_d));
    assert(nested_fma5->is_match(match2) && "Expected match.");
//  fma(a, b*e, fma(c, b, d)) -> fma(b, fma(a, e, c), d)
    auto nested_fma6 = graph::fma(var_a,
                                  var_b*var_e,
                                  graph::fma(var_c, var_b, var_d));
    assert(nested_fma6->is_match(match2) && "Expected match.");
//  fma(a, e*b, fma(b, c, d)) -> fma(b, fma(a, e, c), d)
    auto nested_fma7 = graph::fma(var_a,
                                  var_e*var_b,
                                  graph::fma(var_b, var_c, var_d));
    assert(nested_fma7->is_match(match2) && "Expected match.");
//  fma(a, b*e, fma(c, b, d)) -> fma(b, fma(a, e, c), d)
    auto nested_fma8 = graph::fma(var_a,
                                  var_b*var_e,
                                  graph::fma(var_b, var_c, var_d));
    assert(nested_fma8->is_match(match2) && "Expected match.");

//  fma(e*b, a, fma(c, b, d)) -> fma(b, fma(a, e, c), d)
    auto nested_fma9 = graph::fma(var_e*var_b,
                                  var_a,
                                  graph::fma(var_c, var_b, var_d));
    assert(nested_fma9->is_match(match2) && "Expected match.");
//  fma(b*e, a, fma(c, b, d)) -> fma(b, fma(a, e, c), d)
    auto nested_fma10 = graph::fma(var_b*var_e,
                                   var_a,
                                   graph::fma(var_c, var_b, var_d));
    assert(nested_fma10->is_match(match2) && "Expected match.");
//  fma(e*b, a, fma(b, c, d)) -> fma(b, fma(a, e, c), d)
    auto nested_fma11 = graph::fma(var_e*var_b,
                                   var_a,
                                   graph::fma(var_b, var_c, var_d));
    assert(nested_fma11->is_match(match2) && "Expected match.");
//  fma(e*d, a, fma(b, c, d)) -> fma(b, fma(a, e, c), d)
    auto nested_fma12 = graph::fma(var_a,
                                   var_b*var_e,
                                   graph::fma(var_b, var_c, var_d));
    assert(nested_fma12->is_match(match2) && "Expected match.");

//  fma(a, b, fma(c, e*b, d)) -> fma(b, fma(c, e, a), d)
    auto match3 = graph::fma(var_b, graph::fma(var_c, var_e, var_a), var_d);
    auto nested_fma13 = graph::fma(var_a,
                                   var_b,
                                   graph::fma(var_c, var_e*var_b, var_d));
    assert(nested_fma13->is_match(match3) && "Expected match.");
//  fma(b, a, fma(c, e*b, d)) -> fma(b, fma(c, e, a), d)
    auto nested_fma14 = graph::fma(var_b,
                                   var_a,
                                   graph::fma(var_c, var_e*var_b, var_d));
    assert(nested_fma14->is_match(match3) && "Expected match.");
//  fma(a, b, fma(c, b*e, d)) -> fma(b, fma(c, e, a), d)
    auto nested_fma15 = graph::fma(var_a,
                                   var_b,
                                   graph::fma(var_c, var_b*var_e, var_d));
    assert(nested_fma15->is_match(match3) && "Expected match.");
//  fma(b, a, fma(c, b*e, d)) -> fma(b, fma(c, e, a), d)
    auto nested_fma16 = graph::fma(var_b,
                                   var_a,
                                   graph::fma(var_c, var_b*var_e, var_d));
    assert(nested_fma16->is_match(match3) && "Expected match.");
//  fma(a, b, fma(e*b, c, d)) -> fma(b, fma(c, e, a), d)
    auto nested_fma17 = graph::fma(var_a,
                                   var_b,
                                   graph::fma(var_e*var_b, var_c, var_d));
    assert(nested_fma17->is_match(match3) && "Expected match.");
//  fma(b, a, fma(e*b, c, d)) -> fma(b, fma(c, e, a), d)
    auto nested_fma18 = graph::fma(var_b,
                                   var_a,
                                   graph::fma(var_e*var_b, var_c, var_d));
    assert(nested_fma18->is_match(match3) && "Expected match.");
//  fma(a, b, fma(b*e, c, d)) -> fma(b, fma(c, e, a), d)
    auto nested_fma19 = graph::fma(var_a,
                                   var_b,
                                   graph::fma(var_b*var_e, var_c, var_d));
    assert(nested_fma19->is_match(match3) && "Expected match.");
//  fma(b, a, fma(b*e, c, d)) -> fma(b, fma(c, e, a), d)
    auto nested_fma20 = graph::fma(var_b,
                                   var_a,
                                   graph::fma(var_b*var_e, var_c, var_d));
    assert(nested_fma20->is_match(match3) && "Expected match.");

//  fma(a, f*b, fma(c, e*b, d)) -> fma(b, fma(a, f, c*e), d)
    auto var_f = graph::variable<T> (1, "");
    auto match4 = graph::fma(var_b, graph::fma(var_a, var_f, var_c*var_e), var_d);
    auto nested_fma21 = graph::fma(var_a,
                                   var_f*var_b,
                                   graph::fma(var_c, var_e*var_b, var_d));
    assert(nested_fma21->is_match(match4) && "Expected match.");
//  fma(a, b*f, fma(c, e*b, d)) -> fma(b, fma(a, f, c*e), d)
    auto nested_fma22 = graph::fma(var_a,
                                   var_b*var_f,
                                   graph::fma(var_c, var_e*var_b, var_d));
    assert(nested_fma22->is_match(match4) && "Expected match.");
//  fma(a, f*b, fma(c, b*e, d)) -> fma(b, fma(a, f, c*e), d)
    auto nested_fma23 = graph::fma(var_a,
                                   var_f*var_b,
                                   graph::fma(var_c, var_b*var_e, var_d));
    assert(nested_fma23->is_match(match4) && "Expected match.");
//  fma(a, b*f, fma(c, b*e, d)) -> fma(b, fma(a, f, c*e), d)
    auto nested_fma24 = graph::fma(var_a,
                                   var_b*var_f,
                                   graph::fma(var_c, var_b*var_e, var_d));
    assert(nested_fma24->is_match(match4) && "Expected match.");
//  fma(f*b, a, fma(c, e*b, d)) -> fma(b, fma(a, f, c*e), d)
    auto nested_fma25 = graph::fma(var_f*var_b,
                                   var_a,
                                   graph::fma(var_c, var_e*var_b, var_d));
    assert(nested_fma25->is_match(match4) && "Expected match.");
//  fma(b*f, a, fma(c, e*b, d)) -> fma(b, fma(a, f, c*e), d)
    auto nested_fma26 = graph::fma(var_b*var_f,
                                   var_a,
                                   graph::fma(var_c, var_e*var_b, var_d));
    assert(nested_fma26->is_match(match4) && "Expected match.");
//  fma(f*b, a, fma(c, b*e, d)) -> fma(b, fma(a, f, c*e), d)
    auto nested_fma27 = graph::fma(var_f*var_b,
                                   var_a,
                                   graph::fma(var_c, var_b*var_e, var_d));
    assert(nested_fma27->is_match(match4) && "Expected match.");
//  fma(b*f, a, fma(c, b*e, d)) -> fma(b, fma(a, f, c*e), d)
    auto nested_fma28 = graph::fma(var_b*var_f,
                                   var_a,
                                   graph::fma(var_c, var_b*var_e, var_d));
    assert(nested_fma28->is_match(match4) && "Expected match.");
//  fma(a, f*b, fma(e*b, c, d)) -> fma(b, fma(a, f, c*e), d)
    auto nested_fma29 = graph::fma(var_a,
                                   var_f*var_b,
                                   graph::fma(var_e*var_b, var_c, var_d));
    assert(nested_fma29->is_match(match4) && "Expected match.");
//  fma(a, b*f, fma(e*b, c, d)) -> fma(b, fma(a, f, c*e), d)
    auto nested_fma30 = graph::fma(var_a,
                                   var_b*var_f,
                                   graph::fma(var_e*var_b, var_c, var_d));
    assert(nested_fma30->is_match(match4) && "Expected match.");
//  fma(a, f*b, fma(b*e, c, d)) -> fma(b, fma(a, f, c*e), d)
    auto nested_fma31= graph::fma(var_a,
                                   var_f*var_b,
                                   graph::fma(var_b*var_e, var_c, var_d));
    assert(nested_fma31->is_match(match4) && "Expected match.");
//  fma(a, b*f, fma(b*e, c, d)) -> fma(b, fma(a, f, c*e), d)
    auto nested_fma32 = graph::fma(var_a,
                                   var_b*var_f,
                                   graph::fma(var_b*var_e, var_c, var_d));
    assert(nested_fma32->is_match(match4) && "Expected match.");
//  fma(f*b, a, fma(e*b, c, d)) -> fma(b, fma(a, f, c*e), d)
    auto nested_fma33 = graph::fma(var_f*var_b,
                                   var_a,
                                   graph::fma(var_e*var_b, var_c, var_d));
    assert(nested_fma33->is_match(match4) && "Expected match.");
//  fma(b*f, a, fma(e*b, c, d)) -> fma(b, fma(a, f, c*e), d)
    auto nested_fma34 = graph::fma(var_b*var_f,
                                   var_a,
                                   graph::fma(var_e*var_b, var_c, var_d));
    assert(nested_fma34->is_match(match4) && "Expected match.");
//  fma(f*b, a, fma(b*e, c, d)) -> fma(b, fma(a, f, c*e), d)
    auto nested_fma35 = graph::fma(var_f*var_b,
                                   var_a,
                                   graph::fma(var_b*var_e, var_c, var_d));
    assert(nested_fma35->is_match(match4) && "Expected match.");
//  fma(b*f, a, fma(b*e, c, d)) -> fma(b, fma(a, f, c*e), d)
    auto nested_fma36 = graph::fma(var_b*var_f,
                                   var_a,
                                   graph::fma(var_b*var_e, var_c, var_d));
    assert(nested_fma36->is_match(match4) && "Expected match.");

//  fma(a^b,a^c,d) -> a^(b+c) +d
    assert(graph::fma(graph::pow(var_a, var_b),
                      graph::pow(var_a, var_c),
                      var_d)->is_match(graph::pow(var_a, 
                                                  var_b + var_c) + var_d) &&
           "Expected match");

//  fma(a,x^b,fma(c,x^d,e)) -> fma(x^d,fma(x^(d-b),a,c),e) if b > d
    auto matchv1 = graph::fma(graph::pow(var_b, 2.0),
                              fma(var_b, var_a, var_c),
                              var_d);
    auto matchv2 = graph::fma(graph::pow(var_b, 2.0),
                              fma(var_b, var_c, var_a),
                              var_d);
    auto nested_fmav1 = graph::fma(var_a,
                                   graph::pow(var_b, 3.0),
                                   fma(var_c,
                                       graph::pow(var_b, 2.0),
                                       var_d));
    assert(nested_fmav1->is_match(matchv1) && "Expected match");
//  fma(a,x^b,fma(c,x^d,e)) -> fma(x^b,fma(x^(d-b),c,a),e) if d > b
    auto nested_fmav2 = graph::fma(var_a,
                                   graph::pow(var_b, 2.0),
                                   fma(var_c,
                                       graph::pow(var_b, 3.0),
                                       var_d));
    assert(nested_fmav2->is_match(matchv2) && "Expected match");
//  fma(x^b,a,fma(c,x^d,e)) -> fma(x^d,fma(x^(d-b),a,c),e) if b > d
    auto nested_fmav3 = graph::fma(graph::pow(var_b, 3.0),
                                   var_a,
                                   fma(var_c,
                                       graph::pow(var_b, 2.0),
                                       var_d));
    assert(nested_fmav3->is_match(matchv1) && "Expected match");
//  fma(x^b,a,fma(c,x^d,e)) -> fma(x^b,fma(x^(d-b),c,a),e) if d > b
    auto nested_fmav4 = graph::fma(graph::pow(var_b, 2.0),
                                   var_a,
                                   fma(var_c,
                                       graph::pow(var_b, 3.0),
                                       var_d));
    assert(nested_fmav4->is_match(matchv2) && "Expected match");
//  fma(a,x^b,fma(x^d,c,e)) -> fma(x^d,fma(x^(d-b),a,c),e) if b > d
    auto nested_fmav5 = graph::fma(var_a,
                                   graph::pow(var_b, 3.0),
                                   fma(graph::pow(var_b, 2.0),
                                       var_c,
                                       var_d));
    assert(nested_fmav5->is_match(matchv1) && "Expected match");
//  fma(a,x^b,fma(x^d,c,e)) -> fma(x^b,fma(x^(d-b),c,a),e) if d > b
    auto nested_fmav6 = graph::fma(var_a,
                                   graph::pow(var_b, 2.0),
                                   fma(graph::pow(var_b, 3.0),
                                       var_c,
                                       var_d));
    assert(nested_fmav6->is_match(matchv2) && "Expected match");
//  fma(x^b,a,fma(x^d,c,e)) -> fma(x^d,fma(x^(d-b),a,c),e) if b > d
    auto nested_fmav7 = graph::fma(graph::pow(var_b, 3.0),
                                   var_a,
                                   fma(graph::pow(var_b, 2.0),
                                       var_c,
                                       var_d));
    assert(nested_fmav7->is_match(matchv1) && "Expected match");
//  fma(x^b,a,fma(x^d,c,e)) -> fma(x^b,fma(x^(d-b),c,a),e) if d > b
    auto nested_fmav8 = graph::fma(graph::pow(var_b, 2.0),
                                   var_a,
                                   fma(graph::pow(var_b, 3.0),
                                       var_c,
                                       var_d));
    assert(nested_fmav8->is_match(matchv2) && "Expected match");

//  fma(a, b, a*b) -> 2*a*b
//  fma(b, a, a*b) -> 2*a*b
//  fma(a, b, b*a) -> 2*a*b
    assert(graph::fma(var_a, var_b, var_a*var_b)->is_match(2.0*var_a*var_b) &&
           "Expected to match 2*a*b");
    assert(graph::fma(var_b, var_a, var_a*var_b)->is_match(2.0*var_a*var_b) &&
           "Expected to match 2*a*b");
    assert(graph::fma(var_a, var_b, var_b*var_a)->is_match(2.0*var_a*var_b) &&
           "Expected to match 2*a*b");

//  fma(c1*a,b,c2*d) -> c1*(a*b + c2/c1*d)
    assert(graph::multiply_cast(graph::fma(2.0*var_b,
                                           var_a,
                                           4.0*var_b)).get() &&
           "Expected multiply node.");

//  fma(c1*a,b,c2/d) -> c1*(a*b + c1/(c2*d))
//  fma(c1*a,b,d/c2) -> c1*(a*b + d/(c1*c2))
    assert(graph::multiply_cast(graph::fma(2.0*var_b,
                                           var_a,
                                           4.0/var_b)).get() &&
           "Expected multiply node.");
    assert(graph::multiply_cast(graph::fma(2.0*var_b,
                                           var_a,
                                           var_b/4.0)).get() &&
           "Expected multiply node.");

//  fma(a,v1,b*v2) -> (a + b*v1/v2)*v1
//  fma(a,v1,c*b*v2) -> (a + c*b*v1/v2)*v1
    assert(graph::multiply_cast(graph::fma(2.0,
                                           var_a,
                                           2.0*sqrt(var_a))).get() &&
           "Expected multiply node.");
    assert(graph::multiply_cast(graph::fma(2.0,
                                           var_a,
                                           2.0*(var_b*sqrt(var_a)))).get() &&
           "Expected multiply node.");

//  fma(a,b,fma(a,b,2)) -> 2*fma(a,b,1)
    auto chained_fma = fma(var_a, var_b, fma(var_a, var_b, two));
    auto chained_fma_cast = multiply_cast(chained_fma);
    assert(chained_fma_cast.get() && "Expected muliply node.");
    assert(constant_cast(chained_fma_cast->get_left()) &&
           "Expected constant node.");
//  fma(a,b,fma(b,a,c)) -> 2*fma(a,b,1)
    auto chained_fma2 = fma(var_a, var_b, fma(var_b, var_a, two));
    auto chained_fma_cast2 = multiply_cast(chained_fma2);
    assert(chained_fma_cast2.get() && "Expected muliply node.");
    assert(constant_cast(chained_fma_cast2->get_left()) &&
           "Expected constant node.");

//  fma(a,b/c,fma(d,e/c,g)) -> (a*b + d*e)/c + g
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
    auto power_factor = graph::fma(var_a, graph::pow(var_b, -2.0),
                                   var_c/graph::pow(var_b, 2.0));
    auto power_factor_cast = divide_cast(power_factor);
    assert(power_factor_cast.get() && "Expected a divide node.");
//  fma(b^-c,a,d/b^c) -> (a + d)/b^c
    auto power_factor2 = graph::fma(graph::pow(var_b, -2.0), var_a,
                                    var_c/graph::pow(var_b, 2.0));
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

//  Test node properties.
    assert(one_two_three->is_constant() && "Expected a constant.");
    assert(!one_two_three->is_all_variables() && "Did not expect a variable.");
    assert(one_two_three->is_power_like() && "Expected a power like.");
    auto constant_fma = graph::fma(one_two_three,
                                   graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (1.0),
                                                                            static_cast<T> (2.0)}),
                                                           var_a),
                                   one);
    assert(!constant_fma->is_all_variables() && "Did not expect a variable.");
    assert(constant_fma->is_power_like() && "Expected a power like.");
    auto constant_var_fma = graph::fma(var_a, var_b, 1.0);
    assert(!constant_var_fma->is_constant() && "Did not expect a constant.");
    assert(!constant_var_fma->is_all_variables() && "Did not expect a variable.");
    assert(!constant_var_fma->is_power_like() && "Did not expect a power like.");
    auto var_var_fma = graph::fma(var_a, var_b, var_c);
    assert(!var_var_fma->is_constant() && "Did not expect a constant.");
    assert(var_var_fma->is_all_variables() && "Expected a variable.");
    assert(!var_var_fma->is_power_like() && "Did not expect a power like.");

//  fma(c*a,b,d) -> fma(c,a*b,d)
    auto constant_move = graph::fma(3.0*var_a, var_b, var_c);
    auto constant_move_cast = graph::fma_cast(constant_move);
    assert(constant_move_cast.get() && "Expected an fma cast");
    assert(graph::constant_cast(constant_move_cast->get_left()) &&
           "Expected a constant on the left.");
//  fma(a,c*b,d) -> fma(c,a*b,d)
    auto constant_move2 = graph::fma(var_a, 3.0*var_b, var_c);
    auto constant_move2_cast = graph::fma_cast(constant_move2);
    assert(constant_move2_cast.get() && "Expected an fma cast");
    assert(graph::constant_cast(constant_move2_cast->get_left()) &&
           "Expected a constant on the left.");
    
//  fma(c, pwc*v, d) -> fma(pwc, v, d)
    auto piecewise1 = graph::fma<T> (2.0,
                                     graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (1.0),
                                                                              static_cast<T> (2.0)}),
                                                             var_a)*var_a,
                                     var_b);
    auto piecewise1_cast = graph::fma_cast(piecewise1);
    assert(piecewise1_cast.get() && "Expected a fma node.");
    assert(graph::piecewise_1D_cast(piecewise1_cast->get_left()) &&
           "Expected a piecewise_1D node.");
    auto piecewise2 = graph::fma<T> (2.0,
                                     graph::piecewise_2D<T> (std::vector<T> ({static_cast<T> (1.0),
                                                                              static_cast<T> (2.0)}),
                                                             1, var_a, var_b)*var_a,
                                     var_b);
    auto piecewise2_cast = graph::fma_cast(piecewise2);
    assert(piecewise2_cast.get() && "Expected a fma node.");
    assert(graph::piecewise_2D_cast(piecewise2_cast->get_left()) &&
           "Expected a piecewise_2D node.");

//  fma(a/b,c,(d/b)*e) -> fma(a,c,d*e)/b
    assert(graph::divide_cast(graph::fma(var_a/var_b, 
                                         var_c,
                                         (var_d/var_b)*var_e)).get() &&
           "Expected a divide node.");
//  fma(a/b,c,e*(d/b)) -> fma(a,c,d*e)/b
    assert(graph::divide_cast(graph::fma(var_a/var_b, 
                                         var_c,
                                         var_e*(var_d/var_b))).get() &&
           "Expected a divide node.");
//  fma(a,c/b,(d/b)*e) -> fma(a,c,d*e)/b
    assert(graph::divide_cast(graph::fma(var_a, 
                                         var_c/var_b,
                                         (var_d/var_b)*var_e)).get() &&
           "Expected a divide node.");
//  fma(a,c/b,e*(d/b)) -> fma(a,c,d*e)/b
    assert(graph::divide_cast(graph::fma(var_a, 
                                         var_c/var_b,
                                         var_e*(var_d/var_b))).get() &&
           "Expected a divide node.");
//  fma(a/b*c,d,e/b) -> fma(a*c,d,e)/b
    assert(graph::divide_cast(graph::fma((var_a/var_b)*var_c,
                                         var_d,
                                         var_e/var_b)).get() &&
           "Expected a divide node.");
//  fma(a*c/b,d,e/b) -> fma(a*c,d,e)/b
    assert(graph::divide_cast(graph::fma(var_a*(var_c/var_b),
                                         var_d,
                                         var_e/var_b)).get() &&
           "Expected a divide node.");
//  fma(a,c/b*d,e/b) -> fma(a,c*d,e)/b
    assert(graph::divide_cast(graph::fma(var_a,
                                         (var_c/var_b)*var_d,
                                         var_e/var_b)).get() &&
           "Expected a divide node.");
//  fma(a,c*d/b,e/b) -> fma(a,c*d,e)/b
    assert(graph::divide_cast(graph::fma(var_a,
                                         var_c*(var_d/var_b),
                                         var_e/var_b)).get() &&
           "Expected a divide node.");

//  fma(a, b/c, ((f/c)*e)*d) -> fma(a, b, f*e*d)/c
//  fma(a/c, b, ((f/c)*e)*d) -> fma(a, b, f*e*d)/c
//  fma(a, b/c, (e*(f/c))*d) -> fma(a, b, f*e*d)/c
//  fma(a/c, b, (e*(f/c))*d) -> fma(a, b, f*e*d)/c
//  fma(a, b/c, d*((f/c)*e)) -> fma(a, b, f*e*d)/c
//  fma(a/c, b, d*((f/c)*e)) -> fma(a, b, f*e*d)/c
//  fma(a, b/c, d*(e*(f/c))) -> fma(a, b, f*e*d)/c
//  fma(a/c, b, d*(e*(f/c))) -> fma(a, b, f*e*d)/c
    auto exp_a = 1.0 + var_a;
    auto exp_b = 1.0 + var_b;
    auto exp_c = 1.0 + var_c;
    auto exp_d = 1.0 + var_d;
    auto exp_e = 1.0 + var_e;
    auto exp_f = 1.0 + var_f;
    assert(graph::divide_cast(fma(exp_a, exp_b/exp_c,
                                  ((var_f/exp_c)*exp_e)*exp_d)).get() &&
           "Expected a divide node.");
    assert(graph::divide_cast(fma(exp_a/exp_c, exp_b,
                                  ((exp_f/exp_c)*exp_e)*exp_d)).get() &&
           "Expected a divide node.");
    assert(graph::divide_cast(fma(exp_a, exp_b/exp_c,
                                  (exp_e*(exp_f/exp_c))*exp_d)).get() &&
           "Expected a divide node.");
    assert(graph::divide_cast(fma(exp_a/exp_c, exp_b,
                                  (exp_e*(exp_f/exp_c))*exp_d)).get() &&
           "Expected a divide node.");
    assert(graph::divide_cast(fma(exp_a, exp_b/exp_c,
                                  exp_d*((exp_f/exp_c)*exp_e))).get() &&
           "Expected a divide node.");
    assert(graph::divide_cast(fma(exp_a/exp_c, exp_b,
                                  exp_d*((exp_f/exp_c)*exp_e))).get() &&
           "Expected a divide node.");
    assert(graph::divide_cast(fma(exp_a, exp_b/exp_c,
                                  exp_d*(exp_e*(exp_f/exp_c)))).get() &&
           "Expected a divide node.");
    assert(graph::divide_cast(fma(exp_a/exp_c, exp_b,
                                  exp_d*(exp_e*(exp_f/exp_c)))).get() &&
           "Expected a divide node.");
    auto var_j = graph::variable<T> (1, "");
    auto exp_j = 1.0 + var_j;
    assert(graph::fma_cast(fma(exp_a, exp_b/exp_j,
                               ((var_f/exp_c)*exp_e)*exp_d)).get() &&
           "Expected a divide node.");
    assert(graph::fma_cast(fma(exp_a/exp_j, exp_b,
                               ((exp_f/exp_c)*exp_e)*exp_d)).get() &&
           "Expected a divide node.");
    assert(graph::fma_cast(fma(exp_a, exp_b/exp_j,
                               (exp_e*(exp_f/exp_c))*exp_d)).get() &&
           "Expected a divide node.");
    assert(graph::fma_cast(fma(exp_a/exp_j, exp_b,
                               (exp_e*(exp_f/exp_c))*exp_d)).get() &&
           "Expected a divide node.");
    assert(graph::fma_cast(fma(exp_a, exp_b/exp_j,
                               exp_d*((exp_f/exp_c)*exp_e))).get() &&
           "Expected a divide node.");
    assert(graph::fma_cast(fma(exp_a/exp_j, exp_b,
                               exp_d*((exp_f/exp_c)*exp_e))).get() &&
           "Expected a divide node.");
    assert(graph::fma_cast(fma(exp_a, exp_b/exp_j,
                               exp_d*(exp_e*(exp_f/exp_c)))).get() &&
           "Expected a divide node.");
    assert(graph::fma_cast(fma(exp_a/exp_j, exp_b,
                               exp_d*(exp_e*(exp_f/exp_c)))).get() &&
           "Expected a divide node.");

//  fma(exp(a), exp(b), c) -> exp(a + b) + c
    auto expa = graph::exp(exp_a);
    auto expb = graph::exp(exp_b);
    auto fmaexp = graph::fma(expa, expb, var_c);
    assert(graph::add_cast(fmaexp).get() && "Expected an Add node.");

//  fma(exp(a), exp(b)*c, d) = fma(exp(a + b), c, d)
    auto fmaexp2 = graph::fma(expa, expb*exp_c, exp_b);
    auto fmaexp2_cast = graph::fma_cast(fmaexp2);
    assert(fmaexp2_cast.get() && "Expected a fma node.");
    assert(graph::exp_cast(fmaexp2_cast->get_left()).get() &&
           "Expected a exp node on the left.");
//  fma(exp(a), c*exp(b), d) = fma(exp(a + b), c, d)
    auto fmaexp3 = graph::fma(expa, exp_c*expb, exp_d);
    auto fmaexp3_cast = graph::fma_cast(fmaexp3);
    assert(fmaexp3_cast.get() && "Expected a fma node.");
    assert(graph::exp_cast(fmaexp3_cast->get_left()).get() &&
           "Expected a exp node on the left.");
//  fma(exp(a)*c, exp(b), d) = fma(exp(var_a + var_b), c, d)
    auto fmaexp4 = graph::fma(expa*exp_c, expb, exp_d);
    auto fmaexp4_cast = graph::fma_cast(fmaexp4);
    assert(fmaexp4_cast.get() && "Expected a fma node.");
    assert(graph::exp_cast(fmaexp4_cast->get_left()).get() &&
           "Expected a exp node on the left.");
//  fma(c*exp(a), exp(b), d) = fma(exp(var_a + var_b), c, d)
    auto fmaexp5 = graph::fma(exp_c*expa, expb, exp_d);
    auto fmaexp5_cast = graph::fma_cast(fmaexp5);
    assert(fmaexp5_cast.get() && "Expected a fma node.");
    assert(graph::exp_cast(fmaexp5_cast->get_left()).get() &&
           "Expected a exp node on the left.");

//  fma(exp(a)*c, exp(b)*d, e) -> fma(exp(a + b), c*d, e)
    auto fmaexp6 = graph::fma(expa*exp_c, expb*exp_d, exp_e);
    auto fmaexp6_cast = graph::fma_cast(fmaexp6);
    assert(fmaexp6_cast.get() && "Expected a fma node.");
    assert(graph::exp_cast(fmaexp6_cast->get_left()).get() &&
           "Expected a exp node on the left.");
//  fma(exp(a)*c, d*exp(b), e) -> fma(exp(a + b), c*d, e)
    auto fmaexp7 = graph::fma(expa*exp_c, exp_d*expb, exp_e);
    auto fmaexp7_cast = graph::fma_cast(fmaexp7);
    assert(fmaexp7_cast.get() && "Expected a fma node.");
    assert(graph::exp_cast(fmaexp7_cast->get_left()).get() &&
           "Expected a exp node on the left.");
//  fma(c*exp(a), exp(b)*d, e) -> fma(exp(a + b), c*d, e)
    auto fmaexp8 = graph::fma(exp_c*expa, expb*exp_d, exp_e);
    auto fmaexp8_cast = graph::fma_cast(fmaexp8);
    assert(fmaexp8_cast.get() && "Expected a fma node.");
    assert(graph::exp_cast(fmaexp8_cast->get_left()).get() &&
           "Expected a exp node on the left.");
//  fma(c*exp(a), d*exp(b), e) -> fma(exp(a + b), c*d, e)
    auto fmaexp9 = graph::fma(exp_c*expa, exp_d*expb, exp_e);
    auto fmaexp9_cast = graph::fma_cast(fmaexp9);
    assert(fmaexp9_cast.get() && "Expected a fma node.");
    assert(graph::exp_cast(fmaexp9_cast->get_left()).get() &&
           "Expected a exp node on the left.");

//  fma(exp(a)*c, exp(b)/d, e) -> fma(exp(a + b), c/d, e)
    auto fmaexp10 = graph::fma(expa*exp_c, expb/exp_d, exp_e);
    auto fmaexp10_cast = graph::fma_cast(fmaexp10);
    assert(fmaexp10_cast.get() && "Expected a fma node.");
    assert(graph::exp_cast(fmaexp10_cast->get_left()).get() &&
           "Expected a exp node on the left.");
//  fma(exp(a)*c, d/exp(b), e) -> fma(exp(a - b), c*d, e)
    auto fmaexp11 = graph::fma(expa*exp_c, exp_d/expb, exp_e);
    auto fmaexp11_cast = graph::fma_cast(fmaexp11);
    assert(fmaexp11_cast.get() && "Expected a fma node.");
    assert(graph::exp_cast(fmaexp11_cast->get_left()).get() &&
           "Expected a exp node on the left.");
//  fma(c*exp(a), exp(b)/d, e) -> fma(exp(a + b), c/d, e)
    auto fmaexp12 = graph::fma(exp_c*expa, expb/exp_d, exp_e);
    auto fmaexp12_cast = graph::fma_cast(fmaexp12);
    assert(fmaexp12_cast.get() && "Expected a fma node.");
    assert(graph::exp_cast(fmaexp12_cast->get_left()).get() &&
           "Expected a exp node on the left.");
//  fma(c*exp(a), d/exp(b), e) -> fma(exp(a - b), c*d, e)
    auto fmaexp13 = graph::fma(exp_c*expa, exp_d/expb, exp_e);
    auto fmaexp13_cast = graph::fma_cast(fmaexp13);
    assert(fmaexp13_cast.get() && "Expected a fma node.");
    assert(graph::exp_cast(fmaexp13_cast->get_left()).get() &&
           "Expected a exp node on the left.");

//  fma(exp(a)/c, exp(b)*d, e) -> fma(exp(a + b), d/c, e)
    auto fmaexp14 = graph::fma(expa/exp_c, expb*exp_d, exp_e);
    auto fmaexp14_cast = graph::fma_cast(fmaexp14);
    assert(fmaexp14_cast.get() && "Expected a fma node.");
    assert(graph::exp_cast(fmaexp14_cast->get_left()).get() &&
           "Expected a exp node on the left.");
//  fma(exp(a)/c, d*exp(b), e) -> fma(exp(a + b), d/c, e)
    auto fmaexp15 = graph::fma(expa/exp_c, exp_d*expb, exp_e);
    auto fmaexp15_cast = graph::fma_cast(fmaexp15);
    assert(fmaexp15_cast.get() && "Expected a fma node.");
    assert(graph::exp_cast(fmaexp15_cast->get_left()).get() &&
           "Expected a exp node on the left.");
//  fma(c/exp(a), exp(b)*d, e) -> fma(exp(b - a), c*d, e)
    auto fmaexp16 = graph::fma(exp_c/expa, expb*exp_d, exp_e);
    auto fmaexp16_cast = graph::fma_cast(fmaexp16);
    assert(fmaexp16_cast.get() && "Expected a fma node.");
    assert(graph::exp_cast(fmaexp16_cast->get_left()).get() &&
           "Expected a exp node on the left.");
//  fma(c/exp(a), d*exp(b), e) -> fma(exp(b - a), c*d, e)
    auto fmaexp17 = graph::fma(exp_c/expa, exp_d*expb, exp_e);
    auto fmaexp17_cast = graph::fma_cast(fmaexp17);
    assert(fmaexp17_cast.get() && "Expected a fma node.");
    assert(graph::exp_cast(fmaexp17_cast->get_left()).get() &&
           "Expected a exp node on the left.");

//  fma(exp(a)/c, exp(b)/d, e) -> exp(a + b)/(c*d) + e
    auto fmaexp18 = graph::fma(expa/exp_c, expb/exp_d, exp_e);
    auto fmaexp18_cast = graph::add_cast(fmaexp18);
    assert(fmaexp18_cast.get() && "Expected an add node.");
    assert(graph::divide_cast(fmaexp18_cast->get_left()).get() &&
           "Expected a divide node on the left.");
//  fma(exp(a)/c, d/exp(b), e) -> fma(exp(a - b), d/c, e)
    auto fmaexp19 = graph::fma(expa/exp_c, exp_d/expb, exp_e);
    auto fmaexp19_cast = graph::fma_cast(fmaexp19);
    assert(fmaexp19_cast.get() && "Expected a fma node.");
    assert(graph::exp_cast(fmaexp19_cast->get_left()).get() &&
           "Expected a exp node on the left.");
//  fma(c/exp(a), exp(b)/d, e) -> fma(exp(b - a), c/d, e)
    auto fmaexp20 = graph::fma(exp_c/expa, expb/exp_d, exp_e);
    auto fmaexp20_cast = graph::fma_cast(fmaexp20);
    assert(fmaexp20_cast.get() && "Expected a fma node.");
    assert(graph::exp_cast(fmaexp20_cast->get_left()).get() &&
           "Expected a exp node on the left.");
//  fma(c/exp(a), d/exp(b), e) -> (c*d)/exp(a + b) + e
    auto fmaexp21 = graph::fma(exp_c/expa, exp_d/expb, exp_e);
    auto fmaexp21_cast = graph::add_cast(fmaexp21);
    assert(fmaexp21_cast.get() && "Expected an add node.");
    assert(graph::divide_cast(fmaexp21_cast->get_left()).get() &&
           "Expected a dive node on the left.");

//  fma(p2,p1,a) -> fma(p1,p2,a)
    auto p1 = graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (1.0),
                                                       static_cast<T> (2.0)}),
                                      var_a);
    auto p2 = graph::piecewise_2D<T> (std::vector<T> ({static_cast<T> (1.0),
                                                       static_cast<T> (2.0),
                                                       static_cast<T> (3.0),
                                                       static_cast<T> (4.0)}),
                                      2, var_b, var_c);
    auto fma_promote = graph::fma(p2, p1, var_a);
    auto fma_promote_cast = graph::fma_cast(fma_promote);
    assert(fma_promote_cast.get() && "Expected a fma node.");
    assert(graph::piecewise_1D_cast(fma_promote_cast->get_left()).get() &&
           "Expected a piecewise 1d node on the left.");
    assert(graph::piecewise_2D_cast(fma_promote_cast->get_middle()).get() &&
           "Expected a piecewise 2d node in the middle.");

//  fma(a,b,-c*d) -> a*b - c*d
    auto fma_to_sub = graph::fma(var_a,var_b,-1.0*var_c);
    auto fma_to_sub_cast = graph::subtract_cast(fma_to_sub);
    assert(fma_to_sub_cast.get() && "Expected a subtract node.");

//  Test common denominators.
//  fma(a/(b*c),d,e/c) -> fma(a,d,e*b)/(b*c)
    auto common_denom1 = graph::fma(var_a/(var_b*var_c), var_d, var_e/var_c);
    auto common_denom1_cast = graph::divide_cast(common_denom1);
    assert(common_denom1_cast.get() && "Expected a divide node.");
    assert(common_denom1_cast->get_right()->is_match(var_b*var_c) &&
           "Expected var_b*var_c as common denominator.");
    assert(common_denom1_cast->get_left()->is_match(graph::fma(var_a,
                                                               var_d,
                                                               var_e*var_b)) &&
           "Expected fma(a,d,e*b) as numerator.");
//  fma(a/(c*b),d,e/c) -> fma(a,d,e*b)/(c*b)
    auto common_denom2 = graph::fma(var_a/(var_c*var_b), var_d, var_e/var_c);
    auto common_denom2_cast = graph::divide_cast(common_denom2);
    assert(common_denom2_cast.get() && "Expected a divide node.");
    assert(common_denom2_cast->get_right()->is_match(var_c*var_b) &&
           "Expected var_b*var_c as common denominator.");
    assert(common_denom2_cast->get_left()->is_match(graph::fma(var_a,
                                                               var_d,
                                                               var_e*var_b)) &&
           "Expected fma(a,d,e*b) as numerator.");
//  fma(a/c,d,e/(c*b)) -> fma(a*b,d,e)/(b*c)
    auto common_denom3 = graph::fma(var_a/var_c, var_d, var_e/(var_b*var_c));
    auto common_denom3_cast = graph::divide_cast(common_denom3);
    assert(common_denom3_cast.get() && "Expected a divide node.");
    assert(common_denom3_cast->get_right()->is_match(var_b*var_c) &&
           "Expected var_b*var_c as common denominator.");
    assert(common_denom3_cast->get_left()->is_match(graph::fma(var_a*var_b,
                                                               var_d,
                                                               var_e)) &&
           "Expected fma(a*b,d,e) as numerator.");
//  fma(a/c,d,e/(b*c)) -> fma(a,d,e*b)/(c*b)
    auto common_denom4 = graph::fma(var_a/var_c, var_d, var_e/(var_c*var_b));
    auto common_denom4_cast = graph::divide_cast(common_denom4);
    assert(common_denom4_cast.get() && "Expected a divide node.");
    assert(common_denom4_cast->get_right()->is_match(var_c*var_b) &&
           "Expected var_b*var_c as common denominator.");
    assert(common_denom4_cast->get_left()->is_match(graph::fma(var_a*var_b,
                                                               var_d,
                                                               var_e)) &&
           "Expected fma(a*b,d,e) as numerator.");
//  fma(a,d/(b*c),e/c) -> fma(a,d,e*b)/(b*c)
    auto common_denom5 = graph::fma(var_a, var_d/(var_b*var_c), var_e/var_c);
    auto common_denom5_cast = graph::divide_cast(common_denom5);
    assert(common_denom5_cast.get() && "Expected a divide node.");
    assert(common_denom5_cast->get_right()->is_match(var_b*var_c) &&
           "Expected var_b*var_c as common denominator.");
    assert(common_denom5_cast->get_left()->is_match(graph::fma(var_a,
                                                               var_d,
                                                               var_e*var_b)) &&
           "Expected fma(a,d,e*b) as numerator.");
//  fma(a,d/(c*b),e/c) -> fma(a,d,e*b)/(c*b)
    auto common_denom6 = graph::fma(var_a, var_d/(var_c*var_b), var_e/var_c);
    auto common_denom6_cast = graph::divide_cast(common_denom6);
    assert(common_denom6_cast.get() && "Expected a divide node.");
    assert(common_denom6_cast->get_right()->is_match(var_c*var_b) &&
           "Expected var_b*var_c as common denominator.");
    assert(common_denom6_cast->get_left()->is_match(graph::fma(var_a,
                                                               var_d,
                                                               var_e*var_b)) &&
           "Expected fma(a,d,e*b) as numerator.");
//  fma(a,d/c,e/(b*c)) -> fma(a,d*b,e)/(b*c)
    auto common_denom7 = graph::fma(var_a, var_d/var_c, var_e/(var_b*var_c));
    auto common_denom7_cast = graph::divide_cast(common_denom7);
    assert(common_denom7_cast.get() && "Expected a divide node.");
    assert(common_denom7_cast->get_right()->is_match(var_b*var_c) &&
           "Expected var_b*var_c as common denominator.");
    assert(common_denom7_cast->get_left()->is_match(graph::fma(var_a,
                                                               var_d*var_b,
                                                               var_e)) &&
           "Expected fma(a,d*b,e) as numerator.");
//  fma(a,d/c,e/(c*b)) -> fma(a,d*b,e)/(c*b)
    auto common_denom8 = graph::fma(var_a, var_d/var_c, var_e/(var_c*var_b));
    auto common_denom8_cast = graph::divide_cast(common_denom8);
    assert(common_denom8_cast.get() && "Expected a divide node.");
    assert(common_denom8_cast->get_right()->is_match(var_c*var_b) &&
           "Expected var_b*var_c as common denominator.");
    assert(common_denom8_cast->get_left()->is_match(graph::fma(var_a,
                                                               var_d*var_b,
                                                               var_e)) &&
           "Expected fma(a,d*b,e) as numerator.");
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void run_tests() {
    test_add<T> ();
    test_subtract<T> ();
    test_multiply<T> ();
    test_divide<T> ();
    test_fma<T> ();
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
