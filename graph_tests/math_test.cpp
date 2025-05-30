//------------------------------------------------------------------------------
///  @file math\_test.cpp
///  @brief Tests for math nodes.
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
///  @brief Tests for sqrt nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
void test_sqrt() {
    auto ten = graph::constant(static_cast<T> (10.0));
    auto sqrt_ten = graph::sqrt(ten);
    assert(sqrt_ten->evaluate().at(0) == std::sqrt(static_cast<T> (10.0)));

    assert(graph::constant_cast(sqrt_ten).get() && "Expected a constant type.");

    auto var = graph::variable<T> (1, "");
    auto sqrt_var = graph::sqrt(var);
    assert(graph::sqrt_cast(sqrt_var).get() &&"Expected a variable type.");

    var->set(static_cast<T> (3.0));
    assert(sqrt_var->evaluate().at(0) == std::sqrt(static_cast<T> (3.0)));

    auto var_vec = graph::variable<T> (std::vector<T> ({4.0, 7.0}), "");
    auto sqrt_var_vec = graph::sqrt(var_vec);
    assert(graph::sqrt_cast(sqrt_var_vec).get() && "Expected a variable type.");
    const backend::buffer<T> sqrt_var_vec_result = sqrt_var_vec->evaluate();
    assert(sqrt_var_vec_result.size() == 2);
    assert(sqrt_var_vec_result.at(0) == std::sqrt(static_cast<T> (4.0)));
    assert(sqrt_var_vec_result.at(1) == std::sqrt(static_cast<T> (7.0)));

//  d sqrt(x) / dx = 1/(2 Sqrt(x))
    auto dsqrt_var = sqrt_var->df(var);
    assert(graph::divide_cast(dsqrt_var).get() && "Expected a divide type.");
    assert(dsqrt_var->evaluate().at(0) ==
           static_cast<T> (1.0/2.0) /
           std::sqrt(static_cast<T> (3.0)) &&
           "Expected 0.5*sqrt(3)");

//  Reduction sqrt(c*x*c*y) = c*Sqrt(x*y)
    auto x1 = 2.0*graph::variable<T> (1, "");
    auto x2 = 3.0*graph::variable<T> (1, "");
    auto x = graph::sqrt(x1*x2);
    auto x_cast = graph::multiply_cast(x);
    assert(x_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(x_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::sqrt_cast(x_cast->get_right()).get() &&
           "Expected sqrt node.");
    
//  Reduction Sqrt(x*x) = x
    auto x_var = graph::variable<T> (1, "x");
    auto x2_sqrt = graph::sqrt(x_var*x_var);
    assert(x2_sqrt.get() != x_var.get() && "Expected not to reduce to x_var.");

//  Reduction Sqrt(x*y*x*y) = sqrt((x*y)^2)
    auto y_var = graph::variable<T> (1, "y");
    auto x2y2_sqrt = graph::sqrt(x_var*y_var*x_var*y_var);
    auto x2y2_sqrt_cast = graph::sqrt_cast(x2y2_sqrt);
    assert(x2y2_sqrt_cast.get() && "Expected sqrt node");
    assert(x2y2_sqrt_cast->get_arg()->is_match(graph::pow(y_var*x_var, 2.0)) &&
           "Expected (x_var*y_var)^2.");

//  Reduction Sqrt(x*x/y*y);
    auto sq_reduce = graph::sqrt((x_var*x_var)/(y_var*y_var));
    auto sq_reduce_cast = graph::sqrt_cast(sq_reduce);
    assert(sq_reduce_cast.get() && "Expected sqrt node.");
    assert(sq_reduce_cast->get_arg()->is_match(graph::pow(x_var/y_var, 2.0)) &&
           "Expected (x_var/y_var)^2.");

//  Reduction Sqrt(c*x/b*y) = d*Sqrt(x/y)
    auto cxby_sqrt = graph::sqrt(x1/x2);
    auto cxby_sqrt_cast = graph::multiply_cast(cxby_sqrt);
    assert(cxby_sqrt_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(cxby_sqrt_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::sqrt_cast(cxby_sqrt_cast->get_right()).get() &&
           "Expected sqrt node.");

#if 0
//  Reduction Sqrt(x^a) -> x^(a/2)
    auto sqpow = graph::sqrt(graph::pow(x_var, y_var));
    auto sqpow_cast = graph::pow_cast(sqpow);
    assert(sqpow_cast.get() && "Expected pow node.");
    auto exp_cast = graph::multiply_cast(sqpow_cast->get_right());
    assert(exp_cast.get() && "Expected a mutliply node.");
    auto constant_cast = graph::constant_cast(exp_cast->get_left());
    assert(constant_cast.get() && "Expected constant node on the left.");
    assert(constant_cast->is(0.5) && "Expected a value of 0.5");
#endif
//  Test node properties.
    auto sqrt_const = graph::sqrt(graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (1.0),
                                                                           static_cast<T> (2.0)}),
                                                          var, 1.0, 0.0));
    assert(sqrt_const->is_constant() && "Expected a constant.");
    assert(!sqrt_const->is_all_variables() && "Did not expect a variable.");
    assert(sqrt_const->is_power_like() && "Expected a power like.");
    assert(!sqrt_var->is_constant() && "Did not expect a constant.");
    assert(sqrt_var->is_all_variables() && "Expected a variable.");
    assert(sqrt_var->is_power_like() && "Expected a power like.");
}

//------------------------------------------------------------------------------
///  @brief Tests for exp nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
void test_exp() {
    auto ten = graph::constant(static_cast<T> (10.0));
    auto exp_ten = graph::exp(ten);
    assert(exp_ten->evaluate().at(0) == std::exp(static_cast<T> (10.0)));
    assert(graph::constant_cast(exp_ten).get() &&
           "Expected a constant type.");

    auto var = graph::variable<T> (1, "");
    auto exp_var = graph::exp(var);
    assert(graph::exp_cast(exp_var).get() && "Expected a variable type.");

    var->set(static_cast<T> (3.0));
    assert(exp_var->evaluate().at(0) == std::exp(static_cast<T> (3.0)));

    auto var_vec = graph::variable<T> (std::vector<T> ({4.0, 7.0}), "");
    auto exp_var_vec = graph::exp(var_vec);
    assert(graph::exp_cast(exp_var_vec).get() && "Expected a variable type.");
    const backend::buffer<T> exp_var_vec_result = exp_var_vec->evaluate();
    assert(exp_var_vec_result.size() == 2);
    assert(exp_var_vec_result.at(0) == std::exp(static_cast<T> (4.0)));
    assert(exp_var_vec_result.at(1) == std::exp(static_cast<T> (7.0)));

//  d exp(x) / dx = exp(x)
    auto dexp_var = exp_var->df(var);
    assert(graph::exp_cast(dexp_var).get() && "Expected a exp node.");
    assert(dexp_var->evaluate().at(0) == std::exp(static_cast<T> (3.0)));

//  Test node properties.
    assert(!exp_var->is_constant() && "Did not expect a constant.");
    assert(exp_var->is_all_variables() && "Expected a variable.");
    assert(!exp_var->is_power_like() && "Did not expect a power like.");
}

//------------------------------------------------------------------------------
///  @brief Tests for pow nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
void test_pow() {
//  a^0 = 1
    auto zero = graph::zero<T> ();
    auto ten = graph::variable<T> (1, "10");
    ten->set(static_cast<T> (10.0));
    auto one = graph::pow(ten, zero);
    assert(graph::constant_cast(one).get() && "Expected constant");
    assert(one->evaluate().at(0) == static_cast<T> (1.0) &&
           "Expected 1.");

//  a^1 = a
    assert(graph::pow(ten, one).get() == ten.get() && "Expected ten.");
    assert(graph::pow(ten, one)->evaluate().at(0) == static_cast<T> (10.0) &&
           "Expected ten.");

//  Sqrt(a)^2 = a
    assert(graph::pow(graph::sqrt(ten), 2.0)->is_match(ten) &&
           "Expected ten.");
    assert(graph::pow(graph::sqrt(ten), 2.0)->evaluate().at(0) == static_cast<T> (10.0) &&
           "Expected ten.");

//  (Sqrt(a))^b -> a^(b/2)
    auto powsq = graph::pow(graph::sqrt(ten), ten);
    auto powsq_cast = graph::pow_cast(powsq);
    assert(powsq_cast.get() && "Expected pow node.");
    auto exp_cast = graph::multiply_cast(powsq_cast->get_right());
    assert(exp_cast.get() && "Expected a mutliply node.");
    auto constant_cast = graph::constant_cast(exp_cast->get_left());
    assert(constant_cast.get() && "Expected constant node on the left.");
    assert(constant_cast->is(0.5) && "Expected a value of 0.5");

//  (c1*Sqrt(b))^c2 -> c3*b^c2/2
    assert(graph::multiply_cast(graph::pow(2.0*graph::sqrt(ten), 10.0)).get() &&
           "Expected multiply node.");
//  (Sqrt(b)*c)^a -> c^a*b^a/2
    assert(graph::multiply_cast(graph::pow(graph::sqrt(ten)*2.0, 10.0)).get() &&
           "Expected multiply node.");

//  (c*b^d)^a -> c^a*b^(a*d)
    assert(graph::multiply_cast(graph::pow(2.0*graph::pow(ten, 2.0), 10.0)).get() &&
           "Expected multiply node.");
//  ((b^d)*c)^a -> b^(a*d)*c^a
    assert(graph::multiply_cast(graph::pow(graph::pow(ten, 2.0)*2.0, 10.0)).get() &&
           "Expected multiply node.");

//  (c/Sqrt(b))^a -> c^a/b^a/2
    assert(graph::divide_cast(graph::pow(2.0/graph::sqrt(ten), 10.0)).get() &&
           "Expected divide node.");
//  (Sqrt(b)/c)^a -> (b^a/2)/c^a -> c2*b^a
    assert(graph::multiply_cast(graph::pow(graph::sqrt(ten)/2.0, 10.0)).get() &&
           "Expected multiply node.");

//  (c/(b^d))^a -> c^a/(b^(a*d))
    assert(graph::divide_cast(graph::pow(2.0/graph::pow(ten, 2.0), 10.0)).get() &&
           "Expected divide node.");
//  ((b^d)/c))^a -> (b^(a*d))/c^a -> c2*b^a
    assert(graph::multiply_cast(graph::pow(graph::pow(ten, 2.0)/2.0, 10.0)).get() &&
           "Expected multiply node.");

//  a^1/2 -> sqrt(a);
    assert(graph::sqrt_cast(graph::pow(ten, one/2.0)).get() &&
           "Expected sqrt node.");

    auto hundred = graph::pow(ten, 2.0);
    assert(hundred->evaluate().at(0) == static_cast<T> (100.0) &&
           "Expected 100");
    const auto non_int = static_cast<T> (0.438763);
    auto sqrd = graph::pow(graph::constant(non_int), 2.0);
    assert(sqrd->evaluate().at(0) == static_cast<T> (non_int*non_int) &&
           "Expected x*x");
    const auto non_int_neg = static_cast<T> (-0.438763);
    auto sqrd_neg = graph::pow(graph::constant(non_int_neg), 2.0);
    assert(sqrd_neg->evaluate().at(0) == static_cast<T> (non_int_neg*non_int_neg) &&
           "Expected x*x");

    auto pow_pow1 = graph::pow(graph::pow(ten, 3.0), 2.0);
    auto pow_pow2 = graph::pow(ten, 6.0);
    assert(pow_pow1->is_match(pow_pow2) &&
           "Expected ten to the 6.");

    assert(graph::multiply_cast(graph::pow(2.0*ten, 2.0)).get() &&
           "Expected multiply node.");
    assert(graph::multiply_cast(graph::pow(ten*2.0, 2.0)).get() &&
           "Expected multiply node.");
    assert(graph::divide_cast(graph::pow(2.0/ten, 2.0)).get() &&
           "Expected divide node.");
// (v/c)^a -> v^a/c^a -> c2*v^a
    assert(graph::multiply_cast(graph::pow(ten/2.0, 2.0)).get() &&
           "Expected multiply node.");

//  sqrt(a)^a -> a^(b/c) -> a^(c2*b)
    auto pow_sqrt = graph::pow_cast(graph::pow(graph::sqrt(ten), ten));
    assert(graph::multiply_cast(pow_sqrt->get_right()).get() &&
           "Expected mutliply node.");

//  Test derivatives.
    auto x2 = graph::pow(ten, 2.0);
    auto dx2dx = x2->df(ten);
    assert(graph::multiply_cast(dx2dx).get() && "Expected multiply node.");

    auto x3 = graph::pow(2.0, ten);
    auto dx3dx = x3->df(ten);
    assert(graph::multiply_cast(dx3dx).get() && "Expected multiply node.");

//  Test node properties.
    auto var_a = graph::variable<T> (1, "");
    auto pow_const = graph::pow(3.0, graph::piecewise_1D<T> (std::vector<T> ({static_cast<T> (1.0),
                                                                              static_cast<T> (2.0)}),
                                                             var_a, 1.0, 0.0));
    assert(pow_const->is_constant() && "Expected a constant.");
    assert(!pow_const->is_all_variables() && "Did not expect a variable.");
    assert(pow_const->is_power_like() && "Expected a power like.");
    auto pow_var = graph::pow(var_a, 3.0);
    assert(!pow_var->is_constant() && "Did not expect a constant.");
    assert(pow_var->is_all_variables() && "Expected a variable.");
    assert(pow_var->is_power_like() && "Expected a power like.");
    auto var_b = graph::variable<T> (1, "");
    auto pow_var_var = graph::pow(var_a, var_b);
    assert(!pow_var->is_constant() && "Did not expect a constant.");
    assert(pow_var->is_all_variables() && "Expected a variable.");
    assert(pow_var->is_power_like() && "Expected a power like.");

//  Test power of power
//  (a^b)^n -> a^n*b when n is an integer.
    auto powpow_int = graph::pow(graph::pow(var_a, var_b),
                                 graph::constant<T> (static_cast<T> (3.0)));
    auto powpow_int_cast = graph::pow_cast(powpow_int);
    assert((powpow_int_cast.get() &&
            graph::multiply_cast(powpow_int_cast->get_right())) &&
           "Expected multiply node.");
    auto powpow_float =  graph::pow(graph::pow(var_a, var_b), 1.5);
    auto powpow_float_cast = graph::pow_cast(powpow_float);
    assert((powpow_int_cast.get() &&
            !graph::multiply_cast(powpow_float_cast->get_right())) &&
           "Did not expect multiply node.");
    auto powpow_var =  graph::pow(graph::pow(var_a, var_b),
                                  ten);
    auto powpow_var_cast = graph::pow_cast(powpow_var);
    assert((powpow_int_cast.get() &&
            !graph::multiply_cast(powpow_var_cast->get_right())) &&
           "Did not expect multiply node.");

//  (a/(b*Sqrt(c)))^2 -> a^2/(b^2*c)
    auto var_c = graph::variable<T> (1, "");
    auto denomsqrt = graph::pow(var_a/(var_b*graph::sqrt(var_c)),
                                graph::constant<T> (static_cast<T> (2.0)));
    assert(graph::divide_cast(denomsqrt).get() &&
           "Expected a divide node.");

//  Test pow of exp
//  Exp[x]^n -> Exp[n*x] when n is an integer.
    auto powexp_int = graph::pow(graph::exp(var_a), 3.0);
    auto powexp_int_cast = graph::exp_cast(powexp_int);
    assert((powexp_int_cast.get() &&
            graph::multiply_cast(powexp_int_cast->get_arg())) &&
           "Expected multiply node in exp argument.");
    auto powexp_float = graph::pow(graph::exp(var_a), 1.5);
    auto powexp_float_cast = graph::pow_cast(powexp_float);
    assert(powexp_float_cast.get() &&
           "Expected power cast.");

//  c1^c2
    assert(graph::constant_cast(graph::pow(graph::constant<T> (static_cast<T> (2.0)),
                                           graph::constant<T> (static_cast<T> (3.0)))).get() &&
           "Expected a constant node.");

//  (-a/b)^2 -> (a/b)^2
    assert(graph::pow(-var_a/var_b, 2.0)->is_match(graph::pow(var_a/var_b, 2.0)) &&
           "Expected (a/b)^2.");
//  (-a/b)^4 -> (a/b)^4
    assert(graph::pow(-var_a/var_b, 4.0)->is_match(graph::pow(var_a/var_b, 4.0)) &&
           "Expected (a/b)^4.");
//  (-a/b)^3 -> -(a/b)^3
    assert(graph::pow(-var_a/var_b, 3.0)->is_match(-graph::pow(var_a, 3.0)/graph::pow(var_b, 3.0)) &&
           "Expected -(a/b)^3");

//  (a*sqrt(b^2))^2-> (a*b)^2
    assert(graph::pow(var_a*graph::sqrt(var_b*var_b), 2.0)->is_match(var_a*var_a*var_b*var_b) &&
           "Expected (a*b)^2.");
//  (a*sqrt(b^2))^4-> (a*b)^4
    assert(graph::pow(var_a*graph::sqrt(var_b*var_b), 4.0)->is_match(graph::pow(var_a, 4.0)*graph::pow(var_b, 4.0)) &&
           "Expected (a*b)^4.");
//  (a*sqrt(b^2)/c)^2-> (a*b/c)^2
    assert(graph::pow(var_a*graph::sqrt(var_b*var_b)/var_c, 2.0)->is_match(var_a*var_a*var_b*var_b/(var_c*var_c)) &&
           "Expected (a*b/c)^2.");

//  (a^b*c/Sqrt(a)d)^2 -> (a^(2*b - 1)*c^2)/d^2
    auto var_d = graph::variable<T> (1, "");
    auto pow_combine = graph::pow((graph::pow(var_a, var_b)*var_c) /
                                  (graph::sqrt(var_a)*var_d), 2.0);
    auto pow_combine_cast = graph::divide_cast(pow_combine);
    assert(pow_combine_cast.get() && "Expected a divide node.");
    assert(pow_combine_cast->get_right()->is_match(graph::pow(var_d, 2.0)) &&
           "Expected d^2 in deniminator.");
    assert(pow_combine_cast->get_left()->is_match(graph::pow(var_a,
                                                             2.0*(var_b - 0.5)) *
                                                  var_c*var_c) &&
           "Expected a^(2*b - 1)*c^2 in the numerator.");

//  (b/(c*Sqrt(a)*a))^2 -> b^2/(a^4*c^2)
    auto expr_a = 1.0 - var_a;
    auto expr_b = 1.0 + var_b;
    auto expr_c = 1.0 - var_c;
    auto pow_combine2 = graph::pow(expr_b/(expr_c*expr_a*graph::sqrt(expr_a*expr_a)), 2.0);
    auto pow_combine2_cast = graph::divide_cast(pow_combine2);
    assert(pow_combine2_cast.get() && "Expected a divide node.");
    assert(pow_combine2_cast->get_right()->is_match(graph::pow(expr_c, 2.0)*
                                                    graph::pow(expr_a, 4.0)) &&
           "Expected a^4 in the denominator.");
    assert(pow_combine2_cast->get_left()->is_match(graph::pow(expr_b, 2.0)) &&
           "Expected (b/c)^4 in the numerator.");
    auto pow_combine3 = graph::pow(expr_b/(expr_c*graph::sqrt(expr_a*expr_a)*expr_a), 2.0);
    auto pow_combine3_cast = graph::divide_cast(pow_combine3);
    assert(pow_combine3_cast.get() && "Expected a divide node.");
    assert(pow_combine3_cast->get_left()->is_match(graph::pow(expr_b, 2.0)) &&
           "Expected b^2.");
    assert(pow_combine3_cast->get_right()->is_match(graph::pow(expr_a, 4.0) *
                                                    graph::pow(expr_c, 2.0)) &&
           "Expected a^4*c^2.");
    auto pow_combine4 = graph::pow(expr_b/(graph::sqrt(expr_a*expr_a)*expr_c*expr_a), 2.0);
    auto pow_combine4_cast = graph::divide_cast(pow_combine4);
    assert(pow_combine4_cast.get() && "Expected a divide node.");
    assert(pow_combine4_cast->get_left()->is_match(graph::pow(expr_b, 2.0)) &&
           "Expected b^2.");
    assert(pow_combine4_cast->get_right()->is_match(graph::pow(expr_a, 4.0) *
                                                    graph::pow(expr_c, 2.0)) &&
           "Expected a^4*c^2.");
    auto pow_combine5 = graph::pow(expr_b/(expr_a*graph::sqrt(expr_a*expr_a)*expr_c), 2.0);
    auto pow_combine5_cast = graph::divide_cast(pow_combine5);
    assert(pow_combine5_cast.get() && "Expected a divide node.");
    assert(pow_combine5_cast->get_left()->is_match(graph::pow(expr_b, 2.0)) &&
           "Expected b^2.");
    assert(pow_combine5_cast->get_right()->is_match(graph::pow(expr_a, 4.0) *
                                                    graph::pow(expr_c, 2.0)) &&
           "Expected a^4*c^2.");
    auto pow_combine6 = graph::pow(expr_b/(graph::sqrt(expr_a*expr_a)*expr_a*expr_c), 2.0);
    auto pow_combine6_cast = graph::divide_cast(pow_combine6);
    assert(pow_combine6_cast.get() && "Expected a divide node.");
    assert(pow_combine6_cast->get_left()->is_match(graph::pow(expr_b, 2.0)) &&
           "Expected b^2.");
    assert(pow_combine6_cast->get_right()->is_match(graph::pow(expr_a, 4.0) *
                                                    graph::pow(expr_c, 2.0)) &&
           "Expected a^4*c^2.");

//  (Sqrt(a)*b*c)^d -> a^(d/2)*(b*c)^d
    auto sqrtpow = graph::pow(var_c*var_d*graph::sqrt(var_a), 10.0);
    assert(sqrtpow.get()->is_match(graph::pow(var_a, 5.0) *
                                   graph::pow(var_c*var_d, 10.0)) &&
           "Expected a^(d/2)*(c*b)^d.");

    auto factorconst = graph::pow(-0.5*var_a/var_b, 2.0);
    auto factorconst_cast = graph::multiply_cast(factorconst);
    assert(factorconst_cast.get() && "Expected a multiply node.");
    assert(factorconst_cast->get_left()->is_match(graph::constant<T> (static_cast<T> (0.25))) &&
           "Expected 0.25 on the left.");
    assert(factorconst_cast->get_right()->is_match(graph::pow(var_a/var_b, 2.0)) &&
           "Expected (a/b)^2 on the right.");

//  pow(a/((sqrt(b)*c)*d), 2) -> pow(a,2)/(b*pow(c,2)*pow(d,2))
    auto divid_sqrt = graph::pow(var_a/((graph::sqrt(var_b)*var_c)*var_d), 2.0);
    auto divid_sqrt_cast = graph::divide_cast(divid_sqrt);
    assert(divid_sqrt_cast.get() && "Expected a divide node.");
    assert(divid_sqrt_cast->get_left()->is_match(graph::pow(var_a, 2.0)) &&
           "Expected a^2.");
    assert(divid_sqrt_cast->get_right()->is_match(graph::pow(var_c, 2.0) *
                                                  graph::pow(var_d, 2.0) *
                                                  var_b) &&
           "Expected c^2*d^2*b.");
//  pow(a/((c*sqrt(b))*d), 2) -> pow(a,2)/(b*pow(c,2)*pow(d,2))
    auto divid_sqrt2 = graph::pow(var_a/((var_c*graph::sqrt(var_b))*var_d), 2.0);
    auto divid_sqrt2_cast = graph::divide_cast(divid_sqrt2);
    assert(divid_sqrt2_cast.get() && "Expected a divide node.");
    assert(divid_sqrt2_cast->get_left()->is_match(graph::pow(var_a, 2.0)) &&
           "Expected a^2.");
    assert(divid_sqrt2_cast->get_right()->is_match(graph::pow(var_c, 2.0) *
                                                   graph::pow(var_d, 2.0) *
                                                   var_b) &&
           "Expected c^2*d^2*b.");
//  pow(((sqrt(b)*c)*d)/a, 2) -> (b*pow(c,2)*pow(d,2))/pow(a,2)
    auto divid_sqrt3 = graph::pow(((graph::sqrt(var_b)*var_c)*var_d)/var_a, 2.0);
    auto divid_sqrt3_cast = graph::divide_cast(divid_sqrt3);
    assert(divid_sqrt3_cast.get() && "Expected a divide node.");
    assert(divid_sqrt3_cast->get_right()->is_match(graph::pow(var_a, 2.0)) &&
           "Expected a^2.");
    assert(divid_sqrt3_cast->get_left()->is_match(graph::pow(var_c, 2.0) *
                                                  graph::pow(var_d, 2.0) *
                                                  var_b) &&
           "Expected c^2*d^2*b.");
//  pow(((sqrt(b)*c)*d)/a, 2) -> (b*pow(c,2)*pow(d,2))/pow(a,2)
    auto divid_sqrt4 = graph::pow(((var_c*graph::sqrt(var_b))*var_d)/var_a, 2.0);
    auto divid_sqrt4_cast = graph::divide_cast(divid_sqrt4);
    assert(divid_sqrt4_cast.get() && "Expected a divide node.");
    assert(divid_sqrt4_cast->get_right()->is_match(graph::pow(var_a, 2.0)) &&
           "Expected a^2.");
    assert(divid_sqrt4_cast->get_left()->is_match(graph::pow(var_c, 2.0) *
                                                  graph::pow(var_d, 2.0) *
                                                  var_b) &&
           "Expected c^2*d^2*b.");

//  pow(a/(c*(sqrt(b)*d)), 2) -> pow(a,2)/(b*pow(c,2)*pow(d,2))
    auto divid_sqrt5 = graph::pow(var_a/(expr_c*(graph::sqrt(expr_b)*expr_a)), 2.0);
    auto divid_sqrt5_cast = graph::divide_cast(divid_sqrt5);
    assert(divid_sqrt5_cast.get() && "Expected a divide node.");
    assert(divid_sqrt5_cast->get_left()->is_match(graph::pow(var_a, 2.0)) &&
           "Expected a^2.");
    assert(divid_sqrt5_cast->get_right()->is_match(expr_b*
                                                   graph::pow(expr_a, 2.0) *
                                                   graph::pow(expr_c, 2.0)) &&
           "Expected b*c^2*d^2.");
//  pow(a/(c(d**sqrt(b))), 2) -> pow(a,2)/(b*pow(c,2)*pow(d,2))
    auto divid_sqrt6 = graph::pow(var_a/(expr_c*(expr_a*graph::sqrt(expr_b))), 2.0);
    auto divid_sqrt6_cast = graph::divide_cast(divid_sqrt6);
    assert(divid_sqrt6_cast.get() && "Expected a divide node.");
    assert(divid_sqrt6_cast->get_left()->is_match(graph::pow(var_a, 2.0)) &&
           "Expected a^2.");
    assert(divid_sqrt6_cast->get_right()->is_match(expr_b*
                                                   graph::pow(expr_a, 2.0) *
                                                   graph::pow(expr_c, 2.0)) &&
           "Expected b*c^2*d^2.");
//  pow(((c*sqrt(b))*d), 2)/a -> pow(a,2)/(b*pow(c,2)*pow(d,2))
    auto divid_sqrt7 = graph::pow((expr_c*(graph::sqrt(expr_b)*expr_a)/var_a), 2.0);
    auto divid_sqrt7_cast = graph::divide_cast(divid_sqrt7);
    assert(divid_sqrt7_cast.get() && "Expected a divide node.");
    assert(divid_sqrt7_cast->get_right()->is_match(graph::pow(var_a, 2.0)) &&
           "Expected a^2.");
    assert(divid_sqrt7_cast->get_left()->is_match(expr_b*
                                                  graph::pow(expr_a, 2.0) *
                                                  graph::pow(expr_c, 2.0)) &&
           "Expected b*c^2*d^2.");
//  pow(((c*sqrt(b))*d), 2)/a -> pow(a,2)/(b*pow(c,2)*pow(d,2))
    auto divid_sqrt8 = graph::pow((expr_c*(expr_a*graph::sqrt(expr_b))/var_a), 2.0);
    auto divid_sqrt8_cast = graph::divide_cast(divid_sqrt8);
    assert(divid_sqrt8_cast.get() && "Expected a divide node.");
    assert(divid_sqrt8_cast->get_right()->is_match(graph::pow(var_a, 2.0)) &&
           "Expected a^2.");
    assert(divid_sqrt8_cast->get_left()->is_match(expr_b*
                                                  graph::pow(expr_a, 2.0) *
                                                  graph::pow(expr_c, 2.0)) &&
           "Expected b*c^2*d^2.");
}

//------------------------------------------------------------------------------
///  @brief Tests for log nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
void test_log() {
    assert(graph::constant_cast(graph::log(graph::constant(static_cast<T> (10.0)))) &&
           "Expected constant");

    auto y = graph::variable<T> (1, "y");
    auto logy = graph::log(y);
    assert(graph::log_cast(logy) && "Expected log");

//  Test derivatives.
    auto dlogy = logy->df(y);
    assert(graph::divide_cast(dlogy) && "Expected divide node.");

    assert(!logy->is_constant() && "Did not expect a constant.");
    assert(logy->is_all_variables() && "Expected a variable.");
    assert(!logy->is_power_like() && "Did not expect a power like.");
}

//------------------------------------------------------------------------------
///  @brief Tests for log nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::complex_scalar T>
void test_erfi() {
    auto a = graph::variable<T> (1, "");
    auto erfi = graph::erfi(a);
    
    assert(graph::erfi_cast(erfi) &&
           "Expected an erfi node.");

    auto derfida = erfi->df(a);
    assert(graph::multiply_cast(derfida) &&
           "Expected a multiply node.");
    
    auto erfic = graph::erfi(graph::one<T> ());
    assert(graph::constant_cast(erfic) &&
           "Expected a constant node.");

//  Test node properties.
    assert(!erfi->is_constant() && "Did not expect a constant.");
    assert(erfi->is_all_variables() && "Expected a variable.");
    assert(!erfi->is_power_like() && "Did not expect a power like.");
}

//------------------------------------------------------------------------------
///  @brief Tests function for variable like expressions.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_variable_like() {
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
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void run_tests() {
    test_variable_like<T> ();
    test_sqrt<T> ();
    test_exp<T> ();
    test_pow<T> ();
    test_log<T> ();
    if constexpr (jit::complex_scalar<T>) {
        test_erfi<T> ();
    }
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
