//------------------------------------------------------------------------------
///  @file math_test.cpp
///  @brief Tests for math nodes.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/backend.hpp"
#include "../graph_framework/math.hpp"

//------------------------------------------------------------------------------
///  @brief Tests for sqrt nodes.
//------------------------------------------------------------------------------
template<typename T>
void test_sqrt() {
    auto ten = graph::constant(static_cast<T> (10.0));
    auto sqrt_ten = graph::sqrt(ten);
    assert(sqrt_ten->evaluate().at(0) == std::sqrt(static_cast<T> (10.0)));

#ifdef USE_REDUCE
    assert(graph::constant_cast(sqrt_ten).get() && "Expected a constant type.");
#else
    assert(graph::sqrt_cast(sqrt_ten).get() && "Expected a sqrt node.");
#endif

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
    auto x1 = graph::constant(static_cast<T> (2.0))*graph::variable<T> (1, "x");
    auto x2 = graph::constant(static_cast<T> (3.0))*graph::variable<T> (1, "y");
    auto x = graph::sqrt(x1*x2);
#ifdef USE_REDUCE
    auto x_cast = graph::multiply_cast(x);
    assert(x_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(x_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::sqrt_cast(x_cast->get_right()).get() &&
           "Expected sqrt node.");
#else
    assert(graph::sqrt_cast(x).get() && "Expected a sqrt node.");
#endif
    
//  Reduction Sqrt(x*x) = x
    auto x_var = graph::variable<T> (1, "x");
    auto x2_sqrt = graph::sqrt(x_var*x_var);
#ifdef USE_REDUCE
    assert(x2_sqrt.get() == x_var.get() && "Expected to reduce to x_var.");
#else
    assert(graph::sqrt_cast(x2_sqrt).get() && "Expected a sqrt node.");
#endif

//  Reduction Sqrt(x*y*x*y) = x*y
    auto y_var = graph::variable<T> (1, "y");
    auto x2y2_sqrt = graph::sqrt(x_var*y_var*x_var*y_var);
#ifdef USE_REDUCE
    auto x2y2_sqrt_cast = graph::multiply_cast(x2y2_sqrt);
    assert(x2y2_sqrt_cast.get() && "Expected multiply node");
    assert((x2y2_sqrt_cast->get_left().get() == x_var.get() ||
            x2y2_sqrt_cast->get_left().get() == y_var.get()) &&
           "Expected x_var or y_var.");
    assert((x2y2_sqrt_cast->get_right().get() == x_var.get() ||
            x2y2_sqrt_cast->get_right().get() == y_var.get()) &&
           "Expected x_var or y_var.");
#else
    assert(graph::sqrt_cast(x2_sqrt).get() && "Expected a sqrt node.");
#endif

//  Reduction Sqrt(x*x/y*y);
    auto sq_reduce = graph::sqrt((x_var*x_var)/(y_var*y_var));
#ifdef USE_REDUCE
    auto sq_reduce_cast = graph::divide_cast(sq_reduce);
    assert(sq_reduce_cast.get() && "Expected divide node.");
    assert(sq_reduce_cast->get_left().get() == x_var.get() &&
           "Expected x_var.");
    assert(sq_reduce_cast->get_right().get() == y_var.get() &&
           "Expected y_var.");
#else
    assert(graph::sqrt_cast(sq_reduce).get() && "Expected a sqrt node.");
#endif

//  Reduction Sqrt(c*x/b*y) = d*Sqrt(x/y)
    auto cxby_sqrt = graph::sqrt(x1/x2);
#ifdef USE_REDUCE
    auto cxby_sqrt_cast = graph::multiply_cast(cxby_sqrt);
    assert(cxby_sqrt_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(cxby_sqrt_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::sqrt_cast(cxby_sqrt_cast->get_right()).get() &&
           "Expected sqrt node.");
#else
    assert(graph::sqrt_cast(cxby_sqrt).get() && "Expected a sqrt node.");
#endif
}

//------------------------------------------------------------------------------
///  @brief Tests for exp nodes.
//------------------------------------------------------------------------------
template<typename T>
void test_exp() {
    auto ten = graph::constant(static_cast<T> (10.0));
    auto exp_ten = graph::exp(ten);
    assert(exp_ten->evaluate().at(0) == std::exp(static_cast<T> (10.0)));
#ifdef USE_REDUCE
    assert(graph::constant_cast(exp_ten).get() &&
           "Expected a constant type.");
#else
    assert(graph::exp_cast(exp_ten).get() && "Expected a sqrt node.");
#endif

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
#ifdef USE_REDUCE
    assert(graph::exp_cast(dexp_var).get() && "Expected a exp node.");
#else
    auto dexp_var_cast = graph::multiply_cast(dexp_var);
    assert(dexp_var_cast.get() && "Expected a multiply node.");
    assert(graph::exp_cast(dexp_var_cast->get_left()).get() &&
           "Expected a exp node.");
    assert(graph::constant_cast(dexp_var_cast->get_right()).get() &&
           "Expected a constant node.");
#endif
    assert(dexp_var->evaluate().at(0) == std::exp(static_cast<T> (3.0)));
}

//------------------------------------------------------------------------------
///  @brief Tests for pow nodes.
//------------------------------------------------------------------------------
template<typename T>
void test_pow() {
//  a^0 = 1
    auto zero = graph::constant(static_cast<T> (0.0));
    auto ten = graph::variable<T> (1, "10");
    ten->set(static_cast<T> (10.0));
    auto one = graph::pow(ten, zero);
#ifdef USE_REDUCE
    assert(graph::constant_cast(one).get() && "Expected constant");
#else
    assert(graph::pow_cast(one).get() && "Expected a pow node.");
#endif
    assert(one->evaluate().at(0) == static_cast<T> (1.0) &&
           "Expected 1.");

//  a^1 = a
#ifdef USE_REDUCE
    assert(graph::pow(ten, one).get() == ten.get() && "Expected ten.");
#else
    assert(graph::pow_cast(graph::pow(ten, one)).get() &&
           "Expected a pow node.");
#endif
    assert(graph::pow(ten, one)->evaluate().at(0) == static_cast<T> (10.0) &&
           "Expected ten.");

//  Sqrt(a)^2 = a
    auto two = graph::constant(static_cast<T> (2.0));
#ifdef USE_REDUCE
    assert(graph::pow(graph::sqrt(ten), two)->is_match(ten) &&
           "Expected ten.");
    assert(graph::pow(graph::sqrt(ten), two)->evaluate().at(0) == static_cast<T> (10.0) &&
           "Expected ten.");
#else
    assert(graph::pow_cast(graph::pow(graph::sqrt(ten), two)).get() &&
           "Expected a pow node.");
    assert(graph::pow(graph::sqrt(ten), two)->evaluate().at(0) == static_cast<T> (std::pow(std::sqrt(10.0), 2.0)) &&
           "Expected ten.");
#endif

//  (c*Sqrt(b))^a -> c^a*b^a/2
#ifdef USE_REDUCE
    assert(graph::multiply_cast(graph::pow(two*graph::sqrt(ten), ten)).get() &&
           "Expected multiply node.");
//  (Sqrt(b)*c)^a -> c^a*b^a/2
    assert(graph::multiply_cast(graph::pow(graph::sqrt(ten)*two, ten)).get() &&
           "Expected multiply node.");
#else
    assert(graph::pow_cast(graph::pow(two*graph::sqrt(ten), ten)).get() &&
           "Expected a pow node.");
    assert(graph::pow_cast(graph::pow(graph::sqrt(ten)*two, ten)).get() &&
           "Expected a pow node.");
#endif

//  (c*b^d)^a -> c^a*b^(a*d)
#ifdef USE_REDUCE
    assert(graph::multiply_cast(graph::pow(two*graph::pow(ten, two), ten)).get() &&
           "Expected multiply node.");
//  ((b^d)*c)^a -> b^(a*d)*c^a
    assert(graph::multiply_cast(graph::pow(graph::pow(ten, two)*two, ten)).get() &&
           "Expected multiply node.");
#else
    assert(graph::pow_cast(graph::pow(two*graph::pow(ten, two), ten)).get() &&
           "Expected a pow node.");
    assert(graph::pow_cast(graph::pow(graph::pow(ten, two)*two, ten)).get() &&
           "Expected a pow node.");
#endif

//  (c/Sqrt(b))^a -> c^a/b^a/2
#ifdef USE_REDUCE
    assert(graph::divide_cast(graph::pow(two/graph::sqrt(ten), ten)).get() &&
           "Expected divide node.");
//  (Sqrt(b)/c)^a -> (b^a/2)/c^a -> c2*b^a
    assert(graph::multiply_cast(graph::pow(graph::sqrt(ten)/two, ten)).get() &&
           "Expected multiply node.");
#else
    assert(graph::pow_cast(graph::pow(two/graph::sqrt(ten), ten)).get() &&
           "Expected a pow node.");
    assert(graph::pow_cast(graph::pow(graph::sqrt(ten)/two, ten)).get() &&
           "Expected a pow node.");
#endif

//  (c/(b^d))^a -> c^a/(b^(a*d))
#ifdef USE_REDUCE
    assert(graph::divide_cast(graph::pow(two/graph::pow(ten, two), ten)).get() &&
           "Expected divide node.");
//  ((b^d)/c))^a -> (b^(a*d))/c^a -> c2*b^a
    assert(graph::multiply_cast(graph::pow(graph::pow(ten, two)/two, ten)).get() &&
           "Expected multiply node.");
#else
    assert(graph::pow_cast(graph::pow(two/graph::pow(ten, two), ten)).get() &&
           "Expected a pow node.");
    assert(graph::pow_cast(graph::pow(graph::pow(ten, two)/two, ten)).get() &&
           "Expected a pow node.");
#endif

//  a^1/2 -> sqrt(a);
#ifdef USE_REDUCE
    assert(graph::sqrt_cast(graph::pow(ten, one/two)).get() &&
           "Expected sqrt node.");
#else
    assert(graph::pow_cast(graph::pow(ten, one/two)).get() &&
           "Expected a pow node.");
#endif

    auto hundred = graph::pow(ten, two);
    assert(hundred->evaluate().at(0) == static_cast<T> (100.0) &&
           "Expected 100");
    const auto non_int = static_cast<T> (0.438763);
    auto sqrd = graph::pow(graph::constant(non_int), two);
    assert(sqrd->evaluate().at(0) == static_cast<T> (non_int*non_int) &&
           "Expected x*x");

    auto three = graph::constant(static_cast<T> (2.0));
    auto pow_pow1 = graph::pow(graph::pow(ten, three), two);
    auto pow_pow2 = graph::pow(ten, three*two);
#ifdef USE_REDUCE
    assert(pow_pow1->is_match(pow_pow2) &&
           "Expected ten to the 6.");
#else
    assert(graph::pow_cast(pow_pow2).get() && "Expected a pow node.");
#endif

#ifdef USE_REDUCE
    assert(graph::multiply_cast(graph::pow(two*ten, two)).get() &&
           "Expected multiply node.");
    assert(graph::multiply_cast(graph::pow(ten*two, two)).get() &&
           "Expected multiply node.");
    assert(graph::divide_cast(graph::pow(two/ten, two)).get() &&
           "Expected divide node.");
// (v/c)^a -> v^a/c^a -> c2*v^a
    assert(graph::multiply_cast(graph::pow(ten/two, two)).get() &&
           "Expected multiply node.");
#else
    assert(graph::pow_cast(graph::pow(two*ten, two)).get() &&
           "Expected a pow node.");
    assert(graph::pow_cast(graph::pow(ten*two, two)).get() &&
           "Expected a pow node.");
    assert(graph::pow_cast(graph::pow(two/ten, two)).get() &&
           "Expected a pow node.");
    assert(graph::pow_cast(graph::pow(ten/two, two)).get() &&
           "Expected a pow node.");
#endif

//  sqrt(a)^a -> a^(b/c) -> a^(c2*b)
    auto pow_sqrt = graph::pow_cast(graph::pow(graph::sqrt(ten), ten));
#ifdef USE_REDUCE
    assert(graph::multiply_cast(pow_sqrt->get_right()).get() &&
           "Expected mutliply node.");
#else
    assert(graph::sqrt_cast(pow_sqrt->get_left()).get() &&
           "Expected a sqrt node.");
#endif

//  Test derivatives.
    auto x2 = graph::pow(ten, two);
    auto dx2dx = x2->df(ten);
    assert(graph::multiply_cast(dx2dx).get() && "Expected multiply node.");

    auto x3 = graph::pow(two, ten);
    auto dx3dx = x3->df(ten);
    assert(graph::multiply_cast(dx3dx).get() && "Expected multiply node.");
}

//------------------------------------------------------------------------------
///  @brief Tests for log nodes.
//------------------------------------------------------------------------------
template<typename T>
void test_log() {
#ifdef USE_REDUCE
    assert(graph::constant_cast(graph::log(graph::constant(static_cast<T> (10.0)))) &&
           "Expected constant");
#else
    assert(graph::log_cast(graph::log(graph::constant(static_cast<T> (10.0)))) &&
           "Expected log node");
#endif

    auto y = graph::variable<T> (1, "y");
    auto logy = graph::log(y);
    assert(graph::log_cast(logy) && "Expected log");

//  Test derivatives.
    auto dlogy = logy->df(y);
    assert(graph::divide_cast(dlogy) && "Expected divide node.");
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
//------------------------------------------------------------------------------
template<typename T> void run_tests() {
    test_sqrt<T> ();
    test_exp<T> ();
    test_pow<T> ();
    test_log<T> ();
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
