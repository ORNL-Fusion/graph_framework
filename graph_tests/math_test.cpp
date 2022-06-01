//------------------------------------------------------------------------------
///  @file math_test.cpp
///  @brief Tests for math nodes.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/cpu_backend.hpp"
#include "../graph_framework/math.hpp"

//------------------------------------------------------------------------------
///  @brief Tests for sqrt nodes.
//------------------------------------------------------------------------------
template<typename BACKEND>
void test_sqrt() {
    auto ten = graph::constant<BACKEND> (10);
    auto sqrt_ten = graph::sqrt(ten);
    assert(sqrt_ten->evaluate().at(0) == sqrt(backend::base_cast<BACKEND> (10.0)));

    assert(graph::constant_cast(sqrt_ten).get() && "Expected a constant type.");

    auto vec = graph::constant<BACKEND> (std::vector<typename BACKEND::base> ({5.0, 3.0}));
    auto sqrt_vec = graph::sqrt(vec);
    const BACKEND sqrt_vec_result = sqrt_vec->evaluate();
    assert(sqrt_vec_result.size() == 2);
    assert(sqrt_vec_result.at(0) == sqrt(backend::base_cast<BACKEND> (5.0)));
    assert(sqrt_vec_result.at(1) == sqrt(backend::base_cast<BACKEND> (3.0)));

    auto var = graph::variable<BACKEND> (1, "");
    auto sqrt_var = graph::sqrt(var);
    assert(graph::sqrt_cast(sqrt_var).get() &&"Expected a variable type.");

    var->set(3);
    assert(sqrt_var->evaluate().at(0) == sqrt(backend::base_cast<BACKEND> (3.0)));

    auto var_vec = graph::variable<BACKEND> (std::vector<typename BACKEND::base> ({4.0, 7.0}), "");
    auto sqrt_var_vec = graph::sqrt(var_vec);
    assert(graph::sqrt_cast(sqrt_var_vec).get() && "Expected a variable type.");
    const BACKEND sqrt_var_vec_result = sqrt_var_vec->evaluate();
    assert(sqrt_var_vec_result.size() == 2);
    assert(sqrt_var_vec_result.at(0) == sqrt(backend::base_cast<BACKEND> (4.0)));
    assert(sqrt_var_vec_result.at(1) == sqrt(backend::base_cast<BACKEND> (7.0)));

//  d sqrt(x) / dx = 1/(2 Sqrt(x))
    auto dsqrt_var = sqrt_var->df(var);
    assert(graph::divide_cast(dsqrt_var).get() && "Expected a divide type.");
    assert(dsqrt_var->evaluate().at(0) ==
           backend::base_cast<BACKEND> (1.0) /
           (backend::base_cast<BACKEND> (2.0) *
            sqrt(backend::base_cast<BACKEND> (3.0))) &&
           "Expected 0.5*sqrt(3)");

//  Reduction sqrt(c*x*c*y) = x
    auto x1 = graph::constant<BACKEND> (2)*graph::variable<BACKEND> (1, "x");
    auto x2 = graph::constant<BACKEND> (3)*graph::variable<BACKEND> (1, "y");
    auto x = graph::sqrt(x1*x2);
    auto x_cast = graph::multiply_cast(x);
    assert(x_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(x_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::sqrt_cast(x_cast->get_right()).get() &&
           "Expected sqrt node.");

//  Reduction Sqrt(x*x) = x
    auto x_var = graph::variable<BACKEND> (1, "x");
    auto x2_sqrt = graph::sqrt(x_var*x_var);
    assert(x2_sqrt.get() == x_var.get() && "Expected to reduce to x_var.");

//  Reduction Sqrt(x*y*x*y) = x*y
    auto y_var = graph::variable<BACKEND> (1, "y");
    auto x2y2_sqrt = graph::sqrt(x_var*y_var*x_var*y_var);
    auto x2y2_sqrt_cast = graph::multiply_cast(x2y2_sqrt);
    assert(x2y2_sqrt_cast.get() && "Expected multiply node");
    assert((x2y2_sqrt_cast->get_left().get() == x_var.get() ||
            x2y2_sqrt_cast->get_left().get() == y_var.get()) &&
           "Expected x_var or y_var.");
    assert((x2y2_sqrt_cast->get_right().get() == x_var.get() ||
            x2y2_sqrt_cast->get_right().get() == y_var.get()) &&
           "Expected x_var or y_var.");

//  Reduction Sqrt(x*x/y*y);
    auto sq_reduce = graph::sqrt((x_var*x_var)/(y_var*y_var));
    auto sq_reduce_cast = graph::divide_cast(sq_reduce);
    assert(sq_reduce_cast.get() && "Expected divide node.");
    assert(sq_reduce_cast->get_left().get() == x_var.get() &&
           "Expected x_var.");
    assert(sq_reduce_cast->get_right().get() == y_var.get() &&
           "Expected y_var.");

//  Reduction Sqrt(c*x/b*y) = d*Sqrt(x/y)
    auto cxby_sqrt = graph::sqrt(x1/x2);
    auto cxby_sqrt_cast = graph::multiply_cast(cxby_sqrt);
    assert(cxby_sqrt_cast.get() && "Expected a multiply node.");
    assert(graph::constant_cast(cxby_sqrt_cast->get_left()).get() &&
           "Expected a constant coefficent.");
    assert(graph::sqrt_cast(cxby_sqrt_cast->get_right()).get() &&
           "Expected sqrt node.");
}

//------------------------------------------------------------------------------
///  @brief Tests for exp nodes.
//------------------------------------------------------------------------------
template<typename BACKEND>
void test_exp() {
    auto ten = graph::constant<BACKEND> (10);
    auto exp_ten = graph::exp(ten);
    assert(exp_ten->evaluate().at(0) == exp(backend::base_cast<BACKEND> (10.0)));
    assert(graph::constant_cast(exp_ten).get() &&
           "Expected a constant type.");

    auto vec = graph::constant<BACKEND> (std::vector<typename BACKEND::base> ({5.0, 3.0}));
    auto exp_vec = exp(vec);
    const BACKEND exp_vec_result = exp_vec->evaluate();
    assert(exp_vec_result.size() == 2);
    assert(exp_vec_result.at(0) == exp(backend::base_cast<BACKEND> (5.0)));
    assert(exp_vec_result.at(1) == exp(backend::base_cast<BACKEND> (3.0)));

    auto var = graph::variable<BACKEND> (1, "");
    auto exp_var = graph::exp(var);
    assert(graph::exp_cast(exp_var).get() && "Expected a variable type.");

    var->set(3);
    assert(exp_var->evaluate().at(0) == exp(backend::base_cast<BACKEND> (3.0)));

    auto var_vec = graph::variable<BACKEND> (std::vector<typename BACKEND::base> ({4.0, 7.0}), "");
    auto exp_var_vec = graph::exp(var_vec);
    assert(graph::exp_cast(exp_var_vec).get() && "Expected a variable type.");
    const BACKEND exp_var_vec_result = exp_var_vec->evaluate();
    assert(exp_var_vec_result.size() == 2);
    assert(exp_var_vec_result.at(0) == exp(backend::base_cast<BACKEND> (4.0)));
    assert(exp_var_vec_result.at(1) == exp(backend::base_cast<BACKEND> (7.0)));

//  d exp(x) / dx = exp(x)
    auto dexp_var = exp_var->df(var);
    assert(graph::exp_cast(dexp_var).get() && "Expected a divide type.");
    assert(dexp_var->evaluate().at(0) == exp(backend::base_cast<BACKEND> (3.0)));
}

//------------------------------------------------------------------------------
///  @brief Tests for pow nodes.
//------------------------------------------------------------------------------
template<typename BACKEND>
void test_pow() {
//  a^0 = 1
    auto zero = graph::constant<BACKEND> (0);
    auto ten = graph::variable<BACKEND> (1, "10");
    ten->set(10.0);
    auto one = graph::pow(ten, zero);
    assert(graph::constant_cast(one).get() && "Expected constant");
    assert(one->evaluate().at(0) == backend::base_cast<BACKEND> (1.0) &&
           "Expected 1");
    
//  a^1 = a
    assert(graph::pow(ten, one).get() == ten.get() && "Expected ten.");

//  Sqrt(a)^2 = a
    auto two = graph::constant<BACKEND> (2);
    assert(graph::pow(graph::sqrt(ten), two)->is_match(ten) &&
           "Expected ten.");

//  (c*Sqrt(b))^a -> c^a*b^a/2
    assert(graph::multiply_cast(graph::pow(two*graph::sqrt(ten), ten)).get() &&
           "Expected multiply node.");
//  (Sqrt(b)*c)^a -> c^a*b^a/2
    assert(graph::multiply_cast(graph::pow(graph::sqrt(ten)*two, ten)).get() &&
           "Expected multiply node.");

//  (c*b^d)^a -> c^a*b^(a*d)
    assert(graph::multiply_cast(graph::pow(two*graph::pow(ten, two), ten)).get() &&
           "Expected multiply node.");
//  ((b^d)*c)^a -> b^(a*d)*c^a
    assert(graph::multiply_cast(graph::pow(graph::pow(ten, two)*two, ten)).get() &&
           "Expected multiply node.");

//  (c/Sqrt(b))^a -> c^a/b^a/2
    assert(graph::divide_cast(graph::pow(two/graph::sqrt(ten), ten)).get() &&
           "Expected divide node.");
//  (Sqrt(b)/c)^a -> (b^a/2)/c^a
    assert(graph::divide_cast(graph::pow(graph::sqrt(ten)/two, ten)).get() &&
           "Expected divide node.");

//  (c/(b^d))^a -> c^a/(b^(a*d))
    assert(graph::divide_cast(graph::pow(two/graph::pow(ten, two), ten)).get() &&
           "Expected divide node.");
//  ((b^d)/c))^a -> (b^(a*d))/c^a
    assert(graph::divide_cast(graph::pow(graph::pow(ten, two)/two, ten)).get() &&
           "Expected divide node.");

//  a^1/2 -> sqrt(a);
    assert(graph::sqrt_cast(graph::pow(ten, one/two)).get() &&
           "Expected sqrt node.");

    auto hundred = graph::pow(ten, two);
    assert(hundred->evaluate().at(0) == backend::base_cast<BACKEND> (100.0) &&
           "Expected 100");
    
    auto three = graph::constant<BACKEND> (2);
    auto pow_pow1 = graph::pow(graph::pow(ten, three), two);
    auto pow_pow2 = graph::pow(ten, three*two);
    assert(pow_pow1->is_match(pow_pow2) &&
           "Expected ten to the 6.");

    assert(graph::multiply_cast(graph::pow(two*ten, two)).get() &&
           "Expected multiply node.");
    assert(graph::multiply_cast(graph::pow(ten*two, two)).get() &&
           "Expected multiply node.");
    assert(graph::divide_cast(graph::pow(two/ten, two)).get() &&
           "Expected divide node.");
    assert(graph::divide_cast(graph::pow(ten/two, two)).get() &&
           "Expected divide node.");
    
    auto pow_sqrt = graph::pow_cast(graph::pow(graph::sqrt(ten), ten));
    assert(graph::divide_cast(pow_sqrt->get_right()).get() &&
           "Expected divide node.");
    
//  Test derivatives.
    auto x2 = graph::pow(ten, two);
    auto dx2dx = x2->df(ten);
    assert(graph::multiply_cast(dx2dx).get() && "Expected multiply node.");
    auto x3 = graph::pow(two, ten);
    auto dx3dx = x3->df(ten);
    assert(graph::multiply_cast(dx3dx).get() && "Expected multiply node.");
}

//------------------------------------------------------------------------------
///  @brief Tests for pow nodes.
//------------------------------------------------------------------------------
template<typename BACKEND>
void test_log() {
    assert(graph::constant_cast(graph::log(graph::constant<BACKEND> (10))) &&
           "Expected constant");
    auto y = graph::variable<BACKEND> (1, "y");
    auto logy = graph::log(y);
    assert(graph::log_cast(logy) && "Expected log");

//  Test derivatives.
    auto dlogy = logy->df(y);
    assert(graph::divide_cast(dlogy) && "Expected divide node.");
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
//------------------------------------------------------------------------------
template<typename BACKEND> void run_tests() {
    test_sqrt<BACKEND> ();
    test_exp<BACKEND> ();
    test_pow<BACKEND> ();
    test_log<BACKEND> ();
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
}
