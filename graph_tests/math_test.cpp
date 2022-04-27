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
    assert(sqrt_ten->evaluate().at(0) == sqrt(10.0));

    assert(graph::constant_cast(sqrt_ten).get() != nullptr &&
           "Expected a constant type.");

    auto vec = graph::constant<BACKEND> (std::vector<double> ({5.0, 3.0}));
    auto sqrt_vec = graph::sqrt(vec);
    const BACKEND sqrt_vec_result = sqrt_vec->evaluate();
    assert(sqrt_vec_result.size() == 2);
    assert(sqrt_vec_result.at(0) == sqrt(5.0));
    assert(sqrt_vec_result.at(1) == sqrt(3.0));

    auto var = graph::variable<BACKEND> (1);
    auto sqrt_var = graph::sqrt(var);
    assert(graph::sqrt_cast(sqrt_var).get() != nullptr &&
           "Expected a variable type.");

    var->set(3);
    assert(sqrt_var->evaluate().at(0) == sqrt(3.0));

    auto var_vec = graph::variable<BACKEND> (std::vector<double> ({4.0, 7.0}));
    auto sqrt_var_vec = graph::sqrt(var_vec);
    assert(graph::sqrt_cast(sqrt_var_vec).get() != nullptr &&
           "Expected a variable type.");
    const BACKEND sqrt_var_vec_result = sqrt_var_vec->evaluate();
    assert(sqrt_var_vec_result.size() == 2);
    assert(sqrt_var_vec_result.at(0) == sqrt(4.0));
    assert(sqrt_var_vec_result.at(1) == sqrt(7.0));

//  d sqrt(x) / dx = 1/(2 Sqrt(x))
    auto dsqrt_var = sqrt_var->df(var);
    assert(graph::divide_cast(dsqrt_var).get() != nullptr &&
           "Expected a divide type.");
    assert(dsqrt_var->evaluate().at(0) == 1.0/(2.0*sqrt(3.0)));
}

//------------------------------------------------------------------------------
///  @brief Tests for exp nodes.
//------------------------------------------------------------------------------
template<typename BACKEND>
void test_exp() {
    auto ten = graph::constant<BACKEND> (10);
    auto exp_ten = graph::exp(ten);
    assert(exp_ten->evaluate().at(0) == exp(10.0));
    assert(graph::constant_cast(exp_ten).get() != nullptr &&
           "Expected a constant type.");

    auto vec = graph::constant<BACKEND> (std::vector<double> ({5.0, 3.0}));
    auto exp_vec = exp(vec);
    const BACKEND exp_vec_result = exp_vec->evaluate();
    assert(exp_vec_result.size() == 2);
    assert(exp_vec_result.at(0) == exp(5.0));
    assert(exp_vec_result.at(1) == exp(3.0));

    auto var = graph::variable<BACKEND> (1);
    auto exp_var = graph::exp(var);
    assert(graph::exp_cast(exp_var).get() != nullptr &&
           "Expected a variable type.");

    var->set(3);
    assert(exp_var->evaluate().at(0) == exp(3.0));

    auto var_vec = graph::variable<BACKEND> (std::vector<double> ({4.0, 7.0}));
    auto exp_var_vec = graph::exp(var_vec);
    assert(graph::exp_cast(exp_var_vec).get() != nullptr &&
           "Expected a variable type.");
    const BACKEND exp_var_vec_result = exp_var_vec->evaluate();
    assert(exp_var_vec_result.size() == 2);
    assert(exp_var_vec_result.at(0) == exp(4.0));
    assert(exp_var_vec_result.at(1) == exp(7.0));

//  d exp(x) / dx = exp(x)
    auto dexp_var = exp_var->df(var);
    assert(graph::exp_cast(dexp_var).get() != nullptr &&
           "Expected a divide type.");
    assert(dexp_var->evaluate().at(0) == exp(3.0));
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
//------------------------------------------------------------------------------
template<typename BACKEND> void run_tests() {
    test_sqrt<backend::cpu> ();
    test_exp<backend::cpu> ();
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
