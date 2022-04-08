//------------------------------------------------------------------------------
///  @file math_test.cpp
///  @brief Tests for math nodes.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/math.hpp"

//------------------------------------------------------------------------------
///  @brief Tests for constant nodes.
//------------------------------------------------------------------------------
void test_sqrt() {
    auto ten = graph::constant(10);
    auto sqrt_ten = graph::sqrt(ten);
    assert(sqrt_ten->evaluate().at(0) == sqrt(10.0));

    auto sqrt_ten_cast = std::dynamic_pointer_cast<graph::constant_node> (sqrt_ten);
    assert(sqrt_ten_cast.get() != nullptr && "Expected a constant type.");

    auto vec = graph::constant({5, 3});
    auto sqrt_vec = sqrt(vec);
    const std::vector<double> sqrt_vec_result = sqrt_vec->evaluate();
    assert(sqrt_vec_result.size() == 2);
    assert(sqrt_vec_result.at(0) == sqrt(5.0));
    assert(sqrt_vec_result.at(1) == sqrt(3.0));

    auto var = graph::variable(1);
    auto sqrt_var = graph::sqrt(var);
    auto sqrt_var_cast =
        std::dynamic_pointer_cast<
            graph::sqrt_node<graph::leaf_node>> (sqrt_var);
    assert(sqrt_var_cast.get() != nullptr && "Expected a variable type.");

    var->set(3);
    assert(sqrt_var->evaluate().at(0) == sqrt(3.0));

    auto var_vec = graph::variable({4, 7});
    auto sqrt_var_vec = graph::sqrt(var_vec);
    auto sqrt_var_vec_cast =
        std::dynamic_pointer_cast<
            graph::sqrt_node<graph::leaf_node>> (sqrt_var_vec);
    assert(sqrt_var_vec_cast.get() != nullptr && "Expected a variable type.");
    const std::vector<double> sqrt_var_vec_result = sqrt_var_vec->evaluate();
    assert(sqrt_var_vec_result.size() == 2);
    assert(sqrt_var_vec_result.at(0) == sqrt(4.0));
    assert(sqrt_var_vec_result.at(1) == sqrt(7.0));

//  d sqrt(x) / dx = 1/(2 Sqrt(x))
    auto dsqrt_var = sqrt_var->df(var);
    auto dsqrt_var_cast =
        std::dynamic_pointer_cast<
            graph::divide_node<graph::leaf_node,
                               graph::leaf_node>> (dsqrt_var);
    assert(dsqrt_var_cast.get() != nullptr && "Expected a divide type.");
    assert(dsqrt_var->evaluate().at(0) == 1.0/(2.0*sqrt(3.0)));
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    test_sqrt();
}
