//------------------------------------------------------------------------------
///  @file logical.cpp
///  @brief Tests for logic nodes.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cfloat>

#include "../graph_framework/graph_framework.hpp"

//------------------------------------------------------------------------------
///  @brief Tests for isinf nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_isinf() {
    auto zero = graph::zero<T> ();
    auto one = graph::one<T> ();
    auto nan = graph::constant<T> (NAN);
    auto inf = graph::constant<T> (INFINITY);

    auto true_v = graph::true_constant<T> ();
    auto false_v = graph::false_constant<T> ();

    assert(graph::isinf(zero)->is_match(false_v) && "Expected false.");
    assert(graph::isinf(one)->is_match(false_v) && "Expected false.");
    assert(graph::isinf(nan)->is_match(false_v) && "Expected false.");
    assert(graph::isinf(inf)->is_match(true_v) && "Expected true.");
}

//------------------------------------------------------------------------------
///  @brief Tests for isnan nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_isnan() {
    auto zero = graph::zero<T> ();
    auto one = graph::one<T> ();
    auto nan = graph::constant<T> (NAN);
    auto inf = graph::constant<T> (INFINITY);

    auto true_v = graph::true_constant<T> ();
    auto false_v = graph::false_constant<T> ();

    assert(graph::isnan(zero)->is_match(false_v) && "Expected false.");
    assert(graph::isnan(one)->is_match(false_v) && "Expected false.");
    assert(graph::isnan(nan)->is_match(true_v) && "Expected true.");
    assert(graph::isnan(inf)->is_match(false_v) && "Expected false.");
}

//------------------------------------------------------------------------------
///  @brief Tests for not nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_not() {
    auto true_v = graph::true_constant<T> ();
    auto false_v = graph::false_constant<T> ();

    auto result1 = !true_v;
    assert(result1->is_match(false_v) && "Expected false.");
    auto result2 = !false_v;
    assert(result2->is_match(true_v) && "Expected true.");

//  !(a == b) -> a != b
    auto v1 = graph::variable<T> (1, "");
    auto v2 = graph::variable<T> (1, "");
    auto result3 = !(v1 == v2);
    auto result3_cast = graph::not_equal_cast(result3);
    assert(result3_cast.get() && "Expected a not equal node.");

//  !(a != b) -> a == b
    auto result4 = !(v1 != v2);
    auto result4_cast = graph::equal_cast(result4);
    assert(result4_cast.get() && "Expected an equal node.");

    if constexpr (!jit::complex_scalar<T>) {
//  !(a < b) -> a >= b
        auto result5 = !(v1 < v2);
        auto result5_cast = graph::greater_than_equal_cast(result5);
        assert(result5_cast.get() && "Expected a greater than equal node.");
        
//  !(a <= b) -> a > b
        auto result6 = !(v1 <= v2);
        auto result6_cast = graph::greater_than_cast(result6);
        assert(result6_cast.get() && "Expected a greater than node.");
        
//  !(a > b) -> a <= b
        auto result7 = !(v1 > v2);
        auto result7_cast = graph::less_than_equal_cast(result7);
        assert(result7_cast.get() && "Expected a less than equal node.");
        
//  !(a >= b) -> a < b
        auto result8 = !(v1 >= v2);
        auto result8_cast = graph::less_than_cast(result8);
        assert(result8_cast.get() && "Expected a less than node.");
    }

//  !!a -> a
    auto result9 = !!v1;
    auto result9_cast = graph::variable_cast(result9);
    assert(result9_cast.get() && "Expected v1");
}

//------------------------------------------------------------------------------
///  @brief Tests for equal nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_equal() {
    auto true_v = graph::true_constant<T> ();
    auto false_v = graph::false_constant<T> ();

    auto result1 = true_v == true_v;
    assert(result1->is_match(true_v) && "Expected true.");
    auto result2 = false_v == false_v;
    assert(result2->is_match(true_v) && "Expected true.");
    auto result3 = true_v == false_v;
    assert(result3->is_match(false_v) && "Expected false.");
    auto result4 = false_v == true_v;
    assert(result4->is_match(false_v) && "Expected false.");

    auto v1 = graph::variable<T> (1, "");
    auto v2 = graph::variable<T> (1, "");
    assert((v1 == v2)->is_match(v2 == v1) && "Expected match.");
}

//------------------------------------------------------------------------------
///  @brief Tests for not equal nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_not_equal() {
    auto true_v = graph::true_constant<T> ();
    auto false_v = graph::false_constant<T> ();

    auto result1 = true_v != true_v;
    assert(result1->is_match(false_v) && "Expected false.");
    auto result2 = false_v != false_v;
    assert(result2->is_match(false_v) && "Expected false.");
    auto result3 = true_v != false_v;
    assert(result3->is_match(true_v) && "Expected true.");
    auto result4 = false_v != true_v;
    assert(result4->is_match(true_v) && "Expected true.");

    auto v1 = graph::variable<T> (1, "");
    auto v2 = graph::variable<T> (1, "");
    assert((v1 != v2)->is_match(v2 != v1) && "Expected match.");
}

//------------------------------------------------------------------------------
///  @brief Tests for greater than nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_greater_than() {
    auto true_v = graph::true_constant<T> ();
    auto false_v = graph::false_constant<T> ();

    auto one = graph::one<T> ();
    auto none = graph::none<T> ();

    auto result1 = one > none;
    assert(result1->is_match(true_v) && "Expected true.");
    auto result2 = none > one;
    assert(result2->is_match(false_v) && "Expected false.");
}

//------------------------------------------------------------------------------
///  @brief Tests for less than nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_less_than() {
    auto true_v = graph::true_constant<T> ();
    auto false_v = graph::false_constant<T> ();

    auto one = graph::one<T> ();
    auto none = graph::none<T> ();

    auto result1 = one < none;
    assert(result1->is_match(false_v) && "Expected false.");
    auto result2 = none < one;
    assert(result2->is_match(true_v) && "Expected true.");
}

//------------------------------------------------------------------------------
///  @brief Tests for greater than equal nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_greater_than_equal() {
    auto true_v = graph::true_constant<T> ();
    auto false_v = graph::false_constant<T> ();

    auto one = graph::one<T> ();
    auto none = graph::none<T> ();

    auto result1 = one >= none;
    assert(result1->is_match(true_v) && "Expected true.");
    auto result2 = none >= one;
    assert(result2->is_match(false_v) && "Expected false.");
    auto result3 = one >= one;
    assert(result3->is_match(true_v) && "Expected true.");
}

//------------------------------------------------------------------------------
///  @brief Tests for less than equal nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_less_than_equal() {
    auto true_v = graph::true_constant<T> ();
    auto false_v = graph::false_constant<T> ();

    auto one = graph::one<T> ();
    auto none = graph::none<T> ();

    auto result1 = one <= none;
    assert(result1->is_match(false_v) && "Expected false.");
    auto result2 = none <= one;
    assert(result2->is_match(true_v) && "Expected true.");
    auto result3 = one <= one;
    assert(result3->is_match(true_v) && "Expected true.");
}

//------------------------------------------------------------------------------
///  @brief Tests for and nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_and() {
    auto true_v = graph::true_constant<T> ();
    auto false_v = graph::false_constant<T> ();

    auto result1 = true_v && true_v;
    assert(result1->is_match(true_v) && "Expected true.");
    auto result2 = true_v && false_v;
    assert(result2->is_match(false_v) && "Expected false.");
    auto result3 = false_v && false_v;
    assert(result3->is_match(false_v) && "Expected false.");

    auto v1 = graph::variable<T> (1, "");
    auto v2 = graph::variable<T> (1, "");
    assert((v1 && v2)->is_match(v2 && v1) && "Expected match.");
}

//------------------------------------------------------------------------------
///  @brief Tests for or nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_or() {
    auto true_v = graph::true_constant<T> ();
    auto false_v = graph::false_constant<T> ();

    auto result1 = true_v || true_v;
    assert(result1->is_match(true_v) && "Expected true.");
    auto result2 = true_v || true_v;
    assert(result2->is_match(true_v) && "Expected true.");
    auto result3 = false_v || false_v;
    assert(result3->is_match(false_v) && "Expected false.");

    auto v1 = graph::variable<T> (1, "");
    auto v2 = graph::variable<T> (1, "");
    assert((v1 || v2)->is_match(v2 || v1) && "Expected match.");
}

//------------------------------------------------------------------------------
///  @brief Tests for if nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<std::floating_point T> void test_if() {
    auto true_v = graph::true_constant<T> ();
    auto false_v = graph::false_constant<T> ();

    auto result1 = graph::if_(true_v, true_v, false_v);
    assert(result1->is_match(true_v) && "Exected the true condition.");
    auto result2 = graph::if_(false_v, true_v, false_v);
    assert(result2->is_match(false_v) && "Exected the false condition.");

//  If(c, a, a) -> a
    auto v1 = graph::variable<T> (1, "");
    auto v2 = graph::variable<T> (1, "");
    auto result = graph::if_(v1, v2, v2);
    assert(result->is_match(v2));
    auto result_df = result->df(v1);
    assert(result_df->is_match(false_v) && "Expected 0");

//  If(!a, b, c) -> If(a, c, b)
    auto test_not = graph::if_(graph::not_(v1), v1, v2);
    auto test_not_cast = if_cast(test_not);
    assert(test_not_cast.get() && "Expected if node.");
    assert(test_not_cast->get_left()->is_match(v1) && "Expected v1");
    assert(test_not_cast->get_middle()->is_match(v2) && "Expected v2");
    assert(test_not_cast->get_right()->is_match(v1) && "Expected v1");
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void run_tests() {
    test_equal<T> ();
    test_not_equal<T> ();
    test_not<T> ();
    if constexpr (std::floating_point<T>) {
        test_isinf<T> ();
        test_isnan<T> ();
        test_greater_than<T> ();
        test_less_than<T> ();
        test_and<T> ();
        test_or<T> ();
        test_if<T> ();
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
