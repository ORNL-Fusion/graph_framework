//------------------------------------------------------------------------------
///  @file logical.cpp
///  @brief Tests for logic nodes.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/graph_framework.hpp"

//------------------------------------------------------------------------------
///  @brief Tests for equal nodes.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_not() {
    auto true_v = graph::true_constant<T> ();
    auto false_v = graph::false_constant<T> ();

    auto result1 = !true_v;
    assert(result1->is_match(false_v) && "Expected flase.");
    auto result2 = !false_v;
    assert(result2->is_match(true_v) && "Expected true.");

    auto v1 = graph::variable<T> (1, "");
    auto v2 = graph::variable<T> (1, "");
    auto result3 = !(v1 == v2);
    auto result3_cast = graph::not_equal_cast(result3);
    assert(result3_cast.get() && "Expected a not equal node.");

    auto result4 = !(v1 != v2);
    auto result4_cast = graph::equal_cast(result4);
    assert(result4_cast.get() && "Expected an equal node.");
    
    auto result5 = !(v1 < v2);
    auto result5_cast = graph::greater_than_equal_cast(result5);
    assert(result5_cast.get() && "Expected a greater than equal node.");

    auto result6 = !(v1 > v2);
    auto result6_cast = graph::less_than_equal_cast(result6);
    assert(result6_cast.get() && "Expected a less than equal node.");
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

    auto v1 = graph::variable<T> (1, "");
    auto v2 = graph::variable<T> (1, "");
    auto result = graph::if_(v1, v2, v2);
    assert(result->is_match(v2));
    auto result_df = result->df(v1);
    assert(result_df->is_match(false_v) && "Expected 0");
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void run_tests() {
    test_equal<T> ();
    test_not_equal<T> ();
    if constexpr (std::floating_point<T>) {
        test_not<T> ();
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
