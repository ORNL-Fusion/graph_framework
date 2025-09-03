//------------------------------------------------------------------------------
///  @file backend_test.cpp
///  @brief Tests for the buffer backend.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/backend.hpp"

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void test_backend() {
    backend::buffer<T> size_one(1);
    assert(size_one.size() == 1 && "Expected a size of one.");
    size_one[0] = 10;
    assert(size_one.at(0) == static_cast<T> (10) &&
           "Expected a value of 10.");
    assert(size_one[0] == static_cast<T> (10) &&
           "Expected a value of 10.");

    const backend::buffer<T> size_three(3, static_cast<T> (2.0));
    assert(size_three.size() == 3 && "Expected a size of three.");
    for (size_t i = 0; i < 3; i++) {
        assert(size_three.at(i) == static_cast<T> (2.0) &&
               "Expected a value of 10.");
        assert(size_three[i] == static_cast<T> (2.0) &&
               "Expected a value of 10.");
    }

    backend::buffer<T> vector(std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0),
        static_cast<T> (3.0),
        static_cast<T> (4.0)
    }));
    assert(vector.size() == 4 && "Expected a size of three.");
    for (size_t i = 0; i < 4; i++) {
        assert(vector.at(i) == static_cast<T> (i + 1) &&
               "Expected a value one more than index.");
        assert(vector[i] == static_cast<T> (i + 1) &&
               "Expected a value one more than index.");
    }

    backend::buffer<T> copy(vector);
    assert(copy.size() == 4 && "Expected a size of four.");
    for (size_t i = 0; i < 4; i++) {
        assert(copy.at(i) == static_cast<T> (i + 1) &&
               "Expected a value one more than index.");
        assert(copy[i] == static_cast<T> (i + 1) &&
               "Expected a value one more than index.");
    }

    copy.set(static_cast<T> (5.0));
    for (size_t i = 0; i < 3; i++) {
        assert(copy.at(i) == static_cast<T> (5.0) &&
               "Expected a value of 5.");
        assert(copy[i] == static_cast<T> (5.0) &&
               "Expected a value of 5.");
    }

    assert(!vector.is_same() && "Expected different values.");
    assert(copy.is_same() && "Expected same values.");

    assert(!vector.is_zero() && "Expected non zero values.");
    assert(!copy.is_zero() && "Expected non zero values.");
    copy.set(0);
    assert(copy.is_zero() && "Expected zero values.");

    vector.sqrt();
    for (size_t i = 0; i < 4; i++) {
        assert(vector.at(i) == std::sqrt(static_cast<T> (i + 1)) &&
               "Expected a value sqrt(one more than index).");
    }

    backend::buffer<T> vector2(std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0),
        static_cast<T> (3.0),
        static_cast<T> (4.0)
    }));
    vector2.exp();
    for (size_t i = 0; i < 4; i++) {
        assert(vector2.at(i) == std::exp(static_cast<T> (i + 1)) &&
               "Expected a value exp(one more than index).");
    }

//  Addition tests.
    backend::buffer<T> avec(std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0)
    }));
    backend::buffer<T> bvec(std::vector<T> ({
        static_cast<T> (3.0),
        static_cast<T> (4.0)
    }));
    const backend::buffer<T> add_vec_vec = avec + bvec;
    assert(add_vec_vec.size() == 2 && "Expected a size of 2");
    assert(add_vec_vec.at(0) == static_cast<T> (4.0) &&
           "Expected a value of 4.");
    assert(add_vec_vec.at(1) == static_cast<T> (6.0) &&
           "Expected a value of 6.");

    backend::buffer<T> bscalar(1, static_cast<T> (3.0));
    avec.set(std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0)
    }));
    const backend::buffer<T> add_vec_scale = avec + bscalar;
    assert(add_vec_scale.size() == 2 && "Expected a size of 2");
    assert(add_vec_scale.at(0) == static_cast<T> (4.0) &&
           "Expected a value of 4.");
    assert(add_vec_scale.at(1) == static_cast<T> (5.0) &&
           "Expected a value of 5.");

    const backend::buffer<T> add_scale_vec = bscalar + bvec;
    assert(add_scale_vec.size() == 2 && "Expected a size of 2");
    assert(add_scale_vec.at(0) == static_cast<T> (6.0) &&
           "Expected a value of 6.");
    assert(add_scale_vec.at(1) == static_cast<T> (7.0) &&
           "Expected a value of 7.");

    backend::buffer<T> cscalar(1, static_cast<T> (-5.0));
    const backend::buffer<T> add_scale_scale = bscalar + cscalar;
    assert(add_scale_scale.size() == 1 && "Expected a size of 1");
    assert(add_scale_scale.at(0) == static_cast<T> (-2.0) &&
           "Expected a value of -4.");

//  Subtraction tests.
    const backend::buffer<T> sub_vec_vec = avec - bvec;
    assert(sub_vec_vec.size() == 2 && "Expected a size of 2");
    assert(sub_vec_vec.at(0) == static_cast<T> (-2.0) &&
           "Expected a value of -2.");
    assert(sub_vec_vec.at(1) == static_cast<T> (-2.0) &&
           "Expected a value of -2.");

    avec.set(std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0)
    }));
    bscalar.set(static_cast<T> (3.0));
    const backend::buffer<T> sub_vec_scale = avec - bscalar;
    assert(sub_vec_scale.size() == 2 && "Expected a size of 2");
    assert(sub_vec_scale.at(0) == static_cast<T> (-2.0) &&
           "Expected a value of -2.");
    assert(sub_vec_scale.at(1) == static_cast<T> (-1.0) &&
           "Expected a value of -1.");

    avec.set(std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0)
    }));
    cscalar.set(static_cast<T> (-5.0));
    const backend::buffer<T> sub_scale_vec = cscalar - avec;
    assert(sub_scale_vec.size() == 2 && "Expected a size of 2");
    assert(sub_scale_vec.at(0) == static_cast<T> (-6.0) &&
           "Expected a value of -6.");
    assert(sub_scale_vec.at(1) == static_cast<T> (-7.0) &&
           "Expected a value of -7.");

    const backend::buffer<T> sub_scale_scale = bscalar - cscalar;
    assert(sub_scale_scale.size() == 1 && "Expected a size of 1");
    assert(sub_scale_scale.at(0) == static_cast<T> (8.0) &&
           "Expected a value of 8.");

//  Multiply tests.
    avec.set(std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0)
    }));
    bvec.set(std::vector<T> ({
        static_cast<T> (3.0),
        static_cast<T> (4.0)
    }));
    const backend::buffer<T> mul_vec_vec = avec*bvec;
    assert(mul_vec_vec.size() == 2 && "Expected a size of 2");
    assert(mul_vec_vec.at(0) == static_cast<T> (3.0) &&
           "Expected a value of 3.");
    assert(mul_vec_vec.at(1) == static_cast<T> (8.0) &&
           "Expected a value of 8.");

    avec.set(std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0)
    }));
    bscalar.set(static_cast<T> (3.0));
    const backend::buffer<T> mul_vec_scale = avec*bscalar;
    assert(mul_vec_scale.size() == 2 && "Expected a size of 2");
    assert(mul_vec_scale.at(0) == static_cast<T> (3.0) &&
           "Expected a value of 3.");
    assert(mul_vec_scale.at(1) == static_cast<T> (6.0) &&
           "Expected a value of 6.");

    const backend::buffer<T> mul_scale_vec = cscalar*bvec;
    assert(mul_scale_vec.size() == 2 && "Expected a size of 2");
    assert(mul_scale_vec.at(0) == static_cast<T> (-15.0) &&
           "Expected a value of -15.");
    assert(mul_scale_vec.at(1) == static_cast<T> (-20.0) &&
           "Expected a value of -20.");

    const backend::buffer<T> mul_scale_scale = bscalar*cscalar;
    assert(mul_scale_scale.size() == 1 && "Expected a size of 1");
    assert(mul_scale_scale.at(0) == static_cast<T> (-15.0) &&
           "Expected a value of -15.");

//  Divide tests.
    avec.set(std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0)
    }));
    bvec.set(std::vector<T> ({
        static_cast<T> (3.0),
        static_cast<T> (4.0)
    }));
    const backend::buffer<T> div_vec_vec = avec/bvec;
    assert(div_vec_vec.size() == 2 && "Expected a size of 2");
    assert(div_vec_vec.at(0) == static_cast<T> (1.0) /
                                static_cast<T> (3.0) &&
           "Expected a value of 1/3.");
    assert(div_vec_vec.at(1) == static_cast<T> (2.0) /
                                static_cast<T> (4.0) &&
           "Expected a value of 2/4.");

    avec.set(std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0)
    }));
    bscalar.set(static_cast<T> (3.0));
    const backend::buffer<T> div_vec_scale = avec/bscalar;
    assert(div_vec_scale.size() == 2 && "Expected a size of 2");
    assert(div_vec_scale.at(0) == static_cast<T> (1.0) /
                                  static_cast<T> (3.0) &&
           "Expected a value of 1/3.");
    assert(div_vec_scale.at(1) == static_cast<T> (2.0) /
                                  static_cast<T> (3.0) &&
           "Expected a value of 2/3.");

    avec.set(std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0)
    }));
    const backend::buffer<T> div_scale_vec = cscalar/avec;
    assert(div_scale_vec.size() == 2 && "Expected a size of 2");
    assert(div_scale_vec.at(0) == static_cast<T> (-5.0) &&
           "Expected a value of -5.");
    assert(div_scale_vec.at(1) == static_cast<T> (-5.0) /
                                  static_cast<T> (2.0) &&
           "Expected a value of -5/2.");

    const backend::buffer<T> div_scale_scale = bscalar/cscalar;
    assert(div_scale_scale.size() == 1 && "Expected a size of 1");
    assert(div_scale_scale.at(0) == static_cast<T> (-3.0) /
                                    static_cast<T> (5.0) &&
           "Expected a value of -3/5.");

//  Fused multiply add tests.
    avec.set(std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0)
    }));
    bvec.set(std::vector<T> ({
        static_cast<T> (3.0),
        static_cast<T> (4.0)
    }));
    backend::buffer<T> cvec(std::vector<T> ({
        static_cast<T> (-2.0),
        static_cast<T> (6.0)
    }));
    const backend::buffer<T> fma_vec_vec_vec = backend::fma(avec, bvec, cvec);
    assert(fma_vec_vec_vec.size() == 2 && "Expected a size of 2");
    assert(fma_vec_vec_vec.at(0) == static_cast<T> (1.0) &&
           "Expected a value of 1.");
    assert(fma_vec_vec_vec.at(1) == static_cast<T> (14.0) &&
           "Expected a value of 14.");

    avec.set(std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0)
    }));
    bvec.set(std::vector<T> ({
        static_cast<T> (3.0),
        static_cast<T> (4.0)
    }));
    bscalar.set(static_cast<T> (3.0));
    const backend::buffer<T> fma_vec_vec_scale = backend::fma(avec, bvec, bscalar);
    assert(fma_vec_vec_scale.size() == 2 && "Expected a size of 2");
    assert(fma_vec_vec_scale.at(0) == static_cast<T> (6.0) &&
           "Expected a value of 6.");
    assert(fma_vec_vec_scale.at(1) == static_cast<T> (11.0) &&
           "Expected a value of 11.");

    avec.set(std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0)
    }));
    bvec.set(std::vector<T> ({
        static_cast<T> (3.0),
        static_cast<T> (4.0)
    }));
    const backend::buffer<T> fma_vec_scale_vec = backend::fma(avec, bscalar, bvec);
    assert(fma_vec_scale_vec.size() == 2 && "Expected a size of 2");
    assert(fma_vec_scale_vec.at(0) == static_cast<T> (6.0) &&
           "Expected a value of 6.");
    assert(fma_vec_scale_vec.at(1) == static_cast<T> (10.0) &&
           "Expected a value of 10.");

    avec.set(std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0)
    }));
    bvec.set(std::vector<T> ({
        static_cast<T> (3.0),
        static_cast<T> (4.0)
    }));
    const backend::buffer<T> fma_scale_vec_vec = backend::fma(bscalar, avec, bvec);
    assert(fma_scale_vec_vec.size() == 2 && "Expected a size of 2");
    assert(fma_scale_vec_vec.at(0) == static_cast<T> (6.0) &&
           "Expected a value of 6.");
    assert(fma_scale_vec_vec.at(1) == static_cast<T> (10.0) &&
           "Expected a value of 10.");

    avec.set(std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0)
    }));
    const backend::buffer<T> fma_vec_scale_scale = backend::fma(avec, bscalar, cscalar);
    assert(fma_vec_scale_scale.size() == 2 && "Expected a size of 2");
    assert(fma_vec_scale_scale.at(0) == static_cast<T> (-2.0) &&
           "Expected a value of -1.");
    assert(fma_vec_scale_scale.at(1) == static_cast<T> (1.0) &&
           "Expected a value of -1.");

    avec.set(std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0)
    }));
    bscalar.set(static_cast<T> (3.0));
    const backend::buffer<T> fma_scale_vec_scale = backend::fma(bscalar, avec, cscalar);
    assert(fma_scale_vec_scale.size() == 2 && "Expected a size of 2");
    assert(fma_scale_vec_scale.at(0) == static_cast<T> (-2.0) &&
           "Expected a value of -2.");
    assert(fma_scale_vec_scale.at(1) == static_cast<T> (1.0) &&
           "Expected a value of 1.");

    avec.set(std::vector<T> ({
        static_cast<T> (1.0),
        static_cast<T> (2.0)
    }));
    const backend::buffer<T> fma_scale_scale_vec = backend::fma(bscalar, cscalar, avec);
    assert(fma_scale_scale_vec.size() == 2 && "Expected a size of 2");
    assert(fma_scale_scale_vec.at(0) == static_cast<T> (-14.0) &&
           "Expected a value of -14.");
    assert(fma_scale_scale_vec.at(1) == static_cast<T> (-13.0) &&
           "Expected a value of --13.");

    backend::buffer<T> ascalar(1, static_cast<T> (-4.0));
    cscalar.set(static_cast<T> (-5.0));
    const backend::buffer<T> fma_scale_scale_scale = backend::fma(ascalar, bscalar, cscalar);
    assert(fma_scale_scale_scale.size() == 1 && "Expected a size of 1");
    assert(fma_scale_scale_scale.at(0) == static_cast<T> (-17.0) &&
           "Expected a value of -17.");

//  Pow tests.
    backend::buffer<T> base_scalar(1, static_cast<T> (4.0));
    backend::buffer<T> exp_scalar(1, static_cast<T> (0.5));
    const backend::buffer<T> sqrt = backend::pow(base_scalar, exp_scalar);
    assert(sqrt.size() == 1 && "Expected a size of 1");
    assert(sqrt.at(0) == static_cast<T> (2.0) && "Expected 2.");

    base_scalar.set(static_cast<T> (4.0));
    exp_scalar.set(static_cast<T> (-0.5));
    const backend::buffer<T> one_over_sqrt = backend::pow(base_scalar, exp_scalar);
    assert(one_over_sqrt.size() == 1 && "Expected a size of 1");
    assert(one_over_sqrt.at(0) == static_cast<T> (0.5) &&
           "Expected 1/2.");

    base_scalar.set(static_cast<T> (4.0));
    exp_scalar.set(static_cast<T> (3.0));
    const backend::buffer<T> x_cubed = backend::pow(base_scalar, exp_scalar);
    assert(x_cubed.size() == 1 && "Expected a size of 1");
    assert(x_cubed.at(0) == static_cast<T> (64.0) &&
           "Expected 64.");

    base_scalar.set(static_cast<T> (4.0));
    exp_scalar.set(static_cast<T> (-2.0));
    const backend::buffer<T> x_over_cubed = backend::pow(base_scalar, exp_scalar);
    assert(x_over_cubed.size() == 1 && "Expected a size of 1");
    assert(x_over_cubed.at(0) ==
           static_cast<T> (1.0) /
           (static_cast<T> (4.0)*static_cast<T> (4.0)) &&
           "Expected 1/(4*4).");

    base_scalar.set(static_cast<T> (4.0));
    exp_scalar.set(static_cast<T> (-0.23));
    const backend::buffer<T> pow_gen = backend::pow(base_scalar, exp_scalar);
    assert(pow_gen.size() == 1 && "Expected a size of 1");
    assert(std::abs(pow_gen.at(0) -
                    std::pow(static_cast<T> (4.0),
                             static_cast<T> (-0.23))) <=
           std::abs(static_cast<T> (1.1102230246251565e-16)) &&
           "Expected 4^-0.23.");

    backend::buffer<T> base_vec(std::vector<T> ({
        static_cast<T> (4.0),
        static_cast<T> (2.0)
    }));
    exp_scalar.set(static_cast<T> (0.5));
    const backend::buffer<T> sqrt_vec = backend::pow(base_vec, exp_scalar);
    assert(sqrt_vec.size() == 2 && "Expected a size of 2");
    assert(sqrt_vec.at(0) == static_cast<T> (2.0) &&
           "Expected 2.");
    assert(sqrt_vec.at(1) == std::sqrt(static_cast<T> (2.0)) &&
           "Expected sqrt(2).");

    base_vec.set(std::vector<T> ({
        static_cast<T> (4.0),
        static_cast<T> (2.0)
    }));
    exp_scalar.set(static_cast<T> (-0.5));
    const backend::buffer<T> one_over_sqrt_vec = backend::pow(base_vec, exp_scalar);
    assert(one_over_sqrt_vec.size() == 2 && "Expected a size of 2");
    assert(one_over_sqrt_vec.at(0) == static_cast<T> (0.5) &&
           "Expected 2.");
    assert(one_over_sqrt_vec.at(1) == std::pow(static_cast<T> (2.0),
                                               static_cast<T> (-0.5)) &&
           "Expected 1/sqrt(2).");

    base_vec.set(std::vector<T> ({
        static_cast<T> (4.0),
        static_cast<T> (2.0)
    }));
    exp_scalar.set(static_cast<T> (3.0));
    const backend::buffer<T> x_cubed_vec = backend::pow(base_vec, exp_scalar);
    assert(x_cubed_vec.size() == 2 && "Expected a size of 2");
    assert(x_cubed_vec.at(0) == static_cast<T> (64.0) &&
           "Expected 64.");
    assert(x_cubed_vec.at(1) == static_cast<T> (8.0) &&
           "Expected 8.");

    base_vec.set(std::vector<T> ({
        static_cast<T> (4.0),
        static_cast<T> (2.0)
    }));
    exp_scalar.set(static_cast<T> (-0.23));
    const backend::buffer<T> x_over_cubed_vec = backend::pow(base_vec, exp_scalar);
    assert(x_over_cubed_vec.size() == 2 && "Expected a size of 2");
    assert(std::abs(x_over_cubed_vec.at(0) -
                    std::pow(static_cast<T> (4.0),
                             static_cast<T> (-0.23))) <=
           std::abs(static_cast<T> (1.1102230246251565e-16)) &&
           "Expected 4^-0.23.");
    assert(x_over_cubed_vec.at(1) == std::pow(static_cast<T> (2.0),
                                              static_cast<T> (-0.23)) &&
           "Expected 2^-0.23.");

    base_scalar.set(static_cast<T> (4.0));
    backend::buffer<T> exp_vec(std::vector<T> ({
        static_cast<T> (4.0),
        static_cast<T> (2.0)
    }));
    backend::buffer<T> scale_base = backend::pow(base_scalar, exp_vec);
    assert(scale_base.size() == 2 && "Expected a size of 2");
    assert(std::abs(scale_base.at(0) -
                    std::pow(static_cast<T> (4.0),
                             static_cast<T> (4.0))) <=
           std::abs(static_cast<T> (5.6843418860808015e-14)) &&
           "Expected 4^4.");
    assert(std::abs(scale_base.at(1) -
                    std::pow(static_cast<T> (4.0),
                             static_cast<T> (2.0))) <=
           std::abs(static_cast<T> (1.7763568394002505e-15)) &&
           "Expected 4^2.");

    const auto non_int = static_cast<T> (0.438763);
    base_scalar.set(non_int);
    exp_scalar.set(static_cast<T> (2.0));
    scale_base = backend::pow(base_scalar, exp_scalar);
    assert(scale_base.at(0) == std::pow(non_int,
                                        static_cast<T> (2.0)) &&
           "Expected x*x.");
    assert(scale_base.at(0) == non_int*non_int && "Expected x*x.");

    base_scalar.set(static_cast<T> (10.0));
    exp_scalar.set(static_cast<T> (0.0));
    const backend::buffer<T> pow_zero = backend::pow(base_scalar, exp_scalar);
    assert(pow_zero.at(0) == std::pow(static_cast<T> (10.0),
                                      static_cast<T> (0.0)) &&
           "Expected 10^0.");

    base_vec.set(std::vector<T> ({
        static_cast<T> (4.0),
        static_cast<T> (2.0)
    }));
    exp_vec.set(std::vector<T> ({
        static_cast<T> (-4.0),
        static_cast<T> (0.30)
    }));
    const backend::buffer<T> vec_vec = backend::pow(base_vec, exp_vec);
    assert(vec_vec.size() == 2 && "Expected a size of 2");
    assert(std::abs(vec_vec.at(0) -
                    std::pow(static_cast<T> (4.0),
                             static_cast<T> (-4.0))) <=
           std::abs(static_cast<T> (8.6736173798840355e-19)) &&
           "Expected 4^-4.");
    assert(vec_vec.at(1) == std::pow(static_cast<T> (2.0),
                                     static_cast<T> (0.30)) &&
           "Expected 2^0.30.");

    base_scalar.set(static_cast<T> (4.0));
    base_scalar.log();
    assert(base_scalar.size() == 1 && "Expected a size of 1");
    assert(base_scalar.at(0) == std::log(static_cast<T> (4.0)) &&
           "Expected ln(4).");

    base_vec.set(std::vector<T> ({
        static_cast<T> (4.0),
        static_cast<T> (2.0)
    }));
    base_vec.log();
    assert(base_vec.size() == 2 && "Expected a size of 2");
    assert(base_vec.at(0) == std::log(static_cast<T> (4.0)) &&
           "Expected ln(4).");
    assert(base_vec.at(1) == std::log(static_cast<T> (2.0)) &&
           "Expected ln(2).");

    base_vec.set(std::vector<T> ({
        static_cast<T> (-4.0),
        static_cast<T> (-2.0)
    }));
    assert(base_vec.is_negative() && "Expected true.");
    base_vec.set(std::vector<T> ({
        static_cast<T> (-4.0),
        static_cast<T> (2.0)
    }));
    assert(!base_vec.is_negative() && "Expected false.");

    backend::buffer<T> has_zero_vec(std::vector<T> ({
        static_cast<T> (3.0),
        static_cast<T> (0.0)
    }));
    assert(has_zero_vec.has_zero() && "Expected zero.");
    backend::buffer<T> has_zero_vec2(std::vector<T> ({
        static_cast<T> (3.0),
        static_cast<T> (1.0)
    }));
    assert(!has_zero_vec2.has_zero() && "Expected zero.");
    assert(has_zero_vec2.is_normal() && "Expected normal.");

    backend::buffer<T> inf_vec(std::vector<T> ({
        static_cast<T> (3.0),
        static_cast<T> (INFINITY)
    }));
    assert(!inf_vec.is_normal() && "Expected a inf.");

    backend::buffer<T> nan_vec(std::vector<T> ({
        static_cast<T> (3.0),
        static_cast<T> (NAN)
    }));
    assert(!nan_vec.is_normal() && "Expected a NaN.");
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
    test_backend<float> ();
    test_backend<double> ();
    test_backend<std::complex<float>> ();
    test_backend<std::complex<double>> ();
}
