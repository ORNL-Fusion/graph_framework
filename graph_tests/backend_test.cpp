//------------------------------------------------------------------------------
///  @file backend_test.cpp
///  @brief Tests for the cpu backend.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/cpu_backend.hpp"

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
template<typename BACKEND> void test_backend() {
    BACKEND size_one(1);
    assert(size_one.size() == 1 && "Expected a size of one.");
    size_one[0] = 10;
    assert(size_one.at(0) == 10 && "Expected a value of 10.");
    assert(size_one[0] == 10 && "Expected a value of 10.");

    BACKEND size_three(3, 2.0);
    assert(size_three.size() == 3 && "Expected a size of three.");
    for (size_t i = 0; i < 3; i++) {
        assert(size_three.at(i) == 2.0 && "Expected a value of 10.");
        assert(size_three[i] == 2.0 && "Expected a value of 10.");
    }

    BACKEND vector(std::vector<double> ({1.0, 2.0, 3.0, 4.0}));
    assert(vector.size() == 4 && "Expected a size of three.");
    for (size_t i = 0; i < 4; i++) {
        assert(vector.at(i) == i + 1 &&
               "Expected a value one more than index.");
        assert(vector[i] == i + 1 &&
               "Expected a value one more than index.");
    }

    BACKEND copy(vector);
    assert(copy.size() == 4 && "Expected a size of four.");
    for (size_t i = 0; i < 4; i++) {
        assert(copy.at(i) == i + 1 &&
               "Expected a value one more than index.");
        assert(copy[i] == i + 1 &&
               "Expected a value one more than index.");
    }

    copy.set(5.0);
    for (size_t i = 0; i < 3; i++) {
        assert(copy.at(i) == 5.0 && "Expected a value of 5.");
        assert(copy[i] == 5.0 && "Expected a value of 5.");
    }

    assert(vector.max() == 4.0 && "Expected a value of 4.");

    assert(!vector.is_same() && "Expected different values.");
    assert(copy.is_same() && "Expected same values.");

    assert(!vector.is_zero() && "Expected non zero values.");
    assert(!copy.is_zero() && "Expected non zero values.");
    copy.set(0);
    assert(copy.is_zero() && "Expected zero values.");

    vector.sqrt();
    for (size_t i = 0; i < 4; i++) {
        assert(vector.at(i) == std::sqrt(i + 1) &&
               "Expected a value sqrt(one more than index).");
    }

    BACKEND vector2(std::vector<double> ({1.0, 2.0, 3.0, 4.0}));
    vector2.exp();
    for (size_t i = 0; i < 4; i++) {
        assert(vector2.at(i) == std::exp(i + 1) &&
               "Expected a value exp(one more than index).");
    }

//  Addition tests.
    BACKEND avec(std::vector<double> ({1.0, 2.0}));
    BACKEND bvec(std::vector<double> ({3.0, 4.0}));
    BACKEND add_vec_vec = avec + bvec;
    assert(add_vec_vec.size() == 2 && "Expected a size of 2");
    assert(add_vec_vec.at(0) == 4.0 && "Expected a value of 4.");
    assert(add_vec_vec.at(1) == 6.0 && "Expected a value of 6.");

    BACKEND bscalar(1, 3.0);
    avec.set(std::vector<double> ({1.0, 2.0}));
    BACKEND add_vec_scale = avec + bscalar;
    assert(add_vec_scale.size() == 2 && "Expected a size of 2");
    assert(add_vec_scale.at(0) == 4.0 && "Expected a value of 4.");
    assert(add_vec_scale.at(1) == 5.0 && "Expected a value of 5.");

    BACKEND add_scale_vec = bscalar + bvec;
    assert(add_scale_vec.size() == 2 && "Expected a size of 2");
    assert(add_scale_vec.at(0) == 6.0 && "Expected a value of 6.");
    assert(add_scale_vec.at(1) == 7.0 && "Expected a value of 7.");

    BACKEND cscalar(1, -5.0);
    BACKEND add_scale_scale = bscalar + cscalar;
    assert(add_scale_scale.size() == 1 && "Expected a size of 1");
    assert(add_scale_scale.at(0) == -2.0 && "Expected a value of -4.");

//  Subtraction tests.
    BACKEND sub_vec_vec = avec - bvec;
    assert(sub_vec_vec.size() == 2 && "Expected a size of 2");
    assert(sub_vec_vec.at(0) == -2.0 && "Expected a value of -2.");
    assert(sub_vec_vec.at(1) == -2.0 && "Expected a value of -2.");

    avec.set(std::vector<double> ({1.0, 2.0}));
    bscalar.set(3.0);
    BACKEND sub_vec_scale = avec - bscalar;
    assert(sub_vec_scale.size() == 2 && "Expected a size of 2");
    assert(sub_vec_scale.at(0) == -2.0 && "Expected a value of -2.");
    assert(sub_vec_scale.at(1) == -1.0 && "Expected a value of -1.");

    avec.set(std::vector<double> ({1.0, 2.0}));
    cscalar.set(-5.0);
    BACKEND sub_scale_vec = cscalar - avec;
    assert(sub_scale_vec.size() == 2 && "Expected a size of 2");
    assert(sub_scale_vec.at(0) == -6.0 && "Expected a value of -6.");
    assert(sub_scale_vec.at(1) == -7.0 && "Expected a value of -7.");

    BACKEND sub_scale_scale = bscalar - cscalar;
    assert(sub_scale_scale.size() == 1 && "Expected a size of 1");
    assert(sub_scale_scale.at(0) == 8.0 && "Expected a value of 8.");

//  Multiply tests.
    avec.set(std::vector<double> ({1.0, 2.0}));
    bvec.set(std::vector<double> ({3.0, 4.0}));
    BACKEND mul_vec_vec = avec*bvec;
    assert(mul_vec_vec.size() == 2 && "Expected a size of 2");
    assert(mul_vec_vec.at(0) == 3.0 && "Expected a value of 3.");
    assert(mul_vec_vec.at(1) == 8.0 && "Expected a value of 8.");

    avec.set(std::vector<double> ({1.0, 2.0}));
    bscalar.set(3.0);
    BACKEND mul_vec_scale = avec*bscalar;
    assert(mul_vec_scale.size() == 2 && "Expected a size of 2");
    assert(mul_vec_scale.at(0) == 3.0 && "Expected a value of 3.");
    assert(mul_vec_scale.at(1) == 6.0 && "Expected a value of 6.");

    BACKEND mul_scale_vec = cscalar*bvec;
    assert(mul_scale_vec.size() == 2 && "Expected a size of 2");
    assert(mul_scale_vec.at(0) == -15.0 && "Expected a value of -15.");
    assert(mul_scale_vec.at(1) == -20.0 && "Expected a value of -20.");

    BACKEND mul_scale_scale = bscalar*cscalar;
    assert(mul_scale_scale.size() == 1 && "Expected a size of 1");
    assert(mul_scale_scale.at(0) == -15.0 && "Expected a value of -15.");

//  Divide tests.
    avec.set(std::vector<double> ({1.0, 2.0}));
    bvec.set(std::vector<double> ({3.0, 4.0}));
    BACKEND div_vec_vec = avec/bvec;
    assert(div_vec_vec.size() == 2 && "Expected a size of 2");
    assert(div_vec_vec.at(0) == 1.0/3.0 && "Expected a value of 1/3.");
    assert(div_vec_vec.at(1) == 2.0/4.0 && "Expected a value of 2/4.");

    avec.set(std::vector<double> ({1.0, 2.0}));
    bscalar.set(3.0);
    BACKEND div_vec_scale = avec/bscalar;
    assert(div_vec_scale.size() == 2 && "Expected a size of 2");
    assert(div_vec_scale.at(0) == 1.0/3.0 && "Expected a value of 1/3.");
    assert(div_vec_scale.at(1) == 2.0/3.0 && "Expected a value of 2/3.");

    avec.set(std::vector<double> ({1.0, 2.0}));
    BACKEND div_scale_vec = cscalar/avec;
    assert(div_scale_vec.size() == 2 && "Expected a size of 2");
    assert(div_scale_vec.at(0) == -5.0 && "Expected a value of -5.");
    assert(div_scale_vec.at(1) == -5.0/2.0 && "Expected a value of -5/2.");

    BACKEND div_scale_scale = bscalar/cscalar;
    assert(div_scale_scale.size() == 1 && "Expected a size of 1");
    assert(div_scale_scale.at(0) == -3.0/5.0 && "Expected a value of -3/5.");

#ifdef USE_FMA
//  Fused multiply add tests.
    avec.set(std::vector<double> ({1.0, 2.0}));
    bvec.set(std::vector<double> ({3.0, 4.0}));
    BACKEND cvec(std::vector<double> ({-2.0, 6.0}));
    BACKEND fma_vec_vec_vec = backend::fma(avec, bvec, cvec);
    assert(fma_vec_vec_vec.size() == 2 && "Expected a size of 2");
    assert(fma_vec_vec_vec.at(0) == 1.0 && "Expected a value of 1.");
    assert(fma_vec_vec_vec.at(1) == 14.0 && "Expected a value of 14.");

    avec.set(std::vector<double> ({1.0, 2.0}));
    bvec.set(std::vector<double> ({3.0, 4.0}));
    bscalar.set(3.0);
    BACKEND fma_vec_vec_scale = backend::fma(avec, bvec, bscalar);
    assert(fma_vec_vec_scale.size() == 2 && "Expected a size of 2");
    assert(fma_vec_vec_scale.at(0) == 6.0 && "Expected a value of 6.");
    assert(fma_vec_vec_scale.at(1) == 11.0 && "Expected a value of 11.");

    avec.set(std::vector<double> ({1.0, 2.0}));
    bvec.set(std::vector<double> ({3.0, 4.0}));
    BACKEND fma_vec_scale_vec = backend::fma(avec, bscalar, bvec);
    assert(fma_vec_scale_vec.size() == 2 && "Expected a size of 2");
    assert(fma_vec_scale_vec.at(0) == 6.0 && "Expected a value of 6.");
    assert(fma_vec_scale_vec.at(1) == 10.0 && "Expected a value of 10.");

    avec.set(std::vector<double> ({1.0, 2.0}));
    bvec.set(std::vector<double> ({3.0, 4.0}));
    BACKEND fma_scale_vec_vec = backend::fma(bscalar, avec, bvec);
    assert(fma_scale_vec_vec.size() == 2 && "Expected a size of 2");
    assert(fma_scale_vec_vec.at(0) == 6.0 && "Expected a value of 6.");
    assert(fma_scale_vec_vec.at(1) == 10.0 && "Expected a value of 10.");

    avec.set(std::vector<double> ({1.0, 2.0}));
    BACKEND fma_vec_scale_scale = backend::fma(avec, bscalar, cscalar);
    assert(fma_vec_scale_scale.size() == 2 && "Expected a size of 2");
    assert(fma_vec_scale_scale.at(0) == -2.0 && "Expected a value of -1.");
    assert(fma_vec_scale_scale.at(1) == 1.0 && "Expected a value of -1.");

    avec.set(std::vector<double> ({1.0, 2.0}));
    bscalar.set(3.0);
    BACKEND fma_scale_vec_scale = backend::fma(bscalar, avec, cscalar);
    assert(fma_scale_vec_scale.size() == 2 && "Expected a size of 2");
    assert(fma_scale_vec_scale.at(0) == -2.0 && "Expected a value of -2.");
    assert(fma_scale_vec_scale.at(1) == 1.0 && "Expected a value of 1.");

    avec.set(std::vector<double> ({1.0, 2.0}));
    BACKEND fma_scale_scale_vec = backend::fma(bscalar, cscalar, avec);
    assert(fma_scale_scale_vec.size() == 2 && "Expected a size of 2");
    assert(fma_scale_scale_vec.at(0) == -14.0 && "Expected a value of -14.");
    assert(fma_scale_scale_vec.at(1) == -13.0 && "Expected a value of --13.");

    BACKEND ascalar(1, -4.0);
    cscalar.set(-5.0);
    BACKEND fma_scale_scale_scale = backend::fma(ascalar, bscalar, cscalar);
    assert(fma_scale_scale_scale.size() == 1 && "Expected a size of 1");
    assert(fma_scale_scale_scale.at(0) == -17.0 && "Expected a value of -17.");
#endif
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    test_backend<backend::cpu> ();
}
