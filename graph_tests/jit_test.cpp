//------------------------------------------------------------------------------
///  @file jit_test.cpp
///  @brief Tests for the jit code.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/dispersion.hpp"

//------------------------------------------------------------------------------
///  @brief Assert when difference is greater than the tolerance.
///
///  Specialize to check for complex numbers since complex has not <= operator.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] test      Test value.
///  @param[in] tolerance Test tolerance.
//------------------------------------------------------------------------------
template<jit::float_scalar T> 
void check(const T test, const T tolerance) {
    if constexpr (jit::complex_scalar<T>) {
        assert(std::real(test) <= std::real(tolerance) &&
               "Real GPU and CPU values differ.");
        assert(std::imag(test) <= std::imag(tolerance) &&
               "Imaginary GPU and CPU values differ.");
    } else {
        assert(test <= tolerance && "GPU and CPU values differ.");
    }
}

//------------------------------------------------------------------------------
///  @brief Compile kernal and check the result of the output.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] inputs    Kernel input nodes.
///  @param[in] outputs   Kernel output nodes.
///  @param[in] setters   Kernel set nodes.
///  @param[in] expected  Expected result.
///  @param[in] tolerance Check tolerances.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
void compile(graph::input_nodes<T> inputs,
             graph::output_nodes<T> outputs,
             graph::map_nodes<T> setters,
             const T expected,
             const T tolerance) {
    jit::context<T> source(0);
    source.add_kernel("test_kernel", inputs, outputs, setters,
                      graph::shared_random_state<T> (),
                      inputs.back()->size());

    source.compile();

    auto run = source.create_kernel_call("test_kernel", inputs, outputs,
                                         graph::shared_random_state<T> (), 1);
    run();

    T result;
    source.copy_to_host(outputs.back(), &result);

    const T diff = std::abs(result - expected);
    check(diff, tolerance);
}

//------------------------------------------------------------------------------
///  @brief Run tests to test simple math operations are the same.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void run_math_tests() {
    auto v1 = graph::variable<T> (1, "v1");
    auto v2 = graph::variable<T> (1, "v2");
    auto v3 = graph::variable<T> (1, "v3");

    v1->set(static_cast<T> (2.0));
    v2->set(static_cast<T> (3.0));
    v3->set(static_cast<T> (4.0));

    auto add_node = v1 + v2;
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {add_node}, {},
                add_node->evaluate().at(0), 0.0);

    auto add_node_dfdv1 = add_node->df(v1);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {add_node_dfdv1}, {},
                add_node_dfdv1->evaluate().at(0), 0.0);

    auto add_node_dfdv2 = add_node->df(v2);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {add_node_dfdv2}, {},
                add_node_dfdv2->evaluate().at(0), 0.0);

    auto add_node_dfdv3 = add_node->df(v3);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {add_node_dfdv3}, {},
                add_node_dfdv3->evaluate().at(0), 0.0);

    auto subtract_node = v1 - v2;
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {subtract_node}, {},
                subtract_node->evaluate().at(0), 0.0);

    auto subtract_node_dfdv1 = subtract_node->df(v1);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {subtract_node_dfdv1}, {},
                subtract_node_dfdv1->evaluate().at(0), 0.0);

    auto subtract_node_dfdv2 = subtract_node->df(v2);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {subtract_node_dfdv2}, {},
                subtract_node_dfdv2->evaluate().at(0), 0.0);

    auto subtract_node_dfdv3 = subtract_node->df(v3);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {subtract_node_dfdv3}, {},
                subtract_node_dfdv3->evaluate().at(0), 0.0);

    auto multiply_node = v1*v2;
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {multiply_node}, {},
                multiply_node->evaluate().at(0), 0.0);

    auto multiply_node_dfdv1 = multiply_node->df(v1);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {multiply_node_dfdv1}, {},
                multiply_node_dfdv1->evaluate().at(0), 0.0);

    auto multiply_node_dfdv2 = multiply_node->df(v2);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {multiply_node_dfdv2}, {},
                multiply_node_dfdv2->evaluate().at(0), 0.0);

    auto multiply_node_dfdv3 = multiply_node->df(v3);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {multiply_node_dfdv3}, {},
                multiply_node_dfdv3->evaluate().at(0), 0.0);

    auto divide_node = v1/v2;
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {divide_node}, {},
                divide_node->evaluate().at(0), 0.0);

    T result;
    if constexpr (jit::use_cuda()) {
        result = 2.8E-17;
    } else {
        result = 0.0;
    }

    auto divide_node_dfdv1 = divide_node->df(v1);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {divide_node_dfdv1}, {},
                divide_node_dfdv1->evaluate().at(0),
                result);

    if constexpr (!jit::use_gpu<T> ()) {
        result = 2.8E-17;
    }

    auto divide_node_dfdv2 = divide_node->df(v2);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {divide_node_dfdv2}, {},
                divide_node_dfdv2->evaluate().at(0),
                result);

    auto divide_node_dfdv3 = divide_node->df(v3);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {divide_node_dfdv3}, {},
                divide_node_dfdv3->evaluate().at(0), 0.0);

    auto fma_node = graph::fma(v1, v2, v3);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2),
                 graph::variable_cast(v3)},
                {fma_node}, {},
                fma_node->evaluate().at(0), 0.0);

    auto fma_node_dfdv1 = fma_node->df(v1);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {fma_node_dfdv1}, {},
                fma_node_dfdv1->evaluate().at(0), 0.0);

    auto fma_node_dfdv2 = fma_node->df(v2);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2),
                 graph::variable_cast(v3)},
                {fma_node_dfdv2}, {},
                fma_node_dfdv2->evaluate().at(0), 0.0);

    auto fma_node_dfdv3 = fma_node->df(v3);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2),
                 graph::variable_cast(v3)},
                {fma_node_dfdv2}, {},
                fma_node_dfdv2->evaluate().at(0), 0.0);

    auto fma_node_dfdv4 = fma_node->df(graph::variable<T> (1, ""));
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2),
                 graph::variable_cast(v3)},
                {fma_node_dfdv4}, {},
                fma_node_dfdv4->evaluate().at(0), 0.0);

    auto sqrt_node = graph::sqrt(v2);
    compile<T> ({graph::variable_cast(v2)},
                {sqrt_node}, {},
                sqrt_node->evaluate().at(0), 0.0);

    auto sqrt_node_dfdv1 = sqrt_node->df(v1);
    compile<T> ({graph::variable_cast(v2)},
                {sqrt_node_dfdv1}, {},
                sqrt_node_dfdv1->evaluate().at(0), 0.0);

    auto sqrt_node_dfdv2 = sqrt_node->df(v2);
    compile<T> ({graph::variable_cast(v2)},
                {sqrt_node_dfdv2}, {},
                sqrt_node_dfdv2->evaluate().at(0), 0.0);

    auto log_node = graph::log(v1);
    compile<T> ({graph::variable_cast(v1)},
                {log_node}, {},
                log_node->evaluate().at(0), 0.0);

    auto log_node_dfdv1 = log_node->df(v1);
    compile<T> ({graph::variable_cast(v1)},
                {log_node_dfdv1}, {},
                log_node_dfdv1->evaluate().at(0), 0.0);

    auto log_node_dfdv2 = log_node->df(v2);
    compile<T> ({graph::variable_cast(v1)},
                {log_node_dfdv2}, {},
                log_node_dfdv2->evaluate().at(0), 0.0);

    auto exp_node = graph::exp(v2);
    compile<T> ({graph::variable_cast(v2)},
                {exp_node}, {},
                exp_node->evaluate().at(0), 2.0E-6);

    auto exp_node_dfdv1 = exp_node->df(v1);
    compile<T> ({graph::variable_cast(v2)},
                {exp_node_dfdv1}, {},
                exp_node_dfdv1->evaluate().at(0), 0.0);

    auto exp_node_dfdv2 = exp_node->df(v2);
    compile<T> ({graph::variable_cast(v2)},
                {exp_node_dfdv2}, {},
                exp_node_dfdv2->evaluate().at(0), 2.0E-6);

    if constexpr (jit::use_cuda() || jit::use_hip() || !jit::use_gpu<T> ()) {
        result = 1.8E-15;
    } else {
        result = 0.0;
    }

    auto pow_node = graph::pow(v1, v2);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {pow_node}, {},
                pow_node->evaluate().at(0),
                result);

    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {pow_node}, {},
                2.0*2.0*2.0,
                result);

    auto pow_node_dfdv1 = pow_node->df(v1);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {pow_node_dfdv1}, {},
                pow_node_dfdv1->evaluate().at(0), 0.0);

    if constexpr (jit::use_cuda() || jit::use_hip() || !jit::use_gpu<T> ()) {
        result = 8.9E-16;
    } else {
        result = 0.0;
    }

    auto pow_node_dfdv2 = pow_node->df(v2);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {pow_node_dfdv2}, {},
                pow_node_dfdv2->evaluate().at(0),
                result);

    auto pow_node_dfdv3 = pow_node->df(v3);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v2)},
                {pow_node_dfdv3}, {},
                pow_node_dfdv3->evaluate().at(0), 0.0);

    auto v4 = graph::variable<T> (1, "v4");
    v4->set(static_cast<T> (0.57245));
    auto pow_non_int = graph::pow(v4, v1);
    compile<T> ({graph::variable_cast(v1),
                 graph::variable_cast(v4)},
                {pow_non_int}, {},
                pow_non_int->evaluate().at(0), 3.0E-8);

    if constexpr (jit::use_gpu<T> ()) {
        result = 9.0E-8;
    } else {
        result = 0.0;
    }

    auto sin_node = graph::sin(v1);
    compile<T> ({graph::variable_cast(v1)},
                {sin_node}, {},
                sin_node->evaluate().at(0), 0.0);

    auto cos_node = graph::cos(v1);
    compile<T> ({graph::variable_cast(v1)},
                {cos_node}, {},
                cos_node->evaluate().at(0), result);

    auto atan_node = graph::atan(v1, v2);
    compile<T> ({
        graph::variable_cast(v1),
        graph::variable_cast(v2)
    }, {atan_node}, {}, atan_node->evaluate().at(0), result);
}

//------------------------------------------------------------------------------
///  @brief Run dispersion tests.
///
///  @tparam DISPERSION_FUNCTION Class of dispersion function to use.
///
///  @param[in,out] eq        Equilibrium for the dispersion function.
///  @param[in]     tolerance The test tolerance for check results to.
//------------------------------------------------------------------------------
template<dispersion::function DISPERSION_FUNCTION>
void run_dispersion_test(equilibrium::shared<typename DISPERSION_FUNCTION::base> &eq,
                         const typename DISPERSION_FUNCTION::base tolerance) {

    auto w = graph::variable<typename DISPERSION_FUNCTION::base> (1, "w");
    auto x = graph::variable<typename DISPERSION_FUNCTION::base> (1, "x");
    auto y = graph::variable<typename DISPERSION_FUNCTION::base> (1, "y");
    auto z = graph::variable<typename DISPERSION_FUNCTION::base> (1, "z");
    auto kx = graph::variable<typename DISPERSION_FUNCTION::base> (1, "kx");
    auto ky = graph::variable<typename DISPERSION_FUNCTION::base> (1, "ky");
    auto kz = graph::variable<typename DISPERSION_FUNCTION::base> (1, "kz");
    auto t = graph::variable<typename DISPERSION_FUNCTION::base> (1, "t");

    w->set(static_cast<typename DISPERSION_FUNCTION::base> (1.0));
    x->set(static_cast<typename DISPERSION_FUNCTION::base> (1.0));
    y->set(static_cast<typename DISPERSION_FUNCTION::base> (1.0));
    z->set(static_cast<typename DISPERSION_FUNCTION::base> (1.0));
    kx->set(static_cast<typename DISPERSION_FUNCTION::base> (1.0));
    ky->set(static_cast<typename DISPERSION_FUNCTION::base> (1.0));
    kz->set(static_cast<typename DISPERSION_FUNCTION::base> (1.0));
    t->set(static_cast<typename DISPERSION_FUNCTION::base> (1.0));

    dispersion::dispersion_interface<DISPERSION_FUNCTION> D(w, kx, ky, kz, x, y, z, t, eq);
    auto residual = D.get_d();

    compile<typename DISPERSION_FUNCTION::base> ({graph::variable_cast(w),
                                                  graph::variable_cast(x),
                                                  graph::variable_cast(y),
                                                  graph::variable_cast(z),
                                                  graph::variable_cast(kx),
                                                  graph::variable_cast(ky),
                                                  graph::variable_cast(kz),
                                                  graph::variable_cast(t)},
                                                 {residual}, {},
                                                 residual->evaluate().at(0),
                                                 tolerance);
}

//------------------------------------------------------------------------------
///  @brief Run dispersion tests.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<typename T>
void run_dispersion_tests() {
    auto no_mag_eq = equilibrium::make_no_magnetic_field<T> ();

    run_dispersion_test<dispersion::bohm_gross<T>> (no_mag_eq, 0.0);

    auto slab_eq = equilibrium::make_no_magnetic_field<T> ();

    run_dispersion_test<dispersion::simple<T>> (slab_eq, 0.0);
    run_dispersion_test<dispersion::ordinary_wave<T>> (slab_eq, 0.0);
    if constexpr (jit::use_cuda()) {
        run_dispersion_test<dispersion::extra_ordinary_wave<T>> (slab_eq, 0.032);
    } else {
        run_dispersion_test<dispersion::extra_ordinary_wave<T>> (slab_eq, 0.0);
    }

    if constexpr (jit::use_cuda()) {
        run_dispersion_test<dispersion::cold_plasma<T>> (slab_eq, 1.4E10);
    } else if constexpr (jit::use_metal<T> ()) {
        run_dispersion_test<dispersion::cold_plasma<T>> (slab_eq, 5.0E9);
    } else if constexpr (jit::complex_scalar<T>){
        run_dispersion_test<dispersion::cold_plasma<T>> (slab_eq, 1.5E11);
    } else {
        run_dispersion_test<dispersion::cold_plasma<T>> (slab_eq, 5.1E10);
    }
}

//------------------------------------------------------------------------------
///  @brief Run tests.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T> void run_tests() {
    run_math_tests<T> ();
    run_dispersion_tests<T> ();
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    START_GPU
    (void)argc;
    (void)argv;
    jit::verbose = true;
    run_tests<float> ();
    run_tests<double> ();
    run_tests<std::complex<float>> ();
    run_tests<std::complex<double>> ();
    END_GPU
}
