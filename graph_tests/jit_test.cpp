//------------------------------------------------------------------------------
///  @file jit_test.cpp
///  @brief Tests for the jit code.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include <iomanip>

#include "../graph_framework/jit.hpp"
#include "../graph_framework/math.hpp"
#include "../graph_framework/dispersion.hpp"

//------------------------------------------------------------------------------
///  @brief Compile kernal and check the result of the output.
///
///  @param[in] inputs    Kernel input nodes.
///  @param[in] outputs   Kernel output nodes.
///  @param[in] setters   Kernel set nodes.
///  @param[in] expected  Expected result.
///  @param[in] tolarance Check tolarances.
//------------------------------------------------------------------------------
template<typename BASE> void compile(graph::input_nodes<backend::cpu<BASE>> inputs,
                                     graph::output_nodes<backend::cpu<BASE>> outputs,
                                     graph::map_nodes<backend::cpu<BASE>> setters,
                                     const BASE expected,
                                     const BASE tolarance) {
    jit::kernel<backend::cpu<BASE>> source("test_kernel", inputs, outputs, setters);
    
    source.compile("test_kernel", inputs, outputs, 1);
    source.run();
    
    BASE result;
    source.copy_buffer(inputs.size(), &result);

    assert(std::abs(result - expected) <= tolarance &&
           "GPU and CPU values differ.");
}

//------------------------------------------------------------------------------
///  @brief Run tests to test simple math operations are the same.
//------------------------------------------------------------------------------
template<typename BASE> void run_math_tests() {
    auto v1 = graph::variable<backend::cpu<BASE>> (1, "v1");
    auto v2 = graph::variable<backend::cpu<BASE>> (1, "v2");
    auto v3 = graph::variable<backend::cpu<BASE>> (1, "v3");

    v1->set(2.0);
    v2->set(3.0);
    v3->set(4.0);

    auto add_node = v1 + v2;
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {add_node}, {},
                   add_node->evaluate().at(0), 0.0);

    auto add_node_dfdv1 = add_node->df(v1);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {add_node_dfdv1}, {},
                   add_node_dfdv1->evaluate().at(0), 0.0);

    auto add_node_dfdv2 = add_node->df(v2);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {add_node_dfdv2}, {},
                   add_node_dfdv2->evaluate().at(0), 0.0);

    auto add_node_dfdv3 = add_node->df(v3);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {add_node_dfdv3}, {},
                   add_node_dfdv3->evaluate().at(0), 0.0);

    auto subtract_node = v1 - v2;
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {subtract_node}, {},
                   subtract_node->evaluate().at(0), 0.0);

    auto subtract_node_dfdv1 = subtract_node->df(v1);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {subtract_node_dfdv1}, {},
                   subtract_node_dfdv1->evaluate().at(0), 0.0);

    auto subtract_node_dfdv2 = subtract_node->df(v2);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {subtract_node_dfdv2}, {},
                   subtract_node_dfdv2->evaluate().at(0), 0.0);

    auto subtract_node_dfdv3 = subtract_node->df(v3);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {subtract_node_dfdv3}, {},
                   subtract_node_dfdv3->evaluate().at(0), 0.0);

    auto multiply_node = v1*v2;
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {multiply_node}, {},
                   multiply_node->evaluate().at(0), 0.0);

    auto multiply_node_dfdv1 = multiply_node->df(v1);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {multiply_node_dfdv1}, {},
                   multiply_node_dfdv1->evaluate().at(0), 0.0);

    auto multiply_node_dfdv2 = multiply_node->df(v2);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {multiply_node_dfdv2}, {},
                   multiply_node_dfdv2->evaluate().at(0), 0.0);

    auto multiply_node_dfdv3 = multiply_node->df(v3);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {multiply_node_dfdv3}, {},
                   multiply_node_dfdv3->evaluate().at(0), 0.0);

    auto divide_node = v1/v2;
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {divide_node}, {},
                   divide_node->evaluate().at(0), 0.0);

    auto divide_node_dfdv1 = divide_node->df(v1);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {divide_node_dfdv1}, {},
                   divide_node_dfdv1->evaluate().at(0), 0.0);

    auto divide_node_dfdv2 = divide_node->df(v2);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {divide_node_dfdv2}, {},
                   divide_node_dfdv2->evaluate().at(0), 0.0);

    auto divide_node_dfdv3 = divide_node->df(v3);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {divide_node_dfdv3}, {},
                   divide_node_dfdv3->evaluate().at(0), 0.0);

    auto fma_node = graph::fma(v1, v2, v3);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2),
                    graph::variable_cast(v3)},
                   {fma_node}, {},
                   fma_node->evaluate().at(0), 0.0);

    auto fma_node_dfdv1 = fma_node->df(v1);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {fma_node_dfdv1}, {},
                   fma_node_dfdv1->evaluate().at(0), 0.0);

    auto fma_node_dfdv2 = fma_node->df(v2);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2),
                    graph::variable_cast(v3)},
                   {fma_node_dfdv2}, {},
                   fma_node_dfdv2->evaluate().at(0), 0.0);

    auto fma_node_dfdv3 = fma_node->df(v3);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2),
                    graph::variable_cast(v3)},
                   {fma_node_dfdv2}, {},
                   fma_node_dfdv2->evaluate().at(0), 0.0);

    auto fma_node_dfdv4 = fma_node->df(graph::variable<backend::cpu<BASE>> (1, ""));
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2),
                    graph::variable_cast(v3)},
                   {fma_node_dfdv4}, {},
                   fma_node_dfdv4->evaluate().at(0), 0.0);

    auto sqrt_node = graph::sqrt(v2);
    compile<BASE> ({graph::variable_cast(v2)},
                   {sqrt_node}, {},
                   sqrt_node->evaluate().at(0), 0.0);

    auto sqrt_node_dfdv1 = sqrt_node->df(v1);
    compile<BASE> ({graph::variable_cast(v2)},
                   {sqrt_node_dfdv1}, {},
                   sqrt_node_dfdv1->evaluate().at(0), 0.0);

    auto sqrt_node_dfdv2 = sqrt_node->df(v2);
    compile<BASE> ({graph::variable_cast(v2)},
                   {sqrt_node_dfdv2}, {},
                   sqrt_node_dfdv2->evaluate().at(0), 0.0);

    auto log_node = graph::log(v1);
    compile<BASE> ({graph::variable_cast(v1)},
                   {log_node}, {},
                   log_node->evaluate().at(0), 0.0);

    auto log_node_dfdv1 = log_node->df(v1);
    compile<BASE> ({graph::variable_cast(v1)},
                   {log_node_dfdv1}, {},
                   log_node_dfdv1->evaluate().at(0), 0.0);

    auto log_node_dfdv2 = log_node->df(v2);
    compile<BASE> ({graph::variable_cast(v1)},
                   {log_node_dfdv2}, {},
                   log_node_dfdv2->evaluate().at(0), 0.0);

    auto exp_node = graph::exp(v2);
    compile<BASE> ({graph::variable_cast(v2)},
                   {exp_node}, {},
                   exp_node->evaluate().at(0), 2.0E-6);

    auto exp_node_dfdv1 = exp_node->df(v1);
    compile<BASE> ({graph::variable_cast(v2)},
                   {exp_node_dfdv1}, {},
                   exp_node_dfdv1->evaluate().at(0), 0.0);

    auto exp_node_dfdv2 = exp_node->df(v2);
    compile<BASE> ({graph::variable_cast(v2)},
                   {exp_node_dfdv2}, {},
                   exp_node_dfdv2->evaluate().at(0), 2.0E-6);

    auto pow_node = graph::pow(v1, v2);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {pow_node}, {},
                   pow_node->evaluate().at(0), 0.0);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {pow_node}, {},
                   2.0*2.0*2.0, 0.0);

    auto pow_node_dfdv1 = pow_node->df(v1);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {pow_node_dfdv1}, {},
                   pow_node_dfdv1->evaluate().at(0), 0.0);

    auto pow_node_dfdv2 = pow_node->df(v2);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {pow_node_dfdv2}, {},
                   pow_node_dfdv2->evaluate().at(0), 0.0);

    auto pow_node_dfdv3 = pow_node->df(v3);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {pow_node_dfdv3}, {},
                   pow_node_dfdv3->evaluate().at(0), 0.0);

    auto v4 = graph::variable<backend::cpu<BASE>> (1, "v4");
    v4->set(0.57245);
    auto pow_non_int = graph::pow(v4, v1);
    compile<BASE> ({graph::variable_cast(v1),
                    graph::variable_cast(v4)},
                   {pow_non_int}, {},
                   pow_non_int->evaluate().at(0), 3.0E-8);
}

//------------------------------------------------------------------------------
///  @brief Run dispersion tests.
///
///  @param[in] eq Equilibrium for the dispersion function.
//------------------------------------------------------------------------------
template<class DISPERSION_FUNCTION>
void run_dispersion_test(equilibrium::unique_equilibrium<typename DISPERSION_FUNCTION::backend> &eq,
                         const typename DISPERSION_FUNCTION::backend::base tolarance) {
    
    auto w = graph::variable<typename DISPERSION_FUNCTION::backend> (1, "w");
    auto x = graph::variable<typename DISPERSION_FUNCTION::backend> (1, "x");
    auto y = graph::variable<typename DISPERSION_FUNCTION::backend> (1, "y");
    auto z = graph::variable<typename DISPERSION_FUNCTION::backend> (1, "z");
    auto kx = graph::variable<typename DISPERSION_FUNCTION::backend> (1, "kx");
    auto ky = graph::variable<typename DISPERSION_FUNCTION::backend> (1, "ky");
    auto kz = graph::variable<typename DISPERSION_FUNCTION::backend> (1, "kz");
    auto t = graph::variable<typename DISPERSION_FUNCTION::backend> (1, "t");

    w->set(1.0);
    x->set(1.0);
    y->set(1.0);
    z->set(1.0);
    kx->set(1.0);
    ky->set(1.0);
    kz->set(1.0);
    t->set(1.0);

    dispersion::dispersion_interface<DISPERSION_FUNCTION> D(w, kx, ky, kz, x, y, z, t, eq);
    auto residule = D.get_d();
    
    compile<typename DISPERSION_FUNCTION::backend::base> ({graph::variable_cast(w),
                                                           graph::variable_cast(x),
                                                           graph::variable_cast(y),
                                                           graph::variable_cast(z),
                                                           graph::variable_cast(kx),
                                                           graph::variable_cast(ky),
                                                           graph::variable_cast(kz),
                                                           graph::variable_cast(t)},
                                                          {residule}, {},
                                                          residule->evaluate().at(0),
                                                          tolarance);
}

//------------------------------------------------------------------------------
///  @brief Run dispersion tests.
//------------------------------------------------------------------------------
template<class BASE>
void run_dispersion_tests() {
    auto no_mag_eq = equilibrium::make_no_magnetic_field<backend::cpu<BASE>> ();
    
    run_dispersion_test<dispersion::bohm_gross<backend::cpu<BASE>>> (no_mag_eq, 0.0);
    
    auto slab_eq = equilibrium::make_no_magnetic_field<backend::cpu<BASE>> ();
    
    run_dispersion_test<dispersion::simple<backend::cpu<BASE>>> (slab_eq, 0.0);
    run_dispersion_test<dispersion::ordinary_wave<backend::cpu<BASE>>> (slab_eq, 0.0);
    run_dispersion_test<dispersion::extra_ordinary_wave<backend::cpu<BASE>>> (slab_eq, 0.0);
    run_dispersion_test<dispersion::cold_plasma<backend::cpu<BASE>>> (slab_eq, 5.0E9);
}

//------------------------------------------------------------------------------
///  @brief Run tests.
//------------------------------------------------------------------------------
template<typename BASE> void run_tests() {
    if constexpr (jit::can_jit<backend::cpu<BASE>> ()) {
        run_math_tests<BASE> ();
        run_dispersion_tests<BASE> ();
    }
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    run_tests<float> ();
    run_tests<double> ();
    run_tests<std::complex<float>> ();
    run_tests<std::complex<double>> ();
}
