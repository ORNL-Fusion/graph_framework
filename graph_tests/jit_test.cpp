//------------------------------------------------------------------------------
///  @file jit_test.cpp
///  @brief Tests for the jit code.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include "../graph_framework/jit.hpp"
#include "../graph_framework/math.hpp"

template<typename BASE> void compile(const std::string name,
                                     graph::input_nodes<backend::cpu<BASE>> inputs,
                                     graph::output_nodes<backend::cpu<BASE>> outputs,
                                     graph::map_nodes<backend::cpu<BASE>> setters) {
    for (auto output : outputs) {
        output->to_latex();
        std::cout << std::endl;
    }

    jit::kernel<backend::cpu<BASE>> source(name, inputs, outputs, setters);
    
    source.compile(name, inputs, outputs, 1, 1);
    
    source.print();
}

//------------------------------------------------------------------------------
///  @brief Run tests with a specified backend.
//------------------------------------------------------------------------------
template<typename BASE> void run_tests() {
    auto v1 = graph::variable<backend::cpu<BASE>> (1, "v1");
    auto v2 = graph::variable<backend::cpu<BASE>> (1, "v2");
    auto v3 = graph::variable<backend::cpu<BASE>> (1, "v3");
    
    v1->set(2.0);
    v2->set(3.0);
    v3->set(4.0);
    
    auto add_node = v1 + v2;
    
    compile<BASE> ("add_kernel",
                   {graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {add_node}, {});
    
    auto subtract_node = v1 - v2;
    
    compile<BASE> ("subtract_kernel",
                   {graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {subtract_node}, {});
    
    auto multiply_node = v1*v2;
    
    compile<BASE> ("multiply_kernel",
                   {graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {multiply_node}, {});
    
    auto divide_node = v1/v2;
    
    compile<BASE> ("divide_kernel",
                   {graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {divide_node}, {});
    
    auto fma_node = graph::fma(v1, v2, v3);

    compile<BASE> ("fma_kernel",
                   {graph::variable_cast(v1),
                    graph::variable_cast(v2),
                    graph::variable_cast(v3)},
                   {fma_node}, {});

    auto sqrt_node = graph::sqrt(v2);
    
    compile<BASE> ("sqrt_kernel",
                   {graph::variable_cast(v2)},
                   {sqrt_node}, {});

    auto log_node = graph::log(v1);
    
    compile<BASE> ("log_kernel",
                   {graph::variable_cast(v1)},
                   {log_node}, {});

    auto exp_node = graph::exp(v2);
    
    compile<BASE> ("exp_kernel",
                   {graph::variable_cast(v2)},
                   {exp_node}, {});

    auto pow_node = graph::pow(v1, v2);
    
    compile<BASE> ("pow_kernel",
                   {graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {pow_node}, {});

    auto divide_node_df = divide_node->df(v3);

    compile<BASE> ("divide_df_kernel",
                   {graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {divide_node_df}, {});

    v1->set(0.0);
    v2->set(0.0);
    
    compile<BASE> ("divide_by_zero_kernel",
                   {graph::variable_cast(v1),
                    graph::variable_cast(v2)},
                   {divide_node}, {});
    
    v3->set(0.0);
    
    auto multiply_divide_node = v3*divide_node;
    
    compile<BASE> ("multiply_divide_by_zero_kernel",
                   {graph::variable_cast(v1),
                    graph::variable_cast(v2),
                    graph::variable_cast(v3)},
                   {multiply_divide_node}, {});
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    run_tests<float> ();
}
