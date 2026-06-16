//------------------------------------------------------------------------------
///  @file no_derivative_test.cpp
///  @brief Test for nodes with no derivatives.
//------------------------------------------------------------------------------

#include "../graph_framework/node.hpp"

//------------------------------------------------------------------------------
///  @brief Dummy node.
//------------------------------------------------------------------------------
class dummy : public graph::no_derivative<float, false, graph::leaf_node<float, false>> {
public:
    dummy() : graph::no_derivative<float, false, graph::leaf_node<float, false>> ("") {}

    virtual backend::buffer<float> evaluate() {
        return backend::buffer<float> ();
    };

    virtual graph::shared_leaf<float>
    compile(std::ostringstream &stream,
            jit::register_map &registers,
            jit::register_map &indices,
            const jit::register_usage &usage) {
        return this->shared_from_this();
    }

    virtual graph::shared_leaf<float> to_vizgraph(std::stringstream &stream,
                                                  jit::register_map &registers) {
        return this->shared_from_this();
    }

    virtual bool is_all_variables() const {
        return false;
    }

    virtual graph::shared_leaf<float> get_power_exponent() const {
        return graph::one<float> ();
    }
};

//------------------------------------------------------------------------------
///  @brief Test function.
///
///  This test checks for a failure to compiler if a df method is called on a
///  node without a derivative.
//------------------------------------------------------------------------------
void test() {
    dummy a;
#ifndef CHECK_TEST
    a.df(a);
#endif
}
