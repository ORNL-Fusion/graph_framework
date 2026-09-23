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
//------------------------------------------------------------------------------
///  @brief A dummy constructor.
//------------------------------------------------------------------------------
    dummy() :
    graph::no_derivative<float, false, graph::leaf_node<float, false>> ("") {}

//------------------------------------------------------------------------------
///  @brief Dummy evaluate method.
///
///  @returns An empty buffer.
//------------------------------------------------------------------------------
    virtual backend::buffer<float> evaluate() {
        return backend::buffer<float> ();
    };

//------------------------------------------------------------------------------
///  @brief Dummy reduce method.
///
///  @returns Returns the dummy node.
//------------------------------------------------------------------------------
    virtual graph::shared_leaf<float>
    compile(std::ostringstream &stream,
            jit::register_map &registers,
            const jit::register_map &thread_mem,
            const jit::register_usage &usage) {
        return this->shared_from_this();
    }

//------------------------------------------------------------------------------
///  @brief Dummy to vizgraph method.
///
///  @returns A reference to this.
//------------------------------------------------------------------------------
    virtual graph::shared_leaf<float> to_vizgraph(std::stringstream &stream,
                                                  jit::register_map &registers) {
        return this->shared_from_this();
    }

//------------------------------------------------------------------------------
///  @brief Dummy get power exponent method.
///
///  @returns One.
//------------------------------------------------------------------------------
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
