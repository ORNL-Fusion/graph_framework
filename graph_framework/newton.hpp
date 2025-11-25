//------------------------------------------------------------------------------
///  @file newton.hpp
///  @brief Sets up the kernel for a newtons method.
///
///  Defines a dispersion function.
//------------------------------------------------------------------------------

#ifndef newton_h
#define newton_h

#include "workflow.hpp"

namespace solver {
//------------------------------------------------------------------------------
///  @brief Determine the value of vars to minimize the loss function.
///
///  This uses newtons methods to solver for D(x) = 0.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in,out] work           Workflow manager.
///  @param[in]     vars           The unknowns to solver for.
///  @param[in]     inputs         Inputs for jit compile.
///  @param[in]     func           Function to find the root of.
///  @param[in]     state          Random state node.
///  @param[in]     tolerance      Tolerance to solve the dispersion function
///                                to.
///  @param[in]     max_iterations Maximum number of iterations before giving
///                                up.
///  @param[in]     step           Newton step size.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    void newton(workflow::manager<T, SAFE_MATH> &work,
                graph::output_nodes<T, SAFE_MATH> vars,
                graph::input_nodes<T, SAFE_MATH> inputs,
                graph::shared_leaf<T, SAFE_MATH> func,
                graph::shared_random_state<T, SAFE_MATH> state,
                const T tolerance = 1.0E-30,
                const size_t max_iterations = 1000,
                const T step = 1.0) {
        graph::map_nodes<T, SAFE_MATH> setters;
        for (auto x : vars) {
            setters.push_back({x - step*func/func->df(x),
                               graph::variable_cast(x)});
        }

        work.add_converge_item(inputs, {func*func}, setters, state,
                               "loss_kernel", inputs.back()->size(),
                               tolerance, max_iterations);
    }
}
#endif /* newton_h */
