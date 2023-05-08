//------------------------------------------------------------------------------
///  @file newton.hpp
///  @brief Sets up the kernel for a newtons method.
///
///  Defines a dispersion function.
//------------------------------------------------------------------------------

#ifndef newton_h
#define newton_h

#include "arithmetic.hpp"
#include "workflow.hpp"

namespace solver {
//------------------------------------------------------------------------------
///  @brief Determine the value of vars to minimze the loss function.
///
///  This uses newtons methods to solver for D(x) = 0.
///
///  @params[in,out] work           Workflow manager.
///  @params[in]     vars           The unknowns to solver for.
///  @params[in]     inputs         Inputs for jit compile.
///  @params[in]     loss           Loss function.
///  @params[in]     tolarance      Tolarance to solve the dispersion function to.
///  @params[in]     max_iterations Maximum number of iterations before giving up.
//------------------------------------------------------------------------------
    template<typename T>
    void newton(workflow::manager<T> &work,
                graph::output_nodes<T> vars,
                graph::input_nodes<T> inputs,
                graph::shared_leaf<T> loss,
                const T tolarance = 1.0E-30,
                const size_t max_iterations = 1000) {
        auto fudge = graph::constant(tolarance);

        graph::map_nodes<T> setters;
        for (auto x : vars) {
            setters.push_back({x - loss/(loss->df(x) + fudge),
                               graph::variable_cast(x)});
        }

        work.add_converge_item(inputs, {loss}, setters, "loss_kernel",
                               tolarance, max_iterations);
    }
}
#endif /* newton_h */
