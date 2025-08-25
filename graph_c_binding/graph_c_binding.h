//------------------------------------------------------------------------------
///  @file graph_c_binding.h
///  @brief Header file for the c binding library.
//------------------------------------------------------------------------------

#ifndef graph_c_binding_h
#define graph_c_binding_h

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

//------------------------------------------------------------------------------
///  @def START_GPU
///  Starts a Cocoa auto release pool when using the metal backend. No opt
///  otherwise.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
///  @def END_GPU
///  Ends a Cocoa auto release pool when using the metal backend. No opt
///  otherwise.
//------------------------------------------------------------------------------
#ifdef USE_METAL
#define START_GPU @autoreleasepool {
#define END_GPU }
#else
#define START_GPU
#define END_GPU
#endif

//------------------------------------------------------------------------------
///  @def STRUCT_TAG
///  C++ mode needs to tag a graph_c_context as a struct.
//------------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#define STRUCT_TAG
#else
#define STRUCT_TAG struct
#endif
///  Graph node type for C interface.
    typedef void * graph_node;

//------------------------------------------------------------------------------
///  @brief Graph base type.
//------------------------------------------------------------------------------
    enum graph_type {
        FLOAT,
        DOUBLE,
        COMPLEX_FLOAT,
        COMPLEX_DOUBLE
    };

//------------------------------------------------------------------------------
///  @brief graph_c_context type.
//------------------------------------------------------------------------------
    struct graph_c_context {
///  Type of the context.
        enum graph_type type;
///  Uses safe math.
        bool safe_math;
    };

//------------------------------------------------------------------------------
///  @brief Construct a C context.
///
///  @param[in] type          Base type.
///  @param[in] use_safe_math Control is safe math is used.
///  @returns A contructed C context.
//------------------------------------------------------------------------------
    STRUCT_TAG graph_c_context *graph_construct_context(const enum graph_type type,
                                                        const bool use_safe_math);

//------------------------------------------------------------------------------
///  @brief Destroy C context.
///
///  @param[in,out] c The c context to delete.
//------------------------------------------------------------------------------
    void graph_destroy_context(STRUCT_TAG graph_c_context *c);

//------------------------------------------------------------------------------
///  @brief Create variable node.
///
///  @param[in] c      The graph C context.
///  @param[in] size   Size of the data buffer.
///  @param[in] symbol Symbol of the variable used in equations.
///  @returns The created variable.
//------------------------------------------------------------------------------
    graph_node graph_variable(STRUCT_TAG graph_c_context *c,
                              const size_t size,
                              const char *symbol);

//------------------------------------------------------------------------------
///  @brief Create constant node.
///
///  @param[in] c     The graph C context.
///  @param[in] value The value to create the constant.
///  @returns The created constant.
//------------------------------------------------------------------------------
    graph_node graph_constant(STRUCT_TAG graph_c_context *c,
                              const double value);

//------------------------------------------------------------------------------
///  @brief Set a variable value.
///
///  @param[in] c      The graph C context.
///  @param[in] var    The variable to set.
///  @param[in] source The source pointer.
//------------------------------------------------------------------------------
    void graph_set_variable(STRUCT_TAG graph_c_context *c,
                            graph_node var,
                            const void *source);

//------------------------------------------------------------------------------
///  @brief Create complex constant node.
///
///  @param[in] c          The graph C context.
///  @param[in] real_value The real component.
///  @param[in] img_value  The imaginary component.
///  @returns The complex constant.
//------------------------------------------------------------------------------
    graph_node graph_constant_c(STRUCT_TAG graph_c_context *c,
                                const double real_value,
                                const double img_value);

//------------------------------------------------------------------------------
///  @brief Create a pseudo variable.
///
///  @param[in] c   The graph C context.
///  @param[in] var The variable to set.
///  @returns The pseudo variable.
//------------------------------------------------------------------------------
    graph_node graph_pseudo_variable(STRUCT_TAG graph_c_context *c,
                                     graph_node var);

//------------------------------------------------------------------------------
///  @brief Remove pseudo.
///
///  @param[in] c   The graph C context.
///  @param[in] var The graph to remove pseudo variables.
///  @returns The graph with pseudo variables removed.
//------------------------------------------------------------------------------
    graph_node graph_remove_pseudo(STRUCT_TAG graph_c_context *c,
                                   graph_node var);

//------------------------------------------------------------------------------
///  @brief Create Addition node.
///
///  @param[in] c     The graph C context.
///  @param[in] left  The left opperand.
///  @param[in] right The right opperand.
///  @returns left + right
//------------------------------------------------------------------------------
    graph_node graph_add(STRUCT_TAG graph_c_context *c,
                         graph_node left,
                         graph_node right);

//------------------------------------------------------------------------------
///  @brief Create Substract node.
///
///  @param[in] c     The graph C context.
///  @param[in] left  The left opperand.
///  @param[in] right The right opperand.
///  @returns left - right
//------------------------------------------------------------------------------
    graph_node graph_sub(STRUCT_TAG graph_c_context *c,
                         graph_node left,
                         graph_node right);

//------------------------------------------------------------------------------
///  @brief Create Multiply node.
///
///  @param[in] c     The graph C context.
///  @param[in] left  The left opperand.
///  @param[in] right The right opperand.
///  @returns left*right
//------------------------------------------------------------------------------
    graph_node graph_mul(STRUCT_TAG graph_c_context *c,
                         graph_node left,
                         graph_node right);

//------------------------------------------------------------------------------
///  @brief Create Divide node.
///
///  @param[in] c     The graph C context.
///  @param[in] left  The left opperand.
///  @param[in] right The right opperand.
///  @returns left/right
//------------------------------------------------------------------------------
    graph_node graph_div(STRUCT_TAG graph_c_context *c,
                         graph_node left,
                         graph_node right);

//------------------------------------------------------------------------------
///  @brief Create Sqrt node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg The function argument.
///  @returns sqrt(arg)
//------------------------------------------------------------------------------
    graph_node graph_sqrt(STRUCT_TAG graph_c_context *c,
                          graph_node arg);

//------------------------------------------------------------------------------
///  @brief Create exp node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg The function argument.
///  @returns exp(arg)
//------------------------------------------------------------------------------
    graph_node graph_exp(STRUCT_TAG graph_c_context *c,
                         graph_node arg);

//------------------------------------------------------------------------------
///  @brief Create log node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg The function argument.
///  @returns log(arg)
//------------------------------------------------------------------------------
    graph_node graph_log(STRUCT_TAG graph_c_context *c,
                         graph_node arg);

//------------------------------------------------------------------------------
///  @brief Create Pow node.
///
///  @param[in] c     The graph C context.
///  @param[in] left  The left opperand.
///  @param[in] right The right opperand.
///  @returns pow(left, right)
//------------------------------------------------------------------------------
    graph_node graph_pow(STRUCT_TAG graph_c_context *c,
                         graph_node left,
                         graph_node right);

//------------------------------------------------------------------------------
///  @brief Create imaginary error function node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg The function argument.
///  @returns erfi(arg)
//------------------------------------------------------------------------------
    graph_node graph_erfi(STRUCT_TAG graph_c_context *c,
                          graph_node arg);

//------------------------------------------------------------------------------
///  @brief Create sine node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg The function argument.
///  @returns sin(arg)
//------------------------------------------------------------------------------
    graph_node graph_sin(STRUCT_TAG graph_c_context *c,
                         graph_node arg);

//------------------------------------------------------------------------------
///  @brief Create cosine node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg The function argument.
///  @returns sin(arg)
//------------------------------------------------------------------------------
    graph_node graph_cos(STRUCT_TAG graph_c_context *c,
                         graph_node arg);

//------------------------------------------------------------------------------
///  @brief Create arctangent node.
///
///  @param[in] c     The graph C context.
///  @param[in] left  The left opperand.
///  @param[in] right The right opperand.
///  @returns atan(left, right)
//------------------------------------------------------------------------------
    graph_node graph_atan(STRUCT_TAG graph_c_context *c,
                          graph_node left,
                          graph_node right);

//------------------------------------------------------------------------------
///  @brief Construct a random state node.
///
///  @param[in] c    The graph C context.
///  @param[in] seed Intial random seed.
///  @returns A random state node.
//------------------------------------------------------------------------------
    graph_node graph_random_state(STRUCT_TAG graph_c_context *c,
                                  const uint32_t seed);

//------------------------------------------------------------------------------
///  @brief Create random node.
///
///  @param[in] c     The graph C context.
///  @param[in] state A random state node.
///  @returns random(state)
//------------------------------------------------------------------------------
    graph_node graph_random(STRUCT_TAG graph_c_context *c,
                            graph_node state);

//------------------------------------------------------------------------------
///  @brief Create 1D piecewise node.
///
///  @param[in] c           The graph C context.
///  @param[in] arg         The function argument.
///  @param[in] scale       Scale factor argument.
///  @param[in] offset      Offset factor argument.
///  @param[in] source      Source buffer to fill elements.
///  @param[in] source_size Number of elements in the source buffer.
///  @returns A 1D piecewise node.
//------------------------------------------------------------------------------
    graph_node graph_piecewise_1D(STRUCT_TAG graph_c_context *c,
                                  graph_node arg,
                                  const double scale,
                                  const double offset,
                                  const void *source,
                                  const size_t source_size);

//------------------------------------------------------------------------------
///  @brief Create 2D piecewise node.
///
///  @param[in] c           The graph C context.
///  @param[in] num_cols    Number of columns.
///  @param[in] x_arg       The function x argument.
///  @param[in] x_scale     Scale factor x argument.
///  @param[in] x_offset    Offset factor x argument.
///  @param[in] y_arg       The function y argument.
///  @param[in] y_scale     Scale factor y argument.
///  @param[in] y_offset    Offset factor y argument.
///  @param[in] source      Source buffer to fill elements.
///  @param[in] source_size Number of elements in the source buffer.
///  @returns A 2D piecewise node.
//------------------------------------------------------------------------------
    graph_node graph_piecewise_2D(STRUCT_TAG graph_c_context *c,
                                  const size_t num_cols,
                                  graph_node x_arg,
                                  const double x_scale,
                                  const double x_offset,
                                  graph_node y_arg,
                                  const double y_scale,
                                  const double y_offset,
                                  const void *source,
                                  const size_t source_size);

//------------------------------------------------------------------------------
///  @brief Create 2D piecewise node with complex arguments.
///
///  @param[in] c The graph C context.
///  @returns The number of concurrent devices.
//------------------------------------------------------------------------------
    size_t graph_get_max_concurrency(STRUCT_TAG graph_c_context *c);

//------------------------------------------------------------------------------
///  @brief Choose the device number.
///
///  @param[in] c   The graph C context.
///  @param[in] num The device number.
//------------------------------------------------------------------------------
    void graph_set_device_number(STRUCT_TAG graph_c_context *c,
                                 const size_t num);

//------------------------------------------------------------------------------
///  @brief Add pre workflow item.
///
///  @param[in] c            The graph C context.
///  @param[in] inputs       Array of input nodes.
///  @param[in] num_inputs   Number of inputs.
///  @param[in] outputs      Array of output nodes.
///  @param[in] num_outputs  Number of outputs.
///  @param[in] map_inputs   Array of map input nodes.
///  @param[in] map_outputs  Array of map output nodes.
///  @param[in] num_maps     Number of maps.
///  @param[in] random_state Optional random state, can be NULL if not used.
///  @param[in] name         Name for the kernel.
///  @param[in] size         Number of elements to operate on.
//------------------------------------------------------------------------------
    void graph_add_pre_item(STRUCT_TAG graph_c_context *c,
                            graph_node *inputs, size_t num_inputs,
                            graph_node *outputs, size_t num_outputs,
                            graph_node *map_inputs,
                            graph_node *map_outputs, size_t num_maps,
                            graph_node random_state,
                            const char *name,
                            const size_t size);

//------------------------------------------------------------------------------
///  @brief Add workflow item.
///
///  @param[in] c            The graph C context.
///  @param[in] inputs       Array of input nodes.
///  @param[in] num_inputs   Number of inputs.
///  @param[in] outputs      Array of output nodes.
///  @param[in] num_outputs  Number of outputs.
///  @param[in] map_inputs   Array of map input nodes.
///  @param[in] map_outputs  Array of map output nodes.
///  @param[in] num_maps     Number of maps.
///  @param[in] random_state Optional random state, can be NULL if not used.
///  @param[in] name         Name for the kernel.
///  @param[in] size         Number of elements to operate on.
//------------------------------------------------------------------------------
    void graph_add_item(STRUCT_TAG graph_c_context *c,
                        graph_node *inputs, size_t num_inputs,
                        graph_node *outputs, size_t num_outputs,
                        graph_node *map_inputs,
                        graph_node *map_outputs, size_t num_maps,
                        graph_node random_state,
                        const char *name,
                        const size_t size);

//------------------------------------------------------------------------------
///  @brief Add a converge item.
///
///  @param[in] c            The graph C context.
///  @param[in] inputs       Array of input nodes.
///  @param[in] num_inputs   Number of inputs.
///  @param[in] outputs      Array of output nodes.
///  @param[in] num_outputs  Number of outputs.
///  @param[in] map_inputs   Array of map input nodes.
///  @param[in] map_outputs  Array of map output nodes.
///  @param[in] num_maps     Number of maps.
///  @param[in] random_state Optional random state, can be NULL if not used.
///  @param[in] name         Name for the kernel.
///  @param[in] size         Number of elements to operate on.
///  @param[in] tol          Tolarance to converge the function to.
///  @param[in] max_iter     Maximum number of iterations before giving up.
//------------------------------------------------------------------------------
    void graph_add_converge_item(STRUCT_TAG graph_c_context *c,
                                 graph_node *inputs, size_t num_inputs,
                                 graph_node *outputs, size_t num_outputs,
                                 graph_node *map_inputs,
                                 graph_node *map_outputs, size_t num_maps,
                                 graph_node random_state,
                                 const char *name,
                                 const size_t size,
                                 const double tol,
                                 const size_t max_iter);

//------------------------------------------------------------------------------
///  @brief Compile the work items.
///
///  @param[in] c The graph C context.
//------------------------------------------------------------------------------
    void graph_compile(STRUCT_TAG graph_c_context *c);

//------------------------------------------------------------------------------
///  @brief Run pre work items.
///
///  @param[in] c The graph C context.
//------------------------------------------------------------------------------
    void graph_pre_run(STRUCT_TAG graph_c_context *c);

//------------------------------------------------------------------------------
///  @brief Run work items.
///
///  @param[in] c The graph C context.
//------------------------------------------------------------------------------
    void graph_run(STRUCT_TAG graph_c_context *c);

//------------------------------------------------------------------------------
///  @brief Wait for work items to complete.
///
///  @param[in] c The graph C context.
//------------------------------------------------------------------------------
    void graph_wait(STRUCT_TAG graph_c_context *c);

//------------------------------------------------------------------------------
///  @brief Copy data to a device buffer.
///
///  @param[in] c      The graph C context.
///  @param[in] node   Node to copy to.
///  @param[in] source Source to copy from.
//------------------------------------------------------------------------------
    void graph_copy_to_device(STRUCT_TAG graph_c_context *c,
                              graph_node node,
                              void *source);

//------------------------------------------------------------------------------
///  @brief Copy data to a host buffer.
///
///  @param[in] c           The graph C context.
///  @param[in] node        Node to copy from.
///  @param[in] destination Host side buffer to copy to.
//------------------------------------------------------------------------------
    void graph_copy_to_host(STRUCT_TAG graph_c_context *c,
                            graph_node node,
                            void *destination);

//------------------------------------------------------------------------------
///  @brief Print a value from nodes.
///
///  @param[in] c         The graph C context.
///  @param[in] index     Particle index to print.
///  @param[in] nodes     Nodes to print.
///  @param[in] num_nodes Number of nodes.
//------------------------------------------------------------------------------
    void graph_print(STRUCT_TAG graph_c_context *c,
                     const size_t index,
                     graph_node *nodes,
                     const size_t num_nodes);

//------------------------------------------------------------------------------
///  @brief Take derivative ∂f∂x.
///
///  @param[in] c     The graph C context.
///  @param[in] fnode The function expression to take the derivative of.
///  @param[in] xnode The expression to take the derivative with respect to.
//------------------------------------------------------------------------------
    graph_node graph_df(STRUCT_TAG graph_c_context *c,
                        graph_node fnode,
                        graph_node xnode);
#ifdef __cplusplus
}
#endif

#endif /* graph_c_binding_h */
