//------------------------------------------------------------------------------
///  @file graph_c_binding.h
///  @brief Header file for the c binding library.
//------------------------------------------------------------------------------

#ifndef graph_c_binding_h
#define graph_c_binding_h

extern "C" {
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
    graph_c_context *graph_construct_context(const enum graph_type type,
                                             const bool use_safe_math);

//------------------------------------------------------------------------------
///  @brief Destroy C context.
///
///  @param[inout] The c context to delete.
//------------------------------------------------------------------------------
    void graph_destroy_node_flt(graph_c_context *c);

//------------------------------------------------------------------------------
///  @brief Create variable node.
///
///  @param[in] c      The graph C context.
///  @param[in] size   Size of the data buffer.
///  @param[in] symbol Symbol of the variable used in equations.
///  @returns The created variable.
//------------------------------------------------------------------------------
    graph_node graph_create_variable(graph_c_context *c,
                                     const size_t size,
                                     const char *symbol);

//------------------------------------------------------------------------------
///  @brief Create constant node.
///
///  @param[in] c     The graph C context.
///  @param[in] value The value to create the constant.
///  @returns The created constant.
//------------------------------------------------------------------------------
    graph_node graph_create_constant(graph_c_context *c,
                                     const double value);

//------------------------------------------------------------------------------
///  @brief Set a variable value.
///
///  @param[in] c      The graph C context.
///  @param[in] var    The variable to set.
///  @param[in] source The source pointer.
//------------------------------------------------------------------------------
    void graph_set_variable(graph_c_context *c,
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
    graph_node graph_create_constant_c(graph_c_context *c,
                                       const double real_value,
                                       const double img_value);

//------------------------------------------------------------------------------
///  @brief Create a pseudo variable.
///
///  @param[in] c   The graph C context.
///  @param[in] var The variable to set.
///  @returns THe pseudo variable.
//------------------------------------------------------------------------------
    graph_node graph_create_pseudo_variable(graph_c_context *c,
                                            graph_node var);

//------------------------------------------------------------------------------
///  @brief Remove pseudo.
///
///  @param[in] c   The graph C context.
///  @param[in] var The variable to set.
///  @returns The graph with pseudo variables removed.
//------------------------------------------------------------------------------
    graph_node graph_remove_pseudo(graph_c_context *c,
                                   graph_node var);

//------------------------------------------------------------------------------
///  @brief Remove pseudo.
///
///  @param[in] c     The graph C context.
///  @param[in] left  The left opperand.
///  @param[in] right The right opperand.
///  @returns left + right
//------------------------------------------------------------------------------
    graph_node graph_create_add(graph_c_context *c,
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
    graph_node graph_create_sub(graph_c_context *c,
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
    graph_node graph_create_mul(graph_c_context *c,
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
    graph_node graph_create_div(graph_c_context *c,
                                graph_node left,
                                graph_node right);

//------------------------------------------------------------------------------
///  @brief Create Sqrt node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg The left opperand.
///  @returns sqrt(arg)
//------------------------------------------------------------------------------
    graph_node graph_create_sqrt(graph_c_context *c,
                                 graph_node arg);

//------------------------------------------------------------------------------
///  @brief Create exp node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg The left opperand.
///  @returns exp(arg)
//------------------------------------------------------------------------------
    graph_node graph_create_exp(graph_c_context *c,
                                graph_node arg);

//------------------------------------------------------------------------------
///  @brief Create log node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg The left opperand.
///  @returns log(arg)
//------------------------------------------------------------------------------
    graph_node graph_create_log(graph_c_context *c,
                                graph_node arg);

//------------------------------------------------------------------------------
///  @brief Create imaginary error function node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg The left opperand.
///  @returns erfi(arg)
//------------------------------------------------------------------------------
    graph_node graph_create_erfi(graph_c_context *c,
                                 graph_node arg);

//------------------------------------------------------------------------------
///  @brief Create sine node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg The left opperand.
///  @returns sin(arg)
//------------------------------------------------------------------------------
    graph_node graph_create_sin(graph_c_context *c,
                                graph_node arg);

//------------------------------------------------------------------------------
///  @brief Create cosine node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg The left opperand.
///  @returns sin(arg)
//------------------------------------------------------------------------------
    graph_node graph_create_cos(graph_c_context *c,
                                graph_node arg);

//------------------------------------------------------------------------------
///  @brief Create arctangent node.
///
///  @param[in] c     The graph C context.
///  @param[in] left  The left opperand.
///  @param[in] right The right opperand.
///  @returns atan(left, right)
//------------------------------------------------------------------------------
    graph_node graph_create_atan(graph_c_context *c,
                                 graph_node left,
                                 graph_node right);

//------------------------------------------------------------------------------
///  @brief Construct a random state node.
///
///  @param[in] c    The graph C context.
///  @param[in] size Number of random states.
///  @param[in] seed Intial random seed.
///  @returns A random state node.
//------------------------------------------------------------------------------
    graph_node graph_create_random_state(graph_c_context *c,
                                         const size_t size,
                                         const uint32_t seed);

//------------------------------------------------------------------------------
///  @brief Create random node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg The left opperand.
///  @returns random(arg)
//------------------------------------------------------------------------------
    graph_node graph_create_random(graph_c_context *c,
                                   graph_node arg);

//------------------------------------------------------------------------------
///  @brief Create 1D piecewise node.
///
///  @param[in] c           The graph C context.
///  @param[in] arg         The left opperand.
///  @param[in] scale       Scale factor argument.
///  @param[in] offset      Offset factor argument.
///  @param[in] source      Source buffer to fill elements.
///  @param[in] source_size Number of elements in the source buffer.
///  @returns A 1D piecewise node.
//------------------------------------------------------------------------------
    graph_node graph_create_piecewise_1D(graph_c_context *c,
                                         graph_node arg,
                                         const double scale,
                                         const double offset,
                                         const void *source,
                                         const size_t source_size);

//------------------------------------------------------------------------------
///  @brief Create 1D piecewise node with complex arguments.
///
///  @param[in] c           The graph C context.
///  @param[in] arg         The left opperand.
///  @param[in] scale       Scale factor argument.
///  @param[in] offset      Offset factor argument.
///  @param[in] source      Source buffer to fill elements.
///  @param[in] source_size Number of elements in the source buffer.
///  @returns A 1D piecewise node.
//------------------------------------------------------------------------------
    graph_node graph_create_piecewise_1D_c(graph_c_context *c,
                                           graph_node arg,
                                           const std::complex<double> scale,
                                           const std::complex<double> offset,
                                           const void *source,
                                           const size_t source_size) ;

//------------------------------------------------------------------------------
///  @brief Create 2D piecewise node.
///
///  @param[in] c           The graph C context.
///  @param[in] num_cols    Number of columns.
///  @param[in] x_arg       The left opperand.
///  @param[in] x_scale     Scale factor argument.
///  @param[in] x_offset    Offset factor argument.
///  @param[in] y_arg       The left opperand.
///  @param[in] y_scale     Scale factor argument.
///  @param[in] y_offset    Offset factor argument.
///  @param[in] source      Source buffer to fill elements.
///  @param[in] source_size Number of elements in the source buffer.
///  @returns A 2D piecewise node.
//------------------------------------------------------------------------------
    graph_node graph_create_piecewise_2D(graph_c_context *c,
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
///  @param[in] c           The graph C context.
///  @param[in] num_cols    Number of columns.
///  @param[in] x_arg       The left opperand.
///  @param[in] x_scale     Scale factor argument.
///  @param[in] x_offset    Offset factor argument.
///  @param[in] y_arg       The left opperand.
///  @param[in] y_scale     Scale factor argument.
///  @param[in] y_offset    Offset factor argument.
///  @param[in] source      Source buffer to fill elements.
///  @param[in] source_size Number of elements in the source buffer.
///  @returns A 2D piecewise node.
//------------------------------------------------------------------------------
    graph_node graph_create_piecewise_2D_c(graph_c_context *c,
                                           const size_t num_cols,
                                           graph_node x_arg,
                                           const std::complex<double> x_scale,
                                           const std::complex<double> x_offset,
                                           graph_node y_arg,
                                           const std::complex<double> y_scale,
                                           const std::complex<double> y_offset,
                                           const void *source,
                                           const size_t source_size);

//------------------------------------------------------------------------------
///  @brief Create 2D piecewise node with complex arguments.
///
///  @param[in] c The graph C context.
///  @returns The number of concurrent devices.
//------------------------------------------------------------------------------
    size_t graph_get_max_concurrency(graph_c_context *c);

//------------------------------------------------------------------------------
///  @brief Choose the device number.
///
///  @param[in] c   The graph C context.
///  @param[in] num The device number.
//------------------------------------------------------------------------------
    void graph_set_device_number(graph_c_context *c,
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
    void graph_add_pre_item(graph_c_context *c,
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
    void graph_add_item(graph_c_context *c,
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
    void graph_add_converge_item(graph_c_context *c,
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
///  @brief Compile the work items
///
///  @param[in] c The graph C context.
//------------------------------------------------------------------------------
    void graph_compile(graph_c_context *c);

//------------------------------------------------------------------------------
///  @brief Run pre work items.
///
///  @param[in] c The graph C context.
//------------------------------------------------------------------------------
    void graph_pre_run(graph_c_context *c);

//------------------------------------------------------------------------------
///  @brief Run work items.
///
///  @param[in] c The graph C context.
//------------------------------------------------------------------------------
    void graph_run(graph_c_context *c);

//------------------------------------------------------------------------------
///  @brief Wait for work items to complete.
///
///  @param[in] c The graph C context.
//------------------------------------------------------------------------------
    void graph_wait(graph_c_context *c);

//------------------------------------------------------------------------------
///  @brief Copy data to a device buffer.
///
///  @param[in] c      The graph C context.
///  @param[in] node   Node to copy to.
///  @param[in] source Source to copy from.
//------------------------------------------------------------------------------
    void graph_copy_to_device(graph_c_context *c,
                              graph_node node,
                              void *source);

//------------------------------------------------------------------------------
///  @brief Copy data to a host buffer.
///
///  @param[in] c           The graph C context.
///  @param[in] node        Node to copy from.
///  @param[in] destination Host side buffer to copy to.
//------------------------------------------------------------------------------
    void graph_copy_to_host(graph_c_context *c,
                            graph_node node,
                            void *destination);

//------------------------------------------------------------------------------
///  @brief Copy data to a host buffer.
///
///  @param[in] c         The graph C context.
///  @param[in] index     Particle index to print.
///  @param[in] nodes     Nodes to print.
///  @param[in] num_nodes Number of nodes.
//------------------------------------------------------------------------------
    void graph_print(graph_c_context *c,
                     const size_t index,
                     graph_node *nodes,
                     const size_t num_nodes);
}

#endif /* graph_c_binding_h */
