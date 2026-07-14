//------------------------------------------------------------------------------
///  @file workflow.hpp
///  @brief Classes to manage workflows,
///
///  Defines work items and a manager to run them.
//------------------------------------------------------------------------------

#ifndef workflow_h
#define workflow_h

#include "jit.hpp"

///  Name space for workflows.
namespace workflow {
///  Items order
    enum order {
///  Pre items
        pre_run_item,
///  Items
        run_item,
///  Post items
        post_run_item
    };

//------------------------------------------------------------------------------
///  @brief Interface class representing items.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class item {
    public:
//------------------------------------------------------------------------------
///  @brief Set the kernel function.
///
///  @param[in,out] context Jit context.
//------------------------------------------------------------------------------
        virtual void create_kernel_call(jit::context<T, SAFE_MATH> &context) = 0;

//------------------------------------------------------------------------------
///  @brief Run the work item.
//------------------------------------------------------------------------------
        virtual void run() = 0;
    };

//------------------------------------------------------------------------------
///  @brief Callback item.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class callback_item : public item<T, SAFE_MATH> {
    protected:
///  Callback function.
        std::function<void(void)> callback;
///  Kernel function.
        std::function<void(void)> kernel;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a workflow item.
///
///  @param[in] callback Lambda function to run.
//------------------------------------------------------------------------------
        callback_item(std::function<void(void)> callback) :
        callback(callback) {}

//------------------------------------------------------------------------------
///  @brief Set the kernel function.
///
///  @param[in,out] context Jit context.
//------------------------------------------------------------------------------
        virtual void create_kernel_call(jit::context<T, SAFE_MATH> &context) {
            kernel = context.run_function(callback);
        }

//------------------------------------------------------------------------------
///  @brief Run the work item.
//------------------------------------------------------------------------------
        virtual void run() {
            kernel();
        }
    };

//------------------------------------------------------------------------------
///  @brief Clear buffer item.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class zero_item : public item<T, SAFE_MATH> {
    protected:
///  Kernel function.
        std::function<void(void)> kernel;
///  Input nodes.
        graph::input_nodes<T, SAFE_MATH> inputs;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a workflow item.
///
///  @param[in] in Input variables.
//------------------------------------------------------------------------------
        zero_item(graph::input_nodes<T, SAFE_MATH> in) :
        inputs(in) {}

//------------------------------------------------------------------------------
///  @brief Set the kernel function.
///
///  @param[in,out] context Jit context.
//------------------------------------------------------------------------------
        virtual void create_kernel_call(jit::context<T, SAFE_MATH> &context) {
            kernel = context.create_zero_call(inputs);
        }

//------------------------------------------------------------------------------
///  @brief Run the work item.
//------------------------------------------------------------------------------
        virtual void run() {
            kernel();
        }
    };

//------------------------------------------------------------------------------
///  @brief Copy one buffer item to another.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class copy_item : public item<T, SAFE_MATH> {
    protected:
///  Kernel function.
        std::function<void(void)> kernel;
///  Input nodes.
        graph::copy_nodes<T, SAFE_MATH> maps;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a workflow item.
///
///  @param[in] maps Input variables to copy.
//------------------------------------------------------------------------------
        copy_item(graph::copy_nodes<T, SAFE_MATH> maps) :
        maps(maps) {}

//------------------------------------------------------------------------------
///  @brief Set the kernel function.
///
///  @param[in,out] context Jit context.
//------------------------------------------------------------------------------
        virtual void create_kernel_call(jit::context<T, SAFE_MATH> &context) {
            kernel = context.create_copy_call(maps);
        }

//------------------------------------------------------------------------------
///  @brief Run the work item.
//------------------------------------------------------------------------------
        virtual void run() {
            kernel();
        }
    };

//------------------------------------------------------------------------------
///  @brief Class representing a work item.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class work_item : public item<T, SAFE_MATH> {
    protected:
///  Kernel function.
        std::function<void(void)> kernel;
///  Name of the GPU kernel.
        const std::string kernel_name;
///  Size of the GPU kernel.
        const size_t kernel_size;
///  Input nodes.
        graph::input_nodes<T, SAFE_MATH> inputs;
///  Output nodes.
        graph::output_nodes<T, SAFE_MATH> outputs;
///  Random state node.
        graph::shared_random_state<T, SAFE_MATH> state;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a workflow item.
///
///  @param[in]     in      Input variables.
///  @param[in]     out     Output nodes.
///  @param[in]     maps    Setter maps.
///  @param[in]     state   Random state node.
///  @param[in]     name    Name of the work item.
///  @param[in]     size    Size of the work item.
///  @param[in,out] context Jit context.
//------------------------------------------------------------------------------
        work_item(graph::input_nodes<T, SAFE_MATH> in,
                  graph::output_nodes<T, SAFE_MATH> out,
                  graph::map_nodes<T, SAFE_MATH> maps,
                  graph::shared_random_state<T, SAFE_MATH> state,
                  const std::string name, const size_t size,
                  jit::context<T, SAFE_MATH> &context) :
        inputs(in), outputs(out), state(state),
        kernel_name(name), kernel_size(size) {
            context.add_kernel(name, in, out, maps, state, size);
        }

//------------------------------------------------------------------------------
///  @brief Set the kernel function.
///
///  @param[in,out] context Jit context.
//------------------------------------------------------------------------------
        virtual void create_kernel_call(jit::context<T, SAFE_MATH> &context) {
            kernel = context.create_kernel_call(kernel_name, inputs, outputs,
                                                state, kernel_size);
        }

//------------------------------------------------------------------------------
///  @brief Run the work item.
//------------------------------------------------------------------------------
        virtual void run() {
            kernel();
        }
    };

//------------------------------------------------------------------------------
///  @brief Run a work item in a fixed iteration loop.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class loop_item final : public item<T, SAFE_MATH> {
    protected:
///  Kernel function.
        std::function<void(void)> kernel;
///  Name of the GPU kernel.
        const std::string kernel_name;
///  Size of the GPU kernel.
        const size_t kernel_size;
///  Input nodes.
        graph::input_nodes<T, SAFE_MATH> inputs;
///  Output nodes.
        graph::output_nodes<T, SAFE_MATH> outputs;
///  Random state node.
        graph::shared_random_state<T, SAFE_MATH> state;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a workflow item.
///
///  @param[in]     in      Input variables.
///  @param[in]     out     Output nodes.
///  @param[in]     maps    Setter maps.
///  @param[in]     state   Random state node.
///  @param[in]     name    Name of the work item.
///  @param[in]     size    Size of the work item.
///  @param[in,out] context Jit context.
///  @param[in]     iterations Number of iterations to run the loop.
//------------------------------------------------------------------------------
        loop_item(graph::input_nodes<T, SAFE_MATH> in,
                  graph::output_nodes<T, SAFE_MATH> out,
                  graph::map_nodes<T, SAFE_MATH> maps,
                  graph::shared_random_state<T, SAFE_MATH> state,
                  const std::string name, const size_t size,
                  jit::context<T, SAFE_MATH> &context,
                  const size_t iterations) :
        inputs(in), outputs(out), state(state),
        kernel_name(name), kernel_size(size) {
            context.add_kernel(name, in, out, maps, state, size, iterations);
        }

//------------------------------------------------------------------------------
///  @brief Set the kernel function.
///
///  @param[in,out] context Jit context.
//------------------------------------------------------------------------------
        virtual void create_kernel_call(jit::context<T, SAFE_MATH> &context) {
            kernel = context.create_kernel_call(kernel_name, inputs, outputs,
                                                state, kernel_size);
        }

//------------------------------------------------------------------------------
///  @brief Run the workitem.
//------------------------------------------------------------------------------
        virtual void run() {
            kernel();
        }
    };

//------------------------------------------------------------------------------
///  @brief Class representing a convergence work item.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class converge_item final : public work_item<T, SAFE_MATH> {
    private:
///  Kernel function.
        std::function<T(void)> max_kernel;
///  Convergence tolerance.
        const T tolerance;
///  Total number of iterations.
        const size_t max_iterations;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a workflow item.
///
///  @param[in]     inputs   Input variables.
///  @param[in]     outputs  Output nodes.
///  @param[in]     maps     Setter maps.
///  @param[in]     state    Random state node.
///  @param[in]     name     Name of the work item.
///  @param[in]     size     Size of the work item.
///  @param[in,out] context  Jit context.
///  @param[in]     tol      Tolerance to solve the dispersion function to.
///  @param[in]     max_iter Maximum number of iterations before giving up.
//------------------------------------------------------------------------------
        converge_item(graph::input_nodes<T, SAFE_MATH> inputs,
                      graph::output_nodes<T, SAFE_MATH> outputs,
                      graph::map_nodes<T, SAFE_MATH> maps,
                      graph::shared_random_state<T, SAFE_MATH> state,
                      const std::string name, const size_t size,
                      jit::context<T, SAFE_MATH> &context,
                      const T tol=1.0E-30,
                      const size_t max_iter=1000) :
        work_item<T, SAFE_MATH> (inputs, outputs, maps, state, name, size, context),
        tolerance(tol), max_iterations(max_iter) {
            context.add_max_reduction(size);
        }

//------------------------------------------------------------------------------
///  @brief Set the kernel function.
///
///  @param[in,out] context Jit context.
//------------------------------------------------------------------------------
        virtual void create_kernel_call(jit::context<T, SAFE_MATH> &context) {
            work_item<T, SAFE_MATH>::create_kernel_call(context);
            max_kernel = context.create_max_call(this->outputs.back(),
                                                 this->kernel);
        }

//------------------------------------------------------------------------------
///  @brief Run the workitem.
//------------------------------------------------------------------------------
        virtual void run() {
            size_t iterations = 0;
            T max_residual = max_kernel();
            T last_max = std::numeric_limits<T>::max();
            T off_last_max = std::numeric_limits<T>::max();
            while (std::abs(max_residual) > std::abs(tolerance)                &&
                   std::abs(last_max - max_residual) > std::abs(tolerance)     &&
                   std::abs(off_last_max - max_residual) > std::abs(tolerance) &&
                   iterations++ < max_iterations) {
                last_max = max_residual;
                if (!(iterations%2)) {
                    off_last_max = max_residual;
                }
                max_residual = max_kernel();
            }

//  In release mode asserts are disables so write error to standard err. Need to
//  flip the comparison operator because we want to assert to trip if false.
            assert(iterations < max_iterations &&
                   "Workitem failed to converge.");
            if (iterations > max_iterations) {
                std::cerr << "Workitem failed to converge with in given iterations."
                          << std::endl;
                std::cerr << "Minimum residual reached: " << max_residual
                          << std::endl;
            }
        }
    };

//------------------------------------------------------------------------------
///  @brief Class representing a workflow manager.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class manager {
    private:
///  JIT context.
        jit::context<T, SAFE_MATH> context;
///  List of pre work items.
        std::vector<std::unique_ptr<item<T, SAFE_MATH>>> preitems;
///  List of work items.
        std::vector<std::unique_ptr<item<T, SAFE_MATH>>> items;
///  List of pre work items.
        std::vector<std::unique_ptr<item<T, SAFE_MATH>>> postitems;
///  Use reduction.
        bool add_reduction;

    public:
//------------------------------------------------------------------------------
///  @brief Workflow manager constructor.
///
///  For GPU devices, this select the device number to run on. For CPU devices
///  this parameter is ignored.
///
///  @note It is possible to create multiple workflow managers for the same
///        GPU device and may have performance benefits todo so.
///
///  @param[in] index Device index.
//------------------------------------------------------------------------------
        manager(const size_t index) : context(index), add_reduction(false) {}

//------------------------------------------------------------------------------
///  @brief Add a pre callback function.
///
///  @tparam O The @ref workflow::order
///
///  @param[in] callback Lambda function to run.
//------------------------------------------------------------------------------
        template<order O=run_item>
        void add_callback_item(std::function<void(void)> callback) {
            if constexpr (O == pre_run_item) {
                preitems.push_back(std::make_unique<callback_item<T, SAFE_MATH>> (callback));
            } else if constexpr (O == run_item) {
                items.push_back(std::make_unique<callback_item<T, SAFE_MATH>> (callback));
            } else {
                postitems.push_back(std::make_unique<callback_item<T, SAFE_MATH>> (callback));
            }
        }

//------------------------------------------------------------------------------
///  @brief Add a workflow item.
///
///  @tparam O The @ref workflow::order
///
///  @param[in] in    Input variables.
///  @param[in] out   Output nodes.
///  @param[in] maps  Setter maps.
///  @param[in] state Random state node.
///  @param[in] name  Name of the work item.
///  @param[in] size  Size of the work item.
//------------------------------------------------------------------------------
        template<order O=run_item>
        void add_item(graph::input_nodes<T, SAFE_MATH> in,
                      graph::output_nodes<T, SAFE_MATH> out,
                      graph::map_nodes<T, SAFE_MATH> maps,
                      graph::shared_random_state<T, SAFE_MATH> state,
                      const std::string name, const size_t size) {
            if constexpr (O == pre_run_item) {
                preitems.push_back(std::make_unique<work_item<T, SAFE_MATH>> (in, out,
                                                                              maps, state,
                                                                              name, size,
                                                                              context));
            } else if constexpr (O == run_item) {
                items.push_back(std::make_unique<work_item<T, SAFE_MATH>> (in, out,
                                                                           maps, state,
                                                                           name, size,
                                                                           context));
            } else {
                postitems.push_back(std::make_unique<work_item<T, SAFE_MATH>> (in, out,
                                                                               maps, state,
                                                                               name, size,
                                                                               context));
            }
        }

//------------------------------------------------------------------------------
///  @brief Add a zero item.
///
///  @tparam O The @ref workflow::order
///
///  @param[in] in    Input variables.
//------------------------------------------------------------------------------
        template<order O=run_item>
        void add_zero_item(graph::input_nodes<T, SAFE_MATH> in) {
            if constexpr (O == pre_run_item) {
                preitems.push_back(std::make_unique<zero_item<T, SAFE_MATH>> (in));
            } else if constexpr (O == run_item) {
                items.push_back(std::make_unique<zero_item<T, SAFE_MATH>> (in));
            } else {
                postitems.push_back(std::make_unique<zero_item<T, SAFE_MATH>> (in));
            }
        }

//------------------------------------------------------------------------------
///  @brief Add a copy item.
///
///  @tparam O The @ref workflow::order
///
///  @param[in] maps Copy maps.
//------------------------------------------------------------------------------
        template<order O=run_item>
        void add_copy_item(graph::copy_nodes<T, SAFE_MATH> maps) {
            if constexpr (O == pre_run_item) {
                preitems.push_back(std::make_unique<copy_item<T, SAFE_MATH>> (maps));
            } else if constexpr (O == run_item) {
                items.push_back(std::make_unique<copy_item<T, SAFE_MATH>> (maps));
            } else {
                postitems.push_back(std::make_unique<copy_item<T, SAFE_MATH>> (maps));
            }
        }

//------------------------------------------------------------------------------
///  @brief Add a loop item.
///
///  @tparam O The @ref workflow::order
///
///  @param[in] in         Input variables.
///  @param[in] out        Output nodes.
///  @param[in] maps       Setter maps.
///  @param[in] state      Random state node.
///  @param[in] name       Name of the work item.
///  @param[in] size       Size of the work item.
///  @param[in] iterations Number of iterations.
//------------------------------------------------------------------------------
        template<order O=run_item>
        void add_loop_item(graph::input_nodes<T, SAFE_MATH> in,
                           graph::output_nodes<T, SAFE_MATH> out,
                           graph::map_nodes<T, SAFE_MATH> maps,
                           graph::shared_random_state<T, SAFE_MATH> state,
                           const std::string name, const size_t size,
                           const size_t iterations) {
            if constexpr (O == pre_run_item) {
                preitems.push_back(std::make_unique<loop_item<T, SAFE_MATH>> (in, out,
                                                                              maps, state,
                                                                              name, size,
                                                                              context,
                                                                              iterations));
            } else if constexpr (O == run_item) {
                items.push_back(std::make_unique<loop_item<T, SAFE_MATH>> (in, out,
                                                                           maps, state,
                                                                           name, size,
                                                                           context,
                                                                           iterations));
            } else {
                postitems.push_back(std::make_unique<loop_item<T, SAFE_MATH>> (in, out,
                                                                               maps, state,
                                                                               name, size,
                                                                               context,
                                                                               iterations));
            }
        }

//------------------------------------------------------------------------------
///  @brief Add a converge item.
///
///  @tparam O The @ref workflow::order
///
///  @param[in] in       Input variables.
///  @param[in] out      Output nodes.
///  @param[in] maps     Setter maps.
///  @param[in] state    Random state node.
///  @param[in] name     Name of the work item.
///  @param[in] size     Size of the work item.
///  @param[in] tol      Tolerance to converge the function to.
///  @param[in] max_iter Maximum number of iterations before giving up.
//------------------------------------------------------------------------------
        template<order O=run_item>
        void add_converge_item(graph::input_nodes<T, SAFE_MATH> in,
                               graph::output_nodes<T, SAFE_MATH> out,
                               graph::map_nodes<T, SAFE_MATH> maps,
                               graph::shared_random_state<T, SAFE_MATH> state,
                               const std::string name, const size_t size,
                               const T tol=1.0E-30,
                               const size_t max_iter=1000) {
            add_reduction = true;
            if constexpr (O == pre_run_item) {
                items.push_back(std::make_unique<converge_item<T, SAFE_MATH>> (in, out,
                                                                               maps, state,
                                                                               name, size,
                                                                               context, tol,
                                                                               max_iter));
            } else if constexpr (O == run_item) {
                items.push_back(std::make_unique<converge_item<T, SAFE_MATH>> (in, out,
                                                                               maps, state,
                                                                               name, size,
                                                                               context, tol,
                                                                               max_iter));
            } else {
                postitems.push_back(std::make_unique<converge_item<T, SAFE_MATH>> (in, out,
                                                                                   maps, state,
                                                                                   name, size,
                                                                                   context, tol,
                                                                                   max_iter));
            }
        }

//------------------------------------------------------------------------------
///  @brief  Compile the workflow items.
//------------------------------------------------------------------------------
        void compile() {
            context.compile(add_reduction);

            for (auto &item : preitems) {
                item->create_kernel_call(context);
            }
            for (auto &item : items) {
                item->create_kernel_call(context);
            }
            for (auto &item : postitems) {
                item->create_kernel_call(context);
            }
        }

//------------------------------------------------------------------------------
///  @brief Run work items.
///
///  @tparam O The @ref workflow::order
//------------------------------------------------------------------------------
        template<order O=run_item>
        void run() {
            if constexpr (O == pre_run_item) {
                for (auto &item : preitems) {
                    item->run();
                }
            } else if constexpr (O == run_item) {
                for (auto &item : items) {
                    item->run();
                }
            } else {
                for (auto &item : postitems) {
                    item->run();
                }
            }
        }

//------------------------------------------------------------------------------
///  @brief Wait for GPU queue to finish.
//------------------------------------------------------------------------------
        void wait() {
            context.wait();
        }

//------------------------------------------------------------------------------
///  @brief Copy buffer contents to the device.
///
///  @param[in] node        Node to copy buffer to.
///  @param[in] destination Host side buffer to copy from.
//------------------------------------------------------------------------------
        void copy_to_device(graph::shared_leaf<T, SAFE_MATH> &node,
                            T *destination) {
            context.copy_to_device(node, destination);
        }

//------------------------------------------------------------------------------
///  @brief Copy contexts of buffer to host.
///
///  @param[in]     node        Node to copy buffer from.
///  @param[in,out] destination Host side buffer to copy to.
//------------------------------------------------------------------------------
        void copy_to_host(graph::shared_leaf<T, SAFE_MATH> &node,
                          T *destination) {
            context.copy_to_host(node, destination);
        }

//------------------------------------------------------------------------------
///  @brief Print results.
///
///  @param[in] index Particle index to print.
///  @param[in] nodes Nodes to output.
//------------------------------------------------------------------------------
        void print(const size_t index,
                   const graph::output_nodes<T, SAFE_MATH> &nodes) {
            context.print(index, nodes);
        }

//------------------------------------------------------------------------------
///  @brief Check the value.
///
///  @param[in] index Particle index to check value for.
///  @param[in] node  Node to check the value for.
///  @returns The value at the index.
//------------------------------------------------------------------------------
        T check_value(const size_t index,
                      const graph::shared_leaf<T, SAFE_MATH> &node) {
            return context.check_value(index, node);
        }

//------------------------------------------------------------------------------
///  @brief Get the jit context.
///
///  @returns The jit context.
//------------------------------------------------------------------------------
        jit::context<T, SAFE_MATH> &get_context() {
            return context;
        }
    };
}

#endif /* workflow_h */
