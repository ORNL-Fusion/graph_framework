//------------------------------------------------------------------------------
///  @file workflow.hpp
///  @brief Classes to manage workflows,
///
///  Defines work items and a manager to run them.
//------------------------------------------------------------------------------

#ifndef workflow_h
#define workflow_h

#include "jit.hpp"

namespace workflow {
//------------------------------------------------------------------------------
///  @brief Class representing a workitem.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class work_item {
    protected:
///  Name of the GPU kernel.
        const std::string kernel_name;
///  Input nodes.
        graph::input_nodes<T, SAFE_MATH> inputs;
///  Output nodes.
        graph::output_nodes<T, SAFE_MATH> outputs;
///  Kernel function.
        std::function<void(void)> kernel;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a workflow item.
///
///  @param[in]     in      Input variables.
///  @param[in]     out     Output nodes.
///  @param[in]     maps    Setter maps.
///  @param[in]     name    Name of the workitem.
///  @param[in,out] context Jit context.
//------------------------------------------------------------------------------
        work_item(graph::input_nodes<T, SAFE_MATH> in,
                  graph::output_nodes<T, SAFE_MATH> out,
                  graph::map_nodes<T, SAFE_MATH> maps,
                  const std::string name,
                  jit::context<T, SAFE_MATH> &context) :
        inputs(in), outputs(out), kernel_name(name) {
            context.add_kernel(name, in, out, maps);
        }

//------------------------------------------------------------------------------
///  @brief Set the kernel function.
///
///  @param[in,out] context Jit context.
//------------------------------------------------------------------------------
        virtual void create_kernel_call(jit::context<T, SAFE_MATH> &context) {
            kernel = context.create_kernel_call(kernel_name, inputs, outputs,
                                                inputs.back()->size());
        }

//------------------------------------------------------------------------------
///  @brief Run the workitem.
//------------------------------------------------------------------------------
        virtual void run() {
            kernel();
        }
    };

//------------------------------------------------------------------------------
///  @brief Class representing a convergence workitem.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class converge_item final : public work_item<T, SAFE_MATH> {
    private:
///  Kernel function.
        std::function<T(void)> max_kernel;
///  Convergence tolarance.
        const T tolarance;
///  Total number of iterations.
        const size_t max_iterations;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a workflow item.
///
///  @param[in]     inputs   Input variables.
///  @param[in]     outputs  Output nodes.
///  @param[in]     maps     Setter maps.
///  @param[in]     name     Name of the workitem.
///  @param[in,out] context  Jit context.
///  @param[in]     tol      Tolarance to solve the dispersion function to.
///  @param[in]     max_iter Maximum number of iterations before giving up.
//------------------------------------------------------------------------------
        converge_item(graph::input_nodes<T, SAFE_MATH> inputs,
                      graph::output_nodes<T, SAFE_MATH> outputs,
                      graph::map_nodes<T, SAFE_MATH> maps,
                      const std::string name,
                      jit::context<T, SAFE_MATH> &context,
                      const T tol=1.0E-30,
                      const size_t max_iter=1000) :
        work_item<T, SAFE_MATH> (inputs, outputs, maps, name, context),
        tolarance(tol), max_iterations(max_iter) {
            context.add_max_reduction(inputs.back()->size());
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
            T max_residule = max_kernel();
            T last_max = std::numeric_limits<T>::max();
            T off_last_max = std::numeric_limits<T>::max();
            while (std::abs(max_residule) > std::abs(tolarance)                 &&
                   std::abs(last_max - max_residule) > std::abs(tolarance)      &&
                   std::abs(off_last_max - max_residule) > std::abs(tolarance)  &&
                   iterations++ < max_iterations) {
                last_max = max_residule;
                if (!(iterations%2)) {
                    off_last_max = max_residule;
                }
                max_residule = max_kernel();
            }

//  In release mode asserts are diaables so write error to standard err. Need to
//  flip the comparison operator because we want to assert to trip if false.
            assert(iterations < max_iterations &&
                   "Workitem failed to converge.");
            if (iterations > max_iterations) {
                std::cerr << "Workitem failed to converge with in given iterations."
                          << std::endl;
                std::cerr << "Minimum residule reached: " << max_residule
                          << std::endl;
            }
        }
    };

//------------------------------------------------------------------------------
///  @brief Class representing a workflow manager.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class manager {
    private:
///  JIT context.
        jit::context<T, SAFE_MATH> context;
///  List of prework items.
        std::vector<std::unique_ptr<work_item<T, SAFE_MATH>>> preitems;
///  List of work items.
        std::vector<std::unique_ptr<work_item<T, SAFE_MATH>>> items;
///  Use reduction.
        bool add_reduction;

    public:
//------------------------------------------------------------------------------
///  @brief Workflow manager constructor.
///
///  @param[in] index Concurrent index.
//------------------------------------------------------------------------------
        manager(const size_t index) : context(index), add_reduction(false) {}

//------------------------------------------------------------------------------
///  @brief Add a pre workflow item.
///
///  @param[in] in   Input variables.
///  @param[in] out  Output nodes.
///  @param[in] maps Setter maps.
///  @param[in] name Name of the workitem.
//------------------------------------------------------------------------------
        void add_preitem(graph::input_nodes<T, SAFE_MATH> in,
                         graph::output_nodes<T, SAFE_MATH> out,
                         graph::map_nodes<T, SAFE_MATH> maps,
                         const std::string name) {
            preitems.push_back(std::make_unique<work_item<T, SAFE_MATH>> (in, out,
                                                                          maps, name,
                                                                          context));
        }

//------------------------------------------------------------------------------
///  @brief Add a workflow item.
///
///  @param[in] in   Input variables.
///  @param[in] out  Output nodes.
///  @param[in] maps Setter maps.
///  @param[in] name Name of the workitem.
//------------------------------------------------------------------------------
        void add_item(graph::input_nodes<T, SAFE_MATH> in,
                      graph::output_nodes<T, SAFE_MATH> out,
                      graph::map_nodes<T, SAFE_MATH> maps,
                      const std::string name) {
            items.push_back(std::make_unique<work_item<T, SAFE_MATH>> (in, out,
                                                                       maps, name,
                                                                       context));
        }

//------------------------------------------------------------------------------
///  @brief Add a workflow item.
///
///  @param[in] in       Input variables.
///  @param[in] out      Output nodes.
///  @param[in] maps     Setter maps.
///  @param[in] name     Name of the workitem.
///  @param[in] tol      Tolarance to solve the dispersion function to.
///  @param[in] max_iter Maximum number of iterations before giving up.
//------------------------------------------------------------------------------
        void add_converge_item(graph::input_nodes<T, SAFE_MATH> in,
                               graph::output_nodes<T, SAFE_MATH> out,
                               graph::map_nodes<T, SAFE_MATH> maps,
                               const std::string name,
                               const T tol=1.0E-30,
                               const size_t max_iter=1000) {
            add_reduction = true;
            items.push_back(std::make_unique<converge_item<T, SAFE_MATH>> (in, out,
                                                                           maps, name,
                                                                           context,
                                                                           tol, max_iter));
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
        }

//------------------------------------------------------------------------------
///  @brief Run prework items.
//------------------------------------------------------------------------------
        void pre_run() {
            for (auto &item : preitems) {
                item->run();
            }
        }

//------------------------------------------------------------------------------
///  @brief Run work items.
//------------------------------------------------------------------------------
        void run() {
            for (auto &item : items) {
                item->run();
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
///  @param[in] node        Not to copy buffer to.
///  @param[in] destination Device side buffer to copy to.
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
///  @param[in] index Ray index to check value for.
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
