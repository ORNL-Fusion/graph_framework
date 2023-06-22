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
//------------------------------------------------------------------------------
    template<typename T>
    class work_item {
    protected:
///  Name of the GPU kernel.
        const std::string kernel_name;
///  Input nodes.
        graph::input_nodes<T> inputs;
///  Output nodes.
        graph::output_nodes<T> outputs;
///  Kernel function.
        std::function<void(void)> kernel;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a workflow item.
///
///  @params[in]     in      Input variables.
///  @params[in]     out     Output nodes.
///  @params[in]     maps    Setter maps.
///  @params[in]     name    Name of the workitem.
///  @params[in,out] context Jit context.
//------------------------------------------------------------------------------
        work_item(graph::input_nodes<T> in,
                  graph::output_nodes<T> out,
                  graph::map_nodes<T> maps,
                  const std::string name,
                  jit::context<T> &context) :
        inputs(in), outputs(out), kernel_name(name) {
            context.add_kernel(name, in, out, maps);
        }

//------------------------------------------------------------------------------
///  @brief Set the kernel function.
///
///  @params[in,out] context Jit context.
//------------------------------------------------------------------------------
        virtual void create_kernel_call(jit::context<T> &context) {
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
//------------------------------------------------------------------------------
    template<typename T>
    class converge_item final : public work_item<T> {
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
///  @params[in]     inputs   Input variables.
///  @params[in]     outputs  Output nodes.
///  @params[in]     maps     Setter maps.
///  @params[in]     name     Name of the workitem.
///  @params[in,out] context  Jit context.
///  @params[in]     tol      Tolarance to solve the dispersion function to.
///  @params[in]     max_iter Maximum number of iterations before giving up.
//------------------------------------------------------------------------------
        converge_item(graph::input_nodes<T> inputs,
                      graph::output_nodes<T> outputs,
                      graph::map_nodes<T> maps,
                      const std::string name,
                      jit::context<T> &context,
                      const T tol=1.0E-30,
                      const size_t max_iter=1000) :
        work_item<T> (inputs, outputs, maps, name, context),
        tolarance(tol), max_iterations(max_iter) {
            context.add_max_reduction(inputs.back()->size());
        }

//------------------------------------------------------------------------------
///  @brief Set the kernel function.
///
///  @params[in,out] context Jit context.
//------------------------------------------------------------------------------
        virtual void create_kernel_call(jit::context<T> &context) {
            work_item<T>::create_kernel_call(context);
            max_kernel = context.create_max_call(this->outputs.back(),
                                                 this->kernel);
        }
        
//------------------------------------------------------------------------------
///  @brief Run the workitem.
//------------------------------------------------------------------------------
        virtual void run() {
            size_t iterations = 0;
            T max_residule = max_kernel();
            while (std::abs(max_residule) > std::abs(tolarance) &&
                   iterations++ < max_iterations) {
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
//------------------------------------------------------------------------------
    template<typename T>
    class manager {
    private:
///  JIT context.
        jit::context<T> context;
///  List of work items.
        std::vector<std::unique_ptr<work_item<T>>> items;
///  Use reduction.
        bool add_reduction;

    public:
//------------------------------------------------------------------------------
///  @brief Workflow manager constructor.
//------------------------------------------------------------------------------
        manager() : add_reduction(false) {}

//------------------------------------------------------------------------------
///  @brief Add a workflow item.
///
///  @params[in] in   Input variables.
///  @params[in] out  Output nodes.
///  @params[in] maps Setter maps.
///  @params[in] name Name of the workitem.
//------------------------------------------------------------------------------
        void add_item(graph::input_nodes<T> in,
                      graph::output_nodes<T> out,
                      graph::map_nodes<T> maps,
                      const std::string name) {
            items.push_back(std::make_unique<work_item<T>> (in, out,
                                                            maps, name,
                                                            context));
        }

//------------------------------------------------------------------------------
///  @brief Add a workflow item.
///
///  @params[in] in       Input variables.
///  @params[in] out      Output nodes.
///  @params[in] maps     Setter maps.
///  @params[in] name     Name of the workitem.
///  @params[in] tol      Tolarance to solve the dispersion function to.
///  @params[in] max_iter Maximum number of iterations before giving up.
//------------------------------------------------------------------------------
        void add_converge_item(graph::input_nodes<T> in,
                               graph::output_nodes<T> out,
                               graph::map_nodes<T> maps,
                               const std::string name,
                               const T tol=1.0E-30,
                               const size_t max_iter=1000) {
            add_reduction = true;
            items.push_back(std::make_unique<converge_item<T>> (in, out,
                                                                maps, name,
                                                                context,
                                                                tol, max_iter));
        }

//------------------------------------------------------------------------------
///  @brief  Compile the workflow items.
//------------------------------------------------------------------------------
        void compile() {
            context.compile(add_reduction);
            
            for (auto &item : items) {
                item->create_kernel_call(context);
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
///  @params[in] node        Not to copy buffer to.
///  @params[in] destination Device side buffer to copy to.
//------------------------------------------------------------------------------
        void copy_to_device(graph::shared_leaf<T> &node,
                            T *destination) {
            context.copy_to_device(node, destination);
        }

//------------------------------------------------------------------------------
///  @brief Copy contexts of buffer to host.
///
///  @params[in]     node        Node to copy buffer from.
///  @params[in,out] destination Host side buffer to copy to.
//------------------------------------------------------------------------------
        void copy_to_host(graph::shared_leaf<T> &node,
                         T *destination) {
            context.copy_to_host(node, destination);
        }

//------------------------------------------------------------------------------
///  @brief Print results.
///
///  @params[in] index Particle index to print.
///  @params[in] nodes Nodes to output.
//------------------------------------------------------------------------------
        void print(const size_t index,
                   const graph::output_nodes<T> &nodes) {
            context.print(index, nodes);
        }

//------------------------------------------------------------------------------
///  @brief Get the jit context.
///
///  @returns The jit context.
//------------------------------------------------------------------------------
        jit::context<T> &get_context() {
            return context;
        }
    };
}

#endif /* workflow_h */
