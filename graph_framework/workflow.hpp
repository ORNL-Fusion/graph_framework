//------------------------------------------------------------------------------
///  @file workflow.hpp
///  @brief Classes to manage workflows,
///
///  Defines work items and a manager to run them.
//------------------------------------------------------------------------------

#ifndef workflow_h
#define workflow_h

#include <forward_list>

#include "node.hpp"
#include "jit.hpp"

namespace workflow {
//------------------------------------------------------------------------------
///  @brief Class representing a workitem.
//------------------------------------------------------------------------------
    template<typename T>
    class work_item {
    private:
///  Name of the GPU kernel.
        const std::string kernel_name;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a workflow item.
///
///  @params[in] inputs  Input variables.
///  @params[in] outputs Output nodes.
///  @params[in] maps    Setter maps.
///  @params[in] name    Name of the workitem.
//------------------------------------------------------------------------------
        work_item(graph::input_nodes<T> inputs,
                  graph::output_nodes<T> outputs,
                  graph::map_nodes<T> maps,
                  const std::string name) :
        kernel_name(name) {}

//------------------------------------------------------------------------------
///  @brief Run the workitem.
//------------------------------------------------------------------------------
        virtual void run() {
        }
    };

//------------------------------------------------------------------------------
///  @brief Class representing a convergence workitem.
//------------------------------------------------------------------------------
    template<typename T>
    class converge_item final : public work_item<T> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a workflow item.
///
///  @params[in] inputs  Input variables.
///  @params[in] outputs Output nodes.
///  @params[in] maps    Setter maps.
///  @params[in] name    Name of the workitem.
//------------------------------------------------------------------------------
        converge_item(graph::input_nodes<T> inputs,
                      graph::output_nodes<T> outputs,
                      graph::map_nodes<T> maps,
                      const std::string name) {}
        
//------------------------------------------------------------------------------
///  @brief Run the workitem.
//------------------------------------------------------------------------------
        virtual void run() {
        }
    };

//------------------------------------------------------------------------------
///  @brief Class representing a workflow manager.
//------------------------------------------------------------------------------
    template<typename T>
    class manager {
    private:
///  List of work items.
        std::forward_list<std::unique_ptr<work_item<T>>> items;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a work item.
//------------------------------------------------------------------------------
        manager() {}

//------------------------------------------------------------------------------
///  @brief Add a workflow item.
///
///  @params[in] item Workitem to add.
//------------------------------------------------------------------------------
        void add_item(std::unique_ptr<work_item<T>> &item) {
            item.push_back(item);
        }

//------------------------------------------------------------------------------
///  @brief Run work items.
//------------------------------------------------------------------------------
        void run() {
            for (auto &item : items) {
                item->run();
            }
        }
    };
}

#endif /* workflow_h */
