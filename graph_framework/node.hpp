//------------------------------------------------------------------------------
///  @file node.hpp
///  @brief Base nodes of graph computation framework.
///
///  Defines a tree of operations that allows automatic differentiation.
//------------------------------------------------------------------------------

#ifndef node_h
#define node_h

#include <type_traits>
#include <cassert>
#include <memory>
#include <vector>

namespace graph {
//******************************************************************************
//  Base leaf node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a node leaf.
//------------------------------------------------------------------------------
    class leaf_node : public std::enable_shared_from_this<leaf_node> {
    public:
//------------------------------------------------------------------------------
///  @brief Destructor
//------------------------------------------------------------------------------
        virtual ~leaf_node() {}

//------------------------------------------------------------------------------
///  @brief Evaluate method.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual std::vector<double> evaluate() = 0;

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  For basic nodes, there's nothing to reduce.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> reduce() = 0;

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> df(std::shared_ptr<leaf_node> x) = 0;

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const double d) {}

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const std::vector<double> &d) {}
    };

//******************************************************************************
//  Base straight node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a node branch.
///
///  This ensures that the base leaf type has the common type between the two
///  template arguments.
//------------------------------------------------------------------------------
    template<typename N>
    class straight_node : public leaf_node {
    protected:
///  Argument
        std::shared_ptr<leaf_node> arg;

    public:

//------------------------------------------------------------------------------
///  @brief Class representing a straight node.
///
///  @param[in] a Argument.
//------------------------------------------------------------------------------
        straight_node(std::shared_ptr<N> a) :
        arg(a->reduce()) {}
    };

//******************************************************************************
//  Base branch node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a branch node.
///
///  This ensures that the base leaf type has the common type between the two
///  template arguments.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    class branch_node : public leaf_node {
    protected:
//  Left branch of the tree.
        std::shared_ptr<leaf_node> left;
//  Right branch of the tree.
        std::shared_ptr<leaf_node> right;

    public:

//------------------------------------------------------------------------------
///  @brief Reduces and assigns the left and right branches.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        branch_node(std::shared_ptr<LN> l,
                    std::shared_ptr<RN> r) :
        left(l->reduce()),
        right(r->reduce()) {}
    };

//******************************************************************************
//  Zero node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing zero.
//------------------------------------------------------------------------------
    class zero_node final : public leaf_node {
    public:
//------------------------------------------------------------------------------
///  @brief Evaluate zero.
///
///  @returns Zero
//------------------------------------------------------------------------------
        virtual std::vector<double> evaluate() final {
            return std::vector<double> (1, 0);
        }

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  For basic nodes, there's nothing to reduce.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> reduce() final {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> df(std::shared_ptr<leaf_node> x) final {
            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Construct a zero_node.
//------------------------------------------------------------------------------
    static const std::shared_ptr<leaf_node> zero = std::make_shared<zero_node> ();

//******************************************************************************
//  One node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing one.
//------------------------------------------------------------------------------
    class one_node final : public leaf_node {
    public:
//------------------------------------------------------------------------------
///  @brief Evaluate one.
///
///  @returns One
//------------------------------------------------------------------------------
        virtual std::vector<double> evaluate() final {
            return std::vector<double> (1, 1);
        }

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  For basic nodes, there's nothing to reduce.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> reduce() final {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> df(std::shared_ptr<leaf_node> x) final {
            return zero;
        }
    };

//------------------------------------------------------------------------------
///  @brief Construct a one_node.
//------------------------------------------------------------------------------
    static const std::shared_ptr<leaf_node> one = std::make_shared<one_node> ();

//******************************************************************************
//  Constant node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing data that cannot change.
//------------------------------------------------------------------------------
    class constant_node final : public leaf_node {
    private:
///  Storage buffer for the data.
        const std::vector<double> data;

    public:

//------------------------------------------------------------------------------
///  @brief Construct a variable node from a scalar.
///
///  @param[in] d Scalar data to initalize.
//------------------------------------------------------------------------------
        constant_node(const double &d) :
        data(1, d) {}

//------------------------------------------------------------------------------
///  @brief Construct a variable node from a vector.
///
///  @param[in] d Size of the .
//------------------------------------------------------------------------------
        constant_node(const std::vector<double> &d) :
        data(d) {}

//------------------------------------------------------------------------------
///  @brief Evaluate method.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual std::vector<double> evaluate() final {
            return data;
        }

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  For basic nodes, there's nothing to reduce.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> reduce() final {
            bool is_all_zero = true;
            bool is_all_one = true;
            bool is_all_same = true;

            const double same = data.at(0);
            for (double e: data) {
                is_all_zero = is_all_zero && e == 0;
                is_all_one = is_all_one && e == 1;
                is_all_same = is_all_same && e == same;

                if (!(is_all_zero || is_all_one || is_all_same)) {
                    break;
                }
            }

            if (is_all_zero) {
                return zero;
            } else if (is_all_one) {
                return one;
            } else if (is_all_same) {
                return std::make_shared<constant_node> (same);
            } else {
                return this->shared_from_this();
            }
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> df(std::shared_ptr<leaf_node> x) final {
            return zero;
        }
    };

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] d Scalar data to initalize.
//------------------------------------------------------------------------------
    std::shared_ptr<leaf_node> constant(const double d) {
        return (std::make_shared<constant_node> (1, d))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] d Array buffer.
//------------------------------------------------------------------------------
    std::shared_ptr<leaf_node> constant(const std::vector<double> &d) {
        return (std::make_shared<constant_node> (d))->reduce();
    }

//******************************************************************************
//  Constant node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing data that can change.
//------------------------------------------------------------------------------
    class variable_node final : public leaf_node {
    private:
///  Storage buffer for the data.
        std::vector<double> data;

    public:

//------------------------------------------------------------------------------
///  @brief Construct a variable node with a size.
///
///  @param[in] s Size of the data buffer.
//------------------------------------------------------------------------------
        variable_node(const size_t s) :
        data(s) {}

//------------------------------------------------------------------------------
///  @brief Construct a variable node from a scalar.
///
///  @param[in] s Size of he data buffer.
///  @param[in] d Scalar data to initalize.
//------------------------------------------------------------------------------
        variable_node(const size_t s, const double &d) :
        data(s, d) {}

//------------------------------------------------------------------------------
///  @brief Construct a variable node from a vector.
///
///  @param[in] d Array buffer.
//------------------------------------------------------------------------------
        variable_node(const std::vector<double> &d) :
        data(d) {}

//------------------------------------------------------------------------------
///  @brief Evaluate method.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual std::vector<double> evaluate() final {
            return data;
        }

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  For basic nodes, there's nothing to reduce.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> reduce() final {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> df(std::shared_ptr<leaf_node> x) final {
            if (x.get() == this) {
                return one;
            } else {
                return zero;
            }
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const double d) final {
            data.assign(data.size(), d);
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const std::vector<double> &d) final {
            data = d;
        }
    };

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] s Size of the data buffer.
//------------------------------------------------------------------------------
    std::shared_ptr<leaf_node> variable(const size_t s) {
        return (std::make_shared<variable_node> (s))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] s Size of he data buffer.
///  @param[in] d Scalar data to initalize.
//------------------------------------------------------------------------------
    std::shared_ptr<leaf_node> variable(const size_t s, const double d) {
        return (std::make_shared<variable_node> (s, d))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] d Array buffer.
//------------------------------------------------------------------------------
    std::shared_ptr<leaf_node> variable(const std::vector<double> &d) {
        return (std::make_shared<variable_node> (d))->reduce();
    }
}

#endif /* node_h */
