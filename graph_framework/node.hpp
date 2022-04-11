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
    template<class BACKEND>
    class leaf_node : public std::enable_shared_from_this<leaf_node<BACKEND>> {
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
        virtual BACKEND evaluate() = 0;

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
        virtual std::shared_ptr<leaf_node> df(std::shared_ptr<leaf_node<BACKEND>> x) = 0;

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const double d) {}

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] index Buffer index to set value.
///  @param[in] d     Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const size_t index,
                         const double d) {}

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const std::vector<double> &d) {}

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Backend buffer data to set.
//------------------------------------------------------------------------------
        virtual void set(const BACKEND &d) {}

///  Type def to retrieve the backend type.
        typedef BACKEND backend;
    };

//******************************************************************************
//  Base straight node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a straight node.
///
///  This ensures that the base leaf type has the common type between the two
///  template arguments.
//------------------------------------------------------------------------------
    template<class BACKEND>
    class straight_node : public leaf_node<BACKEND> {
    protected:
///  Argument
        std::shared_ptr<leaf_node<BACKEND>> arg;

    public:
//------------------------------------------------------------------------------
///  @brief Class representing a straight node.
///
///  @param[in] a Argument.
//------------------------------------------------------------------------------
        straight_node(std::shared_ptr<leaf_node<BACKEND>> a) :
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
    template<class BACKEND>
    class branch_node : public leaf_node<BACKEND> {
    protected:
//  Left branch of the tree.
        std::shared_ptr<leaf_node<BACKEND>> left;
//  Right branch of the tree.
        std::shared_ptr<leaf_node<BACKEND>> right;

    public:

//------------------------------------------------------------------------------
///  @brief Reduces and assigns the left and right branches.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        branch_node(std::shared_ptr<leaf_node<BACKEND>> l,
                    std::shared_ptr<leaf_node<BACKEND>> r) :
        left(l->reduce()),
        right(r->reduce()) {}
    };

//******************************************************************************
//  Constant node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing data that cannot change.
//------------------------------------------------------------------------------
    template<class BACKEND>
    class constant_node final : public leaf_node<BACKEND> {
    private:
///  Storage buffer for the data.
        const BACKEND data;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a variable node from a scalar.
///
///  @param[in] d Scalar data to initalize.
//------------------------------------------------------------------------------
        constant_node(const double d) :
        data(1, d) {}

//------------------------------------------------------------------------------
///  @brief Construct a variable node from a vector.
///
///  @param[in] d Size of the .
//------------------------------------------------------------------------------
        constant_node(const std::vector<double> &d) :
        data(d) {}

//------------------------------------------------------------------------------
///  @brief Construct a variable node from a vector.
///
///  @param[in] d Size of the .
//------------------------------------------------------------------------------
        constant_node(const BACKEND &d) :
        data(d) {}

//------------------------------------------------------------------------------
///  @brief Evaluate method.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual BACKEND evaluate() final {
            return data;
        }

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  For basic nodes, there's nothing to reduce.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<BACKEND>> reduce() final {
            if (data.size() > 1 && data.is_same()) {
                return std::make_shared<constant_node<BACKEND>> (data.at(0));
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<BACKEND>>
        df(std::shared_ptr<leaf_node<BACKEND>> x) final {
            return std::make_shared<constant_node> (0);
        }

//------------------------------------------------------------------------------
///  @brief Check if the constant is value.
//------------------------------------------------------------------------------
        bool is(const double d) {
            return data.size() == 1 && data.at(0) == d;
        }
    };

//------------------------------------------------------------------------------
///  @brief Construct a constant.
///
///  @param[in] d Scalar data to initalize.
///  @returns A reduced constant node.
//------------------------------------------------------------------------------
    template<class BACKEND>
    std::shared_ptr<leaf_node<BACKEND>> constant(const double d) {
        return (std::make_shared<constant_node<BACKEND>> (d))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a constant.
///
///  @param[in] d Array buffer.
///  @returns A reduced constant node.
//------------------------------------------------------------------------------
    template<class BACKEND>
    std::shared_ptr<leaf_node<BACKEND>> constant(const std::vector<double> &d) {
        return (std::make_shared<constant_node<BACKEND>> (d))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a constant.
///
///  @param[in] d Array buffer.
///  @returns A reduced constant node.
//------------------------------------------------------------------------------
    template<class BACKEND>
    std::shared_ptr<leaf_node<BACKEND>> constant(const BACKEND &d) {
        return (std::make_shared<constant_node<BACKEND>> (d))->reduce();
    }

//******************************************************************************
//  Constant node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing data that can change.
//------------------------------------------------------------------------------
    template<class BACKEND>
    class variable_node final : public leaf_node<BACKEND> {
    private:
///  Storage buffer for the data.
        BACKEND data;

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
        variable_node(const size_t s, const double d) :
        data(s, d) {}

//------------------------------------------------------------------------------
///  @brief Construct a variable node from a vector.
///
///  @param[in] d Array buffer.
//------------------------------------------------------------------------------
        variable_node(const std::vector<double> &d) :
        data(d) {}

//------------------------------------------------------------------------------
///  @brief Construct a variable node from backend buffer.
///
///  @param[in] d Backend buffer.
//------------------------------------------------------------------------------
        variable_node(const BACKEND &d) :
        data(d) {}

//------------------------------------------------------------------------------
///  @brief Evaluate method.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual BACKEND evaluate() final {
            return data;
        }

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  For basic nodes, there's nothing to reduce.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<BACKEND>> reduce() final {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<BACKEND>>
        df(std::shared_ptr<leaf_node<BACKEND>> x) final {
            return constant<BACKEND> (x.get() == this);
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const double d) final {
            data.set(d);
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] index Index to place the value at.
///  @param[in] d     Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const size_t index, const double d) final {
            data[index] = d;
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const std::vector<double> &d) final {
            data = BACKEND(d);
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const BACKEND &d) final {
            data = d;
        }
    };

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] s Size of the data buffer.
//------------------------------------------------------------------------------
    template<class BACKEND>
    std::shared_ptr<leaf_node<BACKEND>> variable(const size_t s) {
        return (std::make_shared<variable_node<BACKEND>> (s))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] s Size of he data buffer.
///  @param[in] d Scalar data to initalize.
//------------------------------------------------------------------------------
    template<class BACKEND>
    std::shared_ptr<leaf_node<BACKEND>>
    variable(const size_t s, const double d) {
        return (std::make_shared<variable_node<BACKEND>> (s, d))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] d Array buffer.
//------------------------------------------------------------------------------
    template<class BACKEND>
    std::shared_ptr<leaf_node<BACKEND>> variable(const std::vector<double> &d) {
        return (std::make_shared<variable_node<BACKEND>> (d))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] d Array buffer.
//------------------------------------------------------------------------------
    template<class BACKEND>
    std::shared_ptr<leaf_node<BACKEND>> variable(const BACKEND &d) {
        return (std::make_shared<variable_node<BACKEND>> (d))->reduce();
    }
}

#endif /* node_h */
