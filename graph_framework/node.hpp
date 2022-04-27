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
///  @brief Reset the cache.
///
///  For any nodes that are not a cache node this is a no operation.
//------------------------------------------------------------------------------
        virtual void reset_cache() {}

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

///  Convience type alias for shared leaf nodes.
    template<typename BACKEND>
    using shared_leaf = std::shared_ptr<leaf_node<BACKEND>>;

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
        shared_leaf<BACKEND> arg;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a straight node.
///
///  @param[in] a Argument.
//------------------------------------------------------------------------------
        straight_node(shared_leaf<BACKEND> a) :
        arg(a->reduce()) {}

//------------------------------------------------------------------------------
///  @brief Evaluate method.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual BACKEND evaluate() {
            return arg->evaluate();
        }
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
        shared_leaf<BACKEND> left;
//  Right branch of the tree.
        shared_leaf<BACKEND> right;

    public:

//------------------------------------------------------------------------------
///  @brief Reduces and assigns the left and right branches.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        branch_node(shared_leaf<BACKEND> l,
                    shared_leaf<BACKEND> r) :
        left(l->reduce()),
        right(r->reduce()) {}

//------------------------------------------------------------------------------
///  @brief Get the left branch.
//------------------------------------------------------------------------------
        shared_leaf<BACKEND> get_left() {
            return this->left;
        }

//------------------------------------------------------------------------------
///  @brief Get the right branch.
//------------------------------------------------------------------------------
        shared_leaf<BACKEND> get_right() {
            return this->right;
        }
    };

//******************************************************************************
//  Base triple node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a triple branch node.
///
///  This ensures that the base leaf type has the common type between the two
///  template arguments.
//------------------------------------------------------------------------------
    template<class BACKEND>
    class triple_node : public branch_node<BACKEND> {
    protected:
//  Middle branch of the tree.
        shared_leaf<BACKEND> middle;

    public:

//------------------------------------------------------------------------------
///  @brief Reduces and assigns the left and right branches.
///
///  @param[in] l Left branch.
///  @param[in] m Middle branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        triple_node(shared_leaf<BACKEND> l,
                    shared_leaf<BACKEND> m,
                    shared_leaf<BACKEND> r) :
        branch_node<BACKEND> (l, r),
        middle(m->reduce()) {}

//------------------------------------------------------------------------------
///  @brief Get the right branch.
//------------------------------------------------------------------------------
        shared_leaf<BACKEND> get_middle() {
            return this->middle;
        }
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
        virtual shared_leaf<BACKEND> reduce() final {
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
        virtual shared_leaf<BACKEND> df(shared_leaf<BACKEND> x) final {
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
    shared_leaf<BACKEND> constant(const double d) {
        return (std::make_shared<constant_node<BACKEND>> (d))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a constant.
///
///  @param[in] d Array buffer.
///  @returns A reduced constant node.
//------------------------------------------------------------------------------
    template<class BACKEND>
    shared_leaf<BACKEND> constant(const std::vector<double> &d) {
        return (std::make_shared<constant_node<BACKEND>> (d))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a constant.
///
///  @param[in] d Array buffer.
///  @returns A reduced constant node.
//------------------------------------------------------------------------------
    template<class BACKEND>
    shared_leaf<BACKEND> constant(const BACKEND &d) {
        return (std::make_shared<constant_node<BACKEND>> (d))->reduce();
    }

///  Convience type alias for shared constant nodes.
    template<typename N>
    using shared_constant = std::shared_ptr<constant_node<typename N::backend>>;

//------------------------------------------------------------------------------
///  @brief Cast to a constant node.
///
///  @param[in] x Leaf node to attempt cast.
//------------------------------------------------------------------------------
    template<typename N>
    shared_constant<N> constant_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<constant_node<typename N::backend>> (x);
    }

//******************************************************************************
//  Variable node.
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
        virtual shared_leaf<BACKEND> reduce() final {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<BACKEND> df(shared_leaf<BACKEND> x) final {
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
    shared_leaf<BACKEND> variable(const size_t s) {
        return (std::make_shared<variable_node<BACKEND>> (s))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] s Size of he data buffer.
///  @param[in] d Scalar data to initalize.
//------------------------------------------------------------------------------
    template<class BACKEND>
    shared_leaf<BACKEND> variable(const size_t s, const double d) {
        return (std::make_shared<variable_node<BACKEND>> (s, d))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] d Array buffer.
//------------------------------------------------------------------------------
    template<class BACKEND>
    shared_leaf<BACKEND> variable(const std::vector<double> &d) {
        return (std::make_shared<variable_node<BACKEND>> (d))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] d Array buffer.
//------------------------------------------------------------------------------
    template<class BACKEND>
    shared_leaf<BACKEND> variable(const BACKEND &d) {
        return (std::make_shared<variable_node<BACKEND>> (d))->reduce();
    }

///  Convience type alias for shared variable nodes.
    template<typename N>
    using shared_variable = std::shared_ptr<variable_node<typename N::backend>>;

//------------------------------------------------------------------------------
///  @brief Cast to a variable node.
///
///  @param[in] x Leaf node to attempt cast.
//------------------------------------------------------------------------------
    template<typename N>
    shared_variable<N> variable_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<variable_node<typename N::backend>> (x);
    }

//******************************************************************************
//  Cache node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing data that can be cached.
///
///  Cache nodes save the results of evaluate so the subtree does not need to be
///  revaluated.
//------------------------------------------------------------------------------
    template<class BACKEND>
    class cache_node final : public straight_node<BACKEND> {
    private:
///  Storage buffer for the data.
        BACKEND data;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a cache node.
///
///  @param[in] a Argument.
//------------------------------------------------------------------------------
        cache_node(shared_leaf<BACKEND> a) :
        straight_node<BACKEND> (a),
        data(a->evaluate()) {}

//------------------------------------------------------------------------------
///  @brief Evaluate method.
///
///  Only need to evaluate the sub tree if the cache is not set.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual BACKEND evaluate() final {
            return data;
        }

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  When the arg is a constant there's no point to caching anything. Replace
///  cache node with the constant node. Otherwise do nothing.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<BACKEND> reduce() final {
            if (constant_cast(this->arg).get() == nullptr) {
                return this->shared_from_this();
            } else {
                return this->arg;
            }
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  This has the consequence of removing the cache node.
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<BACKEND> df(shared_leaf<BACKEND> x) final {
            if (x.get() == this) {
                return constant<BACKEND> (1);
            } else {
                return this->arg->df(x)->reduce();
            }
        }

//------------------------------------------------------------------------------
///  @brief Reset the cache.
//------------------------------------------------------------------------------
        virtual void reset_cache() final {
            data = this->arg->evaluate();
        }
    };

//------------------------------------------------------------------------------
///  @brief Define cache convience function.
///
///  @param[in] x Argument.
///  @returns A reduced cache node.
//------------------------------------------------------------------------------
    template<typename N>
    shared_leaf<typename N::backend> cache(std::shared_ptr<N> x) {
        return (std::make_shared<cache_node<typename N::backend>> (x))->reduce();
    }

///  Convience type alias for shared cache nodes.
    template<typename N>
    using shared_cache = std::shared_ptr<cache_node<typename N::backend>>;

//------------------------------------------------------------------------------
///  @brief Cast to a cache node.
///
///  @param[in] x Leaf node to attempt cast.
//------------------------------------------------------------------------------
    template<typename N>
    shared_cache<N> cache_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<cache_node<typename N::backend>> (x);
    }

//******************************************************************************
//  Pseudo variable node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a subexpression that acts like a variable.
///
///  Pseudo variable nodes treat sub trees as if they were a variable. This
///  ensures that the expression returns zero when taking a derivative with
///  something that is not itself.
//------------------------------------------------------------------------------
    template<class BACKEND>
    class pseudo_variable_node final : public straight_node<BACKEND> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a cache node.
///
///  @param[in] a Argument.
//------------------------------------------------------------------------------
        pseudo_variable_node(shared_leaf<BACKEND> a) :
        straight_node<BACKEND> (a) {}

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  For basic nodes, there's nothing to reduce.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<BACKEND> reduce() final {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<BACKEND> df(shared_leaf<BACKEND> x) final {
            return constant<BACKEND> (x.get() == this);
        }
    };

//------------------------------------------------------------------------------
///  @brief Define pseudo variable convience function.
///
///  @param[in] x Argument.
///  @returns A reduced cache node.
//------------------------------------------------------------------------------
    template<typename N>
    shared_leaf<typename N::backend> pseudo_variable(std::shared_ptr<N> x) {
        return (std::make_shared<pseudo_variable_node<typename N::backend>> (x))->reduce();
    }

///  Convience type alias for shared pseudo variable nodes.
    template<typename N>
    using shared_pseudo_variable = std::shared_ptr<variable_node<typename N::backend>>;

//------------------------------------------------------------------------------
///  @brief Cast to a pseudo variable node.
///
///  @param[in] x Leaf node to attempt cast.
//------------------------------------------------------------------------------
    template<typename N>
    shared_pseudo_variable<N> pseudo_variable_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<pseudo_variable_node<typename N::backend>> (x);
    }
}

#endif /* node_h */
