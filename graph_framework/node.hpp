//------------------------------------------------------------------------------
///  @file node.hpp
///  @brief Base nodes of graph computation framework.
///
///  Defines a tree of operations that allows automatic differentiation.
//------------------------------------------------------------------------------

#ifndef node_h
#define node_h

#include <iostream>
#include <string>
#include <type_traits>
#include <cassert>
#include <memory>
#include <vector>
#include <iomanip>

#include "register.hpp"
#include "cpu_backend.hpp"

namespace graph {
//******************************************************************************
//  Base leaf node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a node leaf.
//------------------------------------------------------------------------------
    template<typename T>
    class leaf_node : public std::enable_shared_from_this<leaf_node<T>> {
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
        virtual backend::cpu<T> evaluate() = 0;

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
        virtual std::shared_ptr<leaf_node<T>> df(std::shared_ptr<leaf_node<T>> x) = 0;

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<T>> compile(std::stringstream &stream,
                                                      jit::register_map<leaf_node<T>> &registers) = 0;

//------------------------------------------------------------------------------
///  @brief Reset the cache.
///
///  For any nodes that are not a cache node this is a no operation.
//------------------------------------------------------------------------------
        virtual void reset_cache() {}

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(std::shared_ptr<leaf_node<T>> x) = 0;

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const T d) {}

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] index Buffer index to set value.
///  @param[in] d     Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const size_t index,
                         const T d) {}

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const std::vector<T> &d) {}

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Backend buffer data to set.
//------------------------------------------------------------------------------
        virtual void set(const backend::cpu<T> &d) {}

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const = 0;

///  Type def to retrieve the backend type.
        typedef T base;
    };

///  Convenience type alias for shared leaf nodes.
    template<typename T>
    using shared_leaf = std::shared_ptr<leaf_node<T>>;
///  Convenience type alias for a vector of output nodes.
    template<typename T>
    using output_nodes = std::vector<shared_leaf<T>>;

//******************************************************************************
//  Base straight node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a straight node.
///
///  This ensures that the base leaf type has the common type between the two
///  template arguments.
//------------------------------------------------------------------------------
    template<typename T>
    class straight_node : public leaf_node<T> {
    protected:
///  Argument
        shared_leaf<T> arg;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a straight node.
///
///  @param[in] a Argument.
//------------------------------------------------------------------------------
        straight_node(shared_leaf<T> a) :
        arg(a->reduce()) {}

//------------------------------------------------------------------------------
///  @brief Evaluate method.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual backend::cpu<T> evaluate() {
            return this->arg->evaluate();
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in] stream    String buffer stream.
///  @param[in] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> compile(std::stringstream &stream,
                                       jit::register_map<leaf_node<T>> &registers) {
            return this->arg->compile(stream, registers);
        }

//------------------------------------------------------------------------------
///  @brief Get the argument.
//------------------------------------------------------------------------------
        shared_leaf<T> get_arg() {
            return this->arg;
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
    template<typename T>
    class branch_node : public leaf_node<T> {
    protected:
//  Left branch of the tree.
        shared_leaf<T> left;
//  Right branch of the tree.
        shared_leaf<T> right;

    public:

//------------------------------------------------------------------------------
///  @brief Reduces and assigns the left and right branches.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        branch_node(shared_leaf<T> l,
                    shared_leaf<T> r) :
        left(l->reduce()),
        right(r->reduce()) {}

//------------------------------------------------------------------------------
///  @brief Get the left branch.
//------------------------------------------------------------------------------
        shared_leaf<T> get_left() {
            return this->left;
        }

//------------------------------------------------------------------------------
///  @brief Get the right branch.
//------------------------------------------------------------------------------
        shared_leaf<T> get_right() {
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
    template<typename T>
    class triple_node : public branch_node<T> {
    protected:
//  Middle branch of the tree.
        shared_leaf<T> middle;

    public:

//------------------------------------------------------------------------------
///  @brief Reduces and assigns the left and right branches.
///
///  @param[in] l Left branch.
///  @param[in] m Middle branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        triple_node(shared_leaf<T> l,
                    shared_leaf<T> m,
                    shared_leaf<T> r) :
        branch_node<T> (l, r),
        middle(m->reduce()) {}

//------------------------------------------------------------------------------
///  @brief Get the right branch.
//------------------------------------------------------------------------------
        shared_leaf<T> get_middle() {
            return this->middle;
        }
    };

//******************************************************************************
//  Constant node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing data that cannot change.
//------------------------------------------------------------------------------
    template<typename T>
    class constant_node final : public leaf_node<T> {
    private:
///  Storage buffer for the data.
        const backend::cpu<T> data;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a constant node from a scalar.
///
///  @param[in] d Scalar data to initalize.
//------------------------------------------------------------------------------
        constant_node(const T d) :
        data(1, d) {}

//------------------------------------------------------------------------------
///  @brief Construct a constant node from a vector.
///
///  @param[in] d Array buffer.
//------------------------------------------------------------------------------
        constant_node(const backend::cpu<T> &d) :
        data(d) {
            assert(d.size() == 1 && "Constants need to be scalar functions.");
        }

//------------------------------------------------------------------------------
///  @brief Evaluate method.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual backend::cpu<T> evaluate() final {
            return data;
        }

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  For basic nodes, there's nothing to reduce.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> reduce() final {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> df(shared_leaf<T> x) final {
            return std::make_shared<constant_node<T>> (static_cast<T> (0.0));
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in] stream    String buffer stream.
///  @param[in] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> compile(std::stringstream &stream,
                                       jit::register_map<leaf_node<T>> &registers) final {
            if (registers.find(this) == registers.end()) {
                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                const T temp = this->evaluate()[0];

                stream << " " << registers[this] << " = ";
                if constexpr (jit::is_complex<T> ()) {
                    jit::add_type<T> (stream);
                    stream << " (" << std::real(temp) << ","
                                   << std::imag(temp) << ")";
                } else {
                    stream << temp;
                }
                stream << ";" << std::endl;
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T> x) final {
            if (this == x.get()) {
                return true;
            }

            auto x_cast = constant_cast(x);
            if (x_cast.get()) {
                return this->evaluate() == x_cast->evaluate();
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Check if the constant is value.
//------------------------------------------------------------------------------
        bool is(const T d) {
            return data.size() == 1 && data.at(0) == d;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const final {
            std::cout << data.at(0);
        }
    };

//------------------------------------------------------------------------------
///  @brief Construct a constant.
///
///  @param[in] d Scalar data to initalize.
///  @returns A reduced constant node.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> constant(const T d) {
        return (std::make_shared<constant_node<T>> (d))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a constant.
///
///  @param[in] d Array buffer.
///  @returns A reduced constant node.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> constant(const backend::cpu<T> &d) {
        return (std::make_shared<constant_node<T>> (d))->reduce();
    }

//  Define some common constants.
//------------------------------------------------------------------------------
///  @brief Create a zero constant.
///
///  @returns A zero constant.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> zero() {
        return constant(static_cast<T> (0.0));
    }
        
//------------------------------------------------------------------------------
///  @brief Create a one constant.
///
///  @returns A one constant.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> one() {
        return constant(static_cast<T> (1.0));
    }
        
//------------------------------------------------------------------------------
///  @brief Create a negative one constant.
///
///  @returns A negative one constant.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> none() {
        return constant(static_cast<T> (-1.0));
    }
        
//------------------------------------------------------------------------------
///  @brief Create a two constant.
///
///  @returns A two constant.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> two() {
        return constant(static_cast<T> (2.0));
    }

///  Convenience type alias for shared constant nodes.
    template<typename N>
    using shared_constant = std::shared_ptr<constant_node<typename N::base>>;

//------------------------------------------------------------------------------
///  @brief Cast to a constant node.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename N>
    shared_constant<N> constant_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<constant_node<typename N::base>> (x);
    }

//******************************************************************************
//  Variable node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing data that can change.
//------------------------------------------------------------------------------
    template<typename T>
    class variable_node final : public leaf_node<T> {
    private:
///  Storage buffer for the data.
        backend::cpu<T> buffer;
///  Latex Symbol for the variable when pretty printing.
        const std::string symbol;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a variable node with a size.
///
///  @param[in] s      Size of the data buffer.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
        variable_node(const size_t s,
                      const std::string &symbol) :
        buffer(s), symbol(symbol) {}

//------------------------------------------------------------------------------
///  @brief Construct a variable node from a scalar.
///
///  @param[in] s      Size of he data buffer.
///  @param[in] d      Scalar data to initalize.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
        variable_node(const size_t s, const T d,
                      const std::string &symbol) :
        buffer(s, d), symbol(symbol) {}

//------------------------------------------------------------------------------
///  @brief Construct a variable node from a vector.
///
///  @param[in] d Array buffer.
//------------------------------------------------------------------------------
        variable_node(const std::vector<T> &d,
                      const std::string &symbol) :
        buffer(d), symbol(symbol) {}

//------------------------------------------------------------------------------
///  @brief Construct a variable node from backend buffer.
///
///  @param[in] d Backend buffer.
//------------------------------------------------------------------------------
        variable_node(const backend::cpu<T> &d,
                      const std::string &symbol) :
        buffer(d), symbol(symbol) {}

//------------------------------------------------------------------------------
///  @brief Evaluate method.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual backend::cpu<T> evaluate() final {
            return buffer;
        }

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  For basic nodes, there's nothing to reduce.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> reduce() final {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> df(shared_leaf<T> x) final {
            return constant(static_cast<T> (this->is_match(x)));
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in] stream    String buffer stream.
///  @param[in] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> compile(std::stringstream &stream,
                                       jit::register_map<leaf_node<T>> &registers) final {
           return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T> x) final {
            return this == x.get();
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const T d) final {
            buffer.set(d);
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] index Index to place the value at.
///  @param[in] d     Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const size_t index, const T d) final {
            buffer[index] = d;
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const std::vector<T> &d) final {
            buffer.set(d);
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const backend::cpu<T> &d) final {
            buffer = d;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const final {
            std::cout << symbol;
        }

//------------------------------------------------------------------------------
///  @brief Get the size of the variable buffer.
//------------------------------------------------------------------------------
        size_t size() {
            return buffer.size();
        }

//------------------------------------------------------------------------------
///  @brief Get a pointer to raw buffer.
///
///  @returns A buffer to the underlying data.
//------------------------------------------------------------------------------
        T *data() {
            return buffer.data();
        }
    };

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] s      Size of the data buffer.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> variable(const size_t s,
                            const std::string &symbol) {
        return (std::make_shared<variable_node<T>> (s, symbol))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] s      Size of he data buffer.
///  @param[in] d      Scalar data to initalize.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> variable(const size_t s, const T d,
                            const std::string &symbol) {
        return (std::make_shared<variable_node<T>> (s, d, symbol))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] d      Array buffer.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> variable(const std::vector<T> &d,
                            const std::string &symbol) {
        return (std::make_shared<variable_node<T>> (d, symbol))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] d      Array buffer.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> variable(const backend::cpu<T> &d,
                            const std::string &symbol) {
        return (std::make_shared<variable_node<T>> (d, symbol))->reduce();
    }

///  Convenience type alias for shared variable nodes.
    template<typename T>
    using shared_variable = std::shared_ptr<variable_node<T>>;
///  Convenience type alias for a vector of inputs.
    template<typename T>
    using input_nodes = std::vector<shared_variable<T>>;
///  Convenience type alias for maping end codes back to inputs.
    template<typename T>
    using map_nodes = std::vector<std::pair<graph::shared_leaf<T>,
                                            shared_variable<T>>>;

//------------------------------------------------------------------------------
///  @brief Cast to a variable node.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T>
    shared_variable<T> variable_cast(shared_leaf<T> x) {
        return std::dynamic_pointer_cast<variable_node<T>> (x);
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
    template<typename T>
    class cache_node final : public straight_node<T> {
    private:
///  Storage buffer for the data.
        backend::cpu<T> data;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a cache node.
///
///  @param[in] a Argument.
//------------------------------------------------------------------------------
        cache_node(shared_leaf<T> a) :
        straight_node<T> (a),
        data(a->evaluate()) {}

//------------------------------------------------------------------------------
///  @brief Evaluate method.
///
///  Only need to evaluate the sub tree if the cache is not set.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual backend::cpu<T> evaluate() final {
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
        virtual shared_leaf<T> reduce() final {
#ifdef USE_REDUCE
            if (constant_cast(this->arg).get()) {
                return this->arg;
            }
#endif
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  This has the consequence of removing the cache node.
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> df(shared_leaf<T> x) final {
            if (this->is_match(x)) {
                return one<T> ();
            } else {
                return this->arg->df(x)->reduce();
            }
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T> x) final {
            if (this == x.get()) {
                return true;
            }

            auto x_cast = cache_cast(x);
            if (x_cast.get()) {
                return this->arg->is_match(x_cast->get_arg());
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Reset the cache.
//------------------------------------------------------------------------------
        virtual void reset_cache() final {
            data = this->arg->evaluate();
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const final {
            return this->arg->to_latex();
        }
    };

//------------------------------------------------------------------------------
///  @brief Define cache convience function.
///
///  @param[in] x Argument.
///  @returns A reduced cache node.
//------------------------------------------------------------------------------
    template<typename N>
    shared_leaf<typename N::base> cache(std::shared_ptr<N> x) {
        return (std::make_shared<cache_node<typename N::base>> (x))->reduce();
    }

///  Convenience type alias for shared cache nodes.
    template<typename N>
    using shared_cache = std::shared_ptr<cache_node<typename N::base>>;

//------------------------------------------------------------------------------
///  @brief Cast to a cache node.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename N>
    shared_cache<N> cache_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<cache_node<typename N::base>> (x);
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
    template<typename T>
    class pseudo_variable_node final : public straight_node<T> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a cache node.
///
///  @param[in] a Argument.
//------------------------------------------------------------------------------
        pseudo_variable_node(shared_leaf<T> a) :
        straight_node<T> (a) {}

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  For basic nodes, there's nothing to reduce.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> reduce() final {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> df(shared_leaf<T> x) final {
            return constant(static_cast<T> (this->is_match(x)));
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T> x) final {
            return this == x.get();
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const final {
            this->arg->to_latex();
        }
    };

//------------------------------------------------------------------------------
///  @brief Define pseudo variable convience function.
///
///  @param[in] x Argument.
///  @returns A reduced cache node.
//------------------------------------------------------------------------------
    template<typename N>
    shared_leaf<typename N::base> pseudo_variable(std::shared_ptr<N> x) {
        return (std::make_shared<pseudo_variable_node<typename N::base>> (x))->reduce();
    }

///  Convenience type alias for shared pseudo variable nodes.
    template<typename N>
    using shared_pseudo_variable = std::shared_ptr<pseudo_variable_node<typename N::base>>;

//------------------------------------------------------------------------------
///  @brief Cast to a pseudo variable node.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename N>
    shared_pseudo_variable<N> pseudo_variable_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<pseudo_variable_node<typename N::base>> (x);
    }
}

#endif /* node_h */
