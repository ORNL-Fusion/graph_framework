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
#include "backend_protocall.hpp"

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
///  @brief Compile the node.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> compile(std::stringstream &stream,
                                                   jit::register_map<leaf_node<BACKEND>> &registers) = 0;

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
        virtual bool is_match(std::shared_ptr<leaf_node<BACKEND>> x) = 0;

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const typename BACKEND::base d) {}

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] index Buffer index to set value.
///  @param[in] d     Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const size_t index,
                         const typename BACKEND::base d) {}

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const std::vector<typename BACKEND::base> &d) {}

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Backend buffer data to set.
//------------------------------------------------------------------------------
        virtual void set(const BACKEND &d) {}

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const = 0;

///  Type def to retrieve the backend type.
        typedef BACKEND backend;
    };

///  Convenience type alias for shared leaf nodes.
    template<typename BACKEND>
    using shared_leaf = std::shared_ptr<leaf_node<BACKEND>>;
///  Convenience type alias for a vector of output nodes.
    template<class BACKEND>
    using output_nodes = std::vector<shared_leaf<BACKEND>>;

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
            return this->arg->evaluate();
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in] stream    String buffer stream.
///  @param[in] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual shared_leaf<BACKEND> compile(std::stringstream &stream,
                                             jit::register_map<leaf_node<BACKEND>> &registers) {
            return this->arg->compile(stream, registers);
        }

//------------------------------------------------------------------------------
///  @brief Get the argument.
//------------------------------------------------------------------------------
        shared_leaf<BACKEND> get_arg() {
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
///  @brief Construct a constant node from a scalar.
///
///  @param[in] d Scalar data to initalize.
//------------------------------------------------------------------------------
        constant_node(const typename BACKEND::base d) :
        data(1, d) {}

//------------------------------------------------------------------------------
///  @brief Construct a constant node from a vector.
///
///  @param[in] d Array buffer.
//------------------------------------------------------------------------------
        constant_node(const BACKEND &d) :
        data(d) {
            assert(d.size() == 1 && "Constants need to be scalar functions.");
        }

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
#ifdef USE_REDUCE
            if (data.size() > 1 && data.is_same()) {
                return std::make_shared<constant_node<BACKEND>> (data.at(0));
            }
#endif
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<BACKEND> df(shared_leaf<BACKEND> x) final {
            return std::make_shared<constant_node> (backend::base_cast<BACKEND> (0.0));
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in] stream    String buffer stream.
///  @param[in] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual shared_leaf<BACKEND> compile(std::stringstream &stream,
                                             jit::register_map<leaf_node<BACKEND>> &registers) final {
            if (registers.find(this) == registers.end()) {
                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<BACKEND> (stream);
                const auto temp = this->evaluate()[0];

                stream << " " << registers[this] << " = ";
                if constexpr (jit::is_complex<typename BACKEND::base> ()) {
                    jit::add_type<BACKEND> (stream);
                    stream << std::setprecision(jit::max_digits10<typename BACKEND::base> ())
                           << " (" << std::real(temp) << ","
                                   << std::imag(temp) << ")";
                } else {
                    stream << std::setprecision(jit::max_digits10<typename BACKEND::base> ())
                           << temp;
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
        virtual bool is_match(shared_leaf<BACKEND> x) final {
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
        bool is(const typename BACKEND::base d) {
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
    template<class BACKEND>
    shared_leaf<BACKEND> constant(const typename BACKEND::base d) {
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

///  Convenience type alias for shared constant nodes.
    template<typename N>
    using shared_constant = std::shared_ptr<constant_node<typename N::backend>>;

//------------------------------------------------------------------------------
///  @brief Cast to a constant node.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
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
        BACKEND buffer;
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
        variable_node(const size_t s,
                      const typename BACKEND::base d,
                      const std::string &symbol) :
        buffer(s, d), symbol(symbol) {}

//------------------------------------------------------------------------------
///  @brief Construct a variable node from a vector.
///
///  @param[in] d Array buffer.
//------------------------------------------------------------------------------
        variable_node(const std::vector<typename BACKEND::base> &d,
                      const std::string &symbol) :
        buffer(d), symbol(symbol) {}

//------------------------------------------------------------------------------
///  @brief Construct a variable node from backend buffer.
///
///  @param[in] d Backend buffer.
//------------------------------------------------------------------------------
        variable_node(const BACKEND &d,
                      const std::string &symbol) :
        buffer(d), symbol(symbol) {}

//------------------------------------------------------------------------------
///  @brief Evaluate method.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual BACKEND evaluate() final {
            return buffer;
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
            return constant<BACKEND> (backend::base_cast<BACKEND> (this->is_match(x)));
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in] stream    String buffer stream.
///  @param[in] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual shared_leaf<BACKEND> compile(std::stringstream &stream,
                                             jit::register_map<leaf_node<BACKEND>> &registers) final {
           return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<BACKEND> x) final {
            return this == x.get();
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const typename BACKEND::base d) final {
            buffer.set(d);
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] index Index to place the value at.
///  @param[in] d     Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const size_t index,
                         const typename BACKEND::base d) final {
            buffer[index] = d;
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const std::vector<typename BACKEND::base> &d) final {
            buffer.set(d);
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const BACKEND &d) final {
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
        typename BACKEND::base *data() {
            return buffer.data();
        }
    };

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] s      Size of the data buffer.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<class BACKEND>
    shared_leaf<BACKEND> variable(const size_t s,
                                  const std::string &symbol) {
        return (std::make_shared<variable_node<BACKEND>> (s, symbol))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] s      Size of he data buffer.
///  @param[in] d      Scalar data to initalize.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<class BACKEND>
    shared_leaf<BACKEND> variable(const size_t s,
                                  const typename BACKEND::base d,
                                  const std::string &symbol) {
        return (std::make_shared<variable_node<BACKEND>> (s, d, symbol))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] d      Array buffer.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<class BACKEND>
    shared_leaf<BACKEND> variable(const std::vector<typename BACKEND::base> &d,
                                  const std::string &symbol) {
        return (std::make_shared<variable_node<BACKEND>> (d, symbol))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @param[in] d      Array buffer.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<class BACKEND>
    shared_leaf<BACKEND> variable(const BACKEND &d,
                                  const std::string &symbol) {
        return (std::make_shared<variable_node<BACKEND>> (d, symbol))->reduce();
    }

///  Convenience type alias for shared variable nodes.
    template<class BACKEND>
    using shared_variable = std::shared_ptr<variable_node<BACKEND>>;
///  Convenience type alias for a vector of inputs.
    template<class BACKEND>
    using input_nodes = std::vector<shared_variable<BACKEND>>;
///  Convenience type alias for maping end codes back to inputs.
    template<class BACKEND>
    using map_nodes = std::vector<std::pair<graph::shared_leaf<BACKEND>,
                                            shared_variable<BACKEND>>>;

//------------------------------------------------------------------------------
///  @brief Cast to a variable node.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<class BACKEND>
    shared_variable<BACKEND> variable_cast(shared_leaf<BACKEND> x) {
        return std::dynamic_pointer_cast<variable_node<BACKEND>> (x);
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
        virtual shared_leaf<BACKEND> df(shared_leaf<BACKEND> x) final {
            if (this->is_match(x)) {
                return constant<BACKEND> (backend::base_cast<BACKEND> (1.0));
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
        virtual bool is_match(shared_leaf<BACKEND> x) final {
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
    shared_leaf<typename N::backend> cache(std::shared_ptr<N> x) {
        return (std::make_shared<cache_node<typename N::backend>> (x))->reduce();
    }

///  Convenience type alias for shared cache nodes.
    template<typename N>
    using shared_cache = std::shared_ptr<cache_node<typename N::backend>>;

//------------------------------------------------------------------------------
///  @brief Cast to a cache node.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
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
            return constant<BACKEND> (backend::base_cast<BACKEND> (this->is_match(x)));
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<BACKEND> x) final {
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
    shared_leaf<typename N::backend> pseudo_variable(std::shared_ptr<N> x) {
        return (std::make_shared<pseudo_variable_node<typename N::backend>> (x))->reduce();
    }

///  Convenience type alias for shared pseudo variable nodes.
    template<typename N>
    using shared_pseudo_variable = std::shared_ptr<pseudo_variable_node<typename N::backend>>;

//------------------------------------------------------------------------------
///  @brief Cast to a pseudo variable node.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename N>
    shared_pseudo_variable<N> pseudo_variable_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<pseudo_variable_node<typename N::backend>> (x);
    }
}

#endif /* node_h */
