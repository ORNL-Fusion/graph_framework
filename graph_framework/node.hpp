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
#include "backend.hpp"

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
        virtual backend::buffer<T> evaluate() = 0;

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
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<T>> df(std::shared_ptr<leaf_node<T>> x) = 0;

//------------------------------------------------------------------------------
///  @brief Compile preamble.
///
///  Some nodes require additions to the preamble however most don't so define a
///  generic method that does nothing.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @params[in,out] visited   List of visited nodes.
//------------------------------------------------------------------------------
        virtual void compile_preamble(std::stringstream &stream,
                                      jit::register_map &registers,
                                      jit::visiter_map &visited) {}

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<T>> compile(std::stringstream &stream,
                                                      jit::register_map &registers) = 0;

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @params[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(std::shared_ptr<leaf_node<T>> x) = 0;

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @params[in] d Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const T d) {}

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @params[in] index Buffer index to set value.
///  @params[in] d     Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const size_t index,
                         const T d) {}

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @params[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const std::vector<T> &d) {}

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @params[in] d Backend buffer data to set.
//------------------------------------------------------------------------------
        virtual void set(const backend::buffer<T> &d) {}

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
///  Convenience type alias for node caches.
    template<typename T>
    using node_cache = std::vector<shared_leaf<T>>;

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
///  @params[in] a Argument.
//------------------------------------------------------------------------------
        straight_node(shared_leaf<T> a) :
        arg(a->reduce()) {}

//------------------------------------------------------------------------------
///  @brief Evaluate method.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            return this->arg->evaluate();
        }

//------------------------------------------------------------------------------
///  @brief Compile preamble.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
//------------------------------------------------------------------------------
    virtual void compile_preamble(std::stringstream &stream,
                                  jit::register_map &registers,
                                  jit::visiter_map &visited) {
        if (visited.find(this) == visited.end()) {
            this->arg->compile_preamble(stream, registers, visited);
            visited[this] = 0;
        }
    }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> compile(std::stringstream &stream,
                                       jit::register_map &registers) {
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
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
        branch_node(shared_leaf<T> l,
                    shared_leaf<T> r) :
        left(l->reduce()),
        right(r->reduce()) {}

//------------------------------------------------------------------------------
///  @brief Compile preamble.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.///  @params[in,out] visited   List of visited nodes.
//------------------------------------------------------------------------------
        virtual void compile_preamble(std::stringstream &stream,
                                      jit::register_map &registers,
                                      jit::visiter_map &visited) {
            if (visited.find(this) == visited.end()) {
                this->left->compile_preamble(stream, registers, visited);
                this->right->compile_preamble(stream, registers, visited);
                visited[this] = 0;
            }
        }

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
///  @params[in] l Left branch.
///  @params[in] m Middle branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
        triple_node(shared_leaf<T> l,
                    shared_leaf<T> m,
                    shared_leaf<T> r) :
        branch_node<T> (l, r),
        middle(m->reduce()) {}

//------------------------------------------------------------------------------
///  @brief Compile preamble.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.///  @params[in,out] visited   List of visited nodes.
//------------------------------------------------------------------------------
        virtual void compile_preamble(std::stringstream &stream,
                                      jit::register_map &registers,
                                      jit::visiter_map &visited) {
            if (visited.find(this) == visited.end()) {
                this->left->compile_preamble(stream, registers, visited);
                this->middle->compile_preamble(stream, registers, visited);
                this->right->compile_preamble(stream, registers, visited);
                visited[this] = 0;
            }
        }

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
        const backend::buffer<T> data;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a constant node from a scalar.
///
///  @params[in] d Scalar data to initalize.
//------------------------------------------------------------------------------
        constant_node(const T d) :
        data(1, d) {}

//------------------------------------------------------------------------------
///  @brief Construct a constant node from a vector.
///
///  @params[in] d Array buffer.
//------------------------------------------------------------------------------
        constant_node(const backend::buffer<T> &d) :
        data(d) {
            assert(d.size() == 1 && "Constants need to be scalar functions.");
        }

//------------------------------------------------------------------------------
///  @brief Evaluate method.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            return data;
        }

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  For basic nodes, there's nothing to reduce.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> reduce() {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> df(shared_leaf<T> x) {
            auto zero = std::make_shared<constant_node<T>> (static_cast<T> (0.0));
            for (auto &c : constant_node<T>::cache) {
                if (zero->is_match(c)) {
                    return c;
                }
            }
            constant_node<T>::cache.push_back(zero);
            return zero;
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> compile(std::stringstream &stream,
                                       jit::register_map &registers) {
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
///  @params[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T> x) {
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
        virtual void to_latex() const {
            std::cout << data.at(0);
        }

///  Cache for constructed nodes.
        inline static node_cache<T> cache;
    };

//------------------------------------------------------------------------------
///  @brief Construct a constant.
///
///  @params[in] d Scalar data to initalize.
///  @returns A reduced constant node.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> constant(const T d) {
        auto temp = std::make_shared<constant_node<T>> (d)->reduce();
        for (auto &c : constant_node<T>::cache) {
            if (temp->is_match(c)) {
                return c;
            }
        }
        constant_node<T>::cache.push_back(temp);
        return temp;
    }

//------------------------------------------------------------------------------
///  @brief Construct a constant.
///
///  @params[in] d Array buffer.
///  @returns A reduced constant node.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> constant(const backend::buffer<T> &d) {
        auto temp = std::make_shared<constant_node<T>> (d)->reduce();
        for (auto &c : constant_node<T>::cache) {
            if (temp->is_match(c)) {
                return c;
            }
        }
        constant_node<T>::cache.push_back(temp);
        return temp;
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
    template<typename T>
    using shared_constant = std::shared_ptr<constant_node<T>>;

//------------------------------------------------------------------------------
///  @brief Cast to a constant node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T>
    shared_constant<T> constant_cast(shared_leaf<T> x) {
        return std::dynamic_pointer_cast<constant_node<T>> (x);
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
        backend::buffer<T> buffer;
///  Latex Symbol for the variable when pretty printing.
        const std::string symbol;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a variable node with a size.
///
///  @params[in] s      Size of the data buffer.
///  @params[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
        variable_node(const size_t s,
                      const std::string &symbol) :
        buffer(s), symbol(symbol) {}

//------------------------------------------------------------------------------
///  @brief Construct a variable node from a scalar.
///
///  @params[in] s      Size of he data buffer.
///  @params[in] d      Scalar data to initalize.
///  @params[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
        variable_node(const size_t s, const T d,
                      const std::string &symbol) :
        buffer(s, d), symbol(symbol) {}

//------------------------------------------------------------------------------
///  @brief Construct a variable node from a vector.
///
///  @params[in] d Array buffer.
//------------------------------------------------------------------------------
        variable_node(const std::vector<T> &d,
                      const std::string &symbol) :
        buffer(d), symbol(symbol) {}

//------------------------------------------------------------------------------
///  @brief Construct a variable node from backend buffer.
///
///  @params[in] d Backend buffer.
//------------------------------------------------------------------------------
        variable_node(const backend::buffer<T> &d,
                      const std::string &symbol) :
        buffer(d), symbol(symbol) {}

//------------------------------------------------------------------------------
///  @brief Evaluate method.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            return buffer;
        }

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  For basic nodes, there's nothing to reduce.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> reduce() {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> df(shared_leaf<T> x) {
            return constant(static_cast<T> (this->is_match(x)));
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> compile(std::stringstream &stream,
                                       jit::register_map &registers) {
           return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @params[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T> x) {
            return this == x.get();
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @params[in] d Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const T d) {
            buffer.set(d);
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @params[in] index Index to place the value at.
///  @params[in] d     Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const size_t index, const T d) {
            buffer[index] = d;
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @params[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const std::vector<T> &d) {
            buffer.set(d);
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @params[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const backend::buffer<T> &d) {
            buffer = d;
        }

//------------------------------------------------------------------------------
///  @brief Get Symbol.
//------------------------------------------------------------------------------
        std::string get_symbol() const {
            return symbol;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << get_symbol();
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
///  @params[in] s      Size of the data buffer.
///  @params[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> variable(const size_t s,
                            const std::string &symbol) {
        return (std::make_shared<variable_node<T>> (s, symbol))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @params[in] s      Size of he data buffer.
///  @params[in] d      Scalar data to initalize.
///  @params[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> variable(const size_t s, const T d,
                            const std::string &symbol) {
        return (std::make_shared<variable_node<T>> (s, d, symbol))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @params[in] d      Array buffer.
///  @params[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> variable(const std::vector<T> &d,
                            const std::string &symbol) {
        return (std::make_shared<variable_node<T>> (d, symbol))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @params[in] d      Array buffer.
///  @params[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> variable(const backend::buffer<T> &d,
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
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T>
    shared_variable<T> variable_cast(shared_leaf<T> x) {
        return std::dynamic_pointer_cast<variable_node<T>> (x);
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
///  @brief Construct a pseudo variable node.
///
///  @params[in] a Argument.
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
        virtual shared_leaf<T> reduce() {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> df(shared_leaf<T> x) {
            return constant(static_cast<T> (this->is_match(x)));
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @params[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T> x) {
            return this == x.get();
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            this->arg->to_latex();
        }

///  Cache for constructed nodes.
        inline static node_cache<T> cache;
    };

//------------------------------------------------------------------------------
///  @brief Define pseudo variable convience function.
///
///  @params[in] x Argument.
///  @returns A reduced pseudo variable node.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> pseudo_variable(shared_leaf<T> x) {
        auto temp = std::make_shared<pseudo_variable_node<T>> (x)->reduce();
        for (auto &c : pseudo_variable_node<T>::cache) {
            if (temp->is_match(c)) {
                return c;
            }
        }
        pseudo_variable_node<T>::cache.push_back(temp);
        return temp;
    }

///  Convenience type alias for shared pseudo variable nodes.
    template<typename T>
    using shared_pseudo_variable = std::shared_ptr<pseudo_variable_node<T>>;

//------------------------------------------------------------------------------
///  @brief Cast to a pseudo variable node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T>
    shared_pseudo_variable<T> pseudo_variable_cast(shared_leaf<T> x) {
        return std::dynamic_pointer_cast<pseudo_variable_node<T>> (x);
    }
}

#endif /* node_h */
