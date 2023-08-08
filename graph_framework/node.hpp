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
#include <memory>
#include <iomanip>
#include <functional>

#include "backend.hpp"

namespace graph {
//******************************************************************************
//  Base leaf node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a node leaf.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    class leaf_node : public std::enable_shared_from_this<leaf_node<T, SAFE_MATH>> {
    protected:
///  Hash for node.
        const size_t hash;
///  Graph complexity.
        const size_t complexity;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a basic node.
///
///  @params[in] s     Node string to hash.
///  @params[in] count Number of nodes in the subgraph.
//------------------------------------------------------------------------------
        leaf_node(const std::string s,
                  const size_t count) :
        hash(std::hash<std::string>{} (s)),
        complexity(count) {}

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
        virtual std::shared_ptr<leaf_node<T, SAFE_MATH>>
        df(std::shared_ptr<leaf_node<T, SAFE_MATH>> x) = 0;

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
        virtual void compile_preamble(std::ostringstream &stream,
                                      jit::register_map &registers,
                                      jit::visiter_map &visited) {}

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<T, SAFE_MATH>>
        compile(std::ostringstream &stream,
                jit::register_map &registers) = 0;

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @params[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(std::shared_ptr<leaf_node<T, SAFE_MATH>> x) = 0;

//------------------------------------------------------------------------------
///  @brief Check if the base of the powers match.
///
///  @params[in] x Other graph to check if the bases match.
///  @returns True if the powers of the nodes match.
//------------------------------------------------------------------------------
        bool is_power_base_match(std::shared_ptr<leaf_node<T, SAFE_MATH>> x) {
            return this->get_power_base()->is_match(x->get_power_base());
        }

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

//------------------------------------------------------------------------------
///  @brief Test if node acts like a constant.
///
///  @returns True if the node acts like a constant.
//------------------------------------------------------------------------------
        virtual bool is_constant_like() const = 0;

//------------------------------------------------------------------------------
///  @brief Test if all the subnodes terminate in variables.
///
///  @returns True if all the subnodes terminate in variables.
//------------------------------------------------------------------------------
        virtual bool is_all_variables() const = 0;

//------------------------------------------------------------------------------
///  @brief Test if the node acts like a power of variable.
///
///  Most notes are not so default to false.
///
///  @returns True the node is power like and false otherwise.
//------------------------------------------------------------------------------
        virtual bool is_power_like() const {
            return false;
        }

//------------------------------------------------------------------------------
///  @brief Get the base of a power.
///
///  Most node can be treated as x^1 so just return this node.
///
///  @returns The base of a power like node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<T, SAFE_MATH>> get_power_base() {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Get the exponent of a power.
///
///  Most node can be treated as x^1 so just return one for those nodes but we
///  need todo that manually in the derived classes.
///
///  @returns The exponent of a power like node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<T, SAFE_MATH>> get_power_exponent() const = 0;

//------------------------------------------------------------------------------
///  @brief Get the hash for the node.
///
///  @returns The hash for the current node.
//------------------------------------------------------------------------------
        size_t get_hash() const {
            return hash;
        }

//------------------------------------------------------------------------------
///  @brief Get the number of nodes in the subgraph.
///
///  @returns The complexity count.
//------------------------------------------------------------------------------
        size_t get_complexity() const {
            return complexity;
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<T, SAFE_MATH>> remove_pseudo() {
            return this->shared_from_this();
        }

///  Cache for constructed nodes.
        inline thread_local static std::map<size_t,
                                            std::shared_ptr<leaf_node<T, SAFE_MATH>>> cache;
///  Cache for the backend buffers.
        inline thread_local static std::map<size_t,
                                            backend::buffer<T>> backend_cache;
        
///  Type def to retrieve the backend type.
        typedef T base;
    };

///  Convenience type alias for shared leaf nodes.
    template<typename T, bool SAFE_MATH=false>
    using shared_leaf = std::shared_ptr<leaf_node<T, SAFE_MATH>>;
///  Convenience type alias for a vector of output nodes.
    template<typename T, bool SAFE_MATH=false>
    using output_nodes = std::vector<shared_leaf<T, SAFE_MATH>>;

///  Forward declare for zero.
    template<typename T, bool SAFE_MATH=false>
    constexpr shared_leaf<T, SAFE_MATH> zero();
///  Forward declare for one.
    template<typename T, bool SAFE_MATH=false>
    constexpr shared_leaf<T, SAFE_MATH> one();

//******************************************************************************
//  Constant node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing data that cannot change.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    class constant_node final : public leaf_node<T, SAFE_MATH> {
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @params[in] d Scalar data to initalize.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(const T d) {
            return jit::format_to_string<T> (d);
        }
        
    private:
///  Storage buffer for the data.
        const backend::buffer<T> data;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a constant node from a vector.
///
///  @params[in] d Array buffer.
//------------------------------------------------------------------------------
        constant_node(const backend::buffer<T> &d) :
        leaf_node<T, SAFE_MATH> (constant_node::to_string(d.at(0)), 1), data(d) {
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
        virtual shared_leaf<T, SAFE_MATH> reduce() {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> df(shared_leaf<T, SAFE_MATH> x) {
            return zero<T, SAFE_MATH> ();
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                const T temp = this->evaluate().at(0);

                stream << " " << registers[this] << " = ";
                if constexpr (jit::is_complex<T> ()) {
                    jit::add_type<T> (stream);
                }
                stream << temp << ";" << std::endl;
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @params[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T, SAFE_MATH> x) {
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

//------------------------------------------------------------------------------
///  @brief Test if node acts like a constant.
///
///  @returns True if the node acts like a constant.
//------------------------------------------------------------------------------
        virtual bool is_constant_like() const {
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a variable.
///
///  @returns True if the node acts like a variable.
//------------------------------------------------------------------------------
        virtual bool is_all_variables() const {
            return false;
        }

//------------------------------------------------------------------------------
///  @brief Test if the node acts like a power of variable.
///
///  @returns True.
//------------------------------------------------------------------------------
        virtual bool is_power_like() const {
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Get the base of a power.
///
///  @returns The base of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_base() {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Get the exponent of a power.
///
///  @returns The exponent of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_exponent() const {
            return one<T, SAFE_MATH> ();
        }
    };

//------------------------------------------------------------------------------
///  @brief Construct a constant.
///
///  @params[in] d Array buffer.
///  @returns A reduced constant node.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> constant(const backend::buffer<T> &d) {
        auto temp = std::make_shared<constant_node<T, SAFE_MATH>> (d);
//  Test for hash collisions.
        for (size_t i = temp->get_hash(); i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::cache.find(i) ==
                leaf_node<T, SAFE_MATH>::cache.end()) {
                leaf_node<T, SAFE_MATH>::cache[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::cache[i])) {
                return leaf_node<T, SAFE_MATH>::cache[i];
            }
        }
        assert(false && "Should never reach.");
    }

//------------------------------------------------------------------------------
///  @brief Construct a constant.
///
///  @params[in] d Scalar data to initalize.
///  @returns A reduced constant node.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> constant(const T d) {
        return constant<T, SAFE_MATH> (backend::buffer<T> (1, d));
    }

//  Define some common constants.
//------------------------------------------------------------------------------
///  @brief Create a zero constant.
///
///  @returns A zero constant.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH>
    constexpr shared_leaf<T, SAFE_MATH> zero() {
        return constant<T, SAFE_MATH> (static_cast<T> (0.0));
    }

//------------------------------------------------------------------------------
///  @brief Create a one constant.
///
///  @returns A one constant.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH>
    constexpr shared_leaf<T, SAFE_MATH> one() {
        return constant<T, SAFE_MATH> (static_cast<T> (1.0));
    }

//------------------------------------------------------------------------------
///  @brief Create a negative one constant.
///
///  @returns A negative one constant.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
     constexpr shared_leaf<T, SAFE_MATH> none() {
        return constant<T, SAFE_MATH> (static_cast<T> (-1.0));
    }

//------------------------------------------------------------------------------
///  @brief Create a two constant.
///
///  @returns A two constant.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    constexpr shared_leaf<T, SAFE_MATH> two() {
        return constant<T, SAFE_MATH> (static_cast<T> (2.0));
    }

//------------------------------------------------------------------------------
///  @brief Create a two constant.
///
///  @returns A two constant.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    constexpr shared_leaf<T, SAFE_MATH> pi() {
        return constant<T, SAFE_MATH> (static_cast<T> (M_PI));
    }

//------------------------------------------------------------------------------
///  @brief Create a two constant.
///
///  @returns A two constant.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    constexpr shared_leaf<T, SAFE_MATH> half() {
        return constant<T, SAFE_MATH> (static_cast<T> (0.5));
    }

//------------------------------------------------------------------------------
///  @brief Create a machine epsilon constant.
///
///  @returns A two constant.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    constexpr shared_leaf<T, SAFE_MATH> epsilon() {
        return constant(std::numeric_limits<T>::epsilon());
    }

//------------------------------------------------------------------------------
/// @brief Create an imaginary constant.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    constexpr shared_leaf<T, SAFE_MATH> i() {
        static_assert(jit::is_complex<T> (),
                      "Imaginary only valid for complex base types.");
        return constant<T, SAFE_MATH> (T(0.0, 1.0));
    }

///  Convenience type alias for shared constant nodes.
    template<typename T, bool SAFE_MATH=false>
    using shared_constant = std::shared_ptr<constant_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a constant node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    shared_constant<T, SAFE_MATH> constant_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<constant_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Base straight node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a straight node.
///
///  This ensures that the base leaf type has the common type between the two
///  template arguments.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    class straight_node : public leaf_node<T, SAFE_MATH> {
    protected:
///  Argument
        shared_leaf<T, SAFE_MATH> arg;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a straight node.
///
///  @params[in] a Argument.
///  @params[in] s Node string to hash.
//------------------------------------------------------------------------------
        straight_node(shared_leaf<T, SAFE_MATH> a,
                      const std::string s) :
        leaf_node<T, SAFE_MATH> (s, a->get_complexity() + 1), arg(a) {}

//------------------------------------------------------------------------------
///  @brief Construct a straight node with defered argument.
///
///  @params[in] s Node string to hash.
//------------------------------------------------------------------------------
        straight_node(const std::string s) :
        leaf_node<T, SAFE_MATH> (s) {}

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
        virtual void compile_preamble(std::ostringstream &stream,
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
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers) {
            return this->arg->compile(stream, registers);
        }

//------------------------------------------------------------------------------
///  @brief Get the argument.
//------------------------------------------------------------------------------
        shared_leaf<T, SAFE_MATH> get_arg() {
            return this->arg;
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a constant.
///
///  @returns True if the node acts like a constant.
//------------------------------------------------------------------------------
        virtual bool is_constant_like() const {
            return this->arg->is_constant_like();
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a variable.
///
///  @returns True if the node acts like a variable.
//------------------------------------------------------------------------------
        virtual bool is_all_variables() const {
            return this->arg->is_all_variables();
        }

//------------------------------------------------------------------------------
///  @brief Get the exponent of a power.
///
///  @returns Returns a power of one.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_exponent() const {
            return one<T, SAFE_MATH> ();
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
    template<typename T, bool SAFE_MATH=false>
    class branch_node : public leaf_node<T, SAFE_MATH> {
    protected:
//  Left branch of the tree.
        shared_leaf<T, SAFE_MATH> left;
//  Right branch of the tree.
        shared_leaf<T, SAFE_MATH> right;

    public:

//------------------------------------------------------------------------------
///  @brief Assigns the left and right branches.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
///  @params[in] s Node string to hash.
//------------------------------------------------------------------------------
        branch_node(shared_leaf<T, SAFE_MATH> l,
                    shared_leaf<T, SAFE_MATH> r,
                    const std::string s) :
        leaf_node<T, SAFE_MATH> (s, l->get_complexity() + r->get_complexity() + 1),
        left(l), right(r) {}

//------------------------------------------------------------------------------
///  @brief Assigns the left and right branches.
///
///  @params[in] l     Left branch.
///  @params[in] r     Right branch.
///  @params[in] s     Node string to hash.
///  @params[in] count Number of nodes in the subgraph.
//------------------------------------------------------------------------------
                branch_node(shared_leaf<T, SAFE_MATH> l,
                            shared_leaf<T, SAFE_MATH> r,
                            const std::string s,
                            const size_t count) :
                leaf_node<T, SAFE_MATH> (s, count),
                left(l), right(r) {}

//------------------------------------------------------------------------------
///  @brief Compile preamble.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @params[in,out] visited   List of visited nodes.
//------------------------------------------------------------------------------
        virtual void compile_preamble(std::ostringstream &stream,
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
        shared_leaf<T, SAFE_MATH> get_left() {
            return this->left;
        }

//------------------------------------------------------------------------------
///  @brief Get the right branch.
//------------------------------------------------------------------------------
        shared_leaf<T, SAFE_MATH> get_right() {
            return this->right;
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a constant.
///
///  @returns True if the node acts like a constant.
//------------------------------------------------------------------------------
        virtual bool is_constant_like() const {
            return this->left->is_constant_like() &&
                   this->right->is_constant_like();
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a variable.
///
///  @returns True if the node acts like a variable.
//------------------------------------------------------------------------------
        virtual bool is_all_variables() const {
            return this->left->is_all_variables() &&
                   this->right->is_all_variables();
        }

//------------------------------------------------------------------------------
///  @brief Get the exponent of a power.
///
///  @returns Returns a power of one.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<T, SAFE_MATH>>
        get_power_exponent() const {
            return one<T, SAFE_MATH> ();
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
    template<typename T, bool SAFE_MATH=false>
    class triple_node : public branch_node<T, SAFE_MATH> {
    protected:
//  Middle branch of the tree.
        shared_leaf<T, SAFE_MATH> middle;

    public:

//------------------------------------------------------------------------------
///  @brief Reduces and assigns the left and right branches.
///
///  @params[in] l Left branch.
///  @params[in] m Middle branch.
///  @params[in] r Right branch.
///  @params[in] s Node string to hash.
//------------------------------------------------------------------------------
        triple_node(shared_leaf<T, SAFE_MATH> l,
                    shared_leaf<T, SAFE_MATH> m,
                    shared_leaf<T, SAFE_MATH> r,
                    const std::string s) :
        branch_node<T, SAFE_MATH> (l, r, s,
                                   l->get_complexity() +
                                   m->get_complexity() +
                                   r->get_complexity()),
        middle(m) {}

//------------------------------------------------------------------------------
///  @brief Compile preamble.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @params[in,out] visited   List of visited nodes.
//------------------------------------------------------------------------------
        virtual void compile_preamble(std::ostringstream &stream,
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
        shared_leaf<T, SAFE_MATH> get_middle() {
            return this->middle;
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a constant.
///
///  @returns True if the node acts like a constant.
//------------------------------------------------------------------------------
        virtual bool is_constant_like() const {
            return this->left->is_constant_like()   &&
                   this->middle->is_constant_like() &&
                   this->right->is_constant_like();
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a variable.
///
///  @returns True if the node acts like a variable.
//------------------------------------------------------------------------------
        virtual bool is_all_variables() const {
            return this->left->is_all_variables()   &&
                   this->middle->is_all_variables() &&
                   this->right->is_all_variables();
        }
    };

//******************************************************************************
//  Variable node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing data that can change.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    class variable_node final : public leaf_node<T, SAFE_MATH> {
    private:
///  Storage buffer for the data.
        backend::buffer<T> buffer;
///  Latex Symbol for the variable when pretty printing.
        const std::string symbol;

//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @params[in] p Pointer to the node.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(variable_node<T, SAFE_MATH> *p) {
            return jit::format_to_string(reinterpret_cast<size_t> (p));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a variable node with a size.
///
///  @params[in] s      Size of the data buffer.
///  @params[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
        variable_node(const size_t s,
                      const std::string &symbol) :
        leaf_node<T, SAFE_MATH> (variable_node::to_string(this), 1),
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
        leaf_node<T, SAFE_MATH> (variable_node::to_string(this), 1),
        buffer(s, d), symbol(symbol) {}

//------------------------------------------------------------------------------
///  @brief Construct a variable node from a vector.
///
///  @params[in] d      Array buffer.
///  @params[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
        variable_node(const std::vector<T> &d,
                      const std::string &symbol) :
        leaf_node<T, SAFE_MATH> (variable_node::to_string(this), 1),
        buffer(d), symbol(symbol) {}

//------------------------------------------------------------------------------
///  @brief Construct a variable node from backend buffer.
///
///  @params[in] d      Backend buffer.
///  @params[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
        variable_node(const backend::buffer<T> &d,
                      const std::string &symbol) :
        leaf_node<T, SAFE_MATH> (variable_node::to_string(this), 1),
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
        virtual shared_leaf<T, SAFE_MATH> reduce() {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> df(shared_leaf<T, SAFE_MATH> x) {
            return constant<T, SAFE_MATH> (static_cast<T> (this->is_match(x)));
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers) {
           return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @params[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T, SAFE_MATH> x) {
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
    
//------------------------------------------------------------------------------
///  @brief Test if node acts like a constant.
///
///  @returns True if the node acts like a constant.
//------------------------------------------------------------------------------
        virtual bool is_constant_like() const {
            return false;
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a variable.
///
///  @returns True if the node acts like a variable.
//------------------------------------------------------------------------------
        virtual bool is_all_variables() const {
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Test if the node acts like a power of variable.
///
///  @returns True.
//------------------------------------------------------------------------------
        virtual bool is_power_like() const {
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Get the base of a power.
///
///  @returns The base of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_base() {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Get the exponent of a power.
///
///  @returns The exponent of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_exponent() const {
            return one<T, SAFE_MATH> ();
        }
    };

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @params[in] s      Size of the data buffer.
///  @params[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> variable(const size_t s,
                                       const std::string &symbol) {
        return (std::make_shared<variable_node<T, SAFE_MATH>> (s, symbol))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @params[in] s      Size of he data buffer.
///  @params[in] d      Scalar data to initalize.
///  @params[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> variable(const size_t s, const T d,
                                       const std::string &symbol) {
        return (std::make_shared<variable_node<T, SAFE_MATH>> (s, d, symbol))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @params[in] d      Array buffer.
///  @params[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> variable(const std::vector<T> &d,
                                       const std::string &symbol) {
        return (std::make_shared<variable_node<T, SAFE_MATH>> (d, symbol))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @params[in] d      Array buffer.
///  @params[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> variable(const backend::buffer<T> &d,
                                       const std::string &symbol) {
        return std::make_shared<variable_node<T, SAFE_MATH>> (d, symbol)->reduce();
    }

///  Convenience type alias for shared variable nodes.
    template<typename T, bool SAFE_MATH=false>
    using shared_variable = std::shared_ptr<variable_node<T, SAFE_MATH>>;
///  Convenience type alias for a vector of inputs.
    template<typename T, bool SAFE_MATH=false>
    using input_nodes = std::vector<shared_variable<T, SAFE_MATH>>;
///  Convenience type alias for maping end codes back to inputs.
    template<typename T, bool SAFE_MATH=false>
    using map_nodes = std::vector<std::pair<graph::shared_leaf<T, SAFE_MATH>,
                                            shared_variable<T, SAFE_MATH>>>;

//------------------------------------------------------------------------------
///  @brief Cast to a variable node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    shared_variable<T, SAFE_MATH> variable_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<variable_node<T, SAFE_MATH>> (x);
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
    template<typename T, bool SAFE_MATH=false>
    class pseudo_variable_node final : public straight_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @params[in] p Pointer to the node argument.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *p) {
            return jit::format_to_string(reinterpret_cast<size_t> (p));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a pseudo variable node.
///
///  @params[in] a Argument.
//------------------------------------------------------------------------------
        pseudo_variable_node(shared_leaf<T, SAFE_MATH> a) :
        straight_node<T, SAFE_MATH> (a, pseudo_variable_node::to_string(a.get())) {}

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  For basic nodes, there's nothing to reduce.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> df(shared_leaf<T, SAFE_MATH> x) {
            return constant<T, SAFE_MATH> (static_cast<T> (this == x.get()));
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @params[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T, SAFE_MATH> x) {
            return this == x.get();
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << "\\left(";
            this->arg->to_latex();
            std::cout << "\\right)";
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a constant.
///
///  @returns True if the node acts like a constant.
//------------------------------------------------------------------------------
        virtual bool is_constant_like() const {
            return false;
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a variable.
///
///  @returns True if the node acts like a variable.
//------------------------------------------------------------------------------
        virtual bool is_all_variables() const {
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Test if the node acts like a power of variable.
///
///  @returns True.
//------------------------------------------------------------------------------
        virtual bool is_power_like() const {
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Get the base of a power.
///
///  @returns The base of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_base() {
            return this->arg->get_power_base();
        }

//------------------------------------------------------------------------------
///  @brief Get the exponent of a power.
///
///  @returns The exponent of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_exponent() const {
            return this->arg->get_power_exponent();
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> remove_pseudo() {
            return this->arg;
        }
    };

//------------------------------------------------------------------------------
///  @brief Define pseudo variable convience function.
///
///  @params[in] x Argument.
///  @returns A reduced pseudo variable node.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> pseudo_variable(shared_leaf<T, SAFE_MATH> x) {
        return std::make_shared<pseudo_variable_node<T, SAFE_MATH>> (x)->reduce();
    }

///  Convenience type alias for shared pseudo variable nodes.
    template<typename T, bool SAFE_MATH=false>
    using shared_pseudo_variable = std::shared_ptr<pseudo_variable_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a pseudo variable node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    shared_pseudo_variable<T, SAFE_MATH> pseudo_variable_cast(shared_leaf<T, SAFE_MATH> &x) {
        return std::dynamic_pointer_cast<pseudo_variable_node<T, SAFE_MATH>> (x);
    }
}

#endif /* node_h */
