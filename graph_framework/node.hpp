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
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class leaf_node : public std::enable_shared_from_this<leaf_node<T, SAFE_MATH>> {
    protected:
///  Hash for node.
        const size_t hash;
///  Graph complexity.
        const size_t complexity;
///  Cache derivative terms.
        std::map<size_t, std::shared_ptr<leaf_node<T, SAFE_MATH>>> df_cache;
///  Node contains pseudo variables.
        const bool contains_pseudo;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a basic node.
///
///  @param[in] s      Node string to hash.
///  @param[in] count  Number of nodes in the subgraph.
///  @param[in] pseudo Node contains pseudo variable.
//------------------------------------------------------------------------------
        leaf_node(const std::string s,
                  const size_t count,
                  const bool pseudo) :
        hash(std::hash<std::string>{} (s)),
        complexity(count), contains_pseudo(pseudo) {}

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
///  @param[in] x The variable to take the derivative to.
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
///  @param[in,out] stream          String buffer stream.
///  @param[in,out] registers       List of defined registers.
///  @param[in,out] visited         List of visited nodes.
///  @param[in,out] usage           List of register usage count.
///  @param[in,out] textures1d      List of 1D textures.
///  @param[in,out] textures2d      List of 2D textures.
///  @param[in,out] avail_const_mem Available constant memory.
//------------------------------------------------------------------------------
        virtual void compile_preamble(std::ostringstream &stream,
                                      jit::register_map &registers,
                                      jit::visiter_map &visited,
                                      jit::register_usage &usage,
                                      jit::texture1d_list &textures1d,
                                      jit::texture2d_list &textures2d,
                                      int &avail_const_mem) {
#ifdef SHOW_USE_COUNT
            if (usage.find(this) == usage.end()) {
                usage[this] = 1;
            } else {
                ++usage[this];
            }
#endif
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @param[in,out] indices   List of defined indices.
///  @param[in]     usage     List of register usage count.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<T, SAFE_MATH>>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                jit::register_map &indices,
                const jit::register_usage &usage) = 0;

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(std::shared_ptr<leaf_node<T, SAFE_MATH>> x) = 0;

//------------------------------------------------------------------------------
///  @brief Check if the base of the powers match.
///
///  @param[in] x Other graph to check if the bases match.
///  @returns True if the powers of the nodes match.
//------------------------------------------------------------------------------
        bool is_power_base_match(std::shared_ptr<leaf_node<T, SAFE_MATH>> x) {
            return this->get_power_base()->is_match(x->get_power_base());
        }

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
        virtual void set(const backend::buffer<T> &d) {}

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const = 0;

//------------------------------------------------------------------------------
///  @brief Convert the node to vizgraph.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<T, SAFE_MATH>> to_vizgraph(std::stringstream &stream,
                                                                     jit::register_map &registers) = 0;

//------------------------------------------------------------------------------
///  @brief Test if node is a constant.
///
///  @returns True if the node is like a constant.
//------------------------------------------------------------------------------
        virtual bool is_constant() const {
            return false;
        }

//------------------------------------------------------------------------------
///  @brief Test the constant node has a zero.
///
///  @returns True the node has a zero constant value.
//------------------------------------------------------------------------------
        virtual bool has_constant_zero() const {
            return false;
        }

//------------------------------------------------------------------------------
///  @brief Test if the result is normal.
///
///  @returns True if the node is normal.
//------------------------------------------------------------------------------
        bool is_normal() {
            return this->evaluate().is_normal();
        }

//------------------------------------------------------------------------------
///  @brief Test if all the subnodes terminate in variables.
///
///  @returns True if all the subnodes terminate in variables.
//------------------------------------------------------------------------------
        virtual bool is_all_variables() const = 0;

//------------------------------------------------------------------------------
///  @brief Test if the node acts like a power of variable.
///
///  Most nodes are not so default to false.
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
///  @brief Query if the node contains pseudo variables.
///
///  @return True if the node contains pseudo variables.
//------------------------------------------------------------------------------
        virtual bool has_pseudo() const {
            return contains_pseudo;
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<T, SAFE_MATH>> remove_pseudo() {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief End a line in the kernel source.
///
///  @param[in,out] stream String buffer stream.
///  @param[in]     usage  List of register usage count.
//------------------------------------------------------------------------------
        virtual void endline(std::ostringstream &stream,
                             const jit::register_usage &usage)
#ifndef SHOW_USE_COUNT
                             const
#endif
                             final {
            stream << ";"
#ifdef SHOW_USE_COUNT
                   << " // used " << usage.at(this)
#endif
                   << std::endl;
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
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_leaf = std::shared_ptr<leaf_node<T, SAFE_MATH>>;
///  Convenience type alias for a vector of output nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using output_nodes = std::vector<shared_leaf<T, SAFE_MATH>>;

///  Forward declare for zero.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    constexpr shared_leaf<T, SAFE_MATH> zero();
///  Forward declare for one.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    constexpr shared_leaf<T, SAFE_MATH> one();

//------------------------------------------------------------------------------
///  @brief Build the vizgraph input.
///
///  @param[in] node      Node to build the graph of.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    void make_vizgraph(shared_leaf<T, SAFE_MATH> node) {
        std::stringstream stream;
        jit::register_map registers;
        stream << std::setprecision(jit::max_digits10<T> ());

        stream << "graph \"\" {" << std::endl;
        stream << "    node [fontname = \"Helvetica\", ordering = out]" << std::endl << std::endl;
        node->to_vizgraph(stream, registers);
        stream << "}" << std::endl;

        std::cout << stream.str() << std::endl;
    }

//******************************************************************************
//  Constant node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing data that cannot change.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class constant_node final : public leaf_node<T, SAFE_MATH> {
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] d Scalar data to initalize.
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
///  @param[in] d Array buffer.
//------------------------------------------------------------------------------
        constant_node(const backend::buffer<T> &d) :
        leaf_node<T, SAFE_MATH> (constant_node::to_string(d.at(0)), 1, false), data(d) {
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
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> df(shared_leaf<T, SAFE_MATH> x) {
            return zero<T, SAFE_MATH> ();
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @param[in,out] indices   List of defined indices.
///  @param[in]     usage     List of register usage count.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                jit::register_map &indices,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
#ifdef USE_CONSTANT_CACHE
                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                const T temp = this->evaluate().at(0);

                stream << " " << registers[this] << " = ";
                if constexpr (jit::is_complex<T> ()) {
                    jit::add_type<T> (stream);
                }
                stream << temp;
                this->endline(stream, usage);
#else
                if constexpr (jit::is_complex<T> ()) {
                    registers[this] = jit::get_type_string<T> () + "("
                                    + jit::format_to_string(this->evaluate().at(0))
                                    + ")";
                } else {
                    registers[this] = jit::format_to_string(this->evaluate().at(0));
                }
#endif
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
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
///  @brief Check if the value is an integer.
//------------------------------------------------------------------------------
        bool is_integer() {
            const auto temp = this->evaluate().at(0);
            return std::imag(temp) == 0 &&
                   fmod(std::real(temp), 1.0) == 0.0;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << data.at(0);
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to vizgraph.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> to_vizgraph(std::stringstream &stream,
                                                      jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                const std::string name = jit::to_string('r', this);
                registers[this] = name;
                stream << "    " << name
                       << " [label = \"" << this->evaluate().at(0)
                       << "\", shape = box, style = \"rounded,filled\", fillcolor = black, fontcolor = white];" << std::endl;
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Test if node is a constant.
///
///  @returns True if the is a constant.
//------------------------------------------------------------------------------
        virtual bool is_constant() const {
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Test the constant node has a zero.
///
///  @returns True the node has a zero constant value.
//------------------------------------------------------------------------------
        virtual bool has_constant_zero() const {
            return data.has_zero();
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
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] d Array buffer.
///  @returns A reduced constant node.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
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
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

//------------------------------------------------------------------------------
///  @brief Construct a constant.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] d Scalar data to initalize.
///  @returns A reduced constant node.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> constant(const T d) {
        return constant<T, SAFE_MATH> (backend::buffer<T> (1, d));
    }

//  Define some common constants.
//------------------------------------------------------------------------------
///  @brief Create a zero constant.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @returns A zero constant.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH>
    constexpr shared_leaf<T, SAFE_MATH> zero() {
        return constant<T, SAFE_MATH> (static_cast<T> (0.0));
    }

//------------------------------------------------------------------------------
///  @brief Create a one constant.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @returns A one constant.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH>
    constexpr shared_leaf<T, SAFE_MATH> one() {
        return constant<T, SAFE_MATH> (static_cast<T> (1.0));
    }

///  Convinece type for imaginary constant.
    template<jit::complex_scalar T>
    constexpr T i = T(0.0, 1.0);

///  Convenience type alias for shared constant nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_constant = std::shared_ptr<constant_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a constant node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_constant<T, SAFE_MATH> constant_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<constant_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Base straight node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a straight node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  This ensures that the base leaf type has the common type between the two
///  template arguments.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class straight_node : public leaf_node<T, SAFE_MATH> {
    protected:
///  Argument
        shared_leaf<T, SAFE_MATH> arg;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a straight node.
///
///  @param[in] a Argument.
///  @param[in] s Node string to hash.
//------------------------------------------------------------------------------
        straight_node(shared_leaf<T, SAFE_MATH> a,
                      const std::string s) :
        leaf_node<T, SAFE_MATH> (s, a->get_complexity() + 1, a->has_pseudo()),
        arg(a) {}

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
///  @param[in,out] stream          String buffer stream.
///  @param[in,out] registers       List of defined registers.
///  @param[in,out] visited         List of visited nodes.
///  @param[in,out] usage           List of register usage count.
///  @param[in,out] textures1d      List of 1D textures.
///  @param[in,out] textures2d      List of 2D textures.
///  @param[in,out] avail_const_mem Available constant memory.
//------------------------------------------------------------------------------
        virtual void compile_preamble(std::ostringstream &stream,
                                      jit::register_map &registers,
                                      jit::visiter_map &visited,
                                      jit::register_usage &usage,
                                      jit::texture1d_list &textures1d,
                                      jit::texture2d_list &textures2d,
                                      int &avail_const_mem) {
            if (visited.find(this) == visited.end()) {
                this->arg->compile_preamble(stream, registers,
                                            visited, usage,
                                            textures1d, textures2d,
                                            avail_const_mem);
                visited.insert(this);
#ifdef SHOW_USE_COUNT
                usage[this] = 1;
            } else {
                ++usage[this];
#endif
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @param[in,out] indices   List of defined indices.
///  @param[in]     usage     List of register usage count.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                jit::register_map &indices,
                const jit::register_usage &usage) {
            return this->arg->compile(stream, registers, indices, usage);
        }

//------------------------------------------------------------------------------
///  @brief Get the argument.
//------------------------------------------------------------------------------
        shared_leaf<T, SAFE_MATH> get_arg() {
            return this->arg;
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
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  This ensures that the base leaf type has the common type between the two
///  template arguments.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
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
///  @param[in] l Left branch.
///  @param[in] r Right branch.
///  @param[in] s Node string to hash.
//------------------------------------------------------------------------------
        branch_node(shared_leaf<T, SAFE_MATH> l,
                    shared_leaf<T, SAFE_MATH> r,
                    const std::string s) :
        leaf_node<T, SAFE_MATH> (s, l->get_complexity() + r->get_complexity() + 1,
                                 l->has_pseudo() || r->has_pseudo()),
        left(l), right(r) {}

//------------------------------------------------------------------------------
///  @brief Assigns the left and right branches.
///
///  @param[in] l     Left branch.
///  @param[in] r     Right branch.
///  @param[in] s     Node string to hash.
///  @param[in] count Number of nodes in the subgraph.
///  @param[in] pseudo Node contains pseudo variable.
//------------------------------------------------------------------------------
        branch_node(shared_leaf<T, SAFE_MATH> l,
                    shared_leaf<T, SAFE_MATH> r,
                    const std::string s,
                    const size_t count,
                    const bool pseudo) :
        leaf_node<T, SAFE_MATH> (s, count, pseudo),
        left(l), right(r) {}

//------------------------------------------------------------------------------
///  @brief Compile preamble.
///
///  @param[in,out] stream          String buffer stream.
///  @param[in,out] registers       List of defined registers.
///  @param[in,out] visited         List of visited nodes.
///  @param[in,out] usage           List of register usage count.
///  @param[in,out] textures1d      List of 1D textures.
///  @param[in,out] textures2d      List of 2D textures.
///  @param[in,out] avail_const_mem Available constant memory.
//------------------------------------------------------------------------------
        virtual void compile_preamble(std::ostringstream &stream,
                                      jit::register_map &registers,
                                      jit::visiter_map &visited,
                                      jit::register_usage &usage,
                                      jit::texture1d_list &textures1d,
                                      jit::texture2d_list &textures2d,
                                      int &avail_const_mem) {
            if (visited.find(this) == visited.end()) {
                this->left->compile_preamble(stream, registers, 
                                             visited, usage,
                                             textures1d, textures2d,
                                             avail_const_mem);
                this->right->compile_preamble(stream, registers,
                                              visited, usage,
                                              textures1d, textures2d,
                                              avail_const_mem);
                visited.insert(this);
#ifdef SHOW_USE_COUNT
                usage[this] = 1;
            } else {
                ++usage[this];
#endif
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
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  This ensures that the base leaf type has the common type between the two
///  template arguments.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class triple_node : public branch_node<T, SAFE_MATH> {
    protected:
//  Middle branch of the tree.
        shared_leaf<T, SAFE_MATH> middle;

    public:

//------------------------------------------------------------------------------
///  @brief Reduces and assigns the left and right branches.
///
///  @param[in] l Left branch.
///  @param[in] m Middle branch.
///  @param[in] r Right branch.
///  @param[in] s Node string to hash.
//------------------------------------------------------------------------------
        triple_node(shared_leaf<T, SAFE_MATH> l,
                    shared_leaf<T, SAFE_MATH> m,
                    shared_leaf<T, SAFE_MATH> r,
                    const std::string s) :
        branch_node<T, SAFE_MATH> (l, r, s,
                                   l->get_complexity() +
                                   m->get_complexity() +
                                   r->get_complexity(),
                                   l->has_pseudo() ||
                                   m->has_pseudo() ||
                                   r->has_pseudo()),
        middle(m) {}

//------------------------------------------------------------------------------
///  @brief Compile preamble.
///
///  @param[in,out] stream          String buffer stream.
///  @param[in,out] registers       List of defined registers.
///  @param[in,out] visited         List of visited nodes.
///  @param[in,out] usage           List of register usage count.
///  @param[in,out] textures1d      List of 1D textures.
///  @param[in,out] textures2d      List of 2D textures.
///  @param[in,out] avail_const_mem Available constant memory.
//------------------------------------------------------------------------------
        virtual void compile_preamble(std::ostringstream &stream,
                                      jit::register_map &registers,
                                      jit::visiter_map &visited,
                                      jit::register_usage &usage,
                                      jit::texture1d_list &textures1d,
                                      jit::texture2d_list &textures2d,
                                      int &avail_const_mem) {
            if (visited.find(this) == visited.end()) {
                this->left->compile_preamble(stream, registers, 
                                             visited, usage,
                                             textures1d, textures2d,
                                             avail_const_mem);
                this->middle->compile_preamble(stream, registers,
                                               visited, usage,
                                               textures1d, textures2d,
                                               avail_const_mem);
                this->right->compile_preamble(stream, registers,
                                              visited, usage,
                                              textures1d, textures2d,
                                              avail_const_mem);
                visited.insert(this);
#ifdef SHOW_USE_COUNT
                usage[this] = 1;
            } else {
                ++usage[this];
#endif
            }
        }

//------------------------------------------------------------------------------
///  @brief Get the right branch.
//------------------------------------------------------------------------------
        shared_leaf<T, SAFE_MATH> get_middle() {
            return this->middle;
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
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class variable_node final : public leaf_node<T, SAFE_MATH> {
    private:
///  Storage buffer for the data.
        backend::buffer<T> buffer;
///  Latex Symbol for the variable when pretty printing.
        const std::string symbol;

//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] p Pointer to the node.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(variable_node<T, SAFE_MATH> *p) {
            return jit::format_to_string(reinterpret_cast<size_t> (p));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a variable node with a size.
///
///  @param[in] s      Size of the data buffer.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
        variable_node(const size_t s,
                      const std::string &symbol) :
        leaf_node<T, SAFE_MATH> (variable_node::to_string(this), 1, false),
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
        leaf_node<T, SAFE_MATH> (variable_node::to_string(this), 1, false),
        buffer(s, d), symbol(symbol) {
            assert(buffer.is_normal() && "NaN or Inf value.");
        }

//------------------------------------------------------------------------------
///  @brief Construct a variable node from a vector.
///
///  @param[in] d      Array buffer.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
        variable_node(const std::vector<T> &d,
                      const std::string &symbol) :
        leaf_node<T, SAFE_MATH> (variable_node::to_string(this), 1, false),
        buffer(d), symbol(symbol) {
            assert(buffer.is_normal() && "NaN or Inf value.");
        }

//------------------------------------------------------------------------------
///  @brief Construct a variable node from backend buffer.
///
///  @param[in] d      Backend buffer.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
        variable_node(const backend::buffer<T> &d,
                      const std::string &symbol) :
        leaf_node<T, SAFE_MATH> (variable_node::to_string(this), 1, false),
        buffer(d), symbol(symbol) {
            assert(buffer.is_normal() && "NaN or Inf value.");
        }

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
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> df(shared_leaf<T, SAFE_MATH> x) {
            return constant<T, SAFE_MATH> (static_cast<T> (this->is_match(x)));
        }

//------------------------------------------------------------------------------
///  @brief Compile preamble.
///
///  Some nodes require additions to the preamble however most don't so define a
///  generic method that does nothing.
///
///  @param[in,out] stream          String buffer stream.
///  @param[in,out] registers       List of defined registers.
///  @param[in,out] visited         List of visited nodes.
///  @param[in,out] usage           List of register usage count.
///  @param[in,out] textures1d      List of 1D textures.
///  @param[in,out] textures2d      List of 2D textures.
///  @param[in,out] avail_const_mem Available constant memory.
//------------------------------------------------------------------------------
        virtual void compile_preamble(std::ostringstream &stream,
                                      jit::register_map &registers,
                                      jit::visiter_map &visited,
                                      jit::register_usage &usage,
                                      jit::texture1d_list &textures1d,
                                      jit::texture2d_list &textures2d,
                                      int &avail_const_mem) {
            if (usage.find(this) == usage.end()) {
                usage[this] = 1;
#ifdef SHOW_USE_COUNT
            } else {
                ++usage[this];
#endif
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @param[in,out] indices   List of defined indices.
///  @param[in]     usage     List of register usage count.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                jit::register_map &indices,
                const jit::register_usage &usage) {
           return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T, SAFE_MATH> x) {
            return this == x.get();
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const T d) {
            buffer.set(d);
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] index Index to place the value at.
///  @param[in] d     Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const size_t index, const T d) {
            buffer[index] = d;
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const std::vector<T> &d) {
            buffer.set(d);
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Vector data to set.
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
///  @brief Convert the node to vizgraph.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> to_vizgraph(std::stringstream &stream,
                                                      jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                const std::string name = jit::to_string('r', this);
                registers[this] = name;
                stream << "    " << name
                       << " [label = \"" << this->get_symbol()
                       << "\", shape = box];" << std::endl;
            }

            return this->shared_from_this();
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
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] s      Size of the data buffer.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> variable(const size_t s,
                                       const std::string &symbol) {
        return std::make_shared<variable_node<T, SAFE_MATH>> (s, symbol);
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] s      Size of he data buffer.
///  @param[in] d      Scalar data to initalize.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> variable(const size_t s, const T d,
                                       const std::string &symbol) {
        return std::make_shared<variable_node<T, SAFE_MATH>> (s, d, symbol);
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] d      Array buffer.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> variable(const std::vector<T> &d,
                                       const std::string &symbol) {
        return std::make_shared<variable_node<T, SAFE_MATH>> (d, symbol);
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] d      Array buffer.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> variable(const backend::buffer<T> &d,
                                       const std::string &symbol) {
        return std::make_shared<variable_node<T, SAFE_MATH>> (d, symbol);
    }

///  Convenience type alias for shared variable nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_variable = std::shared_ptr<variable_node<T, SAFE_MATH>>;
///  Convenience type alias for a vector of inputs.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using input_nodes = std::vector<shared_variable<T, SAFE_MATH>>;
///  Convenience type alias for maping end codes back to inputs.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using map_nodes = std::vector<std::pair<graph::shared_leaf<T, SAFE_MATH>,
                                            shared_variable<T, SAFE_MATH>>>;

//------------------------------------------------------------------------------
///  @brief Cast to a variable node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
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
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class pseudo_variable_node final : public straight_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] p Pointer to the node argument.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *p) {
            return jit::format_to_string(reinterpret_cast<size_t> (p));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a pseudo variable node.
///
///  @param[in] a Argument.
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
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> df(shared_leaf<T, SAFE_MATH> x) {
            return constant<T, SAFE_MATH> (static_cast<T> (this == x.get()));
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
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
///  @brief Query if the node contains pseudo variables.
///
///  @return True if the node contains pseudo variables.
//------------------------------------------------------------------------------
        virtual bool has_pseudo() const {
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> remove_pseudo() {
            return this->arg;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to vizgraph.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> to_vizgraph(std::stringstream &stream,
                                                      jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                const std::string name = jit::to_string('r', this);
                registers[this] = name;
                stream << "    " << name
                       << " [label = \"pseudo_variable\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto a = this->arg->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[a.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Define pseudo variable convience function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x Argument.
///  @returns A reduced pseudo variable node.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> pseudo_variable(shared_leaf<T, SAFE_MATH> x) {
        return std::make_shared<pseudo_variable_node<T, SAFE_MATH>> (x);
    }

///  Convenience type alias for shared pseudo variable nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_pseudo_variable = std::shared_ptr<pseudo_variable_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a pseudo variable node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_pseudo_variable<T, SAFE_MATH> pseudo_variable_cast(shared_leaf<T, SAFE_MATH> &x) {
        return std::dynamic_pointer_cast<pseudo_variable_node<T, SAFE_MATH>> (x);
    }
}

#endif /* node_h */
