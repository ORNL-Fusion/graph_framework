//------------------------------------------------------------------------------
///  @file logical.hpp
///  @brief Nodes for boolean logic.
///
///  Defines a tree of operations that allows automatic differentiation.
//------------------------------------------------------------------------------
#ifndef logical_h
#define logical_h

#include "node.hpp"

/// Name space for graph nodes.
namespace graph {
///  Convenience type for true constant.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    constexpr shared_leaf<T, SAFE_MATH> true_constant() {
        return one<T, SAFE_MATH> ();
    }

///  Convenience type for false constant.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    constexpr shared_leaf<T, SAFE_MATH> false_constant() {
        return zero<T, SAFE_MATH> ();
    }

//******************************************************************************
//  Not node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Not node.
///
///  Note use templates here to defer this so it can use the operator functions.
///
///  @tparam T         Base type of the operands.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    class not_node final : public no_derivative<T, SAFE_MATH, straight_node<T, SAFE_MATH>> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] arg Argument node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *arg) {
            return "!" + jit::format_to_string(reinterpret_cast<size_t> (arg));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct an not node.
///
///  @param[in] arg Node argument.
//------------------------------------------------------------------------------
        not_node(shared_leaf<T, SAFE_MATH> arg) :
        no_derivative<T, SAFE_MATH,
                      straight_node<T, SAFE_MATH>> (arg,
                                                    not_node::to_string(arg.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of equal.
///
///  result = !a
///
///  @returns The value of !a.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> arg = this->arg->evaluate();
            return !arg;
        }

//------------------------------------------------------------------------------
///  @brief Reduce an equal node.
///
///  @returns A reduced equal node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
//  Constant reductions.
            auto arg = constant_cast(this->arg);

            if (arg.get()) {
                return constant<T, SAFE_MATH> (this->evaluate());
            }

            auto equalc = equal_cast(this->arg);
            if (equalc.get()) {
                return equalc->get_left() != equalc->get_right();
            }

            auto nequalc = not_equal_cast(this->arg);
            if (nequalc.get()) {
                return nequalc->get_left() == nequalc->get_right();
            }

            auto ltc = less_than_cast(this->arg);
            if (ltc.get()) {
                return ltc->get_left() >= ltc->get_right();
            }

            auto gtc = greater_than_cast(this->arg);
            if (gtc.get()) {
                return gtc->get_left() <= gtc->get_right();
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                jit::register_map &indices,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> arg = this->arg->compile(stream,
                                                                   registers,
                                                                   indices,
                                                                   usage);

                registers[this] = jit::to_string('l', this);
                stream << "        const bool ";
                stream << registers[this] << " = !"
                       << registers[arg.get()];
                this->endline(stream, usage);
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            bool arg_brackets = add_cast(this->arg).get() ||
                                subtract_cast(this->arg).get();
            std::cout << "\neg";
            if (arg_brackets) {
                std::cout << "\\left(";
            }
            this->arg->to_latex();
            if (arg_brackets) {
                std::cout << "\\right)";
            }
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
                       << " [label = \"!\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto arg = this->arg->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[arg.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build not node from the argument leaves.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] arg Arguement
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> not_(shared_leaf<T, SAFE_MATH> arg) {
        auto temp = std::make_shared<not_node<T, SAFE_MATH>> (arg)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::caches.nodes.find(i) ==
                leaf_node<T, SAFE_MATH>::caches.nodes.end()) {
                leaf_node<T, SAFE_MATH>::caches.nodes[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::caches.nodes[i])) {
                return leaf_node<T, SAFE_MATH>::caches.nodes[i];
            }
        }
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

//------------------------------------------------------------------------------
///  @brief Build equal node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] arg Arguement
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator!(shared_leaf<T, SAFE_MATH> arg) {
        return not_<T, SAFE_MATH> (arg);
    }

///  Convenience type alias for shared equal nodes.
    template<std::floating_point T, bool SAFE_MATH=false>
    using shared_not = std::shared_ptr<not_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a equal node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attempted dynamic cast.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_not<T, SAFE_MATH> not_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<not_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Equal node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief An equal node.
///
///  Note use templates here to defer this so it can use the operator functions.
///
///  @tparam T         Base type of the operands.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class equal_node final : public no_derivative<T, SAFE_MATH, branch_node<T, SAFE_MATH>> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] l Left node pointer.
///  @param[in] r Right node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *l,
                                     leaf_node<T, SAFE_MATH> *r) {
            return jit::format_to_string(reinterpret_cast<size_t> (l)) + "==" +
                   jit::format_to_string(reinterpret_cast<size_t> (r));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct an equal node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        equal_node(shared_leaf<T, SAFE_MATH> l,
                   shared_leaf<T, SAFE_MATH> r) :
        no_derivative<T, SAFE_MATH,
                      branch_node<T, SAFE_MATH>> (l, r,
                                                  equal_node::to_string(l.get(),
                                                                        r.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of equal.
///
///  result = l == r
///
///  @returns The value of l == r.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> l_result = this->left->evaluate();
            backend::buffer<T> r_result = this->right->evaluate();
            return l_result == r_result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce an equal node.
///
///  @returns A reduced equal node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
//  Constant reductions.
            auto l = constant_cast(this->left);
            auto r = constant_cast(this->right);

            if (l.get() && r.get()) {
                return constant<T, SAFE_MATH> (this->evaluate());
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                jit::register_map &indices,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream,
                                                                  registers,
                                                                  indices,
                                                                  usage);
                shared_leaf<T, SAFE_MATH> r = this->right->compile(stream,
                                                                   registers,
                                                                   indices,
                                                                   usage);

                registers[this] = jit::to_string('l', this);
                stream << "        const bool ";
                stream << registers[this] << " = "
                       << registers[l.get()] << "=="
                       << registers[r.get()];
                this->endline(stream, usage);
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Query if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T, SAFE_MATH> x) {
            if (this == x.get()) {
                return true;
            }

            auto x_cast = equal_cast(x);
            if (x_cast.get()) {
//  equal is commutative.
                if ((this->left->is_match(x_cast->get_left()) &&
                     this->right->is_match(x_cast->get_right())) ||
                    (this->right->is_match(x_cast->get_left()) &&
                     this->left->is_match(x_cast->get_right()))) {
                    return true;
                }
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            bool l_brackets = add_cast(this->left).get() ||
                              subtract_cast(this->left).get();
            bool r_brackets = add_cast(this->right).get() ||
                              subtract_cast(this->right).get();
            if (l_brackets) {
                std::cout << "\\left(";
            }
            this->left->to_latex();
            if (l_brackets) {
                std::cout << "\\right)";
            }
            std::cout << "=";
            if (r_brackets) {
                std::cout << "\\left(";
            }
            this->right->to_latex();
            if (r_brackets) {
                std::cout << "\\right)";
            }
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
                       << " [label = \"==\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto l = this->left->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[l.get()] << ";" << std::endl;
                auto r = this->right->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[r.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build equal node from two leaves.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> equal(shared_leaf<T, SAFE_MATH> l,
                                    shared_leaf<T, SAFE_MATH> r) {
        auto temp = std::make_shared<equal_node<T, SAFE_MATH>> (l, r)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::caches.nodes.find(i) ==
                leaf_node<T, SAFE_MATH>::caches.nodes.end()) {
                leaf_node<T, SAFE_MATH>::caches.nodes[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::caches.nodes[i])) {
                return leaf_node<T, SAFE_MATH>::caches.nodes[i];
            }
        }
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

//------------------------------------------------------------------------------
///  @brief Build equal node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator==(shared_leaf<T, SAFE_MATH> l,
                                         shared_leaf<T, SAFE_MATH> r) {
        return equal<T, SAFE_MATH> (l, r);
    }

//------------------------------------------------------------------------------
///  @brief Build equal node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam L         Float type for the constant.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar L, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator==(const L l,
                                         shared_leaf<T, SAFE_MATH> r) {
        return equal<T, SAFE_MATH> (constant<T, SAFE_MATH> (static_cast<T> (l)), r);
    }

//------------------------------------------------------------------------------
///  @brief Build equal node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam R         Float type for the constant.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar R, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator==(shared_leaf<T, SAFE_MATH> l,
                                         const R r) {
        return equal<T, SAFE_MATH> (l, constant<T, SAFE_MATH> (static_cast<T> (r)));
    }

///  Convenience type alias for shared equal nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_equal = std::shared_ptr<equal_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a equal node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attempted dynamic cast.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_equal<T, SAFE_MATH> equal_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<equal_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Not equal node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A not equal node.
///
///  Note use templates here to defer this so it can use the operator functions.
///
///  @tparam T         Base type of the operands.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class not_equal_node final : public no_derivative<T, SAFE_MATH, branch_node<T, SAFE_MATH>> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] l Left node pointer.
///  @param[in] r Right node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *l,
                                     leaf_node<T, SAFE_MATH> *r) {
            return jit::format_to_string(reinterpret_cast<size_t> (l)) + "!=" +
            jit::format_to_string(reinterpret_cast<size_t> (r));
        }
        
    public:
//------------------------------------------------------------------------------
///  @brief Construct an not equal node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        not_equal_node(shared_leaf<T, SAFE_MATH> l,
                       shared_leaf<T, SAFE_MATH> r) :
        no_derivative<T, SAFE_MATH,
        branch_node<T, SAFE_MATH>> (l, r,
                                    not_equal_node::to_string(l.get(),
                                                              r.get())) {}
    
//------------------------------------------------------------------------------
///  @brief Evaluate the results of not equal.
///
///  result = l != r
///
///  @returns The value of l != r.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> l_result = this->left->evaluate();
            backend::buffer<T> r_result = this->right->evaluate();
            return l_result != r_result;
        }
    
//------------------------------------------------------------------------------
///  @brief Reduce an not equal node.
///
///  @returns A reduced not equal node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
//  Constant reductions.
            auto l = constant_cast(this->left);
            auto r = constant_cast(this->right);
            
            if (l.get() && r.get()) {
                return constant<T, SAFE_MATH> (this->evaluate());
            }
            
            return this->shared_from_this();
        }
    
//------------------------------------------------------------------------------
///  @brief Compile the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                jit::register_map &indices,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream,
                                                                  registers,
                                                                  indices,
                                                                  usage);
                shared_leaf<T, SAFE_MATH> r = this->right->compile(stream,
                                                                   registers,
                                                                   indices,
                                                                   usage);
                
                registers[this] = jit::to_string('l', this);
                stream << "        const bool ";
                stream << registers[this] << " = "
                << registers[l.get()] << "!="
                << registers[r.get()];
                this->endline(stream, usage);
            }
            
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Query if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T, SAFE_MATH> x) {
            if (this == x.get()) {
                return true;
            }

            auto x_cast = not_equal_cast(x);
            if (x_cast.get()) {
//  equal is commutative.
                if ((this->left->is_match(x_cast->get_left()) &&
                     this->right->is_match(x_cast->get_right())) ||
                    (this->right->is_match(x_cast->get_left()) &&
                     this->left->is_match(x_cast->get_right()))) {
                    return true;
                }
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            bool l_brackets = add_cast(this->left).get() ||
                              subtract_cast(this->left).get();
            bool r_brackets = add_cast(this->right).get() ||
                              subtract_cast(this->right).get();
            if (l_brackets) {
                std::cout << "\\left(";
            }
            this->left->to_latex();
            if (l_brackets) {
                std::cout << "\\right)";
            }
            std::cout << "\\ne";
            if (r_brackets) {
                std::cout << "\\left(";
            }
            this->right->to_latex();
            if (r_brackets) {
                std::cout << "\\right)";
            }
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
                       << " [label = \"!=\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto l = this->left->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[l.get()] << ";" << std::endl;
                auto r = this->right->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[r.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build note equal node from two leaves.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> not_equal(shared_leaf<T, SAFE_MATH> l,
                                        shared_leaf<T, SAFE_MATH> r) {
        auto temp = std::make_shared<not_equal_node<T, SAFE_MATH>> (l, r)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::caches.nodes.find(i) ==
                leaf_node<T, SAFE_MATH>::caches.nodes.end()) {
                leaf_node<T, SAFE_MATH>::caches.nodes[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::caches.nodes[i])) {
                return leaf_node<T, SAFE_MATH>::caches.nodes[i];
            }
        }
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

//------------------------------------------------------------------------------
///  @brief Build not equal node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator!=(shared_leaf<T, SAFE_MATH> l,
                                         shared_leaf<T, SAFE_MATH> r) {
        return not_equal<T, SAFE_MATH> (l, r);
    }

//------------------------------------------------------------------------------
///  @brief Build not equal node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam L         Float type for the constant.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar L, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator!=(const L l,
                                         shared_leaf<T, SAFE_MATH> r) {
        return not_equal<T, SAFE_MATH> (constant<T, SAFE_MATH> (static_cast<T> (l)), r);
    }

//------------------------------------------------------------------------------
///  @brief Build not equal node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam R         Float type for the constant.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar R, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator!=(shared_leaf<T, SAFE_MATH> l,
                                         const R r) {
        return not_equal<T, SAFE_MATH> (l, constant<T, SAFE_MATH> (static_cast<T> (r)));
    }

///  Convenience type alias for shared not equal nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_not_equal = std::shared_ptr<not_equal_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a equal node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attempted dynamic cast.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_not_equal<T, SAFE_MATH> not_equal_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<not_equal_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Greater than node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A greater than node.
///
///  Note use templates here to defer this so it can use the operator functions.
///
///  @tparam T         Base type of the operands.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    class greater_than_node final : public no_derivative<T, SAFE_MATH, branch_node<T, SAFE_MATH>> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] l Left node pointer.
///  @param[in] r Right node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *l,
                                     leaf_node<T, SAFE_MATH> *r) {
            return jit::format_to_string(reinterpret_cast<size_t> (l)) + ">" +
                   jit::format_to_string(reinterpret_cast<size_t> (r));
        }
        
    public:
//------------------------------------------------------------------------------
///  @brief Construct a greater than node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        greater_than_node(shared_leaf<T, SAFE_MATH> l,
                          shared_leaf<T, SAFE_MATH> r) :
        no_derivative<T, SAFE_MATH,
        branch_node<T, SAFE_MATH>> (l, r,
                                    greater_than_node::to_string(l.get(),
                                                                 r.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of greater than.
///
///  result = l > r
///
///  @returns The value of l > r.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> l_result = this->left->evaluate();
            backend::buffer<T> r_result = this->right->evaluate();
            return l_result > r_result;
        }
    
//------------------------------------------------------------------------------
///  @brief Reduce greater than node.
///
///  @returns A reduced not equal node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
//  Constant reductions.
            auto l = constant_cast(this->left);
            auto r = constant_cast(this->right);
            
            if (l.get() && r.get()) {
                return constant<T, SAFE_MATH> (this->evaluate());
            }
            
            return this->shared_from_this();
        }
    
//------------------------------------------------------------------------------
///  @brief Compile the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                jit::register_map &indices,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream,
                                                                  registers,
                                                                  indices,
                                                                  usage);
                shared_leaf<T, SAFE_MATH> r = this->right->compile(stream,
                                                                   registers,
                                                                   indices,
                                                                   usage);
                
                registers[this] = jit::to_string('l', this);
                stream << "        const bool ";
                stream << registers[this] << " = "
                << registers[l.get()] << ">"
                << registers[r.get()];
                this->endline(stream, usage);
            }
            
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            bool l_brackets = add_cast(this->left).get() ||
                              subtract_cast(this->left).get();
            bool r_brackets = add_cast(this->right).get() ||
                              subtract_cast(this->right).get();
            if (l_brackets) {
                std::cout << "\\left(";
            }
            this->left->to_latex();
            if (l_brackets) {
                std::cout << "\\right)";
            }
            std::cout << ">";
            if (r_brackets) {
                std::cout << "\\left(";
            }
            this->right->to_latex();
            if (r_brackets) {
                std::cout << "\\right)";
            }
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
                       << " [label = \">\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto l = this->left->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[l.get()] << ";" << std::endl;
                auto r = this->right->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[r.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build greater than node from two leaves.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> greater_than(shared_leaf<T, SAFE_MATH> l,
                                           shared_leaf<T, SAFE_MATH> r) {
        auto temp = std::make_shared<greater_than_node<T, SAFE_MATH>> (l, r)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::caches.nodes.find(i) ==
                leaf_node<T, SAFE_MATH>::caches.nodes.end()) {
                leaf_node<T, SAFE_MATH>::caches.nodes[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::caches.nodes[i])) {
                return leaf_node<T, SAFE_MATH>::caches.nodes[i];
            }
        }
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

//------------------------------------------------------------------------------
///  @brief Build greater than node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator>(shared_leaf<T, SAFE_MATH> l,
                                        shared_leaf<T, SAFE_MATH> r) {
        return greater_than<T, SAFE_MATH> (l, r);
    }

//------------------------------------------------------------------------------
///  @brief Build greater than node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam L         Float type for the constant.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, std::floating_point L, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator>(const L l,
                                        shared_leaf<T, SAFE_MATH> r) {
        return greater_than<T, SAFE_MATH> (constant<T, SAFE_MATH> (static_cast<T> (l)), r);
    }

//------------------------------------------------------------------------------
///  @brief Build greater than node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam R         Float type for the constant.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, std::floating_point R, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator>(shared_leaf<T, SAFE_MATH> l,
                                        const R r) {
        return greater_than<T, SAFE_MATH> (l, constant<T, SAFE_MATH> (static_cast<T> (r)));
    }

///  Convenience type alias for shared greater than nodes.
    template<std::floating_point T, bool SAFE_MATH=false>
    using shared_greater_than = std::shared_ptr<greater_than_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a greater than node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attempted dynamic cast.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_greater_than<T, SAFE_MATH> greater_than_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<greater_than_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Less than node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A less than node.
///
///  Note use templates here to defer this so it can use the operator functions.
///
///  @tparam T         Base type of the operands.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    class less_than_node final : public no_derivative<T, SAFE_MATH, branch_node<T, SAFE_MATH>> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] l Left node pointer.
///  @param[in] r Right node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *l,
                                     leaf_node<T, SAFE_MATH> *r) {
            return jit::format_to_string(reinterpret_cast<size_t> (l)) + "<" +
                   jit::format_to_string(reinterpret_cast<size_t> (r));
        }
        
    public:
//------------------------------------------------------------------------------
///  @brief Construct a less than node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        less_than_node(shared_leaf<T, SAFE_MATH> l,
                       shared_leaf<T, SAFE_MATH> r) :
        no_derivative<T, SAFE_MATH,
        branch_node<T, SAFE_MATH>> (l, r,
                                    less_than_node::to_string(l.get(),
                                                              r.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of less than.
///
///  result = l < r
///
///  @returns The value of l < r.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> l_result = this->left->evaluate();
            backend::buffer<T> r_result = this->right->evaluate();
            return l_result < r_result;
        }
    
//------------------------------------------------------------------------------
///  @brief Reduce less than node.
///
///  @returns A reduced less than node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
//  Constant reductions.
            auto l = constant_cast(this->left);
            auto r = constant_cast(this->right);
            
            if (l.get() && r.get()) {
                return constant<T, SAFE_MATH> (this->evaluate());
            }
            
            return this->shared_from_this();
        }
    
//------------------------------------------------------------------------------
///  @brief Compile the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                jit::register_map &indices,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream,
                                                                  registers,
                                                                  indices,
                                                                  usage);
                shared_leaf<T, SAFE_MATH> r = this->right->compile(stream,
                                                                   registers,
                                                                   indices,
                                                                   usage);
                
                registers[this] = jit::to_string('l', this);
                stream << "        const bool ";
                stream << registers[this] << " = "
                << registers[l.get()] << "<"
                << registers[r.get()];
                this->endline(stream, usage);
            }
            
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            bool l_brackets = add_cast(this->left).get() ||
                              subtract_cast(this->left).get();
            bool r_brackets = add_cast(this->right).get() ||
                              subtract_cast(this->right).get();
            if (l_brackets) {
                std::cout << "\\left(";
            }
            this->left->to_latex();
            if (l_brackets) {
                std::cout << "\\right)";
            }
            std::cout << "<";
            if (r_brackets) {
                std::cout << "\\left(";
            }
            this->right->to_latex();
            if (r_brackets) {
                std::cout << "\\right)";
            }
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
                       << " [label = \"<\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto l = this->left->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[l.get()] << ";" << std::endl;
                auto r = this->right->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[r.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build less than node from two leaves.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> less_than(shared_leaf<T, SAFE_MATH> l,
                                        shared_leaf<T, SAFE_MATH> r) {
        auto temp = std::make_shared<less_than_node<T, SAFE_MATH>> (l, r)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::caches.nodes.find(i) ==
                leaf_node<T, SAFE_MATH>::caches.nodes.end()) {
                leaf_node<T, SAFE_MATH>::caches.nodes[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::caches.nodes[i])) {
                return leaf_node<T, SAFE_MATH>::caches.nodes[i];
            }
        }
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

//------------------------------------------------------------------------------
///  @brief Build less than node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator<(shared_leaf<T, SAFE_MATH> l,
                                        shared_leaf<T, SAFE_MATH> r) {
        return less_than<T, SAFE_MATH> (l, r);
    }

//------------------------------------------------------------------------------
///  @brief Build less than node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam L         Float type for the constant.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, std::floating_point L, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator<(const L l,
                                        shared_leaf<T, SAFE_MATH> r) {
        return less_than<T, SAFE_MATH> (constant<T, SAFE_MATH> (static_cast<T> (l)), r);
    }

//------------------------------------------------------------------------------
///  @brief Build less than node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam R         Float type for the constant.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, std::floating_point R, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator<(shared_leaf<T, SAFE_MATH> l,
                                        const R r) {
        return less_than<T, SAFE_MATH> (l, constant<T, SAFE_MATH> (static_cast<T> (r)));
    }

///  Convenience type alias for shared less than nodes.
    template<std::floating_point T, bool SAFE_MATH=false>
    using shared_less_than = std::shared_ptr<less_than_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a less than node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attempted dynamic cast.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_less_than<T, SAFE_MATH> less_than_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<less_than_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Greater than equal node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A greater than equal node.
///
///  Note use templates here to defer this so it can use the operator functions.
///
///  @tparam T         Base type of the operands.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    class greater_than_equal_node final : public no_derivative<T, SAFE_MATH, branch_node<T, SAFE_MATH>> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] l Left node pointer.
///  @param[in] r Right node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *l,
                                     leaf_node<T, SAFE_MATH> *r) {
            return jit::format_to_string(reinterpret_cast<size_t> (l)) + ">=" +
                   jit::format_to_string(reinterpret_cast<size_t> (r));
        }
        
    public:
//------------------------------------------------------------------------------
///  @brief Construct a greater than equal node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        greater_than_equal_node(shared_leaf<T, SAFE_MATH> l,
                          shared_leaf<T, SAFE_MATH> r) :
        no_derivative<T, SAFE_MATH,
        branch_node<T, SAFE_MATH>> (l, r,
                                    greater_than_equal_node::to_string(l.get(),
                                                                       r.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of greater than equal.
///
///  result = l >= r
///
///  @returns The value of l >= r.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> l_result = this->left->evaluate();
            backend::buffer<T> r_result = this->right->evaluate();
            return l_result >= r_result;
        }
    
//------------------------------------------------------------------------------
///  @brief Reduce greater than equal node.
///
///  @returns A reduced greater than equal node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
//  Constant reductions.
            auto l = constant_cast(this->left);
            auto r = constant_cast(this->right);
            
            if (l.get() && r.get()) {
                return constant<T, SAFE_MATH> (this->evaluate());
            }
            
            return this->shared_from_this();
        }
    
//------------------------------------------------------------------------------
///  @brief Compile the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                jit::register_map &indices,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream,
                                                                  registers,
                                                                  indices,
                                                                  usage);
                shared_leaf<T, SAFE_MATH> r = this->right->compile(stream,
                                                                   registers,
                                                                   indices,
                                                                   usage);
                
                registers[this] = jit::to_string('l', this);
                stream << "        const bool ";
                stream << registers[this] << " = "
                << registers[l.get()] << ">="
                << registers[r.get()];
                this->endline(stream, usage);
            }
            
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            bool l_brackets = add_cast(this->left).get() ||
                              subtract_cast(this->left).get();
            bool r_brackets = add_cast(this->right).get() ||
                              subtract_cast(this->right).get();
            if (l_brackets) {
                std::cout << "\\left(";
            }
            this->left->to_latex();
            if (l_brackets) {
                std::cout << "\\right)";
            }
            std::cout << "\\ge";
            if (r_brackets) {
                std::cout << "\\left(";
            }
            this->right->to_latex();
            if (r_brackets) {
                std::cout << "\\right)";
            }
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
                       << " [label = \">=\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto l = this->left->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[l.get()] << ";" << std::endl;
                auto r = this->right->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[r.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build greater than equal node from two leaves.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> greater_than_equal(shared_leaf<T, SAFE_MATH> l,
                                                 shared_leaf<T, SAFE_MATH> r) {
        auto temp = std::make_shared<greater_than_equal_node<T, SAFE_MATH>> (l, r)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::caches.nodes.find(i) ==
                leaf_node<T, SAFE_MATH>::caches.nodes.end()) {
                leaf_node<T, SAFE_MATH>::caches.nodes[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::caches.nodes[i])) {
                return leaf_node<T, SAFE_MATH>::caches.nodes[i];
            }
        }
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

//------------------------------------------------------------------------------
///  @brief Build greater than equal node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator>=(shared_leaf<T, SAFE_MATH> l,
                                         shared_leaf<T, SAFE_MATH> r) {
        return greater_than_equal<T, SAFE_MATH> (l, r);
    }

//------------------------------------------------------------------------------
///  @brief Build greater than equal node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam L         Float type for the constant.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, std::floating_point L, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator>=(const L l,
                                         shared_leaf<T, SAFE_MATH> r) {
        return greater_than_equal<T, SAFE_MATH> (constant<T, SAFE_MATH> (static_cast<T> (l)), r);
    }

//------------------------------------------------------------------------------
///  @brief Build greater than equal node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam R         Float type for the constant.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, std::floating_point R, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator>=(shared_leaf<T, SAFE_MATH> l,
                                         const R r) {
        return greater_than_equal<T, SAFE_MATH> (l, constant<T, SAFE_MATH> (static_cast<T> (r)));
    }

///  Convenience type alias for shared greater than equal nodes.
    template<std::floating_point T, bool SAFE_MATH=false>
    using shared_greater_than_equal = std::shared_ptr<greater_than_equal_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a greater than equal node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attempted dynamic cast.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_greater_than_equal<T, SAFE_MATH> greater_than_equal_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<greater_than_equal_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Less than equal node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A less than equal node.
///
///  Note use templates here to defer this so it can use the operator functions.
///
///  @tparam T         Base type of the operands.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    class less_than_equal_node final : public no_derivative<T, SAFE_MATH, branch_node<T, SAFE_MATH>> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] l Left node pointer.
///  @param[in] r Right node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *l,
                                     leaf_node<T, SAFE_MATH> *r) {
            return jit::format_to_string(reinterpret_cast<size_t> (l)) + "<=" +
                   jit::format_to_string(reinterpret_cast<size_t> (r));
        }
        
    public:
//------------------------------------------------------------------------------
///  @brief Construct a less than equal node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        less_than_equal_node(shared_leaf<T, SAFE_MATH> l,
                             shared_leaf<T, SAFE_MATH> r) :
        no_derivative<T, SAFE_MATH,
        branch_node<T, SAFE_MATH>> (l, r,
                                    less_than_equal_node::to_string(l.get(),
                                                                    r.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of less than equal.
///
///  result = l <= r
///
///  @returns The value of l <= r.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> l_result = this->left->evaluate();
            backend::buffer<T> r_result = this->right->evaluate();
            return l_result <= r_result;
        }
    
//------------------------------------------------------------------------------
///  @brief Reduce less than equal node.
///
///  @returns A reduced less than equal node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
//  Constant reductions.
            auto l = constant_cast(this->left);
            auto r = constant_cast(this->right);
            
            if (l.get() && r.get()) {
                return constant<T, SAFE_MATH> (this->evaluate());
            }
            
            return this->shared_from_this();
        }
    
//------------------------------------------------------------------------------
///  @brief Compile the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                jit::register_map &indices,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream,
                                                                  registers,
                                                                  indices,
                                                                  usage);
                shared_leaf<T, SAFE_MATH> r = this->right->compile(stream,
                                                                   registers,
                                                                   indices,
                                                                   usage);
                
                registers[this] = jit::to_string('l', this);
                stream << "        const bool ";
                stream << registers[this] << " = "
                << registers[l.get()] << "<="
                << registers[r.get()];
                this->endline(stream, usage);
            }
            
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            bool l_brackets = add_cast(this->left).get() ||
                              subtract_cast(this->left).get();
            bool r_brackets = add_cast(this->right).get() ||
                              subtract_cast(this->right).get();
            if (l_brackets) {
                std::cout << "\\left(";
            }
            this->left->to_latex();
            if (l_brackets) {
                std::cout << "\\right)";
            }
            std::cout << "\\le";
            if (r_brackets) {
                std::cout << "\\left(";
            }
            this->right->to_latex();
            if (r_brackets) {
                std::cout << "\\right)";
            }
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
                       << " [label = \"<=\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto l = this->left->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[l.get()] << ";" << std::endl;
                auto r = this->right->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[r.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build less than equal node from two leaves.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> less_than_equal(shared_leaf<T, SAFE_MATH> l,
                                              shared_leaf<T, SAFE_MATH> r) {
        auto temp = std::make_shared<less_than_equal_node<T, SAFE_MATH>> (l, r)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::caches.nodes.find(i) ==
                leaf_node<T, SAFE_MATH>::caches.nodes.end()) {
                leaf_node<T, SAFE_MATH>::caches.nodes[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::caches.nodes[i])) {
                return leaf_node<T, SAFE_MATH>::caches.nodes[i];
            }
        }
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

//------------------------------------------------------------------------------
///  @brief Build less than equal node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator<=(shared_leaf<T, SAFE_MATH> l,
                                         shared_leaf<T, SAFE_MATH> r) {
        return less_than_equal<T, SAFE_MATH> (l, r);
    }

//------------------------------------------------------------------------------
///  @brief Build less than equal node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam L         Float type for the constant.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, std::floating_point L, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator<=(const L l,
                                         shared_leaf<T, SAFE_MATH> r) {
        return less_than_equal<T, SAFE_MATH> (constant<T, SAFE_MATH> (static_cast<T> (l)), r);
    }

//------------------------------------------------------------------------------
///  @brief Build less than equal node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam R         Float type for the constant.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, std::floating_point R, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator<=(shared_leaf<T, SAFE_MATH> l,
                                         const R r) {
        return less_than_equal<T, SAFE_MATH> (l, constant<T, SAFE_MATH> (static_cast<T> (r)));
    }

///  Convenience type alias for shared less than equal nodes.
    template<std::floating_point T, bool SAFE_MATH=false>
    using shared_less_than_equal = std::shared_ptr<less_than_equal_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a less than equal node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attempted dynamic cast.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_less_than_equal<T, SAFE_MATH> less_than_equal_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<less_than_equal_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  And node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A and node.
///
///  Note use templates here to defer this so it can use the operator functions.
///
///  @tparam T         Base type of the operands.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    class and_node final : public no_derivative<T, SAFE_MATH, branch_node<T, SAFE_MATH>> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] l Left node pointer.
///  @param[in] r Right node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *l,
                                     leaf_node<T, SAFE_MATH> *r) {
            return jit::format_to_string(reinterpret_cast<size_t> (l)) + "&&" +
                   jit::format_to_string(reinterpret_cast<size_t> (r));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct an and node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        and_node(shared_leaf<T, SAFE_MATH> l,
                 shared_leaf<T, SAFE_MATH> r) :
        no_derivative<T, SAFE_MATH,
        branch_node<T, SAFE_MATH>> (l, r,
                                    and_node::to_string(l.get(), r.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of less than equal.
///
///  result = l && r
///
///  @returns The value of l && r.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> l_result = this->left->evaluate();
            backend::buffer<T> r_result = this->right->evaluate();
            return l_result && r_result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce and node.
///
///  @returns A reduced less than equal node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
//  Constant reductions.
            auto l = constant_cast(this->left);
            auto r = constant_cast(this->right);

            if (l.get() && r.get()) {
                return constant<T, SAFE_MATH> (this->evaluate());
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                jit::register_map &indices,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream,
                                                                  registers,
                                                                  indices,
                                                                  usage);
                shared_leaf<T, SAFE_MATH> r = this->right->compile(stream,
                                                                   registers,
                                                                   indices,
                                                                    usage);
                        
                registers[this] = jit::to_string('l', this);
                stream << "        const bool ";
                stream << registers[this] << " = "
                       << registers[l.get()] << "&&"
                       << registers[r.get()];
                this->endline(stream, usage);
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Query if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T, SAFE_MATH> x) {
            if (this == x.get()) {
                return true;
            }

            auto x_cast = and_cast(x);
            if (x_cast.get()) {
//  and is commutative.
                if ((this->left->is_match(x_cast->get_left()) &&
                     this->right->is_match(x_cast->get_right())) ||
                    (this->right->is_match(x_cast->get_left()) &&
                     this->left->is_match(x_cast->get_right()))) {
                    return true;
                }
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            bool l_brackets = add_cast(this->left).get() ||
                              subtract_cast(this->left).get();
            bool r_brackets = add_cast(this->right).get() ||
                              subtract_cast(this->right).get();
            if (l_brackets) {
                std::cout << "\\left(";
            }
            this->left->to_latex();
            if (l_brackets) {
                std::cout << "\\right)";
            }
            std::cout << "\\land";
            if (r_brackets) {
                std::cout << "\\left(";
            }
            this->right->to_latex();
            if (r_brackets) {
                std::cout << "\\right)";
            }
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
                       << " [label = \"&&\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto l = this->left->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[l.get()] << ";" << std::endl;
                auto r = this->right->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[r.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build and node from two leaves.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> and_(shared_leaf<T, SAFE_MATH> l,
                                   shared_leaf<T, SAFE_MATH> r) {
        auto temp = std::make_shared<and_node<T, SAFE_MATH>> (l, r)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::caches.nodes.find(i) ==
                leaf_node<T, SAFE_MATH>::caches.nodes.end()) {
                leaf_node<T, SAFE_MATH>::caches.nodes[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::caches.nodes[i])) {
                return leaf_node<T, SAFE_MATH>::caches.nodes[i];
            }
        }
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

//------------------------------------------------------------------------------
///  @brief Build and node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator&&(shared_leaf<T, SAFE_MATH> l,
                                         shared_leaf<T, SAFE_MATH> r) {
        return and_<T, SAFE_MATH> (l, r);
    }

//------------------------------------------------------------------------------
///  @brief Build and node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator&&(const bool l,
                                         shared_leaf<T, SAFE_MATH> r) {
        return and_<T, SAFE_MATH> (constant<T, SAFE_MATH> (static_cast<T> (l)), r);
    }

//------------------------------------------------------------------------------
///  @brief Build and node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator&&(shared_leaf<T, SAFE_MATH> l,
                                         const bool r) {
        return and_<T, SAFE_MATH> (l, constant<T, SAFE_MATH> (static_cast<T> (r)));
    }

///  Convenience type alias for shared and nodes.
    template<std::floating_point T, bool SAFE_MATH=false>
    using shared_and = std::shared_ptr<and_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to an and node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attempted dynamic cast.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_and<T, SAFE_MATH> and_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<and_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Or node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A or node.
///
///  Note use templates here to defer this so it can use the operator functions.
///
///  @tparam T         Base type of the operands.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    class or_node final : public no_derivative<T, SAFE_MATH, branch_node<T, SAFE_MATH>> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] l Left node pointer.
///  @param[in] r Right node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *l,
                                     leaf_node<T, SAFE_MATH> *r) {
            return jit::format_to_string(reinterpret_cast<size_t> (l)) + "||" +
                   jit::format_to_string(reinterpret_cast<size_t> (r));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a or node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        or_node(shared_leaf<T, SAFE_MATH> l,
                shared_leaf<T, SAFE_MATH> r) :
        no_derivative<T, SAFE_MATH,
        branch_node<T, SAFE_MATH>> (l, r,
                                    or_node::to_string(l.get(), r.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of less than equal.
///
///  result = l || r
///
///  @returns The value of l || r.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> l_result = this->left->evaluate();
            backend::buffer<T> r_result = this->right->evaluate();
            return l_result || r_result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce or node.
///
///  @returns A reduced less than equal node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
//  Constant reductions.
            auto l = constant_cast(this->left);
            auto r = constant_cast(this->right);

            if (l.get() && r.get()) {
                return constant<T, SAFE_MATH> (this->evaluate());
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                jit::register_map &indices,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream,
                                                                  registers,
                                                                  indices,
                                                                  usage);
                shared_leaf<T, SAFE_MATH> r = this->right->compile(stream,
                                                                   registers,
                                                                   indices,
                                                                    usage);
                        
                registers[this] = jit::to_string('l', this);
                stream << "        const bool ";
                stream << registers[this] << " = "
                       << registers[l.get()] << "||"
                       << registers[r.get()];
                this->endline(stream, usage);
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Query if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T, SAFE_MATH> x) {
            if (this == x.get()) {
                return true;
            }

            auto x_cast = and_cast(x);
            if (x_cast.get()) {
//  or is commutative.
                if ((this->left->is_match(x_cast->get_left()) &&
                     this->right->is_match(x_cast->get_right())) ||
                    (this->right->is_match(x_cast->get_left()) &&
                     this->left->is_match(x_cast->get_right()))) {
                    return true;
                }
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            bool l_brackets = add_cast(this->left).get() ||
                              subtract_cast(this->left).get();
            bool r_brackets = add_cast(this->right).get() ||
                              subtract_cast(this->right).get();
            if (l_brackets) {
                std::cout << "\\left(";
            }
            this->left->to_latex();
            if (l_brackets) {
                std::cout << "\\right)";
            }
            std::cout << "\\lor";
            if (r_brackets) {
                std::cout << "\\left(";
            }
            this->right->to_latex();
            if (r_brackets) {
                std::cout << "\\right)";
            }
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
                       << " [label = \"||\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto l = this->left->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[l.get()] << ";" << std::endl;
                auto r = this->right->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[r.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build or node from two leaves.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> or_(shared_leaf<T, SAFE_MATH> l,
                                  shared_leaf<T, SAFE_MATH> r) {
        auto temp = std::make_shared<and_node<T, SAFE_MATH>> (l, r)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::caches.nodes.find(i) ==
                leaf_node<T, SAFE_MATH>::caches.nodes.end()) {
                leaf_node<T, SAFE_MATH>::caches.nodes[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::caches.nodes[i])) {
                return leaf_node<T, SAFE_MATH>::caches.nodes[i];
            }
        }
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

//------------------------------------------------------------------------------
///  @brief Build or node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator||(shared_leaf<T, SAFE_MATH> l,
                                         shared_leaf<T, SAFE_MATH> r) {
        return or_<T, SAFE_MATH> (l, r);
    }

//------------------------------------------------------------------------------
///  @brief Build or node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator||(const bool l,
                                         shared_leaf<T, SAFE_MATH> r) {
        return or_<T, SAFE_MATH> (constant<T, SAFE_MATH> (static_cast<T> (l)), r);
    }

//------------------------------------------------------------------------------
///  @brief Build or node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator||(shared_leaf<T, SAFE_MATH> l,
                                         const bool r) {
        return or_<T, SAFE_MATH> (l, constant<T, SAFE_MATH> (static_cast<T> (r)));
    }

///  Convenience type alias for shared or nodes.
    template<std::floating_point T, bool SAFE_MATH=false>
    using shared_or = std::shared_ptr<or_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a or node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attempted dynamic cast.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool SAFE_MATH=false>
    shared_or<T, SAFE_MATH> or_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<or_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  If node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief An If conditional node.
///
///  Note use templates here to defer this so it can use the operator functions.
///
///  @tparam T         Base type of the operands.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class if_node final : public triple_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] c Condition node.
///  @param[in] t True condition.
///  @param[in] f False condition.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *c,
                                     leaf_node<T, SAFE_MATH> *t,
                                     leaf_node<T, SAFE_MATH> *f) {
            return "if(" +
                   jit::format_to_string(reinterpret_cast<size_t> (c)) + "," +
                   jit::format_to_string(reinterpret_cast<size_t> (t)) + "," +
                   jit::format_to_string(reinterpret_cast<size_t> (f)) + ")";
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct an equal node.
///
///  @param[in] c Condition node.
///  @param[in] t True condition branch.
///  @param[in] f False condition branch.
//------------------------------------------------------------------------------
        if_node(shared_leaf<T, SAFE_MATH> c,
                shared_leaf<T, SAFE_MATH> t,
                shared_leaf<T, SAFE_MATH> f) :
        triple_node<T, SAFE_MATH> (c, t, f,
                                   if_node::to_string(c.get(),
                                                      t.get(),
                                                      f.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of if.
///
///  result = if(c, t, f)
///
///  @returns The value of if(c, t, f).
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> c_result = this->left->evaluate();
            backend::buffer<T> t_result = this->middle->evaluate();
            backend::buffer<T> f_result = this->right->evaluate();
            return c_result.if_(t_result, f_result);
        }

//------------------------------------------------------------------------------
///  @brief Reduce an if node.
///
///  @returns A reduced equal node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
//  Constant reductions.
            auto c = constant_cast(this->left);
            auto t = constant_cast(this->middle);
            auto f = constant_cast(this->right);

            if (c.get() && t.get() && f.get()) {
                return constant<T, SAFE_MATH> (this->evaluate());
            }

//  If(c, a, a) -> a
            if (this->middle->is_match(this->right)) {
                return this->middle;
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d if(c,t,f)/dx = if(c,dt/dx,df/dx)
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        df(shared_leaf<T, SAFE_MATH> x) {
            if (this->is_match(x)) {
                return one<T, SAFE_MATH> ();
            }

            const size_t hash = reinterpret_cast<size_t> (x.get());
            if (this->df_cache.find(hash) == this->df_cache.end()) {
                this->df_cache[hash] = if_<T, SAFE_MATH> (this->left,
                                                          this->middle->df(x),
                                                          this->right->df(x));
            }
            return this->df_cache[hash];
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                jit::register_map &indices,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> c = this->left->compile(stream,
                                                                  registers,
                                                                  indices,
                                                                  usage);
                registers[this] = jit::to_string('r', this);
                shared_leaf<T, SAFE_MATH> t = this->middle->compile(stream,
                                                                    registers,
                                                                    indices,
                                                                    usage);
                shared_leaf<T, SAFE_MATH> f = this->right->compile(stream,
                                                                   registers,
                                                                   indices,
                                                                   usage);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = "
                       << registers[c.get()] << " ? "
                       << registers[t.get()] << " : "
                       << registers[f.get()];
                this->endline(stream, usage);
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << "if\\left(";
            this->left->to_latex();
            std::cout << ",";
            this->middle->to_latex();
            std::cout << ",";
            this->right->to_latex();
            std::cout << "\\right)";
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> remove_pseudo() {
            if (this->has_pseudo()) {
                return if_<T, SAFE_MATH> (this->left,
                                          this->middle->remove_pseudo(),
                                          this->right->remove_pseudo());
            }
            return this->shared_from_this();
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
                       << " [label = \"if\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto c = this->left->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[c.get()] << ";" << std::endl;
                auto t = this->middle->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[t.get()] << ";" << std::endl;
                auto f = this->right->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[f.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build an if node from a condition and two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] c Condition branch.
///  @param[in] t True branch.
///  @param[in] f False branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> if_(shared_leaf<T, SAFE_MATH> c,
                                  shared_leaf<T, SAFE_MATH> t,
                                  shared_leaf<T, SAFE_MATH> f) {
        auto temp = std::make_shared<if_node<T, SAFE_MATH>> (c, t, f)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::caches.nodes.find(i) ==
                leaf_node<T, SAFE_MATH>::caches.nodes.end()) {
                leaf_node<T, SAFE_MATH>::caches.nodes[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::caches.nodes[i])) {
                return leaf_node<T, SAFE_MATH>::caches.nodes[i];
            }
        }
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

///  Convenience type alias for shared add nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_if = std::shared_ptr<if_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to an if node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attempted dynamic cast.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_if<T, SAFE_MATH> if_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<if_node<T, SAFE_MATH>> (x);
    }
}

#endif /* logical_h */
