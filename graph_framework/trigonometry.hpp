//------------------------------------------------------------------------------
///  @file trigonometry.hpp
///  @brief Trigonometry functions.
///
///  Defines trigonometry operations.
//------------------------------------------------------------------------------

#ifndef trigonometry_h
#define trigonometry_h

#include "node.hpp"

namespace graph {

//******************************************************************************
//  Sine node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a sine_node leaf.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class sine_node final : public straight_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @params[in] a Argument node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *a) {
            return "sin" + jit::format_to_string(reinterpret_cast<size_t> (a));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a sine\_node node.
///
///  @params[in] x Argument.
//------------------------------------------------------------------------------
        sine_node(shared_leaf<T, SAFE_MATH> x) :
        straight_node<T, SAFE_MATH> (x, sine_node::to_string(x.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of sine.
///
///  result = sin(x)
///
///  @returns The value of sin(x).
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> result = this->arg->evaluate();
            result.sin();
            return result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the sin(x).
///
///  @returns Reduced graph from sine.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
            if (constant_cast(this->arg).get()) {
                return constant<T, SAFE_MATH> (this->evaluate());
            }

            auto ap1 = piecewise_1D_cast(this->arg);
            if (ap1.get()) {
                return piecewise_1D(this->evaluate(),
                                    ap1->get_arg());
            }

            auto ap2 = piecewise_2D_cast(this->arg);
            if (ap2.get()) {
                return piecewise_2D(this->evaluate(),
                                    ap2->get_num_columns(),
                                    ap2->get_left(),
                                    ap2->get_right());
            }

//  Sin(ArcTan(x, y)) -> y/Sqrt(x^2 + y^2)
            auto temp = atan_cast(this->arg);
            if (temp.get()) {
                return temp->get_right() /
                       (sqrt(temp->get_left()*temp->get_left() +
                             temp->get_right()*temp->get_right()));
            }

//  Remove negative constants from the arguments.
            auto am = multiply_cast(this->arg);
            if (am.get()) {
                auto lc = constant_cast(am->get_left());
                if (lc.get() && lc->evaluate().is_negative()) {
                    return none<T, SAFE_MATH> ()*sin(none<T, SAFE_MATH> ()*this->arg);
                }
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d sin(a)/dx = cos(a)*da/dx
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        df(shared_leaf<T, SAFE_MATH> x) {
            if (this->is_match(x)) {
                return one<T, SAFE_MATH> ();
            }
            
            const size_t hash = reinterpret_cast<size_t> (x.get());
            if (this->df_cache.find(hash) == this->df_cache.end()) {
                this->df_cache[hash] = cos(this->arg)*this->arg->df(x);
            }
            return this->df_cache[hash];
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> compile(std::ostringstream &stream,
                                                  jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> a = this->arg->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = sin("
                       << registers[a.get()] << ");"
                       << std::endl;
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

            auto x_cast = sin_cast(x);
            if (x_cast.get()) {
                return this->arg->is_match(x_cast->get_arg());
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << "\\sin\\left(";
            this->arg->to_latex();
            std::cout << "\\right)";
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> remove_pseudo() {
            return sin(this->arg->remove_pseudo());
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to vizgraph.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> to_vizgraph(std::stringstream &stream,
                                                      jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                const std::string name = jit::to_string('r', this);
                registers[this] = name;
                stream << "    " << name
                       << " [label = \"sin\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto a = this->arg->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[a.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Define sine convience function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @params[in] x Argument.
///  @returns A reduced sin node.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> sin(shared_leaf<T, SAFE_MATH> x) {
        auto temp = std::make_shared<sine_node<T, SAFE_MATH>> (x)->reduce();
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

///  Convenience type alias for shared sine nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_sine = std::shared_ptr<sine_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a sine node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_sine<T, SAFE_MATH> sin_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<sine_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Cosine node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a cosine_node leaf.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class cosine_node final : public straight_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @params[in] a Argument node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *a) {
            return "cos" + jit::format_to_string(reinterpret_cast<size_t> (a));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a cosine_node node.
///
///  @params[in] x Argument.
//------------------------------------------------------------------------------
        cosine_node(shared_leaf<T, SAFE_MATH> x) :
        straight_node<T, SAFE_MATH> (x, cosine_node::to_string(x.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of cosine.
///
///  result = cos(x)
///
///  @returns The value of cos(x).
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> result = this->arg->evaluate();
            result.cos();
            return result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the cos(x).
///
///  @returns Reduced graph from cosine.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
            if (constant_cast(this->arg).get()) {
                return constant<T, SAFE_MATH> (this->evaluate());
            }

            auto ap1 = piecewise_1D_cast(this->arg);
            if (ap1.get()) {
                return piecewise_1D(this->evaluate(),
                                    ap1->get_arg());
            }

            auto ap2 = piecewise_2D_cast(this->arg);
            if (ap2.get()) {
                return piecewise_2D(this->evaluate(),
                                    ap2->get_num_columns(),
                                    ap2->get_left(),
                                    ap2->get_right());
            }

//  Cos(ArcTan(x, y)) -> x/Sqrt(x^2 + y^2)
            auto temp = atan_cast(this->arg);
            if (temp.get()) {
                return temp->get_left() /
                       (sqrt(temp->get_left()*temp->get_left() +
                             temp->get_right()*temp->get_right()));
            }

//  Remove negative constants from the arguments.
            auto am = multiply_cast(this->arg);
            if (am.get()) {
                auto lc = constant_cast(am->get_left());
                if (lc.get() && lc->evaluate().is_negative()) {
                    return cos(none<T, SAFE_MATH> ()*this->arg);
                }
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d sin(a)/dx = cos(a)*da/dx
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        df(shared_leaf<T, SAFE_MATH> x) {
            if (this->is_match(x)) {
                return one<T, SAFE_MATH> ();
            }

            const size_t hash = reinterpret_cast<size_t> (x.get());
            if (this->df_cache.find(hash) == this->df_cache.end()) {
                this->df_cache[hash] = none<T, SAFE_MATH> ()*sin(this->arg)*this->arg->df(x);
            }
            return this->df_cache[hash];
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
                shared_leaf<T, SAFE_MATH> a = this->arg->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = cos("
                       << registers[a.get()] << ");"
                       << std::endl;
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

            auto x_cast = cos_cast(x);
            if (x_cast.get()) {
                return this->arg->is_match(x_cast->get_arg());
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << "\\cos\\left(";
            this->arg->to_latex();
            std::cout << "\\right)";
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> remove_pseudo() {
            return cos(this->arg->remove_pseudo());
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to vizgraph.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> to_vizgraph(std::stringstream &stream,
                                                      jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                const std::string name = jit::to_string('r', this);
                registers[this] = name;
                stream << "    " << name
                       << " [label = \"cos\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto a = this->arg->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[a.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Define cosine convience function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @params[in] x Argument.
///  @returns A reduced cos node.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> cos(shared_leaf<T, SAFE_MATH> x) {
        auto temp = std::make_shared<cosine_node<T, SAFE_MATH>> (x)->reduce();
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

///  Convenience type alias for shared cosine nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_cosine = std::shared_ptr<cosine_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a cosine node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_cosine<T, SAFE_MATH> cos_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<cosine_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Tangent node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Define tangent convience function.
///
///  tan(x) = sin(x)/cos(x)
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @params[in] x Argument.
///  @returns A reduced tan node.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> tan(shared_leaf<T, SAFE_MATH> x) {
        return sin(x)/cos(x);
    }

//******************************************************************************
//  Arctangent node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a sine_node leaf.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class arctan_node final : public branch_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @params[in] l Left pointer.
///  @params[in] r Left pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
    static std::string to_string(leaf_node<T, SAFE_MATH> *l,
                                 leaf_node<T, SAFE_MATH> *r) {
        return "atan" + jit::format_to_string(reinterpret_cast<size_t> (l))
                      + jit::format_to_string(reinterpret_cast<size_t> (r));
    }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a arctan\_node node.
///
///  @params[in] x Argument.
///  @params[in] y Argument.
//------------------------------------------------------------------------------
        arctan_node(shared_leaf<T, SAFE_MATH> x,
                    shared_leaf<T, SAFE_MATH> y) :
        branch_node<T, SAFE_MATH> (x, y, arctan_node::to_string(x.get(), y.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of arctan.
///
///  result = atan2(y, x)
///
///  @returns The value of atan2(y, x).
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> left = this->left->evaluate();
            backend::buffer<T> right = this->right->evaluate();
            return backend::atan(left, right);
        }

//------------------------------------------------------------------------------
///  @brief Reduce a arctan node.
///
///  @returns A reduced arctan node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
            auto l = constant_cast(this->left);
            auto r = constant_cast(this->right);
            if (l.get() && r.get()) {
                return constant<T, SAFE_MATH> (this->evaluate());
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d atan(x,y)/dx = 1/(1 + (y/x)^2)*d (y/x)/dx
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        df(shared_leaf<T, SAFE_MATH> x) {
            auto one_constant = one<T, SAFE_MATH> ();
            if (this->is_match(x)) {
                return one_constant;
            }

            const size_t hash = reinterpret_cast<size_t> (x.get());
            if (this->df_cache.find(hash) == this->df_cache.end()) {
                auto z = this->right/this->left;
                this->df_cache[hash] = (one_constant/(one_constant + z*z))*z->df(x);
            }
            return this->df_cache[hash];
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
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream, registers);
                shared_leaf<T, SAFE_MATH> r = this->right->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                if constexpr (jit::is_complex<T> ()) {
                    stream << " " << registers[this] << " = atan("
                           << registers[r.get()] << "/"
                           << registers[l.get()];
                } else {
                    stream << " " << registers[this] << " = atan2("
                           << registers[r.get()] << ","
                           << registers[l.get()];
                }
                stream << ");" << std::endl;
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

            auto x_cast = atan_cast(x);
            if (x_cast.get()) {
                return this->left->is_match(x_cast->get_left()) &&
                       this->right->is_match(x_cast->get_right());
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << "atan\\left(";
            this->left->to_latex();
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
            return atan(this->left->remove_pseudo(),
                        this->right->remove_pseudo());
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to vizgraph.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> to_vizgraph(std::stringstream &stream,
                                                      jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                const std::string name = jit::to_string('r', this);
                registers[this] = name;
                stream << "    " << name
                       << " [label = \"atan\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto l = this->left->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[l.get()] << ";" << std::endl;
                auto r = this->right->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[r.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build arctan node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> atan(shared_leaf<T, SAFE_MATH> l,
                                   shared_leaf<T, SAFE_MATH> r) {
        auto temp = std::make_shared<arctan_node<T, SAFE_MATH>> (l, r)->reduce();
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

///  Convenience type alias for shared add nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_atan = std::shared_ptr<arctan_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a power node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_atan<T, SAFE_MATH> atan_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<arctan_node<T, SAFE_MATH>> (x);
    }
}

#endif /* trigonometry_h */
