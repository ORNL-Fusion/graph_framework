//------------------------------------------------------------------------------
///  @file math.hpp
///  @brief Defined basic math functions.
//------------------------------------------------------------------------------

#ifndef math_h
#define math_h

#include <cmath>

#include "node.hpp"

namespace graph {
//******************************************************************************
//  Sqrt node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A sqrt node.
///
///  Note use templates here to defer this so it can use the operator functions.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class sqrt_node final : public straight_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] a Argument node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *a) {
            return "sqrt" + jit::format_to_string(reinterpret_cast<size_t> (a));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a sqrt node.
///
///  @param[in] x Argument.
//------------------------------------------------------------------------------
        sqrt_node(shared_leaf<T, SAFE_MATH> x) :
        straight_node<T, SAFE_MATH> (x, sqrt_node::to_string(x.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of sqrt.
///
///  result = sqrt(x)
///
///  @returns The value of sqrt(x).
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> result = this->arg->evaluate();
            result.sqrt();
            return result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the sqrt(x).
///
///  @returns Reduced graph from sqrt.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
            auto ac = constant_cast(this->arg);
            
            if (ac.get()) {
                if (ac->is(0) || ac->is(1)) {
                    return this->arg;
                }
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

//  Handle casses like sqrt(a^b).
            auto ap = pow_cast(this->arg);
            if (ap.get()) {
                auto bc = constant_cast(ap->get_right());
                if ((bc.get() && !bc->is(2)) || !bc.get()) {
                    return pow(ap->get_left(),
                               ap->get_right()/2.0);
                }
            }

//  Handle casses like sqrt(c*x) where c is constant or cases like
//  sqrt((x^a)*y).
            auto am = multiply_cast(this->arg);
            if (am.get()) {
                if (pow_cast(am->get_left()).get()  ||
                    am->get_left()->is_constant()   ||
                    pow_cast(am->get_right()).get() ||
                    am->get_right()->is_constant()) {
                    return sqrt(am->get_left()) *
                           sqrt(am->get_right());
                }
            }

//  Handle casses like sqrt(x^a/b) and sqrt(a/x^b) or sqrt(c/b) and sqrt(a/c)
//  where c is a constant.
            auto ad = divide_cast(this->arg);
            if (ad.get()) {
                if (pow_cast(ad->get_left()).get()  ||
                    ad->get_left()->is_constant()   ||
                    pow_cast(ad->get_right()).get() ||
                    ad->get_right()->is_constant()) {
                    return sqrt(ad->get_left()) /
                           sqrt(ad->get_right());
                }
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d sqrt(a)/dx = 1/(2*sqrt(a))da/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> df(shared_leaf<T, SAFE_MATH> x) {
            if (this->is_match(x)) {
                return one<T, SAFE_MATH> ();
            }

            const size_t hash = reinterpret_cast<size_t> (x.get());
            if (this->df_cache.find(hash) == this->df_cache.end()) {
                this->df_cache[hash] = this->arg->df(x)
                                     / (2.0*this->shared_from_this());
            }
            return this->df_cache[hash];
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @param[in]     usage     List of register usage count.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> a = this->arg->compile(stream,
                                                                 registers,
                                                                 usage);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = sqrt("
                       << registers[a.get()] << ")";
                this->endline(stream, usage);
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

            auto x_cast = sqrt_cast(x);
            if (x_cast.get()) {
                return this->arg->is_match(x_cast->get_arg());
            } else {
                return false;
            }
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << "\\sqrt{";
            this->arg->to_latex();
            std::cout << "}";
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
            return this->arg;
        }

//------------------------------------------------------------------------------
///  @brief Get the exponent of a power.
///
///  @returns The exponent of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_exponent() const {
            return constant<T, SAFE_MATH> (static_cast<T> (0.5));
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> remove_pseudo() {
            if (this->has_pseudo()) {
                return sqrt(this->arg->remove_pseudo());
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
                       << " [label = \"sqrt\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto a = this->arg->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[a.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Define sqrt convience function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x Argument.
///  @returns A reduced sqrt node.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> sqrt(shared_leaf<T, SAFE_MATH> x) {
        auto temp = std::make_shared<sqrt_node<T, SAFE_MATH>> (x)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
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

///  Convenience type alias for shared sqrt nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_sqrt = std::shared_ptr<sqrt_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a sqrt node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_sqrt<T, SAFE_MATH> sqrt_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<sqrt_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Exp node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A exp node.
///
///  Note use templates here to defer this so it can use the operator functions.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class exp_node final : public straight_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] a Argument node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *a) {
            return "exp" + jit::format_to_string(reinterpret_cast<size_t> (a));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a exp node.
///
///  @param[in] x Argument.
//------------------------------------------------------------------------------
        exp_node(shared_leaf<T, SAFE_MATH> x) :
        straight_node<T, SAFE_MATH> (x, exp_node::to_string(x.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of exp.
///
///  result = exp(x)
///
///  @returns The value of exp(x).
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> result = this->arg->evaluate();
            result.exp();
            return result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the exp(x).
///
///  @returns Reduced graph from exp.
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

//  Reduce exp(log(x)) -> x
            auto a = log_cast(this->arg);
            if (a.get()) {
                return a->get_arg();
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d exp(a)/dx = exp(a)*da/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> df(shared_leaf<T, SAFE_MATH> x) {
            if (this->is_match(x)) {
                return one<T, SAFE_MATH> ();
            }

            const size_t hash = reinterpret_cast<size_t> (x.get());
            if (this->df_cache.find(hash) == this->df_cache.end()) {
                this->df_cache[hash] = this->shared_from_this()*this->arg->df(x);
            }
            return this->df_cache[hash];
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @param[in]     usage     List of register usage count.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> a = this->arg->compile(stream, 
                                                                 registers,
                                                                 usage);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = ";
                if constexpr (SAFE_MATH) {
                    if constexpr (jit::is_complex<T> ()) {
                        stream << "real(";
                    }
                    stream << registers[a.get()];
                    if constexpr (jit::is_complex<T> ()) {
                        stream << ")";
                    }
                    stream << " < 709.8 ? ";
                }
                stream << "exp("  << registers[a.get()] << ")";
                if constexpr (SAFE_MATH) {
                    stream << " : ";
                    if constexpr (jit::is_complex<T> ()) {
                        jit::add_type<T> (stream);
                        stream << "(";
                    }
                    if constexpr (jit::is_float<T> ()) {
                        stream << std::numeric_limits<float>::max();
                    } else {
                        stream << std::numeric_limits<double>::max();
                    }
                    if constexpr (jit::is_complex<T> ()) {
                        stream << ")";
                    }
                }
                stream << "";
                this->endline(stream, usage);
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

            auto x_cast = exp_cast(x);
            if (x_cast.get()) {
                return this->arg->is_match(x_cast->get_arg());
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << "e^{\\left(";
            this->arg->to_latex();
            std::cout << "\\right)}";
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> remove_pseudo() {
            if (this->has_pseudo()) {
                return exp(this->arg->remove_pseudo());
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
                       << " [label = \"exp\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto a = this->arg->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[a.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Define exp convience function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x Argument.
///  @returns A reduced exp node.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> exp(shared_leaf<T, SAFE_MATH> x) {
        auto temp = std::make_shared<exp_node<T, SAFE_MATH>> (x)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
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

///  Convenience type alias for shared exp nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_exp = std::shared_ptr<exp_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a exp node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_exp<T, SAFE_MATH> exp_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<exp_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Log node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A log node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  Note use templates here to defer this so it can use the operator functions.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class log_node final : public straight_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] a Argument node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *a) {
            return "log" + jit::format_to_string(reinterpret_cast<size_t> (a));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a log node.
///
///  @param[in] x Argument.
//------------------------------------------------------------------------------
        log_node(shared_leaf<T, SAFE_MATH> x) :
        straight_node<T, SAFE_MATH> (x, log_node::to_string(x.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of log.
///
///  result = log(x)
///
///  @returns The value of log(x).
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> result = this->arg->evaluate();
            result.log();
            return result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the log(x).
///
///  @returns Reduced graph from log.
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

//  Reduce log(exp(x)) -> x
            auto a = exp_cast(this->arg);
            if (a.get()) {
                return a->get_arg();
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d log(a)/dx = (da/dx)/a
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> df(shared_leaf<T, SAFE_MATH> x) {
            if (this->is_match(x)) {
                return one<T, SAFE_MATH> ();
            }

            const size_t hash = reinterpret_cast<size_t> (x.get());
            if (this->df_cache.find(hash) == this->df_cache.end()) {
                this->df_cache[hash] = this->arg->df(x)/this->arg;
            }
            return this->df_cache[hash];
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @param[in]     usage     List of register usage count.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> a = this->arg->compile(stream, 
                                                                 registers,
                                                                 usage);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = log("
                       << registers[a.get()] << ")";
                this->endline(stream, usage);
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

            auto x_cast = log_cast(x);
            if (x_cast.get()) {
                return this->arg->is_match(x_cast->get_arg());
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << "\\ln{\\left(";
            this->arg->to_latex();
            std::cout << "\\right)}";
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> remove_pseudo() {
            if (this->has_pseudo()) {
                return log(this->arg->remove_pseudo());
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
                       << " [label = \"log\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto a = this->arg->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[a.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Define log convience function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x Argument.
///  @returns A reduced log node.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> log(shared_leaf<T, SAFE_MATH> x) {
        auto temp = std::make_shared<log_node<T, SAFE_MATH>> (x)->reduce();
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

///  Convenience type alias for shared log nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_log = std::shared_ptr<log_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a exp node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_log<T, SAFE_MATH> log_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<log_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Pow node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief An power node.
///
///  Note use templates here to defer this so it can use the operator functions.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class pow_node final : public branch_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] l Argument node pointer.
///  @param[in] r Argument node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *l,
                                     leaf_node<T, SAFE_MATH> *r) {
            return "pow" + jit::format_to_string(reinterpret_cast<size_t> (l))
                         + jit::format_to_string(reinterpret_cast<size_t> (r));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct an power node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        pow_node(shared_leaf<T, SAFE_MATH> l,
                 shared_leaf<T, SAFE_MATH> r) :
        branch_node<T, SAFE_MATH> (l, r, pow_node::to_string(l.get(),
                                                             r.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of addition.
///
///  result = l^r
///
///  @returns The value of l^r.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> l_result = this->left->evaluate();
            backend::buffer<T> r_result = this->right->evaluate();
            return backend::pow(l_result, r_result);
        }

//------------------------------------------------------------------------------
///  @brief Reduce a power node.
///
///  @returns A reduced power node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
            auto lc = constant_cast(this->left);
            auto rc = constant_cast(this->right);

            if (rc.get() && rc->is(0)) {
                return one<T, SAFE_MATH> ();
            } else if (rc.get() && rc->is(1)) {
                return this->left;
            } else if (rc.get() && rc->is(0.5)) {
                return sqrt(this->left);
            } else if (rc.get() && rc->is(2)){
                auto sq = sqrt_cast(this->left);
                if (sq.get()) {
                    return sq->get_arg();
                }
            }

            if (lc.get() && rc.get()) {
                return constant<T, SAFE_MATH> (this->evaluate());
            }

            auto pl1 = piecewise_1D_cast(this->left);
            auto pr1 = piecewise_1D_cast(this->right);
            if (pl1.get() && (rc.get() || pl1->is_arg_match(this->right))) {
                return piecewise_1D(this->evaluate(), pl1->get_arg());
            } else if (pr1.get() && (lc.get() || pr1->is_arg_match(this->left))) {
                return piecewise_1D(this->evaluate(), pr1->get_arg());
            }
            
            auto pl2 = piecewise_2D_cast(this->left);
            auto pr2 = piecewise_2D_cast(this->right);
            if (pl2.get() && (rc.get() || pl2->is_arg_match(this->right))) {
                return piecewise_2D(this->evaluate(),
                                    pl2->get_num_columns(),
                                    pl2->get_left(),
                                    pl2->get_right());
            } else if (pr2.get() && (lc.get() || pr2->is_arg_match(this->left))) {
                return piecewise_2D(this->evaluate(),
                                    pr2->get_num_columns(),
                                    pr2->get_left(),
                                    pr2->get_right());
            }

//  Combine 2D and 1D piecewise constants if a row or column matches.
            if (pr2.get() && pr2->is_row_match(this->left)) {
                backend::buffer<T> result = pl1->evaluate();
                result.pow_row(pr2->evaluate());
                return piecewise_2D(result,
                                    pr2->get_num_columns(),
                                    pr2->get_left(),
                                    pr2->get_right());
            } else if (pr2.get() && pr2->is_col_match(this->left)) {
                backend::buffer<T> result = pl1->evaluate();
                result.pow_col(pr2->evaluate());
                return piecewise_2D(result,
                                    pr2->get_num_columns(),
                                    pr2->get_left(),
                                    pr2->get_right());
            } else if (pl2.get() && pl2->is_row_match(this->right)) {
                backend::buffer<T> result = pl2->evaluate();
                result.pow_row(pr1->evaluate());
                return piecewise_2D(result,
                                    pl2->get_num_columns(),
                                    pl2->get_left(),
                                    pl2->get_right());
            } else if (pl2.get() && pl2->is_col_match(this->right)) {
                backend::buffer<T> result = pl2->evaluate();
                result.pow_col(pr1->evaluate());
                return piecewise_2D(result,
                                    pl2->get_num_columns(),
                                    pl2->get_left(),
                                    pl2->get_right());
            }

            auto lp = pow_cast(this->left);
//  Only run this reduction if the right is an integer constant value.
            if (lp.get() && rc.get() && rc->is_integer()) {
                return pow(lp->get_left(), lp->get_right()*this->right);
            }

//  Handle cases where (c*x)^a, (x*c)^a, (a*sqrt(b))^c and (a*b^c)^2.
            auto lm = multiply_cast(this->left);
            if (lm.get()) {
                if (lm->get_left()->is_constant()    ||
                    lm->get_right()->is_constant()   ||
                    sqrt_cast(lm->get_left()).get()  ||
                    sqrt_cast(lm->get_right()).get() ||
                    pow_cast(lm->get_left()).get()   ||
                    pow_cast(lm->get_right()).get()) {
                    return pow(lm->get_left(), this->right) *
                           pow(lm->get_right(), this->right);
                }
            }

//  Handle cases where (c/x)^a, (x/c)^a, (a/sqrt(b))^c and (a/b^c)^2.
            auto ld = divide_cast(this->left);
            if (ld.get()) {
                if (ld->get_left()->is_constant()    ||
                    ld->get_right()->is_constant()   ||
                    sqrt_cast(ld->get_left()).get()  ||
                    sqrt_cast(ld->get_right()).get() ||
                    pow_cast(ld->get_left()).get()   ||
                    pow_cast(ld->get_right()).get()) {
                    return pow(ld->get_left(), this->right) /
                           pow(ld->get_right(), this->right);
                }

//  Handle cases where (a/(b*sqrt(c))), (a/(sqrt(c)*b)), (a/(b*c^d)), (a/(c^d*b))
                auto ldm = multiply_cast(ld->get_right());
                if (ldm.get()) {
                    if (ldm->get_left()->is_constant()    ||
                        ldm->get_right()->is_constant()   ||
                        sqrt_cast(ldm->get_left()).get()  ||
                        sqrt_cast(ldm->get_right()).get() ||
                        pow_cast(ldm->get_left()).get()   ||
                        pow_cast(ldm->get_right()).get()) {
                        return pow(ld->get_left(), this->right) /
                               (pow(ldm->get_left(), this->right) *
                                pow(ldm->get_right(), this->right));
                    }
                }
            }

//  Reduce sqrt(a)^b
            auto lsq = sqrt_cast(this->left);
            if (lsq.get()) {
                return pow(lsq->get_arg(),
                           this->right/2.0);
            }

//  Reduce exp(x)^n -> exp(n*x) when x is an integer.
            auto temp = exp_cast(this->left);
            if (temp.get() && rc.get() && rc->is_integer()) {
                return exp(this->right*temp->get_arg());
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d a^b dx = b*a^(b-1)*da/dx + ln(a)a^b*db/dx
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
                this->df_cache[hash] = pow(this->left, this->right - 1.0)
                                     * (this->right*this->left->df(x) +
                                        this->left*log(this->left)*this->right->df(x));
            }
            return this->df_cache[hash];
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @param[in]     usage     List of register usage count.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream, 
                                                                  registers,
                                                                  usage);
                shared_leaf<T, SAFE_MATH> r;
                auto temp = constant_cast(this->right);
                if (!temp.get() || !temp->is_integer()) {
                    r = this->right->compile(stream, registers, usage);
                }

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = ";
                if (temp.get() && temp->is_integer()) {
                    stream << registers[l.get()];
                    const size_t end = static_cast<size_t> (std::real(this->right->evaluate().at(0)));
                    for (size_t i = 1; i < end; i++) {
                        stream << "*" << registers[l.get()];
                    }
                } else {
                    stream << "pow("
                           << registers[l.get()] << ", "
                           << registers[r.get()] << ")";
                }
                this->endline(stream, usage);
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

            auto x_cast = pow_cast(x);
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
            auto use_brackets = !constant_cast(this->left).get() &&
                                !variable_cast(this->left).get();

            if (use_brackets) {
                std::cout << "\\left(";
            }
            this->left->to_latex();
            if (use_brackets) {
                std::cout << "\\right)";
            }
            std::cout << "^{";
            this->right->to_latex();
            std::cout << "}";
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
                       << " [label = \"pow\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto l = this->left->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[l.get()] << ";" << std::endl;
                auto r = this->right->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[r.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a variable.
///
///  @returns True if the node acts like a variable.
//------------------------------------------------------------------------------
        virtual bool is_all_variables() const {
            return this->left->is_all_variables() &&
                   (this->right->is_all_variables() ||
                    constant_cast(this->right).get());
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
            return this->left;
        }

//------------------------------------------------------------------------------
///  @brief Get the exponent of a power.
///
///  @returns The exponent of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_exponent() const {
            return this->right;
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> remove_pseudo() {
            if (this->has_pseudo()) {
                return pow(this->left->remove_pseudo(),
                           this->right->remove_pseudo());
            }
            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build power node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> pow(shared_leaf<T, SAFE_MATH> l,
                                  shared_leaf<T, SAFE_MATH> r) {
        auto temp = std::make_shared<pow_node<T, SAFE_MATH>> (l, r)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
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
///  @brief Build power node.
///
///  @tparam T         Base type of the calculation.
///  @tparam L         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar L, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> pow(const L l,
                                  shared_leaf<T, SAFE_MATH> r) {
        return pow(constant<T, SAFE_MATH> (static_cast<T> (l)), r);
    }

//------------------------------------------------------------------------------
///  @brief Build power node.
///
///  @tparam T         Base type of the calculation.
///  @tparam R         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar R, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> pow(shared_leaf<T, SAFE_MATH> l,
                                  const R r) {
        return pow(l, constant<T, SAFE_MATH> (static_cast<T> (r)));
    }

///  Convenience type alias for shared add nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_pow = std::shared_ptr<pow_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a power node.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_pow<T, SAFE_MATH> pow_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<pow_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Erfi node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief An imaginary error function node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  Note use templates here to defer this so it can use the operator functions.
//------------------------------------------------------------------------------
    template<jit::complex_scalar T, bool SAFE_MATH=false>
    class erfi_node final : public straight_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] a Argument node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *a) {
            return "erfi" + jit::format_to_string(reinterpret_cast<size_t> (a));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a exp node.
///
///  @param[in] x Argument.
//------------------------------------------------------------------------------
        erfi_node(shared_leaf<T, SAFE_MATH> x) :
        straight_node<T, SAFE_MATH> (x, erfi_node::to_string(x.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of erfi.
///
///  result = erfi(x)
///
///  @returns The value of erfi(x).
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> result = this->arg->evaluate();
            result.erfi();
            return result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the erfi(x).
///
///  @returns Reduced graph from exp.
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

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d erfi(z)/dx = 2/sqrt(pi)Exp(z^2)*dz/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> df(shared_leaf<T, SAFE_MATH> x) {
            if (this->is_match(x)) {
                return one<T, SAFE_MATH> ();
            }

            const size_t hash = reinterpret_cast<size_t> (x.get());
            if (this->df_cache.find(hash) == this->df_cache.end()) {
                this->df_cache[hash] = 2.0/std::sqrt(M_PI)
                                     * exp(this->arg*this->arg)*this->arg->df(x);
            }
            return this->df_cache[hash];
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @param[in]     usage     List of register usage count.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> a = this->arg->compile(stream, 
                                                                 registers,
                                                                 usage);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = special::erfi("
                       << registers[a.get()] << ")";
                this->endline(stream, usage);
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

            auto x_cast = erfi_cast(x);
            if (x_cast.get()) {
                return this->arg->is_match(x_cast->get_arg());
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << "erfi\\left(";
            this->arg->to_latex();
            std::cout << "\\right)";
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> remove_pseudo() {
            if (this->has_pseudo()) {
                return erfi(this->arg->remove_pseudo());
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
                       << " [label = \"erfi\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto a = this->arg->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[a.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Define erfi convience function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x Argument.
///  @returns A reduced exp node.
//------------------------------------------------------------------------------
    template<jit::complex_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> erfi(shared_leaf<T, SAFE_MATH> x) {
        auto temp = std::make_shared<erfi_node<T, SAFE_MATH>> (x)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
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

///  Convenience type alias for shared exp nodes.
    template<jit::complex_scalar T, bool SAFE_MATH=false>
    using shared_erfi = std::shared_ptr<erfi_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a exp node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::complex_scalar T, bool SAFE_MATH=false>
    shared_erfi<T, SAFE_MATH> erfi_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<erfi_node<T, SAFE_MATH>> (x);
    }
}

#endif /* math_h */
