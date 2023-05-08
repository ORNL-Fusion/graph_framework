//------------------------------------------------------------------------------
///  @file math.hpp
///  @brief Defined basic math functions.
//------------------------------------------------------------------------------


#ifndef math_h
#define math_h

#include "arithmetic.hpp"

namespace graph {
//******************************************************************************
//  Sqrt node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A sqrt node.
///
///  Note use templates here to defer this so it can use the operator functions.
//------------------------------------------------------------------------------
    template<typename T>
    class sqrt_node final : public straight_node<T> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a sqrt node.
///
///  @params[in] x Argument.
//------------------------------------------------------------------------------
        sqrt_node(shared_leaf<T> x) :
        straight_node<T> (x->reduce()) {}

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
        virtual shared_leaf<T> reduce() {
#ifdef USE_REDUCE
            auto ac = constant_cast(this->arg);
            if (ac.get()) {
                if (ac->is(0) || ac->is(1)) {
                    return this->arg;
                }
                return constant(this->evaluate());
            }

//  Handle casses like sqrt(a^b).
            auto ap = pow_cast(this->arg);
            if (ap.get()) {
                auto bc = constant_cast(ap->get_right());
                if (bc.get() && bc->is(2)) {
                    return ap->get_left();
                }

                return pow(ap->get_left(),
                           ap->get_right() +
                           constant(static_cast<T> (0.5)));
            }

//  Handle casses like sqrt(c*x) where c is constant or cases like
//  sqrt((x^a)*y).
            auto am = multiply_cast(this->arg);
            if (am.get()) {
                if (pow_cast(am->get_left()).get()      ||
                    constant_cast(am->get_left()).get() ||
                    pow_cast(am->get_right()).get()     ||
                    constant_cast(am->get_right()).get()) {
                    return sqrt(am->get_left()) *
                           sqrt(am->get_right());
                }
            }

//  Handle casses like sqrt(x^a/b) and sqrt(a/x^b) or sqrt(c/b) and sqrt(a/c)
//  where c is a constant.
            auto ad = divide_cast(this->arg);
            if (ad.get()) {
                if (pow_cast(ad->get_left()).get()      ||
                    constant_cast(ad->get_left()).get() ||
                    pow_cast(ad->get_right()).get()     ||
                    constant_cast(ad->get_right()).get()) {
                    return sqrt(ad->get_left()) /
                           sqrt(ad->get_right());
                }
            }
#endif
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d sqrt(a)/dx = 1/(2*sqrt(a))da/dx
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> df(shared_leaf<T> x) {
            if (this->is_match(x)) {
                return one<T> ();
            } else {
                return this->arg->df(x) /
                       (two<T> ()*this->shared_from_this());
            }
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
                shared_leaf<T> a = this->arg->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = sqrt("
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
        virtual bool is_match(shared_leaf<T> x) {
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

///  Cache for constructed nodes.
        inline static node_cache<T> cache;
    };

//------------------------------------------------------------------------------
///  @brief Define sqrt convience function.
///
///  @params[in] x Argument.
///  @returns A reduced sqrt node.
//------------------------------------------------------------------------------
    template<typename T> shared_leaf<T> sqrt(shared_leaf<T> x) {
        auto temp = std::make_shared<sqrt_node<T>> (x)->reduce();
        for (auto &c : sqrt_node<T>::cache) {
            if (temp->is_match(c)) {
                return c;
            }
        }
        sqrt_node<T>::cache.push_back(temp);
        return temp;
    }

///  Convenience type alias for shared sqrt nodes.
    template<typename T>
    using shared_sqrt = std::shared_ptr<sqrt_node<T>>;

//------------------------------------------------------------------------------
///  @brief Cast to a sqrt node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T>
    shared_sqrt<T> sqrt_cast(shared_leaf<T> x) {
        return std::dynamic_pointer_cast<sqrt_node<T>> (x);
    }

//******************************************************************************
//  Exp node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A exp node.
///
///  Note use templates here to defer this so it can use the operator functions.
//------------------------------------------------------------------------------
    template<typename T>
    class exp_node final : public straight_node<T> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a exp node.
///
///  @params[in] x Argument.
//------------------------------------------------------------------------------
        exp_node(shared_leaf<T> x) :
        straight_node<T> (x->reduce()) {}

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
        virtual shared_leaf<T> reduce() {
#ifdef USE_REDUCE
            if (constant_cast(this->arg).get()) {
                return constant(this->evaluate());
            }

//  Reduce exp(log(x)) -> x
            auto a = log_cast(this->arg);
            if (a.get()) {
                return a->get_arg();
            }
#endif
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d exp(a)/dx = exp(a)*da/dx
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> df(shared_leaf<T> x) {
            if (this->is_match(x)) {
                return one<T> ();
            }

            return this->shared_from_this()*this->arg->df(x);
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
                shared_leaf<T> a = this->arg->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = exp("
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
        virtual bool is_match(shared_leaf<T> x) {
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

///  Cache for constructed nodes.
        inline static node_cache<T> cache;
    };

//------------------------------------------------------------------------------
///  @brief Define exp convience function.
///
///  @params[in] x Argument.
///  @returns A reduced exp node.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> exp(shared_leaf<T> x) {
        auto temp = std::make_shared<exp_node<T>> (x)->reduce();
        for (auto &c : exp_node<T>::cache) {
            if (temp->is_match(c)) {
                return c;
            }
        }
        exp_node<T>::cache.push_back(temp);
        return temp;
    }

///  Convenience type alias for shared exp nodes.
    template<typename T>
    using shared_exp = std::shared_ptr<exp_node<T>>;

//------------------------------------------------------------------------------
///  @brief Cast to a exp node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T>
    shared_exp<T> exp_cast(shared_leaf<T> x) {
        return std::dynamic_pointer_cast<exp_node<T>> (x);
    }

//******************************************************************************
//  Log node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A log node.
///
///  Note use templates here to defer this so it can use the operator functions.
//------------------------------------------------------------------------------
    template<typename T>
    class log_node final : public straight_node<T> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a log node.
///
///  @params[in] x Argument.
//------------------------------------------------------------------------------
        log_node(shared_leaf<T> x) :
        straight_node<T> (x->reduce()) {}

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
        virtual shared_leaf<T> reduce() {
#ifdef USE_REDUCE
            if (constant_cast(this->arg).get()) {
                return constant(this->evaluate());
            }

//  Reduce log(exp(x)) -> x
            auto a = exp_cast(this->arg);
            if (a.get()) {
                return a->get_arg();
            }
#endif
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d log(a)/dx = (da/dx)/a
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> df(shared_leaf<T> x) {
            return this->arg->df(x)/this->arg;
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
            if (registers.find(this) == registers.end()) {
                shared_leaf<T> a = this->arg->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = log("
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
        virtual bool is_match(shared_leaf<T> x) {
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

///  Cache for constructed nodes.
        inline static node_cache<T> cache;
    };

//------------------------------------------------------------------------------
///  @brief Define log convience function.
///
///  @params[in] x Argument.
///  @returns A reduced log node.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> log(shared_leaf<T> x) {
        auto temp = std::make_shared<log_node<T>> (x)->reduce();
        for (auto &c : log_node<T>::cache) {
            if (temp->is_match(c)) {
                return c;
            }
        }
        log_node<T>::cache.push_back(temp);
        return temp;
    }

///  Convenience type alias for shared log nodes.
    template<typename T>
    using shared_log = std::shared_ptr<log_node<T>>;

//------------------------------------------------------------------------------
///  @brief Cast to a exp node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T>
    shared_log<T> log_cast(shared_leaf<T> x) {
        return std::dynamic_pointer_cast<log_node<T>> (x);
    }

//******************************************************************************
//  Pow node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief An power node.
///
///  Note use templates here to defer this so it can use the operator functions.
//------------------------------------------------------------------------------
    template<typename T>
    class pow_node final : public branch_node<T> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct an power node.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
        pow_node(shared_leaf<T> l,
                 shared_leaf<T> r) :
        branch_node<T> (l, r) {}

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
        virtual shared_leaf<T> reduce() {
#ifdef USE_REDUCE
            auto rc = constant_cast(this->right);

            if (rc.get()) {
                if (rc->is(0)) {
                    return one<T> ();
                } else if (rc->is(1)) {
                    return this->left;
                } else if (rc->is(0.5)) {
                    return sqrt(this->left);
                } else if (rc->is(2)){
                    auto sq = sqrt_cast(this->left);
                    if (sq.get()) {
                        return sq->get_arg();
                    }
                }

                if (constant_cast(this->left).get()) {
                    return constant(this->evaluate());
                }
            }

            auto lp = pow_cast(this->left);
            if (lp.get()) {
                return pow(lp->get_left(), lp->get_right()*this->right);
            }

//  Handle cases where (c*x)^a, (x*c)^a, (a*sqrt(b))^c and (a*b^c)^2.
            auto lm = multiply_cast(this->left);
            if (lm.get()) {
                if (constant_cast(lm->get_left()).get()  ||
                    constant_cast(lm->get_right()).get() ||
                    sqrt_cast(lm->get_left()).get()      ||
                    sqrt_cast(lm->get_right()).get()     ||
                    pow_cast(lm->get_left()).get()       ||
                    pow_cast(lm->get_right()).get()) {
                    return pow(lm->get_left(), this->right) *
                           pow(lm->get_right(), this->right);
                }
            }

//  Handle cases where (c/x)^a, (x/c)^a, (a/sqrt(b))^c and (a/b^c)^2.
            auto ld = divide_cast(this->left);
            if (ld.get()) {
                if (constant_cast(ld->get_left()).get()  ||
                    constant_cast(ld->get_right()).get() ||
                    sqrt_cast(ld->get_left()).get()      ||
                    sqrt_cast(ld->get_right()).get()     ||
                    pow_cast(ld->get_left()).get()       ||
                    pow_cast(ld->get_right()).get()) {
                    return pow(ld->get_left(), this->right) /
                           pow(ld->get_right(), this->right);
                }
            }

//  Reduce sqrt(a)^b
            auto lsq = sqrt_cast(this->left);
            if (lsq.get()) {
                return pow(lsq->get_arg(),
                           this->right/two<T> ());
            }
#endif
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d a^b dx = b*a^(b-1)*da/dx + ln(a)a^b*db/dx
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T>
        df(shared_leaf<T> x) {
            return pow(this->left, this->right - one<T> ()) *
                   (this->right*this->left->df(x) +
                    this->left*log(this->left)*this->right->df(x));
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
            if (registers.find(this) == registers.end()) {
                shared_leaf<T> l = this->left->compile(stream, registers);
                shared_leaf<T> r = this->right->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = pow("
                       << registers[l.get()] << ", "
                       << registers[r.get()] << ");"
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
        virtual bool is_match(shared_leaf<T> x) {
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

///  Cache for constructed nodes.
        inline static node_cache<T> cache;
    };

//------------------------------------------------------------------------------
///  @brief Build power node.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> pow(shared_leaf<T> l,
                       shared_leaf<T> r) {
        auto temp = std::make_shared<pow_node<T>> (l, r)->reduce();
        for (auto &c : pow_node<T>::cache) {
            if (temp->is_match(c)) {
                return c;
            }
        }
        pow_node<T>::cache.push_back(temp);
        return temp;
    }

///  Convenience type alias for shared add nodes.
    template<typename T>
    using shared_pow = std::shared_ptr<pow_node<T>>;

//------------------------------------------------------------------------------
///  @brief Cast to a power node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T>
    shared_pow<T> pow_cast(shared_leaf<T> x) {
        return std::dynamic_pointer_cast<pow_node<T>> (x);
    }
}

#endif /* math_h */
