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
    template<typename N>
    class sqrt_node : public straight_node<typename N::backend> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a sqrt node.
///
///  @param[in] x Argument.
//------------------------------------------------------------------------------
        sqrt_node(std::shared_ptr<N> x) :
        straight_node<typename N::backend> (x->reduce()) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of sqrt.
///
///  result = sqrt(x)
///
///  @returns The value of sqrt(x).
//------------------------------------------------------------------------------
        virtual typename N::backend evaluate() final {
            typename N::backend result = this->arg->evaluate();
            result.sqrt();
            return result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the sqrt(x).
///
///  @returns Reduced graph from sqrt.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename N::backend> reduce() final {
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
                           constant<typename N::backend> (0.5));
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
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename N::backend>
        df(shared_leaf<typename N::backend> x) final {
            if (this->is_match(x)) {
                return constant<typename N::backend> (1);
            } else {
                return this->arg->df(x) /
                       (constant<typename N::backend> (2)*this->shared_from_this());
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in] stream    String buffer stream.
///  @param[in] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename N::backend> compile(std::stringstream &stream,
                                                         jit::register_map<N> &registers) final {
            if (registers.find(this) == registers.end()) {
                shared_leaf<typename N::backend> a = this->arg->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<typename N::backend> (stream);
                stream << " " << registers[this] << " = sqrt("
                       << registers[a.get()] << ");"
                       << std::endl;
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<typename N::backend> x) final {
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
        virtual void to_latex() const final {
            std::cout << "\\sqrt{";
            this->arg->to_latex();
            std::cout << "}";
        }
    };

//------------------------------------------------------------------------------
///  @brief Define sqrt convience function.
///
///  @param[in] x Argument.
///  @returns A reduced sqrt node.
//------------------------------------------------------------------------------
    template<typename N>
    shared_leaf<typename N::backend> sqrt(std::shared_ptr<N> x) {
        return (std::make_shared<sqrt_node<N>> (x))->reduce();
    }

///  Convenience type alias for shared sqrt nodes.
    template<typename N>
    using shared_sqrt = std::shared_ptr<sqrt_node<N>>;

//------------------------------------------------------------------------------
///  @brief Cast to a sqrt node.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename N>
    shared_sqrt<N> sqrt_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<sqrt_node<N>> (x);
    }

//******************************************************************************
//  Exp node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A exp node.
///
///  Note use templates here to defer this so it can use the operator functions.
//------------------------------------------------------------------------------
    template<typename N>
    class exp_node : public straight_node<typename N::backend> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a exp node.
///
///  @param[in] x Argument.
//------------------------------------------------------------------------------
        exp_node(std::shared_ptr<N> x) :
        straight_node<typename N::backend> (x->reduce()) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of exp.
///
///  result = exp(x)
///
///  @returns The value of exp(x).
//------------------------------------------------------------------------------
        virtual typename N::backend evaluate() final {
            typename N::backend result = this->arg->evaluate();
            result.exp();
            return result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the exp(x).
///
///  @returns Reduced graph from exp.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename N::backend> reduce() final {
#ifdef USE_REDUCE
            if (constant_cast(this->arg).get()) {
                return constant(this->evaluate());
            }
#endif
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
        virtual shared_leaf<typename N::backend>
        df(shared_leaf<typename N::backend> x) final {
            if (this->is_match(x)) {
                return constant<typename N::backend> (1);
            }

            return this->shared_from_this()*this->arg->df(x);
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in] stream    String buffer stream.
///  @param[in] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename N::backend> compile(std::stringstream &stream,
                                                         jit::register_map<N> &registers) final {
            if (registers.find(this) == registers.end()) {
                shared_leaf<typename N::backend> a = this->arg->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<typename N::backend> (stream);
                stream << " " << registers[this] << " = exp("
                       << registers[a.get()] << ");"
                       << std::endl;
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<typename N::backend> x) final {
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
        virtual void to_latex() const final {
            std::cout << "e^{";
            this->arg->to_latex();
            std::cout << "}";
        }
    };

//------------------------------------------------------------------------------
///  @brief Define exp convience function.
///
///  @param[in] x Argument.
///  @returns A reduced exp node.
//------------------------------------------------------------------------------
    template<typename N>
    shared_leaf<typename N::backend> exp(std::shared_ptr<N> x) {
        return (std::make_shared<exp_node<N>> (x))->reduce();
    }

///  Convenience type alias for shared exp nodes.
    template<typename N>
    using shared_exp = std::shared_ptr<exp_node<N>>;

//------------------------------------------------------------------------------
///  @brief Cast to a exp node.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename N>
    shared_exp<N> exp_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<exp_node<N>> (x);
    }

//******************************************************************************
//  Log node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A log node.
///
///  Note use templates here to defer this so it can use the operator functions.
//------------------------------------------------------------------------------
    template<typename N>
    class log_node : public straight_node<typename N::backend> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a log node.
///
///  @param[in] x Argument.
//------------------------------------------------------------------------------
        log_node(std::shared_ptr<N> x) :
        straight_node<typename N::backend> (x->reduce()) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of log.
///
///  result = log(x)
///
///  @returns The value of log(x).
//------------------------------------------------------------------------------
        virtual typename N::backend evaluate() final {
            typename N::backend result = this->arg->evaluate();
            result.log();
            return result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the log(x).
///
///  @returns Reduced graph from log.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename N::backend> reduce() final {
#ifdef USE_REDUCE
            if (constant_cast(this->arg).get()) {
                return constant(this->evaluate());
            }
#endif
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
        virtual shared_leaf<typename N::backend>
        df(shared_leaf<typename N::backend> x) final {
            return this->arg->df(x)/this->arg;
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in] stream    String buffer stream.
///  @param[in] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename N::backend> compile(std::stringstream &stream,
                                                         jit::register_map<N> &registers) final {
            if (registers.find(this) == registers.end()) {
                shared_leaf<typename N::backend> a = this->arg->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<typename N::backend> (stream);
                stream << " " << registers[this] << " = log("
                       << registers[a.get()] << ");"
                       << std::endl;
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<typename N::backend> x) final {
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
        virtual void to_latex() const final {
            std::cout << "\\ln{\\left(";
            this->arg->to_latex();
            std::cout << "\\right)}";
        }
    };

//------------------------------------------------------------------------------
///  @brief Define log convience function.
///
///  @param[in] x Argument.
///  @returns A reduced exp node.
//------------------------------------------------------------------------------
    template<typename N>
    shared_leaf<typename N::backend> log(std::shared_ptr<N> x) {
        return (std::make_shared<log_node<N>> (x))->reduce();
    }

///  Convenience type alias for shared exp nodes.
    template<typename N>
    using shared_log = std::shared_ptr<log_node<N>>;

//------------------------------------------------------------------------------
///  @brief Cast to a exp node.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename N>
    shared_log<N> log_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<log_node<N>> (x);
    }

//******************************************************************************
//  Pow node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief An power node.
///
///  Note use templates here to defer this so it can use the operator functions.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    class pow_node : public branch_node<typename LN::backend> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct an power node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        pow_node(std::shared_ptr<LN> l,
                 std::shared_ptr<RN> r) :
        branch_node<typename LN::backend> (l, r) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of addition.
///
///  result = l^r
///
///  @returns The value of l^r.
//------------------------------------------------------------------------------
        virtual typename LN::backend evaluate() final {
            typename LN::backend l_result = this->left->evaluate();
            typename RN::backend r_result = this->right->evaluate();
            return pow(l_result, r_result);
        }

//------------------------------------------------------------------------------
///  @brief Reduce a power node.
///
///  @returns A reduced power node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::backend> reduce() final {
#ifdef USE_REDUCE
            auto rc = constant_cast(this->right);

            if (rc.get()) {
                if (rc->is(0)) {
                    return constant<typename LN::backend> (1.0);
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
                auto two = constant<typename LN::backend> (2.0);
                return pow(lsq->get_arg(), this->right/two);
            }
#endif
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
        virtual shared_leaf<typename LN::backend>
        df(shared_leaf<typename LN::backend> x) final {
            auto one = constant<typename LN::backend> (1.0);
            return pow(this->left, this->right - one) *
                   (this->right*this->left->df(x) +
                    this->left*log(this->left)*this->right->df(x));
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in] stream    String buffer stream.
///  @param[in] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::backend> compile(std::stringstream &stream,
                                                          jit::register_map<LN> &registers) final {
            if (registers.find(this) == registers.end()) {
                shared_leaf<typename LN::backend> l = this->left->compile(stream, registers);
                shared_leaf<typename RN::backend> r = this->right->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<typename LN::backend> (stream);
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
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<typename LN::backend> x) final {
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
        virtual void to_latex() const final {
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
    };

//------------------------------------------------------------------------------
///  @brief Build power node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    shared_leaf<typename LN::backend> pow(std::shared_ptr<LN> l,
                                          std::shared_ptr<RN> r) {
        return std::make_shared<pow_node<LN, RN>> (l, r)->reduce();
    }

///  Convenience type alias for shared add nodes.
    template<typename LN, typename RN>
    using shared_pow = std::shared_ptr<pow_node<LN, RN>>;

//------------------------------------------------------------------------------
///  @brief Cast to a power node.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename N>
    shared_pow<N, N> pow_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<pow_node<N, N>> (x);
    }
}

#endif /* math_h */
