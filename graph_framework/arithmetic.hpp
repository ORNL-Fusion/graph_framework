//------------------------------------------------------------------------------
///  @file arithmetic.hpp
///  @brief Basic arithmetic operations.
///
///  Defines basic operators.
//------------------------------------------------------------------------------

#ifndef arithmetic_h
#define arithmetic_h

#include "node.hpp"
#include "register.hpp"

namespace graph {
//------------------------------------------------------------------------------
///  @brief Check if an expression is variable like.
///
///  Variable like quantities can be a variable or sqrt and power of one.
///
///  @params[in] a Expression to check.
///  @returns True if a is variable like.
//------------------------------------------------------------------------------
    template<typename N>
    bool is_variable_like(std::shared_ptr<N> a) {
        return variable_cast(a).get() ||
               (sqrt_cast(a).get() && variable_cast(sqrt_cast(a)->get_arg()).get()) ||
               (pow_cast(a).get()  && variable_cast(pow_cast(a)->get_left()).get());
    }

//------------------------------------------------------------------------------
///  @brief Get the argument of a variable like object.
///
///  @params[in] a Expression to check.
///  @returns The agument of a.
//------------------------------------------------------------------------------
    template<typename N>
    std::shared_ptr<N> get_argument(std::shared_ptr<N> a) {
        if (variable_cast(a).get()) {
            return a;
        } else if (sqrt_cast(a).get() &&
                   variable_cast(sqrt_cast(a)->get_arg()).get()) {
            return sqrt_cast(a)->get_arg();
        } else if (pow_cast(a).get()  &&
                   variable_cast(pow_cast(a)->get_left()).get()) {
            return pow_cast(a)->get_left();
        }
        assert(false && "Should never reach this point.");
        return nullptr;
    }

//------------------------------------------------------------------------------
///  @brief Check variable like objects are the same.
///
///  @params[in] a Expression to check.
///  @params[in] b Expression to check.
///  @returns True if a is variable like.
//------------------------------------------------------------------------------
    template<typename N>
    bool is_same_variable_like(std::shared_ptr<N> a,
                               std::shared_ptr<N> b) {
        return is_variable_like(a) &&
               is_variable_like(b) &&
               get_argument(a)->is_match(get_argument(b));
    }

//******************************************************************************
//  Add node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief An addition node.
///
///  Note use templates here to defer this so it can use the operator functions.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    class add_node : public branch_node<typename LN::base> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct an addition node.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
        add_node(std::shared_ptr<LN> l,
                 std::shared_ptr<RN> r) :
        branch_node<typename LN::base> (l, r) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of addition.
///
///  result = l + r
///
///  @returns The value of l + r.
//------------------------------------------------------------------------------
        virtual backend::buffer<typename LN::base> evaluate() final {
            backend::buffer<typename LN::base> l_result = this->left->evaluate();
            backend::buffer<typename RN::base> r_result = this->right->evaluate();
            return l_result + r_result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce an addition node.
///
///  @returns A reduced addition node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::base> reduce() final {
#ifdef USE_REDUCE
//  Idenity reductions.
            if (this->left->is_match(this->right)) {
                return two<typename LN::base> ()*this->left;
            }

//  Constant reductions.
            auto l = constant_cast(this->left);
            auto r = constant_cast(this->right);

            if (l.get() && l->is(0)) {
                return this->right;
            } else if (r.get() && r->is(0)) {
                return this->left;
            } else if (l.get() && r.get()) {
                return constant(this->evaluate());
            }

//  Common factor reduction. If the left and right are both muliply nodes check
//  for a common factor. So you can change a*b + a*c -> a*(b + c).
            auto lm = multiply_cast(this->left);
            auto rm = multiply_cast(this->right);

//  Assume constants are on the left.
//  v1 + -c*v2 -> v1 - c*v2
//  -c*v1 + v2 -> v2 - c*v1
            auto none = graph::none<typename LN::base> ();
            if (rm.get()) {
                auto rmc = constant_cast(rm->get_left());
                if (rmc.get() && rmc->evaluate().is_negative()) {
                    return this->left - none*rm->get_left()*rm->get_right();
                }
            } else if (lm.get()) {
                auto lmc = constant_cast(lm->get_left());
                if (lmc.get() && lmc->evaluate().is_negative()) {
                    return this->right - none*lm->get_left()*lm->get_right();
                }
            }

            if (lm.get() && rm.get()) {
                if (lm->get_left()->is_match(rm->get_left())) {
                    return lm->get_left()*(lm->get_right() + rm->get_right());
                } else if (lm->get_left()->is_match(rm->get_right())) {
                    return lm->get_left()*(lm->get_right() + rm->get_left());
                } else if (lm->get_right()->is_match(rm->get_left())) {
                    return lm->get_right()*(lm->get_left() + rm->get_right());
                } else if (lm->get_right()->is_match(rm->get_right())) {
                    return lm->get_right()*(lm->get_left() + rm->get_left());
                }

//  Change cases like c1*a + c2*b -> c1*(a + c2*b)
                auto lmc = constant_cast(lm->get_left());
                auto rmc = constant_cast(rm->get_left());
                if (lmc.get() && rmc.get()) {
                    return lm->get_left()*(lm->get_right() +
                                           (rm->get_left()/lm->get_left())*rm->get_right());
                }
            }

//  Common denominator reduction. If the left and right are both divide nodes
//  for a common denominator. So you can change a/b + c/b -> (a + c)/d.
            auto ld = divide_cast(this->left);
            auto rd = divide_cast(this->right);

            if (ld.get() && rd.get() &&
                ld->get_right()->is_match(rd->get_right())) {
                return (ld->get_left() + rd->get_left())/ld->get_right();
            }

//  Move cases like
//  (c1 + c2/x) + c3/y -> c1 + (c2/x + c3/y)
//  (c1 - c2/x) + c3/y -> c1 + (c3/y - c2/x)
//  in case of common denominators.
            if (rd.get()) {
                auto la = add_cast(this->left);
                if (la.get() && divide_cast(la->get_right()).get()) {
                    return la->get_left() + (la->get_right() + this->right);
                }

                auto ls = subtract_cast(this->left);
                if (ls.get() && divide_cast(ls->get_right()).get()) {
                    return ls->get_left() + (this->right - ls->get_right());
                }
            }

//  Fused multiply add reductions.
            auto m = multiply_cast(this->left);

            if (m.get()) {
                return fma(m->get_left(),
                           m->get_right(),
                           this->right);
            }

//  Handle cases like:
//  (a/y)^e + b/y^e -> (a^2 + b)/(y^e)
//  b/y^e + (a/y)^e -> (b + a^2)/(y^e)
//  (a/y)^e + (b/y)^e -> (a^2 + b^2)/(y^e)
            auto pl = pow_cast(this->left);
            auto pr = pow_cast(this->right);
            if (pl.get() && rd.get()) {
                auto rdp = pow_cast(rd->get_right());
                if (rdp.get() && pl->get_right()->is_match(rdp->get_right())) {
                    auto plld = divide_cast(pl->get_left());
                    if (plld.get() &&
                        rdp->get_left()->is_match(plld->get_right())) {
                        return (pow(plld->get_left(), pl->get_right()) +
                                rd->get_left()) /
                               pow(rdp->get_left(), pl->get_right());
                    }
                }
            } else if (pr.get() && ld.get()) {
                auto ldp = pow_cast(ld->get_right());
                if (ldp.get() && pr->get_right()->is_match(ldp->get_right())) {
                    auto prld = divide_cast(pr->get_left());
                    if (prld.get() &&
                        ldp->get_left()->is_match(prld->get_right())) {
                        return (pow(prld->get_left(), pr->get_right()) +
                                ld->get_left()) /
                               pow(ldp->get_left(), pr->get_right());
                    }
                }
            } else if (pl.get() && pr.get()) {
                if (pl->get_right()->is_match(pr->get_right())) {
                    auto pld = divide_cast(pl->get_left());
                    auto prd = divide_cast(pr->get_left());
                    if (pld.get() && prd.get() &&
                        pld->get_right()->is_match(prd->get_right())) {
                        return (pow(pld->get_left(), pl->get_right()) +
                                pow(prd->get_left(), pl->get_right())) /
                               pow(pld->get_right(), pl->get_right());
                    }
                }
            }

            auto lfma = fma_cast(this->left);
            auto rfma = fma_cast(this->right);

            if (lfma.get() && rfma.get()) {
                if (lfma->get_middle()->is_match(rfma->get_middle())) {
                    return fma(lfma->get_left() + rfma->get_left(),
                               lfma->get_middle(),
                               lfma->get_right() + rfma->get_right());
                }
            }

            if (lfma.get()) {
                return fma(lfma->get_left(),
                           lfma->get_middle(),
                           lfma->get_right() + this->right);
            }
#endif
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d add(a,b)/dx = da/dx + db/dx
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::base>
        df(shared_leaf<typename LN::base> x) final {
            if (this->is_match(x)) {
                return one<typename LN::base> ();
            } else {
                return this->left->df(x) + this->right->df(x);
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in] stream    String buffer stream.
///  @params[in] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::base> compile(std::stringstream &stream,
                                                       jit::register_map<LN> &registers) final {
            if (registers.find(this) == registers.end()) {
                shared_leaf<typename LN::base> l = this->left->compile(stream, registers);
                shared_leaf<typename RN::base> r = this->right->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<typename LN::base> (stream);
                stream << " " << registers[this] << " = "
                       << registers[l.get()] << " + "
                       << registers[r.get()] << ";"
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
        virtual bool is_match(shared_leaf<typename LN::base> x) final {
            if (this == x.get()) {
                return true;
            }

            auto x_cast = add_cast(x);
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
            std::cout << "+";
            if (r_brackets) {
                std::cout << "\\left(";
            }
            this->right->to_latex();
            if (r_brackets) {
                std::cout << "\\right)";
            }
        }
    };

//------------------------------------------------------------------------------
///  @brief Build add node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    shared_leaf<typename LN::base> add(std::shared_ptr<LN> l,
                                       std::shared_ptr<RN> r) {
        return (std::make_shared<add_node<LN, RN>> (l, r))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Build add node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    shared_leaf<typename LN::base> operator+(std::shared_ptr<LN> l,
                                             std::shared_ptr<RN> r) {
        return add<LN, RN> (l, r);
    }

///  Convenience type alias for shared add nodes.
    template<typename LN, typename RN>
    using shared_add = std::shared_ptr<add_node<LN, RN>>;

//------------------------------------------------------------------------------
///  @brief Cast to a add node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename N>
    shared_add<N, N> add_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<add_node<N, N>> (x);
    }

//******************************************************************************
//  Subtract node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A subtraction node.
///
///  Note use templates here to defer this so it can use the operator functions.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    class subtract_node : public branch_node<typename LN::base> {
    public:
//------------------------------------------------------------------------------
///  @brief Consruct a subtraction node.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
        subtract_node(std::shared_ptr<LN> l,
                      std::shared_ptr<RN> r) :
        branch_node<typename LN::base> (l, r) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of subtraction.
///
///  result = l - r
///
///  @returns The value of l - r.
//------------------------------------------------------------------------------
        virtual backend::buffer<typename LN::base> evaluate() final {
            backend::buffer<typename LN::base> l_result = this->left->evaluate();
            backend::buffer<typename RN::base> r_result = this->right->evaluate();
            return l_result - r_result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce an subtraction node.
///
///  @returns A reduced subtraction node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::base> reduce() final {
#ifdef USE_REDUCE
//  Idenity reductions.
            if (this->left->is_match(this->right)) {
                auto l = constant_cast(this->left);
                if (l.get() && l->is(0)) {
                    return this->left;
                }

                return zero<typename LN::base> ();
            }

//  Constant reductions.
            auto l = constant_cast(this->left);
            auto r = constant_cast(this->right);

            if (l.get() && l->is(0)) {
                return none<typename LN::base> ()*this->right;
            } else if (r.get() && r->is(0)) {
                return this->left;
            } else if (l.get() && r.get()) {
                return constant(this->evaluate());
            }

//  Common factor reduction. If the left and right are both muliply nodes check
//  for a common factor. So you can change a*b - a*c -> a*(b - c).
            auto lm = multiply_cast(this->left);
            auto rm = multiply_cast(this->right);

//  Assume constants are on the left.
//  v1 - -c*v2 -> v1 + c*v2
            if (rm.get()) {
                auto rmc = constant_cast(rm->get_left());
                if (rmc.get() && rmc->evaluate().is_negative()) {
                    return this->left +
                           none<typename LN::base> ()*rm->get_left()*rm->get_right();
                }
            }

            if (lm.get() && rm.get()) {
                if (lm->get_left()->is_match(rm->get_left())) {
                    return lm->get_left()*(lm->get_right() - rm->get_right());
                } else if (lm->get_left()->is_match(rm->get_right())) {
                    return lm->get_left()*(lm->get_right() - rm->get_left());
                } else if (lm->get_right()->is_match(rm->get_left())) {
                    return lm->get_right()*(lm->get_left() - rm->get_right());
                } else if (lm->get_right()->is_match(rm->get_right())) {
                    return lm->get_right()*(lm->get_left() - rm->get_left());
                }

//  Change cases like c1*a - c2*b -> c1*(a - c2*b)
                auto lmc = constant_cast(lm->get_left());
                auto rmc = constant_cast(rm->get_left());
                if (lmc.get() && rmc.get()) {
                    return lm->get_left()*(lm->get_right() -
                                           (rm->get_left()/lm->get_left())*rm->get_right());
                }
            }

//  Common denominator reduction. If the left and right are both divide nodes
//  for a common denominator. So you can change a/b - c/b -> (a - c)/d.
            auto ld = divide_cast(this->left);
            auto rd = divide_cast(this->right);

            if (ld.get() && rd.get() &&
                ld->get_right()->is_match(rd->get_right())) {
                return (ld->get_left() - rd->get_left())/ld->get_right();
            }

//  Move cases like
//  (c1 + c2/x) - c3/y -> c1 + (c2/x - c3/y)
//  (c1 - c2/x) - c3/y -> c1 - (c2/x + c3/y)
//  in case of common denominators.
            if (rd.get()) {
                auto la = add_cast(this->left);
                if (la.get() && divide_cast(la->get_right()).get()) {
                    return la->get_left() + (la->get_right() - this->right);
                }

                auto ls = subtract_cast(this->left);
                if (ls.get() && divide_cast(ls->get_right()).get()) {
                    return ls->get_left() - (this->right + ls->get_right());
                }
            }

//  Handle cases like:
//  (a/y)^e - b/y^e -> (a^2 - b)/(y^e)
//  b/y^e - (a/y)^e -> (b - a^2)/(y^e)
//  (a/y)^e - (b/y)^e -> (a^2 - b^2)/(y^e)
            auto pl = pow_cast(this->left);
            auto pr = pow_cast(this->right);
            if (pl.get() && rd.get()) {
                auto rdp = pow_cast(rd->get_right());
                if (rdp.get() && pl->get_right()->is_match(rdp->get_right())) {
                    auto plld = divide_cast(pl->get_left());
                    if (plld.get() &&
                        rdp->get_left()->is_match(plld->get_right())) {
                        return (pow(plld->get_left(), pl->get_right()) -
                                rd->get_left()) /
                               pow(rdp->get_left(), pl->get_right());
                    }
                }
            } else if (pr.get() && ld.get()) {
                auto ldp = pow_cast(ld->get_right());
                if (ldp.get() && pr->get_right()->is_match(ldp->get_right())) {
                    auto prld = divide_cast(pr->get_left());
                    if (prld.get() &&
                        ldp->get_left()->is_match(prld->get_right())) {
                        return (pow(prld->get_left(), pr->get_right()) -
                                ld->get_left()) /
                               pow(ldp->get_left(), pr->get_right());
                    }
                }
            } else if (pl.get() && pr.get()) {
                if (pl->get_right()->is_match(pr->get_right())) {
                    auto pld = divide_cast(pl->get_left());
                    auto prd = divide_cast(pr->get_left());
                    if (pld.get() && prd.get() &&
                        pld->get_right()->is_match(prd->get_right())) {
                        return (pow(pld->get_left(), pl->get_right()) -
                                pow(prd->get_left(), pl->get_right())) /
                               pow(pld->get_right(), pl->get_right());
                    }
                }
            }

            auto lfma = fma_cast(this->left);
            auto rfma = fma_cast(this->right);

            if (lfma.get() && rfma.get()) {
                if (lfma->get_middle()->is_match(rfma->get_middle())) {
                    return fma(lfma->get_left() - rfma->get_left(),
                               lfma->get_middle(),
                               lfma->get_right() - rfma->get_right());
                }
            }
#endif
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d sub(a,b)/dx = da/dx - db/dx
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::base>
        df(shared_leaf<typename LN::base> x) final {
            if (this->is_match(x)) {
                return one<typename LN::base> ();
            } else {
                return this->left->df(x) - this->right->df(x);
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in] stream    String buffer stream.
///  @params[in] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::base> compile(std::stringstream &stream,
                                                       jit::register_map<LN> &registers) final {
            if (registers.find(this) == registers.end()) {
                shared_leaf<typename LN::base> l = this->left->compile(stream, registers);
                shared_leaf<typename RN::base> r = this->right->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<typename LN::base> (stream);
                stream << " " << registers[this] << " = "
                       << registers[l.get()] << " - "
                       << registers[r.get()] << ";"
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
        virtual bool is_match(shared_leaf<typename LN::base> x) final {
            if (this == x.get()) {
                return true;
            }

            auto x_cast = subtract_cast(x);
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
            std::cout << "-";
            if (r_brackets) {
                std::cout << "\\left(";
            }
            this->right->to_latex();
            if (r_brackets) {
                std::cout << "\\right)";
            }
        }
    };

//------------------------------------------------------------------------------
///  @brief Build subtract node from two leaves.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    shared_leaf<typename LN::base> subtract(std::shared_ptr<LN> l,
                                            std::shared_ptr<RN> r) {
        return (std::make_shared<subtract_node<LN, RN>> (l, r))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Build subtract operator from two leaves.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    shared_leaf<typename LN::base> operator-(std::shared_ptr<LN> l,
                                             std::shared_ptr<RN> r) {
        return subtract<LN, RN> (l, r);
    }

///  Convenience type alias for shared subtract nodes.
    template<typename LN, typename RN>
    using shared_subtract = std::shared_ptr<subtract_node<LN, RN>>;

//------------------------------------------------------------------------------
///  @brief Cast to a subtract node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename N>
    shared_subtract<N, N> subtract_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<subtract_node<N, N>> (x);
    }

//******************************************************************************
//  Multiply node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A multiplcation node.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    class multiply_node : public branch_node<typename LN::base> {
    public:
//------------------------------------------------------------------------------
///  @brief Consruct a multiplcation node.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
        multiply_node(std::shared_ptr<LN> l,
                      std::shared_ptr<RN> r) :
        branch_node<typename LN::base> (l, r) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of multiplcation.
///
///  result = l*r
///
///  @returns The value of l*r.
//------------------------------------------------------------------------------
        virtual backend::buffer<typename LN::base> evaluate() final {
            backend::buffer<typename LN::base> l_result = this->left->evaluate();

//  If the left are right are same don't evaluate the right.
//  NOTE: Do not use is_match here. Remove once power is implimented.
            if (this->left.get() == this->right.get()) {
                return l_result*l_result;
            }

//  If all the elements on the left are zero, return the leftside without
//  revaluating the rightside. Stop this loop early once the first non zero
//  element is encountered.
            if (l_result.is_zero()) {
                return l_result;
            }

            backend::buffer<typename LN::base> r_result = this->right->evaluate();
            return l_result*r_result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce an multiplcation node.
///
///  @returns A reduced multiplcation node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::base> reduce() final {
#ifdef USE_REDUCE
            auto l = constant_cast(this->left);
            auto r = constant_cast(this->right);

            if (l.get() && l->is(1)) {
                return this->right;
            } else if (l.get() && l->is(0)) {
                return this->left;
            } else if (r.get() && r->is(1)) {
                return this->left;
            } else if (r.get() && r->is(0)) {
                return this->right;
            } else if (l.get() && r.get()) {
                return constant(this->evaluate());
            }

//  Move constants to the left.
            if (r.get() && !l.get()) {
                return this->right*this->left;
            }

//  Move variables, sqrt of variables, and powers of variables to the right.
            if (is_variable_like(this->left) &&
                !is_variable_like(this->right)) {
                return this->right*this->left;
            }

//  Reduce x*x to x^2
            if (this->left->is_match(this->right)) {
                return pow(this->left, two<typename LN::base> ());
            }

//  Gather common terms.
//  (a*b)*a -> (a*a)*b
//  (b*a)*a -> (a*a)*b
//  a*(a*b) -> (a*a)*b
//  a*(b*a) -> (a*a)*b
            auto lm = multiply_cast(this->left);
            if (lm.get()) {
                if (this->right->is_match(lm->get_left())) {
                    return (this->right*lm->get_left())*lm->get_right();
                } else if (this->right->is_match(lm->get_right())) {
                    return (this->right*lm->get_right())*lm->get_left();
                }

//  Promote constants before variables.
//  (c*v1)*v2 -> c*(v1*v2)
                if (constant_cast(lm->get_left()).get()) {
                    return lm->get_left()*(lm->get_right()*this->right);
                }

//  Assume variables, sqrt of variables, and powers of variables are on the
//  right.
//  (a*v)*b -> a*(v*b)
                if (is_variable_like(lm->get_right()) &&
                    !is_variable_like(lm->get_left())) {
                    return lm->get_left()*(lm->get_right()*this->right);
                }
            }

            auto rm = multiply_cast(this->right);
            if (rm.get()) {
//  Assume constants are on the left.
//  c1*(c2*v) -> c3*v
                if (constant_cast(rm->get_left()).get() && l.get()) {
                    return (this->left*rm->get_left())*rm->get_right();
                }

                if (this->left->is_match(rm->get_left())) {
                    return (this->left*rm->get_left())*rm->get_right();
                } else if (this->left->is_match(rm->get_right())) {
                    return (this->left*rm->get_right())*rm->get_left();
                }
            }

//  v1*(c*v2) -> c*(v1*v2)
            if (rm.get() && constant_cast(rm->get_left()).get()) {
                return rm->get_left()*(this->left*rm->get_right());
            }

//  Factor out common constants c*b*c*d -> c*c*b*d. c*c will get reduced to c on
//  the second pass.
            if (lm.get() && rm.get()) {
                if (constant_cast(lm->get_left()).get() &&
                    constant_cast(rm->get_left()).get()) {
                    return (lm->get_left()*rm->get_left()) *
                           (lm->get_right()*rm->get_right());
                } else if (constant_cast(lm->get_left()).get() &&
                           constant_cast(rm->get_right()).get()) {
                    return (lm->get_left()*rm->get_right()) *
                           (lm->get_right()*rm->get_left());
                } else if (constant_cast(lm->get_right()).get() &&
                           constant_cast(rm->get_left()).get()) {
                    return (lm->get_right()*rm->get_left()) *
                           (lm->get_left()*rm->get_right());
                } else if (constant_cast(lm->get_right()).get() &&
                           constant_cast(rm->get_right()).get()) {
                    return (lm->get_right()*rm->get_right()) *
                           (lm->get_left()*rm->get_left());
                }

//  Gather common terms. This will help reduce sqrt(a)*sqrt(a).
                if (lm->get_left()->is_match(rm->get_left())) {
                    return (lm->get_left()*rm->get_left()) *
                           (lm->get_right()*rm->get_right());
                } else if (lm->get_right()->is_match(rm->get_left())) {
                    return (lm->get_right()*rm->get_left()) *
                           (lm->get_left()*rm->get_right());
                } else if (lm->get_left()->is_match(rm->get_right())) {
                    return (lm->get_left()*rm->get_right()) *
                           (lm->get_right()*rm->get_left());
                } else if (lm->get_right()->is_match(rm->get_right())) {
                    return (lm->get_right()*rm->get_right()) *
                           (lm->get_left()*rm->get_left());
                }
            }

//  Common factor reduction. (a/b)*(c/a) = c/b.
            auto ld = divide_cast(this->left);
            auto rd = divide_cast(this->right);

//  c1*(c2/v) -> c3/v
//  c1*(v/c2) -> v/c3
            if (rd.get() && l.get()) {
                if (constant_cast(rd->get_left()).get()) {
                    return (this->left*rd->get_left())/rd->get_right();
                } else if (constant_cast(rd->get_right()).get()) {
                    return rd->get_left()/(this->left*rd->get_right());
                }
            }

            if (ld.get() && rd.get()) {
                if (ld->get_left()->is_match(rd->get_right())) {
                    return ld->get_right()/rd->get_left();
                } else if (ld->get_right()->is_match(rd->get_left())) {
                    return ld->get_left()/rd->get_right();
                }

//  Convert (a/b)*(c/d) -> (a*c)/(b*d). This should help reduce cases like.
//  (a/b)*(a/b) + (c/b)*(c/b).
                return (ld->get_left()*rd->get_left()) /
                       (ld->get_right()*rd->get_right());
            }

//  Power reductions. Reduced cases like a^b*a^c == a^(b+c).
            auto lp = pow_cast(this->left);
            auto rp = pow_cast(this->right);
            if (lp.get()) {
//  a^b*a -> a^(b + 1)
                if (lp->get_left()->is_match(this->right)) {
                    return pow(lp->get_left(),
                               lp->get_right() + one<typename LN::base> ());
                }

//  a^b*a^c -> a^(b + c)
                if (rp.get() && lp->get_left()->is_match(rp->get_left())) {
                    return pow(lp->get_left(),
                               lp->get_right() + rp->get_right());
                }

//  a^b*sqrt(a) -> a^(b + 1/2)
                auto rsq = sqrt_cast(this->right);
                if (rsq.get() && lp->get_left()->is_match(rsq->get_arg())) {
                    return pow(lp->get_left(),
                               lp->get_right() + constant(static_cast<typename LN::base> (0.5)));
                }
            } else {
//  a*sqrt(a) -> a^(1 + 1/2)
                auto rsq = sqrt_cast(this->right);
                if (rsq.get() && this->left->is_match(rsq->get_arg())) {
                    return pow(this->left, constant(static_cast<typename LN::base> (1.5)));
                }
            }
            if (rp.get()) {
//  a*a^b -> a^(1 + b)
                if (rp->get_left()->is_match(this->left)) {
                    return pow(rp->get_left(),
                               rp->get_right() + one<typename LN::base> ());
                }

//  sqrt(a)*a^b -> a^(b + 1)
                auto lsq = sqrt_cast(this->left);
                if (lsq.get() && rp->get_left()->is_match(lsq->get_arg())) {
                    return pow(rp->get_left(),
                               rp->get_right() + constant(static_cast<typename LN::base> (0.5)));
                }
            } else {
//  sqrt(a)*a -> a^(1/2 + 1)
                auto lsq = sqrt_cast(this->left);
                if (lsq.get() && this->right->is_match(lsq->get_arg())) {
                    return pow(this->right, constant(static_cast<typename LN::base> (1.5)));
                }
            }
#endif
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d mul(a,b)/dx = da/dx*b + a*db/dx
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::base>
        df(shared_leaf<typename LN::base> x) final {
            if (this->is_match(x)) {
                return one<typename LN::base> ();
            }

            return this->left->df(x)*this->right +
                   this->left*this->right->df(x);
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in] stream    String buffer stream.
///  @params[in] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::base> compile(std::stringstream &stream,
                                                       jit::register_map<LN> &registers) final {
            if (registers.find(this) == registers.end()) {
                shared_leaf<typename LN::base> l = this->left->compile(stream, registers);
                shared_leaf<typename RN::base> r = this->right->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<typename LN::base> (stream);
                stream << " " << registers[this] << " = "
                       << registers[l.get()] << "*"
                       << registers[r.get()] << ";"
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
        virtual bool is_match(shared_leaf<typename LN::base> x) final {
            if (this == x.get()) {
                return true;
            }

            auto x_cast = multiply_cast(x);
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
            if (constant_cast(this->left).get() ||
                add_cast(this->left).get()      ||
                subtract_cast(this->left).get()) {
                std::cout << "\\left(";
                this->left->to_latex();
                std::cout << "\\right)";
            } else {
                this->left->to_latex();
            }
            std::cout << " ";
            if (constant_cast(this->right).get() ||
                add_cast(this->right).get()     ||
                subtract_cast(this->right).get()) {
                std::cout << "\\left(";
                this->right->to_latex();
                std::cout << "\\right)";
            } else {
                this->right->to_latex();
            }
        }
    };

//------------------------------------------------------------------------------
///  @brief Build multiply node from two leaves.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    shared_leaf<typename LN::base> multiply(std::shared_ptr<LN> l,
                                            std::shared_ptr<RN> r) {
        return (std::make_shared<multiply_node<LN, RN>> (l, r))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Build multiply operator from two leaves.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    shared_leaf<typename LN::base> operator*(std::shared_ptr<LN> l,
                                             std::shared_ptr<RN> r) {
        return multiply<LN, RN> (l, r);
    }

///  Convenience type alias for shared multiply nodes.
    template<typename LN, typename RN>
    using shared_multiply = std::shared_ptr<multiply_node<LN, RN>>;

//------------------------------------------------------------------------------
///  @brief Cast to a multiply node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename N>
    shared_multiply<N, N> multiply_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<multiply_node<N, N>> (x);
    }

//******************************************************************************
//  Divide node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A division node.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    class divide_node : public branch_node<typename LN::base> {
    public:
        divide_node(std::shared_ptr<LN> n,
                    std::shared_ptr<RN> d) :
        branch_node<typename LN::base> (n, d) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of division.
///
///  result = n/d
///
///  @returns The value of n/d.
//------------------------------------------------------------------------------
        virtual backend::buffer<typename LN::base> evaluate() final {
            backend::buffer<typename LN::base> l_result = this->left->evaluate();

//  If all the elements on the left are zero, return the leftside without
//  revaluating the rightside. Stop this loop early once the first non zero
//  element is encountered.
            if (l_result.is_zero()) {
                return l_result;
            }

            backend::buffer<typename RN::base> r_result = this->right->evaluate();
            return l_result/r_result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce an division node.
///
///  @returns A reduced division node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::base> reduce() final {
#ifdef USE_REDUCE
//  Constant Reductions.
            auto l = constant_cast(this->left);
            auto r = constant_cast(this->right);

            if ((l.get() && l->is(0)) ||
                (r.get() && r->is(1))) {
                return this->left;
            } else if (l.get() && r.get()) {
                return constant(this->evaluate());
            }

            if (this->left->is_match(this->right)) {
                if (l.get() && l->is(1)) {
                    return this->left;
                }

                return one<typename LN::base> ();
            }

//  Reduce cases of a/c1 -> c2*a
            if (r.get()) {
                return (one<typename LN::base> ()/this->right) *
                       this->left;
            }

//  Reduce fused multiply divided by constant nodes.
            auto lfma = fma_cast(this->left);
            if (r.get() && lfma.get()) {
                return fma(lfma->get_left()/this->right,
                           lfma->get_middle(),
                           lfma->get_right()/this->right);
            }

//  Common factor reduction. (a*b)/(a*c) = b/c.
            auto lm = multiply_cast(this->left);
            auto rm = multiply_cast(this->right);

//  Assume constants are always on the left.
//  c1/(c2*v) -> c3/v
//  (c1*v)/c2 -> c3*v
            if (rm.get() && l.get()) {
                if (constant_cast(rm->get_left()).get()) {
                    return (this->left/rm->get_left())/rm->get_right();
                }
            } else if (lm.get() && r.get()) {
                if (constant_cast(lm->get_left()).get()) {
                    return (lm->get_left()/this->right)*lm->get_right();
                }
            }

            if (lm.get() && rm.get()) {
//  Test for constants that can be reduced out.
                if (constant_cast(lm->get_left()).get() &&
                    constant_cast(rm->get_left()).get()) {
                    return (lm->get_left()/rm->get_left())*(lm->get_right()/rm->get_right());
                } else if (constant_cast(lm->get_left()).get() &&
                           constant_cast(rm->get_right()).get()) {
                    return (lm->get_left()/rm->get_right())*(lm->get_right()/rm->get_left());
                } else if (constant_cast(lm->get_right()).get() &&
                           constant_cast(rm->get_left()).get()) {
                    return (lm->get_right()/rm->get_left())*(lm->get_left()/rm->get_right());
                } else if (constant_cast(lm->get_right()).get() &&
                           constant_cast(rm->get_right()).get()) {
                    return (lm->get_right()/rm->get_right())*(lm->get_left()/rm->get_left());
                }

                if (lm->get_left()->is_match(rm->get_left())) {
                    return lm->get_right()/rm->get_right();
                } else if (lm->get_left()->is_match(rm->get_right())) {
                    return lm->get_right()/rm->get_left();
                } else if (lm->get_right()->is_match(rm->get_left())) {
                    return lm->get_left()/rm->get_right();
                } else if (lm->get_right()->is_match(rm->get_right())) {
                    return lm->get_left()/rm->get_left();
                }
            }

//  (a/b)/c -> a/(b*c)
            auto ld = divide_cast(this->left);
            if (ld.get()) {
                return ld->get_left()/(ld->get_right()*this->right);
            }

//  Assume variables, sqrt of variables, and powers of variables are on the
//  right.
//  (a*v)/c -> a*(v/c)
            if (lm.get() && is_variable_like(lm->get_right()) &&
                !is_variable_like(lm->get_left())) {
                return lm->get_left()*(lm->get_right()/this->right);
            }

//  (c*v1)/v2 -> c*(v1/v2)
            if (lm.get() && constant_cast(lm->get_left()).get()) {
                return lm->get_left()*(lm->get_right()/this->right);
            }

//  Power reductions. Reduced cases like a^b/a^c == a^(b - c).
            auto lp = pow_cast(this->left);
            auto rp = pow_cast(this->right);
            if (lp.get()) {
//  a^b/a -> a^(b - 1)
                if (lp->get_left()->is_match(this->right)) {
                    return pow(lp->get_left(),
                               lp->get_right() - one<typename LN::base> ());
                }

//  a^b/a^c -> a^(b - c)
                if (rp.get() && lp->get_left()->is_match(rp->get_left())) {
                    return pow(lp->get_left(),
                               lp->get_right() - rp->get_right());
                }

//  a^b/sqrt(a) -> a^(b - 1/2)
                auto rsq = sqrt_cast(this->right);
                if (rsq.get() && lp->get_left()->is_match(rsq->get_arg())) {
                    return pow(lp->get_left(),
                               lp->get_right() - constant(static_cast<typename LN::base> (0.5)));
                }
            } else {
//  a/sqrt(a) -> sqrt(a)
                auto rsq = sqrt_cast(this->right);
                if (rsq.get() && this->left->is_match(rsq->get_arg())) {
                    return this->right;
                }
            }
            if (rp.get()) {
//  a/a^b -> a^(1 - b)
                if (rp->get_left()->is_match(this->left)) {
                    return pow(rp->get_left(),
                               one<typename LN::base> () - rp->get_right());
                }

//  sqrt(a)/a^b -> a^(1/2 - b)
                auto lsq = sqrt_cast(this->left);
                if (lsq.get() && rp->get_left()->is_match(lsq->get_arg())) {
                    return pow(rp->get_left(),
                               constant(static_cast<typename LN::base> (0.5)) - rp->get_right());
                }
            } else {
//  sqrt(a)/a -> 1.0/sqrt(a)
                auto lsq = sqrt_cast(this->left);
                if (lsq.get() && this->right->is_match(lsq->get_arg())) {
                    return one<typename LN::base> ()/this->left;
                }
            }
#endif
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d div(n,d)/dx = dn/dx*1/d - n*/(d*d)*db/dx
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::base>
        df(shared_leaf<typename LN::base> x) final {
            if (this->is_match(x)) {
                return one<typename LN::base> ();
            }

            return this->left->df(x)/this->right -
                   this->left*this->right->df(x)/(this->right*this->right);
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in] stream    String buffer stream.
///  @params[in] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::base> compile(std::stringstream &stream,
                                                       jit::register_map<LN> &registers) final {
            if (registers.find(this) == registers.end()) {
                shared_leaf<typename LN::base> l = this->left->compile(stream, registers);
                shared_leaf<typename RN::base> r = this->right->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<typename LN::base> (stream);
                //std::cout << ((registers.find(r.get()) == registers.end()) ? "True" : registers[r.get()])
                //          << std::endl;
                stream << " " << registers[this] << " = "
                       << registers[l.get()] << "/"
                       << registers[r.get()] << ";"
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
        virtual bool is_match(shared_leaf<typename LN::base> x) final {
            if (this == x.get()) {
                return true;
            }

            auto x_cast = divide_cast(x);
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
            std::cout << "\\frac{";
            this->left->to_latex();
            std::cout << "}{";
            this->right->to_latex();
            std::cout << "}";
        }
    };

//------------------------------------------------------------------------------
///  @brief Build divide node from two leaves.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    shared_leaf<typename LN::base> divide(std::shared_ptr<LN> l,
                                          std::shared_ptr<RN> r) {
        return std::make_shared<divide_node<LN, RN>> (l, r)->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Build divide operator from two leaves.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    shared_leaf<typename LN::base> operator/(std::shared_ptr<LN> l,
                                             std::shared_ptr<RN> r) {
        return divide<LN, RN> (l, r);
    }

///  Convenience type alias for shared divide nodes.
    template<typename LN, typename RN>
    using shared_divide = std::shared_ptr<divide_node<LN, RN>>;

//------------------------------------------------------------------------------
///  @brief Cast to a divide node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename N>
    shared_divide<N, N> divide_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<divide_node<N, N>> (x);
    }

//******************************************************************************
//  fused multiply add node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A fused multiply add node.
///
///  Note use templates here to defer this so it can use the operator functions.
//------------------------------------------------------------------------------
    template<typename LN, typename MN, typename RN>
    class fma_node : public triple_node<typename LN::base> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a fused multiply add node.
///
///  @params[in] l Left branch.
///  @params[in] m Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
        fma_node(std::shared_ptr<LN> l,
                 std::shared_ptr<MN> m,
                 std::shared_ptr<RN> r) :
        triple_node<typename LN::base> (l, m, r) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of fused multiply add.
///
///  result = l*m + r
///
///  @returns The value of l*m + r.
//------------------------------------------------------------------------------
        virtual backend::buffer<typename LN::base> evaluate() final {
            backend::buffer<typename LN::base> l_result = this->left->evaluate();
            backend::buffer<typename RN::base> r_result = this->right->evaluate();

//  If all the elements on the left are zero, return the leftside without
//  revaluating the rightside.
            if (l_result.is_zero()) {
                return r_result;
            }

            backend::buffer<typename MN::base> m_result = this->middle->evaluate();
            return fma(l_result, m_result, r_result);
        }

//------------------------------------------------------------------------------
///  @brief Reduce a fused multiply add node.
///
///  @returns A reduced addition node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::base> reduce() final {
#ifdef USE_REDUCE
            auto l = constant_cast(this->left);
            auto m = constant_cast(this->middle);
            auto r = constant_cast(this->right);

            if ((l.get() && l->is(0)) ||
                (m.get() && m->is(0)) ) {
                return this->right;
            } else if (r.get() && r->is(0)) {
                return this->left*this->middle;
            } else if (l.get() && m.get() && r.get()) {
                return constant(this->evaluate());
            } else if (l.get() && m.get()) {
                backend::buffer<typename LN::base> l_result = this->left->evaluate();
                backend::buffer<typename MN::base> m_result = this->middle->evaluate();
                return constant(l_result*m_result) + this->right;
            }

//  Common factor reduction. If the left and right are both multiply nodes check
//  for a common factor. So you can change a*b + (a*c) -> a*(b + c).
            auto rm = multiply_cast(this->right);
            if (rm.get()) {
                if (rm->get_left()->is_match(this->left)) {
                    return this->left*(this->middle + rm->get_right());
                } else if (rm->get_left()->is_match(this->middle)) {
                    return this->middle*(this->left + rm->get_right());
                } else if (rm->get_right()->is_match(this->left)) {
                    return this->left*(this->middle + rm->get_left());
                } else if (rm->get_right()->is_match(this->middle)) {
                    return this->middle*(this->left + rm->get_left());
                }
            }

//  Handle cases like.
//  fma(c1*a,b,c2*d) -> c1*(a*b + c2/c1*d)
            auto lm = multiply_cast(this->left);
            if (lm.get() && rm.get()) {
                auto rmc = constant_cast(rm->get_left());
                if (rmc.get()) {
                    return lm->get_left()*fma(lm->get_right(),
                                              this->middle,
                                              (rm->get_left()/lm->get_left())*rm->get_right());
                }
            }
//  fma(c1*a,b,c2/d) -> c1*(a*b + c1/(c2*d))
//  fma(c1*a,b,d/c2) -> c1*(a*b + d/(c1*c2))
            auto rd = divide_cast(this->right);
            if (lm.get() && rd.get()) {
                if (constant_cast(rd->get_left()).get() ||
                    constant_cast(rd->get_right()).get()) {
                    return lm->get_left()*fma(lm->get_right(),
                                              this->middle,
                                              rd->get_left()/(lm->get_left()*rd->get_right()));
                }
            }

//  Handle cases like.
//  fma(a,v1,b*v2) -> (a + b*v1/v2)*v1
//  fma(a,v1,c*b*v2) -> (a + c*b*v1/v2)*v1
            if (rm.get()) {
                if (is_same_variable_like(this->middle, rm->get_right())) {
                    return (this->left + rm->get_left()*this->middle/rm->get_right()) *
                           this->middle*rm->get_right();
                }
                auto rmm = multiply_cast(rm->get_right());
                if (rmm.get() &&
                    is_same_variable_like(this->middle, rmm->get_right())) {
                    return (this->left + rm->get_left()*rmm->get_left()*this->middle/rmm->get_right()) *
                           this->middle;
                }
            }

//  Promote constants out to the left.
            if (l.get() && r.get()) {
                return this->left*(this->middle + this->right/this->left);
            }
#endif
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d fma(a,b,c)/dx = da*b/dx + dc/dx = da/dx*b + a*db/dx + dc/dx
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::base>
        df(shared_leaf<typename LN::base> x) final {
            if (this->is_match(x)) {
                return one<typename LN::base> ();
            }

            auto temp_right = fma(this->left,
                                  this->middle->df(x),
                                  this->right->df(x));

            return fma(this->left->df(x),
                       this->middle,
                       temp_right);
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in] stream    String buffer stream.
///  @params[in] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::base> compile(std::stringstream &stream,
                                                       jit::register_map<LN> &registers) final {
            if (registers.find(this) == registers.end()) {
                shared_leaf<typename LN::base> l = this->left->compile(stream, registers);
                shared_leaf<typename MN::base> m = this->middle->compile(stream, registers);
                shared_leaf<typename RN::base> r = this->right->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<typename LN::base> (stream);
                stream << " " << registers[this] << " = ";
                if constexpr (jit::is_complex<typename LN::base> ()) {
                    stream << registers[l.get()] << "*"
                           << registers[m.get()] << " + "
                           << registers[r.get()] << ";"
                           << std::endl;
                } else {
                    stream << "fma("
                           << registers[l.get()] << ", "
                           << registers[m.get()] << ", "
                           << registers[r.get()] << ");"
                           << std::endl;
                }
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @params[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<typename LN::base> x) final {
            if (this == x.get()) {
                return true;
            }

            auto x_cast = fma_cast(x);
            if (x_cast.get()) {
                return this->left->is_match(x_cast->get_left()) &&
                       this->middle->is_match(x_cast->get_middle()) &&
                       this->right->is_match(x_cast->get_right());
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const final {
            std::cout << "\\left(";
            if (add_cast(this->left).get() ||
                subtract_cast(this->left).get()) {
                std::cout << "\\left(";
                this->left->to_latex();
                std::cout << "\\right)";
            } else {
                this->left->to_latex();
            }
            std::cout << " ";
            if (add_cast(this->right).get() ||
                subtract_cast(this->right).get()) {
                std::cout << "\\left(";
                this->middle->to_latex();
                std::cout << "\\right)";
            } else {
                this->middle->to_latex();
            }
            std::cout << "+";
            this->right->to_latex();
            std::cout << "\\right)";
        }
    };

//------------------------------------------------------------------------------
///  @brief Build fused multiply add node.
///
///  @params[in] l Left branch.
///  @params[in] m Middle branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename MN, typename RN>
    shared_leaf<typename LN::base> fma(std::shared_ptr<LN> l,
                                       std::shared_ptr<MN> m,
                                       std::shared_ptr<RN> r) {
        return std::make_shared<fma_node<LN, MN, RN>> (l, m, r)->reduce();
    }

///  Convenience type alias for shared add nodes.
    template<typename LN, typename MN, typename RN>
    using shared_fma = std::shared_ptr<fma_node<LN, MN, RN>>;

//------------------------------------------------------------------------------
///  @brief Cast to a fma node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename N>
    shared_fma<N, N, N> fma_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<fma_node<N, N, N>> (x);
    }
}

#endif /* arithmetic_h */
