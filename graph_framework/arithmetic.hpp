//------------------------------------------------------------------------------
///  @file arithmetic.hpp
///  @brief Basic arithmetic operations.
///
///  Defines basic operators.
//------------------------------------------------------------------------------

#ifndef arithmetic_h
#define arithmetic_h

#include "node.hpp"

namespace graph {
//******************************************************************************
//  Add node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief An addition node.
///
///  Note use templates here to defer this so it can use the operator functions.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    class add_node : public branch_node<typename LN::backend> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct an addition node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        add_node(std::shared_ptr<LN> l,
                 std::shared_ptr<RN> r) :
        branch_node<typename LN::backend> (l, r) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of addition.
///
///  result = l + r
///
///  @returns The value of l + r.
//------------------------------------------------------------------------------
        virtual typename LN::backend evaluate() final {
            typename LN::backend l_result = this->left->evaluate();
            typename RN::backend r_result = this->right->evaluate();
            return l_result + r_result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce an addition node.
///
///  @returns A reduced addition node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::backend> reduce() final {
//  Idenity reductions.
            if (this->left->is_match(this->right)) {
                return constant<typename LN::backend> (2)*this->left;
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
            }

//  Common denominator reduction. If the left and right are both divide nodes
//  for a common denominator. So you can change a/b + c/b -> (a + c)/d.
            auto ld = divide_cast(this->left);
            auto rd = divide_cast(this->right);

            if (ld.get() && rd.get() &&
                ld->get_right()->is_match(rd->get_right())) {
                return (ld->get_left() + rd->get_left())/ld->get_right();
            }

//  Fused multiply add reductions.
            auto m = multiply_cast(this->left);

            if (m.get()) {
                return fma<leaf_node<typename LN::backend>,
                           leaf_node<typename LN::backend>,
                           leaf_node<typename RN::backend>> (m->get_left(),
                                                             m->get_right(),
                                                             this->right);
            }

//  Handel cases like:
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

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d add(a,b)/dx = da/dx + db/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::backend>
        df(shared_leaf<typename LN::backend> x) final {
            if (this->is_match(x)) {
                return constant<typename LN::backend> (1);
            } else {
                return this->left->df(x) + this->right->df(x);
            }
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
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    shared_leaf<typename LN::backend> add(std::shared_ptr<LN> l,
                                          std::shared_ptr<RN> r) {
        return (std::make_shared<add_node<LN, RN>> (l, r))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Build add node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    shared_leaf<typename LN::backend> operator+(std::shared_ptr<LN> l,
                                                std::shared_ptr<RN> r) {
        return add<LN, RN> (l, r);
    }

///  Convience type alias for shared add nodes.
    template<typename LN, typename RN>
    using shared_add = std::shared_ptr<add_node<LN, RN>>;

//------------------------------------------------------------------------------
///  @brief Cast to a add node.
///
///  @param[in] x Leaf node to attempt cast.
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
    class subtract_node : public branch_node<typename LN::backend> {
    public:
//------------------------------------------------------------------------------
///  @brief Consruct a subtraction node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        subtract_node(std::shared_ptr<LN> l,
                      std::shared_ptr<RN> r) :
        branch_node<typename LN::backend> (l, r) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of subtraction.
///
///  result = l - r
///
///  @returns The value of l - r.
//------------------------------------------------------------------------------
        virtual typename LN::backend evaluate() final {
            typename LN::backend l_result = this->left->evaluate();
            typename RN::backend r_result = this->right->evaluate();
            return l_result - r_result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce an subtraction node.
///
///  @returns A reduced subtraction node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::backend> reduce() final {
//  Idenity reductions.
            if (this->left->is_match(this->right)) {
                auto l = constant_cast(this->left);
                if (l.get() && l->is(0)) {
                    return this->left;
                }

                return constant<typename LN::backend> (0);
            }

//  Constant reductions.
            auto l = constant_cast(this->left);
            auto r = constant_cast(this->right);

            if (l.get() && l->is(0)) {
                return constant<typename LN::backend> (-1)*this->right;
            } else if (r.get() && r->is(0)) {
                return this->left;
            } else if (l.get() && r.get()) {
                return constant(this->evaluate());
            }

//  Common factor reduction. If the left and right are both muliply nodes check
//  for a common factor. So you can change a*b - a*c -> a*(b - c).
            auto lm = multiply_cast(this->left);
            auto rm = multiply_cast(this->right);

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
            }

//  Common denominator reduction. If the left and right are both divide nodes
//  for a common denominator. So you can change a/b - c/b -> (a - c)/d.
            auto ld = divide_cast(this->left);
            auto rd = divide_cast(this->right);

            if (ld.get() && rd.get() &&
                ld->get_right()->is_match(rd->get_right())) {
                return (ld->get_left() - rd->get_left())/ld->get_right();
            }

//  Handel cases like:
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
            
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d sub(a,b)/dx = da/dx - db/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::backend>
        df(shared_leaf<typename LN::backend> x) final {
            if (this->is_match(x)) {
                return constant<typename LN::backend> (1);
            } else {
                return this->left->df(x) - this->right->df(x);
            }
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
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    shared_leaf<typename LN::backend> subtract(std::shared_ptr<LN> l,
                                               std::shared_ptr<RN> r) {
        return (std::make_shared<subtract_node<LN, RN>> (l, r))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Build subtract operator from two leaves.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    shared_leaf<typename LN::backend> operator-(std::shared_ptr<LN> l,
                                                std::shared_ptr<RN> r) {
        return subtract<LN, RN> (l, r);
    }

///  Convience type alias for shared subtract nodes.
    template<typename LN, typename RN>
    using shared_subtract = std::shared_ptr<subtract_node<LN, RN>>;

//------------------------------------------------------------------------------
///  @brief Cast to a subtract node.
///
///  @param[in] x Leaf node to attempt cast.
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
    class multiply_node : public branch_node<typename LN::backend> {
    public:
//------------------------------------------------------------------------------
///  @brief Consruct a multiplcation node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        multiply_node(std::shared_ptr<LN> l,
                      std::shared_ptr<RN> r) :
        branch_node<typename LN::backend> (l, r) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of multiplcation.
///
///  result = l*r
///
///  @returns The value of l*r.
//------------------------------------------------------------------------------
        virtual typename LN::backend evaluate() final {
            typename LN::backend l_result = this->left->evaluate();

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

            typename LN::backend r_result = this->right->evaluate();
            return l_result*r_result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce an multiplcation node.
///
///  @returns A reduced multiplcation node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::backend> reduce() final {
            auto l = constant_cast(this->left);
            auto r = constant_cast(this->right);

            if (l.get() && l->is(1)) {
                return this->right;
            } else if (l.get() &&  l->is(0)) {
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

//  Move variables to the right.
            auto lv = variable_cast(this->left);
            auto rv = variable_cast(this->right);
            if (lv.get() && !rv.get()) {
                return this->right*this->left;
            }

//  Reduce constants multiplied by fused multiply add nodes.
            auto rfma = fma_cast(this->right);
            if (l.get() && rfma.get()) {
                return fma(this->left*rfma->get_left(),
                           rfma->get_middle(),
                           this->left*rfma->get_right());
            }
            auto lfma = fma_cast(this->left);
            if (r.get() && lfma.get()) {
                return fma(this->right*lfma->get_left(),
                           lfma->get_middle(),
                           this->right*lfma->get_right());
            }

//  Reduce x*x to x^2
            if (this->left->is_match(this->right)) {
                return pow(this->left, constant<typename LN::backend> (2.0));
            }

//  Gather common terms. (a*b)*a -> (a*a)*b, (b*a)*a -> (a*a)*b,
//  a*(a*b) -> (a*a)*b, a*(b*a) -> (a*a)*b
            auto lm = multiply_cast(this->left);
            if (lm.get()) {
                if (this->right->is_match(lm->get_left())) {
                    return (this->right*lm->get_left())*lm->get_right();
                } else if (this->right->is_match(lm->get_right())) {
                    return (this->right*lm->get_right())*lm->get_left();
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
//  (c*v1)*v2 -> c*(v1*v2)
            if (rm.get() && constant_cast(rm->get_left()).get()) {
                return rm->get_left()*(this->left*rm->get_right());
            } else if (lm.get() && constant_cast(lm->get_left()).get()) {
                return lm->get_left()*(lm->get_right()*this->right);
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
                if (lp->get_left()->is_match(this->right)) {
                    return pow(lp->get_left(),
                               lp->get_right() + constant<typename LN::backend> (1.0));
                }

                if (rp.get() && lp->get_left()->is_match(rp->get_left())) {
                    return pow(lp->get_left(),
                               lp->get_right() + rp->get_right());
                }
                
                auto rsq = sqrt_cast(this->right);
                if (rsq.get() && lp->get_left()->is_match(rsq->get_arg())) {
                    return pow(lp->get_left(),
                               lp->get_right() + constant<typename LN::backend> (0.5));
                }
            }
            if (rp.get()) {
                if (rp->get_left()->is_match(this->left)) {
                    return pow(rp->get_left(),
                               rp->get_right() + constant<typename LN::backend> (1.0));
                }
                
                auto lsq = sqrt_cast(this->left);
                if (lsq.get() && rp->get_left()->is_match(lsq->get_arg())) {
                    return pow(rp->get_left(),
                               rp->get_right() + constant<typename LN::backend> (0.5));
                }
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d mul(a,b)/dx = da/dx*b + a*db/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::backend>
        df(shared_leaf<typename LN::backend> x) final {
            if (this->is_match(x)) {
                return constant<typename LN::backend> (1);
            }

            return this->left->df(x)*this->right +
                   this->left*this->right->df(x);
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
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    shared_leaf<typename LN::backend> multiply(std::shared_ptr<LN> l,
                                               std::shared_ptr<RN> r) {
        return (std::make_shared<multiply_node<LN, RN>> (l, r))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Build multiply operator from two leaves.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    shared_leaf<typename LN::backend> operator*(std::shared_ptr<LN> l,
                                                std::shared_ptr<RN> r) {
        return multiply<LN, RN> (l, r);
    }

///  Convience type alias for shared multiply nodes.
    template<typename LN, typename RN>
    using shared_multiply = std::shared_ptr<multiply_node<LN, RN>>;

//------------------------------------------------------------------------------
///  @brief Cast to a multiply node.
///
///  @param[in] x Leaf node to attempt cast.
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
    class divide_node : public branch_node<typename LN::backend> {
    public:
        divide_node(std::shared_ptr<LN> n,
                    std::shared_ptr<RN> d) :
        branch_node<typename LN::backend> (n, d) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of division.
///
///  result = n/d
///
///  @returns The value of n/d.
//------------------------------------------------------------------------------
        virtual typename LN::backend evaluate() final {
            typename LN::backend l_result = this->left->evaluate();

//  If all the elements on the left are zero, return the leftside without
//  revaluating the rightside. Stop this loop early once the first non zero
//  element is encountered.
            if (l_result.is_zero()) {
                return l_result;
            }

            typename RN::backend r_result = this->right->evaluate();
            return l_result/r_result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce an division node.
///
///  @returns A reduced division node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::backend> reduce() final {
//  Constant Reductions.
            auto l = constant_cast(this->left);

            if (this->left->is_match(this->right)) {
                if (l.get() && l->is(1)) {
                    return this->left;
                }

                return constant<typename LN::backend> (1);
            }

            auto r = constant_cast(this->right);

            if ((l.get() && l->is(0)) ||
                (r.get() && r->is(1))) {
                return this->left;
            } else if (l.get() && r.get()) {
                return constant(this->evaluate());
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
//  (c1*v)/c2 -> v/c3
            if (rm.get() && l.get()) {
                if (constant_cast(rm->get_left()).get()) {
                    return (this->left/rm->get_left())/rm->get_right();
                }
            } else if (lm.get() && r.get()) {
                if (constant_cast(lm->get_left()).get()) {
                    return lm->get_right()/(this->right/lm->get_left());
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

//  (c*v1)/v2 -> c*(v1/v2)
            if (lm.get() && constant_cast(lm->get_left()).get()) {
                return lm->get_left()*(lm->get_right()/this->right);
            }

//  Power reductions. Reduced cases like a^b/a^c == a^(b - c).
            auto lp = pow_cast(this->left);
            auto rp = pow_cast(this->right);
            if (lp.get()) {
                if (lp->get_left()->is_match(this->right)) {
                    return pow(lp->get_left(),
                               lp->get_right() - constant<typename LN::backend> (1.0));
                }

                if (rp.get() && lp->get_left()->is_match(rp->get_left())) {
                    return pow(lp->get_left(),
                               lp->get_right() - rp->get_right());
                }
                            
                auto rsq = sqrt_cast(this->right);
                if (rsq.get() && lp->get_left()->is_match(rsq->get_arg())) {
                    return pow(lp->get_left(),
                               lp->get_right() - constant<typename LN::backend> (0.5));
                }
            }
            if (rp.get()) {
                if (rp->get_left()->is_match(this->left)) {
                    return pow(rp->get_left(),
                               constant<typename LN::backend> (1.0) - rp->get_right());
                }
                            
                auto lsq = sqrt_cast(this->left);
                if (lsq.get() && rp->get_left()->is_match(lsq->get_arg())) {
                    return pow(rp->get_left(),
                               constant<typename LN::backend> (0.5) - rp->get_right());
                }
            }
            
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d div(n,d)/dx = dn/dx*1/d - n*/(d*d)*db/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::backend>
        df(shared_leaf<typename LN::backend> x) final {
            if (this->is_match(x)) {
                return constant<typename LN::backend> (1);
            }

            return this->left->df(x)/this->right -
                   this->left*this->right->df(x)/(this->right*this->right);
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
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    shared_leaf<typename LN::backend> divide(std::shared_ptr<LN> l,
                                             std::shared_ptr<RN> r) {
        return std::make_shared<divide_node<LN, RN>> (l, r)->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Build divide operator from two leaves.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    shared_leaf<typename LN::backend> operator/(std::shared_ptr<LN> l,
                                                std::shared_ptr<RN> r) {
        return divide<LN, RN> (l, r);
    }

///  Convience type alias for shared divide nodes.
    template<typename LN, typename RN>
    using shared_divide = std::shared_ptr<divide_node<LN, RN>>;

//------------------------------------------------------------------------------
///  @brief Cast to a divide node.
///
///  @param[in] x Leaf node to attempt cast.
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
    class fma_node : public triple_node<typename LN::backend> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a fused multiply add node.
///
///  @param[in] l Left branch.
///  @param[in] m Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        fma_node(std::shared_ptr<LN> l,
                 std::shared_ptr<MN> m,
                 std::shared_ptr<RN> r) :
        triple_node<typename LN::backend> (l, m, r) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of fused multiply add.
///
///  result = l*m + r
///
///  @returns The value of l*m + r.
//------------------------------------------------------------------------------
        virtual typename LN::backend evaluate() final {
            typename LN::backend l_result = this->left->evaluate();
            typename RN::backend r_result = this->right->evaluate();

//  If all the elements on the left are zero, return the leftside without
//  revaluating the rightside.
            if (l_result.is_zero()) {
                return r_result;
            }

            typename MN::backend m_result = this->middle->evaluate();
            return fma(l_result, m_result, r_result);
        }

//------------------------------------------------------------------------------
///  @brief Reduce a fused multiply add node.
///
///  @returns A reduced addition node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::backend> reduce() final {
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
                typename LN::backend l_result = this->left->evaluate();
                typename MN::backend m_result = this->middle->evaluate();
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

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d fma(a,b,c)/dx = da*b/dx + dc/dx = da/dx*b + a*db/dx + dc/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename LN::backend>
        df(shared_leaf<typename LN::backend> x) final {
            if (this->is_match(x)) {
                return constant<typename LN::backend> (1);
            }

            auto temp_right = fma<LN,
                                  leaf_node<typename MN::backend>,
                                  leaf_node<typename RN::backend>> (this->left,
                                                                    this->middle->df(x),
                                                                    this->right->df(x));

            return fma<leaf_node<typename LN::backend>,
                       MN,
                       leaf_node<typename RN::backend>> (this->left->df(x),
                                                         this->middle,
                                                         temp_right);
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
            this->left->to_latex();
            std::cout << " ";
            this->middle->to_latex();
            std::cout << "+";
            this->right->to_latex();
            std::cout << "\\right)";
        }
    };

//------------------------------------------------------------------------------
///  @brief Build fused multiply add node.
///
///  @param[in] l Left branch.
///  @param[in] m Middle branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename MN, typename RN>
    shared_leaf<typename LN::backend> fma(std::shared_ptr<LN> l,
                                          std::shared_ptr<MN> m,
                                          std::shared_ptr<RN> r) {
        return std::make_shared<fma_node<LN, MN, RN>> (l, m, r)->reduce();
    }

///  Convience type alias for shared add nodes.
    template<typename LN, typename MN, typename RN>
    using shared_fma = std::shared_ptr<fma_node<LN, MN, RN>>;

//------------------------------------------------------------------------------
///  @brief Cast to a fma node.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename N>
    shared_fma<N, N, N> fma_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<fma_node<N, N, N>> (x);
    }
}

#endif /* arithmetic_h */
