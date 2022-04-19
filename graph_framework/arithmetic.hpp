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
    template<typename LN, typename RN> class multiply_node;

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
        virtual std::shared_ptr<leaf_node<typename LN::backend>> reduce() final {
//  Idenity reductions.
            if (this->left.get() == this->right.get()) {
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

            if (lm.get() != nullptr && rm.get() != nullptr) {
                if (lm->get_left().get() == rm->get_left().get()) {
                    return lm->get_left()*(lm->get_right() + rm->get_right());
                } else if (lm->get_left().get() == rm->get_right().get()) {
                    return lm->get_left()*(lm->get_right() + rm->get_left());
                } else if (lm->get_right().get() == rm->get_left().get()) {
                    return lm->get_right()*(lm->get_left() + rm->get_right());
                } else if (lm->get_right().get() == rm->get_right().get()) {
                    return lm->get_right()*(lm->get_left() + rm->get_left());
                }
            }

//  Common denominator reduction. If the left and right are both divide nodes
//  for a common denominator. So you can change a/b + c/b -> (a + c)/d.
            auto ld = divide_cast(this->left);
            auto rd = divide_cast(this->right);

            if (ld.get() != nullptr && rd.get() != nullptr &&
                ld->get_right().get() == rd->get_right().get()) {
                return (ld->get_left() + rd->get_left())/ld->get_right();
            }

//  Fused multiply add reductions.
#ifdef USE_FMA
            auto m = multiply_cast(this->left);

            if (m.get()) {
                return fma<leaf_node<typename LN::backend>,
                           leaf_node<typename LN::backend>,
                           leaf_node<typename RN::backend>> (m->get_left(),
                                                             m->get_right(),
                                                             this->right);
            }
#endif

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
        virtual std::shared_ptr<leaf_node<typename LN::backend>>
        df(std::shared_ptr<leaf_node<typename LN::backend>> x) final {
            if (x.get() == this) {
                return constant<typename LN::backend> (1);
            } else {
                return this->left->df(x) + this->right->df(x);
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
    std::shared_ptr<leaf_node<typename LN::backend> > add(std::shared_ptr<LN> l,
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
    std::shared_ptr<leaf_node<typename LN::backend>> operator+(std::shared_ptr<LN> l,
                                                               std::shared_ptr<RN> r) {
        return add<LN, RN> (l, r);
    }

//------------------------------------------------------------------------------
///  @brief Cast to a add node.
///
///  @param[in] x Leaf node to attempt cast.
//------------------------------------------------------------------------------
    template<typename LEAF>
    std::shared_ptr<add_node<LEAF, LEAF>> add_cast(std::shared_ptr<LEAF> x) {
        return std::dynamic_pointer_cast<add_node<LEAF, LEAF>> (x);
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
        virtual std::shared_ptr<leaf_node<typename LN::backend>> reduce() final {
//  Idenity reductions.
            if (this->left.get() == this->right.get()) {
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

            if (lm.get() != nullptr && rm.get() != nullptr) {
                if (lm->get_left().get() == rm->get_left().get()) {
                    return lm->get_left()*(lm->get_right() - rm->get_right());
                } else if (lm->get_left().get() == rm->get_right().get()) {
                    return lm->get_left()*(lm->get_right() - rm->get_left());
                } else if (lm->get_right().get() == rm->get_left().get()) {
                    return lm->get_right()*(lm->get_left() - rm->get_right());
                } else if (lm->get_right().get() == rm->get_right().get()) {
                    return lm->get_right()*(lm->get_left() - rm->get_left());
                }
            }

//  Common denominator reduction. If the left and right are both divide nodes
//  for a common denominator. So you can change a/b - c/b -> (a - c)/d.
            auto ld = divide_cast(this->left);
            auto rd = divide_cast(this->right);

            if (ld.get() != nullptr && rd.get() != nullptr &&
                ld->get_right().get() == rd->get_right().get()) {
                return (ld->get_left() - rd->get_left())/ld->get_right();
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
        virtual std::shared_ptr<leaf_node<typename LN::backend>>
        df(std::shared_ptr<leaf_node<typename LN::backend>> x) final {
            if (x.get() == this) {
                return constant<typename LN::backend> (1);
            } else {
                return this->left->df(x) - this->right->df(x);
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
    std::shared_ptr<leaf_node<typename LN::backend>> subtract(std::shared_ptr<LN> l,
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
    std::shared_ptr<leaf_node<typename LN::backend>> operator-(std::shared_ptr<LN> l,
                                                               std::shared_ptr<RN> r) {
        return subtract<LN, RN> (l, r);
    }

//------------------------------------------------------------------------------
///  @brief Cast to a subtract node.
///
///  @param[in] x Leaf node to attempt cast.
//------------------------------------------------------------------------------
    template<typename LEAF>
    std::shared_ptr<subtract_node<LEAF, LEAF>> subtract_cast(std::shared_ptr<LEAF> x) {
        return std::dynamic_pointer_cast<subtract_node<LEAF, LEAF>> (x);
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
        virtual std::shared_ptr<leaf_node<typename LN::backend>> reduce() final {
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

//  Common factor reduction. (a/b)*(c/a) = c/b.
            auto ld = divide_cast(this->left);
            auto rd = divide_cast(this->right);

            if (ld.get() != nullptr && rd.get() != nullptr) {
                if (ld->get_left().get() == rd->get_right().get()) {
                    return ld->get_right()/rd->get_left();
                } else if (ld->get_right().get() == rd->get_left().get()) {
                    return ld->get_left()/rd->get_right();
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
        virtual std::shared_ptr<leaf_node<typename LN::backend>>
        df(std::shared_ptr<leaf_node<typename LN::backend>> x) final {
            if (x.get() == this) {
                return constant<typename LN::backend> (1);
            } else {
                return this->left->df(x)*this->right +
                       this->left*this->right->df(x);
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
    std::shared_ptr<leaf_node<typename LN::backend>> multiply(std::shared_ptr<LN> l,
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
    std::shared_ptr<leaf_node<typename LN::backend>> operator*(std::shared_ptr<LN> l,
                                                               std::shared_ptr<RN> r) {
        return multiply<LN, RN> (l, r);
    }

//------------------------------------------------------------------------------
///  @brief Cast to a multiply node.
///
///  @param[in] x Leaf node to attempt cast.
//------------------------------------------------------------------------------
    template<typename LEAF>
    std::shared_ptr<multiply_node<LEAF, LEAF>> multiply_cast(std::shared_ptr<LEAF> x) {
        return std::dynamic_pointer_cast<multiply_node<LEAF, LEAF>> (x);
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
        virtual std::shared_ptr<leaf_node<typename LN::backend>> reduce() final {
//  Constant Reductions.
            auto l = constant_cast(this->left);

            if (this->left.get() == this->right.get()) {
                if (l.get() && l->is(1)) {
                    return this->left;
                } else {
                    return constant<typename LN::backend> (1);
                }
            }

            auto r = constant_cast(this->right);

            if ((l.get() && l->is(0)) ||
                (r.get() && r->is(1))) {
                return this->left;
            } else if (l.get() && r.get()) {
                return constant(this->evaluate());
            }

//  Common factor reduction. (a*b)/(a*c) = b/c.
            auto lm = multiply_cast(this->left);
            auto rm = multiply_cast(this->right);

            if (lm.get() != nullptr && rm.get() != nullptr) {
                if (lm->get_left().get() == rm->get_left().get()) {
                    return lm->get_right()/rm->get_right();
                } else if (lm->get_left().get() == rm->get_right().get()) {
                    return lm->get_right()/rm->get_left();
                } else if (lm->get_right().get() == rm->get_left().get()) {
                    return lm->get_left()/rm->get_right();
                } else if (lm->get_right().get() == rm->get_right().get()) {
                    return lm->get_left()/rm->get_left();
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
        virtual std::shared_ptr<leaf_node<typename LN::backend>>
        df(std::shared_ptr<leaf_node<typename LN::backend>> x) final {
            if (x.get() == this) {
                return constant<typename LN::backend> (1);
            } else {
                return this->left->df(x)/this->right -
                       this->left*this->right->df(x)/(this->right*this->right);
            }
        }
    };

//------------------------------------------------------------------------------
///  @brief Build divide node from two leaves.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    std::shared_ptr<leaf_node<typename LN::backend>> divide(std::shared_ptr<LN> l,
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
    std::shared_ptr<leaf_node<typename LN::backend>> operator/(std::shared_ptr<LN> l,
                                                               std::shared_ptr<RN> r) {
        return divide<LN, RN> (l, r);
    }

//------------------------------------------------------------------------------
///  @brief Cast to a divide node.
///
///  @param[in] x Leaf node to attempt cast.
//------------------------------------------------------------------------------
    template<typename LEAF>
    std::shared_ptr<divide_node<LEAF, LEAF>> divide_cast(std::shared_ptr<LEAF> x) {
        return std::dynamic_pointer_cast<divide_node<LEAF, LEAF>> (x);
    }

#ifdef USE_FMA
//******************************************************************************
//  fused multiply add node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief An fused multiply add node.
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
//  revaluating the rightside. Stop this loop early once the first non zero
//  element is encountered.
            if (l_result.is_zero()) {
                return r_result;
            }

            typename MN::backend m_result = this->middle->evaluate();

//  If all the elements on the left are zero, return the leftside without
//  revaluating the rightside. Stop this loop early once the first non zero
//  element is encountered.
            if (r_result.is_zero()) {
                return l_result*m_result;
            }

            return fma(l_result, m_result, r_result);
        }

//------------------------------------------------------------------------------
///  @brief Reduce an fused multiply add node.
///
///  @returns A reduced addition node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<typename LN::backend>> reduce() final {
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
        virtual std::shared_ptr<leaf_node<typename LN::backend>>
        df(std::shared_ptr<leaf_node<typename LN::backend>> x) final {
            if (x.get() == this) {
                return constant<typename LN::backend> (1);
            } else {
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
        }
    };
#endif

//------------------------------------------------------------------------------
///  @brief Build fused multiply add node.
///
///  @param[in] l Left branch.
///  @param[in] m Middle branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename MN, typename RN>
    std::shared_ptr<leaf_node<typename LN::backend>> fma(std::shared_ptr<LN> l,
                                                         std::shared_ptr<MN> m,
                                                         std::shared_ptr<RN> r) {
#ifdef USE_FMA
        return std::make_shared<fma_node<LN, MN, RN>> (l, m, r)->reduce();
#else
        return l*m + r;
#endif
    }

#ifdef USE_FMA
//------------------------------------------------------------------------------
///  @brief Cast to a fma node.
///
///  @param[in] x Leaf node to attempt cast.
//------------------------------------------------------------------------------
    template<typename LEAF>
    std::shared_ptr<fma_node<LEAF, LEAF, LEAF>> fma_cast(std::shared_ptr<LEAF> x) {
        return std::dynamic_pointer_cast<fma_node<LEAF, LEAF, LEAF>> (x);
    }
#endif
}

#endif /* arithmetic_h */
