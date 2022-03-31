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
    class add_node : public branch_node {
    public:
//------------------------------------------------------------------------------
///  @brief Construct an addition node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        add_node(std::shared_ptr<LN> l,
                 std::shared_ptr<RN> r) :
        branch_node(l, r) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of addition.
///
///  result = l + r
///
///  @returns The value of l + r.
//------------------------------------------------------------------------------
        virtual std::vector<double> evaluate() final {
            const std::vector<double> l_result = this->left->evaluate();
            const std::vector<double> r_result = this->right->evaluate();

            if (l_result.size()*r_result.size() == 1) {
                return std::vector<double> (1, l_result.at(0) + r_result.at(0));
            } else if (l_result.size() > 0) {
                std::vector<double> result(l_result.size());
                for (size_t i = 0, ie = l_result.size(); i < ie; i++) {
                    result[i] = l_result.at(i) + r_result.at(0);
                }
                return result;
            } else {
                std::vector<double> result(r_result.size());
                for (size_t i = 0, ie = r_result.size(); i < ie; i++) {
                    result[i] = l_result.at(0) + r_result.at(i);
                }
                return result;
            }
        }

//------------------------------------------------------------------------------
///  @brief Reduce an addition node.
///
///  @returns A reduced addition node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> reduce() final {
            auto l = std::dynamic_pointer_cast<constant_node> (this->left);
            auto r = std::dynamic_pointer_cast<constant_node> (this->right);

            if (l.get() && r.get()) {
                return constant(this->evaluate());
            } else if (l.get() && l->is(0)) {
                return this->right;
            } else if (r.get() && r->is(0)) {
                return this->left;
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
        virtual std::shared_ptr<leaf_node> df(std::shared_ptr<leaf_node> x) final {
            return this->left->df(x) + this->right->df(x);
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
    std::shared_ptr<leaf_node> add(std::shared_ptr<LN> l,
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
    std::shared_ptr<leaf_node> operator+(std::shared_ptr<LN> l,
                                         std::shared_ptr<RN> r) {
        return add<LN, RN> (l, r);
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
    class subtract_node : public branch_node {
    public:
//------------------------------------------------------------------------------
///  @brief Consruct a subtraction node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        subtract_node(std::shared_ptr<LN> l,
                      std::shared_ptr<RN> r) :
        branch_node(l, r) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of subtraction.
///
///  result = l - r
///
///  @returns The value of l - r.
//------------------------------------------------------------------------------
        virtual std::vector<double> evaluate() final {
            const std::vector<double> l_result = this->left->evaluate();
            const std::vector<double> r_result = this->right->evaluate();

            if (l_result.size()*r_result.size() == 1) {
                return std::vector<double> (1, l_result.at(0) - r_result.at(0));
            } else if (l_result.size() > 0) {
                std::vector<double> result(l_result.size());
                for (size_t i = 0, ie = l_result.size(); i < ie; i++) {
                    result[i] = l_result.at(i) - r_result.at(0);
                }
                return result;
            } else {
                std::vector<double> result(r_result.size());
                for (size_t i = 0, ie = r_result.size(); i < ie; i++) {
                    result[i] = l_result.at(0) - r_result.at(i);
                }
                return result;
            }
        }

//------------------------------------------------------------------------------
///  @brief Reduce an subtraction node.
///
///  @returns A reduced subtraction node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> reduce() final {
            auto l = std::dynamic_pointer_cast<constant_node> (this->left);
            auto r = std::dynamic_pointer_cast<constant_node> (this->right);

            if (l.get() && r.get()) {
                return constant(this->evaluate());
            } else if (l.get() && l->is(0)) {
                return constant(-1)*this->right;
            } else if (r.get() && r->is(0)) {
                return this->left;
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
        virtual std::shared_ptr<leaf_node> df(std::shared_ptr<leaf_node> x) final {
            return this->left->df(x) - this->right->df(x);
        }
    };

//------------------------------------------------------------------------------
///  @brief Build subtract node from two leaves.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    std::shared_ptr<leaf_node> subtract(std::shared_ptr<LN> l,
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
    std::shared_ptr<leaf_node> operator-(std::shared_ptr<LN> l,
                                         std::shared_ptr<RN> r) {
        return subtract<LN, RN> (l, r);
    }

//******************************************************************************
//  Multiply node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A multiplcation node.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    class multiply_node : public branch_node {
    public:
//------------------------------------------------------------------------------
///  @brief Consruct a multiplcation node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        multiply_node(std::shared_ptr<LN> l,
                      std::shared_ptr<RN> r) :
        branch_node(l, r) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of multiplcation.
///
///  result = l*r
///
///  @returns The value of l*r.
//------------------------------------------------------------------------------
        virtual std::vector<double> evaluate() final {
            const std::vector<double> l_result = this->left->evaluate();
            const std::vector<double> r_result = this->right->evaluate();

            if (l_result.size()*r_result.size() == 1) {
                return std::vector<double> (1, l_result.at(0)*r_result.at(0));
            } else if (l_result.size() > 0) {
                std::vector<double> result(l_result.size());
                for (size_t i = 0, ie = l_result.size(); i < ie; i++) {
                    result[i] = l_result.at(i)*r_result.at(0);
                }
                return result;
            } else {
                std::vector<double> result(r_result.size());
                for (size_t i = 0, ie = r_result.size(); i < ie; i++) {
                    result[i] = l_result.at(0)*r_result.at(i);
                }
                return result;
            }
        }

//------------------------------------------------------------------------------
///  @brief Reduce an multiplcation node.
///
///  @returns A reduced multiplcation node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> reduce() final {
            auto l = std::dynamic_pointer_cast<constant_node> (this->left);
            auto r = std::dynamic_pointer_cast<constant_node> (this->right);

            if (l.get() && r.get()) {
                return constant(this->evaluate());
            } else if (l.get()) {
                if (l->is(1)) {
                    return this->right;
                } else if (l->is(0)) {
                    return constant(0);
                }
            } else if (r.get()) {
                if (r->is(1)) {
                    return this->left;
                } else if (r->is(0)) {
                    return constant(0);
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
        virtual std::shared_ptr<leaf_node> df(std::shared_ptr<leaf_node> x) final {
            return this->left->df(x)*this->right +
                   this->left*this->right->df(x);
        }
    };

//------------------------------------------------------------------------------
///  @brief Build multiply node from two leaves.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    std::shared_ptr<leaf_node> multiply(std::shared_ptr<LN> l,
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
    std::shared_ptr<leaf_node> operator*(std::shared_ptr<LN> l,
                                         std::shared_ptr<RN> r) {
        return multiply<LN, RN> (l, r);
    }

//******************************************************************************
//  Divide node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A division node.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    class divide_node : public branch_node {
    public:
        divide_node(std::shared_ptr<LN> n,
                    std::shared_ptr<RN> d) :
        branch_node(n, d) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of division.
///
///  result = n/d
///
///  @returns The value of n/d.
//------------------------------------------------------------------------------
        virtual std::vector<double> evaluate() final {
            const std::vector<double> l_result = this->left->evaluate();
            const std::vector<double> r_result = this->right->evaluate();

            if (l_result.size()*r_result.size() == 1) {
                return std::vector<double> (1, l_result.at(0)/r_result.at(0));
            } else if (l_result.size() > 0) {
                std::vector<double> result(l_result.size());
                for (size_t i = 0, ie = l_result.size(); i < ie; i++) {
                    result[i] = l_result.at(i)/r_result.at(0);
                }
                return result;
            } else {
                std::vector<double> result(r_result.size());
                for (size_t i = 0, ie = r_result.size(); i < ie; i++) {
                    result[i] = l_result.at(0)/r_result.at(i);
                }
                return result;
            }
        }

//------------------------------------------------------------------------------
///  @brief Reduce an division node.
///
///  @returns A reduced division node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> reduce() final {
            if (this->left.get() == this->right.get()) {
                return constant(1);
            }

            auto l = std::dynamic_pointer_cast<constant_node> (this->left);
            auto r = std::dynamic_pointer_cast<constant_node> (this->right);

            if (l.get() && r.get()) {
                return constant(this->evaluate());
            } else if ((l.get() && l->is(0)) ||
                       (r.get() && r->is(1))) {
                return this->left;
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
        virtual std::shared_ptr<leaf_node> df(std::shared_ptr<leaf_node> x) final {
            return this->left->df(x)/this->right -
                   this->left*this->right->df(x)/(this->right*this->right);
        }
    };

//------------------------------------------------------------------------------
///  @brief Build divide node from two leaves.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename LN, typename RN>
    std::shared_ptr<leaf_node> divide(std::shared_ptr<LN> l,
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
    std::shared_ptr<leaf_node> operator/(std::shared_ptr<LN> l,
                                         std::shared_ptr<RN> r) {
        return divide<LN, RN> (l, r);
    }
}

#endif /* arithmetic_h */
