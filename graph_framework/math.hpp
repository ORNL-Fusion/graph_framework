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
        virtual std::shared_ptr<leaf_node<typename N::backend>> reduce() final {
            if (std::dynamic_pointer_cast<constant_node<typename N::backend>> (this->arg)) {
                return constant(this->evaluate());
            } else {
                return this->shared_from_this();
            }
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d sqrt(a)/dx = 1/(2*sqrt(a))da/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<typename N::backend>>
        df(std::shared_ptr<leaf_node<typename N::backend>> x) final {
            if (x.get() == this) {
                return constant<typename N::backend> (1);
            } else {
                return this->arg->df(x) /
                       (constant<typename N::backend> (2)*this->shared_from_this());
            }
        }
    };

//------------------------------------------------------------------------------
///  @brief Define sqrt convience function.
///
///  @param[in] x Argument.
///  @returns A reduced sqrt node.
//------------------------------------------------------------------------------
    template<typename N>
    std::shared_ptr<leaf_node<typename N::backend>> sqrt(std::shared_ptr<N> x) {
        return (std::make_shared<sqrt_node<N>> (x))->reduce();
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
        virtual std::shared_ptr<leaf_node<typename N::backend>> reduce() final {
            if (std::dynamic_pointer_cast<constant_node<typename N::backend>> (this->arg)) {
                return constant(this->evaluate());
            } else {
                return this->shared_from_this();
            }
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d exp(a)/dx = exp(a)*da/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<typename N::backend>>
        df(std::shared_ptr<leaf_node<typename N::backend>> x) final {
            if (x.get() == this) {
                return constant<typename N::backend> (1);
            } else {
                return this->shared_from_this()*this->arg->df(x);
            }
        }
    };

//------------------------------------------------------------------------------
///  @brief Define exp convience function.
///
///  @param[in] x Argument.
///  @returns A reduced exp node.
//------------------------------------------------------------------------------
    template<typename N>
    std::shared_ptr<leaf_node<typename N::backend>> exp(std::shared_ptr<N> x) {
        return (std::make_shared<exp_node<N>> (x))->reduce();
    }
}

#endif /* math_h */
