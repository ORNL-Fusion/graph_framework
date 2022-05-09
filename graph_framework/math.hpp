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
            if (constant_cast(this->arg)) {
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
            if (x_cast != nullptr) {
                return this->arg->is_match(x_cast->get_arg());
            } else {
                return false;
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
    shared_leaf<typename N::backend> sqrt(std::shared_ptr<N> x) {
        return (std::make_shared<sqrt_node<N>> (x))->reduce();
    }

///  Convience type alias for shared sqrt nodes.
    template<typename N>
    using shared_sqrt = std::shared_ptr<sqrt_node<N>>;

//------------------------------------------------------------------------------
///  @brief Cast to a sqrt node.
///
///  @param[in] x Leaf node to attempt cast.
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
            if (constant_cast(this->arg)) {
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
        virtual shared_leaf<typename N::backend>
        df(shared_leaf<typename N::backend> x) final {
            if (this->is_match(x)) {
                return constant<typename N::backend> (1);
            } else {
                return this->shared_from_this()*this->arg->df(x);
            }
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
            if (x_cast != nullptr) {
                return this->arg->is_match(x_cast->get_arg());
            } else {
                return false;
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
    shared_leaf<typename N::backend> exp(std::shared_ptr<N> x) {
        return (std::make_shared<exp_node<N>> (x))->reduce();
    }

///  Convience type alias for shared exp nodes.
    template<typename N>
    using shared_exp = std::shared_ptr<exp_node<N>>;

//------------------------------------------------------------------------------
///  @brief Cast to a exp node.
///
///  @param[in] x Leaf node to attempt cast.
//------------------------------------------------------------------------------
    template<typename N>
    shared_exp<N> exp_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<exp_node<N>> (x);
    }
}

#endif /* math_h */
