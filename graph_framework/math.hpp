//------------------------------------------------------------------------------
///  @file math.hpp
///  @brief Defined basic math functions.
//------------------------------------------------------------------------------


#ifndef math_h
#define math_h

#include <cmath>

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
    class sqrt_node : public straight_node {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a sqrt node.
///
///  @param[in] x Argument.
//------------------------------------------------------------------------------
        sqrt_node(std::shared_ptr<N> x) :
        straight_node(x->reduce()) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of sqrt.
///
///  result = sqrt(x)
///
///  @returns The value of sqrt(x).
//------------------------------------------------------------------------------
        virtual std::vector<double> evaluate() final {
            std::vector<double> result = this->arg->evaluate();
            for (double &e : result) {
                e = sqrt(e);
            }
            return result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the sqrt(x).
///
///  @returns Reduced graph from sine.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> reduce() final {
            if (std::dynamic_pointer_cast<constant_node> (this->arg)) {
                return constant(this->evaluate());
            } else {
                return this->shared_from_this();
            }
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d sqrt(a)/dx = 1/(2*sqrt(x))
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> df(std::shared_ptr<leaf_node> x) final {
            return this->arg->df(x)/(constant(2)*this->shared_from_this());
        }
    };

//------------------------------------------------------------------------------
///  @brief Define sqrt convience function.
///
///  @param[in] x Argument.
///  @returns A reduced sin node.
//------------------------------------------------------------------------------
    template<typename N>
    std::shared_ptr<leaf_node> sqrt(std::shared_ptr<N> x) {
        return (std::make_shared<sqrt_node<N>> (x))->reduce();
    }
}

#endif /* math_h */
