//------------------------------------------------------------------------------
///  @file trigonometry.hpp
///  @brief Trigonometry functions.
///
///  Created by Cianciosa, Mark R. on 7/15/19.
///  Copyright © 2019 Cianciosa, Mark R. All rights reserved.
//------------------------------------------------------------------------------

#ifndef trigonometry_h
#define trigonometry_h

#include <cmath>

#include "arithmetic.hpp"

namespace graph {

//******************************************************************************
//  Sine node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a sine_node leaf.
//------------------------------------------------------------------------------
    template<typename N>
    class sine_node final : public straight_node<N> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a sine_node node.
///
///  @param[in] x Argument.
//------------------------------------------------------------------------------
        sine_node(std::shared_ptr<N> x) :
        straight_node<N> (x) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of sine.
///
///  result = sin(x)
///
///  @returns The value of sin(x).
//------------------------------------------------------------------------------
        virtual std::vector<double> evaluate() final {
            std::vector<double> result = this->arg->evaluate();
            for (double &e : result) {
                e = sin(e);
            }
            return result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the sin(x).
///
///  @returns Reduced graph from sine.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> reduce() final {
            if constexpr (std::is_same<N, zero_node>::value) {
                return this->arg();
            } else if constexpr (std::is_same<N, constant_node>::value) {
                return std::make_shared<constant_node> (this->evaluate());
            } else {
                return this->shared_from_this();
            }
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d sin(a)/dx = cos(a)*da/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> df(std::shared_ptr<leaf_node> x) final {
            return cos(this->arg)*this->arg->df(x);
        }
    };

//------------------------------------------------------------------------------
///  @brief Define sine convience function.
///
///  @param[in] x Argument.
///  @returns A reduced sin node.
//------------------------------------------------------------------------------
    template<typename N>
    std::shared_ptr<leaf_node> sin(std::shared_ptr<N> x) {
        return (std::make_shared<sine_node<N>> (x))->reduce();
    }

//******************************************************************************
//  Cosine node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a cosine_node leaf.
//------------------------------------------------------------------------------
    template<typename N>
    class cosine_node final : public straight_node<N> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a cosine_node node.
///
///  @param[in] x Argument.
//------------------------------------------------------------------------------
        cosine_node(std::shared_ptr<N> x) :
        straight_node<N> (x) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of cosine.
///
///  result = cos(x)
///
///  @returns The value of cos(x).
//------------------------------------------------------------------------------
        virtual std::vector<double> evaluate() final {
            std::vector<double> result = this->arg->evaluate();
            for (double &e : result) {
                e = cos(e);
            }
            return result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the cos(x).
///
///  @returns Reduced graph from cosine.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> reduce() final {
            if constexpr (std::is_same<N, zero_node>::value) {
                return one;
            } else if constexpr (std::is_same<N, constant_node>::value) {
                return std::make_shared<constant_node> (this->evaluate());
            } else {
                return this->shared_from_this();
            }
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d sin(a)/dx = cos(a)*da/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> df(std::shared_ptr<leaf_node> x) final {
            return std::make_shared<constant_node> (-1)*sin(this->arg)*this->arg->df(x);
        }
    };

//------------------------------------------------------------------------------
///  @brief Define cosine convience function.
///
///  @param[in] x Argument.
///  @returns A reduced cos node.
//------------------------------------------------------------------------------
    template<typename N>
    std::shared_ptr<leaf_node> cos(std::shared_ptr<N> x) {
        return (std::make_shared<cosine_node<N>> (x))->reduce();
    }

//******************************************************************************
//  Tangent node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Define tangent convience function.
///
///  tan(x) = sin(x)/cos(x)
///
///  @param[in] x Argument.
///  @returns A reduced tan node.
//------------------------------------------------------------------------------
    template<typename N>
    std::shared_ptr<leaf_node> tan(std::shared_ptr<N> x) {
        return (sin(x)/cos(x))->reduce();
    }
}

#endif /* trigonometry_h */
