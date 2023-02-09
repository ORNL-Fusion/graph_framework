//------------------------------------------------------------------------------
///  @file trigonometry.hpp
///  @brief Trigonometry functions.
///
///  Created by Cianciosa, Mark R. on 7/15/19.
///  Copyright © 2019 Cianciosa, Mark R. All rights reserved.
//------------------------------------------------------------------------------

#ifndef trigonometry_h
#define trigonometry_h

#include "arithmetic.hpp"

namespace graph {

//******************************************************************************
//  Sine node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a sine_node leaf.
//------------------------------------------------------------------------------
    template<typename N>
    class sine_node final : public straight_node<typename N::backend> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a sine_node node.
///
///  @param[in] x Argument.
//------------------------------------------------------------------------------
        sine_node(std::shared_ptr<N> x) :
        straight_node<typename N::backend> (x) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of sine.
///
///  result = sin(x)
///
///  @returns The value of sin(x).
//------------------------------------------------------------------------------
        virtual typename N::backend evaluate() final {
            typename N::backend result = this->arg->evaluate();
            result.sin();
            return this->save_cache(result);
        }

//------------------------------------------------------------------------------
///  @brief Reduce the sin(x).
///
///  @returns Reduced graph from sine.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename N::backend> reduce() final {
#ifdef USE_REDUCE
            if (constant_cast(this->arg)) {
                return constant(this->evaluate());
            }
#endif
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d sin(a)/dx = cos(a)*da/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename N::backend>
        df(shared_leaf<typename N::backend> x) final {
            if (this->is_match(x)) {
                return constant<typename N::backend> (1);
            } else {
                return cos(this->arg)*this->arg->df(x);
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
                stream << " " << registers[this] << " = sin("
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

            auto x_cast = sin_cast(x);
            if (x_cast.get()) {
                return this->arg->is_match(x_cast->get_arg());
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
///
///  @returns The latex string representing the node.
//------------------------------------------------------------------------------
        virtual std::string to_latex() const final {
            return "\\sin\\left(" + this->arg->to_latex() + "\\right)";
        }
    };

//------------------------------------------------------------------------------
///  @brief Define sine convience function.
///
///  @param[in] x Argument.
///  @returns A reduced sin node.
//------------------------------------------------------------------------------
    template<typename N>
    shared_leaf<typename N::backend> sin(std::shared_ptr<N> x) {
        return (std::make_shared<sine_node<N>> (x))->reduce();
    }

///  Convenience type alias for shared sine nodes.
    template<typename N>
    using shared_sine = std::shared_ptr<sine_node<typename N::backend>>;

//------------------------------------------------------------------------------
///  @brief Cast to a sine node.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename N>
    shared_sine<N> sin_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<sine_node<N>> (x);
    }

//******************************************************************************
//  Cosine node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a cosine_node leaf.
//------------------------------------------------------------------------------
    template<typename N>
    class cosine_node final : public straight_node<typename N::backend> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a cosine_node node.
///
///  @param[in] x Argument.
//------------------------------------------------------------------------------
        cosine_node(std::shared_ptr<N> x) :
        straight_node<typename N::backend> (x) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of cosine.
///
///  result = cos(x)
///
///  @returns The value of cos(x).
//------------------------------------------------------------------------------
        virtual typename N::backend evaluate() final {
            typename N::backend result = this->arg->evaluate();
            result.cos();
            return this->save_cache(result);
        }

//------------------------------------------------------------------------------
///  @brief Reduce the cos(x).
///
///  @returns Reduced graph from cosine.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename N::backend> reduce() final {
#ifdef USE_REDUCE
            if (constant_cast(this->arg)) {
                return constant(this->evaluate());
            }
#endif
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d sin(a)/dx = cos(a)*da/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<typename N::backend>
        df(shared_leaf<typename N::backend> x) final {
            if (this->is_match(x)) {
                return constant<typename N::backend> (1);
            } else {
                return constant<typename N::backend> (-1)*sin(this->arg)*this->arg->df(x);
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
                stream << " " << registers[this] << " = cos("
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

            auto x_cast = cos_cast(x);
            if (x_cast.get()) {
                return this->arg->is_match(x_cast->get_arg());
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
///
///  @returns The latex string representing the node.
//------------------------------------------------------------------------------
        virtual std::string to_latex() const final {
            return "\\cos\\left(" + this->arg->to_latex() + "\\right)";
        }
    };

//------------------------------------------------------------------------------
///  @brief Define cosine convience function.
///
///  @param[in] x Argument.
///  @returns A reduced cos node.
//------------------------------------------------------------------------------
    template<typename N>
    shared_leaf<typename N::backend> cos(std::shared_ptr<N> x) {
        return (std::make_shared<cosine_node<N>> (x))->reduce();
    }


///  Convenience type alias for shared cosine nodes.
    template<typename N>
    using shared_cosine = std::shared_ptr<sine_node<typename N::backend>>;

//------------------------------------------------------------------------------
///  @brief Cast to a cosine node.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename N>
    shared_cosine<N> cos_cast(std::shared_ptr<N> x) {
        return std::dynamic_pointer_cast<cosine_node<N>> (x);
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
    std::shared_ptr<leaf_node<typename N::backend>> tan(std::shared_ptr<N> x) {
        return (sin(x)/cos(x))->reduce();
    }
}

#endif /* trigonometry_h */
