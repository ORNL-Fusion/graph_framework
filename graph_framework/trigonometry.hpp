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
    template<typename T>
    class sine_node final : public straight_node<T> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a sine_node node.
///
///  @params[in] x Argument.
//------------------------------------------------------------------------------
        sine_node(shared_leaf<T> x) :
        straight_node<T> (x) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of sine.
///
///  result = sin(x)
///
///  @returns The value of sin(x).
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() final {
            backend::buffer<T> result = this->arg->evaluate();
            result.sin();
            return result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the sin(x).
///
///  @returns Reduced graph from sine.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> reduce() final {
#ifdef USE_REDUCE
            if (constant_cast(this->arg).get()) {
                return constant(this->evaluate());
            }

//  Piecewise constant reductions.
            auto apw = piecewise_1D_cast(this->arg);
            if (piecewise_1D_cast(this->arg).get()) {
                return piecewise_1D(this->evaluate(), apw->get_arg());
            }
#endif
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d sin(a)/dx = cos(a)*da/dx
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T>
        df(shared_leaf<T> x) final {
            if (this->is_match(x)) {
                return one<T> ();
            } else {
                return cos(this->arg)*this->arg->df(x);
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in] stream    String buffer stream.
///  @params[in] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> compile(std::stringstream &stream,
                                       jit::register_map &registers) final {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T> a = this->arg->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = sin("
                       << registers[a.get()] << ");"
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
        virtual bool is_match(shared_leaf<T> x) final {
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
//------------------------------------------------------------------------------
        virtual void to_latex() const final {
            std::cout << "\\sin\\left(";
            this->arg->to_latex();
            std::cout << "\\right)";
        }
    };

//------------------------------------------------------------------------------
///  @brief Define sine convience function.
///
///  @params[in] x Argument.
///  @returns A reduced sin node.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> sin(shared_leaf<T> x) {
        return (std::make_shared<sine_node<T>> (x))->reduce();
    }

///  Convenience type alias for shared sine nodes.
    template<typename T>
    using shared_sine = std::shared_ptr<sine_node<T>>;

//------------------------------------------------------------------------------
///  @brief Cast to a sine node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T>
    shared_sine<T> sin_cast(shared_leaf<T> x) {
        return std::dynamic_pointer_cast<sine_node<T>> (x);
    }

//******************************************************************************
//  Cosine node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a cosine_node leaf.
//------------------------------------------------------------------------------
    template<typename T>
    class cosine_node final : public straight_node<T> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a cosine_node node.
///
///  @params[in] x Argument.
//------------------------------------------------------------------------------
        cosine_node(shared_leaf<T> x) :
        straight_node<T> (x) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of cosine.
///
///  result = cos(x)
///
///  @returns The value of cos(x).
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() final {
            backend::buffer<T> result = this->arg->evaluate();
            result.cos();
            return result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the cos(x).
///
///  @returns Reduced graph from cosine.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> reduce() final {
#ifdef USE_REDUCE
            if (constant_cast(this->arg).get()) {
                return constant(this->evaluate());
            }

//  Piecewise constant reductions.
            auto apw = piecewise_1D_cast(this->arg);
            if (piecewise_1D_cast(this->arg).get()) {
                return piecewise_1D(this->evaluate(), apw->get_arg());
            }
#endif
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d sin(a)/dx = cos(a)*da/dx
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T>
        df(shared_leaf<T> x) final {
            if (this->is_match(x)) {
                return one<T> ();
            } else {
                return none<T> ()*sin(this->arg)*this->arg->df(x);
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in] stream    String buffer stream.
///  @params[in] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> compile(std::stringstream &stream,
                                       jit::register_map &registers) final {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T> a = this->arg->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = cos("
                       << registers[a.get()] << ");"
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
        virtual bool is_match(shared_leaf<T> x) final {
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
//------------------------------------------------------------------------------
        virtual void to_latex() const final {
            std::cout << "\\cos\\left(";
            this->arg->to_latex();
            std::cout << "\\right)";
        }
    };

//------------------------------------------------------------------------------
///  @brief Define cosine convience function.
///
///  @params[in] x Argument.
///  @returns A reduced cos node.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> cos(shared_leaf<T> x) {
        return (std::make_shared<cosine_node<T>> (x))->reduce();
    }


///  Convenience type alias for shared cosine nodes.
    template<typename T>
    using shared_cosine = std::shared_ptr<cosine_node<T>>;

//------------------------------------------------------------------------------
///  @brief Cast to a cosine node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T>
    shared_cosine<T> cos_cast(shared_leaf<T> x) {
        return std::dynamic_pointer_cast<cosine_node<T>> (x);
    }

//******************************************************************************
//  Tangent node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Define tangent convience function.
///
///  tan(x) = sin(x)/cos(x)
///
///  @params[in] x Argument.
///  @returns A reduced tan node.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> tan(shared_leaf<T> x) {
        return (sin(x)/cos(x))->reduce();
    }
}

#endif /* trigonometry_h */
