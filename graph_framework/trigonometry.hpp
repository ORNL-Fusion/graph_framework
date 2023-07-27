//------------------------------------------------------------------------------
///  @file trigonometry.hpp
///  @brief Trigonometry functions.
///
///  Defines trigonometry operations.
//------------------------------------------------------------------------------

#ifndef trigonometry_h
#define trigonometry_h

#include "node.hpp"

namespace graph {

//******************************************************************************
//  Sine node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a sine_node leaf.
//------------------------------------------------------------------------------
    template<typename T>
    class sine_node final : public straight_node<T> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @params[in] a Argument node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T> *a) {
            return "sin" + jit::format_to_string(reinterpret_cast<size_t> (a));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a sine\_node node.
///
///  @params[in] x Argument.
//------------------------------------------------------------------------------
        sine_node(shared_leaf<T> x) :
        straight_node<T> (x, sine_node<T>::to_string(x.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of sine.
///
///  result = sin(x)
///
///  @returns The value of sin(x).
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> result = this->arg->evaluate();
            result.sin();
            return result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the sin(x).
///
///  @returns Reduced graph from sine.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> reduce() {
            if (constant_cast(this->arg).get()) {
                return constant(this->evaluate());
            }
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
        df(shared_leaf<T> x) {
            if (this->is_match(x)) {
                return one<T> ();
            } else {
                return cos(this->arg)*this->arg->df(x);
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> compile(std::ostringstream &stream,
                                       jit::register_map &registers) {
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
        virtual bool is_match(shared_leaf<T> x) {
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
        virtual void to_latex() const {
            std::cout << "\\sin\\left(";
            this->arg->to_latex();
            std::cout << "\\right)";
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> remove_pseudo() {
            return sin(this->arg->remove_pseudo());
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
        auto temp = std::make_shared<sine_node<T>> (x)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash(); i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T>::cache.find(i) ==
                leaf_node<T>::cache.end()) {
                leaf_node<T>::cache[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T>::cache[i])) {
                return leaf_node<T>::cache[i];
            }
        }
        assert(false && "Should never reach.");
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
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @params[in] a Argument node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T> *a) {
            return "cos" + jit::format_to_string(reinterpret_cast<size_t> (a));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a cosine_node node.
///
///  @params[in] x Argument.
//------------------------------------------------------------------------------
        cosine_node(shared_leaf<T> x) :
        straight_node<T> (x, cosine_node<T>::to_string(x.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of cosine.
///
///  result = cos(x)
///
///  @returns The value of cos(x).
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> result = this->arg->evaluate();
            result.cos();
            return result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the cos(x).
///
///  @returns Reduced graph from cosine.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> reduce() {
            if (constant_cast(this->arg).get()) {
                return constant(this->evaluate());
            }
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
        df(shared_leaf<T> x) {
            if (this->is_match(x)) {
                return one<T> ();
            } else {
                return none<T> ()*sin(this->arg)*this->arg->df(x);
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> compile(std::ostringstream &stream,
                                       jit::register_map &registers) {
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
        virtual bool is_match(shared_leaf<T> x) {
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
        virtual void to_latex() const {
            std::cout << "\\cos\\left(";
            this->arg->to_latex();
            std::cout << "\\right)";
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> remove_pseudo() {
            return cos(this->arg->remove_pseudo());
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
        auto temp = std::make_shared<cosine_node<T>> (x)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash(); i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T>::cache.find(i) ==
                leaf_node<T>::cache.end()) {
                leaf_node<T>::cache[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T>::cache[i])) {
                return leaf_node<T>::cache[i];
            }
        }
        assert(false && "Should never reach.");
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
        return sin(x)/cos(x);
    }

//******************************************************************************
//  Arctangent node.
//******************************************************************************
//******************************************************************************
//  arctan node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a sine_node leaf.
//------------------------------------------------------------------------------
    template<typename T>
    class arctan_node final : public branch_node<T> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @params[in] l Left pointer.
///  @params[in] r Left pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
    static std::string to_string(leaf_node<T> *l,
                                 leaf_node<T> *r) {
        return "atan" + jit::format_to_string(reinterpret_cast<size_t> (l))
                      + jit::format_to_string(reinterpret_cast<size_t> (r));
    }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a arctan\_node node.
///
///  @params[in] x Argument.
///  @params[in] y Argument.
//------------------------------------------------------------------------------
        arctan_node(shared_leaf<T> x,
                    shared_leaf<T> y) :
        branch_node<T> (x, y, arctan_node<T>::to_string(x.get(), y.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of arctan.
///
///  result = atan2(y, x)
///
///  @returns The value of atan2(y, x).
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> left = this->left->evaluate();
            backend::buffer<T> right = this->right->evaluate();
            return backend::atan(left, right);
        }

//------------------------------------------------------------------------------
///  @brief Reduce a arctan node.
///
///  @returns A reduced arctan node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> reduce() {
            auto l = constant_cast(this->left);
            auto r = constant_cast(this->right);
            if (l.get() && r.get()) {
                return constant(this->evaluate());
            }
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d tan(x,y)/dx = 1/(1 + (y/x)^2)*d (y/x)/dx
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T>
        df(shared_leaf<T> x) {
            auto one_constant = one<T> ();
            if (this->is_match(x)) {
                return one_constant;
            } else {
                auto z = this->right/this->left;
                return (one_constant/(one_constant + z*z))*z->df(x);
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> compile(std::ostringstream &stream,
                                       jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T> l = this->left->compile(stream, registers);
                shared_leaf<T> r = this->right->compile(stream, registers);
                
                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                if constexpr (jit::is_complex<T> ()) {
                    stream << " " << registers[this] << " = atan("
                           << registers[r.get()] << "/"
                           << registers[l.get()];
                } else {
                    stream << " " << registers[this] << " = atan2("
                           << registers[r.get()] << ","
                           << registers[l.get()];
                }
                stream << ");" << std::endl;
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @params[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T> x) {
            if (this == x.get()) {
                return true;
            }

            auto x_cast = atan_cast(x);
            if (x_cast.get()) {
                return this->left->is_match(x_cast->get_left()) &&
                       this->right->is_match(x_cast->get_right());
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << "atan\\left(";
            this->left->to_latex();
            std::cout << ",";
            this->right->to_latex();
            std::cout << "\\right)";
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> remove_pseudo() {
            return atan(this->left->remove_pseudo(),
                        this->right->remove_pseudo());
        }
    };

//------------------------------------------------------------------------------
///  @brief Build arctan node.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename T>
    shared_leaf<T> atan(shared_leaf<T> l,
                        shared_leaf<T> r) {
        auto temp = std::make_shared<arctan_node<T>> (l, r)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash(); i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T>::cache.find(i) ==
                leaf_node<T>::cache.end()) {
                leaf_node<T>::cache[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T>::cache[i])) {
                return leaf_node<T>::cache[i];
            }
        }
        assert(false && "Should never reach.");
    }

///  Convenience type alias for shared add nodes.
    template<typename T>
    using shared_atan = std::shared_ptr<arctan_node<T>>;

//------------------------------------------------------------------------------
///  @brief Cast to a power node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T>
    shared_atan<T> atan_cast(shared_leaf<T> x) {
        return std::dynamic_pointer_cast<arctan_node<T>> (x);
    }
}

#endif /* trigonometry_h */
