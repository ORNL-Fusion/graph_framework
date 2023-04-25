//------------------------------------------------------------------------------
///  @file piecewise.hpp
///  @brief Piecewise nodes.
///
///  Defines nodes containing piecewise constants.
//------------------------------------------------------------------------------

#ifndef piecewise_h
#define piecewise_h

#include "node.hpp"

namespace graph {
//******************************************************************************
//  1D Piecewise node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a 1D piecewise constant.
///
///  This class is used to impliment the coefficent terms of bicubic spline
///  interpolation. An function is interpolated using
///
///    y'(t) = a_i + b_i*t +c_i*t^2 + d_i*t^3                                  (1)
///
///  The coeffients are defined as
///
///    a_i = y_i                                                               (2)
///    b_i = D_i                                                               (3)
///    c_i = 3*(y_i+1 - y_i) - 2*D_i - D_i+1                                   (4)
///    d_i = 2*(y_i - y_i+1) + D_i + D_i+1                                     (5)
///
///  The agument T is assumed to be the normalized argument
///
///    t = (x + sqrt(y))/((y_n - y_0)/(n - 1))                                 (6)
///
///  To avoid tracking the index which normaizes t to a zero to one interval the
///  coefficients should be normalized to
///
///    a'_i = a_i - b_i*i + c_i*i^2 - d_i*i^3                                  (7)
///    b'_i = b_i - 2*c_i*i+3*d_i*i^2                                          (8)
///    c'_i = c_i - 3*d_i*i                                                    (9)
///    d'_i = d_i                                                             (10)
//------------------------------------------------------------------------------
    template<typename T>
    class piecewise_1D_node : public straight_node<T> {
    protected:
///  Storage buffers for the data.
        backend::buffer<T> data;
    public:
//------------------------------------------------------------------------------
///  @brief Construct a piecewise constant node.
///
///  @params[in] d Data to initalize the piecewise constant.
///  @params[in] x Argument.
//------------------------------------------------------------------------------
        piecewise_1D_node(const std::vector<T> d,
                          shared_leaf<T> x) :
        straight_node<T> (x->reduce()), data(d) {}

//------------------------------------------------------------------------------
///  @brief Construct a piece wise constant.
///
///  @params[in] d Back end buffer to initalize with.
///  @params[in] x Argument.
//------------------------------------------------------------------------------
        piecewise_1D_node(const backend::buffer<T> d,
                          shared_leaf<T> x) :
        straight_node<T> (x->reduce()), data(d) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of the piecewise constant.
///
///  Evaluate functions are only used by the minimization. So this node does not
///  evaluate the argument. Instead this only returs the data as if it were a
///  constant.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() final {
            return data;
        }

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  For basic nodes, there's nothing to reduce.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> reduce() final {
#ifdef USE_REDUCE
            if (data.is_same()) {
                return constant(data.at(0));
            }
#endif
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> df(shared_leaf<T> x) final {
            return std::make_shared<constant_node<T>> (static_cast<T> (0.0));
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  This node first evaluates the value of the argument then chooses the correct
///  in piecewise index. This assumes that the argument is
///
///    x' = (x - Sqrt(y_0^2))/((y_n - y_0)/(n - 1))                            (1)
///
///  and the spline coefficients are of the form.
///
///    a'_i = a_i - b_i*i + c_i*i^2 - d_i*i^3                                  (2)
///    b'_i = b_i - 2*c_i*i+3*d_i*i^2                                          (3)
///    c'_i = c_i - 3*d_i*i                                                    (4)
///    d'_i = d_i                                                              (5)
///
///  @params[in] stream    String buffer stream.
///  @params[in] registers List of defined registers.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> compile(std::stringstream &stream,
                                       jit::register_map &registers) final {
            if (registers.find(this) == registers.end()) {
                if (registers.find(data.data()) == registers.end()) {
                    registers[data.data()] = jit::to_string('a', data.data());
                    stream << "        const ";
                    jit::add_type<T> (stream);
                    stream << " " << registers[data.data()] << "[] = {";
                    if constexpr (jit::is_complex<T> ()) {
                        jit::add_type<T> (stream);
                    }
                    stream << data[0];
                    for (size_t i = 1, ie = data.size(); i < ie; i++) {
                        stream << ", ";
                        if constexpr (jit::is_complex<T> ()) {
                            jit::add_type<T> (stream);
                        }
                        stream << data[i];
                    }
                    stream << "};" << std::endl;
                }
                
                shared_leaf<T> a = this->arg->compile(stream, registers);
                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = " << registers[data.data()];
                stream << "[max(min((int)";
                if constexpr (jit::is_complex<T> ()) {
                    stream << "real(";
                }
                stream << registers[a.get()];
                if constexpr (jit::is_complex<T> ()) {
                    stream << ")";
                }
                stream <<", " << data.size() - 1 << "), 0)];" << std::endl;
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

            auto x_cast = piecewise_1D_cast(x);
            if (x_cast.get()) {
                return this->evaluate() == this->arg->evaluate();
            } else {
                return false;
            }
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const final {
            std::cout << jit::to_string('r', this) << "_{i}";
        }
    };

//------------------------------------------------------------------------------
///  @brief Define piecewise_1D convience function.
///
///  @params[in] d Data to initalize the piecewise constant.
///  @params[in] x Argument.
///  @returns A reduced sqrt node.
//------------------------------------------------------------------------------
    template<typename T> shared_leaf<T> piecewise_1D(const std::vector<T> d,
                                                     shared_leaf<T> x) {
        return (std::make_shared<piecewise_1D_node<T>> (d, x))->reduce();
    }

//------------------------------------------------------------------------------
///  @brief Define piecewise_1D convience function.
///
///  @params[in] d Data to initalize the piecewise constant.
///  @params[in] x Argument.
///  @returns A reduced sqrt node.
//------------------------------------------------------------------------------
    template<typename T> shared_leaf<T> piecewise_1D(const backend::buffer<T> d,
                                                     shared_leaf<T> x) {
        return (std::make_shared<piecewise_1D_node<T>> (d, x))->reduce();
    }

///  Convenience type alias for shared sqrt nodes.
    template<typename T>
    using shared_piecewise_1D = std::shared_ptr<piecewise_1D_node<T>>;

//------------------------------------------------------------------------------
///  @brief Cast to a sqrt node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T>
    shared_piecewise_1D<T> piecewise_1D_cast(shared_leaf<T> x) {
        return std::dynamic_pointer_cast<piecewise_1D_node<T>> (x);
    }
}

#endif /* piecewise_h */
