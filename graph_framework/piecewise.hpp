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
///  This class is used to impliment the coefficent terms of cubic spline
///  interpolation. An function is interpolated using
///
///    y(x) = a_i + b_i*x +c_i*x^2 + d_i*x^3                                   (1)
///
///  The coeffients are defined as
///
///    a_i = y_i                                                               (2)
///    b_i = D_i                                                               (3)
///    c_i = 3*(y_i+1 - y_i) - 2*D_i - D_i+1                                   (4)
///    d_i = 2*(y_i - y_i+1) + D_i + D_i+1                                     (5)
///
///  The agument x is assumed to be the normalized argument
///
///    x_norm = (x - xmin)/dx - i                                              (6)
///
///  To avoid tracking the index i which normaizes x to a zero to one interval the
///  coefficients should be normalized to
///
///    a'_i = a_i - b_i*i + c_i*i^2 - d_i*i^3                                  (7)
///    b'_i = b_i - 2*c_i*i+3*d_i*i^2                                          (8)
///    c'_i = c_i - 3*d_i*i                                                    (9)
///    d'_i = d_i                                                             (10)
///
///  This makes the normalized argument (6) become
///
///    x_norm' = (x - xmin)/dx                                                (11)
//------------------------------------------------------------------------------
    template<typename T>
    class piecewise_1D_node final : public straight_node<T> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @params[in] d Backend buffer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(const std::vector<T> &d) {
            std::stringstream stream;
            stream << std::setprecision(jit::max_digits10<T> ());

            stream << "{" << d[0];
            for (size_t i = 1, ie = d.size(); i < ie; i++) {
                stream << "," << d[i];
            }
            stream << "}";
                    
            return stream.str();
        }

///  Storage buffers for the data.
        backend::buffer<T> data;

    public:
//------------------------------------------------------------------------------
///  @brief Construct 1D a piecewise constant node.
///
///  @params[in] d Data to initalize the piecewise constant.
///  @params[in] x Argument.
//------------------------------------------------------------------------------
        piecewise_1D_node(const std::vector<T> d,
                          shared_leaf<T> x) :
        straight_node<T> (x, piecewise_1D_node<T>::to_string(d)), data(d) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of the piecewise constant.
///
///  Evaluate functions are only used by the minimization. So this node does not
///  evaluate the argument. Instead this only returs the data as if it were a
///  constant.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            return data;
        }

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  If all the values in the data buffer are the same. Reduce to a single
///  constant.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> reduce() {
            if (data.is_same()) {
                return constant(data.at(0));
            }
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> df(shared_leaf<T> x) {
            return zero<T> ();
        }

//------------------------------------------------------------------------------
///  @brief Compile preamble.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @params[in,out] visited   List of visited nodes.
//------------------------------------------------------------------------------
        virtual void compile_preamble(std::stringstream &stream,
                                      jit::register_map &registers,
                                      jit::visiter_map &visited) {
            if (visited.find(this) == visited.end()) {
                if (registers.find(data.data()) == registers.end()) {
                    registers[data.data()] = jit::to_string('a', data.data());
                    if constexpr (jit::use_metal<T> ()) {
                        stream << "constant ";
                    }
                    stream << "const ";
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
                    visited[this] = 0;
                }
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  This node first evaluates the value of the argument then chooses the correct
///  piecewise index. This assumes that the argument is
///
///    x' = (x - xmin)/dx                                                      (1)
///
///  and the spline coefficients are of the form.
///
///    a'_i = a_i - b_i*i + c_i*i^2 - d_i*i^3                                  (2)
///    b'_i = b_i - 2*c_i*i+3*d_i*i^2                                          (3)
///    c'_i = c_i - 3*d_i*i                                                    (4)
///    d'_i = d_i                                                              (5)
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> compile(std::stringstream &stream,
                                       jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
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
        virtual bool is_match(shared_leaf<T> x) {
            if (this == x.get()) {
                return true;
            }

            auto x_cast = piecewise_1D_cast(x);
            if (x_cast.get()) {
                return this->evaluate() == x->evaluate() &&
                       this->arg->is_match(x_cast->get_arg());
            } else {
                return false;
            }
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << "r\\_" << reinterpret_cast<size_t> (this) << "_{i}";
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a constant.
///
///  @returns True if the node acts like a constant.
//------------------------------------------------------------------------------
        virtual bool is_constant_like() const {
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a variable.
///
///  @returns True if the node acts like a variable.
//------------------------------------------------------------------------------
        virtual bool is_variable_like() const {
            return false;
        }

///  Cache for constructed nodes.
        inline thread_local static node_cache<T> cache;
    };

//------------------------------------------------------------------------------
///  @brief Define piecewise\_1D convience function.
///
///  @params[in] d Data to initalize the piecewise constant.
///  @params[in] x Argument.
///  @returns A reduced piecewise\_1D node.
//------------------------------------------------------------------------------
    template<typename T> shared_leaf<T> piecewise_1D(const std::vector<T> d,
                                                     shared_leaf<T> x) {
        auto temp = std::make_shared<piecewise_1D_node<T>> (d, x)->reduce();
        const size_t h = temp->get_hash();
        if (piecewise_1D_node<T>::cache.find(h) ==
            piecewise_1D_node<T>::cache.end()) {
            piecewise_1D_node<T>::cache[h] = temp;
            return temp;
        }
        
        return piecewise_1D_node<T>::cache[h];
    }

///  Convenience type alias for shared piecewise 1D nodes.
    template<typename T>
    using shared_piecewise_1D = std::shared_ptr<piecewise_1D_node<T>>;

//------------------------------------------------------------------------------
///  @brief Cast to a piecewise 1D node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T>
    shared_piecewise_1D<T> piecewise_1D_cast(shared_leaf<T> x) {
        return std::dynamic_pointer_cast<piecewise_1D_node<T>> (x);
    }

//******************************************************************************
//  2D Piecewise node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a 2D piecewise constant.
///
///  This class is used to impliment the coefficent terms of bicubic spline
///  interpolation. An function is interpolated using
///
///    z(x,y) = Σ_i,3Σ_j,3 c_ij*x^i*y^j                                        (1)
///
///  The aguments x and y are assumed to be the normalized arguments
///
///    x_norm = (x - xmin)/dx - i                                              (2)
///    y_norm = (y - ymin)/dy - j                                              (3)
///
///  To avoid tracking the indices i and j which normaizes x and y to a zero to
///  one interval the coefficients should be normalized to
///
///    c00'_ij = Σ_k,3Σ_l,3 (-i)^k*(-j)^l*ckl_ij                               (4)
///    c10'_ij = Σ_k,3Σ_l,3 k*(-i)^(k-1)*(-j)^l*ckl_ij                         (5)
///    c01'_ij = Σ_k,3Σ_l,3 l*(-i)^k*(-j)^(l-1)*ckl_ij                         (6)
///    c20'_ij = Σ_k,3Σ_l,3 Max(2*k-3,0)*(-i)^(k-2)*(-j)^l*ckl_ij              (7)
///    c02'_ij = Σ_k,3Σ_l,3 Max(2*l-3,0)*(-i)^k*(-j)^(l-2)*ckl_ij              (8)
///    c30'_ij = Σ_k,3Σ_l,3 Max(k-2,0)*(-i)^(k-3)*(-j)^l*ckl_ij                (9)
///    c03'_ij = Σ_k,3Σ_l,3 Max(l-2,0)*(-i)^k*(-j)^(l-3)*ckl_ij               (10)
///    c11'_ij = Σ_k,3Σ_l,3 k*l*(-i)^(k-1)*(-j)^(j-1)                         (11)
///    c21'_ij = Σ_k,3Σ_l,3 Max(2*k-3,0)*l*(-i)^(k-2)*(-j)^(j-1)              (12)
///    c12'_ij = Σ_k,3Σ_l,3 k*Max(2*l-3,0)*(-i)^(k-1)*(-j)^(j-2)              (13)
///    c31'_ij = Σ_k,3Σ_l,3 Max(k-2,0)*l*(-i)^(k-3)*(-j)^(j-1)                (14)
///    c13'_ij = Σ_k,3Σ_l,3 k*Max(l-2,0)*(-i)^(k-1)*(-j)^(j-3)                (15)
///    c22'_ij = Σ_k,3Σ_l,3 Max(2*k-3,0)*Max(2*l-3,0)*(-i)^(k-2)*(-j)^(j-2)   (16)
///    c32'_ij = Σ_k,3Σ_l,3 Max(k-2,0)*Max(2*l-3,0)*(-i)^(k-3)*(-j)^(j-2)     (17)
///    c23'_ij = Σ_k,3Σ_l,3 Max(2*k-3,0)*Max(l-2,0)*(-i)^(k-2)*(-j)^(j-3)     (18)
///    c33'_ij = Σ_k,3Σ_l,3 Max(k-2,0)*Max(l-2,0)*(-i)^(k-3)*(-j)^(j-3)       (19)
///
///  This makes the normalized arguments (6,7) become
///
///    x_norm' = (x - xmin)/dx                                                (20)
///    y_norm' = (y - ymin)/dy                                                (21)
//------------------------------------------------------------------------------
    template<typename T>
    class piecewise_2D_node final : public branch_node<T> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @params[in] d Backend buffer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(const std::vector<T> &d) {
            std::stringstream stream;
            stream << std::setprecision(jit::max_digits10<T> ());

            stream << "{" << d[0];
            for (size_t i = 1, ie = d.size(); i < ie; i++) {
                stream << "," << d[i];
            }
            stream << "}";

            return stream.str();
        }

///  Storage buffers for the data.
        backend::buffer<T> data;
///  Number of columns.
        const size_t num_columns;

    public:
//------------------------------------------------------------------------------
///  @brief Construct 2D a piecewise constant node.
///
///  @params[in] d Data to initalize the piecewise constant.
///  @params[in] n Number of columns.
///  @params[in] x X Argument.
///  @params[in] y Y Argument.
//------------------------------------------------------------------------------
        piecewise_2D_node(const std::vector<T> d,
                          const size_t n,
                          shared_leaf<T> x,
                          shared_leaf<T> y) :
        branch_node<T> (x, y, piecewise_2D_node<T>::to_string(d)),
        data(d), num_columns(n) {
            assert(data.size()/n &&
                   "Expected the data buffer to be a multiple of the number of columns.");
        }

//------------------------------------------------------------------------------
///  @brief Get the number of columns.
///
///  @returns The number of columns in the constant.
//------------------------------------------------------------------------------
        size_t get_num_columns() {
            return num_columns;
        }

//------------------------------------------------------------------------------
///  @brief Evaluate the results of the piecewise constant.
///
///  Evaluate functions are only used by the minimization. So this node does not
///  evaluate the argument. Instead this only returs the data as if it were a
///  constant.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            return data;
        }

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  If all the values in the data buffer are the same. Reduce to a single
///  constant.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> reduce() {
            if (data.is_same()) {
                return constant(data.at(0));
            }
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> df(shared_leaf<T> x) {
            return zero<T> ();
        }

//------------------------------------------------------------------------------
///  @brief Compile preamble.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @params[in,out] visited   List of visited nodes.
//------------------------------------------------------------------------------
        virtual void compile_preamble(std::stringstream &stream,
                                      jit::register_map &registers,
                                      jit::visiter_map &visited) {
            if (visited.find(this) == visited.end()) {
                if (registers.find(data.data()) == registers.end()) {
                    registers[data.data()] = jit::to_string('a', data.data());
                    if constexpr (jit::use_metal<T> ()) {
                        stream << "constant ";
                    }
                    stream << "const ";
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
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  This node first evaluates the value of the argument then chooses the correct
///  piecewise index. This assumes that the argument is
///
///    x' = (x - xmin)/dx                                                      (1)
///    y' = (y - ymin)/dy                                                      (2)
///
///  and the spline coefficients are of the form.
///
///    c00'_ij = Σ_k,3Σ_l,3 (-i)^k*(-j)^l*ckl_ij                               (3)
///    c10'_ij = Σ_k,3Σ_l,3 k*(-i)^(k-1)*(-j)^l*ckl_ij                         (4)
///    c01'_ij = Σ_k,3Σ_l,3 l*(-i)^k*(-j)^(l-1)*ckl_ij                         (5)
///    c20'_ij = Σ_k,3Σ_l,3 Max(2*k-3,0)*(-i)^(k-2)*(-j)^l*ckl_ij              (6)
///    c02'_ij = Σ_k,3Σ_l,3 Max(2*l-3,0)*(-i)^k*(-j)^(l-2)*ckl_ij              (7)
///    c30'_ij = Σ_k,3Σ_l,3 Max(k-2,0)*(-i)^(k-3)*(-j)^l*ckl_ij                (8)
///    c03'_ij = Σ_k,3Σ_l,3 Max(l-2,0)*(-i)^k*(-j)^(l-3)*ckl_ij                (9)
///    c11'_ij = Σ_k,3Σ_l,3 k*l*(-i)^(k-1)*(-j)^(j-1)                         (10)
///    c21'_ij = Σ_k,3Σ_l,3 Max(2*k-3,0)*l*(-i)^(k-2)*(-j)^(j-1)              (11)
///    c12'_ij = Σ_k,3Σ_l,3 k*Max(2*l-3,0)*(-i)^(k-1)*(-j)^(j-2)              (12)
///    c31'_ij = Σ_k,3Σ_l,3 Max(k-2,0)*l*(-i)^(k-3)*(-j)^(j-1)                (13)
///    c13'_ij = Σ_k,3Σ_l,3 k*Max(l-2,0)*(-i)^(k-1)*(-j)^(j-3)                (14)
///    c22'_ij = Σ_k,3Σ_l,3 Max(2*k-3,0)*Max(2*l-3,0)*(-i)^(k-2)*(-j)^(j-2)   (15)
///    c32'_ij = Σ_k,3Σ_l,3 Max(k-2,0)*Max(2*l-3,0)*(-i)^(k-3)*(-j)^(j-2)     (16)
///    c23'_ij = Σ_k,3Σ_l,3 Max(2*k-3,0)*Max(l-2,0)*(-i)^(k-2)*(-j)^(j-3)     (17)
///    c33'_ij = Σ_k,3Σ_l,3 Max(k-2,0)*Max(l-2,0)*(-i)^(k-3)*(-j)^(j-3)       (18)
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T> compile(std::stringstream &stream,
                                       jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {                
                shared_leaf<T> x = this->left->compile(stream, registers);
                shared_leaf<T> y = this->right->compile(stream, registers);
                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = " << registers[data.data()];
                stream << "[max(min((int)";
                if constexpr (jit::is_complex<T> ()) {
                    stream << "real(";
                }
                stream << registers[x.get()];
                if constexpr (jit::is_complex<T> ()) {
                    stream << ")";
                }
                stream << "*" << num_columns << " + (int)";
                if constexpr (jit::is_complex<T> ()) {
                    stream << "real(";
                }
                stream << registers[y.get()];
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
        virtual bool is_match(shared_leaf<T> x) {
            if (this == x.get()) {
                return true;
            }

            auto x_cast = piecewise_2D_cast(x);
            if (x_cast.get()) {
                return this->evaluate() == x->evaluate()        &&
                       this->left->is_match(x_cast->get_left()) &&
                       this->right->is_match(x_cast->get_right());
            } else {
                return false;
            }
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout  << "r\\_" << reinterpret_cast<size_t> (this) << "_{ij}";
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a constant.
///
///  @returns True if the node acts like a constant.
//------------------------------------------------------------------------------
        virtual bool is_constant_like() const {
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a variable.
///
///  @returns True if the node acts like a variable.
//------------------------------------------------------------------------------
        virtual bool is_variable_like() const {
            return false;
        }

///  Cache for constructed nodes.
        inline thread_local static node_cache<T> cache;
    };

//------------------------------------------------------------------------------
///  @brief Define piecewise\_2D convience function.
///
///  @params[in] d Data to initalize the piecewise constant.
///  @params[in] n Number of columns.
///  @params[in] x Argument.
///  @params[in] y Argument.
///  @returns A reduced sqrt node.
//------------------------------------------------------------------------------
    template<typename T> shared_leaf<T> piecewise_2D(const std::vector<T> d,
                                                     const size_t n,
                                                     shared_leaf<T> x,
                                                     shared_leaf<T> y) {
        auto temp = std::make_shared<piecewise_2D_node<T>> (d, n, x, y)->reduce();
        const size_t h = temp->get_hash();
        if (piecewise_2D_node<T>::cache.find(h) ==
            piecewise_2D_node<T>::cache.end()) {
            piecewise_2D_node<T>::cache[h] = temp;
            return temp;
        }
        
        return piecewise_2D_node<T>::cache[h];
    }

///  Convenience type alias for shared piecewise 2D nodes.
    template<typename T>
    using shared_piecewise_2D = std::shared_ptr<piecewise_2D_node<T>>;

//------------------------------------------------------------------------------
///  @brief Cast to a piecewise 2D node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T>
    shared_piecewise_2D<T> piecewise_2D_cast(shared_leaf<T> x) {
        return std::dynamic_pointer_cast<piecewise_2D_node<T>> (x);
    }
}

#endif /* piecewise_h */
