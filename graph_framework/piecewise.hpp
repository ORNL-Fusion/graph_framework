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
//------------------------------------------------------------------------------
///  @brief Compile an index.
///
///  @tparam T Base type of the calculation.
///
///  @params[in,out] stream        String buffer stream.
///  @params[in]     register_name Reister for the argument.
///  @params[in]     length        Dimension length of argument.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
void compile_index(std::ostringstream &stream,
                   const std::string &register_name,
                   const size_t length) {
    stream << "min(max((unsigned int)";
    if constexpr (jit::is_complex<T> ()) {
        stream << "real(";
    }
    stream << register_name;
    if constexpr (jit::is_complex<T> ()) {
        stream << ")";
    }
    stream << ",0u)," << length - 1 << "u)";
}

//******************************************************************************
//  1D Piecewise node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a 1D piecewise constant.
///
///  This class is used to impliment the coefficent terms of cubic spline
///  interpolation. An function is interpolated using
///
///    y(x) = a_i + b_i*x +c_i*x^2 + d_i*x^3                                 (1)
///
///  The coeffients are defined as
///
///    a_i = y_i                                                             (2)
///    b_i = D_i                                                             (3)
///    c_i = 3*(y_i+1 - y_i) - 2*D_i - D_i+1                                 (4)
///    d_i = 2*(y_i - y_i+1) + D_i + D_i+1                                   (5)
///
///  The agument x is assumed to be the normalized argument
///
///    x_norm = (x - xmin)/dx - i                                            (6)
///
///  To avoid tracking the index i which normaizes x to a zero to one interval
///  the coefficients should be normalized to
///
///    a'_i = a_i - b_i*i + c_i*i^2 - d_i*i^3                                (7)
///    b'_i = b_i - 2*c_i*i+3*d_i*i^2                                        (8)
///    c'_i = c_i - 3*d_i*i                                                  (9)
///    d'_i = d_i                                                           (10)
///
///  This makes the normalized argument (6) become
///
///    x_norm' = (x - xmin)/dx                                              (11)
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class piecewise_1D_node final : public straight_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @params[in] d Backend buffer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(const backend::buffer<T> &d) {
            std::string temp;
            for (size_t i = 0, ie = d.size(); i < ie; i++) {
                temp += jit::format_to_string(d[i]);
            }

            return temp;
        }

//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string with the argument.
///
///  @params[in] d Backend buffer.
///  @params[in] x Argument.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(const backend::buffer<T> &d,
                                     shared_leaf<T, SAFE_MATH> x) {
            return piecewise_1D_node::to_string(d) +
                   jit::format_to_string(x->get_hash());
        }

//------------------------------------------------------------------------------
///  @brief Stores the data in a hash.
///
///  @params[in] d Backend buffer.
///  @returns The hash the node is stored in.
//------------------------------------------------------------------------------
        static size_t hash_data(const backend::buffer<T> &d) {
            const size_t h = std::hash<std::string>{} (piecewise_1D_node::to_string(d));
            for (size_t i = h; i < std::numeric_limits<size_t>::max(); i++) {
                if (leaf_node<T, SAFE_MATH>::backend_cache.find(i) ==
                    leaf_node<T, SAFE_MATH>::backend_cache.end()) {
                    leaf_node<T, SAFE_MATH>::backend_cache[i] = d;
                    return i;
                } else if (d == leaf_node<T, SAFE_MATH>::backend_cache[i]) {
                    return i;
                }
            }
            assert(false && "Should never reach.");
        }

///  Data buffer hash.
        const size_t data_hash;

    public:
//------------------------------------------------------------------------------
///  @brief Construct 1D a piecewise constant node.
///
///  @params[in] d Data to initalize the piecewise constant.
///  @params[in] x Argument.
//------------------------------------------------------------------------------
        piecewise_1D_node(const backend::buffer<T> &d,
                          shared_leaf<T, SAFE_MATH> x) :
        straight_node<T, SAFE_MATH> (x, piecewise_1D_node::to_string(d, x)),
        data_hash(piecewise_1D_node::hash_data(d)) {}

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
            return leaf_node<T, SAFE_MATH>::backend_cache[data_hash];
        }

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  If all the values in the data buffer are the same. Reduce to a single
///  constant.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
            if (evaluate().is_same()) {
                return constant<T, SAFE_MATH> (evaluate().at(0));
            }
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> df(shared_leaf<T, SAFE_MATH> x) {
            return zero<T, SAFE_MATH> ();
        }

//------------------------------------------------------------------------------
///  @brief Compile preamble.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @params[in,out] visited   List of visited nodes.
///  @params[in,out] textures1d List of 1D textures.
///  @params[in,out] textures2d List of 2D textures.
//------------------------------------------------------------------------------
        virtual void compile_preamble(std::ostringstream &stream,
                                      jit::register_map &registers,
                                      jit::visiter_map &visited,
                                      jit::texture1d_list &textures1d,
                                      jit::texture2d_list &textures2d) {
            if (visited.find(this) == visited.end()) {
                if (registers.find(leaf_node<T, SAFE_MATH>::backend_cache[data_hash].data()) == registers.end()) {
                    registers[leaf_node<T, SAFE_MATH>::backend_cache[data_hash].data()] =
                        jit::to_string('a', leaf_node<T, SAFE_MATH>::backend_cache[data_hash].data());
                    const size_t length = leaf_node<T, SAFE_MATH>::backend_cache[data_hash].size();
                    if constexpr (jit::use_metal<T> () || jit::use_cuda()) {
                        textures1d.emplace_back(leaf_node<T, SAFE_MATH>::backend_cache[data_hash].data(),
                                                length);
                    } else {
                        stream << "const ";
                        jit::add_type<T> (stream);
                        stream << " " << registers[leaf_node<T, SAFE_MATH>::backend_cache[data_hash].data()] << "[] = {";
                        if constexpr (jit::is_complex<T> ()) {
                            jit::add_type<T> (stream);
                        }
                        stream << leaf_node<T, SAFE_MATH>::backend_cache[data_hash][0];
                        for (size_t i = 1; i < length; i++) {
                            stream << ", ";
                            if constexpr (jit::is_complex<T> ()) {
                                jit::add_type<T> (stream);
                            }
                            stream << leaf_node<T, SAFE_MATH>::backend_cache[data_hash][i];
                        }
                        stream << "};" << std::endl;
                    }
                    visited.insert(this);
                }
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  This node first evaluates the value of the argument then chooses the correct
///  piecewise index. This assumes that the argument is
///
///    x' = (x - xmin)/dx                                                    (1)
///
///  and the spline coefficients are of the form.
///
///    a'_i = a_i - b_i*i + c_i*i^2 - d_i*i^3                                (2)
///    b'_i = b_i - 2*c_i*i+3*d_i*i^2                                        (3)
///    c'_i = c_i - 3*d_i*i                                                  (4)
///    d'_i = d_i                                                            (5)
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> a = this->arg->compile(stream, registers);
                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = ";
                if constexpr (jit::use_cuda()) {
                    if constexpr (jit::is_float<T> ()) {
                        stream << "tex1D<float> (";
                    } else if constexpr (jit::is_double<T> ()) {
                        stream << "to_double(tex1D<uint2> (";
                    } else if constexpr (jit::is_complex<T> () && jit::is_float<T> ()) {
                        stream << "to_cmp_float(tex1D<float2> (";
                    } else {
                        stream << "to_cmp_double(tex1D<uint4> (";
                    }
                }
                stream << registers[leaf_node<T, SAFE_MATH>::backend_cache[data_hash].data()];
                const size_t length = leaf_node<T, SAFE_MATH>::backend_cache[data_hash].size();
                if constexpr (jit::use_metal<T> ()) {
                    stream << ".read(";
                    compile_index<T> (stream, registers[a.get()], length);
                    stream << ").r;";
                } else if constexpr (jit::use_cuda()) {
                    stream << ", ";
                    compile_index<T> (stream, registers[a.get()], length);
                    if constexpr (jit::is_complex<T> () || jit::is_double<T> ()) {
                        stream << ")";
                    }
                    stream << ");";
                } else {
                    stream << "[";
                    compile_index<T> (stream, registers[a.get()], length);
                    stream << "];";
                }
                stream << std::endl;
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  The argument of this node can be defered so we need to check if the
///  arguments are null.
///
///  @params[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T, SAFE_MATH> x) {
            auto x_cast = piecewise_1D_cast(x);

            if (x_cast.get()) {
                return this->data_hash == x_cast->data_hash &&
                       this->arg->is_match(x_cast->get_arg());
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << "r\\_" << reinterpret_cast<size_t> (this) << "_{i}";
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to vizgraph.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> to_vizgraph(std::stringstream &stream,
                                                      jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                const std::string name = jit::to_string('r', this);
                registers[this] = name;
                stream << "    " << name
                       << " [label = \"r_" << reinterpret_cast<size_t> (this)
                       << "_{i}\", shape = hexagon, style = filled, fillcolor = black, fontcolor = white];" << std::endl;

                auto a = this->arg->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[a.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
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
        virtual bool is_all_variables() const {
            return false;
        }

//------------------------------------------------------------------------------
///  @brief Test if the node acts like a power of variable.
///
///  @returns True.
//------------------------------------------------------------------------------
        virtual bool is_power_like() const {
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Get the base of a power.
///
///  @returns The base of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_base() {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Get the exponent of a power.
///
///  @returns The exponent of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_exponent() const {
            return one<T, SAFE_MATH> ();
        }

//------------------------------------------------------------------------------
///  @brief Check if the args match.
///
///  @param[in] x Node to match.
///  @returns True if the arguments match.
//------------------------------------------------------------------------------
        bool is_arg_match(shared_leaf<T, SAFE_MATH> x) {
            auto temp = piecewise_1D_cast(x);
            return temp.get() && this->arg->is_match(temp->get_arg());
        }
    };

//------------------------------------------------------------------------------
///  @brief Define piecewise\_1D convience function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @params[in] d Data to initalize the piecewise constant.
///  @params[in] x Argument.
///  @returns A reduced piecewise\_1D node.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> piecewise_1D(const backend::buffer<T> &d,
                                           shared_leaf<T, SAFE_MATH> x) {
        auto temp = std::make_shared<piecewise_1D_node<T, SAFE_MATH>> (d, x)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash(); i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::cache.find(i) ==
                leaf_node<T, SAFE_MATH>::cache.end()) {
                leaf_node<T, SAFE_MATH>::cache[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::cache[i])) {
                return leaf_node<T, SAFE_MATH>::cache[i];
            }
        }
        assert(false && "Should never reach.");
    }

///  Convenience type alias for shared piecewise 1D nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_piecewise_1D = std::shared_ptr<piecewise_1D_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a piecewise 1D node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_piecewise_1D<T, SAFE_MATH> piecewise_1D_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<piecewise_1D_node<T, SAFE_MATH>> (x);
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
///    z(x,y) = Σ_i,3Σ_j,3 c_ij*x^i*y^j                                      (1)
///
///  The aguments x and y are assumed to be the normalized arguments
///
///    x_norm = (x - xmin)/dx - i                                            (2)
///    y_norm = (y - ymin)/dy - j                                            (3)
///
///  To avoid tracking the indices i and j which normaizes x and y to a zero to
///  one interval the coefficients should be normalized to
///
///    c00'_ij = Σ_k,3Σ_l,3 (-i)^k*(-j)^l*ckl_ij                             (4)
///    c10'_ij = Σ_k,3Σ_l,3 k*(-i)^(k-1)*(-j)^l*ckl_ij                       (5)
///    c01'_ij = Σ_k,3Σ_l,3 l*(-i)^k*(-j)^(l-1)*ckl_ij                       (6)
///    c20'_ij = Σ_k,3Σ_l,3 Max(2*k-3,0)*(-i)^(k-2)*(-j)^l*ckl_ij            (7)
///    c02'_ij = Σ_k,3Σ_l,3 Max(2*l-3,0)*(-i)^k*(-j)^(l-2)*ckl_ij            (8)
///    c30'_ij = Σ_k,3Σ_l,3 Max(k-2,0)*(-i)^(k-3)*(-j)^l*ckl_ij              (9)
///    c03'_ij = Σ_k,3Σ_l,3 Max(l-2,0)*(-i)^k*(-j)^(l-3)*ckl_ij             (10)
///    c11'_ij = Σ_k,3Σ_l,3 k*l*(-i)^(k-1)*(-j)^(j-1)                       (11)
///    c21'_ij = Σ_k,3Σ_l,3 Max(2*k-3,0)*l*(-i)^(k-2)*(-j)^(j-1)            (12)
///    c12'_ij = Σ_k,3Σ_l,3 k*Max(2*l-3,0)*(-i)^(k-1)*(-j)^(j-2)            (13)
///    c31'_ij = Σ_k,3Σ_l,3 Max(k-2,0)*l*(-i)^(k-3)*(-j)^(j-1)              (14)
///    c13'_ij = Σ_k,3Σ_l,3 k*Max(l-2,0)*(-i)^(k-1)*(-j)^(j-3)              (15)
///    c22'_ij = Σ_k,3Σ_l,3 Max(2*k-3,0)*Max(2*l-3,0)*(-i)^(k-2)*(-j)^(j-2) (16)
///    c32'_ij = Σ_k,3Σ_l,3 Max(k-2,0)*Max(2*l-3,0)*(-i)^(k-3)*(-j)^(j-2)   (17)
///    c23'_ij = Σ_k,3Σ_l,3 Max(2*k-3,0)*Max(l-2,0)*(-i)^(k-2)*(-j)^(j-3)   (18)
///    c33'_ij = Σ_k,3Σ_l,3 Max(k-2,0)*Max(l-2,0)*(-i)^(k-3)*(-j)^(j-3)     (19)
///
///  This makes the normalized arguments (6,7) become
///
///    x_norm' = (x - xmin)/dx                                              (20)
///    y_norm' = (y - ymin)/dy                                              (21)
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class piecewise_2D_node final : public branch_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @params[in] d Backend buffer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(const backend::buffer<T> &d) {
            std::string temp;
            for (size_t i = 0, ie = d.size(); i < ie; i++) {
                temp += jit::format_to_string(d[i]);
            }

            return temp;
        }

//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string with the argument.
///
///  @params[in] d Backend buffer.
///  @params[in] x X argument.
///  @params[in] y Y argument.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(const backend::buffer<T> &d,
                                     shared_leaf<T, SAFE_MATH> x,
                                     shared_leaf<T, SAFE_MATH> y) {
            return piecewise_2D_node::to_string(d) +
                   jit::format_to_string(x->get_hash()) +
                   jit::format_to_string(y->get_hash());
        }

//------------------------------------------------------------------------------
///  @brief Stores the data in a hash.
///
///  @params[in] d Backend buffer.
///  @returns The hash the node is stored in.
//------------------------------------------------------------------------------
        static size_t hash_data(const backend::buffer<T> &d) {
            const size_t h = std::hash<std::string>{} (piecewise_2D_node::to_string(d));
            for (size_t i = h; i < std::numeric_limits<size_t>::max(); i++) {
                if (leaf_node<T, SAFE_MATH>::backend_cache.find(i) ==
                    leaf_node<T, SAFE_MATH>::backend_cache.end()) {
                    leaf_node<T, SAFE_MATH>::backend_cache[i] = d;
                    return i;
                } else if (d == leaf_node<T, SAFE_MATH>::backend_cache[i]) {
                    return i;
                }
            }
            assert(false && "Should never reach.");
        }

///  Data buffer hash.
        const size_t data_hash;
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
        piecewise_2D_node(const backend::buffer<T> &d,
                          const size_t n,
                          shared_leaf<T, SAFE_MATH> x,
                          shared_leaf<T, SAFE_MATH> y) :
        branch_node<T, SAFE_MATH> (x, y, piecewise_2D_node::to_string(d, x, y)),
        data_hash(piecewise_2D_node::hash_data(d)),
        num_columns(n) {
            assert(d.size()/n &&
                   "Expected the data buffer to be a multiple of the number of columns.");
        }

//------------------------------------------------------------------------------
///  @brief Get the number of columns.
///
///  @returns The number of columns in the constant.
//------------------------------------------------------------------------------
        size_t get_num_columns() const {
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
            return leaf_node<T, SAFE_MATH>::backend_cache[data_hash];
        }

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  If all the values in the data buffer are the same. Reduce to a single
///  constant.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
            if (evaluate().is_same()) {
                return constant<T, SAFE_MATH> (evaluate().at(0));
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> df(shared_leaf<T, SAFE_MATH> x) {
            return zero<T, SAFE_MATH> ();
        }

//------------------------------------------------------------------------------
///  @brief Compile preamble.
///
///  @params[in,out] stream     String buffer stream.
///  @params[in,out] registers  List of defined registers.
///  @params[in,out] visited    List of visited nodes.
///  @params[in,out] textures1d List of 1D textures.
///  @params[in,out] textures2d List of 2D textures.
//------------------------------------------------------------------------------
        virtual void compile_preamble(std::ostringstream &stream,
                                      jit::register_map &registers,
                                      jit::visiter_map &visited,
                                      jit::texture1d_list &textures1d,
                                      jit::texture2d_list &textures2d) {
            if (visited.find(this) == visited.end()) {
                if (registers.find(leaf_node<T, SAFE_MATH>::backend_cache[data_hash].data()) == registers.end()) {
                    registers[leaf_node<T, SAFE_MATH>::backend_cache[data_hash].data()] =
                        jit::to_string('a', leaf_node<T, SAFE_MATH>::backend_cache[data_hash].data());
                    const size_t length = leaf_node<T, SAFE_MATH>::backend_cache[data_hash].size();
                    if constexpr (jit::use_metal<T> () || jit::use_cuda()) {
                        textures2d.emplace_back(leaf_node<T, SAFE_MATH>::backend_cache[data_hash].data(),
                                                std::array<size_t, 2> ({length/num_columns, num_columns}));
                    } else {
                        stream << "const ";
                        jit::add_type<T> (stream);
                        stream << " " << registers[leaf_node<T, SAFE_MATH>::backend_cache[data_hash].data()] << "[] = {";
                        if constexpr (jit::is_complex<T> ()) {
                            jit::add_type<T> (stream);
                        }
                        stream << leaf_node<T, SAFE_MATH>::backend_cache[data_hash][0];
                        for (size_t i = 1; i < length; i++) {
                            stream << ", ";
                            if constexpr (jit::is_complex<T> ()) {
                                jit::add_type<T> (stream);
                            }
                            stream << leaf_node<T, SAFE_MATH>::backend_cache[data_hash][i];
                        }
                        stream << "};" << std::endl;
                    }
                    visited.insert(this);
                }
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  This node first evaluates the value of the argument then chooses the correct
///  piecewise index. This assumes that the argument is
///
///    x' = (x - xmin)/dx                                                    (1)
///    y' = (y - ymin)/dy                                                    (2)
///
///  and the spline coefficients are of the form.
///
///    c00'_ij = Σ_k,3Σ_l,3 (-i)^k*(-j)^l*ckl_ij                             (3)
///    c10'_ij = Σ_k,3Σ_l,3 k*(-i)^(k-1)*(-j)^l*ckl_ij                       (4)
///    c01'_ij = Σ_k,3Σ_l,3 l*(-i)^k*(-j)^(l-1)*ckl_ij                       (5)
///    c20'_ij = Σ_k,3Σ_l,3 Max(2*k-3,0)*(-i)^(k-2)*(-j)^l*ckl_ij            (6)
///    c02'_ij = Σ_k,3Σ_l,3 Max(2*l-3,0)*(-i)^k*(-j)^(l-2)*ckl_ij            (7)
///    c30'_ij = Σ_k,3Σ_l,3 Max(k-2,0)*(-i)^(k-3)*(-j)^l*ckl_ij              (8)
///    c03'_ij = Σ_k,3Σ_l,3 Max(l-2,0)*(-i)^k*(-j)^(l-3)*ckl_ij              (9)
///    c11'_ij = Σ_k,3Σ_l,3 k*l*(-i)^(k-1)*(-j)^(j-1)                       (10)
///    c21'_ij = Σ_k,3Σ_l,3 Max(2*k-3,0)*l*(-i)^(k-2)*(-j)^(j-1)            (11)
///    c12'_ij = Σ_k,3Σ_l,3 k*Max(2*l-3,0)*(-i)^(k-1)*(-j)^(j-2)            (12)
///    c31'_ij = Σ_k,3Σ_l,3 Max(k-2,0)*l*(-i)^(k-3)*(-j)^(j-1)              (13)
///    c13'_ij = Σ_k,3Σ_l,3 k*Max(l-2,0)*(-i)^(k-1)*(-j)^(j-3)              (14)
///    c22'_ij = Σ_k,3Σ_l,3 Max(2*k-3,0)*Max(2*l-3,0)*(-i)^(k-2)*(-j)^(j-2) (15)
///    c32'_ij = Σ_k,3Σ_l,3 Max(k-2,0)*Max(2*l-3,0)*(-i)^(k-3)*(-j)^(j-2)   (16)
///    c23'_ij = Σ_k,3Σ_l,3 Max(2*k-3,0)*Max(l-2,0)*(-i)^(k-2)*(-j)^(j-3)   (17)
///    c33'_ij = Σ_k,3Σ_l,3 Max(k-2,0)*Max(l-2,0)*(-i)^(k-3)*(-j)^(j-3)     (18)
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> x = this->left->compile(stream, registers);
                shared_leaf<T, SAFE_MATH> y = this->right->compile(stream, registers);
                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = ";
                if constexpr (jit::use_cuda()) {
                    if constexpr (jit::is_float<T> ()) {
                        stream << "tex2D<float> (";
                    } else if constexpr (jit::is_double<T> ()) {
                        stream << "to_double(tex2D<uint2> (";
                    } else if constexpr (jit::is_complex<T> () && jit::is_float<T> ()) {
                        stream << "to_cmp_float(tex2D<float2> (";
                    } else {
                        stream << "to_cmp_double(tex2D<uint4> (";
                    }
                }
                stream << registers[leaf_node<T, SAFE_MATH>::backend_cache[data_hash].data()];
                const size_t length = leaf_node<T, SAFE_MATH>::backend_cache[data_hash].size();
                const size_t num_rows = length/num_columns;
                if constexpr (jit::use_metal<T> ()) {
                    stream << ".read(uint2(";
                    compile_index<T> (stream, registers[y.get()], num_columns);
                    stream << ",";
                    compile_index<T> (stream, registers[x.get()], num_rows);
                    stream << ")).r;";
                } else if constexpr (jit::use_cuda()) {
                    stream << ", ";
                    compile_index<T> (stream, registers[y.get()], num_columns);
                    stream << ", ";
                    compile_index<T> (stream, registers[x.get()], num_rows);
                    if constexpr (jit::is_complex<T> () || jit::is_double<T> ()) {
                        stream << ")";
                    }
                    stream << ");";
                }  else {
                    stream << "[min(max((int)";
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
                    stream << ",0), " << length - 1 << ")];";
                }
                stream << std::endl;
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  Assumes both arguments are either set or not set.
///
///  @params[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T, SAFE_MATH> x) {
            auto x_cast = piecewise_2D_cast(x);

            if (x_cast.get()) {
                return this->data_hash == x_cast->data_hash     &&
                       this->left->is_match(x_cast->get_left()) &&
                       this->right->is_match(x_cast->get_right());
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
///
///  Assumes both arguments are either set or not set.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout  << "r\\_" << reinterpret_cast<size_t> (this) << "_{ij}";
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to vizgraph.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> to_vizgraph(std::stringstream &stream,
                                                      jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                const std::string name = jit::to_string('r', this);
                registers[this] = name;
                stream << "    " << name
                       << " [label = \"r_" << reinterpret_cast<size_t> (this)
                       << "_{ij}\", shape = hexagon, style = filled, fillcolor = black, fontcolor = white];" << std::endl;

                auto l = this->left->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[l.get()] << ";" << std::endl;
                auto r = this->right->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[r.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
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
        virtual bool is_all_variables() const {
            return false;
        }

//------------------------------------------------------------------------------
///  @brief Test if the node acts like a power of variable.
///
///  @returns True.
//------------------------------------------------------------------------------
        virtual bool is_power_like() const {
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Get the base of a power.
///
///  @returns The base of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_base() {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Get the exponent of a power.
///
///  @returns The exponent of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_exponent() const {
            return one<T, SAFE_MATH> ();
        }
        
//------------------------------------------------------------------------------
///  @brief Check if the args match.
///
///  @param[in] x Node to match.
///  @returns True if the arguments match.
//------------------------------------------------------------------------------
        bool is_arg_match(shared_leaf<T, SAFE_MATH> x) {
            auto temp = piecewise_2D_cast(x);
            return temp.get()                               &&
                   this->left->is_match(temp->get_left())   &&
                   this->right->is_match(temp->get_right()) &&
                   (num_columns == this->get_num_columns());
        }
    };

//------------------------------------------------------------------------------
///  @brief Define piecewise\_2D convience function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @params[in] d Data to initalize the piecewise constant.
///  @params[in] n Number of columns.
///  @params[in] x Argument.
///  @params[in] y Argument.
///  @returns A reduced sqrt node.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false> 
    shared_leaf<T, SAFE_MATH> piecewise_2D(const backend::buffer<T> &d,
                 const size_t n,
                 shared_leaf<T, SAFE_MATH> x,
                 shared_leaf<T, SAFE_MATH> y) {
        auto temp = std::make_shared<piecewise_2D_node<T, SAFE_MATH>> (d, n, x, y)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash(); i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::cache.find(i) ==
                leaf_node<T, SAFE_MATH>::cache.end()) {
                leaf_node<T, SAFE_MATH>::cache[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::cache[i])) {
                return leaf_node<T, SAFE_MATH>::cache[i];
            }
        }
        assert(false && "Should never reach.");
    }

///  Convenience type alias for shared piecewise 2D nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_piecewise_2D = std::shared_ptr<piecewise_2D_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a piecewise 2D node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_piecewise_2D<T, SAFE_MATH> piecewise_2D_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<piecewise_2D_node<T, SAFE_MATH>> (x);
    }
}

#endif /* piecewise_h */
