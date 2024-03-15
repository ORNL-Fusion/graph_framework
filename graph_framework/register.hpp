//------------------------------------------------------------------------------
///  @file register.hpp
///  @brief Utilities for writing jit source code.
//------------------------------------------------------------------------------

#ifndef register_h
#define register_h

#include <concepts>
#include <cassert>
#include <map>
#include <set>
#include <sstream>
#include <complex>
#include <type_traits>
#include <limits>
#include <charconv>
#include <array>
#include <utility>

namespace jit {
///  Complex scalar concept.
    template<typename T>
    concept complex_scalar = std::same_as<T, std::complex<float>> ||
                             std::same_as<T, std::complex<double>>;

///  Float scalar concept.
    template<typename T>
    concept float_scalar = std::floating_point<T> || complex_scalar<T>;

///  General scalar concept.
    template<typename T>
    concept scalar = float_scalar<T> || std::integral<T>;

///  float base concept.
    template<typename T>
    concept float_base = std::same_as<T, float> ||
                         std::same_as<T, std::complex<float>>;

///  Double base concept.
    template<typename T>
    concept double_base = std::same_as<T, double> ||
                          std::same_as<T, std::complex<double>>;

///  Verbose output.
    static bool verbose = USE_VERBOSE;

//------------------------------------------------------------------------------
///  @brief Convert a base type to a string.
///
///  @tparam T Base type of the calculation.
///
///  @returns A constant string literal of the type.
//------------------------------------------------------------------------------
    template<float_scalar T>
    std::string type_to_string() {
        if constexpr (float_base<T>) {
            return "float";
        } else {
            return "double";
        }
    }

//------------------------------------------------------------------------------
///  @brief Test to use Cuda
//------------------------------------------------------------------------------
    constexpr bool use_cuda() {
#ifdef USE_CUDA
        return true;
#else
        return false;
#endif
    }

//------------------------------------------------------------------------------
///  @brief Test to use Hip
//------------------------------------------------------------------------------
    constexpr bool use_hip() {
#ifdef USE_HIP
        return true;
#else
        return false;
#endif
    }

//------------------------------------------------------------------------------
///  @brief Test to use metal.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
    template<float_scalar T>
    constexpr bool use_metal() {
#if USE_METAL
        return float_base<T> && !complex_scalar<T>;
#else
        return false;
#endif
    }

//------------------------------------------------------------------------------
///  @brief  Test to use the GPU.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
    template<float_scalar T>
    constexpr bool use_gpu() {
        return use_cuda() || use_hip() || use_metal<T> ();
    }

//------------------------------------------------------------------------------
///  @brief Get smallest integer type.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] max_size The maximum size needed.
///  @returns The smallest integer type as a string.
//------------------------------------------------------------------------------
    template<float_scalar T>
    std::string smallest_int_type(const size_t max_size) {
        if (max_size <= std::numeric_limits<unsigned char>::max()) {
            if constexpr (jit::use_metal<T> ()) {
                return "ushort";
            } else {
                return "unsigned char";
            }
        } else if (max_size <= std::numeric_limits<unsigned short>::max()) {
            if constexpr (jit::use_metal<T> ()) {
                return "ushort";
            } else {
                return "unsigned short";
            }
        } else if (max_size <= std::numeric_limits<unsigned int>::max()) {
            if constexpr (jit::use_metal<T> ()) {
                return "uint";
            } else {
                return "unsigned int";
            }
        } else {
            if constexpr (jit::use_metal<T> ()) {
                return "uint";
            } else {
                return "size_t";
            }
        }
    }

//------------------------------------------------------------------------------
///  @brief Get the type string.
///
///  @tparam T Base type of the calculation.
///
///  @returns The type as a string.
//------------------------------------------------------------------------------
    template<float_scalar T>
    std::string get_type_string() {
        if constexpr (complex_scalar<T>) {
            if constexpr (use_cuda()) {
                return "cuda::std::complex<" + type_to_string<T> () + ">";
            } else {
                return "complex<" + type_to_string<T> () + ">";
            }
        } else {
            return type_to_string<T> ();
        }
    }

//------------------------------------------------------------------------------
///  @brief Write out the node base type to a general stream.
///
///  @tparam T Base type of the calculation.
///
///  @param[in, out] stream Generic stream.
//------------------------------------------------------------------------------
    template<float_scalar T>
    void add_type(std::basic_ostream<char> &stream) {
        stream << get_type_string<T> ();
    }

//------------------------------------------------------------------------------
///  @brief The maximum number of digits to represent a type literal.
///
///  @tparam T Base type of the calculation.
///
///  @returns The maximum number of digits needed.
//------------------------------------------------------------------------------
    template<float_scalar T>
    constexpr int max_digits10() {
        if constexpr (float_base<T>) {
            return std::numeric_limits<float>::max_digits10;
        } else {
            return std::numeric_limits<double>::max_digits10;
        }
    }

//------------------------------------------------------------------------------
///  @brief The maximum value for a base type.
///
///  @tparam T Base type of the calculation.
///
///  @returns The maximum number of digits needed.
//------------------------------------------------------------------------------
    template<float_scalar T>
    constexpr int max_base() {
        if constexpr (float_base<T>) {
            return std::numeric_limits<float>::max();
        } else {
            return std::numeric_limits<double>::max();
        }
    }

//------------------------------------------------------------------------------
///  @brief Convert a value to a string while avoiding locale.
///
///  The standard streams use localization that interferes with multiple
///  threads.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] value Value to convert.
///  @returns String with the value.
//------------------------------------------------------------------------------
    template<scalar T>
    std::string format_to_string(const T value) {
        std::array<char, 36> buffer;
        char *end;
        if constexpr (std::is_same<T, size_t>::value) {
            end = std::to_chars(buffer.begin(),
                                buffer.end(),
                                value, 16).ptr;
        } else if constexpr (complex_scalar<T>) {
            return format_to_string(std::real(value)) + "," +
                   format_to_string(std::imag(value));
        } else {
            end = std::to_chars(buffer.begin(), buffer.end(),
                                value, std::chars_format::general,
                                max_digits10<T> ()).ptr;
        }
        return std::string(buffer.data(), end);
    }

//------------------------------------------------------------------------------
///  @brief Convert a @ref graph::leaf_node pointer to a string.
///
///  This converts the point value into a string of format t######. Where t is
///  of type
///  -# v Variable
///  -# r Register
///  -# i Input
///
///  @tparam NODE Node class type.
///
///  @param[in] prefix  Type prefix for the name.
///  @param[in] pointer Address of the @ref graph::leaf_node.
///  @returns The pointer value as a string.
//------------------------------------------------------------------------------
    template<class NODE>
    std::string to_string(const char prefix,
                          const NODE *pointer) {
        assert((prefix == 'r' || prefix == 'v' ||
                prefix == 'o' || prefix == 'a' ||
                prefix == 'i' || prefix == 's') &&
               "Expected a variable (v), register (r), output (o), array (a), index (i), or state (s) prefix.");
        return std::string(1, prefix) +
               format_to_string(reinterpret_cast<size_t> (pointer));
    }

///  Type alias for mapping node pointers to register names.
    typedef std::map<void *, std::string> register_map;
///  Type alias for counting register usage.
    typedef std::map<void *, size_t> register_usage;
///  Type alias for listing visited nodes.
    typedef std::set<void *> visiter_map;
///  Type alias for indexing 1D textures.
    typedef std::map<void *, size_t> texture1d_list;
///  Type alias for indexing 2D textures.
    typedef std::map<void *, std::array<size_t,2>> texture2d_list;

//------------------------------------------------------------------------------
///  @brief  Define a custom comparator class.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
    template<float_scalar T>
    class float_compare {
    public:
//------------------------------------------------------------------------------
///  @brief Call operator.
///
///  @param[in] left  Left hand side.
///  @param[in] right Right hand side.
//------------------------------------------------------------------------------
        bool operator() (const T &left, const T &right) const {
            if constexpr (complex_scalar<T>) {
                return std::abs(left) < std::abs(right);
            } else {
                return left < right;
            }
        }
    };
}

#endif /* register_h */
