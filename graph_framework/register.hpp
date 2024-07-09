//------------------------------------------------------------------------------
///  @file register.hpp
///  @brief Utilities for writting jit source code.
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

///  Verbose output.
    static bool verbose = false;

//------------------------------------------------------------------------------
///  @brief Test if a type is complex.
///
///  @tparam BASE Base type.
///  @tparam T    Type to check against.
///
///  @returns A constant expression true or false type.
//------------------------------------------------------------------------------
    template<float_scalar BASE, std::floating_point T>
    constexpr bool is_complex() {
        return std::is_same<BASE, std::complex<T>>::value;
    }

//------------------------------------------------------------------------------
///  @brief Test if the base type is float.
///
///  @tparam BASE Base type.
///  @tparam T    Type to check against.
///
///  @returns A constant expression true or false type.
//------------------------------------------------------------------------------
    template<float_scalar BASE, std::floating_point T>
    constexpr bool is_base() {
        return is_complex<BASE, T> () || std::is_same<BASE, T>::value;
    }

//------------------------------------------------------------------------------
///  @brief Test if the base type is float.
///
///  @tparam T Base type of the calculation.
///
///  @returns A constant expression true or false type.
//------------------------------------------------------------------------------
    template<float_scalar T>
    constexpr bool is_float() {
        return is_base<T, float> ();
    }

//------------------------------------------------------------------------------
///  @brief Test if the base type is double.
///
///  @tparam T Base type of the calculation.
///
///  @returns A constant expression true or false type.
//------------------------------------------------------------------------------
    template<float_scalar T>
    constexpr bool is_double() {
        return is_base<T, double> ();
    }

//------------------------------------------------------------------------------
///  @brief Test if a type is complex.
///
///  @tparam T Base type of the calculation.
///
///  @returns A constant expression true or false type.
//------------------------------------------------------------------------------
    template<float_scalar T>
    constexpr bool is_complex() {
        return is_complex<T, float> () ||
               is_complex<T, double> ();
    }

//------------------------------------------------------------------------------
///  @brief Convert a base type to a string.
///
///  @tparam T Base type of the calculation.
///
///  @returns A constant string literal of the type.
//------------------------------------------------------------------------------
    template<float_scalar T>
    std::string type_to_string() {
        if constexpr (is_float<T> ()) {
            return "float";
        } else if constexpr (is_double<T> ()) {
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
///  @brief Test to use metal.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
    template<float_scalar T>
    constexpr bool use_metal() {
#if USE_METAL
        return is_float<T>() && !is_complex<T> ();
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
        return use_cuda() || use_metal<T> ();
    }

//------------------------------------------------------------------------------
///  @brief Write out the node base type to a general stream.
///
///  @tparam T Base type of the calculation.
///
///  @params[in, out] stream Generic stream.
//------------------------------------------------------------------------------
    template<float_scalar T>
    void add_type(std::basic_ostream<char> &stream) {
        if constexpr (is_complex<T> ()) {
            if constexpr (use_cuda()) {
                stream << "cuda::";
            }
            stream << "std::complex<";
        }
        stream << type_to_string<T> ();
        if constexpr (is_complex<T> ()) {
            stream << ">";
        }
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
        if constexpr (is_float<T> ()) {
            return std::numeric_limits<float>::max_digits10;
        } else {
            return std::numeric_limits<double>::max_digits10;
        }
    }

//------------------------------------------------------------------------------
///  @brief Convert a value to a string while avoiding locale.
///
///  The standard streams use localizarion that interfers with multiple threads.
///
///  @tparam T Base type of the calculation.
///
///  @params[in] value Value to convert.
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
        } else if constexpr (is_complex<T> ()) {
            return format_to_string(std::real(value)) + " " +
                   format_to_string(std::imag(value));
        } else {
            end = std::to_chars(buffer.begin(), buffer.end(),
                                value, std::chars_format::general,
                                max_digits10<T> ()).ptr;
        }
        return std::string(buffer.data(), end);
    }

//------------------------------------------------------------------------------
///  @brief Convert a leaf\_node pointer to a string.
///
///  This converts the point value into a string of format t\_######. Where t is
///  of type
///  -# v Variable
///  -# r Register
///
///  @tparam NODE Node class type.
///
///  @params[in] prefix  Type prefix for the name.
///  @params[in] pointer Address of the @ref{leaf_node}.
///  @returns The pointer value as a string.
//------------------------------------------------------------------------------
    template<class NODE>
    std::string to_string(const char prefix,
                          const NODE *pointer) {
        assert((prefix == 'r' || prefix == 'v' ||
                prefix == 'o' || prefix == 'a') &&
               "Expected a variable (v), register (r), output (o) or array (a) prefix.");
        return std::string(1, prefix) + "_" +
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
///  @brief  Define a custom comparitor class.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
    template<float_scalar T>
    class float_compare {
    public:
//------------------------------------------------------------------------------
///  @brief Call operator.
///
///  @params[in] left  Left hand side.
///  @params[in] right Right hand side.
//------------------------------------------------------------------------------
        bool operator() (const T &left, const T &right) const {
            if constexpr (is_complex<T> ()) {
                return std::abs(left) < std::abs(right);
            } else {
                return left < right;
            }
        }
    };
}

#endif /* register_h */
