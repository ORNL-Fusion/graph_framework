//
//  register.hpp
//  graph_framework
//
//  Created by Cianciosa, Mark on 12/8/22.
//  Copyright © 2022 Cianciosa, Mark R. All rights reserved.
//

#ifndef register_h
#define register_h

#include <map>
#include <sstream>
#include <complex>
#include <type_traits>

namespace jit {
//------------------------------------------------------------------------------
///  @brief Test if a type is complex.
///
///  @returns A constant expression true or false type.
//------------------------------------------------------------------------------
    template<typename BASE, typename T>
    constexpr bool is_complex() {
        return std::is_same<BASE, std::complex<T>>::value;
    }

//------------------------------------------------------------------------------
///  @brief Test if the base type is float.
///
///  @returns A constant expression true or false type.
//------------------------------------------------------------------------------
    template<typename BASE, typename T>
    constexpr bool is_base() {
        return is_complex<BASE, T> () || std::is_same<BASE, T>::value;
    }

//------------------------------------------------------------------------------
///  @brief Test if the base type is float.
///
///  @returns A constant expression true or false type.
//------------------------------------------------------------------------------
    template<typename T>
    constexpr bool is_float() {
        return is_base<T, float> ();
    }

//------------------------------------------------------------------------------
///  @brief Test if the base type is double.
///
///  @returns A constant expression true or false type.
//------------------------------------------------------------------------------
    template<typename T>
    constexpr bool is_double() {
        return is_base<T, double> ();
    }

//------------------------------------------------------------------------------
///  @brief Test if a type is complex.
///
///  @returns A constant expression true or false type.
//------------------------------------------------------------------------------
    template<typename T>
    constexpr bool is_complex() {
        return is_complex<T, float> () ||
               is_complex<T, double> ();
    }

//------------------------------------------------------------------------------
///  @brief Convert a leaf\_node pointer to a string.
///
///  This converts the point value into a string of format t\_######. Where t is
///  of type
///  -# v Variable
///  -# r Register
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
        std::stringstream stream;
        stream << prefix << "_" << reinterpret_cast<size_t> (pointer);
        return stream.str();
    }

//------------------------------------------------------------------------------
///  @brief Convert a base type to a string.
///
///  @returns A constant string literal of the type.
//------------------------------------------------------------------------------
    template<typename T>
    std::string type_to_string() {
        if constexpr (is_float<T> ()) {
            return "float";
        } else if constexpr (is_double<T> ()) {
            return "double";
        } else {
            static_assert(!is_float<T> () &&
                          !is_double<T> (), "Unsupported base type.");
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
//------------------------------------------------------------------------------
    template<typename T>
    constexpr bool use_metal() {
#if USE_METAL
        return is_float<T>() && !is_complex<T> ();
#else
        return false;
#endif
    }

//------------------------------------------------------------------------------
///  @brief  Test to use the GPU.
//------------------------------------------------------------------------------
    template<typename T>
    constexpr bool use_gpu() {
        return use_cuda() || use_metal<T> ();
    }

//------------------------------------------------------------------------------
///  @brief Write out the node base type to a general stream.
///
///  @params[in, out] stream Generic stream.
//------------------------------------------------------------------------------
    template<typename T>
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
///  @returns The maximum number of digits needed.
//------------------------------------------------------------------------------
    template<typename T>
    constexpr int max_digits10() {
        if constexpr (is_float<T> ()) {
            return std::numeric_limits<float>::max_digits10;
        } else {
            return std::numeric_limits<double>::max_digits10;;
        }
    }

///  Type alias for mapping node pointers to register names.
    using register_map = std::map<void *, std::string>;
}

#endif /* register_h */
