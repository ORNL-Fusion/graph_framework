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
    template<typename BASE>
    constexpr bool is_float() {
        return is_base<BASE, float> ();
    }

//------------------------------------------------------------------------------
///  @brief Test if the base type is double.
///
///  @returns A constant expression true or false type.
//------------------------------------------------------------------------------
    template<typename BASE>
    constexpr bool is_double() {
        return is_base<BASE, double> ();
    }

//------------------------------------------------------------------------------
///  @brief Test if a type is complex.
///
///  @returns A constant expression true or false type.
//------------------------------------------------------------------------------
    template<typename BASE>
    constexpr bool is_complex() {
        return is_complex<BASE, float> () ||
               is_complex<BASE, double> ();
    }

//------------------------------------------------------------------------------
///  @brief Convert a leaf\_node pointer to a string.
///
///  This converts the point value into a string of format t\_######. Where t is
///  of type
///  -# v Variable
///  -# r Register
///
///  @param[in] prefix  Type prefix for the name.
///  @param[in] pointer Address of the @ref{leaf_node}.
///  @returns The pointer value as a string.
//------------------------------------------------------------------------------
    template<class NODE>
    std::string to_string(const char prefix,
                          const NODE *pointer) {
        assert((prefix == 'r' || prefix == 'v' || prefix == 'o' ) &&
               "Expected a variable (v) or register (r) prefix.");
        std::stringstream stream;
        stream << prefix << "_" << reinterpret_cast<size_t> (pointer);
        return stream.str();
    }

//------------------------------------------------------------------------------
///  @brief Convert a base type to a string.
///
///  @returns A constant string literal of the type.
//------------------------------------------------------------------------------
    template<typename BASE>
    std::string type_to_string() {
        if constexpr (is_float<BASE> ()) {
            return "float";
        } else if constexpr (is_double<BASE> ()) {
            return "double";
        } else {
            static_assert(!is_float<BASE> () &&
                          !is_double<BASE> (), "Unsupported base type.");
        }
    }

//------------------------------------------------------------------------------
///  @brief Write out the node base type to a general stream.
///
///  @param[in, out] stream Generic stream.
//------------------------------------------------------------------------------
    template<typename BASE>
    void add_type_base(std::basic_ostream<char> &stream) {
        if constexpr (is_complex<BASE> ()) {
#ifdef USE_CUDA
            stream << "cuda::";
#endif
            stream << "std::complex<";
        }
        stream << type_to_string<BASE> ();
        if constexpr (is_complex<BASE> ()) {
            stream << ">";
        }
    }

//------------------------------------------------------------------------------
///  @brief Write out the node base type to the string buffer.
///
///  @param[in,out] stream String buffer stream.
//------------------------------------------------------------------------------
    template<class BACKEND>
    void add_type(std::stringstream &stream) {
        add_type_base<typename BACKEND::base> (stream);
    }

//------------------------------------------------------------------------------
///  @brief The maximum number of digits to represent a type literal.
///
///  @returns The maximum number of digits needed.
//------------------------------------------------------------------------------
    template<typename BASE>
    constexpr int max_digits10() {
        if constexpr (is_float<BASE> ()) {
            return std::numeric_limits<float>::max_digits10;
        } else {
            return std::numeric_limits<double>::max_digits10;;
        }
    }

///  Type alias for mapping node pointers to register names.
    template<class NODE>
    using register_map = std::map<NODE *, std::string>;
}

#endif /* register_h */
