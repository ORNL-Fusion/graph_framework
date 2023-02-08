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

#include "cpu_backend.hpp"

namespace jit {
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
///  @brief Write out the node base type to a general stream.
///
///  @param[in, out] stream Generic stream.
//------------------------------------------------------------------------------
    template<typename BASE>
    void add_type_base(std::basic_ostream<char> &stream) {
        if constexpr (std::is_same<BASE, float>::value) {
            stream << "float";
        } else if constexpr (std::is_same<BASE, double>::value) {
            stream << "double";
        } else if constexpr (std::is_same<BASE, std::complex<float>>::value) {
#ifdef USE_CUDA
            stream << "cuda::std::complex<float>";
#else
            stream << "std::complex<float>";
#endif
        } else if constexpr (std::is_same<BASE, std::complex<double>>::value) {
#ifdef USE_CUDA
            stream << "cuda::std::complex<double>";
#else
            stream << "std::complex<double>";
#endif
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

///  Type alias for mapping node pointers to register names.
    template<class NODE>
    using register_map = std::map<NODE *, std::string>;
}

#endif /* register_h */
