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
///  @brief Write out the node base type to the string buffer.
///
///  @param[in,out] stream String buffer stream.
//------------------------------------------------------------------------------
    template<class NODE>
    void add_type(std::stringstream &stream) {
        if constexpr (std::is_same<typename NODE::backend::base, float>::value) {
            stream << "float";
        } else if constexpr (std::is_same<typename NODE::backend::base, double>::value) {
            stream << "double";
        } else if constexpr (std::is_same<typename NODE::backend::base,
                                          std::complex<float>>::value) {
            stream << "cuda::std::complex<float>";
        } else if constexpr (std::is_same<typename NODE::backend::base,
                                          std::complex<double>>::value) {
            stream << "cuda::std::complex<double>";
        }
    }

///  Type alias for mapping node pointers to register names.
    template<class NODE>
    using register_map = std::map<NODE *, std::string>;
}

#endif /* register_h */
