//------------------------------------------------------------------------------
///  @file coordinates.hpp
///  @brief Define graph generation routines for different coodinates.
///
///  Defines graphs for different coordinate systems.
//------------------------------------------------------------------------------

#ifndef coordinates_h
#define coordinates_h

#include <array>

#include "vector.hpp"

namespace coordinates {
//******************************************************************************
//  Cartesian
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing cartesian coodinates.
//------------------------------------------------------------------------------
    template<typename T>
    class cartesian {
    protected:
///  Base coordinates.
        std::array<graph::shared_leaf<T>, 3> base;
        
//------------------------------------------------------------------------------
///  @brief Construct cartesian coordinates.
///
///  @params[in] size    Number of elements in the variable.
///  @params[in] symbols Symbols for the coordinates.
//------------------------------------------------------------------------------
        cartesian(const size_t size,
                  const std::array<std::string, 3> symbols) :
        base({graph::variable<T> (size, symbols.at(0)),
              graph::variable<T> (size, symbols.at(1)),
              graph::variable<T> (size, symbols.at(2))}) {}
        
    public:
//------------------------------------------------------------------------------
///  @brief Construct cartesian coordinates.
///
///  @params[in] size Number of elements in the variable.
//------------------------------------------------------------------------------
        cartesian(const size_t size) :
        base({graph::variable<T> (size, "x"),
              graph::variable<T> (size, "y"),
              graph::variable<T> (size, "z")}) {}

//------------------------------------------------------------------------------
///  @brief Convert to cartesian.
///
///  @returns The base in cartesian coordinates.
//------------------------------------------------------------------------------
        virtual std::array<graph::shared_leaf<T>, 3> to_cartesian() const {
            return base;
        }

//------------------------------------------------------------------------------
///  @brief Get basis vectors.
///
///  @returns The basis vectors in cartesian coordinates.
//------------------------------------------------------------------------------
        virtual std::array<graph::shared_vector<T>, 3> get_basis() {
            return {graph::vector(graph::one<T> (),
                                  graph::one<T> (),
                                  graph::one<T> ()),
                    graph::vector(graph::one<T> (),
                                  graph::one<T> (),
                                  graph::one<T> ()),
                    graph::vector(graph::one<T> (),
                                  graph::one<T> (),
                                  graph::one<T> ())};
        }
    };
}

#endif /* coordinates_h */
