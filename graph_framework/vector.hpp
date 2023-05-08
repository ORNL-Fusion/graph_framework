//------------------------------------------------------------------------------
///  vector.hpp
///  graph_framework
///
///  Created by Cianciosa, Mark R. on 3/31/22.
///  Copyright © 2022 Cianciosa, Mark R. All rights reserved.
//------------------------------------------------------------------------------

#ifndef vector_h
#define vector_h

#include "math.hpp"

namespace graph {
//******************************************************************************
//  Vector interface.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class to represent vector quantities.
//------------------------------------------------------------------------------
    template<typename T>
    class vector_quantity {
    protected:
///  X component of the vector.
        shared_leaf<T> x;
///  Y component of the vector.
        shared_leaf<T> y;
///  Z component of the vector.
        shared_leaf<T> z;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a new vector_quantity.
///
///  @params[in] x X vector component.
///  @params[in] y Y vector component.
///  @params[in] z Z Vector component.
//------------------------------------------------------------------------------
        vector_quantity(shared_leaf<T> x,
                        shared_leaf<T> y,
                        shared_leaf<T> z) :
        x(x), y(y), z(z) {}

//------------------------------------------------------------------------------
///  @brief Get the x component.
///
///  @return x
//------------------------------------------------------------------------------
        shared_leaf<T> get_x() const {
            return x;
        }


//------------------------------------------------------------------------------
///  @brief Get the y component.
///
///  @return y
//------------------------------------------------------------------------------
        shared_leaf<T> get_y() const {
            return y;
        }

//------------------------------------------------------------------------------
///  @brief Get the z component.
///
///  @return z
//------------------------------------------------------------------------------
        shared_leaf<T> get_z() const {
            return z;
        }

//------------------------------------------------------------------------------
///  @brief Vector dot product.
///
///  @params[in] v2 Second vector.
///  @returns v1.v2
//------------------------------------------------------------------------------
        shared_leaf<T> dot(std::shared_ptr<vector_quantity<T>> v2) {
            return x*v2->get_x() +
                   y*v2->get_y() +
                   z*v2->get_z();
        }

//------------------------------------------------------------------------------
///  @brief Vector cross product.
///
///  @params[in] v2 Second vector.
///  @returns v1 X v2
//------------------------------------------------------------------------------
        std::shared_ptr<vector_quantity<T>>
        cross(std::shared_ptr<vector_quantity<T>> v2) {
            return std::make_shared<vector_quantity<T>> (y*v2->get_z() - z*v2->get_y(),
                                                         z*v2->get_x() - x*v2->get_z(),
                                                         x*v2->get_y() - y*v2->get_x());
        }

//------------------------------------------------------------------------------
///  @brief Get the length of the vector.
///
///  @returns |V|
//------------------------------------------------------------------------------
        shared_leaf<T> length() {
            return sqrt(x*x + y*y + z*z);
        }

//------------------------------------------------------------------------------
///  @brief Get the unit vector.
///
///  @returns v_hat
//------------------------------------------------------------------------------
        std::shared_ptr<vector_quantity<T>> unit() {
            auto l = length();
            return std::make_shared<vector_quantity<T>> (x/l, y/l, z/l);
        }
    };

///  Convenience type for shared vector quantities.
    template<typename T>
    using shared_vector = std::shared_ptr<vector_quantity<T>>;

//------------------------------------------------------------------------------
///  @brief Build a shared vector quantity.
//------------------------------------------------------------------------------
    template<typename T>
    shared_vector<T> vector(shared_leaf<T> x,
                            shared_leaf<T> y,
                            shared_leaf<T> z) {
        return std::make_shared<vector_quantity<T>> (x, y, z);
    }
}

#endif /* vector_h */
