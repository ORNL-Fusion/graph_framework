//------------------------------------------------------------------------------
///  vector.hpp
///  graph_framework
///
///  Created by Cianciosa, Mark R. on 3/31/22.
///  Copyright © 2022 Cianciosa, Mark R. All rights reserved.
//------------------------------------------------------------------------------

#ifndef vector_h
#define vector_h

#include "node.hpp"

namespace graph {
//******************************************************************************
//  Vector interface.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class to represent vector quantities.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    class vector_quantity : public std::enable_shared_from_this<vector_quantity<T, SAFE_MATH>> {
    protected:
///  X component of the vector.
        shared_leaf<T, SAFE_MATH> x;
///  Y component of the vector.
        shared_leaf<T, SAFE_MATH> y;
///  Z component of the vector.
        shared_leaf<T, SAFE_MATH> z;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a new vector_quantity.
///
///  @params[in] x X vector component.
///  @params[in] y Y vector component.
///  @params[in] z Z Vector component.
//------------------------------------------------------------------------------
        vector_quantity(shared_leaf<T, SAFE_MATH> x,
                        shared_leaf<T, SAFE_MATH> y,
                        shared_leaf<T, SAFE_MATH> z) :
        x(x), y(y), z(z) {}

//------------------------------------------------------------------------------
///  @brief Get the x component.
///
///  @return x
//------------------------------------------------------------------------------
        shared_leaf<T, SAFE_MATH> get_x() const {
            return x;
        }


//------------------------------------------------------------------------------
///  @brief Get the y component.
///
///  @return y
//------------------------------------------------------------------------------
        shared_leaf<T, SAFE_MATH> get_y() const {
            return y;
        }

//------------------------------------------------------------------------------
///  @brief Get the z component.
///
///  @return z
//------------------------------------------------------------------------------
        shared_leaf<T, SAFE_MATH> get_z() const {
            return z;
        }

//------------------------------------------------------------------------------
///  @brief Vector dot product.
///
///  @params[in] v2 Second vector.
///  @returns v1.v2
//------------------------------------------------------------------------------
        shared_leaf<T, SAFE_MATH>
        dot(std::shared_ptr<vector_quantity<T, SAFE_MATH>> v2) {
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
        std::shared_ptr<vector_quantity<T, SAFE_MATH>>
        cross(std::shared_ptr<vector_quantity<T, SAFE_MATH>> v2) {
            return std::make_shared<vector_quantity<T, SAFE_MATH>> (y*v2->get_z() - z*v2->get_y(),
                                                                    z*v2->get_x() - x*v2->get_z(),
                                                                    x*v2->get_y() - y*v2->get_x());
        }

//------------------------------------------------------------------------------
///  @brief Get the length of the vector.
///
///  @returns |V|
//------------------------------------------------------------------------------
        shared_leaf<T, SAFE_MATH> length() {
            return sqrt(this->dot(this->shared_from_this()));
        }

//------------------------------------------------------------------------------
///  @brief Get the unit vector.
///
///  @returns v_hat
//------------------------------------------------------------------------------
        std::shared_ptr<vector_quantity<T, SAFE_MATH>> unit() {
            auto l = length();
            return std::make_shared<vector_quantity<T, SAFE_MATH>> (x/l, y/l, z/l);
        }
    };

///  Convenience type for shared vector quantities.
    template<typename T, bool SAFE_MATH=false>
    using shared_vector = std::shared_ptr<vector_quantity<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Build a shared vector quantity.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    shared_vector<T, SAFE_MATH> vector(shared_leaf<T, SAFE_MATH> x,
                                       shared_leaf<T, SAFE_MATH> y,
                                       shared_leaf<T, SAFE_MATH> z) {
        return std::make_shared<vector_quantity<T, SAFE_MATH>> (x, y, z);
    }
}

#endif /* vector_h */
