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
    template<typename VX, typename VY, typename VZ>
    class vector_quantity {
    protected:
///  X component of the vector.
        VX x;
///  Y component of the vector.
        VY y;
///  Z component of the vector.
        VZ z;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a new vector_quantity.
///
///  @param[in] x X vector component.
///  @param[in] y Y vector component.
///  @param[in] z Z Vector component.
//------------------------------------------------------------------------------
        vector_quantity(VX x, VY y, VZ z) :
        x(x), y(y), z(z) {}

//------------------------------------------------------------------------------
///  @brief Get the x component.
///
///  @return x
//------------------------------------------------------------------------------
        VX get_x() const {
            return x;
        }


//------------------------------------------------------------------------------
///  @brief Get the y component.
///
///  @return y
//------------------------------------------------------------------------------
        VY get_y() const {
            return y;
        }

//------------------------------------------------------------------------------
///  @brief Get the z component.
///
///  @return z
//------------------------------------------------------------------------------
        VZ get_z() const {
            return z;
        }

//------------------------------------------------------------------------------
///  @brief Vector dot product.
///
///  @param[in] v2 Second vector.
///  @returns v1.v2
//------------------------------------------------------------------------------
        VX dot(std::shared_ptr<vector_quantity<VX, VY, VZ>> v2) {
            return x*v2->get_x() +
                   y*v2->get_y() +
                   z*v2->get_z();
        }

//------------------------------------------------------------------------------
///  @brief Vector cross product.
///
///  @param[in] v2 Second vector.
///  @returns v1 X v2
//------------------------------------------------------------------------------
        std::shared_ptr<vector_quantity<VX, VY, VZ>>
        cross(std::shared_ptr<vector_quantity<VX, VY, VZ>> v2) {
            return std::make_shared<vector_quantity<VX, VY, VZ>> (y*v2->get_z() - z*v2->get_y(),
                                                                  z*v2->get_x() - x*v2->get_z(),
                                                                  x*v2->get_y() - y*v2->get_x());
        }

//------------------------------------------------------------------------------
///  @brief Get the length of the vector.
///
///  @returns |V|
//------------------------------------------------------------------------------
        VX length() {
            return sqrt(x*x + y*y + z*z);
        }

//------------------------------------------------------------------------------
///  @brief Get the unit vector.
///
///  @returns v_hat
//------------------------------------------------------------------------------
        std::shared_ptr<vector_quantity<VX, VY, VZ>> unit() {
            auto l = length();
            return std::make_shared<vector_quantity<VX, VY, VZ>> (x/l, y/l, z/l);
        }
    };

///  Convenience type for shared vector quantities.
    template<typename VX, typename VY, typename VZ>
    using shared_vector = std::shared_ptr<vector_quantity<VX, VY, VZ>>;

//------------------------------------------------------------------------------
///  @brief Build a shared vector quantity.
//------------------------------------------------------------------------------
    template<typename VX, typename VY, typename VZ>
    shared_vector<VX, VY, VZ> vector(VX x, VY y, VZ z) {
        return std::make_shared<vector_quantity<VX, VY, VZ>> (x, y, z);
    }
}

#endif /* vector_h */
