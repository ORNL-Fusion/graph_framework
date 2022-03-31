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
//------------------------------------------------------------------------------
///  @brief Vector dot product.
///
///  @param[in] v1x Vector 1 x component.
///  @param[in] v1y Vector 1 y component.
///  @param[in] v1z Vector 1 z component.
///  @param[in] v2x Vector 2 x component.
///  @param[in] v2y Vector 2 y component.
///  @param[in] v2z Vector 2 z component.
///  @returns v1.v2
//------------------------------------------------------------------------------
    template<typename V1X, typename V1Y, typename V1Z,
             typename V2X, typename V2Y, typename V2Z>
    std::shared_ptr<leaf_node> dot(std::shared_ptr<V1X> v1x,
                                   std::shared_ptr<V1Y> v1y,
                                   std::shared_ptr<V1Z> v1z,
                                   std::shared_ptr<V2X> v2x,
                                   std::shared_ptr<V2Y> v2y,
                                   std::shared_ptr<V2Z> v2z) {
        return v1x*v2x + v1y*v2y + v1z*v2z;
    }

//------------------------------------------------------------------------------
///  @brief Vector dot product.
///
///  @param[in] vx Vector x component.
///  @param[in] vy Vector y component.
///  @param[in] vz Vector z component.
///  @returns v1.v2
//------------------------------------------------------------------------------
    template<typename VX, typename VY, typename VZ>
    std::shared_ptr<leaf_node> length(std::shared_ptr<VX> vx,
                                      std::shared_ptr<VY> vy,
                                      std::shared_ptr<VZ> vz) {
        return sqrt(dot(vx, vy, vz, vx, vy, vz));
    }
}

#endif /* vector_h */
