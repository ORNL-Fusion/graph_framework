//------------------------------------------------------------------------------
///  @file cublic\_splines.hpp
///  @brief Creates function for cubic spline interpolated quantities.
///
///  Defines a tree of operations that allows automatic differentiation.
//------------------------------------------------------------------------------

#ifndef cublic_splines_h
#define cublic_splines_h

#include <vector>

#include "arithmetic.hpp"

namespace spline {
//------------------------------------------------------------------------------
///  @brief Build a 1D spline.
///
///  @params[in] min_x     Minimum argument.
///  @params[in] dx        Grid spacing.
///  @params[in] c0_buffer Zeroth coefficient buffer.
///  @params[in] c1_buffer First coefficient buffer.
///  @params[in] c2_buffer Second coefficient buffer.
///  @params[in] c3_buffer Third coefficient buffer.
///  @params[in] x         Non-normalized function argument.
///  @returns The graph of a 1D spline function.
//------------------------------------------------------------------------------
    template<typename T>
    graph::shared_leaf<T> make_1D(graph::shared_leaf<T> min_x,
                                  graph::shared_leaf<T> dx,
                                  const std::vector<T> c0_buffer,
                                  const std::vector<T> c1_buffer,
                                  const std::vector<T> c2_buffer,
                                  const std::vector<T> c3_buffer,
                                  graph::shared_leaf<T> x) {
        auto arg_norm = (x - min_x)/dx;

        auto c0 = graph::piecewise_1D(c0_buffer, arg_norm);
        auto c1 = graph::piecewise_1D(c1_buffer, arg_norm);
        auto c2 = graph::piecewise_1D(c2_buffer, arg_norm);
        auto c3 = graph::piecewise_1D(c3_buffer, arg_norm);
        
        return c0 + c1*arg_norm + c2*arg_norm*arg_norm + c3*arg_norm*arg_norm*arg_norm;
    }

//------------------------------------------------------------------------------
///  @brief Build a 2D spline.
///
///  @params[in] min_x      Minimum argument.
///  @params[in] dx         Grid spacing.
///  @params[in] min_y      Minimum argument.
///  @params[in] dy         Grid spacing.
///  @params[in] c00_buffer Zeroth coefficient buffer.
///  @params[in] c01_buffer First coefficient buffer.
///  @params[in] c02_buffer Second coefficient buffer.
///  @params[in] c03_buffer Third coefficient buffer.
///  @params[in] c10_buffer Zeroth coefficient buffer.
///  @params[in] c11_buffer First coefficient buffer.
///  @params[in] c12_buffer Second coefficient buffer.
///  @params[in] c13_buffer Third coefficient buffer.
///  @params[in] c20_buffer Zeroth coefficient buffer.
///  @params[in] c21_buffer First coefficient buffer.
///  @params[in] c22_buffer Second coefficient buffer.
///  @params[in] c23_buffer Third coefficient buffer.
///  @params[in] c30_buffer Zeroth coefficient buffer.
///  @params[in] c31_buffer First coefficient buffer.
///  @params[in] c32_buffer Second coefficient buffer.
///  @params[in] c33_buffer Third coefficient buffer.
///  @params[in] x          Non-normalized function argument.
///  @params[in] y          Non-normalized function argument.
///  @returns The graph of a 1D spline function.
//------------------------------------------------------------------------------
    template<typename T>
    graph::shared_leaf<T> make_2D(graph::shared_leaf<T> min_x,
                                  graph::shared_leaf<T> dx,
                                  graph::shared_leaf<T> min_y,
                                  graph::shared_leaf<T> dy,
                                  const std::vector<T> c00_buffer,
                                  const std::vector<T> c01_buffer,
                                  const std::vector<T> c02_buffer,
                                  const std::vector<T> c03_buffer,
                                  const std::vector<T> c10_buffer,
                                  const std::vector<T> c11_buffer,
                                  const std::vector<T> c12_buffer,
                                  const std::vector<T> c13_buffer,
                                  const std::vector<T> c20_buffer,
                                  const std::vector<T> c21_buffer,
                                  const std::vector<T> c22_buffer,
                                  const std::vector<T> c23_buffer,
                                  const std::vector<T> c30_buffer,
                                  const std::vector<T> c31_buffer,
                                  const std::vector<T> c32_buffer,
                                  const std::vector<T> c33_buffer,
                                  graph::shared_leaf<T> x,
                                  graph::shared_leaf<T> y,
                                  const size_t num_columns) {
        auto x_norm = (x - min_x)/dx;
        auto y_norm = (y - min_y)/dy;

        auto c00 = graph::piecewise_2D(c00_buffer, num_columns, x_norm, y_norm);
        auto c01 = graph::piecewise_2D(c01_buffer, num_columns, x_norm, y_norm);
        auto c02 = graph::piecewise_2D(c02_buffer, num_columns, x_norm, y_norm);
        auto c03 = graph::piecewise_2D(c03_buffer, num_columns, x_norm, y_norm);
        auto c10 = graph::piecewise_2D(c10_buffer, num_columns, x_norm, y_norm);
        auto c11 = graph::piecewise_2D(c11_buffer, num_columns, x_norm, y_norm);
        auto c12 = graph::piecewise_2D(c12_buffer, num_columns, x_norm, y_norm);
        auto c13 = graph::piecewise_2D(c13_buffer, num_columns, x_norm, y_norm);
        auto c20 = graph::piecewise_2D(c20_buffer, num_columns, x_norm, y_norm);
        auto c21 = graph::piecewise_2D(c21_buffer, num_columns, x_norm, y_norm);
        auto c22 = graph::piecewise_2D(c22_buffer, num_columns, x_norm, y_norm);
        auto c23 = graph::piecewise_2D(c23_buffer, num_columns, x_norm, y_norm);
        auto c30 = graph::piecewise_2D(c30_buffer, num_columns, x_norm, y_norm);
        auto c31 = graph::piecewise_2D(c31_buffer, num_columns, x_norm, y_norm);
        auto c32 = graph::piecewise_2D(c32_buffer, num_columns, x_norm, y_norm);
        auto c33 = graph::piecewise_2D(c33_buffer, num_columns, x_norm, y_norm);
        
        return c00 +
               c01*y_norm +
               c02*y_norm*y_norm +
               c03*y_norm*y_norm*y_norm +
               c10*x_norm +
               c11*x_norm*y_norm +
               c12*x_norm*y_norm*y_norm +
               c13*x_norm*y_norm*y_norm*y_norm +
               c20*x_norm*x_norm +
               c21*x_norm*x_norm*y_norm +
               c22*x_norm*x_norm*y_norm*y_norm +
               c23*x_norm*x_norm*y_norm*y_norm*y_norm +
               c30*x_norm*x_norm*x_norm +
               c31*x_norm*x_norm*x_norm*y_norm +
               c32*x_norm*x_norm*x_norm*y_norm*y_norm +
               c33*x_norm*x_norm*x_norm*y_norm*y_norm*y_norm;
    }
}

#endif /* cublic_splines_h */
