//------------------------------------------------------------------------------
///  @file vector.hpp
///  @brief Defines vectors of graphs.
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
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
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
///  @param[in] x X vector component.
///  @param[in] y Y vector component.
///  @param[in] z Z Vector component.
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
///  @param[in] v2 Second vector.
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
///  @param[in] v2 Second vector.
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
            return std::make_shared<vector_quantity<T, SAFE_MATH>> (x, y, z)/length();
        }
    };

///  Convenience type for shared vector quantities.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_vector = std::shared_ptr<vector_quantity<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Build a shared vector quantity.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x X vector component.
///  @param[in] y Y vector component.
///  @param[in] z Z Vector component.
///  @returns A vector.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_vector<T, SAFE_MATH> vector(shared_leaf<T, SAFE_MATH> x,
                                       shared_leaf<T, SAFE_MATH> y,
                                       shared_leaf<T, SAFE_MATH> z) {
        return std::make_shared<vector_quantity<T, SAFE_MATH>> (x, y, z);
    }

//------------------------------------------------------------------------------
///  @brief Build a shared vector quantity.
///
///  @tparam T         Base type of the calculation.
///  @tparam L         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x X vector component.
///  @param[in] y Y vector component.
///  @param[in] z Z Vector component.
///  @returns A vector.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar L, bool SAFE_MATH=false>
    shared_vector<T, SAFE_MATH> vector(const L x,
                                       shared_leaf<T, SAFE_MATH> y,
                                       shared_leaf<T, SAFE_MATH> z) {
        return std::make_shared<vector_quantity<T, SAFE_MATH>> (constant<T, SAFE_MATH> (static_cast<T> (x)), y, z);
    }

//------------------------------------------------------------------------------
///  @brief Build a shared vector quantity.
///
///  @tparam T         Base type of the calculation.
///  @tparam M         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x X vector component.
///  @param[in] y Y vector component.
///  @param[in] z Z Vector component.
///  @returns A vector.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar M, bool SAFE_MATH=false>
    shared_vector<T, SAFE_MATH> vector(shared_leaf<T, SAFE_MATH> x,
                                       const M y,
                                       shared_leaf<T, SAFE_MATH> z) {
        return std::make_shared<vector_quantity<T, SAFE_MATH>> (x, constant<T, SAFE_MATH> (static_cast<T> (y)), z);
    }

//------------------------------------------------------------------------------
///  @brief Build a shared vector quantity.
///
///  @tparam T         Base type of the calculation.
///  @tparam R         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x X vector component.
///  @param[in] y Y vector component.
///  @param[in] z Z Vector component.
///  @returns A vector.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar R, bool SAFE_MATH=false>
    shared_vector<T, SAFE_MATH> vector(shared_leaf<T, SAFE_MATH> x,
                                       shared_leaf<T, SAFE_MATH> y,
                                       const R z) {
        return std::make_shared<vector_quantity<T, SAFE_MATH>> (x, y, constant<T, SAFE_MATH> (static_cast<T> (z)));
    }

//------------------------------------------------------------------------------
///  @brief Build a shared vector quantity.
///
///  @tparam T         Base type of the calculation.
///  @tparam L         Float type for the constant.
///  @tparam M         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x X vector component.
///  @param[in] y Y vector component.
///  @param[in] z Z Vector component.
///  @returns A vector.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar L, jit::float_scalar M, bool SAFE_MATH=false>
    shared_vector<T, SAFE_MATH> vector(const L x,
                                       const M y,
                                       shared_leaf<T, SAFE_MATH> z) {
        return std::make_shared<vector_quantity<T, SAFE_MATH>> (constant<T, SAFE_MATH> (static_cast<T> (x)),
                                                                constant<T, SAFE_MATH> (static_cast<T> (y)), z);
    }

//------------------------------------------------------------------------------
///  @brief Build a shared vector quantity.
///
///  @tparam T         Base type of the calculation.
///  @tparam M         Float type for the constant.
///  @tparam R         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x X vector component.
///  @param[in] y Y vector component.
///  @param[in] z Z Vector component.
///  @returns A vector.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar M, jit::float_scalar R, bool SAFE_MATH=false>
    shared_vector<T, SAFE_MATH> vector(shared_leaf<T, SAFE_MATH> x,
                                       const M y,
                                       const R z) {
        return std::make_shared<vector_quantity<T, SAFE_MATH>> (x, constant<T, SAFE_MATH> (static_cast<T> (y)),
                                                                constant<T, SAFE_MATH> (static_cast<T> (z)));
    }

//------------------------------------------------------------------------------
///  @brief Build a shared vector quantity.
///
///  @tparam T         Base type of the calculation.
///  @tparam L         Float type for the constant.
///  @tparam R         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x X vector component.
///  @param[in] y Y vector component.
///  @param[in] z Z Vector component.
///  @returns A vector.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar L, jit::float_scalar R, bool SAFE_MATH=false>
    shared_vector<T, SAFE_MATH> vector(const L x,
                                       shared_leaf<T, SAFE_MATH> y,
                                       const R z) {
        return std::make_shared<vector_quantity<T, SAFE_MATH>> (constant<T, SAFE_MATH> (static_cast<T> (x)), y,
                                                                constant<T, SAFE_MATH> (static_cast<T> (z)));
    }

//------------------------------------------------------------------------------
///  @brief Build a shared vector quantity.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x X vector component.
///  @param[in] y Y vector component.
///  @param[in] z Z Vector component.
///  @returns A vector.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_vector<T, SAFE_MATH> vector(const T x,
                                       const T y,
                                       const T z) {
        return std::make_shared<vector_quantity<T, SAFE_MATH>> (constant<T, SAFE_MATH> (static_cast<T> (x)),
                                                                constant<T, SAFE_MATH> (static_cast<T> (y)),
                                                                constant<T, SAFE_MATH> (static_cast<T> (z)));
    }

//------------------------------------------------------------------------------
///  @brief Addition operator.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left vector.
///  @param[in] r Right vector.
///  @returns The vector vector addition.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_vector<T, SAFE_MATH> operator+(shared_vector<T, SAFE_MATH> l,
                                          shared_vector<T, SAFE_MATH> r) {
        return vector(l->get_x() + r->get_x(),
                      l->get_y() + r->get_y(),
                      l->get_z() + r->get_z());
    }

//------------------------------------------------------------------------------
///  @brief Subtraction operator.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left vector.
///  @param[in] r Right vector.
///  @returns The vector vector addition.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_vector<T, SAFE_MATH> operator-(shared_vector<T, SAFE_MATH> l,
                                          shared_vector<T, SAFE_MATH> r) {
        return vector(l->get_x() - r->get_x(),
                      l->get_y() - r->get_y(),
                      l->get_z() - r->get_z());
    }

//------------------------------------------------------------------------------
///  @brief Multiplication operator.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] s Scalar term.
///  @param[in] v Vector term.
///  @returns The scalar vector multiply.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_vector<T, SAFE_MATH> operator*(shared_leaf<T, SAFE_MATH> s,
                                          shared_vector<T, SAFE_MATH> v) {
        return vector(s*v->get_x(),
                      s*v->get_y(),
                      s*v->get_z());
    }

//------------------------------------------------------------------------------
///  @brief Division operator.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] v Vector numerator.
///  @param[in] s Scalar denominator.
///  @returns The vector scalar division.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_vector<T, SAFE_MATH> operator/(shared_vector<T, SAFE_MATH> v,
                                          shared_leaf<T, SAFE_MATH> s) {
        return vector(v->get_x()/s,
                      v->get_y()/s,
                      v->get_z()/s);
    }

//******************************************************************************
//  Matrix interface.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class to represent matrix quantities.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class matrix_quantity : public std::enable_shared_from_this<matrix_quantity<T, SAFE_MATH>> {
    protected:
///  First row of the matrix.
        shared_vector<T, SAFE_MATH> r1;
///  Second row of the matrix.
        shared_vector<T, SAFE_MATH> r2;
///  Third row of the matrix.
        shared_vector<T, SAFE_MATH> r3;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a new matrix_quantity.
///
///  @param[in] r1 Row 1 matrix component.
///  @param[in] r2 Row 2 matrix component.
///  @param[in] r3 Row 3 matrix component.
//------------------------------------------------------------------------------
        matrix_quantity(shared_vector<T, SAFE_MATH> r1,
                        shared_vector<T, SAFE_MATH> r2,
                        shared_vector<T, SAFE_MATH> r3) :
        r1(r1), r2(r2), r3(r3) {}

//------------------------------------------------------------------------------
///  @brief Multiply matrix by vector.
///
///  @param[in] v Vector vector.
///  @returns v1.v2
//------------------------------------------------------------------------------
        shared_vector<T, SAFE_MATH>
        dot(shared_vector<T, SAFE_MATH> v) {
            return vector<T, SAFE_MATH> (r1->dot(v),
                                         r2->dot(v),
                                         r3->dot(v));
        }
    };

///  Convenience type for shared vector quantities.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_matrix = std::shared_ptr<matrix_quantity<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Build a shared vector quantity.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] r1 Row 1 matrix component.
///  @param[in] r2 Row 2 matrix component.
///  @param[in] r3 Row 3 matrix component.
///  @returns A matrix.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_matrix<T, SAFE_MATH> matrix(shared_vector<T, SAFE_MATH> r1,
                                       shared_vector<T, SAFE_MATH> r2,
                                       shared_vector<T, SAFE_MATH> r3) {
        return std::make_shared<matrix_quantity<T, SAFE_MATH>> (r1, r2, r3);
    }
}

#endif /* vector_h */
