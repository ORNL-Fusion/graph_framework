//------------------------------------------------------------------------------
///  @file backend.hpp
///  @brief Class signature to implement compute backends.
///
///  Defined the function interfaces to access compute resources.
//------------------------------------------------------------------------------

#ifndef backend_h
#define backend_h

#include <algorithm>
#include <vector>
#include <cmath>

#include "special_functions.hpp"
#include "register.hpp"

///  Name space for backend buffers.
namespace backend {
//******************************************************************************
//  Data buffer.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a generic buffer.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
    template<jit::float_scalar T>
    class buffer : public std::vector<T> {
    public:
        using std::vector<T>::size;
        using std::vector<T>::data;
        using std::vector<T>::assign;

//------------------------------------------------------------------------------
///  @brief Construct an empty buffer backend.
//------------------------------------------------------------------------------
        buffer() :
        std::vector<T> () {}

//------------------------------------------------------------------------------
///  @brief Construct a buffer backend with a size.
///
///  @param[in] s Size of he data buffer.
//------------------------------------------------------------------------------
        buffer(const size_t s) :
        std::vector<T> (s) {}

//------------------------------------------------------------------------------
///  @brief Construct a buffer backend with a size.
///
///  @param[in] s Size of he data buffer.
///  @param[in] d Scalar data to initialize.
//------------------------------------------------------------------------------
        buffer(const size_t s, const T d) :
        std::vector<T> (s, d) {}

//------------------------------------------------------------------------------
///  @brief Construct a buffer backend from a vector.
///
///  @param[in] d Array buffer.
//------------------------------------------------------------------------------
        buffer(const std::vector<T> &d) :
        std::vector<T> (d) {}

//------------------------------------------------------------------------------
///  @brief Construct a buffer backend from a buffer backend.
///
///  @param[in] d Backend buffer.
//------------------------------------------------------------------------------
        buffer(const buffer &d) :
        std::vector<T> (d) {}

//------------------------------------------------------------------------------
///  @brief Construct a buffer backend linearly.
///
///  @param[in] min Minimum value..
///  @param[in] dx  Step size.
///  @param[in] num Number of mesh points.
//------------------------------------------------------------------------------
        buffer(const T min, const T dx, const size_t num) : std::vector<T> (num) {
            for (size_t i = 0; i < num; i++) {
                (*this)[i] = dx*i + min;
            }
        }

//------------------------------------------------------------------------------
///  @brief Assign a constant value.
///
///  @param[in] d Scalar data to set.
//------------------------------------------------------------------------------
        void set(const T d) {
            assign(size(), d);
        }

//------------------------------------------------------------------------------
///  @brief Assign a vector value.
///
///  @param[in] d Vector data to set.
//------------------------------------------------------------------------------
        void set(const std::vector<T> &d) {
            assign(d.cbegin(), d.cend());
        }

//------------------------------------------------------------------------------
///  @brief Is every element the same.
///
///  @returns Returns true if every element is the same.
//------------------------------------------------------------------------------
        bool is_same() const {
            const T same = (*this)[0];
            for (size_t i = 1, ie = size(); i < ie; i++) {
                if ((*this)[i] != same) {
                    return false;
                }
            }

            return true;
        }

//------------------------------------------------------------------------------
///  @brief Is every element zero.
///
///  @returns Returns true if every element is zero.
//------------------------------------------------------------------------------
        bool is_zero() const {
            for (const T &d : *this) {
                if (d != static_cast<T> (0.0)) {
                    return false;
                }
            }

            return true;
        }
        
//------------------------------------------------------------------------------
///  @brief Is any element zero.
///
///  @returns Returns true if any element is zero.
//------------------------------------------------------------------------------
        bool has_zero() const {
            for (const T &d : *this) {
                if (d == static_cast<T> (0.0)) {
                    return true;
                }
            }

            return false;
        }
        
//------------------------------------------------------------------------------
///  @brief Is every element negative.
///
///  @returns Returns true if every element is negative.
//------------------------------------------------------------------------------
        bool is_negative() const {
            for (const T &d : *this) {
                if (std::real(d) > std::real(static_cast<T> (0.0))) {
                    return false;
                }
            }

            return true;
        }

//------------------------------------------------------------------------------
///  @brief Is every element even.
///
///  @returns Returns true if every element is negative.
//------------------------------------------------------------------------------
        bool is_even() const {
            for (const T &d : *this) {
                if (std::fmod(std::real(d), std::real(static_cast<T> (2.0)))) {
                    return false;
                }
            }

            return true;
        }

//------------------------------------------------------------------------------
///  @brief Is every element negative one.
///
///  @returns Returns true if every element is negative one.
//------------------------------------------------------------------------------
        bool is_none() const {
            for (const T &d : *this) {
                if (d != static_cast<T> (-1.0)) {
                    return false;
                }
            }

            return true;
        }

//------------------------------------------------------------------------------
///  @brief Applies an operation over all elements in the buffer.
///
///  @param op The operation to apply.
//------------------------------------------------------------------------------
#define apply_op(op) \
for (T &d : *this) { \
    d = op(d);       \
}

//------------------------------------------------------------------------------
///  @brief Take sqrt.
//------------------------------------------------------------------------------
        void sqrt() {
            apply_op(std::sqrt)
        }

//------------------------------------------------------------------------------
///  @brief Take exp.
//------------------------------------------------------------------------------
        void exp() {
            apply_op(std::exp)
        }

//------------------------------------------------------------------------------
///  @brief Take log.
//------------------------------------------------------------------------------
        void log() {
            apply_op(std::log)
        }

//------------------------------------------------------------------------------
///  @brief Take sin.
//------------------------------------------------------------------------------
        void sin() {
            apply_op(std::sin)
        }

//------------------------------------------------------------------------------
///  @brief Take cos.
//------------------------------------------------------------------------------
        void cos() {
            apply_op(std::cos)
        }

//------------------------------------------------------------------------------
///  @brief Take erfi.
//------------------------------------------------------------------------------
        void erfi() requires(jit::complex_scalar<T>) {
            apply_op(special::erfi)
        }

//------------------------------------------------------------------------------
///  @brief Check for normal values.
///
///  @returns False if any NaN or Inf is found.
//------------------------------------------------------------------------------
        bool is_normal() const {
            for (const T &x : *this) {
                if constexpr (jit::complex_scalar<T>) {
                    if (std::isnan(std::real(x)) || std::isinf(std::real(x)) ||
                        std::isnan(std::imag(x)) || std::isinf(std::imag(x))) {
                        return false;
                    }
                } else {
                    if (std::isnan(x) || std::isinf(x)) {
                        return false;
                    }
                }
            }
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Index row.
///
///  @param[in] index       The row index.
///  @param[in] num_columns The number of coils.
///  @returns A buffer containing the row.
//------------------------------------------------------------------------------
        buffer<T> index_row(const size_t index, const size_t num_columns) {
            buffer<T> b(num_columns);
            const size_t num_rows = size()/num_columns;
            for (size_t j = 0; j < num_columns; j++) {
                b[j] = (*this)[index*num_rows + j];
            }
            return b;
        }

//------------------------------------------------------------------------------
///  @brief Index column.
///
///  @param[in] index       The row index.
///  @param[in] num_columns The number of coils.
///  @returns A buffer containing the row.
//------------------------------------------------------------------------------
        buffer<T> index_column(const size_t index, const size_t num_columns) {
            const size_t num_rows = size()/num_columns;
            buffer<T> b(num_rows);
            for (size_t i = 0; i < num_rows; i++) {
                b[i] = (*this)[i*num_rows + index];
            }
            return b;
        }

//------------------------------------------------------------------------------
///  @brief Applies an operatator along a row.
///
///  @param opp   The operation to apply.
///  @param oppeq The assignment operator to apply.
//------------------------------------------------------------------------------
#define row_op(opp, oppeq)                                                  \
if (size() > x.size()) {                                                    \
    assert(size()%x.size() == 0 &&                                          \
           "Vector operand size is not a multiple of matrix operand size"); \
                                                                            \
    const size_t num_columns = size()/x.size();                             \
    const size_t num_rows = x.size();                                       \
    for (size_t i = 0; i < num_rows; i++) {                                 \
        for (size_t j = 0; j < num_columns; j++) {                          \
            (*this)[i*num_columns + j] oppeq x[i];                          \
        }                                                                   \
    }                                                                       \
} else {                                                                    \
    assert(x.size()%size() == 0 &&                                          \
           "Vector operand size is not a multiple of matrix operand size"); \
                                                                            \
    std::vector<T> m(x.size());                                             \
    const size_t num_columns = x.size()/size();                             \
    const size_t num_rows = size();                                         \
    for (size_t i = 0; i < num_rows; i++) {                                 \
        for (size_t j = 0; j < num_columns; j++) {                          \
            m[i*num_columns + j] = (*this)[i] opp x[i*num_columns + j];     \
        }                                                                   \
    }                                                                       \
    *this = m;                                                              \
}

//------------------------------------------------------------------------------
///  @brief Add row operation.
///
///  Adds m_ij + v_i or v_i + m_ij. This will resize the buffer if it needs to 
///  be.
///
///  @param[in] x The right operand.
//------------------------------------------------------------------------------
        void add_row(const buffer<T> &x) {
            row_op(+, +=)
        }

//------------------------------------------------------------------------------
///  @brief Applies an operatator along a column.
///
///  @param opp   The operation to apply.
///  @param oppeq The assignment operator to apply.
//------------------------------------------------------------------------------
#define col_op(opp, oppeq)                                                  \
if (size() > x.size()) {                                                    \
    assert(size()%x.size() == 0 &&                                          \
           "Vector operand size is not a multiple of matrix operand size"); \
                                                                            \
    const size_t num_columns = size()/x.size();                             \
    const size_t num_rows = x.size();                                       \
    for (size_t i = 0; i < num_rows; i++) {                                 \
        for (size_t j = 0; j < num_columns; j++) {                          \
            (*this)[i*num_columns + j] oppeq x[j];                          \
        }                                                                   \
    }                                                                       \
} else {                                                                    \
    assert(x.size()%size() == 0 &&                                          \
           "Vector operand size is not a multiple of matrix operand size"); \
                                                                            \
    std::vector<T> m(x.size());                                             \
    const size_t num_columns = x.size()/size();                             \
    const size_t num_rows = size();                                         \
    for (size_t i = 0; i < num_rows; i++) {                                 \
        for (size_t j = 0; j < num_columns; j++) {                          \
            m[i*num_columns + j] = (*this)[j] opp x[i*num_columns + j];     \
        }                                                                   \
    }                                                                       \
    *this = m;                                                              \
}

//------------------------------------------------------------------------------
///  @brief Add col operation.
///
///  Adds m_ij + v_j or v_j + m_ij. This will resize the buffer if it needs to
///  be.
///
///  @param[in] x The other operand.
//------------------------------------------------------------------------------
        void add_col(const buffer<T> &x) {
            col_op(+, +=)
        }

//------------------------------------------------------------------------------
///  @brief Subtract row operation.
///
///  Subtracts m_ij - v_i or v_i - m_ij. This will resize the buffer if it
///  needs to be.
///
///  @param[in] x The right operand.
//------------------------------------------------------------------------------
        void subtract_row(const buffer<T> &x) {
            row_op(-, -=)
        }

//------------------------------------------------------------------------------
///  @brief Subtract col operation.
///
///  Subtracts m_ij - v_j or v_j - m_ij. This will resize the buffer if it
///  needs to be.
///
///  @param[in] x The other operand.
//------------------------------------------------------------------------------
        void subtract_col(const buffer<T> &x) {
            col_op(-, -=)
        }

//------------------------------------------------------------------------------
///  @brief Multiply row operation.
///
///  Multiplies m_ij * v_i or v_i * m_ij. This will resize the buffer if it
///  needs to be.
///
///  @param[in] x The right operand.
//------------------------------------------------------------------------------
        void multiply_row(const buffer<T> &x) {
            row_op(*, *=)
        }

//------------------------------------------------------------------------------
///  @brief Multiply col operation.
///
///  Multiplies m_ij * v_j or v_j * m_ij. This will resize the buffer if it
///  needs to be.
///
///  @param[in] x The other operand.
//------------------------------------------------------------------------------
        void multiply_col(const buffer<T> &x) {
            col_op(*, *=)
        }

//------------------------------------------------------------------------------
///  @brief Divide row operation.
///
///  Divides m_ij / v_i or v_i / m_ij. This will resize the buffer if it needs 
///  to be.
///
///  @param[in] x The right operand.
//------------------------------------------------------------------------------
        void divide_row(const buffer<T> &x) {
            row_op(/, /=)
        }

//------------------------------------------------------------------------------
///  @brief Divide col operation.
///
///  Divides m_ij / v_j or v_j / m_ij. This will resize the buffer if it needs
///  to be.
///
///  @param[in] x The other operand.
//------------------------------------------------------------------------------
        void divide_col(const buffer<T> &x) {
            col_op(/, /=)
        }

//------------------------------------------------------------------------------
///  @brief Atan row operation.
///
///  Computes atan(m_ij, v_i) or atan(v_i, m_ij). This will resize the buffer if
///  it needs to be.
///
///  @param[in] x The right operand.
//------------------------------------------------------------------------------
        void atan_row(const buffer<T> &x) {
            if (size() > x.size()) {
                assert(size()%x.size() == 0 &&
                       "Vector operand size is not a multiple of matrix operand size");

                const size_t num_columns = size()/x.size();
                const size_t num_rows = x.size();
                for (size_t i = 0; i < num_rows; i++) {
                    for (size_t j = 0; j < num_columns; j++) {
                        if constexpr (jit::complex_scalar<T>) {
                            (*this)[i*num_columns + j] = std::atan(x[i]/(*this)[i*num_columns + j]);
                        } else {
                            (*this)[i*num_columns + j] = std::atan2(x[i], (*this)[i*num_columns + j]);
                        }
                    }
                }
            } else {
                assert(x.size()%size() == 0 &&
                       "Vector operand size is not a multiple of matrix operand size");

                std::vector<T> m(x.size());
                const size_t num_columns = x.size()/size();
                const size_t num_rows = size();
                for (size_t i = 0; i < num_rows; i++) {
                    for (size_t j = 0; j < num_columns; j++) {
                        if constexpr (jit::complex_scalar<T>) {
                            m[i*num_columns + j] = std::atan(x[i*num_columns + j]/(*this)[i]);
                        } else {
                            m[i*num_columns + j] = std::atan2(x[i*num_columns + j], (*this)[i]);
                        }
                    }
                }
                *this = m;
            }
        }

//------------------------------------------------------------------------------
///  @brief Atan col operation.
///
///  Computes atan(m_ij, v_j) or atan(v_j, m_ij). This will resize the buffer if
///  it needs to be.
///
///  @param[in] x The other operand.
//------------------------------------------------------------------------------
        void atan_col(const buffer<T> &x) {
            if (size() > x.size()) {
                assert(size()%x.size() == 0 &&
                       "Vector operand size is not a multiple of matrix operand size");

                const size_t num_columns = size()/x.size();
                const size_t num_rows = x.size();
                for (size_t i = 0; i < num_columns; i++) {
                    for (size_t j = 0; j < num_rows; j++) {
                        if constexpr (jit::complex_scalar<T>) {
                            (*this)[i*num_columns + j] = std::atan(x[j]/(*this)[i*num_columns + j]);
                        } else {
                            (*this)[i*num_columns + j] = std::atan2(x[j], (*this)[i*num_columns + j]);
                        }
                    }
                }
            } else {
                assert(x.size()%size() == 0 &&
                       "Vector operand size is not a multiple of matrix operand size");

                std::vector<T> m(x.size());
                const size_t num_columns = x.size()/size();
                const size_t num_rows = size();
                for (size_t i = 0; i < num_rows; i++) {
                    for (size_t j = 0; j < num_columns; j++) {
                        if constexpr (jit::complex_scalar<T>) {
                            m[i*num_columns + j] = std::atan(x[i*num_columns + j]/(*this)[j]);
                        } else {
                            m[i*num_columns + j] = std::atan2(x[i*num_columns + j], (*this)[j]);
                        }
                    }
                }
                *this = m;
            }
        }

//------------------------------------------------------------------------------
///  @brief Pow row operation.
///
///  Computes pow(m_ij, v_i) or pow(v_i, m_ij). This will resize the buffer if
///  it needs to be.
///
///  @param[in] x The right operand.
//------------------------------------------------------------------------------
        void pow_row(const buffer<T> &x) {
            if (size() > x.size()) {
                assert(size()%x.size() == 0 &&
                       "Vector operand size is not a multiple of matrix operand size");

                const size_t num_columns = size()/x.size();
                const size_t num_rows = x.size();
                for (size_t i = 0; i < num_rows; i++) {
                    for (size_t j = 0; j < num_columns; j++) {
                        (*this)[i*num_columns + j] = std::pow((*this)[i*num_columns + j], x[i]);
                    }
                }
            } else {
                assert(x.size()%size() == 0 &&
                       "Vector operand size is not a multiple of matrix operand size");

                std::vector<T> m(x.size());
                const size_t num_columns = x.size()/size();
                const size_t num_rows = size();
                for (size_t i = 0; i < num_columns; i++) {
                    for (size_t j = 0; j < num_rows; j++) {
                        m[i*num_columns + j] = std::pow((*this)[i], x[i*num_columns + j]);
                    }
                }
                *this = m;
            }
        }

//------------------------------------------------------------------------------
///  @brief Pow col operation.
///
///  Computes pow(m_ij, v_j) or pow(v_j, m_ij). This will resize the buffer if
///  it needs to be.
///
///  @param[in] x The other operand.
//------------------------------------------------------------------------------
        void pow_col(const buffer<T> &x) {
            if (size() > x.size()) {
                assert(size()%x.size() == 0 &&
                       "Vector operand size is not a multiple of matrix operand size");

                const size_t num_columns = size()/x.size();
                const size_t num_rows = x.size();
                for (size_t i = 0; i < num_rows; i++) {
                    for (size_t j = 0; j < num_columns; j++) {
                        (*this)[i*num_columns + j] = std::pow((*this)[i*num_columns + j], x[j]);
                    }
                }
            } else {
                assert(x.size()%size() == 0 &&
                       "Vector operand size is not a multiple of matrix operand size");

                std::vector<T> m(x.size());
                const size_t num_columns = x.size()/size();
                const size_t num_rows = size();
                for (size_t i = 0; i < num_rows; i++) {
                    for (size_t j = 0; j < num_columns; j++) {
                        m[i*num_columns + j] = std::pow((*this)[j], x[i*num_columns + j]);
                    }
                }
                *this = m;
            }
        }

//------------------------------------------------------------------------------
///  @brief Not operation.
///
///  @returns The negation of the buffer.
//------------------------------------------------------------------------------
        buffer<T> operator!() requires(std::floating_point<T>) {
            for (T &d : *this) {
                d = !d;
            }
            return *this;
        }

//------------------------------------------------------------------------------
///  @brief Apply condition.
///
///  @params[in] t True condition.
///  @params[in] f False condition.
//------------------------------------------------------------------------------
        buffer<T> if_(const buffer<T> &t,
                      const buffer<T> &f) {
            if (size() == 1) {
                return (*this)[0] ? t : f;
            } else {
                if (t.size() == 1) {
                    if (f.size() == 1) {
                        for (T &d : *this) {
                            d = d ? t[0] : f[0];
                        }
                        return *this;
                    } else {
                        assert(size() == f.size() && "Incompatable buffersize.");
                        for (size_t i = 0, ie = size(); i < ie; i++) {
                            (*this)[i] = (*this)[i] ? t[0] : f[i];
                        }
                        return *this;
                    }
                } else {
                    assert(size() == t.size() && "Incompatable buffersize.");
                    if (f.size() == 1) {
                        for (size_t i = 0, ie = size(); i < ie; i++) {
                            (*this)[i] = (*this)[i] ? t[i] : f[0];
                        }
                        return *this;
                    } else {
                        assert(size() == f.size() && "Incompatable buffersize.");
                        for (size_t i = 0, ie = size(); i < ie; i++) {
                            (*this)[i] = (*this)[i] ? t[i] : f[i];
                        }
                        return *this;
                    }
                }
            }
        }

///  Type def to retrieve the backend T type.
        typedef T base;
    };

//------------------------------------------------------------------------------
///  @brief Equal operation.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] a Left operand.
///  @param[in] b Right operand.
///  @returns a == b.
//------------------------------------------------------------------------------
    template<jit::float_scalar T>
    inline bool operator==(const buffer<T> &a,
                           const buffer<T> &b) {
        if (a.size() != b.size()) {
            return false;
        }

        for (size_t i = 0, ie = a.size(); i < ie; i++) {
            if (a[i] != b[i]) {
                return false;
            }
        }
        return true;
    }

//------------------------------------------------------------------------------
///  @brief Applies an associative operator.
///
///  @param op The operation to apply.
//------------------------------------------------------------------------------
#define build_assoc_op(op)                        \
if (b.size() == 1) {                              \
    const T right = b[0];                         \
    for (T &l : a) {                              \
        l op right;                               \
    }                                             \
    return a;                                     \
} else if (a.size() == 1) {                       \
    const T left = a[0];                          \
    for (T &r : b) {                              \
        r op left;                                \
    }                                             \
    return b;                                     \
}                                                 \
                                                  \
assert(a.size() == b.size() &&                    \
       "Left and right sizes are incompatible."); \
for (size_t i = 0, ie = a.size(); i < ie; i++) {  \
    a[i] op b[i];                                 \
}                                                 \
return a;

//------------------------------------------------------------------------------
///  @brief Add operation.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] a Left operand.
///  @param[in] b Right operand.
///  @returns a + b.
//------------------------------------------------------------------------------
    template<jit::float_scalar T>
    inline buffer<T> operator+(buffer<T> &a,
                               buffer<T> &b) {
        build_assoc_op(+=)
    }

//------------------------------------------------------------------------------
///  @brief Applies a non-associative operator.
///
///  @param op   The operation to apply.
///  @param opeq The assign operation to apply.
//------------------------------------------------------------------------------
#define build_non_assoc_op(op, opeq)              \
if (b.size() == 1) {                              \
    const T right = b[0];                         \
    for (T &l : a) {                              \
        l opeq right;                             \
    }                                             \
    return a;                                     \
} else if (a.size() == 1) {                       \
    const T left = a[0];                          \
    for (T &r : b) {                              \
        r = left op r;                            \
    }                                             \
    return b;                                     \
}                                                 \
                                                  \
assert(a.size() == b.size() &&                    \
       "Left and right sizes are incompatible."); \
for (size_t i = 0, ie = a.size(); i < ie; i++) {  \
    a[i] opeq b[i];                               \
}                                                 \
return a;

//------------------------------------------------------------------------------
///  @brief Subtract operation.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] a Left operand.
///  @param[in] b Right operand.
///  @returns a - b.
//------------------------------------------------------------------------------
    template<jit::float_scalar T>
    inline buffer<T> operator-(buffer<T> &a,
                               buffer<T> &b) {
        build_non_assoc_op(-, -=)
    }

//------------------------------------------------------------------------------
///  @brief Multiply operation.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] a Left operand.
///  @param[in] b Right operand.
///  @returns a * b.
//------------------------------------------------------------------------------
    template<jit::float_scalar T>
    inline buffer<T> operator*(buffer<T> &a,
                               buffer<T> &b) {
        build_assoc_op(*=)
    }

//------------------------------------------------------------------------------
///  @brief Divide operation.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] a Numerator.
///  @param[in] b Denominator.
///  @returns a / b.
//------------------------------------------------------------------------------
    template<jit::float_scalar T>
    inline buffer<T> operator/(buffer<T> &a,
                               buffer<T> &b) {
        build_non_assoc_op(/, /=)
    }

//------------------------------------------------------------------------------
///  @brief Fused multiply add operation.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] a Left operand.
///  @param[in] b Middle operand.
///  @param[in] c Right operand.
///  @returns a*b + c.
//------------------------------------------------------------------------------
    template<jit::float_scalar T>
    inline buffer<T> fma(buffer<T> &a,
                         buffer<T> &b,
                         buffer<T> &c) {
        constexpr bool use_fma = !jit::complex_scalar<T> &&
#ifdef FP_FAST_FMA
                                 true;
#else
                                 false;
#endif
        
        if (a.size() == 1) {
            const T left = a[0];

            if (b.size() == 1) {
                const T middle = b[0];
                for (size_t i = 0, ie = c.size(); i < ie; i++) {
                    if constexpr (use_fma) {
                        c[i] = std::fma(left, middle, c[i]);
                    } else {
                        c[i] = left*middle + c[i];
                    }
                }
                return c;
            } else if (c.size() == 1) {
                const T right = c[0];
                for (size_t i = 0, ie = b.size(); i < ie; i++) {
                    if constexpr (use_fma) {
                        b[i] = std::fma(left, b[i], right);
                    } else {
                        b[i] = left*b[i] + right;
                    }
                }
                return b;
            }

            assert(b.size() == c.size() &&
                   "Size mismatch between middle and right.");
            for (size_t i = 0, ie = b.size(); i < ie; i++) {
                if constexpr (use_fma) {
                    b[i] = std::fma(left, b[i], c[i]);
                } else {
                    b[i] = left*b[i] + c[i];
                }
            }
            return b;
        } else if (b.size() == 1) {
            const T middle = b[0];
            if (c.size() == 1) {
                const T right = c[0];
                for (size_t i = 0, ie = a.size(); i < ie; i++) {
                    if constexpr (use_fma) {
                        a[i] = std::fma(a[i], middle, right);
                    } else {
                        a[i] = a[i]*middle + right;
                    }
                }
                return a;
            }

            assert(a.size() == c.size() &&
                   "Size mismatch between left and right.");
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                if constexpr (use_fma) {
                    a[i] = std::fma(a[i], middle, c[i]);
                } else {
                    a[i] = a[i]*middle + c[i];
                }
            }
            return a;
        } else if (c.size() == 1) {
            assert(a.size() == b.size() &&
                   "Size mismatch between left and middle.");
            const T right = c[0];
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                if constexpr (use_fma) {
                    a[i] = std::fma(a[i], b[i], right);
                } else {
                    a[i] = a[i]*b[i] + right;
                }
            }
            return a;
        }

        assert(a.size() == b.size() &&
               b.size() == c.size() &&
               a.size() == c.size() &&
               "Left, middle and right sizes are incompatible.");
        for (size_t i = 0, ie = a.size(); i < ie; i++) {
            if constexpr (use_fma) {
                a[i] = std::fma(a[i], b[i], c[i]);
            } else {
                a[i] = a[i]*b[i] + c[i];
            }
        }
        return a;
    }

//------------------------------------------------------------------------------
///  @brief Modulo operation.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] a Numerator.
///  @param[in] b Denominator.
///  @returns a % b.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    inline buffer<T> operator%(buffer<T> &a,
                               buffer<T> &b) {
        if (b.size() == 1) {
            const T right = b[0];
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] = std::fmod(a[i], right);
            }
            return a;
        } else if (a.size() == 1) {
            const T left = a[0];
            for (size_t i = 0, ie = b.size(); i < ie; i++) {
                b[i] = std::fmod(left, b[i]);
            }
            return b;
        }

        assert(a.size() == b.size() &&
               "Left and right sizes are incompatible.");
        for (size_t i = 0, ie = a.size(); i < ie; i++) {
            a[i] = std::fmod(a[i], b[i]);
        }
        return a;
    }

//------------------------------------------------------------------------------
///  @brief Applies a logical operator.
///
///  @param op The operation to apply.
//------------------------------------------------------------------------------
#define logic_op(op)                                 \
if (b.size() == 1) {                                 \
    const T right = b[0];                            \
    for (size_t i = 0, ie = a.size(); i < ie; i++) { \
        a[i] = static_cast<T> (a[i] op right);       \
    }                                                \
    return a;                                        \
} else if (a.size() == 1) {                          \
    const T left = a[0];                             \
    for (size_t i = 0, ie = b.size(); i < ie; i++) { \
        b[i] = static_cast<T> (left op b[i]);        \
    }                                                \
    return b;                                        \
}                                                    \
                                                     \
assert(a.size() == b.size() &&                       \
       "Left and right sizes are incompatible.");    \
for (size_t i = 0, ie = a.size(); i < ie; i++) {     \
    a[i] = static_cast<T> (a[i] op b[i]);            \
}                                                    \
return a;

//------------------------------------------------------------------------------
///  @brief Equal operation.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] a Numerator.
///  @param[in] b Denominator.
///  @returns a == b.
//------------------------------------------------------------------------------
    template<jit::float_scalar T>
    inline buffer<T> operator==(buffer<T> &a,
                                buffer<T> &b) {
        logic_op(==)
    }

//------------------------------------------------------------------------------
///  @brief Not equal operation.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] a Numerator.
///  @param[in] b Denominator.
///  @returns a == b.
//------------------------------------------------------------------------------
    template<jit::float_scalar T>
    inline buffer<T> operator!=(buffer<T> &a,
                                buffer<T> &b) {
        logic_op(!=)
    }

//------------------------------------------------------------------------------
///  @brief Greater than operation.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] a Numerator.
///  @param[in] b Denominator.
///  @returns a > b.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    inline buffer<T> operator>(buffer<T> &a,
                               buffer<T> &b) {
        logic_op(>)
    }

//------------------------------------------------------------------------------
///  @brief Less than operation.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] a Numerator.
///  @param[in] b Denominator.
///  @returns a < b.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    inline buffer<T> operator<(buffer<T> &a,
                               buffer<T> &b) {
        logic_op(<)
    }

//------------------------------------------------------------------------------
///  @brief Greater than equal operation.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] a Numerator.
///  @param[in] b Denominator.
///  @returns a >= b.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    inline buffer<T> operator>=(buffer<T> &a,
                                buffer<T> &b) {
        logic_op(>=)
    }

//------------------------------------------------------------------------------
///  @brief Less than equal operation.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] a Numerator.
///  @param[in] b Denominator.
///  @returns a <= b.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    inline buffer<T> operator<=(buffer<T> &a,
                                buffer<T> &b) {
        logic_op(<=)
    }

//------------------------------------------------------------------------------
///  @brief And operation.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] a Numerator.
///  @param[in] b Denominator.
///  @returns a && b.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    inline buffer<T> operator&&(buffer<T> &a,
                                buffer<T> &b) {
        logic_op(&&)
    }

//------------------------------------------------------------------------------
///  @brief Or operation.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] a Numerator.
///  @param[in] b Denominator.
///  @returns a || b.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    inline buffer<T> operator||(buffer<T> &a,
                                buffer<T> &b) {
        logic_op(||)
    }

//------------------------------------------------------------------------------
///  @brief Take the power.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] base     Base to raise to the power of.
///  @param[in] exponent Power to apply to the base.
///  @returns base^exponent.
//------------------------------------------------------------------------------
    template<jit::float_scalar T>
    inline buffer<T> pow(buffer<T> &base,
                         buffer<T> &exponent) {
        if (exponent.size() == 1) {
            const T right = exponent[0];
            if (std::imag(right) == 0) {
                const int64_t right_int = static_cast<int64_t> (std::real(right));
                if (std::real(right) - right_int) {
                    if (right == static_cast<T> (0.5)) {
                        base.sqrt();
                        return base;
                    }

                    for (size_t i = 0, ie = base.size(); i < ie; i++) {
                        base[i] = std::pow(base[i], right);
                    }
                    return base;
                }

                if (right_int > 0) {
                    for (size_t i = 0, ie = base.size(); i < ie; i++) {
                        const T left = base[i];
                        for (size_t j = 0, je = right_int - 1; j < je; j++) {
                            base[i] *= left;
                        }
                    }
                    return base;
                } else if (right_int == 0) {
                    for (size_t i = 0, ie = base.size(); i < ie; i++) {
                        base[i] = 1.0;
                    }
                    return base;
                } else {
                    for (size_t i = 0, ie = base.size(); i < ie; i++) {
                        const T left = static_cast<T> (1.0)/base[i];
                        base[i] = left;
                        for (size_t j = 0, je = std::abs(right_int) - 1; j < je; j++) {
                            base[i] *= left;
                        }
                    }
                    return base;
                }
            } else {
                for (size_t i = 0, ie = base.size(); i < ie; i++) {
                    base[i] = std::pow(base[i], right);
                }
                return base;
            }
        } else if (base.size() == 1) {
            const T left = base[0];
            for (size_t i = 0, ie = exponent.size(); i < ie; i++) {
                exponent[i] = std::pow(left, exponent[i]);
            }
            return exponent;
        }

        assert(base.size() == exponent.size() &&
               "Left and right sizes are incompatible.");
        for (size_t i = 0, ie = base.size(); i < ie; i++) {
            base[i] = std::pow(base[i], exponent[i]);
        }
        return base;
    }

//------------------------------------------------------------------------------
///  @brief Take the inverse tangent.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] x X argument.
///  @param[in] y Y argument.
///  @returns atan2(y, x)
//------------------------------------------------------------------------------
    template<jit::float_scalar T>
    inline buffer<T> atan(buffer<T> &x,
                          buffer<T> &y) {
        if (y.size() == 1) {
            const T right = y[0];
            for (size_t i = 0, ie = x.size(); i < ie; i++) {
                if constexpr (jit::complex_scalar<T>) {
                    x[i] = std::atan(right/x[i]);
                } else {
                    x[i] = std::atan2(right, x[i]);
                }
            }
            return x;
        } else if (x.size() == 1) {
            const T left = x[0];
            for (size_t i = 0, ie = y.size(); i < ie; i++) {
                if constexpr (jit::complex_scalar<T>) {
                    y[i] = std::atan(y[i]/left);
                } else {
                    y[i] = std::atan2(y[i], left);
                }
            }
            return y;
        }

        assert(x.size() == y.size() &&
               "Left and right sizes are incompatible.");
        for (size_t i = 0, ie = x.size(); i < ie; i++) {
            if constexpr (jit::complex_scalar<T>) {
                x[i] = std::atan(y[i]/x[i]);
            } else {
                x[i] = std::atan2(y[i], x[i]);
            }
        }
        return x;
    }
}

#endif /* backend_h */
