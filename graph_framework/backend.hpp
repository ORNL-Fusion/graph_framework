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
    class buffer {
    private:
///  The data buffer to hold the data.
        std::vector<T> memory;

    public:
//------------------------------------------------------------------------------
///  @brief Construct an empty buffer backend.
//------------------------------------------------------------------------------
        buffer() :
        memory() {}

//------------------------------------------------------------------------------
///  @brief Construct a buffer backend with a size.
///
///  @param[in] s Size of he data buffer.
//------------------------------------------------------------------------------
        buffer(const size_t s) :
        memory(s) {}

//------------------------------------------------------------------------------
///  @brief Construct a buffer backend with a size.
///
///  @param[in] s Size of he data buffer.
///  @param[in] d Scalar data to initialize.
//------------------------------------------------------------------------------
        buffer(const size_t s, const T d) :
        memory(s, d) {}

//------------------------------------------------------------------------------
///  @brief Construct a buffer backend from a vector.
///
///  @param[in] d Array buffer.
//------------------------------------------------------------------------------
        buffer(const std::vector<T> &d) :
        memory(d) {}

//------------------------------------------------------------------------------
///  @brief Construct a buffer backend from a buffer backend.
///
///  @param[in] d Backend buffer.
//------------------------------------------------------------------------------
        buffer(const buffer &d) :
        memory(d.memory) {}

//------------------------------------------------------------------------------
///  @brief Index operator.
//------------------------------------------------------------------------------
        T &operator[] (const size_t index) {
            return memory[index];
        }

//------------------------------------------------------------------------------
///  @brief Const index operator.
//------------------------------------------------------------------------------
        const T &operator[] (const size_t index) const {
            return memory[index];
        }

//------------------------------------------------------------------------------
///  @brief Get value at.
//------------------------------------------------------------------------------
        const T at(const size_t index) const {
            return memory.at(index);
        }

//------------------------------------------------------------------------------
///  @brief Assign a constant value.
///
///  @param[in] d Scalar data to set.
//------------------------------------------------------------------------------
        void set(const T d) {
            memory.assign(memory.size(), d);
        }

//------------------------------------------------------------------------------
///  @brief Assign a vector value.
///
///  @param[in] d Vector data to set.
//------------------------------------------------------------------------------
        void set(const std::vector<T> &d) {
            memory.assign(d.cbegin(), d.cend());
        }

//------------------------------------------------------------------------------
///  @brief Get size of the buffer.
//------------------------------------------------------------------------------
        size_t size() const {
            return memory.size();
        }

//------------------------------------------------------------------------------
///  @brief Is every element the same.
///
///  @returns Returns true if every element is the same.
//------------------------------------------------------------------------------
        bool is_same() const {
            const T same = memory.at(0);
            for (size_t i = 1, ie = memory.size(); i < ie; i++) {
                if (memory.at(i) != same) {
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
            for (const T &d : memory) {
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
            for (const T &d : memory) {
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
            for (const T &d : memory) {
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
            for (const T &d : memory) {
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
            for (const T &d : memory) {
                if (d != static_cast<T> (-1.0)) {
                    return false;
                }
            }

            return true;
        }

//------------------------------------------------------------------------------
///  @brief Take sqrt.
//------------------------------------------------------------------------------
        void sqrt() {
            for (T &d : memory) {
                d = std::sqrt(d);
            }
        }

//------------------------------------------------------------------------------
///  @brief Take exp.
//------------------------------------------------------------------------------
        void exp() {
            for (T &d : memory) {
                d = std::exp(d);
            }
        }

//------------------------------------------------------------------------------
///  @brief Take log.
//------------------------------------------------------------------------------
        void log() {
            for (T &d : memory) {
                d = std::log(d);
            }
        }

//------------------------------------------------------------------------------
///  @brief Take sin.
//------------------------------------------------------------------------------
        void sin() {
            for (T &d : memory) {
                d = std::sin(d);
            }
        }

//------------------------------------------------------------------------------
///  @brief Take cos.
//------------------------------------------------------------------------------
        void cos() {
            for (T &d : memory) {
                d = std::cos(d);
            }
        }

//------------------------------------------------------------------------------
///  @brief Take erfi.
//------------------------------------------------------------------------------
        void erfi() requires(jit::complex_scalar<T>) {
            for (T &d : memory) {
                d = special::erfi(d);
            }
        }

//------------------------------------------------------------------------------
///  @brief Get a pointer to the basic memory buffer.
///
///  @returns The pointer to the buffer memory.
//------------------------------------------------------------------------------
        T *data() {
            return memory.data();
        }

//------------------------------------------------------------------------------
///  @brief Check for normal values.
///
///  @returns False if any NaN or Inf is found.
//------------------------------------------------------------------------------
        bool is_normal() const {
            for (const T &x : memory) {
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
///  @brief Add row operation.
///
///  Adds m_ij + v_i or v_i + m_ij. This will resize the buffer if it needs to 
///  be.
///
///  @param[in] x The right operand.
//------------------------------------------------------------------------------
        void add_row(const buffer<T> &x) {
            if (size() > x.size()) {
                assert(size()%x.size() == 0 &&
                       "Vector operand size is not a multiple of matrix operand size");

                const size_t num_columns = size()/x.size();
                const size_t num_rows = x.size();
                for (size_t i = 0; i < num_rows; i++) {
                    for (size_t j = 0; j < num_columns; j++) {
                        memory[i*num_rows + j] += x[i];
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
                        m[i*num_columns + j] = memory[i] + x[i*num_columns + j];
                    }
                }
                memory = m;
            }
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
            if (size() > x.size()) {
                assert(size()%x.size() == 0 &&
                       "Vector operand size is not a multiple of matrix operand size");

                const size_t num_columns = size()/x.size();
                const size_t num_rows = x.size();
                for (size_t i = 0; i < num_rows; i++) {
                    for (size_t j = 0; j < num_columns; j++) {
                        memory[i*num_columns + j] += x[j];
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
                        m[i*num_columns + j] = memory[j] + x[i*num_columns + j];
                    }
                }
                memory = m;
            }
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
            if (size() > x.size()) {
                assert(size()%x.size() == 0 &&
                       "Vector operand size is not a multiple of matrix operand size");

                const size_t num_columns = size()/x.size();
                const size_t num_rows = x.size();
                for (size_t i = 0; i < num_rows; i++) {
                    for (size_t j = 0; j < num_columns; j++) {
                        memory[i*num_columns + j] -= x[i];
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
                        m[i*num_columns + j] = memory[i] - x[i*num_columns + j];
                    }
                }
                memory = m;
            }
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
            if (size() > x.size()) {
                assert(size()%x.size() == 0 &&
                       "Vector operand size is not a multiple of matrix operand size");

                const size_t num_columns = size()/x.size();
                const size_t num_rows = x.size();
                for (size_t i = 0; i < num_rows; i++) {
                    for (size_t j = 0; j < num_columns; j++) {
                        memory[i*num_columns + j] -= x[j];
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
                        m[i*num_columns + j] = memory[j] - x[i*num_columns + j];
                    }
                }
                memory = m;
            }
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
            if (size() > x.size()) {
                assert(size()%x.size() == 0 &&
                       "Vector operand size is not a multiple of matrix operand size");

                const size_t num_columns = size()/x.size();
                const size_t num_rows = x.size();
                for (size_t i = 0; i < num_rows; i++) {
                    for (size_t j = 0; j < num_columns; j++) {
                        memory[i*num_columns + j] *= x[i];
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
                        m[i*num_columns + j] = memory[i]*x[i*num_columns + j];
                    }
                }
                memory = m;
            }
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
            if (size() > x.size()) {
                assert(size()%x.size() == 0 &&
                       "Vector operand size is not a multiple of matrix operand size");

                const size_t num_columns = size()/x.size();
                const size_t num_rows = x.size();
                for (size_t i = 0; i < num_rows; i++) {
                    for (size_t j = 0; j < num_columns; j++) {
                        memory[i*num_columns + j] *= x[j];
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
                        m[i*num_columns + j] = memory[j]*x[i*num_columns + j];
                    }
                }
                memory = m;
            }
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
            if (size() > x.size()) {
                assert(size()%x.size() == 0 &&
                       "Vector operand size is not a multiple of matrix operand size");

                const size_t num_columns = size()/x.size();
                const size_t num_rows = x.size();
                for (size_t i = 0; i < num_rows; i++) {
                    for (size_t j = 0; j < num_columns; j++) {
                        memory[i*num_columns + j] /= x[i];
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
                        m[i*num_columns + j] = memory[i]/x[i*num_columns + j];
                    }
                }
                memory = m;
            }
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
            if (size() > x.size()) {
                assert(size()%x.size() == 0 &&
                       "Vector operand size is not a multiple of matrix operand size");

                const size_t num_columns = size()/x.size();
                const size_t num_rows = x.size();
                for (size_t i = 0; i < num_rows; i++) {
                    for (size_t j = 0; j < num_columns; j++) {
                        memory[i*num_columns + j] /= x[j];
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
                        m[i*num_columns + j] = memory[j]/x[i*num_columns + j];
                    }
                }
                memory = m;
            }
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
                            memory[i*num_columns + j] = std::atan(x[i]/memory[i*num_columns + j]);
                        } else {
                            memory[i*num_columns + j] = std::atan2(x[i], memory[i*num_columns + j]);
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
                            m[i*num_columns + j] = std::atan(x[i*num_columns + j]/memory[i]);
                        } else {
                            m[i*num_columns + j] = std::atan2(x[i*num_columns + j], memory[i]);
                        }
                    }
                }
                memory = m;
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
                            memory[i*num_columns + j] = std::atan(x[j]/memory[i*num_columns + j]);
                        } else {
                            memory[i*num_columns + j] = std::atan2(x[j], memory[i*num_columns + j]);
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
                            m[i*num_columns + j] = std::atan(x[i*num_columns + j]/memory[j]);
                        } else {
                            m[i*num_columns + j] = std::atan2(x[i*num_columns + j], memory[j]);
                        }
                    }
                }
                memory = m;
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
                        memory[i*num_columns + j] = std::pow(memory[i*num_columns + j], x[i]);
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
                        m[i*num_columns + j] = std::pow(memory[i], x[i*num_columns + j]);
                    }
                }
                memory = m;
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
                        memory[i*num_columns + j] = std::pow(memory[i*num_columns + j], x[j]);
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
                        m[i*num_columns + j] = std::pow(memory[j], x[i*num_columns + j]);
                    }
                }
                memory = m;
            }
        }

///  Type def to retrieve the backend T type.
        typedef T base;
    };

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
        if (b.size() == 1) {
            const T right = b.at(0);
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] += right;
            }
            return a;
        } else if (a.size() == 1) {
            const T left = a.at(0);
            for (size_t i = 0, ie = b.size(); i < ie; i++) {
                b[i] += left;
            }
            return b;
        }

        assert(a.size() == b.size() &&
               "Left and right sizes are incompatible.");
        for (size_t i = 0, ie = a.size(); i < ie; i++) {
            a[i] += b.at(i);
        }
        return a;
    }

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
            if (a.at(i) != b.at(i)) {
                return false;
            }
        }
        return true;
    }

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
        if (b.size() == 1) {
            const T right = b.at(0);
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] -= right;
            }
            return a;
        } else if (a.size() == 1) {
            const T left = a.at(0);
            for (size_t i = 0, ie = b.size(); i < ie; i++) {
                b[i] = left - b.at(i);
            }
            return b;
        }

        assert(a.size() == b.size() &&
               "Left and right sizes are incompatible.");
        for (size_t i = 0, ie = a.size(); i < ie; i++) {
            a[i] -= b.at(i);
        }
        return a;
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
        if (b.size() == 1) {
            const T right = b.at(0);
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] *= right;
            }
            return a;
        } else if (a.size() == 1) {
            const T left = a.at(0);
            for (size_t i = 0, ie = b.size(); i < ie; i++) {
                b[i] *= left;
            }
            return b;
        }

        assert(a.size() == b.size() &&
               "Left and right sizes are incompatible.");
        for (size_t i = 0, ie = a.size(); i < ie; i++) {
            a[i] *= b.at(i);
        }
        return a;
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
        if (b.size() == 1) {
            const T right = b.at(0);
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] /= right;
            }
            return a;
        } else if (a.size() == 1) {
            const T left = a.at(0);
            for (size_t i = 0, ie = b.size(); i < ie; i++) {
                b[i] = left/b.at(i);
            }
            return b;
        }

        assert(a.size() == b.size() &&
               "Left and right sizes are incompatible.");
        for (size_t i = 0, ie = a.size(); i < ie; i++) {
            a[i] /= b.at(i);
        }
        return a;
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
            const T left = a.at(0);

            if (b.size() == 1) {
                const T middle = b.at(0);
                for (size_t i = 0, ie = c.size(); i < ie; i++) {
                    if constexpr (use_fma) {
                        c[i] = std::fma(left, middle, c.at(i));
                    } else {
                        c[i] = left*middle + c.at(i);
                    }
                }
                return c;
            } else if (c.size() == 1) {
                const T right = c.at(0);
                for (size_t i = 0, ie = b.size(); i < ie; i++) {
                    if constexpr (use_fma) {
                        b[i] = std::fma(left, b.at(i), right);
                    } else {
                        b[i] = left*b.at(i) + right;
                    }
                }
                return b;
            }

            assert(b.size() == c.size() &&
                   "Size mismatch between middle and right.");
            for (size_t i = 0, ie = b.size(); i < ie; i++) {
                if constexpr (use_fma) {
                    b[i] = std::fma(left, b.at(i), c.at(i));
                } else {
                    b[i] = left*b.at(i) + c.at(i);
                }
            }
            return b;
        } else if (b.size() == 1) {
            const T middle = b.at(0);
            if (c.size() == 1) {
                const T right = c.at(0);
                for (size_t i = 0, ie = a.size(); i < ie; i++) {
                    if constexpr (use_fma) {
                        a[i] = std::fma(a.at(i), middle, right);
                    } else {
                        a[i] = a.at(i)*middle + right;
                    }
                }
                return a;
            }

            assert(a.size() == c.size() &&
                   "Size mismatch between left and right.");
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                if constexpr (use_fma) {
                    a[i] = std::fma(a.at(i), middle, c.at(i));
                } else {
                    a[i] = a.at(i)*middle + c.at(i);
                }
            }
            return a;
        } else if (c.size() == 1) {
            assert(a.size() == b.size() &&
                   "Size mismatch between left and middle.");
            const T right = c.at(0);
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                if constexpr (use_fma) {
                    a[i] = std::fma(a.at(i), b.at(i), right);
                } else {
                    a[i] = a.at(i)*b.at(i) + right;
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
                a[i] = std::fma(a.at(i), b.at(i), c.at(i));
            } else {
                a[i] = a.at(i)*b.at(i) + c.at(i);
            }
        }
        return a;
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
            const T right = exponent.at(0);
            if (std::imag(right) == 0) {
                const int64_t right_int = static_cast<int64_t> (std::real(right));
                if (std::real(right) - right_int) {
                    if (right == static_cast<T> (0.5)) {
                        base.sqrt();
                        return base;
                    }

                    for (size_t i = 0, ie = base.size(); i < ie; i++) {
                        base[i] = std::pow(base.at(i), right);
                    }
                    return base;
                }

                if (right_int > 0) {
                    for (size_t i = 0, ie = base.size(); i < ie; i++) {
                        const T left = base.at(i);
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
                        const T left = static_cast<T> (1.0)/base.at(i);
                        base[i] = left;
                        for (size_t j = 0, je = std::abs(right_int) - 1; j < je; j++) {
                            base[i] *= left;
                        }
                    }
                    return base;
                }
            } else {
                for (size_t i = 0, ie = base.size(); i < ie; i++) {
                    base[i] = std::pow(base.at(i), right);
                }
                return base;
            }
        } else if (base.size() == 1) {
            const T left = base.at(0);
            for (size_t i = 0, ie = exponent.size(); i < ie; i++) {
                exponent[i] = std::pow(left, exponent.at(i));
            }
            return exponent;
        }

        assert(base.size() == exponent.size() &&
               "Left and right sizes are incompatible.");
        for (size_t i = 0, ie = base.size(); i < ie; i++) {
            base[i] = std::pow(base.at(i), exponent.at(i));
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
            const T right = y.at(0);
            for (size_t i = 0, ie = x.size(); i < ie; i++) {
                if constexpr (jit::complex_scalar<T>) {
                    x[i] = std::atan(right/x[i]);
                } else {
                    x[i] = std::atan2(right, x[i]);
                }
            }
            return x;
        } else if (x.size() == 1) {
            const T left = x.at(0);
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
