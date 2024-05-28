//------------------------------------------------------------------------------
///  @file backend.hpp
///  @brief Class signature to impliment compute backends.
///
///  Defined the function interfaces to access compute resources.
//------------------------------------------------------------------------------

#ifndef backend_h
#define backend_h

#include <algorithm>
#include <vector>

#include "special_functions.hpp"
#include "register.hpp"

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
///  @params[in] s Size of he data buffer.
//------------------------------------------------------------------------------
        buffer(const size_t s) :
        memory(s) {}

//------------------------------------------------------------------------------
///  @brief Construct a buffer backend with a size.
///
///  @params[in] s Size of he data buffer.
///  @params[in] d Scalar data to initalize.
//------------------------------------------------------------------------------
        buffer(const size_t s, const T d) :
        memory(s, d) {}

//------------------------------------------------------------------------------
///  @brief Construct a buffer backend from a vector.
///
///  @params[in] d Array buffer.
//------------------------------------------------------------------------------
        buffer(const std::vector<T> &d) :
        memory(d) {}

//------------------------------------------------------------------------------
///  @brief Construct a buffer backend from a buffer backend.
///
///  @params[in] d Backend buffer.
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
///  @params[in] d Scalar data to set.
//------------------------------------------------------------------------------
        void set(const T d) {
            memory.assign(memory.size(), d);
        }

//------------------------------------------------------------------------------
///  @brief Assign a vector value.
///
///  @params[in] d Vector data to set.
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
///  @brief Get the maximum value from the buffer.
///
///  @returns The maximum value.
//------------------------------------------------------------------------------
        T max() const {
            if constexpr (jit::is_complex<T> ()) {
                return *std::max_element(memory.cbegin(), memory.cend(),
                                         [] (const T a, const T b) {
                    return std::abs(a) < std::abs(b);
                });
            } else {
                return *std::max_element(memory.cbegin(), memory.cend());
            }
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
            for (T d : memory) {
                if (d != static_cast<T> (0.0)) {
                    return false;
                }
            }

            return true;
        }

//------------------------------------------------------------------------------
///  @brief Is every element negative.
///
///  @returns Returns true if every element is negative.
//------------------------------------------------------------------------------
        bool is_negative() const {
            for (T d : memory) {
                if (std::real(d) > std::real(static_cast<T> (0.0))) {
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
            for (T d : memory) {
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
        template<jit::float_scalar D=T>
        typename std::enable_if<jit::is_complex<D> (), void>::type erfi() {
            for (D &d : memory) {
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

///  Type def to retrieve the backend T type.
        typedef T base;
    };

//------------------------------------------------------------------------------
///  @brief Add operation.
///
///  @tparam T Base type of the calculation.
///
///  @params[in] a Left operand.
///  @params[in] b Right operand.
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
               "Left and right sizes are incompatable.");
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
///  @params[in] a Left operand.
///  @params[in] b Right operand.
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
///  @params[in] a Left operand.
///  @params[in] b Right operand.
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
               "Left and right sizes are incompatable.");
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
///  @params[in] a Left operand.
///  @params[in] b Right operand.
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
               "Left and right sizes are incompatable.");
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
///  @params[in] a Numerator.
///  @params[in] b Denominator.
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
               "Left and right sizes are incompatable.");
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
///  @params[in] a Left operand.
///  @params[in] b Middle operand.
///  @params[in] c Right operand.
///  @returns a*b + c.
//------------------------------------------------------------------------------
    template<jit::float_scalar T>
    inline buffer<T> fma(buffer<T> &a,
                         buffer<T> &b,
                         buffer<T> &c) {
        constexpr bool use_fma = !jit::is_complex<T> () &&
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
               "Left, middle and right sizes are incompatable.");
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
///  @params[in] base     Base to raise to the power of.
///  @params[in] exponent Power to apply to the base.
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
               "Left and right sizes are incompatable.");
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
///  @params[in] x X argument.
///  @params[in] y Y argument.
///  @returns atan2(y, x)
//------------------------------------------------------------------------------
    template<jit::float_scalar T>
    inline buffer<T> atan(buffer<T> &x,
                          buffer<T> &y) {
        if (y.size() == 1) {
            const T right = y.at(0);
            for (size_t i = 0, ie = x.size(); i < ie; i++) {
                if constexpr (jit::is_complex<T> ()) {
                    x[i] = std::atan(right/x[i]);
                } else {
                    x[i] = std::atan2(right, x[i]);
                }
            }
            return x;
        } else if (x.size() == 1) {
            const T left = x.at(0);
            for (size_t i = 0, ie = y.size(); i < ie; i++) {
                if constexpr (jit::is_complex<T> ()) {
                    y[i] = std::atan(y[i]/left);
                } else {
                    y[i] = std::atan2(y[i], left);
                }
            }
            return y;
        }

        assert(x.size() == y.size() &&
               "Left and right sizes are incompatable.");
        for (size_t i = 0, ie = x.size(); i < ie; i++) {
            if constexpr (jit::is_complex<T> ()) {
                x[i] = std::atan(y[i]/x[i]);
            } else {
                x[i] = std::atan2(y[i], x[i]);
            }
        }
        return x;
    }
}

#endif /* backend_h */
