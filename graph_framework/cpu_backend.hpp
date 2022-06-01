//------------------------------------------------------------------------------
///  @file cpu_backend.hpp
///  @brief Class signature for a cpu backend.
///
///  Defined the function to run kernels on the cpu.
//------------------------------------------------------------------------------

#ifndef cpu_backend_h
#define cpu_backend_h

#include <vector>
#include <cmath>
#include <algorithm>

#include "backend_protocall.hpp"

namespace backend {
//******************************************************************************
//  CPU Data buffer.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a generic buffer.
//------------------------------------------------------------------------------
    template<typename BASE>
    class cpu final : public buffer<BASE> {
    protected:
///  The data buffer to hold the data.
        std::vector<BASE> data;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a cpu backend with a size.
///
///  @param[in] s Size of he data buffer.
//------------------------------------------------------------------------------
        cpu(const size_t s) :
        data(s) {}

//------------------------------------------------------------------------------
///  @brief Construct a cpu backend with a size.
///
///  @param[in] s Size of he data buffer.
///  @param[in] d Scalar data to initalize.
//------------------------------------------------------------------------------
        cpu(const size_t s, const BASE d) :
        data(s, d) {}

//------------------------------------------------------------------------------
///  @brief Construct a cpu backend from a vector.
///
///  @param[in] d Array buffer.
//------------------------------------------------------------------------------
        cpu(const std::vector<BASE> &d) :
        data(d) {}

//------------------------------------------------------------------------------
///  @brief Construct a cpu backend from a cpu backend.
///
///  @param[in] d Backend buffer.
//------------------------------------------------------------------------------
        cpu(const cpu &d) :
        data(d.data) {}

//------------------------------------------------------------------------------
///  @brief Index operator.
//------------------------------------------------------------------------------
        virtual BASE &operator[] (const size_t index) final {
            return data[index];
        }

//------------------------------------------------------------------------------
///  @brief Const index operator.
//------------------------------------------------------------------------------
        virtual const BASE &operator[] (const size_t index) const final {
            return data[index];
        }

//------------------------------------------------------------------------------
///  @brief Get value at.
//------------------------------------------------------------------------------
        virtual const BASE at(const size_t index) const final {
            return data.at(index);
        }

//------------------------------------------------------------------------------
///  @brief Assign a constant value.
///
///  @param[in] d Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const BASE d) final {
            data.assign(data.size(), d);
        }

//------------------------------------------------------------------------------
///  @brief Assign a vector value.
///
///  @param[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const std::vector<BASE> &d) final {
            data.assign(d.cbegin(), d.cend());
        }

//------------------------------------------------------------------------------
///  @brief Get size of the buffer.
//------------------------------------------------------------------------------
        virtual size_t size() const final {
            return data.size();
        }

//------------------------------------------------------------------------------
///  @brief Get the maximum value from the buffer.
///
///  @returns The maximum value.
//------------------------------------------------------------------------------
        virtual BASE max() const final {
            return *std::max_element(data.cbegin(), data.cend());
        }

//------------------------------------------------------------------------------
///  @brief Is every element the same.
///
///  @returns Returns true if every element is the same.
//------------------------------------------------------------------------------
        virtual bool is_same() const final {
            const BASE same = data.at(0);
            for (size_t i = 1, ie = data.size(); i < ie; i++) {
                if (data.at(i) != same) {
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
        virtual bool is_zero() const final {
            for (BASE d : data) {
                if (d != 0.0) {
                    return false;
                }
            }

            return true;
        }

//------------------------------------------------------------------------------
///  @brief Take sqrt.
//------------------------------------------------------------------------------
        virtual void sqrt() final {
            for (BASE &d : data) {
                d = std::sqrt(d);
            }
        }

//------------------------------------------------------------------------------
///  @brief Take exp.
//------------------------------------------------------------------------------
        virtual void exp() final {
            for (BASE &d : data) {
                d = std::exp(d);
            }
        }

//------------------------------------------------------------------------------
///  @brief Take the natural log.
//------------------------------------------------------------------------------
        virtual void log() final {
            for (BASE &d : data) {
                d = std::log(d);
            }
        }

//------------------------------------------------------------------------------
///  @brief Take sine.
//------------------------------------------------------------------------------
        virtual void sin() final {
            for (BASE &d : data) {
                d = std::sin(d);
            }
        }

//------------------------------------------------------------------------------
///  @brief Take cosine.
//------------------------------------------------------------------------------
        virtual void cos() final {
            for (BASE &d : data) {
                d = std::cos(d);
            }
        }
    };

//------------------------------------------------------------------------------
///  @brief Add operation.
//------------------------------------------------------------------------------
    template<typename BASE>
    inline cpu<BASE> operator+(cpu<BASE> &a, cpu<BASE> &b) {
        if (b.size() == 1) {
            const BASE right = b.at(0);
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] += right;
            }
            return a;
        } else if (a.size() == 1) {
            const BASE left = a.at(0);
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
//------------------------------------------------------------------------------
    template<typename BASE>
    inline bool operator==(const cpu<BASE> &a, const cpu<BASE> &b) {
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
//------------------------------------------------------------------------------
    template<typename BASE>
    inline cpu<BASE> operator-(cpu<BASE> &a, cpu<BASE> &b) {
        if (b.size() == 1) {
            const BASE right = b.at(0);
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] -= right;
            }
            return a;
        } else if (a.size() == 1) {
            const BASE left = a.at(0);
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
//------------------------------------------------------------------------------
    template<typename BASE>
    inline cpu<BASE> operator*(cpu<BASE> &a, cpu<BASE> &b) {
        if (b.size() == 1) {
            const BASE right = b.at(0);
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] *= right;
            }
            return a;
        } else if (a.size() == 1) {
            const BASE left = a.at(0);
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
//------------------------------------------------------------------------------
    template<typename BASE>
    inline cpu<BASE> operator/(cpu<BASE> &a, cpu<BASE> &b) {
        if (b.size() == 1) {
            const BASE right = b.at(0);
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] /= right;
            }
            return a;
        } else if (a.size() == 1) {
            const BASE left = a.at(0);
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
///  @brief Multiply operation.
//------------------------------------------------------------------------------
    template<typename BASE>
    inline cpu<BASE> fma(cpu<BASE> &a, cpu<BASE> &b, cpu<BASE> &c) {
        if (a.size() == 1) {
            const BASE left = a.at(0);

            if (b.size() == 1) {
                const BASE middle = b.at(0);
                for (size_t i = 0, ie = c.size(); i < ie; i++) {
#ifdef FP_FAST_FMA
                    c[i] = std::fma(left, middle, c.at(i));
#else
                    c[i] = left*middle + c.at(i);
#endif
                }
                return c;
            } else if (c.size() == 1) {
                const BASE right = c.at(0);
                for (size_t i = 0, ie = b.size(); i < ie; i++) {
#ifdef FP_FAST_FMA
                    b[i] = std::fma(left, b.at(i), right);
#else
                    b[i] = left*b.at(i) + right;
#endif
                }
                return b;
            }

            assert(b.size() == c.size() &&
                   "Size mismatch between middle and right.");
            for (size_t i = 0, ie = b.size(); i < ie; i++) {
#ifdef FP_FAST_FMA
                b[i] = std::fma(left, b.at(i), c.at(i));
#else
                b[i] = left*b.at(i) + c.at(i);
#endif
            }
            return b;
        } else if (b.size() == 1) {
            const BASE middle = b.at(0);
            if (c.size() == 1) {
                const BASE right = c.at(0);
                for (size_t i = 0, ie = a.size(); i < ie; i++) {
#ifdef FP_FAST_FMA
                    a[i] = std::fma(a.at(i), middle, right);
#else
                    a[i] = a.at(i)*middle + right;
#endif
                }
                return a;
            }

            assert(a.size() == c.size() &&
                   "Size mismatch between left and right.");
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
#ifdef FP_FAST_FMA
                a[i] = std::fma(a.at(i), middle, c.at(i));
#else
                a[i] = a.at(i)*middle + c.at(i);
#endif
            }
            return a;
        } else if (c.size() == 1) {
            assert(a.size() == b.size() &&
                   "Size mismatch between left and middle.");
            const BASE right = c.at(0);
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
#ifdef FP_FAST_FMA
                a[i] = std::fma(a.at(i), b.at(i), right);
#else
                a[i] = a.at(i)*b.at(i) + right;
#endif
            }
            return a;
        }

        assert(a.size() == b.size() &&
               b.size() == c.size() &&
               a.size() == c.size() &&
               "Left, middle and right sizes are incompatable.");
        for (size_t i = 0, ie = a.size(); i < ie; i++) {
#ifdef FP_FAST_FMA
            a[i] = std::fma(a.at(i), b.at(i), c.at(i));
#else
            a[i] = a.at(i)*b.at(i) + c.at(i);
#endif
        }
        return a;
    }

//------------------------------------------------------------------------------
///  @brief Take the power.
///
///  @param[in] base     Raise to the power of.
///  @param[in] exponent Raise to the power of.
//------------------------------------------------------------------------------
    template<typename BASE>
    inline cpu<BASE> pow(cpu<BASE> &base,
                         cpu<BASE> &exponent) {
        if (exponent.size() == 1) {
            const BASE right = exponent.at(0);
            const int64_t right_int = static_cast<int64_t> (right);
            if (right - right_int) {
                if (right == 0.5) {
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
                    const BASE left = base.at(i);
                    for (size_t j = 0, je = right_int - 1; j < je; j++) {
                        base[i] *= left;
                    }
                }
                return base;
            } else {
                for (size_t i = 0, ie = base.size(); i < ie; i++) {
                    const BASE left = 1.0/base.at(i);
                    base[i] = left;
                    for (size_t j = 0, je = std::abs(right_int) - 1; j < je; j++) {
                        base[i] *= left;
                    }
                }
                return base;
            }
        } else if (base.size() == 1) {
            const BASE left = base.at(0);
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
}
#endif /* cpu_backend_h */
