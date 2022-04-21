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
    class cpu final : public buffer {
    protected:
///  The data buffer to hold the data.
        std::vector<double> data;

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
        cpu(const size_t s, const double d) :
        data(s, d) {}

//------------------------------------------------------------------------------
///  @brief Construct a cpu backend from a vector.
///
///  @param[in] d Array buffer.
//------------------------------------------------------------------------------
        cpu(const std::vector<double> &d) :
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
        virtual double &operator[] (const size_t index) final {
            return data[index];
        }

//------------------------------------------------------------------------------
///  @brief Const index operator.
//------------------------------------------------------------------------------
        virtual const double &operator[] (const size_t index) const final {
            return data[index];
        }

//------------------------------------------------------------------------------
///  @brief Get value at.
//------------------------------------------------------------------------------
        virtual const double at(const size_t index) const final {
            return data.at(index);
        }

//------------------------------------------------------------------------------
///  @brief Assign a constant value.
///
///  @param[in] d Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const double d) final {
            data.assign(data.size(), d);
        }

//------------------------------------------------------------------------------
///  @brief Assign a vector value.
///
///  @param[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const std::vector<double> &d) final {
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
        virtual double max() const final {
            return *std::max_element(data.cbegin(), data.cend());
        }

//------------------------------------------------------------------------------
///  @brief Is every element the same.
///
///  @returns Returns true if every element is the same.
//------------------------------------------------------------------------------
        virtual bool is_same() const final {
            const double same = data.at(0);
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
            for (double d : data) {
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
            for (double &d : data) {
                d = std::sqrt(d);
            }
        }

//------------------------------------------------------------------------------
///  @brief Take exp.
//------------------------------------------------------------------------------
        virtual void exp() final {
            for (double &d : data) {
                d = std::exp(d);
            }
        }

//------------------------------------------------------------------------------
///  @brief Take sine.
//------------------------------------------------------------------------------
        virtual void sin() final {
            for (double &d : data) {
                d = std::sin(d);
            }
        }

//------------------------------------------------------------------------------
///  @brief Take cosine.
//------------------------------------------------------------------------------
        virtual void cos() final {
            for (double &d : data) {
                d = std::cos(d);
            }
        }
    };

//------------------------------------------------------------------------------
///  @brief Add operation.
//------------------------------------------------------------------------------
    inline cpu operator+(cpu &a, cpu &b) {
        if (b.size() == 1) {
            const double right = b.at(0);
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] += right;
            }
            return a;
        } else if (a.size() == 1) {
            const double left = a.at(0);
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
///  @brief Subtract operation.
//------------------------------------------------------------------------------
    inline cpu operator-(cpu &a, cpu &b) {
        if (b.size() == 1) {
            const double right = b.at(0);
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] -= right;
            }
            return a;
        } else if (a.size() == 1) {
            const double left = a.at(0);
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
    inline cpu operator*(cpu &a, cpu &b) {
        if (b.size() == 1) {
            const double right = b.at(0);
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] *= right;
            }
            return a;
        } else if (a.size() == 1) {
            const double left = a.at(0);
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
///  @brief Multiply operation.
//------------------------------------------------------------------------------
    inline cpu operator/(cpu &a, cpu &b) {
        if (b.size() == 1) {
            const double right = b.at(0);
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] /= right;
            }
            return a;
        } else if (a.size() == 1) {
            const double left = a.at(0);
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
#ifdef USE_FMA
    inline cpu fma(cpu &a, cpu &b, cpu &c) {
        if (a.size() == 1 && b.size() == 1) {
            const double left = a.at(0);
            const double middle = b.at(0);
            for (size_t i = 0, ie = c.size(); i < ie; i++) {
                c[i] = std::fma(left, middle, c.at(i));
            }
            return c;
        } else if (a.size() == 1 && c.size() == 1) {
            const double left = a.at(0);
            const double right = c.at(0);
            for (size_t i = 0, ie = b.size(); i < ie; i++) {
                b[i] = std::fma(left, b.at(i), right);
            }
            return b;
        } else if (b.size() == 1 && c.size() == 1) {
            const double middle = b.at(0);
            const double right = c.at(0);
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] = std::fma(a.at(i), middle, right);
            }
            return a;
        } else if (a.size() == 1) {
            assert(b.size() == c.size() &&
                  "Middle and right sizes are incompatable.");
            const double left = a.at(0);
            for (size_t i = 0, ie = b.size(); i < ie; i++) {
                b[i] = std::fma(left, b.at(i), c.at(i));
            }
            return b;
        } else if (b.size() == 1) {
            assert(a.size() == c.size() &&
                  "Left and right sizes are incompatable.");
            const double middle = b.at(0);
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] = std::fma(a.at(i), middle, c.at(i));
            }
            return a;
        } else if (c.size() == 1) {
            assert(a.size() == b.size() &&
                  "Left and middle sizes are incompatable.");
            const double right = c.at(0);
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] = std::fma(a.at(i), b.at(i), right);
            }
            return a;
        }

        assert(a.size() == b.size() &&
               b.size() == c.size() &&
               a.size() == c.size() &&
               "Left, middle and right sizes are incompatable.");
        for (size_t i = 0, ie = a.size(); i < ie; i++) {
            a[i] = std::fma(a.at(i), b.at(i), c.at(i));
        }
        return a;
    }
#endif
}
#endif /* cpu_backend_h */
