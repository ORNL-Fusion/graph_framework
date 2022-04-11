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
    private:
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
        virtual const double &at(const size_t index) const final {
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
    cpu operator+(cpu a, cpu b) {
        if (b.size() == 1 && a.size() > 1) {
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] = a.at(i) + b.at(0);
            }
            return a;
        } else if (a.size() == 1 && b.size() > 1) {
            for (size_t i = 0, ie = b.size(); i < ie; i++) {
                b[i] = a.at(0) + b.at(i);
            }
            return b;
        }

        assert(a.size() == b.size() &&
               "Left and right sizes are incompatable.");
        for (size_t i = 0, ie = a.size(); i < ie; i++) {
            a[i] = a.at(i) + b.at(i);
        }
        return a;
    }

//------------------------------------------------------------------------------
///  @brief Subtract operation.
//------------------------------------------------------------------------------
    cpu operator-(cpu a, cpu b) {
        if (b.size() == 1 && a.size() > 1) {
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] = a.at(i) - b.at(0);
            }
            return a;
        } else if (a.size() == 1 && b.size() > 1) {
            for (size_t i = 0, ie = b.size(); i < ie; i++) {
                b[i] = a.at(0) - b.at(i);
            }
            return b;
        }

        assert(a.size() == b.size() &&
               "Left and right sizes are incompatable.");
        for (size_t i = 0, ie = a.size(); i < ie; i++) {
            a[i] = a.at(i) - b.at(i);
        }
        return a;
    }

//------------------------------------------------------------------------------
///  @brief Multiply operation.
//------------------------------------------------------------------------------
    cpu operator*(cpu a, cpu b) {
        if (b.size() == 1 && a.size() > 1) {
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] = a.at(i)*b.at(0);
            }
            return a;
        } else if (a.size() == 1 && b.size() > 1) {
            for (size_t i = 0, ie = b.size(); i < ie; i++) {
                b[i] = a.at(0)*b.at(i);
            }
            return b;
        }

        assert(a.size() == b.size() &&
               "Left and right sizes are incompatable.");
        for (size_t i = 0, ie = a.size(); i < ie; i++) {
            a[i] = a.at(i)*b.at(i);
        }
        return a;
    }

//------------------------------------------------------------------------------
///  @brief Multiply operation.
//------------------------------------------------------------------------------
    cpu operator/(cpu a, cpu b) {
        if (b.size() == 1 && a.size() > 1) {
            for (size_t i = 0, ie = a.size(); i < ie; i++) {
                a[i] = a.at(i)/b.at(0);
            }
            return a;
        } else if (a.size() == 1 && b.size() > 1) {
            for (size_t i = 0, ie = b.size(); i < ie; i++) {
                b[i] = a.at(0)/b.at(i);
            }
            return b;
        }

        assert(a.size() == b.size() &&
               "Left and right sizes are incompatable.");
        for (size_t i = 0, ie = a.size(); i < ie; i++) {
            a[i] = a.at(i)/b.at(i);
        }
        return a;
    }
}
#endif /* cpu_backend_h */
