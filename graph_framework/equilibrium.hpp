//------------------------------------------------------------------------------
///  @file equilibrium.hpp
///  @brief Class signature to impliment plasma equilibrium.
///
///  Defined the interfaces to access plasma equilibrium.
//------------------------------------------------------------------------------

#ifndef equilibrium_h
#define equilibrium_h

#include <mutex>

#include <netcdf.h>

#include "vector.hpp"
#include "trigonometry.hpp"
#include "piecewise.hpp"
#include "math.hpp"
#include "arithmetic.hpp"
#include "newton.hpp"

namespace equilibrium {
///  Lock to syncronize netcdf accross threads.
    static std::mutex sync;

//******************************************************************************
//  Equilibrium interface
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a generic equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class generic {
    protected:
///  Ion masses for each species.
        const std::vector<T> ion_masses;
///  Ion charge for each species.
        const std::vector<uint8_t> ion_charges;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a generic equilibrum.
///
///  @param[in] masses  Vector of ion masses.
///  @param[in] charges Vector of ion charges.
//------------------------------------------------------------------------------
        generic(const std::vector<T> &masses,
                const std::vector<uint8_t> &charges) :
        ion_masses(masses), ion_charges(charges) {
            assert(ion_masses.size() == ion_charges.size() &&
                   "Masses and charges need the same number of elements.");
        }

//------------------------------------------------------------------------------
///  @brief Destructor
//------------------------------------------------------------------------------
        virtual ~generic() {}

//------------------------------------------------------------------------------
///  @brief Get the number of ion species.
///
///  @returns The number of ion species.
//------------------------------------------------------------------------------
        size_t get_num_ion_species() const {
            return ion_masses.size();
        }

//------------------------------------------------------------------------------
///  @brief Get the mass for an ion species.
///
///  @param[in] index The species index.
///  @returns The mass for the ion at the index.
//------------------------------------------------------------------------------
        T get_ion_mass(const size_t index) const {
            return ion_masses.at(index);
        }

//------------------------------------------------------------------------------
///  @brief Get the charge for an ion species.
///
///  @param[in] index The species index.
///  @returns The number of ion species.
//------------------------------------------------------------------------------
        uint8_t get_ion_charge(const size_t index) const {
            return ion_charges.at(index);
        }

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron density expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_density(graph::shared_leaf<T, SAFE_MATH> x,
                             graph::shared_leaf<T, SAFE_MATH> y,
                             graph::shared_leaf<T, SAFE_MATH> z) = 0;

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @param[in] index The species index.
///  @param[in] x     X position.
///  @param[in] y     Y position.
///  @param[in] z     Z position.
///  @returns The ion density expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_density(const size_t index,
                        graph::shared_leaf<T, SAFE_MATH> x,
                        graph::shared_leaf<T, SAFE_MATH> y,
                        graph::shared_leaf<T, SAFE_MATH> z) = 0;

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron temperature expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_temperature(graph::shared_leaf<T, SAFE_MATH> x,
                                 graph::shared_leaf<T, SAFE_MATH> y,
                                 graph::shared_leaf<T, SAFE_MATH> z) = 0;

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @param[in] index The species index.
///  @param[in] x     X position.
///  @param[in] y     Y position.
///  @param[in] z     Z position.
///  @returns The ion temperature expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_temperature(const size_t index,
                            graph::shared_leaf<T, SAFE_MATH> x,
                            graph::shared_leaf<T, SAFE_MATH> y,
                            graph::shared_leaf<T, SAFE_MATH> z) = 0;
        
//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_magnetic_field(graph::shared_leaf<T, SAFE_MATH> x,
                           graph::shared_leaf<T, SAFE_MATH> y,
                           graph::shared_leaf<T, SAFE_MATH> z) = 0;

//------------------------------------------------------------------------------
///  @brief Get the characteristic field.
///
///  The characteristic field is equilibrium dependent.
///
///  @returns The characteristic field.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH> get_characteristic_field() = 0;

//------------------------------------------------------------------------------
///  @brief Get the contravariant basis vector in the x1 direction.
///
///  @param[in] x1 X1 posiiton.
///  @param[in] x2 X2 position.
///  @param[in] x3 X3 position.
///  @returns The contravaraiant basis vector in x1.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_esup1(graph::shared_leaf<T, SAFE_MATH> x1,
                  graph::shared_leaf<T, SAFE_MATH> x2,
                  graph::shared_leaf<T, SAFE_MATH> x3) {
            auto one = graph::one<T, SAFE_MATH> ();
            auto zero = graph::zero<T, SAFE_MATH> ();
            return graph::vector(one, zero, zero);
        }

//------------------------------------------------------------------------------
///  @brief Get the contravariant basis vector in the x2 direction.
///
///  @param[in] x1 X1 posiiton.
///  @param[in] x2 X2 position.
///  @param[in] x3 X3 position.
///  @returns The contravaraiant basis vector in x2.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_esup2(graph::shared_leaf<T, SAFE_MATH> x1,
                  graph::shared_leaf<T, SAFE_MATH> x2,
                  graph::shared_leaf<T, SAFE_MATH> x3) {
            auto one = graph::one<T, SAFE_MATH> ();
            auto zero = graph::zero<T, SAFE_MATH> ();
            return graph::vector(zero, one, zero);
        }

//------------------------------------------------------------------------------
///  @brief Get the contravariant basis vector in the x3 direction.
///
///  @param[in] x1 X1 posiiton.
///  @param[in] x2 X2 position.
///  @param[in] x3 X3 position.
///  @returns The contravaraiant basis vector in x3.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_esup3(graph::shared_leaf<T, SAFE_MATH> x1,
                  graph::shared_leaf<T, SAFE_MATH> x2,
                  graph::shared_leaf<T, SAFE_MATH> x3) {
            auto one = graph::one<T, SAFE_MATH> ();
            auto zero = graph::zero<T, SAFE_MATH> ();
            return graph::vector(zero, zero, one);
        }

//------------------------------------------------------------------------------
///  @brief Get the x position.
///
///  @param[in] x1 X1 posiiton.
///  @param[in] x2 X2 position.
///  @param[in] x3 X3 position.
///  @returns The x position.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_x(graph::shared_leaf<T, SAFE_MATH> x1,
              graph::shared_leaf<T, SAFE_MATH> x2,
              graph::shared_leaf<T, SAFE_MATH> x3) {
            return x1;
        }

//------------------------------------------------------------------------------
///  @brief Get the y position.
///
///  @param[in] x1 X1 posiiton.
///  @param[in] x2 X2 position.
///  @param[in] x3 X3 position.
///  @returns The y position.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_y(graph::shared_leaf<T, SAFE_MATH> x1,
              graph::shared_leaf<T, SAFE_MATH> x2,
              graph::shared_leaf<T, SAFE_MATH> x3) {
            return x2;
        }

//------------------------------------------------------------------------------
///  @brief Get the z position.
///
///  @param[in] x1 X1 posiiton.
///  @param[in] x2 X2 position.
///  @param[in] x3 X3 position.
///  @returns The z position.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_z(graph::shared_leaf<T, SAFE_MATH> x1,
              graph::shared_leaf<T, SAFE_MATH> x2,
              graph::shared_leaf<T, SAFE_MATH> x3) {
            return x3;
        }
    };

///  Convenience type alias for shared equilibria.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared = std::shared_ptr<generic<T, SAFE_MATH>>;

//******************************************************************************
//  No Magnetic equilibrium.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Uniform density with no magnetic field equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class no_magnetic_field : public generic<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a linear density with no magnetic field.
//------------------------------------------------------------------------------
        no_magnetic_field() :
        generic<T, SAFE_MATH> ({3.34449469E-27}, {1}) {}

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_density(graph::shared_leaf<T, SAFE_MATH> x,
                             graph::shared_leaf<T, SAFE_MATH> y,
                             graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::constant<T, SAFE_MATH> (static_cast<T> (1.0E19)) *
                   (graph::constant<T, SAFE_MATH> (static_cast<T> (0.1))*x +
                    graph::one<T, SAFE_MATH> ());
        }

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @param[in] index The species index.
///  @param[in] x     X position.
///  @param[in] y     Y position.
///  @param[in] z     Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_density(const size_t index,
                        graph::shared_leaf<T, SAFE_MATH> x,
                        graph::shared_leaf<T, SAFE_MATH> y,
                        graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::constant<T, SAFE_MATH> (static_cast<T> (1.0E19)) *
                   (graph::constant<T, SAFE_MATH> (static_cast<T> (0.1))*x +
                    graph::one<T, SAFE_MATH> ());
        }

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_temperature(graph::shared_leaf<T, SAFE_MATH> x,
                                 graph::shared_leaf<T, SAFE_MATH> y,
                                 graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::constant<T, SAFE_MATH> (static_cast<T> (1000.0));
        }

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @param[in] index The species index.
///  @param[in] x     X position.
///  @param[in] y     Y position.
///  @param[in] z     Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_temperature(const size_t index,
                            graph::shared_leaf<T, SAFE_MATH> x,
                            graph::shared_leaf<T, SAFE_MATH> y,
                            graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::constant<T, SAFE_MATH> (static_cast<T> (1000.0));
        }

//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_magnetic_field(graph::shared_leaf<T, SAFE_MATH> x,
                           graph::shared_leaf<T, SAFE_MATH> y,
                           graph::shared_leaf<T, SAFE_MATH> z) final {
            auto zero = graph::zero<T, SAFE_MATH> ();
            return graph::vector(zero, zero, zero);
        }

//------------------------------------------------------------------------------
///  @brief Get the characteristic field.
///
///  To avoid divide by zeros use the value of 1.
///
///  @returns The characteristic field.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_characteristic_field() final {
            return graph::one<T, SAFE_MATH> ();
        }
    };

//------------------------------------------------------------------------------
///  @brief Convenience function to build a no magnetic field equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @returns A constructed no magnetic field equilibrium.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared<T, SAFE_MATH> make_no_magnetic_field() {
        return std::make_shared<no_magnetic_field<T, SAFE_MATH>> ();
    }

//******************************************************************************
//  Slab equilibrium.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Uniform density with varying magnetic field equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class slab : public generic<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a guassian density with uniform magnetic field.
//------------------------------------------------------------------------------
        slab() :
        generic<T, SAFE_MATH> ({3.34449469E-27}, {1}) {}

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_density(graph::shared_leaf<T, SAFE_MATH> x,
                             graph::shared_leaf<T, SAFE_MATH> y,
                             graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::constant<T, SAFE_MATH> (static_cast<T> (1.0E19));
        }

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @param[in] index The species index.
///  @param[in] x     X position.
///  @param[in] y     Y position.
///  @param[in] z     Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_density(const size_t index,
                        graph::shared_leaf<T, SAFE_MATH> x,
                        graph::shared_leaf<T, SAFE_MATH> y,
                        graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::constant<T, SAFE_MATH> (static_cast<T> (1.0E19));
        }

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_temperature(graph::shared_leaf<T, SAFE_MATH> x,
                                 graph::shared_leaf<T, SAFE_MATH> y,
                                 graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::constant<T, SAFE_MATH> (static_cast<T> (1000.0));
        }

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @param[in] index The species index.
///  @param[in] x     X position.
///  @param[in] y     Y position.
///  @param[in] z     Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_temperature(const size_t index,
                            graph::shared_leaf<T, SAFE_MATH> x,
                            graph::shared_leaf<T, SAFE_MATH> y,
                            graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::constant<T, SAFE_MATH> (static_cast<T> (1000.0));
        }

//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_magnetic_field(graph::shared_leaf<T, SAFE_MATH> x,
                           graph::shared_leaf<T, SAFE_MATH> y,
                           graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::vector(0.0, 0.0, 0.1*x + 1.0);
        }

//------------------------------------------------------------------------------
///  @brief Get the characteristic field.
///
///  Use the value at the y intercept.
///
///  @returns The characteristic field.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_characteristic_field() final {
            return graph::one<T, SAFE_MATH> ();
        }
    };

//------------------------------------------------------------------------------
///  @brief Convenience function to build a slab equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @returns A constructed slab equilibrium.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared<T, SAFE_MATH> make_slab() {
        return std::make_shared<slab<T, SAFE_MATH>> ();
    }

//******************************************************************************
//  Slab density equilibrium.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Vary density with uniform magnetic field equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class slab_density : public generic<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a guassian density with uniform magnetic field.
//------------------------------------------------------------------------------
        slab_density() :
        generic<T, SAFE_MATH> ({3.34449469E-27}, {1}) {}

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_density(graph::shared_leaf<T, SAFE_MATH> x,
                             graph::shared_leaf<T, SAFE_MATH> y,
                             graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::constant<T, SAFE_MATH> (static_cast<T> (1.0E19)) *
                   (graph::constant<T, SAFE_MATH> (static_cast<T> (0.1))*x +
                    graph::one<T, SAFE_MATH> ());
        }

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @param[in] index The species index.
///  @param[in] x     X position.
///  @param[in] y     Y position.
///  @param[in] z     Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_density(const size_t index,
                        graph::shared_leaf<T, SAFE_MATH> x,
                        graph::shared_leaf<T, SAFE_MATH> y,
                        graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::constant<T, SAFE_MATH> (static_cast<T> (1.0E19)) *
                   (graph::constant<T, SAFE_MATH> (static_cast<T> (0.1))*x +
                    graph::one<T, SAFE_MATH> ());
        }

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_temperature(graph::shared_leaf<T, SAFE_MATH> x,
                                 graph::shared_leaf<T, SAFE_MATH> y,
                                 graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::constant<T, SAFE_MATH> (static_cast<T> (1000.0));
        }

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @param[in] index The species index.
///  @param[in] x     X position.
///  @param[in] y     Y position.
///  @param[in] z     Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_temperature(const size_t index,
                            graph::shared_leaf<T, SAFE_MATH> x,
                            graph::shared_leaf<T, SAFE_MATH> y,
                            graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::constant<T, SAFE_MATH> (static_cast<T> (1000.0));
        }
        
//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_magnetic_field(graph::shared_leaf<T, SAFE_MATH> x,
                           graph::shared_leaf<T, SAFE_MATH> y,
                           graph::shared_leaf<T, SAFE_MATH> z) final {
            auto zero = graph::zero<T, SAFE_MATH> ();
            return graph::vector(zero, zero, graph::one<T, SAFE_MATH> ());
        }

//------------------------------------------------------------------------------
///  @brief Get the characteristic field.
///
///  Use the value at the y intercept.
///
///  @returns The characteristic field.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_characteristic_field() final {
            return graph::one<T, SAFE_MATH> ();
        }
    };

//------------------------------------------------------------------------------
///  @brief Convenience function to build a slab density equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @returns A constructed slab density equilibrium.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared<T, SAFE_MATH> make_slab_density() {
        return std::make_shared<slab_density<T, SAFE_MATH>> ();
    }

//******************************************************************************
//  Slab field gradient equilibrium.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Vary density with uniform magnetic field equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class slab_field : public generic<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a guassian density with uniform magnetic field.
//------------------------------------------------------------------------------
        slab_field() :
        generic<T, SAFE_MATH> ({3.34449469E-27}, {1}) {}

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_density(graph::shared_leaf<T, SAFE_MATH> x,
                             graph::shared_leaf<T, SAFE_MATH> y,
                             graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::constant<T, SAFE_MATH> (static_cast<T> (1.0E19)) *
                   (graph::constant<T, SAFE_MATH> (static_cast<T> (0.01))*x +
                    graph::one<T, SAFE_MATH> ());
        }

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @param[in] index The species index.
///  @param[in] x     X position.
///  @param[in] y     Y position.
///  @param[in] z     Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_density(const size_t index,
                        graph::shared_leaf<T, SAFE_MATH> x,
                        graph::shared_leaf<T, SAFE_MATH> y,
                        graph::shared_leaf<T, SAFE_MATH> z) final {
            return get_electron_density(x, y, z);
        }

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_temperature(graph::shared_leaf<T, SAFE_MATH> x,
                                 graph::shared_leaf<T, SAFE_MATH> y,
                                 graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::constant<T, SAFE_MATH> (static_cast<T> (2000.0)) *
                   (graph::constant<T, SAFE_MATH> (static_cast<T> (0.01))*x +
                    graph::one<T, SAFE_MATH> ());
        }

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @param[in] index The species index.
///  @param[in] x     X position.
///  @param[in] y     Y position.
///  @param[in] z     Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_temperature(const size_t index,
                            graph::shared_leaf<T, SAFE_MATH> x,
                            graph::shared_leaf<T, SAFE_MATH> y,
                            graph::shared_leaf<T, SAFE_MATH> z) final {
            return get_electron_temperature(x, y, z);
        }
        
//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_magnetic_field(graph::shared_leaf<T, SAFE_MATH> x,
                           graph::shared_leaf<T, SAFE_MATH> y,
                           graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::vector(0.0, 0.0, 0.01*x + 1.0);
        }

//------------------------------------------------------------------------------
///  @brief Get the characteristic field.
///
///  Use the value at the y intercept.
///
///  @returns The characteristic field.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_characteristic_field() final {
            return graph::one<T, SAFE_MATH> ();
        }
    };

//------------------------------------------------------------------------------
///  @brief Convenience function to build a slab density equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @returns A constructed slab density equilibrium.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared<T, SAFE_MATH> make_slab_field() {
        return std::make_shared<slab_field<T, SAFE_MATH>> ();
    }
//******************************************************************************
//  Guassian density with a uniform magnetic field.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Guassian density with uniform magnetic field equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class guassian_density : public generic<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a guassian density with uniform magnetic field.
//------------------------------------------------------------------------------
        guassian_density() :
        generic<T, SAFE_MATH> ({3.34449469E-27}, {1}) {}

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_density(graph::shared_leaf<T, SAFE_MATH> x,
                             graph::shared_leaf<T, SAFE_MATH> y,
                             graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::constant<T, SAFE_MATH> (static_cast<T> (1.0E19)) *
                   graph::exp((x*x + y*y)/graph::constant<T, SAFE_MATH> (static_cast<T> (-0.2)));
        }

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @param[in] index The species index.
///  @param[in] x     X position.
///  @param[in] y     Y position.
///  @param[in] z     Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_density(const size_t index,
                        graph::shared_leaf<T, SAFE_MATH> x,
                        graph::shared_leaf<T, SAFE_MATH> y,
                        graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::constant<T, SAFE_MATH> (static_cast<T> (1.0E19)) *
                   graph::exp((x*x + y*y)/graph::constant<T, SAFE_MATH> (static_cast<T> (-0.2)));
        }

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_temperature(graph::shared_leaf<T, SAFE_MATH> x,
                                 graph::shared_leaf<T, SAFE_MATH> y,
                                 graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::constant<T, SAFE_MATH> (static_cast<T> (1000.0));
        }

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @param[in] index The species index.
///  @param[in] x     X position.
///  @param[in] y     Y position.
///  @param[in] z     Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_temperature(const size_t index,
                            graph::shared_leaf<T, SAFE_MATH> x,
                            graph::shared_leaf<T, SAFE_MATH> y,
                            graph::shared_leaf<T, SAFE_MATH> z) final {
            return graph::constant<T, SAFE_MATH> (static_cast<T> (1000.0));
        }
        
//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_magnetic_field(graph::shared_leaf<T, SAFE_MATH> x,
                           graph::shared_leaf<T, SAFE_MATH> y,
                           graph::shared_leaf<T, SAFE_MATH> z) final {
            auto zero = graph::zero<T, SAFE_MATH> ();
            return graph::vector(graph::one<T, SAFE_MATH> (), zero, zero);
        }

//------------------------------------------------------------------------------
///  @brief Get the characteristic field.
///
///  Use the value at the y intercept.
///
///  @returns The characteristic field.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_characteristic_field() final {
            return graph::one<T, SAFE_MATH> ();
        }
    };

//------------------------------------------------------------------------------
///  @brief Convenience function to build a guassian density equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @returns A constructed guassian density equilibrium.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared<T, SAFE_MATH> make_guassian_density() {
        return std::make_shared<guassian_density<T, SAFE_MATH>> ();
    }

//******************************************************************************
//  2D EFIT equilibrium.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief 2D EFIT equilibrium.
///
///  This takes a BiCublic spline representation of the psi and cubic splines for
///  ne, te, p, and fpol.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class efit final : public generic<T, SAFE_MATH> {
    private:
///  Minimum psi.
        graph::shared_leaf<T, SAFE_MATH> psimin;
///  Psi grid spacing.
        graph::shared_leaf<T, SAFE_MATH> dpsi;

//  Temperature spline coefficients.
///  Temperature c0.
        const backend::buffer<T> te_c0;
///  Temperature c1.
        const backend::buffer<T> te_c1;
///  Temperature c2.
        const backend::buffer<T> te_c2;
///  Temperature c3.
        const backend::buffer<T> te_c3;
///  Temperature scale factor.
        graph::shared_leaf<T, SAFE_MATH> te_scale;

//  Density spline coefficients.
///  Density c0.
        const backend::buffer<T> ne_c0;
///  Density c1.
        const backend::buffer<T> ne_c1;
///  Density c2.
        const backend::buffer<T> ne_c2;
///  Density c3.
        const backend::buffer<T> ne_c3;
///  Density scale factor.
        graph::shared_leaf<T, SAFE_MATH> ne_scale;

//  Pressure spline coefficients.
///  Pressure c0.
        const backend::buffer<T> pres_c0;
///  Pressure c1.
        const backend::buffer<T> pres_c1;
///  Pressure c2.
        const backend::buffer<T> pres_c2;
///  Pressure c3.
        const backend::buffer<T> pres_c3;
///  Pressure scale factor.
        graph::shared_leaf<T, SAFE_MATH> pres_scale;

///  Minimum R.
        graph::shared_leaf<T, SAFE_MATH> rmin;
///  R grid spacing.
        graph::shared_leaf<T, SAFE_MATH> dr;
///  Minimum Z.
        graph::shared_leaf<T, SAFE_MATH> zmin;
///  Z grid spacing.
        graph::shared_leaf<T, SAFE_MATH> dz;

//  Fpol spline coefficients.
///  Fpol c0.
        const backend::buffer<T> fpol_c0;
///  Fpol c1.
        const backend::buffer<T> fpol_c1;
///  Fpol c2.
        const backend::buffer<T> fpol_c2;
///  Fpol c3.
        const backend::buffer<T> fpol_c3;

//  Psi spline coefficients.
///  Number of columns.
        const size_t num_cols;
///  Psi c00.
        const backend::buffer<T> c00;
///  Psi c01.
        const backend::buffer<T> c01;
///  Psi c02.
        const backend::buffer<T> c02;
///  Psi c03.
        const backend::buffer<T> c03;
///  Psi c10.
        const backend::buffer<T> c10;
///  Psi c11.
        const backend::buffer<T> c11;
///  Psi c12.
        const backend::buffer<T> c12;
///  Psi c13.
        const backend::buffer<T> c13;
///  Psi c20.
        const backend::buffer<T> c20;
///  Psi c21.
        const backend::buffer<T> c21;
///  Psi c22.
        const backend::buffer<T> c22;
///  Psi c23.
        const backend::buffer<T> c23;
///  Psi c30.
        const backend::buffer<T> c30;
///  Psi c31.
        const backend::buffer<T> c31;
///  Psi c32.
        const backend::buffer<T> c32;
///  Psi c33.
        const backend::buffer<T> c33;

//  Cached values.
///  X position cache.
        graph::shared_leaf<T, SAFE_MATH> x_cache;
///  Y position cache.
        graph::shared_leaf<T, SAFE_MATH> y_cache;
///  Z position cache.
        graph::shared_leaf<T, SAFE_MATH> z_cache;

///  Cached electron density value.
        graph::shared_leaf<T, SAFE_MATH> ne_cache;
///  Cached electron density value.
        graph::shared_leaf<T, SAFE_MATH> ni_cache;
///  Cached electron temperature value.
        graph::shared_leaf<T, SAFE_MATH> te_cache;
///  Cached ion temperature value.
        graph::shared_leaf<T, SAFE_MATH> ti_cache;

///  Cached magnetic field vector.
        graph::shared_vector<T, SAFE_MATH> b_cache;

///  Cached magnetic field vector.
        graph::shared_leaf<T, SAFE_MATH> psi_norm_cache;

//------------------------------------------------------------------------------
///  @brief Build psi.
///
///  @param[in] r_norm The normalized radial position.
///  @param[in] z_norm The normalized z position.
///  @returns The psi value.
//------------------------------------------------------------------------------
        graph::shared_leaf<T, SAFE_MATH>
        build_psi(graph::shared_leaf<T, SAFE_MATH> r_norm,
                  graph::shared_leaf<T, SAFE_MATH> z_norm) {
            auto c00_temp = graph::piecewise_2D(c00, num_cols, r_norm, z_norm);
            auto c01_temp = graph::piecewise_2D(c01, num_cols, r_norm, z_norm);
            auto c02_temp = graph::piecewise_2D(c02, num_cols, r_norm, z_norm);
            auto c03_temp = graph::piecewise_2D(c03, num_cols, r_norm, z_norm);

            auto c10_temp = graph::piecewise_2D(c10, num_cols, r_norm, z_norm);
            auto c11_temp = graph::piecewise_2D(c11, num_cols, r_norm, z_norm);
            auto c12_temp = graph::piecewise_2D(c12, num_cols, r_norm, z_norm);
            auto c13_temp = graph::piecewise_2D(c13, num_cols, r_norm, z_norm);

            auto c20_temp = graph::piecewise_2D(c20, num_cols, r_norm, z_norm);
            auto c21_temp = graph::piecewise_2D(c21, num_cols, r_norm, z_norm);
            auto c22_temp = graph::piecewise_2D(c22, num_cols, r_norm, z_norm);
            auto c23_temp = graph::piecewise_2D(c23, num_cols, r_norm, z_norm);

            auto c30_temp = graph::piecewise_2D(c30, num_cols, r_norm, z_norm);
            auto c31_temp = graph::piecewise_2D(c31, num_cols, r_norm, z_norm);
            auto c32_temp = graph::piecewise_2D(c32, num_cols, r_norm, z_norm);
            auto c33_temp = graph::piecewise_2D(c33, num_cols, r_norm, z_norm);

            return   c00_temp
                   + c01_temp*z_norm
                   + c02_temp*(z_norm*z_norm)
                   + c03_temp*(z_norm*z_norm*z_norm)
                   + c10_temp*r_norm
                   + c11_temp*r_norm*z_norm
                   + c12_temp*r_norm*(z_norm*z_norm)
                   + c13_temp*r_norm*(z_norm*z_norm*z_norm)
                   + c20_temp*(r_norm*r_norm)
                   + c21_temp*(r_norm*r_norm)*z_norm
                   + c22_temp*(r_norm*r_norm)*(z_norm*z_norm)
                   + c23_temp*(r_norm*r_norm)*(z_norm*z_norm*z_norm)
                   + c30_temp*(r_norm*r_norm*r_norm)
                   + c31_temp*(r_norm*r_norm*r_norm)*z_norm
                   + c32_temp*(r_norm*r_norm*r_norm)*(z_norm*z_norm)
                   + c33_temp*(r_norm*r_norm*r_norm)*(z_norm*z_norm*z_norm);
        }

//------------------------------------------------------------------------------
///  @brief Set cache values.
///
///  Sets the cached values if x and y do not match.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
//------------------------------------------------------------------------------
        void set_cache(graph::shared_leaf<T, SAFE_MATH> x,
                       graph::shared_leaf<T, SAFE_MATH> y,
                       graph::shared_leaf<T, SAFE_MATH> z) {
            if (!x->is_match(x_cache) ||
                !y->is_match(y_cache) ||
                !z->is_match(z_cache)) {
                x_cache = x;
                y_cache = y;
                z_cache = z;

                auto r = graph::sqrt(x*x + y*y);
                auto r_norm = (r - rmin)/dr;
                auto z_norm = (z - zmin)/dz;

                auto psi = build_psi(r_norm, z_norm);
                psi_norm_cache = (psi - psimin)/dpsi;

                auto n0_temp = graph::piecewise_1D(ne_c0, psi_norm_cache);
                auto n1_temp = graph::piecewise_1D(ne_c1, psi_norm_cache);
                auto n2_temp = graph::piecewise_1D(ne_c2, psi_norm_cache);
                auto n3_temp = graph::piecewise_1D(ne_c3, psi_norm_cache);

                ne_cache = ne_scale*(n0_temp +
                                     n1_temp*psi_norm_cache +
                                     n2_temp*psi_norm_cache*psi_norm_cache +
                                     n3_temp*psi_norm_cache*psi_norm_cache*psi_norm_cache);

                auto t0_temp = graph::piecewise_1D(te_c0, psi_norm_cache);
                auto t1_temp = graph::piecewise_1D(te_c1, psi_norm_cache);
                auto t2_temp = graph::piecewise_1D(te_c2, psi_norm_cache);
                auto t3_temp = graph::piecewise_1D(te_c3, psi_norm_cache);

                te_cache = te_scale*(t0_temp +
                                     t1_temp*psi_norm_cache +
                                     t2_temp*psi_norm_cache*psi_norm_cache +
                                     t3_temp*psi_norm_cache*psi_norm_cache*psi_norm_cache);

                auto p0_temp = graph::piecewise_1D(pres_c0, psi_norm_cache);
                auto p1_temp = graph::piecewise_1D(pres_c1, psi_norm_cache);
                auto p2_temp = graph::piecewise_1D(pres_c2, psi_norm_cache);
                auto p3_temp = graph::piecewise_1D(pres_c3, psi_norm_cache);

                auto pressure = pres_scale*(p0_temp +
                                            p1_temp*psi_norm_cache +
                                            p2_temp*psi_norm_cache*psi_norm_cache +
                                            p3_temp*psi_norm_cache*psi_norm_cache*psi_norm_cache);

                auto q = graph::constant<T, SAFE_MATH> (static_cast<T> (1.60218E-19));

                ni_cache = te_cache;
                ti_cache = (pressure - ne_cache*te_cache*q)/(ni_cache*q);
                
                auto phi = graph::atan(x, y);

                auto br = psi->df(z)/r;

                auto b0_temp = graph::piecewise_1D(fpol_c0, r_norm);
                auto b1_temp = graph::piecewise_1D(fpol_c1, r_norm);
                auto b2_temp = graph::piecewise_1D(fpol_c2, r_norm);
                auto b3_temp = graph::piecewise_1D(fpol_c3, r_norm);

                auto bp = (b0_temp +
                           b1_temp*r_norm +
                           b2_temp*r_norm*r_norm +
                           b3_temp*r_norm*r_norm*r_norm)/r;

                auto bz = -psi->df(r)/r;

                auto cos = graph::cos(phi);
                auto sin = graph::sin(phi);
                
                b_cache = graph::vector(br*cos - bp*sin,
                                        br*sin + bp*cos,
                                        bz);
            }
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a EFIT equilibrium.
///
///  @param[in] psimin     Minimum psi value.
///  @param[in] dpsi       Change in psi value.
///  @param[in] te_c0      Te c0 spline coefficient.
///  @param[in] te_c1      Te c1 spline coefficient.
///  @param[in] te_c2      Te c2 spline coefficient.
///  @param[in] te_c3      Te c3 spline coefficient.
///  @param[in] te_scale   Temperatire scale.
///  @param[in] ne_c0      Ne c0 spline coefficient.
///  @param[in] ne_c1      Ne c1 spline coefficient.
///  @param[in] ne_c2      Ne c2 spline coefficient.
///  @param[in] ne_c3      Ne c3 spline coefficient.
///  @param[in] ne_scale   Denisty scale.
///  @param[in] pres_c0    Pressure c0 spline coefficient.
///  @param[in] pres_c1    Pressure c1 spline coefficient.
///  @param[in] pres_c2    Pressure c2 spline coefficient.
///  @param[in] pres_c3    Pressure c3 spline coefficient.
///  @param[in] pres_scale Pressure scale.
///  @param[in] rmin       Radial gird minimum.
///  @param[in] dr         Radial grid spacing.
///  @param[in] zmin       Vertical grid minimum.
///  @param[in] dz         Vertical grid spacing.
///  @param[in] fpol_c0    Flux function c0 spline coefficient.
///  @param[in] fpol_c1    Flux function c1 spline coefficient.
///  @param[in] fpol_c2    Flux function c2 spline coefficient.
///  @param[in] fpol_c3    Flux function c3 spline coefficient.
///  @param[in] num_cols   Number of columns for the 2D splines.
///  @param[in] c00        Psi c00 spline coefficient.
///  @param[in] c01        Psi c01 spline coefficient.
///  @param[in] c02        Psi c02 spline coefficient.
///  @param[in] c03        Psi c03 spline coefficient.
///  @param[in] c10        Psi c10 spline coefficient.
///  @param[in] c11        Psi c11 spline coefficient.
///  @param[in] c12        Psi c12 spline coefficient.
///  @param[in] c13        Psi c13 spline coefficient.
///  @param[in] c20        Psi c20 spline coefficient.
///  @param[in] c21        Psi c21 spline coefficient.
///  @param[in] c22        Psi c22 spline coefficient.
///  @param[in] c23        Psi c23 spline coefficient.
///  @param[in] c30        Psi c30 spline coefficient.
///  @param[in] c31        Psi c31 spline coefficient.
///  @param[in] c32        Psi c32 spline coefficient.
///  @param[in] c33        Psi c33 spline coefficient.
//------------------------------------------------------------------------------
        efit(graph::shared_leaf<T, SAFE_MATH> psimin,
             graph::shared_leaf<T, SAFE_MATH> dpsi,
             const backend::buffer<T> te_c0,
             const backend::buffer<T> te_c1,
             const backend::buffer<T> te_c2,
             const backend::buffer<T> te_c3,
             graph::shared_leaf<T, SAFE_MATH> te_scale,
             const backend::buffer<T> ne_c0,
             const backend::buffer<T> ne_c1,
             const backend::buffer<T> ne_c2,
             const backend::buffer<T> ne_c3,
             graph::shared_leaf<T, SAFE_MATH> ne_scale,
             const backend::buffer<T> pres_c0,
             const backend::buffer<T> pres_c1,
             const backend::buffer<T> pres_c2,
             const backend::buffer<T> pres_c3,
             graph::shared_leaf<T, SAFE_MATH> pres_scale,
             graph::shared_leaf<T, SAFE_MATH> rmin,
             graph::shared_leaf<T, SAFE_MATH> dr,
             graph::shared_leaf<T, SAFE_MATH> zmin,
             graph::shared_leaf<T, SAFE_MATH> dz,
             const backend::buffer<T> fpol_c0,
             const backend::buffer<T> fpol_c1,
             const backend::buffer<T> fpol_c2,
             const backend::buffer<T> fpol_c3,
             const size_t num_cols,
             const backend::buffer<T> c00,
             const backend::buffer<T> c01,
             const backend::buffer<T> c02,
             const backend::buffer<T> c03,
             const backend::buffer<T> c10,
             const backend::buffer<T> c11,
             const backend::buffer<T> c12,
             const backend::buffer<T> c13,
             const backend::buffer<T> c20,
             const backend::buffer<T> c21,
             const backend::buffer<T> c22,
             const backend::buffer<T> c23,
             const backend::buffer<T> c30,
             const backend::buffer<T> c31,
             const backend::buffer<T> c32,
             const backend::buffer<T> c33) :
        generic<T, SAFE_MATH> ({3.34449469E-27} ,{1}),
        psimin(psimin), dpsi(dpsi), num_cols(num_cols),
        te_c0(te_c0), te_c1(te_c1), te_c2(te_c2), te_c3(te_c3), te_scale(te_scale),
        ne_c0(te_c0), ne_c1(te_c1), ne_c2(ne_c2), ne_c3(ne_c3), ne_scale(ne_scale),
        pres_c0(pres_c0), pres_c1(pres_c1), pres_c2(pres_c2), pres_c3(pres_c3),
        pres_scale(pres_scale), rmin(rmin), dr(dr), zmin(zmin), dz(dz),
        fpol_c0(fpol_c0), fpol_c1(fpol_c1), fpol_c2(fpol_c2), fpol_c3(fpol_c3),
        c00(c00), c01(c01), c02(c02), c03(c03),
        c10(c10), c11(c11), c12(c12), c13(c13),
        c20(c20), c21(c21), c22(c22), c23(c23),
        c30(c30), c31(c31), c32(c32), c33(c33) {
            auto zero = graph::zero<T, SAFE_MATH> ();
            x_cache = zero;
            y_cache = zero;
            z_cache = zero;
        }

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron density expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_density(graph::shared_leaf<T, SAFE_MATH> x,
                             graph::shared_leaf<T, SAFE_MATH> y,
                             graph::shared_leaf<T, SAFE_MATH> z) {
            set_cache(x, y, z);
            return ne_cache;
        }

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @param[in] index The species index.
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The ion density expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_density(const size_t index,
                        graph::shared_leaf<T, SAFE_MATH> x,
                        graph::shared_leaf<T, SAFE_MATH> y,
                        graph::shared_leaf<T, SAFE_MATH> z) {
            set_cache(x, y, z);
            return ni_cache;
        }

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron temperature expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_temperature(graph::shared_leaf<T, SAFE_MATH> x,
                                 graph::shared_leaf<T, SAFE_MATH> y,
                                 graph::shared_leaf<T, SAFE_MATH> z) {
            set_cache(x, y, z);
            return te_cache;
        }

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @param[in] index The species index.
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The ion temperature expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_temperature(const size_t index,
                            graph::shared_leaf<T, SAFE_MATH> x,
                            graph::shared_leaf<T, SAFE_MATH> y,
                            graph::shared_leaf<T, SAFE_MATH> z) {
            set_cache(x, y, z);
            return ti_cache;
        }

//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_magnetic_field(graph::shared_leaf<T, SAFE_MATH> x,
                           graph::shared_leaf<T, SAFE_MATH> y,
                           graph::shared_leaf<T, SAFE_MATH> z) {
            set_cache(x, y, z);
            return b_cache;
        }

//------------------------------------------------------------------------------
///  @brief Get the characteristic field.
///
///  Use the value at the y intercept.
///
///  @returns The characteristic field.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_characteristic_field() final {
            auto x_axis = graph::variable<T, SAFE_MATH> (1, "x");
            auto y_axis = graph::variable<T, SAFE_MATH> (1, "y");
            auto z_axis = graph::variable<T, SAFE_MATH> (1, "z");
            x_axis->set(static_cast<T> (1.7));
            y_axis->set(static_cast<T> (0.0));
            z_axis->set(static_cast<T> (0.0));
            auto b_vec = get_magnetic_field(x_axis, y_axis, z_axis);
            auto b_mod = b_vec->length();

            graph::input_nodes<T, SAFE_MATH> inputs {
                graph::variable_cast(x_axis),
                graph::variable_cast(y_axis),
                graph::variable_cast(z_axis)
            };

            workflow::manager<T, SAFE_MATH> work(0);
            solver::newton(work, {
                x_axis, z_axis
            }, inputs, psi_norm_cache, static_cast<T> (1.0E-30), 1000, static_cast<T> (0.1));
            work.add_item(inputs, {b_mod}, {}, "bmod_at_axis");
            work.compile();
            work.run();

            T result;
            work.copy_to_host(b_mod, &result);

            return graph::constant<T, SAFE_MATH> (result);
        }
    };

//------------------------------------------------------------------------------
///  @brief Convenience function to build an EFIT equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] spline_file File name of contains the spline functions.
///  @returns A constructed EFIT equilibrium.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared<T, SAFE_MATH> make_efit(const std::string &spline_file) {
        int ncid;
        sync.lock();
        nc_open(spline_file.c_str(), NC_NOWRITE, &ncid);

//  Load scalar quantities.
        int varid;

        double rmin_value;
        nc_inq_varid(ncid, "rmin", &varid);
        nc_get_var(ncid, varid, &rmin_value);

        double dr_value;
        nc_inq_varid(ncid, "dr", &varid);
        nc_get_var(ncid, varid, &dr_value);

        double zmin_value;
        nc_inq_varid(ncid, "zmin", &varid);
        nc_get_var(ncid, varid, &zmin_value);

        double dz_value;
        nc_inq_varid(ncid, "dz", &varid);
        nc_get_var(ncid, varid, &dz_value);

        double psimin_value;
        nc_inq_varid(ncid, "psimin", &varid);
        nc_get_var(ncid, varid, &psimin_value);

        double dpsi_value;
        nc_inq_varid(ncid, "dpsi", &varid);
        nc_get_var(ncid, varid, &dpsi_value);

        double pres_scale_value;
        nc_inq_varid(ncid, "pres_scale", &varid);
        nc_get_var(ncid, varid, &pres_scale_value);

        double ne_scale_value;
        nc_inq_varid(ncid, "ne_scale", &varid);
        nc_get_var(ncid, varid, &ne_scale_value);

        double te_scale_value;
        nc_inq_varid(ncid, "te_scale", &varid);
        nc_get_var(ncid, varid, &te_scale_value);

//  Load 1D quantities.
        int dimid;

        size_t numr;
        nc_inq_dimid(ncid, "numr", &dimid);
        nc_inq_dimlen(ncid, dimid, &numr);

        size_t numpsi;
        nc_inq_dimid(ncid, "numpsi", &dimid);
        nc_inq_dimlen(ncid, dimid, &numpsi);

        std::vector<double> fpol_c0_buffer(numr);
        std::vector<double> fpol_c1_buffer(numr);
        std::vector<double> fpol_c2_buffer(numr);
        std::vector<double> fpol_c3_buffer(numr);

        nc_inq_varid(ncid, "fpol_c0", &varid);
        nc_get_var(ncid, varid, fpol_c0_buffer.data());
        nc_inq_varid(ncid, "fpol_c1", &varid);
        nc_get_var(ncid, varid, fpol_c1_buffer.data());
        nc_inq_varid(ncid, "fpol_c2", &varid);
        nc_get_var(ncid, varid, fpol_c2_buffer.data());
        nc_inq_varid(ncid, "fpol_c3", &varid);
        nc_get_var(ncid, varid, fpol_c3_buffer.data());

//  Load psi grids.
        size_t numz;
        nc_inq_dimid(ncid, "numz", &dimid);
        nc_inq_dimlen(ncid, dimid, &numz);
        
        std::vector<double> psi_c00_buffer(numz*numr);
        std::vector<double> psi_c01_buffer(numz*numr);
        std::vector<double> psi_c02_buffer(numz*numr);
        std::vector<double> psi_c03_buffer(numz*numr);
        std::vector<double> psi_c10_buffer(numz*numr);
        std::vector<double> psi_c11_buffer(numz*numr);
        std::vector<double> psi_c12_buffer(numz*numr);
        std::vector<double> psi_c13_buffer(numz*numr);
        std::vector<double> psi_c20_buffer(numz*numr);
        std::vector<double> psi_c21_buffer(numz*numr);
        std::vector<double> psi_c22_buffer(numz*numr);
        std::vector<double> psi_c23_buffer(numz*numr);
        std::vector<double> psi_c30_buffer(numz*numr);
        std::vector<double> psi_c31_buffer(numz*numr);
        std::vector<double> psi_c32_buffer(numz*numr);
        std::vector<double> psi_c33_buffer(numz*numr);

        nc_inq_varid(ncid, "psi_c00", &varid);
        nc_get_var(ncid, varid, psi_c00_buffer.data());
        nc_inq_varid(ncid, "psi_c01", &varid);
        nc_get_var(ncid, varid, psi_c01_buffer.data());
        nc_inq_varid(ncid, "psi_c02", &varid);
        nc_get_var(ncid, varid, psi_c02_buffer.data());
        nc_inq_varid(ncid, "psi_c03", &varid);
        nc_get_var(ncid, varid, psi_c03_buffer.data());
        nc_inq_varid(ncid, "psi_c10", &varid);
        nc_get_var(ncid, varid, psi_c10_buffer.data());
        nc_inq_varid(ncid, "psi_c11", &varid);
        nc_get_var(ncid, varid, psi_c11_buffer.data());
        nc_inq_varid(ncid, "psi_c12", &varid);
        nc_get_var(ncid, varid, psi_c12_buffer.data());
        nc_inq_varid(ncid, "psi_c13", &varid);
        nc_get_var(ncid, varid, psi_c13_buffer.data());
        nc_inq_varid(ncid, "psi_c20", &varid);
        nc_get_var(ncid, varid, psi_c20_buffer.data());
        nc_inq_varid(ncid, "psi_c21", &varid);
        nc_get_var(ncid, varid, psi_c21_buffer.data());
        nc_inq_varid(ncid, "psi_c22", &varid);
        nc_get_var(ncid, varid, psi_c22_buffer.data());
        nc_inq_varid(ncid, "psi_c23", &varid);
        nc_get_var(ncid, varid, psi_c23_buffer.data());
        nc_inq_varid(ncid, "psi_c30", &varid);
        nc_get_var(ncid, varid, psi_c30_buffer.data());
        nc_inq_varid(ncid, "psi_c31", &varid);
        nc_get_var(ncid, varid, psi_c31_buffer.data());
        nc_inq_varid(ncid, "psi_c32", &varid);
        nc_get_var(ncid, varid, psi_c32_buffer.data());
        nc_inq_varid(ncid, "psi_c33", &varid);
        nc_get_var(ncid, varid, psi_c33_buffer.data());

        std::vector<double> pressure_c0_buffer(numpsi);
        std::vector<double> pressure_c1_buffer(numpsi);
        std::vector<double> pressure_c2_buffer(numpsi);
        std::vector<double> pressure_c3_buffer(numpsi);

        nc_inq_varid(ncid, "pressure_c0", &varid);
        nc_get_var(ncid, varid, pressure_c0_buffer.data());
        nc_inq_varid(ncid, "pressure_c1", &varid);
        nc_get_var(ncid, varid, pressure_c1_buffer.data());
        nc_inq_varid(ncid, "pressure_c2", &varid);
        nc_get_var(ncid, varid, pressure_c2_buffer.data());
        nc_inq_varid(ncid, "pressure_c3", &varid);
        nc_get_var(ncid, varid, pressure_c3_buffer.data());

        std::vector<double> te_c0_buffer(numpsi);
        std::vector<double> te_c1_buffer(numpsi);
        std::vector<double> te_c2_buffer(numpsi);
        std::vector<double> te_c3_buffer(numpsi);

        nc_inq_varid(ncid, "te_c0", &varid);
        nc_get_var(ncid, varid, te_c0_buffer.data());
        nc_inq_varid(ncid, "te_c1", &varid);
        nc_get_var(ncid, varid, te_c1_buffer.data());
        nc_inq_varid(ncid, "te_c2", &varid);
        nc_get_var(ncid, varid, te_c2_buffer.data());
        nc_inq_varid(ncid, "te_c3", &varid);
        nc_get_var(ncid, varid, te_c3_buffer.data());

        std::vector<double> ne_c0_buffer(numpsi);
        std::vector<double> ne_c1_buffer(numpsi);
        std::vector<double> ne_c2_buffer(numpsi);
        std::vector<double> ne_c3_buffer(numpsi);

        nc_inq_varid(ncid, "ne_c0", &varid);
        nc_get_var(ncid, varid, ne_c0_buffer.data());
        nc_inq_varid(ncid, "ne_c1", &varid);
        nc_get_var(ncid, varid, ne_c1_buffer.data());
        nc_inq_varid(ncid, "ne_c2", &varid);
        nc_get_var(ncid, varid, ne_c2_buffer.data());
        nc_inq_varid(ncid, "ne_c3", &varid);
        nc_get_var(ncid, varid, ne_c3_buffer.data());
                    
        nc_close(ncid);
        sync.unlock();

        auto rmin = graph::constant<T, SAFE_MATH> (static_cast<T> (rmin_value));
        auto dr = graph::constant<T, SAFE_MATH> (static_cast<T> (dr_value));
        auto zmin = graph::constant<T, SAFE_MATH> (static_cast<T> (zmin_value));
        auto dz = graph::constant<T, SAFE_MATH> (static_cast<T> (dz_value));
        auto psimin = graph::constant<T, SAFE_MATH> (static_cast<T> (psimin_value));
        auto dpsi = graph::constant<T, SAFE_MATH> (static_cast<T> (dpsi_value));
        auto pres_scale = graph::constant<T, SAFE_MATH> (static_cast<T> (pres_scale_value));
        auto ne_scale = graph::constant<T, SAFE_MATH> (static_cast<T> (ne_scale_value));
        auto te_scale = graph::constant<T, SAFE_MATH> (static_cast<T> (te_scale_value));

        const auto fpol_c0 = backend::buffer(std::vector<T> (fpol_c0_buffer.begin(), fpol_c0_buffer.end()));
        const auto fpol_c1 = backend::buffer(std::vector<T> (fpol_c1_buffer.begin(), fpol_c1_buffer.end()));
        const auto fpol_c2 = backend::buffer(std::vector<T> (fpol_c2_buffer.begin(), fpol_c2_buffer.end()));
        const auto fpol_c3 = backend::buffer(std::vector<T> (fpol_c3_buffer.begin(), fpol_c3_buffer.end()));

        const auto c00 = backend::buffer(std::vector<T> (psi_c00_buffer.begin(), psi_c00_buffer.end()));
        const auto c01 = backend::buffer(std::vector<T> (psi_c01_buffer.begin(), psi_c01_buffer.end()));
        const auto c02 = backend::buffer(std::vector<T> (psi_c02_buffer.begin(), psi_c02_buffer.end()));
        const auto c03 = backend::buffer(std::vector<T> (psi_c03_buffer.begin(), psi_c03_buffer.end()));
        const auto c10 = backend::buffer(std::vector<T> (psi_c10_buffer.begin(), psi_c10_buffer.end()));
        const auto c11 = backend::buffer(std::vector<T> (psi_c11_buffer.begin(), psi_c11_buffer.end()));
        const auto c12 = backend::buffer(std::vector<T> (psi_c12_buffer.begin(), psi_c12_buffer.end()));
        const auto c13 = backend::buffer(std::vector<T> (psi_c13_buffer.begin(), psi_c13_buffer.end()));
        const auto c20 = backend::buffer(std::vector<T> (psi_c20_buffer.begin(), psi_c20_buffer.end()));
        const auto c21 = backend::buffer(std::vector<T> (psi_c21_buffer.begin(), psi_c21_buffer.end()));
        const auto c22 = backend::buffer(std::vector<T> (psi_c22_buffer.begin(), psi_c22_buffer.end()));
        const auto c23 = backend::buffer(std::vector<T> (psi_c23_buffer.begin(), psi_c23_buffer.end()));
        const auto c30 = backend::buffer(std::vector<T> (psi_c30_buffer.begin(), psi_c30_buffer.end()));
        const auto c31 = backend::buffer(std::vector<T> (psi_c31_buffer.begin(), psi_c31_buffer.end()));
        const auto c32 = backend::buffer(std::vector<T> (psi_c32_buffer.begin(), psi_c32_buffer.end()));
        const auto c33 = backend::buffer(std::vector<T> (psi_c33_buffer.begin(), psi_c33_buffer.end()));

        const auto pres_c0 = backend::buffer(std::vector<T> (pressure_c0_buffer.begin(), pressure_c0_buffer.end()));
        const auto pres_c1 = backend::buffer(std::vector<T> (pressure_c1_buffer.begin(), pressure_c1_buffer.end()));
        const auto pres_c2 = backend::buffer(std::vector<T> (pressure_c2_buffer.begin(), pressure_c2_buffer.end()));
        const auto pres_c3 = backend::buffer(std::vector<T> (pressure_c3_buffer.begin(), pressure_c3_buffer.end()));

        const auto te_c0 = backend::buffer(std::vector<T> (te_c0_buffer.begin(), te_c0_buffer.end()));
        const auto te_c1 = backend::buffer(std::vector<T> (te_c1_buffer.begin(), te_c1_buffer.end()));
        const auto te_c2 = backend::buffer(std::vector<T> (te_c2_buffer.begin(), te_c2_buffer.end()));
        const auto te_c3 = backend::buffer(std::vector<T> (te_c3_buffer.begin(), te_c3_buffer.end()));

        const auto ne_c0 = backend::buffer(std::vector<T> (ne_c0_buffer.begin(), ne_c0_buffer.end()));
        const auto ne_c1 = backend::buffer(std::vector<T> (ne_c1_buffer.begin(), ne_c1_buffer.end()));
        const auto ne_c2 = backend::buffer(std::vector<T> (ne_c2_buffer.begin(), ne_c2_buffer.end()));
        const auto ne_c3 = backend::buffer(std::vector<T> (ne_c3_buffer.begin(), ne_c3_buffer.end()));

        return std::make_shared<efit<T, SAFE_MATH>> (psimin, dpsi,
                                                     te_c0, te_c1, te_c2, te_c3, te_scale,
                                                     ne_c0, ne_c1, ne_c2, ne_c3, ne_scale,
                                                     pres_c0, pres_c1, pres_c2, pres_c3, pres_scale,
                                                     rmin, dr, zmin, dz,
                                                     fpol_c0, fpol_c1, fpol_c2, fpol_c3, numz,
                                                     c00, c01, c02, c03,
                                                     c10, c11, c12, c13,
                                                     c20, c21, c22, c23,
                                                     c30, c31, c32, c33);
    }

//******************************************************************************
//  3D VMEC equilibrium.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief 3D VMEC equilibrium.
///
///  This takes a Cublic spline interpolations of the vmec quantities.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class vmec final : public generic<T, SAFE_MATH> {
    private:
///  Minimum s on the half grid.
        graph::shared_leaf<T, SAFE_MATH> sminh;
///  Minimum s on the full grid.
        graph::shared_leaf<T, SAFE_MATH> sminf;
///  Change in s grid.
        graph::shared_leaf<T, SAFE_MATH> ds;
///  Sign of the jacobian.
        graph::shared_leaf<T, SAFE_MATH> signj;

//  Poloidal flux coefficients.
///  Poloidal flux c0.
        const backend::buffer<T> chi_c0;
///  Poloidal flux c1.
        const backend::buffer<T> chi_c1;
///  Poloidal flux c2.
        const backend::buffer<T> chi_c2;
///  Poloidal flux c3.
        const backend::buffer<T> chi_c3;

//  Toroidal flux coefficients.
        graph::shared_leaf<T, SAFE_MATH> dphi;

//  Radial coefficients.
///  rmnc c0.
        const std::vector<backend::buffer<T>> rmnc_c0;
///  rmnc c1.
        const std::vector<backend::buffer<T>> rmnc_c1;
///  rmnc c2.
        const std::vector<backend::buffer<T>> rmnc_c2;
///  rmnc c3.
        const std::vector<backend::buffer<T>> rmnc_c3;

//  Vertical coefficients.
///  zmns c0.
        const std::vector<backend::buffer<T>> zmns_c0;
///  zmns c1.
        const std::vector<backend::buffer<T>> zmns_c1;
///  zmns c2.
        const std::vector<backend::buffer<T>> zmns_c2;
///  zmns c3.
        const std::vector<backend::buffer<T>> zmns_c3;
        
//  Lambda coefficients.
///  lmns c0.
        const std::vector<backend::buffer<T>> lmns_c0;
///  lmns c1.
        const std::vector<backend::buffer<T>> lmns_c1;
///  lmns c2.
        const std::vector<backend::buffer<T>> lmns_c2;
///  lmns c3.
        const std::vector<backend::buffer<T>> lmns_c3;

///  Poloidal mode numbers.
        const backend::buffer<T> xm;
///  Toroidal mode numbers.
        const backend::buffer<T> xn;

//  Cached values.
///  s position cache.
        graph::shared_leaf<T, SAFE_MATH> s_cache;
///  u position cache.
        graph::shared_leaf<T, SAFE_MATH> u_cache;
///  v position cache.
        graph::shared_leaf<T, SAFE_MATH> v_cache;
///  x position cache.
        graph::shared_leaf<T, SAFE_MATH> x_cache;
///  y position cache.
        graph::shared_leaf<T, SAFE_MATH> y_cache;
///  z position cache.
        graph::shared_leaf<T, SAFE_MATH> z_cache;

///  Contravaraint s basis cache.
        graph::shared_vector<T, SAFE_MATH> esups_cache;
///  Contravaraint u basis cache.
        graph::shared_vector<T, SAFE_MATH> esupu_cache;
///  Contravaraint v basis cache.
        graph::shared_vector<T, SAFE_MATH> esupv_cache;
 
///  Contravaraint v basis cache.
        graph::shared_vector<T, SAFE_MATH> bvec_cache;

//------------------------------------------------------------------------------
///  @brief Get the covariant basis vectors in the s direction.
///
///  @param[in] r Radial posirtion.
///  @param[in] z Vertical position.
///  @returns The covariant basis vectors.
//------------------------------------------------------------------------------
        graph::shared_vector<T, SAFE_MATH>
        get_esubs(graph::shared_leaf<T, SAFE_MATH> r,
                  graph::shared_leaf<T, SAFE_MATH> z) {
            auto cosv = graph::cos(v_cache);
            auto sinv = graph::sin(v_cache);
            auto one = graph::one<T, SAFE_MATH> ();
            auto zero = graph::zero<T, SAFE_MATH> ();

            auto m = graph::matrix(graph::vector(cosv, -sinv, zero),
                                   graph::vector(sinv, cosv,  zero),
                                   graph::vector(zero, zero,  one ));
            return m->dot(graph::vector(r->df(s_cache),
                                        zero,
                                        z->df(s_cache)));
        }

//------------------------------------------------------------------------------
///  @brief Get the covariant basis vectors in the u direction.
///
///  @param[in] r Radial posirtion.
///  @param[in] z Vertical position.
///  @returns The covariant basis vectors.
//------------------------------------------------------------------------------
        graph::shared_vector<T, SAFE_MATH>
        get_esubu(graph::shared_leaf<T, SAFE_MATH> r,
                  graph::shared_leaf<T, SAFE_MATH> z) {
            auto cosv = graph::cos(v_cache);
            auto sinv = graph::sin(v_cache);
            auto one = graph::one<T, SAFE_MATH> ();
            auto zero = graph::zero<T, SAFE_MATH> ();
                        
            auto m = graph::matrix(graph::vector(cosv, -sinv, zero),
                                   graph::vector(sinv, cosv,  zero),
                                   graph::vector(zero, zero,  one ));
            return m->dot(graph::vector(r->df(u_cache),
                                        zero,
                                        z->df(u_cache)));
        }

//------------------------------------------------------------------------------
///  @brief Get the covariant basis vectors in the u direction.
///
///  @param[in] r Radial posirtion.
///  @param[in] z Vertical position.
///  @returns The covariant basis vectors.
//------------------------------------------------------------------------------
        graph::shared_vector<T, SAFE_MATH>
        get_esubv(graph::shared_leaf<T, SAFE_MATH> r,
                  graph::shared_leaf<T, SAFE_MATH> z) {
            auto cosv = graph::cos(v_cache);
            auto sinv = graph::sin(v_cache);
            auto one = graph::one<T, SAFE_MATH> ();
            auto zero = graph::zero<T, SAFE_MATH> ();

            auto m = graph::matrix(graph::vector(cosv, -sinv, zero),
                                   graph::vector(sinv, cosv,  zero),
                                   graph::vector(zero, zero,  one ));
            return m->dot(graph::vector(r->df(v_cache),
                                        r,
                                        z->df(v_cache)));
        }

//------------------------------------------------------------------------------
///  @brief Get the Jacobian.
///
///  J = e_s.e_u✕e_v
///
///  @param[in] esub_s Covariant s basis.
///  @param[in] esub_u Covariant u basis.
///  @param[in] esub_v Covariant v basis.
///  @returns The jacobian.
//------------------------------------------------------------------------------
        graph::shared_leaf<T, SAFE_MATH>
        get_jacobian(graph::shared_vector<T, SAFE_MATH> esub_s,
                     graph::shared_vector<T, SAFE_MATH> esub_u,
                     graph::shared_vector<T, SAFE_MATH> esub_v) {
            return esub_s->dot(esub_u->cross(esub_v));
        }

//------------------------------------------------------------------------------
///  @brief Get the poloidal flux.
///
///  @param[in] s_norm Normalized S position.
///  @returns χ(s,u,v)
//------------------------------------------------------------------------------
        graph::shared_leaf<T, SAFE_MATH>
        get_chi(graph::shared_leaf<T, SAFE_MATH> s_norm) {
            auto c0_temp = graph::piecewise_1D(chi_c0, s_norm);
            auto c1_temp = graph::piecewise_1D(chi_c1, s_norm);
            auto c2_temp = graph::piecewise_1D(chi_c2, s_norm);
            auto c3_temp = graph::piecewise_1D(chi_c3, s_norm);

            return c0_temp +
                   c1_temp*s_norm +
                   c2_temp*s_norm*s_norm +
                   c3_temp*s_norm*s_norm*s_norm;
        }

//------------------------------------------------------------------------------
///  @brief Get the toroidal flux.
///
///  @param[in] s S position.
///  @returns φ(s,u,v)
//------------------------------------------------------------------------------
        graph::shared_leaf<T, SAFE_MATH>
        get_phi(graph::shared_leaf<T, SAFE_MATH> s) {
            return signj*dphi*s;
        }

//------------------------------------------------------------------------------
///  @brief Set cache values.
///
///  Sets the cached values if s, u, and v do not match.
///
///  @param[in] s S position.
///  @param[in] u U position.
///  @param[in] v V position.
//------------------------------------------------------------------------------
        void set_cache(graph::shared_leaf<T, SAFE_MATH> s,
                       graph::shared_leaf<T, SAFE_MATH> u,
                       graph::shared_leaf<T, SAFE_MATH> v) {
            if (!s->is_match(s_cache) ||
                !u->is_match(u_cache) ||
                !v->is_match(v_cache)) {
                s_cache = s;
                u_cache = u;
                v_cache = v;
                
                auto s_norm_f = (s - sminf)/ds;
                auto s_norm_h = (s - sminh)/ds;

                auto zero = graph::zero<T, SAFE_MATH> ();
                auto r = zero;
                auto z = zero;
                auto l = zero;

                for (size_t i = 0, ie = xm.size(); i < ie; i++) {
                    auto rmnc_c0_temp = graph::piecewise_1D(rmnc_c0[i], s_norm_f);
                    auto rmnc_c1_temp = graph::piecewise_1D(rmnc_c1[i], s_norm_f);
                    auto rmnc_c2_temp = graph::piecewise_1D(rmnc_c2[i], s_norm_f);
                    auto rmnc_c3_temp = graph::piecewise_1D(rmnc_c3[i], s_norm_f);

                    auto zmns_c0_temp = graph::piecewise_1D(zmns_c0[i], s_norm_f);
                    auto zmns_c1_temp = graph::piecewise_1D(zmns_c1[i], s_norm_f);
                    auto zmns_c2_temp = graph::piecewise_1D(zmns_c2[i], s_norm_f);
                    auto zmns_c3_temp = graph::piecewise_1D(zmns_c3[i], s_norm_f);

                    auto lmns_c0_temp = graph::piecewise_1D(lmns_c0[i], s_norm_h);
                    auto lmns_c1_temp = graph::piecewise_1D(lmns_c1[i], s_norm_h);
                    auto lmns_c2_temp = graph::piecewise_1D(lmns_c2[i], s_norm_h);
                    auto lmns_c3_temp = graph::piecewise_1D(lmns_c3[i], s_norm_h);

                    auto rmnc = rmnc_c0_temp
                              + rmnc_c1_temp*s_norm_f
                              + rmnc_c2_temp*s_norm_f*s_norm_f
                              + rmnc_c3_temp*s_norm_f*s_norm_f*s_norm_f;
                    auto zmns = zmns_c0_temp
                              + zmns_c1_temp*s_norm_f
                              + zmns_c2_temp*s_norm_f*s_norm_f
                              + zmns_c3_temp*s_norm_f*s_norm_f*s_norm_f;
                    auto lmns = lmns_c0_temp
                              + lmns_c1_temp*s_norm_h
                              + lmns_c2_temp*s_norm_h*s_norm_h
                              + lmns_c3_temp*s_norm_h*s_norm_h*s_norm_h;

                    auto m = graph::constant<T, SAFE_MATH> (xm[i]);
                    auto n = graph::constant<T, SAFE_MATH> (xn[i]);

                    auto sinmn = graph::sin(m*u - n*v);

                    r = r + rmnc*graph::cos(m*u - n*v);
                    z = z + zmns*sinmn;
                    l = l + lmns*sinmn;
                }

                x_cache = r*graph::cos(v);
                y_cache = r*graph::sin(v);
                z_cache = z;

                auto esubs = get_esubs(r, z);
                auto esubu = get_esubu(r, z);
                auto esubv = get_esubv(r, z);

                auto jacobian = get_jacobian(esubs, esubu, esubv);

                esups_cache = esubu->cross(esubv)/jacobian;
                esupu_cache = esubv->cross(esubs)/jacobian;
                esupv_cache = esubs->cross(esubu)/jacobian;

                auto phip = get_phi(s)->df(s);
                auto jbsupu = get_chi(s_norm_f)->df(s) - phip*l->df(v);
                auto jbsupv = phip*(1.0 + l->df(u));
                bvec_cache = (jbsupu*esubu + jbsupv*esubv)/jacobian;
            }
        }

//------------------------------------------------------------------------------
///  @brief Get the profile function.
///
///  @param[in] s S posiiton.
///  @returns The profile function.
//------------------------------------------------------------------------------
        graph::shared_leaf<T, SAFE_MATH>
        get_profile(graph::shared_leaf<T, SAFE_MATH> s) {
            return graph::pow((1.0 - graph::pow(graph::sqrt(s*s), 1.5)), 2.0);
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a EFIT equilibrium.
///
///  @param[in] sminh   Minimum s on the half grid.
///  @param[in] sminf   Minimum s on the full grid.
///  @param[in] ds      Change in s grid.
///  @param[in] dphi    Change in torodial flux.
///  @param[in] signj   Sign of the jacobian.
///  @param[in] chi_c0  Poloidal flux c0.
///  @param[in] chi_c1  Poloidal flux c1.
///  @param[in] chi_c2  Poloidal flux c2.
///  @param[in] chi_c3  Poloidal flux c3.
///  @param[in] rmnc_c0 rmnc c0.
///  @param[in] rmnc_c1 rmnc c1.
///  @param[in] rmnc_c2 rmnc c2.
///  @param[in] rmnc_c3 rmnc c3.
///  @param[in] zmns_c0 zmns c0.
///  @param[in] zmns_c1 zmns c1.
///  @param[in] zmns_c2 zmns c2.
///  @param[in] zmns_c3 zmns c3.
///  @param[in] lmns_c0 lmns c0.
///  @param[in] lmns_c1 lmns c1.
///  @param[in] lmns_c2 lmns c2.
///  @param[in] lmns_c3 lmns c3.
///  @param[in] xm      Poloidal mode numbers.
///  @param[in] xn      Toroidal mode numbers.
//------------------------------------------------------------------------------
        vmec(graph::shared_leaf<T, SAFE_MATH> sminh,
             graph::shared_leaf<T, SAFE_MATH> sminf,
             graph::shared_leaf<T, SAFE_MATH> ds,
             graph::shared_leaf<T, SAFE_MATH> dphi,
             graph::shared_leaf<T, SAFE_MATH> signj,
             const backend::buffer<T> chi_c0,
             const backend::buffer<T> chi_c1,
             const backend::buffer<T> chi_c2,
             const backend::buffer<T> chi_c3,
             const std::vector<backend::buffer<T>> rmnc_c0,
             const std::vector<backend::buffer<T>> rmnc_c1,
             const std::vector<backend::buffer<T>> rmnc_c2,
             const std::vector<backend::buffer<T>> rmnc_c3,
             const std::vector<backend::buffer<T>> zmns_c0,
             const std::vector<backend::buffer<T>> zmns_c1,
             const std::vector<backend::buffer<T>> zmns_c2,
             const std::vector<backend::buffer<T>> zmns_c3,
             const std::vector<backend::buffer<T>> lmns_c0,
             const std::vector<backend::buffer<T>> lmns_c1,
             const std::vector<backend::buffer<T>> lmns_c2,
             const std::vector<backend::buffer<T>> lmns_c3,
             const backend::buffer<T> xm,
             const backend::buffer<T> xn) :
        generic<T, SAFE_MATH> ({3.34449469E-27} ,{1}),
        sminh(sminh), sminf(sminf), ds(ds), dphi(dphi), signj(signj),
        chi_c0(chi_c0), chi_c1(chi_c1), chi_c2(chi_c2), chi_c3(chi_c3),
        rmnc_c0(rmnc_c0), rmnc_c1(rmnc_c1), rmnc_c2(rmnc_c2), rmnc_c3(rmnc_c3),
        zmns_c0(zmns_c0), zmns_c1(zmns_c1), zmns_c2(zmns_c2), zmns_c3(zmns_c3),
        lmns_c0(lmns_c0), lmns_c1(lmns_c1), lmns_c2(lmns_c2), lmns_c3(lmns_c3),
        xm(xm), xn(xn) {
            auto zero = graph::zero<T, SAFE_MATH> ();
            s_cache = zero;
            u_cache = zero;
            v_cache = zero;
        }

//------------------------------------------------------------------------------
///  @brief Get the contravariant basis vector in the S direction.
///
///  @param[in] s S posiiton.
///  @param[in] u U position.
///  @param[in] v V position.
///  @returns The contravaraiant basis vector in s.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_esup1(graph::shared_leaf<T, SAFE_MATH> s,
                  graph::shared_leaf<T, SAFE_MATH> u,
                  graph::shared_leaf<T, SAFE_MATH> v) {
            set_cache(s, u, v);
            return esups_cache;
        }

//------------------------------------------------------------------------------
///  @brief Get the contravariant basis vector in the U direction.
///
///  @param[in] s S posiiton.
///  @param[in] u U position.
///  @param[in] v V position.
///  @returns The contravaraiant basis vector in u.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_esup2(graph::shared_leaf<T, SAFE_MATH> s,
                  graph::shared_leaf<T, SAFE_MATH> u,
                  graph::shared_leaf<T, SAFE_MATH> v) {
            set_cache(s, u, v);
            return esupu_cache;
        }

//------------------------------------------------------------------------------
///  @brief Get the contravariant basis vector in the V direction.
///
///  @param[in] s S posiiton.
///  @param[in] u U position.
///  @param[in] v V position.
///  @returns The contravaraiant basis vector in v.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_esup3(graph::shared_leaf<T, SAFE_MATH> s,
                  graph::shared_leaf<T, SAFE_MATH> u,
                  graph::shared_leaf<T, SAFE_MATH> v) {
            set_cache(s, u, v);
            return esupv_cache;
        }

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @param[in] s S posiiton.
///  @param[in] u U position.
///  @param[in] v V position.
///  @returns The electron density expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_density(graph::shared_leaf<T, SAFE_MATH> s,
                             graph::shared_leaf<T, SAFE_MATH> u,
                             graph::shared_leaf<T, SAFE_MATH> v) {
            auto ne_scale = graph::constant<T, SAFE_MATH> (static_cast<T> (1.0E19));
            return ne_scale*get_profile(s);
        }

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @param[in] index The species index.
///  @param[in] s S posiiton.
///  @param[in] u U position.
///  @param[in] v V position.
///  @returns The ion density expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_density(const size_t index,
                        graph::shared_leaf<T, SAFE_MATH> s,
                        graph::shared_leaf<T, SAFE_MATH> u,
                        graph::shared_leaf<T, SAFE_MATH> v) {
            return get_electron_density(s, u, v);
        }

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @param[in] s S posiiton.
///  @param[in] u U position.
///  @param[in] v V position.
///  @returns The electron temperature expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_temperature(graph::shared_leaf<T, SAFE_MATH> s,
                                 graph::shared_leaf<T, SAFE_MATH> u,
                                 graph::shared_leaf<T, SAFE_MATH> v) {
            auto te_scale = graph::constant<T, SAFE_MATH> (static_cast<T> (1000.0));
            return te_scale*get_profile(s);
        }

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @param[in] index The species index.
///  @param[in] s S posiiton.
///  @param[in] u U position.
///  @param[in] v V position.
///  @returns The ion temperature expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_temperature(const size_t index,
                            graph::shared_leaf<T, SAFE_MATH> s,
                            graph::shared_leaf<T, SAFE_MATH> u,
                            graph::shared_leaf<T, SAFE_MATH> v) {
            return get_electron_temperature(s, u, v);
        }

//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @param[in] s S posiiton.
///  @param[in] u U position.
///  @param[in] v V position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_magnetic_field(graph::shared_leaf<T, SAFE_MATH> s,
                           graph::shared_leaf<T, SAFE_MATH> u,
                           graph::shared_leaf<T, SAFE_MATH> v) {
            set_cache(s, u, v);
            return bvec_cache;
        }

//------------------------------------------------------------------------------
///  @brief Get the characteristic field.
///
///  Use the value at the y intercept.
///
///  @returns The characteristic field.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_characteristic_field() final {
            auto s_axis = graph::zero<T, SAFE_MATH> ();
            auto u_axis = graph::zero<T, SAFE_MATH> ();
            auto v_axis = graph::zero<T, SAFE_MATH> ();
            auto b_vec = get_magnetic_field(s_axis, u_axis, v_axis);
            return b_vec->length();
        }

//------------------------------------------------------------------------------
///  @brief Get the x position.
///
///  @param[in] s S posiiton.
///  @param[in] u U position.
///  @param[in] v V position.
///  @returns The x position.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_x(graph::shared_leaf<T, SAFE_MATH> s,
              graph::shared_leaf<T, SAFE_MATH> u,
              graph::shared_leaf<T, SAFE_MATH> v) {
            set_cache(s, u, v);
            return x_cache;
        }

//------------------------------------------------------------------------------
///  @brief Get the y position.
///
///  @param[in] s S posiiton.
///  @param[in] u U position.
///  @param[in] v V position.
///  @returns The y position.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_y(graph::shared_leaf<T, SAFE_MATH> s,
              graph::shared_leaf<T, SAFE_MATH> u,
              graph::shared_leaf<T, SAFE_MATH> v) {
            set_cache(s, u, v);
            return y_cache;
        }

//------------------------------------------------------------------------------
///  @brief Get the z position.
///
///  @param[in] s S posiiton.
///  @param[in] u U position.
///  @param[in] v V position.
///  @returns The z position.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_z(graph::shared_leaf<T, SAFE_MATH> s,
              graph::shared_leaf<T, SAFE_MATH> u,
              graph::shared_leaf<T, SAFE_MATH> v) {
            set_cache(s, u, v);
            return z_cache;
        }
    };

//------------------------------------------------------------------------------
///  @brief Convenience function to build an VMEC equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] spline_file File name of contains the spline functions.
///  @returns A constructed VMEC equilibrium.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared<T, SAFE_MATH> make_vmec(const std::string &spline_file) {
        int ncid;
        sync.lock();
        nc_open(spline_file.c_str(), NC_NOWRITE, &ncid);

//  Load scalar quantities.
        int varid;

        double sminf_value;
        nc_inq_varid(ncid, "sminf", &varid);
        nc_get_var(ncid, varid, &sminf_value);

        double sminh_value;
        nc_inq_varid(ncid, "sminh", &varid);
        nc_get_var(ncid, varid, &sminh_value);

        double ds_value;
        nc_inq_varid(ncid, "ds", &varid);
        nc_get_var(ncid, varid, &ds_value);

        double dphi_value;
        nc_inq_varid(ncid, "dphi", &varid);
        nc_get_var(ncid, varid, &dphi_value);

        double signj_value;
        nc_inq_varid(ncid, "signj", &varid);
        nc_get_var(ncid, varid, &signj_value);

//  Load 1D quantities.
        int dimid;

        size_t numsf;
        nc_inq_dimid(ncid, "numsf", &dimid);
        nc_inq_dimlen(ncid, dimid, &numsf);

        std::vector<double> chi_c0_buffer(numsf);
        std::vector<double> chi_c1_buffer(numsf);
        std::vector<double> chi_c2_buffer(numsf);
        std::vector<double> chi_c3_buffer(numsf);

        nc_inq_varid(ncid, "chi_c0", &varid);
        nc_get_var(ncid, varid, chi_c0_buffer.data());
        nc_inq_varid(ncid, "chi_c1", &varid);
        nc_get_var(ncid, varid, chi_c1_buffer.data());
        nc_inq_varid(ncid, "chi_c2", &varid);
        nc_get_var(ncid, varid, chi_c2_buffer.data());
        nc_inq_varid(ncid, "chi_c3", &varid);
        nc_get_var(ncid, varid, chi_c3_buffer.data());

//  Load 2D quantities.
        size_t numsh;
        nc_inq_dimid(ncid, "numsh", &dimid);
        nc_inq_dimlen(ncid, dimid, &numsh);

        size_t nummn;
        nc_inq_dimid(ncid, "nummn", &dimid);
        nc_inq_dimlen(ncid, dimid, &nummn);

        std::vector<std::vector<double>> rmnc_c0_buffer(nummn, std::vector<double> (numsf));
        std::vector<std::vector<double>> rmnc_c1_buffer(nummn, std::vector<double> (numsf));
        std::vector<std::vector<double>> rmnc_c2_buffer(nummn, std::vector<double> (numsf));
        std::vector<std::vector<double>> rmnc_c3_buffer(nummn, std::vector<double> (numsf));

        nc_inq_varid(ncid, "rmnc_c0", &varid);
        for (size_t i = 0; i < nummn; i++) {
            const array<size_t, 2> start = {i, 0};
            const array<size_t, 2> count = {1, numsf};
            nc_get_vara(ncid, varid, start.data(), count.data(),
                        rmnc_c0_buffer[i].data());
        }
        nc_inq_varid(ncid, "rmnc_c1", &varid);
        for (size_t i = 0; i < nummn; i++) {
            const array<size_t, 2> start = {i, 0};
            const array<size_t, 2> count = {1, numsf};
            nc_get_vara(ncid, varid, start.data(), count.data(),
                        rmnc_c1_buffer[i].data());
        }
        nc_inq_varid(ncid, "rmnc_c2", &varid);
        for (size_t i = 0; i < nummn; i++) {
            const array<size_t, 2> start = {i, 0};
            const array<size_t, 2> count = {1, numsf};
            nc_get_vara(ncid, varid, start.data(), count.data(),
                        rmnc_c2_buffer[i].data());
        }
        nc_inq_varid(ncid, "rmnc_c3", &varid);
        for (size_t i = 0; i < nummn; i++) {
            const array<size_t, 2> start = {i, 0};
            const array<size_t, 2> count = {1, numsf};
            nc_get_vara(ncid, varid, start.data(), count.data(),
                        rmnc_c3_buffer[i].data());
        }

        std::vector<std::vector<double>> zmns_c0_buffer(nummn, std::vector<double> (numsf));
        std::vector<std::vector<double>> zmns_c1_buffer(nummn, std::vector<double> (numsf));
        std::vector<std::vector<double>> zmns_c2_buffer(nummn, std::vector<double> (numsf));
        std::vector<std::vector<double>> zmns_c3_buffer(nummn, std::vector<double> (numsf));

        nc_inq_varid(ncid, "zmns_c0", &varid);
        for (size_t i = 0; i < nummn; i++) {
            const array<size_t, 2> start = {i, 0};
            const array<size_t, 2> count = {1, numsf};
            nc_get_vara(ncid, varid, start.data(), count.data(),
                        zmns_c0_buffer[i].data());
        }
        nc_inq_varid(ncid, "zmns_c1", &varid);
        for (size_t i = 0; i < nummn; i++) {
            const array<size_t, 2> start = {i, 0};
            const array<size_t, 2> count = {1, numsf};
            nc_get_vara(ncid, varid, start.data(), count.data(),
                        zmns_c1_buffer[i].data());
        }
        nc_inq_varid(ncid, "zmns_c2", &varid);
        for (size_t i = 0; i < nummn; i++) {
            const array<size_t, 2> start = {i, 0};
            const array<size_t, 2> count = {1, numsf};
            nc_get_vara(ncid, varid, start.data(), count.data(),
                        zmns_c2_buffer[i].data());
        }
        nc_inq_varid(ncid, "zmns_c3", &varid);
        for (size_t i = 0; i < nummn; i++) {
            const array<size_t, 2> start = {i, 0};
            const array<size_t, 2> count = {1, numsf};
            nc_get_vara(ncid, varid, start.data(), count.data(),
                        zmns_c3_buffer[i].data());
        }

        std::vector<std::vector<double>> lmns_c0_buffer(nummn, std::vector<double> (numsh));
        std::vector<std::vector<double>> lmns_c1_buffer(nummn, std::vector<double> (numsh));
        std::vector<std::vector<double>> lmns_c2_buffer(nummn, std::vector<double> (numsh));
        std::vector<std::vector<double>> lmns_c3_buffer(nummn, std::vector<double> (numsh));

        nc_inq_varid(ncid, "lmns_c0", &varid);
        for (size_t i = 0; i < nummn; i++) {
            const array<size_t, 2> start = {i, 0};
            const array<size_t, 2> count = {1, numsh};
            nc_get_vara(ncid, varid, start.data(), count.data(),
                        lmns_c0_buffer[i].data());
        }
        nc_inq_varid(ncid, "lmns_c1", &varid);
        for (size_t i = 0; i < nummn; i++) {
            const array<size_t, 2> start = {i, 0};
            const array<size_t, 2> count = {1, numsh};
            nc_get_vara(ncid, varid, start.data(), count.data(),
                        lmns_c1_buffer[i].data());
        }
        nc_inq_varid(ncid, "lmns_c2", &varid);
        for (size_t i = 0; i < nummn; i++) {
            const array<size_t, 2> start = {i, 0};
            const array<size_t, 2> count = {1, numsh};
            nc_get_vara(ncid, varid, start.data(), count.data(),
                        lmns_c2_buffer[i].data());
        }
        nc_inq_varid(ncid, "lmns_c3", &varid);
        for (size_t i = 0; i < nummn; i++) {
            const array<size_t, 2> start = {i, 0};
            const array<size_t, 2> count = {1, numsh};
            nc_get_vara(ncid, varid, start.data(), count.data(),
                        lmns_c3_buffer[i].data());
        }

        std::vector<double> xm_buffer(nummn);
        nc_inq_varid(ncid, "xm", &varid);
        nc_get_var(ncid, varid, xm_buffer.data());

        std::vector<double> xn_buffer(nummn);
        nc_inq_varid(ncid, "xn", &varid);
        nc_get_var(ncid, varid, xn_buffer.data());

        nc_close(ncid);
        sync.unlock();

        auto sminf = graph::constant<T, SAFE_MATH> (static_cast<T> (sminf_value));
        auto sminh = graph::constant<T, SAFE_MATH> (static_cast<T> (sminh_value));
        auto ds = graph::constant<T, SAFE_MATH> (static_cast<T> (ds_value));
        auto dphi = graph::constant<T, SAFE_MATH> (static_cast<T> (dphi_value));
        auto signj = graph::constant<T, SAFE_MATH> (static_cast<T> (signj_value));

        const backend::buffer<T> chi_c0(std::vector<T> (chi_c0_buffer.begin(), chi_c0_buffer.end()));
        const backend::buffer<T> chi_c1(std::vector<T> (chi_c1_buffer.begin(), chi_c1_buffer.end()));
        const backend::buffer<T> chi_c2(std::vector<T> (chi_c2_buffer.begin(), chi_c2_buffer.end()));
        const backend::buffer<T> chi_c3(std::vector<T> (chi_c3_buffer.begin(), chi_c3_buffer.end()));

        std::vector<backend::buffer<T>> rmnc_c0(nummn);
        std::vector<backend::buffer<T>> rmnc_c1(nummn);
        std::vector<backend::buffer<T>> rmnc_c2(nummn);
        std::vector<backend::buffer<T>> rmnc_c3(nummn);

        std::vector<backend::buffer<T>> zmns_c0(nummn);
        std::vector<backend::buffer<T>> zmns_c1(nummn);
        std::vector<backend::buffer<T>> zmns_c2(nummn);
        std::vector<backend::buffer<T>> zmns_c3(nummn);

        std::vector<backend::buffer<T>> lmns_c0(nummn);
        std::vector<backend::buffer<T>> lmns_c1(nummn);
        std::vector<backend::buffer<T>> lmns_c2(nummn);
        std::vector<backend::buffer<T>> lmns_c3(nummn);
        
        for (size_t i = 0; i < nummn; i++) {
            rmnc_c0[i] = backend::buffer(std::vector<T> (rmnc_c0_buffer[i].begin(), rmnc_c0_buffer[i].end()));
            rmnc_c1[i] = backend::buffer(std::vector<T> (rmnc_c1_buffer[i].begin(), rmnc_c1_buffer[i].end()));
            rmnc_c2[i] = backend::buffer(std::vector<T> (rmnc_c2_buffer[i].begin(), rmnc_c2_buffer[i].end()));
            rmnc_c3[i] = backend::buffer(std::vector<T> (rmnc_c3_buffer[i].begin(), rmnc_c3_buffer[i].end()));

            zmns_c0[i] = backend::buffer(std::vector<T> (zmns_c0_buffer[i].begin(), zmns_c0_buffer[i].end()));
            zmns_c1[i] = backend::buffer(std::vector<T> (zmns_c1_buffer[i].begin(), zmns_c1_buffer[i].end()));
            zmns_c2[i] = backend::buffer(std::vector<T> (zmns_c2_buffer[i].begin(), zmns_c2_buffer[i].end()));
            zmns_c3[i] = backend::buffer(std::vector<T> (zmns_c3_buffer[i].begin(), zmns_c3_buffer[i].end()));

            lmns_c0[i] = backend::buffer(std::vector<T> (lmns_c0_buffer[i].begin(), lmns_c0_buffer[i].end()));
            lmns_c1[i] = backend::buffer(std::vector<T> (lmns_c1_buffer[i].begin(), lmns_c1_buffer[i].end()));
            lmns_c2[i] = backend::buffer(std::vector<T> (lmns_c2_buffer[i].begin(), lmns_c2_buffer[i].end()));
            lmns_c3[i] = backend::buffer(std::vector<T> (lmns_c3_buffer[i].begin(), lmns_c3_buffer[i].end()));
        }

        const backend::buffer<T> xm(std::vector<T> (xm_buffer.begin(), xm_buffer.end()));
        const backend::buffer<T> xn(std::vector<T> (xn_buffer.begin(), xn_buffer.end()));

        return std::make_shared<vmec<T, SAFE_MATH>> (sminh, sminf, ds, dphi, signj,
                                                     chi_c0, chi_c1, chi_c2, chi_c3,
                                                     rmnc_c0, rmnc_c1, rmnc_c2, rmnc_c3,
                                                     zmns_c0, zmns_c1, zmns_c2, zmns_c3,
                                                     lmns_c0, lmns_c1, lmns_c2, lmns_c3,
                                                     xm, xn);
    }
}

#endif /* equilibrium_h */
