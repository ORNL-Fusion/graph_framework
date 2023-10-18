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
///  @params[in] masses  Vector of ion masses.
///  @params[in] charges Vector of ion charges.
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
///  @params[in] index The species index.
///  @returns The mass for the ion at the index.
//------------------------------------------------------------------------------
        T get_ion_mass(const size_t index) const {
            return ion_masses.at(index);
        }

//------------------------------------------------------------------------------
///  @brief Get the charge for an ion species.
///
///  @params[in] index The species index.
///  @returns The number of ion species.
//------------------------------------------------------------------------------
        uint8_t get_ion_charge(const size_t index) const {
            return ion_charges.at(index);
        }

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_density(graph::shared_leaf<T, SAFE_MATH> x,
                             graph::shared_leaf<T, SAFE_MATH> y,
                             graph::shared_leaf<T, SAFE_MATH> z) = 0;

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @params[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_density(const size_t index,
                        graph::shared_leaf<T, SAFE_MATH> x,
                        graph::shared_leaf<T, SAFE_MATH> y,
                        graph::shared_leaf<T, SAFE_MATH> z) = 0;

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_temperature(graph::shared_leaf<T, SAFE_MATH> x,
                                 graph::shared_leaf<T, SAFE_MATH> y,
                                 graph::shared_leaf<T, SAFE_MATH> z) = 0;

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @params[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_temperature(const size_t index,
                            graph::shared_leaf<T, SAFE_MATH> x,
                            graph::shared_leaf<T, SAFE_MATH> y,
                            graph::shared_leaf<T, SAFE_MATH> z) = 0;
        
//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_magnetic_field(graph::shared_leaf<T, SAFE_MATH> x,
                           graph::shared_leaf<T, SAFE_MATH> y,
                           graph::shared_leaf<T, SAFE_MATH> z) = 0;
    };

///  Convenience type alias for shared equilibria.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared = std::shared_ptr<generic<T, SAFE_MATH>>;

//******************************************************************************
//  No Magnetic equilibrium.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Uniform density with varying magnetic field equilibrium.
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
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
///  @params[in] index The species index.
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
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
///  @params[in] index The species index.
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_magnetic_field(graph::shared_leaf<T, SAFE_MATH> x,
                           graph::shared_leaf<T, SAFE_MATH> y,
                           graph::shared_leaf<T, SAFE_MATH> z) final {
            auto zero = graph::zero<T, SAFE_MATH> ();
            return graph::vector(zero, zero, zero);
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
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
///  @params[in] index The species index.
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
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
///  @params[in] index The species index.
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_magnetic_field(graph::shared_leaf<T, SAFE_MATH> x,
                           graph::shared_leaf<T, SAFE_MATH> y,
                           graph::shared_leaf<T, SAFE_MATH> z) final {
            auto zero = graph::zero<T, SAFE_MATH> ();
            return graph::vector(zero, zero,
                                 graph::constant<T, SAFE_MATH> (static_cast<T> (0.1))*x +
                                 graph::one<T, SAFE_MATH> ());
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
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
///  @params[in] index The species index.
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
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
///  @params[in] index The species index.
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_magnetic_field(graph::shared_leaf<T, SAFE_MATH> x,
                           graph::shared_leaf<T, SAFE_MATH> y,
                           graph::shared_leaf<T, SAFE_MATH> z) final {
            auto zero = graph::zero<T, SAFE_MATH> ();
            return graph::vector(zero, zero, graph::one<T, SAFE_MATH> ());
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
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
///  @params[in] index The species index.
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
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
///  @params[in] index The species index.
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_magnetic_field(graph::shared_leaf<T, SAFE_MATH> x,
                           graph::shared_leaf<T, SAFE_MATH> y,
                           graph::shared_leaf<T, SAFE_MATH> z) final {
            auto zero = graph::zero<T, SAFE_MATH> ();
            return graph::vector(zero, zero,
                                 graph::constant<T, SAFE_MATH> (static_cast<T> (0.01))*x +
                                 graph::one<T, SAFE_MATH> ());
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
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
///  @params[in] index The species index.
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
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
///  @params[in] index The species index.
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_magnetic_field(graph::shared_leaf<T, SAFE_MATH> x,
                           graph::shared_leaf<T, SAFE_MATH> y,
                           graph::shared_leaf<T, SAFE_MATH> z) final {
            auto zero = graph::zero<T, SAFE_MATH> ();
            return graph::vector(graph::one<T, SAFE_MATH> (), zero, zero);
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

    public:
//------------------------------------------------------------------------------
///  @brief Construct a EFIT equilibrium.
///
///  @params[in] psimin     Minimum psi value.
///  @params[in] dpsi       Change in psi value.
///  @params[in] te_c0      Te c0 spline coefficient.
///  @params[in] te_c1      Te c1 spline coefficient.
///  @params[in] te_c2      Te c2 spline coefficient.
///  @params[in] te_c3      Te c3 spline coefficient.
///  @params[in] te_scale   Temperatire scale.
///  @params[in] ne_c0      Ne c0 spline coefficient.
///  @params[in] ne_c1      Ne c1 spline coefficient.
///  @params[in] ne_c2      Ne c2 spline coefficient.
///  @params[in] ne_c3      Ne c3 spline coefficient.
///  @params[in] ne_scale   Denisty scale.
///  @params[in] pres_c0    Pressure c0 spline coefficient.
///  @params[in] pres_c1    Pressure c1 spline coefficient.
///  @params[in] pres_c2    Pressure c2 spline coefficient.
///  @params[in] pres_c3    Pressure c3 spline coefficient.
///  @params[in] pres_scale Pressure scale.
///  @params[in] rmin       Radial gird minimum.
///  @params[in] dr         Radial grid spacing.
///  @params[in] zmin       Vertical grid minimum.
///  @params[in] dz         Vertical grid spacing.
///  @params[in] fpol_c0    Flux function c0 spline coefficient.
///  @params[in] fpol_c1    Flux function c1 spline coefficient.
///  @params[in] fpol_c2    Flux function c2 spline coefficient.
///  @params[in] fpol_c3    Flux function c3 spline coefficient.
///  @params[in] num_cols   Number of columns for the 2D splines.
///  @params[in] c00        Psi c00 spline coefficient.
///  @params[in] c01        Psi c01 spline coefficient.
///  @params[in] c02        Psi c02 spline coefficient.
///  @params[in] c03        Psi c03 spline coefficient.
///  @params[in] c10        Psi c10 spline coefficient.
///  @params[in] c11        Psi c11 spline coefficient.
///  @params[in] c12        Psi c12 spline coefficient.
///  @params[in] c13        Psi c13 spline coefficient.
///  @params[in] c20        Psi c20 spline coefficient.
///  @params[in] c21        Psi c21 spline coefficient.
///  @params[in] c22        Psi c22 spline coefficient.
///  @params[in] c23        Psi c23 spline coefficient.
///  @params[in] c30        Psi c30 spline coefficient.
///  @params[in] c31        Psi c31 spline coefficient.
///  @params[in] c32        Psi c32 spline coefficient.
///  @params[in] c33        Psi c33 spline coefficient.
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
        c30(c30), c31(c31), c32(c32), c33(c33) {}

//------------------------------------------------------------------------------
///  @brief Get psi.
///
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns The psi expression.
//------------------------------------------------------------------------------
        graph::shared_leaf<T, SAFE_MATH>
        get_psi(graph::shared_leaf<T, SAFE_MATH> x,
                graph::shared_leaf<T, SAFE_MATH> y,
                graph::shared_leaf<T, SAFE_MATH> z) {
            return get_psi(graph::sqrt(x*x + y*y), z);
        }

//------------------------------------------------------------------------------
///  @brief Get psi.
///
///  @params[in] r R position.
///  @params[in] z Z position.
///  @returns The psi expression.
//------------------------------------------------------------------------------
        graph::shared_leaf<T, SAFE_MATH>
        get_psi(graph::shared_leaf<T, SAFE_MATH> r,
                graph::shared_leaf<T, SAFE_MATH> z) {
            auto r_norm = (r - rmin)/dr;
            auto z_norm = (z - zmin)/dz;

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

            return c00_temp +
                   c01_temp*z_norm +
                   c02_temp*(z_norm*z_norm) +
                   c03_temp*(z_norm*z_norm*z_norm) +
                   c10_temp*r_norm +
                   c11_temp*r_norm*z_norm +
                   c12_temp*r_norm*(z_norm*z_norm) +
                   c13_temp*r_norm*(z_norm*z_norm*z_norm) +
                   c20_temp*(r_norm*r_norm) +
                   c21_temp*(r_norm*r_norm)*z_norm +
                   c22_temp*(r_norm*r_norm)*(z_norm*z_norm) +
                   c23_temp*(r_norm*r_norm)*(z_norm*z_norm*z_norm) +
                   c30_temp*(r_norm*r_norm*r_norm) +
                   c31_temp*(r_norm*r_norm*r_norm)*z_norm +
                   c32_temp*(r_norm*r_norm*r_norm)*(z_norm*z_norm) +
                   c33_temp*(r_norm*r_norm*r_norm)*(z_norm*z_norm*z_norm);
        }

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns The electron density expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_density(graph::shared_leaf<T, SAFE_MATH> x,
                             graph::shared_leaf<T, SAFE_MATH> y,
                             graph::shared_leaf<T, SAFE_MATH> z) {
            auto psi_norm = (get_psi(x, y, z) - psimin)/dpsi;

            auto c0_temp = graph::piecewise_1D(ne_c0, psi_norm);
            auto c1_temp = graph::piecewise_1D(ne_c1, psi_norm);
            auto c2_temp = graph::piecewise_1D(ne_c2, psi_norm);
            auto c3_temp = graph::piecewise_1D(ne_c3, psi_norm);

            return ne_scale*(c0_temp +
                             c1_temp*psi_norm +
                             c2_temp*psi_norm*psi_norm +
                             c3_temp*psi_norm*psi_norm*psi_norm);
        }

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @params[in] index The species index.
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns The ion density expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_density(const size_t index,
                        graph::shared_leaf<T, SAFE_MATH> x,
                        graph::shared_leaf<T, SAFE_MATH> y,
                        graph::shared_leaf<T, SAFE_MATH> z) {
            return get_electron_density(x, y, z);
        }

//------------------------------------------------------------------------------
///  @brief Get the pressure.
///
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns The pressure expression.
//------------------------------------------------------------------------------
        graph::shared_leaf<T, SAFE_MATH>
        get_pressure(graph::shared_leaf<T, SAFE_MATH> x,
                     graph::shared_leaf<T, SAFE_MATH> y,
                     graph::shared_leaf<T, SAFE_MATH> z) {
            auto psi_norm = (get_psi(x, y, z) - psimin)/dpsi;

            auto c0_temp = graph::piecewise_1D(pres_c0, psi_norm);
            auto c1_temp = graph::piecewise_1D(pres_c1, psi_norm);
            auto c2_temp = graph::piecewise_1D(pres_c2, psi_norm);
            auto c3_temp = graph::piecewise_1D(pres_c3, psi_norm);

            return pres_scale*(c0_temp +
                               c1_temp*psi_norm +
                               c2_temp*psi_norm*psi_norm +
                               c3_temp*psi_norm*psi_norm*psi_norm);
        }

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns The electron temperature expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_electron_temperature(graph::shared_leaf<T, SAFE_MATH> x,
                                 graph::shared_leaf<T, SAFE_MATH> y,
                                 graph::shared_leaf<T, SAFE_MATH> z) {
            auto psi_norm = (get_psi(x, y, z) - psimin)/dpsi;

            auto c0_temp = graph::piecewise_1D(te_c0, psi_norm);
            auto c1_temp = graph::piecewise_1D(te_c1, psi_norm);
            auto c2_temp = graph::piecewise_1D(te_c2, psi_norm);
            auto c3_temp = graph::piecewise_1D(te_c3, psi_norm);

            return te_scale*(c0_temp +
                             c1_temp*psi_norm +
                             c2_temp*psi_norm*psi_norm +
                             c3_temp*psi_norm*psi_norm*psi_norm);
        }

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @params[in] index The species index.
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns The ion temperature expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_ion_temperature(const size_t index,
                            graph::shared_leaf<T, SAFE_MATH> x,
                            graph::shared_leaf<T, SAFE_MATH> y,
                            graph::shared_leaf<T, SAFE_MATH> z) {
            auto pressure = get_pressure(x, y, z);
            auto q = graph::constant<T, SAFE_MATH> (static_cast<T> (1.60218E-19));
            return (pressure - get_electron_density(x, y, z)*get_electron_temperature(x, y, z)*q) /
                   (get_ion_density(index, x, y, z)*q);
        }

//------------------------------------------------------------------------------
///  @brief Get the toroidal magnetic field.
///
///  @params[in] r R position.
///  @returns The toroidal magnetic field expression.
//------------------------------------------------------------------------------
        graph::shared_leaf<T, SAFE_MATH>
        get_b_phi(graph::shared_leaf<T, SAFE_MATH> r) {
            auto r_norm = (r - rmin)/dr;

            auto c0_temp = graph::piecewise_1D(fpol_c0, r_norm);
            auto c1_temp = graph::piecewise_1D(fpol_c1, r_norm);
            auto c2_temp = graph::piecewise_1D(fpol_c2, r_norm);
            auto c3_temp = graph::piecewise_1D(fpol_c3, r_norm);

            return (c0_temp +
                    c1_temp*r_norm +
                    c2_temp*r_norm*r_norm +
                    c3_temp*r_norm*r_norm*r_norm)/r;
        }

//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T, SAFE_MATH>
        get_magnetic_field(graph::shared_leaf<T, SAFE_MATH> x,
                           graph::shared_leaf<T, SAFE_MATH> y,
                           graph::shared_leaf<T, SAFE_MATH> z) {
            auto r = graph::sqrt(x*x + y*y);
            auto phi = graph::atan(x, y);
            auto none = graph::none<T, SAFE_MATH> ();
            auto psi = get_psi(r, z);

            auto br = psi->df(z)/r;
            auto bp = get_b_phi(r);
            auto bz = none*psi->df(r)/r;
            
            auto cos = graph::cos(phi);
            auto sin = graph::sin(phi);
            
            return graph::vector(br*cos - bp*sin,
                                 br*sin + bp*cos,
                                 bz);
        }
    };

//------------------------------------------------------------------------------
///  @brief Convenience function to build an EFIT equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @params[in] spline_file File name of contains the spline functions.
///  @returns A constructed EFIT equilibrium.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared<T, SAFE_MATH> make_efit(const std::string spline_file) {
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
}

#endif /* equilibrium_h */
