//------------------------------------------------------------------------------
///  @file equilibrium.hpp
///  @brief Class signature to impliment plasma equilibrium.
///
///  Defined the interfaces to access plasma equilibrium.
//------------------------------------------------------------------------------

#ifndef equilibrium_h
#define equilibrium_h

#include <vector>

#include "math.hpp"

namespace equilibrium {
//******************************************************************************
//  Equilibrium interface
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a generic equilibrium.
//------------------------------------------------------------------------------
    template<typename T>
    class equilibrium {
    protected:
///  Ion masses for each species.
        const std::vector<T> ion_masses;
///  Ion charge for each species.
        const std::vector<uint8_t> ion_charges;

    public:
//------------------------------------------------------------------------------
///  @brief Construct an equilibrum.
///
///  @param[in] masses  Vector of ion masses.
///  @param[in] charges Vector of ion charges.
//------------------------------------------------------------------------------
        equilibrium(const std::vector<T> &masses,
                    const std::vector<uint8_t> &charges) :
        ion_masses(masses), ion_charges(charges) {
            assert(ion_masses.size() == ion_charges.size() &&
                   "Masses and charges need the same number of elements.");
        }

//------------------------------------------------------------------------------
///  @brief Destructor
//------------------------------------------------------------------------------
        virtual ~equilibrium() {}

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
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_electron_density(graph::shared_leaf<T> x,
                                                           graph::shared_leaf<T> y,
                                                           graph::shared_leaf<T> z) = 0;

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @param[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_ion_density(const size_t index,
                                                      graph::shared_leaf<T> x,
                                                      graph::shared_leaf<T> y,
                                                      graph::shared_leaf<T> z) = 0;

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_electron_temperature(graph::shared_leaf<T> x,
                                                               graph::shared_leaf<T> y,
                                                               graph::shared_leaf<T> z) = 0;

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @param[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_ion_temperature(const size_t index,
                                                          graph::shared_leaf<T> x,
                                                          graph::shared_leaf<T> y,
                                                          graph::shared_leaf<T> z) = 0;
        
//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<graph::shared_leaf<T>,
                                     graph::shared_leaf<T>,
                                     graph::shared_leaf<T>>
        get_magnetic_field(graph::shared_leaf<T> x,
                           graph::shared_leaf<T> y,
                           graph::shared_leaf<T> z) = 0;
    };

///  Convenience type alias for unique equilibria.
    template<typename T>
    using unique_equilibrium = std::unique_ptr<equilibrium<T>>;

//******************************************************************************
//  No Magnetic equilibrium.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Uniform density with varying magnetic field equilibrium.
//------------------------------------------------------------------------------
    template<typename T>
    class no_magnetic_field : public equilibrium<T> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a linear density with no magnetic field.
//------------------------------------------------------------------------------
        no_magnetic_field() :
        equilibrium<T> ({3.34449469E-27},
                        {1}) {}

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_electron_density(graph::shared_leaf<T> x,
                                                           graph::shared_leaf<T> y,
                                                           graph::shared_leaf<T> z) final {
            return graph::constant(static_cast<T> (1.0E19)) *
                   (graph::constant(static_cast<T> (0.1))*x + graph::one<T> ());
        }

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @param[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_ion_density(const size_t index,
                                                      graph::shared_leaf<T> x,
                                                      graph::shared_leaf<T> y,
                                                      graph::shared_leaf<T> z) final {
            return graph::constant(static_cast<T> (1.0E19)) *
                   (graph::constant(static_cast<T> (0.1))*x + graph::one<T> ());
        }

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_electron_temperature(graph::shared_leaf<T> x,
                                                               graph::shared_leaf<T> y,
                                                               graph::shared_leaf<T> z) final {
            return graph::constant(static_cast<T> (1000.0));
        }

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @param[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_ion_temperature(const size_t index,
                                                          graph::shared_leaf<T> x,
                                                          graph::shared_leaf<T> y,
                                                          graph::shared_leaf<T> z) final {
            return graph::constant(static_cast<T> (1000.0));
        }

//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<graph::shared_leaf<T>,
                                     graph::shared_leaf<T>,
                                     graph::shared_leaf<T>>
        get_magnetic_field(graph::shared_leaf<T> x,
                           graph::shared_leaf<T> y,
                           graph::shared_leaf<T> z) final {
            auto zero = graph::zero<T> ();
            return graph::vector(zero, zero, zero);
        }
    };

//------------------------------------------------------------------------------
///  @brief Convenience function to build a no magnetic field equilibrium.
///
///  @returns A constructed no magnetic field equilibrium.
//------------------------------------------------------------------------------
    template<typename T>
    std::unique_ptr<equilibrium<T>> make_no_magnetic_field() {
        return std::make_unique<no_magnetic_field<T>> ();
    }

//******************************************************************************
//  Slab equilibrium.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Uniform density with varying magnetic field equilibrium.
//------------------------------------------------------------------------------
    template<typename T>
    class slab : public equilibrium<T> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a guassian density with uniform magnetic field.
//------------------------------------------------------------------------------
        slab() :
        equilibrium<T> ({3.34449469E-27},
                        {1}) {}

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_electron_density(graph::shared_leaf<T> x,
                                                           graph::shared_leaf<T> y,
                                                           graph::shared_leaf<T> z) final {
            return graph::constant(static_cast<T> (1.0E19));
        }

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @param[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_ion_density(const size_t index,
                                                      graph::shared_leaf<T> x,
                                                      graph::shared_leaf<T> y,
                                                      graph::shared_leaf<T> z) final {
            return graph::constant(static_cast<T> (1.0E19));
        }

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_electron_temperature(graph::shared_leaf<T> x,
                                                               graph::shared_leaf<T> y,
                                                               graph::shared_leaf<T> z) final {
            return graph::constant(static_cast<T> (1000.0));
        }

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @param[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_ion_temperature(const size_t index,
                                                          graph::shared_leaf<T> x,
                                                          graph::shared_leaf<T> y,
                                                          graph::shared_leaf<T> z) final {
            return graph::constant(static_cast<T> (1000.0));
        }

//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<graph::shared_leaf<T>,
                                     graph::shared_leaf<T>,
                                     graph::shared_leaf<T>>
        get_magnetic_field(graph::shared_leaf<T> x,
                           graph::shared_leaf<T> y,
                           graph::shared_leaf<T> z) final {
            auto zero = graph::zero<T> ();
            return graph::vector(zero, zero,
                                 graph::constant(static_cast<T> (0.1))*x + graph::one<T> ());
        }
    };

//------------------------------------------------------------------------------
///  @brief Convenience function to build a slab equilibrium.
///
///  @returns A constructed slab equilibrium.
//------------------------------------------------------------------------------
    template<typename T>
    std::unique_ptr<equilibrium<T>> make_slab() {
        return std::make_unique<slab<T>> ();
    }

//******************************************************************************
//  Slab density equilibrium.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Vary density with uniform magnetic field equilibrium.
//------------------------------------------------------------------------------
    template<typename T>
    class slab_density : public equilibrium<T> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a guassian density with uniform magnetic field.
//------------------------------------------------------------------------------
        slab_density() :
        equilibrium<T> ({3.34449469E-27},
                        {1}) {}

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_electron_density(graph::shared_leaf<T> x,
                                                           graph::shared_leaf<T> y,
                                                           graph::shared_leaf<T> z) final {
            return graph::constant(static_cast<T> (1.0E19)) *
                   (graph::constant(static_cast<T> (0.1))*x + graph::one<T> ());
        }

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @param[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_ion_density(const size_t index,
                                                      graph::shared_leaf<T> x,
                                                      graph::shared_leaf<T> y,
                                                      graph::shared_leaf<T> z) final {
            return graph::constant(static_cast<T> (1.0E19)) *
                   (graph::constant(static_cast<T> (0.1))*x + graph::one<T> ());
        }

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_electron_temperature(graph::shared_leaf<T> x,
                                                               graph::shared_leaf<T> y,
                                                               graph::shared_leaf<T> z) final {
            return graph::constant(static_cast<T> (1000.0));
        }

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @param[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_ion_temperature(const size_t index,
                                                          graph::shared_leaf<T> x,
                                                          graph::shared_leaf<T> y,
                                                          graph::shared_leaf<T> z) final {
            return graph::constant(static_cast<T> (1000.0));
        }
        
//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<graph::shared_leaf<T>,
                                     graph::shared_leaf<T>,
                                     graph::shared_leaf<T>>
        get_magnetic_field(graph::shared_leaf<T> x,
                           graph::shared_leaf<T> y,
                           graph::shared_leaf<T> z) final {
            auto zero = graph::zero<T> ();
            return graph::vector(zero, zero, graph::one<T> ());
        }
    };

//------------------------------------------------------------------------------
///  @brief Convenience function to build a slab density equilibrium.
///
///  @returns A constructed slab density equilibrium.
//------------------------------------------------------------------------------
    template<typename T>
    std::unique_ptr<equilibrium<T>> make_slab_density() {
        return std::make_unique<slab_density<T>> ();
    }

//******************************************************************************
//  Guassian density with a uniform magnetic field.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Guassian density with uniform magnetic field equilibrium.
//------------------------------------------------------------------------------
    template<typename T>
    class guassian_density : public equilibrium<T> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a guassian density with uniform magnetic field.
//------------------------------------------------------------------------------
        guassian_density() :
        equilibrium<T> ({3.34449469E-27},
                        {1}) {}

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_electron_density(graph::shared_leaf<T> x,
                                                           graph::shared_leaf<T> y,
                                                           graph::shared_leaf<T> z) final {
            return graph::constant(static_cast<T> (1.0E19))*graph::exp((x*x + y*y)/graph::constant(static_cast<T> (-0.2)));
        }

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @param[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_ion_density(const size_t index,
                                                      graph::shared_leaf<T> x,
                                                      graph::shared_leaf<T> y,
                                                      graph::shared_leaf<T> z) final {
            return graph::constant(static_cast<T> (1.0E19))*graph::exp((x*x + y*y)/graph::constant(static_cast<T> (-0.2)));
        }

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_electron_temperature(graph::shared_leaf<T> x,
                                                               graph::shared_leaf<T> y,
                                                               graph::shared_leaf<T> z) final {
            return graph::constant(static_cast<T> (1000.0));
        }

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @param[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_ion_temperature(const size_t index,
                                                          graph::shared_leaf<T> x,
                                                          graph::shared_leaf<T> y,
                                                          graph::shared_leaf<T> z) final {
            return graph::constant(static_cast<T> (1000.0));
        }
        
//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<graph::shared_leaf<T>,
                                     graph::shared_leaf<T>,
                                     graph::shared_leaf<T>>
        get_magnetic_field(graph::shared_leaf<T> x,
                           graph::shared_leaf<T> y,
                           graph::shared_leaf<T> z) final {
            auto zero = graph::zero<T> ();
            return graph::vector(graph::one<T> (), zero, zero);
        }
    };

//------------------------------------------------------------------------------
///  @brief Convenience function to build a guassian density equilibrium.
///
///  @returns A constructed guassian density equilibrium.
//------------------------------------------------------------------------------
    template<typename T>
    std::unique_ptr<equilibrium<T>> make_guassian_density() {
        return std::make_unique<guassian_density<T>> ();
    }
}

#endif /* equilibrium_h */
