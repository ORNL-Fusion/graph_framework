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
    template<typename BACKEND>
    class equilibrium {
    protected:
///  Ion masses for each species.
        const std::vector<typename BACKEND::base> ion_masses;
///  Ion charge for each species.
        const std::vector<uint8_t> ion_charges;

    public:
//------------------------------------------------------------------------------
///  @brief Construct an equilibrum.
///
///  @param[in] masses  Vector of ion masses.
///  @param[in] charges Vector of ion charges.
//------------------------------------------------------------------------------
        equilibrium(const std::vector<typename BACKEND::base> &masses,
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
        typename BACKEND::base get_ion_mass(const size_t index) const {
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
        virtual graph::shared_leaf<BACKEND> get_electron_density(graph::shared_leaf<BACKEND> x,
                                                                 graph::shared_leaf<BACKEND> y,
                                                                 graph::shared_leaf<BACKEND> z) = 0;

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @param[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> get_ion_density(const size_t index,
                                                            graph::shared_leaf<BACKEND> x,
                                                            graph::shared_leaf<BACKEND> y,
                                                            graph::shared_leaf<BACKEND> z) = 0;

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> get_electron_temperature(graph::shared_leaf<BACKEND> x,
                                                                     graph::shared_leaf<BACKEND> y,
                                                                     graph::shared_leaf<BACKEND> z) = 0;

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @param[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> get_ion_temperature(const size_t index,
                                                                graph::shared_leaf<BACKEND> x,
                                                                graph::shared_leaf<BACKEND> y,
                                                                graph::shared_leaf<BACKEND> z) = 0;
        
//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<graph::shared_leaf<BACKEND>,
                                     graph::shared_leaf<BACKEND>,
                                     graph::shared_leaf<BACKEND>>
        get_magnetic_field(graph::shared_leaf<BACKEND> x,
                           graph::shared_leaf<BACKEND> y,
                           graph::shared_leaf<BACKEND> z) = 0;
    };

///  Convience type alias for unique equilibria.
    template<typename BACKEND>
    using unique_equilibrium = std::unique_ptr<equilibrium<BACKEND>>;

//******************************************************************************
//  Slab equilibrium.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Uniform density with varying magnetic field equilibrium.
//------------------------------------------------------------------------------
    template<typename BACKEND>
    class slab : public equilibrium<BACKEND> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a guassian density with uniform magnetic field.
//------------------------------------------------------------------------------
        slab() :
        equilibrium<BACKEND> ({3.34449469E-27},
                              {1}) {}

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> get_electron_density(graph::shared_leaf<BACKEND> x,
                                                                 graph::shared_leaf<BACKEND> y,
                                                                 graph::shared_leaf<BACKEND> z) final {
            return graph::constant<BACKEND> (1.0E19);
        }

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @param[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> get_ion_density(const size_t index,
                                                            graph::shared_leaf<BACKEND> x,
                                                            graph::shared_leaf<BACKEND> y,
                                                            graph::shared_leaf<BACKEND> z) final {
            return graph::constant<BACKEND> (1.0E19);
        }

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> get_electron_temperature(graph::shared_leaf<BACKEND> x,
                                                                     graph::shared_leaf<BACKEND> y,
                                                                     graph::shared_leaf<BACKEND> z) final {
            return graph::constant<BACKEND> (1000.0);
        }

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @param[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> get_ion_temperature(const size_t index,
                                                                graph::shared_leaf<BACKEND> x,
                                                                graph::shared_leaf<BACKEND> y,
                                                                graph::shared_leaf<BACKEND> z) final {
            return graph::constant<BACKEND> (1000.0);
        }

//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<graph::shared_leaf<BACKEND>,
                                     graph::shared_leaf<BACKEND>,
                                     graph::shared_leaf<BACKEND>>
        get_magnetic_field(graph::shared_leaf<BACKEND> x,
                           graph::shared_leaf<BACKEND> y,
                           graph::shared_leaf<BACKEND> z) final {
            auto zero = graph::constant<BACKEND> (0.0);
            return graph::vector(zero, zero,
                                 graph::constant<BACKEND> (0.1)*x + graph::constant<BACKEND> (1.0));
        }
    };

///  Convience type alias for unique equilibria.
    template<typename BACKEND>
    std::unique_ptr<equilibrium<BACKEND>> make_slab() {
        return std::make_unique<slab<BACKEND>> ();
    }

//******************************************************************************
//  Slab density equilibrium.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Vary density with uniform magnetic field equilibrium.
//------------------------------------------------------------------------------
    template<typename BACKEND>
    class slab_density : public equilibrium<BACKEND> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a guassian density with uniform magnetic field.
//------------------------------------------------------------------------------
        slab_density() :
        equilibrium<BACKEND> ({3.34449469E-27},
                              {1}) {}

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> get_electron_density(graph::shared_leaf<BACKEND> x,
                                                                 graph::shared_leaf<BACKEND> y,
                                                                 graph::shared_leaf<BACKEND> z) final {
            return graph::constant<BACKEND> (1.0E19) *
                   (graph::constant<BACKEND> (0.1)*x + graph::constant<BACKEND> (1.0));
        }

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @param[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> get_ion_density(const size_t index,
                                                            graph::shared_leaf<BACKEND> x,
                                                            graph::shared_leaf<BACKEND> y,
                                                            graph::shared_leaf<BACKEND> z) final {
            return graph::constant<BACKEND> (1.0E19) *
                   (graph::constant<BACKEND> (0.1)*x + graph::constant<BACKEND> (1.0));
        }

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> get_electron_temperature(graph::shared_leaf<BACKEND> x,
                                                                     graph::shared_leaf<BACKEND> y,
                                                                     graph::shared_leaf<BACKEND> z) final {
            return graph::constant<BACKEND> (1000.0);
        }

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @param[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> get_ion_temperature(const size_t index,
                                                                graph::shared_leaf<BACKEND> x,
                                                                graph::shared_leaf<BACKEND> y,
                                                                graph::shared_leaf<BACKEND> z) final {
            return graph::constant<BACKEND> (1000.0);
        }
        
//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<graph::shared_leaf<BACKEND>,
                                     graph::shared_leaf<BACKEND>,
                                     graph::shared_leaf<BACKEND>>
        get_magnetic_field(graph::shared_leaf<BACKEND> x,
                           graph::shared_leaf<BACKEND> y,
                           graph::shared_leaf<BACKEND> z) final {
            auto zero = graph::constant<BACKEND> (0.0);
            return graph::vector(zero, zero, zero);
        }
    };

///  Convience type alias for unique equilibria.
    template<typename BACKEND>
    std::unique_ptr<equilibrium<BACKEND>> make_slab_density() {
        return std::make_unique<slab_density<BACKEND>> ();
    }

//******************************************************************************
//  Guassian density with a uniform magnetic field.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Guassian density with uniform magnetic field equilibrium.
//------------------------------------------------------------------------------
    template<typename BACKEND>
    class guassian_density : public equilibrium<BACKEND> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a guassian density with uniform magnetic field.
//------------------------------------------------------------------------------
        guassian_density() :
        equilibrium<BACKEND> ({3.34449469E-27},
                              {1}) {}

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> get_electron_density(graph::shared_leaf<BACKEND> x,
                                                                 graph::shared_leaf<BACKEND> y,
                                                                 graph::shared_leaf<BACKEND> z) final {
            return graph::constant<BACKEND> (1.0E19)*graph::exp((x*x + y*y)/graph::constant<BACKEND> (-0.2));
        }

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @param[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> get_ion_density(const size_t index,
                                                            graph::shared_leaf<BACKEND> x,
                                                            graph::shared_leaf<BACKEND> y,
                                                            graph::shared_leaf<BACKEND> z) final {
            return graph::constant<BACKEND> (1.0E19)*graph::exp((x*x + y*y)/graph::constant<BACKEND> (-0.2));
        }

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> get_electron_temperature(graph::shared_leaf<BACKEND> x,
                                                                     graph::shared_leaf<BACKEND> y,
                                                                     graph::shared_leaf<BACKEND> z) final {
            return graph::constant<BACKEND> (1000.0);
        }

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @param[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<BACKEND> get_ion_temperature(const size_t index,
                                                                graph::shared_leaf<BACKEND> x,
                                                                graph::shared_leaf<BACKEND> y,
                                                                graph::shared_leaf<BACKEND> z) final {
            return graph::constant<BACKEND> (1000.0);
        }
        
//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @param[in] x X position.
///  @param[in] y Y position.
///  @param[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<graph::shared_leaf<BACKEND>,
                                     graph::shared_leaf<BACKEND>,
                                     graph::shared_leaf<BACKEND>>
        get_magnetic_field(graph::shared_leaf<BACKEND> x,
                           graph::shared_leaf<BACKEND> y,
                           graph::shared_leaf<BACKEND> z) final {
            auto zero = graph::constant<BACKEND> (0.0);
            return graph::vector(graph::constant<BACKEND> (1.0),
                                 zero, zero);
        }
    };

///  Convience type alias for unique equilibria.
    template<typename BACKEND>
    std::unique_ptr<equilibrium<BACKEND>> make_guassian_density() {
        return std::make_unique<guassian_density<BACKEND>> ();
    }
}

#endif /* equilibrium_h */
