//------------------------------------------------------------------------------
///  @file equilibrium.hpp
///  @brief Class signature to impliment plasma equilibrium.
///
///  Defined the interfaces to access plasma equilibrium.
//------------------------------------------------------------------------------

#ifndef equilibrium_h
#define equilibrium_h

#include <vector>
#include <mutex>

#include <netcdf.h>

#include "math.hpp"
#include "trigonometry.hpp"
#include "cublic_splines.hpp"
#include "vector.hpp"

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
///  @params[in] masses  Vector of ion masses.
///  @params[in] charges Vector of ion charges.
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
        virtual graph::shared_leaf<T> get_electron_density(graph::shared_leaf<T> x,
                                                           graph::shared_leaf<T> y,
                                                           graph::shared_leaf<T> z) = 0;

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @params[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_ion_density(const size_t index,
                                                      graph::shared_leaf<T> x,
                                                      graph::shared_leaf<T> y,
                                                      graph::shared_leaf<T> z) = 0;

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_electron_temperature(graph::shared_leaf<T> x,
                                                               graph::shared_leaf<T> y,
                                                               graph::shared_leaf<T> z) = 0;

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @params[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_ion_temperature(const size_t index,
                                                          graph::shared_leaf<T> x,
                                                          graph::shared_leaf<T> y,
                                                          graph::shared_leaf<T> z) = 0;
        
//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T>
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
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
///  @params[in] index The species index.
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
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
///  @params[in] index The species index.
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T>
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
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
///  @params[in] index The species index.
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
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
///  @params[in] index The species index.
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T>
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
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
///  @params[in] index The species index.
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
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
///  @params[in] index The species index.
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T>
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
        equilibrium<T> ({3.34449469E-27}, {1}) {}

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
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
///  @params[in] index The species index.
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
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
///  @params[in] index The species index.
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
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T>
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

//******************************************************************************
//  2D EFIT equilibrium.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief 2D EFIT equilibrium.
///
///  This takes a BiCublic spline representation of the psi and cubic splines for
///  ne, te, p, and fpol.
//------------------------------------------------------------------------------
    template<typename T>
    class efit_equilibrium final : public equilibrium<T> {
    private:
///   Electron temperature profile.
        graph::shared_leaf<T> te;
///   Electron temperature profile.
        graph::shared_leaf<T> ne;
///   Electron temperature profile.
        graph::shared_leaf<T> pressure;
///   Magnetic field in the phi direction.
        graph::shared_leaf<T> b_phi;
///   Magnetic flux.
        graph::shared_leaf<T> psi;

    public:
///  Type alias for the fouier spline coefficients.
        typedef std::vector<std::array<std::vector<T>, 4>> fourier_t;
        
//------------------------------------------------------------------------------
///  @brief Construct a EFIT equilibrium.
///
///  @params[in]     spline_file File name of contains the spline functions.
///  @params[in]     x           X variable.
///  @params[in]     y           Y variable.
///  @params[in]     z           Z variable.
///  @params[in,out] sync        Mutex to ensure the netcdf file is read only by
///                              one thread.
//------------------------------------------------------------------------------
        efit_equilibrium(const std::string spline_file,
                         graph::shared_leaf<T> x,
                         graph::shared_leaf<T> y,
                         graph::shared_leaf<T> z,
                         std::mutex &sync) :
        equilibrium<T> ({3.34449469E-27} ,{1}) {
            int ncid;
            sync.lock();
            nc_open(spline_file.c_str(), NC_NOWRITE, &ncid);

            auto r = graph::sqrt(x*x + y*y);

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

            auto rmin = graph::constant(static_cast<T> (rmin_value));
            auto dr = graph::constant(static_cast<T> (dr_value));
            auto zmin = graph::constant(static_cast<T> (zmin_value));
            auto dz = graph::constant(static_cast<T> (dz_value));
            auto psimin = graph::constant(static_cast<T> (psimin_value));
            auto dpsi = graph::constant(static_cast<T> (dpsi_value));
            auto pres_scale = graph::constant(static_cast<T> (pres_scale_value));
            auto ne_scale = graph::constant(static_cast<T> (ne_scale_value));
            auto te_scale = graph::constant(static_cast<T> (te_scale_value));

            b_phi = spline::make_1D(rmin, dr,
                                    std::vector<T> (fpol_c0_buffer.begin(), fpol_c0_buffer.end()),
                                    std::vector<T> (fpol_c1_buffer.begin(), fpol_c1_buffer.end()),
                                    std::vector<T> (fpol_c2_buffer.begin(), fpol_c2_buffer.end()),
                                    std::vector<T> (fpol_c3_buffer.begin(), fpol_c3_buffer.end()),
                                    r)/r;

            psi = spline::make_2D(rmin, dr, zmin, dz,
                                  std::vector<T> (psi_c00_buffer.begin(), psi_c00_buffer.end()),
                                  std::vector<T> (psi_c01_buffer.begin(), psi_c01_buffer.end()),
                                  std::vector<T> (psi_c02_buffer.begin(), psi_c02_buffer.end()),
                                  std::vector<T> (psi_c03_buffer.begin(), psi_c03_buffer.end()),
                                  std::vector<T> (psi_c10_buffer.begin(), psi_c10_buffer.end()),
                                  std::vector<T> (psi_c11_buffer.begin(), psi_c11_buffer.end()),
                                  std::vector<T> (psi_c12_buffer.begin(), psi_c12_buffer.end()),
                                  std::vector<T> (psi_c13_buffer.begin(), psi_c13_buffer.end()),
                                  std::vector<T> (psi_c20_buffer.begin(), psi_c20_buffer.end()),
                                  std::vector<T> (psi_c21_buffer.begin(), psi_c21_buffer.end()),
                                  std::vector<T> (psi_c22_buffer.begin(), psi_c22_buffer.end()),
                                  std::vector<T> (psi_c23_buffer.begin(), psi_c23_buffer.end()),
                                  std::vector<T> (psi_c30_buffer.begin(), psi_c30_buffer.end()),
                                  std::vector<T> (psi_c31_buffer.begin(), psi_c31_buffer.end()),
                                  std::vector<T> (psi_c32_buffer.begin(), psi_c32_buffer.end()),
                                  std::vector<T> (psi_c33_buffer.begin(), psi_c33_buffer.end()),
                                  r, z, numz);

            pressure = pres_scale*spline::make_1D(psimin, dpsi,
                                                  std::vector<T> (pressure_c0_buffer.begin(), pressure_c0_buffer.end()),
                                                  std::vector<T> (pressure_c1_buffer.begin(), pressure_c1_buffer.end()),
                                                  std::vector<T> (pressure_c2_buffer.begin(), pressure_c2_buffer.end()),
                                                  std::vector<T> (pressure_c3_buffer.begin(), pressure_c3_buffer.end()),
                                                  psi);
            
            te = te_scale*spline::make_1D(psimin, dpsi,
                                          std::vector<T> (te_c0_buffer.begin(), te_c0_buffer.end()),
                                          std::vector<T> (te_c1_buffer.begin(), te_c1_buffer.end()),
                                          std::vector<T> (te_c2_buffer.begin(), te_c2_buffer.end()),
                                          std::vector<T> (te_c3_buffer.begin(), te_c3_buffer.end()),
                                          psi);

            ne = ne_scale*spline::make_1D(psimin, dpsi,
                                          std::vector<T> (ne_c0_buffer.begin(), ne_c0_buffer.end()),
                                          std::vector<T> (ne_c1_buffer.begin(), ne_c1_buffer.end()),
                                          std::vector<T> (ne_c2_buffer.begin(), ne_c2_buffer.end()),
                                          std::vector<T> (ne_c3_buffer.begin(), ne_c3_buffer.end()),
                                          psi);
        }

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_electron_density(graph::shared_leaf<T> x,
                                                           graph::shared_leaf<T> y,
                                                           graph::shared_leaf<T> z) {
            return ne;
        }

//------------------------------------------------------------------------------
///  @brief Get the ion density.
///
///  @params[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_ion_density(const size_t index,
                                                      graph::shared_leaf<T> x,
                                                      graph::shared_leaf<T> y,
                                                      graph::shared_leaf<T> z) {
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
        virtual graph::shared_leaf<T> get_electron_temperature(graph::shared_leaf<T> x,
                                                               graph::shared_leaf<T> y,
                                                               graph::shared_leaf<T> z) {
            return te;
        }

//------------------------------------------------------------------------------
///  @brief Get the ion temperature.
///
///  @params[in] index The species index.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_ion_temperature(const size_t index,
                                                          graph::shared_leaf<T> x,
                                                          graph::shared_leaf<T> y,
                                                          graph::shared_leaf<T> z) {
            auto q = graph::constant(static_cast<T> (1.60218E-19));
            return (pressure - get_electron_density(x, y, z)*get_electron_temperature(x, y, z)/q) /
                   (get_ion_density(index, x, y, z) + graph::constant(static_cast<T> (1.0E-100)));
        }

//------------------------------------------------------------------------------
///  @brief Get the electron temperature.
///
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns The electron expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_psi(graph::shared_leaf<T> x,
                                              graph::shared_leaf<T> y,
                                              graph::shared_leaf<T> z) {
            return psi;
        }

//------------------------------------------------------------------------------
///  @brief Get the magnetic field.
///
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns Magnetic field expression.
//------------------------------------------------------------------------------
        virtual graph::shared_vector<T>
        get_magnetic_field(graph::shared_leaf<T> x,
                           graph::shared_leaf<T> y,
                           graph::shared_leaf<T> z) {
            auto r = graph::sqrt(x*x + y*y);
            auto phi = graph::atan(x, y);
            auto none = graph::none<T> ();
            
            auto br = psi->df(z)/r;
            auto bz = none*psi->df(r)/r;
            
            auto cos = graph::cos(phi);
            auto sin = graph::sin(phi);
            
            return graph::vector(br*cos - b_phi*sin,
                                 br*sin + b_phi*cos,
                                 bz);
        }
    };

//------------------------------------------------------------------------------
///  @brief Convenience function to build an EFIT equilibrium.
///
///  @params[in] spline_file File name of contains the spline functions.
///  @params[in] x           X variable.
///  @params[in] y           Y variable.
///  @params[in] z           Z variable.
///  @params[in,out] sync    Mutex to ensure the netcdf file is read only by one
///                          thread.
///  @returns A constructed EFIT equilibrium.
//------------------------------------------------------------------------------
    template<typename T>
    std::unique_ptr<equilibrium<T>> make_efit(const std::string spline_file,
                                              graph::shared_leaf<T> x,
                                              graph::shared_leaf<T> y,
                                              graph::shared_leaf<T> z,
                                              std::mutex &sync) {
        return std::make_unique<efit_equilibrium<T>> (spline_file, x, y, z, sync);
    }
}

#endif /* equilibrium_h */
