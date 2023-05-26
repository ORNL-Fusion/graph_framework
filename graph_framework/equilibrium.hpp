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

namespace equilibrium {
//******************************************************************************
//  Equilibrium interface
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a generic equilibrium.
//------------------------------------------------------------------------------
    template<typename T>
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

///  Convenience type alias for shared equilibria.
    template<typename T>
    using shared = std::shared_ptr<generic<T>>;

//******************************************************************************
//  No Magnetic equilibrium.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Uniform density with varying magnetic field equilibrium.
//------------------------------------------------------------------------------
    template<typename T>
    class no_magnetic_field : public generic<T> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a linear density with no magnetic field.
//------------------------------------------------------------------------------
        no_magnetic_field() :
        generic<T> ({3.34449469E-27}, {1}) {}

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
    shared<T> make_no_magnetic_field() {
        return std::make_shared<no_magnetic_field<T>> ();
    }

//******************************************************************************
//  Slab equilibrium.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Uniform density with varying magnetic field equilibrium.
//------------------------------------------------------------------------------
    template<typename T>
    class slab : public generic<T> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a guassian density with uniform magnetic field.
//------------------------------------------------------------------------------
        slab() :
        generic<T> ({3.34449469E-27}, {1}) {}

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
    shared<T> make_slab() {
        return std::make_shared<slab<T>> ();
    }

//******************************************************************************
//  Slab density equilibrium.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Vary density with uniform magnetic field equilibrium.
//------------------------------------------------------------------------------
    template<typename T>
    class slab_density : public generic<T> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a guassian density with uniform magnetic field.
//------------------------------------------------------------------------------
        slab_density() :
        generic<T> ({3.34449469E-27}, {1}) {}

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
    shared<T> make_slab_density() {
        return std::make_shared<slab_density<T>> ();
    }

//******************************************************************************
//  Guassian density with a uniform magnetic field.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Guassian density with uniform magnetic field equilibrium.
//------------------------------------------------------------------------------
    template<typename T>
    class guassian_density : public generic<T> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a guassian density with uniform magnetic field.
//------------------------------------------------------------------------------
        guassian_density() :
        generic<T> ({3.34449469E-27}, {1}) {}

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
    shared<T> make_guassian_density() {
        return std::make_shared<guassian_density<T>> ();
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
    class efit final : public generic<T> {
    private:
///  Minimum psi.
        graph::shared_leaf<T> psimin;
///  Psi grid spacing.
        graph::shared_leaf<T> dpsi;

//  Temperature spline coefficients.
///  Temperature c0.
        graph::shared_leaf<T> te_c0;
///  Temperature c1.
        graph::shared_leaf<T> te_c1;
///  Temperature c2.
        graph::shared_leaf<T> te_c2;
///  Temperature c3.
        graph::shared_leaf<T> te_c3;
///  Temperature scale factor.
        graph::shared_leaf<T> te_scale;

//  Density spline coefficients.
///  Density c0.
        graph::shared_leaf<T> ne_c0;
///  Density c1.
        graph::shared_leaf<T> ne_c1;
///  Density c2.
        graph::shared_leaf<T> ne_c2;
///  Density c3.
        graph::shared_leaf<T> ne_c3;
///  Density scale factor.
        graph::shared_leaf<T> ne_scale;

//  Pressure spline coefficients.
///  Pressure c0.
        graph::shared_leaf<T> pres_c0;
///  Pressure c1.
        graph::shared_leaf<T> pres_c1;
///  Pressure c2.
        graph::shared_leaf<T> pres_c2;
///  Pressure c3.
        graph::shared_leaf<T> pres_c3;
///  Pressure scale factor.
        graph::shared_leaf<T> pres_scale;

///  Minimum R.
        graph::shared_leaf<T> rmin;
///  R grid spacing.
        graph::shared_leaf<T> dr;
///  Minimum Z.
        graph::shared_leaf<T> zmin;
///  Z grid spacing.
        graph::shared_leaf<T> dz;

//  Fpol spline coefficients.
///  Fpol c0.
        graph::shared_leaf<T> fpol_c0;
///  Fpol c1.
        graph::shared_leaf<T> fpol_c1;
///  Fpol c2.
        graph::shared_leaf<T> fpol_c2;
///  Fpol c3.
        graph::shared_leaf<T> fpol_c3;

//  Pressure spline coefficients.
///  Psi c00.
        graph::shared_leaf<T> c00;
///  Psi c01.
        graph::shared_leaf<T> c01;
///  Psi c02.
        graph::shared_leaf<T> c02;
///  Psi c03.
        graph::shared_leaf<T> c03;
///  Psi c10.
        graph::shared_leaf<T> c10;
///  Psi c11.
        graph::shared_leaf<T> c11;
///  Psi c12.
        graph::shared_leaf<T> c12;
///  Psi c13.
        graph::shared_leaf<T> c13;
///  Psi c20.
        graph::shared_leaf<T> c20;
///  Psi c21.
        graph::shared_leaf<T> c21;
///  Psi c22.
        graph::shared_leaf<T> c22;
///  Psi c23.
        graph::shared_leaf<T> c23;
///  Psi c30.
        graph::shared_leaf<T> c30;
///  Psi c31.
        graph::shared_leaf<T> c31;
///  Psi c32.
        graph::shared_leaf<T> c32;
///  Psi c33.
        graph::shared_leaf<T> c33;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a EFIT equilibrium.
///
///  @params[in] psimin Minimum psi value.
//------------------------------------------------------------------------------
        efit(graph::shared_leaf<T> psimin,
             graph::shared_leaf<T> dpsi,
             graph::shared_leaf<T> te_c0,
             graph::shared_leaf<T> te_c1,
             graph::shared_leaf<T> te_c2,
             graph::shared_leaf<T> te_c3,
             graph::shared_leaf<T> te_scale,
             graph::shared_leaf<T> ne_c0,
             graph::shared_leaf<T> ne_c1,
             graph::shared_leaf<T> ne_c2,
             graph::shared_leaf<T> ne_c3,
             graph::shared_leaf<T> ne_scale,
             graph::shared_leaf<T> pres_c0,
             graph::shared_leaf<T> pres_c1,
             graph::shared_leaf<T> pres_c2,
             graph::shared_leaf<T> pres_c3,
             graph::shared_leaf<T> pres_scale,
             graph::shared_leaf<T> rmin,
             graph::shared_leaf<T> dr,
             graph::shared_leaf<T> zmin,
             graph::shared_leaf<T> dz,
             graph::shared_leaf<T> fpol_c0,
             graph::shared_leaf<T> fpol_c1,
             graph::shared_leaf<T> fpol_c2,
             graph::shared_leaf<T> fpol_c3,
             graph::shared_leaf<T> c00,
             graph::shared_leaf<T> c01,
             graph::shared_leaf<T> c02,
             graph::shared_leaf<T> c03,
             graph::shared_leaf<T> c10,
             graph::shared_leaf<T> c11,
             graph::shared_leaf<T> c12,
             graph::shared_leaf<T> c13,
             graph::shared_leaf<T> c20,
             graph::shared_leaf<T> c21,
             graph::shared_leaf<T> c22,
             graph::shared_leaf<T> c23,
             graph::shared_leaf<T> c30,
             graph::shared_leaf<T> c31,
             graph::shared_leaf<T> c32,
             graph::shared_leaf<T> c33) :
        generic<T> ({3.34449469E-27} ,{1}),
        psimin(psimin), dpsi(dpsi),
        te_c0(te_c0), te_c1(te_c1), te_c2(te_c2), te_c3(te_c3), te_scale(te_scale),
        ne_c0(te_c0), ne_c1(te_c1), ne_c2(ne_c2), ne_c3(ne_c3), ne_scale(ne_scale),
        pres_c0(pres_c0), pres_c1(pres_c1), pres_c2(pres_c2), pres_c3(pres_c3), pres_scale(pres_scale),
        rmin(rmin), dr(dr), zmin(zmin), dz(dz),
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
        graph::shared_leaf<T> get_psi(graph::shared_leaf<T> x,
                                      graph::shared_leaf<T> y,
                                      graph::shared_leaf<T> z) {
            auto r = graph::sqrt(x*x + y*y);
            return get_psi(r, z);
        }

//------------------------------------------------------------------------------
///  @brief Get psi.
///
///  @params[in] r R position.
///  @params[in] z Z position.
///  @returns The psi expression.
//------------------------------------------------------------------------------
        graph::shared_leaf<T> get_psi(graph::shared_leaf<T> r,
                                      graph::shared_leaf<T> z) {
            auto r_norm = (r - rmin)/dr;
            auto z_norm = (z - zmin)/dz;

            auto c00_temp = graph::piecewise_2D(c00, r_norm, z_norm);
            auto c01_temp = graph::piecewise_2D(c01, r_norm, z_norm);
            auto c02_temp = graph::piecewise_2D(c02, r_norm, z_norm);
            auto c03_temp = graph::piecewise_2D(c03, r_norm, z_norm);

            auto c10_temp = graph::piecewise_2D(c10, r_norm, z_norm);
            auto c11_temp = graph::piecewise_2D(c11, r_norm, z_norm);
            auto c12_temp = graph::piecewise_2D(c12, r_norm, z_norm);
            auto c13_temp = graph::piecewise_2D(c13, r_norm, z_norm);

            auto c20_temp = graph::piecewise_2D(c20, r_norm, z_norm);
            auto c21_temp = graph::piecewise_2D(c21, r_norm, z_norm);
            auto c22_temp = graph::piecewise_2D(c22, r_norm, z_norm);
            auto c23_temp = graph::piecewise_2D(c23, r_norm, z_norm);

            auto c30_temp = graph::piecewise_2D(c30, r_norm, z_norm);
            auto c31_temp = graph::piecewise_2D(c31, r_norm, z_norm);
            auto c32_temp = graph::piecewise_2D(c32, r_norm, z_norm);
            auto c33_temp = graph::piecewise_2D(c33, r_norm, z_norm);

            return c00_temp +
                   c01_temp*z_norm +
                   c02_temp*z_norm*z_norm +
                   c03_temp*z_norm*z_norm*z_norm +
                   c10_temp*r_norm +
                   c11_temp*r_norm*z_norm +
                   c12_temp*r_norm*z_norm*z_norm +
                   c13_temp*r_norm*z_norm*z_norm*z_norm +
                   c20_temp*r_norm*r_norm +
                   c21_temp*r_norm*r_norm*z_norm +
                   c22_temp*r_norm*r_norm*z_norm*z_norm +
                   c23_temp*r_norm*r_norm*z_norm*z_norm*z_norm +
                   c30_temp*r_norm*r_norm*r_norm +
                   c31_temp*r_norm*r_norm*r_norm*z_norm +
                   c32_temp*r_norm*r_norm*r_norm*z_norm*z_norm +
                   c33_temp*r_norm*r_norm*r_norm*z_norm*z_norm*z_norm;
        }

//------------------------------------------------------------------------------
///  @brief Get the electron density.
///
///  @params[in] x X position.
///  @params[in] y Y position.
///  @params[in] z Z position.
///  @returns The electron density expression.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> get_electron_density(graph::shared_leaf<T> x,
                                                           graph::shared_leaf<T> y,
                                                           graph::shared_leaf<T> z) {
            auto psi = get_psi(x, y, z);
            auto psi_norm = (psi - psimin)/dpsi;

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
        virtual graph::shared_leaf<T> get_ion_density(const size_t index,
                                                      graph::shared_leaf<T> x,
                                                      graph::shared_leaf<T> y,
                                                      graph::shared_leaf<T> z) {
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
        graph::shared_leaf<T> get_pressure(graph::shared_leaf<T> x,
                                           graph::shared_leaf<T> y,
                                           graph::shared_leaf<T> z) {
            auto psi = get_psi(x, y, z);
            auto psi_norm = (psi - psimin)/dpsi;

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
        virtual graph::shared_leaf<T> get_electron_temperature(graph::shared_leaf<T> x,
                                                               graph::shared_leaf<T> y,
                                                               graph::shared_leaf<T> z) {
            auto psi = get_psi(x, y, z);
            auto psi_norm = (psi - psimin)/dpsi;

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
        virtual graph::shared_leaf<T> get_ion_temperature(const size_t index,
                                                          graph::shared_leaf<T> x,
                                                          graph::shared_leaf<T> y,
                                                          graph::shared_leaf<T> z) {
            auto pressure = get_pressure(x, y, z);
            auto q = graph::constant(static_cast<T> (1.60218E-19));
            return (pressure - get_electron_density(x, y, z)*get_electron_temperature(x, y, z)/q) /
                   (get_ion_density(index, x, y, z) + graph::constant(static_cast<T> (1.0E-100)));
        }

//------------------------------------------------------------------------------
///  @brief Get the toroidal magnetic field.
///
///  @params[in] r R position.
///  @returns The toroidal magnetic field expression.
//------------------------------------------------------------------------------
        graph::shared_leaf<T> get_b_phi(graph::shared_leaf<T> r) {
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
        virtual graph::shared_vector<T>
        get_magnetic_field(graph::shared_leaf<T> x,
                           graph::shared_leaf<T> y,
                           graph::shared_leaf<T> z) {
            auto r = graph::sqrt(x*x + y*y);
            auto phi = graph::atan(x, y);
            auto none = graph::none<T> ();
            auto psi = get_psi(x, y, z);

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
///  @params[in] spline_file File name of contains the spline functions.
///  @params[in,out] sync    Mutex to ensure the netcdf file is read only by one
///                          thread.
///  @returns A constructed EFIT equilibrium.
//------------------------------------------------------------------------------
    template<typename T>
    shared<T> make_efit(const std::string spline_file,
                        std::mutex &sync) {
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

        auto rmin = graph::constant(static_cast<T> (rmin_value));
        auto dr = graph::constant(static_cast<T> (dr_value));
        auto zmin = graph::constant(static_cast<T> (zmin_value));
        auto dz = graph::constant(static_cast<T> (dz_value));
        auto psimin = graph::constant(static_cast<T> (psimin_value));
        auto dpsi = graph::constant(static_cast<T> (dpsi_value));
        auto pres_scale = graph::constant(static_cast<T> (pres_scale_value));
        auto ne_scale = graph::constant(static_cast<T> (ne_scale_value));
        auto te_scale = graph::constant(static_cast<T> (te_scale_value));

        auto fpol_c0 = graph::piecewise_1D(std::vector<T> (fpol_c0_buffer.begin(), fpol_c0_buffer.end()));
        auto fpol_c1 = graph::piecewise_1D(std::vector<T> (fpol_c1_buffer.begin(), fpol_c1_buffer.end()));
        auto fpol_c2 = graph::piecewise_1D(std::vector<T> (fpol_c2_buffer.begin(), fpol_c2_buffer.end()));
        auto fpol_c3 = graph::piecewise_1D(std::vector<T> (fpol_c3_buffer.begin(), fpol_c3_buffer.end()));

        auto c00 = graph::piecewise_2D(std::vector<T> (psi_c00_buffer.begin(), psi_c00_buffer.end()), numz);
        auto c01 = graph::piecewise_2D(std::vector<T> (psi_c01_buffer.begin(), psi_c01_buffer.end()), numz);
        auto c02 = graph::piecewise_2D(std::vector<T> (psi_c02_buffer.begin(), psi_c02_buffer.end()), numz);
        auto c03 = graph::piecewise_2D(std::vector<T> (psi_c03_buffer.begin(), psi_c03_buffer.end()), numz);
        auto c10 = graph::piecewise_2D(std::vector<T> (psi_c10_buffer.begin(), psi_c10_buffer.end()), numz);
        auto c11 = graph::piecewise_2D(std::vector<T> (psi_c11_buffer.begin(), psi_c11_buffer.end()), numz);
        auto c12 = graph::piecewise_2D(std::vector<T> (psi_c12_buffer.begin(), psi_c12_buffer.end()), numz);
        auto c13 = graph::piecewise_2D(std::vector<T> (psi_c13_buffer.begin(), psi_c13_buffer.end()), numz);
        auto c20 = graph::piecewise_2D(std::vector<T> (psi_c20_buffer.begin(), psi_c20_buffer.end()), numz);
        auto c21 = graph::piecewise_2D(std::vector<T> (psi_c21_buffer.begin(), psi_c21_buffer.end()), numz);
        auto c22 = graph::piecewise_2D(std::vector<T> (psi_c22_buffer.begin(), psi_c22_buffer.end()), numz);
        auto c23 = graph::piecewise_2D(std::vector<T> (psi_c23_buffer.begin(), psi_c23_buffer.end()), numz);
        auto c30 = graph::piecewise_2D(std::vector<T> (psi_c30_buffer.begin(), psi_c30_buffer.end()), numz);
        auto c31 = graph::piecewise_2D(std::vector<T> (psi_c31_buffer.begin(), psi_c31_buffer.end()), numz);
        auto c32 = graph::piecewise_2D(std::vector<T> (psi_c32_buffer.begin(), psi_c32_buffer.end()), numz);
        auto c33 = graph::piecewise_2D(std::vector<T> (psi_c33_buffer.begin(), psi_c33_buffer.end()), numz);

        auto pres_c0 = graph::piecewise_1D(std::vector<T> (pressure_c0_buffer.begin(), pressure_c0_buffer.end()));
        auto pres_c1 = graph::piecewise_1D(std::vector<T> (pressure_c1_buffer.begin(), pressure_c1_buffer.end()));
        auto pres_c2 = graph::piecewise_1D(std::vector<T> (pressure_c2_buffer.begin(), pressure_c2_buffer.end()));
        auto pres_c3 = graph::piecewise_1D(std::vector<T> (pressure_c3_buffer.begin(), pressure_c3_buffer.end()));

        auto te_c0 = graph::piecewise_1D(std::vector<T> (te_c0_buffer.begin(), te_c0_buffer.end()));
        auto te_c1 = graph::piecewise_1D(std::vector<T> (te_c1_buffer.begin(), te_c1_buffer.end()));
        auto te_c2 = graph::piecewise_1D(std::vector<T> (te_c2_buffer.begin(), te_c2_buffer.end()));
        auto te_c3 = graph::piecewise_1D(std::vector<T> (te_c3_buffer.begin(), te_c3_buffer.end()));

        auto ne_c0 = graph::piecewise_1D(std::vector<T> (ne_c0_buffer.begin(), ne_c0_buffer.end()));
        auto ne_c1 = graph::piecewise_1D(std::vector<T> (ne_c1_buffer.begin(), ne_c1_buffer.end()));
        auto ne_c2 = graph::piecewise_1D(std::vector<T> (ne_c2_buffer.begin(), ne_c2_buffer.end()));
        auto ne_c3 = graph::piecewise_1D(std::vector<T> (ne_c3_buffer.begin(), ne_c3_buffer.end()));

        return std::make_shared<efit<T>> (psimin, dpsi,
                                          te_c0, te_c1, te_c2, te_c3, te_scale,
                                          ne_c0, ne_c1, ne_c2, ne_c3, ne_scale,
                                          pres_c0, pres_c1, pres_c2, pres_c3, pres_scale,
                                          rmin, dr, zmin, dz,
                                          fpol_c0, fpol_c1, fpol_c2, fpol_c3,
                                          c00, c01, c02, c03,
                                          c10, c11, c12, c13,
                                          c20, c21, c22, c23,
                                          c30, c31, c32, c33);
    }
}

#endif /* equilibrium_h */
