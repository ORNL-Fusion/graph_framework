//------------------------------------------------------------------------------
///  @file equilibrium.hpp
///  @brief Class signature to implement plasma equilibrium.
///
///  Defined the interfaces to access plasma equilibrium.
//------------------------------------------------------------------------------
///  @page equilibrium_models Equilibrium Models
///  @brief Documentation for formatting equilibrium files.
///  @tableofcontents
///
///  @section equilibrium_models_intro Introduction
///  This page documents the types and formatting of the equilibrium models.
///  xrays currently supports two equilibrium models.
///  * EFIT Are 2D axisymmetric equilibria relevant to tokamak devices.
///  * VMEC Are 3D nested flux surface equilibria relevant for stellarator devices.
///
///  This documentation assumes the user has some familiarity with EFIT or VMEC
///  and focuses instead how quantities from these are formatted.
///
///  @section equilibrium_splines Spline Formatting
///  The equilibrium models used in this section make use of Cubic and Bicubic
///  splines.
///
///  <hr>
///  @subsection equilibrium_splines_1D Cubic Splines
///  Cubic splines are 1D interpolation functions consisting of 4 coefficient
///  arrays. They take the form of
///  @f{equation}{y\left(x\right)=C_{0} + C_{1}x + C_{2}x^2 + C_{3}x^2@f}
///  where @f$x@f$ is a normalized radial index. Cubic splines coefficients can
///  be calculated using
///  <a href="https://mathworld.wolfram.com/CubicSpline.html">Linear Solvers</a>
///  However, to avoid needing account for index offsets, index offsets are pre
///  computed into the spline coefficients.
///  @f{equation}{C'^{i}_{0}=C^{i}_{0} - C^{i}_{1}i + C^{i}_{2}i^2 - C^{i}_{3}i^3@f}
///  @f{equation}{C'^{i}_{1}=C^{i}_{1} -2C^{i}_{2}i + 3C^{i}_{3}i^2@f}
///  @f{equation}{C'^{i}_{2}=C^{i}_{2} - 3C^{i}_{3}i@f}
///  @f{equation}{C'^{i}_{3}=C^{i}_{3}@f}
///  Where @f$i@f$ is the index of the coefficient array. This allows to
///  normalize the spline argument @f$x@f$ so that it can both index the array
///  and evaluate the spline.
///  @f{equation}{x = \frac{x_{real} - x_{min}}{dx}@f}
///  Rounding down the value of @f$x@f$ gives the correct coefficient index.
///
///  <hr>
///  @subsection equilibrium_splines_2D Bicubic Splines
///  Bicubic Splines are computed in a similar way instead they consist of a
///  total of 16 coefficients. These represent 4 spline functions in one
///  dimension which interpolate 4 coefficient values for the other dimension.
///  Like the 1D splines, 2D spline coefficients are normalized so the spline
///  arguments can be used as normalized indices.
///  @f{equation}{C'^{ij}_{00}=C^{ij}_{00}-C^{ij}_{01}j+C^{ij}_{02}j^{2}-C^{ij}_{03}j^{3}-C^{ij}_{10}i+C^{ij}_{11}ij-C^{ij}_{12}ij^{2}+C^{ij}_{13}ij^{3}+C^{ij}_{20}i^{2}-C^{ij}_{21}i^{2}j+C^{ij}_{22}i^{2}j^{2}-C^{ij}_{23}i^{2}j^{3}-C^{ij}_{30}i^{3}+C^{ij}_{31}i^{3}j-C^{ij}_{32}i^{3}j^{2}+C^{ij}_{33}i^{3}j^{3}j@f}
///  @f{equation}{C'^{ij}_{01}=C^{ij}_{01}-2C^{ij}_{02}j+3C^{ij}_{03}j^{2}-C^{ij}_{11}i+2C^{ij}_{12}ij-3C^{ij}_{13}ij^{2}+C^{ij}_{21}i^{2}-2C^{ij}_{22}i^{2}j+3C^{ij}_{23}i^{2}j^{2}-C^{ij}_{31}i^{3}+2C^{ij}_{32}i^{3}j-3C^{ij}_{33}i^{3}j^{2}@f}
///  @f{equation}{C'^{ij}_{02}=C^{ij}_{02}-3C^{ij}_{03}j-C^{ij}_{12}i+3C^{ij}_{13}ij+C^{22}i^{2}-3C^{ij}_{23}i^{2}j-C^{ij}_{32}i^{3}+3C^{ij}i^{3}j @f}
///  @f{equation}{C'^{ij}_{03}=C^{ij}_{03}-C^{ij}_{13}i+C^{ij}_{23}i^{2}-C^{ij}_{33}i^{3}@f}
///  @f{equation}{C'^{ij}_{10}=C^{ij}_{10}-2C^{ij}_{11}j+C^{ij}_{12}j^{2}-C^{ij}_{13}j^{3}-2C^{ij}_{20}i+2C^{ij}_{21}ij-2C^{ij}_{22}ij^{2}+2C^{ij}_{23}ij^{3}j+3C^{ij}_{30}i^{2}-3C^{ij}_{31}i^{2}j+3C^{ij}_{32}i^{2}j^{2}-3C^{ij}_{33}i^{2}j^{3}@f}
///  @f{equation}{C'^{ij}_{11}=C^{ij}_{11}-2C^{ij}_{12}j+3C^{ij}_{13}j^{2}-2C^{ij}_{21}i+4C^{ij}_{22}ij-6C^{ij}_{23}ij^{2}+3C^{ij}_{31}i^{2}-6C^{ij}_{32}i^{2}j+9C^{ij}_{33}i^{2}j^{2}@f}
///  @f{equation}{C'^{ij}_{12}=C^{ij}_{12}-C^{ij}_{13}j-2C^{ij}_{22}i+6C^{ij}_{23}ij+3C^{ij}_{32}j-9C^{ij}_{33}i^{2}j @f}
///  @f{equation}{C'^{ij}_{13}=C^{ij}_{13}-2C^{ij}_{23}i+3C^{ij}_{33}i^{2}@f}
///  @f{equation}{C'^{ij}_{20}=C^{ij}_{20}-C^{ij}_{21}j+C^{ij}_{22}ij^{2}-C^{ij}_{23}j^{3}-3C^{30}i+3C^{ij}_{31}ij-3C^{ij}_{32}ij^{2}+3C^{ij}_{33}ij^{3}@f}
///  @f{equation}{C'^{ij}_{21}=C^{ij}_{21}-2C^{ij}_{22}j+3C^{ij}_{23}j^{2}-3C^{ij}_{31}i+6C^{32}ij-9C^{ij}_{33}ij^{2}@f}
///  @f{equation}{C'^{ij}_{22}=C^{ij}_{22}-3C^{ij}_{23}j-3C^{ij}_{32}i+9C^{ij}_{33}ij @f}
///  @f{equation}{C'^{ij}_{23}=C^{ij}_{23}-3C^{ij}_{33}i @f}
///  @f{equation}{C'^{ij}_{30}=C^{ij}_{30}-C^{ij}_{31}j+C^{ij}_{32}j^{2}-C^{ij}_{33}j^{3}@f}
///  @f{equation}{C'^{ij}_{31}=C^{ij}_{31}-2C^{ij}_{32}j+3C^{ij}_{32}j^{2}@f}
///  @f{equation}{C'^{ij}_{32}=C^{ij}_{32}-3C^{ij}_{33}j @f}
///  @f{equation}{C'^{ij}_{33}=C^{ij}_{33} @f}
///  Bicubic splines are computed by
///  @f{equation}{f\left(x,y\right)=\left(\begin{array}{cccc}1 & x & x^{2} & x^{3}\end{array}\right)\cdot\left(\left(\begin{array}{cccc}C_{00}&C_{01}&C_{02}&C_{03}\\C_{10}&C_{11}&C_{12}&C_{13}\\C_{20}&C_{21}&C_{22}&C_{23}\\C_{30}&C_{31}&C_{32}&C_{33}\end{array}\right)\cdot\left(\begin{array}{c}1\\y\\y^{2}\\y^{3}\end{array}\right)\right)@f}
///  Like the 1D splines @f$x@f$ and @f$y@f$ are normalized.
///  @f{equation}{x = \frac{x_{real} - x_{min}}{dx}@f}
///  @f{equation}{y = \frac{y_{real} - y_{min}}{dy}@f}
///
///  <hr>
///  @section equilibrium_efit EFIT
///  @image{} html Efit.png "Cross section of poloidal flux surfaces."
///  EFIT is an equilibrium that comes from a solution of the
///  <a href="https://en.wikipedia.org/wiki/Grad–Shafranov_equation">Grad–Shafranov equation</a>.
///  The solution gives us a map of the poloidal flux @f$\psi@f$ on 2D grid and
///  a 1D flux function @f$f_{pol}@f$. 1D profiles of electron density
///  @f$n_{e}\left(\psi\right)@f$, electron temperature
///  @f$t_{e}\left(\psi\right)@f$, and pressure @f$p\left(\psi\right)@f$ are
///  mapped as functions of the normalized flux.
///
///  @subsection equilibrium_efit_format EFIT file format
///  Quantities are loaded into the ray tracer via a netcdf file. EFIT NetCDF
///  files must contain the following quantities. Spline quantities have a
///  common format of <i>name</i>_c<i>i</i> or <i>name</i>_c<i>ij</i>.
///  <table>
///  <caption id="equilibrium_efit_format_data">Efit netcdf file quantities</caption>
///  <tr><th colspan="3">Dimensions
///  <tr><th colspan="2">Name                                       <th>Discription
///  <tr><td colspan="2"><tt>numr</tt>                              <td>Size of radial grid.
///  <tr><td colspan="2"><tt>numz</tt>                              <td>Size of vertical grid.
///  <tr><td colspan="2"><tt>numpsi</tt>                            <td>Size of arrays for @f$\psi@f$ mapped quantities.
///  <tr><th colspan="3">Scalar Quantities
///  <tr><td colspan="2"><tt>dpsi</tt>                              <td>Step size of the @f$\psi@f$ grid.
///  <tr><td colspan="2"><tt>dr</tt>                                <td>Step size of the radial grid.
///  <tr><td colspan="2"><tt>dz</tt>                                <td>Step size of the vertical grid.
///  <tr><td colspan="2"><tt>ne_scale</tt>                          <td>Scale of the @f$n_{e}@f$ profile.
///  <tr><td colspan="2"><tt>pres_scale</tt>                        <td>Scale of the pressure profile.
///  <tr><td colspan="2"><tt>psibry</tt>                            <td>Value of @f$\psi@f$ at the boundary.
///  <tr><td colspan="2"><tt>psimin</tt>                            <td>Minimum @f$\psi@f$ value.
///  <tr><td colspan="2"><tt>rmin</tt>                              <td>Minimum radial value.
///  <tr><td colspan="2"><tt>te_scale</tt>                          <td>Scale of the electron temperature profile.
///  <tr><td colspan="2"><tt>zmin</tt>                              <td>Minimum vertical value.
///  <tr><th colspan="3">1D Quantities
///  <tr><th>Name<th>Size                                           <th>Description
///  <tr><td><tt>fpol_c<i>i</i></tt>        <td><tt>numpsi</tt>     <td>Flux function profile coefficients
///  <tr><td><tt>ne_c<i>i</i></tt>          <td><tt>numpsi</tt>     <td>@f$n_{e}@f$ profile coefficients.
///  <tr><td><tt>pressure_c<i>i</i></tt>    <td><tt>numpsi</tt>     <td>Pressure profile coefficients.
///  <tr><td><tt>te_c<i>i</i></tt>          <td><tt>numpsi</tt>     <td>@f$t_{e}@f$ profile coefficients.
///  <tr><th colspan="3">2D Quantities
///  <tr><th>Name<th>Size                                           <th>Description
///  <tr><td><tt>psi_c<i>ij</i></tt>        <td><tt>(numr,numz)</tt><td>@f$\psi\left(r,z\right)@f$ coefficients.
///  </table>
///
///  <hr>
///  @section equilibrium_vmec VMEC
///  @image{} html vmec.png "Cross section of 3D flux surfaces."
///  VMEC is an equilibrium that comes from
///  <a href="https://doi.org/10.1063/1.864116">minimizing mhd energy</a>.
///  The solution gives us set of Fourier coefficients on a discrete radial
///  grid. 1D profiles of electron density
///  @f$n_{e}\left(\psi\right)@f$, electron temperature
///  @f$t_{e}\left(\psi\right)@f$, and pressure @f$p\left(\psi\right)@f$ are
///  mapped as functions of the normalized flux.
///
///  @subsection equilibrium_vmec_format VMEC file format
///  Quantities are loaded into the ray tracer via a netcdf file. VMEC NetCDF
///  files must contain the following quantities. Spline quantities have a
///  common format of <i>name</i>_c<i>i</i>. All radial quantities are splined
///  accross the magnetic axis to the opposite end. That is quantities extend
///  from @f$-s\rightarrow s @f$. Splines of fourier coeffients are one
///  dimensional splines stored in a 2D array. Radial quantities are stored as a
///  full or half grid value.
///  <table>
///  <caption id="equilibrium_vmec_format_data">VMEC netcdf file quantities</caption>
///  <tr><th colspan="3">Dimensions
///  <tr><th colspan="2">Name                                    <th>Description
///  <tr><td colspan="2"><tt>numsf</tt>                          <td>Size of full radial grid.
///  <tr><td colspan="2"><tt>numsh</tt>                          <td>Size of half radial grid.
///  <tr><td colspan="2"><tt>nummn</tt>                          <td>Number of Fourier modes.
///  <tr><th colspan="3">Scalar Quantities
///  <tr><td colspan="2"><tt>dphi</tt>                           <td>Step size of toroidal flux.
///  <tr><td colspan="2"><tt>ds</tt>                             <td>Step size of normalized toroidal flux.
///  <tr><td colspan="2"><tt>signj</tt>                          <td>Sign of the Jacobian.
///  <tr><td colspan="2"><tt>sminf</tt>                          <td>Minimum @f$s @f$ on the full grid.
///  <tr><td colspan="2"><tt>sminh</tt>                          <td>Minimum @f$s @f$ on the half grid.
///  <tr><th colspan="3">1D Quantities
///  <tr><th>Name                      <th>Size                  <th>Description
///  <tr><td><tt>chi_c<i>i</i></tt>    <td><tt>numsf</tt>        <td>Poloidal flux profile.
///  <tr><td><tt>xm</tt>               <td><tt>nummn</tt>        <td>Poloidal modes.
///  <tr><td><tt>xn</tt>               <td><tt>nummn</tt>        <td>Toroidal modes.
///  <tr><th colspan="3">2D Quantities
///  <tr><th>Name                      <th>Size                  <th>Description
///  <tr><td><tt>lmns_c<i>i</i></tt>   <td><tt>(numsh,nummn)</tt><td>@f$\lambda @f$ fourier coefficients.
///  <tr><td><tt>rmnc_c<i>i</i></tt>   <td><tt>(numsf,nummn)</tt><td>@f$r @f$ fourier coefficients.
///  <tr><td><tt>zmns_c<i>i</i></tt>   <td><tt>(numsf,nummn)</tt><td>@f$z @f$ fourier coefficients.
///  </table>
///
///  <hr>
///  @section equilibrium_devel Developing new equilibrium models
///  This section is intended for code developers and outlines how to create new
///  equilibrium models. All equilibrium model use the same
///  @ref equilibrium::generic interface. New equilibrium models can be created
///  from a subclass of @ref equilibrium::generic or any other existing
///  equilibrium class and overloading class methods.
///  @code
///  template<jit::float_scalar T, bool SAFE_MATH=false>
///  class new_equilibrium final : public generic<T, SAFE_MATH> {
///     ...
///  };
///  @endcode
///
///  When a new equilibrium is
///  subclassed from @ref equilibrium::generic implementations must be provided
///  for the following pure virtual methods.
///  * @ref equilibrium::generic::get_characteristic_field
///  * @ref equilibrium::generic::get_electron_density
///  * @ref equilibrium::generic::get_electron_temperature
///  * @ref equilibrium::generic::get_ion_density
///  * @ref equilibrium::generic::get_ion_temperature
///  * @ref equilibrium::generic::get_magnetic_field
///
///  @note @ref equilibrium::generic::get_characteristic_field is only used by
///  the normalized boris method for particle pushing. For most cases this can
///  simply return 1.
///
///  For the remaining methods, or any other methods one wishes to override, the
///  arguments provide expressions for the input position of the ray. The
///  methods return are expressions for the quantity at hand.
///
///  @subsection equilibrium_devel_coordinate Non-cartesian Coordinates
///  While these methods take an @f$x,y,z @f$ as the argument names, there is
///  no reason these need to be assumed to be cartesian coordinates. For
///  instance the @ref equilibrium_vmec treats @f$x,y,z\rightarrow s,u,v @f$ as
///  flux coordinates. In flux coordinate the coordinate system is no longer
///  normalized nor orthogonal. So that other parts of the code can treat
///  @f$\vec{k}@f$ correctly there are methods to return the covariant basis
///  vectors @f$\vec{e}_{s},\vec{e}_{u},\vec{e}_{v}@f$.
///  * @ref equilibrium::generic::get_esup1
///  * @ref equilibrium::generic::get_esup2
///  * @ref equilibrium::generic::get_esup3
///
///  By default, @ref equilibrium::generic return basis vectors for a cartesian
///  system @f$\vec{e}_{1}=\hat{x},\vec{e}_{2}=\hat{y},\vec{e}_{3}=\hat{z}@f$.
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

///  Name space for equilibrium models.
namespace equilibrium {
///  Lock to synchronize netcdf across threads.
    static std::mutex sync;

//******************************************************************************
//  Equilibrium interface
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a generic equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
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
///  @brief Construct a generic equilibrium.
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
///  @param[in] device_number Device to use.
///  @returns The characteristic field.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_characteristic_field(const size_t device_number=0) = 0;

//------------------------------------------------------------------------------
///  @brief Get the contravariant basis vector in the x1 direction.
///
///  @param[in] x1 X1 position.
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
///  @param[in] x1 X1 position.
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
///  @param[in] x1 X1 position.
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
///  @param[in] x1 X1 position.
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
///  @param[in] x1 X1 position.
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
///  @param[in] x1 X1 position.
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
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
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
///  @param[in] device_number Device to use.
///  @returns The characteristic field.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_characteristic_field(const size_t device_number=0) final {
            return graph::one<T, SAFE_MATH> ();
        }
    };

//------------------------------------------------------------------------------
///  @brief Convenience function to build a no magnetic field equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
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
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class slab : public generic<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a gaussian density with uniform magnetic field.
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
///  @param[in] device_number Device to use.
///  @returns The characteristic field.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_characteristic_field(const size_t device_number=0) final {
            return graph::one<T, SAFE_MATH> ();
        }
    };

//------------------------------------------------------------------------------
///  @brief Convenience function to build a slab equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
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
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class slab_density : public generic<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a gaussian density with uniform magnetic field.
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
///  @param[in] device_number Device to use.
///  @returns The characteristic field.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_characteristic_field(const size_t device_number=0) final {
            return graph::one<T, SAFE_MATH> ();
        }
    };

//------------------------------------------------------------------------------
///  @brief Convenience function to build a slab density equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
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
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class slab_field : public generic<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a gaussian density with uniform magnetic field.
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
///  @param[in] device_number Device to use.
///  @returns The characteristic field.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_characteristic_field(const size_t device_number=0) final {
            return graph::one<T, SAFE_MATH> ();
        }
    };

//------------------------------------------------------------------------------
///  @brief Convenience function to build a slab density equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @returns A constructed slab density equilibrium.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared<T, SAFE_MATH> make_slab_field() {
        return std::make_shared<slab_field<T, SAFE_MATH>> ();
    }
//******************************************************************************
//  Gaussian density with a uniform magnetic field.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Gaussian density with uniform magnetic field equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class gaussian_density : public generic<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Construct a gaussian density with uniform magnetic field.
//------------------------------------------------------------------------------
        gaussian_density() :
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
///  @param[in] device_number Device to use.
///  @returns The characteristic field.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_characteristic_field(const size_t device_number=0) final {
            return graph::one<T, SAFE_MATH> ();
        }
    };

//------------------------------------------------------------------------------
///  @brief Convenience function to build a gaussian density equilibrium.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @returns A constructed gaussian density equilibrium.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared<T, SAFE_MATH> make_gaussian_density() {
        return std::make_shared<gaussian_density<T, SAFE_MATH>> ();
    }

//------------------------------------------------------------------------------
///  @brief Build a 1D spline.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] c      Array of spline coefficients.
///  @param[in] x      Spline argument.
///  @param[in] scale  Scale factor for argument.
///  @param[in] offset Offset value for argument.
///  @returns The graph expression for a 1D spline.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    graph::shared_leaf<T, SAFE_MATH> build_1D_spline(graph::output_nodes<T, SAFE_MATH> c,
                                                     graph::shared_leaf<T, SAFE_MATH> x,
                                                     const T scale,
                                                     const T offset) {
        auto c3 = c[3]/(scale*scale*scale);
        auto c2 = c[2]/(scale*scale) - static_cast<T> (3.0)*offset*c[3]/(scale*scale*scale);
        auto c1 = c[1]/scale - static_cast<T> (2.0)*offset*c[2]/(scale*scale) + static_cast<T> (3.0)*offset*offset*c[3]/(scale*scale*scale);
        auto c0 = c[0] - offset*c[1]/scale + offset*offset*c[2]/(scale*scale) - offset*offset*offset*c[3]/(scale*scale*scale);

        return graph::fma(graph::fma(graph::fma(c3, x, c2), x, c1), x, c0);
    }

//******************************************************************************
//  2D EFIT equilibrium.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief 2D EFIT equilibrium.
///
///  This takes a BiCublic spline representation of the psi and cubic splines
///  for ne, te, p, and fpol.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class efit final : public generic<T, SAFE_MATH> {
    private:
///  Minimum psi.
        const T psimin;
///  Psi grid spacing.
        const T dpsi;

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
        const T rmin;
///  R grid spacing.
        const T dr;
///  Minimum Z.
        const T zmin;
///  Z grid spacing.
        const T dz;

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

///  Cached magnetic flux.
        graph::shared_leaf<T, SAFE_MATH> psi_cache;

//------------------------------------------------------------------------------
///  @brief Build psi.
///
///  @param[in] r        The normalized radial position.
///  @param[in] r_scale  Scale factor for r.
///  @param[in] r_offset Offset factor for r.
///  @param[in] z The normalized z position.
///  @param[in] z_scale  Scale factor for z.
///  @param[in] z_offset Offset factor for z.
///  @returns The psi value.
//------------------------------------------------------------------------------
        graph::shared_leaf<T, SAFE_MATH>
        build_psi(graph::shared_leaf<T, SAFE_MATH> r,
                  const T r_scale,
                  const T r_offset,
                  graph::shared_leaf<T, SAFE_MATH> z,
                  const T z_scale,
                  const T z_offset) {
            auto c00_temp = graph::piecewise_2D(c00, num_cols, r, r_scale, r_offset, z, z_scale, z_offset);
            auto c01_temp = graph::piecewise_2D(c01, num_cols, r, r_scale, r_offset, z, z_scale, z_offset);
            auto c02_temp = graph::piecewise_2D(c02, num_cols, r, r_scale, r_offset, z, z_scale, z_offset);
            auto c03_temp = graph::piecewise_2D(c03, num_cols, r, r_scale, r_offset, z, z_scale, z_offset);

            auto c10_temp = graph::piecewise_2D(c10, num_cols, r, r_scale, r_offset, z, z_scale, z_offset);
            auto c11_temp = graph::piecewise_2D(c11, num_cols, r, r_scale, r_offset, z, z_scale, z_offset);
            auto c12_temp = graph::piecewise_2D(c12, num_cols, r, r_scale, r_offset, z, z_scale, z_offset);
            auto c13_temp = graph::piecewise_2D(c13, num_cols, r, r_scale, r_offset, z, z_scale, z_offset);

            auto c20_temp = graph::piecewise_2D(c20, num_cols, r, r_scale, r_offset, z, z_scale, z_offset);
            auto c21_temp = graph::piecewise_2D(c21, num_cols, r, r_scale, r_offset, z, z_scale, z_offset);
            auto c22_temp = graph::piecewise_2D(c22, num_cols, r, r_scale, r_offset, z, z_scale, z_offset);
            auto c23_temp = graph::piecewise_2D(c23, num_cols, r, r_scale, r_offset, z, z_scale, z_offset);

            auto c30_temp = graph::piecewise_2D(c30, num_cols, r, r_scale, r_offset, z, z_scale, z_offset);
            auto c31_temp = graph::piecewise_2D(c31, num_cols, r, r_scale, r_offset, z, z_scale, z_offset);
            auto c32_temp = graph::piecewise_2D(c32, num_cols, r, r_scale, r_offset, z, z_scale, z_offset);
            auto c33_temp = graph::piecewise_2D(c33, num_cols, r, r_scale, r_offset, z, z_scale, z_offset);

            auto r_norm = (r - r_offset)/r_scale;

            auto c0 = build_1D_spline({c00_temp, c01_temp, c02_temp, c03_temp}, z, z_scale, z_offset);
            auto c1 = build_1D_spline({c10_temp, c11_temp, c12_temp, c13_temp}, z, z_scale, z_offset);
            auto c2 = build_1D_spline({c20_temp, c21_temp, c22_temp, c23_temp}, z, z_scale, z_offset);
            auto c3 = build_1D_spline({c30_temp, c31_temp, c32_temp, c33_temp}, z, z_scale, z_offset);

            return ((c3*r_norm + c2)*r_norm + c1)*r_norm + c0;
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

                psi_cache = build_psi(r, dr, rmin, z, dz, zmin);

                auto n0_temp = graph::piecewise_1D(ne_c0, psi_cache, dpsi, psimin);
                auto n1_temp = graph::piecewise_1D(ne_c1, psi_cache, dpsi, psimin);
                auto n2_temp = graph::piecewise_1D(ne_c2, psi_cache, dpsi, psimin);
                auto n3_temp = graph::piecewise_1D(ne_c3, psi_cache, dpsi, psimin);

                ne_cache = ne_scale*build_1D_spline({n0_temp, n1_temp, n2_temp, n3_temp}, psi_cache, dpsi, psimin);

                auto t0_temp = graph::piecewise_1D(te_c0, psi_cache, dpsi, psimin);
                auto t1_temp = graph::piecewise_1D(te_c1, psi_cache, dpsi, psimin);
                auto t2_temp = graph::piecewise_1D(te_c2, psi_cache, dpsi, psimin);
                auto t3_temp = graph::piecewise_1D(te_c3, psi_cache, dpsi, psimin);

                te_cache = te_scale*build_1D_spline({t0_temp, t1_temp, t2_temp, t3_temp}, psi_cache, dpsi, psimin);

                auto p0_temp = graph::piecewise_1D(pres_c0, psi_cache, dpsi, psimin);
                auto p1_temp = graph::piecewise_1D(pres_c1, psi_cache, dpsi, psimin);
                auto p2_temp = graph::piecewise_1D(pres_c2, psi_cache, dpsi, psimin);
                auto p3_temp = graph::piecewise_1D(pres_c3, psi_cache, dpsi, psimin);

                auto pressure = pres_scale*build_1D_spline({p0_temp, p1_temp, p2_temp, p3_temp}, psi_cache, dpsi, psimin);

                auto q = graph::constant<T, SAFE_MATH> (static_cast<T> (1.60218E-19));

                ni_cache = te_cache;
                ti_cache = (pressure - ne_cache*te_cache*q)/(ni_cache*q);
                
                auto phi = graph::atan(x, y);

                auto br = psi_cache->df(z)/r;

                auto b0_temp = graph::piecewise_1D(fpol_c0, psi_cache, dpsi, psimin);
                auto b1_temp = graph::piecewise_1D(fpol_c1, psi_cache, dpsi, psimin);
                auto b2_temp = graph::piecewise_1D(fpol_c2, psi_cache, dpsi, psimin);
                auto b3_temp = graph::piecewise_1D(fpol_c3, psi_cache, dpsi, psimin);
                
                auto bp = build_1D_spline({b0_temp, b1_temp, b2_temp, b3_temp}, psi_cache, dpsi, psimin)/r;

                auto bz = -psi_cache->df(r)/r;

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
///  @param[in] te_scale   Temperature scale.
///  @param[in] ne_c0      Ne c0 spline coefficient.
///  @param[in] ne_c1      Ne c1 spline coefficient.
///  @param[in] ne_c2      Ne c2 spline coefficient.
///  @param[in] ne_c3      Ne c3 spline coefficient.
///  @param[in] ne_scale   Density scale.
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
        efit(const T psimin,
             const T dpsi,
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
             const T rmin,
             const T dr,
             const T zmin,
             const T dz,
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
///  @param[in] device_number Device to use.
///  @returns The characteristic field.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_characteristic_field(const size_t device_number=0) final {
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

            workflow::manager<T, SAFE_MATH> work(device_number);
            solver::newton(work, {
                x_axis, z_axis
            }, inputs, (psi_cache - psimin)/dpsi, graph::shared_random_state<T, SAFE_MATH> (), static_cast<T> (1.0E-30), 1000, static_cast<T> (0.1));
            work.add_item(inputs, {b_mod}, {},
                          graph::shared_random_state<T, SAFE_MATH> (),
                          "bmod_at_axis", inputs.back()->size());
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
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
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

        std::vector<double> fpol_c0_buffer(numpsi);
        std::vector<double> fpol_c1_buffer(numpsi);
        std::vector<double> fpol_c2_buffer(numpsi);
        std::vector<double> fpol_c3_buffer(numpsi);

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

        auto rmin = static_cast<T> (rmin_value);
        auto dr = static_cast<T> (dr_value);
        auto zmin = static_cast<T> (zmin_value);
        auto dz = static_cast<T> (dz_value);
        auto psimin = static_cast<T> (psimin_value);
        auto dpsi = static_cast<T> (dpsi_value);
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
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class vmec final : public generic<T, SAFE_MATH> {
    private:
///  Minimum s on the half grid.
        const T sminh;
///  Minimum s on the full grid.
        const T sminf;
///  Change in s grid.
        const T ds;
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
///  @param[in] r Radial position.
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
///  @param[in] r Radial position.
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
///  @param[in] r Radial position.
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
///  @param[in] s S position.
///  @returns χ(s,u,v)
//------------------------------------------------------------------------------
        graph::shared_leaf<T, SAFE_MATH>
        get_chi(graph::shared_leaf<T, SAFE_MATH> s) {
            auto c0_temp = graph::piecewise_1D(chi_c0, s, ds, sminf);
            auto c1_temp = graph::piecewise_1D(chi_c1, s, ds, sminf);
            auto c2_temp = graph::piecewise_1D(chi_c2, s, ds, sminf);
            auto c3_temp = graph::piecewise_1D(chi_c3, s, ds, sminf);

            return build_1D_spline({c0_temp, c1_temp, c2_temp, c3_temp}, s, ds, sminf);
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

                auto zero = graph::zero<T, SAFE_MATH> ();
                auto r = zero;
                auto z = zero;
                auto l = zero;

                for (size_t i = 0, ie = xm.size(); i < ie; i++) {
                    auto rmnc_c0_temp = graph::piecewise_1D(rmnc_c0[i], s, ds, sminf);
                    auto rmnc_c1_temp = graph::piecewise_1D(rmnc_c1[i], s, ds, sminf);
                    auto rmnc_c2_temp = graph::piecewise_1D(rmnc_c2[i], s, ds, sminf);
                    auto rmnc_c3_temp = graph::piecewise_1D(rmnc_c3[i], s, ds, sminf);

                    auto zmns_c0_temp = graph::piecewise_1D(zmns_c0[i], s, ds, sminf);
                    auto zmns_c1_temp = graph::piecewise_1D(zmns_c1[i], s, ds, sminf);
                    auto zmns_c2_temp = graph::piecewise_1D(zmns_c2[i], s, ds, sminf);
                    auto zmns_c3_temp = graph::piecewise_1D(zmns_c3[i], s, ds, sminf);

                    auto lmns_c0_temp = graph::piecewise_1D(lmns_c0[i], s, ds, sminh);
                    auto lmns_c1_temp = graph::piecewise_1D(lmns_c1[i], s, ds, sminh);
                    auto lmns_c2_temp = graph::piecewise_1D(lmns_c2[i], s, ds, sminh);
                    auto lmns_c3_temp = graph::piecewise_1D(lmns_c3[i], s, ds, sminh);

                    auto rmnc = build_1D_spline({rmnc_c0_temp, rmnc_c1_temp, rmnc_c2_temp, rmnc_c3_temp},
                                                s, ds, sminf);
                    auto zmns = build_1D_spline({zmns_c0_temp, zmns_c1_temp, zmns_c2_temp, zmns_c3_temp},
                                                s, ds, sminf);
                    auto lmns = build_1D_spline({lmns_c0_temp, lmns_c1_temp, lmns_c2_temp, lmns_c3_temp},
                                                s, ds, sminh);

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
///  @param[in] s S position.
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
///  @param[in] dphi    Change in toroidal flux.
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
        vmec(const T sminh,
             const T sminf,
             const T ds,
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
///  @param[in] s S position.
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
///  @param[in] s S position.
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
///  @param[in] s S position.
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
///  @param[in] s S position.
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
///  @param[in] s S position.
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
///  @param[in] s S position.
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
///  @param[in] s S position.
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
///  @param[in] s S position.
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
///  @param[in] device_number Device to use.
///  @returns The characteristic field.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T, SAFE_MATH>
        get_characteristic_field(const size_t device_number=0) final {
            auto s_axis = graph::zero<T, SAFE_MATH> ();
            auto u_axis = graph::zero<T, SAFE_MATH> ();
            auto v_axis = graph::zero<T, SAFE_MATH> ();
            auto b_vec = get_magnetic_field(s_axis, u_axis, v_axis);
            return b_vec->length();
        }

//------------------------------------------------------------------------------
///  @brief Get the x position.
///
///  @param[in] s S position.
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
///  @param[in] s S position.
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
///  @param[in] s S position.
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
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
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

        auto sminf = static_cast<T> (sminf_value);
        auto sminh = static_cast<T> (sminh_value);
        auto ds = static_cast<T> (ds_value);
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
