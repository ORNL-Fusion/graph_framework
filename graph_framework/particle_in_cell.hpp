//------------------------------------------------------------------------------
///  @file particle_in_cell.hpp
///  @brief Utilities needed for a particle in cell code.
///
///  Defines graphs for use in Particle In Cell (PIC) codes.
//------------------------------------------------------------------------------

#ifndef particle_in_cell_h
#define particle_in_cell_h

#include <numbers>

#include "piecewise.hpp"
#include "workflow.hpp"
#include "random.hpp"
#include "logical.hpp"

namespace pic {
// FIXME: This should be in a separate file of physics constants.
///  Speed of light m/s.
    template<std::floating_point T>
    constexpr T c = static_cast<T> (299792458.0);
///  Vacuum permitivity F/m.
    template<std::floating_point T>
    constexpr T epsilon0 = static_cast<T> (8.8541878188E-12);
///  Fundamental charge coulombs.
    template<std::floating_point T>
    constexpr T q = static_cast<T> (1.602176634E-19);
///  Hydrogen mass kg.
    template<std::floating_point T>
    constexpr T m_hydrogen = static_cast<T> (1.67362192595E-27);
///  Electron mass kg.
    template<std::floating_point T>
    constexpr T m_electron = static_cast<T> (9.1093837139E-31);
///  Boltzman constant.
    template<std::floating_point T>
    constexpr T kb = static_cast<T> (1.380649E-23);

//------------------------------------------------------------------------------
///  @brief Characteristic factors.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    class characteristics {
    private:
//------------------------------------------------------------------------------
///  @brief Compute the characteristic mass.
///
///  @param[in] ion_masses Ion masses.
///  @returns (∑(m_i) + me)/(n_i + 1);
//------------------------------------------------------------------------------
        T make_m(const std::vector<T> &ion_masses) {
            T total_m = m_electron<T>;
            for (const T &mass : ion_masses) {
                total_m += mass;
            }
            return total_m/(ion_masses.size() + 1);
        }

//------------------------------------------------------------------------------
///  @brief Compute the characteristic mass.
///
///  @param[in] ion_zs Ion Z.
///  @returns (∑(Z_i)*q + q)/(n_i + 1);
//------------------------------------------------------------------------------
        T make_q(const std::vector<uint8_t> &ion_zs) {
            T total_q = pic::q<T>;
            for (const uint8_t &z : ion_zs) {
                total_q += z*pic::q<T>;
            }
            return total_q/(ion_zs.size() + 1);
        }

    public:
///  Mass
        const T m;
///  Charge
        const T q;
///  Electron density.
        const T ne;
///  Plasma Frequency.
        const T wpe;
///  Time.
        const T t;
///  Length
        const T l;
///  Velocity
        const T v;
///  Electric field;
        const T efield;
///  Magnetic field;
        const T bfield;

//------------------------------------------------------------------------------
///  @brief Construct the characteristics.
///
///  @param[in] ion_masses Ion masses for all species.
///  @param[in] ion_zs     Ion Z effective all species.
///  @param[in] ne         Characteristic density.
//------------------------------------------------------------------------------
        characteristics(const std::vector<T> &ion_masses,
                        const std::vector<uint8_t> &ion_zs,
                        const T ne) :
        m(make_m(ion_masses)), q(make_q(ion_zs)), ne(ne),
        wpe(std::sqrt(ne*q*q/(m*epsilon0<T>))),
        t(1/wpe), l(wpe/c<T>), v(c<T>), efield(m*c<T>/(q*t)),
        bfield(efield/c<T>) {}
    };

//------------------------------------------------------------------------------
///  @brief ion class.
///
///  These values need to be initalized using normalized quantities.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    class ion {
    public:
///  Charge
        const T charge;
///  Particle mass
        const T mass;
///  Normalized Position
        graph::shared_leaf<T> x;
///  Normalized Parallel velocity.
        graph::shared_leaf<T> v_para;
///  Normalized Perpendicular velocity.
        graph::shared_leaf<T> v_perp;
///  Mesh Weights
        std::array<graph::shared_leaf<T>, 3> weights;
///  Mesh index
        graph::shared_leaf<T> indices;

//------------------------------------------------------------------------------
///  @brief Construct an ion object.
///
///  @param[in] mass     Ion mass.
///  @param[in] z        Ion Z.
///  @param[in] num_ions Number of ions.
///  @param[in] norms    A @ref pic::characteristics object.
//------------------------------------------------------------------------------
        ion(const T mass,
            const uint8_t z,
            const size_t num_ions,
            const characteristics<T> &norms) :
        charge(z*pic::q<T>/norms.q), mass(mass/norms.m),
        x(graph::variable<T> (num_ions, "x")),
        v_para(graph::variable<T> (num_ions, "v_{||}")),
        v_perp(graph::variable<T> (num_ions, "v_{\\perp}")),
        weights({
            graph::variable<T> (num_ions, "w_{0}"),
            graph::variable<T> (num_ions, "w_{1}"),
            graph::variable<T> (num_ions, "w_{2}")
        }), indices(graph::variable<T> (num_ions, "m_{i}")) {}
    };

//------------------------------------------------------------------------------
///  @brief Mesh class.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    class mesh {
    public:
///  Min x
        const T xmin;
///  Max x
        const T xmax;
///  Dx
        const T dx;
///  Particle index.
        graph::shared_leaf<T> index;
///  Mesh y values.
        graph::shared_leaf<T> y;

//------------------------------------------------------------------------------
///  @brief Construct a mesh object.
///
///  @param[in] x_min Minimum X postion of mesh.
///  @param[in] x_max Maximum X position of mesh.
///  @param[in] num   Number of mesh points.
///  @param[in] norms A @ref pic::characteristics object.
//------------------------------------------------------------------------------
        mesh(const T x_min,
             const T x_max,
             const size_t num,
             const characteristics<T> &norms) :
        y(graph::variable<T> (num, "y_{m}")),
        index(graph::variable<T> (num, "pi_{m}")),
        xmin(x_min/norms.l), xmax(x_max/norms.l),
        dx((xmax - xmin)/(num - 1)) {}

//------------------------------------------------------------------------------
///  @brief Build x index.
///
///  @param[in] ion A @ref pic::ion object.
///  @returns The indexed mesh X position.
//------------------------------------------------------------------------------
        graph::shared_leaf<T> build_x_index(ion<T> &ion) const {
            const size_t s = graph::variable_cast(y)->size();
            const backend::buffer<T> buffer(xmin, dx, s);
            return piecewise_1D(buffer, ion.x, dx, xmin);
        }

//------------------------------------------------------------------------------
///  @brief Build i index.
///
///  @param[in] ion A @ref pic::ion object.
///  @returns The indexed mesh X position.
//------------------------------------------------------------------------------
        graph::shared_leaf<T> build_i_index(ion<T> &ion) const {
            const size_t s = graph::variable_cast(y)->size();
            const backend::buffer<T> buffer(static_cast<T> (0),
                                            static_cast<T> (1), s);
            return piecewise_1D(buffer, ion.x, dx, xmin);
        }

//------------------------------------------------------------------------------
///  @brief Build mesh accumulation.
///
///  @param[in] ion A @ref pic::ion object.
///  @returns Expressions for mesh accumulation.
//------------------------------------------------------------------------------
        std::array<graph::shared_leaf<T>, 2> build_mesh_solve(ion<T> &ion) const {
            auto next_index = index;
            auto next_weight = y;
            auto kernel_index = graph::index<T> ();

            auto index_i = graph::index_1D(ion.indices, next_index,
                                           static_cast<T> (1),
                                           static_cast<T> (0));
            auto index_w0 = graph::index_1D(ion.weights[0], next_index,
                                            static_cast<T> (1),
                                            static_cast<T> (0));
            auto index_w1 = graph::index_1D(ion.weights[1], next_index,
                                            static_cast<T> (1),
                                            static_cast<T> (0));
            auto index_w2 = graph::index_1D(ion.weights[2], next_index,
                                            static_cast<T> (1),
                                            static_cast<T> (0));
            next_index = next_index + static_cast<T> (1);
            next_weight = graph::if_(index_i - static_cast<T> (1) == kernel_index,
                                     next_weight + index_w0, next_weight);
            next_weight = graph::if_(index_i                      == kernel_index,
                                     next_weight + index_w1, next_weight);
            next_weight = graph::if_(index_i + static_cast<T> (1) == kernel_index,
                                     next_weight + index_w2, next_weight);
            return {next_index, next_weight};
        }
    };

//------------------------------------------------------------------------------
///  @brief Build Magnetic field.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] ion   A @ref pic::ion object.
///  @param[in] norms A @ref pic::characteristics object.
///  @returns The magnetic field expression.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    graph::shared_leaf<T> build_magnetic_field(ion<T> ion,
                                               const characteristics<T> &norms) {
        return (ion.x*ion.x + static_cast<T> (1))/norms.bfield;
    }

//------------------------------------------------------------------------------
///  @brief Build interpolation weights.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] mesh Mesh object.
///  @param[in] ion  A @ref pic::ion object.
///  @returns The interpolated mesh weights.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    std::array<graph::shared_leaf<T>, 3> build_weights(mesh<T> &mesh,
                                                       ion<T> &ion) {
        auto x = mesh.build_x_index(ion) - ion.x;
        auto xnorm1 = static_cast<T> (1.5) + (x - mesh.dx)/mesh.dx;
        auto xnorm2 = x/mesh.dx;
        auto xnorm3 = static_cast<T> (1.5) - (x + mesh.dx)/mesh.dx;

        auto w0 = static_cast<T> (0.5)*xnorm1*xnorm1;
        auto w1 = static_cast<T> (0.75) - xnorm2*xnorm2;
        auto w2 = static_cast<T> (0.5)*xnorm3*xnorm3;
        
        return {w0, w1, w2};
    }

//------------------------------------------------------------------------------
///  @brief Build interpolation expression.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] mesh Mesh object.
///  @param[in] ion  A @ref pic::ion object.
///  @returns The interpolated mesh quantity.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    graph::shared_leaf<T> build_interpolation(mesh<T> &mesh,
                                              ion<T> &ion) {
        auto weights = build_weights<T> (mesh, ion);

        auto ymesh0 = graph::index_1D(mesh.y, ion.x - mesh.dx, mesh.dx, mesh.xmin);
        auto ymesh1 = graph::index_1D(mesh.y, ion.x,           mesh.dx, mesh.xmin);
        auto ymesh2 = graph::index_1D(mesh.y, ion.x + mesh.dx, mesh.dx, mesh.xmin);

        return weights[0]*ymesh0 + weights[1]*ymesh1 + weights[2]*ymesh2;
    }

//------------------------------------------------------------------------------
///  @brief Build F expressions.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] ion   A @ref pic::ion object.
///  @param[in] mesh  A @ref pic::mesh object.
///  @param[in] z     Runga Kutta substep.
///  @param[in] dt    Normalized time step.
///  @param[in] norms A @ref pic::characteristics object.
///  @returns the Forces on the particles.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    std::array<graph::shared_leaf<T>, 3> build_F_expressions(ion<T> &ion,
                                                             mesh<T> &mesh,
                                                             const std::array<graph::shared_leaf<T>, 3> z,
                                                             const T dt,
                                                             const characteristics<T> &norms) {
        auto bfield = build_magnetic_field<T> (z[0], norms);
        auto efield = build_interpolation<T> (mesh, ion.x);
        auto temp = 0.5*z[2]*z[1]*bfield->df(z[0])/bfield;
        return {
            z[1]*dt,
            temp*dt,
            (ion.charge/ion.mass*efield - temp)*dt
        };
    }

//------------------------------------------------------------------------------
///  @brief Build Runga Kutta step update.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] ion   A @ref pic::ion object.
///  @param[in] mesh  A @ref pic::mesh object.
///  @param[in] dt    Normalized time step.
///  @param[in] norms A @ref pic::characteristics object.
///  @returns Step update expressions for x, v_para, and x_perp.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    std::array<graph::shared_leaf<T>, 3> build_rk4_step(ion<T> &ion,
                                                        mesh<T> &mesh,
                                                        const T dt,
                                                        const characteristics<T> &norms) {
        std::array<graph::shared_leaf<T>, 3> ion_norm = ion.normalize(norms);

//  Step 1
        std::array<graph::shared_leaf<T>, 3> Z1{ion.x, ion.v_para, ion.v_perp};
        std::array<graph::shared_leaf<T>, 3> dZ1(build_F_expressions<T> (ion, mesh, Z1, dt, norms));

//  Step 2
        std::array<graph::shared_leaf<T>, 3> Z2{
            Z1[0] + dZ1[0]/static_cast<T> (2),
            Z1[1] + dZ1[1]/static_cast<T> (2),
            Z1[2] + dZ1[2]/static_cast<T> (2)
        };
        std::array<graph::shared_leaf<T>, 3> dZ2(build_F_expressions<T> (ion, mesh, Z2, dt, norms));

//  Step 3
        std::array<graph::shared_leaf<T>, 3> Z3{
            Z1[0] + dZ2[0]/static_cast<T> (2),
            Z1[1] + dZ2[1]/static_cast<T> (2),
            Z1[2] + dZ2[2]/static_cast<T> (2)
        };
        std::array<graph::shared_leaf<T>, 3> dZ3(build_F_expressions<T> (ion, mesh, Z3, dt, norms));

//  Step 4
        std::array<graph::shared_leaf<T>, 3> Z4{
            Z1[0] + dZ3[0],
            Z1[1] + dZ3[1],
            Z1[2] + dZ3[2]
        };
        std::array<graph::shared_leaf<T>, 3> dZ4(build_F_expressions<T> (ion, mesh, Z4, dt, norms));

//  Rk4 Solution
        return {
            Z1[0] + (dZ1[0] + static_cast<T> (2)*(dZ2[0] + dZ3[0]) + dZ4[0])/static_cast<T> (6),
            Z1[1] + (dZ1[1] + static_cast<T> (2)*(dZ2[1] + dZ3[1]) + dZ4[1])/static_cast<T> (6),
            Z1[2] + (dZ1[2] + static_cast<T> (2)*(dZ2[2] + dZ3[2]) + dZ4[2])/static_cast<T> (6)
        };
    }

//------------------------------------------------------------------------------
///  @brief Build magnetic moment.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] ion  A @ref pic::ion object.
///  @param[in] mesh A @ref pic::mesh object.
///  @returns The magnetic moment.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    graph::shared_leaf<T> build_magnetic_moment(ion<T> &ion, mesh<T> &mesh) {
        auto efield = build_interpolation<T> (mesh, ion.x);
        return 0.5*ion.mass*ion.v_perp*ion.v_perp/efield;
    }

//------------------------------------------------------------------------------
///  @brief Build initializtion.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] mesh  A @ref pic::mesh object.
///  @param[in] norms A @ref pic::characteristics object.
///  @param[in] state Random state node.
///  @returns Initialized normalized values for x, v||, and v⟂
//------------------------------------------------------------------------------
    template<std::floating_point T>
    std::array<graph::shared_leaf<T>,3> build_initialization(mesh<T> &mesh,
                                                             const characteristics<T> &norms,
                                                             graph::shared_random_state<T> state) {
//  The mesh is already normalized so position_dist will be a normalized quantity.
        auto position_dist = graph::uniform_random<T> (mesh.xmin, mesh.xmax,
                                                       state);
        auto phi_dist = graph::uniform_random<T> (static_cast<T> (0.0),
                                                  static_cast<T> (2.0)*std::numbers::pi_v<T>,
                                                  state);
        auto r_dist = graph::uniform_random<T> (static_cast<T> (0.0),
                                                static_cast<T> (1.0),
                                                state);

        auto vpara = graph::sqrt(-kb<T>*graph::log(static_cast<T> (1) - r_dist))*graph::sin(phi_dist);

        phi_dist = graph::uniform_random<T> (static_cast<T> (0.0),
                                             static_cast<T> (2.0)*std::numbers::pi_v<T>,
                                             state);
        r_dist = graph::uniform_random<T> (static_cast<T> (0.0),
                                           static_cast<T> (1.0),
                                           state);
        auto vperp1 = graph::sqrt(-pic::kb<T>*graph::log(static_cast<T> (1) - r_dist))*graph::cos(phi_dist);
        auto vperp2 = graph::sqrt(-pic::kb<T>*graph::log(static_cast<T> (1) - r_dist))*graph::sin(phi_dist);
        auto vperp = graph::sqrt(vperp1*vperp1 + vperp2*vperp2);

        return {position_dist, vpara/norms.v, vperp/norms.v};
    }

//------------------------------------------------------------------------------
///  @brief Build reinjected expressions.
///
///  If the particles leave the mesh, reinitalize them using the same
///  initialization.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] ion   A @ref pic::ion object.
///  @param[in] mesh  A @ref pic::mesh object.
///  @param[in] norms A @ref pic::characteristics object.
///  @param[in] state Random state node.
///  @returns Reinjected values for x, v||, and v⟂
//------------------------------------------------------------------------------
    template<std::floating_point T>
    std::array<graph::shared_leaf<T>, 3> build_reinjection(ion<T> &ion,
                                                           mesh<T> &mesh,
                                                           const characteristics<T> &norms,
                                                           graph::shared_random_state<T> state) {
        auto resampled = build_initialization(mesh, state);
        auto is_outside = ion.x <= mesh.xmin || ion.x >= mesh.xmax;

        auto reinject_x     = graph::if_(is_outside, resampled[0], ion.x);
        auto reinject_vpara = graph::if_(is_outside, resampled[1], ion.v_para);
        auto reinject_vperp = graph::if_(is_outside, resampled[2], ion.v_perp);
        return {reinject_x, reinject_vpara, reinject_vperp};
    }
}

#endif /* particle_in_cell_h */
