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

namespace pic {
//------------------------------------------------------------------------------
///  @brief ion class.
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
///  Position
        graph::shared_leaf<T> x;
///  Parallel velocity.
        graph::shared_leaf<T> v_para;
///  Perpendicular velocity.
        graph::shared_leaf<T> v_perp;

//------------------------------------------------------------------------------
///  @brief Construct an ion object.
///
///  @param[in] charge Ion charge.
///  @param[in] mass   Ion mass.
///  @param[in] x      Ion position.
///  @param[in] v_para Parallel velocity.
///  @param[in] v_perp Perpendicular velocity.
//------------------------------------------------------------------------------
        ion(const T charge,
            const T mass,
            graph::shared_leaf<T> x,
            graph::shared_leaf<T> v_para,
            graph::shared_leaf<T> v_perp) :
        charge(charge), mass(mass), x(x), v_para(v_para), v_perp(v_perp) {}
    };

//------------------------------------------------------------------------------
///  @brief Mesh class.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    class mesh {
    public:
///  Mesh x positions.
        graph::shared_leaf<T> x;
///  Mesh y values.
        graph::shared_leaf<T> y;
///  Min x
        const T xmin;
///  Max x
        const T xmax;
///  Min mesh spacing.
        const T dx;

//------------------------------------------------------------------------------
///  @brief Construct a mesh object.
///
///  @param[in] xmesh X position of mesh points.
///  @param[in] ymesh Y position of the mesh.
//------------------------------------------------------------------------------
        mesh(graph::shared_leaf<T> xmesh,
             graph::shared_leaf<T> ymesh) :
        x(xmesh), y(ymesh),
        xmin(graph::variable_cast(xmesh)->data()[0]),
        xmax(graph::variable_cast(xmesh)->data()[graph::variable_cast(xmesh)->size() - 1]),
        dx(graph::variable_cast(xmesh)->data()[1] -
           graph::variable_cast(xmesh)->data()[0]) {}
    };

//------------------------------------------------------------------------------
///  @brief Build Magnetic field.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] xp X position of the particles.
///  @returns The magnetic field expression.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    graph::shared_leaf<T> build_magnetic_field(graph::shared_leaf<T> xp,
                                               const T bchar) {
        return (xp*xp + static_cast<T> (1))/bchar;
    }

//------------------------------------------------------------------------------
///  @brief Build interpolation expression.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] mesh Mesh object.
///  @param[in] xp   X position of the particles.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool UNIT_TEST=false>
    graph::shared_leaf<T> build_interpolation(pic::mesh<T> &mesh,
                                              graph::shared_leaf<T> xp) {
        auto x = graph::index_1D(mesh.x, xp, mesh.dx, mesh.xmin) - xp;
        auto xnorm1 = static_cast<T> (1.5) + (x - mesh.dx)/mesh.dx;
        auto xnorm2 = x/mesh.dx;
        auto xnorm3 = static_cast<T> (1.5) - (x + mesh.dx)/mesh.dx;

        auto w0 = static_cast<T> (0.5)*xnorm1*xnorm1;
        auto w1 = static_cast<T> (0.75) - xnorm2*xnorm2;
        auto w2 = static_cast<T> (0.5)*xnorm3*xnorm3;

        auto ymesh0 = graph::index_1D(mesh.y, xp - mesh.dx, mesh.dx, mesh.xmin);
        auto ymesh1 = graph::index_1D(mesh.y, xp,           mesh.dx, mesh.xmin);
        auto ymesh2 = graph::index_1D(mesh.y, xp + mesh.dx, mesh.dx, mesh.xmin);

//  Run only for unit tests.
        if constexpr (UNIT_TEST) {
            auto xp_cast = graph::variable_cast(xp);
            assert(xp_cast.get() && "Expected a variable.");
            
            auto weight = w0 + w1 + w2;
            
            workflow::manager<T> work(0);
            work.add_item({
                graph::variable_cast(mesh.x),
                graph::variable_cast(xp)
            }, {
                weight
            }, {}, NULL, "build_interpolation_unit_test", xp_cast->size());
            work.compile();
            work.run();
            work.wait();

//  The weights should sum to 1.
            for (size_t i = 0, ie = xp_cast->size(); i < ie; i++) {
                const T recieved = work.check_value(i, weight);
                const T diff = static_cast<T> (1) - recieved;
                if constexpr (std::same_as<T, float>) {
                    assert(diff*diff < static_cast<T> (3.2E-14) &&
                           "Weight not equal to 1±3.2E-14");
                } else {
                    assert(diff*diff < static_cast<T> (5.0E-32) &&
                           "Weight not equal to 1±5.0E-32");
                }
            }
        }

        return w0*ymesh0 + w1*ymesh1 + w2*ymesh2;
    }

//------------------------------------------------------------------------------
///  @brief Build F expressions.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] ion   A @ref pic::ion object.
///  @param[in] mesh  A @ref pic::mesh object.
///  @param[in] bchar Characteristic magnetic field.
///  @param[in] z     Runga Kutta substep.
///  @param[in] dt    Time step.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    std::array<graph::shared_leaf<T>, 3> build_F_expressions(ion<T> &ion,
                                                             mesh<T> &mesh,
                                                             const T bchar,
                                                             const std::array<graph::shared_leaf<T>, 3> z,
                                                             const T dt) {
        auto bfield = build_magnetic_field<T> (z[0], bchar);
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
///  @param[in] bchar Characteristic magnetic field.
///  @returns Step update expressions for x, v_para, and x_perp.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    std::array<graph::shared_leaf<T>, 3> build_rk4_step(ion<T> &ion,
                                                        mesh<T> &mesh,
                                                        const T bchar,
                                                        const T dt) {
//  Step 1
        std::array<graph::shared_leaf<T>, 3> Z1{ion.x, ion.v_para, ion.v_perp};
        std::array<graph::shared_leaf<T>, 3> dZ1(build_F_expressions<T> (ion, mesh, bchar, Z1, dt));

//  Step 2
        std::array<graph::shared_leaf<T>, 3> Z2{
            Z1[0] + dZ1[0]/static_cast<T> (2),
            Z1[1] + dZ1[1]/static_cast<T> (2),
            Z1[2] + dZ1[2]/static_cast<T> (2)
        };
        std::array<graph::shared_leaf<T>, 3> dZ2(build_F_expressions<T> (ion, mesh, bchar, Z2, dt));

//  Step 3
        std::array<graph::shared_leaf<T>, 3> Z3{
            Z1[0] + dZ2[0]/static_cast<T> (2),
            Z1[1] + dZ2[1]/static_cast<T> (2),
            Z1[2] + dZ2[2]/static_cast<T> (2)
        };
        std::array<graph::shared_leaf<T>, 3> dZ3(build_F_expressions<T> (ion, mesh, bchar, Z3, dt));

//  Step 4
        std::array<graph::shared_leaf<T>, 3> Z4{
            Z1[0] + dZ3[0],
            Z1[1] + dZ3[1],
            Z1[2] + dZ3[2]
        };
        std::array<graph::shared_leaf<T>, 3> dZ4(build_F_expressions<T> (ion, mesh, bchar, Z4, dt));

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
///  @param[in] ion  A @ref pic::ion object.
///  @param[in] mesh A @ref pic::mesh object.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    graph::shared_leaf<T> build_magnetic_moment(ion<T> &ion, mesh<T> &mesh) {
        auto efield = build_interpolation<T> (mesh, ion.x);
        return 0.5*ion.mass*ion.v_perp*ion.v_perp/efield;
    }

//------------------------------------------------------------------------------
///  @brief Build
//------------------------------------------------------------------------------
    template<std::floating_point T>
    std::array<graph::shared_leaf<T>,3> build_initialization(mesh<T> &mesh,
                                                             graph::shared_random_state<T> state) {
        auto position_dist = graph::uniform_random<T> (mesh.xmin, mesh.xmax,
                                                       state);
        auto phi_dist = graph::uniform_random<T> (static_cast<T> (0.0),
                                                  static_cast<T> (2.0)*std::numbers::pi_v<T>,
                                                  state);
        auto r_dist = graph::uniform_random<T> (static_cast<T> (0.0),
                                                static_cast<T> (1.0),
                                                state);
// FIXME: This should be in a separate file of physics constants.
        const T kb = static_cast<T> (1.380650E-23);
        auto vpara = graph::sqrt(-kb*graph::log(static_cast<T> (1) - r_dist))%graph::sin(phi_dist);

        phi_dist = graph::uniform_random<T> (static_cast<T> (0.0),
                                             static_cast<T> (2.0)*std::numbers::pi_v<T>,
                                             state);
        r_dist = graph::uniform_random<T> (static_cast<T> (0.0),
                                           static_cast<T> (1.0),
                                           state);
        auto vperp1 = graph::sqrt(-kb*graph::log(static_cast<T> (1) - r_dist))%graph::cos(phi_dist);
        auto vperp2 = graph::sqrt(-kb*graph::log(static_cast<T> (1) - r_dist))%graph::sin(phi_dist);
        auto vperp = graph::sqrt(vperp1*vperp1 + vperp2*vperp2);

        return {position_dist, vpara, vperp};
    }
}

#endif /* particle_in_cell_h */
