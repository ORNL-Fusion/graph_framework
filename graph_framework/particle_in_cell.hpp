//------------------------------------------------------------------------------
///  @file particle_in_cell.hpp
///  @brief Utilities needed for a particle in cell code.
///
///  Defines graphs for use in Particle In Cell (PIC) codes.
//------------------------------------------------------------------------------

#ifndef particle_in_cell_h
#define particle_in_cell_h

#include "piecewise.hpp"
#include "workflow.hpp"

namespace pic {
//------------------------------------------------------------------------------
///  @brief Build interpolation expression.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] xmesh X position of mesh points.
///  @param[in] ymesh Y position of the mesh.
///  @param[in] xp    X position of the particles.
///  @param[in] xmin  Minimum X position of the mesh.
///  @param[in] dx    Size of the mesh cells.
//------------------------------------------------------------------------------
    template<std::floating_point T, bool UNIT_TEST=false>
    graph::shared_leaf<T> build_interpolation(graph::shared_leaf<T> xmesh,
                                              graph::shared_leaf<T> ymesh,
                                              graph::shared_leaf<T> xp,
                                              const T xmin,
                                              const T dx) {
        auto x = graph::index_1D(xmesh, xp, dx, xmin) - xp;
        auto xnorm1 = 1.5 + (x - dx)/dx;
        auto xnorm2 = x/dx;
        auto xnorm3 = 1.5 - (x + dx)/dx;

        auto w0 = 0.5*xnorm1*xnorm1;
        auto w1 = 0.75 - xnorm2*xnorm2;
        auto w2 = 0.5*xnorm3*xnorm3;

        auto ymesh0 = graph::index_1D(ymesh, xp - dx, dx, xmin);
        auto ymesh1 = graph::index_1D(ymesh, xp,      dx, xmin);
        auto ymesh2 = graph::index_1D(ymesh, xp + dx, dx, xmin);

//  Run only for unit tests.
        if constexpr (UNIT_TEST) {
            auto xp_cast = graph::variable_cast(xp);
            assert(xp_cast.get() && "Expected a variable.");
            
            auto weight = w0 + w1 + w2;
            
            workflow::manager<T> work(0);
            work.add_item({
                graph::variable_cast(xmesh),
                graph::variable_cast(xp)
            }, {
                weight
            }, {}, NULL, "Mesh_Interpolation", xp_cast->size());
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
}

#endif /* particle_in_cell_h */
