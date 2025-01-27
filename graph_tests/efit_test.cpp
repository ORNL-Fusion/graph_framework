//------------------------------------------------------------------------------
///  @file efit\_test.cpp
///  @brief Tests for efit splines.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <vector>

#include <netcdf.h>

#include "equilibrium.hpp"

//------------------------------------------------------------------------------
///  @brief Class interface for gold data.
///
///  @tparam T Base type of teh calculation.
//------------------------------------------------------------------------------
template <jit::float_scalar T>
class gold_data {
public:
///  R gird.
    std::vector<T> r_grid;
///  Z grid.
    std::vector<T> z_grid;

///  Bx grid.
    std::vector<T> bx_grid;
///  By grid.
    std::vector<T> by_grid;
///  Bz grid.
    std::vector<T> bz_grid;

///  Pressure grid.
    std::vector<T> pressure_grid;
///  Density grid.
    std::vector<T> ne_grid;
///  Temperature grid.
    std::vector<T> te_grid;

//------------------------------------------------------------------------------
///  @brief Construct a gold data object.
//------------------------------------------------------------------------------
    gold_data() {
        int ncid;

        nc_open(EFIT_GOLD_FILE, NC_NOWRITE, &ncid);

        int varid;

        size_t numr;
        nc_inq_dimid(ncid, "numr", &varid);
        nc_inq_dimlen(ncid, varid, &numr);

        size_t numz;
        nc_inq_dimid(ncid, "numz", &varid);
        nc_inq_dimlen(ncid, varid, &numz);

        r_grid.resize(numr);
        z_grid.resize(numz);

        bx_grid.resize(numr*numz);
        by_grid.resize(numr*numz);
        bz_grid.resize(numr*numz);

        pressure_grid.resize(numr*numz);
        ne_grid.resize(numr*numz);
        te_grid.resize(numr*numz);

        nc_inq_varid(ncid, "r_grid", &varid);
        nc_get_var(ncid, varid, r_grid.data());
        nc_inq_varid(ncid, "z_grid", &varid);
        nc_get_var(ncid, varid, z_grid.data());

        nc_inq_varid(ncid, "bx_grid", &varid);
        nc_get_var(ncid, varid, bx_grid.data());
        nc_inq_varid(ncid, "by_grid", &varid);
        nc_get_var(ncid, varid, by_grid.data());
        nc_inq_varid(ncid, "bz_grid", &varid);
        nc_get_var(ncid, varid, bz_grid.data());

        nc_inq_varid(ncid, "pressure_grid", &varid);
        nc_get_var(ncid, varid, pressure_grid.data());
        nc_inq_varid(ncid, "ne_grid", &varid);
        nc_get_var(ncid, varid, ne_grid.data());
        nc_inq_varid(ncid, "te_grid", &varid);
        nc_get_var(ncid, varid, te_grid.data());

        nc_close(ncid);
    }
};

//------------------------------------------------------------------------------
///  @brief Check error.
///
///  @param[in] test      Test value.
///  @param[in] expected  Expected result.
///  @param[in] tolarance Error tolarance.
///  @param[in] name      Name of the test.
//------------------------------------------------------------------------------
template <jit::float_scalar T>
void check_error(const T test, const T expected, const T tolarance,
                 const char *name) {
    const T diff = test - expected;
    const T error = diff/(diff == 0 ? 1.0 : expected);
    assert(error*error <= tolarance && name);
}

//------------------------------------------------------------------------------
///  @brief Run tests.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<typename T>
void run_test() {
    const gold_data<T> gold;

    auto eq = equilibrium::make_efit<T> (EFIT_FILE);

    std::vector<T> xy_x_grid(gold.bx_grid.size());
    std::vector<T> xy_y_grid(gold.bx_grid.size(),
                             static_cast<T> (0.0));
    std::vector<T> xy_z_grid(gold.bx_grid.size());
    for (size_t i = 0, ie = gold.r_grid.size(); i < ie; i++) {
        for (size_t j = 0, je = gold.z_grid.size(); j < je; j++) {
            const size_t index = i*je + j;
            xy_x_grid[index] = gold.r_grid[i];
            xy_z_grid[index] = gold.z_grid[j];
        }
    }
    
    auto x = graph::variable(xy_x_grid, "x");
    auto y = graph::variable(xy_y_grid, "y");
    auto z = graph::variable(xy_z_grid, "z");

    auto bvec = eq->get_magnetic_field(x, y, z);
    auto ne = eq->get_electron_density(x, y, z);
    auto te = eq->get_electron_temperature(x, y, z);

    workflow::manager<T> work(0);
    work.add_item({
        graph::variable_cast(x),
        graph::variable_cast(y),
        graph::variable_cast(z)
    }, {
        bvec->get_x(), bvec->get_y(), bvec->get_z(), ne, te
    }, {}, "test_kernel");
    work.compile();
    work.run();

    for (size_t i = 0, ie = gold.r_grid.size()*gold.z_grid.size(); i < ie; i++) {
        check_error(work.check_value(i, bvec->get_x()), gold.bx_grid[i], 4.0E-11,
                    "Expected a match in bx.");
        check_error(work.check_value(i, bvec->get_y()), gold.by_grid[i], 1.0E-20,
                    "Expected a match in by.");
        check_error(work.check_value(i, bvec->get_z()), gold.bz_grid[i], 3.0E-12,
                    "Expected a match in bz.");
        check_error(work.check_value(i, ne), gold.ne_grid[i], 5.0E-13,
                    "Expected a match in ne.");
        check_error(work.check_value(i, te), gold.te_grid[i], 5.0E-13,
                    "Expected a match in te.");
    }
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    START_GPU
    (void)argc;
    (void)argv;
    
    run_test<double> ();

    END_GPU
}
