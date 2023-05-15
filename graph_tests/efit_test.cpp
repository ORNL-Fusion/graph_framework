//------------------------------------------------------------------------------
///  @file backend_test.cpp
///  @brief Tests for the buffer backend.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <complex>

#include "../graph_framework/equilibrium.hpp"
#include "../graph_framework/jit.hpp"
#include "../graph_framework/node.hpp"

//------------------------------------------------------------------------------
///  @brief Efit tests
//------------------------------------------------------------------------------
template<typename T>
void test_efit() {
    const size_t num_r = 500;
    const size_t num_z = 250;

    const T r_min = static_cast<T> (0.84);
    const T r_max = r_min + static_cast<T> (1.7);
    const T dr = (r_max - r_min)/static_cast<T> (num_r - 1);

    const T z_min = static_cast<T> (-3.2/2.0);
    const T z_max = z_min + static_cast<T> (3.2);
    const T dz = (z_max - z_min)/static_cast<T> (num_z - 1);

    std::vector<T> x(num_r*num_z);
    std::vector<T> y(num_r*num_z, static_cast<T> (0.0));
    std::vector<T> z(num_r*num_z);
    for (size_t i = 0; i < num_r; i++) {
        for (size_t j = 0; j < num_z; j++) {
            x[i*num_z + j] = r_min + static_cast<T> (i)*dr;
            z[i*num_z + j] = z_min + static_cast<T> (j)*dz;
        }
    }

    auto x_var = graph::variable<T> (x, "x");
    auto y_var = graph::variable<T> (y, "y");
    auto z_var = graph::variable<T> (z, "z");

    std::mutex sync;
    auto eq = equilibrium::efit_equilibrium<T> (NC_FILE, x_var, y_var, z_var, sync);

    auto ne = eq.get_electron_density(x_var, y_var, z_var);
    auto te = eq.get_electron_temperature(x_var, y_var, z_var);
    auto psi = eq.get_psi(x_var, y_var, z_var);
    
    auto b_vec = eq.get_magnetic_field(x_var, y_var, z_var);
    
    jit::context<T> source;
    source.add_kernel("test_ne",
                      {graph::variable_cast(x_var),
                       graph::variable_cast(y_var),
                       graph::variable_cast(z_var)},
                      {psi, ne, te, b_vec->get_x(), b_vec->get_y(), b_vec->get_z()},
                      {});
    
    source.compile();
    auto run = source.create_kernel_call("test_ne",
                                         {graph::variable_cast(x_var),
                                          graph::variable_cast(y_var),
                                          graph::variable_cast(z_var)},
                                         {psi, ne, te, b_vec->get_x(), b_vec->get_y(), b_vec->get_z()},
                                         num_r*num_z);
    run();
    source.wait();
    
    std::cout << std::endl << std::endl;
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @params[in] argc Number of commandline arguments.
///  @params[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    START_GPU
    test_efit<float> ();
    test_efit<double> ();
    test_efit<std::complex<float>> ();
    test_efit<std::complex<double>> ();
    END_GPU
}
