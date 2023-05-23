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

#include "../graph_framework/solver.hpp"

//------------------------------------------------------------------------------
///  @brief Efit tests
//------------------------------------------------------------------------------
template<typename T>
void test_efit() {
    std::mutex sync;

    const size_t num_r = 200;
    const size_t num_z = 400;
    const size_t total = num_r*num_z;

    auto omega = graph::variable<T> (total, "\\omega");
    auto kx = graph::variable<T> (total, "k_{x}");
    auto ky = graph::variable<T> (total, "k_{y}");
    auto kz = graph::variable<T> (total, "k_{z}");
    auto x = graph::variable<T> (total, "x");
    auto y = graph::variable<T> (total, "y");
    auto z = graph::variable<T> (total, "z");
    auto t = graph::variable<T> (total, "t");

    t->set(static_cast<T> (0.0));
    omega->set(static_cast<T> (590.0));
    y->set(static_cast<T> (0.0));
    kx->set(static_cast<T> (0.0));
    ky->set(static_cast<T> (0.0));
    kz->set(static_cast<T> (0.0));

    auto eq = equilibrium::make_efit<T> (NC_FILE, x, y, z, sync);

    dispersion::dispersion_interface<dispersion::ordinary_wave<T>> D(omega, kx, ky, kz, x, y, z, t, eq);

    auto ne = eq->get_electron_density(x, y, z);
    auto b = eq->get_magnetic_field(x, y, z);

    std::vector<T> r_buffer(0);
    std::vector<T> z_buffer(0);

    const T dr = 1.7/(num_r - 1);
    const T dz = 3.2/(num_z - 1);

    for (size_t i = 0; i < num_r; i++) {
        for (size_t j = 0; j < num_z; j++) {
            r_buffer.push_back(i*dr + 0.84);
            z_buffer.push_back(j*dz - 1.6);
        }
    }

    auto q = graph::constant(static_cast<T> (1.602176634E-19));
    auto epsion0 = graph::constant(static_cast<T> (8.8541878138E-12));
    auto mu0 = graph::constant(static_cast<T> (M_PI*4.0E-7));
    auto me = graph::constant(static_cast<T> (9.1093837015E-31));
    auto c = graph::one<T> ()/graph::sqrt(epsion0*mu0);
    auto wpe = dispersion::build_plasma_fequency(ne, q, me, c, epsion0);

    x->set(r_buffer);
    z->set(z_buffer);

    graph::input_nodes<T> inputs = {
        graph::variable_cast<T> (omega),
        graph::variable_cast<T> (kx),
        graph::variable_cast<T> (ky),
        graph::variable_cast<T> (kz),
        graph::variable_cast<T> (x),
        graph::variable_cast<T> (y),
        graph::variable_cast<T> (z),
        graph::variable_cast<T> (t)
    };
    graph::output_nodes<T> outputs = {
        D.get_d(),
        ne,
        b->get_x(),
        b->get_y(),
        b->get_z(),
        wpe
    };

    jit::context<T> context;
    context.add_kernel("residule", inputs, outputs, {});
    context.compile();
    auto kernel = context.create_kernel_call("residule", inputs, outputs, total);

    kernel();

    for (size_t i = 0; i < total; i++) {
        context.print(i);
    }
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @params[in] argc Number of commandline arguments.
///  @params[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    START_GPU
    //test_efit<float> ();
    test_efit<double> ();
    //test_efit<std::complex<float>> ();
    //test_efit<std::complex<double>> ();
    END_GPU
}
