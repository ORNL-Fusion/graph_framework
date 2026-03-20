//------------------------------------------------------------------------------
///  @file xpic.cpp
///  @brief Driver program for the Particle In Cell (PIC) demo.
//------------------------------------------------------------------------------

#include <random>

#include "../graph_framework/graph_framework.hpp"

//------------------------------------------------------------------------------
///  @brief Build density.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] x The particle position.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
graph::shared_leaf<T> build_density(graph::shared_leaf<T> x) {
    return graph::exp(x*x/static_cast<T> (-0.0001));
}

//------------------------------------------------------------------------------
///  @brief Build parallel electric field.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] x The particle position.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
graph::shared_leaf<T> build_parallel_electric_field(graph::shared_leaf<T> x) {
    const T te = 1;
    const T q = 1;//1.602176634E-19;
    auto n = build_density<T> (x);
    auto pe = n*te;
    return static_cast<T> (-1)/(q*n)*pe->df(x);
}

//------------------------------------------------------------------------------
///  @brief Pic code.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
void run_pic() {
    const size_t num_particles = 1000000;
    auto x = graph::variable<T> (num_particles, "x");
    auto vpara = graph::variable<T> (num_particles, "v||");

    std::normal_distribution<T> norm(0, 0.25);
    std::random_device rand_d;
    std::mt19937_64 engine(rand_d());
    backend::buffer<T> a(num_particles);
    backend::buffer<T> b(num_particles);
    for (size_t i = 0; i < num_particles; i++) {
        a[i] = norm(engine);
        b[i] = norm(engine);
    }
    x->set(a);
    vpara->set(b);

    const T m = 1;//9.1093837139E-31;
    const T q = 1;//1.602176634E-19;
    const T te = 1;
    const T dt = 0.00001;

    const size_t num_grid = 1000;
    auto epara = graph::variable<T> (num_grid, "e||");
    auto n = graph::variable<T> (num_grid, "n");
    auto grid_position = graph::variable<T> (num_grid, "x_i");
    auto particle_index = graph::variable<T> (num_grid, "i");

    const T scale = 2.0/999.0;
    const T offset = -1.0;
    backend::buffer<T> c(num_grid);
    for (size_t i = 0; i < num_grid; i++) {
        c[i] = scale*i + offset;
    }
    grid_position->set(c);

    auto x1 = dt*vpara;
    auto vpara1 = -q/m*graph::index_1D(epara, x, scale, offset);

    auto x2 = dt*(vpara + vpara1/2.0);
    auto vpara2 = -q/m*graph::index_1D(epara, x + x1/2.0, scale, offset);

    auto x3 = dt*(vpara + vpara2/2.0);
    auto vpara3 = -q/m*graph::index_1D(epara, x + x2/2.0, scale, offset);

    auto x4 = dt*(vpara + vpara3);
    auto vpara4 = -q/m*graph::index_1D(epara, x + x3, scale, offset);

    auto x_next = x + (x1 + static_cast<T> (2)*(x2 + x3) + x4)/static_cast<T> (6);
    auto vpara_next = vpara + (vpara1 + static_cast<T> (2)*(vpara2 + vpara3) + vpara4)/static_cast<T> (6);

    auto next_index = particle_index;
    auto next_epara = epara;
    auto next_n = n;

    const size_t batch = 1000;
//  Unroll the loop
    for (size_t i = 0; i < batch; i++) {
        auto indexed_particle = graph::index_1D(x, next_index,
                                                static_cast<T> (1),
                                                static_cast<T> (0));
        next_index = next_index + static_cast<T> (1);
        next_epara = next_epara
                   + build_parallel_electric_field<T> (indexed_particle - grid_position);
        next_n = next_n + build_density(indexed_particle - grid_position);
    }

    workflow::manager<T> work(0);
    work.add_item({
        graph::variable_cast(particle_index),
        graph::variable_cast(epara),
        graph::variable_cast(n)
    }, {}, {
        {graph::zero<T> (), graph::variable_cast(particle_index)},
        {graph::zero<T> (), graph::variable_cast(epara)},
        {graph::zero<T> (), graph::variable_cast(n)}
    }, NULL, "Index_reset", num_grid);
    work.add_loop_item({
        graph::variable_cast(epara),
        graph::variable_cast(n),
        graph::variable_cast(grid_position),
        graph::variable_cast(particle_index),
        graph::variable_cast(x)
    }, {}, {
        {next_epara, graph::variable_cast(epara)},
        {next_index, graph::variable_cast(particle_index)},
        {next_n, graph::variable_cast(n)}
    }, NULL, "Compute_efield", num_grid, num_particles/batch);
    work.add_item({
        graph::variable_cast(x),
        graph::variable_cast(vpara),
        graph::variable_cast(epara)
    }, {}, {
        {x_next, graph::variable_cast(x)},
        {vpara_next, graph::variable_cast(vpara)}
    }, NULL, "Particle_Push", num_particles);

    work.compile();
 
    output::result_file particles_file("pic_particles.nc", num_particles);
    output::data_set<T> p_dataset(particles_file);

    p_dataset.create_variable(particles_file, "x", x, work.get_context());
    p_dataset.create_variable(particles_file, "vpara", vpara, work.get_context());

    particles_file.end_define_mode();
    
    output::result_file fields_file("pic_fields.nc", num_grid);
    output::data_set<T> f_dataset(fields_file);

    f_dataset.create_variable(fields_file, "epara", epara, work.get_context());
    f_dataset.create_variable(fields_file, "n", n, work.get_context());

    fields_file.end_define_mode();
    std::thread sync_particles([]{});
    std::thread sync_fields([]{});

    const size_t num_steps = 1000;
    for (size_t i = 0; i < num_steps; i++) {
        sync_particles.join();
        sync_fields.join();
        work.run();
        sync_particles = std::thread([&particles_file, &p_dataset] () -> void {
            p_dataset.write(particles_file);
        });
        sync_fields = std::thread([&fields_file, &f_dataset] () -> void {
            f_dataset.write(fields_file);
        });
    }
    work.wait();
    sync_particles.join();
    sync_fields.join();
}

//------------------------------------------------------------------------------
///  @brief Main program of the driver.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    START_GPU
    (void)argc;
    (void)argv;

    run_pic<float> ();

    END_GPU
}
