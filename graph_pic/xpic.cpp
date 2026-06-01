//------------------------------------------------------------------------------
///  @file xpic.cpp
///  @brief Driver program for the Particle In Cell (PIC) demo.
//------------------------------------------------------------------------------

#include <random>
#include <numbers>

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
//  Constants
    const size_t num_particles = 1000000;
    const size_t num_grid = 200;
    const T c = 299792458.0;
    const T epsilon0 = 8.854E-12;
    const T q = 1.602E-19;
    const T m_hydrogen = 1.6738E-27;
    const T m_electron = 9.1093837139E-31;
    const T kb = 1.380650E-23;
    const uint8_t Z = 1;

//  Characteristic factors
    const T mchar = (m_hydrogen + m_electron)/2;
    const T qchar = (q + q)/2;
    const T nechar = 2.5E19;
    const T wpechar = std::sqrt(nechar*qchar*qchar/(mchar*epsilon0));
    const T tchar = 1/wpechar;
    const T echar = mchar*c/(qchar*tchar);
    const T bchar = echar/c;

//  Particle initalization.
    auto x = graph::variable<T> (num_particles, "x");
    auto vpara = graph::variable<T> (num_particles, "v_{||}");
    auto vperp = graph::variable<T> (num_particles, "v_{\\perp}");

    {
        std::uniform_real_distribution<T> position_dist(-0.25, 0.25);
        std::uniform_real_distribution<T> phi_dist(0.0, 2.0*std::numbers::pi_v<T>);
        std::uniform_real_distribution<T> r_dist(0.0, 1.0);

        std::random_device rand_d;
        std::mt19937_64 engine(rand_d());

        backend::buffer<T> pos_buffer(num_particles);
        backend::buffer<T> vpara_buffer(num_particles);
        backend::buffer<T> vperp_buffer(num_particles);
        
        for (size_t i = 0; i < num_particles; i++) {
            pos_buffer[i] = position_dist(engine);
            T phi = phi_dist(engine);
            T r = r_dist(engine);
            vpara_buffer[i] = std::fmod(std::sqrt(-kb*std::log(1 - r)), std::sin(phi));
            phi = phi_dist(engine);
            r = r_dist(engine);
            const T vperp1 = std::fmod(std::sqrt(-kb*std::log(1 - r)), std::cos(phi));
            const T vperp2 = std::fmod(std::sqrt(-kb*std::log(1 - r)), std::sin(phi));
            vperp_buffer[i] = std::sqrt(vperp1*vperp1 + vperp2*vperp2);
        }
        x->set(pos_buffer);
        vpara->set(vpara_buffer);
        vperp->set(vperp_buffer);
    }

//  Electron initialization.
    auto te = graph::variable<T> (num_grid, static_cast<T> (1.0), "t_{e}");

//  Magnetic field.
    auto bfield = (x*x + static_cast<T> (1))/bchar;

//  Electric field.
    auto efield = graph::variable<T> (num_grid, "E_{||}");

//  Time step
    const T gyro_frequency = q*1*1.2/2;
    const T gyro_period = 2*std::numbers::pi_v<T>/gyro_frequency;
    T dt = 0.25*gyro_period;

    const size_t timeIterations = std::ceil(10/0.25);
    const size_t outputCadence = std::ceil(600/0.25);

//  Normalize
    dt /= tchar;
    vpara = vpara/c;
    vperp = vperp/c;
    x = x*wpechar/c;

    bfield = bfield/bchar;

    

//  Build the field solver;

    auto epara = graph::variable<T> (num_grid, "e||");
    auto n = graph::variable<T> (num_grid, "n");
    auto grid_position = graph::variable<T> (num_grid, "x_i");
    auto particle_index = graph::variable<T> (num_grid, "i");

    const T scale = 2.0/999.0;
    const T offset = -1.0;
    backend::buffer<T> coe(num_grid);
    for (size_t i = 0; i < num_grid; i++) {
        coe[i] = scale*i + offset;
    }
    grid_position->set(coe);

    auto x1 = dt*vpara;
    auto vpara1 = -q/m_electron*graph::index_1D(epara, x, scale, offset);

    auto x2 = dt*(vpara + vpara1/2.0);
    auto vpara2 = -q/m_electron*graph::index_1D(epara, x + x1/2.0, scale, offset);

    auto x3 = dt*(vpara + vpara2/2.0);
    auto vpara3 = -q/m_electron*graph::index_1D(epara, x + x2/2.0, scale, offset);

    auto x4 = dt*(vpara + vpara3);
    auto vpara4 = -q/m_electron*graph::index_1D(epara, x + x3, scale, offset);

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
