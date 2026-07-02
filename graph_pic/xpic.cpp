//------------------------------------------------------------------------------
///  @file xpic.cpp
///  @brief Driver program for the Particle In Cell (PIC) demo.
//------------------------------------------------------------------------------

#include <random>
#include <numbers>

#include "../graph_framework/graph_framework.hpp"

//------------------------------------------------------------------------------
///  @brief Pic code.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
void run_pic() {
//  Sizes
    const size_t num_particles = 1000000;
    const size_t num_grid = 100;
    const size_t num_ions = 1;

    const pic::characteristics<T> norms({
        pic::m_hydrogen<T>
    }, {1}, static_cast<T> (2.5E19));

    std::array<T, num_ions> ion_masses{pic::m_hydrogen<T>};
    std::array<T, num_ions> ion_zs{1};
    std::array<T, num_ions> density_fraction{1};

    const T lmin = static_cast<T> (-3.0);
    const T lmax = static_cast<T> (3.0);
    const T ne0 = 0.4E18;
    const T b0 = 0.050072;
    const T r1 = 0.0;
    const T r2 = 0.5;
    const T a0 = std::numbers::pi_v<T>*(r2*r2 - r1*r1);
    const T ds = (lmax - lmin)/static_cast<T> (num_grid - 1);

    std::vector<pic::ion<T>> ions;
    for(size_t i = 0; i < num_ions; i++) {
        T num_real = 0;
        for (size_t i = 0; i < num_grid; i++) {
            const T x = ds*i + lmin;
            const T b = static_cast<T> (0.1)*x*x + static_cast<T> (0.5);
            const T a = a0*b0/b;
            num_real += ne0*density_fraction[0]*a*ds;
        }
        ions.emplace_back(ion_masses[i], ion_zs[i], num_particles,
                          num_real, norms);
    }

    const T gyro_period = ion_zs[0]*pic::q<T>*b0/ion_masses[0];
    const T dtc = 0.25;
    const pic::parameters<T> params(b0, r1, r2, 3, 1.0E-4,
                                    dtc*gyro_period, norms);

    pic::mesh<T> mesh(lmin, lmax, num_grid, norms);
    
    auto state = graph::random_state<T> (jit::context<T>::random_state_size, 0);

    workflow::manager<T> work(0);
    for (size_t i = 0; i < num_ions; i++) {
        const std::string ion_tag = jit::format_to_string(i);

        auto ion_inits = pic::build_initialization<T> (mesh, norms,
                                                       graph::random_state_cast(state));
        work.add_preitem({
            ions[i].get_x(), ions[i].get_v_para(), ions[i].get_v_perp()
        }, {}, {
            {ion_inits[0], ions[i].get_x()},
            {ion_inits[1], ions[i].get_v_para()},
            {ion_inits[2], ions[i].get_v_perp()}
        }, graph::random_state_cast(state),
        "pre_initization_" + ion_tag, num_particles);

        auto mesh_i = mesh.build_i_index(ions[i].x);
        auto weights = pic::build_weights<T> (ions[i].x, mesh);
        work.add_preitem({
            ions[i].get_x(),
            graph::variable_cast(ions[i].weights[0]),
            graph::variable_cast(ions[i].weights[1]),
            graph::variable_cast(ions[i].weights[2]),
            graph::variable_cast(ions[i].indices)
        }, {}, {
            {weights[0], graph::variable_cast(ions[i].weights[0])},
            {weights[1], graph::variable_cast(ions[i].weights[1])},
            {weights[2], graph::variable_cast(ions[i].weights[2])},
            {mesh_i, graph::variable_cast(ions[i].indices)}
        }, NULL, "pre_compute_weights_" + ion_tag, num_particles);
    }

    work.compile();
    work.pre_run();

    work.wait();
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
