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
    const timing::measure_diagnostic init("Init Time");
//  Sizes
    const size_t num_particles = 3000000;
    const size_t num_grid = 1000;
    const size_t num_batch = 10;
    const size_t num_ions = 1;
    const size_t num_steps = 100;
    const size_t num_sub_steps = 2400;

    const std::vector<T> ion_masses{2*pic::m_atomic<T>};
    const std::vector<uint8_t> ion_zs{1};

    const pic::characteristics<T> norms(ion_masses, ion_zs,
                                        static_cast<T> (2.5E19));

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
    for (size_t i = 0; i < num_ions; i++) {
        T num_real = 0;
        for (size_t i = 0; i < num_grid; i++) {
            const T x = ds*i + lmin;
            const T b = b0*(x*x + static_cast<T> (0.5));
            const T a = a0*b0/b;
            num_real += ne0*density_fraction[0]*a*ds;
        }
        ions.emplace_back(ion_masses[i], ion_zs[i], num_particles,
                          num_real, norms);
    }

    const T b_cv = 1.2;
    const T cyclotron_frequency = ion_zs[0]*pic::q<T>*b_cv/ion_masses[0];
    const T gyro_period = 2*std::numbers::pi_v<T>/cyclotron_frequency;
    const T dtc = 0.25;
    const pic::parameters<T> params(b0, r1, r2, 100, 1.0E-4,
                                    dtc*gyro_period, 2.5, 2.5, norms);

    pic::mesh<T> mesh(lmin, lmax, num_grid, norms);
    
    auto state = graph::random_state<T> (jit::context<T>::random_state_size, 0);

    workflow::manager<T> work(0);

    output::result_file f_file("fields.nc", num_grid);
    output::data_set<T> mesh_dataset(f_file);

    output::result_file p_file("particles.nc", num_particles);
    std::vector<output::data_set<T>> p_datasets(num_ions,
                                                output::data_set<T> (p_file));

    for (size_t i = 0; i < num_ions; i++) {
        const std::string ion_tag = jit::format_to_string(i);

        auto ion_inits = pic::build_initialization<T> (ions[i], mesh,
                                                       norms, params,
                                                       graph::random_state_cast(state));
        work.template add_item<workflow::order::pre_run_item> ({
            ions[i].get_x(), ions[i].get_v_para(), ions[i].get_v_perp()
        }, {}, {
            {ion_inits[0], ions[i].get_x()},
            {ion_inits[1], ions[i].get_v_para()},
            {ion_inits[2], ions[i].get_v_perp()}
        }, graph::random_state_cast(state),
        "pre_initization_" + ion_tag, num_particles);

        auto mesh_i = mesh.build_i_index(ions[i].x);
        auto weights = pic::build_weights<T> (ions[i].x, mesh);
        work.template add_item<workflow::order::pre_run_item> ({
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

        if (i == 0) {
            work.template add_zero_item<workflow::order::pre_run_item> ({
                graph::variable_cast(mesh.index),
                graph::variable_cast(mesh.y[0])
            });
        } else {
            work.template add_zero_item<workflow::order::pre_run_item> ({
                graph::variable_cast(mesh.index)
            });
        }

        auto mesh_solve = mesh.build_mesh_solve(ions[i], num_batch);
        work.template add_loop_item<workflow::order::pre_run_item> ({
            graph::variable_cast(ions[i].indices),
            graph::variable_cast(ions[i].weights[0]),
            graph::variable_cast(ions[i].weights[1]),
            graph::variable_cast(ions[i].weights[2]),
            graph::variable_cast(mesh.index),
            graph::variable_cast(mesh.y[0])
        }, {}, {
            {mesh_solve[0], graph::variable_cast(mesh.index)},
            {mesh_solve[1], graph::variable_cast(mesh.y[0])}
        }, NULL, "pre_sum_weights_" + ion_tag, num_grid, num_particles/num_batch);

        if (i == ions.size() - 1) {
            work.template add_copy_item<workflow::order::pre_run_item> ({
                {graph::variable_cast(mesh.y[0]), graph::variable_cast(mesh.y[1])},
                {graph::variable_cast(mesh.y[0]), graph::variable_cast(mesh.y[2])},
                {graph::variable_cast(mesh.y[0]), graph::variable_cast(mesh.y[3])}
            });
        }

        if (i == 0) {
            work.template add_callback_item<workflow::order::post_run_item> ([&f_file, &mesh_dataset]() {
                mesh_dataset.write(f_file);
            });
        }
        work.template add_callback_item<workflow::order::post_run_item> ([i, &p_file, &p_datasets]() {
            p_datasets[i].write(p_file);
        });

        auto particle_step = pic::build_rk4_step(ions[i], mesh, norms, params);
        work.add_item({
            ions[i].get_x(),
            ions[i].get_v_para(),
            ions[i].get_v_perp(),
            graph::variable_cast(mesh.y[0]),
            graph::variable_cast(mesh.y[1]),
            graph::variable_cast(mesh.y[2]),
            graph::variable_cast(mesh.y[3])
        }, {}, {
            {particle_step[0], ions[i].get_x()},
            {particle_step[1], ions[i].get_v_para()},
            {particle_step[2], ions[i].get_v_perp()}
        }, NULL, "particle_push_" + ion_tag, num_particles);

        auto particle_reinject = pic::build_reinjection(ions[i], mesh, norms, params,
                                                        graph::random_state_cast(state));
        work.add_item({
            ions[i].get_x(),
            ions[i].get_v_para(),
            ions[i].get_v_perp()
        }, {}, {
            {particle_reinject[0], ions[i].get_x()},
            {particle_reinject[1], ions[i].get_v_para()},
            {particle_reinject[2], ions[i].get_v_perp()}
        }, graph::random_state_cast(state),
        "particle_reinjection_" + ion_tag, num_particles);

        work.add_item({
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
        }, NULL, "compute_weights_" + ion_tag, num_particles);

        if (i == 0) {
            work.add_zero_item({
                graph::variable_cast(mesh.index),
                graph::variable_cast(mesh.y[0])
            });
        } else {
            work.add_zero_item({
                graph::variable_cast(mesh.index)
            });
        }
        if (i == 0) {
            work.add_copy_item({
                {graph::variable_cast(mesh.y[2]), graph::variable_cast(mesh.y[3])},
                {graph::variable_cast(mesh.y[1]), graph::variable_cast(mesh.y[2])},
                {graph::variable_cast(mesh.y[0]), graph::variable_cast(mesh.y[1])}
            });
        }

        work.add_loop_item({
            graph::variable_cast(ions[i].indices),
            graph::variable_cast(ions[i].weights[0]),
            graph::variable_cast(ions[i].weights[1]),
            graph::variable_cast(ions[i].weights[2]),
            graph::variable_cast(mesh.index),
            graph::variable_cast(mesh.y[0])
        }, {}, {
            {mesh_solve[0], graph::variable_cast(mesh.index)},
            {mesh_solve[1], graph::variable_cast(mesh.y[0])}
        }, NULL, "sum_weights_" + ion_tag, num_grid, num_particles/num_batch);
    }
    init.print();

    const timing::measure_diagnostic compile("Compile Time");
    work.compile();
    compile.print();

    mesh.define_variables(f_file, mesh_dataset, work);
    f_file.end_define_mode();

    for (size_t i = 0; i < num_ions; i++) {
        const std::string ion_tag = jit::format_to_string(i);
        ions[i].define_variables(p_file, p_datasets[i], work, ion_tag);
    }
    p_file.end_define_mode();

    std::atomic_size_t counter = 0;
#ifndef PROFILE
    std::thread progress = std::thread([&num_steps, &counter]() -> void {
        using namespace std::chrono_literals;
        do {
            const size_t progress = (counter*100.0)/num_steps;
            std::cout << "\33[2K\r" << std::setw(3) << progress << "% Complete"
                      << std::flush;
            std::this_thread::sleep_for(1s);
        } while (counter < num_steps);
    });
#endif
    const timing::measure_diagnostic run("Run Time");
    work.template run<workflow::order::pre_run_item> ();
    work.template run<workflow::order::post_run_item> ();
    work.wait();

    for (; counter < num_steps; counter++) {
        for (size_t i = 0; i < num_sub_steps; i++) {
            work.run();
        }
        work.template run<workflow::order::post_run_item> ();
        work.wait();
    }

    counter = num_steps;
#ifndef PROFILE
    progress.join();
#endif
    std::cout << "\33[2K\r" << "100% Complete" << std::endl;
    run.print();
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

    const timing::measure_diagnostic total("Total Time");
    run_pic<float> ();
    total.print();

    END_GPU
}
