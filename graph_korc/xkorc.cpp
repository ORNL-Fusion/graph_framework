#include <random>

#include "../graph_framework/equilibrium.hpp"
#include "../graph_framework/timing.hpp"
#include "../graph_framework/output.hpp"

//------------------------------------------------------------------------------
///  @brief Run Korc
///
///  @tparam T Base type.
//------------------------------------------------------------------------------
template<jit::float_scalar T>
void run_korc() {
    const timing::measure_diagnostic t_total("Total Time");
    
    const size_t num_particles = 100000;
    std::cout << "Num particles " << num_particles << std::endl;
    std::vector<std::thread> threads(std::max(std::min(static_cast<unsigned int> (jit::context<T>::max_concurrency()),
                                                       static_cast<unsigned int> (num_particles)),
                                              static_cast<unsigned int> (1)));
    
    const size_t batch = num_particles/threads.size();
    const size_t extra = num_particles%threads.size();
    
    for (size_t i = 0, ie = threads.size(); i < ie; i++) {
        threads[i] = std::thread([num_particles, batch, extra] (const size_t thread_number) -> void {
            const size_t local_num_particles = batch + (extra > thread_number ? 1 : 0);
            
            const timing::measure_diagnostic t_setup("Setup Time");
            
            auto eq = equilibrium::make_efit<T> (EFIT_FILE);
            //auto eq = equilibrium::make_slab_density<T> ();
            auto b0 = eq->get_characteristic_field(thread_number);
            const T q = 1.602176634E-19;
            const T me = 9.1093837139E-31;
            const T c = 299792458.0;
            
            auto gryo_period = me/(q*b0);
            std::cout << "gryo_period " << gryo_period->evaluate().at(0) << std::endl;
            auto larmor_radius = c*gryo_period;
            std::cout << "larmor_radius " << larmor_radius->evaluate().at(0) << std::endl;
            
            std::cout << "Local num particles " << local_num_particles << std::endl;

            std::mt19937_64 engine((thread_number + 1)*static_cast<uint64_t> (std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())));
            std::normal_distribution<T> normal(static_cast<T> (0.0), static_cast<T> (0.1));

            auto ux = graph::variable<T> (local_num_particles, "u_{x}");
            auto uy = graph::variable<T> (local_num_particles, "u_{y}");
            auto uz = graph::variable<T> (local_num_particles, "u_{z}");

            ux->set(0.0);
            uy->set(0.9);
            for (size_t j = 0; j < local_num_particles; j++) {
                T temp = normal(engine);
                uz->set(j, temp);
                while (0.9*0.9 + temp*temp > 1.0) {
                    temp = normal(engine);
                    uz->set(j, temp);
                }
            }
            
            auto x = graph::variable<T> (local_num_particles, "x");
            auto y = graph::variable<T> (local_num_particles, "y");
            auto z = graph::variable<T> (local_num_particles, "z");
            auto pos = graph::vector(x, y, z);
            
            for (size_t j = 0; j < local_num_particles; j++) {
                x->set(j, 1.7 + normal(engine));
                y->set(j, 0.0);
                z->set(j, normal(engine));
            }
            
            auto u_vec = graph::vector(ux, uy, uz);
            
            auto gamma = graph::variable<T> (local_num_particles, "\\gamma");
            
            auto dt = graph::constant<T> (0.5);
            
            auto gamma_init = 1.0/graph::sqrt(1.0 - u_vec->dot(u_vec));
            
            auto u_init = gamma_init*u_vec;
            
            auto b_vec = eq->get_magnetic_field(pos->get_x(),
                                                pos->get_y(),
                                                pos->get_z())/b0;
            
            workflow::manager<T> work(thread_number);
            work.add_preitem({
                graph::variable_cast(ux),
                graph::variable_cast(uy),
                graph::variable_cast(uz),
                graph::variable_cast(gamma)
            }, {}, {
                {u_init->get_x(), graph::variable_cast(ux)},
                {u_init->get_y(), graph::variable_cast(uy)},
                {u_init->get_z(), graph::variable_cast(uz)},
                {gamma_init, graph::variable_cast(gamma)}
            }, graph::shared_random_state<T> (), "initalize_gamma", local_num_particles);
            
            auto u_prime = u_vec - dt*u_vec->cross(b_vec)/(2.0*gamma);
            
            auto tau = -0.5*dt*b_vec;
            auto tau_sq = tau->dot(tau);
            auto speed_sq = u_prime->dot(u_prime);
            auto sigma = 1.0 + speed_sq - tau_sq;
            auto ustar = u_prime->dot(tau);
            
            auto gamma_next = graph::sqrt(0.5*(sigma + graph::sqrt(sigma*sigma + 4.0*(tau_sq + ustar*ustar))));
            auto t = tau/gamma_next;
            
            auto s = 1.0 + t->dot(t);
            auto u_prime_dot_t = u_prime->dot(t);
            
            auto u_next = (u_prime + u_prime_dot_t*t + u_prime->cross(t))/s;
            
            auto pos_next = pos + larmor_radius*dt*u_next/gamma_next;
            
            work.add_item({
                graph::variable_cast(x),
                graph::variable_cast(y),
                graph::variable_cast(z),
                graph::variable_cast(ux),
                graph::variable_cast(uy),
                graph::variable_cast(uz),
                graph::variable_cast(gamma)
            }, {}, {
                {pos_next->get_x(), graph::variable_cast(x)},
                {pos_next->get_y(), graph::variable_cast(y)},
                {pos_next->get_z(), graph::variable_cast(z)},
                {u_next->get_x(), graph::variable_cast(ux)},
                {u_next->get_y(), graph::variable_cast(uy)},
                {u_next->get_z(), graph::variable_cast(uz)},
                {gamma_next, graph::variable_cast(gamma)}
            }, graph::shared_random_state<T> (), "step", local_num_particles);
            
            work.compile();
            
            std::ostringstream stream;
            stream << "korc_" << thread_number << ".nc";
            
            output::result_file file(stream.str(), local_num_particles);
            output::data_set<T> dataset(file);
            
            dataset.create_variable(file, "x",     x,     work.get_context());
            dataset.create_variable(file, "y",     y,     work.get_context());
            dataset.create_variable(file, "z",     z,     work.get_context());
            dataset.create_variable(file, "ux",    ux,    work.get_context());
            dataset.create_variable(file, "uy",    uy,    work.get_context());
            dataset.create_variable(file, "uz",    uz,    work.get_context());
            dataset.create_variable(file, "gamma", gamma, work.get_context());
            
            file.end_define_mode();
            std::thread sync([]{});
            
            t_setup.print();
            
            const timing::measure_diagnostic t_run("Run Time");
            work.pre_run();
            for (size_t j = 0; j < 10000; j++) {
                sync.join();
                work.wait();
                sync = std::thread([&file, &dataset] () -> void {
                    dataset.write(file);
                });
                for (size_t k = 0; k < 10; k++) {
                    work.run();
                }
            }
            work.wait();

            sync.join();
            //dataset.write(file);
            work.wait();
            
            t_run.print();
        }, i);
    }
    
    for (std::thread &t : threads) {
        t.join();
    }
    
    std::cout << std::endl << "Timing:" << std::endl;
    t_total.print();
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

    run_korc<double> ();
//    run_korc<float> ();

    END_GPU
}
