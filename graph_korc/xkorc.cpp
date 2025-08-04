//------------------------------------------------------------------------------
//  xkorc.cpp  –  Full-orbit Boris integrator (SI units)
//
//  Reads from  init_particles.nc
//      • x, y, z , ux, uy, uz          (m and m/s)
//      • dt                            (s)
//      • NSTEPS, DUMP_EVERY            (run control)
//      • particle_mass, particle_charge (kg and C)
//
//  Evolves {x,y,z,ux,uy,uz} with the Boris pusher including electric field.
//
//------------------------------------------------------------------------------

#include <cmath>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <cassert>
#include <netcdf.h>

#include "../graph_framework/equilibrium.hpp"
#include "../graph_framework/timing.hpp"
#include "../graph_framework/output.hpp"

static inline void check_nc(int status)
{
    if (status != NC_NOERR) {
        std::cerr << "netCDF error: " << nc_strerror(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

template<jit::float_scalar T>
void run_korc()
{
    const timeing::measure_diagnostic t_total("Total Time");

    // ─────────────────────────────────── 1. READ INITIAL DATA ────────────────
    int ncid;
    check_nc( nc_open("init_particles.nc", NC_NOWRITE, &ncid) );

    int dim_id;
    check_nc( nc_inq_dimid(ncid, "num_particles", &dim_id) );
    size_t Np = 0;
    check_nc( nc_inq_dimlen(ncid, dim_id, &Np) );
    std::cout << "Num particles: " << Np << '\n';

    std::vector<double> h_x (Np), h_y (Np), h_z (Np),
                        h_ux(Np), h_uy(Np), h_uz(Np);

    auto get_vec=[&](const char* n, std::vector<double>& v){
        int vid; check_nc( nc_inq_varid(ncid, n, &vid) );
        check_nc( nc_get_var_double(ncid, vid, v.data()) );
    };
    get_vec("x" ,h_x ); get_vec("y" ,h_y ); get_vec("z" ,h_z );
    get_vec("ux",h_ux); get_vec("uy",h_uy); get_vec("uz",h_uz);

    long long h_NSTEPS=0, h_DUMP=0;
    double    dt_val=0.0;
    double    m_val = 1.0;      // defaults protect against missing attributes
    double    q_val = 1.0;

    int vid;
    check_nc( nc_inq_varid(ncid,"NSTEPS",     &vid) );
    check_nc( nc_get_var_longlong(ncid,vid,&h_NSTEPS) );
    check_nc( nc_inq_varid(ncid,"DUMP_EVERY", &vid) );
    check_nc( nc_get_var_longlong(ncid,vid,&h_DUMP) );
    check_nc( nc_inq_varid(ncid,"dt",         &vid) );
    check_nc( nc_get_var_double  (ncid,vid,&dt_val)  );

    if (nc_inq_varid(ncid,"particle_mass",&vid)==NC_NOERR)
        check_nc( nc_get_var_double(ncid,vid,&m_val) );
    if (nc_inq_varid(ncid,"particle_charge",&vid)==NC_NOERR)
        check_nc( nc_get_var_double(ncid,vid,&q_val) );

    check_nc( nc_close(ncid) );

    const std::size_t NSTEPS     = static_cast<std::size_t>(h_NSTEPS);
    const std::size_t DUMP_EVERY = static_cast<std::size_t>(h_DUMP  );
    const double q_over_m_val    = q_val / m_val;

    // ─────────────────────────────────── 2. CREATE GRAPH VARS ────────────────
    auto x  = graph::variable<T,false>(Np,"x");
    auto y  = graph::variable<T,false>(Np,"y");
    auto z  = graph::variable<T,false>(Np,"z");
    auto ux = graph::variable<T,false>(Np,"u_x");
    auto uy = graph::variable<T,false>(Np,"u_y");
    auto uz = graph::variable<T,false>(Np,"u_z");

    for(size_t i=0;i<Np;++i){
        x->set(i,h_x [i]); y->set(i,h_y [i]); z->set(i,h_z [i]);
        ux->set(i,h_ux[i]); uy->set(i,h_uy[i]); uz->set(i,h_uz[i]);
    }

    // ─────────────────────────────────── 3. THREAD LAYOUT ────────────────────
    unsigned n_thr = std::max(1u,
        std::min<unsigned>(jit::context<T>::max_concurrency(), static_cast<unsigned>(Np)));
    std::vector<std::thread> workers(n_thr);

    const size_t batch = Np / n_thr, extra = Np % n_thr;

    for(unsigned tid=0; tid<n_thr; ++tid)
    {
        workers[tid] = std::thread([=]() mutable
        {
            const size_t Nloc = batch + (extra>tid ? 1 : 0);
            const timeing::measure_diagnostic t_setup("Setup T"+std::to_string(tid));

            /* constants */
            auto dt   = graph::constant<T,false>(static_cast<T>(dt_val));
            auto half = graph::constant<T,false>(0.5);
            auto one  = graph::constant<T,false>(1.0);
            auto two  = graph::constant<T,false>(2.0);
            auto qom  = graph::constant<T,false>(static_cast<T>(q_over_m_val));
            auto dt_half = dt * half;                    // dt/2
            auto coeff   = dt_half * qom;                // (dt/2)*(q/m)

            /* geometry */
            auto eq  = equilibrium::make_efit<T>(EFIT_FILE);

            /* vector wrappers */
            auto pos   = graph::vector(x ,y ,z );
            auto u_vec = graph::vector(ux,uy,uz);

            /* fields */
            auto b_vec = eq->get_magnetic_field(
                            pos->get_x(), pos->get_y(), pos->get_z());
            auto e_vec = eq->get_electric_field(
                            pos->get_x(), pos->get_y(), pos->get_z());

            /* ── Boris pusher with E ─────────────────────────────────────── */
            auto u_minus  = u_vec  + coeff * e_vec;          // first E-kick
            auto t_vec    = coeff * b_vec;                   // t = (qB/m)*(dt/2)
            auto t_sq     = t_vec->dot(t_vec);
            auto s_vec    = (two * t_vec) / (one + t_sq);    // s = 2t/(1+|t|²)
            auto u_prime  = u_minus + u_minus->cross(t_vec);
            auto u_plus   = u_minus + u_prime->cross(s_vec); // after B rotation
            auto u_next   = u_plus  + coeff * e_vec;         // second E-kick

            auto pos_next = pos + dt * u_next;               // advance position

            /* workflow manager */
            workflow::manager<T,false> w(tid);
            w.add_item({graph::variable_cast(x ),
                        graph::variable_cast(y ),
                        graph::variable_cast(z ),
                        graph::variable_cast(ux),
                        graph::variable_cast(uy),
                        graph::variable_cast(uz)},
                       {},
                       {
                           {pos_next->get_x(), graph::variable_cast(x )},
                           {pos_next->get_y(), graph::variable_cast(y )},
                           {pos_next->get_z(), graph::variable_cast(z )},
                           {u_next->get_x(),   graph::variable_cast(ux)},
                           {u_next->get_y(),   graph::variable_cast(uy)},
                           {u_next->get_z(),   graph::variable_cast(uz)}
                       },
                       "boris_EB");
            w.compile();

            /* output file */
            std::ostringstream fn; fn<<"korc_"<<tid<<".nc";
            output::result_file of(fn.str(), Nloc);
            output::data_set<T> ds(of);
            ds.template create_variable<false>(of,"x",  x,  w.get_context());
            ds.template create_variable<false>(of,"y",  y,  w.get_context());
            ds.template create_variable<false>(of,"z",  z,  w.get_context());
            ds.template create_variable<false>(of,"ux", ux, w.get_context());
            ds.template create_variable<false>(of,"uy", uy, w.get_context());
            ds.template create_variable<false>(of,"uz", uz, w.get_context());
            of.end_define_mode();

            /* writer thread */
            std::queue<size_t> q;
            std::mutex mtx;
            std::condition_variable cv;
            bool finish=false;

            std::thread writer([&](){
                std::unique_lock<std::mutex> lk(mtx);
                while(true){
                    cv.wait(lk,[&](){ return finish || !q.empty(); });
                    while(!q.empty()){
                        q.pop();
                        lk.unlock();
                        ds.write(of);         // I/O outside lock
                        lk.lock();
                    }
                    if(finish) break;
                }
            });

            t_setup.print();
            const timeing::measure_diagnostic t_run("Run T"+std::to_string(tid));
            w.pre_run();

            // ── MAIN LOOP ──────────────────────────────────────────────────
            for(size_t s=0; s<NSTEPS; ++s)
            {
                w.wait();

                if(s % DUMP_EVERY == 0){
                    { std::lock_guard<std::mutex> lg(mtx); q.push(s); }
                    cv.notify_one();
                }
                w.run();
            }
            w.wait();

            // final dump + shutdown
            {
                std::lock_guard<std::mutex> lg(mtx);
                q.push(NSTEPS);
                finish=true;
            }
            cv.notify_one();
            writer.join();
        });
    }
    for(auto& th: workers) th.join();

    std::cout << "\nTiming:\n";
    t_total.print();
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, const char* argv[])
{
    START_GPU
    (void)argc; (void)argv;
    run_korc<double>();
    END_GPU
    return 0;
}
