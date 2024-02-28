//------------------------------------------------------------------------------
///  @file absoprtion.hpp
///  @brief Base class for a dispersion relation.
///
///  Defines functions for computing power absorbtion.
//------------------------------------------------------------------------------

#ifndef absorption_h
#define absorption_h

#include <thread>

#include "newton.hpp"

namespace absorption {
//******************************************************************************
//  Root finder.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class interface for the root finder.
///
///  @tparam DISPERSION_FUNCTION Class of dispersion function to use.
//------------------------------------------------------------------------------
    template<dispersion::function DISPERSION_FUNCTION>
    class root_finder {
    private:
///  kamp variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> kamp;

///  w variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                   DISPERSION_FUNCTION::safe_math> w;
///  kx variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> kx;
///  ky variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> ky;
///  kz variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> kz;
///  x variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> x;
///  y variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> y;
///  z variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> z;
///  t variable.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> t;

///  Residule.
        graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                           DISPERSION_FUNCTION::safe_math> residule;

///  Workflow manager.
        workflow::manager<typename DISPERSION_FUNCTION::base,
                          DISPERSION_FUNCTION::safe_math> work;
///  Concurrent index.
        const size_t index;

///  Output file.
        output::result_file file;
///  Output dataset.
        output::data_set<typename DISPERSION_FUNCTION::base> dataset;

///  Async thread to write data files.
        std::thread sync;

    public:
//------------------------------------------------------------------------------
///  @brief Constructor for root finding.
///
///  @params[in] kamp     Inital kamp.
///  @params[in] w        Inital w.
///  @params[in] kx       Inital kx.
///  @params[in] ky       Inital ky.
///  @params[in] kz       Inital kz.
///  @params[in] x        Inital x.
///  @params[in] y        Inital y.
///  @params[in] z        Inital z.
///  @params[in] t        Inital t.
///  @params[in] eq       The plasma equilibrium.
///  @params[in] filename Result filename, empty names will be blank.
///  @params[in] index    Concurrent index.
//------------------------------------------------------------------------------
        root_finder(graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                       DISPERSION_FUNCTION::safe_math> kamp,
                    graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                       DISPERSION_FUNCTION::safe_math> w,
                    graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                       DISPERSION_FUNCTION::safe_math> kx,
                    graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                       DISPERSION_FUNCTION::safe_math> ky,
                    graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                       DISPERSION_FUNCTION::safe_math> kz,
                    graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                       DISPERSION_FUNCTION::safe_math> x,
                    graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                       DISPERSION_FUNCTION::safe_math> y,
                    graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                       DISPERSION_FUNCTION::safe_math> z,
                    graph::shared_leaf<typename DISPERSION_FUNCTION::base,
                                       DISPERSION_FUNCTION::safe_math> t,
                    equilibrium::shared<typename DISPERSION_FUNCTION::base,
                                        DISPERSION_FUNCTION::safe_math> &eq,
                    const std::string &filename="",
                    const size_t index=0) :
        kamp(kamp), w(w), kx(kx), ky(ky), kz(kz), x(x), y(y), z(z), t(t),
        work(index), index(index), file(filename), dataset(file), sync([]{}) {
            auto kvec = kx*eq->get_esup1(x, y, z)
                      + ky*eq->get_esup2(x, y, z)
                      + kz*eq->get_esup3(x, y, z);
            auto klen = kvec->length();

            auto kx_amp = kamp*kx/klen;
            auto ky_amp = kamp*ky/klen;
            auto kz_amp = kamp*kz/klen;

            graph::input_nodes<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> inputs = {
                graph::variable_cast(this->kamp),
                graph::variable_cast(this->kx),
                graph::variable_cast(this->ky),
                graph::variable_cast(this->kz),
                graph::variable_cast(this->x),
                graph::variable_cast(this->y),
                graph::variable_cast(this->z)
            };

            graph::map_nodes<typename DISPERSION_FUNCTION::base,
                             DISPERSION_FUNCTION::safe_math> setters = {
                {graph::zero<typename DISPERSION_FUNCTION::base,
                             DISPERSION_FUNCTION::safe_math> (), graph::variable_cast(this->kamp)}
            };

            work.add_item(inputs, {}, setters, "root_find_init_kernel");

            inputs.push_back(graph::variable_cast(this->t));
            inputs.push_back(graph::variable_cast(this->w));

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D(w, 
                                                                    kx + kx_amp,
                                                                    ky + ky_amp,
                                                                    kz + kz_amp,
                                                                    x, y, z, t, eq);
            solver::newton(work, {kamp}, inputs, {D.get_d()});

            inputs = {
                graph::variable_cast(this->kamp),
                graph::variable_cast(this->kx),
                graph::variable_cast(this->ky),
                graph::variable_cast(this->kz),
                graph::variable_cast(this->x),
                graph::variable_cast(this->y),
                graph::variable_cast(this->z)
            };
            setters = {
                {klen + kamp, graph::variable_cast(this->kamp)}
            };
            work.add_item(inputs, {}, setters, "final_kamp");
        }

//------------------------------------------------------------------------------
///  @brief Destructor.
//------------------------------------------------------------------------------
        ~root_finder() {
            sync.join();
        }

//------------------------------------------------------------------------------
///  @brief Compile the workitems.
//------------------------------------------------------------------------------
        void compile() {
            work.compile();

            dataset.create_variable(file, "kamp", this->kamp, work.get_context());
            
            dataset.reference_variable(file, "w",    graph::variable_cast(this->w));
            dataset.reference_variable(file, "kx",   graph::variable_cast(this->kx));
            dataset.reference_variable(file, "ky",   graph::variable_cast(this->ky));
            dataset.reference_variable(file, "kz",   graph::variable_cast(this->kz));
            dataset.reference_variable(file, "x",    graph::variable_cast(this->x));
            dataset.reference_variable(file, "y",    graph::variable_cast(this->y));
            dataset.reference_variable(file, "z",    graph::variable_cast(this->z));
            dataset.reference_variable(file, "time", graph::variable_cast(this->t));
            file.end_define_mode();
        }

//------------------------------------------------------------------------------
///  @brief Run the workflow.
///
///  @params[in] time_index The time index to run the case for.
//------------------------------------------------------------------------------
        void run(const size_t time_index) {
            dataset.read(file, time_index);
            work.copy_to_device(w,  graph::variable_cast(this->w)->data());
            work.copy_to_device(kx, graph::variable_cast(this->kx)->data());
            work.copy_to_device(ky, graph::variable_cast(this->ky)->data());
            work.copy_to_device(kz, graph::variable_cast(this->kz)->data());
            work.copy_to_device(x,  graph::variable_cast(this->x)->data());
            work.copy_to_device(y,  graph::variable_cast(this->y)->data());
            work.copy_to_device(z,  graph::variable_cast(this->z)->data());
            work.copy_to_device(t,  graph::variable_cast(this->t)->data());

            work.run();
            
            sync.join();
            work.wait();
            sync = std::thread([this] (const size_t i) -> void {
                dataset.write(file, i);
            }, time_index);
        }
    };

//******************************************************************************
//  Weak Damping.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class interface weak damping approximation.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::complex_scalar T, bool SAFE_MATH=true>
    class weak_damping {
    private:
///  kamp variable.
        graph::shared_leaf<T, SAFE_MATH> kamp;

///  w variable.
        graph::shared_leaf<T, SAFE_MATH> w;
///  kx variable.
        graph::shared_leaf<T, SAFE_MATH> kx;
///  ky variable.
        graph::shared_leaf<T, SAFE_MATH> ky;
///  kz variable.
        graph::shared_leaf<T, SAFE_MATH> kz;
///  x variable.
        graph::shared_leaf<T, SAFE_MATH> x;
///  y variable.
        graph::shared_leaf<T, SAFE_MATH> y;
///  z variable.
        graph::shared_leaf<T, SAFE_MATH> z;
///  t variable.
        graph::shared_leaf<T, SAFE_MATH> t;

///  Workflow manager.
        workflow::manager<T, SAFE_MATH> work;

///  Concurrent index.
        const size_t index;

///  Output file.
        output::result_file file;
///  Output dataset.
        output::data_set<T> dataset;

///  Async thread to write data files.
        std::thread sync;

    public:
//------------------------------------------------------------------------------
///  @brief Constructor for weak damping.
///
///  @params[in] kamp     Inital kamp.
///  @params[in] w        Inital w.
///  @params[in] kx       Inital kx.
///  @params[in] ky       Inital ky.
///  @params[in] kz       Inital kz.
///  @params[in] x        Inital x.
///  @params[in] y        Inital y.
///  @params[in] z        Inital z.
///  @params[in] t        Inital t.
///  @params[in] eq       The plasma equilibrium.
///  @params[in] filename Result filename, empty names will be blank.
///  @params[in] index    Concurrent index.
//------------------------------------------------------------------------------
        weak_damping(graph::shared_leaf<T, SAFE_MATH> kamp,
                     graph::shared_leaf<T, SAFE_MATH> w,
                     graph::shared_leaf<T, SAFE_MATH> kx,
                     graph::shared_leaf<T, SAFE_MATH> ky,
                     graph::shared_leaf<T, SAFE_MATH> kz,
                     graph::shared_leaf<T, SAFE_MATH> x,
                     graph::shared_leaf<T, SAFE_MATH> y,
                     graph::shared_leaf<T, SAFE_MATH> z,
                     graph::shared_leaf<T, SAFE_MATH> t,
                     equilibrium::shared<T, SAFE_MATH> &eq,
                     const std::string &filename="",
                     const size_t index=0) :
        kamp(kamp), w(w), kx(kx), ky(ky), kz(kz), x(x), y(y), z(z), t(t),
        work(index), index(index), file(filename), dataset(file), sync([]{}) {
            auto k_vec = kx*eq->get_esup1(x, y, z)
                       + ky*eq->get_esup2(x, y, z)
                       + kz*eq->get_esup3(x, y, z);
            auto k_unit = k_vec->unit();

            auto Dc = dispersion::cold_plasma<T, SAFE_MATH> ().D(w, kx, ky, kz,
                                                                 x, y, z, t, eq);
            auto Dw = dispersion::hot_plasma_expansion<T,
                                                       dispersion::z_erfi<T, SAFE_MATH>,
                                                       SAFE_MATH> ().D(w, kx, ky, kz,
                                                                       x, y, z, t, eq);

            auto none = graph::none<T, SAFE_MATH> ();
            auto kamp1 = k_vec->length() 
                       - Dw/k_unit->dot(Dc->df(kx)*eq->get_esup1(x, y, z) +
                                        Dc->df(ky)*eq->get_esup2(x, y, z) +
                                        Dc->df(kz)*eq->get_esup3(x, y, z));

            graph::input_nodes<T, SAFE_MATH> inputs = {
                graph::variable_cast(this->kamp),
                graph::variable_cast(this->kx),
                graph::variable_cast(this->ky),
                graph::variable_cast(this->kz),
                graph::variable_cast(this->x),
                graph::variable_cast(this->y),
                graph::variable_cast(this->z),
                graph::variable_cast(this->t),
                graph::variable_cast(this->w)
            };

            graph::map_nodes<T, SAFE_MATH> setters = {
                {kamp1, graph::variable_cast(this->kamp)}
            };
            
            work.add_item(inputs, {}, setters, "weak_damping_kimg_kernel");
        }

//------------------------------------------------------------------------------
///  @brief Destructor.
//------------------------------------------------------------------------------
        ~weak_damping() {
            sync.join();
        }

//------------------------------------------------------------------------------
///  @brief Compile the workitems.
//------------------------------------------------------------------------------
        void compile() {
            work.compile();

            dataset.create_variable(file, "kamp", this->kamp, work.get_context());

            dataset.reference_variable(file, "w",    graph::variable_cast(this->w));
            dataset.reference_variable(file, "kx",   graph::variable_cast(this->kx));
            dataset.reference_variable(file, "ky",   graph::variable_cast(this->ky));
            dataset.reference_variable(file, "kz",   graph::variable_cast(this->kz));
            dataset.reference_variable(file, "x",    graph::variable_cast(this->x));
            dataset.reference_variable(file, "y",    graph::variable_cast(this->y));
            dataset.reference_variable(file, "z",    graph::variable_cast(this->z));
            dataset.reference_variable(file, "time", graph::variable_cast(this->t));
            file.end_define_mode();
        }

//------------------------------------------------------------------------------
///  @brief Run the workflow.
///
///  @params[in] time_index The time index to run the case for.
//------------------------------------------------------------------------------
        void run(const size_t time_index) {
            dataset.read(file, time_index);
            work.copy_to_device(w,  graph::variable_cast(this->w)->data());
            work.copy_to_device(kx, graph::variable_cast(this->kx)->data());
            work.copy_to_device(ky, graph::variable_cast(this->ky)->data());
            work.copy_to_device(kz, graph::variable_cast(this->kz)->data());
            work.copy_to_device(x,  graph::variable_cast(this->x)->data());
            work.copy_to_device(y,  graph::variable_cast(this->y)->data());
            work.copy_to_device(z,  graph::variable_cast(this->z)->data());
            work.copy_to_device(t,  graph::variable_cast(this->t)->data());

            work.run();

            sync.join();
            work.wait();
            sync = std::thread([this] (const size_t i) -> void {
                dataset.write(file, i);
            }, time_index);
        }
    };
}

#endif /* absorption_h */
