//------------------------------------------------------------------------------
///  @file absorption.hpp
///  @brief Base class for an absorption model.
///
///  Defines functions for computing power absorbtion.
//------------------------------------------------------------------------------

#ifndef absorption_h
#define absorption_h

#include <thread>

#include "newton.hpp"
#include "output.hpp"

/// Namespace for power absorption models.
namespace absorption {
//******************************************************************************
//  Base class
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Base class for absoption models.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::complex_scalar T, bool SAFE_MATH=true>
    class method {
    public:
///  Type def to retrieve the backend base type.
        typedef T base;
///  Retrieve template parameter of safe math.
        static constexpr bool safe_math = SAFE_MATH;
    };

///  Solver method concept.
    template<class A>
    concept model = std::is_base_of<method<typename A::base, A::safe_math>, A>::value;

//******************************************************************************
//  Root finder.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class interface for the root finder.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::complex_scalar T, bool SAFE_MATH=true>
    class root_finder : public method<T, SAFE_MATH> {
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

///  Residule.
        graph::shared_leaf<T, SAFE_MATH> residule;

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
///  @brief Constructor for root finding.
///
///  @param[in] kamp     Inital kamp.
///  @param[in] w        Inital w.
///  @param[in] kx       Inital kx.
///  @param[in] ky       Inital ky.
///  @param[in] kz       Inital kz.
///  @param[in] x        Inital x.
///  @param[in] y        Inital y.
///  @param[in] z        Inital z.
///  @param[in] t        Inital t.
///  @param[in] eq       The plasma equilibrium.
///  @param[in] filename Result filename, empty names will be blank.
///  @param[in] index    Concurrent index.
//------------------------------------------------------------------------------
        root_finder(graph::shared_leaf<T, SAFE_MATH> kamp,
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
            auto kvec = kx*eq->get_esup1(x, y, z)
                      + ky*eq->get_esup2(x, y, z)
                      + kz*eq->get_esup3(x, y, z);
            auto klen = kvec->length();

            auto kx_amp = kamp*kx/klen;
            auto ky_amp = kamp*ky/klen;
            auto kz_amp = kamp*kz/klen;

            graph::input_nodes<T, SAFE_MATH> inputs = {
                graph::variable_cast(this->kamp),
                graph::variable_cast(this->kx),
                graph::variable_cast(this->ky),
                graph::variable_cast(this->kz),
                graph::variable_cast(this->x),
                graph::variable_cast(this->y),
                graph::variable_cast(this->z)
            };

            graph::map_nodes<T, SAFE_MATH> setters = {
                {graph::zero<T, SAFE_MATH> (), graph::variable_cast(this->kamp)}
            };

            work.add_item(inputs, {}, setters,
                          graph::shared_random_state<T, SAFE_MATH> (),
                          "root_find_init_kernel", inputs.back()->size());

            inputs.push_back(graph::variable_cast(this->t));
            inputs.push_back(graph::variable_cast(this->w));

            auto D = dispersion::hot_plasma<T,
                                            dispersion::z_erfi<T, SAFE_MATH>,
                                            SAFE_MATH>().D(w,
                                                           kx + kx_amp,
                                                           ky + ky_amp,
                                                           kz + kz_amp,
                                                           x, y, z, t, eq);

            solver::newton(work, {kamp}, inputs, {D},
                           graph::shared_random_state<T, SAFE_MATH> ());

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
            work.add_item(inputs, {}, setters,
                          graph::shared_random_state<T, SAFE_MATH> (),
                          "final_kamp", inputs.back()->size());
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
///  @param[in] time_index The time index to run the case for.
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
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::complex_scalar T, bool SAFE_MATH=true>
    class weak_damping : public method<T, SAFE_MATH> {
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
///  @param[in] kamp     Inital kamp.
///  @param[in] w        Inital w.
///  @param[in] kx       Inital kx.
///  @param[in] ky       Inital ky.
///  @param[in] kz       Inital kz.
///  @param[in] x        Inital x.
///  @param[in] y        Inital y.
///  @param[in] z        Inital z.
///  @param[in] t        Inital t.
///  @param[in] eq       The plasma equilibrium.
///  @param[in] filename Result filename, empty names will be blank.
///  @param[in] index    Concurrent index.
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

            auto Dc = dispersion::cold_plasma_expansion<T, SAFE_MATH> ().D(w, kx, ky, kz,
                                                                           x, y, z, t, eq);
            auto Dw = dispersion::hot_plasma_expansion<T,
                                                       dispersion::z_erfi<T, SAFE_MATH>,
                                                       SAFE_MATH> ().D(w, kx, ky, kz,
                                                                       x, y, z, t, eq);

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
            
            work.add_item(inputs, {}, setters,
                          graph::shared_random_state<T, SAFE_MATH> (),
                          "weak_damping_kimg_kernel", inputs.back()->size());
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
///  @param[in] time_index The time index to run the case for.
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
