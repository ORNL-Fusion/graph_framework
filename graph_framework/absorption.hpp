//------------------------------------------------------------------------------
///  @file absorption.hpp
///  @brief Base class for an absorption model.
///
///  Defines functions for computing power absorption.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
///  @page absorption_model Absorption Models
///  @brief A description of the models for power absorption.
///  @tableofcontents
///
///  @section absorption_model_intro Introduction
///  This page documents the types of dispersion functions available. Tracing
///  the ray is only the first step in the process. Along that, ray power is
///  deposited into the plasma. For tracing the ray we assumed the wave number
///  was always real. However, now we need to figure out what the imaginary
///  component is.
///
///  <hr>
///  @subsection absorption_model_hot Hot Plasma Dispersion Function
///  To do this we now assume a hot plasma dispersion function.
///  @f{equation}{D\left(\vec{x},\vec{k},\omega\right)=i\sigma\Gamma_{0}+\Gamma_{1}+\vec{n}_{\perp}\cdot\vec{n}_{\perp}\frac{P\omega}{\omega_{ce}}\left(1+\zeta Z\left(\zeta\right)\right)\left(\Gamma_{2} + \Gamma_{5}F\right)\equiv 0 @f}
///  Where
///  @f{equation}{\Gamma_{0}=\vec{n}_{\perp}\cdot\vec{n}_{\perp}\left(\vec{n}\cdot\vec{n}-2\left(1-2q\right)\right)+\left(1-P\right)\left(2\left(1-2q\right)-\left(\vec{n}\cdot\vec{n}+n_{||}^{2}\right)\right)@f}
///  @f{equation}{\Gamma_{1}=\vec{n}_{\perp}\cdot\vec{n}_{\perp}\left(\left(1-q\right)\vec{n}\cdot\vec{n}-(1-2q)\right)+\left(1-P\right)\left(\vec{n}\cdot\vec{n}n^{2}_{||} - \left(1-q\right)\left(\vec{n}\cdot\vec{n}+n^{2}_{||}\right)+\left(1-2q\right)\right)@f}
///  @f{equation}{\Gamma_{2}=\left(\vec{n}\cdot\vec{n}-\left(1-2q\right)\right)+\frac{P\omega}{4\omega_{ce}n^{2}_{||}}\left(\left(\vec{n}\cdot\vec{n} + n^{2}_{||}\right)-2\left(1-2q\right)\right)@f}
///  @f{equation}{\Gamma_{5}=\vec{n}\cdot\vec{n}n^{2}_{||}-\left(1-q\right)\left(\vec{n}\cdot\vec{n}+n^{2}_{||}\right)+\left(1-2q\right)@f}
///  @f{equation}{i\sigma=\frac{PZ\left(\zeta\right)}{2n_{||}v_{e}}@f}
///  @f{equation}{\zeta=\frac{1-\frac{\omega_{ce}}{\omega}}{n_{||}\frac{v_{e}}{c}}@f}
///  @f{equation}{F=\frac{v_{e}\left(1+\zeta Z\left(\zeta\right)\right)\omega}{2n_{||}\omega_{ce}}@f}
///  @f{equation}{P=\frac{\omega^{2}_{pe}}{\omega^{2}}@f}
///  @f{equation}{q=\frac{P}{2\left(1+\frac{\omega_{ce}}{\omega}\right)}@f}
///  @f{equation}{v_{e}=\sqrt{2n_{e}\frac{t_{e}}{m_{e}}}@f}
///  @f{equation}{Z\left(\zeta\right)=-\sqrt{\pi}e^{-\zeta^{s}}\left(efri\left(\zeta\right)-i\right)@f}
///  Where @f$efri\left(\zeta\right)@f$ is the imaginary error function.
///
///  @subsubsection absorption_model_hotexpand Expansion Terms
///  The hot plasma dispersion function can be split to a hot plasma term and a
///  cold plasma term.
///  @f{equation}{D_{c}\left(\vec{x},\vec{k},\omega\right)=-\frac{P}{2}\left(1+\frac{\omega_{ce}}{\omega}\Gamma_{0}+\left(1-\frac{\omega^{2}_{ce}}{\omega^{2}}\Gamma_{1}\right)\right)@f}
///  Then the hot plasma term is
///  @f{equation}{D_{h}\left(\vec{x},\vec{k},\omega\right)=-\left(1+\frac{\omega_{ce}}{\omega}\right)n_{||}\frac{v_{e}}{c}\left(\Gamma_{1}+\Gamma_{2}+\frac{\vec{n}_{\perp}\cdot\vec{n}_{\perp}}{2n^{2}_{||}}\frac{\omega^{2}}{\omega^{2}_{ce}}\frac{v_{e}}{c}\zeta\Gamma_{5}\right)\left(\frac{1}{Z\left(\zeta\right)}+\zeta\right)@f}
///
///  <hr>
///  @section absorption_model_root Root Find
///  One way to solve for the imaginary component is to locate the root of the
///  hot plasma dispersion function using the cold plasma solution as an initial
///  guess. We start by redefining @f$\vec{k}=k_{amp}\hat{k}@f$ now we can solve
///  for the complex value of @f$k_{amp}@f$ using a
///  @ref solver::newton "Newton method".
///
///  <hr>
///  @section absorption_model_damping Weak Damping
///  Using the cold and hot expansion
///  @f{equation}{k_{amp}=\sqrt{\vec{k}\cdot\vec{k}}-\frac{D_{h}}{\hat{k}\cdot\frac{\partial D_{c}}{\partial \vec{k}}}@f}
///
///  <hr>
///  @section absorption_model_devel Developing new absorption models
///  This section is intended for code developers and outlines how to create new
///  absorption models. New absorption models can be created from a subclass
///  of @ref absorption::method or any other existing
///  absorption class and overloading class methods.
///  @code
///  template<jit::complex_scalar T, bool SAFE_MATH=true>
///  class new_absorption final : public method<T, SAFE_MATH> {
///      new_absorption(graph::shared_leaf<T, SAFE_MATH> kamp,
///                     graph::shared_leaf<T, SAFE_MATH> w,
///                     graph::shared_leaf<T, SAFE_MATH> kx,
///                     graph::shared_leaf<T, SAFE_MATH> ky,
///                     graph::shared_leaf<T, SAFE_MATH> kz,
///                     graph::shared_leaf<T, SAFE_MATH> x,
///                     graph::shared_leaf<T, SAFE_MATH> y,
///                     graph::shared_leaf<T, SAFE_MATH> z,
///                     graph::shared_leaf<T, SAFE_MATH> t,
///                     equilibrium::shared<T, SAFE_MATH> &eq,
///                     const std::string &filename="",
///                     const size_t index=0) {
///          ...
///      }
///
///      void compile() {
///          ...
///      }
///
///      void run(const size_t time_index) {
///          ...
///      }
///  };
///  @endcode
//------------------------------------------------------------------------------
#ifndef absorption_h
#define absorption_h

#include <thread>

#include "newton.hpp"
#include "output.hpp"
#include "dispersion.hpp"

/// Namespace for power absorption models.
namespace absorption {
//******************************************************************************
//  Base class
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Base class for absorption models.
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

//------------------------------------------------------------------------------
///  @brief Compile the work items.
//------------------------------------------------------------------------------
        virtual void compile()=0;

//------------------------------------------------------------------------------
///  @brief Run the workflow.
///
///  @param[in] time_index The time index to run the case for.
//------------------------------------------------------------------------------
        virtual void run(const size_t time_index)=0;
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

///  Residual.
        graph::shared_leaf<T, SAFE_MATH> residual;

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
///  @param[in] kamp     Initial kamp.
///  @param[in] w        Initial w.
///  @param[in] kx       Initial kx.
///  @param[in] ky       Initial ky.
///  @param[in] kz       Initial kz.
///  @param[in] x        Initial x.
///  @param[in] y        Initial y.
///  @param[in] z        Initial z.
///  @param[in] t        Initial t.
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
///  @brief Compile the work items.
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
///  @param[in] kamp     Initial kamp.
///  @param[in] w        Initial w.
///  @param[in] kx       Initial kx.
///  @param[in] ky       Initial ky.
///  @param[in] kz       Initial kz.
///  @param[in] x        Initial x.
///  @param[in] y        Initial y.
///  @param[in] z        Initial z.
///  @param[in] t        Initial t.
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
///  @brief Compile the work items.
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
