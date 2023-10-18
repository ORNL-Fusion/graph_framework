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
///  @params[in] num_rays Number of rays to write.
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
                    const size_t num_rays=0,
                    const size_t index=0) :
        kamp(kamp), w(w), kx(kx), ky(ky), kz(kz), x(x), y(y), z(z), t(t),
        file(filename), dataset(file), index(index), work(index), sync([]{}) {
            auto kvec = graph::vector(kx, ky, kz);
            auto kunit = kvec->unit();
            auto klen = kvec->length();

            auto kx_amp = kamp*kunit->get_x();
            auto ky_amp = kamp*kunit->get_y();
            auto kz_amp = kamp*kunit->get_z();

            graph::input_nodes<typename DISPERSION_FUNCTION::base,
                               DISPERSION_FUNCTION::safe_math> inputs = {
                graph::variable_cast(this->kamp),
                graph::variable_cast(this->kx),
                graph::variable_cast(this->ky),
                graph::variable_cast(this->kz)
            };

            graph::map_nodes<typename DISPERSION_FUNCTION::base,
                             DISPERSION_FUNCTION::safe_math> setters = {
                {klen, graph::variable_cast(this->kamp)}
            };

            work.add_item(inputs, {}, setters, "root_find_init_kernel");

            inputs.push_back(graph::variable_cast(this->x));
            inputs.push_back(graph::variable_cast(this->y));
            inputs.push_back(graph::variable_cast(this->z));
            inputs.push_back(graph::variable_cast(this->t));
            inputs.push_back(graph::variable_cast(this->w));

            dispersion::dispersion_interface<DISPERSION_FUNCTION> D(w, kx_amp, ky_amp, kz_amp, x, y, z, t, eq);
            solver::newton(work, {kamp}, inputs, {D.get_d()});
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
            work.wait();
            
            sync.join();
            sync = std::thread([this] (const size_t index) -> void {
                dataset.write(file, index);
            }, time_index);
        }
    };
}

#endif /* absorption_h */
