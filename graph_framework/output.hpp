//------------------------------------------------------------------------------
///  @file output.hpp
///  @brief Impliments output files in a netcdf format.
//------------------------------------------------------------------------------

#ifndef output_h
#define output_h

#include <netcdf.h>

#include "jit.hpp"

namespace output {
//------------------------------------------------------------------------------
///  @brief Class representing a netcdf based output file.
//------------------------------------------------------------------------------
    template<typename T>
    class result_file {
    private:
///  Netcdf file id.
        int ncid;
///  Unlimited dimension.
        int unlimited_dim;
///  Number of rays dimension.
        int num_rays_dim;
///  Dimension of ray. 1 for real, 2 for complex.
        int ray_dim;
///  Number of rays.
        const size_t num_rays;

//------------------------------------------------------------------------------
///  @brief Struct map variables to a gpu buffer.
//------------------------------------------------------------------------------
        struct variable {
///  Variable id.
            int id;
///  Pointer to the gpu buffer.
            T *buffer;
        };
///  Variable list.
        std::vector<variable> variables;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a result file.
///
///  @params[in] filename Name of the result file.
///  @params[in] num_rays Number of rays.
//------------------------------------------------------------------------------
        result_file(const std::string &filename="",
                    const size_t num_rays=0) : num_rays(num_rays) {
            
            nc_create(filename.c_str(),
                      filename.empty() || num_rays == 0 ? NC_DISKLESS : NC_CLOBBER,
                      &ncid);

            nc_def_dim(ncid, "time", NC_UNLIMITED, &unlimited_dim);
            nc_def_dim(ncid, "num_rays", num_rays, &num_rays_dim);
            if constexpr (jit::is_complex<T> ()) {
                nc_def_dim(ncid, "ray_dim", 2, &ray_dim);
                nc_def_dim(ncid, "num_rays", num_rays*2, &num_rays_dim);
            } else {
                nc_def_dim(ncid, "ray_dim", 1, &ray_dim);
                nc_def_dim(ncid, "num_rays", num_rays*1, &num_rays_dim);
            }
        }

//------------------------------------------------------------------------------
///  @brief Destructor.
//------------------------------------------------------------------------------
        ~result_file() {
            nc_close(ncid);
        }

//------------------------------------------------------------------------------
///  @brief Create a variable.
///
///  @params[in] name    Name of the variable.
///  @params[in] node    Node to create variable for.
///  @params[in] context Context for the gpu.
//------------------------------------------------------------------------------
        template<bool SAFE_MATH=false>
        void create_variable(const std::string &name,
                             graph::shared_leaf<T, SAFE_MATH> &node,
                             jit::context<T, SAFE_MATH> &context) {
            variable var;
            const std::array<int, 3> dims = {unlimited_dim, num_rays_dim, ray_dim};
            if constexpr (jit::is_float<T> ()) {
                nc_def_var(ncid, name.c_str(), NC_FLOAT, dims.size(),
                           dims.data(), &var.id);
            } else {
                nc_def_var(ncid, name.c_str(), NC_DOUBLE, dims.size(),
                           dims.data(), &var.id);
            }

            var.buffer = context.get_buffer(node);

            variables.push_back(var);
        }

//------------------------------------------------------------------------------
///  @brief End define mode.
//------------------------------------------------------------------------------
        void end_define_mode() const {
            nc_enddef(ncid);
        }

//------------------------------------------------------------------------------
///  @brief Write step.
//------------------------------------------------------------------------------
        void write() {
            size_t size;
            nc_inq_dimlen(ncid, unlimited_dim, &size);
            const std::array<size_t, 3> start = {size, 0, 0};
            for (variable &var : variables) {
                if constexpr (jit::is_float<T> ()) {
                    if constexpr (jit::is_complex<T> ()) {
                        const std::array<size_t, 3> count = {1, num_rays, 2};
                        nc_put_vara_float(ncid, var.id, start.data(), count.data(),
                                          reinterpret_cast<float *> (var.buffer));
                    } else {
                        const std::array<size_t, 3> count = {1, num_rays, 1};
                        nc_put_vara_float(ncid, var.id, start.data(), count.data(),
                                          var.buffer);
                    }
                } else {
                    if constexpr (jit::is_complex<T> ()) {
                        const std::array<size_t, 3> count = {1, num_rays, 2};
                        nc_put_vara_double(ncid, var.id, start.data(), count.data(),
                                           reinterpret_cast<double *> (var.buffer));
                    } else {
                        const std::array<size_t, 3> count = {1, num_rays, 1};
                        nc_put_vara_double(ncid, var.id, start.data(), count.data(),
                                           var.buffer);
                    }
                }
            }
            
            nc_sync(ncid);
        }
    };
}

#endif /* output_h */
