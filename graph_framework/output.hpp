//------------------------------------------------------------------------------
///  @file output.hpp
///  @brief Impliments output files in a netcdf format.
//------------------------------------------------------------------------------

#ifndef output_h
#define output_h

#include <mutex>

#include <netcdf.h>

#include "jit.hpp"

namespace output {
///  Lock to syncronize netcdf accross threads.
    static std::mutex sync;

//------------------------------------------------------------------------------
///  @brief Check the error status.
///
///  @param[in] status Error status code.
//------------------------------------------------------------------------------
    static void check_error(const int status) {
        assert(status == NC_NOERR && nc_strerror(status));
    }

//------------------------------------------------------------------------------
///  @brief Class representing a netcdf based output file.
//------------------------------------------------------------------------------
    class result_file {
    private:
///  Netcdf file id.
        int ncid;
///  Unlimited dimension.
        int unlimited_dim;
///  Number of rays dimension.
        int num_rays_dim;
///  Number of rays.
        size_t num_rays;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a new result file.
///
///  @param[in] filename Name of the result file.
///  @param[in] num_rays Number of rays.
//------------------------------------------------------------------------------
        result_file(const std::string &filename,
                    const size_t num_rays) :
        num_rays(num_rays) {
            const std::string temp = filename.empty() ? jit::format_to_string(reinterpret_cast<size_t> (this)) :
                                                        filename;

            sync.lock();
            check_error(nc_create(temp.c_str(),
                                  filename.empty() || num_rays == 0 ? NC_DISKLESS : NC_CLOBBER,
                                  &ncid));

            check_error(nc_def_dim(ncid, "time", NC_UNLIMITED, &unlimited_dim));
            check_error(nc_def_dim(ncid, "num_rays",
                                   num_rays ? num_rays : 1,
                                   &num_rays_dim));
            sync.unlock();
        }

//------------------------------------------------------------------------------
///  @brief Open a new result file.
///
///  @param[in] filename Name of the result file.
//------------------------------------------------------------------------------
        result_file(const std::string &filename) {
            sync.lock();
            check_error(nc_open(filename.c_str(), NC_WRITE, &ncid));

            check_error(nc_inq_dimid(ncid, "time", &unlimited_dim));
            check_error(nc_inq_dimid(ncid, "num_rays", &num_rays_dim));
            check_error(nc_inq_dimlen(ncid, num_rays_dim, &num_rays));
            check_error(nc_redef(ncid));
            sync.unlock();
        }

//------------------------------------------------------------------------------
///  @brief Destructor.
//------------------------------------------------------------------------------
        ~result_file() {
            check_error(nc_close(ncid));
        }

//------------------------------------------------------------------------------
///  @brief End define mode.
//------------------------------------------------------------------------------
        void end_define_mode() const {
            sync.lock();
            check_error(nc_enddef(ncid));
            sync.unlock();
        }

//------------------------------------------------------------------------------
///  @brief Get ncid.
///
///  @returns The netcdf file id.
//------------------------------------------------------------------------------
        int get_ncid() const {
            return ncid;
        }

//------------------------------------------------------------------------------
///  @brief Get the number of rays.
///
///  @returns The number of rays.
//------------------------------------------------------------------------------
        size_t get_num_rays() const {
            return num_rays;
        }

//------------------------------------------------------------------------------
///  @brief Get the number of rays dimension.
///
///  @returns The number of rays dimension.
//------------------------------------------------------------------------------
        int get_num_rays_dim() const {
            return num_rays_dim;
        }

//------------------------------------------------------------------------------
///  @brief Get unlimited dimension.
///
///  @returns The unlimited dimension.
//------------------------------------------------------------------------------
        int get_unlimited_dim() const {
            return unlimited_dim;
        }

//------------------------------------------------------------------------------
///  @brief Get unlimited size.
///
///  @returns The size of the unlimited dimension.
//------------------------------------------------------------------------------
        size_t get_unlimited_size() const {
            size_t size;
            sync.lock();
            check_error(nc_inq_dimlen(ncid, unlimited_dim, &size));
            sync.unlock();

            return size;
        }

//------------------------------------------------------------------------------
///  @brief Sync the file.
//------------------------------------------------------------------------------
        void sync_file() const {
            sync.lock();
            check_error(nc_sync(ncid));
            sync.unlock();
        }
    };

//------------------------------------------------------------------------------
///  @brief Class representing a netcdf dataset.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
    template<jit::float_scalar T>
    class data_set {
    private:
///  Dimension of ray. 1 for real, 2 for complex.
        int ray_dim;
///  Dimension of a data item.
        std::array<int, 3> dims;
///  Data sizes.
        std::array<size_t, 3> count;
///  Get the ray dimension size.
        static constexpr size_t ray_dim_size = 1 + jit::complex_scalar<T>;
///  The NetCDF type.
        static constexpr nc_type type = jit::float_base<T> ? NC_FLOAT : NC_DOUBLE;

//------------------------------------------------------------------------------
///  @brief Struct to map variables to a gpu buffer.
//------------------------------------------------------------------------------
        struct variable {
///  Variable id.
            int id;
///  Pointer to the gpu buffer.
            T *buffer;
        };
///  Variable list.
        std::vector<variable> variables;

//------------------------------------------------------------------------------
///  @brief Struct to map references to a gpu buffer.
//------------------------------------------------------------------------------
        struct reference {
///  Variable id.
            int id;
///  Pointer to the gpu buffer.
            T *buffer;
///  Count stride.
            size_t ray_dim_size;
///  Stride length.
            std::ptrdiff_t stride;
///  Complex index.
            size_t index;
        };
///  References list.
        std::vector<reference> references;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a dataset.
///
///  @param[in] result A result file reference.
//------------------------------------------------------------------------------
        data_set(const result_file &result) {
            sync.lock();
            if constexpr (jit::complex_scalar<T>) {
                if (NC_NOERR != nc_inq_dimid(result.get_ncid(),
                                             "ray_dim_cplx",
                                             &ray_dim)) {
                    check_error(nc_def_dim(result.get_ncid(),
                                           "ray_dim_cplx", ray_dim_size,
                                           &ray_dim));
                }
            } else {
                if (NC_NOERR != nc_inq_dimid(result.get_ncid(),
                                             "ray_dim",
                                             &ray_dim)) {
                    check_error(nc_def_dim(result.get_ncid(),
                                           "ray_dim", ray_dim_size,
                                           &ray_dim));
                }
            }
            sync.unlock();

            dims = {
                result.get_unlimited_dim(),
                result.get_num_rays_dim(),
                ray_dim
            };

            count = {
                1,
                result.get_num_rays(),
                ray_dim_size
            };
        }

//------------------------------------------------------------------------------
///  @brief Create a variable.
///
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] result  A result file reference.
///  @param[in] name    Name of the variable.
///  @param[in] node    Node to create variable for.
///  @param[in] context Context for the gpu.
//------------------------------------------------------------------------------
        template<bool SAFE_MATH=false>
        void create_variable(const result_file &result,
                             const std::string &name,
                             graph::shared_leaf<T, SAFE_MATH> &node,
                             jit::context<T, SAFE_MATH> &context) {
            variable var;
            sync.lock();
            check_error(nc_def_var(result.get_ncid(), name.c_str(), type,
                                   static_cast<int> (dims.size()), dims.data(),
                                   &var.id));
            sync.unlock();

            var.buffer = context.get_buffer(node);
            variables.push_back(var);
        }

//------------------------------------------------------------------------------
///  @brief Load reference.
///
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] result  A result file reference.
///  @param[in] name    Name of the variable.
///  @param[in] node    Node to create variable for.
//------------------------------------------------------------------------------
        template<bool SAFE_MATH=false>
        void reference_variable(const result_file &result,
                                const std::string &name,
                                graph::shared_variable<T, SAFE_MATH> &&node) {
            reference ref;
            nc_type type;
            std::array<int, 3> ref_dims;

            sync.lock();
            check_error(nc_inq_varid(result.get_ncid(),
                                     name.c_str(),
                                     &ref.id));
            check_error(nc_inq_var(result.get_ncid(), ref.id, NULL, &type,
                                   NULL, ref_dims.data(), NULL));
            check_error(nc_inq_dimlen(result.get_ncid(), ref_dims[2],
                                      &ref.ray_dim_size));
            sync.unlock();

            assert(ref.ray_dim_size <= ray_dim_size &&
                   "Context variable too small to read reference.");

            ref.stride = ref.ray_dim_size < ray_dim_size ? 2 : 1;
            ref.buffer = node->data();
            ref.index = 0;
            references.push_back(ref);
        }

//------------------------------------------------------------------------------
///  @brief Load imaginary reference.
///
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] result  A result file reference.
///  @param[in] name    Name of the variable.
///  @param[in] node    Node to create variable for.
//------------------------------------------------------------------------------
        template<bool SAFE_MATH=false>
        void reference_imag_variable(const result_file &result,
                                     const std::string &name,
                                     graph::shared_variable<T, SAFE_MATH> &&node) {
            reference ref;
            nc_type type;
            std::array<int, 3> ref_dims;

            sync.lock();
            check_error(nc_inq_varid(result.get_ncid(),
                                     name.c_str(),
                                     &ref.id));
            check_error(nc_inq_var(result.get_ncid(), ref.id, NULL, &type,
                                   NULL, ref_dims.data(), NULL));
            check_error(nc_inq_dimlen(result.get_ncid(), ref_dims[2],
                                      &ref.ray_dim_size));
            sync.unlock();

            assert(ref.ray_dim_size == 2 &&
                   "Not a complex variable.");

            ref.ray_dim_size = 1;
            ref.stride = ref.ray_dim_size < ray_dim_size ? 2 : 1;
            ref.buffer = node->data();
            ref.index = 1;
            references.push_back(ref);
        }

//------------------------------------------------------------------------------
///  @brief Write step.
///
///  @param[in] result A result file reference.
//------------------------------------------------------------------------------
        void write(const result_file &result) {
            write(result, result.get_unlimited_size());
        }

//------------------------------------------------------------------------------
///  @brief Write step.
///
///  @param[in] result A result file reference.
///  @param[in] index  Time index.
//------------------------------------------------------------------------------
        void write(const result_file &result,
                   const size_t index) {
            const std::array<size_t, 3> start = {
                index, 0, 0
            };

            for (variable &var : variables) {
                sync.lock();
                if constexpr (jit::float_base<T>) {
                    if constexpr (jit::complex_scalar<T>) {
                        check_error(nc_put_vara_float(result.get_ncid(),
                                                      var.id,
                                                      start.data(),
                                                      count.data(),
                                                      reinterpret_cast<float *> (var.buffer)));
                    } else {
                        check_error(nc_put_vara_float(result.get_ncid(),
                                                      var.id,
                                                      start.data(),
                                                      count.data(),
                                                      var.buffer));
                    }
                } else {
                    if constexpr (jit::complex_scalar<T>) {
                        check_error(nc_put_vara_double(result.get_ncid(),
                                                       var.id,
                                                       start.data(),
                                                       count.data(),
                                                       reinterpret_cast<double *> (var.buffer)));
                    } else {
                        check_error(nc_put_vara_double(result.get_ncid(),
                                                       var.id,
                                                       start.data(),
                                                       count.data(),
                                                       var.buffer));
                    }
                }
                sync.unlock();
            }

            result.sync_file();
        }

//------------------------------------------------------------------------------
///  @brief Read step.
///
///  @param[in] result A result file reference.
///  @param[in] index  Time index.
//------------------------------------------------------------------------------
        void read(const result_file &result,
                  const size_t index) {
            const std::array<std::ptrdiff_t, 3> stride = {
                1, 1, 1
            };

            for (reference &ref : references) {
                const std::array<size_t, 3> ref_start = {
                    index, 0, ref.index
                };
                const std::array<size_t, 3> ref_count = {
                    1,
                    result.get_num_rays(),
                    ref.ray_dim_size
                };
                const std::array<std::ptrdiff_t, 3> map = {
                    1, ref.stride, 1
                };

                sync.lock();
                if constexpr (jit::float_base<T>) {
                    if constexpr (jit::complex_scalar<T>) {
                        check_error(nc_get_varm_float(result.get_ncid(),
                                                      ref.id,
                                                      ref_start.data(),
                                                      ref_count.data(),
                                                      stride.data(),
                                                      map.data(),
                                                      reinterpret_cast<float *> (ref.buffer)));
                    } else {
                        check_error(nc_get_varm_float(result.get_ncid(),
                                                      ref.id,
                                                      ref_start.data(),
                                                      ref_count.data(),
                                                      stride.data(),
                                                      map.data(),
                                                      ref.buffer));
                    }
                } else {
                    if constexpr (jit::complex_scalar<T>) {
                        check_error(nc_get_varm_double(result.get_ncid(),
                                                       ref.id,
                                                       ref_start.data(),
                                                       ref_count.data(),
                                                       stride.data(),
                                                       map.data(),
                                                       reinterpret_cast<double *> (ref.buffer)));
                    } else {
                        check_error(nc_get_varm_double(result.get_ncid(),
                                                       ref.id,
                                                       ref_start.data(),
                                                       ref_count.data(),
                                                       stride.data(),
                                                       map.data(),
                                                       ref.buffer));
                    }
                }

                sync.unlock();
            }
        }
    };
}

#endif /* output_h */
