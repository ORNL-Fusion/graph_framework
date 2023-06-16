//------------------------------------------------------------------------------
///  @file erfi\_test.cpp
///  @brief Tests for the buffer backend.
//------------------------------------------------------------------------------

//  Turn on asserts even in release builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#include <netcdf.h>

#include "../graph_framework/special_functions.hpp"

//------------------------------------------------------------------------------
///  @brief Tests of the erfi
///
///  @params[in] tolarance Test tolarance.
//------------------------------------------------------------------------------
template<typename T> void test_erfi(const T tolarance) {
    int ncid;
    nc_open(ERFI_FILE, NC_NOWRITE, &ncid);

    int dimid;
    size_t size;
    nc_inq_dimid(ncid, "size", &dimid);
    nc_inq_dimlen(ncid, dimid, &size);

    std::vector<double> temp_buffer(size);

    int varid;
    nc_inq_varid(ncid, "x", &varid);
    nc_get_var(ncid, varid, temp_buffer.data());

    std::vector<T> x(temp_buffer.begin(), temp_buffer.end());

    nc_inq_varid(ncid, "y", &varid);
    nc_get_var(ncid, varid, temp_buffer.data());

    std::vector<T> y(temp_buffer.begin(), temp_buffer.end());

    nc_inq_varid(ncid, "re", &varid);
    nc_get_var(ncid, varid, temp_buffer.data());

    std::vector<T> re(temp_buffer.begin(), temp_buffer.end());

    nc_inq_varid(ncid, "img", &varid);
    nc_get_var(ncid, varid, temp_buffer.data());

    std::vector<T> img(temp_buffer.begin(), temp_buffer.end());

    for (size_t i = 5; i < size; i++) {
        const std::complex<T> z(x[i], y[i]);
        const std::complex<T> gold(re[i], img[i]);
        const std::complex<T> test = special::erfi<T> (z);
        if (std::isinf(std::real(gold)) && std::isinf(std::imag(gold))) {
            assert(gold == test && "Results don't match.");
        } else if (std::isinf(std::real(gold))) {
            assert(std::real(gold) == std::real(test) &&
                   "Real parts don't match.");
            if (std::imag(test) != std::imag(gold)) {
                assert(std::abs(static_cast<T> (1) - std::imag(test)/std::imag(gold)) <= tolarance &&
                       "Imaginary parts don't match.");
            }
        } else if (std::isinf(std::imag(gold))) {
            assert(std::imag(gold) == std::imag(test) &&
                   "Imaginary parts don't match.");
            if (std::real(test) != std::real(gold)) {
                assert(std::abs(static_cast<T> (1) - std::real(test)/std::real(gold)) <= tolarance &&
                       "Real parts don't match.");
            }
        } else if (!std::isinf(std::real(test)) && !std::isinf(std::imag(test))) {
            std::cout << std::abs(static_cast<T> (1) - test/gold) << std::endl;
            assert(std::abs(static_cast<T> (1) - test/gold) <= tolarance &&
                   "Results don't match.");
        }
    }
}

//------------------------------------------------------------------------------
///  @brief Main program of the test.
///
///  @params[in] argc Number of commandline arguments.
///  @params[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
int main(int argc, const char * argv[]) {
    //test_erfi<float> (2.0E-5);
    test_erfi<double> (2.0E-14);
}
