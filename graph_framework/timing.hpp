//------------------------------------------------------------------------------
///  @file tming.hpp
///  @brief Routines to time the ray execution.
//------------------------------------------------------------------------------

#ifndef timing_h
#define timing_h

#include <chrono>

namespace timeing {
//------------------------------------------------------------------------------
///  @brief A timing object.
//------------------------------------------------------------------------------
    class measure_diagnostic {
    private:
///  Discription of what is being timed.
        const std::string label;
///  Starting time of the measure.
        const std::chrono::high_resolution_clock::time_point start;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a time diagnostic object.
///
///  @param[in] message Discription of what is being timed.
//------------------------------------------------------------------------------
        measure_diagnostic(const std::string message) :
        label(message), start(std::chrono::high_resolution_clock::now()) {}

//------------------------------------------------------------------------------
///  @brief Stop the timer.
//------------------------------------------------------------------------------
        void stop() const {
            const std::chrono::high_resolution_clock::time_point end =
                      std::chrono::high_resolution_clock::now();
            const auto total_time = end - start;
            const std::chrono::nanoseconds total_time_ns =
                      std::chrono::duration_cast<std::chrono::nanoseconds> (total_time);

            std::cout << std::endl << "  " << label << " : ";

            if (total_time_ns.count() < 1000) {
                std::cout << total_time_ns.count()               << " ns"  << std::endl;
            } else if (total_time_ns.count() < 1000000) {
                std::cout << total_time_ns.count()/1000.0        << " μs"  << std::endl;
            } else if (total_time_ns.count() < 1000000000) {
                std::cout << total_time_ns.count()/1000000.0     << " ms"  << std::endl;
            } else if (total_time_ns.count() < 60000000000) {
                std::cout << total_time_ns.count()/1000000000.0  << " s"   << std::endl;
            } else if (total_time_ns.count() < 3600000000000) {
                std::cout << total_time_ns.count()/60000000000.0 << " min" << std::endl;
            } else {
                std::cout << total_time_ns.count()/3600000000000 << " h"   << std::endl;
            }
            std::cout << std::endl;
        }
    };
}

#endif /* timing_h */
