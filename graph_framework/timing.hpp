//------------------------------------------------------------------------------
///  @file tming.hpp
///  @brief Routines to time the ray execution.
//------------------------------------------------------------------------------

#ifndef timing_h
#define timing_h

#include <chrono>
#include <mutex>
#include <map>

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
///  Ending time of the measure.
        std::chrono::high_resolution_clock::time_point end;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a time diagnostic object.
///
///  @param[in] message Discription of what is being timed.
//------------------------------------------------------------------------------
        measure_diagnostic(const std::string message = "") :
        label(message), start(std::chrono::high_resolution_clock::now()) {}

//------------------------------------------------------------------------------
///  @brief Print the result.
//------------------------------------------------------------------------------
        void print() const {
            const auto end = std::chrono::high_resolution_clock::now();
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

    class measure_diagnostic_threaded {
    private:
///  Discription of what is being timed.
        const std::string label;
///  Starting time of the measure.
        std::map<size_t, std::chrono::high_resolution_clock::time_point> start;
///  Starting end of the measure.
        std::map<size_t, std::chrono::high_resolution_clock::time_point> end;
///  Lock to syncronize accross theads for the start time.
        std::mutex sync_start;
///  Lock to syncronize accross theads for the end time.
        std::mutex sync_end;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a time diagnostic object.
///
///  @param[in] message Discription of what is being timed.
//------------------------------------------------------------------------------
        measure_diagnostic_threaded(const std::string message = "") :
        label(message) {}

//------------------------------------------------------------------------------
///  @brief Start time for a given thread.
///
///  @param[in] thread_number The thread number to start the timer for.
//------------------------------------------------------------------------------
        void start_time(const size_t thread_number) {
            sync_start.lock();
            start[thread_number] = std::chrono::high_resolution_clock::now();
            sync_start.unlock();
        }

//------------------------------------------------------------------------------
///  @brief End time for a given thread.
///
///  @param[in] thread_number The thread number to start the timer for.
//------------------------------------------------------------------------------
        void end_time(const size_t thread_number) {
            const auto temp = std::chrono::high_resolution_clock::now();
            sync_end.lock();
            end[thread_number] = temp;
            sync_end.unlock();
        }

//------------------------------------------------------------------------------
///  @brief Print out the average time.
//------------------------------------------------------------------------------
        void print() {
            std::chrono::nanoseconds total_time_ns = static_cast<std::chrono::nanoseconds> (0);
            for (size_t i = 0, ie = start.size(); i < ie; i++) {
                const auto duration = end[i] - start[i];
                total_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds> (duration);
            }

            std::cout << "Average " << label << " time : " << total_time_ns.count()/start.size() << " ns"<< std::endl;
        }
    };
}

#endif /* timing_h */
