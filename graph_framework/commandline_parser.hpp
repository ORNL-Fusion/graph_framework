//------------------------------------------------------------------------------
///  @file commandline_parser.hpp
///  @brief Parsing routines for the command line.
//------------------------------------------------------------------------------

#ifndef commandline_parser_h
#define commandline_parser_h

#include <map>
#include <type_traits>
#include <utility>
#include <mutex>

/// Name space for the command line parsing.
namespace commandline {
///  Lock to syncronize netcdf accross threads.
    static std::mutex sync;

//------------------------------------------------------------------------------
///  @brief Parser class
//------------------------------------------------------------------------------
    class parser {
    private:
///  Command line options with values.
        std::map<std::string, std::pair<bool, std::string>> options;
///  Parsed commands.
        std::map<std::string, std::string> parsed_options;
///  Value options.
        std::map<std::string, std::set<std::string>> option_values;
///  Command name.
        const std::string command;

//------------------------------------------------------------------------------
///  @brief Take end string.
///
///  @param[in] string    String to take.
///  @param[in] character Character to split at.
//------------------------------------------------------------------------------
        static std::string_view take_end(const char *string,
                                         const char character) {
            std::string_view view(string);
            return view.substr(view.find_last_of(character) + 1);
        }

    public:
//------------------------------------------------------------------------------
///  @brief Default constructor
//------------------------------------------------------------------------------
        parser(const char *name) : 
        command(take_end(name, '/')) {
            options.try_emplace("help", false, "Show this help.");
        }

//------------------------------------------------------------------------------
///  @brief Add commandline option.
///
///  @param[in] option      The command option.
///  @param[in] takes_value Flag to indicate the option takes a value.
///  @param[in] help_text   The help text of the option.
///  @param[in] values      An optional set of values.
//------------------------------------------------------------------------------
        void add_option(const std::string &option,
                        const bool takes_value,
                        const std::string &help_text,
                        const std::set<std::string> &values = {}) {
            options.try_emplace(option, takes_value, help_text);
            if (!values.empty()) {
                option_values.try_emplace(option, values);
            }
        }

//------------------------------------------------------------------------------
///  @brief Display help.
///
///  @param[in] command Name of the program.
//------------------------------------------------------------------------------
        void show_help(const std::string &command) const {
            sync.lock();
            size_t longest = 0;
            for (auto &[option, value] : options) {
                longest = std::max(longest, option.size());
            }
            std::cout << "USAGE: " << command << " [--options] [--options=with_value]" << std::endl << std::endl;
            std::cout << "OPTIONS:" << std::endl;
            for (auto &[option, value] : options) {
                std::cout << "  --" << option
                          << (std::get<bool> (value) ? "= " : "  ");
                for (size_t i = option.size(); i < longest; i++) {
                    std::cout << " ";
                }
                std::cout << std::get<std::string> (value) << std::endl;
                if (option_values.find(option) != option_values.cend()) {
                    for (auto &option_value : option_values.at(option)) {
                        std::cout << "    ";
                        for (size_t i = 0; i < longest; i++) {
                            std::cout << " ";
                        }
                        std::cout << "   * " << option_value << std::endl;
                    }
                }
            }
            std::cout << std::endl;
            sync.unlock();
            exit(0);
        }

//------------------------------------------------------------------------------
///  @brief Parse the command line.
///
///  @param[in] argc Number of commandline arguments.
///  @param[in] argv Array of commandline arguments.
//------------------------------------------------------------------------------
        void parse(const int argc, const char * argv[]) {
            for (int i = 1; i < argc; i++) {
                std::string_view view(argv[i]);
                const size_t option_end = view.find_first_of('=');
                std::string option(view.substr(2, option_end - 2));
                if (is_option_set(option)) {
                    std::cout << "Warning --" << option << " set more than once." << std::endl;
                    std::cout << "  Overwriting --" << option << std::endl;
                }
                if (options.find(option) == options.cend()) {
                    std::cout << "UNKNOWN OPTION: " << view << std::endl << std::endl;
                    show_help(std::string(argv[0]));
                }
                if (option_end != view.size()) {
                    parsed_options[option] = std::string(view.substr(option_end + 1));
                    if ((option_values.find(option) != option_values.cend()) &&
                        !(option_values.at(option).find(parsed_options[option]) !=
                          option_values.at(option).cend())) {
                        std::cout << "UNKNOWN VALUE: " << parsed_options[option]
                                  << std::endl << std::endl;
                    }
                } else {
                    parsed_options[option] = "";
                }
            }

            if (is_option_set("help")) {
                show_help(command);
            }
        }

//------------------------------------------------------------------------------
///  @brief Check if option is set.
///
///  @param[in] option The option to check.
//------------------------------------------------------------------------------
        bool is_option_set(const std::string &option) const {
            return parsed_options.find(option) != parsed_options.cend();
        }

//------------------------------------------------------------------------------
///  @brief Get the option value.
///
///  @tparam T Type of the value.
//------------------------------------------------------------------------------
        template<typename T>
        T get_option_value(const std::string &option) const {
            if (is_option_set(option)) {
                std::string value = parsed_options.at(option);
                if constexpr (std::is_same<T, std::string> ()) {
                    return value;
                } else if constexpr (std::is_same<T, float> () ||
                                     std::is_same<T, std::complex<float>> ()) {
                    return static_cast<T> (std::stof(value));
                } else if constexpr (std::is_same<T, double> () ||
                                     std::is_same<T, std::complex<double>> ()) {
                    return static_cast<T> (std::stod(value));
                } else if constexpr (std::is_same<T, int> ()) {
                    return std::stoi(value);
                } else if constexpr (std::is_same<T, long> ()) {
                    return std::stol(value);
                } else if constexpr (std::is_same<T, unsigned long> ()) {
                    return std::stoul(value);
                }
            } else {
                std::cout << "Expected option : --" << option << std::endl << std::endl;
                show_help(command);
            }
            return NULL;
        }
    };
}

#endif /* commandline_parser_h */
