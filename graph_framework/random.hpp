//------------------------------------------------------------------------------
///  @file random.hpp
///  @brief Random constants and distributions.
///
///  Defines random operations.
//------------------------------------------------------------------------------

#ifndef random_h
#define random_h

#include "node.hpp"

namespace graph {
//******************************************************************************
///  @brief Random state
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a random_state_node leaf.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class random_state_node final : public leaf_node<T, SAFE_MATH> {
    public:
//------------------------------------------------------------------------------
///  @brief Random state structure.
//------------------------------------------------------------------------------
        struct mt_state {
///  State array.
            std::array<uint32_t, 624> array;
///  State index.
            uint16_t index;
        };

//------------------------------------------------------------------------------
///  @brief Construct a constant node from a vector.
///
///  @param[in] size Number of random states.
///  @param[in] seed Inital random seed.
//------------------------------------------------------------------------------
        random_state_node(const size_t size,
                          const uint32_t seed=0) :
        leaf_node<T, SAFE_MATH> (random_state_node::to_string(), 1, false) {
            for (uint32_t i = 0; i < size; i++) {
                states.push_back(initalize_state(seed + i));
            }
        }

//------------------------------------------------------------------------------
///  @brief Evaluate the results of random_state_node.
///
///  @returns The value of random_state.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> result;
            return result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the random_state_node.
///
///  @returns Reduced graph from random_state.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> df(shared_leaf<T, SAFE_MATH> x) {
            return zero<T, SAFE_MATH> ();
        }

//------------------------------------------------------------------------------
///  @brief Compile preamble.
///
///  @param[in,out] stream          String buffer stream.
///  @param[in,out] registers       List of defined registers.
///  @param[in,out] visited         List of visited nodes.
///  @param[in,out] usage           List of register usage count.
///  @param[in,out] textures1d      List of 1D textures.
///  @param[in,out] textures2d      List of 2D textures.
///  @param[in,out] avail_const_mem Available constant memory.
//------------------------------------------------------------------------------
        virtual void compile_preamble(std::ostringstream &stream,
                                      jit::register_map &registers,
                                      jit::visiter_map &visited,
                                      jit::register_usage &usage,
                                      jit::texture1d_list &textures1d,
                                      jit::texture2d_list &textures2d,
                                      int &avail_const_mem) {
            if (visited.find(this) == visited.end()) {
                stream << "struct mt_state {"               << std::endl
                       << "    array<uint32_t, 624> array;" << std::endl
                       << "    uint16_t index;"             << std::endl
                       << "};"                              << std::endl;
#ifdef SHOW_USE_COUNT
            } else {
                ++usage[this];
#endif
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @param[in,out] indices   List of defined indices.
///  @param[in]     usage     List of register usage count.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                jit::register_map &indices,
                const jit::register_usage &usage) {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T, SAFE_MATH> x) {
            if (this == x.get()) {
                return true;
            }
            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << "state";
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to vizgraph.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> to_vizgraph(std::stringstream &stream,
                                                      jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                const std::string name = jit::to_string('r', this);
                registers[this] = name;
                stream << "    " << name
                       << " [label = \"state\", shape = box, style = \"rounded,filled\", fillcolor = black, fontcolor = white];" << std::endl;
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Test if all the subnodes terminate in variables.
///
///  @returns True if all the subnodes terminate in variables.
//------------------------------------------------------------------------------
        virtual bool is_all_variables() const {
            return false;
        }

//------------------------------------------------------------------------------
///  @brief Get the exponent of a power.
///
///  @returns The exponent of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_exponent() const {
            return constant<T, SAFE_MATH> (static_cast<T> (1.0));
        }

//------------------------------------------------------------------------------
///  @brief Get the size of the random state vector in bytes.
///
///  @returns The size of the state vector in bytes.
//------------------------------------------------------------------------------
        size_t size() {
            return states.size();
        }

//------------------------------------------------------------------------------
///  @brief Get the size of the random state vector in bytes.
///
///  @returns The size of the state vector in bytes.
//------------------------------------------------------------------------------
        size_t get_size_bytes() {
            return size()*sizeof(mt_state);
        }

//------------------------------------------------------------------------------
///  @brief Get the size of the random state vector in bytes.
///
///  @returns The size of the state vector in bytes.
//------------------------------------------------------------------------------
        mt_state *data() {
            return states.data();
        }

    private:
///  State buffer.
        std::vector<mt_state> states;

//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string() {
            return "random_state";
        }

//------------------------------------------------------------------------------
///  @brief  Initalize the random states.
///
///  @param[in] seed Inital random seed.
///  @returns A seeded state.
//------------------------------------------------------------------------------
        mt_state initalize_state(const uint32_t seed) {
            mt_state state;
            state.array[0] = seed;
            for (uint16_t i = 1, ie = state.array.size(); i < ie; i++) {
                state.array[i] = 1812433253U*(state.array[i - 1]^(state.array[i - 1] >> 30)) + i;
            }
            state.index = 0;

            return state;
        }
    };

//------------------------------------------------------------------------------
///  @brief Define random_state convience function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] size Number of random states.
///  @param[in] seed Inital random seed.
///  @returns A reduced random_state node.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> random_state(const size_t size,
                                           const uint32_t seed=0) {
        auto temp = std::make_shared<random_state_node<T, SAFE_MATH>> (size, seed)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::caches.nodes.find(i) ==
                leaf_node<T, SAFE_MATH>::caches.nodes.end()) {
                leaf_node<T, SAFE_MATH>::caches.nodes[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::caches.nodes[i])) {
                return leaf_node<T, SAFE_MATH>::caches.nodes[i];
            }
        }
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

///  Convenience type alias for shared sqrt nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_random_state = std::shared_ptr<random_state_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a random_state node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_random_state<T, SAFE_MATH> random_state_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<random_state_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Random constant.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a random_node leaf.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class random_node final : public straight_node<T, SAFE_MATH> {
    private:

//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string() {
            return "random";
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a constant node from a vector.
///
///  @param[in] x Argument.
//------------------------------------------------------------------------------
        random_node(shared_random_state<T, SAFE_MATH> x) :
        straight_node<T, SAFE_MATH> (x, random_node::to_string()) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of random node.
///
///  @returns The value of random.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> result;
            return result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the random node.
///
///  @returns Reduced graph from random.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d sqrt(a)/dx = 1/(2*sqrt(a))da/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> df(shared_leaf<T, SAFE_MATH> x) {
            return zero<T, SAFE_MATH> ();
        }

//------------------------------------------------------------------------------
///  @brief Compile preamble.
///
///  @param[in,out] stream          String buffer stream.
///  @param[in,out] registers       List of defined registers.
///  @param[in,out] visited         List of visited nodes.
///  @param[in,out] usage           List of register usage count.
///  @param[in,out] textures1d      List of 1D textures.
///  @param[in,out] textures2d      List of 2D textures.
///  @param[in,out] avail_const_mem Available constant memory.
//------------------------------------------------------------------------------
        virtual void compile_preamble(std::ostringstream &stream,
                                      jit::register_map &registers,
                                      jit::visiter_map &visited,
                                      jit::register_usage &usage,
                                      jit::texture1d_list &textures1d,
                                      jit::texture2d_list &textures2d,
                                      int &avail_const_mem) {
            this->arg->compile_preamble(stream, registers,
                                        visited, usage,
                                        textures1d, textures2d,
                                        avail_const_mem);

            if (visited.find(this) == visited.end()) {
                jit::add_type<T> (stream);
                stream << " random(";
                if constexpr (jit::use_metal<T> ()) {
                    stream << "device ";
                }
                stream <<"mt_state &state) {"                                       << std::endl
                       << "    uint16_t k = state.index;"                           << std::endl
                       << "    int16_t j = k - 623;"                                << std::endl
                       << "    if (j < 0) {"                                        << std::endl
                       << "        j += 624;"                                       << std::endl
                       << "    }"                                                   << std::endl
                       << "    uint32_t x = (state.array[k] & 0xffffffffU << 31) |" << std::endl
                       << "                 (state.array[j] & 0xffffffffU >> 1);"   << std::endl
                       << "    uint32_t xA = x >> 1;"                               << std::endl
                       << "    if (x & 0x00000001U) {"                              << std::endl
                       << "        xA ^= 0x9908b0dfU;"                              << std::endl
                       << "    }"                                                   << std::endl
                       << "    j = k - 227;"                                        << std::endl
                       << "    if (j < 0) {"                                        << std::endl
                       << "        j += 624;"                                       << std::endl
                       << "    }"                                                   << std::endl
                       << "    x = state.array[j]^xA;"                              << std::endl
                       << "    state.array[k++] = x;"                               << std::endl
                       << "    state.index = k;"                                    << std::endl
                       << "    uint32_t y = x^(x >> 11);"                           << std::endl
                       << "    y = y^((y << 7) & 0x9d2c5680U);"                     << std::endl
                       << "    y = y^((y << 15) & 0xefc60000U);"                    << std::endl
                       << "    return static_cast<";
                jit::add_type<T> (stream);
                stream << "> (y^(y >> 18));"                                        << std::endl
                       << "}"                                                       << std::endl;
#ifdef SHOW_USE_COUNT
            } else {
                ++usage[this];
#endif
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @param[in,out] indices   List of defined indices.
///  @param[in]     usage     List of register usage count.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                jit::register_map &indices,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> a = this->arg->compile(stream,
                                                                 registers,
                                                                 indices,
                                                                 usage);

                registers[this] = "random(" + registers[a.get()] + ")";
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(shared_leaf<T, SAFE_MATH> x) {
            if (this == x.get()) {
                return true;
            }
            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << "state";
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to vizgraph.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> to_vizgraph(std::stringstream &stream,
                                                      jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                const std::string name = jit::to_string('r', this);
                registers[this] = name;
                stream << "    " << name
                       << " [label = \"state\", shape = box, style = \"rounded,filled\", fillcolor = black, fontcolor = white];" << std::endl;

                auto a = this->arg->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[a.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Test if all the subnodes terminate in variables.
///
///  @returns True if all the subnodes terminate in variables.
//------------------------------------------------------------------------------
        virtual bool is_all_variables() const {
            return false;
        }

//------------------------------------------------------------------------------
///  @brief Get the exponent of a power.
///
///  @returns The exponent of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_exponent() const {
            return constant<T, SAFE_MATH> (static_cast<T> (1.0));
        }
    };

//------------------------------------------------------------------------------
///  @brief Define random convience function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] state Random state node.
///  @returns A reduced random node.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> random(shared_random_state<T, SAFE_MATH> state) {
        auto temp = std::make_shared<random_node<T, SAFE_MATH>> (state)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::caches.nodes.find(i) ==
                leaf_node<T, SAFE_MATH>::caches.nodes.end()) {
                leaf_node<T, SAFE_MATH>::caches.nodes[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::caches.nodes[i])) {
                return leaf_node<T, SAFE_MATH>::caches.nodes[i];
            }
        }
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

///  Convenience type alias for shared sqrt nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_random = std::shared_ptr<random_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a random node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_random_state<T, SAFE_MATH> random_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<random_node<T, SAFE_MATH>> (x);
    }

//------------------------------------------------------------------------------
///  @brief Create a random_scale constant.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @returns A random_scale constant.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    constexpr shared_leaf<T, SAFE_MATH> random_scale() {
        return constant<T, SAFE_MATH> (static_cast<T> (std::numeric_limits<uint32_t>::max()));
    }
}

#endif /* random_h */
