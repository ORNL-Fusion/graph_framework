//------------------------------------------------------------------------------
///  @file particle_in_cell.hpp
///  @brief Utilities needed for a particle in cell code.
///
///  Defines graphs for use in Particle In Cell (PIC) codes.
//------------------------------------------------------------------------------

#ifndef particle_in_cell_h
#define particle_in_cell_h

#include <numbers>

#include "piecewise.hpp"
#include "workflow.hpp"
#include "random.hpp"
#include "logical.hpp"

namespace pic {
// FIXME: This should be in a separate file of physics constants.
///  Speed of light m/s.
    template<std::floating_point T>
    constexpr T c = static_cast<T> (299792458.0);
///  Vacuum permitivity F/m.
    template<std::floating_point T>
    constexpr T epsilon0 = static_cast<T> (8.8541878188E-12);
///  Fundamental charge coulombs.
    template<std::floating_point T>
    constexpr T q = static_cast<T> (1.602176634E-19);
///  Hydrogen mass kg.
    template<std::floating_point T>
    constexpr T m_hydrogen = static_cast<T> (1.67362192595E-27);
///  Atomic mass
    template<std::floating_point T>
    constexpr T m_atomic = static_cast<T> (1.66053906892E-27);
///  Electron mass kg.
    template<std::floating_point T>
    constexpr T m_electron = static_cast<T> (9.1093837139E-31);
///  Boltzman constant.
    template<std::floating_point T>
    constexpr T kb = static_cast<T> (1.380649E-23);

//------------------------------------------------------------------------------
///  @brief Characteristic factors.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    class characteristics {
    private:
//------------------------------------------------------------------------------
///  @brief Compute the characteristic mass.
///
///  @param[in] ion_masses Ion masses.
///  @returns (∑(m_i) + me)/(n_i + 1);
//------------------------------------------------------------------------------
        T make_m(const std::vector<T> &ion_masses) {
            T total_m = static_cast<T> (0);
            for (const T &mass : ion_masses) {
                total_m += mass;
            }
            return total_m/ion_masses.size();
        }

//------------------------------------------------------------------------------
///  @brief Compute the characteristic mass.
///
///  @param[in] ion_zs Ion Z.
///  @returns (∑(Z_i)*q + q)/(n_i + 1);
//------------------------------------------------------------------------------
        T make_q(const std::vector<uint8_t> &ion_zs) {
            T total_q = static_cast<T> (0);
            for (const uint8_t &z : ion_zs) {
                total_q += z*pic::q<T>;
            }
            return total_q/ion_zs.size();
        }

    public:
///  Mass
        const T m;
///  Charge
        const T q;
///  Electron density.
        const T ne;
///  Plasma Frequency.
        const T wpe;
///  Time.
        const T t;
///  Length
        const T l;
///  Velocity
        const T v;
///  Electron temperature;
        const T te;
///  Electric field;
        const T efield;
///  Magnetic field;
        const T bfield;

//------------------------------------------------------------------------------
///  @brief Construct the characteristics.
///
///  @param[in] ion_masses Ion masses for all species.
///  @param[in] ion_zs     Ion Z effective all species.
///  @param[in] ne         Characteristic density.
//------------------------------------------------------------------------------
        characteristics(const std::vector<T> &ion_masses,
                        const std::vector<uint8_t> &ion_zs,
                        const T ne) :
        m(make_m(ion_masses)), q(make_q(ion_zs)), ne(ne),
        wpe(std::sqrt(ne*q*q/(m*epsilon0<T>))),
        t(1/wpe), l(c<T>/wpe), v(c<T>), te(m*v*v/kb<T>), efield(m*c<T>/(q*t)),
        bfield(efield/c<T>) {}
    };

//------------------------------------------------------------------------------
///  @brief Parameter Class
//------------------------------------------------------------------------------
    template<std::floating_point T>
    class parameters {
    public:
///  Initial magnetic field
        const T b0;
///  Geometry
        const T a0;
///  Filter Iterations.
        const size_t filter_iterations;
///  Smoothing parameters.
        const T smoothing;
///  Time step.
        const T dt;
///  Parallel temperature.
        const T t_para;
///  Perpendicular temperature.
        const T t_perp;

//------------------------------------------------------------------------------
///  @brief Construct a parameters object.
///
///  @param[in] b0                Initial magnetic field.
///  @param[in] r1
///  @param[in] r2
///  @param[in] filter_iterations Number of times to apply smoothing filter.
///  @param[in] smoothing         Smoothing parameter.
///  @param[in] dt                Time step.
///  @param[in] norms A @ref pic::characteristics object
//------------------------------------------------------------------------------
        parameters(const T b0, const T r1, const T r2,
                   const size_t filter_iterations,
                   const T smoothing, const T dt,
                   const T t_para, const T t_perp,
                   const characteristics<T> &norms) :
        b0(b0/norms.bfield),
        a0(std::numbers::pi_v<T>*(r2*r2 - r1*r1)/(norms.l*norms.l)),
        filter_iterations(filter_iterations), smoothing(smoothing),
        dt(dt/norms.t), t_para(t_para), t_perp(t_perp) {}
    };

//------------------------------------------------------------------------------
///  @brief ion class.
///
///  These values need to be initialized using normalized quantities.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    class ion {
    public:
///  Atomic number.
        const T z;
///  Charge
        const T charge;
///  Particle mass
        const T mass;
///  Normalized Position
        graph::shared_leaf<T> x;
///  Normalized Parallel velocity.
        graph::shared_leaf<T> v_para;
///  Normalized Perpendicular velocity.
        graph::shared_leaf<T> v_perp;
///  Mesh Weights
        std::array<graph::shared_leaf<T>, 3> weights;
///  Number of real particles
        const T num_real;

//------------------------------------------------------------------------------
///  @brief Construct an ion object.
///
///  @param[in] mass     Ion mass.
///  @param[in] z        Ion Z.
///  @param[in] num_ions Number of ions.
///  @param[in] num_real Number of real particles.
///  @param[in] norms    A @ref pic::characteristics object.
//------------------------------------------------------------------------------
        ion(const T mass,
            const uint8_t z,
            const size_t num_ions,
            const T num_real,
            const characteristics<T> &norms) :
        z(z), charge(z*pic::q<T>/norms.q),
        mass(mass), num_real(num_real),
        x(graph::variable<T> (num_ions, "x")),
        v_para(graph::variable<T> (num_ions, "v_{||}")),
        v_perp(graph::variable<T> (num_ions, "v_{\\perp}")),
        weights({
            graph::variable<T> (num_ions, "w_{0}"),
            graph::variable<T> (num_ions, "w_{1}"),
            graph::variable<T> (num_ions, "w_{2}")
        }) {}

//------------------------------------------------------------------------------
///  @brief Get x case as variable.
///
///  @return x cast as a variable.
//------------------------------------------------------------------------------
        graph::shared_variable<T> get_x() const {
            return graph::variable_cast(x);
        }

//------------------------------------------------------------------------------
///  @brief Get the number of computational ions.
///
///  @return The number of particles.
//------------------------------------------------------------------------------
        size_t size() const {
            return get_x()->size();
        }

//------------------------------------------------------------------------------
///  @brief Get the data for x.
///
///  @return The number of particles.
//------------------------------------------------------------------------------
        T *x_data() const {
            return get_x()->data();
        }

//------------------------------------------------------------------------------
///  @brief Get x case as variable.
///
///  @return x cast as a variable.
//------------------------------------------------------------------------------
        graph::shared_variable<T> get_v_para() const {
            return graph::variable_cast(v_para);
        }

//------------------------------------------------------------------------------
///  @brief Get the data for the parallel velocity.
///
///  @return The number of particles.
//------------------------------------------------------------------------------
        T *v_para_data() const {
            return get_v_para()->data();
        }

//------------------------------------------------------------------------------
///  @brief Get x case as variable.
///
///  @return x cast as a variable.
//------------------------------------------------------------------------------
        graph::shared_variable<T> get_v_perp() const {
            return graph::variable_cast(v_perp);
        }

//------------------------------------------------------------------------------
///  @brief Get the data for the perpendicular velocity.
///
///  @return The number of particles.
//------------------------------------------------------------------------------
        T *v_perp_data() const {
            return graph::variable_cast(v_perp)->data();
        }

//------------------------------------------------------------------------------
///  @brief Conversion factor from super particles to real particles.
///
///  @returns The super to real conversion factor.
//------------------------------------------------------------------------------
        T super_to_real() const {
            return num_real/size();
        }

//------------------------------------------------------------------------------
///  @brief Define variables.
///
///  @param[in]     file A @ref output::result_file object to define variables.
///  @param[in,out] data A @ref output::data_set object to create variable.
///  @param[in,out] work A @ref workflow::manager object where data was
///                      computed.
///  @param[in]     tag  Unique identity for give the ion species.
//------------------------------------------------------------------------------
        void define_variables(const output::result_file &file,
                              output::data_set<T> &data,
                              workflow::manager<T> &work,
                              const std::string tag) {
            data.create_variable(file, "x_" + tag, x,  work.get_context());
            data.create_variable(file, "vpara_" + tag, v_para,
                                 work.get_context());
            data.create_variable(file, "vperp_" + tag, v_perp,
                                 work.get_context());
        }
    };

//------------------------------------------------------------------------------
///  @brief U Collision node.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    class apply_u_node final : public graph::no_derivative<T, false,
                                                           graph::n_branch_node<7, T, false>,
                                                           7> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] branches Array of branches.
///  @return A string rep of a the node.
//------------------------------------------------------------------------------
        static std::string to_string(std::array<graph::shared_leaf<T>, 7> &branches) {
            std::string s = "apply_u";
            for (auto &b : branches) {
                s += jit::format_to_string(reinterpret_cast<size_t> (b.get()));
            }
            return s;
        }

//------------------------------------------------------------------------------
///  @brief Define a CPU evaluator.
///
///  @param[in] x         Argument to apply collision to.
///  @param[in] i         Number of collision iterations.
///  @param[in] rand      A random value of 32 0s and 1s.
///  @param[in] mof       Mass over 2 e.
///  @param[in] tbnu_e_dt Temperature of species b times normalized collision rate.
///  @param[in] A         A collision factor.
///  @param[in] B         B collision factor.
//------------------------------------------------------------------------------
        static void func(backend::buffer<T> &x,
                         const backend::buffer<T> &i,
                         const backend::buffer<T> &rand,
                         const backend::buffer<T> &mof,
                         const backend::buffer<T> &tbnu_e_dt,
                         const backend::buffer<T> &A,
                         const backend::buffer<T> &B) {
            const size_t size = x.size();
            assert(size == i.size()         &&
                   size == rand.size()      &&
                   size == mof.size()       &&
                   size == tbnu_e_dt.size() &&
                   size == A.size()         &&
                   size == B.size()         &&
                   "Expected all arguments to have the same length.");

            for (size_t j = 0; j < size; j++) {
                const T mof_j = mof[j];
                T temp_x = x[j];
                const T tbnu_e_dt_j = tbnu_e_dt[j];
                if constexpr (std::same_as<T, float>) {
                    uint32_t rand_j = reintrepet_cast<uint32_t> (rand[j]);
                    for (uint8_t k = 0, ke = i[j]; k < ke; k++, rand_j >>= 1) {
                        const T E0 = mof_j*temp_x;
                        const int8_t rm = 4*(rand_j & 1) - 2;
                        const T C = rm*std::sqrt(tbnu_e_dt_j*E0);
                        temp_x = (E0*A + B + C)/mof_j;
                    }
                } else {
                    uint64_t rand_j = reintrepet_cast<uint64_t> (rand[j]);
                    for (uint8_t k = 0, ke = i[j]; k < ke; k++, rand_j >>= 1) {
                        const T E0 = mof_j*temp_x;
                        const int8_t rm = 4*(rand_j & 1) - 2;
                        const T C = rm*std::sqrt(tbnu_e_dt_j*E0);
                        temp_x = (E0*A + B + C)/mof_j;
                    }
                }
                x[j] = temp_x;
            }
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct an apply_u_node.
///
///  @param[in] x         Argument to apply collision to.
///  @param[in] i         Number of collision iterations.
///  @param[in] rand      A random value of 0s and 1s.
///  @param[in] mof       Mass over 2 e.
///  @param[in] tbnu_e_dt Temperature of species b times normalized collision rate.
///  @param[in] A         A collision factor.
///  @param[in] B         B collision factor.
//------------------------------------------------------------------------------
        apply_u_node(graph::shared_leaf<T> x,
                     graph::shared_leaf<T> i,
                     graph::shared_leaf<T> rand,
                     graph::shared_leaf<T> mof,
                     graph::shared_leaf<T> tbnu_e_dt,
                     graph::shared_leaf<T> A,
                     graph::shared_leaf<T> B) :
        graph::no_derivative<T, false,
                             graph::n_branch_node<7, T, false>> ({x, i, rand, mof, tbnu_e_dt, A, B},
                                                                 apply_u_node::to_string({x, i, rand, mof, tbnu_e_dt, A, B})) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of the applying the u operator.
///
///  result = apply_u(x, i, rand, mof, tbnu_e_dt, A, B)
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> x = this->branches[0]->evaluate();
            const backend::buffer<T> i = this->branches[1]->evaluate();
            const backend::buffer<T> rand = this->branches[2]->evaluate();
            const backend::buffer<T> mof = this->branches[3]->evaluate();
            const backend::buffer<T> tbnu_e_dt = this->branches[4]->evaluate();
            const backend::buffer<T> A = this->branches[5]->evaluate();
            const backend::buffer<T> B = this->branches[6]->evaluate();

            apply_u_node::func(x, i, rand, mof, tbnu_e_dt, A, B);
            return x;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the apply_u(x, i, rand, mof, tbnu_e_dt, A, B).
///
///  @returns Reduced graph from apply_u.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> reduce() {
            return this->shared_from_this();
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
                for (auto &b : this->branches) {
                    b->compile_preamble(stream, registers,
                                        visited, usage,
                                        textures1d, textures2d,
                                        avail_const_mem);
                }

                jit::add_type<T> (stream);
                stream << " apply_u(const ";
                jit::add_type<T> (stream);
                stream << " x, const uint8_t i, ";
                if constexpr (std::same_as<T, float>) {
                    stream << "uint32_t";
                } else {
                    stream << "uint64_t";
                }
                stream << " rand, const ";
                jit::add_type<T> (stream);
                stream << " mof, const ";
                jit::add_type<T> (stream);
                stream << " tbnu_e_dt, const ";
                jit::add_type<T> (stream);
                stream << " A, const ";
                jit::add_type<T> (stream);
                stream << " B) {"
                       << "    ";
                jit::add_type<T> (stream);
                stream << " temp_x = x;" << std::endl
                       << "    for (uint8_t j = 0; j < i; j++, rand >>= 1) {" << std::endl
                       << "        const ";
                jit::add_type<T> (stream);
                stream << "E0 = mof*temp_x;" << std::endl
                       << "        const uint8_t rm = 4*(rand & 1) - 2;" << std::endl
                       << "        const ";
                jit::add_type<T> (stream);
                stream << " C = rm*sqrt(tbnu_e_dt*E0);" << std::endl
                       << "        temp_x = (E0*A + B + C)/mof;" << std::endl
                       << "    }" << std::endl
                       << "    return temp_x;"
                       << "}";

                visited.insert(this);
#ifdef SHOW_USE_COUNT
                usage[this] = 1;
            } else {
                ++usage[this];
#endif
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in,out] stream     String buffer stream.
///  @param[in,out] registers  List of defined registers.
///  @param[in]     thread_mem List of defined thread memory registers.
///  @param[in]     usage      List of register usage count.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> compile(std::ostringstream &stream,
                                              jit::register_map &registers,
                                              const jit::register_map &thread_mem,
                                              const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                auto x = this->branches[0]->compile(stream, registers, thread_mem, usage);
                auto i = this->branches[1]->compile(stream, registers, thread_mem, usage);
                auto rand = this->branches[2]->compile(stream, registers, thread_mem, usage);
                auto mof = this->branches[3]->compile(stream, registers, thread_mem, usage);
                auto tbnu_e_dt = this->branches[4]->compile(stream, registers, thread_mem, usage);
                auto A = this->branches[5]->compile(stream, registers, thread_mem, usage);
                auto B = this->branches[6]->compile(stream, registers, thread_mem, usage);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = apply_u("
                       << registers[x.get()] << ", "
                       << registers[i.get()] << ", reinterpret_cast<";
                if constexpr (std::same_as<T, float>) {
                    stream << "uint32_t";
                } else {
                    stream << "uint64_t";
                }
                stream << "> ("
                       << registers[rand.get()] << "), "
                       << registers[mof.get()] << ", "
                       << registers[tbnu_e_dt.get()] << ", "
                       << registers[A.get()] << ", "
                       << registers[B.get()] << ")";
                this->endline(stream, usage);
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Query if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(graph::shared_leaf<T> x) {
            if (this == x.get()) {
                return true;
            }

            auto x_cast = apply_u_cast(x);
            bool temp;
            if (x_cast.get()) {
                temp = this->branches[0]->is_match(x_cast->get_arg(0));
                for (size_t i = 1; i < 7 && temp; i++) {
                    temp = temp && this->branches[i]->is_match(x_cast->get_arg(i));
                }
            }
             
            return temp;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << "\\apply_u{\\left(";
            this->branches[0]->to_latex();
            for (uint8_t i = 1; i < 7; i++) {
                std::cout << ", " << this->branches[i]->to_latex();
            }
            std::cout << "\\right)}";
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> remove_pseudo() {
            if (this->has_pseudo()) {
                return apply_u(this->branches[0]->remove_pseudo(),
                               this->branches[1]->remove_pseudo(),
                               this->branches[2]->remove_pseudo(),
                               this->branches[3]->remove_pseudo(),
                               this->branches[4]->remove_pseudo(),
                               this->branches[5]->remove_pseudo(),
                               this->branches[6]->remove_pseudo());
            }
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to vizgraph.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> to_vizgraph(std::stringstream &stream,
                                                  jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                const std::string name = jit::to_string('r', this);
                registers[this] = name;
                stream << "    " << name
                       << " [label = \"apply_u\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                for (auto &b : this->branches) {
                    auto temp = b->to_vizgraph(stream, registers);
                    stream << "    " << name << " -- " << registers[temp.get()] << ";" << std::endl;
                }
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build apply_u node.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] x         Argument to apply collision to.
///  @param[in] i         Number of collision iterations.
///  @param[in] rand      A random value of 32 0s and 1s.
///  @param[in] mof       Mass over 2 e.
///  @param[in] tbnu_e_dt Temperature of species b times normalized collision rate.
///  @param[in] A         A collision factor.
///  @param[in] B         B collision factor.
///  @returns A reduced apply_u node.
//------------------------------------------------------------------------------
    template<jit::float_scalar T>
    graph::shared_leaf<T> apply_u(graph::shared_leaf<T> x,
                                  graph::shared_leaf<T> i,
                                  graph::shared_leaf<T> rand,
                                  graph::shared_leaf<T> mof,
                                  graph::shared_leaf<T> tbnu_e_dt,
                                  graph::shared_leaf<T> A,
                                  graph::shared_leaf<T> B) {
        auto temp = std::make_shared<apply_u<T>> (x, i, rand, mof,
                                                  tbnu_e_dt, A, B)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
            if (graph::leaf_node<T>::caches.nodes.find(i) ==
                graph::leaf_node<T>::caches.nodes.end()) {
                graph::leaf_node<T>::caches.nodes[i] = temp;
                return temp;
            } else if (temp->is_match(graph::leaf_node<T>::caches.nodes[i])) {
                return graph::leaf_node<T>::caches.nodes[i];
            }
        }
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

///  Convenience type alias for shared sqrt nodes.
    template<jit::float_scalar T>
    using shared_apply_u = std::shared_ptr<apply_u_node<T>>;

//------------------------------------------------------------------------------
///  @brief Cast to a apply_u node.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attempted dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T>
    shared_apply_u<T> apply_u_cast(graph::shared_leaf<T> x) {
        return std::dynamic_pointer_cast<apply_u_node<T>> (x);
    }

//------------------------------------------------------------------------------
///  @brief Xi Collision node.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    class apply_xi_node final : public graph::no_derivative<T, false,
                                                            graph::n_branch_node<4, T, false>,
                                                            4> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] branches Array of branches.
///  @return A string rep of a the node.
//------------------------------------------------------------------------------
        static std::string to_string(std::array<graph::shared_leaf<T>, 4> &branches) {
            std::string s = "apply_xi";
            for (auto &b : branches) {
                s += jit::format_to_string(reinterpret_cast<size_t> (b.get()));
            }
            return s;
        }

//------------------------------------------------------------------------------
///  @brief Define a CPU evaluator.
///
///  @param[in] x       Argument to apply collision to.
///  @param[in] i       Number of collision iterations.
///  @param[in] rand    A random value of 32 0s and 1s.
///  @param[in] nu_D_dt Normalized step rate.
//------------------------------------------------------------------------------
        static void func(backend::buffer<T> &x,
                         const backend::buffer<T> &i,
                         const backend::buffer<T> &rand,
                         const backend::buffer<T> &nu_D_dt) {
            const size_t size = x.size();
            assert(size == i.size()       &&
                   size == rand.size()    &&
                   size == nu_D_dt.size() &&
                   "Expected all arguments to have the same length.");

            for (size_t j = 0; j < size; j++) {
                const T nu_D_dt_j = nu_D_dt[j];
                T temp_x = x[j];
                if constexpr (std::same_as<T, float>) {
                    uint32_t rand_j = reintrepet_cast<uint32_t> (rand[j]);
                    for (uint8_t k = 0, ke = i[j]; k < ke; k++, rand_j >>= 1) {
                        const T A = -temp_x*nu_D_dt_j;
                        const int8_t rm = 2*(rand_j & 1) - 1;
                        const T C = rm*std::sqrt((1 - temp_x*temp_x)*nu_D_dt_j);
                        temp_x += A + C;
                    }
                } else {
                    uint64_t rand_j = reintrepet_cast<uint64_t> (rand[j]);
                    for (uint8_t k = 0, ke = i[j]; k < ke; k++, rand_j >>= 1) {
                        const T A = -temp_x*nu_D_dt_j;
                        const int8_t rm = 2*(rand_j & 1) - 1;
                        const T C = rm*std::sqrt((1 - temp_x*temp_x)*nu_D_dt_j);
                        temp_x += A + C;
                    }
                }
                x[j] = temp_x;
            }
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct an apply_xi_node.
///
///  @param[in] x       Argument to apply collision to.
///  @param[in] i       Number of collision iterations.
///  @param[in] rand    A random value of 32 0s and 1s.
///  @param[in] nu_D_dt Normalized step rate.
//------------------------------------------------------------------------------
        apply_xi_node(graph::shared_leaf<T> x,
                      graph::shared_leaf<T> i,
                      graph::shared_leaf<T> rand,
                      graph::shared_leaf<T> nu_D_dt) :
        graph::no_derivative<T, false,
                             graph::n_branch_node<4, T, false>> ({x, i, rand, nu_D_dt},
                                                                 apply_xi_node::to_string({x, i, rand, nu_D_dt})) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of the applying the xi operator.
///
///  result = apply_xi(x, i, rand, nu_D_dt)
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> x = this->branches[0]->evaluate();
            const backend::buffer<T> i = this->branches[1]->evaluate();
            const backend::buffer<T> rand = this->branches[2]->evaluate();
            const backend::buffer<T> nu_D_dt = this->branches[3]->evaluate();

            apply_xi_node::func(x, i, rand, nu_D_dt);
            return x;
        }

//------------------------------------------------------------------------------
///  @brief Reduce the apply_zi(x, i, rand, nu_D_dt).
///
///  @returns Reduced graph from apply_xi.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> reduce() {
            return this->shared_from_this();
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
                for (auto &b : this->branches) {
                    b->compile_preamble(stream, registers,
                                        visited, usage,
                                        textures1d, textures2d,
                                        avail_const_mem);
                }

                jit::add_type<T> (stream);
                stream << " apply_xi(const ";
                jit::add_type<T> (stream);
                stream << " x, const uint8_t i, ";
                if constexpr (std::same_as<T, float>) {
                    stream << "uint32_t";
                } else {
                    stream << "uint64_t";
                }
                stream << " rand, const ";
                jit::add_type<T> (stream);
                stream << " nu_D_dt) {"
                       << "    ";
                jit::add_type<T> (stream);
                stream << " temp_x = x;" << std::endl
                       << "    for (uint8_t j = 0; j < i; j++, rand >>= 1) {" << std::endl
                       << "        const ";
                jit::add_type<T> (stream);
                stream << "A = -temp_x*nu_D_dt;" << std::endl
                       << "        const uint8_t rm = 2*(rand & 1) - 1;" << std::endl
                       << "        const ";
                jit::add_type<T> (stream);
                stream << " C = rm*sqrt((1 - temp_x*temp_x)*nu_D_dt);" << std::endl
                       << "        temp_x += A + C;" << std::endl
                       << "    }" << std::endl
                       << "    return temp_x;"
                       << "}";

                visited.insert(this);
#ifdef SHOW_USE_COUNT
                usage[this] = 1;
            } else {
                ++usage[this];
#endif
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in,out] stream     String buffer stream.
///  @param[in,out] registers  List of defined registers.
///  @param[in]     thread_mem List of defined thread memory registers.
///  @param[in]     usage      List of register usage count.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> compile(std::ostringstream &stream,
                                              jit::register_map &registers,
                                              const jit::register_map &thread_mem,
                                              const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                auto x = this->branches[0]->compile(stream, registers, thread_mem, usage);
                auto i = this->branches[1]->compile(stream, registers, thread_mem, usage);
                auto rand = this->branches[2]->compile(stream, registers, thread_mem, usage);
                auto nu_D_dt = this->branches[3]->compile(stream, registers, thread_mem, usage);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = apply_xi("
                       << registers[x.get()] << ", "
                       << registers[i.get()] << ", reinterpret_cast<";
                if constexpr (std::same_as<T, float>) {
                    stream << "uint32_t";
                } else {
                    stream << "uint64_t";
                }
                stream << "> ("
                       << registers[rand.get()] << "), "
                       << registers[nu_D_dt.get()] << ")";
                this->endline(stream, usage);
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Query if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(graph::shared_leaf<T> x) {
            if (this == x.get()) {
                return true;
            }

            auto x_cast = apply_xi_cast(x);
            bool temp;
            if (x_cast.get()) {
                temp = this->branches[0]->is_match(x_cast->get_arg(0));
                for (size_t i = 1; i < 4 && temp; i++) {
                    temp = temp && this->branches[i]->is_match(x_cast->get_arg(i));
                }
            }

            return temp;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << "\\apply_xi{\\left(";
            this->branches[0]->to_latex();
            for (uint8_t i = 1; i < 4; i++) {
                std::cout << ", " << this->branches[i]->to_latex();
            }
            std::cout << "\\right)}";
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> remove_pseudo() {
            if (this->has_pseudo()) {
                return apply_xi(this->branches[0]->remove_pseudo(),
                                this->branches[1]->remove_pseudo(),
                                this->branches[2]->remove_pseudo(),
                                this->branches[3]->remove_pseudo());
            }
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to vizgraph.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual graph::shared_leaf<T> to_vizgraph(std::stringstream &stream,
                                                  jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                const std::string name = jit::to_string('r', this);
                registers[this] = name;
                stream << "    " << name
                << " [label = \"apply_xi\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                for (auto &b : this->branches) {
                    auto temp = b->to_vizgraph(stream, registers);
                    stream << "    " << name << " -- " << registers[temp.get()] << ";" << std::endl;
                }
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build apply_xi node.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] x       Argument to apply collision to.
///  @param[in] i       Number of collision iterations.
///  @param[in] rand    A random value of 32 0s and 1s.
///  @param[in] nu_D_dt Normalized step rate.
///  @returns A reduced apply_xi node.
//------------------------------------------------------------------------------
    template<jit::float_scalar T>
    graph::shared_leaf<T> apply_xi(graph::shared_leaf<T> x,
                                   graph::shared_leaf<T> i,
                                   graph::shared_leaf<T> rand,
                                   graph::shared_leaf<T> nu_D_dt) {
        auto temp = std::make_shared<apply_xi<T>> (x, i, rand,
                                                   nu_D_dt)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
            if (graph::leaf_node<T>::caches.nodes.find(i) ==
                graph::leaf_node<T>::caches.nodes.end()) {
                graph::leaf_node<T>::caches.nodes[i] = temp;
                return temp;
            } else if (temp->is_match(graph::leaf_node<T>::caches.nodes[i])) {
                return graph::leaf_node<T>::caches.nodes[i];
            }
        }
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

///  Convenience type alias for shared sqrt nodes.
    template<jit::float_scalar T>
    using shared_apply_xi = std::shared_ptr<apply_xi_node<T>>;

//------------------------------------------------------------------------------
///  @brief Cast to a apply_u node.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attempted dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T>
    shared_apply_xi<T> apply_u_cast(graph::shared_leaf<T> x) {
        return std::dynamic_pointer_cast<apply_xi_node<T>> (x);
    }

//------------------------------------------------------------------------------
///  @brief Mesh class.
///
///  @tparam T Base type of the calculation.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    class mesh {
    private:
//------------------------------------------------------------------------------
///  @brief Build y index.
///
///  @tparam I Mesh index.
///
///  @param[in] x          The x position.
///  @param[in] scale      Scale factor.
///  @param[in] iterations Iterations.
///  @returns The indexed mesh Y position.
//------------------------------------------------------------------------------
        template<size_t I=0>
        graph::shared_leaf<T> build_y_index(graph::shared_leaf<T> x,
                                            const T scale,
                                            const size_t iterations=0) const {
            auto low = iterations ? build_y_index<I> (x - dx, scale, iterations - 1) :
                                    graph::index_1D(y[I], x, dx, xmin + dx);
            auto center = graph::index_1D(y[I], x, dx, xmin);
            auto high = iterations ? build_y_index<I> (x + dx, scale, iterations - 1) :
                                     graph::index_1D(y[I], x, dx, xmin - dx);

            const T center_w = static_cast<T> (0.5);
            const T side_w = static_cast<T> (0.25);

            auto b = center_w*center + side_w*low + side_w*high;
            return (1 - scale)*center + scale*b;
        }

    public:
///  Min x
        const T xmin;
///  Max x
        const T xmax;
///  Dx
        const T dx;
///  Mesh y values.
        std::array<graph::shared_leaf<T>, 4> y;
///  Mesh point.
        enum offset {
///  Lower index.
            low,
///  Center index.
            center,
///  Higher index.
            high
        };

//------------------------------------------------------------------------------
///  @brief Construct a mesh object.
///
///  @param[in] x_min Minimum X postion of mesh.
///  @param[in] x_max Maximum X position of mesh.
///  @param[in] num   Number of mesh points.
///  @param[in] norms A @ref pic::characteristics object.
//------------------------------------------------------------------------------
        mesh(const T x_min,
             const T x_max,
             const size_t num,
             const characteristics<T> &norms) :
        y({
            graph::variable<T> (num, "y^{0}_{m}"),
            graph::variable<T> (num, "y^{1}_{m}"),
            graph::variable<T> (num, "y^{2}_{m}"),
            graph::variable<T> (num, "y^{3}_{m}")
        }), xmin(x_min/norms.l), xmax(x_max/norms.l),
        dx((xmax - xmin)/(num - 1)) {}

//------------------------------------------------------------------------------
///  @brief Build x index.
///
///  @param[in] x The x position.
///  @returns The indexed mesh X position.
//------------------------------------------------------------------------------
        graph::shared_leaf<T> build_x_index(graph::shared_leaf<T> x) const {
            return dx*graph::argument(x, dx, xmin, size()) + xmin;
        }

//------------------------------------------------------------------------------
///  @brief Build i index.
///
///  @param[in] x The x position.
///  @returns The indexed mesh X position.
//------------------------------------------------------------------------------
        graph::shared_leaf<T> build_i_index(graph::shared_leaf<T> x) const {
            return graph::argument(x, dx, xmin, size());
        }

//------------------------------------------------------------------------------
///  @brief Build y index.
///
///  @tparam I Mesh index.
///  @tparam O Mesh offset.
///
///  @param[in] x The x position.
///  @param[in] params A @ref pic::parameters object.
///  @returns The indexed mesh Y position.
//------------------------------------------------------------------------------
        template<size_t I=0, offset O=center>
        graph::shared_leaf<T> build_y_index(graph::shared_leaf<T> x,
                                            const parameters<T> &params) const {
            if constexpr (O == low) {
                return build_y_index<I> (x - dx, params.smoothing,
                                         params.filter_iterations - 1);
            } else if constexpr (O == center) {
                return build_y_index<I> (x, params.smoothing,
                                         params.filter_iterations - 1);
            } else {
                return build_y_index<I> (x + dx, params.smoothing,
                                         params.filter_iterations - 1);
            }
        }

//------------------------------------------------------------------------------
///  @brief Build dy/dx index.
///
///  @tparam I Mesh index.
///  @tparam O Mesh offset.
///
///  @param[in] x      The x position.
///  @param[in] params A @ref pic::parameters object.
///  @returns The indexed mesh Y position.
//------------------------------------------------------------------------------
        template<size_t I=0, offset O=center>
        graph::shared_leaf<T> build_dydx_index(graph::shared_leaf<T> x,
                                               const parameters<T> &params) const {
            const T two = 2;
            if constexpr (O == low) {
                auto low = build_y_index<I> (x - two*dx, params.smoothing,
                                             params.filter_iterations);
                auto high = build_y_index<I> (x, params.smoothing,
                                              params.filter_iterations);
                return (high - low)/two;
            } else if constexpr (O == center) {
                auto low = build_y_index<I> (x - dx, params.smoothing,
                                             params.filter_iterations);
                auto high = build_y_index<I> (x + dx, params.smoothing,
                                              params.filter_iterations);
                return (high - low)/two;
            } else {
                auto low = build_y_index<I> (x, params.smoothing,
                                             params.filter_iterations);
                auto high = build_y_index<I> (x + two*dx, params.smoothing,
                                              params.filter_iterations);
                return (high - low)/two;
            }
        }

//------------------------------------------------------------------------------
///  @brief Build mesh accumulation.
///
///  @param[in] ion A @ref pic::ion object.
///  @returns Expressions for mesh accumulation.
//------------------------------------------------------------------------------
        std::array<graph::shared_leaf<T>, 3> build_mesh_solve(const ion<T> &ion) const {
            auto weights = build_weights(ion.x);
            auto sum_low = graph::atomic_accumulate_1D(y[0], ion.x - dx,
                                                       dx, xmin, weights[0]);
            auto sum = graph::atomic_accumulate_1D(y[0], ion.x,
                                                   dx, xmin, weights[1]);
            auto sum_high = graph::atomic_accumulate_1D(y[0], ion.x + dx,
                                                        dx, xmin, weights[2]);
            return {sum_low, sum, sum_high};
        }

//------------------------------------------------------------------------------
///  @brief Get the number of computational ions.
///
///  @return The number of particles.
//------------------------------------------------------------------------------
        size_t size() const {
            return graph::variable_cast(y[0])->size();
        }

//------------------------------------------------------------------------------
///  @brief Get the number of computational ions.
///
///  @tparam I Mesh index.
///
///  @return The number of particles.
//------------------------------------------------------------------------------
        template<size_t I=0>
        T *data() const {
            return graph::variable_cast(y[I])->data();
        }

//------------------------------------------------------------------------------
///  @brief Define variables.
///
///  @param[in]     file A @ref output::result_file object to define variables.
///  @param[in,out] data A @ref output::data_set object to create variable.
///  @param[in]     work A @ref workflow::manager object where data was
///                      computed.
//------------------------------------------------------------------------------
        void define_variables(const output::result_file &file,
                              output::data_set<T> &data,
                              workflow::manager<T> &work) {
            data.create_variable(file, "y_0", y[0],  work.get_context());
            data.create_variable(file, "y_1", y[1],  work.get_context());
            data.create_variable(file, "y_2", y[2],  work.get_context());
            data.create_variable(file, "y_3", y[3],  work.get_context());
        }

//------------------------------------------------------------------------------
///  @brief Build interpolation weights.
///
///  @param[in] x The x position.
///  @returns The interpolated mesh weights.
//------------------------------------------------------------------------------
        std::array<graph::shared_leaf<T>, 3> build_weights(graph::shared_leaf<T> x) const {
            auto x_off = build_x_index(x) - x;
            auto xnorm1 = static_cast<T> (1.5) + (x_off - dx)/dx;
            auto xnorm2 = x_off/dx;
            auto xnorm3 = static_cast<T> (1.5) - (x_off + dx)/dx;

            auto w0 = static_cast<T> (0.5)*xnorm1*xnorm1;
            auto w1 = static_cast<T> (0.75) - xnorm2*xnorm2;
            auto w2 = static_cast<T> (0.5)*xnorm3*xnorm3;
                
            return {w0, w1, w2};
        }

//------------------------------------------------------------------------------
///  @brief Build interpolation expression.
///
///  @param[in] x The x position.
///  @returns The interpolated mesh quantity.
//------------------------------------------------------------------------------
        graph::shared_leaf<T> build_interpolation(graph::shared_leaf<T> x) const {
            auto weights = build_weights(x);

            auto ymesh0 = graph::index_1D(y[0], x - dx, dx, xmin);
            auto ymesh1 = graph::index_1D(y[0], x,      dx, xmin);
            auto ymesh2 = graph::index_1D(y[0], x + dx, dx, xmin);

            return weights[0]*ymesh0 + weights[1]*ymesh1 + weights[2]*ymesh2;
        }
    };

//------------------------------------------------------------------------------
///  @brief Build initialization.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] ion    A @ref pic::ion object.
///  @param[in] mesh   A @ref pic::mesh object.
///  @param[in] norms  A @ref pic::characteristics object.
///  @param[in] params A @ref pic::parameters object.
///  @param[in] state Random state node.
///  @returns Initialized normalized values for x, v||, and v⟂
//------------------------------------------------------------------------------
    template<std::floating_point T>
    std::array<graph::shared_leaf<T>,3> build_initialization(const pic::ion<T> &ion,
                                                             const mesh<T> &mesh,
                                                             const characteristics<T> &norms,
                                                             const parameters<T> &params,
                                                             const graph::shared_random_state<T> state) {
//  The mesh is already normalized so position_dist will be a normalized quantity.
        auto position_dist = graph::uniform_random<T> (mesh.xmin, mesh.xmax,
                                                       state);
        auto phi_dist = graph::uniform_random<T> (static_cast<T> (0),
                                                  static_cast<T> (2)*std::numbers::pi_v<T>,
                                                  state);
        auto r_dist = graph::uniform_random<T> (std::numeric_limits<T>::min(),
                                                static_cast<T> (1),
                                                state);

        const T vtpara = std::sqrt(2*params.t_para*q<T>/ion.mass);
        auto vpara = vtpara*graph::sqrt(-graph::log(r_dist))
                   * graph::sin(phi_dist);

        phi_dist = graph::uniform_random<T> (static_cast<T> (0),
                                             static_cast<T> (2)*std::numbers::pi_v<T>,
                                             state);
        r_dist = graph::uniform_random<T> (std::numeric_limits<T>::min(),
                                           static_cast<T> (1),
                                           state);

        const T vtperp = std::sqrt(2*params.t_perp*q<T>/ion.mass);
        auto vperp1 = vtperp*graph::sqrt(-graph::log(r_dist))
                    * graph::cos(phi_dist);
        auto vperp2 = vtperp*graph::sqrt(-graph::log(r_dist))
                    * graph::sin(phi_dist);
        auto vperp = graph::sqrt(vperp1*vperp1 + vperp2*vperp2);

        return {position_dist, vpara/norms.v, vperp/norms.v};
    }

//------------------------------------------------------------------------------
///  @brief Build reinjected expressions.
///
///  If the particles leave the mesh, reinitalize them using the same
///  initialization.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] ion    A @ref pic::ion object.
///  @param[in] mesh   A @ref pic::mesh object.
///  @param[in] norms  A @ref pic::characteristics object.
///  @param[in] params A @ref pic::parameters object.
///  @param[in] state Random state node.
///  @returns Reinjected values for x, v||, and v⟂
//------------------------------------------------------------------------------
    template<std::floating_point T>
    std::array<graph::shared_leaf<T>, 3> build_reinjection(const ion<T> &ion,
                                                           const mesh<T> &mesh,
                                                           const characteristics<T> &norms,
                                                           const parameters<T> &params,
                                                           const graph::shared_random_state<T> state) {
        auto resampled = build_initialization(ion, mesh, norms, params, state);
        auto is_outside = ion.x <= mesh.xmin || ion.x >= mesh.xmax;

        auto reinject_x     = graph::if_(is_outside          ||
                                         graph::isnan(ion.x) ||
                                         graph::isinf(ion.x), resampled[0], ion.x);
        auto reinject_vpara = graph::if_(is_outside          ||
                                         graph::isnan(ion.x) ||
                                         graph::isinf(ion.x), resampled[1], ion.v_para);
        auto reinject_vperp = graph::if_(is_outside          ||
                                         graph::isnan(ion.x) ||
                                         graph::isinf(ion.x), resampled[2], ion.v_perp);
        return {reinject_x, reinject_vpara, reinject_vperp};
    }

//------------------------------------------------------------------------------
///  @brief Build a magnetic field.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] x      The x position.
///  @param[in] norms  A @ref pic::characteristics object.
///  @param[in] params A @ref pic::parameters object.
///  @returns The expression for the magnetic field.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    graph::shared_leaf<T> build_magnetic_field(graph::shared_leaf<T> x,
                                               const characteristics<T> &norms,
                                               const parameters<T> &params) {
        return (x*x*(norms.l*norms.l) + static_cast<T> (0.5))*params.b0;
    }

//------------------------------------------------------------------------------
///  @brief Build expressions for the density.
///
///  @tparam T Base type of the calculation.
///  @tparam I Mesh index.
///  @tparam O Mesh offset.
///
///  @param[in] x      The x position.
///  @param[in] ion    A @ref pic::ion object.
///  @param[in] mesh   A @ref pic::mesh object.
///  @param[in] norms  A @ref pic::characteristics object.
///  @param[in] params A @ref pic::parameters object.
///  @returns The expression for the density.
//------------------------------------------------------------------------------
    template<std::floating_point T, size_t I=0,
             mesh<T>::offset O=mesh<T>::center>
    graph::shared_leaf<T> build_density(graph::shared_leaf<T> x,
                                        const ion<T> &ion,
                                        const mesh<T> &mesh,
                                        const characteristics<T> &norms,
                                        const parameters<T> &params) {
        auto y = mesh.template build_y_index<I, O> (x, params);

//  Compression factor.
        auto cf = build_magnetic_field(x, norms, params)/params.b0;
//  Scale factor.
        const T sf = ion.super_to_real()/(params.a0*mesh.dx);

        return ion.z*y*cf*sf;
    }

//------------------------------------------------------------------------------
///  @brief Build expressions for the density gradient.
///
///  @tparam T Base type of the calculation.
///  @tparam I Mesh index.
///  @tparam O Mesh offset.
///
///  @param[in] x      The x position.
///  @param[in] ion    A @ref pic::ion object.
///  @param[in] mesh   A @ref pic::mesh object.
///  @param[in] norms  A @ref pic::characteristics object.
///  @param[in] params A @ref pic::parameters object.
///  @returns The expression for the density.
//------------------------------------------------------------------------------
    template<std::floating_point T, size_t I=0,
             mesh<T>::offset O=mesh<T>::center>
    graph::shared_leaf<T> build_density_gradient(graph::shared_leaf<T> x,
                                                 const ion<T> &ion,
                                                 const mesh<T> &mesh,
                                                 const characteristics<T> &norms,
                                                 const parameters<T> &params) {
        auto y = mesh.template build_dydx_index<I, O> (x, params);

//  Compression factor.
        auto cf = build_magnetic_field(x, norms, params)/params.b0;
//  Scale factor.
        const T sf = ion.super_to_real()/(params.a0*mesh.dx);

        return ion.z*y*cf*sf;
    }

//------------------------------------------------------------------------------
///  @brief Build Expressions for Electron temperature.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] x     The x position.
///  @param[in] norms A @ref pic::characteristics object.
///  @returns The expressions for the electron temperature.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    graph::shared_leaf<T> build_electron_temperature(graph::shared_leaf<T> x,
                                                     const characteristics<T> &norms) {
        return graph::constant<T> (static_cast<T> (2.5)*q<T>/(norms.te*kb<T>));
    }

//------------------------------------------------------------------------------
///  @brief Build expressions for the electric field.
///
///  @tparam T Base type of the calculation.
///  @tparam O Mesh offset.
///
///  @param[in] x      The x position.
///  @param[in] ion    A @ref pic::ion object.
///  @param[in] mesh   A @ref pic::mesh object.
///  @param[in] norms  A @ref pic::characteristics object.
///  @param[in] params A @ref pic::parameters object.
//------------------------------------------------------------------------------
    template<std::floating_point T, mesh<T>::offset O=mesh<T>::center>
    graph::shared_leaf<T> build_electric_efield(graph::shared_leaf<T> x,
                                                const ion<T> &ion,
                                                const mesh<T> &mesh,
                                                const characteristics<T> &norms,
                                                const parameters<T> &params) {
        auto n0 = build_density<T, 0, O> (x, ion, mesh, norms, params);
        auto n1 = build_density<T, 1, O> (x, ion, mesh, norms, params);
        auto n2 = build_density<T, 2, O> (x, ion, mesh, norms, params);
        auto n3 = build_density<T, 3, O> (x, ion, mesh, norms, params);

        auto dn0dx = build_density_gradient<T, 0, O> (x, ion, mesh, norms, params);
        auto dn1dx = build_density_gradient<T, 1, O> (x, ion, mesh, norms, params);
        auto dn2dx = build_density_gradient<T, 2, O> (x, ion, mesh, norms, params);
        auto dn3dx = build_density_gradient<T, 3, O> (x, ion, mesh, norms, params);

        auto n = (n0 + n1 + n2 + n3)/static_cast<T> (4);
        auto dndx = (dn0dx + dn1dx + dn2dx + dn3dx)/static_cast<T> (4);

        auto te = build_electron_temperature(x, norms);
        const T scale = norms.q/q<T>;
        auto pressure = te*n*scale;

        return graph::none<T> ()/n*(dndx*te*scale + pressure->df(x));
    }

//------------------------------------------------------------------------------
///  @brief Build interpolation expression.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] x      The x position.
///  @param[in] ion    A @ref pic::ion object.
///  @param[in] mesh   A @ref pic::mesh object.
///  @param[in] norms  A @ref pic::characteristics object.
///  @param[in] params A @ref pic::parameters object.
///  @returns The interpolated mesh quantity.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    graph::shared_leaf<T> build_interpolate_efield(graph::shared_leaf<T> x,
                                                   const ion<T> &ion,
                                                   const mesh<T> &mesh,
                                                   const characteristics<T> &norms,
                                                   const parameters<T> &params) {
        auto weights = mesh.build_weights(x);

        auto ymesh0 = build_electric_efield<T, ::pic::mesh<T>::low> (x, ion, mesh, norms, params);
        auto ymesh1 = build_electric_efield<T, ::pic::mesh<T>::center> (x, ion, mesh, norms, params);
        auto ymesh2 = build_electric_efield<T, ::pic::mesh<T>::high> (x, ion, mesh, norms, params);

        return weights[0]*ymesh0 + weights[1]*ymesh1 + weights[2]*ymesh2;
    }

//------------------------------------------------------------------------------
///  @brief Build F expressions.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] z      Runga Kutta substep.
///  @param[in] ion    A @ref pic::ion object.
///  @param[in] mesh   A @ref pic::mesh object.
///  @param[in] norms  A @ref pic::characteristics object.
///  @param[in] params A @ref pic::parameters object.
///  @returns the Forces on the particles.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    std::array<graph::shared_leaf<T>, 3> build_F_expressions(const std::array<graph::shared_leaf<T>, 3> z,
                                                             const ion<T> &ion,
                                                             const mesh<T> &mesh,
                                                             const characteristics<T> &norms,
                                                             const parameters<T> &params) {
        auto bfield = build_magnetic_field<T> (z[0], norms, params);
        auto efield = build_interpolate_efield<T> (z[0], ion, mesh, norms, params);
        auto temp = 0.5*z[2]*z[1]*bfield->df(z[0])/bfield;
        return {
            z[1]*params.dt,
            temp*params.dt,
            (ion.charge/ion.mass*norms.m*efield - temp)*params.dt
        };
    }

//------------------------------------------------------------------------------
///  @brief Build Runga Kutta step update.
///
///  @tparam T Base type of the calculation.
///
///  @param[in] ion    A @ref pic::ion object.
///  @param[in] mesh   A @ref pic::mesh object.
///  @param[in] norms  A @ref pic::characteristics object.
///  @param[in] params A @ref pic::parameters object.
///  @returns Step update expressions for x, v_para, and x_perp.
//------------------------------------------------------------------------------
    template<std::floating_point T>
    std::array<graph::shared_leaf<T>, 3> build_rk4_step(const ion<T> &ion,
                                                        const mesh<T> &mesh,
                                                        const characteristics<T> &norms,
                                                        const parameters<T> &params) {
//  Step 1
        std::array<graph::shared_leaf<T>, 3> Z1{ion.x, ion.v_para, ion.v_perp};
        std::array<graph::shared_leaf<T>, 3> dZ1(build_F_expressions<T> (Z1, ion, mesh, norms, params));

//  Step 2
        std::array<graph::shared_leaf<T>, 3> Z2{
            Z1[0] + dZ1[0]/static_cast<T> (2),
            Z1[1] + dZ1[1]/static_cast<T> (2),
            Z1[2] + dZ1[2]/static_cast<T> (2)
        };
        std::array<graph::shared_leaf<T>, 3> dZ2(build_F_expressions<T> (Z2, ion, mesh, norms, params));

//  Step 3
        std::array<graph::shared_leaf<T>, 3> Z3{
            Z1[0] + dZ2[0]/static_cast<T> (2),
            Z1[1] + dZ2[1]/static_cast<T> (2),
            Z1[2] + dZ2[2]/static_cast<T> (2)
        };
        std::array<graph::shared_leaf<T>, 3> dZ3(build_F_expressions<T> (Z3, ion, mesh, norms, params));

//  Step 4
        std::array<graph::shared_leaf<T>, 3> Z4{
            Z1[0] + dZ3[0],
            Z1[1] + dZ3[1],
            Z1[2] + dZ3[2]
        };
        std::array<graph::shared_leaf<T>, 3> dZ4(build_F_expressions<T> (Z4, ion, mesh, norms, params));

//  Rk4 Solution
        return {
            Z1[0] + (dZ1[0] + static_cast<T> (2)*(dZ2[0] + dZ3[0]) + dZ4[0])/static_cast<T> (6),
            Z1[1] + (dZ1[1] + static_cast<T> (2)*(dZ2[1] + dZ3[1]) + dZ4[1])/static_cast<T> (6),
            Z1[2] + (dZ1[2] + static_cast<T> (2)*(dZ2[2] + dZ3[2]) + dZ4[2])/static_cast<T> (6)
        };
    }
}

#endif /* particle_in_cell_h */
