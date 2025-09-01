//------------------------------------------------------------------------------
///  @file node.hpp
///  @brief Base nodes of graph computation framework.
///
///  Defines a tree of operations that allows automatic differentiation.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
///  @page new_operations_tutorial Adding New Operations Tutorial
///  @brief A discription of the models for power absorption.
///  @tableofcontents
///
///  @section new_operations_tutorial_intro Introduction
///  In most cases, physics problems can be generated from combinations of graph
///  nodes. For instance, the @ref graph::tan nodes are built from
///  @f$\frac{\sin\left(x\right)}{\cos\left(x\right)}@f$.
///
///  However, some problems will call for adding new operations. This page
///  provides a basic example of how to impliment a new operator
///  @f$foo\left(x\right)@f$ in the graph framework.
///
///  <hr>
///  @section new_operations_tutorial_node_subclass Node Subclasses
///  All graph nodes are subclasses of @ref graph::leaf_node or subclasses or
///  other nodes. In the case of our @f$foo\left(x\right)@f$ example we can
///  sublass the @ref graph::straight_node instead. If there are two or three
///  operands you can subclass
///  * @ref graph::branch_node
///  * @ref graph::triple_node
///
///  @note Any existing node can be subclassed but do so with caution.
///  Subclasses inherent reduction rules which maybe incorrect.
///
///  In this case, the @ref graph::straight_node (Along with
///  @ref graph::branch_node, @ref graph::triple_node) have no reduction
///  assumputions. For this case since our operation @f$foo\left(x\right)@f$
///  takes one argument, we will subclass the @ref graph::straight_node.
///
///  The basics of subclassing a node, start with a subclass and a constructor.
///  @code
///  template<jit::float_scalar T, bool SAFE_MATH=false>
///  class foo_node : public straight_node {
///  private:
///      static std::string to_string(leaf_node<T, SAFE_MATH> *x) {
///          return "foo(" +
///                 jit::format_to_string(reinterpret_cast<size_t> (l)) +
///                 ")";
///      }
///
///  public:
///      foo_node(shared_leaf<T, SAFE_MATH> x) :
///      straight_node(x, foo_node::to_string(x.get())) {}
///  };
///  @endcode
///  The static <tt>to_string</tt> method provices an idenifier that can be used
///  to generate a hash for the node. This hash will be used later in a factory
///  function to exsure nodes only exist once.
///
///  A factor function constructs a node then immedately reduces it. The reduced
///  node is then checked if it already exists in the
///  @ref leaf_node::caches::node. If the node is a new node, we add it to the
///  cache and return it. Otherwise we discard the node and return the cached
///  node. In it's place.
///  @code
///  template<jit::float_scalar T, bool SAFE_MATH=false>
///  shared_leaf<T, SAFE_MATH> foo(shared_leaf<T, SAFE_MATH> x) {
///      auto temp = std::make_shared<foo_node<T, SAFE_MATH>> (x)->reduce();
///  //  Test for hash collisions.
///      for (size_t i = temp->get_hash();
///           i < std::numeric_limits<size_t>::max(); i++) {
///          if (leaf_node<T, SAFE_MATH>::caches.nodes.find(i) ==
///              leaf_node<T, SAFE_MATH>::caches.nodes.end()) {
///              leaf_node<T, SAFE_MATH>::caches.nodes[i] = temp;
///              return temp;
///          } else if (temp->is_match(leaf_node<T, SAFE_MATH>::caches.nodes[i])) {
///              return leaf_node<T, SAFE_MATH>::caches.nodes[i];
///          }
///      }
///  }
///  @endcode
///
///  To aid in introspection we also need a function to case a generic
///  @ref graph::shared_leaf back to the specifi node tpe. For convience, we
///  also define a type alias for shared type.
///  @code
///  template<jit::float_scalar T, bool SAFE_MATH=false>
///  using shared_foo = std::shared_ptr<add_node<T, SAFE_MATH>>;
///
///  template<jit::float_scalar T, bool SAFE_MATH=false>
///  shared_foo<T, SAFE_MATH> foo_cast(shared_leaf<T, SAFE_MATH> x) {
///      return std::dynamic_pointer_cast<add_node<T, SAFE_MATH>> (x);
///  }
///  @endcode
///
///  <hr>
///  @section new_operations_tutorial_method Methods overloads
///  To subclass a @ref graph::leaf_node there are several methods that need to
///  be provided.
///
///  <hr>
///  @subsection new_operations_tutorial_evalute Evaluate
///  To start, lets provide a way to
///  @ref graph::leaf_node::evaluate "evalute the node". The first step to
///  evaluate a node is to the nodes argument.
///  @code
///  virtual shared_leaf<T, SAFE_MATH> evaluate() {
///      backend::buffer<T> result = this->arg->evaluate();
///  }
///  @endcode
///  @ref backend::buffer are quick ways we can evalute the node on the GPU
///  before needing to generate GPU kernels and is used by the
///  @ref graph::leaf_node::reduce method to precompute constant values. We can
///  extend the @ref backend::buffer class with a new method to evaluate foo or
///  you can use the existing operators. In this case lets assume
///  @f$foo\left(x\right)=x^{2}@f$.
///  @code
///  virtual shared_leaf<T, SAFE_MATH> evaluate() {
///      backend::buffer<T> result = this->arg->evaluate();
///      return result*result;
///  }
///  @endcode
///
///  <hr>
///  @subsection new_operations_tutorial_is_match Is Match
///  This methiod checks if the node matches another node. The first thing to
///  check is if the pointers match. Then we can check if the structure of the
///  graphs match.
///  @code
///  virtual bool is_match(shared_leaf<T, SAFE_MATH> x) {
///      if (this == x.get()) {
///          return true;
///      }
///
///      auto x_cast = foo_cast(x);
///      if (x_cast.get()) {
///          return this->arg->is_match(x_cast->get_arg());
///      }
///
///      return false;
///  }
///  @endcode
///
///  <hr>
///  @subsection new_operations_tutorial_reduce Reduce
///  Lets add a simple reduction method. When the argument @f$x @f$ is a
///  constant we can reduce this node down to a single constant by pre
///  evaluating it.
///  @code
///  virtual shared_leaf<T, SAFE_MATH> reduce() {
///      if (constant_cast(this->arg).get()) {
///          return constant<T, SAFE_MATH> (this->evaluate());
///      }
///
///      return this->shared_from_this();
///  }
///  @endcode
///  In this example we first check if the argument can be cast to a constant.
///  If it was castable, we evalute this node and create a new constant to
///  return in its place. Otherwise we return the current node unchanged.
///  @note Other reductions are possible but not shown here.
///
///  <hr>
///  @subsection new_operations_tutorial_df df
///  Auto differentiation is provided by returning the derivative expression.
///  @f$\frac{\partial}{\partial y}foo\left(x\right)=2x\frac{\partial x}{\partial y}@f$.
///  However, in this frame it is also possible to take a derivative with
///  respect to itself @f$\frac{\partial foo\left(x\right)}{\partial foo\left(x\right)}=1 @f$.
///  @code
///  virtual shared_leaf<T, SAFE_MATH> df(shared_leaf<T, SAFE_MATH> x) {
///      if (this->is_match(x)) {
///          return one<T, SAFE_MATH> ();
///      }
///
///      const size_t hash = reinterpret_cast<size_t> (x.get());
///      if (this->df_cache.find(hash) == this->df_cache.end()) {
///          this->df_cache[hash] = 2.0*this->arg*this->arg->df(x);
///      }
///      return this->df_cache[hash];
///  }
///  @endcode
///  Here we made use of the @ref graph::leaf_node::df_cache to avoid needing
///  to rebuild expressions everytime the same derivative is taken.
///
///  <hr>
///  @subsection new_operations_tutorial_compile_preamble Compile preamble
///  The @ref graph::leaf_node::compile_preamble method provides ways that
///  header files or define functions. Lets use this method to define a function
///  that can be called from the kerne.
///  @code
///  virtual void compile_preamble(std::ostringstream &stream,
///                                jit::register_map &registers,
///                                jit::visiter_map &visited,
///                                jit::register_usage &usage,
///                                jit::texture1d_list &textures1d,
///                                jit::texture2d_list &textures2d,
///                                int &avail_const_mem) {
///      if (visited.find(this) == visited.end()) {
///          this->arg->compile_preamble(stream, registers,
///                                      visited, usage,
///                                      textures1d, textures2d,
///                                      avail_const_mem);
///
///          jit::add_type<T> (stream);
///          stream << " foo(const "
///          jit::add_type<T> (stream);
///          stream << "x) {"
///                 << "    return 2*x;"
///                 << "}";
///
///          visited.insert(this);
///  #ifdef SHOW_USE_COUNT
///          usage[this] = 1;
///      } else {
///          ++usage[this];
///  #endif
///      }
///  }
///  @endcode
///  The compile methods generate kernel source code. In this case we created a
///  function in the preamble to evaluate foo. Since we only want this create
///  this preamble once, we first check if this node has already been visited.
///  The @ref build_system_dev_options "build system option"
///  <tt>SHOW_USE_COUNT</tt> tracks the number of times a node is used in the
///  kernel. When this option is set we need to increment it's usage count.
///  @note Most nodes don't require a preamble so this method can be left out.
///
///  <hr>
///  @subsection new_operations_tutorial_compile Compile
///  The compile method writes a line of source code to the kernel. Here we can
///  use the function defined in the preamble.
///  @code
///  virtual shared_leaf<T, SAFE_MATH>
///  compile(std::ostringstream &stream,
///          jit::register_map &registers,
///          jit::register_map &indices,
///          const jit::register_usage &usage) {
///      if (registers.find(this) == registers.end()) {
///          shared_leaf<T, SAFE_MATH> a = this->arg->compile(stream,
///                                                           registers,
///                                                           indices,
///                                                           usage);
///
///          registers[this] = jit::to_string('r', this);
///          stream << "        const ";
///          jit::add_type<T> (stream);
///          stream << " " << registers[this] << " = foo("
///                 << registers[a.get()] << ")";
///          this->endline(stream, usage);
///      }
///
///      return this->shared_from_this();
///  }
///  @endcode
///  Kernels are created by assuming infinite registers. In this case, a
///  register is a temporary variable. To provide a unquie name, the node
///  pointer value is converted into a string. Since we only want to evaluate
///  this once, we check if the register has already been created.
///
///  <hr>
///  @subsection new_operations_tutorial_to_latex To Latex
///  This method returns the code to generate the @f$\LaTeX @f$ expression for
///  the node.
///  @code
///  virtual void to_latex () const {
///      std::cout << "foo\left(;
///      this->arg->to_latex();
///      std::cout << "\right)";
///  }
///  @endcode
///
///  <hr>
///  @subsection new_operations_tutorial_is_power_like Is Power Like
///  This provides information for other nodes about how this works for
///  reduction methods. In this care we need to set this to true. If this node
///  did not act like a power, this method can be ignored.
///  @code
///  virtual bool is_power_like() const {
///      return true;
///  }
///  @endcode
///
///  <hr>
///  @subsection new_operations_tutorial_get_power_base Get power base
///  Return the base of the power node. This provides information for other
///  nodes about how this works for reduction methods.
///  @code
///  virtual shared_leaf<T, SAFE_MATH> get_power_base() const {
///      return this->arg;
///  }
///  @endcode
///
///  <hr>
///  @subsection new_operations_tutorial_get_power_exponent Get power exponent
///  Return the exponent of the power node. This provides information for other
///  nodes about how this works for reduction methods.
///  @code
///  virtual shared_leaf<T, SAFE_MATH> get_power_exponent() const {
///      return constant<T, SAFE_MATH> (static_cast<T> (2.0));
///  }
///  @endcode
///
///  <hr>
///  @subsection new_operations_tutorial_remove_pseudo Remove Pseudo
///  Return the node with pseduo variables removed.
///  @code
///  virtual shared_leaf<T, SAFE_MATH> remove_pseudo() {
///      if (this->has_pseudo()) {
///          return sqrt(this->arg->remove_pseudo());
///      }
///      return this->shared_from_this();
///  }
///  @endcode
///
///  <hr>
///  @subsection new_operations_tutorial_to_vizgraph To Vizgraph
///  Generates a vizgraph node for visualization.
///  @code
///  virtual shared_leaf<T, SAFE_MATH> to_vizgraph(std::stringstream &stream,
///                                                jit::register_map &registers) {
///      if (registers.find(this) == registers.end()) {
///          const std::string name = jit::to_string('r', this);
///          registers[this] = name;
///          stream << "    " << name
///                 << " [label = \"foo\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;
///
///          auto a = this->arg->to_vizgraph(stream, registers);
///          stream << "    " << name << " -- " << registers[a.get()] << ";" << std::endl;
///      }
///
///      return this->shared_from_this();
///  }
///  @endcode
//------------------------------------------------------------------------------
#ifndef node_h
#define node_h

#include <iostream>
#include <string>
#include <memory>
#include <iomanip>
#include <functional>

#include "backend.hpp"

/// Name space for graph nodes.
namespace graph {
//******************************************************************************
//  Base leaf node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a node leaf.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class leaf_node : public std::enable_shared_from_this<leaf_node<T, SAFE_MATH>> {
    protected:
///  Hash for node.
        const size_t hash;
///  Graph complexity.
        const size_t complexity;
///  Cache derivative terms.
        std::map<size_t, std::shared_ptr<leaf_node<T, SAFE_MATH>>> df_cache;
///  Node contains pseudo variables.
        const bool contains_pseudo;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a basic node.
///
///  @param[in] s      Node string to hash.
///  @param[in] count  Number of nodes in the subgraph.
///  @param[in] pseudo Node contains pseudo variable.
//------------------------------------------------------------------------------
        leaf_node(const std::string s,
                  const size_t count,
                  const bool pseudo) :
        hash(std::hash<std::string>{} (s)),
        complexity(count), contains_pseudo(pseudo) {}

//------------------------------------------------------------------------------
///  @brief Destructor
//------------------------------------------------------------------------------
        virtual ~leaf_node() {}

//------------------------------------------------------------------------------
///  @brief Evaluate method.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() = 0;

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  For basic nodes, there's nothing to reduce.
///
///  @returns A reduced representation of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node> reduce() = 0;

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<T, SAFE_MATH>>
        df(std::shared_ptr<leaf_node<T, SAFE_MATH>> x) = 0;

//------------------------------------------------------------------------------
///  @brief Compile preamble.
///
///  Some nodes require additions to the preamble however most don't so define a
///  generic method that does nothing.
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
#ifdef SHOW_USE_COUNT
            if (usage.find(this) == usage.end()) {
                usage[this] = 1;
            } else {
                ++usage[this];
            }
#endif
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
        virtual std::shared_ptr<leaf_node<T, SAFE_MATH>>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                jit::register_map &indices,
                const jit::register_usage &usage) = 0;

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @param[in] x Other graph to check if it is a match.
///  @returns True if the nodes are a match.
//------------------------------------------------------------------------------
        virtual bool is_match(std::shared_ptr<leaf_node<T, SAFE_MATH>> x) {
            return this == x.get();
        }

//------------------------------------------------------------------------------
///  @brief Check if the base of the powers match.
///
///  @param[in] x Other graph to check if the bases match.
///  @returns True if the powers of the nodes match.
//------------------------------------------------------------------------------
        bool is_power_base_match(std::shared_ptr<leaf_node<T, SAFE_MATH>> x) {
            return this->get_power_base()->is_match(x->get_power_base());
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const T d) {}

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] index Buffer index to set value.
///  @param[in] d     Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const size_t index,
                         const T d) {}

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const std::vector<T> &d) {}

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Backend buffer data to set.
//------------------------------------------------------------------------------
        virtual void set(const backend::buffer<T> &d) {}

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const = 0;

//------------------------------------------------------------------------------
///  @brief Convert the node to vizgraph.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<T, SAFE_MATH>> to_vizgraph(std::stringstream &stream,
                                                                     jit::register_map &registers) = 0;

//------------------------------------------------------------------------------
///  @brief Test if node is a constant.
///
///  @returns True if the node is like a constant.
//------------------------------------------------------------------------------
        virtual bool is_constant() const {
            return false;
        }

//------------------------------------------------------------------------------
///  @brief Test the constant node has a zero.
///
///  @returns True the node has a zero constant value.
//------------------------------------------------------------------------------
        virtual bool has_constant_zero() const {
            return false;
        }

//------------------------------------------------------------------------------
///  @brief Test if the result is normal.
///
///  @returns True if the node is normal.
//------------------------------------------------------------------------------
        bool is_normal() {
            return this->evaluate().is_normal();
        }

//------------------------------------------------------------------------------
///  @brief Test if all the subnodes terminate in variables.
///
///  @returns True if all the subnodes terminate in variables.
//------------------------------------------------------------------------------
        virtual bool is_all_variables() const = 0;

//------------------------------------------------------------------------------
///  @brief Test if the node acts like a power of variable.
///
///  Most nodes are not so default to false.
///
///  @returns True the node is power like and false otherwise.
//------------------------------------------------------------------------------
        virtual bool is_power_like() const {
            return false;
        }

//------------------------------------------------------------------------------
///  @brief Get the base of a power.
///
///  Most node can be treated as x^1 so just return this node.
///
///  @returns The base of a power like node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<T, SAFE_MATH>> get_power_base() {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Get the exponent of a power.
///
///  Most node can be treated as x^1 so just return one for those nodes but we
///  need todo that manually in the derived classes.
///
///  @returns The exponent of a power like node.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<T, SAFE_MATH>> get_power_exponent() const = 0;

//------------------------------------------------------------------------------
///  @brief Get the hash for the node.
///
///  @returns The hash for the current node.
//------------------------------------------------------------------------------
        size_t get_hash() const {
            return hash;
        }

//------------------------------------------------------------------------------
///  @brief Get the number of nodes in the subgraph.
///
///  @returns The complexity count.
//------------------------------------------------------------------------------
        size_t get_complexity() const {
            return complexity;
        }

//------------------------------------------------------------------------------
///  @brief Query if the node contains pseudo variables.
///
///  @return True if the node contains pseudo variables.
//------------------------------------------------------------------------------
        virtual bool has_pseudo() const {
            return contains_pseudo;
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<T, SAFE_MATH>> remove_pseudo() {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief End a line in the kernel source.
///
///  @param[in,out] stream String buffer stream.
///  @param[in]     usage  List of register usage count.
//------------------------------------------------------------------------------
        virtual void endline(std::ostringstream &stream,
                             const jit::register_usage &usage)
#ifndef SHOW_USE_COUNT
                             const
#endif
                             final {
            stream << ";"
#ifdef SHOW_USE_COUNT
                   << " // used " << usage.at(this)
#endif
                   << std::endl;
        }

// Create one struct that holds both caches: for constructed nodes and for the backend buffers
//------------------------------------------------------------------------------
///  @brief Data structure to contain the two caches.
///
///  This a avoids an issue on gnu compilers where it would try to redefine the
///  __tls_guard twice depending on the include order.
//------------------------------------------------------------------------------
        struct caches_t {
///  Cache of node.
            std::map<size_t, std::shared_ptr<leaf_node<T, SAFE_MATH>>> nodes;
///  Cache of backend buffers.
            std::map<size_t, backend::buffer<T>> backends;
        };

///  A per thread instance of the cache structure.
        inline static thread_local caches_t caches;

///  Type def to retrieve the backend type.
        typedef T base;
    };

///  Convenience type alias for shared leaf nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_leaf = std::shared_ptr<leaf_node<T, SAFE_MATH>>;
//------------------------------------------------------------------------------
///  @brief Create a null leaf.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @returns A null leaf.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    constexpr shared_leaf<T, SAFE_MATH> null_leaf() {
        return shared_leaf<T, SAFE_MATH> ();
    }
///  Convenience type alias for a vector of output nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using output_nodes = std::vector<shared_leaf<T, SAFE_MATH>>;

///  Forward declare for zero.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    constexpr shared_leaf<T, SAFE_MATH> zero();
///  Forward declare for one.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    constexpr shared_leaf<T, SAFE_MATH> one();

//------------------------------------------------------------------------------
///  @brief Build the vizgraph input.
///
///  @param[in] node      Node to build the graph of.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    void make_vizgraph(shared_leaf<T, SAFE_MATH> node) {
        std::stringstream stream;
        jit::register_map registers;
        stream << std::setprecision(jit::max_digits10<T> ());

        stream << "graph \"\" {" << std::endl;
        stream << "    node [fontname = \"Helvetica\", ordering = out]" << std::endl << std::endl;
        node->to_vizgraph(stream, registers);
        stream << "}" << std::endl;

        std::cout << stream.str() << std::endl;
    }

//******************************************************************************
//  Constant node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing data that cannot change.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class constant_node final : public leaf_node<T, SAFE_MATH> {
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] d Scalar data to initalize.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(const T d) {
            return jit::format_to_string<T> (d);
        }

    private:
///  Storage buffer for the data.
        const backend::buffer<T> data;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a constant node from a vector.
///
///  @param[in] d Array buffer.
//------------------------------------------------------------------------------
        constant_node(const backend::buffer<T> &d) :
        leaf_node<T, SAFE_MATH> (constant_node::to_string(d.at(0)), 1, false), data(d) {
            assert(d.size() == 1 && "Constants need to be scalar functions.");
        }

//------------------------------------------------------------------------------
///  @brief Evaluate method.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            return data;
        }

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  For basic nodes, there's nothing to reduce.
///
///  @returns A reduced representation of the node.
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
            return this->is_match(x) ? one<T, SAFE_MATH> () : zero<T, SAFE_MATH> ();
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
#ifdef USE_CONSTANT_CACHE
                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                const T temp = this->evaluate().at(0);

                stream << " " << registers[this] << " = ";
                if constexpr (jit::complex_scalar<T>) {
                    jit::add_type<T> (stream);
                }
                stream << temp;
                this->endline(stream, usage);
#else
                if constexpr (jit::complex_scalar<T>) {
                    registers[this] = jit::get_type_string<T> () + "("
                                    + jit::format_to_string(this->evaluate().at(0))
                                    + ")";
                } else {
                    registers[this] = "(" + jit::get_type_string<T> () + ")"
                                    + jit::format_to_string(this->evaluate().at(0));
                }
#endif
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

            auto x_cast = constant_cast(x);
            if (x_cast.get()) {
                return this->evaluate() == x_cast->evaluate();
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Check if the constant is value.
//------------------------------------------------------------------------------
        bool is(const T d) {
            return data.size() == 1 && data.at(0) == d;
        }

//------------------------------------------------------------------------------
///  @brief Check if the value is an integer.
//------------------------------------------------------------------------------
        bool is_integer() {
            const auto temp = this->evaluate().at(0);
            return std::imag(temp) == 0 &&
                   fmod(std::real(temp), 1.0) == 0.0;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << data.at(0);
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
                       << " [label = \"" << this->evaluate().at(0)
                       << "\", shape = box, style = \"rounded,filled\", fillcolor = black, fontcolor = white];" << std::endl;
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Test if node is a constant.
///
///  @returns True if the is a constant.
//------------------------------------------------------------------------------
        virtual bool is_constant() const {
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Test the constant node has a zero.
///
///  @returns True the node has a zero constant value.
//------------------------------------------------------------------------------
        virtual bool has_constant_zero() const {
            return data.has_zero();
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a variable.
///
///  @returns True if the node acts like a variable.
//------------------------------------------------------------------------------
        virtual bool is_all_variables() const {
            return false;
        }

//------------------------------------------------------------------------------
///  @brief Test if the node acts like a power of variable.
///
///  @returns True.
//------------------------------------------------------------------------------
        virtual bool is_power_like() const {
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Get the base of a power.
///
///  @returns The base of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_base() {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Get the exponent of a power.
///
///  @returns The exponent of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_exponent() const {
            return one<T, SAFE_MATH> ();
        }
    };

//------------------------------------------------------------------------------
///  @brief Construct a constant.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] d Array buffer.
///  @returns A reduced constant node.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> constant(const backend::buffer<T> &d) {
        auto temp = std::make_shared<constant_node<T, SAFE_MATH>> (d);
//  Test for hash collisions.
        for (size_t i = temp->get_hash(); i < std::numeric_limits<size_t>::max(); i++) {
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

//------------------------------------------------------------------------------
///  @brief Construct a constant.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] d Scalar data to initalize.
///  @returns A reduced constant node.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> constant(const T d) {
        return constant<T, SAFE_MATH> (backend::buffer<T> (1, d));
    }

//  Define some common constants.
//------------------------------------------------------------------------------
///  @brief Create a zero constant.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @returns A zero constant.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH>
    constexpr shared_leaf<T, SAFE_MATH> zero() {
        return constant<T, SAFE_MATH> (static_cast<T> (0.0));
    }

//------------------------------------------------------------------------------
///  @brief Create a one constant.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @returns A one constant.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH>
    constexpr shared_leaf<T, SAFE_MATH> one() {
        return constant<T, SAFE_MATH> (static_cast<T> (1.0));
    }

//------------------------------------------------------------------------------
///  @brief Create a one constant.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @returns A one constant.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    constexpr shared_leaf<T, SAFE_MATH> none() {
        return constant<T, SAFE_MATH> (static_cast<T> (-1.0));
    }

///  Convinece type for imaginary constant.
    template<jit::complex_scalar T>
    constexpr T i = T(0.0, 1.0);

///  Convenience type alias for shared constant nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_constant = std::shared_ptr<constant_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a constant node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_constant<T, SAFE_MATH> constant_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<constant_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Base straight node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a straight node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  This ensures that the base leaf type has the common type between the two
///  template arguments.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class straight_node : public leaf_node<T, SAFE_MATH> {
    protected:
///  Argument
        shared_leaf<T, SAFE_MATH> arg;

    public:
//------------------------------------------------------------------------------
///  @brief Construct a straight node.
///
///  @param[in] a Argument.
///  @param[in] s Node string to hash.
//------------------------------------------------------------------------------
        straight_node(shared_leaf<T, SAFE_MATH> a,
                      const std::string s) :
        leaf_node<T, SAFE_MATH> (s, a->get_complexity() + 1, a->has_pseudo()),
        arg(a) {}

//------------------------------------------------------------------------------
///  @brief Evaluate method.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            return this->arg->evaluate();
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
                this->arg->compile_preamble(stream, registers,
                                            visited, usage,
                                            textures1d, textures2d,
                                            avail_const_mem);
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
            return this->arg->compile(stream, registers, indices, usage);
        }

//------------------------------------------------------------------------------
///  @brief Get the argument.
//------------------------------------------------------------------------------
        shared_leaf<T, SAFE_MATH> get_arg() {
            return this->arg;
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a variable.
///
///  @returns True if the node acts like a variable.
//------------------------------------------------------------------------------
        virtual bool is_all_variables() const {
            return this->arg->is_all_variables();
        }

//------------------------------------------------------------------------------
///  @brief Get the exponent of a power.
///
///  @returns Returns a power of one.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_exponent() const {
            return one<T, SAFE_MATH> ();
        }
    };

//******************************************************************************
//  Base branch node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a branch node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  This ensures that the base leaf type has the common type between the two
///  template arguments.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class branch_node : public leaf_node<T, SAFE_MATH> {
    protected:
///  Left branch of the tree.
        shared_leaf<T, SAFE_MATH> left;
///  Right branch of the tree.
        shared_leaf<T, SAFE_MATH> right;

    public:

//------------------------------------------------------------------------------
///  @brief Assigns the left and right branches.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
///  @param[in] s Node string to hash.
//------------------------------------------------------------------------------
        branch_node(shared_leaf<T, SAFE_MATH> l,
                    shared_leaf<T, SAFE_MATH> r,
                    const std::string s) :
        leaf_node<T, SAFE_MATH> (s, l->get_complexity() + r->get_complexity() + 1,
                                 l->has_pseudo() || r->has_pseudo()),
        left(l), right(r) {}

//------------------------------------------------------------------------------
///  @brief Assigns the left and right branches.
///
///  @param[in] l     Left branch.
///  @param[in] r     Right branch.
///  @param[in] s     Node string to hash.
///  @param[in] count Number of nodes in the subgraph.
///  @param[in] pseudo Node contains pseudo variable.
//------------------------------------------------------------------------------
        branch_node(shared_leaf<T, SAFE_MATH> l,
                    shared_leaf<T, SAFE_MATH> r,
                    const std::string s,
                    const size_t count,
                    const bool pseudo) :
        leaf_node<T, SAFE_MATH> (s, count, pseudo),
        left(l), right(r) {}

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
                this->left->compile_preamble(stream, registers, 
                                             visited, usage,
                                             textures1d, textures2d,
                                             avail_const_mem);
                this->right->compile_preamble(stream, registers,
                                              visited, usage,
                                              textures1d, textures2d,
                                              avail_const_mem);
                visited.insert(this);
#ifdef SHOW_USE_COUNT
                usage[this] = 1;
            } else {
                ++usage[this];
#endif
            }
        }

//------------------------------------------------------------------------------
///  @brief Get the left branch.
//------------------------------------------------------------------------------
        shared_leaf<T, SAFE_MATH> get_left() {
            return this->left;
        }

//------------------------------------------------------------------------------
///  @brief Get the right branch.
//------------------------------------------------------------------------------
        shared_leaf<T, SAFE_MATH> get_right() {
            return this->right;
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a variable.
///
///  @returns True if the node acts like a variable.
//------------------------------------------------------------------------------
        virtual bool is_all_variables() const {
            return this->left->is_all_variables() &&
                   this->right->is_all_variables();
        }

//------------------------------------------------------------------------------
///  @brief Get the exponent of a power.
///
///  @returns Returns a power of one.
//------------------------------------------------------------------------------
        virtual std::shared_ptr<leaf_node<T, SAFE_MATH>>
        get_power_exponent() const {
            return one<T, SAFE_MATH> ();
        }
    };

//******************************************************************************
//  Base triple node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a triple branch node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  This ensures that the base leaf type has the common type between the two
///  template arguments.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class triple_node : public branch_node<T, SAFE_MATH> {
    protected:
///  Middle branch of the tree.
        shared_leaf<T, SAFE_MATH> middle;

    public:

//------------------------------------------------------------------------------
///  @brief Reduces and assigns the left and right branches.
///
///  @param[in] l Left branch.
///  @param[in] m Middle branch.
///  @param[in] r Right branch.
///  @param[in] s Node string to hash.
//------------------------------------------------------------------------------
        triple_node(shared_leaf<T, SAFE_MATH> l,
                    shared_leaf<T, SAFE_MATH> m,
                    shared_leaf<T, SAFE_MATH> r,
                    const std::string s) :
        branch_node<T, SAFE_MATH> (l, r, s,
                                   l->get_complexity() +
                                   m->get_complexity() +
                                   r->get_complexity(),
                                   l->has_pseudo() ||
                                   m->has_pseudo() ||
                                   r->has_pseudo()),
        middle(m) {}

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
                this->left->compile_preamble(stream, registers, 
                                             visited, usage,
                                             textures1d, textures2d,
                                             avail_const_mem);
                this->middle->compile_preamble(stream, registers,
                                               visited, usage,
                                               textures1d, textures2d,
                                               avail_const_mem);
                this->right->compile_preamble(stream, registers,
                                              visited, usage,
                                              textures1d, textures2d,
                                              avail_const_mem);
                visited.insert(this);
#ifdef SHOW_USE_COUNT
                usage[this] = 1;
            } else {
                ++usage[this];
#endif
            }
        }

//------------------------------------------------------------------------------
///  @brief Get the right branch.
//------------------------------------------------------------------------------
        shared_leaf<T, SAFE_MATH> get_middle() {
            return this->middle;
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a variable.
///
///  @returns True if the node acts like a variable.
//------------------------------------------------------------------------------
        virtual bool is_all_variables() const {
            return this->left->is_all_variables()   &&
                   this->middle->is_all_variables() &&
                   this->right->is_all_variables();
        }
    };

//******************************************************************************
//  Variable node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing data that can change.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class variable_node final : public leaf_node<T, SAFE_MATH> {
    private:
///  Storage buffer for the data.
        backend::buffer<T> buffer;
///  Latex Symbol for the variable when pretty printing.
        const std::string symbol;

//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] p Pointer to the node.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(variable_node<T, SAFE_MATH> *p) {
            return jit::format_to_string(reinterpret_cast<size_t> (p));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a variable node with a size.
///
///  @param[in] s      Size of the data buffer.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
        variable_node(const size_t s,
                      const std::string &symbol) :
        leaf_node<T, SAFE_MATH> (variable_node::to_string(this), 1, false),
        buffer(s), symbol(symbol) {}

//------------------------------------------------------------------------------
///  @brief Construct a variable node from a scalar.
///
///  @param[in] s      Size of he data buffer.
///  @param[in] d      Scalar data to initalize.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
        variable_node(const size_t s, const T d,
                      const std::string &symbol) :
        leaf_node<T, SAFE_MATH> (variable_node::to_string(this), 1, false),
        buffer(s, d), symbol(symbol) {
            assert(buffer.is_normal() && "NaN or Inf value.");
        }

//------------------------------------------------------------------------------
///  @brief Construct a variable node from a vector.
///
///  @param[in] d      Array buffer.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
        variable_node(const std::vector<T> &d,
                      const std::string &symbol) :
        leaf_node<T, SAFE_MATH> (variable_node::to_string(this), 1, false),
        buffer(d), symbol(symbol) {
            assert(buffer.is_normal() && "NaN or Inf value.");
        }

//------------------------------------------------------------------------------
///  @brief Construct a variable node from backend buffer.
///
///  @param[in] d      Backend buffer.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
        variable_node(const backend::buffer<T> &d,
                      const std::string &symbol) :
        leaf_node<T, SAFE_MATH> (variable_node::to_string(this), 1, false),
        buffer(d), symbol(symbol) {
            assert(buffer.is_normal() && "NaN or Inf value.");
        }

//------------------------------------------------------------------------------
///  @brief Evaluate method.
///
///  @returns The evaluated value of the node.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            return buffer;
        }

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  For basic nodes, there's nothing to reduce.
///
///  @returns A reduced representation of the node.
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
            return constant<T, SAFE_MATH> (static_cast<T> (this->is_match(x)));
        }

//------------------------------------------------------------------------------
///  @brief Compile preamble.
///
///  Some nodes require additions to the preamble however most don't so define a
///  generic method that does nothing.
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
            if (usage.find(this) == usage.end()) {
                usage[this] = 1;
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
///  @brief Set the value of variable data.
///
///  @param[in] d Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const T d) {
            buffer.set(d);
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] index Index to place the value at.
///  @param[in] d     Scalar data to set.
//------------------------------------------------------------------------------
        virtual void set(const size_t index, const T d) {
            buffer[index] = d;
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const std::vector<T> &d) {
            buffer.set(d);
        }

//------------------------------------------------------------------------------
///  @brief Set the value of variable data.
///
///  @param[in] d Vector data to set.
//------------------------------------------------------------------------------
        virtual void set(const backend::buffer<T> &d) {
            buffer = d;
        }

//------------------------------------------------------------------------------
///  @brief Get Symbol.
//------------------------------------------------------------------------------
        std::string get_symbol() const {
            return symbol;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << get_symbol();
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
                       << " [label = \"" << this->get_symbol()
                       << "\", shape = box];" << std::endl;
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Get the size of the variable buffer.
//------------------------------------------------------------------------------
        size_t size() {
            return buffer.size();
        }

//------------------------------------------------------------------------------
///  @brief Get a pointer to raw buffer.
///
///  @returns A buffer to the underlying data.
//------------------------------------------------------------------------------
        T *data() {
            return buffer.data();
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a variable.
///
///  @returns True if the node acts like a variable.
//------------------------------------------------------------------------------
        virtual bool is_all_variables() const {
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Test if the node acts like a power of variable.
///
///  @returns True.
//------------------------------------------------------------------------------
        virtual bool is_power_like() const {
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Get the base of a power.
///
///  @returns The base of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_base() {
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Get the exponent of a power.
///
///  @returns The exponent of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_exponent() const {
            return one<T, SAFE_MATH> ();
        }
    };

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] s      Size of the data buffer.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> variable(const size_t s,
                                       const std::string &symbol) {
        return std::make_shared<variable_node<T, SAFE_MATH>> (s, symbol);
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] s      Size of he data buffer.
///  @param[in] d      Scalar data to initalize.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> variable(const size_t s, const T d,
                                       const std::string &symbol) {
        return std::make_shared<variable_node<T, SAFE_MATH>> (s, d, symbol);
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] d      Array buffer.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> variable(const std::vector<T> &d,
                                       const std::string &symbol) {
        return std::make_shared<variable_node<T, SAFE_MATH>> (d, symbol);
    }

//------------------------------------------------------------------------------
///  @brief Construct a variable.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] d      Array buffer.
///  @param[in] symbol Symbol of the variable used in equations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> variable(const backend::buffer<T> &d,
                                       const std::string &symbol) {
        return std::make_shared<variable_node<T, SAFE_MATH>> (d, symbol);
    }

///  Convenience type alias for shared variable nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_variable = std::shared_ptr<variable_node<T, SAFE_MATH>>;
///  Convenience type alias for a vector of inputs.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using input_nodes = std::vector<shared_variable<T, SAFE_MATH>>;
///  Convenience type alias for maping end codes back to inputs.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using map_nodes = std::vector<std::pair<shared_leaf<T, SAFE_MATH>,
                                            shared_variable<T, SAFE_MATH>>>;

//------------------------------------------------------------------------------
///  @brief Cast to a variable node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_variable<T, SAFE_MATH> variable_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<variable_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Pseudo variable node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Class representing a subexpression that acts like a variable.
///
///  Pseudo variable nodes treat sub trees as if they were a variable. This
///  ensures that the expression returns zero when taking a derivative with
///  something that is not itself.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class pseudo_variable_node final : public straight_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] p Pointer to the node argument.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *p) {
            return jit::format_to_string(reinterpret_cast<size_t> (p));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a pseudo variable node.
///
///  @param[in] a Argument.
//------------------------------------------------------------------------------
        pseudo_variable_node(shared_leaf<T, SAFE_MATH> a) :
        straight_node<T, SAFE_MATH> (a, pseudo_variable_node::to_string(a.get())) {}

//------------------------------------------------------------------------------
///  @brief Reduction method.
///
///  For basic nodes, there's nothing to reduce.
///
///  @returns A reduced representation of the node.
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
            return constant<T, SAFE_MATH> (static_cast<T> (this->is_match(x)));
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << "\\left(";
            this->arg->to_latex();
            std::cout << "\\right)";
        }

//------------------------------------------------------------------------------
///  @brief Test if node acts like a variable.
///
///  @returns True if the node acts like a variable.
//------------------------------------------------------------------------------
        virtual bool is_all_variables() const {
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Test if the node acts like a power of variable.
///
///  @returns True.
//------------------------------------------------------------------------------
        virtual bool is_power_like() const {
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Get the base of a power.
///
///  @returns The base of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_base() {
            return this->arg->get_power_base();
        }

//------------------------------------------------------------------------------
///  @brief Get the exponent of a power.
///
///  @returns The exponent of a power like node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> get_power_exponent() const {
            return this->arg->get_power_exponent();
        }

//------------------------------------------------------------------------------
///  @brief Query if the node contains pseudo variables.
///
///  @return True if the node contains pseudo variables.
//------------------------------------------------------------------------------
        virtual bool has_pseudo() const {
            return true;
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> remove_pseudo() {
            return this->arg;
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
                       << " [label = \"pseudo_variable\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto a = this->arg->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[a.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Define pseudo variable convience function.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] x Argument.
///  @returns A reduced pseudo variable node.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> pseudo_variable(shared_leaf<T, SAFE_MATH> x) {
        return std::make_shared<pseudo_variable_node<T, SAFE_MATH>> (x);
    }

///  Convenience type alias for shared pseudo variable nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_pseudo_variable = std::shared_ptr<pseudo_variable_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a pseudo variable node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use @ref general_concepts_safe_math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_pseudo_variable<T, SAFE_MATH> pseudo_variable_cast(shared_leaf<T, SAFE_MATH> &x) {
        return std::dynamic_pointer_cast<pseudo_variable_node<T, SAFE_MATH>> (x);
    }
}

#endif /* node_h */
