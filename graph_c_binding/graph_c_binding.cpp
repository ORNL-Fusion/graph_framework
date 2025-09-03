//------------------------------------------------------------------------------
///  @file graph_c_binding.cpp
///  @brief Implimentation of the c binding library.
//------------------------------------------------------------------------------

#include "graph_c_binding.h"

#include "../graph_framework/register.hpp"
#include "../graph_framework/node.hpp"
#include "../graph_framework/workflow.hpp"
#include "../graph_framework/arithmetic.hpp"
#include "../graph_framework/math.hpp"
#include "../graph_framework/trigonometry.hpp"
#include "../graph_framework/piecewise.hpp"

//------------------------------------------------------------------------------
///  @brief C context with specific type.
//------------------------------------------------------------------------------
template<jit::float_scalar T, bool SAFE_MATH=false>
struct graph_c_context_type : public graph_c_context {
///  Variables nodes.
    std::map<graph_node, graph::shared_leaf<T, SAFE_MATH>> nodes;
///  Workflow manager.
    workflow::manager<T, SAFE_MATH> work;

//------------------------------------------------------------------------------
///  @brief Construct a typed c context.
//------------------------------------------------------------------------------
    graph_c_context_type() : work(0) {}
};

extern "C" {
//------------------------------------------------------------------------------
///  @brief Construct a C context.
///
///  @param[in] type          Base type.
///  @param[in] use_safe_math Control is safe math is used.
///  @returns A contructed C context.
//------------------------------------------------------------------------------
    graph_c_context *graph_construct_context(const enum graph_type type,
                                             const bool use_safe_math) {
        graph_c_context *temp;
        switch (type) {
            case FLOAT:
                if (use_safe_math) {
                    temp = new graph_c_context_type<float, true> ();
                } else {
                    temp = new graph_c_context_type<float> ();
                }
                break;

            case DOUBLE:
                if (use_safe_math) {
                    temp = new graph_c_context_type<double, true> ();
                } else {
                    temp = new graph_c_context_type<double> ();
                }
                break;

            case COMPLEX_FLOAT:
                if (use_safe_math) {
                    temp = new graph_c_context_type<std::complex<float>, true> ();
                } else {
                    temp = new graph_c_context_type<std::complex<float>> ();
                }
                break;

            case COMPLEX_DOUBLE:
                if (use_safe_math) {
                    temp = new graph_c_context_type<std::complex<double>, true> ();
                } else {
                    temp = new graph_c_context_type<std::complex<double>> ();
                }
                break;
        }

        temp->type = type;
        temp->safe_math = use_safe_math;
        return temp;
    }

//------------------------------------------------------------------------------
///  @brief Destroy C context.
///
///  @param[in,out] c The c context to delete.
//------------------------------------------------------------------------------
    void graph_destroy_context(graph_c_context *c) {
        delete c;
    }

//------------------------------------------------------------------------------
///  @brief Create variable node.
///
///  @param[in] c      The graph C context.
///  @param[in] size   Size of the data buffer.
///  @param[in] symbol Symbol of the variable used in equations.
///  @returns The created variable.
//------------------------------------------------------------------------------
    graph_node graph_variable(STRUCT_TAG graph_c_context *c,
                              const size_t size,
                              const char *symbol) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    auto temp = graph::variable<float, true> (size, symbol);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    auto temp = graph::variable<float> (size, symbol);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    auto temp = graph::variable<double, true> (size, symbol);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    auto temp = graph::variable<double> (size, symbol);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto temp = graph::variable<std::complex<float>, true> (size, symbol);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto temp = graph::variable<std::complex<float>> (size, symbol);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto temp = graph::variable<std::complex<double>, true> (size, symbol);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto temp = graph::variable<std::complex<double>> (size, symbol);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }
        }
    }

//------------------------------------------------------------------------------
///  @brief Create constant node.
///
///  @param[in] c     The graph C context.
///  @param[in] value The value to create the constant.
///  @returns The created constant.
//------------------------------------------------------------------------------
    graph_node graph_constant(STRUCT_TAG graph_c_context *c,
                              const double value) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    auto temp = graph::constant<float, true> (value);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    auto temp = graph::constant<float> (value);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    auto temp = graph::constant<double, true> (value);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    auto temp = graph::constant<double> (value);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto temp = graph::constant<std::complex<float>, true> (std::complex<float> (value));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto temp = graph::constant<std::complex<float>> (std::complex<float> (value));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto temp = graph::constant<std::complex<double>, true> (std::complex<double> (value));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto temp = graph::constant<std::complex<double>> (std::complex<double> (value));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }
        }
    }

//------------------------------------------------------------------------------
///  @brief Set a variable value.
///
///  @param[in] c      The graph C context.
///  @param[in] var    The variable to set.
///  @param[in] source The source pointer.
//------------------------------------------------------------------------------
    void graph_set_variable(STRUCT_TAG graph_c_context *c,
                            graph_node var,
                            const void *source) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    auto temp = graph::variable_cast(d->nodes[var]);
                    if (temp.get()) {
                        std::memcpy(temp->data(), source, sizeof(float)*temp->size());
                    } else {
                        std::cerr << "Node is not a variable.";
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    auto temp = graph::variable_cast(d->nodes[var]);
                    if (temp.get()) {
                        std::memcpy(temp->data(), source, sizeof(float)*temp->size());
                    } else {
                        std::cerr << "Node is not a variable.";
                    }
                }
                break;

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    auto temp = graph::variable_cast(d->nodes[var]);
                    if (temp.get()) {
                        std::memcpy(temp->data(), source, sizeof(double)*temp->size());
                    } else {
                        std::cerr << "Node is not a variable.";
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    auto temp = graph::variable_cast(d->nodes[var]);
                    if (temp.get()) {
                        std::memcpy(temp->data(), source, sizeof(double)*temp->size());
                    } else {
                        std::cerr << "Node is not a variable.";
                    }
                }
                break;

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto temp = graph::variable_cast(d->nodes[var]);
                    if (temp.get()) {
                        std::memcpy(temp->data(), source, sizeof(std::complex<float>)*temp->size());
                    } else {
                        std::cerr << "Node is not a variable.";
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto temp = graph::variable_cast(d->nodes[var]);
                    if (temp.get()) {
                        std::memcpy(temp->data(), source, sizeof(std::complex<float>)*temp->size());
                    } else {
                        std::cerr << "Node is not a variable.";
                    }
                }
                break;

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto temp = graph::variable_cast(d->nodes[var]);
                    if (temp.get()) {
                        std::memcpy(temp->data(), source, sizeof(std::complex<double>)*temp->size());
                    } else {
                        std::cerr << "Node is not a variable.";
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto temp = graph::variable_cast(d->nodes[var]);
                    if (temp.get()) {
                        std::memcpy(temp->data(), source, sizeof(std::complex<double>)*temp->size());
                    } else {
                        std::cerr << "Node is not a variable.";
                    }
                }
                break;
        }
    }

//------------------------------------------------------------------------------
///  @brief Create complex constant node.
///
///  @param[in] c          The graph C context.
///  @param[in] real_value The real component.
///  @param[in] img_value  The imaginary component.
///  @returns The complex constant.
//------------------------------------------------------------------------------
    graph_node graph_constant_c(STRUCT_TAG graph_c_context *c,
                                const double real_value,
                                const double img_value) {
        switch (c->type) {
            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto temp = graph::constant<std::complex<float>, true> (std::complex<float> (real_value, img_value));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto temp = graph::constant<std::complex<float>> (std::complex<float> (real_value, img_value));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto temp = graph::constant<std::complex<double>, true> (std::complex<double> (real_value, img_value));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto temp = graph::constant<std::complex<double>> (std::complex<double> (real_value, img_value));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }
            
            case FLOAT:
            case DOUBLE:
                std::cerr << "Error: Context is non-complex." << std::endl;
                exit(1);
        }
    }

//------------------------------------------------------------------------------
///  @brief Create a pseudo variable.
///
///  @param[in] c   The graph C context.
///  @param[in] var The variable to set.
///  @returns THe pseudo variable.
//------------------------------------------------------------------------------
    graph_node graph_pseudo_variable(STRUCT_TAG graph_c_context *c,
                                     graph_node var) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    auto temp = graph::pseudo_variable<float, true> (d->nodes[var]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    auto temp = graph::pseudo_variable<float> (d->nodes[var]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    auto temp = graph::pseudo_variable<double, true> (d->nodes[var]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    auto temp = graph::pseudo_variable<double> (d->nodes[var]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto temp = graph::pseudo_variable<std::complex<float>, true> (d->nodes[var]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto temp = graph::pseudo_variable<std::complex<float>> (d->nodes[var]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto temp = graph::pseudo_variable<std::complex<double>, true> (d->nodes[var]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto temp = graph::pseudo_variable<std::complex<double>> (d->nodes[var]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }
        }
    }

//------------------------------------------------------------------------------
///  @brief Remove pseudo.
///
///  @param[in] c   The graph C context.
///  @param[in] var The variable to set.
///  @returns The graph with pseudo variables removed.
//------------------------------------------------------------------------------
    graph_node graph_remove_pseudo(STRUCT_TAG graph_c_context *c,
                                   graph_node var) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    auto temp = d->nodes[var]->remove_pseudo();
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    auto temp = d->nodes[var]->remove_pseudo();
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    auto temp = d->nodes[var]->remove_pseudo();
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    auto temp = d->nodes[var]->remove_pseudo();
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto temp = d->nodes[var]->remove_pseudo();
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto temp = d->nodes[var]->remove_pseudo();
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto temp = d->nodes[var]->remove_pseudo();
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto temp = d->nodes[var]->remove_pseudo();
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }
        }
    }

//******************************************************************************
//  Arithmetic
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Create add node.
///
///  @param[in] c     The graph C context.
///  @param[in] left  The left opperand.
///  @param[in] right The right opperand.
///  @returns left + right
//------------------------------------------------------------------------------
    graph_node graph_add(STRUCT_TAG graph_c_context *c,
                         graph_node left,
                         graph_node right) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    auto temp = d->nodes[left] + d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    auto temp = d->nodes[left] + d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    auto temp = d->nodes[left] + d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    auto temp = d->nodes[left] + d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto temp = d->nodes[left] + d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto temp = d->nodes[left] + d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto temp = d->nodes[left] + d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto temp = d->nodes[left] + d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }
        }
    }

//------------------------------------------------------------------------------
///  @brief Create Substract node.
///
///  @param[in] c     The graph C context.
///  @param[in] left  The left opperand.
///  @param[in] right The right opperand.
///  @returns left - right
//------------------------------------------------------------------------------
    graph_node graph_sub(STRUCT_TAG graph_c_context *c,
                         graph_node left,
                         graph_node right) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    auto temp = d->nodes[left] - d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    auto temp = d->nodes[left] - d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    auto temp = d->nodes[left] - d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    auto temp = d->nodes[left] - d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto temp = d->nodes[left] - d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto temp = d->nodes[left] - d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto temp = d->nodes[left] - d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto temp = d->nodes[left] - d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }
        }
    }

//------------------------------------------------------------------------------
///  @brief Create Multiply node.
///
///  @param[in] c     The graph C context.
///  @param[in] left  The left opperand.
///  @param[in] right The right opperand.
///  @returns left*right
//------------------------------------------------------------------------------
    graph_node graph_mul(STRUCT_TAG graph_c_context *c,
                         graph_node left,
                         graph_node right) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    auto temp = d->nodes[left]*d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    auto temp = d->nodes[left]*d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    auto temp = d->nodes[left]*d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    auto temp = d->nodes[left]*d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto temp = d->nodes[left]*d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto temp = d->nodes[left]*d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto temp = d->nodes[left]*d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto temp = d->nodes[left]*d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }
        }
    }

//------------------------------------------------------------------------------
///  @brief Create Divide node.
///
///  @param[in] c     The graph C context.
///  @param[in] left  The left opperand.
///  @param[in] right The right opperand.
///  @returns left/right
//------------------------------------------------------------------------------
    graph_node graph_div(STRUCT_TAG graph_c_context *c,
                         graph_node left,
                         graph_node right) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    auto temp = d->nodes[left]/d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    auto temp = d->nodes[left]/d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    auto temp = d->nodes[left]/d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    auto temp = d->nodes[left]/d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto temp = d->nodes[left]/d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto temp = d->nodes[left]/d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto temp = d->nodes[left]/d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto temp = d->nodes[left]/d->nodes[right];
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }
        }
    }

//******************************************************************************
//  Math
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Create Sqrt node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg The left opperand.
///  @returns sqrt(arg)
//------------------------------------------------------------------------------
    graph_node graph_sqrt(STRUCT_TAG graph_c_context *c,
                          graph_node arg) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    auto temp = graph::sqrt(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    auto temp = graph::sqrt(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    auto temp = graph::sqrt(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    auto temp = graph::sqrt(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto temp = graph::sqrt(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto temp = graph::sqrt(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto temp = graph::sqrt(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto temp = graph::sqrt(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }
        }
    }

//------------------------------------------------------------------------------
///  @brief Create exp node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg The left opperand.
///  @returns exp(arg)
//------------------------------------------------------------------------------
    graph_node graph_exp(STRUCT_TAG graph_c_context *c,
                         graph_node arg) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    auto temp = graph::exp(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    auto temp = graph::exp(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    auto temp = graph::exp(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    auto temp = graph::exp(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto temp = graph::exp(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto temp = graph::exp(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto temp = graph::exp(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto temp = graph::exp(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }
        }
    }

//------------------------------------------------------------------------------
///  @brief Create log node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg The left opperand.
///  @returns log(arg)
//------------------------------------------------------------------------------
    graph_node graph_log(STRUCT_TAG graph_c_context *c,
                         graph_node arg) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    auto temp = graph::log(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    auto temp = graph::log(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    auto temp = graph::log(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    auto temp = graph::log(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto temp = graph::log(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto temp = graph::log(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto temp = graph::log(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto temp = graph::log(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }
        }
    }

//------------------------------------------------------------------------------
///  @brief Create Pow node.
///
///  @param[in] c     The graph C context.
///  @param[in] left  The left opperand.
///  @param[in] right The right opperand.
///  @returns pow(left, right)
//------------------------------------------------------------------------------
    graph_node graph_pow(STRUCT_TAG graph_c_context *c,
                         graph_node left,
                         graph_node right) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    auto temp = graph::pow(d->nodes[left], d->nodes[right]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    auto temp = graph::pow(d->nodes[left], d->nodes[right]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    auto temp = graph::pow(d->nodes[left], d->nodes[right]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    auto temp = graph::pow(d->nodes[left], d->nodes[right]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto temp = graph::pow(d->nodes[left], d->nodes[right]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto temp = graph::pow(d->nodes[left], d->nodes[right]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto temp = graph::pow(d->nodes[left], d->nodes[right]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto temp = graph::pow(d->nodes[left], d->nodes[right]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }
        }
    }

//------------------------------------------------------------------------------
///  @brief Create imaginary error function node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg The left opperand.
///  @returns erfi(arg)
//------------------------------------------------------------------------------
    graph_node graph_erfi(STRUCT_TAG graph_c_context *c,
                          graph_node arg) {
        switch (c->type) {
            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto temp = graph::erfi(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto temp = graph::erfi(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto temp = graph::erfi(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto temp = graph::erfi(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case FLOAT:
            case DOUBLE:
                std::cerr << "Error: Imaginary error function requires complex context." << std::endl;
                exit(1);
        }
    }

//******************************************************************************
//  Trigonometry
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Create sine node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg The left opperand.
///  @returns sin(arg)
//------------------------------------------------------------------------------
    graph_node graph_sin(STRUCT_TAG graph_c_context *c,
                         graph_node arg) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    auto temp = graph::sin(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    auto temp = graph::sin(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    auto temp = graph::sin(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    auto temp = graph::sin(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto temp = graph::sin(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto temp = graph::sin(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto temp = graph::sin(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto temp = graph::sin(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }
        }
    }

//------------------------------------------------------------------------------
///  @brief Create cosine node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg The left opperand.
///  @returns cos(arg)
//------------------------------------------------------------------------------
    graph_node graph_cos(STRUCT_TAG graph_c_context *c,
                         graph_node arg) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    auto temp = graph::cos(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    auto temp = graph::cos(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    auto temp = graph::cos(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    auto temp = graph::cos(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto temp = graph::cos(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto temp = graph::cos(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto temp = graph::cos(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto temp = graph::cos(d->nodes[arg]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }
        }
    }

//------------------------------------------------------------------------------
///  @brief Create arctangent node.
///
///  @param[in] c     The graph C context.
///  @param[in] left  The left opperand.
///  @param[in] right The right opperand.
///  @returns atan(left, right)
//------------------------------------------------------------------------------
    graph_node graph_atan(STRUCT_TAG graph_c_context *c,
                          graph_node left,
                          graph_node right) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    auto temp = graph::atan<float, true> (d->nodes[left], d->nodes[right]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    auto temp = graph::atan(d->nodes[left], d->nodes[right]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    auto temp = graph::atan(d->nodes[left], d->nodes[right]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    auto temp = graph::atan(d->nodes[left], d->nodes[right]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto temp = graph::atan(d->nodes[left], d->nodes[right]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto temp = graph::atan(d->nodes[left], d->nodes[right]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto temp = graph::atan(d->nodes[left], d->nodes[right]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto temp = graph::atan(d->nodes[left], d->nodes[right]);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }
        }
    }

//******************************************************************************
//  Random
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Construct a random state node.
///
///  @param[in] c    The graph C context.
///  @param[in] seed Intial random seed.
///  @returns A random state node.
//------------------------------------------------------------------------------
    graph_node graph_random_state(STRUCT_TAG graph_c_context *c,
                                  const uint32_t seed) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    auto temp = graph::random_state<float, true> (jit::context<float, true>::random_state_size,
                                                                  seed);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    auto temp = graph::random_state<float> (jit::context<float>::random_state_size,
                                                            seed);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    auto temp = graph::random_state<double, true> (jit::context<double, true>::random_state_size,
                                                                   seed);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    auto temp = graph::random_state<double> (jit::context<double>::random_state_size,
                                                             seed);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto temp = graph::random_state<std::complex<float>, true> (jit::context<std::complex<float>, true>::random_state_size,
                                                                                seed);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto temp = graph::random_state<std::complex<float>> (jit::context<std::complex<float>>::random_state_size, seed);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto temp = graph::random_state<std::complex<double>, true> (jit::context<std::complex<double>, true>::random_state_size,
                                                                                 seed);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto temp = graph::random_state<std::complex<double>> (jit::context<std::complex<double>>::random_state_size, seed);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }
        }
    }

//------------------------------------------------------------------------------
///  @brief Create random node.
///
///  @param[in] c   The graph C context.
///  @param[in] arg A random state node.
///  @returns random(state)
//------------------------------------------------------------------------------
    graph_node graph_random(STRUCT_TAG graph_c_context *c,
                            graph_node arg) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    auto state = graph::random_state_cast(d->nodes[arg]);
                    if (state.get()) {
                        auto temp = graph::random(state);
                        d->nodes[temp.get()] = temp;
                        return temp.get();
                    } else {
                        std::cerr << "Arg failed cast to state." << std::endl;
                        exit(1);
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    auto state = graph::random_state_cast(d->nodes[arg]);
                    if (state.get()) {
                        auto temp = graph::random(state);
                        d->nodes[temp.get()] = temp;
                        return temp.get();
                    } else {
                        std::cerr << "Arg failed cast to state." << std::endl;
                        exit(1);
                    }
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    auto state = graph::random_state_cast(d->nodes[arg]);
                    if (state.get()) {
                        auto temp = graph::random(state);
                        d->nodes[temp.get()] = temp;
                        return temp.get();
                    } else {
                        std::cerr << "Arg failed cast to state." << std::endl;
                        exit(1);
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    auto state = graph::random_state_cast(d->nodes[arg]);
                    if (state.get()) {
                        auto temp = graph::random(state);
                        d->nodes[temp.get()] = temp;
                        return temp.get();
                    } else {
                        std::cerr << "Arg failed cast to state." << std::endl;
                        exit(1);
                    }
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto state = graph::random_state_cast(d->nodes[arg]);
                    if (state.get()) {
                        auto temp = graph::random(state);
                        d->nodes[temp.get()] = temp;
                        return temp.get();
                    } else {
                        std::cerr << "Arg failed cast to state." << std::endl;
                        exit(1);
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto state = graph::random_state_cast(d->nodes[arg]);
                    if (state.get()) {
                        auto temp = graph::random(state);
                        d->nodes[temp.get()] = temp;
                        return temp.get();
                    } else {
                        std::cerr << "Arg failed cast to state." << std::endl;
                        exit(1);
                    }
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto state = graph::random_state_cast(d->nodes[arg]);
                    if (state.get()) {
                        auto temp = graph::random(state);
                        d->nodes[temp.get()] = temp;
                        return temp.get();
                    } else {
                        std::cerr << "Arg failed cast to state." << std::endl;
                        exit(1);
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto state = graph::random_state_cast(d->nodes[arg]);
                    if (state.get()) {
                        auto temp = graph::random(state);
                        d->nodes[temp.get()] = temp;
                        return temp.get();
                    } else {
                        std::cerr << "Arg failed cast to state." << std::endl;
                        exit(1);
                    }
                }
        }
    }

//******************************************************************************
//  Piecewise
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Create 1D piecewise node.
///
///  @param[in] c           The graph C context.
///  @param[in] arg         The function argument.
///  @param[in] scale       Scale factor argument.
///  @param[in] offset      Offset factor argument.
///  @param[in] source      Source buffer to fill elements.
///  @param[in] source_size Number of elements in the source buffer.
///  @returns A 1D piecewise node.
//------------------------------------------------------------------------------
    graph_node graph_piecewise_1D(STRUCT_TAG graph_c_context *c,
                                  graph_node arg,
                                  const double scale,
                                  const double offset,
                                  const void *source,
                                  const size_t source_size) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    backend::buffer<float> buffer(source_size);
                    std::memcpy(buffer.data(), source, sizeof(float)*source_size);
                    auto temp = graph::piecewise_1D(buffer, d->nodes[arg],
                                                    static_cast<float> (scale),
                                                    static_cast<float> (offset));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    backend::buffer<float> buffer(source_size);
                    std::memcpy(buffer.data(), source, sizeof(float)*source_size);
                    auto temp = graph::piecewise_1D(buffer, d->nodes[arg],
                                                    static_cast<float> (scale),
                                                    static_cast<float> (offset));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    backend::buffer<double> buffer(source_size);
                    std::memcpy(buffer.data(), source, sizeof(double)*source_size);
                    auto temp = graph::piecewise_1D(buffer, d->nodes[arg], scale, offset);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    backend::buffer<double> buffer(source_size);
                    std::memcpy(buffer.data(), source, sizeof(double)*source_size);
                    auto temp = graph::piecewise_1D(buffer, d->nodes[arg], scale, offset);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    backend::buffer<std::complex<float>> buffer(source_size);
                    std::memcpy(buffer.data(), source, sizeof(std::complex<float>)*source_size);
                    auto temp = graph::piecewise_1D(buffer, d->nodes[arg],
                                                    std::complex<float> (scale),
                                                    std::complex<float> (offset));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    backend::buffer<std::complex<float>> buffer(source_size);
                    std::memcpy(buffer.data(), source, sizeof(std::complex<float>)*source_size);
                    auto temp = graph::piecewise_1D(buffer, d->nodes[arg],
                                                    std::complex<float> (scale),
                                                    std::complex<float> (offset));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    backend::buffer<std::complex<double>> buffer(source_size);
                    std::memcpy(buffer.data(), source, sizeof(std::complex<double>)*source_size);
                    auto temp = graph::piecewise_1D(buffer, d->nodes[arg],
                                                    std::complex<double> (scale),
                                                    std::complex<double> (offset));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    backend::buffer<std::complex<double>> buffer(source_size);
                    std::memcpy(buffer.data(), source, sizeof(std::complex<double>)*source_size);
                    auto temp = graph::piecewise_1D(buffer, d->nodes[arg],
                                                    std::complex<double> (scale),
                                                    std::complex<double> (offset));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }
        }
    }

//------------------------------------------------------------------------------
///  @brief Create 2D piecewise node.
///
///  @param[in] c           The graph C context.
///  @param[in] num_cols    Number of columns.
///  @param[in] x_arg       The function x argument.
///  @param[in] x_scale     Scale factor x argument.
///  @param[in] x_offset    Offset factor x argument.
///  @param[in] y_arg       The function y argument.
///  @param[in] y_scale     Scale factor y argument.
///  @param[in] y_offset    Offset factor y argument.
///  @param[in] source      Source buffer to fill elements.
///  @param[in] source_size Number of elements in the source buffer.
///  @returns A 2D piecewise node.
//------------------------------------------------------------------------------
    graph_node graph_piecewise_2D(STRUCT_TAG graph_c_context *c,
                                  const size_t num_cols,
                                  graph_node x_arg,
                                  const double x_scale,
                                  const double x_offset,
                                  graph_node y_arg,
                                  const double y_scale,
                                  const double y_offset,
                                  const void *source,
                                  const size_t source_size) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    backend::buffer<float> buffer(source_size);
                    std::memcpy(buffer.data(), source, sizeof(float)*source_size);
                    auto temp = graph::piecewise_2D(buffer, num_cols,
                                                    d->nodes[x_arg],
                                                    static_cast<float> (x_scale),
                                                    static_cast<float> (x_offset),
                                                    d->nodes[y_arg],
                                                    static_cast<float> (y_scale),
                                                    static_cast<float> (y_offset));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    backend::buffer<float> buffer(source_size);
                    std::memcpy(buffer.data(), source, sizeof(float)*source_size);
                    auto temp = graph::piecewise_2D(buffer, num_cols,
                                                    d->nodes[x_arg],
                                                    static_cast<float> (x_scale),
                                                    static_cast<float> (x_offset),
                                                    d->nodes[y_arg],
                                                    static_cast<float> (y_scale),
                                                    static_cast<float> (y_offset));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    backend::buffer<double> buffer(source_size);
                    std::memcpy(buffer.data(), source, sizeof(double)*source_size);
                    auto temp = graph::piecewise_2D(buffer, num_cols,
                                                    d->nodes[x_arg], y_scale, y_offset,
                                                    d->nodes[y_arg], y_scale, y_offset);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    backend::buffer<double> buffer(source_size);
                    std::memcpy(buffer.data(), source, sizeof(double)*source_size);
                    auto temp = graph::piecewise_2D(buffer, num_cols,
                                                    d->nodes[x_arg], y_scale, y_offset,
                                                    d->nodes[y_arg], y_scale, y_offset);
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    backend::buffer<std::complex<float>> buffer(source_size);
                    std::memcpy(buffer.data(), source, sizeof(std::complex<float>)*source_size);
                    auto temp = graph::piecewise_2D(buffer, num_cols,
                                                    d->nodes[x_arg],
                                                    std::complex<float> (x_scale),
                                                    std::complex<float> (x_offset),
                                                    d->nodes[y_arg],
                                                    std::complex<float> (y_scale),
                                                    std::complex<float> (y_offset));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    backend::buffer<std::complex<float>> buffer(source_size);
                    std::memcpy(buffer.data(), source, sizeof(std::complex<float>)*source_size);
                    auto temp = graph::piecewise_2D(buffer, num_cols,
                                                    d->nodes[x_arg],
                                                    std::complex<float> (x_scale),
                                                    std::complex<float> (x_offset),
                                                    d->nodes[y_arg],
                                                    std::complex<float> (y_scale),
                                                    std::complex<float> (y_offset));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    backend::buffer<std::complex<double>> buffer(source_size);
                    std::memcpy(buffer.data(), source, sizeof(std::complex<double>)*source_size);
                    auto temp = graph::piecewise_2D(buffer, num_cols,
                                                    d->nodes[x_arg],
                                                    std::complex<double> (x_scale),
                                                    std::complex<double> (x_offset),
                                                    d->nodes[y_arg],
                                                    std::complex<double> (y_scale),
                                                    std::complex<double> (y_offset));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    backend::buffer<std::complex<double>> buffer(source_size);
                    std::memcpy(buffer.data(), source, sizeof(std::complex<double>)*source_size);
                    auto temp = graph::piecewise_2D(buffer, num_cols,
                                                    d->nodes[x_arg],
                                                    std::complex<double> (x_scale),
                                                    std::complex<double> (x_offset),
                                                    d->nodes[y_arg],
                                                    std::complex<double> (y_scale),
                                                    std::complex<double> (y_offset));
                    d->nodes[temp.get()] = temp;
                    return temp.get();
                }
        }
    }

//******************************************************************************
//  JIT
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Create 2D piecewise node with complex arguments.
///
///  @param[in] c The graph C context.
///  @returns The number of concurrent devices.
//------------------------------------------------------------------------------
    size_t graph_get_max_concurrency(graph_c_context *c) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    return jit::context<float, true>::max_concurrency();
                } else {
                    return jit::context<float>::max_concurrency();
                }

            case DOUBLE:
                if (c->safe_math) {
                    return jit::context<double, true>::max_concurrency();
                } else {
                    return jit::context<double>::max_concurrency();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    return jit::context<std::complex<float>, true>::max_concurrency();
                } else {
                    return jit::context<std::complex<float>>::max_concurrency();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    return jit::context<std::complex<double>, true>::max_concurrency();
                } else {
                    return jit::context<std::complex<double>>::max_concurrency();
                }
        }
    }

//******************************************************************************
//  Workflows
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief Choose the device number.
///
///  @param[in] c   The graph C context.
///  @param[in] num The device number.
//------------------------------------------------------------------------------
    void graph_set_device_number(STRUCT_TAG graph_c_context *c,
                                 const size_t num) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    d->work = workflow::manager<float, true> (num);
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    d->work = workflow::manager<float> (num);
                }
                break;

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    d->work = workflow::manager<double, true> (num);
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    d->work = workflow::manager<double> (num);
                }
                break;

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    d->work = workflow::manager<std::complex<float>, true> (num);
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    d->work = workflow::manager<std::complex<float>> (num);
                }
                break;

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    d->work = workflow::manager<std::complex<double>, true> (num);
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    d->work = workflow::manager<std::complex<double>> (num);
                }
                break;
        }
    }

//------------------------------------------------------------------------------
///  @brief Add pre workflow item.
///
///  @param[in] c            The graph C context.
///  @param[in] inputs       Array of input nodes.
///  @param[in] num_inputs   Number of inputs.
///  @param[in] outputs      Array of output nodes.
///  @param[in] num_outputs  Number of outputs.
///  @param[in] map_inputs   Array of map input nodes.
///  @param[in] map_outputs  Array of map output nodes.
///  @param[in] num_maps     Number of maps.
///  @param[in] random_state Optional random state, can be NULL if not used.
///  @param[in] name         Name for the kernel.
///  @param[in] size         Number of elements to operate on.
//------------------------------------------------------------------------------
    void graph_add_pre_item(STRUCT_TAG graph_c_context *c,
                            graph_node *inputs, size_t num_inputs,
                            graph_node *outputs, size_t num_outputs,
                            graph_node *map_inputs,
                            graph_node *map_outputs, size_t num_maps,
                            graph_node random_state,
                            const char *name,
                            const size_t size) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    graph::input_nodes<float, true> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Preitem input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<float, true> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<float, true> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Preitem map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_preitem(in, out, map, rand, name, size);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_preitem(in, out, map, NULL, name, size);
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    graph::input_nodes<float> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Preitem input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<float> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<float> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Preitem map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_preitem(in, out, map, rand, name, size);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_preitem(in, out, map, NULL, name, size);
                    }
                }
                break;

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    graph::input_nodes<double, true> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Preitem input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<double, true> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<double, true> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Preitem map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_preitem(in, out, map, rand, name, size);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_preitem(in, out, map, NULL, name, size);
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    graph::input_nodes<double> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Preitem input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<double> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<double> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Preitem map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_preitem(in, out, map, rand, name, size);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_preitem(in, out, map, NULL, name, size);
                    }
                }
                break;

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    graph::input_nodes<std::complex<float>, true> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Preitem input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<std::complex<float>, true> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<std::complex<float>, true> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Preitem map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_preitem(in, out, map, rand, name, size);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_preitem(in, out, map, NULL, name, size);
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    graph::input_nodes<std::complex<float>> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Preitem input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<std::complex<float>> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<std::complex<float>> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Preitem map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_preitem(in, out, map, rand, name, size);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_preitem(in, out, map, NULL, name, size);
                    }
                }
                break;

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    graph::input_nodes<std::complex<double>, true> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Preitem input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<std::complex<double>, true> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<std::complex<double>, true> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Preitem map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_preitem(in, out, map, rand, name, size);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_preitem(in, out, map, NULL, name, size);
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    graph::input_nodes<std::complex<double>> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Preitem input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<std::complex<double>> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<std::complex<double>> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Preitem map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_preitem(in, out, map, rand, name, size);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_preitem(in, out, map, NULL, name, size);
                    }
                }
                break;
        }
    }

//------------------------------------------------------------------------------
///  @brief Add workflow item.
///
///  @param[in] c            The graph C context.
///  @param[in] inputs       Array of input nodes.
///  @param[in] num_inputs   Number of inputs.
///  @param[in] outputs      Array of output nodes.
///  @param[in] num_outputs  Number of outputs.
///  @param[in] map_inputs   Array of map input nodes.
///  @param[in] map_outputs  Array of map output nodes.
///  @param[in] num_maps     Number of maps.
///  @param[in] random_state Optional random state, can be NULL if not used.
///  @param[in] name         Name for the kernel.
///  @param[in] size         Number of elements to operate on.
//------------------------------------------------------------------------------
    void graph_add_item(STRUCT_TAG graph_c_context *c,
                        graph_node *inputs, size_t num_inputs,
                        graph_node *outputs, size_t num_outputs,
                        graph_node *map_inputs,
                        graph_node *map_outputs, size_t num_maps,
                        graph_node random_state,
                        const char *name,
                        const size_t size) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    graph::input_nodes<float, true> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Work input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<float, true> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<float, true> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Work map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_item(in, out, map, rand, name, size);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_item(in, out, map, NULL, name, size);
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    graph::input_nodes<float> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Work input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<float> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<float> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Work map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_item(in, out, map, rand, name, size);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_item(in, out, map, NULL, name, size);
                    }
                }
                break;

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    graph::input_nodes<double, true> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Work input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<double, true> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<double, true> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Work map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_item(in, out, map, rand, name, size);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_item(in, out, map, NULL, name, size);
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    graph::input_nodes<double> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Work input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<double> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<double> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Work map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_item(in, out, map, rand, name, size);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_item(in, out, map, NULL, name, size);
                    }
                }
                break;

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    graph::input_nodes<std::complex<float>, true> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Work input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<std::complex<float>, true> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<std::complex<float>, true> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Work map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_item(in, out, map, rand, name, size);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_item(in, out, map, NULL, name, size);
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    graph::input_nodes<std::complex<float>> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Work input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<std::complex<float>> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<std::complex<float>> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Work map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_item(in, out, map, rand, name, size);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_item(in, out, map, NULL, name, size);
                    }
                }
                break;

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    graph::input_nodes<std::complex<double>, true> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Work input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<std::complex<double>, true> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<std::complex<double>, true> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Work map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_item(in, out, map, rand, name, size);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_item(in, out, map, NULL, name, size);
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    graph::input_nodes<std::complex<double>> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Work input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<std::complex<double>> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<std::complex<double>> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Work map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_item(in, out, map, rand, name, size);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_item(in, out, map, NULL, name, size);
                    }
                }
                break;
        }
    }

//------------------------------------------------------------------------------
///  @brief Add a converge item.
///
///  @param[in] c            The graph C context.
///  @param[in] inputs       Array of input nodes.
///  @param[in] num_inputs   Number of inputs.
///  @param[in] outputs      Array of output nodes.
///  @param[in] num_outputs  Number of outputs.
///  @param[in] map_inputs   Array of map input nodes.
///  @param[in] map_outputs  Array of map output nodes.
///  @param[in] num_maps     Number of maps.
///  @param[in] random_state Optional random state, can be NULL if not used.
///  @param[in] name         Name for the kernel.
///  @param[in] size         Number of elements to operate on.
///  @param[in] tol          Tolarance to converge the function to.
///  @param[in] max_iter     Maximum number of iterations before giving up.
//------------------------------------------------------------------------------
    void graph_add_converge_item(STRUCT_TAG graph_c_context *c,
                                 graph_node *inputs, size_t num_inputs,
                                 graph_node *outputs, size_t num_outputs,
                                 graph_node *map_inputs,
                                 graph_node *map_outputs, size_t num_maps,
                                 graph_node random_state,
                                 const char *name,
                                 const size_t size,
                                 const double tol,
                                 const size_t max_iter) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    graph::input_nodes<float, true> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Work input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<float, true> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<float, true> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Work map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_converge_item(in, out, map, rand, name,
                                                      size, tol, max_iter);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_converge_item(in, out, map, NULL, name,
                                                  size, tol, max_iter);
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    graph::input_nodes<float> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Work input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<float> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<float> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Work map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_converge_item(in, out, map, rand, name,
                                                      size, tol, max_iter);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_converge_item(in, out, map, NULL, name,
                                                  size, tol, max_iter);
                    }
                }
                break;

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    graph::input_nodes<double, true> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Work input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<double, true> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<double, true> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Work map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_converge_item(in, out, map, rand, name,
                                                      size, tol, max_iter);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_converge_item(in, out, map, NULL, name,
                                                  size, tol, max_iter);
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    graph::input_nodes<double> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Work input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<double> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<double> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Work map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_converge_item(in, out, map, rand, name,
                                                      size, tol, max_iter);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_converge_item(in, out, map, NULL, name,
                                                  size, tol, max_iter);
                    }
                }
                break;

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    graph::input_nodes<std::complex<float>, true> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Work input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<std::complex<float>, true> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<std::complex<float>, true> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Work map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_converge_item(in, out, map, rand, name,
                                                      size, tol, max_iter);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_converge_item(in, out, map, NULL, name,
                                                  size, tol, max_iter);
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    graph::input_nodes<std::complex<float>> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Work input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<std::complex<float>> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<std::complex<float>> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Work map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_converge_item(in, out, map, rand, name,
                                                      size, tol, max_iter);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_converge_item(in, out, map, NULL, name,
                                                  size, tol, max_iter);
                    }
                }
                break;

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    graph::input_nodes<std::complex<double>, true> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Work input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<std::complex<double>, true> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<std::complex<double>, true> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Work map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_converge_item(in, out, map, rand, name,
                                                      size, tol, max_iter);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_converge_item(in, out, map, NULL, name,
                                                  size, tol, max_iter);
                    }
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    graph::input_nodes<std::complex<double>> in;
                    for (size_t i = 0; i < num_inputs; i++) {
                        auto temp = graph::variable_cast(d->nodes[inputs[i]]);
                        if (temp.get()) {
                            in.push_back(temp);
                        } else {
                            std::cerr << "Work input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    graph::output_nodes<std::complex<double>> out;
                    for (size_t i = 0; i < num_outputs; i++) {
                        out.push_back(d->nodes[outputs[i]]);
                    }
                    graph::map_nodes<std::complex<double>> map;
                    for (size_t i = 0; i < num_maps; i++) {
                        auto temp = graph::variable_cast(d->nodes[map_inputs[i]]);
                        if (temp.get()) {
                            map.push_back({d->nodes[map_outputs[i]], temp});
                        } else {
                            std::cerr << "Work map input " << i << " is not a variable." << std::endl;
                            exit(1);
                        }
                    }
                    if (random_state) {
                        auto rand = graph::random_state_cast(d->nodes[random_state]);
                        if (rand.get()) {
                            d->work.add_converge_item(in, out, map, rand, name,
                                                      size, tol, max_iter);
                        } else {
                            std::cerr << "Invalid random state." << std::endl;
                            exit(1);
                        }
                    } else {
                        d->work.add_converge_item(in, out, map, NULL, name,
                                                  size, tol, max_iter);
                    }
                }
                break;
        }
    }

//------------------------------------------------------------------------------
///  @brief Compile the work items
///
///  @param[in] c The graph C context.
//------------------------------------------------------------------------------
    void graph_compile(graph_c_context *c) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    d->work.compile();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    d->work.compile();
                }
                break;

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    d->work.compile();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    d->work.compile();
                }
                break;

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    d->work.compile();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    d->work.compile();
                }
                break;

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    d->work.compile();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    d->work.compile();
                }
                break;
        }
    }

//------------------------------------------------------------------------------
///  @brief Run pre work items.
///
///  @param[in] c The graph C context.
//------------------------------------------------------------------------------
    void graph_pre_run(graph_c_context *c) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    d->work.pre_run();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    d->work.pre_run();
                }
                break;

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    d->work.pre_run();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    d->work.pre_run();
                }
                break;

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    d->work.pre_run();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    d->work.pre_run();
                }
                break;

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    d->work.pre_run();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    d->work.pre_run();
                }
                break;
        }
    }

//------------------------------------------------------------------------------
///  @brief Run work items.
///
///  @param[in] c The graph C context.
//------------------------------------------------------------------------------
    void graph_run(graph_c_context *c) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    d->work.run();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    d->work.run();
                }
                break;

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    d->work.run();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    d->work.run();
                }
                break;

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    d->work.run();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    d->work.run();
                }
                break;

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    d->work.run();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    d->work.run();
                }
                break;
        }
    }

//------------------------------------------------------------------------------
///  @brief Wait for work items to complete.
///
///  @param[in] c The graph C context.
//------------------------------------------------------------------------------
    void graph_wait(graph_c_context *c) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    d->work.wait();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    d->work.wait();
                }
                break;

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    d->work.wait();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    d->work.wait();
                }
                break;

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    d->work.wait();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    d->work.wait();
                }
                break;

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    d->work.wait();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    d->work.wait();
                }
                break;
        }
    }

//------------------------------------------------------------------------------
///  @brief Copy data to a device buffer.
///
///  @param[in] c      The graph C context.
///  @param[in] node   Node to copy to.
///  @param[in] source Source to copy from.
//------------------------------------------------------------------------------
    void graph_copy_to_device(STRUCT_TAG graph_c_context *c,
                              graph_node node,
                              void *source) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    d->work.copy_to_device(d->nodes[node], static_cast<float *> (source));
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    d->work.copy_to_device(d->nodes[node], static_cast<float *> (source));
                }
                break;

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    d->work.copy_to_device(d->nodes[node], static_cast<double *> (source));
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    d->work.copy_to_device(d->nodes[node], static_cast<double *> (source));
                }
                break;

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    d->work.copy_to_device(d->nodes[node], static_cast<std::complex<float> *> (source));
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    d->work.copy_to_device(d->nodes[node], static_cast<std::complex<float> *> (source));
                }
                break;

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    d->work.copy_to_device(d->nodes[node], static_cast<std::complex<double> *> (source));
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    d->work.copy_to_device(d->nodes[node], static_cast<std::complex<double> *> (source));
                }
                break;
        }
    }

//------------------------------------------------------------------------------
///  @brief Copy data to a host buffer.
///
///  @param[in] c           The graph C context.
///  @param[in] node        Node to copy from.
///  @param[in] destination Host side buffer to copy to.
//------------------------------------------------------------------------------
    void graph_copy_to_host(STRUCT_TAG graph_c_context *c,
                            graph_node node,
                            void *destination) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    d->work.copy_to_host(d->nodes[node], static_cast<float *> (destination));
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    d->work.copy_to_host(d->nodes[node], static_cast<float *> (destination));
                }
                break;

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    d->work.copy_to_host(d->nodes[node], static_cast<double *> (destination));
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    d->work.copy_to_host(d->nodes[node], static_cast<double *> (destination));
                }
                break;

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    d->work.copy_to_host(d->nodes[node], static_cast<std::complex<float> *> (destination));
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    d->work.copy_to_host(d->nodes[node], static_cast<std::complex<float> *> (destination));
                }
                break;

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    d->work.copy_to_host(d->nodes[node], static_cast<std::complex<double> *> (destination));
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    d->work.copy_to_host(d->nodes[node], static_cast<std::complex<double> *> (destination));
                }
                break;
        }
    }

//------------------------------------------------------------------------------
///  @brief Print a value from nodes.
///
///  @param[in] c         The graph C context.
///  @param[in] index     Particle index to print.
///  @param[in] nodes     Nodes to print.
///  @param[in] num_nodes Number of nodes.
//------------------------------------------------------------------------------
    void graph_print(STRUCT_TAG graph_c_context *c,
                     const size_t index,
                     graph_node *nodes,
                     const size_t num_nodes) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    graph::output_nodes<float, true> out;
                    for (size_t i = 0; i < num_nodes; i++) {
                        out.push_back(d->nodes[nodes[i]]);
                    }
                    d->work.print(index, out);
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    graph::output_nodes<float> out;
                    for (size_t i = 0; i < num_nodes; i++) {
                        out.push_back(d->nodes[nodes[i]]);
                    }
                    d->work.print(index, out);
                }
                break;

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    graph::output_nodes<double, true> out;
                    for (size_t i = 0; i < num_nodes; i++) {
                        out.push_back(d->nodes[nodes[i]]);
                    }
                    d->work.print(index, out);
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    graph::output_nodes<double> out;
                    for (size_t i = 0; i < num_nodes; i++) {
                        out.push_back(d->nodes[nodes[i]]);
                    }
                    d->work.print(index, out);
                }
                break;

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    graph::output_nodes<std::complex<float>, true> out;
                    for (size_t i = 0; i < num_nodes; i++) {
                        out.push_back(d->nodes[nodes[i]]);
                    }
                    d->work.print(index, out);
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    graph::output_nodes<std::complex<float>> out;
                    for (size_t i = 0; i < num_nodes; i++) {
                        out.push_back(d->nodes[nodes[i]]);
                    }
                    d->work.print(index, out);
                }
                break;

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    graph::output_nodes<std::complex<double>, true> out;
                    for (size_t i = 0; i < num_nodes; i++) {
                        out.push_back(d->nodes[nodes[i]]);
                    }
                    d->work.print(index, out);
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    graph::output_nodes<std::complex<double>> out;
                    for (size_t i = 0; i < num_nodes; i++) {
                        out.push_back(d->nodes[nodes[i]]);
                    }
                    d->work.print(index, out);
                }
                break;
        }
    }

//------------------------------------------------------------------------------
///  @brief Take derivative ∂f∂x.
///
///  @param[in] c     The graph C context.
///  @param[in] fnode The function expression to take the derivative of.
///  @param[in] xnode The expression to take the derivative with respect to.
//------------------------------------------------------------------------------
    graph_node graph_df(STRUCT_TAG graph_c_context *c,
                        graph_node fnode,
                        graph_node xnode) {
        switch (c->type) {
            case FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<float, true> *> (c);
                    auto dfdx = d->nodes[fnode]->df(d->nodes[xnode]);
                    d->nodes[dfdx.get()] = dfdx;
                    return dfdx.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<float> *> (c);
                    auto dfdx = d->nodes[fnode]->df(d->nodes[xnode]);
                    d->nodes[dfdx.get()] = dfdx;
                    return dfdx.get();
                }

            case DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<double, true> *> (c);
                    auto dfdx = d->nodes[fnode]->df(d->nodes[xnode]);
                    d->nodes[dfdx.get()] = dfdx;
                    return dfdx.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<double> *> (c);
                    auto dfdx = d->nodes[fnode]->df(d->nodes[xnode]);
                    d->nodes[dfdx.get()] = dfdx;
                    return dfdx.get();
                }

            case COMPLEX_FLOAT:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>, true> *> (c);
                    auto dfdx = d->nodes[fnode]->df(d->nodes[xnode]);
                    d->nodes[dfdx.get()] = dfdx;
                    return dfdx.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<float>> *> (c);
                    auto dfdx = d->nodes[fnode]->df(d->nodes[xnode]);
                    d->nodes[dfdx.get()] = dfdx;
                    return dfdx.get();
                }

            case COMPLEX_DOUBLE:
                if (c->safe_math) {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>, true> *> (c);
                    auto dfdx = d->nodes[fnode]->df(d->nodes[xnode]);
                    d->nodes[dfdx.get()] = dfdx;
                    return dfdx.get();
                } else {
                    auto d = reinterpret_cast<graph_c_context_type<std::complex<double>> *> (c);
                    auto dfdx = d->nodes[fnode]->df(d->nodes[xnode]);
                    d->nodes[dfdx.get()] = dfdx;
                    return dfdx.get();
                }
        }
    }
}
