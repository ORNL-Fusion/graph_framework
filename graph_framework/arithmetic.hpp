//------------------------------------------------------------------------------
///  @file arithmetic.hpp
///  @brief Basic arithmetic operations.
///
///  Defines basic operators.
//------------------------------------------------------------------------------

#ifndef arithmetic_h
#define arithmetic_h

#include "trigonometry.hpp"

namespace graph {
//------------------------------------------------------------------------------
///  @brief Check if nodes are constant combineable.
///
///  @tparam T         Base type of the nodes.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] a Opperand A
///  @param[in] b Opperand B
///  @returns True if a and b are combinable.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    bool is_constant_combineable(shared_leaf<T, SAFE_MATH> a,
                                 shared_leaf<T, SAFE_MATH> b) {
        if (a->is_constant() && b->is_constant()) {
            auto a1 = piecewise_1D_cast(a);
            auto a2 = piecewise_2D_cast(a);
            auto b2 = piecewise_2D_cast(b);

            return constant_cast(a).get()                                     ||
                   constant_cast(b).get()                                     ||
                   (a1.get() && a1->is_arg_match(b))                          ||
                   (a2.get() && a2->is_arg_match(b))                          ||
                   (a2.get() && (a2->is_row_match(b) || a2->is_col_match(b))) ||
                   (b2.get() && (b2->is_row_match(a) || b2->is_col_match(a)));
        }
        return false;
    }

//------------------------------------------------------------------------------
///  @brief Check if the constants are promotable.
///
///  @tparam T         Base type of the nodes.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] a Opperand A
///  @param[in] b Opperand B
///  @returns True if a is promoteable over b.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    bool is_constant_promotable(shared_leaf<T, SAFE_MATH> a,
                                shared_leaf<T, SAFE_MATH> b) {
        auto b1 = piecewise_1D_cast(b);
        auto b2 = piecewise_2D_cast(b);

        return a->is_constant() &&
               (!b->is_constant()                                  ||
                (constant_cast(a).get() && (b1.get() || b2.get())) ||
                (piecewise_1D_cast(a).get() && b2.get()));
    }

//------------------------------------------------------------------------------
///  @brief Check if the variable is combinable.
///
///  @tparam T         Base type of the nodes.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] a Opperand A
///  @param[in] b Opperand B
///  @returns True if a and b are combinable.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    bool is_variable_combineable(shared_leaf<T, SAFE_MATH> a,
                                 shared_leaf<T, SAFE_MATH> b) {
        return a->is_power_base_match(b);
    }

//------------------------------------------------------------------------------
///  @brief Check if the variable is variable is promotable.
///
///  @tparam T         Base type of the nodes.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] a Opperand A
///  @param[in] b Opperand B
///  @returns True if a and b are combinable.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    bool is_variable_promotable(shared_leaf<T, SAFE_MATH> a,
                                 shared_leaf<T, SAFE_MATH> b) {
        return !b->is_constant()                           &&
               (a->is_all_variables()                       &&
                (!b->is_all_variables()                     ||
                 (b->is_all_variables() &&
                  a->get_complexity() < b->get_complexity())));
    }

//------------------------------------------------------------------------------
///  @brief Check if the exponent is greater than the other.
///
///  @tparam T         Base type of the nodes.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] a Opperand A
///  @param[in] b Opperand B
///  @returns True if a and b are combinable.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    bool is_greater_exponent(shared_leaf<T, SAFE_MATH> a,
                             shared_leaf<T, SAFE_MATH> b) {
        auto ae = constant_cast(a->get_power_exponent());
        auto be = constant_cast(b->get_power_exponent());
        
        return ae.get() && be.get() &&
               std::abs(ae->evaluate().at(0)) > std::abs(be->evaluate().at(0));
    }

//******************************************************************************
//  Add node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief An addition node.
///
///  Note use templates here to defer this so it can use the operator functions.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class add_node final : public branch_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] l Left node pointer.
///  @param[in] r Right node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *l,
                                     leaf_node<T, SAFE_MATH> *r) {
            return jit::format_to_string(reinterpret_cast<size_t> (l)) + "+" +
                   jit::format_to_string(reinterpret_cast<size_t> (r));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct an addition node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        add_node(shared_leaf<T, SAFE_MATH> l,
                 shared_leaf<T, SAFE_MATH> r) :
        branch_node<T, SAFE_MATH> (l, r, add_node::to_string(l.get(),
                                                             r.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of addition.
///
///  result = l + r
///
///  @returns The value of l + r.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> l_result = this->left->evaluate();
            backend::buffer<T> r_result = this->right->evaluate();
            return l_result + r_result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce an addition node.
///
///  @returns A reduced addition node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
//  Constant reductions.
            auto l = constant_cast(this->left);
            auto r = constant_cast(this->right);

            if (l.get() && l->is(0)) {
                return this->right;
            } else if (r.get() && r->is(0)) {
                return this->left;
            } else if (l.get() && r.get()) {
                return constant<T, SAFE_MATH> (this->evaluate());
            } else if (r.get() && !l.get()) {
                return this->right + this->left;
            }

            auto pl1 = piecewise_1D_cast(this->left);
            auto pr1 = piecewise_1D_cast(this->right);

            if (pl1.get() && (r.get() || pl1->is_arg_match(this->right))) {
                return piecewise_1D(this->evaluate(), pl1->get_arg());
            } else if (pr1.get() && (l.get() || pr1->is_arg_match(this->left))) {
                return piecewise_1D(this->evaluate(), pr1->get_arg());
            }

            auto pl2 = piecewise_2D_cast(this->left);
            auto pr2 = piecewise_2D_cast(this->right);

            if (pl2.get() && (r.get() || pl2->is_arg_match(this->right))) {
                return piecewise_2D(this->evaluate(),
                                    pl2->get_num_columns(),
                                    pl2->get_left(),
                                    pl2->get_right());
            } else if (pr2.get() && (l.get() || pr2->is_arg_match(this->left))) {
                return piecewise_2D(this->evaluate(),
                                    pr2->get_num_columns(),
                                    pr2->get_left(),
                                    pr2->get_right());
            }

//  Combine 2D and 1D piecewise constants if a row or column matches.
            if (pr2.get() && pr2->is_row_match(this->left)) {
                backend::buffer<T> result = pl1->evaluate();
                result.add_row(pr2->evaluate());
                return piecewise_2D(result,
                                    pr2->get_num_columns(),
                                    pr2->get_left(),
                                    pr2->get_right());
            } else if (pr2.get() && pr2->is_col_match(this->left)) {
                backend::buffer<T> result = pl1->evaluate();
                result.add_col(pr2->evaluate());
                return piecewise_2D(result,
                                    pr2->get_num_columns(),
                                    pr2->get_left(),
                                    pr2->get_right());
            } else if (pl2.get() && pl2->is_row_match(this->right)) {
                backend::buffer<T> result = pl2->evaluate();
                result.add_row(pr1->evaluate());
                return piecewise_2D(result,
                                    pl2->get_num_columns(),
                                    pl2->get_left(),
                                    pl2->get_right());
            } else if (pl2.get() && pl2->is_col_match(this->right)) {
                backend::buffer<T> result = pl2->evaluate();
                result.add_col(pr1->evaluate());
                return piecewise_2D(result,
                                    pl2->get_num_columns(),
                                    pl2->get_left(),
                                    pl2->get_right());
            }

//  Idenity reductions.
            if (this->left->is_match(this->right)) {
                return 2.0*this->left;
            }
            
//  Common factor reduction. If the left and right are both muliply nodes check
//  for a common factor. So you can change a*b + a*c -> a*(b + c).
            auto lm = multiply_cast(this->left);
            auto rm = multiply_cast(this->right);

//  v1 + -c*v2 -> v1 - c*v2
//  -c*v1 + v2 -> v2 - c*v1
            if (rm.get()                      &&
                rm->get_left()->is_constant() &&
                rm->get_left()->evaluate().is_negative()) {
                return this->left - (-this->right);
            } else if (rm.get()                      &&
                       rm->get_left()->is_constant() &&
                       rm->get_left()->evaluate().is_negative()) {
                return this->right - (-this->left);
            }

//  a*b + c -> fma(a,b,c)
//  a + b*c -> fma(b,c,a)
            if (lm.get()) {
                return fma(lm->get_left(), lm->get_right(), this->right);
            } else if (rm.get()) {
                return fma(rm->get_left(), rm->get_right(), this->left);
            }

//  Common denominator reduction. If the left and right are both divide nodes
//  for a common denominator. So you can change a/b + c/b -> (a + c)/d.
            auto ld = divide_cast(this->left);
            auto rd = divide_cast(this->right);

//  c is a constant.
//  a + -c/b -> a - c/b
//  a + (-c*d)/b -> a - (c*d)/b
//  -c/a + b -> b - c/a
//  (-c*d)/a + b -> b - (c*d)/a
            if (rd.get()) {
                auto rdlm = multiply_cast(rd->get_left());
                if ((rd->get_left()->is_constant() &&
                     rd->get_left()->evaluate().is_negative()) ||
                    (rdlm.get() &&
                     (rdlm->get_left()->is_constant() &&
                      rdlm->get_left()->evaluate().is_negative()))) {
                    return this->left - (-rd->get_left())/rd->get_right();
                }
            } else if (ld.get()) {
                auto ldlm = multiply_cast(ld->get_left());
                if ((ld->get_left()->is_constant() &&
                     ld->get_left()->evaluate().is_negative()) ||
                    (ldlm.get() &&
                     (ldlm->get_left()->is_constant() &&
                      ldlm->get_left()->evaluate().is_negative()))) {
                    return this->right - (-ld->get_left())/ld->get_right();
                }
            }

            if (ld.get() && rd.get()) {
                if (ld->get_right()->is_match(rd->get_right())) {
                    return (ld->get_left() + rd->get_left())/ld->get_right();
                }

                auto ldlm = multiply_cast(ld->get_left());
                auto rdlm = multiply_cast(rd->get_left());
//  a/b - c*a/d -> (1/b - c/d)*a
//  a/b - a*c/d -> (1/b - c/d)*a
//  c*a/b - a/d -> (c/b - 1/d)*a
//  a*c/b - a/d -> (c/b - 1/d)*a
                if (rdlm.get()) {
                    if (ld->get_left()->is_match(rdlm->get_left())) {
                        return (1.0/ld->get_right() +
                                rdlm->get_right()/rd->get_right())*rdlm->get_left();
                    } else if (ld->get_left()->is_match(rdlm->get_right())) {
                        return (1.0/ld->get_right() +
                                rdlm->get_left()/rd->get_right())*rdlm->get_right();
                    }
                } else if (ldlm.get()) {
                    if (rd->get_left()->is_match(ldlm->get_left())) {
                        return (ldlm->get_right()/ld->get_right() +
                                1.0/rd->get_right())*ldlm->get_left();
                    } else if (rd->get_left()->is_match(ldlm->get_right())) {
                        return (ldlm->get_left()/ld->get_right() +
                                1.0/rd->get_right())*ldlm->get_right();
                    }
                }

//  c1*a/b + c2*a/d = c3*(a/b + c4*a/d)
//  a*b/c + d*b/e -> (a/c + d/e)*b
//  Make sure we prevent combining constants when we just need to factor out a
//  common term.
//  c1*a/b + c2*a/d -> (c1/b + c2/d)*a
                if (ldlm.get() && rdlm.get()) {
                    if (is_constant_combineable(ldlm->get_left(),
                                                rdlm->get_left()) &&
                        !ldlm->get_right()->is_match(rdlm->get_right())) {
                        return (ldlm->get_right()/ld->get_right() +
                                rdlm->get_left()/ldlm->get_left() *
                                rdlm->get_right()/rd->get_right())*ldlm->get_left();
                    }

                    if (ldlm->get_right()->is_match(rdlm->get_right())) {
                        return (ldlm->get_left()/ld->get_right() +
                                rdlm->get_left()/rd->get_right())*ldlm->get_right();
                    } else if (ldlm->get_right()->is_match(rdlm->get_left())) {
                        return (ldlm->get_left()/ld->get_right() +
                                rdlm->get_right()/rd->get_right())*ldlm->get_right();
                    } else if (ldlm->get_left()->is_match(rdlm->get_right())) {
                        return (ldlm->get_right()/ld->get_right() +
                                rdlm->get_left()/rd->get_right())*ldlm->get_left();
                    } else if (ldlm->get_left()->is_match(rdlm->get_left())) {
                        return (ldlm->get_right()/ld->get_right() +
                                rdlm->get_right()/rd->get_right())*ldlm->get_left();
                    }
                }

//  (a/(c*b) + d/(e*c)) -> (a/b + d/e)/c
//  (a/(b*c) + d/(e*c)) -> (a/b + d/e)/c
//  (a/(c*b) + d/(c*e)) -> (a/b + d/e)/c
//  (a/(b*c) + d/(c*e)) -> (a/b + d/e)/c
                auto ldrm = multiply_cast(ld->get_right());
                auto rdrm = multiply_cast(rd->get_right());
                if (ldrm.get() && rdrm.get()) {
                    if (ldrm->get_right()->is_match(rdrm->get_right())) {
                        return (ld->get_left()/ldrm->get_left() +
                                rd->get_left()/rdrm->get_left())/ldrm->get_right();
                    } else if (ldrm->get_right()->is_match(rdrm->get_left())) {
                        return (ld->get_left()/ldrm->get_left() +
                                rd->get_left()/rdrm->get_right())/ldrm->get_right();
                    } else if (ldrm->get_left()->is_match(rdrm->get_right())) {
                        return (ld->get_left()/ldrm->get_right() +
                                rd->get_left()/rdrm->get_left())/ldrm->get_left();
                    } else if (ldrm->get_left()->is_match(rdrm->get_left())) {
                        return (ld->get_left()/ldrm->get_right() +
                                rd->get_left()/rdrm->get_right())/ldrm->get_left();
                    }
                }

//  a/b + c/(b*d) -> (a*b + c)/(b*d)
//  a/b + c/(d*b) -> (a*b + c)/(b*d)
//  a/(b*d) + c/b -> (c*b + a)/(b*d)
//  a/(d*b) + c/b -> (c*b + a)/(b*d)
                if (rdrm.get()) {
                    if (ld->get_right()->is_match(rdrm->get_left())) {
                        return fma(ld->get_left(),
                                   rdrm->get_right(),
                                   rd->get_left()) /
                               rd->get_right();
                    } else if (ld->get_right()->is_match(rdrm->get_right())) {
                        return fma(ld->get_left(),
                                   rdrm->get_left(),
                                   rd->get_left()) /
                               rd->get_right();
                    }
                } else if (ldrm.get()) {
                    if (rd->get_right()->is_match(ldrm->get_left())) {
                        return fma(rd->get_left(),
                                   ldrm->get_right(),
                                   ld->get_left()) /
                               ld->get_right();
                    } else if (rd->get_right()->is_match(ldrm->get_right())) {
                        return fma(rd->get_left(),
                                   ldrm->get_left(),
                                   ld->get_left()) /
                               ld->get_right();
                    }
                }
            }

//  Chained addition reductions.
//  a + (a + b) = fma(2,a,b)
//  a + (b + a) = fma(2,a,b)
//  (a + b) + a = fma(2,a,b)
//  (b + a) + a = fma(2,a,b)
            auto la = add_cast(this->left);
            if (la.get()) {
                if (this->right->is_match(la->get_left())) {
                    return fma(2.0, this->right, la->get_right());
                } else if (this->right->is_match(la->get_right())) {
                    return fma(2.0, this->right, la->get_left());
                }
            }
            auto ra = add_cast(this->right);
            if (ra.get()) {
                if (this->left->is_match(ra->get_left())) {
                    return fma(2.0, this->left, ra->get_right());
                } else if (this->left->is_match(ra->get_right())) {
                    return fma(2.0, this->left, ra->get_left());
                }
            }

//  Move cases like
//  (c1 + c2/x) + c3/y -> c1 + (c2/x + c3/y)
//  (c1 - c2/x) + c3/y -> c1 + (c3/y - c2/x)
//  in case of common denominators.
            if (rd.get()) {
                if (la.get() && divide_cast(la->get_right()).get()) {
                    return la->get_left() + (la->get_right() + this->right);
                }

                auto ls = subtract_cast(this->left);
                if (ls.get() && divide_cast(ls->get_right()).get()) {
                    return ls->get_left() + (this->right - ls->get_right());
                }
            }

            auto lfma = fma_cast(this->left);
            auto rfma = fma_cast(this->right);
            if (lfma.get()) {
//  fma(c,d,e) + a -> fma(c,d,e + a)
                return fma(lfma->get_left(),
                           lfma->get_middle(),
                           lfma->get_right() + this->right);
            } else if (rfma.get()) {
//  a + fma(c,d,e) -> fma(c,d,a + e)
                return fma(rfma->get_left(),
                           rfma->get_middle(),
                           this->left + rfma->get_right());
            }

//  fma(b,a,d) + fma(c,a,e) -> fma(a,b + c, d + e)
//  fma(a,b,d) + fma(c,a,e) -> fma(a,b + c, d + e)
//  fma(b,a,d) + fma(a,c,e) -> fma(a,b + c, d + e)
//  fma(a,b,d) + fma(a,c,e) -> fma(a,b + c, d + e)
            if (lfma.get() && rfma.get()) {
                if (lfma->get_middle()->is_match(rfma->get_middle())) {
                    return fma(lfma->get_middle(),
                               lfma->get_left() + rfma->get_left(),
                               lfma->get_right() + rfma->get_right());
                } else if (lfma->get_left()->is_match(rfma->get_middle())) {
                    return fma(lfma->get_left(),
                               lfma->get_middle() + rfma->get_left(),
                               lfma->get_right() + rfma->get_right());
                } else if (lfma->get_middle()->is_match(rfma->get_left())) {
                    return fma(lfma->get_middle(),
                               lfma->get_left() + rfma->get_middle(),
                               lfma->get_right() + rfma->get_right());
                } else if (lfma->get_left()->is_match(rfma->get_left())) {
                    return fma(lfma->get_left(),
                               lfma->get_middle() + rfma->get_middle(),
                               lfma->get_right() + rfma->get_right());
                }
            }

//  Handle cases like:
//  (a/y)^e + b/y^e -> (a^2 + b)/(y^e)
//  b/y^e + (a/y)^e -> (b + a^2)/(y^e)
//  (a/y)^e + (b/y)^e -> (a^2 + b^2)/(y^e)
            auto pl = pow_cast(this->left);
            auto pr = pow_cast(this->right);
            if (pl.get() && rd.get()) {
                auto rdp = pow_cast(rd->get_right());
                if (rdp.get() && pl->get_right()->is_match(rdp->get_right())) {
                    auto plld = divide_cast(pl->get_left());
                    if (plld.get() &&
                        rdp->get_left()->is_match(plld->get_right())) {
                        return (pow(plld->get_left(), pl->get_right()) +
                                rd->get_left()) /
                               pow(rdp->get_left(), pl->get_right());
                    }
                }
            } else if (pr.get() && ld.get()) {
                auto ldp = pow_cast(ld->get_right());
                if (ldp.get() && pr->get_right()->is_match(ldp->get_right())) {
                    auto prld = divide_cast(pr->get_left());
                    if (prld.get() &&
                        ldp->get_left()->is_match(prld->get_right())) {
                        return (pow(prld->get_left(), pr->get_right()) +
                                ld->get_left()) /
                               pow(ldp->get_left(), pr->get_right());
                    }
                }
            } else if (pl.get() && pr.get()) {
                if (pl->get_right()->is_match(pr->get_right())) {
                    auto pld = divide_cast(pl->get_left());
                    auto prd = divide_cast(pr->get_left());
                    if (pld.get() && prd.get() &&
                        pld->get_right()->is_match(prd->get_right())) {
                        return (pow(pld->get_left(), pl->get_right()) +
                                pow(prd->get_left(), pl->get_right())) /
                               pow(pld->get_right(), pl->get_right());
                    }
                }
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d add(a,b)/dx = da/dx + db/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        df(shared_leaf<T, SAFE_MATH> x) {
            if (this->is_match(x)) {
                return one<T, SAFE_MATH> ();
            }

            const size_t hash = reinterpret_cast<size_t> (x.get());
            if (this->df_cache.find(hash) == this->df_cache.end()) {
                this->df_cache[hash] = this->left->df(x) + this->right->df(x);
            }
            return this->df_cache[hash];
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @param[in]     usage     List of register usage count.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream, 
                                                                  registers,
                                                                  usage);
                shared_leaf<T, SAFE_MATH> r = this->right->compile(stream,
                                                                   registers,
                                                                   usage);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = "
                       << registers[l.get()] << " + "
                       << registers[r.get()];
                this->endline(stream, usage);
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

            auto x_cast = add_cast(x);
            if (x_cast.get()) {
//  Addition is commutative.
                if ((this->left->is_match(x_cast->get_left()) &&
                     this->right->is_match(x_cast->get_right())) ||
                    (this->right->is_match(x_cast->get_left()) &&
                     this->left->is_match(x_cast->get_right()))) {
                    return true;
                }
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            bool l_brackets = add_cast(this->left).get() ||
                              subtract_cast(this->left).get();
            bool r_brackets = add_cast(this->right).get() ||
                              subtract_cast(this->right).get();
            if (l_brackets) {
                std::cout << "\\left(";
            }
            this->left->to_latex();
            if (l_brackets) {
                std::cout << "\\right)";
            }
            std::cout << "+";
            if (r_brackets) {
                std::cout << "\\left(";
            }
            this->right->to_latex();
            if (r_brackets) {
                std::cout << "\\right)";
            }
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> remove_pseudo() {
            if (this->has_pseudo()) {
                return this->left->remove_pseudo() +
                       this->right->remove_pseudo();
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
        virtual shared_leaf<T, SAFE_MATH> to_vizgraph(std::stringstream &stream,
                                                      jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                const std::string name = jit::to_string('r', this);
                registers[this] = name;
                stream << "    " << name
                       << " [label = \"+\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto l = this->left->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[l.get()] << ";" << std::endl;
                auto r = this->right->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[r.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build add node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> add(shared_leaf<T, SAFE_MATH> l,
                                  shared_leaf<T, SAFE_MATH> r) {
        auto temp = std::make_shared<add_node<T, SAFE_MATH>> (l, r)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::cache.find(i) ==
                leaf_node<T, SAFE_MATH>::cache.end()) {
                leaf_node<T, SAFE_MATH>::cache[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::cache[i])) {
                return leaf_node<T, SAFE_MATH>::cache[i];
            }
        }
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

//------------------------------------------------------------------------------
///  @brief Build add node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator+(shared_leaf<T, SAFE_MATH> l,
                                        shared_leaf<T, SAFE_MATH> r) {
        return add<T, SAFE_MATH> (l, r);
    }

//------------------------------------------------------------------------------
///  @brief Build add node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam L         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar L, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator+(const L l,
                                        shared_leaf<T, SAFE_MATH> r) {
        return add<T, SAFE_MATH> (constant<T, SAFE_MATH> (static_cast<T> (l)), r);
    }

//------------------------------------------------------------------------------
///  @brief Build add node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam R         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar R, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator+(shared_leaf<T, SAFE_MATH> l,
                                        const R r) {
        return add<T, SAFE_MATH> (l, constant<T, SAFE_MATH> (static_cast<T> (r)));
    }

///  Convenience type alias for shared add nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_add = std::shared_ptr<add_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a add node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_add<T, SAFE_MATH> add_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<add_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Subtract node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A subtraction node.
///
///  Note use templates here to defer this so it can use the operator functions.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class subtract_node final : public branch_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] l Left node pointer.
///  @param[in] r Right node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *l,
                                     leaf_node<T, SAFE_MATH> *r) {
            return jit::format_to_string(reinterpret_cast<size_t> (l)) + "-" +
                   jit::format_to_string(reinterpret_cast<size_t> (r));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Consruct a subtraction node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        subtract_node(shared_leaf<T, SAFE_MATH> l,
                      shared_leaf<T, SAFE_MATH> r) :
        branch_node<T, SAFE_MATH> (l, r, subtract_node::to_string(l.get(),
                                                                  r.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of subtraction.
///
///  result = l - r
///
///  @returns The value of l - r.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> l_result = this->left->evaluate();
            backend::buffer<T> r_result = this->right->evaluate();
            return l_result - r_result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce an subtraction node.
///
///  @returns A reduced subtraction node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
//  Idenity reductions.
            auto l = constant_cast(this->left);
            if (this->left->is_match(this->right)) {
                auto l = constant_cast(this->left);
                if (l.get() && l->is(0)) {
                    return this->left;
                }

                return zero<T, SAFE_MATH> ();
            }

//  Constant reductions.
            auto r = constant_cast(this->right);

            if (l.get() && l->is(0)) {
                return -this->right;
            } else if (r.get() && r->is(0)) {
                return this->left;
            } else if (l.get() && r.get()) {
                return constant<T, SAFE_MATH> (this->evaluate());
            } else if (r.get() && r->evaluate().is_negative()) {
                return this->left + -this->right;
            }

            auto pl1 = piecewise_1D_cast(this->left);
            auto pr1 = piecewise_1D_cast(this->right);

            if (pl1.get() && (r.get() || pl1->is_arg_match(this->right))) {
                return piecewise_1D(this->evaluate(), pl1->get_arg());
            } else if (pr1.get() && (l.get() || pr1->is_arg_match(this->left))) {
                return piecewise_1D(this->evaluate(), pr1->get_arg());
            }

            auto pl2 = piecewise_2D_cast(this->left);
            auto pr2 = piecewise_2D_cast(this->right);

            if (pl2.get() && (r.get() || pl2->is_arg_match(this->right))) {
                return piecewise_2D(this->evaluate(),
                                    pl2->get_num_columns(),
                                    pl2->get_left(),
                                    pl2->get_right());
            } else if (pr2.get() && (l.get() || pr2->is_arg_match(this->left))) {
                return piecewise_2D(this->evaluate(),
                                    pr2->get_num_columns(),
                                    pr2->get_left(),
                                    pr2->get_right());
            }

//  Combine 2D and 1D piecewise constants if a row or column matches.
            if (pr2.get() && pr2->is_row_match(this->left)) {
                backend::buffer<T> result = pl1->evaluate();
                result.subtract_row(pr2->evaluate());
                return piecewise_2D(result,
                                    pr2->get_num_columns(),
                                    pr2->get_left(),
                                    pr2->get_right());
            } else if (pr2.get() && pr2->is_col_match(this->left)) {
                backend::buffer<T> result = pl1->evaluate();
                result.subtract_col(pr2->evaluate());
                return piecewise_2D(result,
                                    pr2->get_num_columns(),
                                    pr2->get_left(),
                                    pr2->get_right());
            } else if (pl2.get() && pl2->is_row_match(this->right)) {
                backend::buffer<T> result = pl2->evaluate();
                result.subtract_row(pr1->evaluate());
                return piecewise_2D(result,
                                    pl2->get_num_columns(),
                                    pl2->get_left(),
                                    pl2->get_right());
            } else if (pl2.get() && pl2->is_col_match(this->right)) {
                backend::buffer<T> result = pl2->evaluate();
                result.subtract_col(pr1->evaluate());
                return piecewise_2D(result,
                                    pl2->get_num_columns(),
                                    pl2->get_left(),
                                    pl2->get_right());
            }

//  Common factor reduction. If the left and right are both muliply nodes check
//  for a common factor. So you can change a*b - a*c -> a*(b - c).
            auto lm = multiply_cast(this->left);
            auto rm = multiply_cast(this->right);

//  c1*(c2 + a) - c3 -> fma(c1,a,c4)
            if (lm.get()) {
                auto lmra = add_cast(lm->get_right());
                if (lmra.get()) {
                    if (is_constant_combineable(lm->get_left(),
                                                lmra->get_left()) &&
                        is_constant_combineable(lm->get_left(),
                                                this->right)) {
                        return fma(lm->get_left(),
                                   lmra->get_right(),
                                   lm->get_left()*lmra->get_left() - this->right);
                    }
                }

                auto lmrs = subtract_cast(lm->get_right());
                if (lmrs.get()) {
                    if (is_constant_combineable(lm->get_left(),
                                                lmrs->get_left()) &&
                        is_constant_combineable(lm->get_left(),
                                                this->right)) {
                        return lm->get_left()*lmrs->get_left() - this->right -
                               lm->get_left()*lmrs->get_right();
                    }
                }
            }

//  Assume constants are on the left.
//  v1 - -c*v2 -> v1 + c*v2
            if (rm.get()                      &&
                rm->get_left()->is_constant() &&
                rm->get_left()->evaluate().is_negative()) {
                return this->left + (-this->right);
            }

            if (lm.get()) {
//  Assume constants are on the left.
//  -a - b -> -(a + b)
                auto lmc = constant_cast(lm->get_left());
                if (lmc.get() && lmc->is(-1)) {
                    return lm->get_left()*(lm->get_right() + this->right);
                }

//  a*v - v = (a - 1)*v
//  v*a - v = (a - 1)*v
                if (this->right->is_match(lm->get_right())) {
                    return (lm->get_left() - 1.0)*this->right;
                } else if (this->right->is_match(lm->get_left())) {
                    return (lm->get_right() - 1.0)*this->right;
                }
            }
//  v - a*v = (1 - a)*v
//  v - v*a = (1 - a)*v
            if (rm.get()) {
                if (this->left->is_match(rm->get_right())) {
                    return (1.0 - rm->get_left())*this->left;
                } else if (this->left->is_match(rm->get_left())) {
                    return (1.0 - rm->get_right())*this->left;
                }
            }

            if (lm.get() && rm.get()) {
                if (lm->get_left()->is_match(rm->get_left())) {
//  a*b - a*c -> a*(b - c)
                    return lm->get_left()*(lm->get_right() - rm->get_right());
                } else if (lm->get_left()->is_match(rm->get_right())) {
//  a*b - c*a -> a*(b - c)
                    return lm->get_left()*(lm->get_right() - rm->get_left());
                } else if (lm->get_right()->is_match(rm->get_left())) {
//  b*a - a*c -> a*(b - c)
                    return lm->get_right()*(lm->get_left() - rm->get_right());
                } else if (lm->get_right()->is_match(rm->get_right())) {
//  b*a - c*a -> a*(b - c)
                    return lm->get_right()*(lm->get_left() - rm->get_left());
                }

//  Change cases like c1*a - c2*b -> c1*(a - c2/c1*b)
//  Note need to make sure c1 doesn't contain any zeros.
                if (lm->get_left()->is_constant() &&
                    rm->get_left()->is_constant() &&
                    !lm->get_left()->has_constant_zero()) {
                    return lm->get_left()*(lm->get_right() -
                                           (rm->get_left()/lm->get_left())*rm->get_right());
                }

//  Handle case
                auto rmrm = multiply_cast(rm->get_right());
                if (rmrm.get()) {
//  a*b - c*(d*b) -> (a - c*d)*b
                    if (lm->get_right()->is_match(rmrm->get_right())) {
                        return (lm->get_left() - rm->get_left()*rmrm->get_left())*lm->get_right();
                    }
//  a*b - c*(b*d) -> (a - c*d)*b
                    if (lm->get_right()->is_match(rmrm->get_left())) {
                        return (lm->get_left() - rm->get_left()*rmrm->get_right())*lm->get_right();
                    }
//  b*a - c*(d*b) -> (a - c*d)*b
                    if (lm->get_left()->is_match(rmrm->get_right())) {
                        return (lm->get_right() - rm->get_left()*rmrm->get_left())*lm->get_left();
                    }
//  b*a - c*(b*d) -> (a - c*d)*b
                    if (lm->get_left()->is_match(rmrm->get_left())) {
                        return (lm->get_right() - rm->get_left()*rmrm->get_right())*lm->get_left();
                    }
                }
                auto lmrm = multiply_cast(lm->get_right());
                if (lmrm.get()) {
//  c*(d*b) - a*b -> (c*d - a)*b
                    if (rm->get_right()->is_match(lmrm->get_right())) {
                        return (lm->get_left()*lmrm->get_left() - rm->get_left())*rm->get_right();
                    }
//  c*(b*d) - a*b -> (c*d - a)*b
                    if (rm->get_right()->is_match(lmrm->get_left())) {
                        return (lm->get_left()*lmrm->get_right() - rm->get_left())*rm->get_right();
                    }
//  c*(d*b) - b*a -> (c*d - a)*b
                    if (rm->get_left()->is_match(lmrm->get_right())) {
                        return (lm->get_left()*lmrm->get_left() - rm->get_right())*rm->get_left();
                    }
//  c*(b*d) - b*a -> (c*d - a)*b
                    if (rm->get_left()->is_match(lmrm->get_left())) {
                        return (lm->get_left()*lmrm->get_right() - rm->get_right())*rm->get_left();
                    }
                }

//  a/b*c - d/b*e -> (a*b - d*e)/b
//  a/b*c - d*e/b -> (a*b - d*e)/b
//  a*c/b - d/b*e -> (a*b - d*e)/b
//  a*c/b - d*e/b -> (a*b - d*e)/b
                auto lmld = divide_cast(lm->get_left());
                auto rmld = divide_cast(rm->get_left());
                auto lmrd = divide_cast(lm->get_right());
                auto rmrd = divide_cast(rm->get_right());
                if (lmld.get() && rmld.get() &&
                    lmld->get_right()->is_match(rmld->get_right())) {
                    return (lmld->get_left()*lm->get_right() -
                            rmld->get_left()*rm->get_right())/lmld->get_right();
                } else if (lmld.get() && rmrd.get() &&
                           lmld->get_right()->is_match(rmrd->get_right())) {
                    return (lmld->get_left()*lm->get_right() -
                            rmrd->get_left()*rm->get_left())/lmld->get_right();
                } else if (lmrd.get() && rmld.get() &&
                           lmrd->get_right()->is_match(rmld->get_right())) {
                    return (lmrd->get_left()*lm->get_left() -
                            rmld->get_left()*rm->get_right())/lmrd->get_right();
                } else if (lmrd.get() && rmrd.get() &&
                           lmrd->get_right()->is_match(rmrd->get_right())) {
                    return (lmrd->get_left()*lm->get_left() -
                            rmrd->get_left()*rm->get_left())/lmrd->get_right();
                }
            }

//  Chained subtraction reductions.
            auto ls = subtract_cast(this->left);
            if (ls.get()) {
                auto lrm = multiply_cast(ls->get_right());
                if (lrm.get() && rm.get()) {
                    if (lrm->get_left()->is_match(rm->get_left())) {
//  (a - c*b) - c*d -> a - (b + d)*c
                        return ls->get_left() -
                               (lrm->get_right() +
                                rm->get_right())*rm->get_left();
                    } else if (lrm->get_left()->is_match(rm->get_right())) {
//  (a - c*b) - d*c -> a - (b + d)*c
                        return ls->get_left() -
                               (lrm->get_right() +
                                rm->get_left())*rm->get_right();
                    } else if (lrm->get_right()->is_match(rm->get_left())) {
//  (a - c*b) - c*d -> a - (b + d)*c
                        return ls->get_left() -
                               (lrm->get_left() +
                                rm->get_right())*rm->get_left();
                    } else if (lrm->get_right()->is_match(rm->get_right())) {
//  (a - c*b) - d*c -> a - (b + d)*c
                        return ls->get_left() -
                               (lrm->get_left() +
                                rm->get_left())*rm->get_right();
                    }
                }
            }

//  Common denominator reduction. If the left and right are both divide nodes
//  for a common denominator. So you can change a/b - c/b -> (a - c)/d.
            auto ld = divide_cast(this->left);
            auto rd = divide_cast(this->right);

//  c is a constant.
//  a - -c/b -> a + c/b
//  a - (-c*d)/b -> a + (c*d)/b
//  -c/a - b -> -(b + c/a)
//  (-c*d)/a - b -> -(b + (c*d)/a)
            if (rd.get()) {
                auto rdlm = multiply_cast(rd->get_left());
                if ((rd->get_left()->is_constant() &&
                     rd->get_left()->evaluate().is_negative()) ||
                    (rdlm.get() &&
                     (rdlm->get_left()->is_constant() &&
                      rdlm->get_left()->evaluate().is_negative()))) {
                    return this->left + -this->right;
                }
            } else if (ld.get()) {
                auto ldlm = multiply_cast(ld->get_left());
                if ((ld->get_left()->is_constant() &&
                     ld->get_left()->evaluate().is_negative()) ||
                    (ldlm.get() &&
                     (ldlm->get_left()->is_constant() &&
                      ldlm->get_left()->evaluate().is_negative()))) {
                    return -(-this->left + this->right);
                }
            }

            if (ld.get() && rd.get()) {
                if (ld->get_right()->is_match(rd->get_right())) {
                    return (ld->get_left() - rd->get_left())/ld->get_right();
                }

                auto ldlm = multiply_cast(ld->get_left());
                auto rdlm = multiply_cast(rd->get_left());
//  a/b - c*a/d -> (1/b - c/d)*a
//  a/b - a*c/d -> (1/b - c/d)*a
//  c*a/b - a/d -> (c/b - 1/d)*a
//  a*c/b - a/d -> (c/b - 1/d)*a
                if (rdlm.get()) {
                    if (ld->get_left()->is_match(rdlm->get_left())) {
                        return (1.0/ld->get_right() -
                                rdlm->get_right()/rd->get_right())*rdlm->get_left();
                    } else if (ld->get_left()->is_match(rdlm->get_right())) {
                        return (1.0/ld->get_right() -
                                rdlm->get_left()/rd->get_right())*rdlm->get_right();
                    }
                } else if (ldlm.get()) {
                    if (rd->get_left()->is_match(ldlm->get_left())) {
                        return (ldlm->get_right()/ld->get_right() -
                                1.0/rd->get_right())*ldlm->get_left();
                    } else if (rd->get_left()->is_match(ldlm->get_right())) {
                        return (ldlm->get_left()/ld->get_right() -
                                1.0/rd->get_right())*ldlm->get_right();
                    }
                }

//  c1*a/b - c2*e/d = c3*(a/b - c4*e/d)
//  a*b/c - d*b/e -> (a/c - d/e)*b
//  Make sure we prevent combining constants when we just need to factor out a
//  common term.
//  c1*a/b - c2*a/d -> (c1/b - c2/d)*a
                if (ldlm.get() && rdlm.get()) {
                    if (is_constant_combineable(ldlm->get_left(),
                                                rdlm->get_left()) &&
                        !ldlm->get_right()->is_match(rdlm->get_right())) {
                        return (ldlm->get_right()/ld->get_right() -
                                rdlm->get_left()/ldlm->get_left() *
                                rdlm->get_right()/rd->get_right())*ldlm->get_left();
                    }

                    if (ldlm->get_right()->is_match(rdlm->get_right())) {
                        return (ldlm->get_left()/ld->get_right() -
                                rdlm->get_left()/rd->get_right())*ldlm->get_right();
                    } else if (ldlm->get_right()->is_match(rdlm->get_left())) {
                        return (ldlm->get_left()/ld->get_right() -
                                rdlm->get_right()/rd->get_right())*ldlm->get_right();
                    } else if (ldlm->get_left()->is_match(rdlm->get_right())) {
                        return (ldlm->get_right()/ld->get_right() -
                                rdlm->get_left()/rd->get_right())*ldlm->get_left();
                    } else if (ldlm->get_left()->is_match(rdlm->get_left())) {
                        return (ldlm->get_right()/ld->get_right() -
                                rdlm->get_right()/rd->get_right())*ldlm->get_left();
                    }
                }

//  (a/(c*b) - d/(e*c)) -> (a/b - d/e)/c
//  (a/(b*c) - d/(e*c)) -> (a/b - d/e)/c
//  (a/(c*b) - d/(c*e)) -> (a/b - d/e)/c
//  (a/(b*c) - d/(c*e)) -> (a/b - d/e)/c
                auto ldrm = multiply_cast(ld->get_right());
                auto rdrm = multiply_cast(rd->get_right());
                if (ldrm.get() && rdrm.get()) {
                    if (ldrm->get_right()->is_match(rdrm->get_right())) {
                        return (ld->get_left()/ldrm->get_left() -
                                rd->get_left()/rdrm->get_left())/ldrm->get_right();
                    } else if (ldrm->get_right()->is_match(rdrm->get_left())) {
                        return (ld->get_left()/ldrm->get_left() -
                                rd->get_left()/rdrm->get_right())/ldrm->get_right();
                    } else if (ldrm->get_left()->is_match(rdrm->get_right())) {
                        return (ld->get_left()/ldrm->get_right() -
                                rd->get_left()/rdrm->get_left())/ldrm->get_left();
                    } else if (ldrm->get_left()->is_match(rdrm->get_left())) {
                        return (ld->get_left()/ldrm->get_right() -
                                rd->get_left()/rdrm->get_right())/ldrm->get_left();
                    }
                }

//  a/b - c/(b*d) -> (a*d - c)/(b*d)
//  a/b - c/(d*b) -> (a*d - c)/(b*d)
//  a/(b*d) - c/b -> (a - c*d)/(b*d)
//  a/(d*b) - c/b -> (a - c*d)/(b*d)
                if (rdrm.get()) {
                    if (ld->get_right()->is_match(rdrm->get_left())) {
                        return (ld->get_left()*rdrm->get_right() - rd->get_left()) /
                               rd->get_right();
                    } else if (ld->get_right()->is_match(rdrm->get_right())) {
                        return (ld->get_left()*rdrm->get_left() - rd->get_left()) /
                               rd->get_right();
                    }
                } else if (ldrm.get()) {
                    if (rd->get_right()->is_match(ldrm->get_left())) {
                        return (ld->get_left() - rd->get_left()*ldrm->get_right()) /
                               ld->get_right();
                    } else if (rd->get_right()->is_match(ldrm->get_right())) {
                        return (ld->get_left() - rd->get_left()*ldrm->get_left()) /
                               ld->get_right();
                    }
                }
            }

//  Move cases like
//  (c1 + c2/x) - c3/y -> c1 + (c2/x - c3/y)
//  (c1 - c2/x) - c3/y -> c1 - (c2/x + c3/y)
//  in case of common denominators.
            if (rd.get()) {
                auto la = add_cast(this->left);
                if (la.get() && divide_cast(la->get_right()).get()) {
                    return la->get_left() + (la->get_right() - this->right);
                } else if (ls.get() && divide_cast(ls->get_right()).get()) {
                    return ls->get_left() - (this->right + ls->get_right());
                }
            }

//  Handle cases like:
//  (a/y)^e - b/y^e -> (a^2 - b)/(y^e)
//  b/y^e - (a/y)^e -> (b - a^2)/(y^e)
//  (a/y)^e - (b/y)^e -> (a^2 - b^2)/(y^e)
            auto pl = pow_cast(this->left);
            auto pr = pow_cast(this->right);
            if (pl.get() && rd.get()) {
                auto rdp = pow_cast(rd->get_right());
                if (rdp.get() && pl->get_right()->is_match(rdp->get_right())) {
                    auto plld = divide_cast(pl->get_left());
                    if (plld.get() &&
                        rdp->get_left()->is_match(plld->get_right())) {
                        return (pow(plld->get_left(), pl->get_right()) -
                                rd->get_left()) /
                               pow(rdp->get_left(), pl->get_right());
                    }
                }
            } else if (pr.get() && ld.get()) {
                auto ldp = pow_cast(ld->get_right());
                if (ldp.get() && pr->get_right()->is_match(ldp->get_right())) {
                    auto prld = divide_cast(pr->get_left());
                    if (prld.get() &&
                        ldp->get_left()->is_match(prld->get_right())) {
                        return (pow(prld->get_left(), pr->get_right()) -
                                ld->get_left()) /
                               pow(ldp->get_left(), pr->get_right());
                    }
                }
            } else if (pl.get() && pr.get()) {
                if (pl->get_right()->is_match(pr->get_right())) {
                    auto pld = divide_cast(pl->get_left());
                    auto prd = divide_cast(pr->get_left());
                    if (pld.get() && prd.get() &&
                        pld->get_right()->is_match(prd->get_right())) {
                        return (pow(pld->get_left(), pl->get_right()) -
                                pow(prd->get_left(), pl->get_right())) /
                               pow(pld->get_right(), pl->get_right());
                    }
                }
            }

            auto lfma = fma_cast(this->left);
            auto rfma = fma_cast(this->right);

            if (lfma.get() && rfma.get()) {
                if (lfma->get_middle()->is_match(rfma->get_middle())) {
                    return fma(lfma->get_left() - rfma->get_left(),
                               lfma->get_middle(),
                               lfma->get_right() - rfma->get_right());
                }
            }

//  fma(c,d,e) - a -> fma(c,d,e - a)
            if (lfma.get() && !this->right->is_all_variables()) {
                return fma(lfma->get_left(),
                           lfma->get_middle(),
                           lfma->get_right() - this->right);
            }

//  Reduce cases chained subtract multiply divide.
            if (ls.get()) {
//  (a - b*c) - d*e -> a - (b*c + d*e)
//  (a - b/c) - d/e -> a - (b/c + d/e)
                auto lsrd = divide_cast(ls->get_right());
                if ((multiply_cast(ls->get_right()).get() && (rm.get() || rd.get())) ||
                    (divide_cast(ls->get_right()).get()   && (rm.get() || rd.get()))) {
                    return ls->get_left() - (ls->get_right() + this->right);
                }
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d sub(a,b)/dx = da/dx - db/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        df(shared_leaf<T, SAFE_MATH> x) {
            if (this->is_match(x)) {
                return one<T, SAFE_MATH> ();
            }

            const size_t hash = reinterpret_cast<size_t> (x.get());
            if (this->df_cache.find(hash) == this->df_cache.end()) {
                this->df_cache[hash] = this->left->df(x) - this->right->df(x);
            }
            return this->df_cache[hash];
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @param[in]     usage     List of register usage count.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream, 
                                                                  registers,
                                                                  usage);
                shared_leaf<T, SAFE_MATH> r = this->right->compile(stream,
                                                                   registers,
                                                                   usage);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = "
                       << registers[l.get()] << " - "
                       << registers[r.get()];
                this->endline(stream, usage);
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

            auto x_cast = subtract_cast(x);
            if (x_cast.get()) {
                return this->left->is_match(x_cast->get_left()) &&
                       this->right->is_match(x_cast->get_right());
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            bool l_brackets = add_cast(this->left).get() ||
                              subtract_cast(this->left).get();
            bool r_brackets = add_cast(this->right).get() ||
                              subtract_cast(this->right).get();
            if (l_brackets) {
                std::cout << "\\left(";
            }
            this->left->to_latex();
            if (l_brackets) {
                std::cout << "\\right)";
            }
            std::cout << "-";
            if (r_brackets) {
                std::cout << "\\left(";
            }
            this->right->to_latex();
            if (r_brackets) {
                std::cout << "\\right)";
            }
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> remove_pseudo() {
            if (this->has_pseudo()) {
                return this->left->remove_pseudo() -
                       this->right->remove_pseudo();
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
        virtual shared_leaf<T, SAFE_MATH> to_vizgraph(std::stringstream &stream,
                                                      jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                const std::string name = jit::to_string('r', this);
                registers[this] = name;
                stream << "    " << name
                       << " [label = \"-\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto l = this->left->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[l.get()] << ";" << std::endl;
                auto r = this->right->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[r.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build subtract node from two leaves.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
///  @returns l - r
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> subtract(shared_leaf<T, SAFE_MATH> l,
                                       shared_leaf<T, SAFE_MATH> r) {
        auto temp = std::make_shared<subtract_node<T, SAFE_MATH>> (l, r)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::cache.find(i) ==
                leaf_node<T, SAFE_MATH>::cache.end()) {
                leaf_node<T, SAFE_MATH>::cache[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::cache[i])) {
                return leaf_node<T, SAFE_MATH>::cache[i];
            }
        }
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

//------------------------------------------------------------------------------
///  @brief Build subtract node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
///  @returns l - r
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator-(shared_leaf<T, SAFE_MATH> l,
                                        shared_leaf<T, SAFE_MATH> r) {
        return subtract<T, SAFE_MATH> (l, r);
    }

//------------------------------------------------------------------------------
///  @brief Build subtract node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam L         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
///  @returns l - r
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar L, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator-(const L l,
                                        shared_leaf<T, SAFE_MATH> r) {
        return subtract<T, SAFE_MATH> (constant<T, SAFE_MATH> (static_cast<T> (l)), r);
    }

//------------------------------------------------------------------------------
///  @brief Build subtract node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam R         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
///  @returns l - r
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar R, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator-(shared_leaf<T, SAFE_MATH> l,
                                        const R r) {
        return subtract<T, SAFE_MATH> (l, constant<T, SAFE_MATH> (static_cast<T> (r)));
    }

//------------------------------------------------------------------------------
///  @brief Negate a node.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] a Argument to negate.
///  @returns -1.0*a
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator-(shared_leaf<T, SAFE_MATH> a) {
        return -1.0*a;
    }

///  Convenience type alias for shared subtract nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_subtract = std::shared_ptr<subtract_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a subtract node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_subtract<T, SAFE_MATH> subtract_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<subtract_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Multiply node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A multiplcation node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class multiply_node final : public branch_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] l Left node pointer.
///  @param[in] r Right node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *l,
                                     leaf_node<T, SAFE_MATH> *r) {
            return jit::format_to_string(reinterpret_cast<size_t> (l)) + "*" +
                   jit::format_to_string(reinterpret_cast<size_t> (r));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Consruct a multiplcation node.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        multiply_node(shared_leaf<T, SAFE_MATH> l,
                      shared_leaf<T, SAFE_MATH> r) :
        branch_node<T, SAFE_MATH> (l, r, multiply_node::to_string(l.get(), r.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of multiplcation.
///
///  result = l*r
///
///  @returns The value of l*r.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> l_result = this->left->evaluate();

//  If the left are right are same don't evaluate the right.
//  NOTE: Do not use is_match here. Remove once power is implimented.
            if (this->left.get() == this->right.get()) {
                return l_result*l_result;
            }

//  If all the elements on the left are zero, return the leftside without
//  revaluating the rightside. Stop this loop early once the first non zero
//  element is encountered.
            if (l_result.is_zero()) {
                return l_result;
            }

            backend::buffer<T> r_result = this->right->evaluate();
            return l_result*r_result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce an multiplcation node.
///
///  @returns A reduced multiplcation node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
            auto l = constant_cast(this->left);
            auto r = constant_cast(this->right);

            if (l.get() && l->is(1)) {
                return this->right;
            } else if (l.get() && l->is(0)) {
                return this->left;
            } else if (r.get() && r->is(1)) {
                return this->left;
            } else if (r.get() && r->is(0)) {
                return this->right;
            } else if (l.get() && r.get()) {
                return constant<T, SAFE_MATH> (this->evaluate());
            }

            auto pl1 = piecewise_1D_cast(this->left);
            auto pr1 = piecewise_1D_cast(this->right);

            if (pl1.get() && (r.get() || pl1->is_arg_match(this->right))) {
                return piecewise_1D(this->evaluate(), pl1->get_arg());
            } else if (pr1.get() && (l.get() || pr1->is_arg_match(this->left))) {
                return piecewise_1D(this->evaluate(), pr1->get_arg());
            }

            auto pl2 = piecewise_2D_cast(this->left);
            auto pr2 = piecewise_2D_cast(this->right);

            if (pl2.get() && (r.get() || pl2->is_arg_match(this->right))) {
                return piecewise_2D(this->evaluate(),
                                    pl2->get_num_columns(),
                                    pl2->get_left(),
                                    pl2->get_right());
            } else if (pr2.get() && (l.get() || pr2->is_arg_match(this->left))) {
                return piecewise_2D(this->evaluate(),
                                    pr2->get_num_columns(),
                                    pr2->get_left(),
                                    pr2->get_right());
            }

//  Combine 2D and 1D piecewise constants if a row or column matches.
            if (pr2.get() && pr2->is_row_match(this->left)) {
                backend::buffer<T> result = pl1->evaluate();
                result.multiply_row(pr2->evaluate());
                return piecewise_2D(result,
                                    pr2->get_num_columns(),
                                    pr2->get_left(),
                                    pr2->get_right());
            } else if (pr2.get() && pr2->is_col_match(this->left)) {
                backend::buffer<T> result = pl1->evaluate();
                result.multiply_col(pr2->evaluate());
                return piecewise_2D(result,
                                    pr2->get_num_columns(),
                                    pr2->get_left(),
                                    pr2->get_right());
            } else if (pl2.get() && pl2->is_row_match(this->right)) {
                backend::buffer<T> result = pl2->evaluate();
                result.multiply_row(pr1->evaluate());
                return piecewise_2D(result,
                                    pl2->get_num_columns(),
                                    pl2->get_left(),
                                    pl2->get_right());
            } else if (pl2.get() && pl2->is_col_match(this->right)) {
                backend::buffer<T> result = pl2->evaluate();
                result.multiply_col(pr1->evaluate());
                return piecewise_2D(result,
                                    pl2->get_num_columns(),
                                    pl2->get_left(),
                                    pl2->get_right());
            }

//  Move constants to the left.
            if (is_constant_promotable(this->right, this->left)) {
                return this->right*this->left;
            }

//  Disable if the right is power like to avoid infinite loop.
            if (is_variable_promotable(this->left, this->right)) {
                return this->right*this->left;
            }

//  Move trig to the right.
            auto cl = graph::cos_cast(this->left);
            auto sl = graph::sin_cast(this->left);
            if ((cl.get() && !this->right->is_power_like() &&
                 !this->right->is_all_variables() &&
                 !sin_cast(this->right).get()) ||
                (sl.get() && !this->right->is_power_like() &&
                 !this->right->is_all_variables()) ||
                (sl.get() && cos_cast(this->right).get())) {
                return this->right*this->left;
            }

//  Reduce x*x to x^2
            if (this->left->is_match(this->right)) {
                return pow(this->left, 2.0);
            }

//  Gather common terms.
            auto lm = multiply_cast(this->left);
            if (lm.get()) {
//  Promote constants before variables.
//  (c*v1)*v2 -> c*(v1*v2)
                if (is_constant_promotable(lm->get_left(),
                                           lm->get_right())) {
                    return lm->get_left()*(lm->get_right()*this->right);
                }

//  (a^c*b)*a^d -> a^(c+d)*b
//  (b*a^c)*a^d -> a^(c+d)*b
                if (is_variable_combineable(this->right, lm->get_left())) {
                    return (this->right*lm->get_left())*lm->get_right();
                } else if (is_variable_combineable(this->right, lm->get_right())) {
                    return (this->right*lm->get_right())*lm->get_left();
                }

//  Assume variables, sqrt of variables, and powers of variables are on the
//  right.
//  (a*v)*b -> (a*b)*v
                if (is_variable_promotable(lm->get_right(), this->right)) {
                    return (lm->get_left()*this->right)*lm->get_right();
                }
                
//  (a*(b*c)^e)*c^f -> a*b^e*c^(e+f)
                auto lmrp = pow_cast(lm->get_right());
                if (lmrp.get()) {
                    auto lmrplm = multiply_cast(lmrp->get_left());
                    if (lmrplm.get() &&
                        is_variable_combineable(lmrplm->get_right(),
                                                this->right)) {
                        return (lm->get_left()*pow(lmrplm->get_left(),
                                                   lmrp->get_right()))*pow(this->right->get_power_base(),
                                                                           lmrp->get_right() +
                                                                           this->right->get_power_exponent());
                    }
                }
            }

            auto rm = multiply_cast(this->right);
            if (rm.get()) {
//  Assume constants are on the left.
//  c1*(c2*v) -> c3*v
                if (is_constant_combineable(this->left,
                                            rm->get_left())) {
                    auto temp = this->left*rm->get_left();
                    if (temp->is_normal()) {
                        return temp*rm->get_right();
                    }
                }

//  a*(a*b) -> a^2*b
//  a*(b*a) -> a^2*b
                if (is_variable_combineable(this->left, rm->get_left())) {
                    return (this->left*rm->get_left())*rm->get_right();
                } else if (is_variable_combineable(this->left, rm->get_right())) {
                    return (this->left*rm->get_right())*rm->get_left();
                }
                
//  Assume variables are on the left.
//  a*(b*v) -> (a*b)*v
                if (is_variable_promotable(rm->get_right(), this->left)) {
                    return (this->left*rm->get_left())*rm->get_right();
                }
            }

//  v1*(c*v2) -> c*(v1*v2)
            if (rm.get() &&
                is_constant_promotable(rm->get_left(), this->left)) {
                return rm->get_left()*(this->left*rm->get_right());
            }

// Assume trig on the right.
//  a*(b*sin) -> (a*b)*sin
//  a*(b*cos) -> (a*b)*cos
//  (a*sin)*b -> (a*b)*sin
//  (a*cos)*b -> (a*b)*cos
            if (lm.get() && 
                (sin_cast(lm->get_right()).get() ||
                 cos_cast(lm->get_right()).get()) &&
                !sin_cast(this->right).get() &&
                !this->right->is_power_like()) {
                return (lm->get_left()*this->right)*lm->get_right();
            } else if (rm.get() && 
                       (sin_cast(rm->get_right()).get() ||
                        cos_cast(rm->get_right()).get()) &&
                       !this->left->is_constant()) {
                return (this->left*rm->get_left())*rm->get_right();
            }

//  Factor out common constants c*b*c*d -> c*c*b*d. c*c will get reduced to c on
//  the second pass.
            if (lm.get() && rm.get()) {
                if (is_constant_combineable(lm->get_left(),
                                            rm->get_left())) {
                    auto temp = lm->get_left()*rm->get_left();
                    if (temp->is_normal()) {
                        return temp*(lm->get_right()*rm->get_right());
                    }
                } else if (is_constant_combineable(lm->get_left(),
                                                   rm->get_right())) {
                    auto temp = lm->get_left()*rm->get_right();
                    if (temp->is_normal()) {
                        return temp*(lm->get_right()*rm->get_left());
                    }
                } else if (is_constant_combineable(lm->get_right(),
                                                   rm->get_left())) {
                    auto temp = lm->get_right()*rm->get_left();
                    if (temp->is_normal()) {
                        return temp*(lm->get_left()*rm->get_right());
                    }
                } else if (is_constant_combineable(lm->get_right(),
                                                   rm->get_right())) {
                    auto temp = lm->get_right()*rm->get_right();
                    if (temp->is_normal()) {
                        return temp*(lm->get_left()*rm->get_left());
                    }
                }

//  Gather common terms. This will help reduce sqrt(a)*sqrt(a).
                if (lm->get_left()->is_match(rm->get_left())) {
                    return (lm->get_left()*rm->get_left()) *
                           (lm->get_right()*rm->get_right());
                } else if (lm->get_right()->is_match(rm->get_left())) {
                    return (lm->get_right()*rm->get_left()) *
                           (lm->get_left()*rm->get_right());
                } else if (lm->get_left()->is_match(rm->get_right())) {
                    return (lm->get_left()*rm->get_right()) *
                           (lm->get_right()*rm->get_left());
                } else if (lm->get_right()->is_match(rm->get_right())) {
                    return (lm->get_right()*rm->get_right()) *
                           (lm->get_left()*rm->get_left());
                }
            }

//  Common factor reduction. (a/b)*(c/a) = c/b.
            auto ld = divide_cast(this->left);
            auto rd = divide_cast(this->right);

//  a*(b/c) -> (a*b)/c
//  (a/c)*b -> (a*b)/c
            if (rd.get()) {
                return (this->left*rd->get_left())/rd->get_right();
            } else if (ld.get()) {
                return (ld->get_left()*this->right)/ld->get_right();
            }

//  (a/b)*(c/a) -> c/b
//  (b/a)*(a/c) -> c/b
            if (ld.get() && rd.get()) {
                if (ld->get_left()->is_match(rd->get_right())) {
                    return rd->get_left()/ld->get_right();
                } else if (ld->get_right()->is_match(rd->get_left())) {
                    return ld->get_left()/rd->get_right();
                }

//  Convert (a/b)*(c/d) -> (a*c)/(b*d). This should help reduce cases like.
//  (a/b)*(a/b) + (c/b)*(c/b).
                return (ld->get_left()*rd->get_left()) /
                       (ld->get_right()*rd->get_right());
            }

//  Power reductions.
            if (is_variable_combineable(this->left, this->right)) {
                return pow(this->left->get_power_base(),
                           this->left->get_power_exponent() +
                           this->right->get_power_exponent());
            }

//  a*b^-c -> a/b^c
            auto rp = pow_cast(this->right);
            if (rp.get()) {
                auto exponent = constant_cast(rp->get_right());
                if (exponent.get() && exponent->evaluate().is_negative()) {
                    return this->left/pow(rp->get_left(), -rp->get_right());
                }
            }
//  b^-c*a -> a/b^c
            auto lp = pow_cast(this->left);
            if (lp.get()) {
                auto exponent = constant_cast(lp->get_right());
                if (exponent.get() && exponent->evaluate().is_negative()) {
                    return this->right/pow(lp->get_left(), -lp->get_right());
                }
            }

//  (b*a)^c*a^d -> b^c*a^(c + d)
//  (a*b)^c*a^d -> b^c*a^(c + d)
//  a^d*(b*a)^c -> b^c*a^(c + d)
//  a^d*(a*b)^c -> b^c*a^(c + d)
            if (lp.get() && rp.get()) {
                auto lplm = multiply_cast(lp->get_left());
                auto rplm = multiply_cast(rp->get_left());
                if (lplm.get()) {
                    if (is_variable_combineable(lplm->get_right(),
                                                this->right)) {
                        return pow(lplm->get_left()->get_power_base(),
                                   this->left->get_power_exponent())*
                               pow(this->right->get_power_base(),
                                   this->left->get_power_exponent() +
                                   this->right->get_power_exponent());
                    } else if (is_variable_combineable(lplm->get_left(),
                                                       this->right)) {
                        return pow(lplm->get_right()->get_power_base(),
                                   this->left->get_power_exponent())*
                               pow(this->right->get_power_base(),
                                   this->left->get_power_exponent() +
                                   this->right->get_power_exponent());
                    }
                }

                if (rplm.get()) {
                    if (is_variable_combineable(rplm->get_right(),
                                                this->left)) {
                        return pow(rplm->get_left()->get_power_base(),
                                   this->right->get_power_exponent())*
                               pow(this->left->get_power_base(),
                                   this->left->get_power_exponent() +
                                   this->right->get_power_exponent());
                    } else if (is_variable_combineable(rplm->get_left(),
                                                       this->left)) {
                        return pow(rplm->get_right()->get_power_base(),
                                   this->right->get_power_exponent())*
                               pow(this->left->get_power_base(),
                                   this->left->get_power_exponent() +
                                   this->right->get_power_exponent());
                    }
                }
            }

            auto lpd = divide_cast(this->left->get_power_base());
            if (lpd.get()) {
//  (a/b)^c*b^d -> a^c*b^(c-d)
                if (is_variable_combineable(lpd->get_right(),
                                            this->right)) {
                    return pow(lpd->get_left(), this->left->get_power_exponent()) *
                           pow(this->right->get_power_base(),
                               this->right->get_power_exponent() -
                               this->left->get_power_exponent()*lpd->get_right()->get_power_exponent());
                }
//  (b/a)^c*b^d -> b^(c+d)/a^c
                if (is_variable_combineable(lpd->get_left(), this->right)) {
                    return pow(this->right->get_power_base(),
                               this->right->get_power_exponent() +
                               this->left->get_power_exponent()*lpd->get_left()->get_power_exponent()) /
                           pow(lpd->get_right(), this->left->get_power_exponent());
                }
            }
            auto rpd = divide_cast(this->right->get_power_base());
            if (rpd.get()) {
//  b^d*(a/b)^c -> a^c*b^(c-d)
                if (is_variable_combineable(rpd->get_right(),
                                            this->left)) {
                    return pow(rpd->get_left(), this->right->get_power_exponent()) *
                           pow(this->left->get_power_base(),
                               this->left->get_power_exponent() -
                               this->right->get_power_exponent()*rpd->get_right()->get_power_exponent());
                }
//  b^d*(b/a)^c -> b^(c+d)/a^c
                if (is_variable_combineable(rpd->get_left(),
                                            this->left)) {
                    return pow(this->right->get_power_base(),
                               this->right->get_power_exponent() +
                               this->right->get_power_exponent()*rpd->get_left()->get_power_exponent()) /
                           pow(rpd->get_right(), this->right->get_power_exponent());
                }
            }

//  exp(a)*exp(b) -> exp(a + b)
            auto le = exp_cast(this->left);
            auto re = exp_cast(this->right);
            if (le.get() && re.get()) {
                return exp(le->get_arg() + re->get_arg());
            }

//  exp(a)*(exp(b)*c) -> c*(exp(a)*exp(b))
//  exp(a)*(c*exp(b)) -> c*(exp(a)*exp(b))
            if (le.get() && rm.get()) {
                auto rmle = exp_cast(rm->get_left());
                if (rmle.get()) {
                    return rm->get_right()*(this->left*rm->get_left());
                }
                auto rmre = exp_cast(rm->get_right());
                if (rmre.get()) {
                    return rm->get_left()*(this->left*rm->get_right());
                }
            }
//  (exp(a)*c)*exp(b) -> c*(exp(a)*exp(b))
//  (c*exp(a))*exp(b) -> c*(exp(a)*exp(b))
            if (re.get() && lm.get()) {
                auto lmle = exp_cast(lm->get_left());
                if (lmle.get()) {
                    return lm->get_right()*(this->right*lm->get_left());
                }
                auto lmre = exp_cast(lm->get_right());
                if (lmre.get()) {
                    return lm->get_left()*(this->right*lm->get_right());
                }
            }
//  (exp(a)*c)*(exp(b)*d) -> (c*d)*(exp(a)*exp(b))
//  (exp(a)*c)*(d*exp(b)) -> (c*d)*(exp(a)*exp(b))
//  (c*exp(a))*(exp(b)*d) -> (c*d)*(exp(a)*exp(b))
//  (c*exp(a))*(d*exp(b)) -> (c*d)*(exp(a)*exp(b))
            if (lm.get() && rm.get()) {
                auto lmle = exp_cast(lm->get_left());
                if (lmle.get()) {
                    auto rmle = exp_cast(rm->get_left());
                    if (rmle.get()) {
                        return (lm->get_right()*rm->get_right()) *
                               (lm->get_left()*rm->get_left());
                    }
                    auto rmre = exp_cast(rm->get_right());
                    if (rmre.get()) {
                        return (lm->get_right()*rm->get_left()) *
                               (lm->get_left()*rm->get_right());
                    }
                }
                auto lmre = exp_cast(lm->get_right());
                if (lmre.get()) {
                    auto rmle = exp_cast(rm->get_left());
                    if (rmle.get()) {
                        return (lm->get_left()*rm->get_right()) *
                               (lm->get_right()*rm->get_left());
                    }
                    auto rmre = exp_cast(rm->get_right());
                    if (rmre.get()) {
                        return (lm->get_left()*rm->get_left()) *
                               (lm->get_right()*rm->get_right());
                    }
                }
            }

            if (ld.get() && re.get()) {
//  (c/exp(a))*exp(b) -> c*(exp(b)/exp(a))
                auto ldre = exp_cast(ld->get_right());
                if (ldre.get()) {
                    return ld->get_left()*(this->right/ld->get_right());
                }
//  (exp(a)/c)*exp(b) -> (exp(a)*exp(b))/c
                auto ldle = exp_cast(ld->get_left());
                if (ldle.get()) {
                    return (ld->get_left()*this->right)/ld->get_right();
                }
            }
            if (rd.get() && le.get()) {
//  exp(a)*(c/exp(a)) -> c*(exp(a)/exp(b))
                auto rdre = exp_cast(rd->get_right());
                if (rdre.get()) {
                    return rd->get_left()*(this->left/rd->get_right());
                }
//  exp(a)*(exp(b)/c) -> (exp(a)*exp(b))/c
                auto rdle = exp_cast(rd->get_left());
                if (rdle.get()) {
                    return (this->left*rd->get_left())/rd->get_right();
                }
            }

            if (ld.get() && rm.get()) {
                auto rmle = exp_cast(rm->get_left());
                if (rmle.get()) {
//  (c/exp(a))*(exp(b)*d) -> (c*d)*(exp(b)/exp(a))
                    auto ldre = exp_cast(ld->get_right());
                    if (ldre.get()) {
                        return (ld->get_left()*rm->get_right()) *
                               (rm->get_left()/ld->get_right());
                    }
//  (exp(a)/c)*(exp(b)*d) -> (d/c)*(exp(a)*exp(b))
                    auto ldle = exp_cast(ld->get_left());
                    if (ldle.get()) {
                        return (rm->get_right()/ld->get_right()) *
                               (ld->get_left()*rm->get_left());
                    }
                }
                auto rmre = exp_cast(rm->get_right());
                if (rmre.get()) {
//  (c/exp(a))*(d*exp(b)) -> (c*d)*(exp(b)/exp(a))
                    auto ldre = exp_cast(ld->get_right());
                    if (ldre.get()) {
                        return (ld->get_left()*rm->get_left()) *
                               (rm->get_right()/ld->get_right());
                    }
//  (exp(a)/c)*(d*exp(b)) -> (d/c)*(exp(a)*exp(b))
                    auto ldle = exp_cast(ld->get_left());
                    if (ldle.get()) {
                        return (rm->get_left()/ld->get_right()) *
                               (ld->get_left()*rm->get_right());
                    }
                }
            } else if (rd.get() && lm.get()) {
                auto lmre = exp_cast(lm->get_right());
                if (lmre.get()) {
//  (c*exp(a))*(exp(b)/d) -> (c/d)*(exp(a)*exp(b))
                    auto rdre = exp_cast(rd->get_left());
                    if (rdre.get()) {
                        return (lm->get_left()/rd->get_right()) *
                               (lm->get_right()*rd->get_left());
                    }
//  (c*exp(a))*(d/exp(b)) -> (c*d)*(exp(a)/exp(b))
                    auto rdle = exp_cast(rd->get_right());
                    if (rdle.get()) {
                        return (lm->get_left()*rd->get_left()) *
                               (lm->get_right()/rd->get_right());
                    }
                }
                auto lmle = exp_cast(lm->get_left());
                if (lmle.get()) {
//  (exp(a)*c)*(d/exp(b)) -> (c*d)*(exp(a)/exp(b))
                    auto rdle = exp_cast(rd->get_right());
                    if (rdle.get()) {
                        return (lm->get_right()*rd->get_left()) *
                               (lm->get_left()/rd->get_right());
                    }
//  (exp(a)*c)*(exp(b)/d) -> (c/d)*(exp(a)*exp(b))
                    auto rdre = exp_cast(rd->get_left());
                    if (rdre.get()) {
                        return (lm->get_right()/rd->get_right()) *
                               (lm->get_left()*rd->get_left());
                    }
                }
            }

//  Cases like
//  (c/exp(a))*(exp(b)/d) -> (c/d)*(exp(b)/exp(a))
//  (c/exp(a))*(d/exp(b)) -> (c*e)/(exp(b)*exp(a))
//  (exp(a)/c)*(d/exp(b)) -> (d/c)*(exp(a)/exp(b))
//  (exp(a)/c)*(exp(b)/d) -> (exp(a)*exp(b))/(c*d)
//  Are taken care of by (a/b)*(c/d) -> (a*c)/(b*d) conversion above.

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d mul(a,b)/dx = da/dx*b + a*db/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> df(shared_leaf<T, SAFE_MATH> x) {
            if (this->is_match(x)) {
                return one<T, SAFE_MATH> ();
            }

            const size_t hash = reinterpret_cast<size_t> (x.get());
            if (this->df_cache.find(hash) == this->df_cache.end()) {
                this->df_cache[hash] = this->left->df(x)*this->right 
                                     + this->left*this->right->df(x);
            }
            return this->df_cache[hash];
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @param[in]     usage     List of register usage count.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream,
                                                                  registers,
                                                                  usage);
                shared_leaf<T, SAFE_MATH> r = this->right->compile(stream,
                                                                   registers,
                                                                   usage);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = ";
                if constexpr (SAFE_MATH) {
                    stream << "(" << registers[l.get()] << " == ";
                    if constexpr (jit::is_complex<T> ()) {
                        jit::add_type<T> (stream);
                        stream << "(0, 0)";
                    } else {
                        stream << "0";
                    }
                    stream << " || " << registers[r.get()] << " == ";
                    if constexpr (jit::is_complex<T> ()) {
                        jit::add_type<T> (stream);
                        stream << "(0, 0)";
                    } else {
                        stream << "0";
                    }
                    stream << ") ? ";
                    if constexpr (jit::is_complex<T> ()) {
                        jit::add_type<T> (stream);
                        stream << "(0, 0)";
                    } else {
                        stream << "0";
                    }
                    stream << " : ";
                }
                stream << registers[l.get()] << "*"
                       << registers[r.get()];
                this->endline(stream, usage);
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

            auto x_cast = multiply_cast(x);
            if (x_cast.get()) {
//  Multiplication is commutative.
                if ((this->left->is_match(x_cast->get_left()) &&
                     this->right->is_match(x_cast->get_right())) ||
                    (this->right->is_match(x_cast->get_left()) &&
                     this->left->is_match(x_cast->get_right()))) {
                    return true;
                }
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            if (constant_cast(this->left).get() ||
                add_cast(this->left).get()      ||
                subtract_cast(this->left).get()) {
                std::cout << "\\left(";
                this->left->to_latex();
                std::cout << "\\right)";
            } else {
                this->left->to_latex();
            }
            std::cout << " ";
            if (constant_cast(this->right).get() ||
                add_cast(this->right).get()      ||
                subtract_cast(this->right).get()) {
                std::cout << "\\left(";
                this->right->to_latex();
                std::cout << "\\right)";
            } else {
                this->right->to_latex();
            }
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> remove_pseudo() {
            if (this->has_pseudo()) {
                return this->left->remove_pseudo() *
                       this->right->remove_pseudo();
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
        virtual shared_leaf<T, SAFE_MATH> to_vizgraph(std::stringstream &stream,
                                                      jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                const std::string name = jit::to_string('r', this);
                registers[this] = name;
                stream << "    " << name
                       << " [label = \"⨉\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto l = this->left->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[l.get()] << ";" << std::endl;
                auto r = this->right->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[r.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build multiply node from two leaves.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> multiply(shared_leaf<T, SAFE_MATH> l,
                                       shared_leaf<T, SAFE_MATH> r) {
        auto temp = std::make_shared<multiply_node<T, SAFE_MATH>> (l, r)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::cache.find(i) ==
                leaf_node<T, SAFE_MATH>::cache.end()) {
                leaf_node<T, SAFE_MATH>::cache[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::cache[i])) {
                return leaf_node<T, SAFE_MATH>::cache[i];
            }
        }
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

//------------------------------------------------------------------------------
///  @brief Build multiply node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator*(shared_leaf<T, SAFE_MATH> l,
                                        shared_leaf<T, SAFE_MATH> r) {
        return multiply<T, SAFE_MATH> (l, r);
    }

//------------------------------------------------------------------------------
///  @brief Build multiply node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam L         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar L, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator*(const L l,
                                        shared_leaf<T, SAFE_MATH> r) {
        return multiply<T, SAFE_MATH> (constant<T, SAFE_MATH> (static_cast<T> (l)), r);
    }

//------------------------------------------------------------------------------
///  @brief Build multiply node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam R         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar R, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator*(shared_leaf<T, SAFE_MATH> l,
                                        const R r) {
        return multiply<T, SAFE_MATH> (l, constant<T, SAFE_MATH> (static_cast<T> (r)));
    }

///  Convenience type alias for shared multiply nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_multiply = std::shared_ptr<multiply_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a multiply node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_multiply<T, SAFE_MATH> multiply_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<multiply_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Divide node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A division node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class divide_node final : public branch_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] l Left node pointer.
///  @param[in] r Right node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *l,
                                     leaf_node<T, SAFE_MATH> *r) {
            return jit::format_to_string(reinterpret_cast<size_t> (l)) + "/" +
                   jit::format_to_string(reinterpret_cast<size_t> (r));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct an addition node.
///
///  @param[in] n Numerator branch.
///  @param[in] d Denominator branch.
//------------------------------------------------------------------------------
        divide_node(shared_leaf<T, SAFE_MATH> n,
                    shared_leaf<T, SAFE_MATH> d) :
        branch_node<T, SAFE_MATH> (n, d, divide_node::to_string(n.get(),
                                                                d.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of division.
///
///  result = n/d
///
///  @returns The value of n/d.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> l_result = this->left->evaluate();

//  If all the elements on the left are zero, return the leftside without
//  revaluating the rightside. Stop this loop early once the first non zero
//  element is encountered.
            if (l_result.is_zero()) {
                return l_result;
            }

            backend::buffer<T> r_result = this->right->evaluate();
            return l_result/r_result;
        }

//------------------------------------------------------------------------------
///  @brief Reduce an division node.
///
///  @returns A reduced division node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
//  Constant Reductions.
            auto l = constant_cast(this->left);
            auto r = constant_cast(this->right);

            if ((l.get() && l->is(0)) ||
                (r.get() && r->is(1))) {
                return this->left;
            } else if (l.get() && r.get()) {
                return constant<T, SAFE_MATH> (this->evaluate());
            }

            auto pl1 = piecewise_1D_cast(this->left);
            auto pr1 = piecewise_1D_cast(this->right);

            if (pl1.get() && (r.get() || pl1->is_arg_match(this->right))) {
                return piecewise_1D(this->evaluate(), pl1->get_arg());
            } else if (pr1.get() && (l.get() || pr1->is_arg_match(this->left))) {
                return piecewise_1D(this->evaluate(), pr1->get_arg());
            }

            auto pl2 = piecewise_2D_cast(this->left);
            auto pr2 = piecewise_2D_cast(this->right);

            if (pl2.get() && (r.get() || pl2->is_arg_match(this->right))) {
                return piecewise_2D(this->evaluate(),
                                    pl2->get_num_columns(),
                                    pl2->get_left(),
                                    pl2->get_right());
            } else if (pr2.get() && (l.get() || pr2->is_arg_match(this->left))) {
                return piecewise_2D(this->evaluate(),
                                    pr2->get_num_columns(),
                                    pr2->get_left(),
                                    pr2->get_right());
            }

//  Combine 2D and 1D piecewise constants if a row or column matches.
            if (pr2.get() && pr2->is_row_match(this->left)) {
                backend::buffer<T> result = pl1->evaluate();
                result.divide_row(pr2->evaluate());
                return piecewise_2D(result,
                                    pr2->get_num_columns(),
                                    pr2->get_left(),
                                    pr2->get_right());
            } else if (pr2.get() && pr2->is_col_match(this->left)) {
                backend::buffer<T> result = pl1->evaluate();
                result.divide_col(pr2->evaluate());
                return piecewise_2D(result,
                                    pr2->get_num_columns(),
                                    pr2->get_left(),
                                    pr2->get_right());
            } else if (pl2.get() && pl2->is_row_match(this->right)) {
                backend::buffer<T> result = pl2->evaluate();
                result.divide_row(pr1->evaluate());
                return piecewise_2D(result,
                                    pl2->get_num_columns(),
                                    pl2->get_left(),
                                    pl2->get_right());
            } else if (pl2.get() && pl2->is_col_match(this->right)) {
                backend::buffer<T> result = pl2->evaluate();
                result.divide_col(pr1->evaluate());
                return piecewise_2D(result,
                                    pl2->get_num_columns(),
                                    pl2->get_left(),
                                    pl2->get_right());
            }

            if (this->left->is_match(this->right)) {
                return one<T, SAFE_MATH> ();
            }

//  Reduce cases of a/c1 -> c2*a
            if (this->right->is_constant()) {
                return (1.0/this->right)*this->left;
            }

//  a/(b/c + d) -> a*c/(c*d + b)
//  a/(d + b/c) -> a*c/(c*d + b)
            auto ra = add_cast(this->right);
            if (ra.get()) {
                auto rald = divide_cast(ra->get_left());
                auto rard = divide_cast(ra->get_right());
                if (rald.get()) {
                    return this->left*rald->get_right() /
                           fma(rald->get_right(),
                               ra->get_right(),
                               rald->get_left());
                } else if (rard.get()) {
                    return this->left*rard->get_right() /
                           fma(rard->get_right(),
                               ra->get_left(),
                               rard->get_left());
                }
            }

//  a/(b/c - d) -> a*c/(b - c*d)
//  a/(d - b/c) -> a*c/(c*d - b)
            auto rs = subtract_cast(this->right);
            if (rs.get()) {
                auto rsld = divide_cast(rs->get_left());
                auto rsrd = divide_cast(rs->get_right());
                if (rsld.get()) {
                    return this->left*rsld->get_right() /
                           (rsld->get_left() -
                            rsld->get_right()*rs->get_right());
                } else if (rsrd.get()) {
                    return this->left*rsrd->get_right() /
                           (rsrd->get_right()*rs->get_left() -
                            rsrd->get_left());
                }
            }

//  fma(a,d,c*d)/d -> a + c
//  fma(a,d,d*c)/d -> a + c
//  fma(d,a,c*d)/d -> a + c
//  fma(d,a,d*c)/d -> a + c
            auto lfma = fma_cast(this->left);
            if (lfma.get()) {
                auto fmarm = multiply_cast(lfma->get_right());
                if (fmarm.get()) {
                    if (lfma->get_middle()->is_match(this->right) &&
                        fmarm->get_right()->is_match(this->right)) {
                        return lfma->get_left() + fmarm->get_left();
                    } else if (lfma->get_middle()->is_match(this->right) &&
                               fmarm->get_left()->is_match(this->right)) {
                        return lfma->get_left() + fmarm->get_right();
                    } else if (lfma->get_left()->is_match(this->right) &&
                               fmarm->get_right()->is_match(this->right)) {
                        return lfma->get_middle() + fmarm->get_left();
                    } else if (lfma->get_left()->is_match(this->right) &&
                               fmarm->get_left()->is_match(this->right)) {
                        return lfma->get_middle() + fmarm->get_right();
                    }
                }
            }

//  Common factor reduction. (a*b)/(a*c) = b/c.
            auto lm = multiply_cast(this->left);
            auto rm = multiply_cast(this->right);

            if (lm.get() && rm.get()) {
                if (is_variable_combineable(lm->get_left(),
                                            rm->get_left()) ||
                    is_variable_combineable(lm->get_right(),
                                            rm->get_right())) {
                    return (lm->get_left()/rm->get_left()) *
                           (lm->get_right()/rm->get_right());
                } else if (is_variable_combineable(lm->get_left(),
                                                   rm->get_right()) ||
                           is_variable_combineable(lm->get_right(),
                                                   rm->get_left())) {
                    return (lm->get_left()/rm->get_right()) *
                           (lm->get_right()/rm->get_left());
                }
            }

//  Move constants to the numerator.
//  a/(c1*b) -> (c2*a)/b
//  a/(b*c1) -> (c2*a)/b
            if (rm.get()) {
                if (rm->get_left()->is_constant() &&
                    rm->get_left()->is_normal()) {
                    return ((1.0/rm->get_left())*this->left)/rm->get_right();
                } else if (rm->get_right()->is_constant() &&
                           rm->get_right()->is_normal()) {
                    return ((1.0/rm->get_right())*this->left)/rm->get_left();
                }

//  a/((b/c + d)*e) -> a*c/((c*d + b)*e)
//  a/((d + b/c)*e) -> a*c/((c*d + b)*e)
//  a/(e*(b/c + d)) -> a*c/((c*d + b)*e)
//  a/(e*(d + b/c)) -> a*c/((c*d + b)*e)
                auto rmla = add_cast(rm->get_left());
                auto rmra = add_cast(rm->get_right());
                if (rmla.get()) {
                    auto rmlald = divide_cast(rmla->get_left());
                    auto rmlard = divide_cast(rmla->get_right());
                    if (rmlald.get()) {
                        return this->left*rmlald->get_right() /
                               (fma(rmlald->get_right(),
                                    rmla->get_right(),
                                    rmlald->get_left())*rm->get_right());
                    } else if (rmlard.get()) {
                        return this->left*rmlard->get_right() /
                               (fma(rmlard->get_right(),
                                    rmla->get_left(),
                                    rmlard->get_left())*rm->get_right());
                    }
                }
                if (rmra.get()) {
                    auto rmrald = divide_cast(rmra->get_left());
                    auto rmrard = divide_cast(rmra->get_right());
                    if (rmrald.get()) {
                        return this->left*rmrald->get_right() /
                               (fma(rmrald->get_right(),
                                    rmra->get_right(),
                                    rmrald->get_left())*rm->get_left());
                    } else if (rmrard.get()) {
                        return this->left*rmrard->get_right() /
                               (fma(rmrard->get_right(),
                                    rmra->get_left(),
                                    rmrard->get_left())*rm->get_left());
                    }
                }

//  a/((b/c - d)*e) -> a*c/((b - c*d)*e)
//  a/(e*(b/c - d)) -> a*c/((b - c*d)*e)
//  a/((d - b/c)*e) -> a*c/((c*d - b)*e)
//  a/(e*(d - b/c)) -> a*c/((c*d - b)*e)
                auto rmls = subtract_cast(rm->get_left());
                auto rmrs = subtract_cast(rm->get_right());
                if (rmls.get()) {
                    auto rmlsld = divide_cast(rmls->get_left());
                    auto rmlsrd = divide_cast(rmls->get_right());
                    if (rmlsld.get()) {
                        return this->left*rmlsld->get_right() /
                               ((rmlsld->get_left() -
                                 rmlsld->get_right()*rmls->get_right())*rm->get_right());
                    } else if (rmlsrd.get()) {
                        return this->left*rmlsrd->get_right() /
                               ((rmlsrd->get_right()*rmls->get_left() -
                                 rmlsrd->get_left())*rm->get_right());
                    }
                }
                if (rmrs.get()) {
                    auto rmrsld = divide_cast(rmrs->get_left());
                    auto rmrsrd = divide_cast(rmrs->get_right());
                    if (rmrsld.get()) {
                        return this->left*rmrsld->get_right() /
                               ((rmrsld->get_left() -
                                 rmrsld->get_right()*rmrs->get_right())*rm->get_left());
                    } else if (rmrsrd.get()) {
                        return this->left*rmrsrd->get_right() /
                               ((rmrsrd->get_right()*rmrs->get_left() -
                                 rmrsrd->get_left())*rm->get_left());
                    }
                }
            }

            if (lm.get() && rm.get()) {
//  (a*b)/(a*c) -> b/c
//  (b*a)/(a*c) -> b/c
//  (a*b)/(c*a) -> b/c
//  (b*a)/(c*a) -> b/c
                if (lm->get_left()->is_match(rm->get_left())) {
                    return lm->get_right()/rm->get_right();
                } else if (lm->get_left()->is_match(rm->get_right())) {
                    return lm->get_right()/rm->get_left();
                } else if (lm->get_right()->is_match(rm->get_left())) {
                    return lm->get_left()/rm->get_right();
                } else if (lm->get_right()->is_match(rm->get_right())) {
                    return lm->get_left()/rm->get_left();
                }
            }

            if (lm.get()) {
//  (v1*v2)/v1 -> v2
//  (v2*v1)/v1 -> v2
                if (lm->get_left()->is_match(this->right)) {
                    return lm->get_right();
                } else if (lm->get_right()->is_match(this->right)) {
                    return lm->get_left();
                }

//  (v1^a*v2)/v1^b -> v2*(v1^a/v1^b)
//  (v2*v1^a)/v1^b -> v2*(v1^a/v1^b)
                if (is_variable_combineable(lm->get_left(),
                                            this->right)) {
                    return lm->get_right()*(lm->get_left()/this->right);
                } else if (is_variable_combineable(lm->get_right(),
                                                   this->right)) {
                    return lm->get_left()*(lm->get_right()/this->right);
                }
            }

//  (a/b)/c -> a/(b*c)
            auto ld = divide_cast(this->left);
            if (ld.get()) {
                return ld->get_left()/(ld->get_right()*this->right);
            }

//  Power reductions.
            if (is_variable_combineable(this->left,
                                        this->right)) {
                return pow(this->left->get_power_base(),
                           this->left->get_power_exponent() -
                           this->right->get_power_exponent());
            }

//  a/b^-c -> a*b^c
            auto rp = pow_cast(this->right);
            if (rp.get()) {
                auto exponent = constant_cast(rp->get_right());
                if (exponent.get() && exponent->evaluate().is_negative()) {
                    return this->left*pow(rp->get_left(), -rp->get_right());
                }
            }

//  (a*b)^c/(a^d) = a^(c - d)*b^c
//  (b*a)^c/(a^d) = a^(c - d)*b^c
            auto lp = pow_cast(this->left);
            if (lp.get()) {
                auto lpm = multiply_cast(this->left->get_power_base());
                if (lpm.get()) {
                    if (lpm->get_left()->is_match(this->right->get_power_base())) {
                        return pow(this->right->get_power_base(),
                                   this->left->get_power_exponent() -
                                   this->right->get_power_exponent()) *
                               pow(lpm->get_right(),
                                   this->left->get_power_exponent());
                    } else if (lpm->get_right()->is_match(this->right->get_power_base())) {
                        return pow(this->right->get_power_base(),
                                   this->left->get_power_exponent() -
                                   this->right->get_power_exponent()) *
                               pow(lpm->get_left(),
                                   this->left->get_power_exponent());
                    }
                }
            }
//  (a*b)^c/((a^d)*e) = a^(c - d)*b^c/e
//  (b*a)^c/((a^d)*e) = a^(c - d)*b^c/e
//  (a*b)^c/(e*(a^d)) = a^(c - d)*b^c/e
//  (b*a)^c/(e*(a^d)) = a^(c - d)*b^c/e
            if (lp.get() && rm.get()) {
                auto lpm = multiply_cast(this->left->get_power_base());
                if (lpm.get()) {
                    if (lpm->get_left()->is_match(rm->get_left()->get_power_base())) {
                        return (pow(rm->get_left()->get_power_base(),
                                    this->left->get_power_exponent() -
                                    rm->get_left()->get_power_exponent()) *
                                pow(lpm->get_right(),
                                    this->left->get_power_exponent())) /
                               rm->get_right();
                    } else if (lpm->get_right()->is_match(rm->get_left()->get_power_base())) {
                        return (pow(rm->get_left()->get_power_base(),
                                    this->left->get_power_exponent() -
                                    rm->get_left()->get_power_exponent()) *
                                pow(lpm->get_left(),
                                    this->left->get_power_exponent())) /
                               rm->get_right();
                    } else if (lpm->get_left()->is_match(rm->get_right()->get_power_base())) {
                        return (pow(rm->get_right()->get_power_base(),
                                    this->left->get_power_exponent() -
                                    rm->get_right()->get_power_exponent()) *
                                pow(lpm->get_right(),
                                    this->left->get_power_exponent())) /
                               rm->get_left();
                    } else if (lpm->get_right()->is_match(rm->get_right()->get_power_base())) {
                        return (pow(rm->get_right()->get_power_base(),
                                    this->left->get_power_exponent() -
                                    rm->get_right()->get_power_exponent()) *
                                pow(lpm->get_left(),
                                    this->left->get_power_exponent())) /
                               rm->get_left();
                    }
                }
            }

            if (lm.get()) {
//  a*(b*c)/c -> a*b
//  a*(c*b)/c -> a*b
//  (a*c)*b/c -> a*b
//  (c*a)*b/c -> a*b
                auto lmrm = multiply_cast(lm->get_right());
                auto lmlm = multiply_cast(lm->get_left());
                if (lmrm.get()) {
                    if (is_variable_combineable(lmrm->get_right(),
                                                this->right)) {
                        return lm->get_left()*lmrm->get_left() *
                               (lmrm->get_right()/this->right);
                    } else if (is_variable_combineable(lmrm->get_left(),
                                                       this->right)) {
                        return lm->get_left()*lmrm->get_right() *
                               (lmrm->get_left()/this->right);
                    }
                } else if (lmlm.get()) {
                    if (is_variable_combineable(lmlm->get_right(),
                                                this->right)) {
                        return lm->get_right()*lmlm->get_left() *
                               (lmlm->get_right()/this->right);
                    } else if (is_variable_combineable(lmlm->get_left(),
                                                       this->right)) {
                        return lm->get_right()*lmlm->get_right() *
                               (lmlm->get_left()/this->right);
                    }
                }

//  (f*(a*b)^c)/(a^d) = f*a^(c - d)*b^c
//  (f*(b*a)^c)/(a^d) = f*a^(c - d)*b^c
//  (((a*b)^c)*f)/(a^d) = f*a^(c - d)*b^c
//  (((b*a)^c)*f)/(a^d) = f*a^(c - d)*b^c
                auto lmlp = pow_cast(lm->get_left());
                auto lmrp = pow_cast(lm->get_right());
                if (lmlp.get()) {
                    auto lmlpm = multiply_cast(lmlp->get_power_base());
                    if (lmlpm.get()) {
                        if (lmlpm->get_left()->is_match(this->right->get_power_base())) {
                            return lm->get_right() *
                                   pow(this->right->get_power_base(),
                                       lmlp->get_power_exponent() -
                                       this->right->get_power_exponent()) *
                                   pow(lmlpm->get_right(),
                                       lmlp->get_power_exponent());
                        } else if (lmlpm->get_right()->is_match(this->right->get_power_base())) {
                            return lm->get_right() *
                                   pow(this->right->get_power_base(),
                                       lmlp->get_power_exponent() -
                                       this->right->get_power_exponent()) *
                                   pow(lmlpm->get_left(),
                                       lmlp->get_power_exponent());
                        }
                    }
                } else if (lmrp.get()) {
                    auto lmrpm = multiply_cast(lmrp->get_power_base());
                    if (lmrpm.get()) {
                        if (lmrpm->get_left()->is_match(this->right->get_power_base())) {
                            return lm->get_left() *
                                   pow(this->right->get_power_base(),
                                       lmrp->get_power_exponent() -
                                       this->right->get_power_exponent()) *
                                   pow(lmrpm->get_right(),
                                       lmrp->get_power_exponent());
                        } else if (lmrpm->get_right()->is_match(this->right->get_power_base())) {
                            return lm->get_left() *
                                   pow(this->right->get_power_base(),
                                       lmrp->get_power_exponent() -
                                       this->right->get_power_exponent()) *
                                   pow(lmrpm->get_left(),
                                       lmrp->get_power_exponent());
                        }
                    }
                }
            }

//  f*(a*b)^c/((a^d)*e) = a^(c - d)*b^c/e
//  f*(b*a)^c/((a^d)*e) = a^(c - d)*b^c/e
//  f*(a*b)^c/(e*(a^d)) = a^(c - d)*b^c/e
//  f*(b*a)^c/(e*(a^d)) = a^(c - d)*b^c/e
//  (a*b)^c*f/((a^d)*e) = a^(c - d)*b^c/e
//  (b*a)^c*f/((a^d)*e) = a^(c - d)*b^c/e
//  (a*b)^c*f/(e*(a^d)) = a^(c - d)*b^c/e
//  (b*a)^c*f/(e*(a^d)) = a^(c - d)*b^c/e
            if (lm.get() && rm.get()) {
                auto lmlp = pow_cast(lm->get_left());
                auto lmrp = pow_cast(lm->get_right());
                if (lmlp.get()) {
                    auto lmlpm = multiply_cast(lmlp->get_power_base());
                    if (lmlpm.get()) {
                        if (lmlpm->get_left()->is_match(rm->get_left()->get_power_base())) {
                            return lm->get_right() *
                                   (pow(rm->get_left()->get_power_base(),
                                        lmlp->get_power_exponent() -
                                        rm->get_left()->get_power_exponent())) *
                                   pow(lmlpm->get_right(),
                                       lmlp->get_power_exponent()) /
                                   rm->get_right();
                        } else if (lmlpm->get_right()->is_match(rm->get_left()->get_power_base())) {
                            return lm->get_right() *
                                   (pow(rm->get_left()->get_power_base(),
                                        lmlp->get_power_exponent() -
                                        rm->get_left()->get_power_exponent())) *
                                   pow(lmlpm->get_left(),
                                       lmlp->get_power_exponent()) /
                                   rm->get_right();
                        } else if (lmlpm->get_left()->is_match(rm->get_right()->get_power_base())) {
                            return lm->get_right() *
                                   (pow(rm->get_left()->get_power_base(),
                                        lmlp->get_power_exponent() -
                                        rm->get_right()->get_power_exponent())) *
                                   pow(lmlpm->get_right(),
                                       lmlp->get_power_exponent()) /
                                   rm->get_left();
                        } else if (lmlpm->get_right()->is_match(rm->get_right()->get_power_base())) {
                            return lm->get_right() *
                                   (pow(rm->get_left()->get_power_base(),
                                        lmlp->get_power_exponent() -
                                        rm->get_right()->get_power_exponent())) *
                                   pow(lmlpm->get_left(),
                                       lmlp->get_power_exponent()) /
                                   rm->get_left();
                        }
                    }
                } else if (lmrp.get()) {
                    auto lmrpm = multiply_cast(lmrp->get_power_base());
                    if (lmrpm.get()) {
                        if (lmrpm->get_left()->is_match(rm->get_left()->get_power_base())) {
                            return lm->get_left() *
                                   (pow(rm->get_left()->get_power_base(),
                                        lmrp->get_power_exponent() -
                                        rm->get_left()->get_power_exponent())) *
                                   pow(lmrpm->get_right(),
                                       lmrp->get_power_exponent()) /
                                   rm->get_right();
                        } else if (lmrpm->get_right()->is_match(rm->get_left()->get_power_base())) {
                            return lm->get_left() *
                                   (pow(rm->get_left()->get_power_base(),
                                        lmrp->get_power_exponent() -
                                        rm->get_left()->get_power_exponent())) *
                                   pow(lmrpm->get_left(),
                                       lmrp->get_power_exponent()) /
                                   rm->get_right();
                        } else if (lmrpm->get_left()->is_match(rm->get_right()->get_power_base())) {
                            return lm->get_left() *
                                   (pow(rm->get_left()->get_power_base(),
                                        lmrp->get_power_exponent() -
                                        rm->get_right()->get_power_exponent())) *
                                   pow(lmrpm->get_right(),
                                       lmrp->get_power_exponent()) /
                                   rm->get_left();
                        } else if (lmrpm->get_right()->is_match(rm->get_right()->get_power_base())) {
                            return lm->get_left() *
                                   (pow(rm->get_left()->get_power_base(),
                                        lmrp->get_power_exponent() -
                                        rm->get_right()->get_power_exponent())) *
                                   pow(lmrpm->get_left(),
                                       lmrp->get_power_exponent()) /
                                   rm->get_left();
                        }
                    }
                }
            }

//  exp(a)/exp(b) -> exp(a - b)
            auto lexp = exp_cast(this->left);
            auto rexp = exp_cast(this->right);
            if (lexp.get() && rexp.get()) {
                return exp(lexp->get_arg() - rexp->get_arg());
            }

//  (c*exp(a))/exp(b) -> c*(exp(a)/exp(b))
//  (exp(a)*c)/exp(b) -> c*(exp(a)/exp(b))
            if (rexp.get() && lm.get()) {
                auto lmre = exp_cast(lm->get_right());
                if (lmre.get()) {
                    return lm->get_left()*(lm->get_right()/this->right);
                }
                auto lmle = exp_cast(lm->get_left());
                if (lmle.get()) {
                    return lm->get_right()*(lm->get_left()/this->right);
                }
            }
//  ((c*exp(a))*d)/exp(b)
//  ((exp(a)*c)*d)/exp(b)
//  (c*(exp(a)*d))/exp(b)
//  (c*(d*exp(a)))/exp(b)
            if (rexp.get() && lm.get()) {
                auto lmlm = multiply_cast(lm->get_left());
                auto lmrm = multiply_cast(lm->get_right());

                if (lmlm.get()) {
                    if (exp_cast(lmlm->get_right()).get()) {
                        return lmlm->get_left()*lm->get_right() *
                               (lmlm->get_right()/this->right);
                    } else if (exp_cast(lmlm->get_left()).get()) {
                        return lmlm->get_right()*lm->get_right() *
                               (lmlm->get_left()/this->right);
                    }
                } else if (lmrm.get()) {
                    if (exp_cast(lmrm->get_right()).get()) {
                        return lmrm->get_left()*lm->get_left() *
                               (lmrm->get_right()/this->right);
                    } else if (exp_cast(lmrm->get_left()).get()) {
                        return lmrm->get_right()*lm->get_left() *
                               (lmrm->get_left()/this->right);
                    }
                }
            }

//  exp(a)/(c*exp(b)) -> (exp(a)/exp(b))/c
//  exp(a)/(exp(b)*c) -> (exp(a)/exp(b))/c
            if (lexp.get() && rm.get()) {
                auto rmre = exp_cast(rm->get_right());
                if (rmre.get()) {
                    return (this->left/rm->get_right())/rm->get_left();
                }
                auto rmle = exp_cast(rm->get_left());
                if (rmle.get()) {
                    return (this->left/rm->get_left())/rm->get_right();
                }
            }

//  (c*exp(a))/(d*exp(b)) -> (c/d)*(exp(a)/exp(b))
//  (c*exp(a))/(exp(b)*d) -> (c/d)*(exp(a)/exp(b))
//  (exp(a)*c)/(d*exp(b)) -> (c/d)*(exp(a)/exp(b))
//  (exp(a)*c)/(exp(b)*d) -> (c/d)*(exp(a)/exp(b))
            if (lm.get() && rm.get()) {
                auto lmre = exp_cast(lm->get_right());
                if (lmre.get()) {
                    auto rmre = exp_cast(rm->get_right());
                    if (rmre.get()) {
                        return (lm->get_left()/rm->get_left()) *
                               (lm->get_right()/rm->get_right());
                    }
                    auto rmle = exp_cast(rm->get_left());
                    if (rmle.get()) {
                        return (lm->get_left()/rm->get_right()) *
                               (lm->get_right()/rm->get_left());
                    }
                }
                auto lmle = exp_cast(lm->get_left());
                if (lmle.get()) {
                    auto rmre = exp_cast(rm->get_right());
                    if (rmre.get()) {
                        return (lm->get_right()/rm->get_left()) *
                               (lm->get_left()/rm->get_right());
                    }
                    auto rmle = exp_cast(rm->get_left());
                    if (rmle.get()) {
                        return (lm->get_right()/rm->get_right()) *
                               (lm->get_left()/rm->get_left());
                    }
                }
            }

//  exp(a)/(c/exp(b)) -> (exp(a)*exp(b))/c
//  exp(a)/(exp(b)/c) -> c*(exp(a)/exp(b))
            auto rd = divide_cast(this->right);
            if (rd.get() && lexp.get()) {
                auto rdre = exp_cast(rd->get_right());
                if (rdre.get()) {
                    return (this->left*rd->get_right())/rd->get_left();
                }
                auto rdle = exp_cast(rd->get_left());
                if (rdle.get()) {
                    return rd->get_right()*(this->left/rd->get_left());
                }
            }

//  (c/exp(a))/exp(b) -> c/(exp(a)*exp(b))
//  (exp(a)/c)/exp(b) -> exp(a)/(c*exp(b))
//  (c/exp(a))/(d/exp(b)) -> (c*exp(b))/(d*exp(a))
//  (c/exp(a))/(exp(b)/d) -> (c*d)/(exp(b)*exp(a))
//  (exp(a)/c)/(d/exp(b)) -> (exp(a)*exp(b))/(d*c)
//  (exp(a)/c)/(exp(b)/d) -> (exp(a)*d)/(exp(b)*c)
//  Note cases like this are already transformed by the (a/b)/c -> a/(b*c)
//  above.

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d div(n,d)/dx = dn/dx*1/d - n*/(d*d)*db/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        df(shared_leaf<T, SAFE_MATH> x) {
            if (this->is_match(x)) {
                return one<T, SAFE_MATH> ();
            }

            const size_t hash = reinterpret_cast<size_t> (x.get());
            if (this->df_cache.find(hash) == this->df_cache.end()) {
                this->df_cache[hash] = this->left->df(x)/this->right 
                                     - this->left*this->right->df(x)/(this->right*this->right);
            }
            return this->df_cache[hash];
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @param[in]     usage     List of register usage count.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream,
                                                                  registers,
                                                                  usage);
                shared_leaf<T, SAFE_MATH> r = this->right->compile(stream,
                                                                   registers,
                                                                   usage);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = ";
                if constexpr (SAFE_MATH) {
                    stream << registers[l.get()] << " == ";
                    if constexpr (jit::is_complex<T> ()) {
                        jit::add_type<T> (stream);
                        stream << "(0, 0)";
                    } else {
                        stream << "0";
                    }
                    stream << " ? ";
                    if constexpr (jit::is_complex<T> ()) {
                        jit::add_type<T> (stream);
                        stream << "(0, 0)";
                    } else {
                        stream << "0";
                    }
                    stream << " : ";
                }
                stream << registers[l.get()] << "/"
                       << registers[r.get()];
                this->endline(stream, usage);
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

            auto x_cast = divide_cast(x);
            if (x_cast.get()) {
                return this->left->is_match(x_cast->get_left()) &&
                       this->right->is_match(x_cast->get_right());
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << "\\frac{";
            this->left->to_latex();
            std::cout << "}{";
            this->right->to_latex();
            std::cout << "}";
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> remove_pseudo() {
            if (this->has_pseudo()) {
                return this->left->remove_pseudo() /
                       this->right->remove_pseudo();
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
        virtual shared_leaf<T, SAFE_MATH> to_vizgraph(std::stringstream &stream,
                                                      jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                const std::string name = jit::to_string('r', this);
                registers[this] = name;
                stream << "    " << name
                       << " [label = \"\\\\\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto l = this->left->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[l.get()] << ";" << std::endl;
                auto r = this->right->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[r.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build divide node from two leaves.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> divide(shared_leaf<T, SAFE_MATH> l,
                                     shared_leaf<T, SAFE_MATH> r) {
        auto temp = std::make_shared<divide_node<T, SAFE_MATH>> (l, r)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::cache.find(i) ==
                leaf_node<T, SAFE_MATH>::cache.end()) {
                leaf_node<T, SAFE_MATH>::cache[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::cache[i])) {
                return leaf_node<T, SAFE_MATH>::cache[i];
            }
        }
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

//------------------------------------------------------------------------------
///  @brief Build divide node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator/(shared_leaf<T, SAFE_MATH> l,
                                        shared_leaf<T, SAFE_MATH> r) {
        return divide<T, SAFE_MATH> (l, r);
    }

//------------------------------------------------------------------------------
///  @brief Build divide node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam L         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar L, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator/(const L l,
                                        shared_leaf<T, SAFE_MATH> r) {
        return divide<T, SAFE_MATH> (constant<T, SAFE_MATH> (static_cast<T> (l)), r);
    }

//------------------------------------------------------------------------------
///  @brief Build multiply node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam R         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar R, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator/(shared_leaf<T, SAFE_MATH> l,
                                        const R r) {
        return divide<T, SAFE_MATH> (l, constant<T, SAFE_MATH> (static_cast<T> (r)));
    }

///  Convenience type alias for shared divide nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_divide = std::shared_ptr<divide_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a divide node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_divide<T, SAFE_MATH> divide_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<divide_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  fused multiply add node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A fused multiply add node.
///
///  Note use templates here to defer this so it can use the operator functions.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    class fma_node final : public triple_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @param[in] l Left node pointer.
///  @param[in] m Middle node pointer.
///  @param[in] r Right node pointer.
///  @return A string rep of the node.
//------------------------------------------------------------------------------
        static std::string to_string(leaf_node<T, SAFE_MATH> *l,
                                     leaf_node<T, SAFE_MATH> *m,
                                     leaf_node<T, SAFE_MATH> *r) {
            return "fma" + jit::format_to_string(reinterpret_cast<size_t> (l))
                         + jit::format_to_string(reinterpret_cast<size_t> (m))
                         + jit::format_to_string(reinterpret_cast<size_t> (r));
        }

    public:
//------------------------------------------------------------------------------
///  @brief Construct a fused multiply add node.
///
///  @param[in] l Left branch.
///  @param[in] m Middle branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
        fma_node(shared_leaf<T, SAFE_MATH> l,
                 shared_leaf<T, SAFE_MATH> m,
                 shared_leaf<T, SAFE_MATH> r) :
        triple_node<T, SAFE_MATH> (l, m, r, fma_node::to_string(l.get(),
                                                                m.get(),
                                                                r.get())) {}

//------------------------------------------------------------------------------
///  @brief Evaluate the results of fused multiply add.
///
///  result = l*m + r
///
///  @returns The value of l*m + r.
//------------------------------------------------------------------------------
        virtual backend::buffer<T> evaluate() {
            backend::buffer<T> l_result = this->left->evaluate();
            backend::buffer<T> r_result = this->right->evaluate();

//  If all the elements on the left are zero, return the leftside without
//  revaluating the rightside.
            if (l_result.is_zero()) {
                return r_result;
            }

            backend::buffer<T> m_result = this->middle->evaluate();
            return backend::fma(l_result, m_result, r_result);
        }

//------------------------------------------------------------------------------
///  @brief Reduce a fused multiply add node.
///
///  @returns A reduced fused multiply add node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> reduce() {
            auto l = constant_cast(this->left);
            auto m = constant_cast(this->middle);
            auto r = constant_cast(this->right);

            if ((l.get() && l->is(0)) ||
                (m.get() && m->is(0))) {
                return this->right;
            } else if (r.get() && r->is(0)) {
                return this->left*this->middle;
            } else if (l.get() && m.get() && r.get()) {
                return constant<T, SAFE_MATH> (this->evaluate());
            } else if (l.get() && m.get()) {
                return this->left*this->middle + this->right;
            } else if (l.get() && l->is(-1)) {
                return this->right - this->middle;
            } else if (m.get() && m->is(-1)) {
                return this->right - this->left;
            } else if (l.get() && l->is(1)) {
                return this->middle + this->right;
            } else if (m.get() && m->is(1)) {
                return this->left + this->right;
            }

//  Check if the left and middle are combinable. This will be constant merged in
//  multiply reduction.
            if (is_constant_combineable(this->left, this->middle) ||
                is_variable_combineable(this->left, this->middle)) {
                return (this->left*this->middle)  + this->right;
            }

//  fma(c2,c1,a) -> fma(c1,c2,a)
            if (is_constant_promotable(this->middle, 
                                       this->left)) {
                return fma(this->middle, this->left, this->right);
            }

//  fma(a,b,a) -> a*(1 + b)
//  fma(b,a,a) -> a*(1 + b)
            if (this->left->is_match(this->right)) {
                return this->left*(1.0 + this->middle);
            } else if (this->middle->is_match(this->right)) {
                return this->middle*(1.0 + this->left);
            }

//  fma(c1,c2 + a,c3) -> fma(c4,a,c5)
            auto ma = add_cast(this->middle);
            if (ma.get()) {
                if (is_constant_combineable(this->left, ma->get_left()) &&
                    is_constant_combineable(this->left, this->right)) {
                    return fma(this->left,
                               ma->get_right(),
                               fma(this->left, ma->get_left(), this->right));
                }
            }

//  fma(c1,c2 - a,c3) -> c4 - c5*a
            auto ms = subtract_cast(this->middle);
            if (ms.get()) {
                if (is_constant_combineable(this->left, ms->get_left()) &&
                    is_constant_combineable(this->left, this->right)) {
                    return fma(this->left, ms->get_left(), this->right) -
                           this->left*ms->get_right();
                }
            }

//  Common factor reduction. If the left and right are both multiply nodes check
//  for a common factor. So you can change a*b + (a*c) -> a*(b + c).
            auto lm = multiply_cast(this->left);
            auto mm = multiply_cast(this->middle);
            auto rm = multiply_cast(this->right);
            if (rm.get()) {
                if (rm->get_left()->is_match(this->left)) {
                    return this->left*(this->middle + rm->get_right());
                } else if (rm->get_left()->is_match(this->middle)) {
                    return this->middle*(this->left + rm->get_right());
                } else if (rm->get_right()->is_match(this->left)) {
                    return this->left*(this->middle + rm->get_left());
                } else if (rm->get_right()->is_match(this->middle)) {
                    return this->middle*(this->left + rm->get_left());
                }

//  Chnage case of
//  fma(a,b,-c1*b) -> a*b - c1*b
                auto rmlc = constant_cast(rm->get_left());
                if (rmlc.get() && rmlc->evaluate().is_negative()) {
                    return this->left*this->middle -
                           (-1.0*rm->get_left())*rm->get_right();
                }

//  Change cases like
//  fma(c1,a,c2*b) -> c1*fma(c3,b,a)
//  fma(a,c1,c2*b) -> c1*fma(c3,b,a)
//  fma(c1,a,b*c2) -> c1*fma(c3,b,a)
//  fma(a,c1,b*c2) -> c1*fma(c3,b,a)
                if (is_constant_combineable(this->left,
                                            rm->get_left()) &&
                    !this->left->has_constant_zero()) {
                    auto temp = rm->get_left()/this->left;
                    if (temp->is_normal()) {
                        return this->left*fma(temp,
                                              rm->get_right(),
                                              this->middle);
                    }
                }
                if (is_constant_combineable(this->middle,
                                            rm->get_left()) &&
                    !this->middle->has_constant_zero()) {
                    auto temp = rm->get_left()/this->middle;
                    if (temp->is_normal()) {
                        return this->middle*fma(temp,
                                                rm->get_right(),
                                                this->left);
                    }
                }
                if (is_constant_combineable(this->left,
                                            rm->get_right()) &&
                    !this->left->has_constant_zero()) {
                    auto temp = rm->get_right()/this->left;
                    if (temp->is_normal()) {
                        return this->left*fma(temp,
                                              rm->get_left(),
                                              this->middle);
                    }
                }
                if (is_constant_combineable(this->middle,
                                            rm->get_right()) &&
                    !this->middle->has_constant_zero()) {
                    auto temp = rm->get_right()/this->middle;
                    if (temp->is_normal()) {
                        return this->middle*fma(temp,
                                                rm->get_left(),
                                                this->left);
                    }
                }

//  Convert fma(a*b,c,d*e) -> fma(d,e,a*b*c)
//  Convert fma(a,b*c,d*e) -> fma(d,e,a*b*c)
                if ((lm.get() || mm.get()) &&
                    (this->left->get_complexity() + this->middle->get_complexity() >
                     this->right->get_complexity())) {
                    return fma(rm->get_left(), rm->get_right(),
                               this->left*this->middle);
                }
            }

//  Handle cases like.
//  fma(c1*a,b,c2*d) -> c1*(a*b + c2/c1*d)
//  fma(a*c1,b,c2*d) -> c1*(a*b + c2/c1*d)
//  fma(c1*a,b,d*c2*d) -> c1*(a*b + c2/c1*d)
//  fma(a*c1,b,d*c2*d) -> c1*(a*b + c2/c1*d)
            if (lm.get() && rm.get()) {
                if (is_constant_combineable(rm->get_left(),
                                            lm->get_left()) &&
                    !lm->get_left()->has_constant_zero()) {
                    auto temp = rm->get_left()/lm->get_left();
                    if (temp->is_normal()){
                        return lm->get_left()*fma(lm->get_right(),
                                                  this->middle,
                                                  temp*rm->get_right());
                    }
                }
                if (is_constant_combineable(rm->get_left(),
                                            lm->get_right()) &&
                    !lm->get_right()->has_constant_zero()) {
                    auto temp = rm->get_left()/lm->get_right();
                    if (temp->is_normal()){
                        return lm->get_right()*fma(lm->get_left(),
                                                   this->middle,
                                                   temp*rm->get_right());
                    }
                }
                if (is_constant_combineable(rm->get_right(),
                                            lm->get_left()) &&
                    !lm->get_left()->has_constant_zero()) {
                    auto temp = rm->get_right()/lm->get_left();
                    if (temp->is_normal()) {
                        return lm->get_left()*fma(lm->get_right(),
                                                  this->middle,
                                                  temp*rm->get_left());
                    }
                }
                if (is_constant_combineable(rm->get_right(),
                                            lm->get_right()) &&
                    !lm->get_right()->has_constant_zero()) {
                    auto temp = rm->get_right()/lm->get_right();
                    if (temp->is_normal()) {
                        return lm->get_right()*fma(lm->get_left(),
                                                   this->middle,
                                                   temp*rm->get_left());
                    }
                }
            }

//  Move constant multiplies to the left.
            if (lm.get()) {
// fma(c1*a,b,c) -> fma(c1,a*b,c)
                if (is_constant_promotable(lm->get_left(),
                                           lm->get_right())) {
                    return fma(lm->get_left(),
                               lm->get_right()*this->middle,
                               this->right);
                }
            } else if (mm.get()) {
// fma(c1,c2*a,b) -> fma(c3,a,b)
// fma(c1,a*c2,b) -> fma(c3,a,b)
// fma(a,c1*b,c) -> fma(c1,a*b,c)
                if (is_constant_combineable(this->left,
                                            mm->get_left())) {
                    auto temp = this->left*mm->get_left();
                    if (temp->is_normal()) {
                        return fma(temp,
                                   mm->get_right(),
                                   this->right);
                    }
                }
                if (is_constant_combineable(this->left,
                                            mm->get_right())) {
                    auto temp = this->left*mm->get_right();
                    if (temp->is_normal()) {
                        return fma(temp,
                                   mm->get_left(),
                                   this->right);
                    }
                }
                if (is_constant_promotable(mm->get_left(),
                                           this->left)) {
                    return fma(mm->get_left(),
                               this->left*mm->get_right(),
                               this->right);
                }
            }

//  fma(c1,a,c2/b) -> c1*(a + c3/b)
//  fma(a,c1,c2/b) -> c1*(a + c3/b)
            auto rd = divide_cast(this->right);
            if (rd.get()) {
                if (is_constant_combineable(this->left,
                                            rd->get_left()) &&
                    !this->left->has_constant_zero()) {
                    auto temp = rd->get_left()/this->left;
                    if (temp->is_normal()) {
                        return this->left*(this->middle +
                                           temp/rd->get_right());
                    }
                }
                if (is_constant_combineable(this->middle,
                                                   rd->get_left()) &&
                           !this->middle->has_constant_zero()) {
                    auto temp = rd->get_left()/this->middle;
                    if (temp->is_normal()) {
                        return this->middle*(this->left +
                                             temp/rd->get_right());
                    }
                }
            }

//  Reduce fma(a/b,b,c) -> a + c
//  Reduce fma(a,b/a,c) -> b + c
            auto ld = divide_cast(this->left);
            if (ld.get() && ld->get_right()->is_match(this->middle)) {
                return ld->get_left() + this->right;
            }
            auto md = divide_cast(this->middle);
            if (md.get() && md->get_right()->is_match(this->left)) {
                return md->get_left() + this->right;
            }

//  Common denominator reductions.
            if (ld.get() && rd.get()) {
//  fma(a/(b*c),d,e/c) -> fma(a,d,e*b)/(b*c)
//  fma(a/(c*b),d,e/c) -> fma(a,d,e*b)/(c*b)
//  fma(a/c,d,e/(c*b)) -> fma(a*b,d,e)/(b*c)
//  fma(a/c,d,e/(b*c)) -> fma(a*b,d,e)/(c*b)
                auto ldrm = multiply_cast(ld->get_right());
                auto rdrm = multiply_cast(rd->get_right());

                if (ldrm.get()) {
                    if (ldrm->get_right()->is_match(rd->get_right())) {
                        return fma(ld->get_left(), this->middle,
                                   rd->get_left()*ldrm->get_left()) /
                               ld->get_right();
                    } else if (ldrm->get_left()->is_match(rd->get_right())) {
                        return fma(ld->get_left(), this->middle,
                                   rd->get_left()*ldrm->get_right()) /
                               ld->get_right();
                    }
                } else if (rdrm.get()) {
                    if (rdrm->get_right()->is_match(ld->get_right())) {
                        return fma(ld->get_left()*rdrm->get_left(),
                                   this->middle, rd->get_left()) /
                               rd->get_right();
                    } else if (rdrm->get_left()->is_match(ld->get_right())) {
                        return fma(ld->get_left()*rdrm->get_right(),
                                   this->middle, rd->get_left()) /
                               rd->get_right();
                    }
                }
            } else if (md.get() && rd.get()) {
//  fma(a,d/(b*c),e/c) -> fma(a,d,e*b)/(b*c)
//  fma(a,d/(c*b),e/c) -> fma(a,d,e*b)/(c*b)
//  fma(a,d/c,e/(c*b)) -> fma(a,d*b,e)/(b*c)
//  fma(a,d/c,e/(b*c)) -> fma(a,d*b,e)/(c*b)
                auto mdrm = multiply_cast(md->get_right());
                auto rdrm = multiply_cast(rd->get_right());

                if (mdrm.get()) {
                    if (mdrm->get_right()->is_match(rd->get_right())) {
                        return fma(this->left, md->get_left(),
                                   rd->get_left()*mdrm->get_left()) /
                               md->get_right();
                    } else if (mdrm->get_left()->is_match(rd->get_right())) {
                        return fma(this->left, md->get_left(),
                                   rd->get_left()*mdrm->get_right()) /
                               md->get_right();
                    }
                } else if (rdrm.get()) {
                    if (rdrm->get_right()->is_match(md->get_right())) {
                        return fma(this->left, md->get_left()*rdrm->get_left(),
                                   rd->get_left()) /
                               rd->get_right();
                    } else if (rdrm->get_left()->is_match(md->get_right())) {
                        return fma(this->left, md->get_left()*rdrm->get_right(),
                                   rd->get_left()) /
                               rd->get_right();
                    }
                }
            }

//  Chained fma reductions.
            auto rfma = fma_cast(this->right);
            if (rfma.get()) {
//  fma(a, b, fma(c, b, d)) -> fma(b, a + c, d)
//  fma(b, a, fma(c, b, d)) -> fma(b, a + c, d)
//  fma(a, b, fma(b, c, d)) -> fma(b, a + c, d)
//  fma(b, a, fma(b, c, d)) -> fma(b, a + c, d)
                if (this->middle->is_match(rfma->get_middle())) {
                    return fma(this->middle,
                               this->left + rfma->get_left(),
                               rfma->get_right());
                } else if (this->left->is_match(rfma->get_middle())) {
                    return fma(this->left,
                               this->middle + rfma->get_left(),
                               rfma->get_right());
                } else if (this->middle->is_match(rfma->get_left())) {
                    return fma(this->middle,
                               this->left + rfma->get_middle(),
                               rfma->get_right());
                } else if (this->left->is_match(rfma->get_left())) {
                    return fma(this->left,
                               this->middle + rfma->get_middle(),
                               rfma->get_right());
                }
     
                if (mm.get()) {
//  fma(a, e*b, fma(c, b, d)) -> fma(b, fma(a, e, c), d)
//  fma(a, b*e, fma(c, b, d)) -> fma(b, fma(a, e, c), d)
//  fma(a, e*b, fma(b, c, d)) -> fma(b, fma(a, e, c), d)
//  fma(a, b*e, fma(b, c, d)) -> fma(b, fma(a, e, c), d)
                    if (mm->get_right()->is_match(rfma->get_middle())) {
                        return fma(mm->get_right(),
                                   fma(this->left,
                                       mm->get_left(),
                                       rfma->get_left()),
                                   rfma->get_right());
                    } else if (mm->get_left()->is_match(rfma->get_middle())) {
                        return fma(mm->get_left(),
                                   fma(this->left,
                                       mm->get_right(),
                                       rfma->get_left()),
                                   rfma->get_right());
                    } else if (mm->get_right()->is_match(rfma->get_left())) {
                        return fma(mm->get_right(),
                                   fma(this->left,
                                       mm->get_left(),
                                       rfma->get_middle()),
                                   rfma->get_right());
                    } else if (mm->get_left()->is_match(rfma->get_left())) {
                        return fma(mm->get_left(),
                                   fma(this->left,
                                       mm->get_right(),
                                       rfma->get_middle()),
                                   rfma->get_right());
                    }
                } else if (lm.get()) {
//  fma(e*b, a, fma(c, b, d)) -> fma(b, fma(a, e, c), d)
//  fma(b*e, a, fma(c, b, d)) -> fma(b, fma(a, e, c), d)
//  fma(e*b, a, fma(b, c, d)) -> fma(b, fma(a, e, c), d)
//  fma(e*d, a, fma(b, c, d)) -> fma(b, fma(a, e, c), d)
                    if (lm->get_right()->is_match(rfma->get_middle())) {
                        return fma(lm->get_right(),
                                   fma(this->middle,
                                       lm->get_left(),
                                       rfma->get_left()),
                                   rfma->get_right());
                    } else if (lm->get_left()->is_match(rfma->get_middle())) {
                        return fma(lm->get_left(),
                                   fma(this->middle,
                                       lm->get_right(),
                                       rfma->get_left()),
                                   rfma->get_right());
                    } else if (lm->get_right()->is_match(rfma->get_left())) {
                        return fma(lm->get_right(),
                                   fma(this->middle,
                                       lm->get_left(),
                                       rfma->get_middle()),
                                   rfma->get_right());
                    } else if (lm->get_left()->is_match(rfma->get_left())) {
                        return fma(lm->get_left(),
                                   fma(this->middle,
                                       lm->get_right(),
                                       rfma->get_middle()),
                                   rfma->get_right());
                    }
                }

                auto rfmamm = multiply_cast(rfma->get_middle());
                auto rfmalm = multiply_cast(rfma->get_left());
                if (rfmamm.get()) {
//  fma(a, b, fma(c, e*b, d)) -> fma(b, fma(c, e, a), d)
//  fma(b, a, fma(c, e*b, d)) -> fma(b, fma(c, e, a), d)
//  fma(a, b, fma(c, b*e, d)) -> fma(b, fma(c, e, a), d)
//  fma(b, a, fma(c, b*e, d)) -> fma(b, fma(c, e, a), d)
                    if (rfmamm->get_right()->is_match(this->middle)) {
                        return fma(this->middle,
                                   fma(rfma->get_left(),
                                       rfmamm->get_left(),
                                       this->left),
                                   rfma->get_right());
                    } else if (rfmamm->get_right()->is_match(this->left)) {
                        return fma(this->left,
                                   fma(rfma->get_left(),
                                       rfmamm->get_left(),
                                       this->middle),
                                   rfma->get_right());
                    } else if (rfmamm->get_left()->is_match(this->middle)) {
                        return fma(this->middle,
                                   fma(rfma->get_left(),
                                       rfmamm->get_right(),
                                       this->left),
                                   rfma->get_right());
                    } else if (rfmamm->get_left()->is_match(this->left)) {
                        return fma(this->left,
                                   fma(rfma->get_left(),
                                       rfmamm->get_right(),
                                       this->middle),
                                   rfma->get_right());
                    }
                } else if (rfmalm.get()) {
//  fma(a, b, fma(e*b, c, d)) -> fma(b, fma(c, e, a), d)
//  fma(b, a, fma(e*b, c, d)) -> fma(b, fma(c, e, a), d)
//  fma(a, b, fma(b*e, c, d)) -> fma(b, fma(c, e, a), d)
//  fma(b, a, fma(b*e, c, d)) -> fma(b, fma(c, e, a), d)
                    if (rfmalm->get_right()->is_match(this->middle)) {
                        return fma(this->middle,
                                   fma(rfma->get_middle(),
                                       rfmalm->get_left(),
                                       this->left),
                                   rfma->get_right());
                    } else if (rfmalm->get_right()->is_match(this->left)) {
                        return fma(this->left,
                                   fma(rfma->get_middle(),
                                       rfmalm->get_left(),
                                       this->middle),
                                   rfma->get_right());
                    } else if (rfmalm->get_left()->is_match(this->middle)) {
                        return fma(this->middle,
                                   fma(rfma->get_middle(),
                                       rfmalm->get_right(),
                                       this->left),
                                   rfma->get_right());
                    } else if (rfmalm->get_left()->is_match(this->left)) {
                        return fma(this->left,
                                   fma(rfma->get_middle(),
                                       rfmalm->get_right(),
                                       this->middle),
                                   rfma->get_right());
                    }
                }

                if (mm.get() && rfmamm.get()) {
//  fma(a, f*b, fma(c, e*b, d)) -> fma(b, fma(a, f, c*e), d)
//  fma(a, b*f, fma(c, e*b, d)) -> fma(b, fma(a, f, c*e), d)
//  fma(a, f*b, fma(c, b*e, d)) -> fma(b, fma(a, f, c*e), d)
//  fma(a, b*f, fma(c, b*e, d)) -> fma(b, fma(a, f, c*e), d)
                    if (mm->get_right()->is_match(rfmamm->get_right())) {
                        return fma(mm->get_right(),
                                   fma(this->left,
                                       mm->get_left(),
                                       rfma->get_left()*rfmamm->get_left()),
                                   rfma->get_right());
                    } else if (mm->get_left()->is_match(rfmamm->get_right())) {
                        return fma(mm->get_left(),
                                   fma(this->left,
                                       mm->get_right(),
                                       rfma->get_left()*rfmamm->get_left()),
                                   rfma->get_right());
                    } else if (mm->get_right()->is_match(rfmamm->get_left())) {
                        return fma(mm->get_right(),
                                   fma(this->left,
                                       mm->get_left(),
                                       rfma->get_left()*rfmamm->get_right()),
                                   rfma->get_right());
                    } else if (mm->get_left()->is_match(rfmamm->get_left())) {
                        return fma(mm->get_left(),
                                   fma(this->left,
                                       mm->get_right(),
                                       rfma->get_left()*rfmamm->get_right()),
                                   rfma->get_right());
                    }
                } else if (lm.get() && rfmamm.get()) {
//  fma(f*b, a, fma(c, e*b, d)) -> fma(b, fma(a, f, c*e), d)
//  fma(b*f, a, fma(c, e*b, d)) -> fma(b, fma(a, f, c*e), d)
//  fma(f*b, a, fma(c, b*e, d)) -> fma(b, fma(a, f, c*e), d)
//  fma(b*f, a, fma(c, b*e, d)) -> fma(b, fma(a, f, c*e), d)
                    if (lm->get_right()->is_match(rfmamm->get_right())) {
                        return fma(lm->get_right(),
                                   fma(this->middle,
                                       lm->get_left(),
                                       rfma->get_left()*rfmamm->get_left()),
                                   rfma->get_right());
                    } else if (lm->get_left()->is_match(rfmamm->get_right())) {
                        return fma(lm->get_left(),
                                   fma(this->middle,
                                       lm->get_right(),
                                       rfma->get_left()*rfmamm->get_left()),
                                   rfma->get_right());
                    } else if (lm->get_right()->is_match(rfmamm->get_left())) {
                        return fma(lm->get_right(),
                                   fma(this->middle,
                                       lm->get_left(),
                                       rfma->get_left()*rfmamm->get_right()),
                                   rfma->get_right());
                    } else if (lm->get_left()->is_match(rfmamm->get_left())) {
                        return fma(lm->get_left(),
                                   fma(this->middle,
                                       lm->get_right(),
                                       rfma->get_left()*rfmamm->get_right()),
                                   rfma->get_right());
                    }
                } else if (mm.get() && rfmalm.get()) {
//  fma(a, f*b, fma(e*b, c, d)) -> fma(b, fma(a, f, c*e), d)
//  fma(a, b*f, fma(e*b, c, d)) -> fma(b, fma(a, f, c*e), d)
//  fma(a, f*b, fma(b*e, c, d)) -> fma(b, fma(a, f, c*e), d)
//  fma(a, b*f, fma(b*e, c, d)) -> fma(b, fma(a, f, c*e), d)
                    if (mm->get_right()->is_match(rfmalm->get_right())) {
                        return fma(mm->get_right(),
                                   fma(this->left,
                                       mm->get_left(),
                                       rfma->get_middle()*rfmalm->get_left()),
                                   rfma->get_right());
                    } else if (mm->get_left()->is_match(rfmalm->get_right())) {
                        return fma(mm->get_left(),
                                   fma(this->left,
                                       mm->get_right(),
                                       rfma->get_middle()*rfmalm->get_left()),
                                   rfma->get_right());
                    } else if (mm->get_right()->is_match(rfmalm->get_left())) {
                        return fma(mm->get_right(),
                                   fma(this->left,
                                       mm->get_left(),
                                       rfma->get_middle()*rfmalm->get_right()),
                                   rfma->get_right());
                    } else if (mm->get_left()->is_match(rfmalm->get_left())) {
                        return fma(mm->get_left(),
                                   fma(this->left,
                                       mm->get_right(),
                                       rfma->get_middle()*rfmalm->get_right()),
                                   rfma->get_right());
                    }
                } else if (lm.get() && rfmalm.get()) {
//  fma(f*b, a, fma(e*b, c, d)) -> fma(b, fma(a, f, c*e), d)
//  fma(b*f, a, fma(e*b, c, d)) -> fma(b, fma(a, f, c*e), d)
//  fma(f*b, a, fma(b*e, c, d)) -> fma(b, fma(a, f, c*e), d)
//  fma(b*f, a, fma(b*e, c, d)) -> fma(b, fma(a, f, c*e), d)
                    if (lm->get_right()->is_match(rfmalm->get_right())) {
                        return fma(lm->get_right(),
                                   fma(this->middle,
                                       lm->get_left(),
                                       rfma->get_middle()*rfmalm->get_left()),
                                   rfma->get_right());
                    } else if (lm->get_left()->is_match(rfmalm->get_right())) {
                        return fma(lm->get_left(),
                                   fma(this->middle,
                                       lm->get_right(),
                                       rfma->get_middle()*rfmalm->get_left()),
                                   rfma->get_right());
                    } else if (lm->get_right()->is_match(rfmalm->get_left())) {
                        return fma(lm->get_right(),
                                   fma(this->middle,
                                       lm->get_left(),
                                       rfma->get_middle()*rfmalm->get_right()),
                                   rfma->get_right());
                    } else if (lm->get_left()->is_match(rfmalm->get_left())) {
                        return fma(lm->get_left(),
                                   fma(this->middle,
                                       lm->get_right(),
                                       rfma->get_middle()*rfmalm->get_right()),
                                   rfma->get_right());
                    }
                }

                if (is_variable_combineable(this->middle, rfma->get_middle())) {
                    if (is_greater_exponent(this->middle, rfma->get_middle())) {
//  fma(a,x^b,fma(c,x^d,e)) -> fma(x^d,fma(x^(d-b),a,c),e) if b > d
                        return fma(rfma->get_middle(),
                                   fma(this->middle/rfma->get_middle(),
                                       this->left,
                                       rfma->get_left()),
                                   rfma->get_right());
                    } else {
//  fma(a,x^b,fma(c,x^d,e)) -> fma(x^b,fma(x^(d-b),c,a),e) if d > b
                        return fma(this->middle,
                                   fma(rfma->get_middle()/this->middle,
                                       rfma->get_left(),
                                       this->left),
                                   rfma->get_right());
                    }
                } else if (is_variable_combineable(this->left, rfma->get_middle())) {
                    if (is_greater_exponent(this->left, rfma->get_middle())) {
//  fma(x^b,a,fma(c,x^d,e)) -> fma(x^d,fma(x^(d-b),a,c),e) if b > d
                        return fma(rfma->get_middle(),
                                   fma(this->left/rfma->get_middle(),
                                       this->middle,
                                       rfma->get_left()),
                                   rfma->get_right());
                    } else {
//  fma(x^b,a,fma(c,x^d,e)) -> fma(x^b,fma(x^(d-b),c,a),e) if d > b
                        return fma(this->left,
                                   fma(rfma->get_middle()/this->left,
                                       rfma->get_left(),
                                       this->middle),
                                   rfma->get_right());
                    }
                } else if (is_variable_combineable(this->middle, rfma->get_left())) {
                    if (is_greater_exponent(this->middle, rfma->get_left())) {
//  fma(a,x^b,fma(x^d,c,e)) -> fma(x^d,fma(x^(d-b),a,c),e) if b > d
                        return fma(rfma->get_left(),
                                   fma(this->middle/rfma->get_left(),
                                       this->left,
                                       rfma->get_middle()),
                                   rfma->get_right());
                    } else {
//  fma(a,x^b,fma(x^d,c,e)) -> fma(x^b,fma(x^(d-b),c,a),e) if d > b
                        return fma(this->middle,
                                   fma(rfma->get_left()/this->middle,
                                       rfma->get_middle(),
                                       this->left),
                                   rfma->get_right());
                    }
                } else if (is_variable_combineable(this->left, rfma->get_left())) {
                    if (is_greater_exponent(this->left, rfma->get_left())) {
//  fma(x^b,a,fma(x^d,c,e)) -> fma(x^d,fma(x^(d-b),a,c),e) if b > d
                        return fma(rfma->get_left(),
                                   fma(this->left/rfma->get_left(),
                                       this->middle,
                                       rfma->get_middle()),
                                   rfma->get_right());
                    } else {
//  fma(x^b,a,fma(x^d,c,e)) -> fma(x^b,fma(x^(d-b),c,a),e) if d > b
                        return fma(this->left,
                                   fma(rfma->get_left()/this->left,
                                       rfma->get_middle(),
                                       this->middle),
                                   rfma->get_right());
                    }
                }

//  fma(a,b,fma(a,b,c)) -> fma(2*a,b,c)
//  fma(a,b,fma(b,a,c)) -> fma(2*a,b,c)
                if (this->left->is_match(rfma->get_left()) &&
                    this->middle->is_match(rfma->get_middle())) {
                    return fma(2.0*this->left, this->middle, rfma->get_right());
                } else if (this->left->is_match(rfma->get_middle()) &&
                           this->middle->is_match(rfma->get_left())) {
                    return fma(2.0*this->left, this->middle, rfma->get_right());
                }

//  fma(a,b/c,fma(e,f/c,g)) -> (a*b + e*f)/c + g
//  fma(a,b/c,fma(e/c,f,g)) -> (a*b + e*f)/c + g
//  fma(a/c,b,fma(e,f/c,g)) -> (a*b + e*f)/c + g
//  fma(a/c,b,fma(e/c,f,g)) -> (a*b + e*f)/c + g
                auto fmald = divide_cast(rfma->get_left());
                auto fmamd = divide_cast(rfma->get_middle());
                if (ld.get()) {
                    if (fmald.get() && ld->get_right()->is_match(fmald->get_right())) {
                        return (ld->get_left()*this->middle +
                                fmald->get_left()*rfma->get_middle())/ld->get_right() +
                               rfma->get_right();
                    } else if (fmamd.get() && ld->get_right()->is_match(fmamd->get_right())) {
                        return (ld->get_left()*this->middle +
                                fmamd->get_left()*rfma->get_left())/ld->get_right() +
                               rfma->get_right();
                    }
                } else if (md.get()) {
                    if (fmald.get() && md->get_right()->is_match(fmald->get_right())) {
                        return (md->get_left()*this->left +
                                fmald->get_left()*rfma->get_middle())/md->get_right() +
                               rfma->get_right();
                    } else if (fmamd.get() && md->get_right()->is_match(fmamd->get_right())) {
                        return (md->get_left()*this->left +
                                fmamd->get_left()*rfma->get_left())/md->get_right() +
                               rfma->get_right();
                    }
                }
            }

//  Check to see if it is worth moving nodes out of a fma nodes. These should be
//  restricted to variable like nodes. Only do this reduction if the complexity
//  reduces.
            if (this->left->is_all_variables()) {
                auto rdl = this->right/this->left;
                if (rdl->get_complexity() < this->left->get_complexity() +
                                            this->right->get_complexity()) {
                    return (this->middle + rdl)*this->left;
                }
            } else if (this->middle->is_all_variables()) {
                auto rdm = this->right/this->middle;
                if (rdm->get_complexity() < this->middle->get_complexity() +
                                            this->right->get_complexity()) {
                    return (this->left + rdm)*this->middle;
                }
            }

//  Change negative exponents to divide so that can be factored out.
//  fma(a,b^-c,d) = a/b^c + d
//  fma(b^-c,a,d) = a/b^c + d
            auto lp = pow_cast(this->left);
            if (lp.get()) {
                auto exponent = constant_cast(lp->get_right());
                if (exponent.get() && exponent->evaluate().is_negative()) {
                    return this->middle/pow(lp->get_left(), -lp->get_right()) +
                           this->right;
                }
            }
            auto mp = pow_cast(this->middle);
            if (mp.get()) {
                auto exponent = constant_cast(mp->get_right());
                if (exponent.get() && exponent->evaluate().is_negative()) {
                    return this->left/pow(mp->get_left(), -mp->get_right()) +
                           this->right;
                }
            }

//  fma(a,b/c,b/d) -> b*(a/c + 1/d)
//  fma(a,c/b,d/b) -> (a*c + d)/b
            if (md.get() && rd.get()) {
                if (md->get_left()->is_match(rd->get_left())) {
                    return md->get_left()*(this->left/md->get_right() +
                                           1.0/rd->get_right());
                } else if (md->get_right()->is_match(rd->get_right())) {
                    return (this->left*md->get_left() +
                            rd->get_left())/md->get_right();
                }
            }
//  fma(b/c,a,b/d) -> b*(a/c + 1/d)
//  fma(c/b,a,d/b) -> (a*c + d)/b
            if (ld.get() && rd.get()) {
                if (ld->get_left()->is_match(rd->get_left())) {
                    return ld->get_left()*(this->middle/ld->get_right() +
                                           1.0/rd->get_right());
                } else if (ld->get_right()->is_match(rd->get_right())) {
                    return (this->middle*ld->get_left() +
                            rd->get_left())/ld->get_right();
                }
            }

//  fma(a/b,c,(d/b)*e) -> fma(a,c,d*e)/b
//  fma(a/b,c,e*(d/b)) -> fma(a,c,d*e)/b
            if (rm.get() && ld.get()) {
                auto rmld = divide_cast(rm->get_left());
                if (rmld.get() && ld->get_right()->is_match(rmld->get_right())) {
                    return fma(ld->get_left(), this->middle, rmld->get_left()*rm->get_right())/ld->get_right();
                }
                auto rmrd = divide_cast(rm->get_right());
                if (rmrd.get() && ld->get_right()->is_match(rmrd->get_right())) {
                    return fma(ld->get_left(), this->middle, rmrd->get_left()*rm->get_left())/ld->get_right();
                }
            }
//  fma(a,c/b,(d/b)*e) -> fma(a,c,d*e)/b
//  fma(a,c/b,e*(d/b)) -> fma(a,c,d*e)/b
            if (rm.get() && md.get()) {
                auto rmld = divide_cast(rm->get_left());
                if (rmld.get() && md->get_right()->is_match(rmld->get_right())) {
                    return fma(this->left, md->get_left(), rmld->get_left()*rm->get_right())/md->get_right();
                }
                auto rmrd = divide_cast(rm->get_right());
                if (rmrd.get() && md->get_right()->is_match(rmrd->get_right())) {
                    return fma(this->left, md->get_left(), rmrd->get_left()*rm->get_left())/md->get_right();
                }
            }

//  fma(a/b*c,d,e/b) -> fma(a*c,d,e)/b
//  fma(a*c/b,d,e/b) -> fma(a*c,d,e)/b
            if (rd.get() && lm.get()) {
                auto lmld = divide_cast(lm->get_left());
                if (lmld.get() && rd->get_right()->is_match(lmld->get_right())) {
                    return fma(lmld->get_left()*lm->get_right(), this->middle, rd->get_left())/rd->get_right();
                }
                auto lmrd = divide_cast(lm->get_right());
                if (lmrd.get() && rd->get_right()->is_match(lmrd->get_right())) {
                    return fma(lmld->get_left()*lm->get_left(), this->middle, rd->get_left())/rd->get_right();
                }
            }
//  fma(a,c/b*d,e/b) -> fma(a,c*d,e)/b
//  fma(a,c*d/b,e/b) -> fma(a,c*d,e)/b
            if (rd.get() && mm.get()) {
                auto mmld = divide_cast(mm->get_left());
                if (mmld.get() && rd->get_right()->is_match(mmld->get_right())) {
                    return fma(this->left, mmld->get_left()*mm->get_right(), rd->get_left())/rd->get_right();
                }
                auto mmrd = divide_cast(mm->get_right());
                if (mmrd.get() && rd->get_right()->is_match(mmrd->get_right())) {
                    return fma(this->left, mmrd->get_left()*mm->get_left(), rd->get_left())/rd->get_right();
                }
            }

//  fma(a, b/c, ((f/c)*e)*d) -> fma(a, b, f*e*d)/c
//  fma(a/c, b, ((f/c)*e)*d) -> fma(a, b, f*e*d)/c
//  fma(a, b/c, (e*(f/c))*d) -> fma(a, b, f*e*d)/c
//  fma(a/c, b, (e*(f/c))*d) -> fma(a, b, f*e*d)/c
//  fma(a, b/c, d*((f/c)*e)) -> fma(a, b, f*e*d)/c
//  fma(a/c, b, d*((f/c)*e)) -> fma(a, b, f*e*d)/c
//  fma(a, b/c, d*(e*(f/c))) -> fma(a, b, f*e*d)/c
//  fma(a/c, b, d*(e*(f/c))) -> fma(a, b, f*e*d)/c
            if (md.get() && rm.get()) {
                auto rmlm = multiply_cast(rm->get_left());
                if (rmlm.get()) {
                    auto rmlmld = divide_cast(rmlm->get_left());
                    if (rmlmld.get() && rmlmld->get_right()->is_match(md->get_right())) {
                        return fma(this->left, md->get_left(),
                                   rmlmld->get_left()*rmlm->get_right()*rm->get_right())/md->get_right();
                    }
                    auto rmlmrd = divide_cast(rmlm->get_right());
                    if (rmlmrd.get() && rmlmrd->get_right()->is_match(md->get_right())) {
                        return fma(this->left, md->get_left(),
                                   rmlmrd->get_left()*rmlm->get_left()*rm->get_right())/md->get_right();
                    }
                }
                auto rmrm = multiply_cast(rm->get_right());
                if (rmrm.get()) {
                    auto rmrmld = divide_cast(rmrm->get_left());
                    if (rmrmld.get() && rmrmld->get_right()->is_match(md->get_right())) {
                        return fma(this->left, md->get_left(),
                                   rmrmld->get_left()*rmrm->get_right()*rm->get_left())/md->get_right();
                    }
                    auto rmrmrd = divide_cast(rmrm->get_right());
                    if (rmrmrd.get() && rmrmrd->get_right()->is_match(md->get_right())) {
                        return fma(this->left, md->get_left(),
                                   rmrmrd->get_left()*rmrm->get_left()*rm->get_left())/md->get_right();
                    }
                }
            } else if (ld.get() && rm.get()) {
                auto rmlm = multiply_cast(rm->get_left());
                if (rmlm.get()) {
                    auto rmlmld = divide_cast(rmlm->get_left());
                    if (rmlmld.get() && rmlmld->get_right()->is_match(ld->get_right())) {
                        return fma(ld->get_left(), this->middle,
                                   rmlmld->get_left()*rmlm->get_right()*rm->get_right())/ld->get_right();
                    }
                    auto rmlmrd = divide_cast(rmlm->get_right());
                    if (rmlmrd.get() && rmlmrd->get_right()->is_match(ld->get_right())) {
                        return fma(ld->get_left(), this->middle,
                                   rmlmrd->get_left()*rmlm->get_right()*rm->get_right())/ld->get_right();
                    }
                }
                auto rmrm = multiply_cast(rm->get_right());
                if (rmrm.get()) {
                    auto rmrmld = divide_cast(rmrm->get_left());
                    if (rmrmld.get() && rmrmld->get_right()->is_match(ld->get_right())) {
                        return fma(ld->get_left(), this->middle,
                                   rmrmld->get_left()*rmrm->get_right()*rm->get_left())/ld->get_right();
                    }
                    auto rmrmrd = divide_cast(rmrm->get_right());
                    if (rmrmrd.get() && rmrmrd->get_right()->is_match(ld->get_right())) {
                        return fma(ld->get_left(), this->middle,
                                   rmrmrd->get_left()*rmrm->get_left()*rm->get_left())/ld->get_right();
                    }
                }
            }

//  fma(exp(a), exp(b), c) -> exp(a + b) + c
            auto le = exp_cast(this->left);
            auto me = exp_cast(this->middle);
            if (le.get() && me.get()) {
                return exp(le->get_arg() + me->get_arg()) + this->right;
            }

//  fma(exp(a), exp(b)*c, d) -> fma(exp(a)*exp(b), c, d)
//  fma(exp(a), c*exp(b), d) -> fma(exp(a)*exp(b), c, d)
            if (mm.get() && le.get()) {
                auto mmle = exp_cast(mm->get_left());
                if (mmle.get()) {
                    return fma(this->left*mm->get_left(), 
                               mm->get_right(),
                               this->right);
                }
                auto mmre = exp_cast(mm->get_right());
                if (mmre.get()) {
                    return fma(this->left*mm->get_right(), 
                               mm->get_left(),
                               this->right);
                }
            }
//  fma(exp(a)*c, exp(b), d) -> fma(exp(a)*exp(b), c, d)
//  fma(c*exp(a), exp(b), d) -> fma(exp(a)*exp(b), c, d)
            if (lm.get() && me.get()) {
                auto lmle = exp_cast(lm->get_left());
                if (lmle.get()) {
                    return fma(lm->get_left()*this->middle, 
                               lm->get_right(),
                               this->right);
                }
                auto lmre = exp_cast(lm->get_right());
                if (lmre.get()) {
                    return fma(lm->get_right()*this->middle, 
                               lm->get_left(),
                               this->right);
                }
            }

//  fma(exp(a)*c, exp(b)*d, e) -> fma(exp(a)*exp(b), c*d, e)
//  fma(exp(a)*c, d*exp(b), e) -> fma(exp(a)*exp(b), c*d, e)
//  fma(c*exp(a), exp(b)*d, e) -> fma(exp(a)*exp(b), c*d, e)
//  fma(c*exp(a), d*exp(b), e) -> fma(exp(a)*exp(b), c*d, e)
            if (lm.get() && mm.get()) {
                auto lmle = exp_cast(lm->get_left());
                if (lmle.get()) {
                    auto mmle = exp_cast(mm->get_left());
                    if (mmle.get()) {
                        return fma(lm->get_left()*mm->get_left(),
                                   lm->get_right()*mm->get_right(),
                                   this->right);
                    }
                    auto mmre = exp_cast(mm->get_right());
                    if (mmre.get()) {
                        return fma(lm->get_left()*mm->get_right(),
                                   lm->get_right()*mm->get_left(),
                                   this->right);
                    }
                }
                auto lmre = exp_cast(lm->get_right());
                if (lmre.get()) {
                    auto mmle = exp_cast(mm->get_left());
                    if (mmle.get()) {
                        return fma(lm->get_right()*mm->get_left(),
                                   lm->get_left()*mm->get_right(),
                                   this->right);
                    }
                    auto mmre = exp_cast(mm->get_right());
                    if (mmre.get()) {
                        return fma(lm->get_right()*mm->get_right(),
                                   lm->get_left()*mm->get_left(),
                                   this->right);
                    }
                }
            }

//  fma(exp(a)*c, exp(b)/d, e) -> fma(exp(a)*exp(b), c/d, e)
//  fma(exp(a)*c, d/exp(b), e) -> fma(exp(a)/exp(b), c*d, e)
//  fma(c*exp(a), exp(b)/d, e) -> fma(exp(a)*exp(b), c/d, e)
//  fma(c*exp(a), d/exp(b), e) -> fma(exp(a)/exp(b), c*d, e)
            if (lm.get() && md.get()) {
                auto lmle = exp_cast(lm->get_left());
                if (lmle.get()) {
                    auto mdle = exp_cast(md->get_left());
                    if (mdle.get()) {
                        return fma(lm->get_left()*md->get_left(),
                                   lm->get_right()/md->get_right(),
                                   this->right);
                    }
                    auto mdre = exp_cast(md->get_right());
                    if (mdre.get()) {
                        return fma(lm->get_left()/md->get_right(),
                                   lm->get_right()*md->get_left(),
                                   this->right);
                    }
                }
                auto lmre = exp_cast(lm->get_right());
                if (lmre.get()) {
                    auto mdle = exp_cast(md->get_left());
                    if (mdle.get()) {
                        return fma(lm->get_right()*md->get_left(),
                                   lm->get_left()/md->get_right(),
                                   this->right);
                    }
                    auto mdre = exp_cast(md->get_right());
                    if (mdre.get()) {
                        return fma(lm->get_right()/md->get_right(),
                                   lm->get_left()*md->get_left(),
                                   this->right);
                    }
                }
            }

//  fma(exp(a)/c, exp(b)*d, e) -> fma(exp(a)*exp(b), d/c, e)
//  fma(exp(a)/c, d*exp(b), e) -> fma(exp(a)*exp(b), d/c, e)
//  fma(c/exp(a), exp(b)*d, e) -> fma(exp(b)/exp(a), c*d, e)
//  fma(c/exp(a), d*exp(b), e) -> fma(exp(b)/exp(a), c*d, e)
            if (ld.get() && mm.get()) {
                auto ldle = exp_cast(ld->get_left());
                if (ldle.get()) {
                    auto mmle = exp_cast(mm->get_left());
                    if (mmle.get()) {
                        return fma(ld->get_left()*mm->get_left(),
                                   mm->get_right()/ld->get_right(),
                                   this->right);
                    }
                    auto mmre = exp_cast(mm->get_right());
                    if (mmre.get()) {
                        return fma(ld->get_left()*mm->get_right(),
                                   mm->get_left()/ld->get_right(),
                                   this->right);
                    }
                }
                auto ldre = exp_cast(ld->get_right());
                if (ldre.get()) {
                    auto mmle = exp_cast(mm->get_left());
                    if (mmle.get()) {
                        return fma(mm->get_left()/ld->get_right(),
                                   ld->get_left()*mm->get_right(),
                                   this->right);
                    }
                    auto mmre = exp_cast(mm->get_right());
                    if (mmre.get()) {
                        return fma(mm->get_right()/ld->get_right(),
                                   ld->get_left()*mm->get_left(),
                                   this->right);
                    }
                }
            }

//  fma(exp(a)/c, exp(b)/d, e) -> (exp(a)*exp(b))/(c*d) + e
//  fma(exp(a)/c, d/exp(b), e) -> fma(exp(a)/exp(b), d/c, e)
//  fma(c/exp(a), exp(b)/d, e) -> fma(exp(b)/exp(a), c/d, e)
//  fma(c/exp(a), d/exp(b), e) -> (c*d)/(exp(a)*exp(b)) + e
            if (ld.get() && md.get()) {
                auto ldle = exp_cast(ld->get_left());
                if (ldle.get()) {
                    auto mdle = exp_cast(md->get_left());
                    if (mdle.get()) {
                        return ((ld->get_left()*md->get_left()) /
                                (ld->get_right()*md->get_right())) +
                               this->right;
                    }
                    auto mdre = exp_cast(md->get_right());
                    if (mdre.get()) {
                        return fma(ld->get_left()/md->get_right(),
                                   md->get_left()/ld->get_right(),
                                   this->right);
                    }
                }
                auto ldre = exp_cast(ld->get_right());
                if (ldre.get()) {
                    auto mdle = exp_cast(md->get_left());
                    if (mdle.get()) {
                        return fma(md->get_left()/ld->get_right(),
                                   ld->get_left()/md->get_right(),
                                   this->right);
                    }
                    auto mdre = exp_cast(md->get_right());
                    if (mdre.get()) {
                        return ((ld->get_left()*md->get_left()) /
                                (ld->get_right()*md->get_right())) +
                               this->right;
                    }
                }
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d fma(a,b,c)/dx = da*b/dx + dc/dx = da/dx*b + a*db/dx + dc/dx
///
///  @param[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        df(shared_leaf<T, SAFE_MATH> x) {
            if (this->is_match(x)) {
                return one<T, SAFE_MATH> ();
            }

            const size_t hash = reinterpret_cast<size_t> (x.get());
            if (this->df_cache.find(hash) == this->df_cache.end()) {
                auto temp_right = fma(this->left,
                                      this->middle->df(x),
                                      this->right->df(x));

                this->df_cache[hash] = fma(this->left->df(x),
                                           this->middle,
                                           temp_right);
            }
            return this->df_cache[hash];
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @param[in,out] stream    String buffer stream.
///  @param[in,out] registers List of defined registers.
///  @param[in]     usage     List of register usage count.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers,
                const jit::register_usage &usage) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream,
                                                                  registers,
                                                                  usage);
                shared_leaf<T, SAFE_MATH> m = this->middle->compile(stream,
                                                                    registers,
                                                                    usage);
                shared_leaf<T, SAFE_MATH> r = this->right->compile(stream,
                                                                   registers,
                                                                   usage);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = ";
                if constexpr (SAFE_MATH) {
                    stream << "(" << registers[l.get()] << " == ";
                    if constexpr (jit::is_complex<T> ()) {
                        jit::add_type<T> (stream);
                        stream << "(0, 0)";
                    } else {
                        stream << "0";
                    }
                    stream << " || " << registers[m.get()] << " == ";
                    if constexpr (jit::is_complex<T> ()) {
                        jit::add_type<T> (stream);
                        stream << "(0, 0)";
                    } else {
                        stream << "0";
                    }
                    stream << ") ? " << registers[r.get()] << " : ";
                }
                if constexpr (jit::is_complex<T> ()) {
                    stream << registers[l.get()] << "*"
                           << registers[m.get()] << " + "
                           << registers[r.get()];
                } else {
                    stream << "fma("
                           << registers[l.get()] << ", "
                           << registers[m.get()] << ", "
                           << registers[r.get()] << ")";
                }
                this->endline(stream, usage);
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

            auto x_cast = fma_cast(x);
            if (x_cast.get()) {
                return this->left->is_match(x_cast->get_left()) &&
                       this->middle->is_match(x_cast->get_middle()) &&
                       this->right->is_match(x_cast->get_right());
            }

            return false;
        }

//------------------------------------------------------------------------------
///  @brief Convert the node to latex.
//------------------------------------------------------------------------------
        virtual void to_latex() const {
            std::cout << "\\left(";
            if (add_cast(this->left).get() ||
                subtract_cast(this->left).get()) {
                std::cout << "\\left(";
                this->left->to_latex();
                std::cout << "\\right)";
            } else {
                this->left->to_latex();
            }
            std::cout << " ";
            if (add_cast(this->right).get() ||
                subtract_cast(this->right).get()) {
                std::cout << "\\left(";
                this->middle->to_latex();
                std::cout << "\\right)";
            } else {
                this->middle->to_latex();
            }
            std::cout << "+";
            this->right->to_latex();
            std::cout << "\\right)";
        }

//------------------------------------------------------------------------------
///  @brief Remove pseudo variable nodes.
///
///  @returns A tree without variable nodes.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> remove_pseudo() {
            if (this->has_pseudo()) {
                return fma(this->left->remove_pseudo(),
                           this->middle->remove_pseudo(),
                           this->right->remove_pseudo());
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
        virtual shared_leaf<T, SAFE_MATH> to_vizgraph(std::stringstream &stream,
                                                      jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                const std::string name = jit::to_string('r', this);
                registers[this] = name;
                stream << "    " << name
                       << " [label = \"fma\", shape = oval, style = filled, fillcolor = blue, fontcolor = white];" << std::endl;

                auto l = this->left->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[l.get()] << ";" << std::endl;
                auto m = this->middle->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[m.get()] << ";" << std::endl;
                auto r = this->right->to_vizgraph(stream, registers);
                stream << "    " << name << " -- " << registers[r.get()] << ";" << std::endl;
            }

            return this->shared_from_this();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build fused multiply add node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] m Middle branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> fma(shared_leaf<T, SAFE_MATH> l,
                                  shared_leaf<T, SAFE_MATH> m,
                                  shared_leaf<T, SAFE_MATH> r) {
        auto temp = std::make_shared<fma_node<T, SAFE_MATH>> (l, m, r)->reduce();
//  Test for hash collisions.
        for (size_t i = temp->get_hash();
             i < std::numeric_limits<size_t>::max(); i++) {
            if (leaf_node<T, SAFE_MATH>::cache.find(i) ==
                leaf_node<T, SAFE_MATH>::cache.end()) {
                leaf_node<T, SAFE_MATH>::cache[i] = temp;
                return temp;
            } else if (temp->is_match(leaf_node<T, SAFE_MATH>::cache[i])) {
                return leaf_node<T, SAFE_MATH>::cache[i];
            }
        }
#if defined(__clang__) || defined(__GNUC__)
        __builtin_unreachable();
#else
        assert(false && "Should never reach.");
#endif
    }

//------------------------------------------------------------------------------
///  @brief Build divide node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam L         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] m Middle branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar L, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> fma(const L l,
                                  shared_leaf<T, SAFE_MATH> m,
                                  shared_leaf<T, SAFE_MATH> r) {
        return fma<T, SAFE_MATH> (constant<T, SAFE_MATH> (static_cast<T> (l)), m, r);
    }

//------------------------------------------------------------------------------
///  @brief Build divide node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam M         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] m Middle branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar M, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> fma(shared_leaf<T, SAFE_MATH> l,
                                  const M m,
                                  shared_leaf<T, SAFE_MATH> r) {
        return fma<T, SAFE_MATH> (l, constant<T, SAFE_MATH> (static_cast<T> (m)), r);
    }

//------------------------------------------------------------------------------
///  @brief Build multiply node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam R         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] m Middle branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar R, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> fma(shared_leaf<T, SAFE_MATH> l,
                                  shared_leaf<T, SAFE_MATH> m,
                                  const R r) {
        return fma<T, SAFE_MATH> (l, m, constant<T, SAFE_MATH> (static_cast<T> (r)));
    }

//------------------------------------------------------------------------------
///  @brief Build divide node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam L         Float type for the constant.
///  @tparam M         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] m Middle branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar L, jit::float_scalar M, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> fma(const L l,
                                  const M m,
                                  shared_leaf<T, SAFE_MATH> r) {
        return fma<T, SAFE_MATH> (constant<T, SAFE_MATH> (static_cast<T> (l)),
                                  constant<T, SAFE_MATH> (static_cast<T> (m)), r);
    }

//------------------------------------------------------------------------------
///  @brief Build divide node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam M         Float type for the constant.
///  @tparam R         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] m Middle branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar M, jit::float_scalar R, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> fma(shared_leaf<T, SAFE_MATH> l,
                                  const M m,
                                  const R r) {
        return fma<T, SAFE_MATH> (l, constant<T, SAFE_MATH> (static_cast<T> (m)),
                                  constant<T, SAFE_MATH> (static_cast<T> (r)));
    }

//------------------------------------------------------------------------------
///  @brief Build multiply node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @tparam T         Base type of the calculation.
///  @tparam L         Float type for the constant.
///  @tparam R         Float type for the constant.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] l Left branch.
///  @param[in] m Middle branch.
///  @param[in] r Right branch.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, jit::float_scalar L, jit::float_scalar R, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> fma(const L l,
                                  shared_leaf<T, SAFE_MATH> m,
                                  const R r) {
        return fma<T, SAFE_MATH> (constant<T, SAFE_MATH> (static_cast<T> (l)), m,
                                  constant<T, SAFE_MATH> (static_cast<T> (r)));
    }

///  Convenience type alias for shared add nodes.
    template<jit::float_scalar T, bool SAFE_MATH=false>
    using shared_fma = std::shared_ptr<fma_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a fma node.
///
///  @tparam T         Base type of the calculation.
///  @tparam SAFE_MATH Use safe math operations.
///
///  @param[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<jit::float_scalar T, bool SAFE_MATH=false>
    shared_fma<T, SAFE_MATH> fma_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<fma_node<T, SAFE_MATH>> (x);
    }
}

#endif /* arithmetic_h */
