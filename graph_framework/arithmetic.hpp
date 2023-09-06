//------------------------------------------------------------------------------
///  @file arithmetic.hpp
///  @brief Basic arithmetic operations.
///
///  Defines basic operators.
//------------------------------------------------------------------------------

#ifndef arithmetic_h
#define arithmetic_h

#include "node.hpp"

namespace graph {
//******************************************************************************
//  Add node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief An addition node.
///
///  Note use templates here to defer this so it can use the operator functions.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    class add_node final : public branch_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @params[in] l Left node pointer.
///  @params[in] r Right node pointer.
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
///  @params[in] l Left branch.
///  @params[in] r Right branch.
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

//  Idenity reductions.
            if (this->left->is_match(this->right)) {
                return two<T, SAFE_MATH> ()*this->left;
            }
            
//  Common factor reduction. If the left and right are both muliply nodes check
//  for a common factor. So you can change a*b + a*c -> a*(b + c).
            auto lm = multiply_cast(this->left);
            auto rm = multiply_cast(this->right);

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

            if (ld.get() && rd.get()) {
                if (ld->get_right()->is_match(rd->get_right())) {
                    return (ld->get_left() + rd->get_left())/ld->get_right();
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
            }

//  Chained addition reductions.
//  a + (a + b) = fma(2,a,b)
//  a + (b + a) = fma(2,a,b)
//  (a + b) + a = fma(2,a,b)
//  (b + a) + a = fma(2,a,b)
            auto la = add_cast(this->left);
            if (la.get()) {
                if (this->right->is_match(la->get_left())) {
                    return fma(two<T, SAFE_MATH> (), this->right, la->get_right());
                } else if (this->right->is_match(la->get_right())) {
                    return fma(two<T, SAFE_MATH> (), this->right, la->get_left());
                }
            }
            auto ra = add_cast(this->right);
            if (ra.get()) {
                if (this->left->is_match(ra->get_left())) {
                    return fma(two<T, SAFE_MATH> (), this->left, ra->get_right());
                } else if (this->left->is_match(ra->get_right())) {
                    return fma(two<T, SAFE_MATH> (), this->left, ra->get_left());
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
                return fma(lfma->get_left(), lfma->get_middle(),
                           lfma->get_right() + this->right);
            } else if (rfma.get()) {
//  a + fma(c,d,e) -> fma(c,d,a + e)
                return fma(rfma->get_left(), rfma->get_middle(),
                           this->left + rfma->get_right());
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
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        df(shared_leaf<T, SAFE_MATH> x) {
            if (this->is_match(x)) {
                return one<T, SAFE_MATH> ();
            } else {
                return this->left->df(x) + this->right->df(x);
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream, registers);
                shared_leaf<T, SAFE_MATH> r = this->right->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = "
                       << registers[l.get()] << " + "
                       << registers[r.get()] << ";"
                       << std::endl;
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @params[in] x Other graph to check if it is a match.
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
            return this->left->remove_pseudo() +
                   this->right->remove_pseudo();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build add node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
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
        assert(false && "Should never reach.");
    }

//------------------------------------------------------------------------------
///  @brief Build add node from two leaves.
///
///  Note use templates here to defer this so it can be used in the above
///  classes.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator+(shared_leaf<T, SAFE_MATH> l,
                                        shared_leaf<T, SAFE_MATH> r) {
        return add<T, SAFE_MATH> (l, r);
    }

///  Convenience type alias for shared add nodes.
    template<typename T, bool SAFE_MATH=false>
    using shared_add = std::shared_ptr<add_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a add node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
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
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    class subtract_node final : public branch_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @params[in] l Left node pointer.
///  @params[in] r Right node pointer.
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
///  @params[in] l Left branch.
///  @params[in] r Right branch.
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
                return none<T, SAFE_MATH> ()*this->right;
            } else if (r.get() && r->is(0)) {
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

//  Common factor reduction. If the left and right are both muliply nodes check
//  for a common factor. So you can change a*b - a*c -> a*(b - c).
            auto lm = multiply_cast(this->left);
            auto rm = multiply_cast(this->right);

//  Assume constants are on the left.
//  v1 - -1*v2 -> v1 + v2
            if (rm.get()) {
                auto rmc = constant_cast(rm->get_left());
                if (rmc.get() && rmc->evaluate().is_none()) {
                    return this->left + rm->get_right();
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

//  Change cases like c1*a - c2*b -> c1*(a - c2*b)
                auto lmc = constant_cast(lm->get_left());
                auto rmc = constant_cast(rm->get_left());
                if (lmc.get() && rmc.get()) {
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
                        return (lm->get_left()*lmrm->get_left() - lm->get_left())*rm->get_right();
                    }
//  c*(b*d) - a*b -> (c*d - a)*b
                    if (rm->get_right()->is_match(lmrm->get_left())) {
                        return (lm->get_left()*lmrm->get_right() - lm->get_left())*rm->get_right();
                    }
//  c*(d*b) - b*a -> (c*d - a)*b
                    if (rm->get_left()->is_match(lmrm->get_right())) {
                        return (lm->get_left()*lmrm->get_left() - lm->get_right())*rm->get_left();
                    }
//  c*(b*d) - b*a -> (c*d - a)*b
                    if (rm->get_left()->is_match(lmrm->get_left())) {
                        return (lm->get_left()*lmrm->get_right() - lm->get_right())*rm->get_left();
                    }
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
                               (lrm->get_right() + rm->get_right())*rm->get_left();
                    } else if (lrm->get_left()->is_match(rm->get_right())) {
//  (a - c*b) - d*c -> a - (b + d)*c
                        return ls->get_left() -
                               (lrm->get_right() + rm->get_left())*rm->get_right();
                    } else if (lrm->get_right()->is_match(rm->get_left())) {
//  (a - c*b) - c*d -> a - (b + d)*c
                        return ls->get_left() -
                               (lrm->get_left() + rm->get_right())*rm->get_left();
                    } else if (lrm->get_right()->is_match(rm->get_right())) {
//  (a - c*b) - d*c -> a - (b + d)*c
                        return ls->get_left() -
                               (lrm->get_left() + rm->get_left())*rm->get_right();
                    }
                }
            }

//  Common denominator reduction. If the left and right are both divide nodes
//  for a common denominator. So you can change a/b - c/b -> (a - c)/d.
            auto ld = divide_cast(this->left);
            auto rd = divide_cast(this->right);

            if (ld.get() && rd.get() &&
                ld->get_right()->is_match(rd->get_right())) {
                return (ld->get_left() - rd->get_left())/ld->get_right();
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
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        df(shared_leaf<T, SAFE_MATH> x) {
            if (this->is_match(x)) {
                return one<T, SAFE_MATH> ();
            } else {
                return this->left->df(x) - this->right->df(x);
            }
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream, registers);
                shared_leaf<T, SAFE_MATH> r = this->right->compile(stream, registers);

                registers[this] = jit::to_string('r', this);
                stream << "        const ";
                jit::add_type<T> (stream);
                stream << " " << registers[this] << " = "
                       << registers[l.get()] << " - "
                       << registers[r.get()] << ";"
                       << std::endl;
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @params[in] x Other graph to check if it is a match.
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
            return this->left->remove_pseudo() -
                   this->right->remove_pseudo();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build subtract node from two leaves.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
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
        assert(false && "Should never reach.");
    }

//------------------------------------------------------------------------------
///  @brief Build subtract operator from two leaves.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator-(shared_leaf<T, SAFE_MATH> l,
                                        shared_leaf<T, SAFE_MATH> r) {
        return subtract<T, SAFE_MATH> (l, r);
    }

///  Convenience type alias for shared subtract nodes.
    template<typename T, bool SAFE_MATH=false>
    using shared_subtract = std::shared_ptr<subtract_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a subtract node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    shared_subtract<T, SAFE_MATH> subtract_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<subtract_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Multiply node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A multiplcation node.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    class multiply_node final : public branch_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @params[in] l Left node pointer.
///  @params[in] r Right node pointer.
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
///  @params[in] l Left branch.
///  @params[in] r Right branch.
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

//  Move constants to the left.
            if (r.get() && !l.get()) {
                return this->right*this->left;
            }

//  Move piecewise constants to the left.
            auto lpw1d = piecewise_1D_cast(this->left);
            auto rpw1d = piecewise_1D_cast(this->right);
            auto lpw2d = piecewise_2D_cast(this->left);
            auto rpw2d = piecewise_2D_cast(this->right);
            if ((rpw1d.get() || rpw2d.get()) &&
                (!lpw1d.get() && !lpw2d.get() && !l.get())) {
                return this->right*this->left;
            }

//  Move constant like to the left.
            if (this->right->is_constant_like() &&
                !this->left->is_constant_like()) {
                return this->right*this->left;
            }

//  Move variables, sqrt of variables, and powers of variables to the right.
//  Disable if the left is a constant like to avoid an infinite loop.
            if (this->left->is_power_like()   &&
                !this->right->is_power_like() &&
                !this->left->is_constant_like()) {
                return this->right*this->left;
            }

//  Disable if the right is power like to avoid infinite loop.
            if (this->left->is_all_variables()   &&
                !this->right->is_all_variables() &&
                !this->right->is_power_like()) {
                return this->right*this->left;
            }

//  Reduce x*x to x^2
            if (this->left->is_match(this->right)) {
                return pow(this->left, two<T, SAFE_MATH> ());
            }

//  Gather common terms.
//  (a*b)*a -> (a*a)*b
//  (b*a)*a -> (a*a)*b
//  a*(a*b) -> (a*a)*b
//  a*(b*a) -> (a*a)*b
            auto lm = multiply_cast(this->left);
            if (lm.get()) {
                if (this->right->is_match(lm->get_left())) {
                    return (this->right*lm->get_left())*lm->get_right();
                } else if (this->right->is_match(lm->get_right())) {
                    return (this->right*lm->get_right())*lm->get_left();
                }

//  Promote constants before variables.
//  (c*v1)*v2 -> c*(v1*v2)
                if (lm->get_left()->is_constant_like()) {
                    return lm->get_left()*(lm->get_right()*this->right);
                }

//  Assume variables, sqrt of variables, and powers of variables are on the
//  right.
//  (a*v)*b -> (a*b)*v
                if (lm->get_right()->is_power_like() &&
                    !(this->right->is_power_like() ||
                      this->right->is_all_variables())) {
                    return (lm->get_left()*this->right)*lm->get_right();
                }
                if (lm->get_right()->is_all_variables() &&
                    !(this->right->is_power_like()   ||
                      this->right->is_all_variables())) {
                    return (lm->get_left()*this->right)*lm->get_right();
                }
            }

            auto rm = multiply_cast(this->right);
            if (rm.get()) {
//  Assume constants are on the left.
//  c1*(c2*v) -> c3*v
                if (constant_cast(rm->get_left()).get() && l.get()) {
                    return (this->left*rm->get_left())*rm->get_right();
                }

                if (this->left->is_match(rm->get_left())) {
                    return (this->left*rm->get_left())*rm->get_right();
                } else if (this->left->is_match(rm->get_right())) {
                    return (this->left*rm->get_right())*rm->get_left();
                }
            }

//  v1*(c*v2) -> c*(v1*v2)
            if (rm.get() && constant_cast(rm->get_left()).get()) {
                return rm->get_left()*(this->left*rm->get_right());
            }

//  Factor out common constants c*b*c*d -> c*c*b*d. c*c will get reduced to c on
//  the second pass.
            if (lm.get() && rm.get()) {
                if (constant_cast(lm->get_left()).get() &&
                    constant_cast(rm->get_left()).get()) {
                    return (lm->get_left()*rm->get_left()) *
                           (lm->get_right()*rm->get_right());
                } else if (constant_cast(lm->get_left()).get() &&
                           constant_cast(rm->get_right()).get()) {
                    return (lm->get_left()*rm->get_right()) *
                           (lm->get_right()*rm->get_left());
                } else if (constant_cast(lm->get_right()).get() &&
                           constant_cast(rm->get_left()).get()) {
                    return (lm->get_right()*rm->get_left()) *
                           (lm->get_left()*rm->get_right());
                } else if (constant_cast(lm->get_right()).get() &&
                           constant_cast(rm->get_right()).get()) {
                    return (lm->get_right()*rm->get_right()) *
                           (lm->get_left()*rm->get_left());
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

            if (ld.get()) {
//  (c/v1)*v2 -> c*(v2/v1)
                if (constant_cast(ld->get_left()).get()     ||
                    piecewise_1D_cast(ld->get_left()).get() ||
                    piecewise_2D_cast(ld->get_left()).get()) {
                    return ld->get_left()*(this->right/ld->get_right());
                }
            }

//  c1*(c2/v) -> c3/v
//  c1*(v/c2) -> v/c3
            if (rd.get() && l.get()) {
                if (constant_cast(rd->get_left()).get()) {
                    return (this->left*rd->get_left())/rd->get_right();
                } else if (constant_cast(rd->get_right()).get()) {
                    return rd->get_left()/(this->left*rd->get_right());
                }
            }

            if (ld.get() && rd.get()) {
                if (ld->get_left()->is_match(rd->get_right())) {
                    return ld->get_right()/rd->get_left();
                } else if (ld->get_right()->is_match(rd->get_left())) {
                    return ld->get_left()/rd->get_right();
                }

//  Convert (a/b)*(c/d) -> (a*c)/(b*d). This should help reduce cases like.
//  (a/b)*(a/b) + (c/b)*(c/b).
                return (ld->get_left()*rd->get_left()) /
                       (ld->get_right()*rd->get_right());
            }

//  Power reductions.
            if (this->left->is_power_base_match(this->right)) {
                return pow(this->left->get_power_base(),
                           this->left->get_power_exponent() +
                           this->right->get_power_exponent());
            }
//  (a*b^c)*b^d -> a*b^(c + d)
            if (lm.get() &&
                lm->get_right()->is_power_base_match(this->right)) {
                return lm->get_left()*pow(this->right->get_power_base(),
                                          lm->get_right()->get_power_exponent() +
                                          this->right->get_power_exponent());
            }

//  a*b^-c -> a/b^c
            auto rp = pow_cast(this->right);
            if (rp.get()) {
                auto exponent = constant_cast(rp->get_right());
                if (exponent.get() && exponent->evaluate().is_negative()) {
                    return this->left/pow(rp->get_left(),
                                          none<T, SAFE_MATH> ()*rp->get_right());
                }
            }
//  b^-c*a -> a/b^c
            auto lp = pow_cast(this->left);
            if (lp.get()) {
                auto exponent = constant_cast(lp->get_right());
                if (exponent.get() && exponent->evaluate().is_negative()) {
                    return this->right/pow(lp->get_left(),
                                           none<T, SAFE_MATH> ()*lp->get_right());
                }
            }

            auto lpd = divide_cast(this->left->get_power_base());
            if (lpd.get()) {
//  (a/b)^c*b^d -> a^c*b^(c-d)
                if (lpd->get_right()->is_power_base_match(this->right)) {
                    return pow(lpd->get_left(), this->left->get_power_exponent()) *
                           pow(this->right->get_power_base(),
                               this->right->get_power_exponent() -
                               this->left->get_power_exponent()*lpd->get_right()->get_power_exponent());
                }
//  (b/a)^c*b^d -> b^(c+d)/a^c
                if (lpd->get_left()->is_power_base_match(this->right)) {
                    return pow(this->right->get_power_base(),
                               this->right->get_power_exponent() +
                               this->left->get_power_exponent()*lpd->get_left()->get_power_exponent()) /
                           pow(lpd->get_right(), this->left->get_power_exponent());
                }
            }
            auto rpd = divide_cast(this->right->get_power_base());
            if (rpd.get()) {
//  b^d*(a/b)^c -> a^c*b^(c-d)
                if (rpd->get_right()->is_power_base_match(this->left)) {
                    return pow(rpd->get_left(), this->right->get_power_exponent()) *
                           pow(this->left->get_power_base(),
                               this->left->get_power_exponent() -
                               this->right->get_power_exponent()*rpd->get_right()->get_power_exponent());
                }
//  b^d*(b/a)^c -> b^(c+d)/a^c
                if (rpd->get_left()->is_power_base_match(this->left)) {
                    return pow(this->right->get_power_base(),
                               this->right->get_power_exponent() +
                               this->right->get_power_exponent()*rpd->get_left()->get_power_exponent()) /
                           pow(rpd->get_right(), this->right->get_power_exponent());
                }
            }

//  Exp(a)*Exp(b) -> Exp(a + b)
            auto le = exp_cast(this->left);
            auto re = exp_cast(this->right);
            if (le.get() && re.get()) {
                return exp(le->get_arg() + re->get_arg());
            }

//  Exp(a)*(Exp(b)*c) -> (Exp(a)*Exp(b))*c
//  Exp(a)*(c*Exp(b)) -> (Exp(a)*Exp(b))*c
            if (le.get() && rm.get()) {
                auto rmle = exp_cast(rm->get_left());
                if (rmle.get()) {
                    return (this->left*rm->get_left())*rm->get_right();
                }
                auto rmre = exp_cast(rm->get_right());
                if (rmre.get()) {
                    return (this->left*rm->get_right())*rm->get_left();
                }
            }
//  (Exp(a)*c)*Exp(b) -> (Exp(a)*Exp(b))*c
//  (c*Exp(a))*Exp(b) -> (Exp(a)*Exp(b))*c
            if (re.get() && lm.get()) {
                auto lmle = exp_cast(lm->get_left());
                if (lmle.get()) {
                    return (this->right*lm->get_left())*lm->get_right();
                }
                auto lmre = exp_cast(rm->get_right());
                if (lmre.get()) {
                    return (this->right*lm->get_right())*lm->get_left();
                }
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d mul(a,b)/dx = da/dx*b + a*db/dx
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH> df(shared_leaf<T, SAFE_MATH> x) {
            if (this->is_match(x)) {
                return one<T, SAFE_MATH> ();
            }

            return this->left->df(x)*this->right +
                   this->left*this->right->df(x);
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream, registers);
                shared_leaf<T, SAFE_MATH> r = this->right->compile(stream, registers);

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
                       << registers[r.get()] << ";"
                       << std::endl;
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @params[in] x Other graph to check if it is a match.
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
            return this->left->remove_pseudo() *
                   this->right->remove_pseudo();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build multiply node from two leaves.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
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
        assert(false && "Should never reach.");
    }

//------------------------------------------------------------------------------
///  @brief Build multiply operator from two leaves.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator*(shared_leaf<T, SAFE_MATH> l,
                                        shared_leaf<T, SAFE_MATH> r) {
        return multiply<T, SAFE_MATH> (l, r);
    }

///  Convenience type alias for shared multiply nodes.
    template<typename T, bool SAFE_MATH=false>
    using shared_multiply = std::shared_ptr<multiply_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a multiply node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    shared_multiply<T, SAFE_MATH> multiply_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<multiply_node<T, SAFE_MATH>> (x);
    }

//******************************************************************************
//  Divide node.
//******************************************************************************
//------------------------------------------------------------------------------
///  @brief A division node.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    class divide_node final : public branch_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @params[in] l Left node pointer.
///  @params[in] r Right node pointer.
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
///  @params[in] n Numerator branch.
///  @params[in] d Denominator branch.
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

            if (this->left->is_match(this->right)) {
                if (l.get() && l->is(1)) {
                    return this->left;
                }

                return one<T, SAFE_MATH> ();
            }

//  Reduce cases of a/c1 -> c2*a
            if (r.get()) {
                return (one<T, SAFE_MATH> ()/this->right) *
                       this->left;
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

//  Assume constants are always on the left.
//  c1/(c2*v) -> c3/v
//  (c1*v)/c2 -> c3*v
            if (rm.get() && l.get()) {
                if (constant_cast(rm->get_left()).get()) {
                    return (this->left/rm->get_left())/rm->get_right();
                }
            } else if (lm.get() && r.get()) {
                if (constant_cast(lm->get_left()).get()) {
                    return (lm->get_left()/this->right)*lm->get_right();
                }
            }

            if (lm.get() && rm.get()) {
//  Test for constants that can be reduced out.
                if (constant_cast(lm->get_left()).get() &&
                    constant_cast(rm->get_left()).get()) {
                    return (lm->get_left()/rm->get_left())*(lm->get_right()/rm->get_right());
                } else if (constant_cast(lm->get_left()).get() &&
                           constant_cast(rm->get_right()).get()) {
                    return (lm->get_left()/rm->get_right())*(lm->get_right()/rm->get_left());
                } else if (constant_cast(lm->get_right()).get() &&
                           constant_cast(rm->get_left()).get()) {
                    return (lm->get_right()/rm->get_left())*(lm->get_left()/rm->get_right());
                } else if (constant_cast(lm->get_right()).get() &&
                           constant_cast(rm->get_right()).get()) {
                    return (lm->get_right()/rm->get_right())*(lm->get_left()/rm->get_left());
                }

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
                if (lm->get_left()->is_power_base_match(this->right)) {
                    return lm->get_right()*(lm->get_left()/this->right);
                } else if (lm->get_right()->is_power_base_match(this->right)) {
                    return lm->get_left()*(lm->get_right()/this->right);
                }
            }

//  (a/b)/c -> a/(b*c)
            auto ld = divide_cast(this->left);
            if (ld.get()) {
                return ld->get_left()/(ld->get_right()*this->right);
            }

//  Assume variables, sqrt of variables, and powers of variables are on the
//  right.
//  (a*v)/c -> (a/c)*v
            if (lm.get() && lm->get_right()->is_all_variables() &&
                !lm->get_left()->is_all_variables()) {
                return (lm->get_left()/this->right)*lm->get_right();
            }

//  (c*v1)/v2 -> c*(v1/v2)
            if (lm.get() && constant_cast(lm->get_left()).get()) {
                return lm->get_left()*(lm->get_right()/this->right);
            }

            if (lm.get() && lm->get_left()->is_constant_like()) {
                return lm->get_left()*(lm->get_right()/this->right);
            }

//  Power reductions.
            if (this->left->is_power_base_match(this->right)) {
                return pow(this->left->get_power_base(),
                           this->left->get_power_exponent() -
                           this->right->get_power_exponent());
            }

//  a/b^-c -> a*b^c
            auto rp = pow_cast(this->right);
            if (rp.get()) {
                auto exponent = constant_cast(rp->get_right());
                if (exponent.get() && exponent->evaluate().is_negative()) {
                    return this->left*pow(rp->get_left(),
                                          none<T, SAFE_MATH> ()*rp->get_right());
                }
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d div(n,d)/dx = dn/dx*1/d - n*/(d*d)*db/dx
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        df(shared_leaf<T, SAFE_MATH> x) {
            if (this->is_match(x)) {
                return one<T, SAFE_MATH> ();
            }

            return this->left->df(x)/this->right -
                   this->left*this->right->df(x)/(this->right*this->right);
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream, registers);
                shared_leaf<T, SAFE_MATH> r = this->right->compile(stream, registers);

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
                       << registers[r.get()] << ";"
                       << std::endl;
            }
            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @params[in] x Other graph to check if it is a match.
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
            return this->left->remove_pseudo() /
                   this->right->remove_pseudo();
        }
    };

//------------------------------------------------------------------------------
///  @brief Build divide node from two leaves.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
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
        assert(false && "Should never reach.");
    }

//------------------------------------------------------------------------------
///  @brief Build divide operator from two leaves.
///
///  @params[in] l Left branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    shared_leaf<T, SAFE_MATH> operator/(shared_leaf<T, SAFE_MATH> l,
                                        shared_leaf<T, SAFE_MATH> r) {
        return divide<T, SAFE_MATH> (l, r);
    }

///  Convenience type alias for shared divide nodes.
    template<typename T, bool SAFE_MATH=false>
    using shared_divide = std::shared_ptr<divide_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a divide node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
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
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    class fma_node final : public triple_node<T, SAFE_MATH> {
    private:
//------------------------------------------------------------------------------
///  @brief Convert node pointer to a string.
///
///  @params[in] l Left node pointer.
///  @params[in] m Middle node pointer.
///  @params[in] r Right node pointer.
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
///  @params[in] l Left branch.
///  @params[in] m Middle branch.
///  @params[in] r Right branch.
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
            } else if (l.get() && l->evaluate().is_none()) {
                return this->right - this->middle;
            } else if (m.get() && m->evaluate().is_none()) {
                return this->right - this->left;
            }

            auto pl1 = piecewise_1D_cast(this->left);
            auto pm1 = piecewise_1D_cast(this->middle);
            auto pl2 = piecewise_2D_cast(this->left);
            auto pm2 = piecewise_2D_cast(this->middle);

            if ((pl1.get() && (m.get() || pl1->is_arg_match(this->middle))) ||
                (pm1.get() && (l.get() || pm1->is_arg_match(this->left)))   ||
                (pl2.get() && (m.get() || pl2->is_arg_match(this->middle))) ||
                (pm2.get() && (l.get() || pm2->is_arg_match(this->left)))) {
                return (this->left*this->middle) + this->right;
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

//  Change cases like c1*a + c2*b -> c1*(c3*b + a)
                auto rmc = constant_cast(rm->get_left());
                if (rmc.get() && l.get()) {
                    return this->left*fma(rm->get_left()/this->left,
                                          rm->get_right(),
                                          this->middle);
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
            if (lm.get() && rm.get()) {
                auto rmc = constant_cast(rm->get_left());
                if (rmc.get()) {
                    return lm->get_left()*fma(lm->get_right(),
                                              this->middle,
                                              (rm->get_left()/lm->get_left())*rm->get_right());
                }
            }

//  Move constant multiplies to the left.
            if (lm.get()) {
                auto lmc = constant_cast(lm->get_left());
                if (lmc.get()) {
                    return fma(lm->get_left(),
                               lm->get_right()*this->middle,
                               this->right);
                }
            } else if (mm.get()) {
                auto mmc = constant_cast(mm->get_left());
                auto mmpw1c = piecewise_1D_cast(mm->get_left());
                auto mmpw2c = piecewise_2D_cast(mm->get_left());
                if (mmc.get() || mmpw1c.get() || mmpw2c.get()) {
                    if (l.get() || pl1.get() || pl2.get()) {
                        return fma(this->left*mm->get_left(),
                                   mm->get_right(),
                                   this->right);
                    } else {
                        return fma(mm->get_left(),
                                   this->left*mm->get_right(),
                                   this->right);
                    }
                }
            }

//  fma(c1,a,c2/b) -> c1*(a + c1/(c2*b))
//  fma(c1,a,b/c2) -> c1*(a + b/(c1*c2))
            auto rd = divide_cast(this->right);
            if (l.get() && rd.get()) {
                if (constant_cast(rd->get_left()).get() ||
                    constant_cast(rd->get_right()).get()) {
                    return this->left*(this->middle +
                                       rd->get_left()/(this->left*rd->get_right()));
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

//  Special case divide by variables.

//  Move fma(a/c, b, e) -> fma(a,b,e*c)/c
//  Move fma(a, b/c, e) -> fma(a,b,e*c)/c
            if (ld.get() && ld->get_left()->is_all_variables()) {
                auto temp = fma(ld->get_left(), this->middle,
                                this->right*ld->get_right())/ld->get_right();
                if (temp->get_complexity() < this->get_complexity()) {
                    return temp;
                }
            }
            if (md.get() && md->get_left()->is_all_variables()) {
                auto temp = fma(this->left, md->get_left(),
                                this->right*md->get_right())/md->get_right();
                if (temp->get_complexity() < this->get_complexity()) {
                    return temp;
                }
            }

//  Chained fma reductions.
            auto rfma = fma_cast(this->right);
            if (rfma.get()) {
//  fma(a,b,fma(a,b,c)) -> fma(2*a,b,c)
//  fma(a,b,fma(b,a,c)) -> fma(2*a,b,c)
                if (this->left->is_match(rfma->get_left()) &&
                    this->middle->is_match(rfma->get_middle())) {
                    return fma(two<T, SAFE_MATH> ()*this->left, this->middle,
                               rfma->get_right());
                } else if (this->left->is_match(rfma->get_middle()) &&
                           this->middle->is_match(rfma->get_left())) {
                    return fma(two<T, SAFE_MATH> ()*this->left, this->middle,
                               rfma->get_right());
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
//  restricted to variable like nodes. Only do this reduction is the complexity
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

//  Promote constants out to the left.
            if (l.get() && r.get()) {
                return this->left*(this->middle + this->right/this->left);
            }


//  Change negative eponents to divide so that can be factored out.
//  fma(a,b^-c,d) = a/b^c + d
//  fma(b^-c,a,d) = a/b^c + d
            auto lp = pow_cast(this->left);
            if (lp.get()) {
                auto exponent = constant_cast(lp->get_right());
                if (exponent.get() && exponent->evaluate().is_negative()) {
                    return this->middle/pow(lp->get_left(),
                                            none<T, SAFE_MATH> ()*lp->get_right()) + this->right;
                }
            }
            auto mp = pow_cast(this->middle);
            if (mp.get()) {
                auto exponent = constant_cast(mp->get_right());
                if (exponent.get() && exponent->evaluate().is_negative()) {
                    return this->left/pow(mp->get_left(),
                                            none<T, SAFE_MATH> ()*mp->get_right()) + this->right;
                }
            }

//  fma(a,b/c,b/d) -> b*(a/c + 1/d)
//  fma(a,c/b,d/b) -> (a*c + d)/b
            if (md.get() && rd.get()) {
                if (md->get_left()->is_match(rd->get_left())) {
                    return md->get_left()*(this->left/md->get_right() +
                                           one<T, SAFE_MATH> ()/rd->get_right());
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
                                           one<T, SAFE_MATH> ()/rd->get_right());
                } else if (ld->get_right()->is_match(rd->get_right())) {
                    return (this->middle*ld->get_left() +
                            rd->get_left())/ld->get_right();
                }
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Transform node to derivative.
///
///  d fma(a,b,c)/dx = da*b/dx + dc/dx = da/dx*b + a*db/dx + dc/dx
///
///  @params[in] x The variable to take the derivative to.
///  @returns The derivative of the node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        df(shared_leaf<T, SAFE_MATH> x) {
            if (this->is_match(x)) {
                return one<T, SAFE_MATH> ();
            }

            auto temp_right = fma(this->left,
                                  this->middle->df(x),
                                  this->right->df(x));

            return fma(this->left->df(x),
                       this->middle,
                       temp_right);
        }

//------------------------------------------------------------------------------
///  @brief Compile the node.
///
///  @params[in,out] stream    String buffer stream.
///  @params[in,out] registers List of defined registers.
///  @returns The current node.
//------------------------------------------------------------------------------
        virtual shared_leaf<T, SAFE_MATH>
        compile(std::ostringstream &stream,
                jit::register_map &registers) {
            if (registers.find(this) == registers.end()) {
                shared_leaf<T, SAFE_MATH> l = this->left->compile(stream, registers);
                shared_leaf<T, SAFE_MATH> m = this->middle->compile(stream, registers);
                shared_leaf<T, SAFE_MATH> r = this->right->compile(stream, registers);

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
                           << registers[r.get()] << ";"
                           << std::endl;
                } else {
                    stream << "fma("
                           << registers[l.get()] << ", "
                           << registers[m.get()] << ", "
                           << registers[r.get()] << ");"
                           << std::endl;
                }
            }

            return this->shared_from_this();
        }

//------------------------------------------------------------------------------
///  @brief Querey if the nodes match.
///
///  @params[in] x Other graph to check if it is a match.
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
            return fma(this->left->remove_pseudo(),
                       this->middle->remove_pseudo(),
                       this->right->remove_pseudo());
        }
    };

//------------------------------------------------------------------------------
///  @brief Build fused multiply add node.
///
///  @params[in] l Left branch.
///  @params[in] m Middle branch.
///  @params[in] r Right branch.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
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
        assert(false && "Should never reach.");
    }

///  Convenience type alias for shared add nodes.
    template<typename T, bool SAFE_MATH=false>
    using shared_fma = std::shared_ptr<fma_node<T, SAFE_MATH>>;

//------------------------------------------------------------------------------
///  @brief Cast to a fma node.
///
///  @params[in] x Leaf node to attempt cast.
///  @returns An attemped dynamic case.
//------------------------------------------------------------------------------
    template<typename T, bool SAFE_MATH=false>
    shared_fma<T, SAFE_MATH> fma_cast(shared_leaf<T, SAFE_MATH> x) {
        return std::dynamic_pointer_cast<fma_node<T, SAFE_MATH>> (x);
    }
}

#endif /* arithmetic_h */
