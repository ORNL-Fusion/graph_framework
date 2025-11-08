//------------------------------------------------------------------------------
///  @file graph_framework.hpp
///  @brief Single include header for the entire framework.
//------------------------------------------------------------------------------

#ifndef graph_framework_h
#define graph_framework_h

#include "absorption.hpp"
#include "arithmetic.hpp"
#include "backend.hpp"
#include "commandline_parser.hpp"
#include "cpu_context.hpp"
#include "dispersion.hpp"
#include "equilibrium.hpp"
#include "jit.hpp"
#include "math.hpp"
#include "newton.hpp"
#include "node.hpp"
#include "output.hpp"
#include "piecewise.hpp"
#include "random.hpp"
#include "register.hpp"
#include "solver.hpp"
#include "special_functions.hpp"
#include "timing.hpp"
#include "trigonometry.hpp"
#include "vector.hpp"
#include "workflow.hpp"

#ifdef USE_CUDA
#include "cuda_context.hpp"
#endif
#ifdef USE_METAL
#include "metal_context.hpp"
#endif
#ifdef USE_HIP
#include "hip_context.hpp"
#endif

#endif /* graph_framework_h */
