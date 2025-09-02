!-------------------------------------------------------------------------------
!>  @file graph_fortran_binding.f90
!>  @brief Implimentation of the Fortran binding library.
!-------------------------------------------------------------------------------
!>  @page graph_fortran_binding Embedding in Fortran code
!>  @brief Documentation for linking into Fortran code base.
!>  @tableofcontents
!>
!>  @section graph_fortran_binding_into Introduction
!>  This section assumes the reader is already familar with developing Fortran
!>  codes. The simplist method is to create a C callable function like the
!>  @ref graph_c_binding_into "C binding exmaple". Then create a fortran
!>  interface for it.
!>  @code
!>  INTERFACE
!>     SUBROUTINE Fortran_Callable BIND(C, NAME='c_callable_function')
!>     USE, INTRINSIC :: ISO_C_BINDING
!>     IMPLICIT NONE
!>     END SUBROUTINE
!>  END INTERFACE
!>  @endcode
!>  This subroutine can be called like any other Fortran subroutine.
!>  @code
!>  CALL Fortran_Callable
!>  @endcode
!>
!>  <hr>
!>  @section graph_fortran_binding_interface Fortran Binding Interface
!>  An alternative is to use the
!>  @ref graph_fortran "Fortran Language interface". The Fortran binding
!>  interface can be enabled as one of the <tt>cmake</tt>
!>  @ref build_system_user_options "conifgure options". As an example, we will
!>  convert the @ref tutorial_workflow "making workflows" turorial to use the
!>  Fortran language bindings.
!>  @code
!>  SUBROUTINE fortran_binding
!>  USE graph_fortran
!>  USE, INTRINSIC :: ISO_C_BINDING
!>
!>  IMPLICIT NONE
!>
!>  CLASS(graph_context), POINTER :: graph
!>  TYPE(C_PTR)                   :: x
!>  TYPE(C_PTR)                   :: m
!>  TYPE(C_PTR)                   :: b
!>  TYPE(C_PTR)                   :: y
!>  TYPE(C_PTR)                   :: dydx
!>
!>  LOGICAL(C_BOOL), PARAMETER :: use_safe_math = .false.
!>
!>  graph => graph_double_context(use_safe_math);
!>
!>  x = graph%variable(1_C_LONG, 'x' // C_NULL_CHAR)
!>  m = graph%constant(0.4_C_DOUBLE)
!>  b = graph%constant(0.6_C_DOUBLE)
!>
!>  y = graph%add(graph%mul(m, x), b)
!>  dydx = graph%df(y, x);
!>
!>  CALL graph%set_variable(x, (/ 1.0_C_DOUBLE, 2.0_C_DOUBLE, 3.0_C_DOUBLE /))
!>
!>  CALL graph%set_device_number(0)
!>  CALL graph%add_item((/ graph_ptr(x) /),                                    &
!>                      (/ graph_ptr(y), graph_ptr(dydx) /))                   &
!>                      graph_null_array, graph_null_array, C_NULL_PTR,        &
!>                      'my_first_kernel' // C_NULL_CHAR, 3_C_LONG)
!>  CALL graph%compile
!>  CALL graph%run
!>  CALL graph%print(0, (/ graph_ptr(x), graph_ptr(y), graph_ptr(dydx) /))
!>  CALL graph%print(1, (/ graph_ptr(x), graph_ptr(y), graph_ptr(dydx) /))
!>  CALL graph%print(2, (/ graph_ptr(x), graph_ptr(y), graph_ptr(dydx) /))
!>
!>  DEALLOCATE(graph)
!>  END SUBROUTINE
!>  @endcode
!>
!>  @note Graphs need to use the @ref graph_fortran::graph_ptr function to get
!>  the pointer address of node.
!>  @note The @ref graph_fortran::graph_null_array allows set arrays of nodes to
!>  null.
!-------------------------------------------------------------------------------
!> Module contains subroutines for calling this from fortran.
      MODULE graph_fortran
      USE, INTRINSIC :: ISO_C_BINDING

      IMPLICIT NONE

!>  A null array for empty
      INTEGER(C_INTPTR_T), DIMENSION(0) :: graph_null_array

!-------------------------------------------------------------------------------
!>  @brief Class object for the binding.
!-------------------------------------------------------------------------------
      TYPE :: graph_context
#ifdef USE_METAL
!>  The auto release pool context.
         TYPE(C_PTR) :: arp_context
#endif
!>  The graph c context.
         TYPE(C_PTR) :: c_context
      CONTAINS
         FINAL :: graph_destruct
         PROCEDURE :: variable => graph_context_variable
         PROCEDURE :: constant_real => graph_context_constant_real
         PROCEDURE :: constant_complex => graph_context_constant_complex
         GENERIC   :: constant => constant_real, constant_complex
         PROCEDURE :: set_variable_float => graph_context_set_variable_float
         PROCEDURE :: set_variable_double => graph_context_set_variable_double
         PROCEDURE :: set_variable_cfloat => graph_context_set_variable_cfloat
         PROCEDURE :: set_variable_cdouble => graph_context_set_variable_cdouble
         GENERIC   :: set_variable => set_variable_float,                      &
                                      set_variable_double,                     &
                                      set_variable_cfloat,                     &
                                      set_variable_cdouble
         PROCEDURE :: pseudo_variable => graph_context_pseudo_variable
         PROCEDURE :: remove_pseudo => graph_context_remove_pseudo
         PROCEDURE :: add => graph_context_add
         PROCEDURE :: sub => graph_context_sub
         PROCEDURE :: mul => graph_context_mul
         PROCEDURE :: div => graph_context_div
         PROCEDURE :: sqrt => graph_context_sqrt
         PROCEDURE :: exp => graph_context_exp
         PROCEDURE :: log => graph_context_log
         PROCEDURE :: pow => graph_context_pow
         PROCEDURE :: erfi => graph_context_erfi
         PROCEDURE :: sin => graph_context_sin
         PROCEDURE :: cos => graph_context_cos
         PROCEDURE :: atan => graph_context_atan
         PROCEDURE :: random_state => graph_context_random_state
         PROCEDURE :: random => graph_context_random
         PROCEDURE :: piecewise_1D_float => graph_context_piecewise_1D_float
         PROCEDURE :: piecewise_1D_double => graph_context_piecewise_1D_double
         PROCEDURE :: piecewise_1D_cfloat => graph_context_piecewise_1D_cfloat
         PROCEDURE :: piecewise_1D_cdouble =>                                  &
                         graph_context_piecewise_1D_cdouble
         GENERIC   :: piecewise_1D => piecewise_1D_float,                      &
                                      piecewise_1D_double,                     &
                                      piecewise_1D_cfloat,                     &
                                      piecewise_1D_cdouble
         PROCEDURE :: piecewise_2D_float => graph_context_piecewise_2D_float
         PROCEDURE :: piecewise_2D_double => graph_context_piecewise_2D_double
         PROCEDURE :: piecewise_2D_cfloat => graph_context_piecewise_2D_cfloat
         PROCEDURE :: piecewise_2D_cdouble => graph_context_piecewise_2D_cdouble
         GENERIC   :: piecewise_2D => piecewise_2D_float,                      &
                                      piecewise_2D_double,                     &
                                      piecewise_2D_cfloat,                     &
                                      piecewise_2D_cdouble
         PROCEDURE :: get_max_concurrency => graph_context_get_max_concurrency
         PROCEDURE :: set_device_number => graph_context_set_device_number
         PROCEDURE :: add_pre_item => graph_context_add_pre_item
         PROCEDURE :: add_item => graph_context_add_item
         PROCEDURE :: add_converge_item => graph_context_add_converge_item
         PROCEDURE :: df => graph_context_df
         PROCEDURE :: compile => graph_context_compile
         PROCEDURE :: pre_run => graph_context_pre_run
         PROCEDURE :: run => graph_context_run
         PROCEDURE :: wait => graph_context_wait
         PROCEDURE :: copy_to_device_float => graph_context_copy_to_device_float
         PROCEDURE :: copy_to_device_double =>                                 &
                         graph_context_copy_to_device_double
         PROCEDURE :: copy_to_device_cfloat =>                                 &
                         graph_context_copy_to_device_cfloat
         PROCEDURE :: copy_to_device_cdouble =>                                &
                         graph_context_copy_to_device_cdouble
         GENERIC   :: copy_to_device => copy_to_device_float,                  &
                                        copy_to_device_double,                 &
                                        copy_to_device_cfloat,                 &
                                        copy_to_device_cdouble
         PROCEDURE :: copy_to_host_float => graph_context_copy_to_host_float
         PROCEDURE :: copy_to_host_double => graph_context_copy_to_host_double
         PROCEDURE :: copy_to_host_cfloat => graph_context_copy_to_host_cfloat
         PROCEDURE :: copy_to_host_cdouble => graph_context_copy_to_host_cdouble
         GENERIC   :: copy_to_host => copy_to_host_float,                      &
                                      copy_to_host_double,                     &
                                      copy_to_host_cfloat,                     &
                                      copy_to_host_cdouble
         PROCEDURE :: print => graph_context_print
      END TYPE

!*******************************************************************************
!  ENUMERATED TYPES
!*******************************************************************************
!-------------------------------------------------------------------------------
!>  @brief
!-------------------------------------------------------------------------------
      ENUM, BIND(C)
         ENUMERATOR :: FLOAT_T
         ENUMERATOR :: DOUBLE_T
         ENUMERATOR :: COMPLEX_FLOAT_T
         ENUMERATOR :: COMPLEX_DOUBLE_T
      END ENUM

!*******************************************************************************
!  INTERFACE BLOCKS
!*******************************************************************************
!-------------------------------------------------------------------------------
!>  @brief Interface for the graph_context constructor with float type.
!-------------------------------------------------------------------------------
      INTERFACE graph_float_context
         MODULE PROCEDURE graph_construct_float
      END INTERFACE

!-------------------------------------------------------------------------------
!>  @brief Interface for the graph_context constructor with double type.
!-------------------------------------------------------------------------------
      INTERFACE graph_double_context
         MODULE PROCEDURE graph_construct_double
      END INTERFACE

!-------------------------------------------------------------------------------
!>  @brief Interface for the graph_context constructor with complex float type.
!-------------------------------------------------------------------------------
      INTERFACE graph_complex_float_context
         MODULE PROCEDURE graph_construct_complex_float
      END INTERFACE

!-------------------------------------------------------------------------------
!>  @brief Interface for the graph_context constructor with complex double type.
!-------------------------------------------------------------------------------
      INTERFACE graph_complex_double_context
         MODULE PROCEDURE graph_construct_complex_double
      END INTERFACE

!*******************************************************************************
!  C Binding Interface.
!*******************************************************************************
      INTERFACE
!-------------------------------------------------------------------------------
!>  @brief Auto release pool push interface.
!>
!>  @returns An auto release pool context.
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION objc_autoreleasePoolPush()                      &
         BIND(C, NAME='objc_autoreleasePoolPush')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Auto release pool pop interface.
!>
!>  @param[in,out] ctx Auto Release pool context.
!-------------------------------------------------------------------------------
         SUBROUTINE objc_autoreleasePoolPop(ctx)                               &
         BIND(C, NAME='objc_autoreleasePoolPop')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: ctx
         END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Construct a C context.
!>
!>  @param[in] c_type        The type of the context @ref graph_type.
!>  @param[in] use_safe_math C context uses safemath.
!>  @returns The constructed C context.
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_construct_context(c_type, use_safe_math)   &
         BIND(C, NAME='graph_construct_context')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         INTEGER(C_INT), VALUE  :: c_type
         LOGICAL(C_BOOL), VALUE :: use_safe_math
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Destroy C context.
!>
!>  @param[in] c The c context to delete.
!-------------------------------------------------------------------------------
         SUBROUTINE graph_destroy_context(c)                                   &
         BIND(C, NAME='graph_destroy_context')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Create a variable node.
!>
!>  @param[in,out] c      The c context.
!>  @param[in]     size   Size of the data buffer.
!>  @param[in]     symbol Symbol of the variable used in equations.
!>  @returns The created variable.
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_variable(c, size, symbol)                  &
         BIND(C, NAME='graph_variable')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE                   :: c
         INTEGER(C_LONG), VALUE               :: size
         CHARACTER(kind=C_CHAR), DIMENSION(*) :: symbol
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create a constant node.
!>
!>  @param[in,out] c     The c context.
!>  @param[in]     value Value of the constant.
!>  @returns The created constant.
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_constant(c, value)                         &
         BIND(C, NAME='graph_constant')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE    :: c
         REAL(C_DOUBLE), VALUE :: value
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Set a variable value.
!>
!>  @param[in,out] c     The c context.
!>  @param[in]     var   Variable to set.
!>  @param[in]     value The buffer to the variable with.
!-------------------------------------------------------------------------------
         SUBROUTINE graph_set_variable(c, var, value)                          &
         BIND(C, NAME='graph_set_variable')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE         :: c
         TYPE(C_PTR), VALUE         :: var
         INTEGER(C_INTPTR_T), VALUE :: value
         END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Create a constant node with complex values.
!>
!>  @param[in] c          The graph C context.
!>  @param[in] real_value The real component.
!>  @param[in] img_value  The imaginary component.
!>  @returns The complex constant.
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_constant_c(c, real_value, img_value)       &
         BIND(C, NAME='graph_constant_c')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE    :: c
         REAL(C_DOUBLE), VALUE :: real_value
         REAL(C_DOUBLE), VALUE :: img_value
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create a pseudo variable node.
!>
!>  @param[in] c   The graph C context.
!>  @param[in] var The variable to set.
!>  @returns The pseudo variable.
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_pseudo_variable(c, var)                    &
         BIND(C, NAME='graph_pseudo_variable')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         TYPE(C_PTR), VALUE :: var
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Remove pseudo.
!>
!>  @param[in] c   The graph C context.
!>  @param[in] var The graph to remove pseudo variables.
!>  @returns The graph with pseudo variables removed.
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_remove_pseudo(c, var)                      &
         BIND(C, NAME='graph_remove_pseudo')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         TYPE(C_PTR), VALUE :: var
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create Addition node.
!>
!>  @param[in] c     The graph C context.
!>  @param[in] left  The left opperand.
!>  @param[in] right The right opperand.
!>  @returns left + right
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_add(c, left, right)                        &
         BIND(C, NAME='graph_add')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         TYPE(C_PTR), VALUE :: left
         TYPE(C_PTR), VALUE :: right
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create Substract node.
!>
!>  @param[in] c     The graph C context.
!>  @param[in] left  The left opperand.
!>  @param[in] right The right opperand.
!>  @returns left - right
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_sub(c, left, right)                        &
         BIND(C, NAME='graph_sub')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         TYPE(C_PTR), VALUE :: left
         TYPE(C_PTR), VALUE :: right
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create Multiply node.
!>
!>  @param[in] c     The graph C context.
!>  @param[in] left  The left opperand.
!>  @param[in] right The right opperand.
!>  @returns left*right
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_mul(c, left, right)                        &
         BIND(C, NAME='graph_mul')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         TYPE(C_PTR), VALUE :: left
         TYPE(C_PTR), VALUE :: right
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create Divide node.
!>
!>  @param[in] c     The graph C context.
!>  @param[in] left  The left opperand.
!>  @param[in] right The right opperand.
!>  @returns left/right
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_div(c, left, right)                        &
         BIND(C, NAME='graph_div')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         TYPE(C_PTR), VALUE :: left
         TYPE(C_PTR), VALUE :: right
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create Sqrt node.
!>
!>  @param[in] c   The graph C context.
!>  @param[in] arg The function argument.
!>  @returns sqrt(arg)
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_sqrt(c, arg)                               &
         BIND(C, NAME='graph_sqrt')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         TYPE(C_PTR), VALUE :: arg
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create Exp node.
!>
!>  @param[in] c   The graph C context.
!>  @param[in] arg The function argument.
!>  @returns exp(arg)
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_exp(c, arg)                               &
         BIND(C, NAME='graph_exp')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         TYPE(C_PTR), VALUE :: arg
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create Log node.
!>
!>  @param[in] c   The graph C context.
!>  @param[in] arg The function argument.
!>  @returns log(arg)
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_log(c, arg)                               &
         BIND(C, NAME='graph_log')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         TYPE(C_PTR), VALUE :: arg
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create pow node.
!>
!>  @param[in] c     The graph C context.
!>  @param[in] left  The left opperand.
!>  @param[in] right The right opperand.
!>  @returns pow(left, right)
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_pow(c, left, right)                        &
         BIND(C, NAME='graph_pow')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         TYPE(C_PTR), VALUE :: left
         TYPE(C_PTR), VALUE :: right
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create Erfi node.
!>
!>  @param[in] c   The graph C context.
!>  @param[in] arg The function argument.
!>  @returns erfi(arg)
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_erfi(c, arg)                               &
         BIND(C, NAME='graph_erfi')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         TYPE(C_PTR), VALUE :: arg
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create Sine node.
!>
!>  @param[in] c   The graph C context.
!>  @param[in] arg The function argument.
!>  @returns sin(arg)
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_sin(c, arg)                                &
         BIND(C, NAME='graph_sin')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         TYPE(C_PTR), VALUE :: arg
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create Cosine node.
!>
!>  @param[in] c   The graph C context.
!>  @param[in] arg The function argument.
!>  @returns cos(arg)
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_cos(c, arg)                                &
         BIND(C, NAME='graph_cos')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         TYPE(C_PTR), VALUE :: arg
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create atan node.
!>
!>  @param[in] c     The graph C context.
!>  @param[in] left  The left opperand.
!>  @param[in] right The right opperand.
!>  @returns pow(left, right)
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_atan(c, left, right)                       &
         BIND(C, NAME='graph_atan')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         TYPE(C_PTR), VALUE :: left
         TYPE(C_PTR), VALUE :: right
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Construct a random state node.
!>
!>  @param[in] c    The graph C context.
!>  @param[in] seed Intial random seed.
!>  @returns A random state node.
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_random_state(c, seed)                      &
         BIND(C, NAME='graph_random_state')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE        :: c
         INTEGER(C_INT32_T), VALUE :: seed
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create Random node.
!>
!>  @param[in] c     The graph C context.
!>  @param[in] state A random state node.
!>  @returns random(state)
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_random(c, state)                           &
         BIND(C, NAME='graph_random')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         TYPE(C_PTR), VALUE :: state
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create 1D piecewise node with complex double buffer.
!>
!>  @param[in] c           The graph C context.
!>  @param[in] arg         The left opperand.
!>  @param[in] scale       Scale factor argument.
!>  @param[in] offset      Offset factor argument.
!>  @param[in] source      Source buffer to fill elements.
!>  @param[in] source_size Number of elements in the source buffer.
!>  @returns A 1D piecewise node.
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_piecewise_1D(c, arg, scale, offset,        &
                                                 source, source_size)          &
         BIND(C, NAME='graph_piecewise_1D')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE         :: c
         TYPE(C_PTR), VALUE         :: arg
         REAL(C_DOUBLE), VALUE      :: scale
         REAL(C_DOUBLE), VALUE      :: offset
         INTEGER(C_INTPTR_T), VALUE :: source
         INTEGER(C_LONG), VALUE     :: source_size
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create 2D piecewise node.
!>
!>  @param[in] c           The graph C context.
!>  @param[in] num_cols    Number of columns.
!>  @param[in] x_arg       The function x argument.
!>  @param[in] x_scale     Scale factor x argument.
!>  @param[in] x_offset    Offset factor x argument.
!>  @param[in] y_arg       The function y argument.
!>  @param[in] y_scale     Scale factor y argument.
!>  @param[in] y_offset    Offset factor y argument.
!>  @param[in] source      Source buffer to fill elements.
!>  @param[in] source_size Number of elements in the source buffer.
!>  @returns A 2D piecewise node.
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_piecewise_2D(c, num_cols,                  &
                                                 x_arg, x_scale, x_offset,     &
                                                 y_arg, y_scale, y_offset,     &
                                                 source, source_size)          &
         BIND(C, NAME='graph_piecewise_2D')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE         :: c
         INTEGER(C_LONG), VALUE     :: num_cols
         TYPE(C_PTR), VALUE         :: x_arg
         REAL(C_DOUBLE), VALUE      :: x_scale
         REAL(C_DOUBLE), VALUE      :: x_offset
         TYPE(C_PTR), VALUE         :: y_arg
         REAL(C_DOUBLE), VALUE      :: y_scale
         REAL(C_DOUBLE), VALUE      :: y_offset
         INTEGER(C_INTPTR_T), VALUE :: source
         INTEGER(C_LONG), VALUE     :: source_size
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Get the maximum number of concurrent devices.
!>
!>  @param[in] c The graph C context.
!>  @returns The number of devices.
!-------------------------------------------------------------------------------
         INTEGER(C_LONG) FUNCTION graph_get_max_concurrency(c)                 &
         BIND(C, NAME='graph_get_max_concurrency')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Choose the device number.
!>
!>  @param[in] c   The graph C context.
!>  @param[in] num The device number.
!-------------------------------------------------------------------------------
         SUBROUTINE graph_set_device_number(c, num)                            &
         BIND(C, NAME='graph_set_device_number')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE     :: c
         INTEGER(C_LONG), VALUE :: num
         END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Add pre workflow item.
!>
!>  @param[in] c             The graph C context.
!>  @param[in] inputs        Array of input nodes.
!>  @param[in] num_inputs    Number of inputs.
!>  @param[in] outputs       Array of output nodes.
!>  @param[in] num_outputs   Number of outputs.
!>  @param[in] map_inputs    Array of map input nodes.
!>  @param[in] map_outputs   Array of map output nodes.
!>  @param[in] num_maps      Number of maps.
!>  @param[in] random_state  Optional random state, can be NULL if not used.
!>  @param[in] name          Name for the kernel.
!>  @param[in] num_particles Number of elements to operate on.
!-------------------------------------------------------------------------------
         SUBROUTINE graph_add_pre_item(c, inputs, num_inputs,                  &
                                       outputs, num_outputs,                   &
                                       map_inputs, map_outputs, num_maps,      &
                                       random_state, name, num_particles)      &
         BIND(C, NAME='graph_add_pre_item')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE                   :: c
         INTEGER(C_INTPTR_T), VALUE           :: inputs
         INTEGER(C_LONG), VALUE               :: num_inputs
         INTEGER(C_INTPTR_T), VALUE           :: outputs
         INTEGER(C_LONG), VALUE               :: num_outputs
         INTEGER(C_INTPTR_T), VALUE           :: map_inputs
         INTEGER(C_INTPTR_T), VALUE           :: map_outputs
         INTEGER(C_LONG), VALUE               :: num_maps
         TYPE(C_PTR), VALUE                   :: random_state
         CHARACTER(kind=C_CHAR), DIMENSION(*) :: name
         INTEGER(C_LONG), VALUE               :: num_particles
         END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Add workflow item.
!>
!>  @param[in] c             The graph C context.
!>  @param[in] inputs        Array of input nodes.
!>  @param[in] num_inputs    Number of inputs.
!>  @param[in] outputs       Array of output nodes.
!>  @param[in] num_outputs   Number of outputs.
!>  @param[in] map_inputs    Array of map input nodes.
!>  @param[in] map_outputs   Array of map output nodes.
!>  @param[in] num_maps      Number of maps.
!>  @param[in] random_state  Optional random state, can be NULL if not used.
!>  @param[in] name          Name for the kernel.
!>  @param[in] num_particles Number of elements to operate on.
!-------------------------------------------------------------------------------
         SUBROUTINE graph_add_item(c, inputs, num_inputs,                      &
                                   outputs, num_outputs,                       &
                                   map_inputs, map_outputs, num_maps,          &
                                   random_state, name, num_particles)          &
         BIND(C, NAME='graph_add_item')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE                   :: c
         INTEGER(C_INTPTR_T), VALUE           :: inputs
         INTEGER(C_LONG), VALUE               :: num_inputs
         INTEGER(C_INTPTR_T), VALUE           :: outputs
         INTEGER(C_LONG), VALUE               :: num_outputs
         INTEGER(C_INTPTR_T), VALUE           :: map_inputs
         INTEGER(C_INTPTR_T), VALUE           :: map_outputs
         INTEGER(C_LONG), VALUE               :: num_maps
         TYPE(C_PTR), VALUE                   :: random_state
         CHARACTER(kind=C_CHAR), DIMENSION(*) :: name
         INTEGER(C_LONG), VALUE               :: num_particles
         END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Add workflow converge item.
!>
!>  @param[in] c             The graph C context.
!>  @param[in] inputs        Array of input nodes.
!>  @param[in] num_inputs    Number of inputs.
!>  @param[in] outputs       Array of output nodes.
!>  @param[in] num_outputs   Number of outputs.
!>  @param[in] map_inputs    Array of map input nodes.
!>  @param[in] map_outputs   Array of map output nodes.
!>  @param[in] num_maps      Number of maps.
!>  @param[in] random_state  Optional random state, can be NULL if not used.
!>  @param[in] name          Name for the kernel.
!>  @param[in] num_particles Number of elements to operate on.
!>  @param[in] tol           Tolarance to converge the function to.
!>  @param[in] max_iter      Maximum number of iterations before giving up.
!-------------------------------------------------------------------------------
         SUBROUTINE graph_add_converge_item(c, inputs, num_inputs,             &
                                            outputs, num_outputs,              &
                                            map_inputs, map_outputs, num_maps, &
                                            random_state, name, num_particles, &
                                            tol, max_iter)                     &
         BIND(C, NAME='graph_add_converge_item')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE                   :: c
         INTEGER(C_INTPTR_T), VALUE           :: inputs
         INTEGER(C_LONG), VALUE               :: num_inputs
         INTEGER(C_INTPTR_T), VALUE           :: outputs
         INTEGER(C_LONG), VALUE               :: num_outputs
         INTEGER(C_INTPTR_T), VALUE           :: map_inputs
         INTEGER(C_INTPTR_T), VALUE           :: map_outputs
         INTEGER(C_LONG), VALUE               :: num_maps
         TYPE(C_PTR), VALUE                   :: random_state
         CHARACTER(kind=C_CHAR), DIMENSION(*) :: name
         INTEGER(C_LONG), VALUE               :: num_particles
         REAL(C_DOUBLE), VALUE                :: tol
         INTEGER(C_LONG), VALUE               :: max_iter
         END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Compile the work items.
!>
!>  @param[in] c The graph C context.
!-------------------------------------------------------------------------------
         SUBROUTINE graph_compile(c)                                           &
         BIND(C, NAME='graph_compile')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Run pre work items.
!>
!>  @param[in] c The graph C context.
!-------------------------------------------------------------------------------
         SUBROUTINE graph_pre_run(c)                                           &
         BIND(C, NAME='graph_pre_run')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Run work items.
!>
!>  @param[in] c The graph C context.
!-------------------------------------------------------------------------------
         SUBROUTINE graph_run(c)                                               &
         BIND(C, NAME='graph_run')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Wait for work items to complete.
!>
!>  @param[in] c The graph C context.
!-------------------------------------------------------------------------------
         SUBROUTINE graph_wait(c)                                              &
         BIND(C, NAME='graph_wait')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE :: c
         END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Take derivative ∂f∂x.
!>
!>  @param[in] c     The graph C context.
!>  @param[in] fnode The function expression to take the derivative of.
!>  @param[in] xnode The expression to take the derivative with respect to.
!-------------------------------------------------------------------------------
         TYPE(C_PTR) FUNCTION graph_df(c, fnode, xnode)                        &
         BIND(C, NAME='graph_df')
         USE, INTRINSIC :: ISO_C_BINDING
         TYPE(C_PTR), VALUE :: c
         TYPE(C_PTR), VALUE :: fnode
         TYPE(C_PTR), VALUE :: xnode
         END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Copy data to a device buffer.
!>
!>  @param[in] c      The c context.
!>  @param[in] node   Node to copy to.
!>  @param[in] source Source to copy from.
!-------------------------------------------------------------------------------
         SUBROUTINE graph_copy_to_device(c, node, source)                      &
         BIND(C, NAME='graph_copy_to_device')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE     :: c
         TYPE(C_PTR), VALUE     :: node
         INTEGER(C_LONG), VALUE :: source
         END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Copy data to a host buffer.
!>
!>  @param[in]     c           The graph C context.
!>  @param[in]     node        Node to copy from.
!>  @param[in,out] destination Host side buffer to copy to.
!-------------------------------------------------------------------------------
         SUBROUTINE graph_copy_to_host(c, node, destination)                   &
         BIND(C, NAME='graph_copy_to_host')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE     :: c
         TYPE(C_PTR), VALUE     :: node
         INTEGER(C_LONG), VALUE :: destination
         END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Print a value from nodes.
!>
!>  @param[in] c         The graph C context.
!>  @param[in] index     Particle index to print.
!>  @param[in] nodes     Nodes to print.
!>  @param[in] num_nodes Number of nodes.
!-------------------------------------------------------------------------------
         SUBROUTINE graph_print(c, index, nodes, num_nodes)                    &
         BIND(C, NAME='graph_print')
         USE, INTRINSIC :: ISO_C_BINDING
         IMPLICIT NONE
         TYPE(C_PTR), VALUE         :: c
         INTEGER(C_LONG), VALUE     :: index
         INTEGER(C_INTPTR_T), VALUE :: nodes
         INTEGER(C_LONG), VALUE     :: num_nodes
         END SUBROUTINE

      END INTERFACE

      CONTAINS

!*******************************************************************************
!  Utilities
!*******************************************************************************
!-------------------------------------------------------------------------------
!>  @brief Convert a node to the pointer value.
!>
!>  @return The pointer value.
!-------------------------------------------------------------------------------
      FUNCTION graph_ptr(node)

      IMPLICIT NONE

!  Declare Arguments
      INTEGER(C_INTPTR_T)     :: graph_ptr
      TYPE(C_PTR), INTENT(IN) :: node

!  Start of executable code.
      graph_ptr = TRANSFER(node, 0_C_INTPTR_T)

      END FUNCTION

!*******************************************************************************
!  CONSTRUCTION SUBROUTINES
!*******************************************************************************
!-------------------------------------------------------------------------------
!>  @brief Construct a @ref graph_context object with float type.
!>
!>  Allocate memory for the @ref graph_context and initalize the c context with
!>  a double type.
!>
!>  @param[in] use_safe_math Optional use safe math.
!-------------------------------------------------------------------------------
      FUNCTION graph_construct_float(use_safe_math)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), POINTER :: graph_construct_float
      LOGICAL(C_BOOL), INTENT(IN)   :: use_safe_math

!  Start of executable code.
      ALLOCATE(graph_construct_float)
#ifdef USE_METAL
      graph_construct_float%arp_context = objc_autoreleasePoolPush()
#endif
      graph_construct_float%c_context =                                        &
         graph_construct_context(FLOAT_T, use_safe_math)

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Construct a @ref graph_context object with double type.
!>
!>  Allocate memory for the @ref graph_context and initalize the c context with
!>  a double type.
!>
!>  @param[in] use_safe_math Use safe math.
!-------------------------------------------------------------------------------
      FUNCTION graph_construct_double(use_safe_math)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), POINTER :: graph_construct_double
      LOGICAL(C_BOOL), INTENT(IN)   :: use_safe_math

!  Start of executable code.
      ALLOCATE(graph_construct_double)
#ifdef USE_METAL
      graph_construct_double%arp_context = objc_autoreleasePoolPush()
#endif
      graph_construct_double%c_context =                                       &
         graph_construct_context(DOUBLE_T, use_safe_math)

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Construct a @ref graph_context object with complex float type.
!>
!>  Allocate memory for the @ref graph_context and initalize the c context with
!>  a complex float type.
!>
!>  @param[in] use_safe_math Use safe math.
!-------------------------------------------------------------------------------
      FUNCTION graph_construct_complex_float(use_safe_math)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), POINTER :: graph_construct_complex_float
      LOGICAL(C_BOOL), INTENT(IN)   :: use_safe_math

!  Start of executable code.
      ALLOCATE(graph_construct_complex_float)
#ifdef USE_METAL
      graph_construct_complex_float%arp_context = objc_autoreleasePoolPush()
#endif
      graph_construct_complex_float%c_context =                                &
         graph_construct_context(COMPLEX_FLOAT_T, use_safe_math)

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Construct a @ref graph_context object with complex double type.
!>
!>  Allocate memory for the @ref graph_context and initalize the c context with
!>  a complex double type.
!>
!>  @param[in] use_safe_math Use safe math.
!-------------------------------------------------------------------------------
      FUNCTION graph_construct_complex_double(use_safe_math)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), POINTER :: graph_construct_complex_double
      LOGICAL(C_BOOL), INTENT(IN)   :: use_safe_math

!  Start of executable code.
      ALLOCATE(graph_construct_complex_double)
#ifdef USE_METAL
      graph_construct_complex_double%arp_context = objc_autoreleasePoolPush()
#endif
      graph_construct_complex_double%c_context =                               &
         graph_construct_context(COMPLEX_DOUBLE_T, use_safe_math)

      END FUNCTION

!*******************************************************************************
!  DESTRUCTION SUBROUTINES
!*******************************************************************************
!-------------------------------------------------------------------------------
!>  @brief Deconstruct a @ref graph_context object.
!>
!>  Deallocate memory and unitialize a @ref graph_context object.
!>
!>  @param[in,out] this A @ref graph_context instance.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_destruct(this)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(graph_context), INTENT(INOUT) :: this

!  Start of executable.
#ifdef USE_METAL
      CALL objc_autoreleasePoolPop(this%arp_context)
#endif
      CALL graph_destroy_context(this%c_context)

      END SUBROUTINE

!*******************************************************************************
!  Basic Nodes
!*******************************************************************************
!-------------------------------------------------------------------------------
!>  @brief Create variable node.
!>
!>  @param[in,out] this   @ref graph_context instance.
!>  @param[in]     size   Size of the data buffer.
!>  @param[in]     symbol Symbol of the variable.
!>  @returns A variable node.
!-------------------------------------------------------------------------------
      FUNCTION graph_context_variable(this, size, symbol)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                              :: graph_context_variable
      CLASS(graph_context), INTENT(INOUT)      :: this
      INTEGER(C_LONG), INTENT(IN)              :: size
      CHARACTER(kind=C_CHAR,len=*), INTENT(IN) :: symbol

!  Start of executable.
      graph_context_variable = graph_variable(this%c_context, size, symbol)

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create a constant node.
!>
!>  @param[in,out] this  @ref graph_context instance.
!>  @param[in]     value Size of the data buffer.
!>  @returns A constant node.
!-------------------------------------------------------------------------------
      FUNCTION graph_context_constant_real(this, value)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_constant_real
      CLASS(graph_context), INTENT(INOUT) :: this
      REAL(C_DOUBLE), INTENT(IN)          :: value

!  Start of executable.
      graph_context_constant_real = graph_constant(this%c_context, value)

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Set the value of a variable float types.
!>
!>  @param[in,out] this  @ref graph_context instance.
!>  @param[in]     var   The variable to set
!>  @param[in]     value THe value to set.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_set_variable_float(this, var, value)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(INOUT)     :: this
      TYPE(C_PTR), INTENT(IN)                 :: var
      REAL(C_FLOAT), DIMENSION(:), INTENT(IN) :: value

!  Start of executable.
      CALL graph_set_variable(this%c_context, var, LOC(value))

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Set the value of a variable double types.
!>
!>  @param[in,out] this  @ref graph_context instance.
!>  @param[in]     var   The variable to set
!>  @param[in]     value THe value to set.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_set_variable_double(this, var, value)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(INOUT)      :: this
      TYPE(C_PTR), INTENT(IN)                  :: var
      REAL(C_DOUBLE), DIMENSION(:), INTENT(IN) :: value

!  Start of executable.
      CALL graph_set_variable(this%c_context, var, LOC(value))

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Set the value of a variable complex float types.
!>
!>  @param[in,out] this  @ref graph_context instance.
!>  @param[in]     var   The variable to set
!>  @param[in]     value THe value to set.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_set_variable_cfloat(this, var, value)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(INOUT)                :: this
      TYPE(C_PTR), INTENT(IN)                            :: var
      COMPLEX(C_FLOAT_COMPLEX), DIMENSION(:), INTENT(IN) :: value

!  Start of executable.
      CALL graph_set_variable(this%c_context, var, LOC(value))

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Set the value of a variable complex double types.
!>
!>  @param[in,out] this  @ref graph_context instance.
!>  @param[in]     var   The variable to set
!>  @param[in]     value THe value to set.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_set_variable_cdouble(this, var, value)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(INOUT)                 :: this
      TYPE(C_PTR), INTENT(IN)                             :: var
      COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(:), INTENT(IN) :: value

!  Start of executable.
      CALL graph_set_variable(this%c_context, var, LOC(value))

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Create constant node with complex values.
!>
!>  @param[in,out] this       @ref graph_context instance.
!>  @param[in]     real_value The real component.
!>  @param[in]     img_value  The imaginary component.
!>  @returns A constant node.
!-------------------------------------------------------------------------------
      FUNCTION graph_context_constant_complex(this, real_value, img_value)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_constant_complex
      CLASS(graph_context), INTENT(INOUT) :: this
      REAL(C_DOUBLE), INTENT(IN)          :: real_value
      REAL(C_DOUBLE), INTENT(IN)          :: img_value

!  Start of executable.
      graph_context_constant_complex = graph_constant_c(this%c_context,        &
                                                        real_value, img_value)

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create pseudo variable node.
!>
!>  @param[in,out] this @ref graph_context instance.
!>  @param[in]     var  The variable to set.
!>  @returns The pseudo variable.
!-------------------------------------------------------------------------------
      FUNCTION graph_context_pseudo_variable(this, var)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_pseudo_variable
      CLASS(graph_context), INTENT(INOUT) :: this
      TYPE(C_PTR), INTENT(IN)             :: var

!  Start of executable.
      graph_context_pseudo_variable = graph_pseudo_variable(this%c_context, var)

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Remove pseudo.
!>
!>  @param[in,out] this @ref graph_context instance.
!>  @param[in]     var  The graph to remove pseudo variables.
!>  @returns The graph with pseudo variables removed.
!-------------------------------------------------------------------------------
      FUNCTION graph_context_remove_pseudo(this, var)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_remove_pseudo
      CLASS(graph_context), INTENT(INOUT) :: this
      TYPE(C_PTR), INTENT(IN)             :: var

!  Start of executable.
      graph_context_remove_pseudo = graph_remove_pseudo(this%c_context, var)

      END FUNCTION

!*******************************************************************************
!  Arithmetic Nodes
!*******************************************************************************
!-------------------------------------------------------------------------------
!>  @brief Create Addition node.
!>
!>  @param[in,out] this  @ref graph_context instance.
!>  @param[in]     left  The graph to remove pseudo variables.
!>  @param[in]     right The graph to remove pseudo variables.
!>  @returns left + right
!-------------------------------------------------------------------------------
      FUNCTION graph_context_add(this, left, right)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_add
      CLASS(graph_context), INTENT(INOUT) :: this
      TYPE(C_PTR), INTENT(IN)             :: left
      TYPE(C_PTR), INTENT(IN)             :: right

!  Start of executable.
      graph_context_add = graph_add(this%c_context, left, right)

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create Subtract node.
!>
!>  @param[in,out] this  @ref graph_context instance.
!>  @param[in]     left  The graph to remove pseudo variables.
!>  @param[in]     right The graph to remove pseudo variables.
!>  @returns left - right
!-------------------------------------------------------------------------------
      FUNCTION graph_context_sub(this, left, right)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_sub
      CLASS(graph_context), INTENT(INOUT) :: this
      TYPE(C_PTR), INTENT(IN)             :: left
      TYPE(C_PTR), INTENT(IN)             :: right

!  Start of executable.
      graph_context_sub = graph_sub(this%c_context, left, right)

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create Multiply node.
!>
!>  @param[in,out] this  @ref graph_context instance.
!>  @param[in]     left  The graph to remove pseudo variables.
!>  @param[in]     right The graph to remove pseudo variables.
!>  @returns left*right
!-------------------------------------------------------------------------------
      FUNCTION graph_context_mul(this, left, right)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_mul
      CLASS(graph_context), INTENT(INOUT) :: this
      TYPE(C_PTR), INTENT(IN)             :: left
      TYPE(C_PTR), INTENT(IN)             :: right

!  Start of executable.
      graph_context_mul = graph_mul(this%c_context, left, right)

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create Divide node.
!>
!>  @param[in,out] this  @ref graph_context instance.
!>  @param[in]     left  The graph to remove pseudo variables.
!>  @param[in]     right The graph to remove pseudo variables.
!>  @returns left/right
!-------------------------------------------------------------------------------
      FUNCTION graph_context_div(this, left, right)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_div
      CLASS(graph_context), INTENT(INOUT) :: this
      TYPE(C_PTR), INTENT(IN)             :: left
      TYPE(C_PTR), INTENT(IN)             :: right

!  Start of executable.
      graph_context_div = graph_div(this%c_context, left, right)

      END FUNCTION

!*******************************************************************************
!  Math Nodes
!*******************************************************************************
!-------------------------------------------------------------------------------
!>  @brief Create Sqrt node.
!>
!>  @param[in,out] this @ref graph_context instance.
!>  @param[in]     arg  The function argument.
!>  @returns sqrt(arg)
!-------------------------------------------------------------------------------
      FUNCTION graph_context_sqrt(this, arg)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_sqrt
      CLASS(graph_context), INTENT(INOUT) :: this
      TYPE(C_PTR), INTENT(IN)             :: arg

!  Start of executable.
      graph_context_sqrt = graph_sqrt(this%c_context, arg)

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create Exp node.
!>
!>  @param[in,out] this @ref graph_context instance.
!>  @param[in]     arg  The function argument.
!>  @returns exp(arg)
!-------------------------------------------------------------------------------
      FUNCTION graph_context_exp(this, arg)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_exp
      CLASS(graph_context), INTENT(INOUT) :: this
      TYPE(C_PTR), INTENT(IN)             :: arg

!  Start of executable.
      graph_context_exp = graph_exp(this%c_context, arg)

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create Log node.
!>
!>  @param[in,out] this @ref graph_context instance.
!>  @param[in]     arg  The function argument.
!>  @returns log(arg)
!-------------------------------------------------------------------------------
      FUNCTION graph_context_log(this, arg)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_log
      CLASS(graph_context), INTENT(INOUT) :: this
      TYPE(C_PTR), INTENT(IN)             :: arg

!  Start of executable.
      graph_context_log = graph_log(this%c_context, arg)

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create Pow node.
!>
!>  @param[in,out] this  @ref graph_context instance.
!>  @param[in]     left  The graph to remove pseudo variables.
!>  @param[in]     right The graph to remove pseudo variables.
!>  @returns pow(left, right)
!-------------------------------------------------------------------------------
      FUNCTION graph_context_pow(this, left, right)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_pow
      CLASS(graph_context), INTENT(INOUT) :: this
      TYPE(C_PTR), INTENT(IN)             :: left
      TYPE(C_PTR), INTENT(IN)             :: right

!  Start of executable.
      graph_context_pow = graph_pow(this%c_context, left, right)

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create erfi node.
!>
!>  @param[in,out] this @ref graph_context instance.
!>  @param[in]     arg  The function argument.
!>  @returns erfi(arg)
!-------------------------------------------------------------------------------
      FUNCTION graph_context_erfi(this, arg)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_erfi
      CLASS(graph_context), INTENT(INOUT) :: this
      TYPE(C_PTR), INTENT(IN)             :: arg

!  Start of executable.
      graph_context_erfi = graph_erfi(this%c_context, arg)

      END FUNCTION

!*******************************************************************************
!  Trigonometry Nodes
!*******************************************************************************
!-------------------------------------------------------------------------------
!>  @brief Create Sine node.
!>
!>  @param[in,out] this @ref graph_context instance.
!>  @param[in]     arg  The function argument.
!>  @returns sin(arg)
!-------------------------------------------------------------------------------
      FUNCTION graph_context_sin(this, arg)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_sin
      CLASS(graph_context), INTENT(INOUT) :: this
      TYPE(C_PTR), INTENT(IN)             :: arg

!  Start of executable.
      graph_context_sin = graph_sin(this%c_context, arg)

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create Cosine node.
!>
!>  @param[in,out] this @ref graph_context instance.
!>  @param[in]     arg  The function argument.
!>  @returns cos(arg)
!-------------------------------------------------------------------------------
      FUNCTION graph_context_cos(this, arg)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_cos
      CLASS(graph_context), INTENT(INOUT) :: this
      TYPE(C_PTR), INTENT(IN)             :: arg

!  Start of executable.
      graph_context_cos = graph_cos(this%c_context, arg)

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create atan node.
!>
!>  @param[in,out] this  @ref graph_context instance.
!>  @param[in]     left  The graph to remove pseudo variables.
!>  @param[in]     right The graph to remove pseudo variables.
!>  @returns atan(left, right)
!-------------------------------------------------------------------------------
      FUNCTION graph_context_atan(this, left, right)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_atan
      CLASS(graph_context), INTENT(INOUT) :: this
      TYPE(C_PTR), INTENT(IN)             :: left
      TYPE(C_PTR), INTENT(IN)             :: right

!  Start of executable.
      graph_context_atan = graph_atan(this%c_context, left, right)

      END FUNCTION

!*******************************************************************************
!  Random Nodes
!*******************************************************************************
!-------------------------------------------------------------------------------
!>  @brief Get random size.
!>
!>  @param[in,out] this @ref graph_context instance.
!>  @param[in]     seed Intial random seed.
!>  @returns The random size.
!-------------------------------------------------------------------------------
      FUNCTION graph_context_random_state(this, seed)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_random_state
      CLASS(graph_context), INTENT(INOUT) :: this
      INTEGER(C_INT32_T), INTENT(IN)      :: seed

!  Start of executable.
      graph_context_random_state = graph_random_state(this%c_context, seed)

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create random node.
!>
!>  @param[in,out] this  @ref graph_context instance.
!>  @param[in]     state A random state node.
!>  @returns random(state)
!-------------------------------------------------------------------------------
      FUNCTION graph_context_random(this, state)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_random
      CLASS(graph_context), INTENT(INOUT) :: this
      TYPE(C_PTR), INTENT(IN)             :: state

!  Start of executable.
      graph_context_random = graph_random(this%c_context, state)

      END FUNCTION

!*******************************************************************************
!  Piecewise Nodes
!*******************************************************************************
!-------------------------------------------------------------------------------
!>  @brief Create 1D piecewise node with float buffer.
!>
!>  @param[in,out] this   @ref graph_context instance.
!>  @param[in]     arg    The function argument.
!>  @param[in]     scale  Scale factor argument.
!>  @param[in]     offset Offset factor argument.
!>  @param[in]     source Source buffer to fill elements.
!>  @returns random(state)
!-------------------------------------------------------------------------------
      FUNCTION graph_context_piecewise_1D_float(this, arg, scale, offset,      &
                                                source)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_piecewise_1D_float
      CLASS(graph_context), INTENT(INOUT) :: this
      TYPE(C_PTR), INTENT(IN)             :: arg
      REAL(C_DOUBLE)                      :: scale
      REAL(C_DOUBLE)                      :: offset
      REAL(C_FLOAT), DIMENSION(:)         :: source

!  Start of executable.
      graph_context_piecewise_1D_float =                                       &
         graph_piecewise_1D(this%c_context, arg, scale, offset, LOC(source),   &
                            INT(SIZE(source), KIND=C_LONG))

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create 1D piecewise node with double buffer.
!>
!>  @param[in,out] this   @ref graph_context instance.
!>  @param[in]     arg    The function argument.
!>  @param[in]     scale  Scale factor argument.
!>  @param[in]     offset Offset factor argument.
!>  @param[in]     source Source buffer to fill elements.
!>  @returns random(state)
!-------------------------------------------------------------------------------
      FUNCTION graph_context_piecewise_1D_double(this, arg, scale, offset,     &
                                                 source)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                          :: graph_context_piecewise_1D_double
      CLASS(graph_context), INTENT(INOUT)  :: this
      TYPE(C_PTR), INTENT(IN)              :: arg
      REAL(C_DOUBLE)                       :: scale
      REAL(C_DOUBLE)                       :: offset
      REAL(C_DOUBLE), DIMENSION(:)         :: source

!  Start of executable.
      graph_context_piecewise_1D_double =                                      &
         graph_piecewise_1D(this%c_context, arg, scale, offset, LOC(source),   &
                            INT(SIZE(source), KIND=C_LONG))

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create 1D piecewise node with complex float buffer.
!>
!>  @param[in,out] this   @ref graph_context instance.
!>  @param[in]     arg    The function argument.
!>  @param[in]     scale  Scale factor argument.
!>  @param[in]     offset Offset factor argument.
!>  @param[in]     source Source buffer to fill elements.
!>  @returns random(state)
!-------------------------------------------------------------------------------
      FUNCTION graph_context_piecewise_1D_cfloat(this, arg, scale, offset,     &
                                                 source)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR) :: graph_context_piecewise_1D_cfloat
      CLASS(graph_context), INTENT(INOUT)    :: this
      TYPE(C_PTR), INTENT(IN)                :: arg
      REAL(C_DOUBLE)                         :: scale
      REAL(C_DOUBLE)                         :: offset
      COMPLEX(C_FLOAT_COMPLEX), DIMENSION(:) :: source

!  Start of executable.
      graph_context_piecewise_1D_cfloat =                                      &
         graph_piecewise_1D(this%c_context, arg, scale, offset, LOC(source),   &
                            INT(SIZE(source), KIND=C_LONG))

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create 1D piecewise node with complex double buffer.
!>
!>  @param[in,out] this   @ref graph_context instance.
!>  @param[in]     arg    The function argument.
!>  @param[in]     scale  Scale factor argument.
!>  @param[in]     offset Offset factor argument.
!>  @param[in]     source Source buffer to fill elements.
!>  @returns random(state)
!-------------------------------------------------------------------------------
      FUNCTION graph_context_piecewise_1D_cdouble(this, arg, scale, offset,    &
                                                  source)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR) :: graph_context_piecewise_1D_cdouble
      CLASS(graph_context), INTENT(INOUT)     :: this
      TYPE(C_PTR), INTENT(IN)                 :: arg
      REAL(C_DOUBLE)                          :: scale
      REAL(C_DOUBLE)                          :: offset
      COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(:) :: source

!  Start of executable.
      graph_context_piecewise_1D_cdouble =                                     &
         graph_piecewise_1D(this%c_context, arg, scale, offset, LOC(source),   &
                            INT(SIZE(source), KIND=C_LONG))

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create 2D piecewise node with float buffer.
!>
!>  @param[in,out] this     @ref graph_context instance.
!>  @param[in]     x_arg    The function x argument.
!>  @param[in]     x_scale  Scale factor for x argument.
!>  @param[in]     x_offset Offset factor for x argument.
!>  @param[in]     y_arg    The function y argument.
!>  @param[in]     y_scale  Scale factor for y argument.
!>  @param[in]     y_offset Offset factor for y argument.
!>  @param[in]     source   Source buffer to fill elements.
!>  @returns random(state)
!-------------------------------------------------------------------------------
      FUNCTION graph_context_piecewise_2D_float(this,                          &
                                                x_arg, x_scale, x_offset,      &
                                                y_arg, y_scale, y_offset,      &
                                                source)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR) :: graph_context_piecewise_2D_float
      CLASS(graph_context), INTENT(INOUT) :: this
      TYPE(C_PTR), INTENT(IN)             :: x_arg
      REAL(C_DOUBLE)                      :: x_scale
      REAL(C_DOUBLE)                      :: x_offset
      TYPE(C_PTR), INTENT(IN)             :: y_arg
      REAL(C_DOUBLE)                      :: y_scale
      REAL(C_DOUBLE)                      :: y_offset
      REAL(C_FLOAT), DIMENSION(:,:)       :: source

!  Start of executable.
      graph_context_piecewise_2D_float =                                       &
         graph_piecewise_2D(this%c_context,                                    &
                            INT(SIZE(source, 1), KIND=C_LONG),                 &
                            x_arg, x_scale, x_offset,                          &
                            y_arg, y_scale, y_offset,                          &
                            LOC(source), INT(SIZE(source), KIND=C_LONG))

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create 2D piecewise node with double buffer.
!>
!>  @param[in,out] this     @ref graph_context instance.
!>  @param[in]     x_arg    The function x argument.
!>  @param[in]     x_scale  Scale factor for x argument.
!>  @param[in]     x_offset Offset factor for x argument.
!>  @param[in]     y_arg    The function y argument.
!>  @param[in]     y_scale  Scale factor for y argument.
!>  @param[in]     y_offset Offset factor for y argument.
!>  @param[in]     source   Source buffer to fill elements.
!>  @returns random(state)
!-------------------------------------------------------------------------------
      FUNCTION graph_context_piecewise_2D_double(this,                         &
                                                 x_arg, x_scale, x_offset,     &
                                                 y_arg, y_scale, y_offset,     &
                                                 source)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR) :: graph_context_piecewise_2D_double
      CLASS(graph_context), INTENT(INOUT) :: this
      TYPE(C_PTR), INTENT(IN)             :: x_arg
      REAL(C_DOUBLE)                      :: x_scale
      REAL(C_DOUBLE)                      :: x_offset
      TYPE(C_PTR), INTENT(IN)             :: y_arg
      REAL(C_DOUBLE)                      :: y_scale
      REAL(C_DOUBLE)                      :: y_offset
      REAL(C_DOUBLE), DIMENSION(:,:)      :: source

!  Start of executable.
      graph_context_piecewise_2D_double =                                      &
         graph_piecewise_2D(this%c_context,                                    &
                            INT(SIZE(source, 1), KIND=C_LONG),                 &
                            x_arg, x_scale, x_offset,                          &
                            y_arg, y_scale, y_offset,                          &
                            LOC(source), INT(SIZE(source), KIND=C_LONG))

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create 2D piecewise node with complex float buffer.
!>
!>  @param[in,out] this     @ref graph_context instance.
!>  @param[in]     x_arg    The function x argument.
!>  @param[in]     x_scale  Scale factor for x argument.
!>  @param[in]     x_offset Offset factor for x argument.
!>  @param[in]     y_arg    The function y argument.
!>  @param[in]     y_scale  Scale factor for y argument.
!>  @param[in]     y_offset Offset factor for y argument.
!>  @param[in]     source   Source buffer to fill elements.
!>  @returns random(state)
!-------------------------------------------------------------------------------
      FUNCTION graph_context_piecewise_2D_cfloat(this,                         &
                                                 x_arg, x_scale, x_offset,     &
                                                 y_arg, y_scale, y_offset,     &
                                                 source)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR) :: graph_context_piecewise_2D_cfloat
      CLASS(graph_context), INTENT(INOUT)      :: this
      TYPE(C_PTR), INTENT(IN)                  :: x_arg
      REAL(C_DOUBLE)                           :: x_scale
      REAL(C_DOUBLE)                           :: x_offset
      TYPE(C_PTR), INTENT(IN)                  :: y_arg
      REAL(C_DOUBLE)                           :: y_scale
      REAL(C_DOUBLE)                           :: y_offset
      COMPLEX(C_FLOAT_COMPLEX), DIMENSION(:,:) :: source

!  Start of executable.
      graph_context_piecewise_2D_cfloat =                                      &
         graph_piecewise_2D(this%c_context,                                    &
                            INT(SIZE(source, 1), KIND=C_LONG),                 &
                            x_arg, x_scale, x_offset,                          &
                            y_arg, y_scale, y_offset,                          &
                            LOC(source), INT(SIZE(source), KIND=C_LONG))

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Create 2D piecewise node with complex double buffer.
!>
!>  @param[in,out] this     @ref graph_context instance.
!>  @param[in]     x_arg    The function x argument.
!>  @param[in]     x_scale  Scale factor for x argument.
!>  @param[in]     x_offset Offset factor for x argument.
!>  @param[in]     y_arg    The function y argument.
!>  @param[in]     y_scale  Scale factor for y argument.
!>  @param[in]     y_offset Offset factor for y argument.
!>  @param[in]     source   Source buffer to fill elements.
!>  @returns random(state)
!-------------------------------------------------------------------------------
      FUNCTION graph_context_piecewise_2D_cdouble(this,                        &
                                                  x_arg, x_scale, x_offset,    &
                                                  y_arg, y_scale, y_offset,    &
                                                  source)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR) :: graph_context_piecewise_2D_cdouble
      CLASS(graph_context), INTENT(INOUT)       :: this
      TYPE(C_PTR), INTENT(IN)                   :: x_arg
      REAL(C_DOUBLE)                            :: x_scale
      REAL(C_DOUBLE)                            :: x_offset
      TYPE(C_PTR), INTENT(IN)                   :: y_arg
      REAL(C_DOUBLE)                            :: y_scale
      REAL(C_DOUBLE)                            :: y_offset
      COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(:,:) :: source

!  Start of executable.
      graph_context_piecewise_2D_cdouble =                                     &
         graph_piecewise_2D(this%c_context,                                    &
                            INT(SIZE(source, 1), KIND=C_LONG),                 &
                            x_arg, x_scale, x_offset,                          &
                            y_arg, y_scale, y_offset,                          &
                            LOC(source), INT(SIZE(source), KIND=C_LONG))

      END FUNCTION

!*******************************************************************************
!  JIT
!*******************************************************************************
!-------------------------------------------------------------------------------
!>  @brief Get the maximum number of concurrent devices.
!>
!>  @param[in] this @ref graph_context instance.
!>  @returns The number of devices.
!-------------------------------------------------------------------------------
      FUNCTION graph_context_get_max_concurrency(this)

      IMPLICIT NONE

!  Declare Arguments
      INTEGER(C_LONG)                     :: graph_context_get_max_concurrency
      CLASS(graph_context), INTENT(IN) :: this

!  Start of executable.
      graph_context_get_max_concurrency =                                      &
         graph_get_max_concurrency(this%c_context)

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Choose the device number.
!>
!>  @param[in] this @ref graph_context instance.
!>  @param[in] num The device number.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_set_device_number(this, num)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(INOUT) :: this
      INTEGER(C_LONG), INTENT(IN)         :: num

!  Start of executable.
      CALL graph_set_device_number(this%c_context, num)

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Add pre workflow item.
!>
!>  @param[in,out] this          @ref graph_context instance.
!>  @param[in]     inputs        Array of input nodes.
!>  @param[in]     outputs       Array of output nodes.
!>  @param[in]     map_inputs    Array of map input nodes.
!>  @param[in]     map_outputs   Array of map output nodes.
!>  @param[in]     random_state  Optional random state, can be NULL if not used.
!>  @param[in]     name          Name for the kernel.
!>  @param[in]     num_particles Number of elements to operate on.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_add_pre_item(this, inputs, outputs,             &
                                            map_inputs, map_outputs,           &
                                            random_state, name, num_particles)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(INOUT)           :: this
      INTEGER(C_INTPTR_T), DIMENSION(:), INTENT(IN) :: inputs
      INTEGER(C_INTPTR_T), DIMENSION(:), INTENT(IN) :: outputs
      INTEGER(C_INTPTR_T), DIMENSION(:), INTENT(IN) :: map_inputs
      INTEGER(C_INTPTR_T), DIMENSION(:), INTENT(IN) :: map_outputs
      TYPE(C_PTR), INTENT(IN)                       :: random_state
      CHARACTER(kind=C_CHAR,len=*), INTENT(IN)      :: name
      INTEGER(C_LONG), INTENT(IN)                   :: num_particles

!  Start of executable.
      CALL graph_add_pre_item(this%c_context,                                  &
                              LOC(inputs), INT(SIZE(inputs), KIND=C_LONG),     &
                              LOC(outputs), INT(SIZE(outputs), KIND=C_LONG),   &
                              LOC(map_inputs), LOC(map_outputs),               &
                              INT(SIZE(map_inputs), KIND=C_LONG),              &
                              random_state, name, num_particles)

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Add workflow item.
!>
!>  @param[in,out] this          @ref graph_context instance.
!>  @param[in]     inputs        Array of input nodes.
!>  @param[in]     outputs       Array of output nodes.
!>  @param[in]     map_inputs    Array of map input nodes.
!>  @param[in]     map_outputs   Array of map output nodes.
!>  @param[in]     random_state  Optional random state, can be NULL if not used.
!>  @param[in]     name          Name for the kernel.
!>  @param[in]     num_particles Number of elements to operate on.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_add_item(this, inputs, outputs,                 &
                                        map_inputs, map_outputs,               &
                                        random_state, name, num_particles)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(INOUT)           :: this
      INTEGER(C_INTPTR_T), DIMENSION(:), INTENT(IN) :: inputs
      INTEGER(C_INTPTR_T), DIMENSION(:), INTENT(IN) :: outputs
      INTEGER(C_INTPTR_T), DIMENSION(:), INTENT(IN) :: map_inputs
      INTEGER(C_INTPTR_T), DIMENSION(:), INTENT(IN) :: map_outputs
      TYPE(C_PTR), INTENT(IN)                       :: random_state
      CHARACTER(kind=C_CHAR,len=*), INTENT(IN)      :: name
      INTEGER(C_LONG), INTENT(IN)                   :: num_particles

!  Start of executable.
      CALL graph_add_item(this%c_context,                                      &
                          LOC(inputs), INT(SIZE(inputs), KIND=C_LONG),         &
                          LOC(outputs), INT(SIZE(outputs), KIND=C_LONG),       &
                          LOC(map_inputs), LOC(map_outputs),                   &
                          INT(SIZE(map_inputs), KIND=C_LONG),                  &
                          random_state, name, num_particles)

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Add workflow converge item.
!>
!>  @param[in,out] this          @ref graph_context instance.
!>  @param[in]     inputs        Array of input nodes.
!>  @param[in]     outputs       Array of output nodes.
!>  @param[in]     map_inputs    Array of map input nodes.
!>  @param[in]     map_outputs   Array of map output nodes.
!>  @param[in]     random_state  Optional random state, can be NULL if not used.
!>  @param[in]     name          Name for the kernel.
!>  @param[in]     num_particles Number of elements to operate on.
!>  @param[in]     tol           Maximum tolarance to converge to.
!>  @param[in]     max_iter      Maximum number of iterations for convergence.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_add_converge_item(this, inputs, outputs,        &
                                                 map_inputs, map_outputs,      &
                                                 random_state, name,           &
                                                 num_particles, tol, max_iter)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(INOUT)           :: this
      INTEGER(C_INTPTR_T), DIMENSION(:), INTENT(IN) :: inputs
      INTEGER(C_INTPTR_T), DIMENSION(:), INTENT(IN) :: outputs
      INTEGER(C_INTPTR_T), DIMENSION(:), INTENT(IN) :: map_inputs
      INTEGER(C_INTPTR_T), DIMENSION(:), INTENT(IN) :: map_outputs
      TYPE(C_PTR), INTENT(IN)                       :: random_state
      CHARACTER(kind=C_CHAR,len=*), INTENT(IN)      :: name
      INTEGER(C_LONG), INTENT(IN)                   :: num_particles
      REAL(C_DOUBLE), VALUE                         :: tol
      INTEGER(C_LONG), VALUE                        :: max_iter

!  Start of executable.
      CALL graph_add_converge_item(this%c_context, LOC(inputs),                &
                                   INT(SIZE(inputs), KIND=C_LONG),             &
                                   LOC(outputs),                               &
                                   INT(SIZE(outputs), KIND=C_LONG),            &
                                   LOC(map_inputs), LOC(map_outputs),          &
                                   INT(SIZE(map_inputs), KIND=C_LONG),         &
                                   random_state, name, num_particles,          &
                                   tol, max_iter)

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Compile the work items.
!>
!>  @param[in] this  @ref graph_context instance.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_compile(this)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(IN) :: this

!  Start of executable.
      CALL graph_compile(this%c_context)

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Run pre work items.
!>
!>  @param[in] this  @ref graph_context instance.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_pre_run(this)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(IN) :: this

!  Start of executable.
      CALL graph_pre_run(this%c_context)

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Run work items.
!>
!>  @param[in] this  @ref graph_context instance.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_run(this)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(IN) :: this

!  Start of executable.
      CALL graph_run(this%c_context)

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Wait for work items to complete.
!>
!>  @param[in] this  @ref graph_context instance.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_wait(this)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(IN) :: this

!  Start of executable.
      CALL graph_wait(this%c_context)

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Take derivative ∂f∂x.
!>
!>  @param[in,out] this  @ref graph_context instance.
!>  @param[in]     fnode The function expression to take the derivative of.
!>  @param[in]     xnode The expression to take the derivative with respect to.
!-------------------------------------------------------------------------------
      FUNCTION graph_context_df(this, fnode, xnode)

      IMPLICIT NONE

!  Declare Arguments
      TYPE(C_PTR)                         :: graph_context_df
      CLASS(graph_context), INTENT(INOUT) :: this
      TYPE(C_PTR), INTENT(IN)             :: fnode
      TYPE(C_PTR), INTENT(IN)             :: xnode

!  Start of executable.
      graph_context_df = graph_df(this%c_context, fnode, xnode)

      END FUNCTION

!-------------------------------------------------------------------------------
!>  @brief Copy float data to a device buffer.
!>
!>  @param[in] this   @ref graph_context instance.
!>  @param[in] node   Node to copy to.
!>  @param[in] source Source to copy from.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_copy_to_device_float(this, node, source)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(IN)        :: this
      TYPE(C_PTR), INTENT(IN)                 :: node
      REAL(C_FLOAT), DIMENSION(:), INTENT(IN) :: source

!  Start of executable.
      CALL graph_copy_to_device(this%c_context, node, LOC(source))

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Copy double data to a device buffer.
!>
!>  @param[in] this   @ref graph_context instance.
!>  @param[in] node   Node to copy to.
!>  @param[in] source Source to copy from.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_copy_to_device_double(this, node, source)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(IN)         :: this
      TYPE(C_PTR), INTENT(IN)                  :: node
      REAL(C_DOUBLE), DIMENSION(:), INTENT(IN) :: source

!  Start of executable.
      CALL graph_copy_to_device(this%c_context, node, LOC(source))

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Copy complex float data to a device buffer.
!>
!>  @param[in] this   @ref graph_context instance.
!>  @param[in] node   Node to copy to.
!>  @param[in] source Source to copy from.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_copy_to_device_cfloat(this, node, source)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(IN)                   :: this
      TYPE(C_PTR), INTENT(IN)                            :: node
      COMPLEX(C_FLOAT_COMPLEX), DIMENSION(:), INTENT(IN) :: source

!  Start of executable.
      CALL graph_copy_to_device(this%c_context, node, LOC(source))

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Copy complex double data to a device buffer.
!>
!>  @param[in] this   @ref graph_context instance.
!>  @param[in] node   Node to copy to.
!>  @param[in] source Source to copy from.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_copy_to_device_cdouble(this, node, source)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(IN)                    :: this
      TYPE(C_PTR), INTENT(IN)                             :: node
      COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(:), INTENT(IN) :: source

!  Start of executable.
      CALL graph_copy_to_device(this%c_context, node, LOC(source))

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Copy data to a host float buffer.
!>
!>  @param[in]     this        @ref graph_context instance.
!>  @param[in]     node        Node to copy from.
!>  @param[in,out] destination Host side buffer to copy to.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_copy_to_host_float(this, node, destination)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(IN)           :: this
      TYPE(C_PTR), VALUE                         :: node
      REAL(C_FLOAT), DIMENSION(:), INTENT(INOUT) :: destination

!  Start of executable.
      CALL graph_copy_to_host(this%c_context, node, LOC(destination))

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Copy data to a host double buffer.
!>
!>  @param[in]     c           The graph C context.
!>  @param[in]     node        Node to copy from.
!>  @param[in,out] destination Host side buffer to copy to.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_copy_to_host_double(this, node, destination)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(IN)            :: this
      TYPE(C_PTR), VALUE                          :: node
      REAL(C_DOUBLE), DIMENSION(:), INTENT(INOUT) :: destination

!  Start of executable.
      CALL graph_copy_to_host(this%c_context, node, LOC(destination))

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Copy data to a host complex float buffer.
!>
!>  @param[in]     c           The graph C context.
!>  @param[in]     node        Node to copy from.
!>  @param[in,out] destination Host side buffer to copy to.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_copy_to_host_cfloat(this, node, destination)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(IN)                      :: this
      TYPE(C_PTR), VALUE                                    :: node
      COMPLEX(C_FLOAT_COMPLEX), DIMENSION(:), INTENT(INOUT) :: destination

!  Start of executable.
      CALL graph_copy_to_host(this%c_context, node, LOC(destination))

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Copy data to a host complex double buffer.
!>
!>  @param[in]     c           The graph C context.
!>  @param[in]     node        Node to copy from.
!>  @param[in,out] destination Host side buffer to copy to.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_copy_to_host_cdouble(this, node, destination)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(IN)                       :: this
      TYPE(C_PTR), VALUE                                     :: node
      COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(:), INTENT(INOUT) :: destination

!  Start of executable.
      CALL graph_copy_to_host(this%c_context, node, LOC(destination))

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Print a value from nodes.
!>
!>  @param[in] c         The graph C context.
!>  @param[in] index     Particle index to print.
!>  @param[in] nodes     Nodes to print.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_context_print(this, index, nodes)

      IMPLICIT NONE

!  Declare Arguments
      CLASS(graph_context), INTENT(IN)              :: this
      INTEGER(C_LONG), INTENT(IN)                   :: index
      INTEGER(C_INTPTR_T), DIMENSION(:), INTENT(IN) :: nodes

!  Start of executable.
      CALL graph_print(this%c_context, index, LOC(nodes),                      &
                       INT(SIZE(nodes), KIND=C_LONG))

      END SUBROUTINE

      END MODULE
