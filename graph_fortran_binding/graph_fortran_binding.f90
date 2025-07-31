!-------------------------------------------------------------------------------
!>  @file graph_fortran_binding.f90
!>  @brief Implimentation of the Fortran binding library.
!
!  Note separating the Doxygen comment block here so the detailed description is
!  found in the Module not the file.
!
!> Module contains subroutines for calling this from fortran.
!-------------------------------------------------------------------------------
      MODULE graph_fortran
      USE, INTRINSIC :: ISO_C_BINDING

      IMPLICIT NONE

!*******************************************************************************
!  PARAMETERS
!*******************************************************************************
!>  Float type.
    INTEGER(C_INT8_T), PARAMETER :: FLOAT_T = 0
!>  Double type.
    INTEGER(C_INT8_T), PARAMETER :: DOUBLE_T = 1
!>  Complex Float type.
    INTEGER(C_INT8_T), PARAMETER :: COMPLEX_FLOAT_T = 2
!>  Complex Double type.
    INTEGER(C_INT8_T), PARAMETER :: COMPLEX_DOUBLE_T = 3

!-------------------------------------------------------------------------------
!>  @brief Class object for the binding.
!-------------------------------------------------------------------------------
      TYPE :: graph_context
!>  The auto release pool context.
         TYPE(C_PTR) :: arp_context
!>  The graph c context.
         TYPE(C_PTR) :: c_context
      CONTAINS
         FINAL :: graph_destruct
      END TYPE

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
         INTEGER(C_INT8_T), VALUE :: c_type
         LOGICAL(C_BOOL), VALUE    :: use_safe_math
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

      END INTERFACE

      CONTAINS

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
      CLASS (graph_context), POINTER :: graph_construct_float
      LOGICAL(C_BOOL), INTENT(IN)    :: use_safe_math

!  Start of executable code.
      ALLOCATE(graph_construct_float)
      graph_construct_float%arp_context = objc_autoreleasePoolPush()
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
      CLASS (graph_context), POINTER :: graph_construct_double
      LOGICAL(C_BOOL), INTENT(IN)    :: use_safe_math

!  Start of executable code.
      ALLOCATE(graph_construct_double)
      graph_construct_double%arp_context = objc_autoreleasePoolPush()
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
      CLASS (graph_context), POINTER :: graph_construct_complex_float
      LOGICAL(C_BOOL), INTENT(IN)    :: use_safe_math

!  Start of executable code.
      ALLOCATE(graph_construct_complex_float)
      graph_construct_complex_float%arp_context = objc_autoreleasePoolPush()
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
      CLASS (graph_context), POINTER :: graph_construct_complex_double
      LOGICAL(C_BOOL), INTENT(IN)    :: use_safe_math

!  Start of executable code.
      ALLOCATE(graph_construct_complex_double)
      graph_construct_complex_double%arp_context = objc_autoreleasePoolPush()
      graph_construct_complex_double%c_context =                               &
         graph_construct_context(COMPLEX_DOUBLE_T, use_safe_math)

      END FUNCTION

!*******************************************************************************
! DESTRUCTION SUBROUTINES
!*******************************************************************************
!-------------------------------------------------------------------------------
!>  @brief Deconstruct a @ref graph_context object.
!>
!>  Deallocate memory and unitialize a @ref graph_context object.
!>
!>  @param[input] this A @ref graph_context instance.
!-------------------------------------------------------------------------------
      SUBROUTINE graph_destruct(this)

      IMPLICIT NONE

!  Declare Arguments
      TYPE (graph_context), INTENT(INOUT) :: this

!  Start of executable.
      CALL objc_autoreleasePoolPop(this%arp_context)
      CALL graph_destroy_context(this%c_context)

      END SUBROUTINE

      END MODULE
