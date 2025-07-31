!-------------------------------------------------------------------------------
!>  @file f_binding_test.f90
!>  @brief Test for fortran bindings.
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
!>  @brief Main test program.
!-------------------------------------------------------------------------------
      PROGRAM f_binding_test
      USE, INTRINSIC :: ISO_C_BINDING

      IMPLICIT NONE

!  Define parameters.
      LOGICAL(C_BOOL), PARAMETER :: c_true = .true.
      LOGICAL(C_BOOL), PARAMETER :: c_false = .false.

!  Start of executable code.
      CALL run_test_float(c_true)
      CALL run_test_float(c_false)
      CALL run_test_double(c_true)
      CALL run_test_double(c_false)
      CALL run_test_complex_float(c_true)
      CALL run_test_complex_float(c_false)
      CALL run_test_complex_double(c_true)
      CALL run_test_complex_double(c_false)

      END PROGRAM

!-------------------------------------------------------------------------------
!>  @brief Run float tests.
!>
!>  @param[in] c_type        Base type of the calculation.
!>  @param[in] use_safe_math Use safe math.
!-------------------------------------------------------------------------------
      SUBROUTINE run_test_float(use_safe_math)

      USE graph_fortran
      USE, INTRINSIC :: ISO_C_BINDING

      IMPLICIT NONE

!  Declare Arguments
      LOGICAL(C_BOOL), INTENT(IN)   :: use_safe_math

!  Local variables.
      CLASS(graph_context), POINTER :: graph

!  Start of executable code.
      graph => graph_float_context(use_safe_math)
      DEALLOCATE(graph)

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Run double tests.
!>
!>  @param[in] c_type        Base type of the calculation.
!>  @param[in] use_safe_math Use safe math.
!-------------------------------------------------------------------------------
      SUBROUTINE run_test_double(use_safe_math)

      USE graph_fortran
      USE, INTRINSIC :: ISO_C_BINDING

      IMPLICIT NONE

!  Declare Arguments
      LOGICAL(C_BOOL), INTENT(IN)   :: use_safe_math

!  Local variables.
      CLASS(graph_context), POINTER :: graph

!  Start of executable code.
      graph => graph_double_context(use_safe_math)
      DEALLOCATE(graph)

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Run complex float tests.
!>
!>  @param[in] c_type        Base type of the calculation.
!>  @param[in] use_safe_math Use safe math.
!-------------------------------------------------------------------------------
      SUBROUTINE run_test_complex_float(use_safe_math)

      USE graph_fortran
      USE, INTRINSIC :: ISO_C_BINDING

      IMPLICIT NONE

!  Declare Arguments
      LOGICAL(C_BOOL), INTENT(IN)   :: use_safe_math

!  Local variables.
      CLASS(graph_context), POINTER :: graph

!  Start of executable code.
      graph => graph_complex_float_context(use_safe_math)
      DEALLOCATE(graph)

      END SUBROUTINE

!-------------------------------------------------------------------------------
!>  @brief Run double tests.
!>
!>  @param[in] c_type        Base type of the calculation.
!>  @param[in] use_safe_math Use safe math.
!-------------------------------------------------------------------------------
      SUBROUTINE run_test_complex_double(use_safe_math)

      USE graph_fortran
      USE, INTRINSIC :: ISO_C_BINDING

      IMPLICIT NONE

!  Declare Arguments
      LOGICAL(C_BOOL), INTENT(IN)   :: use_safe_math

!  Local variables.
      CLASS(graph_context), POINTER :: graph

!  Start of executable code.
      graph => graph_complex_double_context(use_safe_math)
      DEALLOCATE(graph)

      END SUBROUTINE
