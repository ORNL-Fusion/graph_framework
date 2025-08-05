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
!>  @brief Assert check.
!>
!>  If the assert check does not pass write error to standard error and exit.
!>
!>  @param[in] test    The check test.
!>  @param[in] message Message to report if check fails.
!-------------------------------------------------------------------------------
      SUBROUTINE assert(test, message)
      USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY : error_unit

      IMPLICIT NONE

!  Declare Arguments
      LOGICAL, INTENT(IN) :: test
      CHARACTER(len=*)    :: message

!  Start of executable code.
      IF (.not.test) THEN
         WRITE(error_unit,*) message
         CALL exit(1)
      END IF

      END SUBROUTINE

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
      TYPE(C_PTR)                   :: x
      TYPE(C_PTR)                   :: m
      TYPE(C_PTR)                   :: b
      REAL(C_FLOAT), DIMENSION(1)   :: value
      TYPE(C_PTR)                   :: px
      TYPE(C_PTR)                   :: y
      TYPE(C_PTR)                   :: one
      TYPE(C_PTR)                   :: zero
      INTEGER(C_LONG)               :: size
      TYPE(C_PTR)                   :: rand
      REAL(C_FLOAT), DIMENSION(3)   :: buffer1D
      TYPE(C_PTR)                   :: p1
      TYPE(C_PTR)                   :: i
      REAL(C_FLOAT), DIMENSION(3,3) :: buffer2D
      TYPE(C_PTR)                   :: p2
      TYPE(C_PTR)                   :: j

!  Start of executable code.
      graph => graph_float_context(use_safe_math)

      x = graph%variable(1_C_LONG, 'x' // C_NULL_CHAR)
      m = graph%constant(0.5_C_DOUBLE)
      b = graph%constant(0.2_C_DOUBLE)

      value(1) = 2.0
      CALL graph%set_variable(x, value)

      px = graph%pseudo_variable(x)
      CALL assert(TRANSFER(px, 0) .ne. TRANSFER(x, 0),                         &
                  'Expected different nodes.')
      CALL assert(TRANSFER(graph%remove_pseudo(px), 0) .eq. TRANSFER(x, 0),    &
                  'Remove pseudo failed.')

      y = graph%add(graph%mul(m, x), b)

      one = graph%constant(1.0_C_DOUBLE)
      zero = graph%constant(0.0_C_DOUBLE)

      CALL assert(TRANSFER(graph%sub(one, one), 0) .eq. TRANSFER(zero, 0),     &
                  'Expected 1 - 1 = 0.')
      CALL assert(TRANSFER(graph%div(one, one), 0) .eq. TRANSFER(one, 0),      &
                  'Expected 1/1 = 1.')
      CALL assert(TRANSFER(graph%sqrt(one), 0) .eq. TRANSFER(one, 0),          &
                  'Expected sqrt(1) = 1.')
      CALL assert(TRANSFER(graph%exp(zero), 0) .eq. TRANSFER(one, 0),          &
                  'Expected exp(0) = 1.')
      CALL assert(TRANSFER(graph%log(one), 0) .eq. TRANSFER(zero, 0),          &
                  'Expected log(1) = 0.')
      CALL assert(TRANSFER(graph%pow(one, zero), 0) .eq. TRANSFER(one, 0),     &
                  'Expected pow(1, 0) = 1.')
      CALL assert(TRANSFER(graph%sin(zero), 0) .eq. TRANSFER(zero, 0),         &
                  'Expected sin(0) = 0.')
      CALL assert(TRANSFER(graph%cos(zero), 0) .eq. TRANSFER(one, 0),          &
                  'Expected cos(0) = 1.')
      CALL assert(TRANSFER(graph%atan(one, zero), 0) .eq. TRANSFER(zero, 0),   &
                  'Expected atan(one, zero) = zero.')

      rand = graph%random(graph%random_state(0))

      i = graph%variable(1_C_LONG, 'i' // C_NULL_CHAR)
      buffer1D = (/ 2.0, 4.0, 6.0 /)
      p1 = graph%piecewise_1D(i, 1.0_C_DOUBLE, 0.0_C_DOUBLE, buffer1D)

      j = graph%variable(1_C_LONG, 'j' // C_NULL_CHAR)
      buffer2D = RESHAPE((/ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 /),    &
                         SHAPE(buffer2D))
      p2 = graph%piecewise_2D(i, 1.0_C_DOUBLE, 0.0_C_DOUBLE,                   &
                              j, 1.0_C_DOUBLE, 0.0_C_DOUBLE, buffer2D)

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
      LOGICAL(C_BOOL), INTENT(IN)    :: use_safe_math

!  Local variables.
      CLASS(graph_context), POINTER  :: graph
      TYPE(C_PTR)                    :: x
      TYPE(C_PTR)                    :: m
      TYPE(C_PTR)                    :: b
      REAL(C_DOUBLE), DIMENSION(1)   :: value
      TYPE(C_PTR)                    :: px
      TYPE(C_PTR)                    :: y
      TYPE(C_PTR)                    :: one
      TYPE(C_PTR)                    :: zero
      INTEGER(C_LONG)                :: size
      TYPE(C_PTR)                    :: rand
      REAL(C_DOUBLE), DIMENSION(3)   :: buffer1D
      TYPE(C_PTR)                    :: p1
      TYPE(C_PTR)                    :: i
      REAL(C_DOUBLE), DIMENSION(3,3) :: buffer2D
      TYPE(C_PTR)                    :: p2
      TYPE(C_PTR)                    :: j

!  Start of executable code.
      graph => graph_double_context(use_safe_math)

      x = graph%variable(1_C_LONG, 'x' // C_NULL_CHAR)
      m = graph%constant(0.5_C_DOUBLE)
      b = graph%constant(0.2_C_DOUBLE)

      value(1) = 2.0
      CALL graph%set_variable(x, value)

      px = graph%pseudo_variable(x)
      CALL assert(TRANSFER(px, 0) .ne. TRANSFER(x, 0),                         &
                  'Expected different nodes.')
      CALL assert(TRANSFER(graph%remove_pseudo(px), 0) .eq. TRANSFER(x, 0),    &
                  'Remove pseudo failed.')

      y = graph%add(graph%mul(m, x), b)

      one = graph%constant(1.0_C_DOUBLE)
      zero = graph%constant(0.0_C_DOUBLE)

      CALL assert(TRANSFER(graph%sub(one, one), 0) .eq. TRANSFER(zero, 0),     &
                  'Expected 1 - 1 = 0.')
      CALL assert(TRANSFER(graph%div(one, one), 0) .eq. TRANSFER(one, 0),      &
                  'Expected 1/1 = 1.')
      CALL assert(TRANSFER(graph%sqrt(one), 0) .eq. TRANSFER(one, 0),          &
                  'Expected sqrt(1) = 1.')
      CALL assert(TRANSFER(graph%exp(zero), 0) .eq. TRANSFER(one, 0),          &
                  'Expected exp(0) = 1.')
      CALL assert(TRANSFER(graph%log(one), 0) .eq. TRANSFER(zero, 0),          &
                  'Expected log(1) = 0.')
      CALL assert(TRANSFER(graph%pow(one, zero), 0) .eq. TRANSFER(one, 0),     &
                  'Expected pow(1, 0) = 1.')
      CALL assert(TRANSFER(graph%sin(zero), 0) .eq. TRANSFER(zero, 0),         &
                  'Expected sin(0) = 0.')
      CALL assert(TRANSFER(graph%cos(zero), 0) .eq. TRANSFER(one, 0),          &
                  'Expected cos(0) = 1.')
      CALL assert(TRANSFER(graph%atan(one, zero), 0) .eq. TRANSFER(zero, 0),   &
                  'Expected atan(one, zero) = zero.')

      rand = graph%random(graph%random_state(0))

      i = graph%variable(1_C_LONG, 'i' // C_NULL_CHAR)
      buffer1D = (/ 2.0, 4.0, 6.0 /)
      p1 = graph%piecewise_1D(i, 1.0_C_DOUBLE, 0.0_C_DOUBLE, buffer1D)

      j = graph%variable(1_C_LONG, 'j' // C_NULL_CHAR)
      buffer2D = RESHAPE((/ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 /),    &
                         SHAPE(buffer2D))
      p2 = graph%piecewise_2D(i, 1.0_C_DOUBLE, 0.0_C_DOUBLE,                   &
                              j, 1.0_C_DOUBLE, 0.0_C_DOUBLE, buffer2D)

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
      LOGICAL(C_BOOL), INTENT(IN)              :: use_safe_math

!  Local variables.
      CLASS(graph_context), POINTER            :: graph
      TYPE(C_PTR)                              :: x
      TYPE(C_PTR)                              :: m
      TYPE(C_PTR)                              :: b
      COMPLEX(C_FLOAT_COMPLEX), DIMENSION(1)   :: value
      TYPE(C_PTR)                              :: px
      TYPE(C_PTR)                              :: y
      TYPE(C_PTR)                              :: one
      TYPE(C_PTR)                              :: zero
      INTEGER(C_LONG)                          :: size
      TYPE(C_PTR)                              :: rand
      COMPLEX(C_FLOAT_COMPLEX), DIMENSION(3)   :: buffer1D
      TYPE(C_PTR)                              :: p1
      TYPE(C_PTR)                              :: i
      COMPLEX(C_FLOAT_COMPLEX), DIMENSION(3,3) :: buffer2D
      TYPE(C_PTR)                              :: p2
      TYPE(C_PTR)                              :: j

!  Start of executable code.
      graph => graph_complex_float_context(use_safe_math)

      x = graph%variable(1_C_LONG, 'x' // C_NULL_CHAR)
      m = graph%constant(0.5_C_DOUBLE)
      b = graph%constant(0.2_C_DOUBLE, 0.0_C_DOUBLE)

      value(1) = 2.0
      CALL graph%set_variable(x, value)

      px = graph%pseudo_variable(x)
      CALL assert(TRANSFER(px, 0) .ne. TRANSFER(x, 0),                         &
                  'Expected different nodes.')
      CALL assert(TRANSFER(graph%remove_pseudo(px), 0) .eq. TRANSFER(x, 0),    &
                  'Remove pseudo failed.')

      y = graph%add(graph%mul(m, x), b)

      one = graph%constant(1.0_C_DOUBLE)
      zero = graph%constant(0.0_C_DOUBLE)

      CALL assert(TRANSFER(graph%sub(one, one), 0) .eq. TRANSFER(zero, 0),     &
                  'Expected 1 - 1 = 0.')
      CALL assert(TRANSFER(graph%div(one, one), 0) .eq. TRANSFER(one, 0),      &
                  'Expected 1/1 = 1.')
      CALL assert(TRANSFER(graph%sqrt(one), 0) .eq. TRANSFER(one, 0),          &
                  'Expected sqrt(1) = 1.')
      CALL assert(TRANSFER(graph%exp(zero), 0) .eq. TRANSFER(one, 0),          &
                  'Expected exp(0) = 1.')
      CALL assert(TRANSFER(graph%log(one), 0) .eq. TRANSFER(zero, 0),          &
                  'Expected log(1) = 0.')
      CALL assert(TRANSFER(graph%pow(one, zero), 0) .eq. TRANSFER(one, 0),     &
                  'Expected pow(1, 0) = 1.')
      CALL assert(TRANSFER(graph%erfi(zero), 0) .eq. TRANSFER(zero, 0),        &
                  'Expected erfi(0) = 0.')
      CALL assert(TRANSFER(graph%sin(zero), 0) .eq. TRANSFER(zero, 0),         &
                  'Expected sin(0) = 0.')
      CALL assert(TRANSFER(graph%cos(zero), 0) .eq. TRANSFER(one, 0),          &
                  'Expected cos(0) = 1.')
      CALL assert(TRANSFER(graph%atan(one, zero), 0) .eq. TRANSFER(zero, 0),   &
                  'Expected atan(one, zero) = zero.')

      rand = graph%random(graph%random_state(0))

      i = graph%variable(1_C_LONG, 'i' // C_NULL_CHAR)
      buffer1D = (/ CMPLX(2.0, 0.0), CMPLX(4.0, 0.0), CMPLX(6.0, 0.0) /)
      p1 = graph%piecewise_1D(i, 1.0_C_DOUBLE, 0.0_C_DOUBLE, buffer1D)

      j = graph%variable(1_C_LONG, 'j' // C_NULL_CHAR)
      buffer2D = RESHAPE((/ CMPLX(1.0, 0.0), CMPLX(2.0, 0.0), CMPLX(3.0, 0.0), &
                            CMPLX(4.0, 0.0), CMPLX(5.0, 0.0), CMPLX(6.0, 0.0), &
                            CMPLX(7.0, 0.0), CMPLX(8.0, 0.0), CMPLX(9.0, 0.0)  &
                         /), SHAPE(buffer2D))
      p2 = graph%piecewise_2D(i, 1.0_C_DOUBLE, 0.0_C_DOUBLE,                   &
                              j, 1.0_C_DOUBLE, 0.0_C_DOUBLE, buffer2D)

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
      LOGICAL(C_BOOL), INTENT(IN)               :: use_safe_math

!  Local variables.
      CLASS(graph_context), POINTER             :: graph
      TYPE(C_PTR)                               :: x
      TYPE(C_PTR)                               :: m
      TYPE(C_PTR)                               :: b
      COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(1)   :: value
      TYPE(C_PTR)                               :: px
      TYPE(C_PTR)                               :: y
      TYPE(C_PTR)                               :: one
      TYPE(C_PTR)                               :: zero
      INTEGER(C_LONG)                           :: size
      TYPE(C_PTR)                               :: rand
      COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(3)   :: buffer1D
      TYPE(C_PTR)                               :: p1
      TYPE(C_PTR)                               :: i
      COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(3,3) :: buffer2D
      TYPE(C_PTR)                               :: p2
      TYPE(C_PTR)                               :: j

!  Start of executable code.
      graph => graph_complex_double_context(use_safe_math)

      x = graph%variable(1_C_LONG, 'x' // C_NULL_CHAR)
      m = graph%constant(0.5_C_DOUBLE)
      b = graph%constant(0.2_C_DOUBLE, 0.0_C_DOUBLE)

      value(1) = 2.0
      CALL graph%set_variable(x, value)

      px = graph%pseudo_variable(x)
      CALL assert(TRANSFER(px, 0) .ne. TRANSFER(x, 0),                         &
                  'Expected different nodes.')
      CALL assert(TRANSFER(graph%remove_pseudo(px), 0) .eq. TRANSFER(x, 0),    &
                  'Remove pseudo failed.')

      y = graph%add(graph%mul(m, x), b)

      one = graph%constant(1.0_C_DOUBLE)
      zero = graph%constant(0.0_C_DOUBLE)

      CALL assert(TRANSFER(graph%sub(one, one), 0) .eq. TRANSFER(zero, 0),     &
                  'Expected 1 - 1 = 0.')
      CALL assert(TRANSFER(graph%div(one, one), 0) .eq. TRANSFER(one, 0),      &
                  'Expected 1/1 = 1.')
      CALL assert(TRANSFER(graph%sqrt(one), 0) .eq. TRANSFER(one, 0),          &
                  'Expected sqrt(1) = 1.')
      CALL assert(TRANSFER(graph%exp(zero), 0) .eq. TRANSFER(one, 0),          &
                  'Expected exp(0) = 1.')
      CALL assert(TRANSFER(graph%log(one), 0) .eq. TRANSFER(zero, 0),          &
                  'Expected log(1) = 0.')
      CALL assert(TRANSFER(graph%pow(one, zero), 0) .eq. TRANSFER(one, 0),     &
                  'Expected pow(1, 0) = 1.')
      CALL assert(TRANSFER(graph%erfi(zero), 0) .eq. TRANSFER(zero, 0),        &
                  'Expected erfi(0) = 0.')
      CALL assert(TRANSFER(graph%sin(zero), 0) .eq. TRANSFER(zero, 0),         &
                  'Expected sin(0) = 0.')
      CALL assert(TRANSFER(graph%cos(zero), 0) .eq. TRANSFER(one, 0),          &
                  'Expected cos(0) = 1.')
      CALL assert(TRANSFER(graph%atan(one, zero), 0) .eq. TRANSFER(zero, 0),   &
                  'Expected atan(one, zero) = zero.')

      rand = graph%random(graph%random_state(0))

      i = graph%variable(1_C_LONG, 'i' // C_NULL_CHAR)
      buffer1D = (/ CMPLX(2.0, 0.0), CMPLX(4.0, 0.0), CMPLX(6.0, 0.0) /)
      p1 = graph%piecewise_1D(i, 1.0_C_DOUBLE, 0.0_C_DOUBLE, buffer1D)

      j = graph%variable(1_C_LONG, 'j' // C_NULL_CHAR)
      buffer2D = RESHAPE((/ CMPLX(1.0, 0.0), CMPLX(2.0, 0.0), CMPLX(3.0, 0.0), &
                            CMPLX(4.0, 0.0), CMPLX(5.0, 0.0), CMPLX(6.0, 0.0), &
                            CMPLX(7.0, 0.0), CMPLX(8.0, 0.0), CMPLX(9.0, 0.0)  &
                         /), SHAPE(buffer2D))
      p2 = graph%piecewise_2D(i, 1.0_C_DOUBLE, 0.0_C_DOUBLE,                   &
                              j, 1.0_C_DOUBLE, 0.0_C_DOUBLE, buffer2D)

      DEALLOCATE(graph)

      END SUBROUTINE
