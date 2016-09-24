! Library containing a number ad hoc routines relating to linear algebra.
!
! Description
! -----------
!
! For the most part, this code wraps existing LAPACK routines to
! perform linear algebra operations in a more straightforward
! fashion.
!
! Author
! ------
!
! Stanislav Sokolenko, University of Waterloo
!
! History
! -------
!
! June 23/2014 
!   - brought over "v_norm" from previous code
!   - brought over "m_svd" from previous code
!   - brought over "m_rank" from previous code
!   - brought over "m_nullity" from previous code
!   - brought over "m_kernel" from previous code
! July 27/2014
!   - added "v_ssq"

module algebra 

  implicit none
  private

  !=====================================================================>
  ! Interfaces with LAPACK

  !---------------------------------------------------------------------
  ! singular value decomposition
  interface
    subroutine dgesdd(jobz, m, n, A, lda, s, U, ldu, VT, ldvt,                &
                      work, lwork, iwork, info)
      character :: jobz
      integer :: info, lda, ldu, ldvt, lwork, m, n
      integer :: iwork(*)
      double precision :: A(lda, *), s(*), U(ldu, *), VT(ldvt, *), work(*)
    end subroutine dgesdd
  end interface

  !---------------------------------------------------------------------
  ! sum of squares
  interface
    subroutine dlassq(n, x, incx, scale, sumsq)
      integer, intent(in) :: n, incx
      double precision, intent(in) :: x(*)
      double precision, intent(inout) :: scale, sumsq
    end subroutine dlassq
  end interface

  public :: v_norm, v_ssq, m_svd, m_rank, m_nullity, m_kernel

  contains

    !===================================================================>
    ! Convenience functions and subroutines relating to basic linear
    ! algebra

    !-------------------------------------------------------------------
    function v_ssq(x)
      ! Calculates the squared sum of a vector using Lapack's dlassq
      !
      ! Parameters
      ! ----------
      ! x: double precision(:)
      !   Vector whose norm is to be calculated
      !
      ! Returns
      ! -------
      ! v_norm: double precision
      !   Vector norm of x.

      ! Function arguments
      double precision, intent(in) :: x(:)
      double precision :: v_ssq

      ! Local variables
      integer :: n, incx
      double precision :: scale, sumsq

      n = size(x)
      incx = 1

      scale = 0d0
      sumsq = 1d0

      call dlassq(n, x, incx, scale, sumsq)

      v_ssq = scale**2 * sumsq

    end function v_ssq

    !-------------------------------------------------------------------
    function v_norm(x)
      ! Calculates the norm of a vector using Lapack's dlassq
      !
      ! Parameters
      ! ----------
      ! x: double precision(:)
      !   Vector whose norm is to be calculated
      !
      ! Returns
      ! -------
      ! v_norm: double precision
      !   Vector norm of x.

      ! Function arguments
      double precision, intent(in) :: x(:)
      double precision :: v_norm

      v_norm = sqrt(v_ssq(x))

    end function v_norm

    !===================================================================>
    ! Convenience functions and subroutines relating to singular value
    ! decomposition and eigenvalue calculation.

    !-------------------------------------------------------------------
    subroutine m_svd(A, m, n, U, s, VT, job)
      ! Performs singular value decomposition of A.
      !
      ! A = U * sigma * transpose(V)
      !
      ! Parameters
      ! ----------
      ! A: double precision (m, n)
      !   The array on which the singular value decomposition is to be
      !   performed.
      ! m: integer
      !   Row number of A.
      ! n: integer
      !   Column number of A.
      ! job: character (1)
      !   'A' to compute U, s, and VT; 'N' to compute only s
      !
      ! Returns
      ! -------
      ! U: double precision (m, m)
      ! s: double precision (min(m, n)) 
      !   Diagonal element of sigma and singular values of A.
      ! VT: double precision (n, n)
      !   The transpose of V.

      ! Subroutine arguments
      character, intent(in) :: job
      integer, intent(in) :: m, n
      double precision, intent(in) :: A(m, n)
      double precision, intent(out) :: U(m, m), s(min(m, n)), VT(n, n)

      ! Local variables
      integer :: lwork, lda, ldu, ldvt, info
      double precision :: work_query(1), A_work(m, n)

      integer, allocatable :: iwork(:)
      double precision, allocatable :: work(:)

      ! Copying A so it's not destroyed
      A_work = A

      ! Variable assignment
      lda = m
      ldu = m
      ldvt = n

      allocate(iwork(8*min(m, n)))

      ! Determining optimal lwork
      lwork = -1
      info = 0
      call dgesdd(job, m, n, A_work, lda, s, U, ldu, VT, ldvt,                &
                  work_query, lwork, iwork, info)

      ! Allocating appropriately sized work array
      lwork = nint(work_query(1))
      allocate(work(lwork))

      ! Performing SVD
      call dgesdd(job, m, n, A_work, lda, s, U, ldu, VT, ldvt,                &
                  work, lwork, iwork, info)

    end subroutine m_svd

    !-------------------------------------------------------------------
    function m_rank(A)
      ! Calculates rank of matrix A by performing singular value
      ! decomposition and counting non-zero singular values.
      !
      ! Parameters
      ! ----------
      ! A: double precision (m, n)
      !   The array for which the rank is calculated.
      !
      ! Returns
      ! -------
      ! m_rank: integer
      !   Rank of A.

      ! Subroutine arguments
      double precision, intent(in) :: A(:, :)
      integer :: m_rank

      ! Local variables
      integer :: A_dim(2), m, n
      double precision, allocatable :: U(:, :), s(:), VT(:, :) 

      ! Variable assignment
      A_dim = shape(A)
      m = A_dim(1)
      n = A_dim(2)

      allocate(U(m, m))
      allocate(s(min(m, n)))
      allocate(VT(n, n))

      call m_svd(A, m, n, U, s, VT, 'N')

      m_rank = count(abs(s) > 1d-8)

    end function m_rank

    !-------------------------------------------------------------------
    function m_nullity(A)
      ! Calculates rank of the nullspace of matrix A by first
      ! calculating the rank through singular value decomposition.
      !
      ! Parameters
      ! ----------
      ! A: double precision (m, n)
      !   The array for which the nullity is calculated.
      !
      ! Returns
      ! -------
      ! m_nullity: integer
      !   Nullity of A.

      ! Subroutine arguments
      double precision, intent(in) :: A(:, :)
      integer ::m_nullity

      ! Local variables
      integer :: A_dim(2)

      A_dim = shape(A)
      m_nullity = A_dim(2) - m_rank(A)

    end function m_nullity

    !-------------------------------------------------------------------
    function m_kernel(A)
      ! Calculates the nullspace of A.
      !
      ! Parameters
      ! ----------
      ! A: double precision (m, n)
      !   The array for which the nullspace is calculated.
      !
      ! Returns
      ! -------
      ! m_kernel: double precision
      !   Nullspace of A.

      ! Subroutine arguments
      double precision, intent(in) :: A(:, :)
      double precision, allocatable ::m_kernel(:, :)

      ! Local variables
      integer :: A_dim(2), m, n, nullity
      double precision, allocatable :: U(:, :), s(:), VT(:, :) 

      ! Getting nullspace dimensions
      nullity = m_nullity(A)

      ! Calculating VT matrix
      A_dim = shape(A)
      m = A_dim(1)
      n = A_dim(2)

      allocate(U(m, m))
      allocate(s(min(m, n)))
      allocate(VT(n, n))

      call m_svd(A, m, n, U, s, VT, 'A')

      ! Extracting nullspace
      allocate(m_kernel(n, nullity))

      m_kernel = transpose(VT((n - nullity + 1):n, :))

    end function m_kernel

end module algebra

