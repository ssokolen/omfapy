! Library containing a number of routines used to generate constrained
! samples from stoichiometry models.
!
! Description
! -----------
!
! The code is meant to be used as part of the OMFA Python package for
! metabolic flux analysis. A couple of algorithms for random sampling 
! of a constrained (stoichiometric) model solution space are
! implemented here in Fortran as a way to speed up the original pure
! Python implementation.
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
!   - brought over "rda_sample" and "rda_step" from previous code
! July 25/2014
!   - modified "rda_sample" and "convert_constraints" to work on 
!     nullspace rather than stoichiometry input
!

module realization

  use random, only : r_init, r_uniform, r_normal_v
  use algebra, only : v_norm, v_ssq

  implicit none

  contains

    !===================================================================>
    ! Misc. routines 

    !-------------------------------------------------------------------
    subroutine convert_constraints(K, xo, lb, i_lb, ub, i_ub,                 &
                                   G, h, q)
      ! Converts constraints on the lower and upper values of fluxes
      ! defined by the stoichiometry matrix:
      ! Sx >= lb, Sx <= ub 
      ! into strictly lower values of the basis variables, such that:
      ! Gq >= h
      !
      ! Parameters
      ! ----------
      ! K: double precision (:, :)
      !   Nullspace of the stoichiometry matrix (S) in the form of Sx = 0.
      ! xo: double precision(:)
      !   Starting point for sampling algorithm that satisfies: 
      !   Sx = 0, x(i_ub) <= ub, x(i_lb) >= lb.
      ! lb: double precision (:)
      !   Lower boundaries on a subset of fluxes x in Sx = 0.
      ! i_lb: integer (:)
      !   The columns of K to which lb is applied.
      ! ub: double precision (:)
      !   Upper boundaries on a subset of fluxes x in Sx = 0.
      ! i_ub: integer (:)
      !   The columns of K to which ub is applied.
      !
      ! Returns
      ! -------
      ! G: double precision (:, :)
      !   The transformation from the basis vector q to meet the
      !   constraints on x such that Gq >= h.
      ! h: double precision (:)
      !   Combination of lb and ub into a single lower boundary,
      !   redefined as the deviation from initial position.
      ! q: double precision (:)
      !   The initial sampling position following boundary
      !   redefinition.

      ! Subroutine variables
      integer, intent(in) :: i_lb(:), i_ub(:)
      double precision, intent(in) :: K(:, :), xo(:), lb(:), ub(:)
      double precision, intent(out) :: G(:,:), h(:), q(:)

      ! Local variables
      integer :: n_lb, n_ub, n_constraint

      ! Calculating dimensions
      n_lb = size(lb)
      n_ub = size(ub)

      n_constraint = n_lb + n_ub

      !
      ! Combining the constraints as lower bounds: 
      ! x >= lb, -x >= -ub
      !

      ! Combining lb and ub into h and re-defining the constraints as
      ! deviations from the starting point
      h(1:n_lb) = lb - xo(i_lb)
      h((n_lb + 1):(n_constraint)) = -(ub - xo(i_ub))

      ! Starting point shifted to zero
      q = 0

      ! Defining G as the transformation from the basis vector q to meet
      ! the constraints on x such that Gq >= h
      G(1:n_lb, :) = K(i_lb, :)
      G((n_lb + 1):(n_constraint), :) = -K(i_ub, :)

    end subroutine convert_constraints

    !===================================================================>
    ! Random direction algorithm

    !-------------------------------------------------------------------
    function rda_sample(K, m, n, xo, lb, i_lb, n_lb, ub, i_ub, n_ub,          &
                        n_out, param, seed)
      ! Samples from the stoichiometry matrix, S, constrained by upper
      ! and lower bounds, ub and lb, using the random direction
      ! algorithm 
      !
      ! Parameters
      ! ----------
      ! K: double precision (m, n)
      !   Nullspace of the stoichiometry matrix (S) in the form of Sx = 0.
      ! m: integer
      !   Row number of K.
      ! n: integer
      !   Column number of K.
      ! xo: double precision(n)
      !   Starting point for the algorithm that satisfies: 
      !   Sx = 0, x(i_ub) <= ub, x(i_lb) >= lb.
      ! lb: double precision (n_lb)
      !   Lower boundaries on a subset of fluxes x in Sx = 0.
      ! i_lb: integer (n_lb)
      !   The columns of S (rows of K) to which lb is applied.
      ! n_lb: integer
      !   Total number of lower constraints.
      ! ub: double precision (n_ub)
      !   Upper boundaries on a subset of fluxes x in Sx = 0.
      ! i_ub: integer (n_ub)
      !   The columns of S (rows of K) to which ub is applied.
      ! n_ub: integer
      !   Total number of upper constraints.
      ! n_out: integer
      !   Number of samples to return.
      ! param: integer(3)
      !   An array of basic parameters that govern sample generation.
      !
      !     n_burnin: Number of burn-in steps to take before starting sampling.
      !     n_iter: Number of steps to take for sampling.
      !     n_max: The maximum number burn-in and sampling steps to take,
      !            including failed steps.
      !
      ! seed: integer
      !   Seed for the random number generator used for selecting
      !   direction.
      !
      ! Returns
      ! -------
      ! rda_sample: double precision (m, n_out)
      !   A random sample of x values that satisfy Sx = 0 and are
      !   constrained by lb and ub.
      !
      !f2py depend(m) K, rda_sample, xo
      !f2py depend(n) K

      ! Function variables
      integer, intent(in) :: m, n, n_lb, n_ub, n_out, param(3), seed 
      integer, intent(in) :: i_lb(n_lb), i_ub(n_ub)
      double precision, intent(in) :: K(m, n), xo(m), lb(n_lb), ub(n_ub)
      double precision :: rda_sample(m, n_out)

      ! Local variables
      integer :: n_burnin, n_iter, n_max
      integer :: i, j, total_steps, n_constraint
      integer :: i_out(n_out)
      double precision :: G(n_lb + n_ub, n)
      double precision :: h(n_lb + n_ub), q(n), step(n)

      ! Setting seed
      call r_init(seed)

      ! Unpacking parameters
      n_burnin = param(1)
      n_iter = param(2)
      n_max = param(3)

      ! Calculating some dimensions
      n_constraint = n_lb + n_ub

      ! Transforming constraints to satisfy algorithm requirements
      call convert_constraints(K, xo, lb, i_lb, ub, i_ub, G, h, q)

      ! Initializing output to zeros
      rda_sample = 0

      ! Initializing total steps
      total_steps = 0

      ! Burn-in steps
      do i = 1, n_burnin
        
        step = 0

        ! Only counting non-zero steps
        do while (sum(abs(step)) == 0) 
          step = rda_step(q, G, h, n_constraint, n) 
          total_steps = total_steps + 1

          if (total_steps > n_max) then
            return
          end if
        end do

        q = q + step

      end do

      ! Determining which steps to sample ahead of time
      i_out = (/(i, i=1, n_iter - mod(n_iter, n_out), n_iter/n_out)/)  

      ! Offsetting untouched iterations to the beginning to act like
      ! extra burnin
      i_out = i_out + mod(n_iter, n_out) + n_iter/n_out - 1

      ! Starting sampling index
      j = 1

      ! Real steps
      do i = 1, n_iter

        step = 0

        ! Only counting non-zero steps
        do while (sum(abs(step)) == 0) 
          step = rda_step(q, G, h, n_constraint, n)
          total_steps = total_steps + 1

          if (total_steps > n_max) then
            return
          end if
        end do

        q = q + step

        if (i == i_out(j)) then
          rda_sample(:, j) = matmul(K, q) + xo

          j = j + 1

        end if

      end do

    end function rda_sample

    !-------------------------------------------------------------------
    function rda_step(q, G, h, m, n)
      ! Takes a single random direction step defined by the constraints:
      ! Gq >= h
      !
      ! Parameters
      ! ----------
      ! q: double precision (n)
      !   Current position.
      ! G: double precision (m, n)
      !   
      ! h: double precision (m)
      !
      ! m: integer
      !
      ! n: integer
      !
      ! Returns
      ! -------
      ! rda_step: double precision (n)
      !   The direction and magnitude of new step which results in a new
      !   position when added to the current step.

      ! Function variables
      integer, intent(in) :: m, n
      double precision, intent(in) :: G(m, n), h(m), q(n)
      double precision :: rda_step(n)

      ! Local variables
      integer :: i
      double precision :: q_low(n), q_high(n), d(n), infinity
      double precision :: alpha(m), alpha_low, alpha_high, alpha_final, step(m)

      ! Generating random direction
      d = r_normal_v(0d0, 1d0, n)
      d = d/v_norm(d)

      infinity = huge(0d0)

      ! Assuming that no feasible step size exists
      alpha_low = -infinity
      alpha_high = infinity

      ! Updating feasible step sizes
      alpha = (h - matmul(G, q)) / matmul(G, d)

      do i = 1, m
        if (alpha(i) < 0) then
          if (alpha(i) > alpha_low) then
            alpha_low = alpha(i)
          end if
        else if (alpha(i) > 0) then
          if (alpha(i) < alpha_high) then
            alpha_high = alpha(i)
          end if
        end if
      end do

      q_low = q + alpha_low * d
      q_high = q + alpha_high * d

      step = matmul(G, q_low)

      do i = 1, m
        if (step(i) < h(i)) then
          alpha_low = 0
          exit
        end if
      end do

      step = matmul(G, q_high)

      do i = 1, m
        if (step(i) < h(i)) then
          alpha_high = 0
          exit
        end if
      end do

      ! Generating step magnitude from between alpha_low and alpha_high
      if (alpha_high == alpha_low) then
        rda_step = 0
      else
        alpha_final = r_uniform(alpha_low, alpha_high)
        rda_step = alpha_final * d
      end if

    end function rda_step

    !===================================================================>
    ! Mirror algorithm

    !-------------------------------------------------------------------
    function ma_sample(K, m, n, xo, lb, i_lb, n_lb, ub, i_ub, n_ub, sd,       &
                       n_out, param, seed)
      ! Samples from the stoichiometry matrix, S, constrained by upper
      ! and lower bounds, ub and lb, using the mirror algorithm 
      !
      ! Parameters
      ! ----------
      ! K: double precision (m, n)
      !   Nullspace of the stoichiometry matrix (S) in the form of Sx = 0.
      ! m: integer
      !   Row number of K.
      ! n: integer
      !   Column number of K.
      ! xo: double precision(n)
      !   Starting point for the algorithm that satisfies: 
      !   Sx = 0, x(i_ub) <= ub, x(i_lb) >= lb.
      ! lb: double precision (n_lb)
      !   Lower boundaries on a subset of fluxes x in Sx = 0.
      ! i_lb: integer (n_lb)
      !   The columns of S (rows of K) to which lb is applied.
      ! n_lb: integer
      !   Total number of lower constraints.
      ! ub: double precision (n_ub)
      !   Upper boundaries on a subset of fluxes x in Sx = 0.
      ! i_ub: integer (n_ub)
      !   The columns of S (rows of K) to which ub is applied.
      ! n_ub: integer
      !   Total number of upper constraints.
      ! sd: double precision(n)
      !   Standard deviation vector for determining step direction and 
      !   magnitude.
      ! n_out: integer
      !   Number of samples to return.
      ! param: integer(2)
      !   An array of basic parameters that govern sample generation.
      !
      !     n_burnin: Number of burn-in steps to take before starting 
      !               sampling.
      !     n_iter:   Number of steps to take for sampling.
      !
      ! seed: integer
      !   Seed for the random number generator used for selecting
      !   direction.
      !
      ! Returns
      ! -------
      ! ma_sample: double precision (m, n_out)
      !   A random sample of x values that satisfy Sx = 0 and are
      !   constrained by lb and ub.
      !
      !f2py depend(m) K, ma_sample, xo
      !f2py depend(n) K, sd

      ! Function variables
      integer, intent(in) :: m, n, n_lb, n_ub, n_out, seed 
      integer, intent(in) :: param(2), i_lb(n_lb), i_ub(n_ub)
      double precision, intent(in) :: K(m, n), xo(m), lb(n_lb), ub(n_ub), sd(n)
      double precision :: ma_sample(m, n_out)

      ! Local variables
      integer :: n_burnin, n_iter
      integer :: i, j, n_constraint
      integer :: i_out(n_out)
      double precision :: G(n_lb + n_ub, n)
      double precision :: h(n_lb + n_ub), q(n)

      ! Setting seed
      call r_init(seed)

      ! Unpacking parameters
      n_burnin = param(1)
      n_iter = param(2)

      ! Calculating some dimensions
      n_constraint = n_lb + n_ub

      ! Transforming constraints to satisfy algorithm requirements
      call convert_constraints(K, xo, lb, i_lb, ub, i_ub, G, h, q)

      ! Initializing output to zeros
      ma_sample = 0

      ! Burn-in steps
      do i = 1, n_burnin
        q = q + ma_step(q, G, h, n_constraint, n, sd) 
      end do

      ! Determining which steps to sample ahead of time
      i_out = (/(i, i=1, n_iter - mod(n_iter, n_out), n_iter/n_out)/)  

      ! Offsetting untouched iterations to the beginning to act like
      ! extra burnin
      i_out = i_out + mod(n_iter, n_out) + n_iter/n_out - 1
      j = 1

      ! Real steps
      do i = 1, n_iter
        q = q + ma_step(q, G, h, n_constraint, n, sd)

        if (i == i_out(j)) then
          ma_sample(:, j) = matmul(K, q) + xo

          j = j + 1
        end if

      end do

    end function ma_sample

    !-------------------------------------------------------------------
    function ma_step(q, G, h, m, n, sd)
      ! Takes a single random direction step defined, reflecting off
      ! the constraints:
      ! Gq >= h
      !
      ! Parameters
      ! ----------
      ! q: double precision (n)
      !   Current position.
      ! G: double precision (m, n)
      !   
      ! h: double precision (m)
      !
      ! m: integer
      !
      ! n: integer
      !
      ! sd: double precision
      !   Standard deviations for each dimension used to generate step
      !   direction.
      !
      ! Returns
      ! -------
      ! rda_step: double precision (n)
      !   The direction and magnitude of new step which results in a new
      !   position when added to the current step.

      ! Function variables
      integer, intent(in) :: m, n
      double precision, intent(in) :: G(m, n), h(m), q(n), sd(n)
      double precision :: ma_step(n)

      ! Local variables
      logical :: mask(m)
      integer :: i_alpha(1)
      double precision :: s
      double precision :: d(n), step(n), origin(n), ray(n)
      double precision :: residual(m), alpha(m)

      ! Generating normally distributed deviations
      d = r_normal_v(0d0, 1d0, n)*sd

      step = q + d

      ! Initializing origin as the current position
      origin = q

      ! Checking if new step fits criteria
      residual = matmul(G, step) - h

      do while (any(residual < 0))
        
        ray = step - origin

        ! Indexes of blocked directions
        mask = residual < 0

        ! Point of contact with constraints
        alpha = (h - matmul(G, origin)) / matmul(G, ray)

        ! Closest point of contact
        i_alpha = minloc(alpha, 1, mask)

        ! Calculating reflection from constraint 
        s = -residual(i_alpha(1)) / v_ssq(G(i_alpha(1), :))

        step = step + 2*s*G(i_alpha(1), :)

        ! Updating origin as last point of reflection
        origin = origin + alpha(i_alpha(1))*ray

        ! Re-calculating residual
        residual = matmul(G, step) - h

      end do

      ! Converting final position into step to match the syntax
      ! of rda_step function
      ma_step = step - q

    end function ma_step

end module realization
