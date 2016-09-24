! Library containing a number ad hoc routines relating to random numbers.
!
! Description
! -----------
!
! The majority of the code is a direct adaptation/expansion of the
! "randgen.f" library written by Richard Chandler and Paul Northrop.
! The primary purpose of this file is to repackage a number of 
! subprograms as a module for ease in extending the OMFA python
! pacakge. Deviations from "randgen.f" are generally stylistic with
! a couple of extra subprograms added for convenience.
!
! As with the "randgen.f" library, all subprograms rely on the
! generation of uniformly distributed numbers on the interval (0,1)
! using a Marsaglia-Zaman subtract-with-borrow algorithm that has been
! modified by Chandler and Northorp.
!
! Author
! ------
!
! Stanislav Sokolenko, University of Waterloo
!
! History
! -------
!
! June 22/2014 
!   - "zbqlini" implemented as "r_init"
!   - "zbqlu01" and "zbqluab" implemented as "r_uniform"
!   - "zbqlnor" implemented as "r_normal"
!   - added vector version of "r_uniform" as "r_uniform_v"
!   - added vector version of "r_normal" as "r_normal_v"

module random

  implicit none
  private

  ! Values needed for psuedo-random number generation
  double precision, save :: seed_vector(43) =                                 &
    [8.001441D7,   5.5321801D8,  1.69570999D8, 2.88589940D8,                  &
     2.91581871D8, 1.03842493D8, 7.9952507D7,  3.81202335D8,                  &
     3.11575334D8, 4.02878631D8, 2.49757109D8, 1.15192595D8,                  &
     2.10629619D8, 3.99952890D8, 4.12280521D8, 1.33873288D8,                  &
     7.1345525D7,  2.23467704D8, 2.82934796D8, 9.9756750D7,                   &
     1.68564303D8, 2.86817366D8, 1.14310713D8, 3.47045253D8,                  &
     9.3762426D7,  1.09670477D8, 3.20029657D8, 3.26369301D8,                  &
     9.441177D6,   3.53244738D8, 2.44771580D8, 1.59804337D8,                  &
     2.07319904D8, 3.37342907D8, 3.75423178D8, 7.0893571D7,                   &
     4.26059785D8, 3.95854390D8, 2.0081010D7,  5.9250059D7,                   &
     1.62176640D8, 3.20429173D8, 2.63576576D8]
  double precision, save :: b = 4.294967291D9, c = 0.0D0 
  integer, save :: index_1 = 22, index_2 = 43

  ! Normal distribution parameters
  double precision, parameter :: pi = 3.14159265358979323846D0
  double precision, save :: normal_spare
  integer, save :: normal_stored = 0

  public :: r_init, r_uniform, r_normal, r_uniform_v, r_normal_v

    contains

      !===================================================================>
      ! Routines

      !-------------------------------------------------------------------
      subroutine r_init(seed)
        ! Initializes seed for random number generation.
        !
        ! Parameters
        ! ----------
        ! seed: integer
        !   Number that results in a unique stream of pseudo-random 
        !   values.

        ! Subroutine variables
        integer, intent(in) :: seed

        ! Local variabes
        double precision :: temp
        integer :: i

        ! Module variables
        ! double precision :: seed_vector(43), b, c
        ! integer :: index_1, index_2

        temp = dmod(dble(seed), b)
        seed_vector(1) = temp

        do i = 2, 43
          temp = seed_vector(i-1)*3.0269D4
          temp = dmod(temp, b)
          seed_vector(i) = temp
        end do
        
        b = 4.294967291D9
        c = 0.0D0

        index_1 = 22
        index_2 = 43

        ! Dumping stored normal variables
        normal_stored = 0

      end subroutine r_init

      !-------------------------------------------------------------------
      function r_uniform(lb, ub)
        ! Generates value between lb and ub.
        !
        ! Parameters
        ! ----------
        ! lb: double precision
        !   Lower boundary.
        ! ub: double precision
        !   Upper boundary.
        !
        ! Returns
        ! ----------
        ! r_uniform: double precision 
        !   Number on the interval (0, 1).
        !
        ! Details
        ! -------
        ! The function implements a Marsaglia-Zaman subtract-with-borrow 
        ! algorithm that has been modified by Chandler and Northorp to
        ! use double precision numbers.

        ! Function variables
        double precision, intent(in) :: lb, ub
        double precision :: r_uniform

        ! Local variables
        double precision :: b_2, x

        ! Module variables
        ! double precision :: seed_vector(43), b, c
        ! integer :: index_1, index_2

        x = 0
        b_2 = 1

        do while (x < 1.0D0/b)
          
          b_2 = b_2 * b
          x = seed_vector(index_1) - seed_vector(index_2) - c

          if (x < 0.0D0) then
            x = x + b
            c = 1.0D0
          else
            c = 0.0D0
          end if
          
          seed_vector(index_2) = x

          index_1 = index_1 - 1
          index_2 = index_2 - 1

          if (index_1 == 0) then
            index_1 = 43
          else if (index_2 == 0) then
            index_2 = 43
          end if

        end do

        r_uniform = lb + (ub-lb)*(x/b_2) 

      end function r_uniform

      !-------------------------------------------------------------------
      function r_normal(mu, sigma)
        ! Generates value from a normal distribution with mean mu and
        ! standard deviation sigma
        !
        ! Parameters
        ! ----------
        ! mu: double precision
        !   Distribution mean.
        ! sigma: double precision
        !   Distribution standard deviation.
        !
        ! Returns
        ! ----------
        ! r_normal: double precision 
        !   Number from a normal distribution. 
        !
        ! Details
        ! -------
        ! The function implements the Box-Muller transform to generate
        ! pairs of random normal deviates from random uniform ones. The
        ! extra deviate is stored as a module variable until use.
        !
        ! Box, G. E. P. & Muller, M. E. (1958). A Note on the generation
        ! of random normal deviates. Annals of Mathematical Statistics,
        ! vol. 29, no. 2, pp. 610–611

        ! Function variables
        double precision, intent(in) :: mu, sigma
        double precision :: r_normal

        ! Local variables
        double precision :: r, theta

        ! Module variables
        ! double precision :: pi, normal_spare
        ! integer :: normal_stored

        if (normal_stored == 0) then
          theta = 2.0D0*pi*r_uniform(0.0D0, 1.0D0)
          r = dsqrt( -2.0D0*dlog(r_uniform(0.0D0, 1.0D0)) )
          
          r_normal = r*dcos(theta)
          normal_spare = r*dsin(theta)

          normal_stored = 1
        else
          r_normal = normal_spare

          normal_stored = 0
        end if

        r_normal = mu + sigma * r_normal

      end function r_normal

      !-------------------------------------------------------------------
      function r_uniform_v(lb, ub, n)
        ! Generates n values between lb and ub.
        !
        ! Parameters
        ! ----------
        ! lb: double precision
        !   Lower boundary.
        ! ub: double precision
        !   Upper boundary.
        ! n: integer
        !   Number of values to generate.
        !
        ! Returns
        ! ----------
        ! r_uniform_v: double precision(n) 
        !   n numbers on the interval (0, 1).
        !
        ! Details
        ! -------
        ! Convenience function to loop over r_uniform

        ! Function variables
        double precision, intent(in) :: lb, ub
        integer, intent(in) :: n
        double precision :: r_uniform_v(n)

        ! Local variables
        integer :: i

        do i = 1, n
          r_uniform_v(i) = r_uniform(lb, ub)
        end do

      end function r_uniform_v

      !-------------------------------------------------------------------
      function r_normal_v(mu, sigma, n)
        ! Generates n values from a normal distribution with mean mu
        ! and standard deviation sigma
        !
        ! Parameters
        ! ----------
        ! mu: double precision
        !   Distribution mean.
        ! sigma: double precision
        !   Distribution standard deviation.
        ! n: integer
        !   Number of values to generate.
        !
        ! Returns
        ! ----------
        ! r_normal: double precision(n) 
        !   n numbers from a normal distribution. 
        !
        ! Details
        ! -------
        ! Using the same algorithm as in r_normal,
        ! this function takes advantage of the fact that numbers are 
        ! generated in pairs to fill up the vector two at a time.  

        ! Function variables
        double precision, intent(in) :: mu, sigma
        integer, intent(in) :: n
        double precision :: r_normal_v(n)

        ! Local variables
        double precision :: r, theta
        integer :: i, n_stop

        ! Module variables
        ! double precision :: pi, normal_spare
        ! integer :: normal_stored

        if (mod(n, 2) == 1) then
          n_stop = n - 1
        else
          n_stop = n
        end if

        do i = 1, n_stop, 2
          theta = 2.0D0*pi*r_uniform(0.0D0, 1.0D0)
          r = dsqrt( -2.0D0*dlog(r_uniform(0.0D0, 1.0D0)) )

          r_normal_v(i) = r*dcos(theta)
          r_normal_v(i+1) = r*dsin(theta)
        end do

        if (mod(n, 2) == 1) then
          r_normal_v(n) = r_normal(0.0D0, 1.0D0)
        end if

        r_normal_v = mu + sigma * r_normal_v

      end function r_normal_v

end module random

