"""
Collection of functions for exploring constrained flux spaces.
"""

from cvxopt import solvers, blas, matrix, spdiag, log as xlog
from cvxopt.modeling import variable, op, sum as xsum
import numpy as np
import pandas as pd

# Local imports
import omfapy as omfa

#--------------------------------------------------------------------------
def chebyshev_center(G, h, progress=False):
    """
    Calculates the center point of the largest sphere that can fit within
    constraints specified by Gx <= h.

    Parameters
    ----------
    G: pandas.DataFrame
        Array that specifies Gx <= h.
    h: pandas.Series
        The limits specifying Gx <= h.
    progress: bool
        True if detailed progress text from optimization should be shown.

    Returns
    -------
    x: pd.Series
        Centroid with index names equal to the column names of G.
    """
    
    # Aligning input
    h = h.ix[G.index]

    if h.isnull().values.any() or G.isnull().values.any():
        msg = 'Row indeces of G and h must match and contain no NaN entries.'
        omfa.logger.error(msg)
        raise ValueError(msg)

    if progress:
        solvers.options['show_progress'] = True
    else:
        solvers.options['show_progress'] = False

    # Setting up LP problem
    m, n = G.shape

    R = variable()
    x = variable(n)

    G_opt = matrix(np.array(G, dtype=np.float64))
    h_opt = matrix(np.array(h, dtype=np.float64))

    inequality_constraints = [G_opt[k,:]*x + R*blas.nrm2(G_opt[k,:]) <= h_opt[k] 
                              for k in range(m)]

    model = op(-R, inequality_constraints + [ R >= 0] )
    model.solve()

    x = pd.Series(x.value, index=G.columns)

    # Checking output
    if model.status != 'optimal':
        if all(G.dot(x) <= h):
            msg = ('Centroid was not found, '
                   'but the last calculated point is feasible.')
            omfa.logger.warn(msg)
        else:
            msg = 'Optimization calculatoin failed on a non-feasible point.'
            sol = '\nSet progress=True for more details.'
            omfa.logger.error(msg + sol)
            raise RuntimeError(msg + sol)

    return(x)

#------------------------------------------------------------------------
def analytic_center(G, h, progress=False):
    """
    Finds a point within constraints specified by Gx <= h that is furthest
    away from the constraint by maximizing prod(h - Gx).

    Parameters
    ----------
    G: pandas.DataFrame
        Array that specifies Gx <= h.
    h: pandas.Series
        The limits specifying Gx <= h.
    progress: bool
        True if detailed progress text from optimization should be shown.

    Returns
    -------
    x: pd.Series
        Centroid with index names equal to the column names of G.
    """

    # Aligning input
    h = h.ix[G.index]

    if h.isnull().values.any() or G.isnull().values.any():
        msg = 'Row indeces of G and h must match and contain no NaN entries.'
        omfa.logger.error(msg)
        raise ValueError(msg)

    if progress:
        solvers.options['show_progress'] = True
    else:
        solvers.options['show_progress'] = False

    # Initial point necessary for optimization
    if progress:
        omfa.logger.debug('Solving for feasible starting point')

    start = feasible_point(G, h, progress)

    # Setting up LP problem
    def F(x=None, z=None):
        if x is None:  
            return 0, matrix(start)
       
        y = h_opt - G_opt*x
        if min(y) <= 0.0: 
            return None

        f = -xsum(xlog(y))
        Df = (y**-1).T * G_opt

        if z is None: 
            return matrix(f), Df

        H = z[0] * G_opt.T * spdiag(y**-2) * G_opt
        return matrix(f), Df, H

    if progress:
        omfa.logger.debug('Calculating analytic center')

    sol = solvers.cp(F, G_opt, h_opt)

    x = pd.Series(sol['x'], index=G.columns)

    # Checking output
    if sol['status'] != 'optimal':
        if all(G.dot(x) <= h):
            msg = ('Centroid was not found, '
                   'but the last calculated point is feasible')
            omfa.logger.warn(msg)
        else:
            msg = 'Optimization failed on a non-feasible point'
            omfa.logger.error(msg)
            raise omfa.ModelError(msg)

    return(x)

#------------------------------------------------------------------------
def feasible_point(G, h, progress=False):
    """
    Finds a single point within constraints specified by Gx <= h.

    Parameters
    ----------
    G: pandas.DataFrame
        Array that specifies Gx <= h.
    h: pandas.Series
        The limits specifying Gx <= h.
    progress: bool
        True if detailed progress text from optimization should be shown.

    Returns
    -------
    x: pd.Series
        Point with index names equal to the column names of G.
    """

    # Aligning input
    h = h.ix[G.index]

    if h.isnull().values.any() or G.isnull().values.any():
        msg = 'Row indeces of G and h must match and contain no NaN entries.'
        omfa.logger.error(msg)
        raise ValueError(msg)

    if progress:
        solvers.options['show_progress'] = True
    else:
        solvers.options['show_progress'] = False

    # Setting up LP problem
    m, n = G.shape

    s = variable()
    x = variable(n)

    G_opt = matrix(np.array(G, dtype=np.float64))
    h_opt = matrix(np.array(h, dtype=np.float64))

    inequality_constraints = [G_opt[k,:]*x + s <= h_opt[k] for k in range(m)]

    # Run LP to find feasible point
    model = op(-s, inequality_constraints)
    model.solve()

    if s.value[0] <= 0:
        msg = 'Could not find feasible starting point to calculate centroid.'
        omfa.logger.error(msg)
        raise omfa.ModelError(msg)

    out = pd.Series(x.value, index=G.columns)
    return(out)

#------------------------------------------------------------------------
def check_feasibility(G, h, progress=False, silent=False):
    """
    Determines if specified constraints of the form Gx <= h are feasible. 

    Parameters
    ----------
    G: pandas.DataFrame
        Array that specifies Gx <= h.
    h: pandas.Series
        The limits specifying Gx <= h.
    progress: bool
        True if detailed progress text from optimization should be shown.
    silent: bool
        True if diagnostic messages should bre printed (through logger.warn).

    Returns
    -------
    check_passed: bool 
        True if constraints are feasible, False if they aren't.
    """

    # Aligning input
    h = h.ix[G.index]

    if h.isnull().values.any() or G.isnull().values.any():
        msg = 'Row indeces of G and h must match and contain no NaN entries.'
        omfa.logger.error(msg)
        raise ValueError(msg)

    if progress:
        solvers.options['show_progress'] = True
    else:
        solvers.options['show_progress'] = False

    # Setting up LP problem
    m, n = G.shape

    x = variable(n)

    G_opt = matrix(np.array(G, dtype=np.float64))
    h_opt = matrix(np.array(h, dtype=np.float64))

    inequality_constraints = [G_opt[k,:]*x <= h_opt[k] for k in range(m)]

    # Arbitrary, simple minimization problem
    model = op(xsum(abs(x)), inequality_constraints)
    model.solve()

    if 'primal infeasible' in model.status:
        msg = 'The problem is infeasible.'
        omfa.logger.warn(msg)
        check_passed = False
    elif 'dual infeasible' in model.status:
        msg = 'The problem is unbounded.'
        omfa.logger.warn(msg)
        check_passed = False
    elif 'unknown' in model.status:
        msg = 'An unknown optimization error occured.'
        sol = ('\nSome constraints may be near parallel or the problem '
               'needs scaling.')
        omfa.logger.warn(msg + sol)
        check_passed = False
    else:
        check_passed = True

    return(check_passed)

#------------------------------------------------------------------------
def variable_range(G, h, A=None, b=None, progress=False):
    """
    Determines the net upper and lower constraints on x given that Gx <= h
    and Ax == b.
    
    Parameters
    ----------
    G: pandas.DataFrame
        Array that specifies Gx <= h.
    h: pandas.Series
        The limits specifying Gx <= h.
    A: pandas.DataFrame
        Array that specifies Ax == b.
    b: pandas.Series
        The limits specifying Ax == b.
    progress: bool
        True if detailed progress text from optimization should be shown.

    Returns
    -------
    ranges: pd.DataFrame
        A dataframe with columns indicating the lowest and highest
        values each basis variable can take.
    """
    # Aligning input
    h = h.ix[G.index]

    if h.isnull().values.any() or G.isnull().values.any():
        msg = 'Row indeces of G and h must match and contain no NaN entries.'
        omfa.logger.error(msg)
        raise ValueError(msg)

    if sum([A is None, b is None]) == 1:
        msg = 'If one of A or b is specified, the other one must be too'
        omfa.logger.error(msg)
        raise ValueError(msg)

    if A is not None: 
        b = b.ix[A.index]

        if set(A.columns) != set(G.columns):
            msg = 'A and G must have the same column names'
            omfa.logger.error(msg)
            raise ValueError(msg)

    if progress:
        solvers.options['show_progress'] = True
    else:
        solvers.options['show_progress'] = False

    # Setting up LP problem
    m_G, n_G = G.shape

    x = variable(n_G)

    G_opt = matrix(np.array(G, dtype=np.float64))
    h_opt = matrix(np.array(h, dtype=np.float64))

    constraints = [G_opt[k,:]*x <= h_opt[k] for k in range(m_G)]

    if A is not None:
        m_A, n_A = A.shape

        A_opt = matrix(np.array(A, dtype=np.float64))
        b_opt = matrix(np.array(b, dtype=np.float64))

        constraints = constraints + [A_opt[k,:]*x == b_opt[k] 
                                     for k in range(m_A)]

    # Initializing output
    ranges = pd.DataFrame(None, index=G.columns, columns=['low', 'high'])

    # Looping through the individual objectives
    for basis in range(n_G):

        # Minimum
        model = op(x[basis], constraints)
        model.solve()
        status = model.status

        if 'optimal' in model.status:
            ranges.ix[basis, 'low'] = x.value[basis]
        elif 'dual infeasible' in model.status:
            ranges.ix[basis, 'low'] = -float('inf')
        elif 'primal infeasible' in model.status:
            msg = 'Specified constraints are infeasible.'
            omfa.logger.error(msg)
            raise ValueError(msg)
        elif 'unknown' in model.status:
            msg = 'Minimum not found due to unknown optimization error.'
            omfa.logger.warn(msg)
            ranges.ix[basis, 'low'] = None
            
        # Maximum
        model = op(-x[basis], constraints)
        model.solve()
        status = model.status

        if 'optimal' in model.status:
            ranges.ix[basis, 'high'] = x.value[basis]
        elif 'dual infeasible' in model.status:
            ranges.ix[basis, 'high'] = -float('inf')
        elif 'primal infeasible' in model.status:
            msg = 'Specified constraints are infeasible.'
            omfa.logger.error(msg)
            raise ValueError(msg)
        elif 'unknown' in model.status:
            msg = 'Minimum not found due to unknown optimization error.'
            omfa.logger.warn(msg)
            ranges.ix[basis, 'low'] = None

    return(ranges)


