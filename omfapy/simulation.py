"""
A collection of functions that simulate model fluxes or observations.
"""

import numpy as np
import pandas as pd

# Local imports
import omfapy as omfa

#=========================================================================>
# Flux realizations from stoichiometric model.

#-------------------------------------------------------------------------
def generate_sample(model, lower, upper, method='rda', 
                    fortran=False, seed=None, 
                    **param):
    """
    Generates a random sample from the model solution space defined by
    lower and upper constraints.

    Parameters
    ----------
    model: omfa.Model
    lower: pandas.Series
        A set of lower limits on fluxes found in the model.
    upper: pandas.Series
        A set of upper limits on fluxes found in the model.
    method: str
        One of either 'rda' for random direction algorithm or 'ma' for
        mirror algorithm.
    fortran: bool
        Whether to use a python or fortran implementation. The fortran
        implementation is much faster, but requires the successful
        compilation of the provided fortran source code.
    seed: hashable 
        By default, a random number generator is initialized on package
        import. A new generator can be initialized with the provided
        seed to be used for this operation alone.
    **param:
        Parameters that depend on the algorithm.

        Generic:

        xo: pandas.Series
            A set of flux values that meets constraint restrictions
            to use as a sampling starting point.
        n_burnin: int
            The number of steps to take before sampling.
        n_iter: int
            The number of steps to take for drawing samples.
        n_out: int
            The number of steps to return from those taken.

        method='rda':

        n_max: int
            The maximum total number of steps that can be missed
            due to a direction being picked with no available steps.

        method='ma':
        
        sd: pandas.Series
            Standard deviations to be used for choosing a random
            direction. The standard deviation values correspond
            to basis variables rather than fluxes.
        
    Returns
    -------
    sample: pd.DataFrame
        A data frame where each column corresponds to a random sample.
    """

    stoich = model.stoichiometry

    # Checking input
    valid_param = ['n_burnin', 'n_iter', 'n_out', 'xo']

    if method == 'rda':
        valid_param.append('n_max')
    elif method == 'ma':
        valid_param.append('sd')
    else:
        msg = 'Random sampling method must be one of "rda" or "ma"'
        raise ValueError(msg)

    for p in param:
        if p not in valid_param:
            msg = '"{0}" is not a valid parameter'.format(p)
            raise ValueError(msg)

    # Default parameter values
    if 'n_burnin' not in param:
        param['n_burnin'] = 1000

    if 'n_iter' not in param:
        param['n_iter'] = 1000

    if 'n_out' in param:
        if param['n_out'] > param['n_iter']:
            msg = '"n_out" must be less than "n_iter"'
            raise ValueError
    else:
        param['n_out'] = param['n_iter']/2

    if 'xo' in param:
        if not set(stoich.columns).issubset(param['xo'].index):
            msg = 'A value must be provided for all fluxes'
            raise ValueError(msg)

        lower_fail = not all(param['xo'][lower.index] >= lower)
        upper_fail = not all(param['xo'][upper.index] <= upper)
        bounds_fail = sum(abs(stoich.dot(param['xo']))) > 1e-6

        if any([lower_fail, upper_fail, bounds_fail]):
            msg = 'Starting value must meet specified constraints'
            raise ValueError(msg)

    else:
        param['xo'] = model.generate_flux_centroid(lower, upper)

    if method == 'rda':
        if fortran:
            sample = _rda_sample_f(model, lower, upper, seed, **param)
        else:
            sample = _rda_sample(model, lower, upper, seed, **param)
    elif method == 'ma':
        if fortran:
            sample = _ma_sample_f(model, lower, upper, seed, **param)
        else:
            sample = _ma_sample(model, lower, upper, seed, **param)

    return(sample)

#-------------------------------------------------------------------------
def _rda_sample(model, lower, upper, seed=None, **param):
    """
    Generates a random sample from the model solution space defined by
    lower and upper constraints using a random direction algorithm.

    see generate_random_sample()...
    """

    kernel = model.nullspace

    # Unpacking parameters
    n_burnin = param['n_burnin']
    n_iter = param['n_iter']
    n_out = param['n_out']

    xo = param['xo']

    # Checking rda-specific input
    if 'n_max' in param:
        n_max = param['n_max']
        if n_max <= (n_burnin + n_iter):
            msg = '"n_max" must be greater than "n_burnin" + "n_iter"'
            raise ValueError(msg)
    else:
        n_max = (n_burnin + n_iter)*2

    # Setting random number generator
    if seed is None:
        prng = omfa.prng
    else:
        prng = np.random.RandomState(seed)

    # Recalculating inequality constraints as deviation from central point
    new_lower = lower - xo[lower.index]
    new_upper = upper - xo[upper.index]

    # Round small values down to zero
    new_lower[abs(new_lower) < 1e-8] = 0
    new_upper[abs(new_upper) < 1e-8] = 0

    G, h = model.generate_basis_constraints(new_lower, new_upper)

    G = np.array(G)
    h = np.array(h)

    m, n = G.shape

    # Re-defining starting location as zero
    current_step = np.zeros(n)

    # Defining the random step function
    def rd_step(current_step, G, h, n):

        if any(G.dot(current_step) < h):
            msg = 'Initial step out of bounds'
            raise Exception(msg)

        # Generate random direction
        d = prng.normal(size=n)
        d = d/np.linalg.norm(d)
       
        # Calculating feasible step size
        alpha = (h - G.dot(current_step)) / G.dot(d)
       
        try:
            alpha_low = np.max(alpha[alpha < 0])
        except ValueError:
            alpha_low = 0

        try:
            alpha_high = np.min(alpha[alpha > 0])
        except ValueError:
            alpha_high = 0

        step_low = current_step + alpha_low * d
        step_high = current_step + alpha_high * d

        if any(G.dot(step_low) < h): 
            alpha_low = 0
        if any(G.dot(step_high) < h):
            alpha_high = 0

        # If no stepping options, return None
        if alpha_high == alpha_low:
            return(None)
            
        new_step = current_step + prng.uniform(alpha_low, alpha_high) * d

        return(new_step)

    # Burnin
    n_failed = 0
    for i in range(n_burnin):
        attempted_step = rd_step(current_step, G, h, n)

        if attempted_step is None:
            n_failed += 1
            if n_failed == n_max:
                msg = 'Maximum failed threshold reached'
                raise omfa.ModelError(msg)
        else:
            current_step = attempted_step

    # Initializing output
    sample = pd.DataFrame(index=kernel.index, 
                          columns=['S{0}'.format(i) for i in range(n_out)])

    # Predetermining which iterations will be kept
    kept = sorted(list(prng.choice(range(n_iter), n_out, replace=False)))
    columns = list(sample.columns)

    next_kept = kept.pop(0)
    next_column = columns.pop(0)

    for i in range(n_iter):

        attempted_step = None
        while attempted_step is None:
            attempted_step = rd_step(current_step, G, h, n)

            if attempted_step is None:
                n_failed += 1
                if n_failed == n_max:
                    msg = 'Maximum failed threshold reached'
                    raise omfa.ModelError(msg)

        current_step = attempted_step

        if i == next_kept:
            sample[next_column] = kernel.dot(current_step) + xo 

            try:
                next_kept = kept.pop(0)
                next_column = columns.pop(0)
            except IndexError:
                pass

    return(sample)

#-------------------------------------------------------------------------
def _ma_sample(model, lower, upper, seed=None, **param):
    """
    Generates a random sample from the model solution space defined by
    lower and upper constraints using a mirror algorithm.

    see generate_random_sample()...
    """

    kernel = model.nullspace

    # Unpacking parameters
    n_burnin = param['n_burnin']
    n_iter = param['n_iter']
    n_out = param['n_out']

    xo = param['xo']

    # Checking ma-specific input
    if 'sd' in param:
        sd = param['sd']
        if not set(sd.index).issuperset(set(kernel.columns)):
            msg = 'A standard deviation must be provided for each basis'
            raise ValueError(msg)
    else:
        ranges = model.generate_basis_ranges(lower, upper)
        sd = abs(ranges['high'] - ranges['low'])/5.0
        sd[pd.isnull(sd)] = 1 

    # Setting random number generator
    if seed is None:
        prng = omfa.prng
    else:
        prng = np.random.RandomState(seed)

    # Recalculating inequality constraints as deviation from central point
    new_lower = lower - xo[lower.index]
    new_upper = upper - xo[upper.index]

    # Round small values down to zero
    new_lower[abs(new_lower) < 1e-8] = 0
    new_upper[abs(new_upper) < 1e-8] = 0

    G, h = model.generate_basis_constraints(new_lower, new_upper)

    G = np.array(G)
    h = np.array(h)

    m, n = G.shape

    # Re-defining starting location as zero
    current_step = np.zeros(n)

    # Defining the random step function
    def m_step(current_step, G, h, n, sd):

        if any(G.dot(current_step) < h):
            msg = 'Initial step out of bounds'
            raise ValueError(msg)

        # Generate new step to be corrected
        deviation = np.array([prng.normal(0, sd[i]) 
                              for i in range(n)])

        new_step = current_step + deviation

        # Initializing origin at current_step
        origin = current_step

        # Checking if new step fits criteria
        residual = G.dot(new_step) - h

        while any(residual < 0):
            ray = new_step - origin

            # Indexes of blocked directions
            indexes = np.where(residual < 0)[0]

            # Point of contact with constraints
            alpha = ((h - G.dot(origin))/(G.dot(ray)))[indexes]

            # Closest point of contact
            alpha_min = min(alpha[np.logical_not(np.isnan(alpha))])
            alpha_index = np.where(alpha == alpha_min)[0][0]
            index = indexes[alpha_index]

            # Calculating reflection from constraint
            d = -residual[index] / (G[index, :]**2).sum()

            new_step = new_step + (2*d*G[index, :]).squeeze()

            # Updating origin as last point of reflection
            origin = origin + alpha[alpha_index]*ray

            # Re-calculating residual
            residual = G.dot(new_step) - h

        return(new_step)

    # Burnin
    for i in range(n_burnin):
        current_step = m_step(current_step, G, h, n, sd)

    # Initializing output
    sample = pd.DataFrame(index=kernel.index, 
                          columns=['S{0}'.format(i) for i in range(n_out)])

    # Predetermining which iterations will be kept
    kept = sorted(list(prng.choice(range(n_iter), n_out, replace=False)))
    columns = list(sample.columns)

    next_kept = kept.pop(0)
    next_column = columns.pop(0)

    for i in range(n_iter):
        current_step = m_step(current_step, G, h, n, sd)

        if i == next_kept:
            sample[next_column] = kernel.dot(current_step) + xo 

            try:
                next_kept = kept.pop(0)
                next_column = columns.pop(0)
            except IndexError:
                pass

    return(sample)


#========================================================================>
# Fortran flux realizations from stoichiometric model.

#------------------------------------------------------------------------
def _rda_sample_f(model, lower, upper, seed=None, **param):
    """
    Generates a random sample from the model solution space defined by
    lower and upper constraints using a random direction algorithm.

    Wrapper for Fortran implementation.

    see generate_random_sample()...
    """

    kernel = model.nullspace

    # Unpacking parameters
    n_burnin = param['n_burnin']
    n_iter = param['n_iter']
    n_out = param['n_out']

    xo = param['xo']

    # Checking rda-specific input
    if 'n_max' in param:
        n_max = param['n_max']
        if n_max <= (n_burnin + n_iter):
            msg = '"n_max" must be greater than "n_burnin" + "n_iter"'
            raise ValueError(msg)
    else:
        n_max = (n_burnin + n_iter)*2

    # Setting random number seed for Fortran routine. If seed isn't
    # provided, one is generated from the current random state.
    if seed is None:
        seed = omfa.prng.tomaxint()
    else:
        seed = int(seed)

    # Formulating arguments to Fortran function
    flux_names = list(kernel.index)
    xo = np.array(xo[flux_names])

    kernel = np.array(kernel)
    m, n = kernel.shape

    lb = np.array(lower)
    lb_index = lower.index
    ub = np.array(upper)
    ub_index = upper.index

    i_lb = [flux_names.index(i) + 1 for i in lb_index if i in flux_names]
    i_ub = [flux_names.index(i) + 1 for i in ub_index if i in flux_names]

    param = [n_burnin, n_iter, n_max]

    sample = omfa.realization.rda_sample(
                kernel, m, n, xo, lb, i_lb, ub, i_ub, 
                n_out, param, seed)

    columns = ['S{0}'.format(i+1) for i in range(n_out)] 
    out = pd.DataFrame(sample, index=flux_names, columns=columns)

    return(out)

#-------------------------------------------------------------------------
def _ma_sample_f(model, lower, upper, seed=None, **param):
    """
    Generates a random sample from the model solution space defined by
    lower and upper constraints using a mirror algorithm.

    Wrapper for Fortran implementation.

    see generate_random_sample()...
    """

    kernel = model.nullspace

    # Unpacking parameters
    n_burnin = param['n_burnin']
    n_iter = param['n_iter']
    n_out = param['n_out']

    xo = param['xo']

    # Checking ma-specific input
    if 'sd' in param:
        sd = param['sd']
        if not set(sd.index).issuperset(set(kernel.columns)):
            msg = 'A standard deviation must be provided for each basis'
            raise ValueError(msg)
    else:
        ranges = model.generate_basis_ranges(lower, upper)
        sd = abs(ranges['high'] - ranges['low'])/5.0
        sd[pd.isnull(sd)] = 1 

    # Setting random number seed for Fortran routine. If seed isn't
    # provided, one is generated from the current random state.
    if seed is None:
        seed = omfa.prng.tomaxint()
    else:
        seed = int(seed)

    # Formulating arguments to Fortran function
    flux_names = list(kernel.index)
    basis_names = list(kernel.columns)
    xo = np.array(xo[flux_names])
    sd = np.array(sd[basis_names])

    kernel = np.array(kernel)
    m, n = kernel.shape

    lb = np.array(lower)
    lb_index = lower.index
    ub = np.array(upper)
    ub_index = upper.index

    i_lb = [flux_names.index(i) + 1 for i in lb_index if i in flux_names]
    i_ub = [flux_names.index(i) + 1 for i in ub_index if i in flux_names]

    param = [n_burnin, n_iter]

    sample = omfa.realization.ma_sample(
                kernel, m, n, xo, lb, i_lb, ub, i_ub, sd, 
                n_out, param, seed)

    columns = ['S{0}'.format(i+1) for i in range(n_out)] 
    out = pd.DataFrame(sample, index=flux_names, columns=columns)

    return(out)

#=========================================================================>
# Application of noise to simulate flux observations.

#-------------------------------------------------------------------------
def generate_observations(flux, n_observations, abs_bias=None, rel_bias=None, 
                          abs_sd=None, rel_sd=None, seed=None):
    """
    Generates n observations per set of flux values by perturbing the
    flux input by normally distributed noise. Any noise parameters
    that are not None must be set for all fluxes. The function does
    not make any assumptions.
    
    Parameters
    ----------
    flux: pandas.Series or pandas.DataFrame
        The flux values to perturb with noise. Each flux will be perturbed
        n_observations times.
    n_observations: int
        The number of observations to generate from each set of fluxes.
    abs_bias: pandas.Series
        A set of bias values corresponding to each flux.
    rel_bias: pandas.Series
        A set of bias values corresponding to each flux, represented
        as a fraction of the observed flux.
    abs_sd: pandas.Series
        A set of standard deviation values corresponding to each flux.
    rel_sd: pandas.Series
        A set of standard deviation values corresponding to each flux,
        represented as a fraction of the observed flux.
    seed: hashable 
        By default, a random number generator is initialized on package
        import. A new generator can be initialized with the provided
        seed to be used for this operation alone.

    Returns
    -------
    observations: pandas.DataFrame
        A data frame where each column is an observation.
    """

    # Coding parameters as dictionary
    param = {'abs_bias':abs_bias,
             'rel_bias':rel_bias,
             'abs_sd':abs_sd,
             'rel_sd':rel_sd}

    # Checking input
    for parameter in param:
        if not param[parameter] is None:
            if set(param[parameter].index) != set(flux.index):
                msg = 'Noise parameters must have values for each flux'
                raise ValueError(msg)
        else:
            param[parameter] = pd.Series(0, index=flux.index)

    for parameter in ['rel_bias', 'rel_sd']:
        if any(param[parameter] > 1) or any(param[parameter] < 0):
            msg = 'Relative parameters must be fractions'
            raise ValueError(msg)

    # Setting random number generator
    if seed is None:
        prng = omfa.prng
    else:
        prng = np.random.RandomState(seed)

    # Preparing index before concatenation
    observation_names = ['O{0}'.format(i) for i in range(1, n_observations+1)]
    
    try:
        n_samples = len(flux.columns)
        sample_array = list(flux.columns)*n_observations

        observation_array = sorted(observation_names*n_samples)

        columns = pd.MultiIndex.from_arrays([sample_array, observation_array])
    except AttributeError:
        columns = observation_names

    # Concatenating input
    flux = pd.concat([flux]*n_observations, axis=1)
    flux.columns = columns

    try:
        flux.sortlevel(level=0, axis=1, inplace=True)
    except TypeError:
        pass

    # Generating noise
    noise = flux.copy()
    n_row, n_col = noise.shape
    
    noise.iloc[:, :] = (prng.randn(n_row*n_col)
                            .reshape((n_row, n_col), order='F'))
    noise_abs = noise.mul(param['abs_sd'], axis=0)

    noise.iloc[:, :] = (prng.randn(n_row*n_col)
                            .reshape((n_row, n_col), order='F'))
    noise_rel = noise * flux.mul(param['rel_sd'], axis=0)

    noise = noise_abs + noise_rel

    # Adding bias
    observation = (flux.add(param['abs_bias'], axis=0) + 
                   flux.mul(param['rel_bias'], axis=0) +
                   noise)

    return(observation)





