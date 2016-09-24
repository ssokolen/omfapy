import logging
import numpy as np
import pandas as pd
import pickle
import sys
import time

# Local imports
import omfapy as omfa

# Loading all observed fluxes
fluxes = pd.DataFrame.from_csv('../observed_flux.csv', index_col=0)

# Loading simulation constraints
constraints = pd.DataFrame.from_csv('./local_constraints.csv', index_col=0)

#------------------------------------------------------------------------
# Generating model from csv input

m = omfa.Model.from_csv('../model.csv')

# Dropping redundant compounds
redundant = ['mCoA', 'FH4', 'mFAD', 'NAD', 'NADP']
m.remove_compounds(redundant)

# Dropping unbalanced compounds
unbalanced = ['ADP', 'ATP', 'mFADH', 
              'Pi', 'H2O', 'O2', 'CO2']
m.remove_compounds(unbalanced)

# Dropping oxidative phosphorylation reactions
dangling = ['OX_1', 'OX_2', 'T_O2', 'T_CO2']
m.remove_reactions(dangling)

# Validating
observed_fluxes = fluxes.index
problem_compounds, problem_reactions = \
    m.validate(transport=observed_fluxes)

if len(problem_compounds + problem_reactions) != 0:
    msg = 'Unresolved problems with model, aborting...'
    raise ValueError(msg)

#------------------------------------------------------------------------
# Basic analysis
sd = fluxes['sd']
flux = fluxes['mean']

# Calculating chi squared
covar = m.generate_covariance(flux, abs_measurement=sd)

fit_pi = m.fit_pi(flux)
p_chi2 = fit_pi.validate(covar)

msg = 'Chi squared test p value = {:.2}'
print(msg.format(p_chi2))

# GLS framework
covar = m.generate_covariance(flux, abs_measurement=sd, abs_balance=0.1) 

fit_gls = m.fit_gls(flux, covar)
p_chi2, p_t, ci, pi = fit_gls.validate()

combined = pd.concat([ci, p_t], axis=1)
combined.columns = ['lower', 'predicted', 'upper', 'p_t']
combined.to_csv('observed_mfa.csv')
print('Calculated flux confidence intervals:')
print(combined)
msg = '{0:d} fluxes significant of {1:d}'
print(msg.format(sum(combined['p_t'] < 0.05), len(combined['p_t'])))

#------------------------------------------------------------------------
# Simulation

# Initializing analysis
analysis = omfa.Analysis('local_simulation.db', m, seed=1111, overwrite=True)

# Generating samples
m.check_constraints(constraints['min'], constraints['max'])

ranges = m.generate_flux_ranges(constraints['min'], constraints['max'])

new_lower, new_upper = m.reduce_constraints(constraints['min'], 
                                            constraints['max'])

xo = m.generate_flux_centroid(constraints['min'], 
                              constraints['max'], progress=False)
analysis.generate_samples(100, constraints['min'], constraints['max'], 
                          n_burnin=10000, n_iter=10000, xo=xo)

# Generating observations at current levels of uncertainty
analysis.generate_observations(100, 'observed', abs_sd=sd)

# GLS validation
analysis.calculate_gls('observed', rel_measurement=sd/flux)

# Chi2 validation
analysis.calculate_pi('observed', rel_measurement=sd/flux)
