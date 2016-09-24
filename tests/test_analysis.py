import logging
import numpy as np
import pandas as pd
import sys
import time

sys.path.append('..')

# Local imports
import omfapy as omfa

# Defining model
m = {'A':{'F1':-1, 'TA':1},
     'B':{'F1':1, 'B1':-1, 'B2':-1},
     'C':{'B1':1, 'F3':-1},
     'D':{'B2':1, 'F4':-1},
     'E':{'F3':1, 'TE':-1, 'TE2':-1},
     'F':{'F4':1, 'TF':-1},
     'G':{'F4':1, 'TG':-1}}
m = pd.DataFrame(m).fillna(0).T
m = omfa.Model(m)

lower = pd.Series({'TA':-5, 'TE':-5, 'TE2':-5, 'TF':-5, 'TG':-5})
upper = pd.Series({'TA':5, 'TE':5, 'TE2':5, 'TF':5, 'TG':5})

# Initializing analysis
analysis = omfa.Analysis('test.db', m, seed=1111, overwrite=True)

# Generating samples
analysis.generate_samples(1, lower, upper, n_burnin=2, n_iter=2)
print(analysis._xo)

# Generating observations
rel_sd = pd.Series(0.05, lower.index)
analysis.generate_observations(1, 'test', rel_sd=rel_sd)

# GLS validation
analysis.calculate_gls('test', rel_measurement=0.05)

# GLS validation
analysis.calculate_pi('test', rel_measurement=0.05)

