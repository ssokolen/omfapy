Please note, this github page is still under development. The documentation should be ironed out within the next couple of weeks. If you spot any issues or need extra information, email me at stanislav 'at' sokolenko.net.

## omfapy

This package is meant support the Metabolic Flux Analysis methodology
described in the article "Identifying model error in metabolic flux
analysis - A generalized least squares approach." The article is open
access and available
[here](https://bmcsystbiol.biomedcentral.com/articles/10.1186/s12918-016-0335-7).
Please consult the article for the background and theoretical principles
behind this package.  

## Installation

Fortran code has to be compiled for a number of important package
functions. Python fall-back functions have been implemented, but may run
100-1000 times slower. You will need `liblapack` and `gfortran`. Assuming that
the `omfapy` code is downloaded and the working directory of the terminal is
set to omfapy:

    cd ./fortran
    gfortran -fPIC -c -llapack ./misc/*.f90
    f2py -c -m libomfa -llapack ./*.o ./libomfa/*.f90

The python package can the be installed as normal. I recommend using
`pip`. Setting the working directory of the terminal to omfapy:

    sudo pip install '.'

## Basic usage

The `omfapy` functions are written around the `pandas` package data
structures of DataFrames and Series. Please consult `pandas` documentation
for more information. The 'analysis' directory has model and data
information from the manuscript that can be used to get started. Launch
python and set the working directory to the 'analysis' directory:

Models are generated from simple `pandas` DataFrames. A simple script is
available to parse model structure from a more human readable csv file:

    import omfapy as omfa
    import pandas as pd    

    # Loading all observed fluxes
    fluxes = pd.DataFrame.from_csv('observed_flux.csv', index_col=0)

    # Generating model from csv input
    m = omfa.Model.from_csv('model.csv')

    # Dropping redundant compounds
    redundant = ['mCoA', 'FH4', 'mFAD', 'NAD', 'NADP']
    m.remove_compounds(redundant)

    # Dropping unbalanced compounds
    unbalanced = ['ADP', 'ATP', 'mFADH', 'Pi', 'H2O', 'O2', 'CO2']
    m.remove_compounds(unbalanced)

    # Dropping oxidative phosphorylation reactions
    dangling = ['OX_1', 'OX_2', 'T_O2', 'T_CO2']
    m.remove_reactions(dangling)

    # Validating (transport fluxes have to specified so that they are not
    # seen as "dangling"
    observed_fluxes = fluxes.index
    problem_compounds, problem_reactions = \
        m.validate(transport=observed_fluxes)

    # The validate function should raise warnings, but an explicit check
    # can also be run here
    if len(problem_compounds + problem_reactions) != 0:
        msg = 'Unresolved problems with model, aborting...'
        raise ValueError(msg)

The loaded flux data comes with means and standard deviations. The
standard deviations are necessary to generate a covariance matrix for
validation.

    sd = fluxes['sd']
    flux = fluxes['mean']

    # Preparing the covariance matrix
    covar = m.generate_covariance(flux, abs_measurement=sd, abs_balance=0.1) 

    # Fitting observed fluxes
    fit_gls = m.fit_gls(flux, covar)

    # Validation includes prediction and confidence intervals as well as
    # p-values for t-tests
    p_chi2, p_t, ci, pi = fit_gls.validate()

    # Combinging the data and adding some column names
    combined = pd.concat([ci, p_t], axis=1)
    combined.columns = ['lower', 'predicted', 'upper', 'p_t']
    print('Calculated flux confidence intervals:')
    print(combined)
    msg = '{0:d} fluxes significant of {1:d}'
    print(msg.format(sum(combined['p_t'] < 0.05), len(combined['p_t'])))

## Full analysis code

All the analysis performed in the manuscript is available as a set of
scripts in the 'analysis' directory.

### Model formulation

The file `model.csv` contains the MFA model described in the manuscript. The
format can be directly read by the `omfapy` package. Two sets of analysis
were performed in the manuscript. The 'original' formulation excluded NADH
and NADPH from the material balance (as typical for many models). The
'modified' formulation assumed no electron transport chain and balanced
both compounds.

### Simulations

MFA analysis centered on the 66 hour time point of a CHO cell culture, with 
the observed fluxes collected in `observed_flux.csv`. "Local"
simulations were performed around the 66 hour time point observations to relate
theoretical model performance with the observed results. An "extended"
simulation was performed on broader constraints to get a more general
understanding of model performance. Both types of simulations were carried
out on original and modified model formulations.

### Running the code

Once omfapy is installed, it should be possible to navigate to `original
formulation` or `modified formulation` directories and run
`local_analysis.py` or `extended_analysis.py`. Both will output observed
flux results in an `observed_mfa.csv` files as well as a corresponding
SQLite database. Note, the extended simulation may take up to 4 hours to run 
on a mid-tier laptop (2016). Once both local and extended simulations have 
been run, it's possible to execute the `format.R` code within the `plots`
directory to generate most of the figures in the manuscript.

### Database structure

The generated SQLite database files summarize simulation results in the
following tables: `samples`, `observations`, `gls_calculated`,
`gls_predicted`, `gls_overall`, `pi_calculated`, and `pi_overall`, where
gls stands for Generalized Least Squares and pi for Pseudo-Inverse.
Theoretically balanced fluxes are all stored in the `samples` table. Values
perturbed by simulated measurement noise are stored in the `observations`
table. Calculated fluxes are entered into `gls_calculated` and `pi_calculated`,
with the `gls_calculated` also featuring a t-test probability and 95%
confidence intervals. `gls_predicted` contains prediction intervals on
each balance (net accumulation / consumption not including
transport fluxes). `gls_overall` and `pi_overall` contain chi squared
results, with the `pi_overall` values calculated using a reduced redundancy
matrix and `gls_overall` values calculated on all residuals.
