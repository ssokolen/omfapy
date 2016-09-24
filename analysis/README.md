# Analysis code

## Model formulation

The file `model.csv` contains the MFA model described in the manuscript. The
format can be directly read by the `omfapy` package. Two sets of analysis
were performed in the manuscript. The "original" formulation excluded NADH
and NADPH from the material balance (as typical for many models). The
"modified" formulation assumed no electron transport chain and balanced
both compounds.

## Simulations

MFA analysis centered on the 66 hour time point of a CHO cell culture, with 
the observed fluxes collected in `observed_flux.csv`. "Local"
simulations were performed around the 66 hour time point observations to relate
theoretical model performance with the observed results. An "extended"
simulation was performed on broader constraints to get a more general
understanding of model performance. Both types of simulations were carried
out on original and modified model formulations.

## Running the code

Once omfapy is installed, it should be possible to navigate to `original
formulation` or `modified formulation` directories and run
`local_analysis.py` or `extended_analysis.py`. Both will output observed
flux results in an `observed_mfa.csv` files as well as a corresponding
SQLite database. Note, the extended simulation may take up to 4 hours to run 
on a mid-tier laptop (2016). Once both local and extended simulations have 
been run, it's possible to execute the `format.R` code within the `plots`
directory to generate most of the figures in the manuscript.

## Database structure

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
