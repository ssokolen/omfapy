"""
Definition of omfa.PIFit and omfa.GLSFit classes
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, t
from scipy.linalg import qr

# Local imports
import omfapy as omfa

#=======================================================================>
# Pseudo-inverse fit

#-----------------------------------------------------------------------
class PIFit:
    """
    Uses a simple Moore-Penrose pseudo-inverse for flux calculation and
    defines a chi-squared test for fit validation.
    """

    # Initializing basic model variables 
    stoichiometry = None

    calculated_matrix = None
    observed_matrix = None
    calculated_fluxes = None
    observed_fluxes = None
    
    def __init__(self, model, observations):
        """
        Initializes PIFit and partitions model into known and unknown 
        matrices.

        Paremeters
        ----------
        model: omfa.Model
            The model to which data is being fit
        observations: pandas.Series
            Observed values
        """

        S = model.stoichiometry.copy()

        observations_names = observations.index
        calculated_fluxes_names = [i for i in S.columns 
                                     if i not in observations_names]

        # Partitioning model (and switching to common nomenclature)
        S_o = S[observations_names]
        S_c = S[calculated_fluxes_names]

        v_o = observations

        # Calculating
        v_c = -np.linalg.pinv(S_c).dot(S_o).dot(v_o)

        # Whether v_c acts like a pandas.Series or pandas.DataFrame
        # depends on the input observations
        try:
            v_c = pd.DataFrame(v_c, index=calculated_fluxes_names,
                                    columns=v_o.columns)
        except AttributeError:
            v_c = pd.Series(v_c, index=calculated_fluxes_names)

        # Storing local copies
        self.stoichiometry = S

        self.observed_fluxes = v_o.copy()
        self.calculated_fluxes = v_c.copy()
        self.observed_matrix = S_o.copy()
        self.calculated_matrix = S_c.copy()

    #-------------------------------------------------------------------
    # Model validation
    def validate(self, covar):
        """
        Performs PIFit validation using a Chi-squared test.

        Parameters
        ----------
        covar: pd.DataFrame
            Covariance matrix corresponding to measurement errors 
            (used to weigh model residuals).

        Returns
        ----------
        p: float
            p-value for Chi-squared test with p < alpha indicating
            a significant probability of gross measurement error.
        """

        # Renaming to follow common nomenclature
        S_o = self.observed_matrix
        S_c = self.calculated_matrix
        v_o = self.observed_fluxes
        v_c = self.calculated_fluxes

        F = covar

        # Checking that covariance corresponds to observed fluxes
        if not set(F.index) == set(v_o.index):
            msg = 'Covariance matrix must correspond to observed fluxes'
            raise omfa.ModelError(msg)
        
        # Calculating redundancy matrix
        R = S_o - np.dot(S_c, np.linalg.pinv(S_c)).dot(S_o)

        # Eliminating linearly dependent rows
        R_t = R.T

        m, n = R_t.shape
        
        zeros = np.zeros((n - m, n))
        padded = np.vstack((R_t, zeros))
        
        q, r, P = qr(padded, pivoting=True)
        independent = P[np.abs(r.diagonal()) > 1e-10]

        Rr = R.ix[independent,:]

        # Calculated residuals
        epsilon = Rr.dot(v_o)

        # Transforming observation covariance matrix
        P = Rr.dot(F).dot(Rr.T)

        # Calculating statistic
        epsilon_scaled = np.dot(epsilon.T, np.linalg.inv(P)).dot(epsilon)

        if len(epsilon_scaled.shape) == 2:
            h = np.diag(epsilon_scaled)
            p = 1 - chi2.cdf(h, len(epsilon.index))
            p = pd.Series(p, index=v_o.columns)
        else:
            h = epsilon_scaled
            p = 1 - chi2.cdf(h, len(epsilon.index))

        return(p)

#=======================================================================>
# Generalized least squares fit

#-----------------------------------------------------------------------
class GLSFit:
    """
    Uses a generalized least squares framework for flux calculation and
    validation.
    """

    # Initializing basic model variables
    stoichiometry = None
    covar = None
    model = None

    calculated_matrix = None
    observed_matrix = None
    calculated_fluxes = None
    observed_fluxes = None

    # Initializing intermediate variables
    _X = None
    _y = None

    _X_w = None
    _y_w = None
    _P = None

    def __init__(self, model, observations, covar):
        """
        Initializes GLSFit and partitions model into known and unknown 
        matrices.

        Paremeters
        ----------
        model: omfa.Model
            The model to which data is being fit
        observations:pandas.Series
            Observed values
        covar: pandas.DataFrame
            Covariance matrix for observed values
        """

        # Checking that covariance corresponds to flux balances
        if sorted(covar.index) != sorted(model.stoichiometry.index):
            msg = 'Covariance matrix entries must correspond to flux balances'
            raise omfa.ModelError(msg)

        S = model.stoichiometry.copy()

        observations_names = observations.index
        calculated_fluxes_names = [i for i in S.columns 
                                     if i not in observations_names]

        # Partitioning model (and switching to common nomenclature)
        S_o = S[observations_names]
        S_c = S[calculated_fluxes_names]

        v_o = observations

        # Renaming to least squares notation
        X = S_c
        y = -S_o.dot(v_o)

        m, n = X.shape
        df = m - n

        V = covar

        # Finding weights through singular value decomposition
        gamma, s, gamma_inv = np.linalg.svd(covar, compute_uv=True)

        Dp = np.diag(np.sqrt(s))
        P = pd.DataFrame(gamma.dot(Dp).dot(gamma_inv),
                         index=V.index, columns=V.columns)

        P_inv = pd.DataFrame(np.linalg.inv(P), 
                             index=V.index, columns=V.columns)

        # Weighing X and y
        X_w = P_inv.dot(X)
        y_w = P_inv.dot(y)

        # Calculating fluxes
        B = np.linalg.inv(X_w.T.dot(X_w)).dot(X_w.T).dot(y_w)

        # Whether B acts like a pandas.Series or pandas.DataFrame
        # depends on the input observations
        try:
            B = pd.DataFrame(B, index=calculated_fluxes_names,
                                      columns=v_o.columns)
        except AttributeError:
            B = pd.Series(B, index=calculated_fluxes_names)

        # Storing local copies
        self.stoichiometry = S

        self.observed_fluxes = observations.copy()
        self.calculated_fluxes = B.copy()
        self.observed_matrix = S_o.copy()
        self.calculated_matrix = S_c.copy()

        self._X = X
        self._y = y

        self._X_w = X_w
        self._y_w = y_w
        self._P = P

    #-------------------------------------------------------------------
    # Model validation
    def validate(self):
        """
        Performs GLSFit validation using least squares principles.

        Returns
        ----------
        p: pd.Series
            p-value for t-tests conducted on calculated flux values with
            p < alpha indicating a significant probability of a non-zero
            flux.
        B_interval: pd.DataFrame
            95% confidence intervals on calculated flux values.
        yp_interval: pd.DataFrame
            95% confidence intervals on predicted mass balances.
        """

        # Renaming to follow common nomenclature
        X_w = self._X_w
        y_w = self._y_w
        B = self.calculated_fluxes
        P = self._P

        m, n = X_w.shape
        df = m - n - 1

        # Calculating residuals
        epsilon = y_w - X_w.dot(B)
        
        if len(epsilon.shape) == 2:
            sse = (epsilon**2).apply(np.sum, axis=0)
        else:
            sse = np.sum(epsilon**2)

        # Chi squared test on residuals
        p_chi2 = 1 - chi2.cdf(sse, m)

        try:
            p_chi2 = pd.Series(p_chi2, index=y_w.columns)
        except AttributeError:
            pass

        # Confidence intervals
        sigma2 = sse/float(df)

        # To generate the confidence intervals, we need standard deviation
        # values for each calculated flux. The covariance structure is
        # calculated from the X matrix -- np.linalg.inv(X_w.T.dot(X_w)),
        # and the only part that changes is the sse estimate from the
        # residuals. Thus, the covariance diagonals must be expanded
        # to provide one set per variance estimate.

        if len(sigma2.shape) == 1:
            covar = (np.diag(np.linalg.inv(X_w.T.dot(X_w)))
                        .repeat(sigma2.size)
                        .reshape((-1, sigma2.size), order='C'))
            covar = pd.DataFrame(covar, index=B.index, columns=sigma2.index)
        else:
            covar = np.diag(np.linalg.inv(X_w.T.dot(X_w)))
            covar = pd.Series(covar, index=B.index)
        
        var_B = covar * sigma2
        sd_B = np.sqrt(var_B)
        
        error = t.ppf(0.975, df) * sd_B
        B_lower = B - error
        B_upper = B + error
        B_predicted = B.copy()

        n_col = len(B.index)
        B_lower.index = pd.MultiIndex.from_arrays(
                            [B.index, ['lower']*n_col])
        B_upper.index = pd.MultiIndex.from_arrays(
                            [B.index, ['upper']*n_col])
        B_predicted.index = pd.MultiIndex.from_arrays(
                                [B.index, ['predicted']*n_col])

        B_interval = pd.concat([B_lower, B_upper, B_predicted]).unstack()

        to = B / sd_B

        try:
            p_t = pd.DataFrame(1.00, index=B.index,
                                  columns=B.columns)
        except AttributeError:
            p_t = pd.Series(1.00, index=B.index)
        
        p_t[B < 0] = t.cdf(to[B < 0], df)*2
        p_t[B > 0] = (1 - t.cdf(to[B > 0], df))*2

        # Prediction intervals
        H = np.dot(X_w.dot(np.linalg.inv(X_w.T.dot(X_w))), X_w.T)

        # Working in a similar fashion to confidence intervals
        if len(sigma2.shape) == 1:
            covar = (np.diag(1 + H)
                        .repeat(sigma2.size)
                        .reshape((-1, sigma2.size), order='C'))
            covar = pd.DataFrame(covar, index=y_w.index, columns=sigma2.index)
        else:
            covar = np.diag(1 + H)
            covar = pd.Series(covar, index=y_w.index)

        yp_w = H.dot(y_w)

        var_yp_w = covar * sigma2
        sd_yp_w = np.sqrt(var_yp_w)

        error = t.ppf(0.975, df) * sd_yp_w
        yp_w_lower = yp_w - error
        yp_w_upper = yp_w + error

        yp_lower = P.dot(yp_w_lower)
        yp_upper = P.dot(yp_w_upper)
        yp_observed = self._y.copy()
        
        yp_predicted = P.dot(yp_w)
	if len(yp_predicted.shape) > 1:
            yp_predicted.columns = yp_observed.columns

        n_col = len(y_w.index)
        yp_lower.index = pd.MultiIndex.from_arrays(
                             [yp_lower.index, ['lower']*n_col])
        yp_upper.index = pd.MultiIndex.from_arrays(
                             [yp_upper.index, ['upper']*n_col])
        yp_predicted.index = pd.MultiIndex.from_arrays(
                                 [yp_predicted.index, ['predicted']*n_col])
        yp_observed.index = pd.MultiIndex.from_arrays(
                                [yp_observed.index, ['observed']*n_col])

        yp_interval = (pd
                           .concat([yp_lower, yp_upper, 
                                    yp_predicted, yp_observed])
                           .unstack())

        # Adding observed and predicted values to intervals
        return(p_chi2, p_t, B_interval, yp_interval)
        
