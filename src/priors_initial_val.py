import numpy as np

from parameters import nper
from data_estimation import scale_y, n_y


"""
Step 1.a.1 - Gibbs-within-Gibbs
Draw: tau_c, tau_i, epsilon_c
Given:
- alpha_tau, alpha_epsilon: factor loadings of tau_c and epsilon_c
- lambda_i_tau, lambda_i_epsilon: scale parameters of factor loadings
- sigma_c_dtau: standard deviation of common factor
- sigma_i_dtau: standard deviation of individual factors
- sigma_c_epsilon: standard deviation of common error
- sigma_i_epsilon: standard deviation of individual errors
- scale_c_epsilon: outliers of common error
- scale_i_epsilon: outliers of individual errors
"""

# tau_c_0 = 0
# epsilon_c_0 = 0
# tau_i_0 = Diffuse prior, N(0, 1e6)
# Starting values of Kalman Filter / Smoother
# Uses "y" to recover the "tau"'s, given loadings and variance of these loadings


"""
Step 1.a.2 - Gibbs-within-Gibbs
Draw: alpha_tau, alpha_epsilon (loadings only affect common factor)
Given:
- alpha_tau, alpha_epsilon: factor loadings of tau_c and epsilon_c
- lambda_i_tau, lambda_i_epsilon: scale parameters of factor loadings
- sigma_c_dtau: standard deviation of common factor
- sigma_i_dtau: standard deviation of individual factors
- sigma_c_epsilon: standard deviation of common error
- sigma_i_epsilon: standard deviation of individual errors
- scale_c_epsilon: outliers of common error
- scale_i_epsilon: outliers of individual errors
"""

kappa_1, kappa_2 = 10, 0.4

kappa_1 = kappa_1 / scale_y
kappa_2 = kappa_1 / scale_y

var_alpha_tau = ((kappa_1**2) * np.ones([n_y, n_y])) + ((kappa_2**2) * np.eye(n_y))
var_alpha_eps = ((kappa_1**2) * np.ones([n_y, n_y])) + ((kappa_2**2) * np.eye(n_y))

prior_var_alpha = np.block(
    [[var_alpha_tau, np.zeros([n_y, n_y])], [np.zeros([n_y, n_y]), var_alpha_eps]]
)


### Prior for scale mixture of epsilon component
scl_eps_vec = np.linspace(1, 10, 10)
ps_mean = 1 - 1 / (4 * nper)  # Outlier every 4 years
ps_prior_obs = nper * 10  # Sample size of 10 years for prior
ps_prior_a = ps_mean * ps_prior_obs  # "alpha" in beta prior
ps_prior_b = (1 - ps_mean) * ps_prior_obs  # "beta" in beta prior


### Prior for Gamma (ln(sˆ2) = 2*ln(s))
ng = 5  # Number of grid points for approximate uniform prior


def g_values_prob(min, max):
    g_values = 2 * np.linspace(min, max, ng) / np.sqrt(nper)
    g_prob = np.ones(ng) / ng

    return {"values": g_values, "prob": g_prob}


g_dtau_common_prior = g_values_prob(min=0.01, max=0.2)
g_eps_common_prior = g_values_prob(min=0.01, max=0.2)
g_dtau_unique_prior = g_values_prob(min=0.01, max=0.2)
g_eps_unique_prior = g_values_prob(min=0.01, max=0.2)
