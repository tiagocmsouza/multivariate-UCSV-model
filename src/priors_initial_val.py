import numpy as np

from parameters import nper, rgn
from data_estimation import scale_y, n_y, notim


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
- tau_c, tau_i, epsilon_c: factors
- sigma_i_epsilon: standard deviation of individual errors
- scale_i_epsilon: outliers of individual errors
- prior_var_alpha: initial values volatility of alpha_dtau and alpha_epsilon - Normal Prior
- sigma_dalpha: lambda_i_tau, lambda_i_epsilon - volatility scale of alpha (not time varying) - Inverse Gamma Prior
"""

# Uses "y" to recover the "alpha"'s, given the "tau"'s as loadings and variance (lambda) is not time varying

# Initial values
alpha_tau = np.ones((notim, n_y))
alpha_eps = np.ones_like(alpha_tau)

# prior_var_alpha
kappa_1, kappa_2 = 10, 0.4

kappa_1 = kappa_1 / scale_y
kappa_2 = kappa_1 / scale_y

var_alpha_tau = ((kappa_1**2) * np.ones([n_y, n_y])) + ((kappa_2**2) * np.eye(n_y))
var_alpha_eps = ((kappa_1**2) * np.ones([n_y, n_y])) + ((kappa_2**2) * np.eye(n_y))

prior_var_alpha = np.block(
    [[var_alpha_tau, np.zeros([n_y, n_y])], [np.zeros([n_y, n_y]), var_alpha_eps]]
)


"""
Step 1.a.3 - Gibbs-within-Gibbs
Draw: lambda_tau, lambda_epsilon
Given:
- dalpha: change in factor loadings
- nu_prior_alpha: prior of number of observations
- s2_prior_alpha: prior of variance per period
- That is why SSR is the product of nu and s2
"""

# sigma_dalpha
nu_prior_alpha = 0.1 * notim
s2_prior_alpha = ((0.25 / np.sqrt(notim)) ** 2) / (scale_y**2)

a = nu_prior_alpha / 2
ssr = nu_prior_alpha * s2_prior_alpha
b = 2 / ssr
var_dalpha = 1 / rgn.gamma(
    shape=a, scale=1 / b, size=2 * n_y
)  # n_y epsilons and n_y alphas
sigma_dalpha = np.sqrt(var_dalpha)


"""
Step 1.b
Draw: log chi-squared indicators
Given:
- epsilon_unique (= y - y_epsilon_common - y_tau_common - y_tau_unique)
- dtau_common (sigma_dtau_common): divided by their respective standatd deviations
- dtau_unique (sigma_dtau_unique): divided by their respective standatd deviations
- epsilon_common (sigma_epsilon_common, scale_eps_common): also divided by their respective outlier scales
- epsilon_unique (sigma_epsilon_unique, scale_eps_unique): also divided by their respective outlier scales
- r_p, r_m, r_s: prior, mean and standard deviation of normals to be mixtured to indicators to deliver log chi-squared distribution
"""

### Distribution Parameters

# 10-component mixture approximation to log chi-squared(1) from Omori, Chib, Shepard, and Nakajima JOE (2007)
r_p = np.array(
    (
        0.00609,
        0.04775,
        0.13057,
        0.20674,
        0.22715,
        0.18842,
        0.12047,
        0.05591,
        0.01575,
        0.00115,
    )
)  # Prior mixture probabilities
r_m = np.array(
    (
        1.92677,
        1.34744,
        0.73504,
        0.02266,
        -0.85173,
        -1.97278,
        -3.46788,
        -5.55246,
        -8.68384,
        -14.65000,
    )
)  # Means
r_v = np.array(
    (
        0.11265,
        0.17788,
        0.26768,
        0.40611,
        0.62699,
        0.98583,
        1.57469,
        2.54498,
        4.16591,
        7.33342,
    )
)
r_s = np.sqrt(r_v)  # Standard deviations


"""
Step 2.a
Draw: gamma, standard deviation of innovation for ln(volatility)
Given:
- epsilon_unique (= y - y_epsilon_common - y_tau_common - y_tau_unique)
- dtau_common (sigma_dtau_common): divided by their respective standatd deviations
- dtau_unique (sigma_dtau_unique): divided by their respective standatd deviations
- epsilon_common (sigma_epsilon_common, scale_eps_common): also divided by their respective outlier scales
- epsilon_unique (sigma_epsilon_unique, scale_eps_unique): also divided by their respective outlier scales
- r_p, r_m, r_s: prior, mean and standard deviation of normals to be mixetured to indicators to deliver log chi-squared distribution
- gamma_eps_common_prior, gamma_dtau_common_prior, gamma_eps_unique_prior, gamma_dtau_unique_prior: gamma priors
- ind_eps_common, ind_dtau_common, ind_eps_unique, ind_dtau_unique: mixture indicators
- i_init: 1 if vague prior for variance used; i_init = 0 if ln(sig2) = 0 is used
"""

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

"""
Step 3
Draw: scale parameters for epsilon (common and uniques)
Given:
- epsilon_common
- epsilon_unique (= y - y_epsilon_common - y_tau_common - y_tau_unique)
- sigma_epsilon_common, sigma_epsilon_unique
- ind_eps_common, ind_eps_unique: from Step 1.b
- r_m, r_s: mean and standard deviation of normals to be mixetured to indicators to deliver log chi-squared distribution
- scl_eps_vec: list of scale values
- prob_scl_eps_vec_common: prior probabilities on scale values
"""

### Prior for scale mixture of epsilon component
scl_eps_vec = np.linspace(1, 10, 10)


"""
Step 4
Draw: probability of outlier
Given:
- scale_eps_common, scale_eps_unique: from Step 3
- ps_prior_a, ps_prior_b: alpha and beta, respectively, in beta prior
- n_scl_eps
"""

### Prior for scale mixture of epsilon component
ps_mean = 1 - 1 / (4 * nper)  # Outlier every 4 years (THIS IS FOR QUARTERLY DATA!)
ps_prior_obs = nper * 10  # Sample size of 10 years for prior
ps_prior_a = ps_mean * ps_prior_obs  # "alpha" in beta prior
ps_prior_b = (1 - ps_mean) * ps_prior_obs  # "beta" in beta prior
n_scl_eps = scl_eps_vec.size
