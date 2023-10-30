import numpy as np

from parameters import nper


### Parameters for scale mixture of epsilon component
scl_eps_vec = np.linspace(1, 10, 10)
ps_mean = 1 - 1 / (4 * nper)  # Outlier every 4 years
ps_prior_obs = nper * 10  # Sample size of 10 years for prior
ps_prior_a = ps_mean * ps_prior_obs  # "alpha" in beta prior
ps_prior_b = (1 - ps_mean) * ps_prior_obs  # "beta" in beta prior


### Prior for g (ln(sˆ2) = 2*ln(s))
ng = 5  # Number of grid points for approximate uniform prior


def g_values_prob(min, max):
    g_values = 2 * np.linspace(min, max, ng) / np.sqrt(nper)
    g_prob = np.ones(ng) / ng

    return {"values": g_values, "prob": g_prob}


g_dtau_common_prior = g_values_prob(min=0.01, max=0.2)
g_eps_common_prior = g_values_prob(min=0.01, max=0.2)
g_dtau_unique_prior = g_values_prob(min=0.01, max=0.2)
g_eps_unique_prior = g_values_prob(min=0.01, max=0.2)
