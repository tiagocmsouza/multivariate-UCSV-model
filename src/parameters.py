import numpy as np


### General Parameters
small = 1e-10
big = 1e6
nper = 12  # Units of time per year (Monthly: 12, Quarterly: 4)
rgn = np.random.default_rng(1010)


### Parameters for UCSV Draws
n_burnin = 5000  # Discarted Draws
n_draws_save = 5000  # Number of Draws to Save
k_draws = 10  # Save results every k_draws
n_draws = n_draws_save * k_draws  # Total Number of Draws after burning


### Data Parameters
start_sample = "1960-1"

pce_groups = [
    "Personal consumption expenditures",
    "PCE excluding food and energy",
    "PCE excluding energy",
    "Motor vehicles and parts",
    "Furnishings and durable household equipment",
    "Recreational goods and vehicles",
    "Other durable goods",
    "Food and beverages purchased for off-premises consumption",
    "Clothing and footwear",
    "Gasoline and other energy goods",
    "Other nondurable goods",
    "Housing",
    "Water supply and sanitation (25)",
    "Electricity and gas",
    "Health care",
    "Transportation services",
    "Recreation services",
    "Food services and accommodations",
    "Financial services and insurance",
    "Other services",
    "Final consumption expenditures of nonprofit institutions serving households (NPISHs) (132)",
]


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
