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
