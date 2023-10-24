import pandas as pd
import numpy as np
from numba import njit

# PCE Groups

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

df_index.loc[:, pce_groups]
