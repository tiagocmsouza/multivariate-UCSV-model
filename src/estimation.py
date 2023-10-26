import pandas as pd
import numpy as np
from numba import njit

### DATA

df_index = pd.read_csv("df_index.csv", index_col=[0], parse_dates=[0])
df_weights = pd.read_csv("df_weights.csv", index_col=[0], parse_dates=[0])

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

df_index = df_index.loc[:, pce_groups]
df_weights = df_weights.loc[:, pce_groups]


# Parameters
small = 1e-10
big = 1e6
nper = 12  # Units of time per year (Monthly: 12, Quarterly: 4)
rgn = np.random.default_rng(1010)


# Prepare data
# PCE - mom annualized
dp_agg = (
    df_index.loc[:, ["Personal consumption expenditures"]].to_period("M").pct_change()
)
dp_agg = 100 * ((1 + dp_agg) ** nper - 1)

dp_agg_np = dp_agg.values

# Core PCE (Ex-Food and Energy) - mom annualized
dp_agg_xfe = (
    df_index.loc[:, ["PCE excluding food and energy"]].to_period("M").pct_change()
)
dp_agg_xfe = 100 * ((1 + dp_agg_xfe) ** nper - 1)

dp_agg_xfe_np = dp_agg_xfe.values

# PCE Ex-Energy - mom annualized
dp_agg_xe = df_index.loc[:, ["PCE excluding energy"]].to_period("M").pct_change()
dp_agg_xe = 100 * ((1 + dp_agg_xe) ** nper - 1)

dp_agg_xe_np = dp_agg_xe.values

# PCE components - qoq annualized
dp_disagg = (
    df_index.drop(
        [
            "Personal consumption expenditures",
            "PCE excluding food and energy",
            "PCE excluding energy",
        ],
        axis="columns",
    )
    .to_period("m")
    .pct_change()
)
dp_disagg = 100 * ((1 + dp_disagg) ** nper - 1)

dp_disagg_np = dp_disagg.values

# PCE - shares
share_avg = (
    df_weights.div(df_weights.loc[:, "Personal consumption expenditures"], axis=0)
    .drop(
        [
            "Personal consumption expenditures",
            "PCE excluding food and energy",
            "PCE excluding energy",
        ],
        axis="columns",
    )
    .to_period("M")
)

share_avg_np = share_avg.values

# Core PCE (Ex-Food and Energy) - shares
share_avg_xfe = share_avg.drop(
    [
        "Electricity and gas",
        "Gasoline and other energy goods",
        "Food and beverages purchased for off-premises consumption",
    ],
    axis="columns",
)

share_avg_xfe = share_avg_xfe.div(share_avg_xfe.sum(axis=1), axis=0)

share_avg_xfe_np = share_avg_xfe.values

# PCE Ex-Energy - shares
share_avg_xe = share_avg.drop(
    ["Electricity and gas", "Gasoline and other energy goods"], axis="columns"
)

share_avg_xe = share_avg_xe.div(share_avg_xe.sum(axis=1), axis=0)

share_avg_xe_np = share_avg_xe.values

# Number of observations - includes NaN in the first entry
dnobs = len(dp_agg)

# Sample Period for Analysis
start_sample = "1960-1"
sample = dp_agg.loc[start_sample:].index  # calvec_ismpl
notim = len(sample)  # notim
