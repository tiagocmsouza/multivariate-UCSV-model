import pandas as pd
import numpy as np

from parameters import *


### DATA

df_index = pd.read_csv("df_index.csv", index_col=[0], parse_dates=[0])
df_weights = pd.read_csv("df_weights.csv", index_col=[0], parse_dates=[0])

df_index = df_index.loc[:, pce_groups]
df_weights = df_weights.loc[:, pce_groups]

# PCE components - mom annualized
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

# Number of observations - includes NaN in the first entry
dnobs = len(dp_disagg)

# Sample Period for Analysis
sample = dp_disagg.loc[start_sample:].index  # calvec_ismpl
notim = len(sample)  # notim

dp_mat = dp_disagg.loc[sample].values
n_y = dp_mat.shape[1]  # Number of y variables

# Scale Data
dy = np.diff(dp_mat, axis=0)
sd_ddp = np.std(dy, axis=0)
sd_ddp_median = np.median(sd_ddp)
scale_y = sd_ddp_median / 5
dp_mat_n = dp_mat / scale_y
