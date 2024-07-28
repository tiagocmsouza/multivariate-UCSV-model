import pandas as pd
import numpy as np
from numba import int64, float64
from numba.experimental import jitclass

from parameters import pce_groups, nper, start_sample


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
    .to_period("M")
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

spec = [
    ("big", int64),
    ("data", float64[:, :]),
    ("y", float64[:, :]),
    ("scale_y", float64),
    ("nobs", int64),
    ("ny", int64),
    ("prior_var_alpha", float64[:, :]),
    ("alpha_tau", float64[:, :]),
    ("alpha_eps", float64[:, :]),
    ("dalpha", float64[:, :]),
    ("eps_common", float64[:]),
    ("tau_common", float64[:]),
    ("tau_unique", float64[:, :]),
    ("dtau_common", float64[:]),
    ("dtau_unique", float64[:, :]),
    ("tau_f_common", float64[:]),
    ("tau_f_unique", float64[:, :]),
    ("sigma_eps_common", float64[:]),
    ("scale_eps_common", float64[:]),
    ("sigma_eps_common_scl", float64[:]),
    ("sigma_eps_unique", float64[:, :]),
    ("scale_eps_unique", float64[:, :]),
    ("sigma_eps_unique_scl", float64[:, :]),
    ("sigma_dtau_common", float64[:]),
    ("sigma_dtau_unique", float64[:, :]),
    ("sigma_dalpha", float64[:]),
]


@jitclass(spec)
class mcmcTrend:

    def __init__(self, data):

        self.big = 1e6

        self.data = data
        self.y, self.scale_y = self._prepare_y()
        self.nobs, self.ny = self.y.shape

        self.prior_var_alpha = self._prepare_prior_var_alpha()

        self.alpha_eps = np.ones((self.nobs, self.ny))
        self.alpha_tau = np.ones((self.nobs, self.ny))
        self.dalpha = np.ones((self.nobs, 2 * self.ny))

        self.eps_common = np.zeros(self.nobs)
        self.tau_common = np.zeros(self.nobs)
        self.tau_unique = np.zeros((self.nobs, self.ny))
        self.dtau_common = np.zeros(self.nobs)
        self.dtau_unique = np.zeros((self.nobs, self.ny))
        self.tau_f_common = np.zeros(self.nobs)
        self.tau_f_unique = np.zeros((self.nobs, self.ny))

        self.sigma_dalpha = self._prepare_sigma_dalpha()

        self.sigma_eps_common = np.random.random(self.nobs)
        self.scale_eps_common = np.ones(self.nobs)
        self.sigma_eps_common_scl = np.multiply(
            self.sigma_eps_common, self.scale_eps_common
        )

        self.sigma_eps_unique = np.random.random((self.nobs, self.ny))
        self.scale_eps_unique = np.ones((self.nobs, self.ny))
        self.sigma_eps_unique_scl = np.multiply(
            self.sigma_eps_unique, self.scale_eps_unique
        )

        self.sigma_dtau_common = np.random.random(self.nobs)
        self.sigma_dtau_unique = np.random.random((self.nobs, self.ny))

    def _prepare_y(self):

        dy = self.data[1:, :] - self.data[:-1, :]
        n = dy.shape[1]
        sd_ddp = np.empty(n)
        for i in np.arange(n):
            sd_ddp[i] = np.std(dy[:, i])
        sd_ddp_median = np.median(sd_ddp)
        scale_y = sd_ddp_median / 5
        y = self.data / scale_y

        return y, scale_y

    def _prepare_sigma_dalpha(self):

        nu_prior_alpha = 0.1 * self.nobs
        s2_prior_alpha = ((0.25 / np.sqrt(self.nobs)) ** 2) / (self.scale_y**2)

        a = nu_prior_alpha / 2
        ssr = nu_prior_alpha * s2_prior_alpha
        b = 2 / ssr
        var_dalpha = 1 / np.random.gamma(
            shape=a, scale=1 / b, size=2 * self.ny
        )  # ny epsilons and ny alphas
        sigma_dalpha = np.sqrt(var_dalpha)

        return sigma_dalpha

    def _prepare_prior_var_alpha(self):

        omega_tau = 10 / self.scale_y
        omega_eps = 10 / self.scale_y
        sigma_tau = 0.4 / self.scale_y
        sigma_eps = 0.4 / self.scale_y

        var_alpha_tau = ((omega_tau**2) * np.ones((self.ny, self.ny))) + (
            (sigma_tau**2) * np.eye(self.ny)
        )
        var_alpha_eps = ((omega_eps**2) * np.ones((self.ny, self.ny))) + (
            (sigma_eps**2) * np.eye(self.ny)
        )

        prior_var_alpha = np.concatenate(
            (
                np.concatenate((var_alpha_tau, np.zeros((self.ny, self.ny)))),
                np.concatenate((np.zeros((self.ny, self.ny)), var_alpha_eps)),
            ),
            axis=1,
        )

        return prior_var_alpha

    def _kalman_filter(self, y, X1, P1, H, F, R, Q):
        """
        Kalman Filter Procedure - Hamilton Notation

        y(t) = H' x(t) + w(t)
        x(t) = F x(t-1) + v(t)

        var(w(t)) = R
        var(v(t)) = Q

        X1 = x(t-1|t-1)
        P1 = p(t-1|t-1)
        """

        nstate = X1.shape[0]  # Number of states (ns)
        # State Estimation: Time Update
        X2 = F @ X1  # State Prediction X(t+1|t) ; F(ns x ns) X1(ns x 1)
        Z = np.transpose(H) @ X2  # Measurement Prediction ; H(ns x n_y) X2(ns x 1)
        # State Estimation: Measurement Update
        e = y - Z  # Measurement Residual ; y(ns x 1)
        # State Covariance Estimation
        P2 = (
            F @ P1 @ np.transpose(F) + Q
        )  # State Prediction Covariance ; F(ns x ns) P1(ns x ns) Q(ns x ns)
        ht = (
            np.transpose(H) @ P2 @ H + R
        )  # Measurement Prediction Covariace ; H(ns x ny) P2(ns x ns) R(ny x ny) ; Forecast error MSE ; R comes from 'sigma_eps_unique_scl', the scale parameter

        try:
            # hti = np.linalg.pinv(ht, rcond=0, hermitian=True)
            hti = np.linalg.pinv(ht, rcond=0)
        except:
            # hti = np.linalg.pinv(ht, rcond=1e-12, hermitian=True)
            hti = np.linalg.pinv(ht, rcond=1e-12)

        K = P2 @ H @ hti  # Kalman (Filter) Gain
        # State Estimation: Measurement Update
        X1 = X2 + K @ e  # Updated State Estimate
        # State Covariance Estimation
        P1 = (np.eye(nstate) - K @ np.transpose(H)) @ P2  # Update State Covariance
        P1 = 0.5 * (P1 + np.transpose(P1))  # Guarantee that P1 is symetrical

        return X1, P1, X2, P2

    def mdraw_eps_tau(self):

        # Set up State Vector:
        # --- State Vector
        #       (1) esp(t)
        #       (2) tau(t)
        #       (3) tau_u(t)
        ns = (
            2 + self.ny
        )  # Size of state (eps + tau + tau_u, where tau_u is for each sector)
        Q = np.zeros((ns, ns))
        F = np.eye(ns)
        F[0][0] = 0
        H = np.zeros((ns, self.ny))
        H[2:, :] = np.eye(self.ny)

        # Set up KF to run
        # Initial conditions
        X1_init = np.zeros(ns)
        P1_init = np.zeros((ns, ns))
        P1_init[2:, 2:] = self.big * np.eye(
            self.ny
        )  # Vague prior for tau_unique initial values

        X1 = X1_init.copy()
        P1 = P1_init.copy()
        X1t = np.zeros(
            (ns, self.nobs + 1)
        )  # +1 observations because of the initial value
        P1t = np.zeros((ns, ns, self.nobs + 1))
        X2t = np.zeros((ns, self.nobs + 1))
        P2t = np.zeros((ns, ns, self.nobs + 1))
        X1t[:, 0] = X1
        P1t[:, :, 0] = P1
        X_Draw_Filt = np.empty(
            (self.nobs + 1, ns)
        )  # Draw from filtered; these are marginal draws, not joint (same as ucsv)

        # KALMAN FILTER
        for t in np.arange(self.nobs):

            y_t = self.y[t, :]
            mask = ~np.isnan(y_t)  # Control for missing data
            y_t = y_t[mask]

            H_t = H.copy()
            H_t[0, :] = self.alpha_eps[t, :]
            H_t[1, :] = self.alpha_tau[t, :]
            H_t = H_t[:, mask]

            R_t = np.diag(self.sigma_eps_unique[t, :] ** 2)
            R_t = R_t[mask, :][:, mask]

            Q_t = np.empty(ns)
            Q_t[0] = self.sigma_eps_common[t] ** 2
            Q_t[1] = self.sigma_dtau_common[t] ** 2
            Q_t[2:] = self.sigma_dtau_unique[t, :] ** 2
            Q_t = np.diag(Q_t)

            X1, P1, X2, P2 = self._kalman_filter(y_t, X1, P1, H_t, F, R_t, Q_t)

            X1t[:, t + 1] = X1
            P1t[:, :, t + 1] = P1
            X2t[:, t + 1] = X2
            P2t[:, :, t + 1] = P2
            chol_P1 = np.linalg.cholesky(P1)
            X = X1 + chol_P1 @ np.random.standard_normal(size=ns)
            X_Draw_Filt[t + 1, :] = X

        # KALMAN SMOOTHER
        # Draw From State
        X_Draw = np.empty((self.nobs + 1, ns))

        # Initial Draw
        P3 = P1.copy()
        X3 = X1.copy()
        chol_P3 = np.linalg.cholesky(P3)
        X = X3 + chol_P3 @ np.random.standard_normal(size=ns)
        X_Draw[self.nobs, :] = X

        for t in np.arange(self.nobs)[::-1]:  # Working backwards

            X1 = X1t[:, t]
            X2 = X2t[:, t + 1]
            P1 = np.ascontiguousarray(P1t[:, :, t])
            P2 = np.ascontiguousarray(P2t[:, :, t + 1])

            try:
                P2i = np.linalg.pinv(P2, rcond=0)
            except:
                P2i = np.linalg.pinv(P2, rcond=1e-12)

            AS = P2i @ F @ P1

            P3 = P1 - np.transpose(AS) @ F @ P1

            P3 = 0.5 * (P3 + np.transpose(P3))
            X3 = X1 + np.transpose(AS) @ (X - X2)
            X = X3.copy()

            if t > 1:
                chol_P3 = np.linalg.cholesky(P3)
                X = X + chol_P3 @ np.random.standard_normal(size=ns)
            else:  # In the first period, eps and tau common have no pertubation
                P3 = P3[2:, 2:]
                chol_P3 = np.linalg.cholesky(P3)
                X[2:] = X[2:] + chol_P3 @ np.random.standard_normal(size=ns - 2)

            X_Draw[t, :] = X

        self.eps_common = X_Draw[1:, 0]
        self.tau_common = X_Draw[1:, 1]
        self.tau_unique = X_Draw[1:, 2:]
        self.dtau_common = X_Draw[1:, 1] - X_Draw[:-1, 1]
        self.dtau_unique = X_Draw[1:, 2:] - X_Draw[:-1, 2:]
        self.tau_f_common = X_Draw_Filt[1:, 1]
        self.tau_f_unique = X_Draw_Filt[1:, 2:]

    def draw_alpha_tvp(self):

        y = self.y - self.tau_unique  # Eliminates tau_unique from y

        # Brute force calculation (comment from the original code)
        ns = (
            2 * self.ny
        )  # First n_y of state are alpha_eps; second n_y elements are of alpha_tau
        Q = np.diag(self.sigma_dalpha**2)
        F = np.eye(ns)
        H = np.zeros((ns, self.ny))

        # Set up KF to run
        # Initial conditions
        X1_init = np.zeros(ns)
        P1_init = self.prior_var_alpha

        X1 = X1_init.copy()
        P1 = P1_init.copy()
        X1t = np.zeros(
            (ns, self.nobs + 1)
        )  # +1 observations because of the initial value
        P1t = np.zeros((ns, ns, self.nobs + 1))
        X2t = np.zeros((ns, self.nobs + 1))
        P2t = np.zeros((ns, ns, self.nobs + 1))
        X1t[:, 0] = X1
        P1t[:, :, 0] = P1

        # KALMAN FILTER
        for t in np.arange(self.nobs):

            y_t = y[t, :]
            mask = ~np.isnan(y_t)  # Control for missing data
            y_t = y_t[mask]

            H_t = H.copy()
            H_t[: self.ny, :] = self.eps_common[t] * np.eye(self.ny)
            H_t[self.ny :, :] = self.tau_common[t] * np.eye(self.ny)
            H_t = H_t[:, mask]

            R_t = np.diag(self.sigma_eps_unique[t, :] ** 2)
            R_t = R_t[mask, :][:, mask]

            Q_t = Q.copy()

            X1, P1, X2, P2 = self._kalman_filter(y_t, X1, P1, H_t, F, R_t, Q_t)

            X1t[:, t + 1] = X1
            P1t[:, :, t + 1] = P1
            X2t[:, t + 1] = X2
            P2t[:, :, t + 1] = P2

        # KALMAN SMOOTHER
        # Draw From State
        X_Draw = np.empty((self.nobs + 1, ns))

        # Initial Draw
        P3 = P1.copy()
        X3 = X1.copy()
        chol_P3 = np.linalg.cholesky(P3)
        X = X3 + chol_P3 @ np.random.standard_normal(size=ns)
        X_Draw[self.nobs, :] = X

        for t in np.arange(self.nobs)[::-1]:  # Working backwards

            X1 = X1t[:, t]
            X2 = X2t[:, t + 1]
            P1 = np.ascontiguousarray(P1t[:, :, t])
            P2 = np.ascontiguousarray(P2t[:, :, t + 1])

            try:
                P2i = np.linalg.pinv(P2, rcond=0)
            except:
                P2i = np.linalg.pinv(P2, rcond=1e-12)

            AS = P2i @ F @ P1
            P3 = P1 - np.transpose(AS) @ F @ P1
            P3 = 0.5 * (P3 + np.transpose(P3))
            X3 = X1 + np.transpose(AS) @ (X - X2)
            chol_P3 = np.linalg.cholesky(P3)
            X = X3 + chol_P3 @ np.random.standard_normal(size=ns)
            X_Draw[t, :] = X

        alpha_eps = X_Draw[1:, : self.ny]
        alpha_tau = X_Draw[1:, self.ny :]
        dalpha_eps = X_Draw[1:, : self.ny] - X_Draw[:-1, : self.ny]
        dalpha_tau = X_Draw[1:, self.ny :] - X_Draw[:-1, self.ny :]

        self.alpha_eps = alpha_eps
        self.alpha_tau = alpha_tau
        self.dalpha = np.concatenate((dalpha_eps, dalpha_tau), axis=1)


test = mcmcTrend(dp_mat)

test.dalpha
test.mdraw_eps_tau()
test.draw_alpha_tvp()
test.dalpha
