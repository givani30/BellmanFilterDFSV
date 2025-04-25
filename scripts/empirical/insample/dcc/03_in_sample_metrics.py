import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.special import gamma
import joblib
import os
import pathlib
from numpy.linalg import cholesky, solve
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch # Import necessary statsmodels functions
import json # To save metrics as JSON

# Get the script's directory
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
# Define paths relative to the script location
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_FILE = os.path.join(DATA_DIR, "model.pkl")
EPS_FILE = os.path.join(DATA_DIR, "garch_outputs.npz")
RT_FILE = os.path.join(DATA_DIR, "Rt.npy")
DATE_INDEX_FILE = os.path.join(DATA_DIR, "date_index.txt") # To load date index

# Define output directory relative to the project root
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "empirical", "insample", "dcc")
METRICS_OUTPUT_FILE = os.path.join(OUTPUTS_DIR, "metrics_summary.json")
RESIDUALS_OUTPUT_FILE = os.path.join(OUTPUTS_DIR, "standardized_residuals.csv")

# Ensure output directory exists
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Load fitted model, eps, and Rt
try:
    dcc = joblib.load(MODEL_FILE)
    eps = np.load(EPS_FILE)['eps']
    # Load the correlation matrix directly as a 3D array
    Rt = np.load(RT_FILE)

    # Load date index
    date_index = []
    with open(DATE_INDEX_FILE, 'r') as f:
        for line in f:
            date_index.append(pd.to_datetime(line.strip()))

except Exception as e:
    print(f"Error loading data or model: {e}")
    exit()

T, N = eps.shape
k = N * 3 + 3  # 3 per univariate GARCH + (a, b, nu)

# Calculate log-likelihood using the model's parameters
# For DCC-GARCH with t-distribution, we need to calculate it manually
# since it's not stored in the model
a = dcc.a
b = dcc.b
dof = dcc.dof
D_t = dcc.D_t

# Calculate standardized residuals
eps_tilde = np.zeros((T, N))
for i in range(T):
    for j in range(N):
        eps_tilde[i, j] = dcc.rt[i, j] / D_t[i, j]

# Calculate log-likelihood
llik = 0
for t in range(T):
    # Use the correlation matrix we already calculated
    R_t = Rt[t]
    # Calculate H_t (conditional covariance matrix)
    D_t_diag = np.diag(D_t[t])
    H_t = D_t_diag @ R_t @ D_t_diag

    # Calculate log-likelihood contribution for time t
    try:
        # For t-distribution
        log_det_H = np.log(np.linalg.det(H_t))
        eps_t = eps[t]
        quad_form = eps_t.T @ np.linalg.inv(H_t) @ eps_t

        # Log-likelihood for multivariate t-distribution
        llik_t = (
            np.log(gamma((N + dof) / 2)) -
            np.log(gamma(dof / 2)) -
            (N / 2) * np.log(np.pi * (dof - 2)) -
            0.5 * log_det_H -
            ((N + dof) / 2) * np.log(1 + quad_form / (dof - 2))
        )
        llik += llik_t
    except Exception as e:
        # Skip if there's a numerical issue
        print(f"Warning: Skipping log-likelihood calculation for time {t}: {e}")
        continue

print(f"Calculated log-likelihood: {llik}")

# Calculate AIC and BIC
AIC = -2 * llik + 2 * k
BIC = -2 * llik + k * np.log(T)

print(f"Calculated AIC: {AIC}")
print(f"Calculated BIC: {BIC}")

# one-step-ahead Σ̂_{t|t-1}
# dcc.D_t is the conditional standard deviations
Sigma_t = np.zeros((T, N, N))
for t in range(T):
    # Create diagonal matrix of conditional standard deviations
    D_t_diag = np.diag(dcc.D_t[t])
    # Calculate conditional covariance matrix
    Sigma_t[t] = D_t_diag @ Rt[t] @ D_t_diag

# standardised residuals ẑ_t  = Σ̂_t^{-1/2} ε_t
z = np.empty_like(eps)
for t, (e, S) in enumerate(zip(eps, Sigma_t)):
    try:
        L = cholesky(S)
        z[t] = solve(L.T, solve(L, e))
    except np.linalg.LinAlgError:
        print(f"Cholesky decomposition failed at time step {t}. Skipping standardization for this step.")
        z[t] = np.full(N, np.nan) # Fill with NaN if decomposition fails


# Perform residual diagnostics
metrics = {
    "LogLikelihood": llik,
    "AIC": AIC,
    "BIC": BIC
}

print("Performing residual diagnostics...")

# Ljung-Box test on standardized residuals
try:
    # Perform Ljung-Box test on each column of z
    ljungbox_results = {}
    for i in range(N):
        lb_test = acorr_ljungbox(z[:, i], lags=[10], return_df=True) # Using lag 10 as an example
        ljungbox_results[f"LjungBox_pvalue_series_{i+1}"] = lb_test.iloc[0]['lb_pvalue']
    metrics["LjungBox_pvalues"] = ljungbox_results
    print("Ljung-Box test completed.")
except Exception as e:
    print(f"Error during Ljung-Box test: {e}")
    metrics["LjungBox_pvalues"] = "Error during test"


# ARCH-LM test on standardized residuals
try:
    # Perform ARCH-LM test on each column of z
    archlm_results = {}
    for i in range(N):
        # het_arch returns a tuple: (lm_statistic, p_value, f_statistic, f_p_value)
        arch_test = het_arch(z[:, i], nlags=10) # Using lag 10 as an example
        archlm_results[f"ARCHLM_pvalue_series_{i+1}"] = arch_test[1]
    metrics["ARCHLM_pvalues"] = archlm_results
    print("ARCH-LM test completed.")
except Exception as e:
    print(f"Error during ARCH-LM test: {e}")
    metrics["ARCHLM_pvalues"] = "Error during test"


# Jarque-Bera test on standardized residuals
try:
    # Perform Jarque-Bera test on each column of z
    jarquebera_results = {}
    for i in range(N):
        jb_test = st.jarque_bera(z[:, i])
        jarquebera_results[f"JarqueBera_pvalue_series_{i+1}"] = jb_test.pvalue
    metrics["JarqueBera_pvalues"] = jarquebera_results
    print("Jarque-Bera test completed.")
except Exception as e:
    print(f"Error during Jarque-Bera test: {e}")
    metrics["JarqueBera_pvalues"] = "Error during test"


# Save metrics to JSON
try:
    with open(METRICS_OUTPUT_FILE, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"In-sample metrics saved to {METRICS_OUTPUT_FILE}")
except Exception as e:
    print(f"Error saving metrics to JSON: {e}")

# Save standardized residuals to CSV
try:
    z_df = pd.DataFrame(z, index=date_index, columns=[f"z_{i+1}" for i in range(N)])
    z_df.to_csv(RESIDUALS_OUTPUT_FILE)
    print(f"Standardized residuals saved to {RESIDUALS_OUTPUT_FILE}")
except Exception as e:
    print(f"Error saving standardized residuals to CSV: {e}")
