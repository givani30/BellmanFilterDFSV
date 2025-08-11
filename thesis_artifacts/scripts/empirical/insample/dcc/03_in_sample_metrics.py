import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.special import gamma
import joblib
import os
import pathlib
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch # Import necessary statsmodels functions
import json # To save metrics as JSON

# Get the script's directory
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
# Define paths relative to the script location
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_FILE = os.path.join(DATA_DIR, "model.pkl")
RETURNS_FILE = os.path.join(SCRIPT_DIR.parent.parent, "vw_returns_final_with_date.csv")
EPS_TILDE_FILE = os.path.join(DATA_DIR, "eps_tilde.npy")
RT_FILE = os.path.join(DATA_DIR, "Rt.npy")
COV_FILE = os.path.join(DATA_DIR, "Ht.npy")
METADATA_FILE = os.path.join(DATA_DIR, "model_metadata.json")
DATE_INDEX_FILE = os.path.join(DATA_DIR, "date_index.txt") # To load date index

# Define output directory relative to the project root
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "empirical", "insample", "dcc")
METRICS_OUTPUT_FILE = os.path.join(OUTPUTS_DIR, "metrics_summary.json")
RESIDUALS_OUTPUT_FILE = os.path.join(OUTPUTS_DIR, "standardized_residuals.csv")

# Ensure output directory exists
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Load fitted model, returns, standardized residuals, correlation matrices, and metadata
try:
    # Load the DCC model
    dcc = joblib.load(MODEL_FILE)

    # Load the original returns (without scaling)
    df = pd.read_csv(RETURNS_FILE)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    returns = df.values  # Using original decimal returns

    # Load the standardized residuals from the DCC model
    eps_tilde = np.load(EPS_TILDE_FILE)

    # Load the correlation and covariance matrices
    Rt = np.load(RT_FILE)
    Sigma_t = np.load(COV_FILE)

    # Load metadata
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)

    # Use the date index from the returns dataframe
    date_index = df.index

except Exception as e:
    print(f"Error loading data or model: {e}")
    exit()

T, N = returns.shape
k = N * 3 + 3  # 3 per univariate GARCH + (a, b, nu)

# Check if log-likelihood is already available in metadata
if "log_likelihood" in metadata:
    # Use the log-likelihood from the model
    llik = metadata["log_likelihood"]
    print(f"Using log-likelihood from model: {llik}")
else:
    # Calculate log-likelihood manually
    print("Log-likelihood not found in metadata. Calculating manually...")

    # Get model parameters
    a = dcc.a
    b = dcc.b
    dof = dcc.dof
    D_t = dcc.D_t

    # Calculate log-likelihood
    llik = 0
    for t in range(T):
        # Use the covariance matrix we already calculated
        H_t = Sigma_t[t]

        # Calculate log-likelihood contribution for time t
        try:
            # Check if H_t is positive definite
            try:
                # Try Cholesky decomposition (will fail if not positive definite)
                np.linalg.cholesky(H_t)
            except np.linalg.LinAlgError:
                # If not positive definite, add a small ridge
                print(f"Warning: Covariance matrix at time {t} is not positive definite. Adding a small ridge.")
                H_t = H_t + 1e-6 * np.eye(N)

            # For t-distribution
            # Use a more numerically stable approach for the determinant
            sign, logdet = np.linalg.slogdet(H_t)
            if sign <= 0:
                # If determinant is non-positive, add a small ridge
                print(f"Warning: Non-positive determinant at time {t}. Adding a small ridge.")
                H_t = H_t + 1e-6 * np.eye(N)
                sign, logdet = np.linalg.slogdet(H_t)
                if sign <= 0:
                    # If still non-positive, skip this time point
                    print(f"Warning: Still non-positive determinant at time {t}. Skipping.")
                    continue

            log_det_H = logdet

            # Use the original returns for the quadratic form
            r_t = returns[t]

            # Use a more numerically stable approach for the quadratic form
            try:
                # Try using Cholesky decomposition for the quadratic form
                L = np.linalg.cholesky(H_t)
                y = np.linalg.solve(L, r_t)
                quad_form = np.sum(y**2)
            except np.linalg.LinAlgError:
                # If Cholesky fails, try direct inversion
                try:
                    H_inv = np.linalg.inv(H_t)
                    quad_form = r_t @ H_inv @ r_t
                except np.linalg.LinAlgError:
                    # If inversion fails, skip this time point
                    print(f"Warning: Failed to compute quadratic form at time {t}. Skipping.")
                    continue

            # Log-likelihood for multivariate t-distribution
            try:
                llik_t = (
                    np.log(gamma((N + dof) / 2)) -
                    np.log(gamma(dof / 2)) -
                    (N / 2) * np.log(np.pi * (dof - 2)) -
                    0.5 * log_det_H -
                    ((N + dof) / 2) * np.log(1 + quad_form / (dof - 2))
                )

                # Check for invalid values
                if np.isnan(llik_t) or np.isinf(llik_t):
                    print(f"Warning: Invalid log-likelihood value at time {t}: {llik_t}. Skipping.")
                    continue

                llik += llik_t
            except Exception as e:
                print(f"Warning: Error calculating log-likelihood at time {t}: {e}. Skipping.")
                continue
        except Exception as e:
            # Skip if there's a numerical issue
            print(f"Warning: Skipping log-likelihood calculation for time {t}: {e}")
            continue

    # Check for invalid log-likelihood
    if np.isnan(llik) or np.isinf(llik):
        print(f"Warning: Invalid final log-likelihood: {llik}. Setting to a large negative value.")
        llik = -1e10  # Use a large negative value instead of infinity

    print(f"Manually calculated log-likelihood: {llik}")

# Calculate AIC and BIC
AIC = -2 * llik + 2 * k
BIC = -2 * llik + k * np.log(T)

print(f"Calculated AIC: {AIC}")
print(f"Calculated BIC: {BIC}")

# Update metadata with log-likelihood
metadata["log_likelihood"] = float(llik)
metadata["AIC"] = float(AIC)
metadata["BIC"] = float(BIC)

# Save updated metadata
with open(METADATA_FILE, 'w') as f:
    json.dump(metadata, f, indent=4)
print("Updated metadata with log-likelihood information")

# Use the standardized residuals from the DCC model for diagnostics
z = eps_tilde

# Verify that the standardized residuals have reasonable values
z_mean = np.nanmean(z)
z_std = np.nanstd(z)
print(f"Standardized residuals statistics: mean={z_mean:.4f}, std={z_std:.4f}")

# If the values are still extreme, we might need to rescale them
if np.abs(z_mean) > 10 or z_std > 10:
    print("WARNING: Standardized residuals have extreme values. Rescaling...")
    # Rescale to have mean 0 and std 1
    z = (z - np.nanmean(z, axis=0)) / np.nanstd(z, axis=0)
    print(f"After rescaling: mean={np.nanmean(z):.4f}, std={np.nanstd(z):.4f}")


# Perform residual diagnostics
metrics = {
    "model_type": metadata["model_type"],
    "distribution": metadata["distribution"],
    "num_params": metadata["num_params"],
    "estimation_time": metadata["estimation_time"],
    "convergence_status": metadata["convergence_status"],
    "sample_size": metadata["sample_size"],
    "num_series": metadata["num_series"],
    "log_likelihood": metadata["log_likelihood"],
    "AIC": metadata["AIC"],
    "BIC": metadata["BIC"],
    "parameters": metadata["parameters"]
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
