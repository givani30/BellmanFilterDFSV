# %%
import polars as pl
import numpy as np
import os
import time
import pandas as pd
import jax
import jax.numpy as jnp
import cloudpickle
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy.stats import kurtosis, chi2, norm
from scipy.special import digamma
from scipy.optimize import brentq
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.multivariate import test_cov
import warnings
from rich import print

# Import custom multivariate Portmanteau tests
from multivariate_portmanteau import multivariate_portmanteau_tests

from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter

# Enable double precision
jax.config.update("jax_enable_x64", True)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.title_fontsize': 11,
    'figure.dpi': 300,
    'lines.markersize': 6,
    'lines.linewidth': 1.5,
})
df =pl.read_csv("scripts/empirical/vw_returns_final.csv")
df.head()
df_with_date=pl.read_csv("scripts/empirical/vw_returns_final_with_date.csv")
# os.getcwd()
returns=df.to_jax()
N=95
K=5
T=returns.shape[0]

# %%
#Load pickle file
with open('scripts/empirical/insample/bfpf/bif_full_result_20250425_144625.pkl', 'rb') as f:
    result_bif = cloudpickle.load(f)
bif_filter=DFSVBellmanInformationFilter(N,K)
# bif_filter=DFSVParticleFilter(N,K,num_particles=10000)
bif_params=result_bif.final_params
print(round(bif_params.Phi_f,3))
filtered_states_bf,filtered_infos_bf,log_likelihood_bf=bif_filter.filter(bif_params,returns)

# %%
filtered_covs_bf=np.array(bif_filter.get_filtered_covariances())
filtered_factors_bf=filtered_states_bf[:,:K]
predicted_states_bf=np.array(bif_filter.get_predicted_states())
predicted_factors_bf=predicted_states_bf[:,:K]
predicted_covs_bf=np.array(bif_filter.get_predicted_covariances())

# %% [markdown]
# #Calculate standardized residuals z_t

# %%
returns_arr=np.array(returns)
lambda_hat_bif=np.array(bif_params.lambda_r)
sigma_eps_hat_diag_bif=np.array(bif_params.sigma2)
sigma_eps_hat_mat_bif=np.diag(sigma_eps_hat_diag_bif)

# Extract factor components from filtered states and covariances
filtered_factor_covs_bif=filtered_covs_bf[:,:K,:K]
predicted_factor_covs_bif=predicted_covs_bf[:,:K,:K]

# Arrays to store results
standardized_residuals_bif = np.full((T,N),np.nan)
conditional_covariance_H_bif = np.full((T, N, N), np.nan) # Store Sigma_t|t
predicted_covariance_H_bif = np.full((T, N, N), np.nan)   # Store Sigma_t|t-1 (for completeness)

# Calculate both filtered and predicted covariances and standardized residuals
print("Calculating standardized residuals using FILTERED states (for diagnostics)...")
for t in range(T):
    # Get filtered factor state and covariance at time t
    f_t_filt = filtered_factors_bf[t, :]        # Shape (K,) - Filtered factor
    P_f_t_filt = filtered_factor_covs_bif[t, :, :]  # Shape (K, K) - Filtered factor covariance

    # Calculate conditional observation mean and covariance Sigma_{t|t}
    mu_t_filt = lambda_hat_bif @ f_t_filt  # Shape (N,) - Filtered mean
    Sigma_t_filt = (lambda_hat_bif @ P_f_t_filt @ lambda_hat_bif.T
                   + sigma_eps_hat_mat_bif)  # Shape (N, N) - Filtered covariance
    Sigma_t_filt = (Sigma_t_filt + Sigma_t_filt.T) / 2  # Ensure symmetry

    # Also calculate predicted covariance for reference
    if t < T-1:  # Only for t < T-1 since we don't have prediction for last time point
        f_t_pred = predicted_factors_bf[t, :]        # Shape (K,)
        P_f_t_pred = predicted_factor_covs_bif[t, :, :]  # Shape (K, K)

        mu_t_pred = lambda_hat_bif @ f_t_pred
        Sigma_t_pred = (lambda_hat_bif @ P_f_t_pred @ lambda_hat_bif.T
                       + sigma_eps_hat_mat_bif)
        Sigma_t_pred = (Sigma_t_pred + Sigma_t_pred.T) / 2  # Ensure symmetry
        predicted_covariance_H_bif[t, :, :] = Sigma_t_pred

    # Store filtered covariance
    conditional_covariance_H_bif[t, :, :] = Sigma_t_filt

    try:
        # Cholesky decomposition: Sigma_{t|t} = L * L'
        L_t_filt = np.linalg.cholesky(Sigma_t_filt + 1e-7 * np.eye(N))  # Add jitter for numerical stability

        # Raw residual e_t = r'_t - mu_t|t (using filtered mean)
        e_t = returns_arr[t, :] - mu_t_filt  # Shape (N,)

        # Standardized residual: z_t = L^{-1} * e_t
        z_t = np.linalg.solve(L_t_filt, e_t)
        standardized_residuals_bif[t, :] = z_t
    except np.linalg.LinAlgError:
        print(f"Warning: Cholesky failed for Sigma_t|t at t={t} for BIF.")
        # standardized_residuals_bif[t, :] remains NaN

# Convert to DataFrame (similar to 02_factor_cv_fit.py)
standardized_residuals_bif_df = pd.DataFrame(
    standardized_residuals_bif,
)
# Define burn-in for analysis if needed (e.g., analysis_burn_in = 50)
analysis_burn_in = 0
standardized_residuals_bif_post_burn = standardized_residuals_bif_df.iloc[analysis_burn_in:]

# %%
standardized_residuals_bif_df

# %%
# --- Univariate Diagnostic Tests ---
#  Also assume N (number of series) and T_eff (effective number of observations post-burn)
if 'standardized_residuals_bif_post_burn' in locals():
    residuals_df = standardized_residuals_bif_post_burn
    N = residuals_df.shape[1]
    T_eff = residuals_df.shape[0]
    print(f"Using BIF standardized residuals with shape: {residuals_df.shape}")
else:
    print("Error: standardized_residuals_bif_post_burn DataFrame not found.")
    # As a placeholder for testing the code structure:
    T_eff, N = 500, 95 # Example dimensions
    print(f"Creating placeholder DataFrame with shape ({T_eff}, {N})")
    residuals_df = pd.DataFrame(np.random.randn(T_eff, N), columns=[f'Asset_{i+1}' for i in range(N)])

# --- Dictionary to store test results ---
diagnostic_results_bif = {
    "ljung_box_squared": {},
    "arch_lm": {},
    "jarque_bera": {}
}

# Significance level for counting rejections/passes
alpha = 0.05

# %% [markdown]
# Ljung-Box

# %%
print("\n--- Running Ljung-Box Test (Squared Residuals) ---")
lags_lb = [5, 10, 15, 20] # Lags to test (similar to DFM script) [cite: 2]

for lag in lags_lb:
    lb_results = {
        "lag": lag,
        "pass_count": 0,
        "reject_count": 0,
        "error_count": 0,
        "total": N,
        "pass_rate": 0.0
    }
    print(f"Testing with lag = {lag}...")
    for i in range(N):
        col_name = residuals_df.columns[i]
        series_sq = residuals_df.iloc[:, i]**2
        # Drop NaN values which can cause issues, though standardized residuals shouldn't have them
        series_sq = series_sq.dropna()

        if len(series_sq) <= lag:
             print(f"  Skipping Series {i+1} ({col_name}): Not enough observations ({len(series_sq)}) for lag {lag}")
             lb_results["error_count"] += 1
             continue

        try:
            # Run the Ljung-Box test
            lb_test = acorr_ljungbox(series_sq, lags=[lag], return_df=True)
            p_value = lb_test.iloc[0, 1] # Get p-value for the specified lag

            # Check if null hypothesis is rejected (p-value <= alpha means reject H0 -> autocorrelation exists)
            if p_value <= alpha:
                lb_results["reject_count"] += 1
            else:
                # Fail to reject H0 -> No significant autocorrelation detected
                lb_results["pass_count"] += 1
        except Exception:
            # Handle potential errors during the test (e.g., constant series)
            # print(f"  Error testing Series {i+1} ({col_name}): {e}") # Optional: more verbose error
            lb_results["error_count"] += 1

    # Calculate pass rate (proportion of series where H0 is NOT rejected)
    if N > 0:
         # Calculate rate based on successfully tested series
        tested_count = N - lb_results["error_count"]
        if tested_count > 0:
            lb_results["pass_rate"] = lb_results["pass_count"] / tested_count
        else:
            lb_results["pass_rate"] = np.nan # Or 0.0 if preferred

    diagnostic_results_bif["ljung_box_squared"][f"lag_{lag}"] = lb_results
    print(f"Lag {lag}: Pass Rate = {lb_results['pass_rate']:.3f} ({lb_results['pass_count']}/{tested_count})")

print("Ljung-Box Test Complete.")

# %%
print("\n--- Running ARCH-LM Test ---")
lags_arch = [5, 10] # Lags to test

for lag in lags_arch:
    arch_results = {
        "lag": lag,
        "pass_count": 0,
        "reject_count": 0,
        "error_count": 0,
        "total": N,
        "pass_rate": 0.0
    }
    print(f"Testing with lag = {lag}...")
    for i in range(N):
        col_name = residuals_df.columns[i]
        series = residuals_df.iloc[:, i].dropna()

        # het_arch requires length > nlags
        if len(series) <= lag:
             print(f"  Skipping Series {i+1} ({col_name}): Not enough observations ({len(series)}) for lag {lag}")
             arch_results["error_count"] += 1
             continue

        try:
            # Run the ARCH-LM test
            # Note: het_arch returns (lm_stat, lm_p_value, f_stat, f_p_value)
            lm_stat, p_value, f_stat, f_p_value = het_arch(series, nlags=lag)

            # Check if null hypothesis is rejected (p-value <= alpha means reject H0 -> ARCH effects exist)
            if p_value <= alpha:
                arch_results["reject_count"] += 1
            else:
                # Fail to reject H0 -> No significant ARCH effects detected
                arch_results["pass_count"] += 1
        except Exception:
            # Handle potential errors (e.g., perfect multicollinearity if lags are too high for T)
            # print(f"  Error testing Series {i+1} ({col_name}): {e}")
            arch_results["error_count"] += 1

    # Calculate pass rate
    if N > 0:
        tested_count = N - arch_results["error_count"]
        if tested_count > 0:
            arch_results["pass_rate"] = arch_results["pass_count"] / tested_count
        else:
             arch_results["pass_rate"] = np.nan

    diagnostic_results_bif["arch_lm"][f"lag_{lag}"] = arch_results
    print(f"Lag {lag}: Pass Rate = {arch_results['pass_rate']:.3f} ({arch_results['pass_count']}/{tested_count})")

print("ARCH-LM Test Complete.")

# %%
print("\n--- Running Jarque-Bera Normality Test ---")
jb_results = {
    "pass_count": 0,
    "reject_count": 0,
    "error_count": 0,
    "total": N,
    "pass_rate": 0.0
}

for i in range(N):
    col_name = residuals_df.columns[i]
    series = residuals_df.iloc[:, i].dropna()

    # Jarque-Bera test requires at least 2 observations
    if len(series) < 2:
         print(f"  Skipping Series {i+1} ({col_name}): Not enough observations ({len(series)})")
         jb_results["error_count"] += 1
         continue

    try:
        # Run the Jarque-Bera test
        # Returns (jb_statistic, jb_p_value, skewness, kurtosis)
        jb_stat, p_value, skew, kurt = jarque_bera(series)

        # Check if null hypothesis is rejected (p-value <= alpha means reject H0 -> not normal)
        if p_value <= alpha:
            jb_results["reject_count"] += 1
        else:
            # Fail to reject H0 -> Distribution is potentially normal
            jb_results["pass_count"] += 1
    except Exception:
        # Handle potential errors (e.g., zero variance series)
        # print(f"  Error testing Series {i+1} ({col_name}): {e}")
        jb_results["error_count"] += 1

# Calculate pass rate
if N > 0:
    tested_count = N - jb_results["error_count"]
    if tested_count > 0:
         jb_results["pass_rate"] = jb_results["pass_count"] / tested_count
    else:
        jb_results["pass_rate"] = np.nan

diagnostic_results_bif["jarque_bera"] = jb_results
print(f"Jarque-Bera: Pass Rate = {jb_results['pass_rate']:.3f} ({jb_results['pass_count']}/{tested_count})")

print("Jarque-Bera Test Complete.")

# --- Display aggregated results ---
print("\n--- BIF Diagnostic Test Summary ---")
print(json.dumps(diagnostic_results_bif, indent=2))

# %%
print((np.abs(predicted_factors_bf[2,:] - filtered_factors_bf[1,:])))




# %%
# Extract the date column (ensure it's in pandas datetime format)
# If df_with_date is polars:
time_column_pd = df_with_date.select(pl.col("Date").cast(pl.Date)).to_pandas()['Date']
# If df_with_date is already pandas:
# time_column_pd = df_with_date['Date']

# Ensure output directories exist for saving results
main_output_dir = 'outputs/empirical/insample/'
bif_output_dir = os.path.join(main_output_dir, 'bif')
data_dir = os.path.join(bif_output_dir, 'data')
os.makedirs(main_output_dir, exist_ok=True)
os.makedirs(bif_output_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)


# Extract factors (first K columns) and log-vols (next K columns)
filtered_factors = filtered_states_bf[:, :K]
filtered_log_vols = filtered_states_bf[:, K:]

# Create meaningful column names
factor_cols = [f'Factor_{i+1}' for i in range(K)]
logvol_cols = [f'LogVol_{i+1}' for i in range(K)]

# Create pandas DataFrame
plot_df = pd.DataFrame(filtered_factors, columns=factor_cols, index=time_column_pd)
for i, col_name in enumerate(logvol_cols):
    plot_df[col_name] = filtered_log_vols[:, i]

print("DataFrame head for plotting:")
print(plot_df.head())

# %%
# --- Plotting ---
num_states = K # Number of factors OR log-vols to plot

# Create figures directory if it doesn't exist
figures_dir = os.path.join(bif_output_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# Create figure with 2 rows, 1 column
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True) # Share x-axis

# Plot Factors on the first subplot (axes[0])
axes[0].set_title(f'Estimated Latent Factors (K={K})')
for i in range(num_states):
    col_name = factor_cols[i]
    axes[0].plot(plot_df.index, plot_df[col_name], label=col_name, linewidth=1)
axes[0].set_ylabel('Factor Value')
axes[0].legend(loc='upper right')
axes[0].grid(True, linestyle='--', alpha=0.6)

# Plot Log-Volatilities on the second subplot (axes[1])
axes[1].set_title(f'Estimated Log-Volatilities (K={K})')
for i in range(num_states):
    col_name = logvol_cols[i]
    axes[1].plot(plot_df.index, plot_df[col_name], label=col_name, linewidth=1)
axes[1].set_ylabel('Log-Volatility (h_t)')
axes[1].legend(loc='upper right')
axes[1].grid(True, linestyle='--', alpha=0.6)

# Common X-axis label
axes[1].set_xlabel('Date')

# Improve layout and display
plt.tight_layout()

# Save the figure to the figures directory
plt.savefig(os.path.join(figures_dir, 'filtered_states.png'), dpi=300)
print(f"Filtered states plot saved to {os.path.join(figures_dir, 'filtered_states.png')}")

# Display the figure
plt.show()

# Create individual plots for each factor and log-volatility
print("\nCreating individual factor and log-volatility plots...")

# Individual factor plots
for i in range(K):
    plt.figure(figsize=(12, 6))
    col_name = factor_cols[i]
    plt.plot(plot_df.index, plot_df[col_name], linewidth=1.5)
    plt.title(f'Estimated Latent Factor {i+1}')
    plt.xlabel('Date')
    plt.ylabel('Factor Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the figure
    factor_file = os.path.join(figures_dir, f'factor_{i+1}.png')
    plt.savefig(factor_file, dpi=300)
    print(f"Factor {i+1} plot saved to {factor_file}")
    plt.close()

# Individual log-volatility plots
for i in range(K):
    plt.figure(figsize=(12, 6))
    col_name = logvol_cols[i]
    plt.plot(plot_df.index, plot_df[col_name], linewidth=1.5)
    plt.title(f'Estimated Log-Volatility {i+1}')
    plt.xlabel('Date')
    plt.ylabel('Log-Volatility Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the figure
    logvol_file = os.path.join(figures_dir, f'logvol_{i+1}.png')
    plt.savefig(logvol_file, dpi=300)
    print(f"Log-Volatility {i+1} plot saved to {logvol_file}")
    plt.close()

# %%
# Create heatmaps of model parameters
print("\nCreating parameter heatmaps...")

# Create figures directory if it doesn't exist
figures_dir = os.path.join(bif_output_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# 1. Factor loadings (Lambda)
lambda_r = np.array(bif_params.lambda_r)
plt.figure(figsize=(16, 12))
sns.heatmap(lambda_r, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap of Factor Loadings (Lambda)')
plt.xlabel('Factors')
plt.ylabel('Assets')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'lambda_heatmap_interactive.png'), dpi=300)
print(f"Lambda heatmap saved to {os.path.join(figures_dir, 'lambda_heatmap_interactive.png')}")
plt.show()

# 2. Factor transition matrix (Phi_f)
phi_f = np.array(bif_params.Phi_f)
plt.figure(figsize=(10, 8))
sns.heatmap(phi_f, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5)
plt.title('Heatmap of Factor Transition Matrix (Phi_f)')
plt.xlabel('Factor (t-1)')
plt.ylabel('Factor (t)')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'phi_f_heatmap_interactive.png'), dpi=300)
print(f"Phi_f heatmap saved to {os.path.join(figures_dir, 'phi_f_heatmap_interactive.png')}")
plt.show()

# 3. Log-volatility transition matrix (Phi_h)
phi_h = np.array(bif_params.Phi_h)
plt.figure(figsize=(10, 8))
sns.heatmap(phi_h, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5)
plt.title('Heatmap of Log-Volatility Transition Matrix (Phi_h)')
plt.xlabel('Log-Vol (t-1)')
plt.ylabel('Log-Vol (t)')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'phi_h_heatmap_interactive.png'), dpi=300)
print(f"Phi_h heatmap saved to {os.path.join(figures_dir, 'phi_h_heatmap_interactive.png')}")
plt.show()

# %%

# %%
# --- Extended Metrics: Multivariate Tests, Generalized Variance, Average Correlation ---
print("\n--- Running Extended Metrics and Multivariate Tests ---")

# Create a dictionary to store extended metrics
extended_metrics_bif = {}

# 1. Multivariate Tests
print("\n1. Multivariate Tests")
try:
    # Prepare residuals matrix for multivariate tests (T_eff x N)
    residuals_matrix = standardized_residuals_bif_post_burn.values

    # Calculate sample covariance matrix of residuals
    residual_cov = np.cov(residuals_matrix, rowvar=False)

    # Calculate Mardia's multivariate skewness and kurtosis
    # Reference: Mardia, K. V. (1970), Measures of multivariate skewness and kurtosis with applications
    n_obs = residuals_matrix.shape[0]

    # Center the data
    centered_data = residuals_matrix - residuals_matrix.mean(axis=0)

    # Calculate Mahalanobis distances
    try:
        # Use robust approach to compute inverse of covariance matrix
        cov_inv = np.linalg.inv(residual_cov + 1e-8 * np.eye(N))

        # Calculate squared Mahalanobis distances
        mahal_dist_sq = np.zeros(n_obs)
        for i in range(n_obs):
            x_i = centered_data[i, :]
            mahal_dist_sq[i] = x_i @ cov_inv @ x_i

        # Mardia's multivariate skewness
        skewness = 0
        for i in range(n_obs):
            for j in range(n_obs):
                skewness += (mahal_dist_sq[i] * mahal_dist_sq[j]) ** (3/2)
        skewness = skewness / (n_obs ** 2)

        # Mardia's multivariate kurtosis
        kurtosis_mardia = np.mean(mahal_dist_sq ** 2)
        kurtosis_expected = N * (N + 2)  # Expected value under multivariate normality

        # Test statistics
        skewness_stat = (n_obs / 6) * skewness
        kurtosis_stat = (kurtosis_mardia - kurtosis_expected) / np.sqrt(8 * N * (N + 2) / n_obs)

        # p-values
        skewness_df = N * (N + 1) * (N + 2) / 6
        skewness_p = 1 - chi2.cdf(skewness_stat, skewness_df)
        kurtosis_p = 2 * (1 - norm.cdf(abs(kurtosis_stat)))  # Two-tailed test

        # Store results
        extended_metrics_bif["multivariate_normality"] = {
            "mardia_skewness": {
                "statistic": float(skewness_stat),
                "p_value": float(skewness_p),
                "reject_normality": bool(skewness_p < alpha)
            },
            "mardia_kurtosis": {
                "statistic": float(kurtosis_stat),
                "p_value": float(kurtosis_p),
                "reject_normality": bool(kurtosis_p < alpha)
            }
        }

        print(f"Mardia's multivariate skewness: statistic={skewness_stat:.4f}, p-value={skewness_p:.4f}")
        print(f"Mardia's multivariate kurtosis: statistic={kurtosis_stat:.4f}, p-value={kurtosis_p:.4f}")
        print(f"Multivariate normality based on skewness {'rejected' if skewness_p < alpha else 'not rejected'} at {alpha} significance level")
        print(f"Multivariate normality based on kurtosis {'rejected' if kurtosis_p < alpha else 'not rejected'} at {alpha} significance level")

    except np.linalg.LinAlgError:
        print("Warning: Covariance matrix is singular, cannot compute Mardia's tests")
        extended_metrics_bif["multivariate_normality"] = {"error": "Covariance matrix is singular"}

    # Multivariate Portmanteau tests (Hosking/Li-McLeod)
    print("\nRunning multivariate Portmanteau tests (Hosking/Li-McLeod)...")
    try:
        # Run multivariate Portmanteau tests with a subset of series due to high dimensionality
        portmanteau_lags = [5, 10, 15]
        max_dimension = 95  # Maximum number of series to include in the test

        # For DFSV model, df_model would be related to the VAR order (1) and number of factors (K)
        # For a VAR(1) model with K factors, df_model = K^2
        df_model = K**2

        portmanteau_results = multivariate_portmanteau_tests(
            residuals_matrix,
            lags=portmanteau_lags,
            df_model=df_model,
            max_dimension=max_dimension,
            random_state=42  # For reproducibility
        )

        # Store results in extended metrics
        extended_metrics_bif["multivariate_portmanteau"] = portmanteau_results

        # Print summary of results
        print("\nMultivariate Portmanteau Test Results:")
        for lag, result in portmanteau_results.items():
            if "error" in result:
                print(f"  {lag}: Error - {result['error']}")
                continue

            h_pvalue = result["hosking"]["p_value"]
            lm_pvalue = result["li_mcleod"]["p_value"]

            print(f"  {lag} (testing {result['N_tested']} of {N} series):")
            print(f"    Hosking: stat={result['hosking']['statistic']:.2f}, p-value={h_pvalue:.4f}, "
                  f"{'REJECT H0' if h_pvalue < 0.05 else 'FAIL TO REJECT H0'}")
            print(f"    Li-McLeod: stat={result['li_mcleod']['statistic']:.2f}, p-value={lm_pvalue:.4f}, "
                  f"{'REJECT H0' if lm_pvalue < 0.05 else 'FAIL TO REJECT H0'}")

        print("\nNote: H0 = No autocorrelation in residuals. Rejection indicates presence of autocorrelation.")
        print("Note: Tests performed on a random subset of series due to high dimensionality (N=95).")

    except Exception as e:
        print(f"Error in multivariate Portmanteau tests: {e}")
        extended_metrics_bif["multivariate_portmanteau"] = {"error": str(e)}

except Exception as e:
    print(f"Error in multivariate tests: {e}")
    extended_metrics_bif["multivariate_normality"] = {"error": str(e)}

# 2. Generalized Variance (determinant of conditional covariance matrices)
print("\n2. Calculating Generalized Variance Time Series")
try:
    # Calculate determinant of each conditional covariance matrix
    generalized_variance = np.zeros(T)
    for t in range(T):
        # Use log determinant for numerical stability with large matrices
        sign, logdet = np.linalg.slogdet(conditional_covariance_H_bif[t])
        generalized_variance[t] = sign * np.exp(logdet)

    # Create DataFrame with date index for plotting
    gv_df = pd.DataFrame({
        'Generalized_Variance': generalized_variance
    }, index=time_column_pd)

    # Plot generalized variance over time
    plt.figure(figsize=(12, 6))
    plt.plot(gv_df.index, gv_df['Generalized_Variance'])
    plt.title('Generalized Variance (Determinant of Conditional Covariance Matrix)')
    plt.xlabel('Date')
    plt.ylabel('Determinant')
    plt.yscale('log')  # Log scale often helps visualize this better
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(bif_output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Save the plot
    plt.figure(figsize=(12, 6))
    plt.plot(gv_df.index, gv_df['Generalized_Variance'])
    plt.title('Generalized Variance (Determinant of Conditional Covariance Matrix)')
    plt.xlabel('Date')
    plt.ylabel('Determinant')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'generalized_variance.png'), dpi=300)
    plt.close()

    # Save generalized variance to extended metrics
    extended_metrics_bif["generalized_variance"] = {
        "mean": float(np.mean(generalized_variance)),
        "std": float(np.std(generalized_variance)),
        "min": float(np.min(generalized_variance)),
        "max": float(np.max(generalized_variance)),
        "values": generalized_variance.tolist()  # Full time series
    }

    # Save generalized variance to file
    gv_path = os.path.join(data_dir, 'generalized_variance.npy')
    np.save(gv_path, generalized_variance)

except Exception as e:
    print(f"Error calculating generalized variance: {e}")
    extended_metrics_bif["generalized_variance"] = {"error": str(e)}

# 3. Average Conditional Correlation
print("\n3. Calculating Average Conditional Correlation Time Series")
try:
    # Calculate average correlation at each time point
    avg_correlation = np.zeros(T)
    correlation_matrices = np.zeros((T, N, N))

    for t in range(T):
        # Get the covariance matrix at time t
        cov_t = conditional_covariance_H_bif[t]

        # Calculate the correlation matrix
        # Correlation = Cov_ij / sqrt(Var_i * Var_j)
        std_devs = np.sqrt(np.diag(cov_t))
        std_outer = np.outer(std_devs, std_devs)
        corr_t = cov_t / std_outer
        np.fill_diagonal(corr_t, 1.0)  # Ensure diagonal is exactly 1

        # Store the correlation matrix
        correlation_matrices[t] = corr_t

        # Calculate average of off-diagonal elements
        n_off_diag = N * (N - 1) / 2  # Number of unique off-diagonal elements
        avg_correlation[t] = (np.sum(corr_t) - N) / n_off_diag  # Subtract diagonal elements (N ones)

    # Create DataFrame with date index for plotting
    corr_df = pd.DataFrame({
        'Average_Correlation': avg_correlation
    }, index=time_column_pd)

    # Plot average correlation over time
    plt.figure(figsize=(12, 6))
    plt.plot(corr_df.index, corr_df['Average_Correlation'])
    plt.title('Average Conditional Correlation')
    plt.xlabel('Date')
    plt.ylabel('Average Correlation')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(bif_output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Save the plot
    plt.figure(figsize=(12, 6))
    plt.plot(corr_df.index, corr_df['Average_Correlation'])
    plt.title('Average Conditional Correlation')
    plt.xlabel('Date')
    plt.ylabel('Average Correlation')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'average_correlation.png'), dpi=300)
    plt.close()

    # Save average correlation to extended metrics
    extended_metrics_bif["average_correlation"] = {
        "mean": float(np.mean(avg_correlation)),
        "std": float(np.std(avg_correlation)),
        "min": float(np.min(avg_correlation)),
        "max": float(np.max(avg_correlation)),
        "values": avg_correlation.tolist()  # Full time series
    }

    # Save correlation matrices and average correlation to file
    corr_path = os.path.join(data_dir, 'correlation_matrices.npy')
    np.save(corr_path, correlation_matrices)

    avg_corr_path = os.path.join(data_dir, 'average_correlation.npy')
    np.save(avg_corr_path, avg_correlation)

except Exception as e:
    print(f"Error calculating average correlation: {e}")
    extended_metrics_bif["average_correlation"] = {"error": str(e)}

# 4. Factor Variance Contribution Ratio
print("\n4. Calculating Factor Variance Contribution Ratio")
try:
    # Calculate factor and idiosyncratic contributions to variance
    factor_contribution = np.zeros(T)
    idiosyncratic_contribution = np.zeros(T)
    total_variance = np.zeros(T)

    for t in range(T):
        # Get the covariance matrix at time t
        cov_t = conditional_covariance_H_bif[t]

        # Total variance is the trace of the covariance matrix
        total_var_t = np.trace(cov_t)

        # Factor contribution: Λ P_f Λ'
        factor_cov_t = lambda_hat_bif @ filtered_factor_covs_bif[t] @ lambda_hat_bif.T
        factor_var_t = np.trace(factor_cov_t)

        # Idiosyncratic contribution: diag(σ²)
        idio_var_t = np.sum(sigma_eps_hat_diag_bif)

        # Store values
        factor_contribution[t] = factor_var_t
        idiosyncratic_contribution[t] = idio_var_t
        total_variance[t] = total_var_t

    # Calculate ratio of factor variance to total variance
    factor_ratio = factor_contribution / total_variance

    # Create DataFrame with date index for plotting
    ratio_df = pd.DataFrame({
        'Factor_Ratio': factor_ratio,
        'Idiosyncratic_Ratio': idiosyncratic_contribution / total_variance
    }, index=time_column_pd)

    # Plot factor variance contribution ratio over time
    plt.figure(figsize=(12, 6))
    plt.stackplot(ratio_df.index,
                 ratio_df['Factor_Ratio'],
                 ratio_df['Idiosyncratic_Ratio'],
                 labels=['Factor Contribution', 'Idiosyncratic Contribution'],
                 alpha=0.7)
    plt.title('Variance Decomposition: Factor vs. Idiosyncratic')
    plt.xlabel('Date')
    plt.ylabel('Proportion of Total Variance')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(bif_output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Save the plot
    plt.figure(figsize=(12, 6))
    plt.stackplot(ratio_df.index,
                 ratio_df['Factor_Ratio'],
                 ratio_df['Idiosyncratic_Ratio'],
                 labels=['Factor Contribution', 'Idiosyncratic Contribution'],
                 alpha=0.7)
    plt.title('Variance Decomposition: Factor vs. Idiosyncratic')
    plt.xlabel('Date')
    plt.ylabel('Proportion of Total Variance')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'variance_decomposition.png'), dpi=300)
    plt.close()

    # Save factor variance contribution to extended metrics
    extended_metrics_bif["factor_variance_contribution"] = {
        "mean_factor_ratio": float(np.mean(factor_ratio)),
        "std_factor_ratio": float(np.std(factor_ratio)),
        "min_factor_ratio": float(np.min(factor_ratio)),
        "max_factor_ratio": float(np.max(factor_ratio)),
        "factor_ratio_values": factor_ratio.tolist(),  # Full time series
        "mean_total_variance": float(np.mean(total_variance)),
        "mean_factor_variance": float(np.mean(factor_contribution)),
        "mean_idiosyncratic_variance": float(np.mean(idiosyncratic_contribution))
    }

    # Save variance components to file
    variance_components_path = os.path.join(data_dir, 'variance_components.npz')
    np.savez(variance_components_path,
             factor_contribution=factor_contribution,
             idiosyncratic_contribution=idiosyncratic_contribution,
             total_variance=total_variance,
             factor_ratio=factor_ratio)

except Exception as e:
    print(f"Error calculating factor variance contribution: {e}")
    extended_metrics_bif["factor_variance_contribution"] = {"error": str(e)}

print("\nExtended metrics calculation complete.")

# %%
# --- Store Results for Evaluation ---
print("\nStoring BIF results...")

# Extract key parameters from bif_params
lambda_hat_bif = np.array(bif_params.lambda_r)
phi_f_hat_bif = np.array(bif_params.Phi_f)
phi_h_hat_bif = np.array(bif_params.Phi_h)
mu_hat_bif = np.array(bif_params.mu)
sigma_eps_hat_diag_bif = np.array(bif_params.sigma2)
Q_h_hat_bif = np.array(bif_params.Q_h)

# Calculate eigenvalues for stability check
final_phi_f_max_eig = np.max(np.abs(np.linalg.eigvals(phi_f_hat_bif)))
final_phi_h_max_eig = np.max(np.abs(np.linalg.eigvals(phi_h_hat_bif)))

# Get estimation time from result_bif if available, otherwise use placeholder
estimation_time = getattr(result_bif, 'estimation_time', 0.0)
convergence_success = getattr(result_bif, 'convergence_success', True)
convergence_message = getattr(result_bif, 'convergence_message', "Success")

# Get log-likelihood
log_likelihood_base = log_likelihood_bf
log_likelihood_penalized = log_likelihood_bf  # No penalty in current implementation

# Calculate AIC and BIC
num_params = N*K-K*(K+1)/2+2*K**2+N+K #lamda-lambda_constraints+(phi matrixes)+sigma+Q_h
aic_bif = -2 * log_likelihood_base + 2 * num_params
bic_bif = -2 * log_likelihood_base + np.log(T) * num_params

# Main results pickle file (for comparison with other models)
main_output_path = os.path.join(main_output_dir, 'bif_results.pkl')

try:
    # Create comprehensive results dictionary
    bif_results_for_comparison = {
        "model_name": "BIF-DFSV",
        "model_type": "Bellman Information Filter",
        "log_likelihood_penalized": float(log_likelihood_penalized),
        "log_likelihood_base": float(log_likelihood_base),
        "num_params": int(num_params),
        "estimation_time": float(estimation_time),
        "convergence_success": bool(convergence_success),
        "convergence_message": str(convergence_message),
        "standardized_residuals": standardized_residuals_bif,  # Full series
        "standardized_residuals_post_burn": standardized_residuals_bif_post_burn.values,  # Post-burn series
        "estimated_params_constrained": bif_params,
        "estimated_params_unconstrained": None,  # Not applicable for BIF
        "lambda_hat": lambda_hat_bif,
        "phi_f_hat": phi_f_hat_bif,
        "phi_h_hat": phi_h_hat_bif,
        "mu_hat": mu_hat_bif,
        "sigma_eps_hat_diag": sigma_eps_hat_diag_bif,
        "Q_h_hat": Q_h_hat_bif,
        "filtered_states": filtered_states_bf,  # T x (K+K)
        "filtered_factors": filtered_factors_bf,  # T x K
        "filtered_log_vols": filtered_states_bf[:, K:],  # T x K
        "filtered_state_covariances": filtered_covs_bf,  # T x (K+K) x (K+K)
        "conditional_covariance_H": conditional_covariance_H_bif,  # T x N x N
        "predicted_covariance_H": predicted_covariance_H_bif,  # T x N x N (for reference)
        "aic": float(aic_bif),
        "bic": float(bic_bif),
        "final_phi_f_max_eig": float(final_phi_f_max_eig),
        "final_phi_h_max_eig": float(final_phi_h_max_eig),
        "diagnostic_results": diagnostic_results_bif,
        "extended_metrics": extended_metrics_bif,  # Added extended metrics
    }

    # Save main results pickle
    with open(main_output_path, 'wb') as f:
        pickle.dump(bif_results_for_comparison, f)
    print(f"Results dictionary saved to {main_output_path}")

    # Save individual components in DCC-like format

    # 2. Save model parameters
    params_path = os.path.join(data_dir, 'params.pkl')
    with open(params_path, 'wb') as f:
        pickle.dump(bif_params, f)
    print(f"BIF parameters saved to {params_path}")

    # 3. Save model metadata as JSON
    metadata = {
        "model_type": "Bellman Information Filter DFSV",
        "distribution": "Gaussian",  # Current implementation assumes Gaussian errors
        "num_params": int(num_params),
        "estimation_time": float(estimation_time),
        "convergence_success": bool(convergence_success),
        "convergence_message": str(convergence_message),
        "sample_size": int(T),
        "num_series": int(N),
        "k_factors": int(K),
        "log_likelihood": float(log_likelihood_base),
        "aic": float(aic_bif),
        "bic": float(bic_bif),
        "final_phi_f_max_eig": float(final_phi_f_max_eig),
        "final_phi_h_max_eig": float(final_phi_h_max_eig),
    }

    metadata_path = os.path.join(bif_output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata saved to {metadata_path}")

    # 4. Save standardized residuals (eps_tilde.npy in DCC)
    eps_path = os.path.join(data_dir, 'eps_tilde.npy')
    np.save(eps_path, standardized_residuals_bif_post_burn.values)
    print(f"Standardized residuals saved to {eps_path}")

    # Also save as CSV with date index (like in DCC/Factor-CV)
    # First create a DataFrame with date index
    time_column_pd = df_with_date.select(pl.col("Date").cast(pl.Date)).to_pandas()['Date']
    standardized_residuals_df = pd.DataFrame(
        standardized_residuals_bif_post_burn.values,
        index=time_column_pd[analysis_burn_in:],
        columns=[f'Asset_{i+1}' for i in range(N)]
    )
    eps_csv_path = os.path.join(bif_output_dir, 'standardized_residuals.csv')
    standardized_residuals_df.to_csv(eps_csv_path)
    print(f"Standardized residuals CSV saved to {eps_csv_path}")

    # 5. Save conditional covariance matrices (Ht.npy in DCC)
    ht_path = os.path.join(data_dir, 'Ht.npy')
    # Extract post-burn covariances to match DCC format
    post_burn_H = conditional_covariance_H_bif[analysis_burn_in:, :, :]
    np.save(ht_path, post_burn_H)
    print(f"Conditional covariance matrices saved to {ht_path}")

    # 6. Save date index (date_index.txt in DCC)
    date_path = os.path.join(data_dir, 'date_index.txt')
    with open(date_path, 'w') as f:
        for date in time_column_pd[analysis_burn_in:]:
            f.write(f"{date.strftime('%Y-%m-%d')}\n")
    print(f"Date index saved to {date_path}")

    # 7. Save filtered states (factors and log-vols)
    states_path = os.path.join(data_dir, 'filtered_states.npy')
    np.save(states_path, filtered_states_bf[analysis_burn_in:])
    print(f"Filtered states saved to {states_path}")

    # 8. Save diagnostic results
    diagnostic_path = os.path.join(bif_output_dir, 'diagnostic_results.json')
    with open(diagnostic_path, 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        diagnostic_json = json.dumps(diagnostic_results_bif, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
        f.write(diagnostic_json)
    print(f"Diagnostic results saved to {diagnostic_path}")

    # 9. Save metrics summary (similar to Factor-CV)
    metrics_summary = {
        "model_name": "BIF-DFSV",
        "convergence_success": bool(convergence_success),
        "convergence_message": str(convergence_message),
        "estimation_time": float(estimation_time),
        "num_params": int(num_params),
        "log_likelihood_penalized": float(log_likelihood_penalized),
        "log_likelihood_base": float(log_likelihood_base),
        "aic": float(aic_bif),
        "bic": float(bic_bif),
        "final_phi_f_max_eig": float(final_phi_f_max_eig),
        "final_phi_h_max_eig": float(final_phi_h_max_eig),
        "diagnostic_tests": diagnostic_results_bif,
        "extended_metrics": extended_metrics_bif
    }

    # Also save extended metrics separately for easier access
    extended_metrics_path = os.path.join(bif_output_dir, 'extended_metrics.json')
    with open(extended_metrics_path, 'w') as f:
        json.dump(extended_metrics_bif, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    print(f"Extended metrics saved to {extended_metrics_path}")

    # 10. Save parameter heatmaps
    print("\nSaving parameter heatmaps...")

    # Create figures directory if not already created
    figures_dir = os.path.join(bif_output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Factor loadings (Lambda)
    plt.figure(figsize=(16, 12))
    sns.heatmap(lambda_hat_bif, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Heatmap of Factor Loadings (Lambda)')
    plt.xlabel('Factors')
    plt.ylabel('Assets')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'lambda_heatmap.png'), dpi=300)
    plt.close()

    # 2. Factor transition matrix (Phi_f)
    plt.figure(figsize=(10, 8))
    sns.heatmap(phi_f_hat_bif, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5)
    plt.title('Heatmap of Factor Transition Matrix (Phi_f)')
    plt.xlabel('Factor (t-1)')
    plt.ylabel('Factor (t)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'phi_f_heatmap.png'), dpi=300)
    plt.close()

    # 3. Log-volatility transition matrix (Phi_h)
    plt.figure(figsize=(10, 8))
    sns.heatmap(phi_h_hat_bif, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5)
    plt.title('Heatmap of Log-Volatility Transition Matrix (Phi_h)')
    plt.xlabel('Log-Vol (t-1)')
    plt.ylabel('Log-Vol (t)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'phi_h_heatmap.png'), dpi=300)
    plt.close()

    print(f"Parameter heatmaps saved to {figures_dir}")

    metrics_path = os.path.join(bif_output_dir, 'metrics_summary.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    print(f"Metrics summary saved to {metrics_path}")

    print("All DCC-compatible outputs saved successfully")

except Exception as e:
    print(f"Error saving outputs: {e}")
    import traceback
    traceback.print_exc()

print("\nBIF Extraction Script Finished.")

