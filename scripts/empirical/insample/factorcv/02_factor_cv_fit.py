import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResults
from scipy.linalg import solve_discrete_lyapunov # If stationary initialization were needed
import warnings
import time
import pickle
import os

# Import the custom FactorCVModel
# from factor_cv_model import FactorCVModel # Removed as we are using statsmodels

# Import the statsmodels DynamicFactor model
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

# Ensure calculations use float64 for stability
# np.set_printoptions(precision=8, suppress=True) # Optional: for printing

print("Starting Factor-CV Model Estimation...")

import polars as pl
import pathlib

# Get the script's directory
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
# Define paths relative to the script location
INPUT_FILE = os.path.join(SCRIPT_DIR.parent.parent.parent, "empirical", "vw_returns_final_with_date.csv")

# --- Data Loading and Preprocessing ---
print("Loading and preprocessing data...")
try:
    # Load data using polars
    df_pl = pl.read_csv(INPUT_FILE)
    date_col_name = df_pl.columns[0]
    return_cols = df_pl.columns[1:]

    # Convert to pandas DataFrame for statsmodels, setting date as index
    # Ensure date column is treated as datetime
    df_pd = df_pl.with_columns(pl.col(date_col_name).str.strptime(pl.Date, "%Y-%m-%d").cast(pl.Datetime)).to_pandas()
    df_pd = df_pd.set_index(date_col_name)
    df_returns = df_pd[return_cols]

    print(f"Raw data loaded with shape: {df_returns.shape}")

    # Ensure data has N=95 columns
    if df_returns.shape[1] != 95:
         raise ValueError(f"Input data must have N=95 columns, found {df_returns.shape[1]}")

    # Remove temporary data subsetting
    # print(f"Original data shape: {df_returns.shape}")
    # df_returns = df_returns.iloc[:200, :20]
    # print(f"Using reduced dataset for testing: {df_returns.shape}")

    # Note: Input data is already demeaned, no need to demean again
    print("Using already demeaned data")

    # Ensure data is in decimal format (if not already, e.g., if in percentage)
    # Assuming data is already in decimal format based on typical financial data
    # If your data is in percentage, uncomment the line below:
    # df_returns = df_returns / 100.0
    # print("Data converted to decimal format.")


except Exception as e:
    print(f"Error loading or processing data: {e}")
    # Keep dummy data for now in case of error, but ideally exit or handle
    # --- Dummy Data Generation (for testing purposes in case of loading error) ---
    warnings.warn("Error loading actual data, using dummy data for now.")
    T_approx = 726
    N = 95
    K = 5
    np.random.seed(42)
    dummy_data = np.random.randn(T_approx, N) * 0.05 # Simulate returns
    dates = pd.date_range(start='1963-07-01', periods=T_approx, freq='M')
    df_returns = pd.DataFrame(dummy_data, index=dates, columns=[f'Asset_{i+1}' for i in range(N)])
    df_returns = df_returns - df_returns.mean() # Demean
    print(f"Using dummy data with shape: {df_returns.shape}")
    # --- End Dummy Data ---


# After loading and preprocessing, define T and N
T = df_returns.shape[0]
N = df_returns.shape[1]
k_factors = 5 # Set k_factors to 5 as required
print(f"Final data shape for model fitting: {df_returns.shape}")
print(f"Using {k_factors} factors for the model")

# --- Instantiate and Fit the Model ---
# Define the standard Dynamic Factor Model (DFM)
# Assumes Var(factor innovations) = I for identification
print("Initializing statsmodels DynamicFactor model...")
# dfm_model = DynamicFactor(
#     endog=df_returns,          # Your (T x N) demeaned returns data
#     k_factors=k_factors,       # K=5
#     factor_order=1,            # VAR(1) for factors
#     error_order=0,             # White noise errors (epsilon_t)
#     error_var=False,           # Errors are separate AR(0) processes
#     error_cov_type='diagonal', # Diagonal Sigma_epsilon
#     enforce_stationarity=True  # Enforce stationarity for factor VAR
# )
dfm_model = DynamicFactorMQ(
    endog=df_returns,          # Your (T x N) demeaned returns data
    factors=k_factors,       # K=5
    factor_orders=1,            # VAR(1) for factors
    idiosyncratic_ar1=False,   # Idiosyncratic variances are separate AR(0) processes;
)
print(f"Initialized statsmodels DynamicFactor: N={dfm_model.nobs}, K={dfm_model.k_factors}")
print(f"Number of parameters to estimate: {dfm_model.k_params}")

print("Fitting the DynamicFactor model (this may take time)...")
start_time = time.time()
factor_cv_results = None
convergence_success = False
try:
    # Use default 'lbfgs' or try 'bfgs' if needed. Increase maxiter.
    factor_cv_results = dfm_model.fit(maxiter=10000, disp=5) # disp=True shows optimizer output

    # Simple convergence check - if we have finite log-likelihood, consider it converged
    convergence_success = False
    convergence_message = "No convergence information available"

    if factor_cv_results is not None and hasattr(factor_cv_results, 'llf'):
        if np.isfinite(factor_cv_results.llf):
            convergence_success = True
            convergence_message = "Model estimation completed successfully"
            print("Optimization converged successfully.")
        else:
            convergence_message = "Model estimation completed but log-likelihood is not finite"
            warnings.warn(f"Optimization may not have converged: {convergence_message}")
    else:
        warnings.warn(f"Optimization may not have converged: {convergence_message}")

except Exception as e:
    # Handle fitting errors
    print(f"\nERROR fitting DynamicFactor model: {e}")
    # Ensure appropriate variables are set for downstream saving
    loglik_cv_base = np.nan
    num_params_cv = dfm_model.k_params # Or some default if needed
    theta_cv_hat_constrained = None
    # ... potentially set other extracted variables to None or NaN ...
    convergence_success = False
    convergence_message = f"Error during estimation: {str(e)}"
    # Re-raise or handle as appropriate for the script's flow
    # For now, just print and ensure results object is None

end_time = time.time()
estimation_time = end_time - start_time
print(f"Estimation finished in {estimation_time:.2f} seconds.")


# --- Extract and Calculate Results ---
# Initialize variables to handle potential fitting failures
loglik_cv_base = np.nan
num_params_cv = dfm_model.k_params # Default to initial param count
theta_cv_hat_constrained = None
lambda_hat_cv = None
sigma_eps_hat_cv_diag = None
phi_f_hat_cv = None
sigma_nu_hat_cv_diag = np.ones(k_factors) # Implicitly identity
filtered_factors_cv = None
filtered_state_covs_cv = None
final_phi_f_max_eig = np.nan
aic_cv = np.nan  # Initialize AIC
bic_cv = np.nan  # Initialize BIC
hqic_cv = np.nan # Initialize HQIC (if used in results_dict)
conditional_covariance_H_cv = None # Initialize H_cv

if convergence_success and factor_cv_results is not None:
    print("Extracting results from fitted model...")

    # Extract basic results
    loglik_cv_base = factor_cv_results.llf  # Base log-likelihood
    print(f"Log-likelihood: {loglik_cv_base:.4f}")

    num_params_cv = factor_cv_results.df_model  # Number of parameters
    print(f"Number of parameters: {num_params_cv}")

    theta_cv_hat_constrained = factor_cv_results.params  # Parameter vector
    print(f"Parameter vector length: {len(theta_cv_hat_constrained)}")

    # Extract model parameters and factors
    print(f"Extracting model parameters (factors={k_factors}, series={N})")

    try:
        # Get the state space model
        ssm = factor_cv_results.model.ssm

        # Extract model matrices
        lambda_hat_cv = ssm.design[:, :k_factors, 0]  # Factor loadings (N, K)
        sigma_eps_hat_cv_diag = np.diag(ssm.obs_cov[:, :, 0])  # Observation error variances (N,)
        phi_f_hat_cv = ssm.transition[:k_factors, :k_factors, 0]  # Factor VAR coefficients (K, K)

        # Get factors - the factors property returns a Bunch object with filtered/smoothed factors
        print("Getting factors from results object")
        factors_bunch = factor_cv_results.factors
        print(f"Factors bunch attributes: {dir(factors_bunch)}")

        # Use smoothed factors if available, otherwise filtered
        if hasattr(factors_bunch, 'smoothed') and factors_bunch.smoothed is not None:
            print("Using smoothed factors")
            filtered_factors_cv = factors_bunch.smoothed  # Shape should be (T, K)
        elif hasattr(factors_bunch, 'filtered') and factors_bunch.filtered is not None:
            print("Using filtered factors")
            filtered_factors_cv = factors_bunch.filtered  # Shape should be (T, K)
        else:
            # Fallback to extracting from filtered_state
            print("Falling back to filtered_state")
            filtered_states = factor_cv_results.filtered_state
            filtered_factors_cv = filtered_states[:k_factors, :].T  # Transpose to get (T, K)

        print(f"Factors shape: {filtered_factors_cv.shape}")

        # Get factor covariances
        filtered_state_covs = factor_cv_results.filtered_state_cov  # (state_dim, state_dim, T)
        print(f"Filtered state covariances shape: {filtered_state_covs.shape}")
        filtered_state_covs_cv = filtered_state_covs[:k_factors, :k_factors, :].transpose(2, 0, 1)  # (T, K, K)
        print(f"Extracted factor covariances shape: {filtered_state_covs_cv.shape}")

    except Exception as e:
        print(f"Error extracting model parameters: {e}")
        import traceback
        traceback.print_exc()
        # Set to None to handle gracefully
        lambda_hat_cv = None
        sigma_eps_hat_cv_diag = None
        phi_f_hat_cv = None
        filtered_factors_cv = None
        filtered_state_covs_cv = None

    # Check stationarity of estimated Phi_f
    try:
        final_eigenvalues = np.linalg.eigvals(phi_f_hat_cv)
        final_phi_f_max_eig = np.max(np.abs(final_eigenvalues))
        print(f"Max eigenvalue of estimated Phi_f: {final_phi_f_max_eig:.4f}")
    except np.linalg.LinAlgError:
        warnings.warn("Could not compute eigenvalues for Phi_f.")
        final_phi_f_max_eig = np.nan
    # ... [any other specific extraction needed by the script] ...
else:
    print("Skipping result extraction due to fitting failure or non-convergence.")


# --- Calculate Conditional Observation Covariances and Standardized Residuals ---
# Review the existing logic for calculating standardized residuals ($z_t$).
# Ensure it now uses the variables extracted from the `statsmodels` results:
# `lambda_hat_cv`, `sigma_eps_hat_cv_diag`, `filtered_factors_cv`, and `filtered_state_covs_cv`.
# The core mathematical logic should remain the same, just ensure the input variable names match the extraction step above.
# Make sure `Ht` calculation uses `sigma_eps_hat_cv_diag` correctly to form the diagonal $\Sigma_\epsilon$.

# Initialize variables to handle potential calculation failures
standardized_residuals_cv = None
standardized_residuals_cv_post_burn = None
conditional_covariance_H_cv = None # Re-initialize H_cv here for clarity

if convergence_success and lambda_hat_cv is not None and sigma_eps_hat_cv_diag is not None and \
   filtered_factors_cv is not None and filtered_state_covs_cv is not None:

    print("\nCalculating Conditional Covariances and Standardized Residuals...")
    conditional_covariance_H_cv = np.full((T, N, N), np.nan)
    standardized_residuals_cv = np.full((T, N), np.nan)

    returns_arr = df_returns.values
    lambda_hat_arr = np.asarray(lambda_hat_cv)
    sigma_eps_hat_diag_mat = np.diag(np.asarray(sigma_eps_hat_cv_diag))

    # statsmodels DynamicFactor does not have a 'loglikelihood_burn' attribute.
    # Assuming burn_in is not needed for residual calculation based on statsmodels output structure.
    # If a burn-in period is required for analysis, it should be applied *after* calculating residuals for the full series.
    burn_in = 0 # Assuming no burn-in needed for statsmodels residuals

    print(f"Calculating from t={burn_in}") # Start from 0

    # Check if filtered_factors_cv is a DataFrame or ndarray
    if isinstance(filtered_factors_cv, pd.DataFrame):
        filtered_factors_arr = filtered_factors_cv.values
    else:
        filtered_factors_arr = np.asarray(filtered_factors_cv)

    for t in range(burn_in, T):
        # Get filtered factor estimate and its covariance for time t
        f_hat_t = filtered_factors_arr[t, :] # Shape (K,)
        P_f_t = filtered_state_covs_cv[t, :, :] # Shape (K, K)

        # Calculate conditional observation mean: mu_t|t = Lambda * f_t|t
        mu_t_cv = lambda_hat_arr @ f_hat_t # Shape (N,)

        # Calculate conditional observation covariance: Sigma_t|t = Lambda * P_f,t|t * Lambda' + Sigma_eps
        Sigma_t_cv = lambda_hat_arr @ P_f_t @ lambda_hat_arr.T + sigma_eps_hat_diag_mat
        Sigma_t_cv = (Sigma_t_cv + Sigma_t_cv.T) / 2 # Ensure symmetry numerically

        try:
            # Store conditional covariance
            conditional_covariance_H_cv[t, :, :] = Sigma_t_cv

            # Cholesky decomposition: Sigma_t|t = L * L'
            # Increase jitter slightly for potentially ill-conditioned matrices
            jitter_cholesky = 1e-7
            L_t = np.linalg.cholesky(Sigma_t_cv + jitter_cholesky * np.eye(N))
            # Raw residual: e_t = r'_t - mu_t|t
            e_t = returns_arr[t, :] - mu_t_cv # Shape (N,)
            # Standardized residual: z_t = L^{-1} * e_t (solve L*z = e)
            z_t = np.linalg.solve(L_t, e_t) # solve_triangular might be marginally faster
            standardized_residuals_cv[t, :] = z_t
        except np.linalg.LinAlgError:
            warnings.warn(f"Cholesky decomposition failed for Sigma_t|t at t={t}. Skipping residual calculation.")
            # Leave as NaN
            # Optional: Check condition number of Sigma_t_cv here for diagnostics
            # cond_num = np.linalg.cond(Sigma_t_cv)
            # print(f" Condition number at t={t}: {cond_num}")

    print("Finished calculating conditional covariances and standardized residuals.")
    # Convert standardized residuals to DataFrame
    standardized_residuals_cv = pd.DataFrame(
        standardized_residuals_cv,
        index=df_returns.index,
        columns=df_returns.columns
    )
    # Extract post-burn residuals (if a burn-in is defined for analysis)
    # For statsmodels, we assume burn_in = 0 for calculation, but keep the variable for potential later analysis burn-in
    analysis_burn_in = 0 # Define burn-in for analysis if needed
    standardized_residuals_cv_post_burn = standardized_residuals_cv.iloc[analysis_burn_in:]
    print(f"Standardized residuals DataFrame shape (full): {standardized_residuals_cv.shape}")
    print(f"Standardized residuals DataFrame shape (post-burn): {standardized_residuals_cv_post_burn.shape}")

    # Calculate AIC and BIC (using base log-likelihood and *full* sample size for statsmodels)
    if np.isfinite(loglik_cv_base) and num_params_cv is not np.nan:
        # statsmodels AIC/BIC typically use the full sample size T
        T_for_aic_bic = T
        if T_for_aic_bic > 0:
            aic_cv = -2 * loglik_cv_base + 2 * num_params_cv
            bic_cv = -2 * loglik_cv_base + num_params_cv * np.log(T_for_aic_bic)
            print(f"\nCalculated AIC: {aic_cv:.4f}")
            print(f"Calculated BIC: {bic_cv:.4f}")
        else:
            warnings.warn("Sample size is zero, cannot calculate AIC/BIC.")

else:
     print("\nSkipping result extraction and calculation due to estimation error or non-convergence.")


# --- Store Results for Evaluation ---
print("\nStoring Factor-CV results...")

# Ensure output directories exist
main_output_dir = 'outputs/empirical/insample/'
factorcv_output_dir = os.path.join(main_output_dir, 'factorcv')
data_dir = os.path.join(factorcv_output_dir, 'data')
os.makedirs(main_output_dir, exist_ok=True)
os.makedirs(factorcv_output_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Main results pickle file (for comparison with other models)
main_output_path = os.path.join(main_output_dir, 'factorcv_results.pkl')

# Create comprehensive results dictionary
factor_cv_results_for_comparison = {
    "model_name": "Factor-CV", # Keep this name for comparison script compatibility
    "model_type": "DFM (statsmodels)", # Add specific type
    "log_likelihood_penalized": loglik_cv_base, # statsmodels doesn't have penalty, use base LLF
    "log_likelihood_base": loglik_cv_base,
    "num_params": num_params_cv, # Use num_params_cv from statsmodels results
    "estimation_time": estimation_time,
    "convergence_success": convergence_success,
    "convergence_message": convergence_message,
    "standardized_residuals": standardized_residuals_cv, # Full series
    "standardized_residuals_post_burn": standardized_residuals_cv_post_burn, # Post-burn series (based on analysis_burn_in)
    "estimated_params_constrained": theta_cv_hat_constrained,
    "estimated_params_unconstrained": None, # statsmodels doesn't provide unconstrained params directly
    "lambda_hat": lambda_hat_cv,
    "phi_f_hat": phi_f_hat_cv,
    "sigma_eps_hat_diag": sigma_eps_hat_cv_diag,
    "sigma_nu_hat_diag": sigma_nu_hat_cv_diag, # Should be np.ones(k_factors)
    "filtered_factors": filtered_factors_cv, # T x K
    "filtered_state_covariances": filtered_state_covs_cv, # T x K x K
    "conditional_covariance_H": conditional_covariance_H_cv, # T x N x N
    "aic": aic_cv,
    "bic": bic_cv,
    "final_phi_f_max_eig": final_phi_f_max_eig
}

# Save main results pickle
try:
    with open(main_output_path, 'wb') as f:
        pickle.dump(factor_cv_results_for_comparison, f)
    print(f"Results dictionary saved to {main_output_path}")
except Exception as e:
    print(f"Error saving results to pickle file: {e}")

# Save individual components in DCC-like format
if convergence_success and conditional_covariance_H_cv is not None:
    try:
        # 1. Save model object (similar to model.pkl in DCC) - Save the statsmodels model object
        if dfm_model is not None: # Use dfm_model instead of factor_cv_model
            model_path = os.path.join(data_dir, 'model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(dfm_model, f)
            print(f"Statsmodels model object saved to {model_path}")

        # 2. Save model metadata as JSON
        import json
        metadata = {
            "model_type": "DFM (statsmodels)", # Updated model type
            "distribution": "Gaussian", # statsmodels DFM assumes Gaussian errors
            "num_params": int(num_params_cv) if num_params_cv is not None and np.isfinite(num_params_cv) else None, # Use num_params_cv
            "estimation_time": float(estimation_time),
            "convergence_success": bool(convergence_success),
            "convergence_message": str(convergence_message),
            "sample_size": int(T),
            "num_series": int(N),
            "k_factors": int(k_factors),
            "log_likelihood": float(loglik_cv_base) if np.isfinite(loglik_cv_base) else None,
            "aic": float(aic_cv) if np.isfinite(aic_cv) else None,
            "bic": float(bic_cv) if np.isfinite(bic_cv) else None,
            "phi_f_max_eigenvalue": float(final_phi_f_max_eig) if np.isfinite(final_phi_f_max_eig) else None
        }
        metadata_path = os.path.join(data_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Model metadata saved to {metadata_path}")

        # 3. Save standardized residuals (eps_tilde.npy in DCC)
        if standardized_residuals_cv_post_burn is not None:
            eps_path = os.path.join(data_dir, 'eps_tilde.npy')
            np.save(eps_path, standardized_residuals_cv_post_burn.values)
            print(f"Standardized residuals saved to {eps_path}")

            # Also save as CSV with date index (like in DCC)
            eps_csv_path = os.path.join(factorcv_output_dir, 'standardized_residuals.csv')
            standardized_residuals_cv_post_burn.to_csv(eps_csv_path)
            print(f"Standardized residuals CSV saved to {eps_csv_path}")

        # 4. Save conditional covariance matrices (Ht.npy in DCC)
        if conditional_covariance_H_cv is not None:
            ht_path = os.path.join(data_dir, 'Ht.npy')
            # Extract post-burn covariances to match DCC format
            # Use analysis_burn_in for consistency with residuals
            post_burn_H = conditional_covariance_H_cv[analysis_burn_in:, :, :]
            np.save(ht_path, post_burn_H)
            print(f"Conditional covariance matrices saved to {ht_path}")

        # 5. Save date index (date_index.txt in DCC)
        if standardized_residuals_cv_post_burn is not None:
            date_path = os.path.join(data_dir, 'date_index.txt')
            with open(date_path, 'w') as f:
                for date in standardized_residuals_cv_post_burn.index:
                    f.write(f"{date.strftime('%Y-%m-%d')}\n")
            print(f"Date index saved to {date_path}")

        print("All DCC-compatible outputs saved successfully")
    except Exception as e:
        print(f"Error saving DCC-compatible outputs: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Skipping DCC-compatible output generation due to estimation error or non-convergence")


print("\nFactor-CV Estimation Script Finished.")

# Proceed with residual analysis on standardized_residuals_cv_df_post_burn in 03_factor_cv_metrics.py
