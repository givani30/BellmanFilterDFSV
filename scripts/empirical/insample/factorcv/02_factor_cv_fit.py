import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResults, MLEModelResultsWrapper
from scipy.linalg import solve_discrete_lyapunov # If stationary initialization were needed
import warnings
import time
import pickle
import os

# Import the custom FactorCVModel
from .factor_cv_model import FactorCVModel

# Ensure calculations use float64 for stability
# np.set_printoptions(precision=8, suppress=True) # Optional: for printing

print("Starting Factor-CV Model Estimation...")

import polars as pl
import pathlib

# Get the script's directory
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
# Define paths relative to the script location
INPUT_FILE = os.path.join(SCRIPT_DIR.parent.parent, "vw_returns_final_with_date.csv")

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

    # Ensure data is demeaned (important for Factor-CV)

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
k_factors = 5 # Define k_factors here as it's used below
print(f"Final data shape for model fitting: {df_returns.shape}")

# --- Instantiate and Fit the Model ---
factor_cv_model = FactorCVModel(df_returns, k_factors=k_factors)

print("Fitting the model (this may take time)...")
start_time = time.time()
try:
    # Using L-BFGS-B which often works well for high-dimensional problems
    # ftol=1e-08, gtol=1e-06 are typical precision settings
    factor_cv_results = factor_cv_model.fit(method='lbfgs', maxiter=1500,
                                            disp=True, # Show optimizer output
                                            pgtol=1e-06, factr=1e7) # L-BFGS specific options

    end_time = time.time()
    estimation_time = end_time - start_time

    print("\nFactor-CV Estimation Summary:")
    print(factor_cv_results.summary())

    # Check convergence
    convergence_success = factor_cv_results.mle_retvals['warnflag'] == 0
    convergence_message = factor_cv_results.mle_retvals['task']

    if not convergence_success:
         warnings.warn(f"Optimization may not have converged: {convergence_message}")
    else:
         print("\nOptimization converged successfully.")

    # Sanity check final LLF
    if not np.isfinite(factor_cv_results.llf):
        warnings.warn("Final log-likelihood is non-finite. Estimation likely failed.")
        convergence_success = False # Treat non-finite LLF as failure

except Exception as e:
    print(f"\nERROR fitting Factor-CV model: {e}")
    import traceback
    traceback.print_exc()
    factor_cv_results = None
    estimation_time = time.time() - start_time # Record time until error
    convergence_success = False
    convergence_message = f"Error during fitting: {e}"


# --- Extract and Calculate Results ---
loglik_cv_penalized = -np.inf
loglik_cv_base = np.nan
num_params_cv = np.nan
theta_cv_hat_constrained = None
theta_cv_hat_unconstrained = None
lambda_hat_cv = None
phi_f_hat_cv = None
sigma_eps_hat_cv_diag = None
sigma_nu_hat_cv_diag = None
filtered_factors_cv = None
filtered_state_covs_cv = None
conditional_covariance_H_cv = None
standardized_residuals_cv = None
standardized_residuals_cv_post_burn = None
aic_cv = np.nan
bic_cv = np.nan
final_phi_f_max_eig = np.nan
matrices_extracted = False


if convergence_success and factor_cv_results is not None:
    print("\nExtracting and Calculating Results for Factor-CV...")

    # Estimated Parameters
    loglik_cv_penalized = factor_cv_results.llf
    num_params_cv = factor_cv_model.k_params
    theta_cv_hat_constrained = factor_cv_results.params
    theta_cv_hat_unconstrained = factor_cv_model.untransform_params(theta_cv_hat_constrained)

    # Reconstruct estimated system matrices (using final constrained params)
    try:
         factor_cv_model.update(theta_cv_hat_constrained) # Use final constrained params

         lambda_hat_cv = factor_cv_model.ssm['design', 0, 0].copy()
         sigma_eps_hat_cv_diag = np.diag(factor_cv_model.ssm['obs_cov', 0, 0]).copy()
         phi_f_hat_cv = factor_cv_model.ssm['transition', 0, 0].copy()
         sigma_nu_hat_cv_diag = np.diag(factor_cv_model.ssm['state_cov', 0, 0]).copy()

         print("\nExtracted Final System Matrices (Constrained):")
         print(f"Lambda_hat shape: {lambda_hat_cv.shape}")
         print(f"Sigma_eps_hat (diag) shape: {sigma_eps_hat_cv_diag.shape}")
         print(f"Phi_f_hat shape: {phi_f_hat_cv.shape}")
         print(f"Sigma_nu_hat (diag) shape: {sigma_nu_hat_cv_diag.shape}")

         # Verify stationarity of the final estimated Phi_f
         final_eigenvalues = np.linalg.eigvals(phi_f_hat_cv)
         final_phi_f_max_eig = np.max(np.abs(final_eigenvalues))
         print(f"Max eigenvalue magnitude of final Phi_f: {final_phi_f_max_eig:.6f}")
         if final_phi_f_max_eig >= 1.0:
             warnings.warn("Final estimated Phi_f is non-stationary despite penalty!")
         matrices_extracted = True

    except Exception as e:
         print(f"\nError reconstructing final matrices: {e}")
         matrices_extracted = False
         lambda_hat_cv, sigma_eps_hat_cv_diag, phi_f_hat_cv, sigma_nu_hat_cv_diag = None, None, None, None


    # Calculate Base Log-Likelihood (without penalty)
    if theta_cv_hat_unconstrained is not None:
        original_penalty_weight = factor_cv_model.STABILITY_PENALTY_WEIGHT
        factor_cv_model.STABILITY_PENALTY_WEIGHT = 0
        try:
            # Need to call update first with unconstrained params to set ssm correctly
            factor_cv_model.update(theta_cv_hat_unconstrained)
            loglik_cv_base = factor_cv_model.loglike(theta_cv_hat_unconstrained)
            print(f"Base Log-Likelihood (no penalty): {loglik_cv_base:.4f}")
        except Exception as e:
            print(f"Could not calculate base log-likelihood: {e}")
            loglik_cv_base = np.nan
        factor_cv_model.STABILITY_PENALTY_WEIGHT = original_penalty_weight # Restore penalty weight


    # Filtered States and Covariances
    if factor_cv_results.filter_results is not None:
        filtered_factors_cv = factor_cv_results.filter_results.filtered_state # T x K
        filtered_state_covs_cv = factor_cv_results.filter_results.filtered_state_cov # T x K x K
        print(f"Filtered states shape: {filtered_factors_cv.shape}")
        print(f"Filtered state covs shape: {filtered_state_covs_cv.shape}")


    # Calculate Conditional Observation Covariances and Standardized Residuals
    if matrices_extracted and filtered_factors_cv is not None and filtered_state_covs_cv is not None:
        print("\nCalculating Conditional Covariances and Standardized Residuals...")
        conditional_covariance_H_cv = np.full((T, N, N), np.nan)
        standardized_residuals_cv = np.full((T, N), np.nan)

        returns_arr = df_returns.values
        lambda_hat_arr = np.asarray(lambda_hat_cv)
        sigma_eps_hat_diag_mat = np.diag(np.asarray(sigma_eps_hat_cv_diag))

        burn_in = factor_cv_model.loglikelihood_burn
        print(f"Calculating from t={burn_in} (post burn-in)")

        for t in range(burn_in, T):
            # Get filtered factor estimate and its covariance for time t
            f_hat_t = filtered_factors_cv[t, :] # Shape (K,)
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
        # Extract post-burn residuals
        standardized_residuals_cv_post_burn = standardized_residuals_cv.iloc[burn_in:]
        print(f"Standardized residuals DataFrame shape (full): {standardized_residuals_cv.shape}")
        print(f"Standardized residuals DataFrame shape (post-burn): {standardized_residuals_cv_post_burn.shape}")

    # Calculate AIC and BIC (using base log-likelihood and post-burn observations)
    if np.isfinite(loglik_cv_base) and num_params_cv is not np.nan:
        T_post_burn = T - factor_cv_model.loglikelihood_burn
        if T_post_burn > 0:
            aic_cv = -2 * loglik_cv_base + 2 * num_params_cv
            bic_cv = -2 * loglik_cv_base + num_params_cv * np.log(T_post_burn)
            print(f"\nCalculated AIC: {aic_cv:.4f}")
            print(f"Calculated BIC: {bic_cv:.4f}")
        else:
            warnings.warn("Not enough post-burn observations to calculate AIC/BIC.")


else:
     print("\nSkipping result extraction and calculation due to estimation error or non-convergence.")


# --- Store Results for Evaluation ---
print("\nStoring Factor-CV results...")

# Ensure output directory exists
output_dir = 'outputs/empirical/insample/'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'factorcv_results.pkl')

factor_cv_results_for_comparison = {
    "model_name": "Factor-CV",
    "log_likelihood_penalized": loglik_cv_penalized,
    "log_likelihood_base": loglik_cv_base,
    "num_params": num_params_cv,
    "estimation_time": estimation_time,
    "convergence_success": convergence_success,
    "convergence_message": convergence_message,
    "standardized_residuals": standardized_residuals_cv, # Full series
    "standardized_residuals_post_burn": standardized_residuals_cv_post_burn, # Post-burn series
    "estimated_params_constrained": theta_cv_hat_constrained,
    "estimated_params_unconstrained": theta_cv_hat_unconstrained, # Added for completeness
    "lambda_hat": lambda_hat_cv,
    "phi_f_hat": phi_f_hat_cv,
    "sigma_eps_hat_diag": sigma_eps_hat_cv_diag,
    "sigma_nu_hat_diag": sigma_nu_hat_cv_diag,
    "filtered_factors": filtered_factors_cv, # T x K
    "filtered_state_covariances": filtered_state_covs_cv, # T x K x K
    "conditional_covariance_H": conditional_covariance_H_cv, # T x N x N
    "aic": aic_cv,
    "bic": bic_cv,
    "final_phi_f_max_eig": final_phi_f_max_eig
}

try:
    with open(output_path, 'wb') as f:
        pickle.dump(factor_cv_results_for_comparison, f)
    print(f"Results dictionary saved to {output_path}")
except Exception as e:
    print(f"Error saving results to pickle file: {e}")


print("\nFactor-CV Estimation Script Finished.")

# Proceed with residual analysis on standardized_residuals_cv_df_post_burn in 03_factor_cv_metrics.py
