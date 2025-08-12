# Factor-CV Benchmark Implementation Plan (25-04-2025)

## 1. Goal

Implement the Factor model with Constant Volatility (Factor-CV) as a benchmark model for the empirical analysis, following the structure of the existing DCC-GARCH benchmark and using the `statsmodels` library. Ensure all required outputs for comparison are saved.

## 2. File Structure

The implementation will reside in `scripts/empirical/insample/factorcv/` with the following files:

*   **`scripts/empirical/insample/factorcv/01_factor_cv_model.py`**: Contains the Python class definition for `FactorCVModel`.
*   **`scripts/empirical/insample/factorcv/02_factor_cv_fit.py`**: Handles data loading, model fitting, calculation of derived outputs, and saving of comprehensive results.
*   **`scripts/empirical/insample/factorcv/03_factor_cv_metrics.py`**: Loads the saved results and performs residual diagnostics and generates summary metrics/plots.

## 3. Implementation Details

### 3.1. `01_factor_cv_model.py`

*   This file will contain the `FactorCVModel` class as provided in the initial prompt.
*   Key components:
    *   Initialization (`__init__`) setting up state-space dimensions and fixed matrices.
    *   Parameter indexing (`_initialize_parameter_indices`).
    *   Start parameters (`start_params`).
    *   Parameter names (`param_names`).
    *   Parameter transformations (`transform_params`, `untransform_params`).
    *   State-space matrix update logic (`update`).
    *   Stationarity penalty calculation (`_calculate_stationarity_penalty`).
    *   Penalized log-likelihood calculation (`loglike`).

### 3.2. `02_factor_cv_fit.py`

This script performs the main estimation and result generation:

1.  **Imports:** `pandas`, `numpy`, `statsmodels.api`, `time`, `pickle`, `warnings`, and `FactorCVModel` from `01_factor_cv_model.py`.
2.  **Data Loading:**
    *   Load the final, preprocessed `df_returns` (T x N, N=95) dataset used for empirical analysis. (Replace dummy data).
    *   Ensure data is demeaned and in decimal format.
3.  **Model Instantiation:** `factor_cv_model = FactorCVModel(df_returns, k_factors=5)`.
4.  **Estimation & Timing:**
    *   `start_time = time.time()`
    *   Fit the model: `factor_cv_results = factor_cv_model.fit(method='lbfgs', maxiter=1500, disp=True, pgtol=1e-06, factr=1e7)` (adjust optimizer/settings as needed).
    *   `end_time = time.time()`
    *   `estimation_time = end_time - start_time`
    *   Determine `convergence_success` (boolean) based on `factor_cv_results.mle_retvals['warnflag'] == 0`.
    *   Store `convergence_message = factor_cv_results.mle_retvals['task']`.
5.  **Result Extraction (Conditional on `convergence_success`):**
    *   `log_likelihood_penalized = factor_cv_results.llf`
    *   `num_params = factor_cv_model.k_params`
    *   `estimated_params_constrained = factor_cv_results.params`
    *   `estimated_params_unconstrained = factor_cv_model.untransform_params(estimated_params_constrained)`
    *   Reconstruct final system matrices by calling `factor_cv_model.update(estimated_params_constrained)`:
        *   `lambda_hat = factor_cv_model.ssm['design', 0, 0].copy()`
        *   `phi_f_hat = factor_cv_model.ssm['transition', 0, 0].copy()`
        *   `sigma_eps_hat_diag = np.diag(factor_cv_model.ssm['obs_cov', 0, 0]).copy()`
        *   `sigma_nu_hat_diag = np.diag(factor_cv_model.ssm['state_cov', 0, 0]).copy()`
    *   Calculate `log_likelihood_base`: Temporarily set `factor_cv_model.STABILITY_PENALTY_WEIGHT = 0`, call `factor_cv_model.loglike(estimated_params_unconstrained)`, then restore the penalty weight.
    *   Extract filtered states: `filtered_factors = factor_cv_results.filter_results.filtered_state` (T x K)
    *   Extract filtered state covariances: `filtered_state_covariances = factor_cv_results.filter_results.filtered_state_cov` (T x K x K)
6.  **Calculate Derived Outputs (Conditional on `convergence_success`):**
    *   **Conditional Observation Covariances (`conditional_covariance_H`):** Initialize array (T x N x N). Loop `t` from `burn_in` to `T`. Calculate `Sigma_t_cv = lambda_hat @ filtered_state_covariances[t] @ lambda_hat.T + np.diag(sigma_eps_hat_diag)`. Store `Sigma_t_cv` in the array.
    *   **Standardized Residuals (`standardized_residuals`):** Initialize array (T x N). In the same loop (or a new one), calculate `mu_t_cv = lambda_hat @ filtered_factors[t]`, `e_t = returns_arr[t] - mu_t_cv`. Compute Cholesky `L_t` of `Sigma_t_cv` (handle `LinAlgError`). Calculate `z_t = np.linalg.solve(L_t, e_t)`. Store `z_t`. Create DataFrame.
    *   `standardized_residuals_post_burn`: Slice the `standardized_residuals` DataFrame to exclude the burn-in period.
7.  **Save Results:**
    *   Create `results_dict` containing:
        *   `model_name`: "Factor-CV"
        *   `log_likelihood_penalized`
        *   `log_likelihood_base`
        *   `num_params`
        *   `estimation_time`
        *   `convergence_success`
        *   `convergence_message`
        *   `standardized_residuals` (DataFrame, T x N)
        *   `standardized_residuals_post_burn` (DataFrame, (T-burn) x N)
        *   `estimated_params_constrained` (Array)
        *   `conditional_covariance_H` (Array, T x N x N)
        *   `lambda_hat` (Array, N x K)
        *   `phi_f_hat` (Array, K x K)
        *   `sigma_eps_hat_diag` (Array, N)
        *   `sigma_nu_hat_diag` (Array, K)
        *   `filtered_factors` (DataFrame/Array, T x K)
        *   `filtered_state_covariances` (Array, T x K x K)
        *   `aic` (calculated from `log_likelihood_base`, `num_params`, T-burn)
        *   `bic` (calculated from `log_likelihood_base`, `num_params`, T-burn)
    *   Define `output_path = 'outputs/empirical/insample/factorcv_results.pkl'`.
    *   Use `pickle.dump(results_dict, open(output_path, 'wb'))`.

### 3.3. `03_factor_cv_metrics.py`

1.  **Load Results:** Load the `results_dict` from `outputs/empirical/insample/factorcv_results.pkl`.
2.  **Residual Analysis:** Perform diagnostics (Ljung-Box, Jarque-Bera, ARCH-LM tests) on `standardized_residuals_post_burn`.
3.  **Reporting:** Generate summary tables (including AIC, BIC, convergence status) and potentially plots (e.g., ACF of residuals). Save outputs to `outputs/empirical/insample/` or `outputs/tables/`.

## 4. Next Steps

*   User to review and approve this plan.
*   User to toggle to ACT mode.
*   Cline will create the files (`01_...`, `02_...`) and populate them according to this plan.
