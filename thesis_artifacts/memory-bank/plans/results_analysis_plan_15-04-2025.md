# Plan: Comprehensive Analysis of Batch Optimization Results (v3)

**Date:** 15-04-2025

**Analysis Goal:** Evaluate and compare the performance of the Bellman Information Filter (BIF) and Particle Filter (PF) in estimating the DFSV model parameters (scalars, vectors, and matrices) and latent states across different simulation configurations (N, K, T). Assess convergence behavior, computational time, parameter estimation accuracy (using element-wise, norm-based, and decomposition-based metrics), identification difficulty, and state estimation accuracy, focusing on the comparison between filter types.

**Input Data:**

*   `outputs/aggregated_optimization_metrics_14-04-2025.csv` (Scalar metrics, config, status, linked by `unique_id`)
*   `outputs/aggregated_optimization_params_14-04-2025.npz` (Dictionaries mapping `unique_id` to `true_params` and `estimated_params` `DFSVParamsDataclass` objects)

**Critical Questions Addressed by this Plan:**

1.  **Parameter Convergence (BIF vs PF):** How do various accuracy metrics (RMSE, MAE, Bias, Frobenius Norm, Eigenvalue Errors, Det/Trace Diffs) compare between BIF and PF across different configurations? (Phase 2 & 4)
2.  **Identification Difficulty:** Which specific parameters (e.g., `Phi_f`, `Q_h`, `lambda_r`, `sigma2`) or aspects of parameters (e.g., specific elements, overall scale, dynamic properties via eigenvalues) are hardest to estimate accurately for each filter type? (Phase 2 & 4)
3.  **Impact of N, K, T:** How do parameter estimation accuracy and state estimation accuracy change for BIF and PF as N, K, and T vary? Does the relative performance difference between BIF and PF change with configuration? (Phase 4)
4.  **Numerical Analysis & Figures:** The plan includes generating both summary tables (numerical analysis) and visualizations (figures) to present the findings clearly. (Phase 4 & 5)

**Analysis Phases:**

**Phase 1: Data Loading and Preparation**

1.  **Load Data:**
    *   Load `outputs/aggregated_optimization_metrics_14-04-2025.csv` into a Pandas DataFrame (`df_metrics`).
    *   Load `outputs/aggregated_optimization_params_14-04-2025.npz` using `np.load(..., allow_pickle=True)`. Extract the dictionaries: `true_params_dict = loaded_data['true_params'].item()`, `estimated_params_dict = loaded_data['estimated_params'].item()`.
2.  **Initial Cleaning & Filtering:**
    *   Inspect `df_metrics` for data types and potential inconsistencies.
    *   Filter out rows where essential data loading failed (`json_read_error == True` or `pkl_read_error == True`). Log the number of discarded rows.
    *   **Filter for successful runs:** Create a primary analysis DataFrame `df_success = df_metrics[df_metrics['results_success'] == True].copy()`. Failed runs can be analyzed separately later if needed.
    *   Verify `unique_id` integrity.

**Phase 2: Comprehensive Parameter Estimation Accuracy Analysis (on Successful Runs)**

1.  **Define Enhanced Error Calculation Function:**
    *   Create/import a function `calculate_comprehensive_param_errors(true_params: DFSVParamsDataclass, estimated_params: DFSVParamsDataclass) -> dict`.
    *   This function will compare corresponding fields in the two dataclass objects.
    *   **For All Parameters:**
        *   Element-wise Bias: `mean(Est - True)`
        *   Element-wise MAE: `mean(abs(Est - True))`
        *   Element-wise RMSE: `sqrt(mean((Est - True)**2))`
    *   **For Matrix Parameters (`lambda_r`, `Phi_f`, `Phi_h`, `Q_h`):**
        *   Frobenius Norm Difference: `||Est - True||_F` (using `jnp.linalg.norm`)
        *   Relative Frobenius Norm Difference: `||Est - True||_F / ||True||_F` (handle potential zero norm for `True`).
    *   **For Square Matrices (`Phi_f`, `Phi_h`, `Q_h`):**
        *   Eigenvalue Comparison:
            *   Calculate eigenvalues for `True` and `Est` (use `jnp.linalg.eigvals`, handle errors/complex results gracefully).
            *   Sort eigenvalues (e.g., by real part or magnitude).
            *   Calculate RMSE, MAE, Bias between sorted eigenvalues.
    *   **For Covariance Matrix (`Q_h`):**
        *   Log Determinant Difference: `log(det(Est)) - log(det(True))` (use `jnp.linalg.slogdet` for stability, handle non-PSD cases).
        *   Trace Difference: `trace(Est) - trace(True)`
    *   The function returns a flat dictionary including all calculated metrics (e.g., `param_lambda_r_rmse`, `param_Phi_f_frobenius_rel_diff`, `param_Phi_h_eigenvalue_mae`, `param_Q_h_logdet_diff`, `param_mu_bias`, etc.). Implement robustly with error handling (e.g., return NaN if a calculation fails).
2.  **Calculate Per-Replicate Errors:**
    *   Initialize an empty list `param_errors_list`.
    *   Iterate through the rows of `df_success`.
    *   For each row (`unique_id`):
        *   Retrieve `true_params = true_params_dict.get(unique_id)` and `estimated_params = estimated_params_dict.get(unique_id)`.
        *   If both exist:
            *   Call `errors = calculate_comprehensive_param_errors(true_params, estimated_params)`.
            *   Add `{'unique_id': unique_id, **errors}` to `param_errors_list`.
        *   Else: Log a warning about missing parameter objects for this `unique_id`.
3.  **Merge and Aggregate Errors:**
    *   Convert `param_errors_list` to a DataFrame `df_param_errors`.
    *   Merge `df_param_errors` with `df_success` on `unique_id`.
    *   Group the merged DataFrame by configuration (`filter_type`, `N`, `K`, `config_T`) and calculate aggregate statistics (mean, median, std dev) for each parameter error metric (e.g., mean `param_lambda_r_rmse`).

**Phase 3: Scalar Metric Analysis (on Successful Runs)**

1.  **Convergence & Timing:**
    *   Group `df_success` by configuration.
    *   Calculate success rate (should be 100% by definition of `df_success`, but good to verify).
    *   Analyze distributions of steps (`results_steps`) and total script time (`timing_total_script_duration_s`) using mean, median, std dev, box plots.
2.  **State Estimation Accuracy:**
    *   Group `df_success` by configuration.
    *   Analyze distributions of pre-calculated state accuracy metrics (`accuracy_state_estimation_...`) using mean, median, std dev, box plots.
3.  **Relative Loss Analysis:**
    *   Calculate a new column in `df_success`: `loss_diff = results_final_loss - results_loss_at_true_params`. *Note: This difference is meaningful only within a specific filter type and configuration.*
    *   Group `df_success` by configuration and filter type.
    *   Analyze the distribution of `loss_diff` (e.g., mean, median, box plots) to see how close the optimization got to the loss at true parameters *for that specific filter's objective function*.

**Phase 4: Comparative Analysis & Visualization**

1.  **Filter Comparison (BIF vs PF):**
    *   Using the aggregated results (parameter errors from Phase 2, scalar metrics from Phase 3):
        *   For each configuration (N, K, T), create tables/plots directly comparing BIF and PF on:
            *   Mean/Median Parameter Errors (RMSE, MAE, Bias, Frobenius Norm, Eigenvalue Errors, Det/Trace Diffs for each parameter).
            *   Mean/Median State Estimation Accuracy (RMSE, Correlation).
            *   Mean/Median Optimization Steps & Time.
2.  **Identification Difficulty:**
    *   Analyze the aggregated parameter errors (Phase 2) across all configurations. Identify which parameters consistently show the highest relative errors (e.g., RMSE scaled by parameter magnitude if appropriate, Relative Frobenius Norm) or largest bias for BIF and PF. Use eigenvalue errors for dynamic matrices (`Phi_f`, `Phi_h`).
    *   Visualize this (e.g., heatmap or grouped bar chart of average errors per parameter per filter).
3.  **Impact of N, K, T:**
    *   Plot key performance metrics (e.g., mean Frobenius norm diff for `Phi_f`, mean eigenvalue RMSE for `Phi_h`, mean `Q_h` log-det diff, mean state correlation) against N, K, and T, faceted by filter type.
    *   Analyze trends: Does estimation accuracy degrade faster for one filter as N/K increases? Does the time difference widen?
4.  **Visualization (Expanded):**
    *   **Scatter Plots:** Estimated vs. True elements for key parameters (e.g., diagonals of `Phi_f`, `Q_h`, elements of `sigma2`).
    *   **Heatmaps:** Average element-wise difference matrix (`mean(Est - True)`) for matrix parameters.
    *   **Box Plots:** Comparing distributions of key error metrics (RMSE, Frobenius diff, Eigenvalue RMSE) between BIF/PF, faceted by N/K/T.
    *   **Ellipse Plots (K=2 specific):** For configurations where K=2, generate plots comparing the 2x2 covariance ellipses implied by the true vs. estimated `Phi_f`, `Phi_h`, `Q_h` matrices (e.g., average estimate or representative examples).

**Phase 5: Reporting and Output**

1.  **Summary Tables:** Create concise tables summarizing the key findings, directly addressing the critical questions (comparative parameter/state accuracy, identification difficulty, N/K/T impact). Use Polars for table creation if preferred (`.clinerules`).
2.  **Save Plots:** Save generated visualizations (prefer Seaborn over Matplotlib - `.clinerules`) with clear labels to `outputs/analysis_plots_15-04-2025/`.
3.  **Save Processed DataFrames:** Save the main DataFrame containing merged metrics and calculated parameter errors (`df_success_merged`) to a CSV or Feather file for potential later use.
4.  **Documentation:** Write a summary document (e.g., Markdown) interpreting the tables and plots, answering the critical questions, and noting any limitations or interesting observations.

**Implementation Notes:**

*   **Tooling:** Jupyter Notebook or a Python script (`scripts/analysis/analyze_optimization_results.py`).
*   **Libraries:** Pandas (or Polars - `.clinerules`), NumPy, Matplotlib/Seaborn (prefer Seaborn - `.clinerules`), Cloudpickle. Need `DFSVParamsDataclass` definition (from `src/bellman_filter_dfsv/models/dfsv.py`). Use JAX/`jnp` for numerical calculations within error functions where feasible.
*   **Mode:** `EmpiricalAnalyst` is the most suitable mode for implementing this analysis.

**Flowchart:**

```mermaid
graph TD
    A[Start Analysis] --> B(Load CSV & NPZ Data);
    B --> C(Filter for Successful Runs);
    C --> D{Calculate Per-Replicate Comprehensive Parameter Errors (Element-wise, Norms, Eigen, Det/Trace)};
    D --> E(Merge Errors into Success DataFrame);
    E --> F{Aggregate Comprehensive Errors by Config};
    C --> G{Analyze Scalar Metrics by Config (Convergence, State Acc., Time, Relative Loss)};
    F --> H(Compare BIF vs PF: Comprehensive Param Errors);
    G --> I(Compare BIF vs PF: State Acc., Time);
    H & I --> J(Analyze Identification Difficulty using Comprehensive Metrics);
    J --> K(Analyze Impact of N, K, T on Comprehensive Metrics);
    K --> L(Generate Visualizations - Scatter, Heatmap, Boxplot, Ellipse);
    F & G & J & K --> M(Create Summary Tables - Comprehensive Metrics);
    L --> N(Save Plots);
    M --> O(Document Findings);
    E --> P{Optionally Save Merged Error DF};
    N & O & P --> Q[End Analysis];