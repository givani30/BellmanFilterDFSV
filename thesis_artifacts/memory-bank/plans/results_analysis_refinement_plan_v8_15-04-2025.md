# Plan: Refine Analysis Script (v8)

**Date:** 15-04-2025

**Goal:** Overhaul `scripts/analysis/analyze_optimization_results.py` to:

* Treat BIF, PF (1000 particles), and PF (5000 particles) as distinct configurations.
* Correct plot faceting and legends to accurately represent N, K, T combinations.
* Improve clarity and structure of scatter plots and eigenvalue/ellipse plots.
* Remove complexity analysis from time scaling plots.
* Ensure 'fix_mu' analysis applies to all relevant filter configurations.
* Generate comprehensive summary tables.
* Address Seaborn warnings.
* Implement final output saving.

**Starting Point:** Script `scripts/analysis/analyze_optimization_results.py` with initial implementations of Phases 1-4 (data loading, comprehensive error calculation, scalar aggregation, basic comparative analysis logging, and initial plot generation).

**Input Data:**

* `outputs/aggregated_optimization_metrics_14-04-2025.csv`
* `outputs/aggregated_optimization_params_14-04-2025.npz`

**Refinement Steps:**

1. **Refactor Filter Configuration Handling:**
    * **Action:** Modify the script (likely near the start of `main` or in helper functions).
    * **Logic:**
        * Create a new column `filter_config` in `df_success` (e.g., "BIF", "PF-1000", "PF-5000") based on `filter_type` and `config_num_particles`. Fill nulls appropriately (e.g., `config_num_particles` -> 0 for BIF).
        * Update *all* subsequent `group_by`, `filter`, `hue`, and `style` operations in aggregation (`calculate_scalar_metrics`, `generate_summary_tables`, aggregation within `analyze_fix_mu_effect`), analysis (`analyze_identification_difficulty`), and plotting functions to use `filter_config` instead of `filter_type` where appropriate to distinguish the three configurations.
        * Ensure `config_fix_mu` nulls are handled (e.g., fill with -1) before grouping.

2. **Address Seaborn Warnings & Faceting:**
    * **Action:** Modify plotting functions.
    * **Logic:** Add explicit `.cast()` calls in Polars for columns used as numerical axes or categorical distinctions (N, K, config_T, config_num_particles) *before* passing the DataFrame to Seaborn functions. Ensure faceting logic correctly uses the specific (N, K, T) combinations present in the data.

3. **Revise Time Scaling Plots (`plot_time_scaling`):**
    * **Action:** Modify the function.
    * **Remove:** Delete the code related to polynomial fitting, R², complexity class annotation.
    * **Update:** Ensure `hue='filter_config'` and `style='config_fix_mu'`. Correct faceting to show valid (N, K, T) combinations. Remove automatic log scaling unless clearly needed.

4. **Overhaul Scatter Plots (`plot_scatter_comparison`):**
    * **Action:** Refactor the function.
    * **Structure:** Generate a separate plot *for each parameter* (e.g., `lambda_r`, `Phi_f` diagonals, `Phi_h` diagonals, `Q_h` diagonals, `sigma2`).
    * **Plot Content:** Bias vs. RMSE.
    * **Encoding:** Use `hue='filter_config'`, `style='config_T'`.
    * **Faceting:** Use `col='N'`, `row='K'`, ensuring only valid (N, K) pairs are displayed.

5. **Revise Eigenvalue/Ellipse Plots (`plot_k2_eigenvalue_distributions`):**
    * **Action:** Modify the function (assuming K=2 data exists).
    * **Ellipse Plot:**
        * Calculate mean estimated matrices (`Phi_f`, `Phi_h`, `Q_h`) grouped by the *full* refined configuration.
        * Use distinct, easily distinguishable colors and/or line styles for the ellipses based on `filter_config` and `config_fix_mu`. Add a clear legend.
        * Plot the true ellipse for reference.
        * Review and adjust the `xlim`/`ylim` based on the actual range of eigenvalues, especially for `Phi_h`.
    * **Eigenvalue RMSE Boxplot:** Update to use `hue='filter_config'` and facet correctly by (N, T) for K=2.

6. **Update Fix Mu Analysis (`analyze_fix_mu_effect`):**
    * **Action:** Modify the function.
    * **Grouping/Comparison:** Ensure comparisons are made between Fixed vs. Unfixed runs *within* each `filter_config` (BIF, PF-1000, PF-5000) and other config keys. Update logging/outputs.

7. **Update Summary Tables (`generate_summary_tables`):**
    * **Action:** Modify the function.
    * **Aggregation/Structure:** Ensure aggregation uses `filter_config` and other refined keys. Format tables to clearly compare the three `filter_config` types across N, K, T, and `config_fix_mu`. Use pivoting or multi-level indexing.
    * **Save:** Save tables as CSV files in `outputs/`.

8. **Update Identification Difficulty Analysis (`analyze_identification_difficulty`):**
    * **Action:** Modify the function.
    * **Grouping:** Group by `filter_config` and `config_fix_mu`.
    * **Logging:** Update logging to reflect the three distinct filter configurations.

9. **Implement Final Output Saving (Phase 5):**
    * **Action:** Add code at the end of the `main` function.
    * **Aggregate Param Errors:** Aggregate `param_*` columns from `df_success` using the refined grouping keys into `df_agg_param_errors`.
    * **Save DataFrames:** Save `df_success`, `df_agg_scalars`, and `df_agg_param_errors` to timestamped CSV files in `outputs/`.
    * **Final Logging:** Add concluding log message.

**Implementation Strategy:**

Delegate these refinements to Code mode iteratively:

1. Steps 1 & 2 (Filter Config Column, Grouping Keys, Data Types).
2. Step 3 & 4 (Revise Time Scaling & Scatter Plots).
3. Step 5 & 6 (Revise Eigenvalue/Ellipse & Fix Mu Analysis).
4. Step 7 & 8 (Update Summary Tables & Identification Difficulty).
5. Step 9 (Implement Final Output Saving).

**Implementation Notes:**

* **Script:** Modify `scripts/analysis/analyze_optimization_results.py`.
* **Libraries:** Polars, NumPy, JAX/jnp, Seaborn, Matplotlib, Cloudpickle, `DFSVParamsDataclass`.
* **Mode:** Use `Code` mode for implementation.
