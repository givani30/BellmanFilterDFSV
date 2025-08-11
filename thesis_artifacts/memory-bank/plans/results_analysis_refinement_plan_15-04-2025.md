# Plan: Refine Analysis of Batch Optimization Results (v6)

**Date:** 15-04-2025

**Goal:** Refine the analysis script (`scripts/analysis/analyze_optimization_results.py`) to incorporate detailed configuration distinctions (T, num_particles, fix_mu), improve plot clarity, add time scaling analysis, add specific 'fix_mu' comparison (for both filters), generate summary tables, and save final outputs.

**Starting Point:** Script `scripts/analysis/analyze_optimization_results.py` with Phases 1-4 implemented (data loading, comprehensive error calculation, scalar aggregation, basic comparative analysis logging, and initial plot generation).

**Input Data:**

*   `outputs/aggregated_optimization_metrics_14-04-2025.csv`
*   `outputs/aggregated_optimization_params_14-04-2025.npz`

**Key Refinements based on Feedback:**

*   **Grouping:** Analysis will group/facet by `filter_type`, `N`, `K`, `config_T`, `config_num_particles` (for PF), and `config_fix_mu` (for both filters) where relevant.
*   **Plot Legends/Labels:** Will use combined labels (e.g., "N5-K2-T500") for clarity.
*   **Time Scaling Plots:** Added line plots for computation time vs. N, K, T.
*   **PF Distinction:** Will differentiate PF results based on `config_num_particles`.
*   **Scatter Plots:** Will use marker styles/colors to differentiate configurations (N, K, T, fix_mu, num_particles).
*   **Heatmaps:** Clarify purpose (mean element-wise bias `Est - True`) and remove for diagonal `Q_h`.
*   **Fix Mu Analysis:** Added specific comparison for BIF and PF runs based on `config_fix_mu`.
*   **Tables:** Added generation of summary tables.

**Refinement Steps (Modifications to Phase 4 & Addition of Phase 5):**

1.  **Refine Grouping Keys:**
    *   Identify the primary grouping keys used in `aggregate_scalar_metrics` (Phase 3) and `perform_comparative_analysis` / plotting functions (Phase 4).
    *   Update these keys to consistently include: `filter_type`, `N`, `K`, `config_T`, `config_num_particles` (handle nulls/fill appropriately, e.g., 0 for BIF), `config_fix_mu` (handle nulls/fill appropriately).
    *   Re-run the aggregation steps in the script (`aggregate_scalar_metrics` and potentially aggregation within `perform_comparative_analysis` if done there) using the refined keys.

2.  **Enhance Plotting Functions:**
    *   **General:** Modify existing plotting functions (`plot_scatter_comparison`, `plot_error_boxplots`, `plot_k2_eigenvalue_distributions`, `plot_error_heatmaps`) to accept the refined `df_success` or aggregated data.
    *   **Legends/Faceting:** Update `seaborn` calls to use combined configuration labels for legends or facet titles where appropriate (e.g., create a temporary 'config_label' column like "N5-K2-T500"). Facet plots primarily by N, K, T combinations. Use `hue` for `filter_type` and `style` for `config_num_particles` or `config_fix_mu` where relevant and visually feasible.
    *   **Scatter Plots:** Modify `plot_scatter_comparison` to use `style` and/or `hue` arguments in `sns.scatterplot` to differentiate points based on (N, K, T, fix_mu, num_particles) combinations or other relevant config variables.
    *   **Heatmaps:** Update `plot_error_heatmaps` title to "Mean Element-wise Bias (Est - True)". Remove the generation of the heatmap for `Q_h`.
    *   **Ellipse Plots (K=2 specific):** Ensure `plot_k2_eigenvalue_distributions` (or a similar function for ellipses) correctly handles K=2 filtering and generates comparison ellipses for `Phi_f`, `Phi_h`, `Q_h` based on *mean* estimated matrices per configuration group. Use `matplotlib.patches.Ellipse`.

3.  **Add Time Scaling Plots:**
    *   Implement a new function `plot_time_scaling(df_agg, output_dir)`.
    *   Use `seaborn.lineplot` to plot `timing_total_script_duration_s_mean` against N (with `col`/`row` facets for K/T), against K (facets N/T), and against T (facets N/K).
    *   Use `hue='filter_type'` and potentially `style` for `config_fix_mu` or `config_num_particles` on these plots.
    *   Save plots to the output directory.

4.  **Add Fix Mu Analysis (BIF & PF):**
    *   Implement a new function `analyze_fix_mu_effect(df_success, output_dir)`.
    *   Group `df_success` by `config_fix_mu` *and* other config keys (`filter_type`, N, K, T, `config_num_particles`).
    *   Calculate aggregate statistics (mean/median/std) for key parameter errors (excluding `mu` itself) and scalar metrics.
    *   Generate comparison tables (using Polars `pivot` or similar) or box plots comparing performance between `config_fix_mu=True` and `config_fix_mu=False` within each filter type and configuration. Log findings and save tables/plots.

5.  **Generate Summary Tables:**
    *   Implement a new function `generate_summary_tables(df_agg_param_errors, df_agg_scalars, output_dir)`.
    *   Use Polars to select key aggregated metrics (e.g., mean RMSE for major parameters, mean state correlation, mean time) from the refined aggregated DataFrames.
    *   Format these results into clear tables, likely pivoted to compare BIF vs PF across configurations.
    *   Save tables as CSV files in the `outputs` directory (e.g., `outputs/summary_table_param_errors_15-04-2025.csv`).

6.  **Refine Identification Difficulty Analysis:**
    *   Modify the logic within `perform_comparative_analysis` (or a dedicated function).
    *   Calculate relative error metrics (e.g., Relative Frobenius Norm) if not already done.
    *   Group by `filter_type` and `config_fix_mu` and identify parameters with the highest *average* relative error or bias across all configurations. Log these findings clearly.

7.  **Phase 5: Final Reporting and Output:**
    *   **Save Processed DataFrames:** Save the final `df_success` (with all errors) and the refined aggregated DataFrames (`df_agg_param_errors_refined`, `df_agg_scalars_refined`) to CSV/Feather files in `outputs/`.
    *   **Final Logging:** Add a concluding log message summarizing the analysis completion and locations of saved tables, plots, and processed data.

**Implementation Notes:**

*   **Script:** Modify `scripts/analysis/analyze_optimization_results.py`.
*   **Libraries:** Polars, NumPy, JAX/jnp, Seaborn, Matplotlib, Cloudpickle, `DFSVParamsDataclass`.
*   **Mode:** Use `Code` mode for implementation.

**Flowchart (Focus on Modifications & Phase 5):**

```mermaid
graph TD
    A[Start Refinement] --> B(Update Grouping Keys in Aggregation Functions);
    B --> C(Re-run Aggregation -> df_agg_refined);
    C --> D(Enhance Existing Plot Functions);
    D --> E(Update Heatmap Logic);
    E --> F(Implement Time Scaling Plots);
    C --> G(Implement Fix Mu Analysis - Group by All Keys);
    C --> H(Implement Summary Table Generation);
    F & G & H --> I(Save Tables & New Plots);
    C --> J(Refine Identification Difficulty Analysis);
    I & J --> K[Phase 4 Complete];
    K --> L(Phase 5: Save Final DataFrames - df_success_merged, df_agg_refined);
    L --> M(Add Final Summary Logging);
    M --> N[End Analysis Script];