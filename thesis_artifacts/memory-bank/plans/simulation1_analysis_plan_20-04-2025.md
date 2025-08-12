# Analysis Plan: Simulation 1 Aggregated Results (20-04-2025)

This plan outlines the steps for a detailed analysis of the aggregated results from `outputs/simulation1/aggregated_simulation1_results.csv`, based on the user's requirements.

**Data File:** `outputs/simulation1/aggregated_simulation1_results.csv`

**I. Data Loading and Preprocessing:**

1.  **Load Data:** Read `outputs/simulation1/aggregated_simulation1_results.csv` into a Polars DataFrame.
2.  **Validation:**
    *   Verify column data types.
    *   Check for and handle any missing values or outliers in key metric columns (timing, RMSE, correlation). Document the handling strategy.
3.  **Preparation:** Create necessary subsets or filtered views of the DataFrame for specific analyses (e.g., isolating BIF vs. PF, specific N/K values).

**II. Computational Performance Analysis:**

1.  **Timing vs. Dimensions (N & K):**
    *   Calculate mean, standard deviation, and median `filter_time_mean` for each `filter_type` (BIF, BF, PF variants) grouped by `N` and `K` separately.
    *   **Visualization:**
        *   Generate log-scale line plots (Seaborn) showing mean `filter_time_mean` vs. `N` (for fixed K, if appropriate) and vs. `K` (for fixed N, if appropriate) for each filter.
        *   Generate box plots (Seaborn) illustrating the distribution of `filter_time_mean` across replications for different `N` and `K` values for each filter.
2.  **Comparative Timing Analysis:**
    *   **Visualization:**
        *   Create bar charts (Seaborn) comparing the mean `filter_time_mean` (with error bars representing standard deviation) for BIF, BF, and PF (grouped by `num_particles`) across representative (N, K) combinations.
        *   Generate heatmaps (Seaborn) showing mean `filter_time_mean` as a function of (`N`, `K`) pairs for BIF, BF, and each PF variant.
3.  **Output:**
    *   Generate a LaTeX table summarizing key computational performance statistics (mean, std dev, median `filter_time_mean`) across dimensions and filter types.

**III. State Estimation Accuracy Analysis:**

1.  **Accuracy vs. Dimensions (N & K):**
    *   Calculate mean, standard deviation, and median for factor RMSE (`rmse_f_mean`), factor correlation (`corr_f_mean`), volatility RMSE (`rmse_h_mean`), and volatility correlation (`corr_h_mean`) for each `filter_type` grouped by `N` and `K`.
    *   **Visualization:**
        *   Generate line plots (Seaborn) showing mean RMSE (`rmse_f_mean`, `rmse_h_mean`) vs. `N` and vs. `K` for each filter.
        *   Generate line plots (Seaborn) showing mean Correlation (`corr_f_mean`, `corr_h_mean`) vs. `N` and vs. `K` for each filter.
2.  **Comparative Accuracy Analysis:**
    *   **Visualization:**
        *   Create plots (e.g., scatter or bar plots, Seaborn) directly comparing mean RMSE and Correlation metrics between BIF, BF, and PF variants (grouped by `num_particles`).
        *   Analyze and visualize the impact of `num_particles` on PF accuracy metrics.
3.  **Statistical Analysis:**
    *   Calculate standard errors for key accuracy metrics (`rmse_*_mean`, `corr_*_mean`).
    *   *Optional:* If required for specific comparisons, perform statistical significance tests (e.g., t-tests or ANOVA) between filter performances.
4.  **Output:**
    *   Generate LaTeX tables summarizing key accuracy statistics (mean, std dev, median RMSE/Correlation) across dimensions and filter types.
    *   Save all generated accuracy plots.

**IV. Scalability and Efficiency Analysis:**

1.  **Computational Complexity:**
    *   Analyze the trends from the timing plots (Stage II) to discuss the empirical computational complexity (e.g., O(N), O(K)) for each filter.
2.  **Cost vs. Accuracy Trade-off:**
    *   **Visualization:** Generate scatter plots (Seaborn) showing mean `filter_time_mean` vs. mean accuracy metrics (e.g., `rmse_f_mean`, `rmse_h_mean`) for all filters, potentially colored by `N`, `K`, or `num_particles`.
3.  **Output:**
    *   Generate a LaTeX table summarizing observed scaling behavior.
    *   Save trade-off plots.
    *   Include discussion points on complexity, efficiency, and practical limitations in the summary report.

**V. Output Generation and Documentation:**

1.  **Directory Structure:** Create `outputs/simulation1_analysis/` containing subdirectories: `tables/`, `figures/`, `data/` (if intermediate data is saved), and `reports/`.
2.  **Tables:** Save all generated LaTeX tables in `outputs/simulation1_analysis/tables/`.
3.  **Figures:** Save all generated plots (publication quality: clear labels, titles, legends) in `outputs/simulation1_analysis/figures/` (e.g., as PNG or PDF).
4.  **Summary Report:** Create `outputs/simulation1_analysis/reports/analysis_summary_20-04-2025.md`. This report will include:
    *   Brief methodology description.
    *   Summary of key findings from each analysis stage.
    *   Discussion of limitations.
    *   Recommendations/conclusions.
    *   Embed or link to key tables and figures.

**VI. Quality Assurance:**

1.  **Validation:** Incorporate data validation checks within the analysis script(s).
2.  **Reproducibility:** Document the Python environment (library versions) used for the analysis in the summary report or a separate `requirements.txt`.

**High-Level Workflow Diagram:**

```mermaid
graph TD
    A[Load & Preprocess Data\n(aggregated_simulation1_results.csv)] --> B{Analysis Stages};
    B --> C[Computational Performance];
    B --> D[State Estimation Accuracy];
    B --> E[Scalability & Efficiency];
    C --> F{Outputs};
    D --> F;
    E --> F;
    F --> G[Tables (LaTeX)];
    F --> H[Figures (Seaborn)];
    F --> I[Summary Report (MD)];
    I --> J[Final Review];

    subgraph "Computational Performance"
        C1[Timing vs N/K Plots & Stats]
        C2[Comparative Timing Plots]
        C3[Heatmaps/Boxplots]
    end

    subgraph "State Estimation Accuracy"
        D1[RMSE/Corr vs N/K Plots & Stats (f & h)]
        D2[BIF vs PF Comparison Plots]
        D3[Particle Count Effect Analysis]
        D4[Statistical Measures]
    end

     subgraph "Scalability & Efficiency"
        E1[Complexity Discussion]
        E2[Cost vs Accuracy Trade-off Plots]
    end

    subgraph "Output Generation"
        G --> K[Save in outputs/simulation1_analysis/tables/]
        H --> L[Save in outputs/simulation1_analysis/figures/]
        I --> M[Save in outputs/simulation1_analysis/reports/]
    end