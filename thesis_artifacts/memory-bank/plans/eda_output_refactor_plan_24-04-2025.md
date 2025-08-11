# Plan: Refactor EDA Script Output Paths

**Date:** 24-04-2025

**Goal:** Modify `scripts/empirical/eda.py` to save all specified figures (`.png`) and tables (`.tex`) to the `outputs/eda/` directory, and also save the calculated PCA results.

**Plan:**

1.  **Define Output Directory:**
    *   Near the beginning of the script (e.g., after imports), define a variable for the output directory:
        ```python
        output_dir = "outputs/eda/"
        ```
2.  **Create Output Directory:**
    *   Immediately after defining `output_dir`, ensure the directory exists using the `os` module (which is already imported):
        ```python
        os.makedirs(output_dir, exist_ok=True)
        ```
3.  **Modify Figure Saving Paths:**
    *   **Figure 1 (Returns Time Series):** Update path to `os.path.join(output_dir, "figure1_returns_timeseries.png")`.
    *   **Figure 2 (Correlation Heatmap):** Update path to `os.path.join(output_dir, "figure2_avg_correlation_heatmap.png")`.
    *   **Figure 5 (PCA Scree Plot):** Update path to `os.path.join(output_dir, "figure5_pca_scree_plot.png")`.
    *   **Figure 3 (Average Characteristics Time Series):** Update path to `os.path.join(output_dir, "figure3_avg_characteristics_timeseries.png")`.
    *   **Figure 4 (Average Market Cap Bar Chart):** Update path to `os.path.join(output_dir, "figure4_avg_mkt_cap_by_size.png")`.
4.  **Add Table Saving (LaTeX):**
    *   **Table 1 (Returns Summary Stats):** Add code to save `table1_data_pandas` to `os.path.join(output_dir, 'table1_summary_stats_returns.tex')` using `to_latex()`.
    *   **Table 2 (Characteristics Summary Stats):** Add code to save `formatted_table2` (converted to pandas) to `os.path.join(output_dir, 'table2_summary_stats_characteristics.tex')` using `to_latex()`.
5.  **Add PCA Results Saving:**
    *   Add code to save the `pca_results` Polars DataFrame to `os.path.join(output_dir, 'pca_variance_explained.csv')` using `write_csv()`.

**Workflow Diagram:**

```mermaid
graph TD
    A[Start eda.py] --> B{Define output_dir = "outputs/eda/"};
    B --> C[os.makedirs(output_dir)];
    C --> D[Load & Process Data];

    subgraph "Returns Analysis"
        D --> E{Calculate Table 1 Stats};
        E --> F[Save Table 1 (.tex) --> output_dir];
        D --> G{Generate Figure 1};
        G --> H[Save Figure 1 (.png) --> output_dir];
        D --> I{Generate Figure 2};
        I --> J[Save Figure 2 (.png) --> output_dir];
        D --> K{Perform PCA};
        K --> L[Save PCA Results (.csv) --> output_dir];
        L --> M{Generate Figure 5};
        M --> N[Save Figure 5 (.png) --> output_dir];
    end

    subgraph "Characteristics Analysis"
        D --> O{Calculate Table 2 Stats};
        O --> P[Save Table 2 (.tex) --> output_dir];
        D --> Q{Generate Figure 3};
        Q --> R[Save Figure 3 (.png) --> output_dir];
        D --> S{Generate Figure 4};
        S --> T[Save Figure 4 (.png) --> output_dir];
    end

    F --> U[End];
    H --> U;
    J --> U;
    N --> U;
    P --> U;
    R --> U;
    T --> U;