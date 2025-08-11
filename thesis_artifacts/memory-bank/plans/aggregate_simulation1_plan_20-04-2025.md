# Plan: Aggregate Simulation 1 Results

**Date:** 20-04-2025

**Objective:** Create a Python script (`scripts/simulation1/aggregate_simulation1_results.py`) that reads the large CSV file `outputs/simulation1/simulation1_merged_results.csv`, aggregates the data based on configuration parameters, calculates specified summary statistics (mean/std/median), renames filter-specific columns to generic names based on the `filter_type` within each group, and saves the aggregated results to `outputs/simulation1/aggregated_simulation1_results.csv`.

**Libraries:** Polars

**Input File:** `outputs/simulation1/simulation1_merged_results.csv`
**Output File:** `outputs/simulation1/aggregated_simulation1_results.csv`
**Script Location:** `scripts/simulation1/aggregate_simulation1_results.py`

**Plan Details:**

1.  **Setup:**
    *   Import `polars` as `pl`, `pathlib`.
    *   Define paths, grouping columns (`grouping_cols = ["N", "K", "T", "filter_type", "num_particles"]`), timing columns (`timing_cols`), and array columns (`array_cols`).
    *   Define mappings for generic names (e.g., `generic_timing_map`, `generic_array_map`).

2.  **Load Data (Lazy):**
    *   Use `pl.scan_csv(input_file_path)`.

3.  **Pre-processing (Array Columns):**
    *   Iterate through `array_cols`, parse string representations (`.str.json_decode()`), calculate intra-array means (`.list.mean()`), and create new columns (e.g., `bf_rmse_f_mean`).

4.  **Aggregation (Revised for Generic Names):**
    *   Group the LazyFrame using `.group_by(grouping_cols)`.
    *   Apply aggregation functions using `.agg()`:
        *   Aggregate non-filter-specific timing columns directly.
        *   Use conditional aggregation (`pl.when().then().otherwise()`) based on `pl.col("filter_type")` to select the correct filter-specific time column (`bf_filter_time`, etc.) *before* applying `mean`, `std`, `median`. Alias the result to the generic name (e.g., `filter_time_mean`).
        *   Use conditional aggregation similarly on the pre-calculated mean columns (e.g., `bf_rmse_f_mean`) based on `filter_type` before applying `mean`, `std`, `median`. Alias the result to the generic name (e.g., `rmse_f_mean_mean`). Repeat for all generic metrics (`rmse_f`, `rmse_h`, `corr_f`, `corr_h`).

5.  **Collect & Save:**
    *   Execute the query using `.collect()`.
    *   Save the resulting aggregated DataFrame using `aggregated_df.write_csv(output_file_path)`.

6.  **Script Structure & Standards:**
    *   Organize within `main()`, use `if __name__ == "__main__":`.
    *   Consider `argparse` for flexibility.
    *   Add docstrings (Google style), type hints.
    *   Ensure PEP 8 compliance.
    *   Add basic error handling (e.g., file existence check).

**Workflow Diagram (Mermaid):**

```mermaid
graph TD
    A[Start Script] --> B{Define Paths, Cols, Mappings};
    B --> C[Scan CSV (LazyFrame)];
    C --> D{Parse Array Strings};
    D --> E[Calculate Intra-Array Means (e.g., bf_rmse_f_mean)];
    E --> F{Group By Config Vars};
    F --> G[Aggregate Non-Filter Timing Cols];
    G --> H{Conditionally Select & Aggregate Filter Timing Cols (-> filter_time_mean, etc.)};
    H --> I{Conditionally Select & Aggregate Array Mean Cols (-> rmse_f_mean_mean, etc.)};
    I --> J[Collect Results (DataFrame)];
    J --> K{Write Aggregated CSV};
    K --> L[End Script];

    subgraph Preprocessing
        D; E;
    end

    subgraph Aggregation
        F; G; H; I;
    end