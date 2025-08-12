# Plan: BIF Residual Analysis (29-04-2025)

This plan outlines the steps to analyze the standardized residuals from the Bellman Information Filter (BIF) empirical run, fit GARCH(1,1) models, and compare results.

**Input Data:** `outputs/empirical/insample/bif/standardized_residuals.csv`
**Output Directory:** `outputs/empirical/insample/bif/residual_analysis/`
**Significance Level:** α = 0.05

## Phase 1: Initial Residual Analysis

1.  **Setup & Data Loading:**
    *   Import necessary libraries: `polars`, `pandas`, `numpy`, `statsmodels.stats.diagnostic` (`acorr_ljungbox`, `het_arch`), `statsmodels.stats.stattools` (`jarque_bera`), `arch` (`arch_model`), `os`, `json`.
    *   Define constants: Input file path, significance level (`alpha`), output directory path. Ensure output directory exists.
    *   Load the standardized residuals CSV file into a pandas DataFrame.
    *   Determine the number of series (N) and observations (T).

2.  **Run Initial Univariate Tests:**
    *   Create data structures (e.g., dictionaries or DataFrames) to store results (p-values, pass/fail status) for each test and series.
    *   Iterate through each column (residual series):
        *   **Ljung-Box (Squared):** Apply `acorr_ljungbox` for lags 5, 10, 15, 20. Store p-values and pass status (p > alpha).
        *   **ARCH-LM:** Apply `het_arch` for lags 5, 10. Store p-values and pass status (p > alpha).
        *   **Jarque-Bera:** Apply `jarque_bera`. Store p-value and pass status (p > alpha).
        *   Handle potential errors.

3.  **Aggregate & Store Initial Results:**
    *   Calculate the pass rate (percentage of series passing the test) for each test type and lag combination.
    *   Store these aggregated pass rates.

## Phase 2: GARCH Modeling and Analysis

4.  **Fit GARCH(1,1) Models:**
    *   Create data structures to store GARCH results: standardized residuals, parameter estimates (α₁, β₁), parameter p-values, parameter variance-covariance matrix, α₁ + β₁ sum.
    *   Iterate through each original residual series:
        *   Instantiate and fit an `arch_model` (GARCH(1,1), mean='Zero', vol='Garch', p=1, q=1).
        *   Handle potential convergence errors.
        *   If converged:
            *   Extract standardized residuals.
            *   Extract parameter estimates (alpha[1], beta[1]) and their p-values.
            *   Extract the variance-covariance matrix of the parameters.
            *   Calculate the sum `alpha[1] + beta[1]`.
            *   Store all extracted information.
        *   If not converged, store NaNs or indicate failure.

5.  **Run Post-GARCH Univariate Tests:**
    *   Create data structures similar to Step 2 for GARCH residuals.
    *   Iterate through each *GARCH* standardized residual series obtained in Step 4:
        *   Perform Ljung-Box (Squared), ARCH-LM, and Jarque-Bera tests as in Step 2.
        *   Store p-values and pass/fail status.

6.  **Aggregate & Store Post-GARCH Results:**
    *   Calculate the pass rate for each test type and lag combination based on the GARCH residuals.
    *   Store these aggregated pass rates.

## Phase 3: Comparison and Reporting

7.  **Create Comparison Table:**
    *   Generate a pandas DataFrame comparing the pass rates from Step 3 (Initial Residuals) and Step 6 (GARCH Residuals) for each test type and lag.

8.  **Analyze GARCH Parameters:**
    *   Perform Wald test for H₀: α₁ + β₁ = 0 vs H₁: α₁ + β₁ > 0 for each converged model using the stored variance-covariance matrix. Store the p-value.
    *   Calculate the percentage of successfully fitted models where:
        *   α₁ is significant (p < 0.05).
        *   β₁ is significant (p < 0.05).
        *   Wald test for sum rejects H₀ (p < 0.05).
        *   Sum α₁ + β₁ > 0.9.
        *   Sum α₁ + β₁ > 0.95.
        *   Wald test rejects H₀ (p < 0.05) AND Sum α₁ + β₁ > 0.9.
    *   Store these percentages.

9.  **Save Outputs:**
    *   Save the comparison table (Step 7) as `outputs/empirical/insample/bif/residual_analysis/test_pass_rate_comparison.csv`.
    *   Save the GARCH parameter analysis results (Step 8) as `outputs/empirical/insample/bif/residual_analysis/garch_parameter_analysis.csv`.
    *   Optionally, save detailed results per series if needed later.

## Workflow Diagram

```mermaid
graph TD
    A[Start: Load BIF Residuals CSV] --> B{Run Initial Univariate Tests};
    B -- Ljung-Box (Sq) --> C[Store Initial LB Results];
    B -- ARCH-LM --> D[Store Initial ARCH Results];
    B -- Jarque-Bera --> E[Store Initial JB Results];
    C --> F{Aggregate Initial Pass Rates};
    D --> F;
    E --> F;
    F --> G[Store Aggregated Initial Results];
    A --> H{Fit GARCH(1,1) per Series};
    H --> I[Extract GARCH Residuals];
    H --> J[Extract GARCH Params, p-vals, Cov Matrix];
    I --> K{Run Post-GARCH Univariate Tests};
    K -- Ljung-Box (Sq) --> L[Store Post-GARCH LB Results];
    K -- ARCH-LM --> M[Store Post-GARCH ARCH Results];
    K -- Jarque-Bera --> N[Store Post-GARCH JB Results];
    L --> O{Aggregate Post-GARCH Pass Rates};
    M --> O;
    N --> O;
    O --> P[Store Aggregated Post-GARCH Results];
    J --> Q{Analyze GARCH Parameters (inc. Wald Test)};
    Q -- α₁ Significance --> R[Calc % Sig α₁];
    Q -- β₁ Significance --> S[Calc % Sig β₁];
    Q -- Wald Test & Sum --> T[Calc % Sig Sum & >0.9];
    G --> U{Create Comparison Table};
    P --> U;
    U --> V[Save Comparison Table CSV];
    R --> W[Save GARCH Param Analysis CSV];
    S --> W;
    T --> W;
    V --> X[End];
    W --> X;

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#ccf,stroke:#333,stroke-width:2px
    style Q fill:#ccf,stroke:#333,stroke-width:2px
    style U fill:#cfc,stroke:#333,stroke-width:2px
    style X fill:#f9f,stroke:#333,stroke-width:2px