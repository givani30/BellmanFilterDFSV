# Plan: DCC-GARCH(1,1) Implementation for Empirical Analysis

**Date:** 25-04-2025

**Goal:** Implement the DCC-GARCH(1,1) workflow using the `mgarch` library to analyze the processed returns found in `scripts/empirical/vw_returns_final_with_date.csv`, adapting the provided draft plan to the existing project structure and requirements.

**Input Data:** `scripts/empirical/vw_returns_final_with_date.csv` (Assumed to contain a date column and pre-processed decimal, demeaned returns in all other columns).

**Directory Structure:**

```
scripts/empirical/
└─ insample/
   └─ dcc/                         # New directory for DCC-GARCH scripts
      ├─ 01_univariate_garch.py
      ├─ 02_dcc_fit.py
      ├─ 03_in_sample_metrics.py
      ├─ 04_forecast_gmv.py
      └─ data/                      # Intermediate data storage
         ├─ garch_outputs.npz
         ├─ Rt.arrow
         └─ w_tplus1.npy
outputs/
└─ empirical/
   └─ insample/
      └─ dcc/                      # Final output storage
         ├─ metrics_summary.json
         └─ standardized_residuals.csv
```

**Workflow Steps:**

1.  **`scripts/empirical/insample/dcc/01_univariate_garch.py`:**
    *   Load `../../vw_returns_final_with_date.csv` using `polars`.
    *   Identify date and return columns.
    *   Convert to pandas DataFrame with date index.
    *   Fit univariate GARCH(1,1)-t models in parallel (`joblib`, `arch`).
    *   Save `sigma2` (conditional variances) and `eps` (standardized residuals) as numpy arrays to `data/garch_outputs.npz`. Save date index if needed.

2.  **`scripts/empirical/insample/dcc/02_dcc_fit.py`:**
    *   Load `eps` from `data/garch_outputs.npz`.
    *   Fit DCC(1,1)-t model (`mgarch.mgarch`).
    *   Save fitted model object to `model.pkl` (`joblib`).
    *   Save conditional correlations (`dcc.corr_t`) to `data/Rt.arrow`.

3.  **`scripts/empirical/insample/dcc/03_in_sample_metrics.py`:**
    *   Load model (`model.pkl`), `eps`, `Rt`, and date index.
    *   Calculate AIC, BIC.
    *   Reconstruct conditional covariance matrices `Sigma_t`.
    *   Calculate standardized residuals `z_t = Sigma_t^{-1/2} eps_t`.
    *   Perform Ljung-Box, ARCH-LM, Jarque-Bera tests (`scipy.stats`, `statsmodels`).
    *   Save metrics (LogLik, AIC, BIC, p-values) to `../../../outputs/empirical/insample/dcc/metrics_summary.json`.
    *   Save standardized residuals `z_t` (as pandas DataFrame with dates) to `../../../outputs/empirical/insample/dcc/standardized_residuals.csv`.

4.  **`scripts/empirical/insample/dcc/04_forecast_gmv.py`:**
    *   Load model (`model.pkl`), `eps`.
    *   **Implement `forecast_sigma2` as a helper function within this script.**
    *   Calculate 1-step-ahead `Sigma_next` using DCC recursion and the helper function.
    *   Calculate GMV weights `w_gmv` based on `Sigma_next`.
    *   Save weights `w_gmv` to `data/w_tplus1.npy`.

**Dependencies:**
`polars`, `pandas`, `arch`, `mgarch`, `joblib`, `numpy`, `scipy`, `statsmodels`, `pyarrow`. Ensure these are in `requirements.txt`.

**Execution:**
Run scripts `01` through `04` sequentially. Assume execution from within the `scripts/empirical/insample/dcc/` directory.

**Coding Standards:**
Adhere to PEP 8.
