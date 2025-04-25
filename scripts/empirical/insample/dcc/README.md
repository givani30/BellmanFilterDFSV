# DCC-GARCH Model Implementation

This folder contains scripts for estimating a Dynamic Conditional Correlation (DCC) GARCH model on multivariate financial returns data. The implementation uses the `mgarch` package for model fitting and produces various outputs for model evaluation and comparison.

## Scripts Overview

1. **00_run_dcc_estimation.py**: Wrapper script that runs the entire estimation process and tracks total execution time.
2. **01_univariate_garch.py**: (Not used in current workflow) Previously used for first-stage univariate GARCH estimation.
3. **02_dcc_fit.py**: Fits the DCC-GARCH model to the original returns data and saves model outputs.
4. **03_in_sample_metrics.py**: Calculates in-sample metrics and performs diagnostic tests.
5. **04_forecast_gmv.py**: Forecasts Global Minimum Variance (GMV) portfolio weights.

## Data Outputs

The implementation produces the following outputs:

### In `data/` Directory

1. **model.pkl** (Joblib file)
   - Description: Serialized DCC-GARCH model object
   - Content: Fitted model with parameters and internal state
   - Size: ~1.1 MB

2. **model_metadata.json** (JSON file)
   - Description: Model metadata including parameters and fit statistics
   - Content:
     - Model type and distribution
     - Number of parameters (288 = 3 per univariate GARCH × 95 series + 3 DCC parameters)
     - Estimation time (total and per stage)
     - Convergence status
     - Sample size and number of series
     - Model parameters (a, b, dof)
     - Log-likelihood, AIC, and BIC values

3. **eps_tilde.npy** (NumPy array)
   - Description: Standardized residuals from the DCC model
   - Shape: (T, N) = (738, 95)
   - Content: Residuals standardized by the model's estimated conditional standard deviations
   - Used for: Diagnostic tests and DCC recursion

4. **Rt.npy** (NumPy array)
   - Description: Time series of conditional correlation matrices
   - Shape: (T, N, N) = (738, 95, 95)
   - Content: Dynamic conditional correlations between series
   - Properties: Symmetric matrices with ones on the diagonal

5. **Ht.npy** (NumPy array)
   - Description: Time series of conditional covariance matrices
   - Shape: (T, N, N) = (738, 95, 95)
   - Content: Dynamic conditional covariances between series
   - Properties: Symmetric positive definite matrices

6. **w_tplus1.npy** (NumPy array)
   - Description: One-step-ahead forecast of Global Minimum Variance portfolio weights
   - Shape: (N,) = (95,)
   - Content: Portfolio weights that minimize the forecasted portfolio variance
   - Properties: Sums to 1, can include negative values (short positions)

7. **date_index.txt** (Text file)
   - Description: Dates corresponding to the time series data
   - Format: One date per line in YYYY-MM-DD format
   - Used for: Aligning time series data with calendar dates

### In `outputs/empirical/insample/dcc/` Directory

1. **metrics_summary.json** (JSON file)
   - Description: Comprehensive summary of model fit and diagnostic metrics
   - Content:
     - All metadata from model_metadata.json
     - Results of diagnostic tests (Ljung-Box, ARCH-LM, Jarque-Bera)
     - Information criteria (AIC, BIC)

2. **standardized_residuals.csv** (CSV file)
   - Description: Standardized residuals with date index
   - Format: CSV with date column and one column per series
   - Used for: External analysis and visualization of residuals

## Key Model Parameters

- **a** (DCC parameter): Controls the impact of past standardized shocks on current correlations
- **b** (DCC parameter): Controls the persistence of correlations
- **dof** (Degrees of freedom): Parameter of the multivariate t-distribution

## Notes on Implementation

- The implementation directly uses original returns as input to the DCC model
- Numerical stability measures are implemented for log-likelihood calculation
- The model uses a multivariate t-distribution to account for fat tails
- Correlation matrices are stored in 3D format (T,N,N) rather than reshaped to 2D (T,N*N)

## Usage for Model Comparison

When comparing this DCC-GARCH model with other models (e.g., DFSV-BIF, DFSV-PF, Factor-CV), the following outputs are particularly relevant:

1. **Log-likelihood**: For comparing fit to data (higher is better)
2. **Number of parameters**: For assessing model complexity
3. **AIC/BIC**: For comparing models with different numbers of parameters
4. **Estimation time**: For comparing computational efficiency
5. **Standardized residuals**: For diagnostic tests and residual analysis
6. **Conditional covariance matrices**: For comparing volatility and correlation dynamics
7. **GMV weights**: For comparing portfolio allocation implications

The outputs are structured to facilitate direct comparison with other models using the same metrics.
