import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
import os
import json
import warnings

# --- Configuration ---
# INPUT_FILE = "outputs/empirical/insample/bif_predicted_residuals/standardized_residuals.csv"
INPUT_FILE = "outputs/empirical/insample/bif_predicted_residuals/standardized_residuals.csv"
OUTPUT_DIR = "outputs/empirical/insample/bif/residual_analysis/"
ALPHA = 0.05
LB_LAGS = [5, 10, 15, 20]
ARCH_LAGS = [5, 10]

# --- Helper Function ---
def run_tests_for_series(series, alpha, lb_lags, arch_lags):
    """Runs diagnostic tests on a single residual series."""
    results = {
        'ljung_box_sq': {'p_value': {}, 'pass': {}},
        'arch_lm': {'p_value': {}, 'pass': {}},
        'jarque_bera': {'p_value': None, 'pass': None}
    }
    series = series.dropna()
    if len(series) < max(lb_lags + arch_lags): # Ensure enough data points
        warnings.warn(f"Skipping series {series.name} due to insufficient data points ({len(series)}).")
        return None

    # Ljung-Box on Squared Residuals
    try:
        lb_test_result = acorr_ljungbox(series**2, lags=lb_lags, return_df=True)
        for lag in lb_lags:
            p_val = lb_test_result.loc[lag, 'lb_pvalue']
            results['ljung_box_sq']['p_value'][lag] = p_val
            results['ljung_box_sq']['pass'][lag] = p_val > alpha
    except Exception as e:
        warnings.warn(f"Ljung-Box test failed for series {series.name}: {e}")
        for lag in lb_lags:
            results['ljung_box_sq']['p_value'][lag] = np.nan
            results['ljung_box_sq']['pass'][lag] = np.nan # Treat failure as non-pass for aggregation

    # ARCH-LM Test
    try:
        # Note: het_arch returns lm_stat, lm_p_value, f_stat, f_p_value
        arch_test_result = het_arch(series, nlags=max(arch_lags))
        # We need p-values for specific lags, statsmodels doesn't directly provide this
        # Re-running for each lag set required by the plan
        for lag in arch_lags:
             arch_test_lag_result = het_arch(series, nlags=lag)
             p_val = arch_test_lag_result[1] # lm_p_value
             results['arch_lm']['p_value'][lag] = p_val
             results['arch_lm']['pass'][lag] = p_val > alpha
    except Exception as e:
        warnings.warn(f"ARCH-LM test failed for series {series.name}: {e}")
        for lag in arch_lags:
            results['arch_lm']['p_value'][lag] = np.nan
            results['arch_lm']['pass'][lag] = np.nan # Treat failure as non-pass

    # Jarque-Bera Test
    try:
        jb_test_result = jarque_bera(series)
        p_val = jb_test_result[1]
        results['jarque_bera']['p_value'] = p_val
        results['jarque_bera']['pass'] = p_val > alpha
    except Exception as e:
        warnings.warn(f"Jarque-Bera test failed for series {series.name}: {e}")
        results['jarque_bera']['p_value'] = np.nan
        results['jarque_bera']['pass'] = np.nan # Treat failure as non-pass

    return results

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    try:
        residuals_df = pd.read_csv(INPUT_FILE, index_col=0) # Assuming first col is index
        residuals_df.index = pd.to_datetime(residuals_df.index) # Ensure datetime index if applicable
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        exit(1)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit(1)

    N = residuals_df.shape[1] # Number of series
    T = residuals_df.shape[0] # Number of observations
    print(f"Loaded {N} series with {T} observations each from {INPUT_FILE}")

    # Store results per series
    all_series_results = {}
    valid_series_count = 0

    # Run tests for each series
    print("Running initial univariate tests...")
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always") # Capture all warnings
        for col in residuals_df.columns:
            series_result = run_tests_for_series(residuals_df[col], ALPHA, LB_LAGS, ARCH_LAGS)
            if series_result is not None:
                all_series_results[col] = series_result
                valid_series_count += 1

        # Print captured warnings
        if caught_warnings:
            print("\n--- Warnings during test execution ---")
            for w in caught_warnings:
                print(w.message)
            print("-------------------------------------\n")


    if valid_series_count == 0:
        print("Error: No valid series found or processed. Cannot calculate pass rates.")
        exit(1)

    print(f"Finished tests for {valid_series_count} valid series.")

    # Aggregate results (calculate pass rates)
    aggregated_results = {
        'ljung_box_sq': {lag: 0 for lag in LB_LAGS},
        'arch_lm': {lag: 0 for lag in ARCH_LAGS},
        'jarque_bera': 0
    }
    total_jb_tests = 0

    for series_name, results in all_series_results.items():
        # Ljung-Box Squared
        for lag_str, passed in results['ljung_box_sq']['pass'].items():
            # Convert string lag to int, handling both string and int keys
            lag = int(lag_str) if isinstance(lag_str, str) else lag_str

            if passed and passed is not np.nan: # Check for True and not NaN
                aggregated_results['ljung_box_sq'][lag] += 1

        # ARCH-LM
        for lag_str, passed in results['arch_lm']['pass'].items():
            # Convert string lag to int, handling both string and int keys
            lag = int(lag_str) if isinstance(lag_str, str) else lag_str

            if passed and passed is not np.nan: # Check for True and not NaN
                aggregated_results['arch_lm'][lag] += 1

        # Jarque-Bera

        if results['jarque_bera']['pass'] is not None: # Check if test ran successfully
            total_jb_tests += 1
            jb_pass = results['jarque_bera']['pass']
            if jb_pass and jb_pass is not np.nan: # Check for True and not NaN
                aggregated_results['jarque_bera'] += 1

    # Calculate pass rates
    pass_rates = {
        'ljung_box_sq': {lag: (count / valid_series_count) * 100 for lag, count in aggregated_results['ljung_box_sq'].items()},
        'arch_lm': {lag: (count / valid_series_count) * 100 for lag, count in aggregated_results['arch_lm'].items()},
        'jarque_bera': (aggregated_results['jarque_bera'] / total_jb_tests) * 100 if total_jb_tests > 0 else 0
    }

    print("\n--- Aggregated Initial Test Pass Rates (%) ---")
    print(json.dumps(pass_rates, indent=4))
    print("--------------------------------------------")

    # Save detailed results
    with open(os.path.join(OUTPUT_DIR, "initial_detailed_results.json"), 'w') as f:
        # Convert numpy types for JSON serialization if necessary
        json.dump(all_series_results, f, indent=4, default=lambda x: x.item() if isinstance(x, np.generic) else x)

    # Save aggregated pass rates
    with open(os.path.join(OUTPUT_DIR, "initial_aggregated_pass_rates.json"), 'w') as f:
        json.dump(pass_rates, f, indent=4)

    print("\nPhase 1 analysis complete. Aggregated results printed above.")