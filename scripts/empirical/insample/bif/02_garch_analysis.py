import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from arch import arch_model
import os
import json
from scipy.stats import chi2

# --- Configuration ---
INPUT_FILE = "outputs/empirical/insample/bif/standardized_residuals.csv"
OUTPUT_DIR = "outputs/empirical/insample/bif/residual_analysis/"
ALPHA = 0.05
LB_LAGS = [5, 10, 15, 20]
ARCH_LAGS = [5, 10]

# --- Helper Functions ---
def run_tests(residuals_series, alpha, lb_lags, arch_lags):
    """Runs diagnostic tests on a single residual series."""
    results = {
        'ljung_box_sq': {'p_value': {}, 'pass': {}},
        'arch_lm': {'p_value': {}, 'pass': {}},
        'jarque_bera': {'p_value': None, 'pass': None}
    }
    series_clean = residuals_series.dropna()
    if len(series_clean) < max(lb_lags + arch_lags) + 1: # Check for sufficient data
         # Mark all as failed if insufficient data
        for lag in lb_lags:
            results['ljung_box_sq']['p_value'][lag] = np.nan
            results['ljung_box_sq']['pass'][lag] = False
        for lag in arch_lags:
            results['arch_lm']['p_value'][lag] = np.nan
            results['arch_lm']['pass'][lag] = False
        results['jarque_bera']['p_value'] = np.nan
        results['jarque_bera']['pass'] = False
        return results

    # Ljung-Box on Squared Residuals
    try:
        lb_test = acorr_ljungbox(series_clean**2, lags=lb_lags, return_df=True)
        for lag in lb_lags:
            p_val = lb_test.loc[lag, 'lb_pvalue']
            results['ljung_box_sq']['p_value'][lag] = p_val
            results['ljung_box_sq']['pass'][lag] = p_val > alpha
    except Exception:
        for lag in lb_lags:
            results['ljung_box_sq']['p_value'][lag] = np.nan
            results['ljung_box_sq']['pass'][lag] = False

    # ARCH-LM Test
    try:

        # Correct implementation within the loop:
        for i, lag in enumerate(arch_lags): # arch_lags is [5, 10]
             # Calling het_arch with a single integer lag value:
             arch_test_lag_result = het_arch(series_clean, nlags=lag)
             p_val = arch_test_lag_result[1] # lm_p_value for this specific lag test
             results['arch_lm']['p_value'][lag] = p_val
             results['arch_lm']['pass'][lag] = p_val > alpha
    except Exception:
         for lag in arch_lags:
            results['arch_lm']['p_value'][lag] = np.nan
            results['arch_lm']['pass'][lag] = False


    # Jarque-Bera Test
    try:
        jb_test = jarque_bera(series_clean)
        results['jarque_bera']['p_value'] = jb_test[1]
        results['jarque_bera']['pass'] = jb_test[1] > alpha
    except Exception:
        results['jarque_bera']['p_value'] = np.nan
        results['jarque_bera']['pass'] = False

    return results

def wald_test_sum_garch_zero(params, cov_matrix):
    """Performs Wald test for H0: alpha_1 + beta_1 = 0 vs H1: alpha_1 + beta_1 > 0."""
    # params should contain 'alpha[1]' and 'beta[1]'
    # cov_matrix is the variance-covariance matrix of parameters
    try:
        idx_alpha1 = list(params.index).index('alpha[1]')
        idx_beta1 = list(params.index).index('beta[1]')

        alpha1 = params['alpha[1]']
        beta1 = params['beta[1]']
        sum_ab = alpha1 + beta1

        # Variance of the sum: Var(a+b) = Var(a) + Var(b) + 2*Cov(a,b)
        var_alpha1 = cov_matrix.iloc[idx_alpha1, idx_alpha1]
        var_beta1 = cov_matrix.iloc[idx_beta1, idx_beta1]
        cov_alpha1_beta1 = cov_matrix.iloc[idx_alpha1, idx_beta1]
        var_sum = var_alpha1 + var_beta1 + 2 * cov_alpha1_beta1

        if var_sum <= 0: # Avoid division by zero or sqrt of negative
             return np.nan # Cannot perform test if variance is non-positive

        # Wald statistic for H0: sum = 0: (sum - 0)^2 / Var(sum) ~ Chi2(1)
        wald_stat = (sum_ab - 0)**2 / var_sum

        # Calculate the one-sided p-value for H1: sum > 0
        # P(Z > (sum_ab - 0) / sqrt(Var(sum))) where Z ~ N(0,1)
        # Equivalent to P(Chi2(1) > wald_stat) / 2 if sum_ab > 0, else 1 - P(...)/2
        # Using scipy.stats.norm.sf for the right tail probability (1 - cdf)
        from scipy.stats import norm
        z_stat = sum_ab / np.sqrt(var_sum)
        p_value_one_sided = norm.sf(z_stat) # Survival function = 1 - cdf

        return p_value_one_sided
    except (ValueError, IndexError, KeyError):
        # Handle cases where parameters are missing or matrix indexing fails
        return np.nan


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    try:
        residuals_df = pd.read_csv(INPUT_FILE, index_col=0)
        # Attempt to parse dates if the index is date-like, otherwise keep as is
        try:
            residuals_df.index = pd.to_datetime(residuals_df.index)
        except (ValueError, TypeError):
            print("Index could not be parsed as datetime. Proceeding with original index.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        exit()
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    N = residuals_df.shape[1]
    print(f"Loaded {N} residual series.")

    # --- Phase 2: GARCH Modeling and Analysis ---
    garch_results = {}
    garch_std_residuals_df = pd.DataFrame(index=residuals_df.index)
    post_garch_test_results = {}
    successful_fits = 0

    print("Starting GARCH(1,1) fitting...")
    for i, col in enumerate(residuals_df.columns):
        print(f"  Processing series {i+1}/{N}: {col}")
        series = residuals_df[col].dropna() * 100 # Scale for GARCH stability

        if series.empty:
            print(f"    Skipping series {col} due to only NaNs.")
            garch_results[col] = {'status': 'skipped_nan', 'params': None, 'pvalues': None, 'cov_matrix': None, 'sum_alpha_beta': np.nan, 'wald_p_value_sum_1': np.nan}
            garch_std_residuals_df[col] = np.nan
            post_garch_test_results[col] = None # Indicate failure
            continue

        try:
            # Fit GARCH(1,1) model
            model = arch_model(series, vol='Garch', p=1, q=1, mean='Zero', dist='t',rescale=True)
            res = model.fit(disp='off', show_warning=False)

            if res.convergence_flag == 0:
                # Process results first, increment counter only on success
                std_resid = res.resid / res.conditional_volatility
                garch_std_residuals_df[col] = std_resid / 100 # Rescale back
                params = res.params
                pvalues = res.pvalues
                cov_matrix = res.param_cov

                # Calculate sum and Wald test p-value for H0: sum=0
                sum_alpha_beta = params.get('alpha[1]', np.nan) + params.get('beta[1]', np.nan)
                wald_p_sum_zero = wald_test_sum_garch_zero(params, pd.DataFrame(cov_matrix, index=params.index, columns=params.index))


                garch_results[col] = {
                    'status': 'converged',
                    'params': params.to_dict(),
                    'pvalues': pvalues.to_dict(),
                    'cov_matrix': cov_matrix.values.tolist(), # Convert DataFrame -> numpy -> list
                    'sum_alpha_beta': sum_alpha_beta,
                    'wald_p_value_sum_0': wald_p_sum_zero # Updated variable name
                }
                successful_fits += 1 # Increment counter after successful processing
                # Run post-GARCH tests
                post_garch_test_results[col] = run_tests(std_resid, ALPHA, LB_LAGS, ARCH_LAGS)
            else:
                print(f"    Convergence failed for series {col}.")
                garch_results[col] = {'status': 'failed_convergence', 'params': None, 'pvalues': None, 'cov_matrix': None, 'sum_alpha_beta': np.nan, 'wald_p_value_sum_0': np.nan}
                garch_std_residuals_df[col] = np.nan
                post_garch_test_results[col] = None # Indicate failure

        except Exception as e:
            print(f"    Error fitting GARCH for series {col}: {e}")
            garch_results[col] = {'status': 'error', 'error_msg': str(e), 'params': None, 'pvalues': None, 'cov_matrix': None, 'sum_alpha_beta': np.nan, 'wald_p_value_sum_0': np.nan}
            garch_std_residuals_df[col] = np.nan
            post_garch_test_results[col] = None # Indicate failure

    print(f"Finished GARCH fitting. Successful fits: {successful_fits}/{N}")

    # --- Aggregate Post-GARCH Test Results ---
    print("Aggregating post-GARCH test results...")
    post_garch_pass_rates = {
        'ljung_box_sq': {lag: 0 for lag in LB_LAGS},
        'arch_lm': {lag: 0 for lag in ARCH_LAGS},
        'jarque_bera': 0
    }
    valid_test_series_count = 0

    for col, results in post_garch_test_results.items():
        if results is not None: # Only count series where tests could be run
            valid_test_series_count += 1
            for lag in LB_LAGS:
                if results['ljung_box_sq']['pass'].get(lag, False):
                    post_garch_pass_rates['ljung_box_sq'][lag] += 1
            for lag in ARCH_LAGS:
                if results['arch_lm']['pass'].get(lag, False):
                    post_garch_pass_rates['arch_lm'][lag] += 1
            if results['jarque_bera']['pass']:
                post_garch_pass_rates['jarque_bera'] += 1

    if valid_test_series_count > 0:
        for test_type, lags in post_garch_pass_rates.items():
            if isinstance(lags, dict):
                for lag, count in lags.items():
                    post_garch_pass_rates[test_type][lag] = count / valid_test_series_count
            else:
                 post_garch_pass_rates[test_type] = lags / valid_test_series_count
    else:
         print("Warning: No valid post-GARCH tests could be run.")
         # Set all rates to 0 or NaN if no valid tests
         for test_type, lags in post_garch_pass_rates.items():
            if isinstance(lags, dict):
                for lag in lags:
                    post_garch_pass_rates[test_type][lag] = 0.0
            else:
                 post_garch_pass_rates[test_type] = 0.0


    print("Post-GARCH Pass Rates:", post_garch_pass_rates)


    # --- Analyze GARCH Parameters (Step 8 from Plan) ---
    print("Analyzing GARCH parameters...")
    analysis_results = {
        'total_series': N,
        'successful_fits': successful_fits,
        'perc_alpha1_significant': 0,
        'perc_beta1_significant': 0,
        'perc_wald_sum_rejects_H0': 0, # H0: alpha1 + beta1 = 0 (vs H1: > 0)
        'perc_sum_gt_0_9': 0,
        'perc_sum_gt_0_95': 0,
        'perc_wald_rejects_and_sum_gt_0_9': 0, # Wald rejects H0: sum=0 AND sum > 0.9
    }

    if successful_fits > 0:
        count_alpha1_sig = 0
        count_beta1_sig = 0
        count_wald_rejects = 0
        count_sum_gt_0_9 = 0
        count_sum_gt_0_95 = 0
        count_wald_rejects_and_sum_gt_0_9 = 0

        for col, result in garch_results.items():
            if result['status'] == 'converged':
                pvals = result['pvalues']
                params = result['params']
                wald_p_sum_zero = result['wald_p_value_sum_0'] # Use correct p-value
                sum_ab = result['sum_alpha_beta']

                # Check significance (p < ALPHA)
                if pvals.get('alpha[1]', 1) < ALPHA:
                    count_alpha1_sig += 1
                if pvals.get('beta[1]', 1) < ALPHA:
                    count_beta1_sig += 1

                # Check Wald test result (p < ALPHA means reject H0: sum=0 in favor of H1: sum > 0)
                if wald_p_sum_zero is not None and not np.isnan(wald_p_sum_zero) and wald_p_sum_zero < ALPHA:
                     count_wald_rejects += 1 # Count rejections of sum=0

                # Check sum thresholds
                if not np.isnan(sum_ab):
                    if sum_ab > 0.9:
                        count_sum_gt_0_9 += 1
                    if sum_ab > 0.95:
                        count_sum_gt_0_95 += 1
                    # Combined condition: Wald rejects H0: sum=0 AND sum > 0.9
                    if wald_p_sum_zero is not None and not np.isnan(wald_p_sum_zero) and wald_p_sum_zero < ALPHA and sum_ab > 0.9:
                         count_wald_rejects_and_sum_gt_0_9 += 1


        analysis_results['perc_alpha1_significant'] = count_alpha1_sig / successful_fits
        analysis_results['perc_beta1_significant'] = count_beta1_sig / successful_fits
        analysis_results['perc_wald_sum_rejects_H0'] = count_wald_rejects / successful_fits
        analysis_results['perc_sum_gt_0_9'] = count_sum_gt_0_9 / successful_fits
        analysis_results['perc_sum_gt_0_95'] = count_sum_gt_0_95 / successful_fits
        analysis_results['perc_wald_rejects_and_sum_gt_0_9'] = count_wald_rejects_and_sum_gt_0_9 / successful_fits

    print("GARCH Parameter Analysis:", analysis_results)

    # --- Prepare Output for Memory Bank ---
    # Combine results into a single dictionary for printing
    final_output = {
        "post_garch_pass_rates": post_garch_pass_rates,
        "garch_parameter_analysis": analysis_results
        # Optionally include raw garch_results if needed, but it can be large
        # "raw_garch_results": garch_results
    }

    # Print the final output as JSON to stdout
    print("\n--- MEMORY BANK OUTPUT ---")
    print(json.dumps(final_output, indent=4))
    print("--- END MEMORY BANK OUTPUT ---")

    # Note: Saving CSVs is part of Phase 3 in the plan, not done here.
    print("\nScript finished.")