import pandas as pd
import numpy as np
import pickle
import os
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera

# Ensure calculations use float64 for stability
# np.set_printoptions(precision=8, suppress=True) # Optional: for printing

print("Starting Factor-CV Metrics Calculation...")

# --- Load Results ---
output_path = 'outputs/empirical/insample/factorcv_results.pkl'

if not os.path.exists(output_path):
    print(f"Error: Results file not found at {output_path}")
    print("Please run 02_factor_cv_fit.py first to generate the results.")
else:
    try:
        with open(output_path, 'rb') as f:
            results = pickle.load(f)
        print(f"Successfully loaded results from {output_path}")

        # Extract necessary data from results
        model_name = results.get("model_name", "Factor-CV")
        log_likelihood_penalized = results.get("log_likelihood_penalized")
        log_likelihood_base = results.get("log_likelihood_base")
        num_params = results.get("num_params")
        estimation_time = results.get("estimation_time")
        convergence_success = results.get("convergence_success")
        convergence_message = results.get("convergence_message")
        standardized_residuals_post_burn = results.get("standardized_residuals_post_burn")
        aic = results.get("aic")
        bic = results.get("bic")
        final_phi_f_max_eig = results.get("final_phi_f_max_eig")

        if standardized_residuals_post_burn is None:
            print("Error: Standardized residuals (post-burn) not found in results.")
        else:
            print(f"\nAnalyzing standardized residuals ({model_name})...")
            print(f"Residuals shape (post-burn): {standardized_residuals_post_burn.shape}")

            # --- Residual Diagnostics ---
            # TODO: Implement Ljung-Box, Jarque-Bera, ARCH-LM tests
            # Example: Ljung-Box test on squared residuals for autocorrelation
            # For multivariate residuals, you might need to perform tests column-wise
            # or use multivariate extensions if available/appropriate.
            # For simplicity here, we'll show a basic example on one series.

            # Example: Ljung-Box test on the first residual series
            if standardized_residuals_post_burn.shape[1] > 0:
                print("\nLjung-Box test on squared residuals (first series):")
                try:
                    # Test for autocorrelation in squared residuals
                    lb_test_sq = acorr_ljungbox(standardized_residuals_post_burn.iloc[:, 0]**2, lags=[10, 20], return_df=True)
                    print(lb_test_sq)
                except Exception as e:
                    print(f"Error during Ljung-Box test: {e}")

                # TODO: Implement tests for all series or multivariate tests

                # Example: Jarque-Bera test for normality (column-wise)
                print("\nJarque-Bera test for normality (first 5 series):")
                try:
                    for i in range(min(5, standardized_residuals_post_burn.shape[1])):
                        jb_test = jarque_bera(standardized_residuals_post_burn.iloc[:, i])
                        print(f"Series {i+1}: JB Stat={jb_test[0]:.4f}, p-value={jb_test[1]:.4f}")
                except Exception as e:
                    print(f"Error during Jarque-Bera test: {e}")

                # Example: ARCH-LM test for heteroskedasticity (column-wise)
                print("\nARCH-LM test for heteroskedasticity (first 5 series):")
                try:
                    for i in range(min(5, standardized_residuals_post_burn.shape[1])):
                        arch_test = het_arch(standardized_residuals_post_burn.iloc[:, i], nlags=10)
                        print(f"Series {i+1}: LM Stat={arch_test[0]:.4f}, p-value={arch_test[1]:.4f}")
                except Exception as e:
                    print(f"Error during ARCH-LM test: {e}")


            # --- Reporting and Summary ---
            print("\n--- Model Summary ---")
            print(f"Model Name: {model_name}")
            print(f"Convergence Success: {convergence_success}")
            print(f"Convergence Message: {convergence_message}")
            print(f"Estimation Time (s): {estimation_time:.4f}")
            print(f"Number of Parameters: {num_params}")
            print(f"Penalized Log-Likelihood: {log_likelihood_penalized:.4f}")
            print(f"Base Log-Likelihood: {log_likelihood_base:.4f}")
            print(f"AIC: {aic:.4f}")
            print(f"BIC: {bic:.4f}")
            print(f"Final Phi_f Max Eigenvalue Magnitude: {final_phi_f_max_eig:.6f}")

            # Save metrics summary in JSON format (similar to DCC)
            try:
                import json

                # Prepare diagnostic test results
                diagnostic_results = {}

                # Run tests on all series and collect results
                if standardized_residuals_post_burn is not None:
                    # Ljung-Box test on squared residuals
                    lb_results = {}
                    for lag in [5, 10, 15, 20]:
                        lb_results[f"lag_{lag}"] = {}
                        lb_results[f"lag_{lag}"]["pass_rate"] = 0
                        lb_results[f"lag_{lag}"]["total"] = standardized_residuals_post_burn.shape[1]

                        pass_count = 0
                        for i in range(standardized_residuals_post_burn.shape[1]):
                            try:
                                lb_test = acorr_ljungbox(standardized_residuals_post_burn.iloc[:, i]**2, lags=[lag], return_df=True)
                                # Consider test passed if p-value > 0.05 (no autocorrelation)
                                if lb_test.iloc[0, 1] > 0.05:
                                    pass_count += 1
                            except:
                                pass

                        lb_results[f"lag_{lag}"]["pass_count"] = pass_count
                        lb_results[f"lag_{lag}"]["pass_rate"] = pass_count / standardized_residuals_post_burn.shape[1]

                    diagnostic_results["ljung_box_squared"] = lb_results

                    # ARCH-LM test
                    arch_results = {}
                    for lag in [5, 10]:
                        arch_results[f"lag_{lag}"] = {}
                        arch_results[f"lag_{lag}"]["pass_rate"] = 0
                        arch_results[f"lag_{lag}"]["total"] = standardized_residuals_post_burn.shape[1]

                        pass_count = 0
                        for i in range(standardized_residuals_post_burn.shape[1]):
                            try:
                                arch_test = het_arch(standardized_residuals_post_burn.iloc[:, i], nlags=lag)
                                # Consider test passed if p-value > 0.05 (no ARCH effects)
                                if arch_test[1] > 0.05:
                                    pass_count += 1
                            except:
                                pass

                        arch_results[f"lag_{lag}"]["pass_count"] = pass_count
                        arch_results[f"lag_{lag}"]["pass_rate"] = pass_count / standardized_residuals_post_burn.shape[1]

                    diagnostic_results["arch_lm"] = arch_results

                    # Jarque-Bera test
                    jb_results = {}
                    jb_results["pass_rate"] = 0
                    jb_results["total"] = standardized_residuals_post_burn.shape[1]

                    pass_count = 0
                    for i in range(standardized_residuals_post_burn.shape[1]):
                        try:
                            jb_test = jarque_bera(standardized_residuals_post_burn.iloc[:, i])
                            # Consider test passed if p-value > 0.05 (normality)
                            if jb_test[1] > 0.05:
                                pass_count += 1
                        except:
                            pass

                    jb_results["pass_count"] = pass_count
                    jb_results["pass_rate"] = pass_count / standardized_residuals_post_burn.shape[1]

                    diagnostic_results["jarque_bera"] = jb_results

                # Create comprehensive metrics summary
                metrics_summary = {
                    "model_name": model_name,
                    "convergence_success": convergence_success,
                    "convergence_message": str(convergence_message),
                    "estimation_time": float(estimation_time),
                    "num_params": int(num_params) if not np.isnan(num_params) else None,
                    "log_likelihood_penalized": float(log_likelihood_penalized) if np.isfinite(log_likelihood_penalized) else None,
                    "log_likelihood_base": float(log_likelihood_base) if np.isfinite(log_likelihood_base) else None,
                    "aic": float(aic) if np.isfinite(aic) else None,
                    "bic": float(bic) if np.isfinite(bic) else None,
                    "final_phi_f_max_eig": float(final_phi_f_max_eig) if np.isfinite(final_phi_f_max_eig) else None,
                    "diagnostic_tests": diagnostic_results
                }

                # Ensure output directory exists
                output_dir = 'outputs/empirical/insample/factorcv'
                os.makedirs(output_dir, exist_ok=True)

                # Save metrics summary
                metrics_path = os.path.join(output_dir, 'metrics_summary.json')
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_summary, f, indent=2)

                print(f"\nMetrics summary saved to {metrics_path}")

            except Exception as e:
                print(f"Error saving metrics summary: {e}")
                import traceback
                traceback.print_exc()

            # TODO: Generate summary tables (e.g., LaTeX tables) and plots (e.g., ACF plots)
            # Save tables/plots to outputs/tables/ or outputs/figures/

    except Exception as e:
        print(f"Error loading or processing results: {e}")
        import traceback
        traceback.print_exc()


print("\nFactor-CV Metrics Script Finished.")
