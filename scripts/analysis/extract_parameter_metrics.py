# scripts/analysis/extract_parameter_metrics.py
import polars as pl
import os

# Define the path to the results directory and the specific CSV file
results_dir = "outputs/analysis_plots_20250416_022441"
csv_file = "analysis_agg_param_errors_20250416_022543.csv"
file_path = os.path.join(results_dir, csv_file)

# Define the parameters and the corresponding metric columns to extract
# Using mean RMSE/FrobDiff and mean Bias columns
metrics_to_extract = {
    "Lambda": ["param_lambda_r_rmse_mean", "param_lambda_r_bias_mean"],
    "Mu": ["param_mu_rmse_mean", "param_mu_bias_mean"],
    "Sigma2": ["param_sigma2_rmse_mean", "param_sigma2_bias_mean"],
    "Phi_f": ["param_Phi_f_frob_diff_mean", "param_Phi_f_bias_mean"],
    "Phi_h": ["param_Phi_h_frob_diff_mean", "param_Phi_h_bias_mean"],
    "Q_h": ["param_Q_h_frob_diff_mean", "param_Q_h_bias_mean"],
}

# Flatten the list of metric columns
metric_columns = [col for cols in metrics_to_extract.values() for col in cols]

# Columns to identify the configuration
config_columns = ["filter_config", "N", "K", "config_T", "config_fix_mu"]

# Columns to select
columns_to_select = config_columns + metric_columns

try:
    # Load the CSV file
    df = pl.read_csv(file_path)

    # Select the relevant columns
    df_selected = df.select(columns_to_select)

    # Optional: Filter for specific configurations if needed
    # Example: Filter for N=10, K=3, T=2000
    df_filtered = df_selected.filter(
        (pl.col("N") == 10) &
        (pl.col("K") == 3) &
        (pl.col("config_T") == 2000)
    )
    print("Filtered Results (N=10, K=3, T=2000):")
    # print(df_filtered)

    # Print the full selected dataframe for verification
    print(f"Extracted Metrics from: {file_path}")
    # Configure polars display options for better readability
    pl.Config.set_tbl_rows(100) # Show more rows
    pl.Config.set_tbl_cols(20) # Show more columns
    pl.Config.set_float_precision(5) # Set float precision

    print(df_filtered)

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")