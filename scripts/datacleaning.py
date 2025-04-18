import pandas as pd

def clean_data(df):
    # Drop all parameter accuracy columns for N and K
    # Drop all specified columns
    columns_to_drop = [
        'accuracy_parameter_estimation_K_rmse',
        'accuracy_parameter_estimation_K_mean_error',
        'accuracy_parameter_estimation_N_mean_error',
        'accuracy_parameter_estimation_N_rmse',
        'opt_message', 'opt_nfev', 'opt_nit', 'opt_status',
        'results_error_message', 'metadata_parse_error',
        'json_read_error', 'pkl_read_error',
        'config_save_format'
    ]
    df = df.drop(columns=columns_to_drop)
    return df

# Loaded variable 'df' from URI: /home/givanib/Documents/BellmanFilterDFSV/outputs/aggregated_optimization_metrics_14-04-2025.csv
df = pd.read_csv(r'/home/givanib/Documents/BellmanFilterDFSV/outputs/aggregated_optimization_metrics_14-04-2025.csv')

df_clean = clean_data(df.copy())
df_clean.head()