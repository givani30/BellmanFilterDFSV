import pandas as pd
import numpy as np

def parse_array_string(s):
    """
    Parse string representation of arrays without commas between values.
    Example: "[0.47707086 0.61443548]" -> np.array([0.47707086, 0.61443548])
    """
    if not isinstance(s, str):
        return s
    if pd.isna(s):
        return np.array([])
    
    try:
        # Clean the string and split by whitespace
        s = s.strip('[]')
        return np.array([float(x) for x in s.split()])
    except Exception as e:
        print(f"Failed to parse: {s}, Error: {e}")
        return np.array([])

def process_array_columns(df, array_columns):
    """
    Process all array columns in the dataframe using the parse_array_string function.
    Also calculates mean values for each array.
    """
    # Convert string arrays to numpy arrays
    for col in array_columns:
        df[col] = df[col].apply(parse_array_string)
    
    # Calculate mean values for non-empty arrays
    for filt in ['bf', 'pf']:
        for metric in ['rmse', 'corr']:
            for state in ['f', 'h']:
                col_name = f'{filt}_{metric}_{state}'
                if col_name in df.columns:
                    df[f'{col_name}_mean'] = df[col_name].apply(
                        lambda x: np.mean(x) if isinstance(x, np.ndarray) and len(x) > 0 else np.nan
                    )
    
    return df

# Example usage:
# array_columns = ['bf_rmse_f', 'bf_corr_f', 'bf_rmse_h', 'bf_corr_h', 
#                 'pf_rmse_f', 'pf_corr_f', 'pf_rmse_h', 'pf_corr_h']
# results_df = process_array_columns(results_df, array_columns)