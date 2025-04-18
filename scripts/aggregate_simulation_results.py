#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to aggregate simulation results from merged_metrics.csv.
Each configuration has 1000 replications.

Handles array/matrix metrics by calculating statistics for each element.
"""

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import ast


def parse_array_column(df, column_name):
    """
    Parse string representation of arrays/lists into actual arrays.
    
    Args:
        df: DataFrame containing the column
        column_name: Name of the column to parse
        
    Returns:
        DataFrame with the column parsed
    """
    # Skip if column doesn't exist
    if column_name not in df.columns:
        return df
    
    # Skip if column is already parsed or contains all NaN values
    if df[column_name].isna().all():
        return df
    
    try:
        # Convert string representation of arrays to actual arrays
        df[column_name] = df[column_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Error parsing column {column_name}: {e}")
        # If there's an error, keep the column as is
        pass
    
    return df


def calculate_array_statistics(arrays):
    """
    Calculate element-wise mean and standard error for a series of arrays.
    
    Args:
        arrays: Series of arrays
        
    Returns:
        Tuple of (mean_array, se_array)
    """
    # Filter out non-array values and NaN
    valid_arrays = [arr for arr in arrays if isinstance(arr, list) and not any(np.isnan(x) if isinstance(x, float) else False for x in arr)]
    
    if not valid_arrays:
        return None, None
    
    # Convert to numpy array for calculations
    try:
        array_stack = np.array(valid_arrays)
        mean_array = np.mean(array_stack, axis=0).tolist()
        # Standard error = standard deviation / sqrt(n)
        se_array = (np.std(array_stack, axis=0) / np.sqrt(len(valid_arrays))).tolist()
        return mean_array, se_array
    except Exception as e:
        print(f"Error calculating array statistics: {e}")
        return None, None


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Aggregate simulation results from merged_metrics.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to the merged metrics CSV file (e.g., merged_metrics.csv)."
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to save the aggregated results (e.g., aggregated_metrics.csv)."
    )
    parser.add_argument(
        "--expanded-output",
        type=Path,
        help="Optional path to save expanded results with array elements as separate columns."
    )
    args = parser.parse_args()
    
    # --- Input Validation ---
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # --- Load Data ---
    print(f"Loading data from {args.input_file}...")
    try:
        df = pd.read_csv(args.input_file)
        print(f"Loaded {len(df)} rows.")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # --- Process Data ---
    print("Processing data...")
    
    # Handle NaN in num_particles
    if 'num_particles' in df.columns:
        # Fill NaN with 0 for grouping purposes
        df['num_particles'] = df['num_particles'].fillna(0)
    
    # Define columns to group by (configuration)
    group_cols = ['N', 'K', 'filter_type']
    if 'num_particles' in df.columns:
        group_cols.append('num_particles')
    
    # Identify array columns
    array_cols = [
        'bf_rmse_f', 'bf_corr_f', 'bf_rmse_h', 'bf_corr_h',
        'pf_rmse_f', 'pf_corr_f', 'pf_rmse_h', 'pf_corr_h',
        'bif_rmse_f', 'bif_corr_f', 'bif_rmse_h', 'bif_corr_h'
    ]
    array_cols = [col for col in array_cols if col in df.columns]
    
    # Parse array columns
    for col in array_cols:
        df = parse_array_column(df, col)
    
    # --- Aggregate Data ---
    print("Aggregating data...")
    
    # Create a list to store aggregated data
    aggregated_data = []
    
    # Get unique configurations
    unique_configs = df[group_cols].drop_duplicates().reset_index(drop=True)
    
    # Process each configuration
    for _, config in unique_configs.iterrows():
        # Create a filter for this configuration
        config_filter = True
        for col in group_cols:
            config_filter = config_filter & (df[col] == config[col])
        
        # Get data for this configuration
        config_data = df[config_filter]
        
        # Initialize result dictionary with configuration values
        result = {col: config[col] for col in group_cols}
        
        # Process scalar columns (time columns)
        scalar_cols = [col for col in df.columns if col.endswith('_time') and col != 'param_gen_time']
        for col in scalar_cols:
            if col in config_data.columns:
                result[f"{col}_mean"] = config_data[col].mean()
                result[f"{col}_se"] = config_data[col].sem()
        
        # Process array columns
        for col in array_cols:
            if col in config_data.columns:
                mean_array, se_array = calculate_array_statistics(config_data[col])
                result[f"{col}_mean"] = mean_array
                result[f"{col}_se"] = se_array
        
        # Add to aggregated data
        aggregated_data.append(result)
    
    # Convert to DataFrame
    aggregated_df = pd.DataFrame(aggregated_data)
    
    # --- Save Results ---
    print(f"Saving aggregated results to {args.output_file}...")
    try:
        # Ensure output directory exists
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        aggregated_df.to_csv(args.output_file, index=False)
        print("Save complete.")
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)
    
    # --- Optionally Create Expanded Version ---
    if args.expanded_output:
        print(f"Creating expanded version with array elements as separate columns...")
        expanded_df = aggregated_df.copy()
        
        # Identify array columns in the aggregated data
        array_result_cols = [col for col in expanded_df.columns if col.endswith('_mean') or col.endswith('_se')]
        array_result_cols = [col for col in array_result_cols if expanded_df[col].apply(lambda x: isinstance(x, list)).any()]
        
        # Expand array columns
        for col in array_result_cols:
            # Find first non-None value
            first_valid_idx = None
            for idx, val in enumerate(expanded_df[col]):
                if isinstance(val, list):
                    first_valid_idx = idx
                    break
            
            if first_valid_idx is None:
                continue
                
            # Get the length of the arrays
            array_length = len(expanded_df.loc[first_valid_idx, col])
            
            # Create new columns for each element in the array
            for i in range(array_length):
                expanded_df[f"{col}_{i}"] = expanded_df[col].apply(
                    lambda x: x[i] if isinstance(x, list) and i < len(x) else np.nan
                )
            
            # Drop the original array column
            expanded_df = expanded_df.drop(columns=[col])
        
        # Save expanded version
        try:
            expanded_df.to_csv(args.expanded_output, index=False)
            print(f"Expanded version saved to {args.expanded_output}")
        except Exception as e:
            print(f"Error saving expanded output file: {e}")
    
    print(f"Aggregated {len(df)} rows into {len(aggregated_df)} configurations.")


if __name__ == "__main__":
    main()
