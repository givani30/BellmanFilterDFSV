#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to finalize aggregated metrics by:
1. Parsing any remaining arrays
2. Merging filter-specific metrics into common columns
"""

import argparse
import pandas as pd
import numpy as np
import ast
from pathlib import Path
import sys


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
        df[column_name] = df[column_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Error parsing column {column_name}: {e}")
        # If there's an error, keep the column as is
        pass
    
    return df


def expand_array_column(df, column_name):
    """
    Expand array column into separate columns for each element.
    
    Args:
        df: DataFrame containing the column
        column_name: Name of the column to expand
        
    Returns:
        DataFrame with the column expanded
    """
    # Skip if column doesn't exist
    if column_name not in df.columns:
        return df
    
    # Skip if column is not an array
    if not df[column_name].apply(lambda x: isinstance(x, list)).any():
        return df
    
    # Find first non-None value
    first_valid_idx = None
    for idx, val in enumerate(df[column_name]):
        if isinstance(val, list):
            first_valid_idx = idx
            break
    
    if first_valid_idx is None:
        return df
    
    # Get the length of the arrays
    array_length = len(df.loc[first_valid_idx, column_name])
    
    # Create new columns for each element in the array
    for i in range(array_length):
        df[f"{column_name}_{i}"] = df[column_name].apply(
            lambda x: x[i] if isinstance(x, list) and i < len(x) else np.nan
        )
    
    # Drop the original array column
    df = df.drop(columns=[column_name])
    
    return df


def merge_filter_specific_metrics(df):
    """
    Merge filter-specific metrics into common columns.
    
    Args:
        df: DataFrame with filter-specific metrics
        
    Returns:
        DataFrame with merged metrics
    """
    # Create a copy of the DataFrame
    result_df = df.copy()
    
    # Define metric types to merge
    metric_types = [
        'filter_time',
        'rmse_f',
        'corr_f',
        'rmse_h',
        'corr_h'
    ]
    
    # Define filter prefixes
    filter_prefixes = ['bf', 'pf', 'bif']
    
    # Process each metric type
    for metric_type in metric_types:
        # Create new columns for mean and standard error
        mean_col = f"{metric_type}_mean"
        se_col = f"{metric_type}_se"
        
        # Initialize new columns with NaN
        result_df[mean_col] = np.nan
        result_df[se_col] = np.nan
        
        # Fill in values based on filter type
        for prefix in filter_prefixes:
            filter_mean_col = f"{prefix}_{metric_type}_mean"
            filter_se_col = f"{prefix}_{metric_type}_se"
            
            # Skip if columns don't exist
            if filter_mean_col not in df.columns or filter_se_col not in df.columns:
                continue
            
            # Fill in values for rows with matching filter type
            filter_mask = result_df['filter_type'] == prefix.upper()
            result_df.loc[filter_mask, mean_col] = df.loc[filter_mask, filter_mean_col]
            result_df.loc[filter_mask, se_col] = df.loc[filter_mask, filter_se_col]
            
            # Drop the filter-specific columns
            if filter_mean_col in result_df.columns:
                result_df = result_df.drop(columns=[filter_mean_col])
            if filter_se_col in result_df.columns:
                result_df = result_df.drop(columns=[filter_se_col])
    
    return result_df


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Finalize aggregated metrics by parsing arrays and merging filter-specific metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to the aggregated metrics CSV file (e.g., aggregated_metrics.csv)."
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to save the finalized metrics (e.g., finalized_metrics.csv)."
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
    
    # --- Parse Array Columns ---
    print("Parsing array columns...")
    array_columns = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, str) and x.startswith('[')).any()]
    
    for col in array_columns:
        df = parse_array_column(df, col)
    
    # --- Expand Array Columns ---
    print("Expanding array columns...")
    array_columns = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, list)).any()]
    
    expanded_df = df.copy()
    for col in array_columns:
        expanded_df = expand_array_column(expanded_df, col)
    
    # --- Merge Filter-Specific Metrics ---
    print("Merging filter-specific metrics...")
    finalized_df = merge_filter_specific_metrics(expanded_df)
    
    # --- Save Results ---
    print(f"Saving finalized metrics to {args.output_file}...")
    try:
        # Ensure output directory exists
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        finalized_df.to_csv(args.output_file, index=False)
        print("Save complete.")
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)
    
    print(f"Finalized {len(df)} configurations.")


if __name__ == "__main__":
    main()
