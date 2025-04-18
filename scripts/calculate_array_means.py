#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to calculate the mean of each array in the finalized metrics file.
For example, bf_rmse_f_mean_0 through bf_rmse_f_mean_9 will be averaged to create rmse_f_mean.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re


def calculate_array_means(df):
    """
    Calculate the mean of each array in the finalized metrics file.
    
    Args:
        df: DataFrame with the finalized metrics
        
    Returns:
        DataFrame with the array means calculated
    """
    # Create a copy of the DataFrame
    result_df = df.copy()
    
    # Define the metric types
    metric_types = [
        'rmse_f',
        'corr_f',
        'rmse_h',
        'corr_h'
    ]
    
    # Define filter prefixes
    filter_prefixes = ['bf', 'pf', 'bif']
    
    # Process each filter type and metric type
    for prefix in filter_prefixes:
        for metric_type in metric_types:
            # Define the base column name
            base_col = f"{prefix}_{metric_type}"
            
            # Find all mean columns for this metric
            mean_cols = [col for col in df.columns if col.startswith(f"{base_col}_mean_")]
            se_cols = [col for col in df.columns if col.startswith(f"{base_col}_se_")]
            
            if not mean_cols or not se_cols:
                continue
            
            # Calculate the mean of the mean columns
            result_df.loc[df['filter_type'] == prefix.upper(), f"{metric_type}_mean"] = df.loc[
                df['filter_type'] == prefix.upper(), mean_cols
            ].mean(axis=1, skipna=True)
            
            # Calculate the mean of the standard error columns
            result_df.loc[df['filter_type'] == prefix.upper(), f"{metric_type}_se"] = df.loc[
                df['filter_type'] == prefix.upper(), se_cols
            ].mean(axis=1, skipna=True)
    
    # Drop the individual array columns
    array_cols = []
    for prefix in filter_prefixes:
        for metric_type in metric_types:
            array_cols.extend([col for col in df.columns if col.startswith(f"{prefix}_{metric_type}_mean_")])
            array_cols.extend([col for col in df.columns if col.startswith(f"{prefix}_{metric_type}_se_")])
    
    # Keep only the necessary columns
    keep_cols = [col for col in df.columns if col not in array_cols]
    result_df = result_df[keep_cols]
    
    return result_df


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Calculate the mean of each array in the finalized metrics file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to the finalized metrics CSV file (e.g., finalized_metrics.csv)."
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to save the metrics with array means (e.g., final_metrics.csv)."
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
    
    # --- Calculate Array Means ---
    print("Calculating array means...")
    result_df = calculate_array_means(df)
    
    # --- Save Results ---
    print(f"Saving results to {args.output_file}...")
    try:
        # Ensure output directory exists
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        result_df.to_csv(args.output_file, index=False)
        print("Save complete.")
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)
    
    print(f"Processed {len(df)} configurations.")


if __name__ == "__main__":
    main()
