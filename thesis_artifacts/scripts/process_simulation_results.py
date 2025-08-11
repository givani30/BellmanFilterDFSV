#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to process simulation results from raw data to final metrics.

This script combines the functionality of:
1. merge_simulation_results.py - Merges metrics.json files from multiple simulation studies
2. aggregate_simulation_results.py - Aggregates metrics by configuration
3. finalize_aggregated_metrics.py - Parses arrays and merges filter-specific metrics
4. calculate_array_means.py - Calculates the mean of each array metric

Usage:
    python process_simulation_results.py --input-dir simulation_data --output-dir output_thesis
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import ast
import re


def find_metrics_files(base_dir: Path, study_pattern: str, metrics_filename: str) -> list[Path]:
    """
    Finds all metrics files within study directories matching the pattern.

    Args:
        base_dir: The directory containing the study_* folders.
        study_pattern: Glob pattern for study directories (e.g., "study_*").
        metrics_filename: Name of the metrics file (e.g., "metrics.json").

    Returns:
        A list of Path objects pointing to the found metrics files.
    """
    metrics_files = []
    print(f"Searching for study directories matching '{study_pattern}' in '{base_dir}'...")
    study_dirs = list(base_dir.glob(study_pattern))
    print(f"Found {len(study_dirs)} potential study directories.")

    if not study_dirs:
        print("Warning: No study directories found matching the pattern.")
        return []

    for study_dir in study_dirs:
        if not study_dir.is_dir():
            continue
        print(f"  Scanning study directory: {study_dir.name}")
        # Recursively search for the metrics file within each study directory
        # Assumes structure: base_dir/study_*/config_*/metrics.json
        found_in_study = list(study_dir.rglob(f"*/{metrics_filename}"))
        if found_in_study:
            print(f"    Found {len(found_in_study)} '{metrics_filename}' files in {study_dir.name}.")
            metrics_files.extend(found_in_study)
        else:
            print(f"    No '{metrics_filename}' files found in {study_dir.name}.")

    return metrics_files


def load_metric(file_path: Path, study_id: str) -> dict:
    """
    Loads a single metrics JSON file and adds the study ID.

    Args:
        file_path: Path to the metrics.json file.
        study_id: The name of the parent study directory.

    Returns:
        A dictionary containing the metrics data and the study_id,
        or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        data['study_id'] = study_id  # Add the study identifier
        # Add the specific run label as well for more granular identification
        data['run_label'] = file_path.parent.name  # Assumes parent dir is the run label
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def merge_metrics_files(input_dir: Path, study_pattern: str, metrics_filename: str) -> pd.DataFrame:
    """
    Merge metrics files from multiple simulation studies into a single DataFrame.

    Args:
        input_dir: Directory containing the study_* folders.
        study_pattern: Glob pattern for study directories.
        metrics_filename: Name of the metrics file.

    Returns:
        DataFrame containing the merged metrics data.
    """
    # Find metrics files
    metrics_files = find_metrics_files(input_dir, study_pattern, metrics_filename)
    print(f"\nFound {len(metrics_files)} total '{metrics_filename}' files across all studies.")

    if not metrics_files:
        print("No metrics files found. Exiting.")
        sys.exit(0)

    # Load metrics data
    all_metrics_data = []
    processed_count = 0
    error_count = 0
    for file_path in metrics_files:
        # Determine study_id based on the path relative to input_dir
        try:
            relative_path = file_path.relative_to(input_dir)
            # The first part of the relative path should be the study directory
            study_id = relative_path.parts[0]
        except ValueError:
            print(f"Warning: Could not determine study_id for {file_path}. Skipping.")
            error_count += 1
            continue

        metric_data = load_metric(file_path, study_id)
        if metric_data:
            all_metrics_data.append(metric_data)
            processed_count += 1
        else:
            error_count += 1

    print(f"\nSuccessfully processed {processed_count} metrics files.")
    if error_count > 0:
        print(f"Warning: Failed to process or determine study ID for {error_count} files (check logs above).")

    # Create DataFrame
    if not all_metrics_data:
        print("No metrics data successfully loaded to merge. Exiting.")
        sys.exit(0)

    print("\nCreating DataFrame...")
    try:
        df = pd.DataFrame(all_metrics_data)

        # Optional: Reorder columns for clarity - put identifiers first
        id_cols = ['study_id', 'run_label']
        other_cols = [col for col in df.columns if col not in id_cols]
        # Handle case where 'run_label' might not exist if load_metric failed partially
        final_id_cols = [col for col in id_cols if col in df.columns]
        df = df[final_id_cols + other_cols]

        return df
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        sys.exit(1)


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


def aggregate_metrics(df, group_cols):
    """
    Aggregate metrics by calculating mean and standard error for each group.
    Handles both scalar and array metrics.

    Args:
        df: DataFrame with the data
        group_cols: Columns to group by

    Returns:
        DataFrame with aggregated metrics
    """
    # Identify metric columns (time and performance metrics)
    time_cols = [col for col in df.columns if col.endswith('_time') and col != 'param_gen_time']
    array_metric_cols = [
        'bf_rmse_f', 'bf_corr_f', 'bf_rmse_h', 'bf_corr_h',
        'pf_rmse_f', 'pf_corr_f', 'pf_rmse_h', 'pf_corr_h',
        'bif_rmse_f', 'bif_corr_f', 'bif_rmse_h', 'bif_corr_h'
    ]

    # Filter to only include columns that exist in the DataFrame
    time_cols = [col for col in time_cols if col in df.columns]
    array_metric_cols = [col for col in array_metric_cols if col in df.columns]

    # Parse array columns
    for col in array_metric_cols:
        df = parse_array_column(df, col)

    # Get unique configurations
    unique_configs = df[group_cols].drop_duplicates().reset_index(drop=True)

    # Initialize results DataFrame with the unique configurations
    results = unique_configs.copy()

    # Calculate statistics for scalar metrics (time columns)
    for col in time_cols:
        # Calculate mean and standard error for each configuration
        means = []
        ses = []

        for _, config_row in unique_configs.iterrows():
            # Create a filter for this configuration
            config_filter = True
            for group_col in group_cols:
                config_filter = config_filter & (df[group_col] == config_row[group_col])

            # Get data for this configuration
            config_data = df.loc[config_filter, col]

            # Calculate mean and standard error
            mean_val = config_data.mean() if not config_data.empty else None
            se_val = config_data.sem() if not config_data.empty else None

            means.append(mean_val)
            ses.append(se_val)

        # Add to results DataFrame
        results[f"{col}_mean"] = means
        results[f"{col}_se"] = ses

    # Calculate statistics for array metrics
    for col in array_metric_cols:
        # Calculate mean and standard error arrays for each configuration
        mean_arrays = []
        se_arrays = []

        for _, config_row in unique_configs.iterrows():
            # Create a filter for this configuration
            config_filter = True
            for group_col in group_cols:
                config_filter = config_filter & (df[group_col] == config_row[group_col])

            # Get data for this configuration
            config_data = df.loc[config_filter, col]

            # Skip if no data or all NaN
            if config_data.empty or config_data.isna().all():
                mean_arrays.append(None)
                se_arrays.append(None)
                continue

            # Calculate array statistics
            mean_array, se_array = calculate_array_statistics(config_data)

            mean_arrays.append(mean_array)
            se_arrays.append(se_array)

        # Add to results DataFrame
        results[f"{col}_mean"] = mean_arrays
        results[f"{col}_se"] = se_arrays

    return results


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


def expand_arrays(df):
    """
    Expand all array columns in the DataFrame.

    Args:
        df: DataFrame with array columns

    Returns:
        DataFrame with expanded arrays
    """
    # Identify array columns
    array_cols = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, list)).any()]

    # Expand each array column
    expanded_df = df.copy()
    for col in array_cols:
        expanded_df = expand_array_column(expanded_df, col)

    return expanded_df


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

    # Handle filter time columns separately
    # Create filter time columns
    result_df['filter_time_mean'] = np.nan
    result_df['filter_time_se'] = np.nan

    # Map filter-specific time columns to the common filter time column
    time_column_mapping = {
        'BF': ['bf_filter_time_mean', 'bf_filter_time_se'],
        'PF': ['pf_filter_time_mean', 'pf_filter_time_se'],
        'BIF': ['bif_filter_time_mean', 'bif_filter_time_se']
    }

    # Fill in filter time values based on filter type
    for filter_type, (time_mean_col, time_se_col) in time_column_mapping.items():
        # Skip if columns don't exist
        if time_mean_col not in df.columns or time_se_col not in df.columns:
            continue

        # Fill in values for rows with matching filter type
        filter_mask = result_df['filter_type'] == filter_type
        result_df.loc[filter_mask, 'filter_time_mean'] = df.loc[filter_mask, time_mean_col]
        result_df.loc[filter_mask, 'filter_time_se'] = df.loc[filter_mask, time_se_col]

    return result_df


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

    # Drop the individual array columns and filter-specific columns
    drop_cols = []

    # Add array columns to drop
    for prefix in filter_prefixes:
        for metric_type in metric_types:
            drop_cols.extend([col for col in df.columns if col.startswith(f"{prefix}_{metric_type}_mean_")])
            drop_cols.extend([col for col in df.columns if col.startswith(f"{prefix}_{metric_type}_se_")])

    # Add filter-specific time columns to drop
    for prefix in filter_prefixes:
        time_mean_col = f"{prefix}_filter_time_mean"
        time_se_col = f"{prefix}_filter_time_se"
        if time_mean_col in df.columns:
            drop_cols.append(time_mean_col)
        if time_se_col in df.columns:
            drop_cols.append(time_se_col)

    # Keep only the necessary columns
    keep_cols = [col for col in df.columns if col not in drop_cols]
    result_df = result_df[keep_cols]

    return result_df


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Process simulation results from raw data to final metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the study_* folders (e.g., simulation_data)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the output files."
    )
    parser.add_argument(
        "--study-pattern",
        default="batch_study_*",
        help="Glob pattern for identifying study directories within the input directory."
    )
    parser.add_argument(
        "--metrics-filename",
        default="metrics.json",
        help="Name of the metrics file to search for within each run's subdirectory."
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate files during processing."
    )
    args = parser.parse_args()

    # --- Input Validation ---
    if not args.input_dir.is_dir():
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    # --- Create Output Directory ---
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Merge Metrics Files ---
    print("\n=== Step 1: Merging Metrics Files ===")
    merged_df = merge_metrics_files(args.input_dir, args.study_pattern, args.metrics_filename)

    if args.save_intermediate:
        merged_file = args.output_dir / "merged_metrics.csv"
        merged_df.to_csv(merged_file, index=False)
        print(f"Saved merged metrics to {merged_file}")

    # --- Step 2: Aggregate Metrics ---
    print("\n=== Step 2: Aggregating Metrics ===")
    # Define columns to group by (configuration)
    group_cols = ['N', 'K', 'filter_type']
    if 'num_particles' in merged_df.columns:
        # Fill NaN with 0 for grouping purposes
        merged_df['num_particles'] = merged_df['num_particles'].fillna(0)
        group_cols.append('num_particles')

    aggregated_df = aggregate_metrics(merged_df, group_cols)

    if args.save_intermediate:
        aggregated_file = args.output_dir / "aggregated_metrics.csv"
        aggregated_df.to_csv(aggregated_file, index=False)
        print(f"Saved aggregated metrics to {aggregated_file}")

    # --- Step 3: Expand Arrays ---
    print("\n=== Step 3: Expanding Arrays ===")
    expanded_df = expand_arrays(aggregated_df)

    if args.save_intermediate:
        expanded_file = args.output_dir / "expanded_metrics.csv"
        expanded_df.to_csv(expanded_file, index=False)
        print(f"Saved expanded metrics to {expanded_file}")

    # --- Step 4: Merge Filter-Specific Metrics ---
    print("\n=== Step 4: Merging Filter-Specific Metrics ===")
    finalized_df = merge_filter_specific_metrics(expanded_df)

    if args.save_intermediate:
        finalized_file = args.output_dir / "finalized_metrics.csv"
        finalized_df.to_csv(finalized_file, index=False)
        print(f"Saved finalized metrics to {finalized_file}")

    # --- Step 5: Calculate Array Means ---
    print("\n=== Step 5: Calculating Array Means ===")
    final_df = calculate_array_means(finalized_df)

    # --- Save Final Results ---
    final_file = args.output_dir / "final_metrics.csv"
    final_df.to_csv(final_file, index=False)
    print(f"Saved final metrics to {final_file}")

    print(f"\nProcessed {len(merged_df)} raw metrics into {len(final_df)} aggregated configurations.")


if __name__ == "__main__":
    main()
