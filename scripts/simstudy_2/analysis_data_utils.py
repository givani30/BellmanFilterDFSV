"""Data loading, validation, and preparation utilities for simulation study 2.

This module provides functions for loading and validating simulation study results,
including metrics data from CSV files and parameter data from NPZ files. It also
includes functions for filtering and transforming the data for analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import polars as pl
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass


def load_and_validate_metrics(csv_path: Path) -> pl.DataFrame:
    """Load metrics CSV file and perform initial validation.

    Args:
        csv_path: Path to the metrics CSV file

    Returns:
        Polars DataFrame containing the validated metrics data

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the data validation fails
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV file not found: {csv_path}")

    logging.info("Loading metrics from %s", csv_path)
    df = pl.read_csv(csv_path)

    # Log initial data shape and inspect data types
    logging.info("Initial metrics shape: %s", df.shape)
    logging.debug("DataFrame schema:\n%s", df.schema)

    # Check for required columns
    required_cols = ["unique_id", "json_read_error", "pkl_read_error", "results_success"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df


def load_and_validate_params(npz_path: Path) -> Tuple[Dict, Dict]:
    """Load parameters NPZ file and perform initial validation.

    Args:
        npz_path: Path to the parameters NPZ file

    Returns:
        Tuple of (true_params_dict, estimated_params_dict)

    Raises:
        FileNotFoundError: If the NPZ file doesn't exist
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Parameters NPZ file not found: {npz_path}")

    logging.info("Loading parameters from %s", npz_path)
    with np.load(npz_path, allow_pickle=True) as data:
        true_params_dict = data['true_params'].item()
        estimated_params_dict = data['estimated_params'].item()

    logging.info("Loaded %d parameter sets", len(true_params_dict))
    return true_params_dict, estimated_params_dict


def filter_successful_runs(df: pl.DataFrame) -> pl.DataFrame:
    """Filter metrics for successful optimization runs.

    Args:
        df: Input Polars DataFrame

    Returns:
        Filtered Polars DataFrame containing only successful runs
    """
    # Filter out rows with data loading errors
    df_clean = df.filter(
        (pl.col('json_read_error') == False) &
        (pl.col('pkl_read_error') == False)
    )
    logging.info(
        "Removed %d rows with data loading errors",
        df.shape[0] - df_clean.shape[0]
    )

    # Filter for successful runs
    df_success = df_clean.filter(pl.col('results_success') == True)
    logging.info(
        "Filtered to %d successful runs (removed %d failed runs)",
        df_success.shape[0],
        df_clean.shape[0] - df_success.shape[0]
    )

    # Check unique_id integrity
    n_unique = df_success.select(pl.col('unique_id')).n_unique()
    if n_unique != df_success.shape[0]:
        logging.warning(
            "Found %d duplicate unique_ids in successful runs",
            df_success.shape[0] - n_unique
        )

    return df_success


def calculate_loss_diff(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate the loss difference between final and true parameter loss.

    Args:
        df: Input DataFrame containing results_final_loss and results_loss_at_true_params

    Returns:
        DataFrame with additional loss_diff column
    """
    return df.with_columns(
        (pl.col('results_final_loss') - pl.col('results_loss_at_true_params')).alias('loss_diff')
    )