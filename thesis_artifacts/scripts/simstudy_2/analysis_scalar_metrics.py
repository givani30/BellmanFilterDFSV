"""Utilities for calculating and aggregating scalar metrics from optimization results.

This module provides functions for computing and aggregating scalar metrics from
optimization results data. It handles the calculation of performance metrics,
timing statistics, and accuracy measures across different filter configurations
and parameter settings.
"""

import polars as pl
import logging
from typing import Dict


def calculate_scalar_metrics(df_success: pl.DataFrame) -> pl.DataFrame:
    """Calculate scalar metrics for Phase 3 analysis.

    Args:
        df_success: DataFrame containing successful optimization runs

    Returns:
        DataFrame with aggregated scalar metrics
    """
    logging.info("Starting Phase 3: Scalar Metric Analysis")

    # Calculate loss_diff column
    df_success = df_success.with_columns([
        (pl.col("results_final_loss") - pl.col("results_loss_at_true_params")).alias("loss_diff")
    ])

    # Define expanded grouping columns using filter_config
    group_cols = [
        "filter_config",  # Use the new refined filter config column
        "N",
        "K",
        "config_T",
        "config_fix_mu"
    ]
    logging.debug(f"Grouping columns for scalar metrics: {group_cols}")

    # Define metrics to aggregate with their statistics
    scalar_metrics = [
        "results_steps",
        "timing_total_script_duration_s",
        "accuracy_state_estimation_factor_rmse_mean",
        "accuracy_state_estimation_factor_correlation_mean",
        "accuracy_state_estimation_volatility_rmse_mean",
        "accuracy_state_estimation_volatility_correlation_mean",
        "loss_diff"
    ]

    # Create expressions for aggregation
    agg_exprs = []
    for metric in scalar_metrics:
        agg_exprs.extend([
            pl.col(metric).mean().alias(f"{metric}_mean"),
            pl.col(metric).median().alias(f"{metric}_median"),
            pl.col(metric).std().alias(f"{metric}_std")
        ])

    # Add success rate calculation (should be 100% by definition)
    agg_exprs.append(
        (pl.len().cast(pl.Float64) / pl.len().cast(pl.Float64) * 100.0).alias("success_rate")
    )

    # Perform grouping and aggregation
    df_agg = df_success.group_by(group_cols).agg(agg_exprs)

    logging.info("Completed Phase 3 aggregation with shape: %s", df_agg.shape)

    return df_agg


def aggregate_scalar_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate aggregate statistics for scalar metrics grouped by configuration.

    Args:
        df: Input DataFrame containing scalar metrics

    Returns:
        DataFrame with aggregated metrics by configuration
    """
    # Define metrics to aggregate with their types
    scalar_metrics = {
        'results_steps': pl.Int64,
        'timing_total_script_duration_s': pl.Float64,
        'accuracy_state_estimation_factor_rmse_mean': pl.Float64,
        'accuracy_state_estimation_factor_correlation_mean': pl.Float64,
        'accuracy_state_estimation_volatility_rmse_mean': pl.Float64,
        'accuracy_state_estimation_volatility_correlation_mean': pl.Float64,
        'loss_diff': pl.Float64
    }

    # Create cast expressions for metrics and grouping columns
    cast_expressions = [
        # Cast grouping columns
        pl.col("N").cast(pl.Int64),
        pl.col("K").cast(pl.Int64),
        pl.col("config_T").cast(pl.Int64),
        pl.col("filter_config").cast(pl.Categorical),
        pl.col("config_fix_mu").cast(pl.Boolean),

        # Cast metrics with proper types
        *[pl.col(metric).cast(dtype).fill_null(0)
          for metric, dtype in scalar_metrics.items()
          if metric in df.columns]
    ]

    # Cast columns and fill nulls
    df_clean = df.with_columns(cast_expressions)

    # Create aggregation expressions
    agg_expressions = []
    for metric in scalar_metrics.keys():
        if metric in df_clean.columns:
            agg_expressions.extend([
                pl.col(metric).mean().alias(f"{metric}_mean"),
                pl.col(metric).median().alias(f"{metric}_median"),
                pl.col(metric).std().alias(f"{metric}_std"),
                pl.col(metric).count().alias(f"{metric}_count")  # Track number of observations
            ])

    # Group by refined configuration and calculate aggregates
    result = df_clean.group_by([
        'filter_config',  # Use filter_config instead of filter_type
        'N',
        'K',
        'config_T',
        'config_fix_mu'
    ]).agg(agg_expressions).sort([
        'filter_config',
        'N',
        'K',
        'config_T',
        'config_fix_mu'
    ])

    # Log summary statistics
    logging.info("Scalar metrics aggregation summary:")
    for metric in scalar_metrics.keys():
        if f"{metric}_count" in result.columns:
            total = result.select(pl.col(f"{metric}_count").sum()).item()
            missing = len(df) - total
            logging.info(f"  {metric}: {total} observations ({missing} missing)")

    return result