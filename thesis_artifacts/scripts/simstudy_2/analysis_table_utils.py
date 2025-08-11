"""Utility functions for generating summary tables from simulation study results.

This module provides functions for creating comparative summary tables from simulation
study results, focusing on parameter errors and scalar metrics across different filter
configurations. The functions process Polars DataFrames and return structured table
comparisons without handling file I/O operations.
"""

import polars as pl
from typing import Dict


def generate_summary_tables(
    df_success: pl.DataFrame,
    df_agg_scalars: pl.DataFrame
) -> Dict[str, pl.DataFrame]:
    """Generate summary tables comparing performance across filter configurations.

    Args:
        df_success: DataFrame containing successful runs with parameter errors
        df_agg_scalars: DataFrame containing aggregated scalar metrics

    Returns:
        Dict[str, pl.DataFrame]: Dictionary mapping table identifiers to summary DataFrames.
        Keys follow the pattern: summary_{table_type}_N{N}_K{K}_T{T}{fix_mu_suffix}.csv
    """
    # Cast columns for type safety
    df_success = df_success.with_columns([
        pl.col("filter_config").cast(pl.Categorical),
        pl.col("N").cast(pl.Int64),
        pl.col("K").cast(pl.Int64),
        pl.col("config_T").cast(pl.Int64),
        pl.col("config_fix_mu").cast(pl.Boolean)
    ])

    # 1. Parameter Error Summary Table with types
    param_metrics = {
        "param_lambda_r_rmse": ("Risk Premium RMSE", pl.Float64),
        "param_Phi_f_frob_rel_diff": ("State Transition Matrix Rel. Frob. Diff", pl.Float64),
        "param_Q_h_logdet_diff": ("Volatility Covar. LogDet Diff", pl.Float64),
        "param_Phi_h_frob_rel_diff": ("Volatility Trans. Matrix Rel. Frob. Diff", pl.Float64)
    }

    # Create parameter aggregation expressions with proper casting
    param_error_exprs = []
    for metric, (_, dtype) in param_metrics.items():
        param_error_exprs.extend([
            pl.col(metric).cast(dtype).mean().alias(f"{metric}_mean"),
            pl.col(metric).cast(dtype).std().alias(f"{metric}_std")
        ])

    # Aggregate parameter errors with new grouping
    df_param_summary = (
        df_success
        .group_by([
            "filter_config",
            "N",
            "K",
            "config_T",
            "config_fix_mu"
        ])
        .agg(param_error_exprs)
        .sort([
            "filter_config",
            "N",
            "K",
            "config_T",
            "config_fix_mu"
        ])
    )

    # 2. Scalar Metrics Summary Table with types
    scalar_metrics = {
        "accuracy_state_estimation_factor_correlation_mean":
            ("Factor State Correlation", pl.Float64),
        "accuracy_state_estimation_volatility_correlation_mean":
            ("Volatility State Correlation", pl.Float64),
        "timing_total_script_duration_s":
            ("Computation Time (s)", pl.Float64),
        "results_steps":
            ("Optimization Steps", pl.Int64)
    }

    summary_tables = {}

    # Create nested comparison tables
    for table_type, metrics in [("param_errors", param_metrics), ("scalar_metrics", scalar_metrics)]:
        # Get unique configurations
        configs = df_param_summary if table_type == "param_errors" else df_agg_scalars
        config_groups = (
            configs.group_by(["N", "K", "config_T", "config_fix_mu"])
            .agg([])
            .sort(["N", "K", "config_T", "config_fix_mu"])
        )

        # Process configurations for each fix_mu setting
        for fix_mu in [True, False]:
            fix_mu_suffix = "_fixmu" if fix_mu else "_freeμ"
            filtered_groups = config_groups.filter(pl.col("config_fix_mu") == fix_mu)

            # Process each configuration
            for config_row in filtered_groups.iter_rows(named=True):
                N, K, T = config_row["N"], config_row["K"], config_row["config_T"]

                # Filter data for this configuration
                config_filter = (
                    (pl.col("N") == N) &
                    (pl.col("K") == K) &
                    (pl.col("config_T") == T) &
                    (pl.col("config_fix_mu") == fix_mu)
                )

                config_data = (
                    df_param_summary if table_type == "param_errors"
                    else df_agg_scalars
                ).filter(config_filter)

                # Prepare comparison data
                comparison_rows = []
                filter_configs = config_data.select("filter_config").unique().sort("filter_config")

                for metric, (metric_name, dtype) in metrics.items():
                    mean_col = f"{metric}_mean"
                    std_col = f"{metric}_std"

                    # Get values for each filter configuration
                    filter_values = {}
                    for filter_cfg in filter_configs.to_series():
                        values = (
                            config_data.filter(pl.col("filter_config") == filter_cfg)
                            .select([
                                pl.col(mean_col).cast(dtype),
                                pl.col(std_col).cast(dtype)
                            ])
                        )

                        if values.height > 0:
                            mean_val, std_val = values.row(0)
                            filter_values[filter_cfg] = f"{mean_val:.6f} ± {std_val:.6f}"

                    if filter_values:
                        row = {"Metric": metric_name}
                        row.update({
                            f"{cfg} (Mean ± Std)": val
                            for cfg, val in filter_values.items()
                        })
                        comparison_rows.append(row)

                # Create the comparison DataFrame
                if comparison_rows:
                    table_key = f"summary_{table_type}_N{N}_K{K}_T{T}{fix_mu_suffix}.csv"
                    summary_tables[table_key] = pl.DataFrame(comparison_rows)

    return summary_tables
