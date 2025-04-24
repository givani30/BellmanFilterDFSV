"""Specialized utility functions for optimization analysis.

This module provides specialized utility functions specific to the analysis of
optimization results from Simulation Study 2, including custom calculations
and experiment-specific data transformations.

Note:
    Part of the optimization results analysis refactoring for Simulation Study 2.
"""

import polars as pl
import logging
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_identification_difficulty(df: pl.DataFrame) -> None:
    """Analyze parameter identification difficulty across filter types and fix_mu settings.

    Args:
        df: DataFrame containing parameter error metrics and configuration details
    """
    # Define relative error metrics to analyze
    relative_metrics = [
        "param_Phi_f_frob_rel_diff",
        "param_Phi_h_frob_rel_diff",
        "param_Q_h_frob_rel_diff",
        "param_lambda_r_rmse"  # Already relative as lambda_r is dimensionless
    ]

    # Define bias metrics
    bias_metrics = [
        "param_Phi_f_bias",
        "param_Phi_h_bias",
        "param_Q_h_bias",
        "param_lambda_r_bias"
    ]

    # Group by filter_config and config_fix_mu
    group_cols = ["filter_config", "config_fix_mu"]

    # Calculate comprehensive error statistics for each group
    agg_exprs = []
    for metric in relative_metrics + bias_metrics:
        if metric in df.columns:
            agg_exprs.extend([
                pl.col(metric).mean().alias(f"{metric}_mean"),
                pl.col(metric).std().alias(f"{metric}_std"),
                pl.col(metric).median().alias(f"{metric}_median")
            ])

    df_grouped = df.group_by(group_cols).agg(agg_exprs)

    # Get unique filter configurations and sort them
    filter_configs = df_grouped.select("filter_config").unique().sort("filter_config").to_series()

    # Log findings
    logging.info("\n=== Parameter Identification Difficulty Analysis ===")

    for filter_cfg in filter_configs:
        for fix_mu in [True, False]:
            group_data = df_grouped.filter(
                (pl.col("filter_config") == filter_cfg) &
                (pl.col("config_fix_mu") == fix_mu)
            )

            if group_data.height > 0:
                logging.info(f"\n{filter_cfg} (fix_mu={fix_mu}):")

                # Analyze relative errors
                rel_errors = {
                    metric: group_data.select(pl.col(f"{metric}_mean"))[0, 0]
                    for metric in relative_metrics
                    if f"{metric}_mean" in group_data.columns
                }

                # Sort parameters by relative error magnitude
                sorted_params = sorted(
                    rel_errors.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )

                logging.info("Parameters ranked by relative error magnitude:")
                for param, error in sorted_params:
                    param_name = param.split("_")[1]  # Extract parameter name
                    logging.info(f"  {param_name}: {error:.6f}")

                # Check for significant biases
                for metric in bias_metrics:
                    if f"{metric}_mean" in group_data.columns:
                        bias = group_data.select(pl.col(f"{metric}_mean"))[0, 0]
                        std = group_data.select(pl.col(f"{metric}_std"))[0, 0]
                        if abs(bias) > 2 * std:  # Check if bias is significant
                            param_name = metric.split("_")[1]
                            logging.info(f"  Note: {param_name} shows significant bias: {bias:.6f} (std: {std:.6f})")


def analyze_fix_mu_effect(df_success: pl.DataFrame) -> Dict[str, Any]:
    """Analyze the effect of fixing mu during optimization across filter configurations.

    Args:
        df_success: DataFrame containing successful optimization runs

    Returns:
        Dict containing analysis results:
            - 'fix_mu_param_errors_plot': Figure object for parameter error plots
            - 'fix_mu_time_series_parameter_errors_plot': Figure for parameter error time series
            - 'fix_mu_time_series_state_estimation_plot': Figure for state estimation time series
            - 'fix_mu_time_series_timing_plot': Figure for timing analysis time series
            - 'fix_mu_analysis_dataframe': DataFrame with detailed aggregated statistics
    """
    logging.info("Analyzing effect of fixing mu parameter")

    # Cast columns and create config label
    df = df_success.filter(pl.col("config_fix_mu").is_not_null()).with_columns([
        pl.col("filter_config").cast(pl.Categorical),
        pl.col("config_fix_mu").cast(pl.Boolean),
        pl.col("N").cast(pl.Int64),
        pl.col("K").cast(pl.Int64),
        pl.col("config_T").cast(pl.Int64),
        pl.concat_str([
            pl.lit("N"), pl.col("N").cast(pl.Utf8),
            pl.lit("-K"), pl.col("K").cast(pl.Utf8),
            pl.lit("-T"), pl.col("config_T").cast(pl.Utf8)
        ]).alias("config_label").cast(pl.Categorical)
    ])

    # Define analysis metrics
    params_to_analyze = {
        "lambda_r": "Risk Premium",
        "Phi_f": "State Transition",
        "Phi_h": "Volatility Transition",
        "Q_h": "Volatility Covariance"
    }

    scalar_metrics = [
        "timing_total_script_duration_s",
        "accuracy_state_estimation_factor_correlation_mean",
        "accuracy_state_estimation_volatility_correlation_mean"
    ]

    # Set up color palettes and styles
    filter_configs = df.select("filter_config").unique().sort("filter_config").to_series()
    color_palette = sns.color_palette("husl", n_colors=len(filter_configs))
    color_dict = dict(zip(filter_configs, color_palette))

    # Create parameter error plots
    fig_param_errors, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    plt.suptitle("Parameter Error Analysis by Filter Configuration and Fix μ Setting",
                 y=1.02, fontsize=14, weight='bold')

    for idx, (param, title) in enumerate(params_to_analyze.items()):
        if idx < len(axes):
            error_col = f"param_{param}_rmse"

            if error_col in df.columns:
                plot_data = df.with_columns([
                    pl.col(error_col).cast(pl.Float64),
                    pl.col("config_label").cast(pl.Categorical)
                ]).to_pandas()

                sns.violinplot(
                    data=plot_data,
                    x="filter_config",
                    y=error_col,
                    hue="config_fix_mu",
                    palette=sns.color_palette("husl", n_colors=2),
                    split=True,
                    inner="box",
                    density_norm="width",
                    cut=0,
                    ax=axes[idx]
                )

                axes[idx].set_title(f"{title} RMSE Distribution", pad=10, fontsize=12)
                # Get current tick positions and labels
                ticks = axes[idx].get_xticks()
                labels = axes[idx].get_xticklabels()
                # Set ticks first, then labels
                axes[idx].set_xticks(ticks)
                axes[idx].set_xticklabels(labels, rotation=45, ha='right')
                legend = axes[idx].legend(title="Fix μ", labels=["False", "True"])
                legend.get_frame().set_alpha(0.9)
                axes[idx].grid(True, axis='y', linestyle='--', alpha=0.7)

                if plot_data[error_col].min() >= 0:
                    axes[idx].set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Prepare metrics for refined aggregation
    agg_metrics = [
        *[f"param_{p}_rmse" for p in params_to_analyze],
        *[f"param_{p}_bias" for p in params_to_analyze],
        *scalar_metrics
    ]

    # Cast all metrics to float for aggregation
    cast_exprs = [pl.col(m).cast(pl.Float64) for m in agg_metrics if m in df.columns]
    df_analysis = df.with_columns(cast_exprs)

    # Group by refined configuration
    group_cols = ["filter_config", "N", "K", "config_T", "config_fix_mu", "config_label"]

    # Create aggregation expressions
    agg_expressions = []
    for metric in agg_metrics:
        if metric in df_analysis.columns:
            agg_expressions.extend([
                pl.col(metric).mean().alias(f"{metric}_mean"),
                pl.col(metric).std().alias(f"{metric}_std"),
                pl.col(metric).median().alias(f"{metric}_median")
            ])

    df_agg = df_analysis.group_by(group_cols).agg(agg_expressions)

    # Log detailed analysis by filter configuration
    logging.info("\nFix Mu Analysis Results by Filter Configuration:")
    filter_configs = df_agg.select("filter_config").unique().sort("filter_config")

    for filter_cfg in filter_configs.to_series():
        logging.info(f"\n{filter_cfg} Results:")
        df_filter = df_agg.filter(pl.col("filter_config") == filter_cfg)

        for config in df_filter.select("config_label").unique().sort("config_label").to_series():
            df_config = df_filter.filter(pl.col("config_label") == config)
            logging.info(f"\nConfiguration: {config}")

            for metric in [m for m in agg_metrics if f"{m}_mean" in df_config.columns]:
                metric_stats = df_config.select([
                    pl.col(f"{metric}_mean"),
                    pl.col(f"{metric}_std"),
                    pl.col(f"{metric}_median"),
                    pl.col("config_fix_mu")
                ])

                fix_true = metric_stats.filter(pl.col("config_fix_mu") == "1")
                fix_false = metric_stats.filter(pl.col("config_fix_mu") == "0")

                if fix_true.height > 0 and fix_false.height > 0:
                    t_mean, t_std = fix_true.row(0)[:2]
                    f_mean, f_std = fix_false.row(0)[:2]
                    diff_pct = ((t_mean - f_mean) / f_mean) * 100 if f_mean != 0 else float('inf')

                    logging.info(
                        f"{metric}:\n"
                        f"  Fix=True:  {t_mean:.6f} ± {t_std:.6f}\n"
                        f"  Fix=False: {f_mean:.6f} ± {f_std:.6f}\n"
                        f"  Effect:    {diff_pct:+.2f}%"
                    )

    # Create time series plots for different metric groups
    metric_groups = {
        "Parameter Errors": [m for m in agg_metrics if "param_" in m and "_rmse" in m],
        "State Estimation": [m for m in scalar_metrics if "accuracy_state_estimation" in m],
        "Timing": [m for m in scalar_metrics if "timing_" in m]
    }

    time_series_figures = {}

    for group_name, metrics in metric_groups.items():
        if not metrics:
            continue

        fig = plt.figure(figsize=(15, 5 * len(metrics)))
        plt.suptitle(f"{group_name} Analysis by Time Series Length", y=1.02, fontsize=14, weight='bold')

        for idx, metric in enumerate(metrics, 1):
            if metric in df.columns:
                ax = plt.subplot(len(metrics), 1, idx)

                plot_data = df.with_columns([
                    pl.col(metric).cast(pl.Float64),
                    pl.col("config_T").cast(pl.Int64),
                    pl.col("filter_config").cast(pl.Categorical),
                    pl.col("config_fix_mu").cast(pl.Boolean)
                ]).to_pandas()

                g = sns.lineplot(
                    data=plot_data,
                    x="config_T",
                    y=metric,
                    hue="filter_config",
                    style="config_fix_mu",
                    markers=True,
                    dashes=True,
                    err_style="band",
                    errorbar=('ci', 95),
                    palette=color_dict,
                    ax=ax
                )

                title = metric.replace('_', ' ').title()
                if "correlation" in metric:
                    title = title.replace("Mean", "")
                    ax.set_ylim(-1, 1)
                elif "timing" in metric:
                    ax.set_yscale('log')
                    ax.set_ylabel("Time (seconds, log scale)")

                ax.set_title(title, pad=10, fontsize=12)
                ax.set_xlabel("Time Series Length (T)")
                ax.grid(True, linestyle='--', alpha=0.7)

                legend = ax.legend(
                    title="Filter Configuration / Fix μ",
                    bbox_to_anchor=(1.05, 1),
                    loc='upper left',
                    frameon=True,
                    framealpha=0.95
                )
                legend.get_frame().set_alpha(0.9)

        plt.tight_layout()
        time_series_figures[f"fix_mu_time_series_{group_name.lower().replace(' ', '_')}_plot"] = fig

        # Log analysis results
        logging.info(f"\n{group_name} Analysis Results:")
        for metric in metrics:
            if metric in df.columns:
                stats = df.group_by(["filter_config", "config_fix_mu"]).agg([
                    pl.col(metric).cast(pl.Float64).mean().alias("mean"),
                    pl.col(metric).cast(pl.Float64).std().alias("std")
                ])

                logging.info(f"\nMetric: {metric}")
                for row in stats.iter_rows(named=True):
                    fix_mu = "Fix μ" if row["config_fix_mu"] else "Free μ"
                    logging.info(
                        f"  {row['filter_config']} ({fix_mu}): "
                        f"{row['mean']:.6f} ± {row['std']:.6f}"
                    )

    return {
        'fix_mu_param_errors_plot': fig_param_errors,
        **time_series_figures,
        'fix_mu_analysis_dataframe': df_agg
    }