#!/usr/bin/env python
"""Analysis script for batch optimization results.

This script implements Phases 1-3 of the analysis plan:
1. Data Loading and Preparation
2. Parameter Error Analysis
3. Scalar Metric Analysis

The script loads and processes optimization results, calculates parameter errors,
and performs aggregation of scalar metrics (convergence, timing, and accuracy)
grouped by experimental configuration.
"""

import argparse
from pathlib import Path
import logging
import sys
from typing import Dict, Tuple, List, Optional

import polars as pl
import numpy as np
import jax.numpy as jnp
import cloudpickle
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging with the specified log level.

    Args:
        log_level: The logging level to use (default: "INFO")
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Analyze batch optimization results - Phases 1-3: Data Loading, "
                   "Parameter Error Analysis, and Scalar Metric Analysis"
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="outputs/aggregated_optimization_metrics_14-04-2025.csv",
        help="Path to the metrics CSV file"
    )
    parser.add_argument(
        "--params-npz",
        type=str,
        default="outputs/aggregated_optimization_params_14-04-2025.npz",
        help="Path to the parameters NPZ file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    return parser.parse_args()


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


def calculate_comprehensive_param_errors(
    true_params: DFSVParamsDataclass,
    estimated_params: DFSVParamsDataclass
) -> Dict[str, float]:
    """Calculate comprehensive error metrics between true and estimated parameters.

    Args:
        true_params: True parameter values
        estimated_params: Estimated parameter values

    Returns:
        Dictionary containing all error metrics
    """
    if true_params is None or estimated_params is None:
        return {}  # Return empty dict if either params object is None

    errors = {}

    def safe_array(x: Optional[np.ndarray]) -> np.ndarray:
        """Convert None to zeros array of appropriate shape."""
        return np.zeros_like(x) if x is None else np.array(x)

    # Helper functions for error metrics
    def calc_elementwise_metrics(true: np.ndarray, est: np.ndarray, name: str) -> None:
        """Calculate element-wise error metrics."""
        diff = est - true
        if len(true.shape) == 2 and true.shape[0] == true.shape[1]:  # Square matrix
            K = true.shape[0]
            for i in range(K):
                for j in range(K):
                    errors[f"param_{name}_element_{i}_{j}_bias"] = float(diff[i, j])
            logging.debug(f"Calculated element-wise metrics for {name} with K={K}")

        errors[f"param_{name}_bias"] = float(np.mean(diff))
        errors[f"param_{name}_mae"] = float(np.mean(np.abs(diff)))
        errors[f"param_{name}_rmse"] = float(np.sqrt(np.mean(diff**2)))

    def calc_matrix_metrics(true: np.ndarray, est: np.ndarray, name: str) -> None:
        """Calculate matrix-specific error metrics."""
        # Frobenius norm differences
        frob_diff = float(jnp.linalg.norm(est - true))
        errors[f"param_{name}_frob_diff"] = frob_diff
        true_norm = float(jnp.linalg.norm(true))
        if true_norm > 1e-10:  # Avoid division by zero
            errors[f"param_{name}_frob_rel_diff"] = frob_diff / true_norm
        else:
            errors[f"param_{name}_frob_rel_diff"] = np.nan

    def calc_square_matrix_metrics(true: np.ndarray, est: np.ndarray, name: str) -> None:
        """Calculate square matrix-specific error metrics."""
        try:
            true_eigs = jnp.linalg.eigvals(true)
            est_eigs = jnp.linalg.eigvals(est)

            # Sort by magnitude for comparison
            true_eigs = jnp.sort(jnp.abs(true_eigs))
            est_eigs = jnp.sort(jnp.abs(est_eigs))

            eig_diff = est_eigs - true_eigs
            errors[f"param_{name}_eig_bias"] = float(jnp.mean(eig_diff))
            errors[f"param_{name}_eig_mae"] = float(jnp.mean(jnp.abs(eig_diff)))
            errors[f"param_{name}_eig_rmse"] = float(jnp.sqrt(jnp.mean(eig_diff**2)))
        except Exception as e:
            logging.warning(f"Failed to calculate eigenvalue metrics for {name}: {str(e)}")
            errors[f"param_{name}_eig_bias"] = np.nan
            errors[f"param_{name}_eig_mae"] = np.nan
            errors[f"param_{name}_eig_rmse"] = np.nan

    def calc_covariance_metrics(true: np.ndarray, est: np.ndarray, name: str) -> None:
        """Calculate covariance matrix-specific error metrics."""
        try:
            # Log determinant difference using slogdet for stability
            true_sign, true_logdet = jnp.linalg.slogdet(true)
            est_sign, est_logdet = jnp.linalg.slogdet(est)
            if true_sign > 0 and est_sign > 0:
                errors[f"param_{name}_logdet_diff"] = float(est_logdet - true_logdet)
            else:
                errors[f"param_{name}_logdet_diff"] = np.nan

            # Trace difference
            errors[f"param_{name}_trace_diff"] = float(jnp.trace(est) - jnp.trace(true))
        except Exception as e:
            logging.warning(f"Failed to calculate covariance metrics for {name}: {str(e)}")
            errors[f"param_{name}_logdet_diff"] = np.nan
            errors[f"param_{name}_trace_diff"] = np.nan

    # Calculate metrics for each parameter
    # lambda_r (vector)
    true_lambda_r = safe_array(true_params.lambda_r)
    est_lambda_r = safe_array(estimated_params.lambda_r)
    calc_elementwise_metrics(true_lambda_r, est_lambda_r, "lambda_r")

    # Phi_f (square matrix)
    true_phi_f = safe_array(true_params.Phi_f)
    est_phi_f = safe_array(estimated_params.Phi_f)
    calc_elementwise_metrics(true_phi_f, est_phi_f, "Phi_f")
    calc_matrix_metrics(true_phi_f, est_phi_f, "Phi_f")
    calc_square_matrix_metrics(true_phi_f, est_phi_f, "Phi_f")

    # Phi_h (square matrix)
    true_phi_h = safe_array(true_params.Phi_h)
    est_phi_h = safe_array(estimated_params.Phi_h)
    calc_elementwise_metrics(true_phi_h, est_phi_h, "Phi_h")
    calc_matrix_metrics(true_phi_h, est_phi_h, "Phi_h")
    calc_square_matrix_metrics(true_phi_h, est_phi_h, "Phi_h")

    # Q_h (covariance matrix)
    true_q_h = safe_array(true_params.Q_h)
    est_q_h = safe_array(estimated_params.Q_h)
    calc_elementwise_metrics(true_q_h, est_q_h, "Q_h")
    calc_matrix_metrics(true_q_h, est_q_h, "Q_h")
    calc_square_matrix_metrics(true_q_h, est_q_h, "Q_h")
    calc_covariance_metrics(true_q_h, est_q_h, "Q_h")

    # mu (scalar)
    if hasattr(true_params, 'mu') and hasattr(estimated_params, 'mu'):
        true_mu = np.array([true_params.mu]) if true_params.mu is not None else np.array([0.0])
        est_mu = np.array([estimated_params.mu]) if estimated_params.mu is not None else np.array([0.0])
        calc_elementwise_metrics(true_mu, est_mu, "mu")

    # sigma2 (scalar)
    if hasattr(true_params, 'sigma2') and hasattr(estimated_params, 'sigma2'):
        true_sigma2 = np.array([true_params.sigma2]) if true_params.sigma2 is not None else np.array([0.0])
        est_sigma2 = np.array([estimated_params.sigma2]) if estimated_params.sigma2 is not None else np.array([0.0])
        calc_elementwise_metrics(true_sigma2, est_sigma2, "sigma2")

    return errors


def calculate_param_errors_for_replicate(
    unique_id: str,
    true_params_dict: Dict[str, DFSVParamsDataclass],
    estimated_params_dict: Dict[str, DFSVParamsDataclass]
) -> Optional[Dict[str, float]]:
    """Calculate parameter errors for a single replicate.

    Args:
        unique_id: The unique identifier for this replicate
        true_params_dict: Dictionary of true parameter objects
        estimated_params_dict: Dictionary of estimated parameter objects

    Returns:
        Dictionary with error metrics or None if parameters not found
    """
    true_params = true_params_dict.get(unique_id)
    estimated_params = estimated_params_dict.get(unique_id)

    if true_params is None or estimated_params is None:
        logging.warning(f"Missing parameters for unique_id: {unique_id}")
        return None

    errors = calculate_comprehensive_param_errors(true_params, estimated_params)
    return {'unique_id': unique_id, **errors}


def create_param_errors_df(
    df_success: pl.DataFrame,
    true_params_dict: Dict[str, DFSVParamsDataclass],
    estimated_params_dict: Dict[str, DFSVParamsDataclass]
) -> pl.DataFrame:
    """Create a DataFrame with parameter error metrics for all successful runs.

    Args:
        df_success: DataFrame with successful runs
        true_params_dict: Dictionary of true parameter objects
        estimated_params_dict: Dictionary of estimated parameter objects

    Returns:
        DataFrame with parameter error metrics
    """
    param_errors_list = []

    for row in df_success.iter_rows(named=True):
        unique_id = row['unique_id']
        errors = calculate_param_errors_for_replicate(
            unique_id, true_params_dict, estimated_params_dict
        )
        if errors is not None:
            param_errors_list.append(errors)

    if not param_errors_list:
        raise ValueError("No valid parameter error calculations found")

    # Convert to Polars DataFrame
    df_param_errors = pl.from_dicts(param_errors_list)
    logging.info("Created parameter errors DataFrame with shape: %s", df_param_errors.shape)

    return df_param_errors


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

   # Nulls for config_fix_mu and config_num_particles are handled in main now.

   # Define expanded grouping columns using filter_config
    group_cols = [
       "filter_config", # Use the new refined filter config column
       "N",
       "K",
       "config_T",
       # "config_num_particles", # No longer needed for grouping, included in filter_config
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


def create_output_directory() -> Path:
    """Create a timestamped output directory for analysis plots.

    Returns:
        Path object pointing to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/analysis_plots_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Created output directory: %s", output_dir)
    return output_dir


def plot_scatter_comparison(df: pl.DataFrame, output_dir: Path) -> None:
    """Create scatter plots comparing Bias vs. RMSE for each parameter.

    Each parameter gets its own figure with facets by N and K. The plots show
    the relationship between estimation bias and RMSE, with filter configuration
    encoded by color and time series length (T) encoded by marker style.

    Args:
        df: DataFrame containing parameter errors
        output_dir: Directory to save plots
    """
    # Set up the style
    sns.set_style("whitegrid")

    # Parameters to analyze
    params_to_plot = {
        "lambda_r": "Risk Premium Parameters",
        "Phi_f": "State Transition Matrix",
        "Phi_h": "Volatility Transition Matrix",
        "Q_h": "Volatility Covariance Matrix",
        "mu": "Mean Level",
        "sigma2": "Observation Noise Variance"
    }

    # Ensure correct types for grouping columns
    df = df.with_columns([
        pl.col("N").cast(pl.Int64),
        pl.col("K").cast(pl.Int64),
        pl.col("config_T").cast(pl.Int64),
        pl.col("filter_config").cast(pl.Categorical)
    ])

    # Add verification of bias/RMSE columns
    for param, title in params_to_plot.items():
        bias_col = f"param_{param}_bias"
        rmse_col = f"param_{param}_rmse"
        
        logging.info(f"\nChecking {param} metrics:")
        for cfg in df.select("filter_config").unique().sort("filter_config").to_series():
            cfg_data = df.filter(pl.col("filter_config") == cfg)
            valid_metrics = cfg_data.filter(
                pl.col(bias_col).is_not_null() & 
                pl.col(rmse_col).is_not_null()
            ).height
            total = cfg_data.height
            if valid_metrics < total:
                logging.warning(f"  {cfg}: Only {valid_metrics}/{total} runs have valid {param} metrics")

    for param, title in params_to_plot.items():
        bias_col = f"param_{param}_bias"
        rmse_col = f"param_{param}_rmse"

        # Skip if either column is missing
        if bias_col not in df.columns or rmse_col not in df.columns:
            logging.info(f"Skipping {param} plot - required columns not found")
            continue

        # Get unique N, K combinations
        configs = df.select(["N", "K"]).unique().sort(["N", "K"])
        if configs.height == 0:
            logging.warning(f"No valid configurations found for {param}")
            continue

        # Create plot with subplots for each N, K combination
        n_configs = configs.height
        n_cols = min(3, n_configs)  # Maximum 3 columns
        n_rows = (n_configs + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        plt.suptitle(f"{title} - Bias vs. RMSE Analysis", y=1.02, fontsize=14)

        # Create a list to store legend handles and labels
        legend_handles = []
        legend_labels = []

        for idx, (N, K) in enumerate(configs.iter_rows(), 1):
            ax = plt.subplot(n_rows, n_cols, idx)

            # Filter data for this N, K combination and prepare for plotting
            plot_data = df.filter(
                (pl.col("N") == N) &
                (pl.col("K") == K)
            ).select([
                pl.col(bias_col).cast(pl.Float64),
                pl.col(rmse_col).cast(pl.Float64),
                pl.col("filter_config"),
                pl.col("config_T").cast(pl.Int64)  # Cast T as integer for style
            ]).to_pandas()

            # Create scatter plot with encoding
            scatter = sns.scatterplot(
                data=plot_data,
                x=bias_col,
                y=rmse_col,
                hue="filter_config",
                style="config_T",  # Use T for marker style
                alpha=0.7,
                ax=ax,
                legend=False  # Don't show legend for individual subplots
            )

            # Store legend handles and labels from the first subplot only
            if idx == 1:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            # Customize subplot
            ax.set_title(f"N={N}, K={K}")
            ax.set_xlabel("Bias")
            ax.set_ylabel("RMSE")
            ax.grid(True, alpha=0.3)

        # Add a single legend to the figure
        fig.legend(legend_handles, legend_labels,
                  title="Filter Config / Time Length",
                  bbox_to_anchor=(1.02, 0.5),
                  loc='center left')

        plt.tight_layout()
        plt.savefig(output_dir / f"scatter_{param}_comparison.png",
                   dpi=300,
                   bbox_inches="tight")
        plt.close()


def calculate_matrix_element_errors(
    true_params: DFSVParamsDataclass,
    estimated_params: DFSVParamsDataclass,
    param_name: str
) -> Dict[str, float]:
    """Calculate element-wise errors for matrix parameters.

    Args:
        true_params: True parameter values
        estimated_params: Estimated parameter values
        param_name: Name of the parameter (e.g., 'Phi_f')

    Returns:
        Dictionary containing errors for each matrix element
    """
    true_matrix = getattr(true_params, param_name)
    est_matrix = getattr(estimated_params, param_name)

    if true_matrix is None or est_matrix is None:
        return {}

    errors = {}
    K = true_matrix.shape[0]  # Get actual matrix dimension
    for i in range(K):
        for j in range(K):
            key = f"param_{param_name}_element_{i}_{j}_bias"
            errors[key] = float(est_matrix[i, j] - true_matrix[i, j])

    logging.debug(f"Calculated element-wise errors for {param_name} with K={K}")
    return errors

def plot_error_heatmaps(df: pl.DataFrame, output_dir: Path) -> None:
    """Create heatmaps showing average element-wise differences for matrix parameters.

    Args:
        df: DataFrame containing parameter errors
        output_dir: Directory to save plots
    """
    # Only plot Phi_f and Phi_h (excluding Q_h as per requirements)
    matrix_params = ["Phi_f", "Phi_h"]
    param_titles = {
        "Phi_f": "State Transition Matrix",
        "Phi_h": "Volatility Transition Matrix"
    }

    # Cast faceting columns to appropriate types before getting unique configs
    df_casted = df.with_columns([
        pl.col("N").cast(pl.Int64),
        pl.col("K").cast(pl.Int64),
        pl.col("config_T").cast(pl.Int64),
        pl.col("filter_config").cast(pl.Categorical)
    ])
    configs = df_casted.select(["N", "K", "config_T"]).unique().sort(["N", "K", "config_T"])
    logging.info(f"Generating heatmaps for {configs.height} unique (N, K, T) configurations.")

    for param in matrix_params:
        for config_row in configs.iter_rows(named=True):
            N, K, T = config_row["N"], config_row["K"], config_row["config_T"]
            config_label = f"N{N}-K{K}-T{T}"

            # Determine filter configs present for this specific N, K, T combination
            filter_configs_to_plot = df_casted.filter(
                (pl.col("N") == N) & (pl.col("K") == K) & (pl.col("config_T") == T)
            ).select("filter_config").unique().sort("filter_config").to_series().to_list()

            if not filter_configs_to_plot:
                logging.warning(f"No filter configs found for heatmap: {param}, {config_label}. Skipping.")
                continue

            num_filters = len(filter_configs_to_plot)
            fig_width = min(16, max(8, num_filters * K * 1.5))  # Scale width with K and filters
            fig_height = min(8, max(5, K * 1.5))  # Scale height with K
            fig, axes = plt.subplots(1, num_filters, figsize=(fig_width, fig_height), squeeze=False)

            for idx, filter_cfg in enumerate(filter_configs_to_plot):
                ax = axes[0, idx]
                fontsize = max(8, 12 - (K - 2))  # Decrease font size as K increases

                try:
                    filter_data = df_casted.filter(
                        (pl.col("filter_config") == filter_cfg) &
                        (pl.col("N") == N) &
                        (pl.col("K") == K) &
                        (pl.col("config_T") == T)
                    )

                    if filter_data.height == 0:
                        logging.warning(f"Data unexpectedly empty for heatmap: {param}, {filter_cfg}, {config_label}")
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=fontsize)
                        ax.set_title(f"{filter_cfg} - {config_label}\nMean Element-wise Bias",
                                   fontsize=fontsize + 2, pad=10)
                        continue

                    # Construct matrix from element-wise errors
                    matrix_data = np.full((K, K), np.nan)
                    for i in range(K):
                        for j in range(K):
                            col_name = f"param_{param}_element_{i}_{j}_bias"
                            if col_name in filter_data.columns:
                                bias_val = filter_data.select(
                                    pl.col(col_name).cast(pl.Float64)
                                ).mean().item()
                                matrix_data[i, j] = bias_val if bias_val is not None else np.nan

                    # Create heatmap
                    sns.heatmap(
                        data=matrix_data,
                        cmap="RdBu_r",
                        center=0,
                        annot=True,
                        fmt=".3f",
                        annot_kws={'size': fontsize},
                        square=True,
                        cbar_kws={'label': 'Bias'},
                        ax=ax
                    )

                    ax.tick_params(axis='x', labelsize=fontsize)
                    ax.tick_params(axis='y', labelsize=fontsize)
                    ax.set_title(
                        f"{filter_cfg} - {config_label}\nMean Element-wise Bias (Est - True)",
                        fontsize=fontsize + 2,
                        pad=10
                    )

                except Exception as e:
                    logging.error(f"Error creating heatmap for {param} ({filter_cfg}, {config_label}): {str(e)}")
                    ax.text(0.5, 0.5, 'Error generating heatmap',
                           ha='center', va='center', fontsize=fontsize)
                    ax.set_title(f"{filter_cfg} - {config_label}\nError",
                               fontsize=fontsize + 2, pad=10)

            plt.suptitle(
                f"{param_titles[param]} Bias Analysis (K={K})",
                y=1.05,
                fontsize=fontsize + 4
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(output_dir / f"heatmap_{param}_{config_label}_comparison.png",
                       dpi=300, bbox_inches="tight")
            plt.close()


def plot_error_boxplots(df: pl.DataFrame, output_dir: Path) -> None:
    """Create box plots comparing error distributions between filter configurations.

    Args:
        df: DataFrame containing error metrics
        output_dir: Directory to save plots
    """
    # Ensure correct types and create config label
    df_plot = df.with_columns([
        # Cast grouping columns
        pl.col("N").cast(pl.Int64),
        pl.col("K").cast(pl.Int64),
        pl.col("config_T").cast(pl.Int64),
        pl.col("filter_config").cast(pl.Categorical),
        pl.col("config_fix_mu").cast(pl.Boolean),
        
        # Create config label
        pl.concat_str([
            pl.lit("N"), pl.col("N").cast(pl.Utf8),
            pl.lit("-K"), pl.col("K").cast(pl.Utf8),
            pl.lit("-T"), pl.col("config_T").cast(pl.Utf8)
        ]).alias("config_label")
    ])

    # Define metric details with proper titles
    metric_details = {
        "param_Phi_f_frob_diff": "State Transition Matrix Frobenius Norm Error",
        "accuracy_state_estimation_factor_rmse_mean": "Factor State Estimation RMSE",
        "accuracy_state_estimation_volatility_rmse_mean": "Volatility State Estimation RMSE"
    }

    for metric, title in metric_details.items():
        # Cast metric to float and get K values
        df_metric = df_plot.with_columns([
            pl.col(metric).cast(pl.Float64)
        ])
        k_values = df_metric.select("K").unique().sort("K")
        
        # Create figure with subplots for different K values
        fig, axes = plt.subplots(
            nrows=len(k_values),
            figsize=(12, 5 * len(k_values)),
            squeeze=False
        )

        for idx, k in enumerate(k_values.to_series()):
            # Filter and convert data for this K value
            k_data = df_metric.filter(pl.col("K") == k).to_pandas()

            # Create boxplot using filter_config for hue
            sns.boxplot(
                data=k_data,
                x="N",
                y=metric,
                hue="filter_config",  # Use filter_config instead of filter_type
                palette="Set2",
                ax=axes[idx, 0]
            )

            # Add style attribute based on config_fix_mu
            if "config_fix_mu" in k_data.columns:
                # Add markers to boxes based on config_fix_mu
                for i, patch in enumerate(axes[idx, 0].patches):
                    if k_data.iloc[i].config_fix_mu == 1:
                        patch.set_hatch('/')

            # Customize subplot
            axes[idx, 0].set_title(f"K = {k}")
            axes[idx, 0].set_xlabel("Number of Assets (N)")
            axes[idx, 0].tick_params(axis='x', rotation=45)

            # Update legend title
            axes[idx, 0].get_legend().set_title("Filter Configuration")

            # Add T value annotations
            unique_T = sorted(k_data["config_T"].unique())
            T_text = f"T values: {', '.join(map(str, unique_T))}"
            axes[idx, 0].text(
                0.02, 0.98,
                T_text,
                transform=axes[idx, 0].transAxes,
                fontsize=8,
                verticalalignment='top'
            )

        # Overall title and layout
        fig.suptitle(title, y=1.02, fontsize=14)
        plt.tight_layout()

        # Save plot
        plt.savefig(
            output_dir / f"boxplot_{metric}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()


def plot_k2_eigenvalue_distributions(
    df: pl.DataFrame,
    output_dir: Path,
    true_params_dict: Dict[str, DFSVParamsDataclass],
    estimated_params_dict: Dict[str, DFSVParamsDataclass]
) -> None:
    """Create distribution plots and ellipse visualizations for K=2 configurations.

    Args:
        df: DataFrame containing parameter estimates
        output_dir: Directory to save plots
        true_params_dict: Dictionary of true parameter objects
        estimated_params_dict: Dictionary of estimated parameter objects
    """
    # Cast columns and filter for K=2 configurations
    df_k2 = df.with_columns([
        pl.col("K").cast(pl.Int64),
        pl.col("N").cast(pl.Int64),
        pl.col("config_T").cast(pl.Int64),
        pl.col("filter_config").cast(pl.Categorical),
        pl.col("config_fix_mu").cast(pl.Boolean),
        pl.col("unique_id")  # Ensure we have the unique_id for parameter lookup
    ]).filter(pl.col("K") == 2)

    if df_k2.height == 0:
        logging.info("No K=2 configurations found for eigenvalue plots")
        return

    # Create config label column
    df_k2 = df_k2.with_columns([
        pl.concat_str([
            pl.lit("N"), pl.col("N").cast(pl.Utf8),
            pl.lit("-K2-T"), pl.col("config_T").cast(pl.Utf8)
        ]).alias("config_label").cast(pl.Categorical)
    ])

    param_titles = {
        "Phi_f": "State Transition Matrix",
        "Phi_h": "Volatility Transition Matrix"
    }

    # Set up distinct color palettes for filter configurations
    filter_configs = df_k2.select("filter_config").unique().sort("filter_config").to_series()
    color_palette = sns.color_palette("husl", n_colors=len(filter_configs))
    color_dict = dict(zip(filter_configs, color_palette))

    # Set up line styles for fix_mu
    style_dict = {True: "-", False: "--"}
    hatch_dict = {True: "///", False: None}

    for param, title in param_titles.items():
        eig_col = f"param_{param}_eig_rmse"
        
        # Cast eigenvalue metric to float
        df_plot = df_k2.with_columns([
            pl.col(eig_col).cast(pl.Float64)
        ])
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(15, 6))

        # 1. Eigenvalue plot
        plt.subplot(1, 2, 1)
        sns.boxplot(
            data=df_plot.to_pandas(),
            x="config_label",
            y=eig_col,
            hue="filter_config",  # Use filter_config instead of filter_type
            palette="Set2"
        )
        plt.title(f"Eigenvalue RMSE Distribution")
        plt.xticks(rotation=45)
        plt.ylabel("RMSE")
        plt.xlabel("Configuration")

        # Update legend title
        plt.gca().get_legend().set_title("Filter Configuration")

        # 2. Ellipse visualization
        plt.subplot(1, 2, 2)
        
        # Group by configuration and filter config
        configs = df_plot.unique(["config_label", "filter_config"]).sort(["config_label", "filter_config"])

        for config_row in configs.iter_rows(named=True):
            filter_cfg = config_row["filter_config"]
            config = config_row["config_label"]

            # Get mean matrix for this configuration
            mean_data = df_plot.filter(
                (pl.col("filter_config") == filter_cfg) &
                (pl.col("config_label") == config)
            )

            # Construct mean 2x2 matrix from elements with proper casting
            matrix = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    col_name = f"param_{param}_element_{i}_{j}_bias"
                    if col_name in mean_data.columns:
                        matrix[i, j] = mean_data.select(
                            pl.col(col_name).cast(pl.Float64)
                        ).mean().item()

            # Construct mean matrix
            matrix = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    col_name = f"param_{param}_element_{i}_{j}_bias"
                    if col_name in mean_data.columns:
                        matrix[i, j] = mean_data.select(
                            pl.col(col_name).cast(pl.Float64)
                        ).mean().item()

            # Calculate eigenvalues and eigenvectors
            try:
                eigvals, eigvecs = np.linalg.eig(matrix)
                eigvecs_np = jnp.asarray(eigvecs).astype(float)
                angle = float(jnp.degrees(jnp.arctan2(eigvecs_np[1, 0], eigvecs_np[0, 0])))
                width = 2 * np.abs(eigvals[0])
                height = 2 * np.abs(eigvals[1])

                # Create ellipse patch with visual distinction for fix_mu
                fix_mu = mean_data.select(pl.col("config_fix_mu")).unique()[0, 0]
                
                # Get color from filter configuration
                color = color_dict[filter_cfg]
                
                style_kwargs = {
                    "alpha": 0.3,
                    "color": color,
                    "label": f"{filter_cfg} ({config}{'†' if fix_mu else ''})",
                    "linestyle": style_dict[fix_mu],
                    "hatch": hatch_dict[fix_mu],
                    "fill": True
                }
                
                ellipse = plt.matplotlib.patches.Ellipse(
                    (0, 0), width, height, angle=angle, **style_kwargs
                )
                plt.gca().add_patch(ellipse)

            except np.linalg.LinAlgError as e:
                logging.warning(f"Failed to compute ellipse for {filter_cfg}-{config}: {e}")

        plt.title("Mean Matrix Eigenstructure Visualization")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

        # Add legend with improved formatting
        legend = plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            title="Filter Config (N,K,T)\n† = Fix μ",
            frameon=True,
            framealpha=0.95,
            edgecolor='lightgray'
        )
        legend.get_frame().set_alpha(0.9)

        # Overall title and layout
        fig.suptitle(f"{title} Analysis (K=2)", y=1.05, fontsize=14, weight='bold')
        plt.tight_layout()

        # Save plot with improved quality settings
        output_path = output_dir / f"eigenval_dist_{param}_k2.png"
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor='white',
            edgecolor='none',
            pad_inches=0.1
        )
        plt.close()

        # Log completion
        logging.info(f"Generated eigenvalue distribution plot for {param}")
        logging.info(f"  - Configurations analyzed: {len(configs)}")
        logging.info(f"  - Output file: {output_path}")


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


def perform_comparative_analysis(df: pl.DataFrame, df_agg: pl.DataFrame) -> None:
    """Perform comparative analysis between BIF and PF performance.

    Args:
        df: Full DataFrame with all metrics
        df_agg: Aggregated scalar metrics DataFrame
    """
    # Analyze parameter identification difficulty
    analyze_identification_difficulty(df)

    # Cast columns and select metrics
    df_analysis = df.with_columns([
        pl.col("filter_config").cast(pl.Categorical),
        pl.col("config_fix_mu").cast(pl.Boolean)
    ])

    param_metrics = [col for col in df_analysis.columns if col.startswith("param_") and col.endswith(("_rmse", "_bias"))]

    # Compare metrics across filter configurations
    filter_configs = df_analysis.select("filter_config").unique().sort("filter_config")
    logging.info("\nMetric Comparison Across Filter Configurations:")
    
    for metric in param_metrics:
        logging.info(f"\nMetric: {metric}")
        stats = []
        base_stats = None
        
        for config in filter_configs.to_series():
            config_stats = df_analysis.filter(
                pl.col("filter_config") == config
            ).select(
                pl.col(metric).cast(pl.Float64).mean()
            ).item()
            
            if "BIF" in config:  # Use BIF as baseline for relative comparisons
                base_stats = config_stats
            
            stats.append(f"{config}: {config_stats:.6f}")
            
            if base_stats is not None and "PF" in config:
                rel_diff = ((config_stats - base_stats) / base_stats) if base_stats != 0 else float('inf')
                stats.append(f"(vs BIF: {rel_diff:+.2%})")
        
        logging.info("  " + " | ".join(stats))

    # 2. Analyze computational efficiency
    timing_cols = ["results_steps", "timing_total_script_duration_s"]
    logging.info("\nComputational Efficiency Analysis:")
    
    for col in timing_cols:
        logging.info(f"\nMetric: {col}")
        base_time = None
        
        for config in filter_configs.to_series():
            time_stat = df_analysis.filter(
                pl.col("filter_config") == config
            ).select(
                pl.col(col).cast(pl.Float64).mean()
            ).item()
            
            if "BIF" in config:
                base_time = time_stat
                logging.info(f"  {config}: {time_stat:.2f}")
            else:
                rel_diff = ((time_stat - base_time) / base_time) if base_time != 0 else float('inf')
                logging.info(f"  {config}: {time_stat:.2f} (vs BIF: {rel_diff:+.2%})")

    # 3. Analyze state estimation accuracy
    accuracy_cols = [col for col in df_analysis.columns if col.startswith("accuracy_")]
    logging.info("\nState Estimation Accuracy Analysis:")
    
    for col in accuracy_cols:
        logging.info(f"\nMetric: {col}")
        base_acc = None
        
        # Group by both filter_config and config_fix_mu for detailed analysis
        acc_stats = (
            df_analysis.group_by(["filter_config", "config_fix_mu"])
            .agg([
                pl.col(col).cast(pl.Float64).mean().alias("mean"),
                pl.col(col).cast(pl.Float64).std().alias("std")
            ])
            .sort(["filter_config", "config_fix_mu"])
        )
        
        for row in acc_stats.iter_rows(named=True):
            config = row["filter_config"]
            fix_mu = row["config_fix_mu"]
            mean_val = row["mean"]
            std_val = row["std"]
            
            if "BIF" in config:
                base_acc = mean_val
                logging.info(f"  {config} (fix_mu={fix_mu}): {mean_val:.6f} ± {std_val:.6f}")
            else:
                rel_diff = ((mean_val - base_acc) / base_acc) if base_acc != 0 else float('inf')
                logging.info(f"  {config} (fix_mu={fix_mu}): {mean_val:.6f} ± {std_val:.6f} (vs BIF: {rel_diff:+.2%})")


def analyze_fix_mu_effect(df_success: pl.DataFrame, output_dir: Path) -> None:
    """Analyze the effect of fixing mu during optimization across filter configurations.

    Args:
        df_success: DataFrame containing successful optimization runs
        output_dir: Directory to save plots and results
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

    # Create parameter error plots with enhanced visualization
    fig = plt.figure(figsize=(16, 12))
    plt.suptitle("Parameter Error Analysis by Filter Configuration and Fix μ Setting",
                 y=1.02, fontsize=14, weight='bold')

    for idx, (param, title) in enumerate(params_to_analyze.items(), 1):
        plt.subplot(2, 2, idx)
        error_col = f"param_{param}_rmse"

        if error_col in df.columns:
            # Calculate statistics for violin plots
            plot_data = df.with_columns([
                pl.col(error_col).cast(pl.Float64),
                pl.col("config_label").cast(pl.Categorical)
            ]).to_pandas()

            # Create enhanced violin plot
            ax = plt.gca()
            sns.violinplot(
                data=plot_data,
                x="filter_config",
                y=error_col,
                hue="config_fix_mu",
                palette=[color_dict[cfg] for cfg in filter_configs],
                split=True,
                inner="box",
                scale="width",
                cut=0
            )

            # Customize plot
            plt.title(f"{title} RMSE Distribution", pad=10, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            legend = plt.legend(title="Fix μ", labels=["False", "True"])
            legend.get_frame().set_alpha(0.9)
            
            # Add grid
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Adjust y-axis to start at 0 if all values are positive
            if plot_data[error_col].min() >= 0:
                ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_dir / "fix_mu_param_errors.png", dpi=300, bbox_inches="tight",
                facecolor='white', edgecolor='none')
    plt.close()

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

        for config in df_filter.select("config_label").unique().sort().to_series():
            df_config = df_filter.filter(pl.col("config_label") == config)
            logging.info(f"\nConfiguration: {config}")

            # Analyze each metric
            for metric in [m for m in agg_metrics if f"{m}_mean" in df_config.columns]:
                metric_stats = df_config.select([
                    pl.col(f"{metric}_mean"),
                    pl.col(f"{metric}_std"),
                    pl.col(f"{metric}_median"),
                    pl.col("config_fix_mu")
                ])

                # Compare fix_mu True vs False
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

    # Create enhanced time series plots with faceting by metric type
    metric_groups = {
        "Timing": ["timing_total_script_duration_s"],
        "State Estimation": [
            "accuracy_state_estimation_factor_correlation_mean",
            "accuracy_state_estimation_volatility_correlation_mean"
        ]
    }

    for group_name, metrics in metric_groups.items():
        # Create figure with subplots for each metric
        fig = plt.figure(figsize=(15, 6 * len(metrics)))
        plt.suptitle(f"{group_name} Analysis by Time Series Length",
                    y=1.02, fontsize=14, weight='bold')

        for idx, metric in enumerate(metrics, 1):
            if metric in df.columns:
                plt.subplot(len(metrics), 1, idx)
                
                # Prepare data with proper casting
                plot_data = df.with_columns([
                    pl.col(metric).cast(pl.Float64),
                    pl.col("config_T").cast(pl.Int64),
                    pl.col("filter_config").cast(pl.Categorical),
                    pl.col("config_fix_mu").cast(pl.Boolean)
                ]).to_pandas()
                
                # Create enhanced line plot
                g = sns.lineplot(
                    data=plot_data,
                    x="config_T",
                    y=metric,
                    hue="filter_config",
                    style="config_fix_mu",
                    markers=True,
                    dashes=True,
                    err_style="band",
                    ci=95,
                    palette=color_dict
                )
                
                # Customize plot
                title = metric.replace('_', ' ').title()
                if "correlation" in metric:
                    title = title.replace("Mean", "")
                    plt.ylim(-1, 1)
                elif "timing" in metric:
                    plt.yscale('log')
                    plt.ylabel("Time (seconds, log scale)")
                
                plt.title(title, pad=10, fontsize=12)
                plt.xlabel("Time Series Length (T)")
                
                # Enhance grid
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Improve legend
                legend = g.legend(
                    title="Filter Configuration / Fix μ",
                    bbox_to_anchor=(1.05, 1),
                    loc='upper left',
                    frameon=True,
                    framealpha=0.95
                )
                legend.get_frame().set_alpha(0.9)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"fix_mu_time_series_{group_name.lower()}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor='white',
            edgecolor='none'
        )
        plt.close()

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

    # Save detailed aggregated stats
    output_path = output_dir / "fix_mu_analysis.csv"
    df_agg.write_csv(output_path)
    logging.info(f"\nDetailed fix mu analysis saved to: {output_path}")


def plot_time_scaling(df_agg: pl.DataFrame, output_dir: Path) -> None:
    # Validate inputs
    if df_agg.shape[0] == 0:
        raise ValueError("Empty DataFrame provided")

    required_cols = ["N", "K", "config_T", "filter_config", "config_fix_mu",
                    "timing_total_script_duration_s_mean"]
    missing_cols = [col for col in required_cols if col not in df_agg.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logging.info("Creating time scaling plots for %d configurations", df_agg.shape[0])
    """Create line plots showing how computation time scales with different dimensions.

    Creates separate plots for each dimension (N: assets, K: factors, T: time series length),
    with lines differentiated by filter configuration and fix_mu setting. Includes error bands
    and automatically switches to log scale when the timing spread is large.

    Args:
        df_agg: DataFrame containing aggregated scalar metrics
        output_dir: Directory to save plots

    Generated files will be named:
        - time_scaling_N.png: Scaling with number of assets
        - time_scaling_K.png: Scaling with number of factors
        - time_scaling_config_T.png: Scaling with time series length
    """
    # Set style for all plots
    sns.set_style("whitegrid")
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10

    # Define dimensions to plot against
    dimensions = [
        ("N", "Number of Assets"),
        ("K", "Number of Factors"),
        ("config_T", "Time Series Length")
    ]

    for x_col, x_label in dimensions:
        # Ensure numerical types for plotting and get valid combinations
        df_plot = df_agg.with_columns([
            pl.col("N").cast(pl.Int64),
            pl.col("K").cast(pl.Int64),
            pl.col("config_T").cast(pl.Int64),
            pl.col("filter_config").cast(pl.Categorical),
            pl.col("config_fix_mu").cast(pl.Boolean),
            pl.col("timing_total_script_duration_s_mean").cast(pl.Float64)
        ])

        # Create figure and plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=df_plot.to_pandas(),
            x=x_col,
            y="timing_total_script_duration_s_mean",
            hue="filter_config",
            style="config_fix_mu",
            errorbar=("ci", 95),
            markers=True,
            dashes=True,
            palette="Set2",
            ax=ax
        )

        # Customize axes and labels
        ax.set_title(f"Computation Time Scaling with {x_label}", pad=20, fontsize=14, weight='bold')
        ax.set_xlabel(x_label if x_col != "config_T" else "Time Series Length (T)")
        ax.set_ylabel("Mean Computation Time (s)")
        
        # Rotate x-axis labels and add grid
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Set log scale if spread is large
        y_vals = df_plot.select(pl.col("timing_total_script_duration_s_mean"))
        if y_vals.max()[0, 0] / y_vals.min()[0, 0] > 10:
            ax.set_yscale('log')
            ax.set_ylabel("Mean Computation Time (s, log scale)")

        # Add legend with positioning
        ax.legend(
            title="Filter Configuration / Fix μ",
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            frameon=True,
            framealpha=0.95,
            edgecolor='lightgray'
        )

        # Adjust layout to prevent legend overlap
        plt.tight_layout()

        # Save plot
        output_path = output_dir / f"time_scaling_{x_col}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        # Log details about the plot generation
        logging.info("Generated time scaling plot for %s:", x_label)
        logging.info("  - Unique filter configs: %d",
                    df_plot.select("filter_config").n_unique())
        logging.info("  - Data points: %d", df_plot.height)
        logging.info("  - Output file: %s", output_path)


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

def generate_summary_tables(df_success: pl.DataFrame, df_agg_scalars: pl.DataFrame, output_dir: Path) -> None:
    """Generate summary tables comparing performance across filter configurations.

    Args:
        df_success: DataFrame containing successful runs with parameter errors
        df_agg_scalars: DataFrame containing aggregated scalar metrics
        output_dir: Directory to save the summary tables
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
        "timing_total_script_duration_s":  # Remove "_mean" suffix here
            ("Computation Time (s)", pl.Float64),
        "results_steps":  # Remove "_mean" suffix here
            ("Optimization Steps", pl.Int64)
    }

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
                mean_col = f"{metric}_mean"  # This will add the "_mean" suffix
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

            # Create and save the comparison table if we have data
            if comparison_rows:
                df_comparison = pl.DataFrame(comparison_rows)
                output_file = output_dir / f"summary_{table_type}_N{N}_K{K}_T{T}{fix_mu_suffix}_{timestamp}.csv"
                df_comparison.write_csv(output_file)
                logging.info(
                    f"Saved {table_type} summary table for N={N}, K={K}, T={T} "
                    f"with {len(filter_configs)} configurations to {output_file}"
                )



def main() -> None:
    """Main execution function for optimization results analysis.
    
    This function orchestrates the complete analysis pipeline:
    1. Data Loading and Preparation
    2. Parameter Error Analysis
    3. Scalar Metric Analysis
    4. Visualization Generation
    5. Final Report Generation
    """
    args = parse_args()
    setup_logging(args.log_level)

    # Convert paths
    metrics_path = Path(args.metrics_csv)
    params_path = Path(args.params_npz)

    try:
        # Create output directory for all results
        output_dir = create_output_directory()
        logging.info("\n=== Starting Analysis Pipeline ===")
        logging.info("Created output directory: %s", output_dir)

        # Phase 1: Load and clean metrics data
        logging.info("\nPhase 1: Data Loading and Preparation")
        logging.info("Loading metrics from: %s", metrics_path)
        logging.info("Loading parameters from: %s", params_path)
        
        start_time = datetime.now()
        df_metrics = load_and_validate_metrics(metrics_path)
        df_success = filter_successful_runs(df_metrics)

        # --- Refinement Step 1: Create filter_config and handle nulls ---
        logging.info("\nData Refinement Step:")
        logging.info("Creating filter configurations and handling missing values")
        df_success = df_success.with_columns([
            # Create filter_config based on filter_type and config_num_particles
            pl.when(pl.col("filter_type") == "BIF")
            .then(pl.lit("BIF"))
            .when((pl.col("filter_type") == "PF") & pl.col("config_num_particles").is_not_null())
            .then(pl.concat_str([pl.lit("PF-"), pl.col("config_num_particles").cast(pl.Int64).cast(pl.Utf8)]))
            .otherwise(pl.lit("Unknown")) # Handle unexpected cases
            .alias("filter_config"),

            # Fill nulls for config_fix_mu early
            pl.col("config_fix_mu").fill_null(-1).cast(pl.Int8).alias("config_fix_mu"),

            # Fill nulls for config_num_particles (used later, good practice to handle early)
            pl.col("config_num_particles").fill_null(0).cast(pl.Int64).alias("config_num_particles")
        ])
        # Log data summary after refinement
        filter_configs = df_success.select("filter_config").unique().sort("filter_config")
        logging.info("\nFilter Configurations Found:")
        for cfg in filter_configs.to_series():
            count = df_success.filter(pl.col("filter_config") == cfg).height
            logging.info(f"  {cfg}: {count} runs")
            
        fix_mu_counts = df_success.group_by("config_fix_mu").agg(
            pl.count().alias("count")
        ).sort("config_fix_mu")
        logging.info("\nFix Mu Settings:")
        for row in fix_mu_counts.iter_rows(named=True):
            setting = "True" if row["config_fix_mu"] == 1 else "False"
            logging.info(f"  {setting}: {row['count']} runs")

        # Load parameter data
        true_params_dict, estimated_params_dict = load_and_validate_params(params_path)
        logging.info("\nPhase 1 completed successfully")
        phase1_time = datetime.now() - start_time
        logging.info(f"Time taken: {phase1_time}")

        # Phase 2: Parameter Error Analysis
        logging.info("\nPhase 2: Parameter Error Analysis")
        start_time = datetime.now()
        df_param_errors = create_param_errors_df(df_success, true_params_dict, estimated_params_dict)
        df_success = df_success.join(df_param_errors, on="unique_id", how="left")
        
        # Log summary of parameter errors
        error_cols = [col for col in df_param_errors.columns if col.endswith(("_rmse", "_bias"))]
        logging.info("\nParameter Error Summary:")
        for col in error_cols:
            stats = df_param_errors.select([
                pl.col(col).mean().alias("mean"),
                pl.col(col).std().alias("std"),
                pl.col(col).median().alias("median")
            ])
            row = stats.row(0)
            logging.info(f"  {col}:")
            logging.info(f"    Mean ± Std: {row[0]:.6f} ± {row[1]:.6f}")
            logging.info(f"    Median: {row[2]:.6f}")
            
        logging.info("\nPhase 2 completed successfully")
        phase2_time = datetime.now() - start_time
        logging.info(f"Time taken: {phase2_time}")

        # Phase 3: Scalar Metric Analysis
        logging.info("\nPhase 3: Scalar Metric Analysis")
        start_time = datetime.now()
        df_agg_scalars = calculate_scalar_metrics(df_success)

        # Save and log Phase 3 results
        output_path = output_dir / "scalar_metrics_analysis.csv"
        df_agg_scalars.write_csv(output_path)
        
        # Log summary statistics for key metrics
        logging.info("\nScalar Metrics Summary by Filter Configuration:")
        for cfg in filter_configs.to_series():
            cfg_data = df_agg_scalars.filter(pl.col("filter_config") == cfg)
            if cfg_data.height > 0:
                logging.info(f"\n  {cfg}:")
                # Log timing stats
                timing_mean = cfg_data.select(pl.col("timing_total_script_duration_s_mean").mean()).item()
                logging.info(f"    Average computation time: {timing_mean:.2f}s")
                # Log accuracy stats
                acc_cols = [col for col in cfg_data.columns if "correlation_mean_mean" in col]
                for col in acc_cols:
                    mean_val = cfg_data.select(pl.col(col).mean()).item()
                    logging.info(f"    {col}: {mean_val:.4f}")
                    
        logging.info("\nPhase 3 completed successfully")
        phase3_time = datetime.now() - start_time
        logging.info(f"Time taken: {phase3_time}")

        # Phase 4: Comparative Analysis & Visualization
        logging.info("\nPhase 4: Comparative Analysis & Visualization")
        start_time = datetime.now()
        
        logging.info("Generating visualization plots...")
        plot_scatter_comparison(df_success, output_dir)
        plot_error_heatmaps(df_success, output_dir)
        plot_error_boxplots(df_success, output_dir)
        plot_k2_eigenvalue_distributions(
            df_success,
            output_dir,
            true_params_dict,
            estimated_params_dict
        )

        # Generate time scaling plots
        logging.info("\nGenerating time scaling plots")
        plot_time_scaling(df_agg_scalars, output_dir)
        logging.info("Time scaling plots generated successfully")

        logging.info("\nPhase 4 completed successfully")
        phase4_time = datetime.now() - start_time
        logging.info(f"Time taken: {phase4_time}")

        # Phase 5: Final Reporting
        logging.info("\nPhase 5: Final Reporting and Output")
        start_time = datetime.now()
        generate_summary_tables(df_success, df_agg_scalars, output_dir)
        logging.info("Summary tables generated successfully")

        # Aggregate parameter errors using refined grouping
        param_error_cols = [col for col in df_success.columns if col.startswith("param_")]
        group_cols_refined = [
            "filter_config",  # Use filter_config instead of filter_type
            "N",
            "K",
            "config_T",
            "config_fix_mu"  # config_num_particles is included in filter_config
        ]

        agg_param_exprs = []
        for metric in param_error_cols:
            agg_param_exprs.extend([
                pl.col(metric).mean().alias(f"{metric}_mean"),
                pl.col(metric).median().alias(f"{metric}_median"),
                pl.col(metric).std().alias(f"{metric}_std")
            ])

        df_agg_param_errors = df_success.group_by(group_cols_refined).agg(agg_param_exprs)
        # Log aggregation summary
        logging.info(f"\nAggregated {len(param_error_cols)} parameter error metrics")
        logging.info(f"Generated summary for {df_agg_param_errors.height} unique configurations")

        # Step 9: Final Output Saving
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.info("\nStep 9: Saving final outputs...")
        
        # Identify all parameter error columns
        param_error_cols = [col for col in df_success.columns if col.startswith("param_")]
        logging.info("Found %d parameter error metrics to aggregate", len(param_error_cols))
        
        # Create final aggregation expressions for parameter errors
        agg_expressions = []
        for col in param_error_cols:
            agg_expressions.extend([
                pl.col(col).mean().alias(f"{col}_mean"),
                pl.col(col).median().alias(f"{col}_median"),
                pl.col(col).std().alias(f"{col}_std"),
                pl.col(col).count().alias(f"{col}_count")
            ])
        
        # Refined aggregation of parameter errors
        df_agg_param_errors = df_success.group_by([
            "filter_config",
            "N",
            "K",
            "config_T",
            "config_fix_mu"
        ]).agg(agg_expressions).sort([
            "filter_config",
            "N",
            "K",
            "config_T",
            "config_fix_mu"
        ])
        
        # Save output files with detailed logging
        output_base = Path("outputs")
        full_data_path = output_base / f"analysis_full_data_{timestamp_str}.csv"
        agg_scalars_path = output_base / f"analysis_agg_scalars_{timestamp_str}.csv"
        agg_param_errors_path = output_base / f"analysis_agg_param_errors_{timestamp_str}.csv"

        # Save with informative logging
        df_success.write_csv(full_data_path)
        logging.info("1. Saved full per-replicate data:")
        logging.info("   - Path: %s", full_data_path)
        logging.info("   - Rows: %d, Columns: %d", df_success.height, df_success.width)
        
        df_agg_scalars.write_csv(agg_scalars_path)
        logging.info("\n2. Saved aggregated scalar metrics:")
        logging.info("   - Path: %s", agg_scalars_path)
        logging.info("   - Configurations: %d", df_agg_scalars.height)
        logging.info("   - Metrics: %d", len([col for col in df_agg_scalars.columns if col.endswith(('_mean', '_std', '_median'))]))
        
        df_agg_param_errors.write_csv(agg_param_errors_path)
        logging.info("\n3. Saved aggregated parameter errors:")
        logging.info("   - Path: %s", agg_param_errors_path)
        logging.info("   - Configurations: %d", df_agg_param_errors.height)
        logging.info("   - Parameter metrics: %d", len(param_error_cols))
        logging.info("   - Total statistics: %d", len([col for col in df_agg_param_errors.columns if col.endswith(('_mean', '_std', '_median', '_count'))]))

        phase5_time = datetime.now() - start_time
        logging.info("\nPhase 5 completed successfully")
        logging.info(f"Time taken: {phase5_time}")

        # Final timing summary
        total_time = phase1_time + phase2_time + phase3_time + phase4_time + phase5_time
        logging.info("\n=== Analysis Pipeline Complete ===")
        logging.info("Phase timing summary:")
        logging.info(f"  Phase 1 (Data Loading):     {phase1_time}")
        logging.info(f"  Phase 2 (Parameter Error):  {phase2_time}")
        logging.info(f"  Phase 3 (Scalar Metrics):   {phase3_time}")
        logging.info(f"  Phase 4 (Visualization):    {phase4_time}")
        logging.info(f"  Phase 5 (Final Report):     {phase5_time}")
        logging.info(f"  Total Time:                 {total_time}")

        # Final summary log messages
        logging.info("\nAnalysis Results Summary:")
        logging.info("------------------------")
        logging.info("1. Data Processing Outputs:")
        logging.info("  - Full Data with Errors: %s", full_data_path)
        logging.info("  - Aggregated Scalar Metrics: %s", agg_scalars_path)
        logging.info("  - Aggregated Parameter Errors: %s", agg_param_errors_path)
        logging.info("\n2. Analysis Outputs:")
        logging.info("  - Visualization Plots: %s/", output_dir)
        logging.info("  - Time Scaling Plots: %s/time_scaling_*.png", output_dir)
        logging.info("  - Scalar Metrics Analysis: %s", output_dir / "scalar_metrics_analysis.csv")
        logging.info("  - Summary Tables: %s/", output_dir)
        logging.info("\n3. Additional Information:")
        logging.info(f"  - Total successful runs analyzed: {df_success.height}")
        logging.info(f"  - Unique filter configurations: {len(filter_configs)}")
        logging.info(f"  - Configuration combinations: {df_agg_scalars.height}")
        logging.info(f"  - Total parameter error metrics: {len(param_error_cols)}")
        logging.info(f"  - Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"\nAll outputs saved to: {output_dir}")

    except Exception as e:
        logging.error("\n=== Analysis Pipeline Failed ===")
        logging.error("Error during analysis:")
        logging.error(str(e))
        logging.error("\nTraceback:")
        import traceback
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
