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

    # Fill null values for config_num_particles and config_fix_mu
    df_success = df_success.with_columns([
        pl.col("config_num_particles").fill_null(0),
        pl.col("config_fix_mu").fill_null(-1)
    ])

    # Define expanded grouping columns
    group_cols = [
        "filter_type",
        "N",
        "K",
        "config_T",
        "config_num_particles",
        "config_fix_mu"
    ]

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
    """Create scatter plots comparing estimated vs. true parameter elements.

    Args:
        df: DataFrame containing parameter errors
        output_dir: Directory to save plots
    """
    # Set up the style
    sns.set_style("whitegrid")

    # Create config label column using concat_str with explicit literals
    df = df.with_columns([
        pl.concat_str([
            pl.lit("N"), pl.col("N").cast(pl.Utf8),
            pl.lit("-K"), pl.col("config_K").cast(pl.Utf8), # Use config_K and pl.lit()
            pl.lit("-T"), pl.col("config_T").cast(pl.Utf8)
        ]).alias("config_label")
    ])

    # Parameters to plot (diagonals and key elements)
    params_to_plot = {
        "Phi_f": "State Transition Matrix",
        "Q_h": "Volatility Covariance Matrix",
        "sigma2": "Observation Noise Variance"
    }

    for param, title in params_to_plot.items():
        # Create faceted plot
        g = sns.FacetGrid(
            data=df.to_pandas(),
            col="config_label",
            col_wrap=3,
            height=4,
            aspect=1.2
        )

        # Extract bias and RMSE for parameter
        bias_col = f"param_{param}_bias"
        rmse_col = f"param_{param}_rmse"

        # Map scatter plot
        g.map_dataframe(
            sns.scatterplot,
            x=bias_col,
            y=rmse_col,
            hue="filter_type",
            style="config_num_particles",
            alpha=0.7
        )

        # Customize plot
        g.fig.suptitle(f"{title} - Estimation Errors by Configuration", y=1.02)
        g.set_axis_labels("Bias", "RMSE")

        # Add legend
        g.add_legend(title="Filter Type")

        # Save plot
        plt.savefig(output_dir / f"scatter_{param}_comparison.png", dpi=300, bbox_inches="tight")
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

    # Group by config for subplot faceting
    configs = df.select(["N", "K", "config_T"]).unique()

    for param in matrix_params:
        for config_row in configs.iter_rows(named=True):
            N, K, T = config_row["N"], config_row["K"], config_row["config_T"]
            config_label = f"N{N}-K{K}-T{T}"

            # Adjust figure size based on K
            fig_width = min(16, max(12, K * 3))  # Scale width with K, but cap at 16
            fig_height = min(8, max(5, K * 1.5))  # Scale height with K, but cap at 8
            plt.figure(figsize=(fig_width, fig_height))

            # Create subplots for BIF and PF
            for idx, filter_type in enumerate(["BIF", "PF"]):
                plt.subplot(1, 2, idx + 1)
                
                # Adjust font size based on K
                fontsize = max(8, 12 - (K - 2))  # Decrease font size as K increases

                try:
                    # Get data for this filter type and configuration
                    filter_data = df.filter(
                        (pl.col("filter_type") == filter_type) &
                        (pl.col("N") == N) &
                        (pl.col("K") == K) &
                        (pl.col("config_T") == T)
                    )

                    # Get K value for this configuration
                    K = config_row["K"]
                    
                    # Construct matrix from element-wise errors with dynamic size
                    matrix_data = np.zeros((K, K))
                    for i in range(K):
                        for j in range(K):
                            col_name = f"param_{param}_element_{i}_{j}_bias"
                            if col_name in filter_data.columns:
                                matrix_data[i, j] = filter_data.select(pl.col(col_name)).mean()[0, 0]

                    # Create heatmap with K-dependent formatting
                    sns.heatmap(
                        data=matrix_data,
                        cmap="RdBu_r",
                        center=0,
                        annot=True,
                        fmt=".3f",
                        annot_kws={'size': fontsize},
                        square=True,  # Keep cells square
                        cbar_kws={'label': 'Bias'}
                    )

                    # Adjust tick label sizes
                    plt.xticks(fontsize=fontsize)
                    plt.yticks(fontsize=fontsize)

                    title = f"{filter_type} - {config_label}\nMean Element-wise Bias (Est - True)"
                    plt.title(title, fontsize=fontsize + 2, pad=10)

                except Exception as e:
                    logging.error(f"Error creating heatmap for {param} ({filter_type}, {config_label}): {str(e)}")
                    plt.text(0.5, 0.5, 'Error generating heatmap',
                            ha='center', va='center', fontsize=fontsize)

            # Adjust layout based on K
            plt.suptitle(f"{param_titles[param]} Bias Analysis (K={K})", y=1.05, fontsize=fontsize + 4)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave room for suptitle
            plt.savefig(output_dir / f"heatmap_{param}_{config_label}_comparison.png", dpi=300, bbox_inches="tight")
            plt.close()


def plot_error_boxplots(df: pl.DataFrame, output_dir: Path) -> None:
    """Create box plots comparing error distributions between BIF and PF.

    Args:
        df: DataFrame containing error metrics
        output_dir: Directory to save plots
    """
    # Create config label column using concat_str with explicit literals
    df = df.with_columns([
        pl.concat_str([
            pl.lit("N"), pl.col("N").cast(pl.Utf8),
            pl.lit("-K"), pl.col("config_K").cast(pl.Utf8), # Use config_K and pl.lit()
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
        # Create figure with subplots for different K values
        k_values = df.select("K").unique().sort("K")
        fig, axes = plt.subplots(
            nrows=len(k_values),
            figsize=(12, 5 * len(k_values)),
            squeeze=False
        )

        for idx, k in enumerate(k_values.to_series()):
            # Filter data for this K value
            k_data = df.filter(pl.col("K") == k).to_pandas()

            # Create boxplot
            sns.boxplot(
                data=k_data,
                x="N",
                y=metric,
                hue="filter_type",
                # style="config_num_particles", # Removed invalid argument for boxplot
                palette="Set2",
                ax=axes[idx, 0]
            )

            # Customize subplot
            axes[idx, 0].set_title(f"K = {k}")
            axes[idx, 0].set_xlabel("Number of Assets (N)")
            axes[idx, 0].tick_params(axis='x', rotation=45)

            # Add T value annotations
            for i, t in enumerate(sorted(k_data["config_T"].unique())):
                axes[idx, 0].text(
                    0.02, 0.98 - i*0.05,
                    f"T={t}",
                    transform=axes[idx, 0].transAxes,
                    fontsize=8
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


def plot_k2_eigenvalue_distributions(df: pl.DataFrame, output_dir: Path) -> None:
    """Create distribution plots and ellipse visualizations for K=2 configurations.

    Args:
        df: DataFrame containing parameter estimates
        output_dir: Directory to save plots
    """
    # Filter for K=2 configurations
    df_k2 = df.filter(pl.col("K") == 2)

    if df_k2.height == 0:
        logging.info("No K=2 configurations found for eigenvalue plots")
        return

    # Create config label column using concat_str with explicit literals
    df_k2 = df_k2.with_columns([
        pl.concat_str([
            pl.lit("N"), pl.col("N").cast(pl.Utf8),
            pl.lit("-T"), pl.col("config_T").cast(pl.Utf8)
        ]).alias("config_label")
    ])

    param_titles = {
        "Phi_f": "State Transition Matrix",
        "Phi_h": "Volatility Transition Matrix"
    }

    for param, title in param_titles.items():
        # Create figure with two subplots (eigenvalues and ellipses)
        fig = plt.figure(figsize=(15, 6))

        # 1. Eigenvalue plot
        plt.subplot(1, 2, 1)
        sns.boxplot(
            data=df_k2.to_pandas(),
            x="config_label",
            y=f"param_{param}_eig_rmse",
            hue="filter_type",
            palette="Set2"
        )
        plt.title(f"Eigenvalue RMSE Distribution")
        plt.xticks(rotation=45)
        plt.ylabel("RMSE")
        plt.xlabel("Configuration")

        # 2. Ellipse visualization for mean matrices
        plt.subplot(1, 2, 2)

        # Group by configuration and filter type
        configs = df_k2.unique(["config_label", "filter_type"]).sort(["config_label", "filter_type"])

        for config_row in configs.iter_rows(named=True):
            filter_type = config_row["filter_type"]
            config = config_row["config_label"]

            # Get mean matrix for this configuration
            mean_data = df_k2.filter(
                (pl.col("filter_type") == filter_type) &
                (pl.col("config_label") == config)
            )

            # Construct mean 2x2 matrix from elements
            matrix = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    col_name = f"param_{param}_element_{i}_{j}_bias"
                    if col_name in mean_data.columns:
                        matrix[i, j] = mean_data.select(pl.col(col_name)).mean()[0, 0]

            # Calculate eigenvalues and eigenvectors
            try:
                eigvals, eigvecs = np.linalg.eig(matrix)

                # Calculate ellipse parameters
                # Convert JAX arrays to numpy for angle calculation
                eigvecs_np = jnp.asarray(eigvecs).astype(float)
                angle = float(jnp.degrees(jnp.arctan2(eigvecs_np[1, 0], eigvecs_np[0, 0])))
                width = 2 * np.abs(eigvals[0])
                height = 2 * np.abs(eigvals[1])

                # Create ellipse patch
                ellipse = plt.matplotlib.patches.Ellipse(
                    (0, 0), width, height,
                    angle=angle,
                    alpha=0.3,
                    label=f"{filter_type}-{config}"
                )
                plt.gca().add_patch(ellipse)

            except np.linalg.LinAlgError as e:
                logging.warning(f"Failed to compute ellipse for {filter_type}-{config}: {e}")

        plt.title("Mean Matrix Eigenstructure Visualization")
        plt.axis('equal')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Overall title and layout
        fig.suptitle(f"{title} Analysis (K=2)", y=1.05, fontsize=14)
        plt.tight_layout()

        # Save plot
        plt.savefig(
            output_dir / f"eigenval_dist_{param}_k2.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()


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

    # Group by filter_type and config_fix_mu
    group_cols = ["filter_type", "config_fix_mu"]

    # Calculate mean errors for each group
    agg_exprs = []
    for metric in relative_metrics + bias_metrics:
        if metric in df.columns:
            agg_exprs.append(pl.col(metric).mean().alias(f"{metric}_mean"))
            agg_exprs.append(pl.col(metric).std().alias(f"{metric}_std"))

    df_grouped = df.group_by(group_cols).agg(agg_exprs)

    # Log findings
    logging.info("\n=== Parameter Identification Difficulty Analysis ===")

    for filter_type in ["BIF", "PF"]:
        for fix_mu in [True, False]:
            group_data = df_grouped.filter(
                (pl.col("filter_type") == filter_type) &
                (pl.col("config_fix_mu") == fix_mu)
            )

            if group_data.height > 0:
                logging.info(f"\n{filter_type} (fix_mu={fix_mu}):")

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

    # Original comparative analysis
    param_metrics = [col for col in df.columns if col.startswith("param_") and col.endswith(("_rmse", "_bias"))]

    for metric in param_metrics:
        bif_stats = df.filter(pl.col("filter_type") == "BIF").select(pl.col(metric).mean())
        pf_stats = df.filter(pl.col("filter_type") == "PF").select(pl.col(metric).mean())

        logging.info(f"\nMetric: {metric}")
        logging.info(f"BIF mean: {bif_stats[0, 0]:.6f}")
        logging.info(f"PF mean: {pf_stats[0, 0]:.6f}")

    # 2. Analyze computational efficiency with detailed timing breakdown
    timing_cols = ["results_steps", "timing_total_script_duration_s"]
    for col in timing_cols:
        bif_time = df.filter(pl.col("filter_type") == "BIF").select(pl.col(col).mean())
        pf_time = df.filter(pl.col("filter_type") == "PF").select(pl.col(col).mean())

        logging.info(f"\nMetric: {col}")
        logging.info(f"BIF mean: {bif_time[0, 0]:.2f}")
        logging.info(f"PF mean: {pf_time[0, 0]:.2f}")
        logging.info(f"Relative difference: {((pf_time[0, 0] - bif_time[0, 0]) / bif_time[0, 0]):.2%}")

    # 3. Analyze state estimation accuracy with relative performance indicators
    accuracy_cols = [col for col in df.columns if col.startswith("accuracy_")]
    for col in accuracy_cols:
        bif_acc = df.filter(pl.col("filter_type") == "BIF").select(pl.col(col).mean())
        pf_acc = df.filter(pl.col("filter_type") == "PF").select(pl.col(col).mean())

        logging.info(f"\nMetric: {col}")
        logging.info(f"BIF mean: {bif_acc[0, 0]:.6f}")
        logging.info(f"PF mean: {pf_acc[0, 0]:.6f}")
        if bif_acc[0, 0] != 0:
            logging.info(f"Relative difference: {((pf_acc[0, 0] - bif_acc[0, 0]) / bif_acc[0, 0]):.2%}")


def analyze_fix_mu_effect(df_success: pl.DataFrame, output_dir: Path) -> None:
    """Analyze the effect of fixing mu during optimization for both BIF and PF filters.

    Args:
        df_success: DataFrame containing successful optimization runs
        output_dir: Directory to save plots and results
    """
    logging.info("Analyzing effect of fixing mu parameter")

    # Ensure config_fix_mu is properly formatted
    df = df_success.filter(pl.col("config_fix_mu").is_not_null())
    df = df.with_columns([
        pl.col("config_fix_mu").cast(pl.Int8).alias("config_fix_mu"),
        pl.concat_str([
            "N", pl.col("N").cast(pl.Utf8),
            "-K", pl.col("K").cast(pl.Utf8),
            "-T", pl.col("config_T").cast(pl.Utf8)
        ]).alias("config_label")
    ])

    # Define parameters to analyze (excluding mu itself)
    params_to_analyze = {
        "lambda_r": "Risk Premium",
        "Phi_f": "State Transition",
        "Phi_h": "Volatility Transition",
        "Q_h": "Volatility Covariance"
    }

    # Define scalar metrics to analyze
    scalar_metrics = [
        "timing_total_script_duration_s",
        "accuracy_state_estimation_factor_correlation_mean",
        "accuracy_state_estimation_volatility_correlation_mean"
    ]

    # Group data
    group_cols = ["filter_type", "N", "K", "config_T", "config_num_particles", "config_fix_mu", "config_label"]

    # Create figure for parameter errors
    plt.figure(figsize=(15, 10))
    for idx, (param, title) in enumerate(params_to_analyze.items(), 1):
        plt.subplot(2, 2, idx)
        error_col = f"param_{param}_rmse"

        if error_col in df.columns:
            sns.boxplot(
                data=df.to_pandas(),
                x="filter_type",
                y=error_col,
                hue="config_fix_mu",
                palette="Set2",
                showfliers=False
            )
            plt.title(f"{title} RMSE by Fix Mu Setting")
            plt.xticks(rotation=45)
            plt.legend(title="Fix Mu", labels=["False", "True"])

    plt.tight_layout()
    plt.savefig(output_dir / "fix_mu_param_errors.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Aggregate statistics
    agg_expressions = []
    for metric in [*[f"param_{p}_rmse" for p in params_to_analyze], *scalar_metrics]:
        if metric in df.columns:
            agg_expressions.extend([
                pl.col(metric).mean().alias(f"{metric}_mean"),
                pl.col(metric).std().alias(f"{metric}_std")
            ])

    df_agg = df.group_by(group_cols).agg(agg_expressions)

    # Log aggregated findings
    logging.info("\nFix Mu Analysis Results:")
    for filter_type in ["BIF", "PF"]:
        logging.info(f"\n{filter_type} Results:")
        df_filter = df_agg.filter(pl.col("filter_type") == filter_type)

        for config in df_filter.select("config_label").unique().to_series():
            df_config = df_filter.filter(pl.col("config_label") == config)
            logging.info(f"\nConfiguration: {config}")

            for metric in [*[f"param_{p}_rmse_mean" for p in params_to_analyze], *[f"{m}_mean" for m in scalar_metrics]]:
                if metric in df_config.columns:
                    fix_true = df_config.filter(pl.col("config_fix_mu") == 1).select(pl.col(metric))[0, 0]
                    fix_false = df_config.filter(pl.col("config_fix_mu") == 0).select(pl.col(metric))[0, 0]
                    diff_pct = ((fix_true - fix_false) / fix_false) * 100
                    logging.info(f"{metric}: Fix=True: {fix_true:.6f}, Fix=False: {fix_false:.6f} (Diff: {diff_pct:+.2f}%)")

    # Create time series plot
    plt.figure(figsize=(12, 8))
    for metric in scalar_metrics:
        if metric in df.columns:
            plt.subplot(len(scalar_metrics), 1, scalar_metrics.index(metric) + 1)
            sns.lineplot(
                data=df.to_pandas(),
                x="config_T",
                y=metric,
                hue="filter_type",
                style="config_fix_mu",
                markers=True,
                dashes=False
            )
            plt.title(f"{metric.replace('_', ' ').title()} vs Time Series Length")
            plt.legend(title="Filter Type / Fix Mu")

    plt.tight_layout()
    plt.savefig(output_dir / "fix_mu_time_series.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save aggregated stats
    output_path = output_dir / "fix_mu_analysis.csv"
    df_agg.write_csv(output_path)
    logging.info(f"\nAggregated fix mu analysis saved to: {output_path}")


def plot_time_scaling(df_agg: pl.DataFrame, output_dir: Path) -> List[Dict]:
    """Create line plots showing how computation time scales with N, K, and T.

    Args:
        df_agg: DataFrame containing aggregated scalar metrics
        output_dir: Directory to save plots

    Returns:
        List of dictionaries containing complexity analysis results for all dimensions.
    """
    # Set style for all plots
    sns.set_style("whitegrid")
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    
    all_complexity_results = [] # Initialize list to store results across dimensions

    # Create faceted plots for each dimension (N, K, T)
    dimensions = [
        ("N", "K", "config_T", "Number of Assets"),  # Plot vs N, facet by K and T
        ("K", "N", "config_T", "Number of Factors"),  # Plot vs K, facet by N and T
        ("config_T", "N", "K", "Time Series Length")  # Plot vs T, facet by N and K
    ]

    for x_col, row_col, col_col, x_label in dimensions:
        # Ensure numerical types for plotting
        df_plot = df_agg.with_columns([
            pl.col(x_col).cast(pl.Int64),
            pl.col(row_col).cast(pl.Int64),
            pl.col(col_col).cast(pl.Int64)
        ])

        # Create figure with specified size
        g = sns.FacetGrid(
            data=df_plot.to_pandas(),
            row=row_col,
            col=col_col,
            hue="filter_type",
            height=4,
            aspect=1.5,
            palette="Set2"  # Use consistent color palette
        )

        # Plot timing vs dimension with error bands
        # Use hue for color and style for markers/dashes if needed, but avoid redundant style="filter_type"
        g.map_dataframe(
            sns.lineplot,
            x=x_col,
            y="timing_total_script_duration_s_mean",
            # style="filter_type", # Removed redundant style mapping
            errorbar=("ci", 95),  # Add 95% confidence intervals
            markers=True,  # Add markers to lines
            dashes=False  # Use solid lines
        )

        # Customize axes and labels
        g.set_axis_labels(
            x_label,
            "Mean Computation Time (s)"
        )
        
        # Set plot title
        g.fig.suptitle(
            f"Computation Time Scaling with {x_label}",
            y=1.02,
            fontsize=14,
            weight='bold'
        )
        
        # Customize each subplot
        for ax in g.axes.flat:
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=45)
            
            # Add gridlines
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Set log scale for timing if the spread is large
            if df_plot.select(pl.col("timing_total_script_duration_s_mean")).max()[0, 0] / \
               df_plot.select(pl.col("timing_total_script_duration_s_mean")).min()[0, 0] > 10:
                ax.set_yscale('log')
                ax.set_ylabel("Mean Computation Time (s, log scale)")
            
            # Add subplot dimensions as text
            row_val = ax.get_title().split(" = ")[-1] if ax.get_title() else ""
            col_val = g.col_names[ax.get_subplotspec().colspan.start] if g.col_names else ""
            ax.text(0.02, 0.98, f"{row_col}={row_val}\n{col_col}={col_val}",
                   transform=ax.transAxes, fontsize=8, va='top')
            
            def fit_and_analyze_complexity(x_vals, y_vals):
                """Helper function to fit polynomial and determine complexity."""
                # Fit polynomials of different degrees
                fits = []
                r2_scores = []
                for degree in [1, 2, 3]:
                    coeffs = np.polyfit(x_vals, y_vals, degree)
                    y_pred = np.polyval(coeffs, x_vals)
                    r2 = 1 - np.sum((y_vals - y_pred)**2) / np.sum((y_vals - np.mean(y_vals))**2)
                    fits.append((coeffs, r2))
                    r2_scores.append(r2)

                # Select best fit based on R² improvement threshold
                best_degree = 1
                for i in range(1, len(r2_scores)):
                    if r2_scores[i] > r2_scores[i-1] * 1.1:  # 10% improvement threshold
                        best_degree = i + 1
                
                coeffs, r2 = fits[best_degree - 1]
                
                # Determine complexity class
                if best_degree == 3:
                    complexity = "O(n³)"
                elif best_degree == 2:
                    complexity = "O(n²)"
                else:
                    complexity = "O(n)"
                
                return coeffs, r2, complexity, best_degree

            # Fit polynomials and add trendlines for each filter type
            filter_types = df_plot["filter_type"].unique()
            for idx, filter_type in enumerate(filter_types):
                filter_data = df_plot.filter(pl.col("filter_type") == filter_type)
                x_vals = filter_data.select(pl.col(x_col)).to_numpy().flatten()
                y_vals = filter_data.select(pl.col("timing_total_script_duration_s_mean")).to_numpy().flatten()
                
                if len(x_vals) > 3:  # Need at least 4 points for cubic fit
                    # Analyze complexity
                    coeffs, r2, complexity, degree = fit_and_analyze_complexity(x_vals, y_vals)
                    
                    # Generate smooth fit line
                    x_fit = np.linspace(min(x_vals), max(x_vals), 100)
                    y_fit = np.polyval(coeffs, x_fit)
                    
                    # Add trendline (without explicit label)
                    color = sns.color_palette("Set2")[idx]
                    ax.plot(x_fit, y_fit, '--', alpha=0.5, color=color) # Removed label argument
                    
                    # Add complexity and R² annotation
                    ax.text(0.98, 0.98 - 0.1 * idx,
                           f"{filter_type}: {complexity}\n(Fit R²={r2:.3f}, Deg={degree})", # Added degree info
                           transform=ax.transAxes, fontsize=8,
                           ha='right', va='top', color=color)
            
        # Collect complexity analysis results
        complexity_results = []
        for ax in g.axes.flat:
            row_val = ax.get_title().split(" = ")[-1] if ax.get_title() else ""
            col_val = g.col_names[ax.get_subplotspec().colspan.start] if g.col_names else ""
            
            for idx, filter_type in enumerate(filter_types):
                filter_data = df_plot.filter(
                    (pl.col("filter_type") == filter_type) &
                    (pl.col(row_col) == int(row_val)) &
                    (pl.col(col_col) == int(col_val))
                )
                
                x_vals = filter_data.select(pl.col(x_col)).to_numpy().flatten()
                y_vals = filter_data.select(pl.col("timing_total_script_duration_s_mean")).to_numpy().flatten()
                
                if len(x_vals) > 3:
                    coeffs, r2, complexity, degree = fit_and_analyze_complexity(x_vals, y_vals)
                    complexity_results.append({
                        "filter_type": filter_type,
                        "dimension": x_label,
                        f"{row_col}_value": int(row_val),
                        f"{col_col}_value": int(col_val),
                        "complexity": complexity,
                        "polynomial_degree": degree,
                        "r_squared": r2,
                        "coefficients": list(coeffs),
                    })
        
        # Save complexity analysis results
        if complexity_results:
            df_complexity = pl.DataFrame(complexity_results)
            output_file = output_dir / f"complexity_analysis_{x_col}.csv"
            df_complexity.write_csv(output_file)
            logging.info(f"Saved complexity analysis for {x_label} to {output_file}")
            all_complexity_results.extend(complexity_results) # Append results for this dimension
        
        # Add legend with better positioning and formatting
        g.add_legend(
            title="Filter Type",
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            frameon=True,
            framealpha=0.95,
            edgecolor='lightgray'
        )

        # Save plot
        plt.savefig(
            output_dir / f"time_scaling_{x_col}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    return all_complexity_results # Return the consolidated list


def aggregate_scalar_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate aggregate statistics for scalar metrics grouped by configuration.

    Args:
        df: Input DataFrame containing scalar metrics

    Returns:
        DataFrame with aggregated metrics by configuration
    """
    # Define metrics to aggregate
    scalar_metrics = [
        'results_steps',
        'timing_total_script_duration_s',
        'accuracy_state_estimation_factor_rmse_mean',
        'accuracy_state_estimation_factor_correlation_mean',
        'accuracy_state_estimation_volatility_rmse_mean',
        'accuracy_state_estimation_volatility_correlation_mean',
        'loss_diff'
    ]

    # Create expressions for mean, median, and std for each metric
    agg_expressions = []
    for metric in scalar_metrics:
        agg_expressions.extend([
            pl.col(metric).mean().alias(f"{metric}_mean"),
            pl.col(metric).median().alias(f"{metric}_median"),
            pl.col(metric).std().alias(f"{metric}_std")
        ])

    # Fill null values for config_num_particles and config_fix_mu
    df = df.with_columns([
        pl.col("config_num_particles").fill_null(0),
        pl.col("config_fix_mu").fill_null(-1)
    ])

    # Group by expanded configuration and calculate aggregates
    return df.group_by([
        'filter_type', 'N', 'K', 'config_T',
        'config_num_particles', 'config_fix_mu']
    ).agg(
        agg_expressions
    ).sort(
            ['filter_type', 'N', 'K', 'config_T', 'config_num_particles', 'config_fix_mu']
    )


def generate_summary_tables(df_success: pl.DataFrame, df_agg_scalars: pl.DataFrame, output_dir: Path) -> None:
    """Generate summary tables comparing BIF and PF performance.

    Args:
        df_success: DataFrame containing successful runs with parameter errors
        df_agg_scalars: DataFrame containing aggregated scalar metrics
        output_dir: Directory to save the summary tables
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Parameter Error Summary Table
    param_metrics = {
        "param_lambda_r_rmse": "Risk Premium RMSE",
        "param_Phi_f_frob_rel_diff": "State Transition Matrix Rel. Frob. Diff",
        "param_Q_h_logdet_diff": "Volatility Covar. LogDet Diff",
        "param_Phi_h_frob_rel_diff": "Volatility Trans. Matrix Rel. Frob. Diff"
    }

    # Aggregate parameter errors by filter type and configuration
    param_error_exprs = []
    for metric in param_metrics.keys():
        param_error_exprs.extend([
            pl.col(metric).mean().alias(f"{metric}_mean"),
            pl.col(metric).std().alias(f"{metric}_std")
        ])

    df_param_summary = (
        df_success
        .group_by(["filter_type", "N", "K", "config_T", "config_num_particles", "config_fix_mu"])
        .agg(param_error_exprs)
        .sort(["filter_type", "N", "K", "config_T", "config_num_particles", "config_fix_mu"])
    )

    # 2. Scalar Metrics Summary Table
    scalar_metrics = {
        "accuracy_state_estimation_factor_correlation_mean_mean": "Factor State Correlation", # Added _mean suffix
        "accuracy_state_estimation_volatility_correlation_mean_mean": "Volatility State Correlation", # Added _mean suffix
        "timing_total_script_duration_s_mean": "Computation Time (s)",
        "results_steps_mean": "Optimization Steps"
    }

    # Create pivoted comparison tables
    for table_type, metrics in [("param_errors", param_metrics), ("scalar_metrics", scalar_metrics)]:
        # For each distinct configuration
        configs = df_param_summary if table_type == "param_errors" else df_agg_scalars
        config_groups = configs.group_by(["N", "K", "config_T"]).agg([])

        for config_row in config_groups.iter_rows(named=True):
            logging.info(f"Columns available in df_agg_scalars for summary tables: {df_agg_scalars.columns}") # Indentation fixed
            N, K, T = config_row["N"], config_row["K"], config_row["config_T"]

            # Filter data for this configuration
            if table_type == "param_errors":
                config_data = df_param_summary.filter(
                    (pl.col("N") == N) &
                    (pl.col("K") == K) &
                    (pl.col("config_T") == T)
                )
            else:
                config_data = df_agg_scalars.filter(
                    (pl.col("N") == N) &
                    (pl.col("K") == K) &
                    (pl.col("config_T") == T)
                )

            # Pivot the data for BIF vs PF comparison
            comparison_rows = []
            for metric, metric_name in metrics.items():
                if table_type == "param_errors":
                    mean_col = f"{metric}_mean"
                    std_col = f"{metric}_std"
                else:
                    # Correctly handle double _mean suffix in column names
                    if "_mean_mean" in metric:
                        mean_col = metric
                        std_col = metric.replace("_mean_mean", "_mean_std")
                    else:
                        mean_col = metric
                        std_col = metric.replace("_mean", "_std")

                bif_values = config_data.filter(pl.col("filter_type") == "BIF").select([mean_col, std_col])
                pf_values = config_data.filter(pl.col("filter_type") == "PF").select([mean_col, std_col])

                if bif_values.height > 0 and pf_values.height > 0:
                    row = {
                        "Metric": metric_name,
                        "BIF (Mean ± Std)": f"{bif_values[0, 0]:.6f} ± {bif_values[0, 1]:.6f}",
                        "PF (Mean ± Std)": f"{pf_values[0, 0]:.6f} ± {pf_values[0, 1]:.6f}"
                    }
                    comparison_rows.append(row)

            # Create and save the comparison table
            if comparison_rows:
                df_comparison = pl.DataFrame(comparison_rows)
                output_file = output_dir / f"summary_{table_type}_N{N}_K{K}_T{T}_{timestamp}.csv"
                df_comparison.write_csv(output_file)
                logging.info(f"Saved {table_type} summary table for N={N}, K={K}, T={T} to {output_file}")


def main() -> None:
    """Main execution function."""
    args = parse_args()
    setup_logging(args.log_level)

    # Convert paths
    metrics_path = Path(args.metrics_csv)
    params_path = Path(args.params_npz)

    try:
        # Create output directory for all results
        output_dir = create_output_directory()
        logging.info("Created output directory: %s", output_dir)

        # Phase 1: Load and clean metrics data
        df_metrics = load_and_validate_metrics(metrics_path)
        df_success = filter_successful_runs(df_metrics)

        # Load parameter data
        true_params_dict, estimated_params_dict = load_and_validate_params(params_path)
        logging.info("Phase 1 completed successfully")

        # Phase 2: Parameter Error Analysis
        df_param_errors = create_param_errors_df(df_success, true_params_dict, estimated_params_dict)
        df_success = df_success.join(df_param_errors, on="unique_id", how="left")
        logging.info("Phase 2 completed successfully")

        # Phase 3: Scalar Metric Analysis
        df_agg_scalars = calculate_scalar_metrics(df_success)

        # Save Phase 3 results to output_dir (not directly to outputs/)
        output_path = output_dir / "scalar_metrics_analysis.csv"
        df_agg_scalars.write_csv(output_path)
        logging.info("Phase 3 results saved to: %s", output_path)

        # Phase 4: Comparative Analysis & Visualization
        logging.info("Starting Phase 4: Comparative Analysis & Visualization")

        # Create output directory for plots
        output_dir = create_output_directory()

        # Perform comparative analysis
        logging.info("Performing comparative analysis between BIF and PF")
        perform_comparative_analysis(df_success, df_agg_scalars)

        # Generate visualizations
        logging.info("Generating visualization plots")
        plot_scatter_comparison(df_success, output_dir)
        plot_error_heatmaps(df_success, output_dir)
        plot_error_boxplots(df_success, output_dir)
        plot_k2_eigenvalue_distributions(df_success, output_dir)

        # Add time scaling plots and collect complexity results
        logging.info("Generating time scaling plots")
        all_complexity_results = plot_time_scaling(df_agg_scalars, output_dir)

        # Save consolidated complexity analysis
        if all_complexity_results:
            df_consolidated_complexity = pl.DataFrame(all_complexity_results)
            complexity_output_path = output_dir / "consolidated_complexity_analysis.csv"
            df_consolidated_complexity.write_csv(complexity_output_path)
            logging.info(f"Saved consolidated complexity analysis to {complexity_output_path}")
        else:
            logging.warning("No complexity analysis results generated.")

        logging.info("Phase 4 completed successfully. Plots saved to: %s", output_dir)
        # Phase 5: Generate Summary Tables
        logging.info("Starting Phase 5: Final Reporting and Output")
        # Generate summary tables
        generate_summary_tables(df_success, df_agg_scalars, output_dir)
        logging.info("Generated summary tables")

        # Aggregate parameter errors using refined grouping
        param_error_cols = [col for col in df_success.columns if col.startswith("param_")]
        group_cols_refined = [
            "filter_type",
            "N",
            "K",
            "config_T",
            "config_num_particles",
            "config_fix_mu"
        ]

        agg_param_exprs = []
        for metric in param_error_cols:
            agg_param_exprs.extend([
                pl.col(metric).mean().alias(f"{metric}_mean"),
                pl.col(metric).median().alias(f"{metric}_median"),
                pl.col(metric).std().alias(f"{metric}_std")
            ])

        df_agg_param_errors = df_success.group_by(group_cols_refined).agg(agg_param_exprs)
        logging.info("Created aggregated parameter errors DataFrame with shape: %s", df_agg_param_errors.shape)

        # Define timestamped output filenames
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = Path("outputs")
        full_data_path = output_base / f"analysis_full_data_{timestamp_str}.csv"
        agg_scalars_path = output_base / f"analysis_agg_scalars_{timestamp_str}.csv"
        agg_param_errors_path = output_base / f"analysis_agg_param_errors_{timestamp_str}.csv"

        # Save DataFrames
        logging.info("Saving processed DataFrames...")
        df_success.write_csv(full_data_path)
        logging.info("Saved full per-replicate data (with errors) to %s", full_data_path)
        df_agg_scalars.write_csv(agg_scalars_path)
        logging.info("Saved aggregated scalar metrics to %s", agg_scalars_path)
        df_agg_param_errors.write_csv(agg_param_errors_path)
        logging.info("Saved aggregated parameter errors to %s", agg_param_errors_path)

        # Final summary log messages
        logging.info("Phase 5: Final Reporting and Output completed successfully")
        logging.info("\nAnalysis Results Summary:")
        logging.info("------------------------")
        logging.info("1. Data Processing Outputs:")
        logging.info("  - Full Data with Errors: %s", full_data_path)
        logging.info("  - Aggregated Scalar Metrics: %s", agg_scalars_path)
        logging.info("  - Aggregated Parameter Errors: %s", agg_param_errors_path)
        logging.info("\n2. Analysis Outputs:")
        logging.info("  - Visualization Plots: %s/", output_dir)
        logging.info("  - Scalar Metrics Analysis: %s", output_dir / "scalar_metrics_analysis.csv")
        logging.info("  - Summary Tables: %s/", output_dir)
        logging.info("  - Complexity Analysis: %s", output_dir / "consolidated_complexity_analysis.csv")
        logging.info("\n3. Additional Information:")
        logging.info("  - Total successful runs analyzed: %d", df_success.height)
        logging.info("  - Configuration combinations: %d", df_agg_scalars.height)
        logging.info("  - Timestamp: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logging.info("\nAnalysis completed successfully. All outputs saved to: %s", output_dir)

    except Exception as e:
        logging.error("Error during analysis: %s", str(e))
        raise


if __name__ == "__main__":
    main()