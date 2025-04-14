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
        if len(true.shape) == 2 and true.shape == (3, 3):  # Matrix parameter
            for i in range(3):
                for j in range(3):
                    errors[f"param_{name}_element_{i}_{j}_bias"] = float(diff[i, j])
        
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
    
    # Define grouping columns
    group_cols = ["filter_type", "N", "K", "config_T"]
    
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
    
    # Parameters to plot (diagonals and key elements)
    params_to_plot = {
        "Phi_f": "State Transition Matrix",
        "Q_h": "Volatility Covariance Matrix",
        "sigma2": "Observation Noise Variance"
    }
    
    for param, title in params_to_plot.items():
        plt.figure(figsize=(10, 8))
        
        # Extract bias and RMSE for parameter
        bias_col = f"param_{param}_bias"
        rmse_col = f"param_{param}_rmse"
        
        # Create scatter plot by filter type
        sns.scatterplot(
            data=df.to_pandas(),
            x=bias_col,
            y=rmse_col,
            hue="filter_type",
            style="filter_type",
            alpha=0.7
        )
        
        plt.title(f"{title} - Estimation Errors")
        plt.xlabel("Bias")
        plt.ylabel("RMSE")
        
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
    for i in range(3):
        for j in range(3):
            key = f"param_{param_name}_element_{i}_{j}_bias"
            errors[key] = float(est_matrix[i, j] - true_matrix[i, j])
    
    return errors

def plot_error_heatmaps(df: pl.DataFrame, output_dir: Path) -> None:
    """Create heatmaps showing average element-wise differences for matrix parameters.
    
    Args:
        df: DataFrame containing parameter errors
        output_dir: Directory to save plots
    """
    matrix_params = ["Phi_f", "Phi_h", "Q_h"]
    
    for param in matrix_params:
        plt.figure(figsize=(12, 5))
        
        # Create subplots for BIF and PF
        for idx, filter_type in enumerate(["BIF", "PF"]):
            plt.subplot(1, 2, idx + 1)
            
            try:
                # Get data for this filter type
                filter_data = df.filter(pl.col("filter_type") == filter_type)
                
                # Construct matrix from element-wise errors
                matrix_data = np.zeros((3, 3))
                for i in range(3):
                    for j in range(3):
                        col_name = f"param_{param}_element_{i}_{j}_bias"
                        if col_name in filter_data.columns:
                            matrix_data[i, j] = filter_data.select(pl.col(col_name)).mean()[0, 0]
                
                # Create heatmap
                sns.heatmap(
                    data=matrix_data,
                    cmap="RdBu_r",
                    center=0,
                    annot=True,
                    fmt=".3f"
                )
                
                plt.title(f"{filter_type} - {param} Mean Errors")
                
            except Exception as e:
                logging.error(f"Error creating heatmap for {param} ({filter_type}): {str(e)}")
                plt.text(0.5, 0.5, 'Error generating heatmap',
                        ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_{param}_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_error_boxplots(df: pl.DataFrame, output_dir: Path) -> None:
    """Create box plots comparing error distributions between BIF and PF.
    
    Args:
        df: DataFrame containing error metrics
        output_dir: Directory to save plots
    """
    # Error metrics to plot
    error_metrics = [
        "param_Phi_f_frob_diff",
        "param_Q_h_frob_diff",
        "accuracy_state_estimation_factor_rmse_mean",
        "accuracy_state_estimation_volatility_rmse_mean"
    ]
    
    for metric in error_metrics:
        plt.figure(figsize=(12, 6))
        
        sns.boxplot(
            data=df.to_pandas(),
            x="N",
            y=metric,
            hue="filter_type",
            palette="Set2"
        )
        
        plt.title(f"Distribution of {metric} by N and Filter Type")
        plt.xticks(rotation=45)
        
        plt.savefig(output_dir / f"boxplot_{metric}.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_k2_eigenvalue_distributions(df: pl.DataFrame, output_dir: Path) -> None:
    """Create distribution plots for K=2 configurations comparing eigenvalue errors.
    
    Args:
        df: DataFrame containing parameter estimates
        output_dir: Directory to save plots
    """
    # Filter for K=2 configurations
    df_k2 = df.filter(pl.col("K") == 2)
    
    if df_k2.height == 0:
        logging.info("No K=2 configurations found for eigenvalue plots")
        return
    
    for param in ["Phi_f", "Phi_h", "Q_h"]:
        plt.figure(figsize=(10, 6))
        
        # Plot eigenvalue error distributions
        metric = f"param_{param}_eig_rmse"
        
        sns.boxplot(
            data=df_k2.to_pandas(),
            x="filter_type",
            y=metric,
            palette="Set2"
        )
        
        plt.title(f"{param} Eigenvalue RMSE Distribution (K=2)")
        plt.ylabel("RMSE")
        plt.xlabel("Filter Type")
        
        plt.savefig(output_dir / f"eigenval_dist_{param}_k2.png", dpi=300, bbox_inches="tight")
        plt.close()


def perform_comparative_analysis(df: pl.DataFrame, df_agg: pl.DataFrame) -> None:
    """Perform comparative analysis between BIF and PF performance.
    
    Args:
        df: Full DataFrame with all metrics
        df_agg: Aggregated scalar metrics DataFrame
    """
    # 1. Analyze parameter estimation accuracy
    param_metrics = [col for col in df.columns if col.startswith("param_") and col.endswith(("_rmse", "_bias"))]
    
    for metric in param_metrics:
        bif_stats = df.filter(pl.col("filter_type") == "BIF").select(pl.col(metric).mean())
        pf_stats = df.filter(pl.col("filter_type") == "PF").select(pl.col(metric).mean())
        
        logging.info(f"\nMetric: {metric}")
        logging.info(f"BIF mean: {bif_stats[0, 0]:.6f}")
        logging.info(f"PF mean: {pf_stats[0, 0]:.6f}")
    
    # 2. Analyze computational efficiency
    timing_cols = ["results_steps", "timing_total_script_duration_s"]
    for col in timing_cols:
        bif_time = df.filter(pl.col("filter_type") == "BIF").select(pl.col(col).mean())
        pf_time = df.filter(pl.col("filter_type") == "PF").select(pl.col(col).mean())
        
        logging.info(f"\nMetric: {col}")
        logging.info(f"BIF mean: {bif_time[0, 0]:.2f}")
        logging.info(f"PF mean: {pf_time[0, 0]:.2f}")
    
    # 3. Analyze state estimation accuracy
    accuracy_cols = [col for col in df.columns if col.startswith("accuracy_")]
    for col in accuracy_cols:
        bif_acc = df.filter(pl.col("filter_type") == "BIF").select(pl.col(col).mean())
        pf_acc = df.filter(pl.col("filter_type") == "PF").select(pl.col(col).mean())
        
        logging.info(f"\nMetric: {col}")
        logging.info(f"BIF mean: {bif_acc[0, 0]:.6f}")
        logging.info(f"PF mean: {pf_acc[0, 0]:.6f}")


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
    
    # Group by configuration and calculate aggregates
    return df.group_by(
        ['filter_type', 'N', 'K', 'config_T']
    ).agg(
        agg_expressions
    ).sort(
        ['filter_type', 'N', 'K', 'config_T']
    )


def main() -> None:
    """Main execution function."""
    args = parse_args()
    setup_logging(args.log_level)

    # Convert paths
    metrics_path = Path(args.metrics_csv)
    params_path = Path(args.params_npz)

    try:
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
        
        # Save Phase 3 results
        output_path = Path("outputs/scalar_metrics_analysis.csv")
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
        
        logging.info("Phase 4 completed successfully. Plots saved to: %s", output_dir)
        
        # Placeholder for Phase 5: Results Visualization
        # TODO: Implement final reporting and summary
        
    except Exception as e:
        logging.error("Error during analysis: %s", str(e))
        raise


if __name__ == "__main__":
    main()