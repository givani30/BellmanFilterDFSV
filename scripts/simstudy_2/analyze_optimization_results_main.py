 #!/usr/bin/env python
"""Main orchestration script for optimization results analysis.

This script coordinates the complete analysis pipeline by utilizing specialized utility modules:
1. Data Loading and Preparation (analysis_data_utils)
2. Parameter Error Analysis (analysis_error_metrics)
3. Scalar Metric Analysis (analysis_scalar_metrics)
4. Visualization Generation (analysis_plotting_utils)
5. Final Report Generation (analysis_table_utils)

The script focuses on orchestration, delegating specific functionality to dedicated utility modules
while maintaining a clear overview of the analysis workflow.
"""

import argparse
from datetime import datetime
import logging
import sys
from pathlib import Path

import polars as pl

from analysis_data_utils import (
    load_and_validate_metrics,
    load_and_validate_params,
    filter_successful_runs
)
from analysis_error_metrics import create_param_errors_df
from analysis_scalar_metrics import calculate_scalar_metrics
from analysis_plotting_utils import (
    plot_scatter_comparison,
    plot_error_heatmaps,
    plot_k2_eigenvalue_distributions,
    create_time_scaling_plots,
    create_error_scaling_plots
)
from analysis_specific_utils import analyze_fix_mu_effect
from analysis_table_utils import generate_summary_tables
from analysis_io_utils import create_output_directory, save_analysis_output
import matplotlib.pyplot as plt


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
        default="outputs/aggregated_optimization_metrics_21-04-2025.csv",
        help="Path to the metrics CSV file"
    )
    parser.add_argument(
        "--params-npz",
        type=str,
        default="outputs/aggregated_optimization_params_21-04-2025.npz",
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
            .otherwise(pl.lit("Unknown"))  # Handle unexpected cases
            .alias("filter_config"),

            # Fill nulls for config_fix_mu early
            pl.col("config_fix_mu").fill_null(-1).cast(pl.Int8).alias("config_fix_mu"),

            # Fill nulls for config_num_particles
            pl.col("config_num_particles").fill_null(0).cast(pl.Int64).alias("config_num_particles")
        ])

        # Log data summary
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
        save_analysis_output(df_agg_scalars, "scalar_metrics_analysis", output_dir)

        # Log summary statistics
        logging.info("\nScalar Metrics Summary by Filter Configuration:")
        for cfg in filter_configs.to_series():
            cfg_data = df_agg_scalars.filter(pl.col("filter_config") == cfg)
            if cfg_data.height > 0:
                logging.info(f"\n  {cfg}:")
                timing_mean = cfg_data.select(pl.col("timing_total_script_duration_s_mean").mean()).item()
                logging.info(f"    Average computation time: {timing_mean:.2f}s")
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
        scatter_figures = plot_scatter_comparison(df_success) # Use scatter_figures as per user example
        logging.info("Saving scatter comparison plots...")
        for param_name, figure in scatter_figures.items(): # Use param_name, figure as per user example
            save_analysis_output(figure, f"scatter_{param_name}_comparison", output_dir) # Use scatter_{param_name}_comparison as per user example
            plt.close(figure) # Add plt.close()
        logging.info("Scatter comparison plots saved successfully")

        # heatmap_fig = plot_error_heatmaps(df_success)
        # save_analysis_output(heatmap_fig, "error_heatmaps", output_dir)

        eigenval_fig = plot_k2_eigenvalue_distributions(
            df_success,
            true_params_dict,
            estimated_params_dict
        )
        save_analysis_output(eigenval_fig, "eigenvalue_distributions", output_dir)

        logging.info("\nGenerating time scaling plots")
        time_scaling_figs_dict = create_time_scaling_plots(df_agg_scalars)
        logging.info("Saving time scaling plots...")
        for fig_name, fig in time_scaling_figs_dict.items():
            save_analysis_output(fig, f"time_{fig_name}", output_dir)
            plt.close(fig) # Ensure figure is closed after saving
        logging.info("Time scaling plots generated and saved successfully")

        logging.info("\nGenerating fix mu effect analysis plots")
        fixmu_fig = analyze_fix_mu_effect(df_success)
        save_analysis_output(fixmu_fig, "fix_mu_effect", output_dir)
        logging.info("Fix mu effect analysis plots generated successfully")

        # Generate error scaling plots for specified metrics
        logging.info("\nGenerating error scaling plots for various metrics")

        # First, let's log the available columns to help with debugging
        logging.info("Available columns in df_success:")
        for col in df_success.columns:
            if "rmse" in col.lower():
                logging.info(f"  {col}")

        # Map the requested metrics to actual column names
        metric_mapping = {
            'accuracy_parameter_estimation_K_rmse': 'param_K_rmse',
            'accuracy_parameter_estimation_N_rmse': 'param_N_rmse',
            'accuracy_parameter_estimation_Phi_f_rmse': 'param_Phi_f_rmse',
            'accuracy_parameter_estimation_Phi_h_rmse': 'param_Phi_h_rmse',
            'accuracy_parameter_estimation_Q_h_rmse': 'param_Q_h_rmse',
            'accuracy_parameter_estimation_lambda_r_rmse': 'param_lambda_r_rmse',
            'accuracy_parameter_estimation_mu_rmse': 'param_mu_rmse',
            'accuracy_parameter_estimation_sigma2_rmse': 'param_sigma2_rmse',
            'accuracy_state_estimation_factor_rmse_mean': 'accuracy_state_estimation_factor_rmse_mean',
            'accuracy_state_estimation_volatility_rmse_mean': 'accuracy_state_estimation_volatility_rmse_mean'
        }

        # Create a copy of df_success with renamed columns for T
        df_for_scaling = df_success.clone()
        df_for_scaling = df_for_scaling.with_columns([
            pl.col("config_T").alias("T")
        ])

        for requested_metric, actual_column in metric_mapping.items():
            try:
                if actual_column in df_for_scaling.columns:
                    logging.info(f"  Creating error scaling plot for {requested_metric}")
                    error_scaling_fig = create_error_scaling_plots(df_for_scaling, metric_col=actual_column)
                    save_analysis_output(error_scaling_fig, f"error_scaling_{requested_metric}", output_dir)
                    plt.close(error_scaling_fig)  # Close figure after saving
                    logging.info(f"  Error scaling plot for {requested_metric} saved successfully")
                else:
                    logging.warning(f"  Column {actual_column} not found in DataFrame for {requested_metric}")
            except Exception as e:
                logging.warning(f"  Failed to create error scaling plot for {requested_metric}: {str(e)}")

        logging.info("\nPhase 4 completed successfully")
        phase4_time = datetime.now() - start_time
        logging.info(f"Time taken: {phase4_time}")

        # Phase 5: Final Reporting
        logging.info("\nPhase 5: Final Reporting and Output")
        start_time = datetime.now()
        summary_tables = generate_summary_tables(df_success, df_agg_scalars)
        for name, table in summary_tables.items():
            save_analysis_output(table, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", output_dir)
        logging.info("Summary tables generated successfully")

        # Save final outputs
        save_analysis_output(df_success, "analysis_full_data", output_dir)
        save_analysis_output(df_agg_scalars, "analysis_agg_scalars", output_dir)

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

        logging.info("\nAll outputs saved to: %s", output_dir)

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
