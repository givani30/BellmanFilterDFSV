#!/usr/bin/env python
"""
In-Sample Analysis - Model Estimation (Full Sample)

This script estimates parameters for the DFSV model using the BIF filter
and saves the entire result object to a pickle file.
"""

import polars as pl
import numpy as np
import os
import time
import jax
import jax.numpy as jnp
from pathlib import Path
from datetime import datetime
import cloudpickle
import pickle

# Import DFSV model and filter components
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.utils.optimization import run_optimization, FilterType, OptimizerResult
from bellman_filter_dfsv.filters.particle import DFSVParticleFilter
from bellman_filter_dfsv.utils.optimization_helpers import create_stable_initial_params


def main():
    """Main function to run the model fitting and save results."""
    print("## In-Sample Analysis")
    print("Phase 1: Model Estimation (Full Sample)")

    # Create output directory
    output_dir = Path("outputs/empirical/insample")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("### Loading Data")
    df = pl.read_csv("scripts/empirical/vw_returns_final.csv")
    print(f"Loaded data shape: {df.shape}")

    N = 95  # Number of assets
    K = 5   # Number of factors
    returns_jax = df.to_jax()
    print(f"Returns data shape: {returns_jax.shape}")

    # Create initial parameters
    print("### Estimating DFSV-BIF Model")
    initial_parameters = create_stable_initial_params(N, K)

    # Create BIF filter
    bif_filter = DFSVBellmanInformationFilter(N, K)

    # Run optimization with BIF
    print("Starting BIF optimization...")
    bif_result = run_optimization(
        filter_type=FilterType.BIF,
        returns=returns_jax,
        initial_params=initial_parameters,
        optimizer_name="DampedTrustRegionBFGS",
        use_transformations=True,
        max_steps=1000,
        stability_penalty_weight=1000.0,
        verbose=True,
        log_params=False,  # Enable parameter logging for analysis
        log_interval=1,
        fix_mu=False
    )

    # Extract results
    loglik_bif = -bif_result.final_loss  # Convert minimization objective to log-likelihood
    conv_bif = bif_result.success  # Convergence status

    print(f"BIF Optimization completed in {bif_result.time_taken:.2f} seconds")
    print(f"Convergence status: {conv_bif}")
    print(f"Final log-likelihood: {loglik_bif:.4f}")
    print(f"Number of optimization steps: {bif_result.steps}")

    # Save the entire result object
    save_result(bif_result, output_dir)

    # Optional: Run other models (PF, DCC-GARCH, Factor-CV)
    # These sections are commented out as they were not implemented in the notebook

    print("Model estimation completed successfully.")

def save_result(result: OptimizerResult, output_dir: Path,filter_type: str = "BIF"):
    """
    Save the entire optimization result object using cloudpickle.

    Args:
        result: The optimization result object
        output_dir: Directory to save the results
    """
    try:
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save using cloudpickle (better for complex objects with JAX arrays)
        pickle_file = output_dir / f"{filter_type.lower()}_full_result_{timestamp}.pkl"
        with open(pickle_file, 'wb') as f:
            cloudpickle.dump(result, f)
        print(f"Full result object saved to {pickle_file}")

    except Exception as e:
        print(f"Error saving results with cloudpickle: {e}")

        # Try with regular pickle as fallback
        try:
            pickle_file = output_dir / f"{filter_type.lower()}_full_result_{timestamp}_fallback.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(result, f)
            print(f"Full result object saved to {pickle_file} using standard pickle")
        except Exception as e2:
            print(f"Error saving results with standard pickle: {e2}")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()
