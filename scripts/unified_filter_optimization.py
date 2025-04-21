#!/usr/bin/env python
"""
Unified Filter Optimization Script for DFSV Models.

This script provides a unified interface for optimizing DFSV model parameters
using any of the three implemented filters:
1. Bellman Information Filter (BIF)
2. Bellman Filter (BF)
3. Particle Filter (PF)

It supports parameter transformations, various optimizers, priors, and stability
penalties, allowing for comprehensive comparison of filter performance in
parameter estimation.
"""

import csv
import os
import cloudpickle
from typing import List

# JAX imports
import jax
import jax.numpy as jnp
import numpy as np

# Project specific imports
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.models.simulation_helpers import create_stable_dfsv_params
from bellman_filter_dfsv.utils.transformations import apply_identification_constraint
from bellman_filter_dfsv.utils.optimization import (
    run_optimization,
    FilterType,
    OptimizerResult
)
from bellman_filter_dfsv.utils.optimization_helpers import create_stable_initial_params



# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)


# No need to redefine enums and data structures - using the ones from the main code


# --- Model and Data Generation Functions ---

# Using the imported create_stable_dfsv_params function from simulation_helpers


def create_training_data(params: DFSVParamsDataclass, T: int = 1000, seed: int = 42) -> jnp.ndarray:
    """Generate simulated data for training.

    Args:
        params: Model parameters.
        T: Number of time steps.
        seed: Random seed.

    Returns:
        jnp.ndarray: Simulated returns with shape (T, N).
    """
    returns, _, _ = simulate_DFSV(params, T=T, seed=seed)
    return jnp.asarray(returns)  # Convert to JAX array


# Using create_filter from the main code


# Using the imported create_stable_initial_params function from optimization_helpers


# Using run_optimization from the main code


def determine_optimizer(filter_type_str):
    """Determine the appropriate optimizer based on filter type.

    Args:
        filter_type_str: String representation of filter type.

    Returns:
        String name of the optimizer to use.
    """
    # For BIF and BF, use BFGS
    # For PF, use AdamW
    if filter_type_str in ["BIF", "BF"]:
        return "DampedTrustRegionBFGS"
    elif filter_type_str == "PF":
        return "ArmijoBFGS"
    else:
        raise ValueError(f"Unknown filter type: {filter_type_str}")



# --- Results Analysis Functions ---

def print_results_table(results: List[OptimizerResult]):
    """Print a formatted table of optimization results.

    Args:
        results: List of optimization results.
    """
    print("\n\n--- Optimization Results ---")
    # Header
    print(f"{'Filter':<6} | {'Optimizer':<10} | {'Transform':<10} | {'Fix_mu':<7} | {'Prior Config':<20} | {'Success':<8} | {'Final Loss':<15} | {'Steps':<8} | {'Time (s)':<10} | {'Error Message'}")
    print("-" * 140)

    # Rows
    for res in sorted(results, key=lambda x: (x.filter_type.name, x.optimizer_name, x.uses_transformations, x.fix_mu)):
        filter_str = res.filter_type.name
        success_str = "Yes" if res.success else "No"
        fix_mu_str = "Yes" if res.fix_mu else "No"
        loss_str = f"{res.final_loss:.4e}" if jnp.isfinite(res.final_loss) else "Inf/NaN"
        steps_str = str(res.steps) if res.steps >= 0 else "N/A"
        time_str = f"{res.time_taken:.2f}"
        error_str = res.error_message if not res.success else "N/A"
        print(f"{filter_str:<6} | {res.optimizer_name:<10} | {'Yes' if res.uses_transformations else 'No':<10} | {fix_mu_str:<7} | {res.prior_config_name:<20} | {success_str:<8} | {loss_str:<15} | {steps_str:<8} | {time_str:<10} | {error_str}")

    print("-" * 140)


def print_parameter_comparison(results: List[OptimizerResult], true_params: DFSVParamsDataclass):
    """Print a comparison between true and estimated parameters.

    Args:
        results: List of optimization results.
        true_params: True model parameters.
    """
    import dataclasses

    print("\n\n--- Parameter Estimation Comparison ---")

    # Get parameter names from the dataclass, excluding N and K
    param_names = [f.name for f in dataclasses.fields(DFSVParamsDataclass) if f.name not in ['N', 'K']]

    # Set numpy print options for better readability
    np.set_printoptions(precision=4, suppress=True)

    for res in sorted(results, key=lambda x: (x.filter_type.name, x.optimizer_name, x.uses_transformations, x.fix_mu)):
        if res.final_params is not None:
            print(f"\n-- Run: Filter='{res.filter_type.name}' | Optimizer='{res.optimizer_name}' | Fix_mu='{'Yes' if res.fix_mu else 'No'}' | Success='{'Yes' if res.success else 'No'}' --")
            print("-" * 80)
            print(f"{'Parameter':<10} | {'True Value':<35} | {'Estimated Value'}")
            print("-" * 80)

            for name in param_names:
                true_val = getattr(true_params, name)
                est_val = getattr(res.final_params, name)

                # Convert to numpy for consistent printing
                true_val_np = np.asarray(true_val)
                est_val_np = np.asarray(est_val)

                # Format for printing (handle multi-line arrays)
                true_str_lines = str(true_val_np).split('\n')
                est_str_lines = str(est_val_np).split('\n')

                # Print first line with parameter name
                print(f"{name:<10} | {true_str_lines[0]:<35} | {est_str_lines[0]}")

                # Print subsequent lines aligned
                max_lines = max(len(true_str_lines), len(est_str_lines))
                for i in range(1, max_lines):
                    true_line = true_str_lines[i] if i < len(true_str_lines) else ""
                    est_line = est_str_lines[i] if i < len(est_str_lines) else ""
                    print(f"{'':<10} | {true_line:<35} | {est_line}")

            print("-" * 80)
        else:
            print(f"\n-- Run: Filter='{res.filter_type.name}' | Optimizer='{res.optimizer_name}' --")
            print("  No final parameters available for comparison (likely failed early).")


def save_results_to_csv(results: List[OptimizerResult], filename: str = "filter_optimization_results.csv"):
    """Save optimization results to a CSV file.

    Args:
        results: List of optimization results.
        filename: Name of the CSV file to save.
    """
    if not results:
        print("No results to save.")
        return

    # Get headers from the namedtuple fields, excluding final_params
    headers = [field for field in OptimizerResult._fields if field != 'final_params']

    try:
        # Ensure outputs directory exists
        os.makedirs("outputs", exist_ok=True)
        filepath = os.path.join("outputs", filename)

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(headers)
            # Write data rows
            for result in results:
                # Prepare row data, excluding final_params
                row_data = {field: getattr(result, field) for field in headers}
                # Convert filter_type enum to string
                if 'filter_type' in row_data:
                    row_data['filter_type'] = row_data['filter_type'].name
                # Convert JAX arrays/scalars if necessary
                row = [float(item) if isinstance(item, (jnp.ndarray, jnp.generic)) else item for item in row_data.values()]
                writer.writerow(row)
        print(f"Results successfully saved to {filepath}")
    except IOError as e:
        print(f"Error saving results to CSV: {e}")


def save_estimated_params(results: List[OptimizerResult], true_params: DFSVParamsDataclass):
    """Save estimated parameters to pickle files.

    Args:
        results: List of optimization results.
        true_params: True model parameters.
    """
    print("\nSaving estimated parameters...")

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Save true parameters for reference
    true_params_path = os.path.join("outputs", "true_params.pkl")
    with open(true_params_path, 'wb') as f:
        cloudpickle.dump(true_params, f)
    print(f"  Saved true parameters to {true_params_path}")

    # Save estimated parameters for each successful run
    for res in results:
        if res.final_params is not None:
            status_str = 'Success' if res.success else 'Failure'
            filter_str = res.filter_type.name
            opt_name = res.optimizer_name.replace(' ', '_')
            transform_str = 'Transformed' if res.uses_transformations else 'Untransformed'
            fix_mu_str = 'FixedMu' if res.fix_mu else 'FreeMu'

            filename = f"estimated_params_{filter_str}_{opt_name}_{transform_str}_{fix_mu_str}_{status_str}.pkl"
            filepath = os.path.join("outputs", filename)

            try:
                with open(filepath, 'wb') as f:
                    cloudpickle.dump(res.final_params, f)
                print(f"  Saved parameters to {filepath}")
            except Exception as e:
                print(f"  ERROR saving parameters to {filepath}: {e}")
        else:
            print(f"  Skipping parameter saving for {res.filter_type.name}/{res.optimizer_name} (final_params is None)")


# --- Main Function ---

def main():
    """Main function to run the unified filter optimization."""
    print("Starting Unified Filter Optimization...")

    # Configuration parameters - easily modifiable
    N = 5           # Number of observed variables
    K = 3           # Number of factors
    T = 500         # Number of time steps
    seed = 123      # Random seed for reproducibility
    max_steps = 200 # Maximum optimization steps
    stability_penalty_weight = 1000.0  # Weight for stability penalty
    num_particles = 5000  # Number of particles for PF

    print(f"Configuration: N={N}, K={K}, T={T}, seed={seed}, max_steps={max_steps}")

    # 1. Create model parameters
    true_params = create_stable_dfsv_params(N=N, K=K)
    print(f"Created model with N={true_params.N}, K={true_params.K}")
    print("True Parameters:")
    print(true_params)

    # 2. Generate simulation data
    print(f"\nGenerating {T} time steps of simulation data...")
    returns = create_training_data(true_params, T=T, seed=seed)
    print("Simulation data generated.")

    # 3. Define optimization configurations
    # Run all three filters
    filter_types = [FilterType.BIF, FilterType.PF]

    # Always use transformations for stability
    use_transformations = True

    # Never fix mu as requested
    fix_mu = False
    # Create initial parameters for optimization
    initial_params = create_stable_initial_params(N=N, K=K)
    # 4. Run optimizations
    results = []

    print(f"\nRunning optimizations with max_steps={max_steps}, stability_penalty_weight={stability_penalty_weight}...")

    for filter_type in filter_types:
        # Determine the appropriate optimizer for this filter type
        optimizer_name = determine_optimizer(filter_type.name)

        print(f"\n--- Running: Filter={filter_type.name} | Optimizer={optimizer_name} | Transform={'Yes' if use_transformations else 'No'} | Fix_mu={'Yes' if fix_mu else 'No'} ---")

        try:
            # Run optimization with error handling
            result = run_optimization(
                filter_type=filter_type,
                returns=returns,
                initial_params=initial_params,
                true_params=true_params, # Pass true_params for comparison
                fix_mu=False,  # Explicitly set fix_mu to False
                use_transformations=use_transformations,
                optimizer_name=optimizer_name,
                priors=None,
                stability_penalty_weight=stability_penalty_weight,
                max_steps=max_steps,
                num_particles=num_particles,
                verbose=True,
                log_params=True,  # Enable parameter logging
                prior_config_name="No Priors"
            )

            # Add result to list
            results.append(result)
        except Exception as e:
            print(f"Error running optimization: {e}")
            # Create a failure result
            error_result = OptimizerResult(
                filter_type=filter_type,
                optimizer_name=optimizer_name,
                uses_transformations=use_transformations,
                fix_mu=fix_mu,  # Add fix_mu parameter
                prior_config_name="No Priors",
                success=False,
                final_loss=float('inf'),
                steps=-1,
                time_taken=0.0,
                error_message=f"Exception: {str(e)}",
                final_params=None
            )
            results.append(error_result)

    # 5. Print results table
    if results:
        print_results_table(results)

        # 6. Print parameter comparison
        print_parameter_comparison(results, true_params)

        # 7. Save results
        save_results_to_csv(results)
        save_estimated_params(results, true_params)
    else:
        print("\nNo results to display. All optimizations failed.")

    print("\nUnified Filter Optimization completed.")


if __name__ == "__main__":
    main()

