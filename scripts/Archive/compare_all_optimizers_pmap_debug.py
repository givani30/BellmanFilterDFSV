#!/usr/bin/env python
"""
Parallelized Comprehensive Optimizer Comparison Script for BIF using jax.pmap.

This script runs a comprehensive experiment comparing all available optimizers
for the Bellman Information Filter (BIF) on a DFSV model, parallelizing
the optimizer runs using jax.pmap for potential performance gains. It tracks:
- Final loss value
- Number of steps taken
- Success status
- Computation time
- Final parameter estimates

Results are saved to CSV files and printed as tables for easy comparison.
"""

import os
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from functools import partial

# JAX imports
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

# Project specific imports
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.utils.transformations import apply_identification_constraint
from bellman_filter_dfsv.utils.optimization import (
    run_optimization,
    FilterType,
    OptimizerResult
)
from bellman_filter_dfsv.utils.solvers import get_available_optimizers

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)


# --- Model and Data Generation Functions (Copied from original) ---

def create_simple_model(N: int = 3, K: int = 2) -> DFSVParamsDataclass:
    """Create a simple DFSV model with reasonable parameters.

    Args:
        N: Number of observed series.
        K: Number of factors.

    Returns:
        DFSVParamsDataclass: Model parameters.
    """
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    key, subkey1 = jax.random.split(key)

    # Factor loadings (lower triangular with diagonal fixed to 1)
    lambda_r_init = jax.random.normal(subkey1, (N, K)) * 0.5 + 0.5
    lambda_r = jnp.tril(lambda_r_init)
    diag_indices = jnp.diag_indices(n=min(N, K), ndim=2)
    lambda_r = lambda_r.at[diag_indices].set(1.0)

    # Factor persistence (diagonal-dominant with eigenvalues < 1)
    # Using lower persistence values for Phi_f (0.3-0.5 range)
    key, subkey1 = jax.random.split(key)
    Phi_f_off = jax.random.uniform(subkey1, (K, K), minval=-0.05, maxval=0.05)
    key, subkey1 = jax.random.split(key)
    Phi_f_diag = jax.random.uniform(subkey1, (K,), minval=0.3, maxval=0.5)
    Phi_f = Phi_f_off
    # Use diag_indices based on K for Phi_f
    phi_diag_indices = jnp.diag_indices(n=K, ndim=2)
    Phi_f = Phi_f.at[phi_diag_indices].set(Phi_f_diag)

    # Log-volatility persistence (diagonal-dominant with eigenvalues < 1)
    key, subkey1 = jax.random.split(key)
    Phi_h_off = jax.random.uniform(subkey1, (K, K), minval=-0.1, maxval=0.1)
    key, subkey1 = jax.random.split(key)
    Phi_h_diag = jax.random.uniform(subkey1, (K,), minval=0.8, maxval=0.95)
    Phi_h = Phi_h_off
    Phi_h = Phi_h.at[phi_diag_indices].set(Phi_h_diag) # Use phi_diag_indices here too

    # Long-run mean for log-volatilities
    mu = jnp.array([-1.0, -0.5] if K == 2 else [-1.0] * K)

    # Idiosyncratic variance (diagonal)
    key, subkey1 = jax.random.split(key)
    sigma2 = jax.random.uniform(subkey1, (N,), minval=0.05, maxval=0.1)

    # Log-volatility noise covariance (diagonal)
    key, subkey1 = jax.random.split(key)
    Q_h_diag = jax.random.uniform(subkey1, (K,), minval=0.1, maxval=0.3)
    Q_h = jnp.diag(Q_h_diag)

    # Create parameter object
    params = DFSVParamsDataclass(
        N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, Phi_h=Phi_h,
        mu=mu, sigma2=sigma2, Q_h=Q_h
    )

    # Ensure constraint is applied correctly
    params = apply_identification_constraint(params)
    return params


def create_training_data(params: DFSVParamsDataclass, T: int = 1000, seed: int = 123) -> jnp.ndarray:
    """Generate simulation data for training.

    Args:
        params: Model parameters.
        T: Number of time steps.
        seed: Random seed.

    Returns:
        jnp.ndarray: Simulated returns.
    """
    # Simulate data
    returns, factors, log_vols = simulate_DFSV(
        params=params,
        T=T,
        seed=seed
    )

    return returns


def create_uninformed_parameters(true_params: DFSVParamsDataclass, returns: jnp.ndarray) -> DFSVParamsDataclass:
    """Create uninformed initial parameters for optimization.

    Args:
        true_params: True model parameters (used only for dimensions).
        returns: Observed returns data.

    Returns:
        DFSVParamsDataclass: Uninformed parameters.
    """
    N, K = true_params.N, true_params.K

    # Calculate data variance for reasonable sigma2 initialization
    data_variance = jnp.var(returns, axis=0)

    # Create lambda_r with lower triangular constraint
    lambda_r_init = jnp.ones((N, K)) * 0.5
    lambda_r_init = jnp.tril(lambda_r_init)
    diag_indices = jnp.diag_indices(n=min(N, K), ndim=2)
    lambda_r_init = lambda_r_init.at[diag_indices].set(1.0)

    # Create initial parameters
    initial_params = DFSVParamsDataclass(
        N=N, K=K,
        lambda_r=lambda_r_init,
        Phi_f=jnp.eye(K) * 0.4,  # Lower persistence for factors
        Phi_h=jnp.eye(K) * 0.8,  # Moderate persistence
        mu=jnp.zeros(K),  # Zero mean for log volatility
        sigma2=0.1 * jnp.ones(N),  # Moderate idiosyncratic variance
        Q_h=0.2 * jnp.eye(K)  # Moderate volatility of volatility
    )

    return initial_params


# --- Results Analysis Functions (Copied from original) ---

def print_results_table(results: List[OptimizerResult], max_steps: int):
    """Print a formatted table of optimization results.

    Args:
        results: List of optimization results.
        max_steps: Maximum number of steps used in optimization, used to determine if an optimizer reached the step limit.
    """
    print("\n\n--- Optimization Results ---")
    # Header
    print(f"{'Optimizer':<20} | {'Transform':<10} | {'Fix_mu':<7} | {'Status':<20} | {'Final Loss':<15} | {'Steps':<8} | {'Time (s)':<10} | {'Error Message'}")
    print("-" * 160)

    # Rows
    for res in sorted(results, key=lambda x: (x.optimizer_name, x.uses_transformations, x.fix_mu)):
        # Determine status message based on result code
        if hasattr(res, 'result_code') and res.result_code is not None:
            # Use the result code to determine status
            # Get the result code as an enum value

            # Check the result code against known enum values
            if res.result_code == optx.RESULTS.successful:
                status_str = "Converged"
            elif res.result_code == optx.RESULTS.max_steps_reached:
                status_str = "Max steps reached"
            elif res.result_code == optx.RESULTS.nonlinear_max_steps_reached:
                status_str = "Nonlinear max steps"
            elif res.result_code == optx.RESULTS.nonlinear_divergence:
                status_str = "Diverged"
            elif res.result_code == optx.RESULTS.singular:
                status_str = "Singular matrix"
            elif res.result_code == optx.RESULTS.breakdown:
                status_str = "Iterative breakdown"
            elif res.result_code == optx.RESULTS.stagnation:
                status_str = "Stagnation"
            else:
                status_str = str(res.result_code)
        else:
            # Fall back to the old method if result_code is not available
            if res.success:
                status_str = "Converged"
            else:
                # Check if we reached max steps or had an error
                if res.steps >= max_steps and not res.error_message:
                    status_str = "Max steps reached"
                else:
                    status_str = "Failed"

        fix_mu_str = "Yes" if res.fix_mu else "No"
        loss_str = f"{res.final_loss:.4e}" if jnp.isfinite(res.final_loss) else "Inf/NaN"
        steps_str = str(res.steps) if res.steps >= 0 else "N/A"
        time_str = f"{res.time_taken:.2f}"
        error_str = res.error_message if res.error_message else "N/A"
        print(f"{res.optimizer_name:<20} | {'Yes' if res.uses_transformations else 'No':<10} | {fix_mu_str:<7} | {status_str:<20} | {loss_str:<15} | {steps_str:<8} | {time_str:<10} | {error_str}")

    print("-" * 160)


def print_parameter_comparison(results: List[OptimizerResult], true_params: DFSVParamsDataclass):
    """Print a comparison of estimated parameters to true values.

    Args:
        results: List of optimization results.
        true_params: True model parameters.
    """
    print("\n\n--- Parameter Comparison ---")

    # Get true parameter values as flat arrays for easier comparison
    true_values = {
        "lambda_r": true_params.lambda_r.flatten(),
        "Phi_f": true_params.Phi_f.flatten(),
        "Phi_h": true_params.Phi_h.flatten(),
        "mu": true_params.mu.flatten(),
        "sigma2": true_params.sigma2.flatten(),
        "Q_h": true_params.Q_h.flatten()
    }

    for res in sorted(results, key=lambda x: (x.optimizer_name, x.uses_transformations, x.fix_mu)):
        if res.final_params is not None:
            print(f"\n-- Run: Optimizer='{res.optimizer_name}' | Fix_mu='{'Yes' if res.fix_mu else 'No'}' | Success='{'Yes' if res.success else 'No'}' --")
            print("-" * 80)
            print(f"{'Parameter':<10} | {'True Value':<35} | {'Estimated Value'}")
            print("-" * 80)

            # Get estimated parameter values as flat arrays
            est_values = {
                "lambda_r": res.final_params.lambda_r.flatten(),
                "Phi_f": res.final_params.Phi_f.flatten(),
                "Phi_h": res.final_params.Phi_h.flatten(),
                "mu": res.final_params.mu.flatten(),
                "sigma2": res.final_params.sigma2.flatten(),
                "Q_h": res.final_params.Q_h.flatten()
            }

            # Print comparison for each parameter
            for param_name, true_val in true_values.items():
                est_val = est_values[param_name]

                # Format arrays for display
                true_str = np.array2string(true_val, precision=4, separator=', ')
                est_str = np.array2string(est_val, precision=4, separator=', ')

                print(f"{param_name:<10} | {true_str:<35} | {est_str}")


def save_results_to_csv(results: List[OptimizerResult]):
    """Save optimization results to a CSV file.

    Args:
        results: List of optimization results.
    """
    print("\nSaving results to CSV...")

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Create CSV file
    csv_path = os.path.join("outputs", "optimizer_comparison_parallel_results.csv") # Changed filename
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow([
            "Optimizer", "Transformations", "Fix_mu", "Success",
            "Final Loss", "Steps", "Time (s)", "Error Message"
        ])

        # Write rows
        for res in results:
            writer.writerow([
                res.optimizer_name,
                "Yes" if res.uses_transformations else "No",
                "Yes" if res.fix_mu else "No",
                "Yes" if res.success else "No",
                res.final_loss if jnp.isfinite(res.final_loss) else "Inf/NaN",
                res.steps if res.steps >= 0 else "N/A",
                f"{res.time_taken:.2f}",
                res.error_message if not res.success else "N/A"
            ])

    print(f"  Results saved to {csv_path}")


def save_parameter_errors_to_csv(results: List[OptimizerResult], true_params: DFSVParamsDataclass):
    """Save parameter estimation errors to a CSV file.

    Args:
        results: List of optimization results.
        true_params: True model parameters.
    """
    print("\nSaving parameter errors to CSV...")

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Get true parameter values as flat arrays for easier comparison
    true_values = {
        "lambda_r": true_params.lambda_r.flatten(),
        "Phi_f": true_params.Phi_f.flatten(),
        "Phi_h": true_params.Phi_h.flatten(),
        "mu": true_params.mu.flatten(),
        "sigma2": true_params.sigma2.flatten(),
        "Q_h": true_params.Q_h.flatten()
    }

    # Create CSV file
    csv_path = os.path.join("outputs", "parameter_estimation_errors_parallel.csv") # Changed filename
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = ["Optimizer", "Transformations", "Fix_mu", "Success", "Final Loss"]
        for param_name in true_values.keys():
            header.append(f"{param_name}_RMSE")
            header.append(f"{param_name}_MAE")
        writer.writerow(header)

        # Write rows
        for res in results:
            if res.final_params is not None:
                row = [
                    res.optimizer_name,
                    "Yes" if res.uses_transformations else "No",
                    "Yes" if res.fix_mu else "No",
                    "Yes" if res.success else "No",
                    res.final_loss if jnp.isfinite(res.final_loss) else "Inf/NaN"
                ]

                # Calculate errors for each parameter
                for param_name, true_val in true_values.items():
                    est_val = getattr(res.final_params, param_name).flatten()

                    # Calculate RMSE and MAE
                    rmse = jnp.sqrt(jnp.mean((true_val - est_val) ** 2))
                    mae = jnp.mean(jnp.abs(true_val - est_val))

                    row.append(f"{rmse:.6f}")
                    row.append(f"{mae:.6f}")

                writer.writerow(row)

    print(f"  Parameter errors saved to {csv_path}")


def plot_loss_history(results: List[OptimizerResult]):
    """Plot loss history for each optimizer.

    Args:
        results: List of optimization results.
    """
    print("\nGenerating loss history plots...")

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot loss history for each optimizer
    for res in results:
        if res.loss_history is not None and len(res.loss_history) > 0:
            label = f"{res.optimizer_name} ({'T' if res.uses_transformations else 'U'})"
            # Convert potential JAX arrays in loss_history to numpy for plotting
            loss_hist_np = np.array(res.loss_history)
            plt.plot(loss_hist_np, label=label)

    # Add labels and legend
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss History by Optimizer (Parallel Run)')
    plt.legend()
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(True)

    # Save figure
    plot_path = os.path.join("outputs", "loss_history_parallel.png") # Changed filename
    plt.savefig(plot_path)
    print(f"  Loss history plot saved to {plot_path}")


# --- Parallel Execution Function ---

def run_single_optimizer_config(
    optimizer_name: str,
    filter_type: FilterType,
    returns: jnp.ndarray,
    true_params: Optional[DFSVParamsDataclass],
    use_transformations: bool,
    priors: Optional[Dict[str, Any]],
    stability_penalty_weight: float,
    max_steps: int,
    verbose: bool,
    log_params: bool,
    log_interval: int,
    fix_mu: bool # Added fix_mu here for result tracking
) -> OptimizerResult:
    """
    Runs optimization for a single optimizer configuration.
    Designed to be mapped by jax.pmap.

    Args:
        optimizer_name: Name of the optimizer to use.
        filter_type: Type of filter (e.g., BIF, PF).
        returns: Observed returns data.
        true_params: True model parameters (used if fix_mu is True).
        use_transformations: Whether to use parameter transformations.
        priors: Optional priors for parameters.
        stability_penalty_weight: Weight for the stability penalty.
        max_steps: Maximum number of optimization steps.
        verbose: Whether to print verbose output during optimization.
        log_params: Whether to log parameter history.
        log_interval: Interval for logging.
        fix_mu: Whether mu was fixed (for result tracking).

    Returns:
        OptimizerResult: The result of the optimization run.
                         Returns a dummy result on error.
    """
    print(f"--- Starting: Optimizer={optimizer_name} (on device {jax.process_index()}) ---")
    start_time = time.time()
    try:
        result = run_optimization(
            filter_type=filter_type,
            returns=returns,
            true_params=true_params if fix_mu else None,
            use_transformations=use_transformations,
            optimizer_name=optimizer_name,
            priors=priors,
            stability_penalty_weight=stability_penalty_weight,
            max_steps=max_steps,
            verbose=verbose,
            log_params=log_params,
            log_interval=log_interval
        )
        # Ensure fix_mu is correctly set in the result
        # We need to recreate the result object if run_optimization doesn't store fix_mu
        # Assuming OptimizerResult is a simple dataclass or similar
        result_dict = result.__dict__ # Or however you access its fields
        result_dict['fix_mu'] = fix_mu # Add/overwrite fix_mu
        final_result = OptimizerResult(**result_dict)

        # Print immediate result status
        if hasattr(final_result, 'result_code') and final_result.result_code is not None:
            if final_result.result_code == optx.RESULTS.successful: success_str = "converged"
            elif final_result.result_code == optx.RESULTS.max_steps_reached: success_str = "reached max steps"
            elif final_result.result_code == optx.RESULTS.nonlinear_max_steps_reached: success_str = "reached nonlinear max steps"
            elif final_result.result_code == optx.RESULTS.nonlinear_divergence: success_str = "diverged"
            elif final_result.result_code == optx.RESULTS.singular: success_str = "singular matrix"
            elif final_result.result_code == optx.RESULTS.breakdown: success_str = "iterative breakdown"
            elif final_result.result_code == optx.RESULTS.stagnation: success_str = "stagnation"
            else: success_str = str(final_result.result_code)
        else:
            success_str = "completed (success=True)" if final_result.success else "failed (success=False)"

        print(f"--- Finished: Optimizer={optimizer_name} ({success_str}) | Loss={final_result.final_loss:.4e} | Steps={final_result.steps} | Time={final_result.time_taken:.2f}s ---")
        return final_result

    except Exception as e:
        end_time = time.time()
        print(f"--- Error: Optimizer={optimizer_name} failed after {end_time - start_time:.2f}s: {e} ---")
        # Return a dummy result indicating failure
        # Create a dummy DFSVParamsDataclass if needed for the structure
        dummy_params = DFSVParamsDataclass(
            N=true_params.N if true_params else 0,
            K=true_params.K if true_params else 0,
            lambda_r=jnp.empty((0,0)), Phi_f=jnp.empty((0,0)), Phi_h=jnp.empty((0,0)),
            mu=jnp.empty(0), sigma2=jnp.empty(0), Q_h=jnp.empty((0,0))
        )
        return OptimizerResult(
            optimizer_name=optimizer_name,
            uses_transformations=use_transformations,
            fix_mu=fix_mu,
            success=False,
            final_loss=jnp.inf,
            final_params=dummy_params, # Use dummy params
            steps=-1,
            time_taken=end_time - start_time,
            error_message=str(e),
            loss_history=None, # Or jnp.array([])
            param_history=None, # Or []
            result_code=None # Indicate error state if possible
        )


# --- Main Execution ---

def main():
    """Main function to run the parallel optimizer comparison."""
    print("Starting Parallel Comprehensive Optimizer Comparison...")
    print(f"JAX running on {jax.device_count()} devices.")
    if jax.device_count() == 1:
        print("Warning: Only one JAX device detected. Parallel execution will offer limited speedup.")

    # 1. Create model parameters
    N, K = 5, 2
    true_params = create_simple_model(N=N, K=K)
    print(f"Created model with N={true_params.N}, K={true_params.K}")
    # print("True Parameters:") # Keep output concise for parallel runs
    # print(true_params)

    # 2. Generate simulation data
    T = 1500  # Full experiment with longer time series
    print(f"\nGenerating {T} time steps of simulation data...")
    returns = create_training_data(true_params, T=T, seed=123)
    print("Simulation data generated.")

    # 3. Get all available optimizers
    available_optimizers_dict = get_available_optimizers()
    optimizer_names = list(available_optimizers_dict.keys())
    print(f"\nFound {len(optimizer_names)} available optimizers to test in parallel:")
    # for name, desc in available_optimizers_dict.items():
    #     print(f"  - {name}: {desc}") # Keep output concise

    # 4. Define optimization configurations (mostly static)
    filter_type = FilterType.BIF
    use_transformations = True
    fix_mu = True # Set the desired fix_mu strategy here
    max_steps = 500
    stability_penalty_weight = 1000.0
    verbose = False # Keep verbose off for parallel runs to avoid cluttered output
    log_params = False # Keep param logging off for performance
    log_interval = 1 # Doesn't matter much if log_params is False
    priors = None

    # 5. Prepare for pmap
    # Ensure the number of optimizers is suitable for devices
    num_devices = jax.device_count()
    num_optimizers = len(optimizer_names)
    if num_optimizers < num_devices:
        print(f"Warning: Number of optimizers ({num_optimizers}) is less than number of devices ({num_devices}). Padding optimizers for pmap.")
        # Pad the list to match device count for pmap efficiency, will run duplicates
        optimizer_names_padded = (optimizer_names * (num_devices // num_optimizers + 1))[:num_devices]
    elif num_optimizers % num_devices != 0:
         print(f"Warning: Number of optimizers ({num_optimizers}) is not a multiple of devices ({num_devices}). Padding optimizers for pmap.")
         padding_needed = num_devices - (num_optimizers % num_devices)
         optimizer_names_padded = optimizer_names + optimizer_names[:padding_needed] # Pad with first few
    else:
        optimizer_names_padded = optimizer_names

    # Convert list of names to a JAX array for pmap
    # Note: pmap typically expects JAX arrays, but might handle lists of strings.
    # If issues arise, consider mapping names to integers and back.
    # For now, let's try passing strings directly.
    optimizer_names_array = np.array(optimizer_names_padded) # Use numpy array

    print(f"\nRunning optimizations in parallel across {num_devices} devices...")
    print(f"Optimizers to run (padded): {optimizer_names_padded}")

    # Define which arguments are static (hashable or Pytrees)
    # 'returns' is removed as it's non-hashable and will be broadcast via in_axes=None
    static_argnames_for_pmap = (
        'filter_type', 'true_params', 'use_transformations',
        'priors', 'stability_penalty_weight', 'max_steps', 'verbose',
        'log_params', 'log_interval', 'fix_mu'
    )

    # Create the pmapped function
    pmapped_run = jax.pmap(
        run_single_optimizer_config,
        # Map over optimizer_name (axis 0), broadcast returns (axis None), broadcast true_params (axis None)
        in_axes=(0, None, None),
        # Specify remaining static arguments by their positional index
        # Indices correspond to: filter_type, use_transformations, priors,
    # Create a partial function with static arguments pre-filled
    run_partial = partial(
        run_single_optimizer_config,
        # Fill in static args by keyword
        filter_type=filter_type,
        use_transformations=use_transformations,
        priors=priors,
        stability_penalty_weight=stability_penalty_weight,
        max_steps=max_steps,
        verbose=verbose,
        log_params=log_params,
        log_interval=log_interval,
        fix_mu=fix_mu
    )

    # pmap the partial function. It now effectively only takes:
    # optimizer_name, returns, true_params
    pmapped_run = jax.pmap(
        run_partial,
        # Map over optimizer_name (axis 0), broadcast returns (axis None), broadcast true_params (axis None)
        in_axes=(0, None, None)
        # No need for static_broadcasted_argnums here, as static args are baked in
    )

    # 6. Run optimizations in parallel
    overall_start_time = time.time()

    # Execute the pmapped function, passing only the non-static args for the partial function
    all_results_padded = pmapped_run(
        optimizer_names_array,      # Corresponds to optimizer_name
        returns,                    # Corresponds to returns
        true_params                 # Corresponds to true_params
    )

    overall_end_time = time.time()
    print(f"\nParallel execution finished in {overall_end_time - overall_start_time:.2f} seconds.")

    # 7. Process results
    # The result is potentially replicated across devices. Take the first instance.
    # Also, remove padding if applied.
    # Note: pmap might return results structured per device. We need to flatten.
    # Let's assume it returns a flat list/array matching optimizer_names_padded
    if isinstance(all_results_padded, list) and len(all_results_padded) > 0 and isinstance(all_results_padded[0], list):
         # If results are nested per device, flatten
         results_flat = [item for sublist in all_results_padded for item in sublist]
    else:
         # Assume already flat or handle other structures as needed
         results_flat = list(all_results_padded) # Convert JAX array/tuple if necessary

    # Remove padding results
    results = results_flat[:num_optimizers]

    # Filter out potential dummy results if needed (though error handling inside pmap is better)
    valid_results = [res for res in results if res.steps != -1] # Example filter

    if valid_results:
        print(f"\nProcessing {len(valid_results)} valid results...")
        # 8. Print results table
        print_results_table(valid_results, max_steps)

        # 9. Print parameter comparison
        print_parameter_comparison(valid_results, true_params)

        # 10. Save results
        save_results_to_csv(valid_results)
        save_parameter_errors_to_csv(valid_results, true_params)
        plot_loss_history(valid_results)
    else:
        print("\nNo valid results to display. All parallel optimizations may have failed.")

    print("\nParallel Comprehensive Optimizer Comparison completed.")


if __name__ == "__main__":
    main()