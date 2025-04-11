#!/usr/bin/env python
"""
Comprehensive Optimizer Comparison Script for BIF.

This script runs a comprehensive experiment comparing all available optimizers
for the Bellman Information Filter (BIF) on a DFSV model. It tracks:
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
from typing import List, Dict, Any, Optional

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
from bellman_filter_dfsv.utils.optimization_helpers import create_stable_initial_params
# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)


# --- Model and Data Generation Functions ---

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
    Phi_f = Phi_f.at[diag_indices].set(Phi_f_diag)

    # Log-volatility persistence (diagonal-dominant with eigenvalues < 1)
    key, subkey1 = jax.random.split(key)
    Phi_h_off = jax.random.uniform(subkey1, (K, K), minval=-0.1, maxval=0.1)
    key, subkey1 = jax.random.split(key)
    Phi_h_diag = jax.random.uniform(subkey1, (K,), minval=0.8, maxval=0.95)
    Phi_h = Phi_h_off
    Phi_h = Phi_h.at[diag_indices].set(Phi_h_diag)

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


# --- Results Analysis Functions ---

def print_results_table(results: List[OptimizerResult], max_steps: int):
    """Print a formatted table of optimization results.

    Args:
        results: List of optimization results.
        max_steps: Maximum number of steps used in optimization, used to determine if an optimizer reached the step limit.
    """
    print("\n\n--- Optimization Results ---")
    # Header
    print(f"{'Optimizer':<20} | {'Transform':<10} | {'Fix_mu':<7} | {'Status':<20} | {'Final Loss':<15} | {'Steps':<8} | {'Time (s)':<10} | {'Error'}")
    print("-" * 120)

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
        # Truncate error message if it's too long
        error_str = "N/A"
        if res.error_message:
            error_str = res.error_message
            if len(error_str) > 30:
                error_str = error_str[:27] + "..."

        print(f"{res.optimizer_name:<20} | {'Yes' if res.uses_transformations else 'No':<10} | {fix_mu_str:<7} | {status_str:<20} | {loss_str:<15} | {steps_str:<8} | {time_str:<10} | {error_str}")

    print("-" * 120)


def print_parameter_comparison(results: List[OptimizerResult], true_params: DFSVParamsDataclass):
    """Print a comparison of estimated parameters to true values.

    Args:
        results: List of optimization results.
        true_params: True model parameters.
    """
    print("\n\n--- Parameter Comparison ---")

    # We'll compare the parameters directly as matrices

    for res in sorted(results, key=lambda x: (x.optimizer_name, x.uses_transformations, x.fix_mu)):
        if res.final_params is not None:
            print(f"\n-- Run: Optimizer='{res.optimizer_name}' | Fix_mu='{'Yes' if res.fix_mu else 'No'}' | Success='{'Yes' if res.success else 'No'}' --")
            print("-" * 80)
            print("Parameter Comparison:")
            print("-" * 80)

            # If optimization failed, make sure we're using untransformed parameters
            final_params = res.final_params

            # If the optimization failed and we're using transformations, try to untransform the parameters
            if not res.success and res.uses_transformations:
                try:
                    from bellman_filter_dfsv.utils.transformations import untransform_params, apply_identification_constraint
                    final_params = untransform_params(final_params)
                    final_params = apply_identification_constraint(final_params)
                except Exception:
                    # If untransformation fails, just use the parameters as is
                    pass

            # Function to format matrix parameters nicely
            def format_matrix(matrix, precision=4):
                if matrix.ndim == 1:  # Vector
                    return np.array2string(matrix, precision=precision, separator=', ', suppress_small=True)
                else:  # Matrix
                    rows = []
                    for i in range(matrix.shape[0]):
                        row = np.array2string(matrix[i], precision=precision, separator=', ', suppress_small=True)
                        rows.append(row)
                    return '\n'.join(rows)

            # Print comparison for each parameter
            for param_name in ["lambda_r", "Phi_f", "Phi_h", "mu", "sigma2", "Q_h"]:
                # Get the true and estimated values
                true_val = getattr(true_params, param_name)
                est_val = getattr(final_params, param_name)

                # Print parameter name with a more descriptive title
                param_descriptions = {
                    "lambda_r": "Factor Loadings (Lambda)",
                    "Phi_f": "Factor Transition Matrix (Phi_f)",
                    "Phi_h": "Log-Volatility Transition Matrix (Phi_h)",
                    "mu": "Log-Volatility Mean (mu)",
                    "sigma2": "Observation Noise Variance (sigma2)",
                    "Q_h": "Log-Volatility Noise Covariance (Q_h)"
                }

                print(f"\n{param_descriptions.get(param_name, param_name)}:")
                print("-" * 60)

                # Print true value
                print("True Value:")
                print(format_matrix(true_val))

                # Print estimated value
                print("\nEstimated Value:")
                print(format_matrix(est_val))


def save_results_to_csv(results: List[OptimizerResult]):
    """Save optimization results to a CSV file.

    Args:
        results: List of optimization results.
    """
    print("\nSaving results to CSV...")

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Create CSV file
    csv_path = os.path.join("outputs", "optimizer_comparison_results.csv")
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
    csv_path = os.path.join("outputs", "parameter_estimation_errors.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = ["Optimizer", "Transformations", "Fix_mu", "Success", "Final Loss"]
        for param_name in true_values.keys():
            header.append(f"{param_name}_RMSE")
            header.append(f"{param_name}_MAE")
            header.append(f"{param_name}_ME")  # Mean Error (bias)
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

                    # Calculate RMSE, MAE, and ME (bias)
                    rmse = jnp.sqrt(jnp.mean((true_val - est_val) ** 2))
                    mae = jnp.mean(jnp.abs(true_val - est_val))
                    me = jnp.mean(true_val - est_val)  # Mean Error (bias)

                    row.append(f"{rmse:.6f}")
                    row.append(f"{mae:.6f}")
                    row.append(f"{me:.6f}")

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
            plt.plot(res.loss_history, label=label)

    # Add labels and legend
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss History by Optimizer')
    plt.legend()
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(True)

    # Save figure
    plot_path = os.path.join("outputs", "loss_history.png")
    plt.savefig(plot_path)
    print(f"  Loss history plot saved to {plot_path}")


def main():
    """Main function to run the optimizer comparison."""
    print("Starting Comprehensive Optimizer Comparison...")

    # 1. Create model parameters
    N, K = 5, 2
    true_params = create_simple_model(N=N, K=K)
    print(f"Created model with N={true_params.N}, K={true_params.K}")
    print("True Parameters:")
    print(true_params)

    # 2. Generate simulation data
    T = 1500  # Full experiment with longer time series
    print(f"\nGenerating {T} time steps of simulation data...")
    returns = create_training_data(true_params, T=T, seed=123)
    print("Simulation data generated.")

    # 3. Get all available optimizers
    available_optimizers = get_available_optimizers()
    print(f"\nFound {len(available_optimizers)} available optimizers:")
    for name, desc in available_optimizers.items():
        print(f"  - {name}: {desc}")

    # 4. Define optimization configurations
    filter_type = FilterType.BIF  # Focus on BIF filter
    use_transformations = True  # Always use transformations for stability
    max_steps = 1000  # Number of optimization steps
    stability_penalty_weight = 1e4  # Weight for stability penalty
    verbose = True  # Enable verbose output for detailed information
    log_params = False  # Disable parameter logging to avoid potential issues

    #4,5 generate initial parameter guess
    initial_params = create_stable_initial_params(N, K)
    # 5. Run optimizations
    results = []
    test_optimizers=["BFGS", "ArmijoBFGS", "RMSProp", "AdamW", "DampedTrustRegionBFGS"]
    print(f"\nRunning optimizations with max_steps={max_steps}, stability_penalty_weight={stability_penalty_weight}...")
    # Run all available optimizers with both fixed and unfixed mu
    for optimizer_name in test_optimizers:
        for fix_mu in [True, False]:  # Run with both fixed and unfixed mu
            print(f"\n--- Running: Optimizer={optimizer_name} | Transform={'Yes' if use_transformations else 'No'} | Fix_mu={'Yes' if fix_mu else 'No'} ---")

            try:
                # Run optimization with error handling
                result = run_optimization(
                    filter_type=filter_type,
                    returns=returns,
                    initial_params=initial_params, # Added
                    true_params=true_params if fix_mu else None,  # Only pass true_params if fix_mu is True
                    use_transformations=use_transformations,
                    optimizer_name=optimizer_name,
                    priors=None,
                    stability_penalty_weight=stability_penalty_weight,
                    max_steps=max_steps,
                    verbose=verbose,
                    log_params=log_params,
                    log_interval=1,  # Log at every step
                    rtol=1e-6,
                    atol=1e-6,
                    fix_mu=fix_mu
                )

                results.append(result)

                # Print immediate result based on result code
                if hasattr(result, 'result_code') and result.result_code is not None:
                    # Use the result code to determine status
                    # Get the result code as an enum value

                    # Check the result code against known enum values
                    if result.result_code == optx.RESULTS.successful:
                        success_str = "converged"
                    elif result.result_code == optx.RESULTS.max_steps_reached:
                        success_str = "reached max steps (did not converge)"
                    elif result.result_code == optx.RESULTS.nonlinear_max_steps_reached:
                        success_str = "reached nonlinear max steps (did not converge)"
                    elif result.result_code == optx.RESULTS.nonlinear_divergence:
                        success_str = "diverged"
                    elif result.result_code == optx.RESULTS.singular:
                        success_str = "singular matrix encountered"
                    elif result.result_code == optx.RESULTS.breakdown:
                        success_str = "iterative breakdown"
                    elif result.result_code == optx.RESULTS.stagnation:
                        success_str = "stagnation in iterative solve"
                    else:
                        success_str = str(result.result_code)
                else:
                    # Fall back to the old method if result_code is not available
                    # For short runs, we know it's unlikely to have actually converged
                    if max_steps <= 10:
                        if result.success:
                            success_str = "completed (likely not converged)"
                        else:
                            success_str = "failed"
                    else:
                        if result.success:
                            success_str = "converged"
                        else:
                            # Check if we reached max steps or had an error
                            if result.steps >= max_steps:
                                success_str = "reached max steps (did not converge)"
                            else:
                                success_str = "failed"

                # Print a more detailed summary of the optimization result
                if result.success:
                    print(f"Optimization {success_str} with final loss: {result.final_loss:.4e}")
                else:
                    if jnp.isfinite(result.final_loss):
                        print(f"Optimization {success_str} with final loss: {result.final_loss:.4e} (not converged)")
                    else:
                        print(f"Optimization {success_str} with final loss: {result.final_loss} (not converged)")
                print(f"Steps: {result.steps}, Time: {result.time_taken:.2f}s")

            except Exception as e:
                # Create a minimal result object with error information
                error_message = str(e)
                # Truncate error message if it's too long
                if len(error_message) > 100:
                    error_message = error_message[:97] + "..."

                # Create a dummy result with error information
                dummy_result = OptimizerResult(
                    filter_type=filter_type,
                    optimizer_name=optimizer_name,
                    uses_transformations=use_transformations,
                    fix_mu=fix_mu,
                    prior_config_name="No Priors",
                    success=False,
                    result_code=None,
                    final_loss=float('nan'),  # Set to NaN initially
                    steps=0,
                    time_taken=0.0,
                    error_message=error_message,
                    final_params=None,
                    param_history=None,
                    loss_history=None
                )

                results.append(dummy_result)
                print(f"Error running optimization with {optimizer_name}: {error_message}")

    # 6. Print results table
    if results:
        print_results_table(results, max_steps)

        # 7. Print parameter comparison
        print_parameter_comparison(results, true_params)

        # 8. Save results
        save_results_to_csv(results)
        save_parameter_errors_to_csv(results, true_params)
        plot_loss_history(results)
    else:
        print("\nNo results to display. All optimizations failed.")

    print("\nComprehensive Optimizer Comparison completed.")


if __name__ == "__main__":
    main()
