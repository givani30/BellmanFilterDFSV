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
from bellman_filter_dfsv.utils.transformations import apply_identification_constraint
from bellman_filter_dfsv.utils.optimization import (
    run_optimization,
    FilterType,
    OptimizerResult
)



# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)


# No need to redefine enums and data structures - using the ones from the main code


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
    # Off-diagonal elements
    key, subkey1 = jax.random.split(key)
    Phi_f = jax.random.uniform(subkey1, (K, K), minval=0.01, maxval=0.1)
    # Diagonal elements
    key, subkey1 = jax.random.split(key)
    diag_values = jax.random.uniform(subkey1, (K,), minval=0.15, maxval=0.35)
    Phi_f = Phi_f.at[jnp.diag_indices(K)].set(diag_values)
    #Make sure Phi_f is stable by normalizing
    Phi_f = Phi_f / jnp.linalg.norm(Phi_f, ord=2)*0.999

    # Log-volatility persistence (diagonal-dominant with eigenvalues < 1)
    # Off-diagonal elements
    key, subkey1 = jax.random.split(key)
    Phi_h = jax.random.uniform(subkey1, (K, K), minval=0.01, maxval=0.1)
    # Diagonal elements
    key, subkey1 = jax.random.split(key)
    diag_values = jax.random.uniform(subkey1, (K,), minval=0.9, maxval=0.99)
    Phi_h = Phi_h.at[jnp.diag_indices(K)].set(diag_values)
    #Make sure Phi_h is stable by normalizing
    Phi_h = Phi_h / jnp.linalg.norm(Phi_h, ord=2)*0.999
    
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


def create_initial_params(N: int, K: int) -> DFSVParamsDataclass:
    """Create initial parameter values for optimization.

    Args:
        N: Number of observed series.
        K: Number of factors.
        data_variance: Variance of the data (for scaling sigma2).

    Returns:
        DFSVParamsDataclass: Initial parameter values.
    """
    # Create lower triangular lambda_r with diagonal fixed to 1
    lambda_r_init = jnp.zeros((N, K))
    diag_indices = jnp.diag_indices(min(N, K))
    lambda_r_init = lambda_r_init.at[diag_indices].set(1.0)
    lambda_r_init = jnp.tril(lambda_r_init)

    # Different initial values for factor and volatility persistence
    # Lower persistence for factors, higher for volatilities (following working script)
    # Initialize with small non-zero off-diagonal elements to encourage their estimation
    phi_f_init = 0.3 * jnp.eye(K) + 0.05 * jnp.ones((K, K))  # Add small off-diagonal values
    phi_h_init = 0.8 * jnp.eye(K) + 0.02 * jnp.ones((K, K))  # Add small off-diagonal values

    # Create initial parameters
    initial_params = DFSVParamsDataclass(
        N=N, K=K,
        lambda_r=lambda_r_init,
        Phi_f=phi_f_init,
        Phi_h=phi_h_init,
        mu=jnp.zeros(K),
        sigma2=0.1 * jnp.ones(N),
        Q_h=0.2 * jnp.eye(K)
    )

    return initial_params


# Using run_optimization from the main code



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

    # 1. Create model parameters
    N, K = 5, 3
    true_params = create_simple_model(N=N, K=K)
    print(f"Created model with N={true_params.N}, K={true_params.K}")
    print("True Parameters:")
    print(true_params)

    # 2. Generate simulation data
    T = 500  # Full experiment with longer time series
    print(f"\nGenerating {T} time steps of simulation data...")
    returns = create_training_data(true_params, T=T, seed=123)
    print("Simulation data generated.")

    # 3. Define optimization configurations
    # Run all three filters
    filter_types = [FilterType.BIF, FilterType.PF]

    # Use both AdamW and DampedTrustRegionBFGS optimizers
    optimizers = ["AdamW", "DampedTrustRegionBFGS"]
    use_transformations_options = [True]  # Always use transformations for stability
    fix_mu_options = [True]  # Run only with fixed fixed mu
    max_steps = 200  # Increased for better convergence
    stability_penalty_weight = 1000.0
    num_particles = 5000

    # 4. Run optimizations
    results = []

    print(f"\nRunning optimizations with max_steps={max_steps}, stability_penalty_weight={stability_penalty_weight}...")

    for filter_type in filter_types:
        for optimizer_name in optimizers:
            for use_transformations in use_transformations_options:
                for fix_mu in fix_mu_options:
                    print(f"\n--- Running: Filter={filter_type.name} | Optimizer={optimizer_name} | Transform={'Yes' if use_transformations else 'No'} | Fix_mu={'Yes' if fix_mu else 'No'} ---")

                    # Skip certain filter-optimizer combinations that are known to be problematic
                    # if (filter_type == FilterType.PF and optimizer_name == "BFGS") or \
                    #    (filter_type == FilterType.BF and optimizer_name == "AdamW"):
                    #     print(f"Skipping {filter_type.name} with {optimizer_name} (known compatibility issue)")
                    #     continue

                    try:
                        # Run optimization with error handling
                        result = run_optimization(
                            filter_type=filter_type,
                            returns=returns,
                            true_params=true_params if fix_mu else None,  # Only pass true_params if fix_mu is True
                            use_transformations=use_transformations,
                            optimizer_name=optimizer_name,
                            priors=None,
                            stability_penalty_weight=stability_penalty_weight,
                            max_steps=max_steps,
                            num_particles=num_particles,
                            verbose=True,
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

