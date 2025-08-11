#!/usr/bin/env python
"""
Minimal test script to compare the runtime of minimize_with_lax_while vs minimize_with_logging.

This script creates a simple DFSV model and runs optimization with both minimizers,
measuring the time taken for each.
"""

import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Project specific imports
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.utils.transformations import apply_identification_constraint, transform_params
from bellman_filter_dfsv.utils.optimization import (
    FilterType,
    minimize_with_lax_while,
    minimize_with_logging,
    get_objective_function
)
from bellman_filter_dfsv.utils.solvers import create_optimizer
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)


def create_simple_model(N: int = 3, K: int = 2) -> DFSVParamsDataclass:
    """Create a simple DFSV model with reasonable parameters."""
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
    Phi_f_off = jax.random.uniform(subkey1, (K, K), minval=0.01, maxval=0.05)
    key, subkey1 = jax.random.split(key)
    Phi_f_diag = jax.random.uniform(subkey1, (K,), minval=0.3, maxval=0.5)
    Phi_f = Phi_f_off
    Phi_f = Phi_f.at[diag_indices].set(Phi_f_diag)

    # Log-volatility persistence (diagonal-dominant with eigenvalues < 1)
    key, subkey1 = jax.random.split(key)
    Phi_h_off = jax.random.uniform(subkey1, (K, K), minval=0.01, maxval=0.1)
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


def create_training_data(params: DFSVParamsDataclass, T: int = 500, seed: int = 123) -> jnp.ndarray:
    """Generate simulation data for training."""
    # Simulate data
    returns, _, _ = simulate_DFSV(
        params=params,
        T=T,
        seed=seed
    )
    return returns


def run_speed_test(optimizer_name: str, max_steps: int, use_lax_while: bool, verbose: bool):
    """Run a speed test for the specified minimizer."""
    # Create model and data
    true_params = create_simple_model(N=3, K=2)
    returns = create_training_data(true_params, T=500, seed=123)

    print(f"Model created with N={true_params.N}, K={true_params.K}")
    print(f"Generated {len(returns)} time steps of data")

    # Create filter instance
    filter_instance = DFSVBellmanInformationFilter(N=true_params.N, K=true_params.K)

    # Transform parameters if needed
    initial_params = transform_params(true_params)

    # Get objective function
    objective_fn = get_objective_function(
        filter_type=FilterType.BIF,
        filter_instance=filter_instance,
        stability_penalty_weight=1000.0,
        priors=None,
        is_transformed=True,
        fix_mu=True,
        true_mu=true_params.mu
    )

    # Create optimizer
    optimizer = create_optimizer(
        optimizer_name=optimizer_name,
        learning_rate=1e-3,
        rtol=1e-5,
        atol=1e-5,
        verbose=verbose
    )

    # Run optimization with specified minimizer
    print(f"Starting optimization with {optimizer_name}, max_steps={max_steps}, use_lax_while={use_lax_while}, verbose={verbose}")
    start_time = time.time()
    try:
        if use_lax_while:
            print("Using minimize_with_lax_while")
            sol, param_history = minimize_with_lax_while(
                objective_fn=objective_fn,
                initial_params=initial_params,
                solver=optimizer,
                static_args=returns,
                max_steps=max_steps,
                log_interval=1,
                throw=False,
                options={},
                verbose=verbose
            )
        else:
            print("Using minimize_with_logging")
            sol, param_history = minimize_with_logging(
                objective_fn=objective_fn,
                initial_params=initial_params,
                solver=optimizer,
                static_args=returns,
                max_steps=max_steps,
                log_interval=1,
                throw=False,
                options={},
                verbose=verbose
            )

        end_time = time.time()
        print("Optimization completed successfully")
        print(f"Result: {sol.result}")
        print(f"Steps: {sol.stats.get('num_steps', 0)}")
        print(f"Final value shape: {jax.tree_map(lambda x: x.shape, sol.value)}")

        # Create a result object similar to what run_optimization returns
        result = type('OptimizerResult', (), {
            'final_loss': float(objective_fn(sol.value, returns)[0]),
            'steps': sol.stats.get('num_steps', 0),
            'result_code': sol.result,
            'loss_history': [float(objective_fn(p, returns)[0]) for p in param_history] if param_history else None
        })

    except Exception as e:
        end_time = time.time()
        print(f"Error during optimization: {e}")
        raise

    # Return result and time taken
    return result, end_time - start_time


def main():
    """Main function to run the speed test."""
    print("Starting minimizer speed comparison...")

    # Define test parameters
    optimizer_name = "DampedTrustRegionBFGS"  # Use BFGS as it worked well in previous tests

    # Create dictionaries to store results for each step count
    results = {}
    times = {}

    # Test with different step counts
    for max_steps in [50]:
        print(f"\n=== Test with {max_steps} steps ===")
        results[max_steps] = {}
        times[max_steps] = {}

        # Test with lax_while
        print("\nRunning minimize_with_lax_while (verbose=False, use_lax_while=True)")
        result_lax, time_lax = run_speed_test(
            optimizer_name=optimizer_name,
            max_steps=max_steps,
            use_lax_while=True,
            verbose=False
        )
        results[max_steps]["lax_while"] = result_lax
        times[max_steps]["lax_while"] = time_lax
        print(f"Time taken: {time_lax:.2f} seconds")
        print(f"Final loss: {result_lax.final_loss:.4e}")
        print(f"Steps completed: {result_lax.steps}")
        print(f"Result code: {result_lax.result_code}")

        # Test with standard logging
        print("\nRunning minimize_with_logging (verbose=False, use_lax_while=False)")
        result_std, time_std = run_speed_test(
            optimizer_name=optimizer_name,
            max_steps=max_steps,
            use_lax_while=False,
            verbose=False
        )
        results[max_steps]["logging"] = result_std
        times[max_steps]["logging"] = time_std
        print(f"Time taken: {time_std:.2f} seconds")
        print(f"Final loss: {result_std.final_loss:.4e}")
        print(f"Steps completed: {result_std.steps}")
        print(f"Result code: {result_std.result_code}")

        # Calculate speedup
        speedup = time_std / time_lax
        print(f"\nSpeedup of lax_while vs logging: {speedup:.2f}x")

    # Plot loss history for each step count
    for max_steps in [50]:
        if max_steps in results and "lax_while" in results[max_steps] and "logging" in results[max_steps]:
            plt.figure(figsize=(10, 6))
            if results[max_steps]["lax_while"].loss_history is not None:
                plt.plot(results[max_steps]["lax_while"].loss_history, label="minimize_with_lax_while")
            if results[max_steps]["logging"].loss_history is not None:
                plt.plot(results[max_steps]["logging"].loss_history, label="minimize_with_logging")

            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title(f"Loss History Comparison - {max_steps} steps")
            plt.legend()
            plt.yscale("log")
            plt.grid(True)

            # Save plot
            plt.savefig(f"outputs/minimizer_comparison_{max_steps}_steps.png")
            print(f"\nLoss history plot saved to outputs/minimizer_comparison_{max_steps}_steps.png")

    # Print summary
    print("\n=== Summary ===")
    print("Time taken (seconds):")
    print(f"{'Steps':<10} | {'Lax While':<15} | {'Logging':<15} | {'Speedup':<10} | {'Lax While Result':<30} | {'Logging Result':<30}")
    print("-" * 120)
    for max_steps in [50]:
        if max_steps in times and "lax_while" in times[max_steps] and "logging" in times[max_steps]:
            time_lax = times[max_steps]["lax_while"]
            time_std = times[max_steps]["logging"]
            speedup = time_std / time_lax
            result_lax = results[max_steps]["lax_while"].result_code
            result_std = results[max_steps]["logging"].result_code
            print(f"{max_steps:<10} | {time_lax:<15.2f} | {time_std:<15.2f} | {speedup:<10.2f} | {str(result_lax):<30} | {str(result_std):<30}")


if __name__ == "__main__":
    main()
