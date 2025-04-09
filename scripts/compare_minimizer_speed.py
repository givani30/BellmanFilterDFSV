#!/usr/bin/env python
"""
Minimal test script to compare the runtime of minimize_with_lax_while vs minimize_with_logging.

This script creates a simple DFSV model and runs optimization with both minimizers,
measuring the time taken for each.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

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

    # Run optimization with specified minimizer
    start_time = time.time()
    result = run_optimization(
        filter_type=FilterType.BIF,
        returns=returns,
        true_params=true_params,  # Fix mu to true values
        use_transformations=True,
        optimizer_name=optimizer_name,
        priors=None,
        stability_penalty_weight=1000.0,
        max_steps=max_steps,
        verbose=verbose,
        log_params=True,
        log_interval=1,
        use_lax_while=use_lax_while
    )
    end_time = time.time()

    # Return result and time taken
    return result, end_time - start_time


def main():
    """Main function to run the speed test."""
    print("Starting minimizer speed comparison...")

    # Define test parameters
    optimizer_name = "BFGS"  # Use BFGS as it worked well in previous tests

    # Create dictionaries to store results for each step count
    results = {}
    times = {}

    for max_steps in [10, 50]:
        print(f"\n=== Test with {max_steps} steps ===")
        results[max_steps] = {}
        times[max_steps] = {}

        # Test 1: minimize_with_logging (verbose=True, use_lax_while=False)
        print("\nTest 1: minimize_with_logging (verbose=True, use_lax_while=False)")
        result1, time1 = run_speed_test(
            optimizer_name=optimizer_name,
            max_steps=max_steps,
            use_lax_while=False,
            verbose=True
        )
        results[max_steps]["logging_verbose"] = result1
        times[max_steps]["logging_verbose"] = time1
        print(f"Time taken: {time1:.2f} seconds")
        print(f"Final loss: {result1.final_loss:.4e}")
        print(f"Steps completed: {result1.steps}")

        # Test 2: minimize_with_lax_while (verbose=False, use_lax_while=True)
        print("\nTest 2: minimize_with_lax_while (verbose=False, use_lax_while=True)")
        result2, time2 = run_speed_test(
            optimizer_name=optimizer_name,
            max_steps=max_steps,
            use_lax_while=True,
            verbose=False
        )
        results[max_steps]["lax_while"] = result2
        times[max_steps]["lax_while"] = time2
        print(f"Time taken: {time2:.2f} seconds")
        print(f"Final loss: {result2.final_loss:.4e}")
        print(f"Steps completed: {result2.steps}")

        # Test 3: minimize_with_logging (verbose=False, use_lax_while=False)
        # This isolates the effect of verbose output
        print("\nTest 3: minimize_with_logging (verbose=False, use_lax_while=False)")
        result3, time3 = run_speed_test(
            optimizer_name=optimizer_name,
            max_steps=max_steps,
            use_lax_while=False,
            verbose=False
        )
        results[max_steps]["logging_silent"] = result3
        times[max_steps]["logging_silent"] = time3
        print(f"Time taken: {time3:.2f} seconds")
        print(f"Final loss: {result3.final_loss:.4e}")
        print(f"Steps completed: {result3.steps}")

        # Compare results for this step count
        print(f"\nSpeed Comparison for {max_steps} steps:")
        print(f"minimize_with_logging (verbose=True): {time1:.2f} seconds")
        print(f"minimize_with_lax_while: {time2:.2f} seconds")
        print(f"minimize_with_logging (verbose=False): {time3:.2f} seconds")

        # Calculate speedup/slowdown
        speedup1 = time1 / time2
        speedup2 = time3 / time2
        print(f"\nSpeedup of lax_while vs logging (verbose=True): {speedup1:.2f}x")
        print(f"Speedup of lax_while vs logging (verbose=False): {speedup2:.2f}x")

    # Plot loss history for each step count
    for max_steps in [10, 50]:
        plt.figure(figsize=(10, 6))
        if results[max_steps]["logging_verbose"].loss_history is not None:
            plt.plot(results[max_steps]["logging_verbose"].loss_history, label="minimize_with_logging (verbose=True)")
        if results[max_steps]["lax_while"].loss_history is not None:
            plt.plot(results[max_steps]["lax_while"].loss_history, label="minimize_with_lax_while")
        if results[max_steps]["logging_silent"].loss_history is not None:
            plt.plot(results[max_steps]["logging_silent"].loss_history, label="minimize_with_logging (verbose=False)")

        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"Loss History Comparison ({max_steps} steps)")
        plt.legend()
        plt.yscale("log")
        plt.grid(True)

        # Save plot
        plt.savefig(f"outputs/minimizer_speed_comparison_{max_steps}_steps.png")
        print(f"\nLoss history plot saved to outputs/minimizer_speed_comparison_{max_steps}_steps.png")

    # Print summary
    print("\n=== Summary ===")
    print("Time taken (seconds):")
    print(f"{'Steps':<10} | {'Logging (verbose)':<20} | {'Lax While':<15} | {'Logging (silent)':<20} | {'Lax While Speedup vs Verbose':<30} | {'Lax While Speedup vs Silent':<30}")
    print("-" * 120)
    for max_steps in [10, 50]:
        time1 = times[max_steps]["logging_verbose"]
        time2 = times[max_steps]["lax_while"]
        time3 = times[max_steps]["logging_silent"]
        speedup1 = time1 / time2
        speedup2 = time3 / time2
        print(f"{max_steps:<10} | {time1:<20.2f} | {time2:<15.2f} | {time3:<20.2f} | {speedup1:<30.2f} | {speedup2:<30.2f}")


if __name__ == "__main__":
    main()
