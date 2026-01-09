#!/usr/bin/env python
"""
Basic Filtering Example for DFSV Models

This example demonstrates how to:
1. Create a DFSV model and simulate data
2. Apply different filters to estimate the latent states:
   - Bellman Information Filter (BIF)
   - Particle Filter (PF)
3. Compare filter performance and visualize results

v2.0.0 - Updated for new architecture using Equinox and functional patterns.
"""

import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from bellman_filter_dfsv import BellmanFilter, DFSVParams, ParticleFilter, simulate_dfsv


def create_simple_dfsv_model(N=3, K=1):
    """
    Create a simple DFSV model with K factors and N observed series.

    Args:
        N (int): Number of observed series
        K (int): Number of latent factors

    Returns:
        DFSVParams: Parameters for the DFSV model
    """
    # Factor loadings - how each observed series is affected by the factors
    lambda_r = np.random.uniform(0.3, 0.9, size=(N, K))

    # Factor persistence - how strongly factors depend on their previous values
    Phi_f = np.eye(K) * 0.7  # Moderate persistence

    # Volatility persistence - how strongly log-volatilities depend on their previous values
    Phi_h = np.eye(K) * 0.97  # High persistence

    # Long-run mean of log-volatilities
    mu = np.ones(K) * -1.0  # Negative means lower volatility

    # Idiosyncratic variance of observed series
    sigma2 = np.ones(N) * 0.1  # Low idiosyncratic variance

    # Covariance matrix of log-volatility innovations
    Q_h = np.eye(K) * 0.05  # Low volatility of volatility

    # Create parameter object using JAX arrays
    params = DFSVParams(
        lambda_r=jnp.array(lambda_r),
        Phi_f=jnp.array(Phi_f),
        Phi_h=jnp.array(Phi_h),
        mu=jnp.array(mu),
        sigma2=jnp.array(sigma2),
        Q_h=jnp.array(Q_h),
    )

    return params


def run_filters(params, returns, true_factors, true_log_vols):
    """
    Run different filters on the simulated data and compare their performance.

    Args:
        params (DFSVParams): Model parameters
        returns (np.ndarray): Simulated returns with shape (T, N)
        true_factors (np.ndarray): True factors with shape (T, K)
        true_log_vols (np.ndarray): True log-volatilities with shape (T, K)

    Returns:
        dict: Dictionary containing filter results and performance metrics
    """
    N, K = params.lambda_r.shape

    # Initialize filters
    bif = BellmanFilter(params)
    pf = ParticleFilter(params, num_particles=1000)

    # Run Bellman Information Filter
    print("Running Bellman Information Filter...")
    start_time = time.time()
    bif_result = bif.filter(returns)
    bif_time = time.time() - start_time
    print(f"Bellman Information Filter completed in {bif_time:.4f} seconds")

    # Run Particle Filter
    print("Running Particle Filter...")
    start_time = time.time()
    pf_result = pf.filter(returns)
    pf_time = time.time() - start_time
    print(f"Particle Filter completed in {pf_time:.4f} seconds")

    # Calculate RMSE for factors and log-volatilities
    def calculate_rmse(estimated, true):
        """Calculate Root Mean Squared Error."""
        return np.sqrt(np.mean((estimated - true) ** 2, axis=0))

    # Extract factors and log-volatilities from states
    # State vector is [factors, log_vols]
    bif_factors = np.array(bif_result.means[:, :K])
    bif_log_vols = np.array(bif_result.means[:, K:])

    pf_factors = np.array(pf_result.means[:, :K])
    pf_log_vols = np.array(pf_result.means[:, K:])

    # Calculate RMSE
    bif_factor_rmse = calculate_rmse(bif_factors, true_factors)
    bif_log_vol_rmse = calculate_rmse(bif_log_vols, true_log_vols)

    pf_factor_rmse = calculate_rmse(pf_factors, true_factors)
    pf_log_vol_rmse = calculate_rmse(pf_log_vols, true_log_vols)

    # Print performance metrics
    print("\nFilter Performance Metrics:")
    print(
        f"Bellman Information Filter - Log-Likelihood: {bif_result.log_likelihood:.2f}, Time: {bif_time:.4f}s"
    )
    print(f"  Factor RMSE: {bif_factor_rmse}")
    print(f"  Log-Vol RMSE: {bif_log_vol_rmse}")

    print(
        f"Particle Filter - Log-Likelihood: {pf_result.log_likelihood:.2f}, Time: {pf_time:.4f}s"
    )
    print(f"  Factor RMSE: {pf_factor_rmse}")
    print(f"  Log-Vol RMSE: {pf_log_vol_rmse}")

    # Return results
    results = {
        "bif": {
            "result": bif_result,
            "ll": float(bif_result.log_likelihood),
            "time": bif_time,
            "factor_rmse": bif_factor_rmse,
            "log_vol_rmse": bif_log_vol_rmse,
        },
        "pf": {
            "result": pf_result,
            "ll": float(pf_result.log_likelihood),
            "time": pf_time,
            "factor_rmse": pf_factor_rmse,
            "log_vol_rmse": pf_log_vol_rmse,
        },
    }

    return results


def plot_filter_comparison(results, true_factors, true_log_vols, K):
    """
    Plot comparison of filter estimates against true states.

    Args:
        results (dict): Dictionary containing filter results
        true_factors (np.ndarray): True factors with shape (T, K)
        true_log_vols (np.ndarray): True log-volatilities with shape (T, K)
        K (int): Number of factors
    """
    T = true_factors.shape[0]
    time_axis = np.arange(T)

    # Create figure for factor comparison
    plt.figure(figsize=(12, 4 * K))

    for k in range(K):
        plt.subplot(K, 1, k + 1)
        plt.plot(time_axis, true_factors[:, k], "k-", label="True", alpha=0.7)
        plt.plot(
            time_axis,
            results["bif"]["result"].means[:, k],
            "b-",
            label="BIF",
            alpha=0.7,
        )
        plt.plot(
            time_axis,
            results["pf"]["result"].means[:, k],
            "r-",
            label="PF",
            alpha=0.7,
        )
        plt.title(f"Factor {k + 1} Comparison")
        plt.xlabel("Time")
        plt.ylabel("Factor Value")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Create figure for filter performance comparison
    plt.figure(figsize=(10, 6))

    # Prepare data for bar chart
    filter_names = ["BIF", "PF"]
    times = [results["bif"]["time"], results["pf"]["time"]]
    lls = [results["bif"]["ll"], results["pf"]["ll"]]

    # Plot computation time
    plt.subplot(1, 2, 1)
    plt.bar(filter_names, times, color=["blue", "red"])
    plt.title("Computation Time")
    plt.ylabel("Time (seconds)")
    plt.grid(True, axis="y")

    # Plot log-likelihood
    plt.subplot(1, 2, 2)
    plt.bar(filter_names, lls, color=["blue", "red"])
    plt.title("Log-Likelihood")
    plt.ylabel("Log-Likelihood")
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.show()


def main():
    """Run the basic filtering example."""
    print("Basic Filtering Example for DFSV Models (v2.0.0)")
    print("==================================================")

    # Create model parameters
    N, K = 3, 1  # 3 observed series, 1 factor
    params = create_simple_dfsv_model(N, K)
    print(f"Created DFSV model with {N} observed series and {K} factors")

    # Set simulation parameters
    T = 500  # Number of time periods (shorter for faster filtering)
    seed = 42  # Random seed for reproducibility

    # Run simulation
    print(f"Simulating DFSV model for T={T} time periods...")
    returns, factors, log_vols = simulate_dfsv(params, T=T, key=seed)

    # Convert to numpy
    returns = np.array(returns)
    factors = np.array(factors)
    log_vols = np.array(log_vols)

    print("Simulation complete!")

    # Run filters and compare performance
    filter_results = run_filters(params, returns, factors, log_vols)

    # Plot filter comparison
    plot_filter_comparison(filter_results, factors, log_vols, K)

    return returns, factors, log_vols, params, filter_results


if __name__ == "__main__":
    main()
