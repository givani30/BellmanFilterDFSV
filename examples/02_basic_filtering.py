#!/usr/bin/env python
"""
Basic Filtering Example for DFSV Models

This example demonstrates how to:
1. Create a DFSV model and simulate data
2. Apply different filters to estimate the latent states:
   - Bellman Filter (BF)
   - Bellman Information Filter (BIF)
   - Particle Filter (PF)
3. Compare filter performance and visualize results
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import time
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.filters.bellman import DFSVBellmanFilter
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.filters.particle import DFSVParticleFilter

# Set random seed for reproducibility
np.random.seed(42)


def create_simple_dfsv_model(N=3, K=1):
    """
    Create a simple DFSV model with K factors and N observed series.

    Args:
        N (int): Number of observed series
        K (int): Number of latent factors

    Returns:
        DFSVParamsDataclass: Parameters for the DFSV model
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
    params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=jnp.array(lambda_r),
        Phi_f=jnp.array(Phi_f),
        Phi_h=jnp.array(Phi_h),
        mu=jnp.array(mu),
        sigma2=jnp.array(sigma2),
        Q_h=jnp.array(Q_h)
    )

    return params


def run_filters(params, returns, true_factors, true_log_vols):
    """
    Run different filters on the simulated data and compare their performance.

    Args:
        params (DFSVParamsDataclass): Model parameters
        returns (np.ndarray): Simulated returns with shape (T, N)
        true_factors (np.ndarray): True factors with shape (T, K)
        true_log_vols (np.ndarray): True log-volatilities with shape (T, K)

    Returns:
        dict: Dictionary containing filter results and performance metrics
    """
    T, N = returns.shape
    K = params.K

    # Convert returns to JAX array for filter compatibility
    jax_returns = jnp.array(returns)

    # Initialize filters
    bf = DFSVBellmanFilter(N, K)
    bif = DFSVBellmanInformationFilter(N, K)
    pf = DFSVParticleFilter(N, K, num_particles=1000)

    # Run Bellman Filter
    print("Running Bellman Filter...")
    start_time = time.time()
    bf_states, bf_covs, bf_ll = bf.filter(params, jax_returns)
    bf_time = time.time() - start_time
    print(f"Bellman Filter completed in {bf_time:.4f} seconds")

    # Run Bellman Information Filter
    print("Running Bellman Information Filter...")
    start_time = time.time()
    bif_states, bif_covs, bif_ll = bif.filter(params, jax_returns)
    bif_time = time.time() - start_time
    print(f"Bellman Information Filter completed in {bif_time:.4f} seconds")

    # Run Particle Filter
    print("Running Particle Filter...")
    start_time = time.time()
    pf_states, pf_covs, pf_ll = pf.filter(params, jax_returns)
    pf_time = time.time() - start_time
    print(f"Particle Filter completed in {pf_time:.4f} seconds")

    # Calculate RMSE for factors and log-volatilities
    def calculate_rmse(estimated, true):
        """Calculate Root Mean Squared Error."""
        return np.sqrt(np.mean((estimated - true) ** 2, axis=0))

    # Extract factors and log-volatilities from states
    # State vector is [factors, log_vols]
    bf_factors = np.array(bf_states[:, :K])
    bf_log_vols = np.array(bf_states[:, K:])

    bif_factors = np.array(bif_states[:, :K])
    bif_log_vols = np.array(bif_states[:, K:])

    pf_factors = np.array(pf_states[:, :K])
    pf_log_vols = np.array(pf_states[:, K:])

    # Calculate RMSE
    bf_factor_rmse = calculate_rmse(bf_factors, true_factors)
    bf_log_vol_rmse = calculate_rmse(bf_log_vols, true_log_vols)

    bif_factor_rmse = calculate_rmse(bif_factors, true_factors)
    bif_log_vol_rmse = calculate_rmse(bif_log_vols, true_log_vols)

    pf_factor_rmse = calculate_rmse(pf_factors, true_factors)
    pf_log_vol_rmse = calculate_rmse(pf_log_vols, true_log_vols)

    # Print performance metrics
    print("\nFilter Performance Metrics:")
    print(f"Bellman Filter - Log-Likelihood: {bf_ll:.2f}, Time: {bf_time:.4f}s")
    print(f"  Factor RMSE: {bf_factor_rmse}")
    print(f"  Log-Vol RMSE: {bf_log_vol_rmse}")

    print(f"Bellman Information Filter - Log-Likelihood: {bif_ll:.2f}, Time: {bif_time:.4f}s")
    print(f"  Factor RMSE: {bif_factor_rmse}")
    print(f"  Log-Vol RMSE: {bif_log_vol_rmse}")

    print(f"Particle Filter - Log-Likelihood: {pf_ll:.2f}, Time: {pf_time:.4f}s")
    print(f"  Factor RMSE: {pf_factor_rmse}")
    print(f"  Log-Vol RMSE: {pf_log_vol_rmse}")

    # Return results
    results = {
        'bf': {
            'states': bf_states,
            'covs': bf_covs,
            'll': bf_ll,
            'time': bf_time,
            'factor_rmse': bf_factor_rmse,
            'log_vol_rmse': bf_log_vol_rmse
        },
        'bif': {
            'states': bif_states,
            'covs': bif_covs,
            'll': bif_ll,
            'time': bif_time,
            'factor_rmse': bif_factor_rmse,
            'log_vol_rmse': bif_log_vol_rmse
        },
        'pf': {
            'states': pf_states,
            'covs': pf_covs,
            'll': pf_ll,
            'time': pf_time,
            'factor_rmse': pf_factor_rmse,
            'log_vol_rmse': pf_log_vol_rmse
        }
    }

    return results


def plot_filter_comparison(results, true_factors, true_log_vols, params):
    """
    Plot comparison of filter estimates against true states.

    Args:
        results (dict): Dictionary containing filter results
        true_factors (np.ndarray): True factors with shape (T, K)
        true_log_vols (np.ndarray): True log-volatilities with shape (T, K)
        params (DFSVParamsDataclass): Model parameters
    """
    T = true_factors.shape[0]
    time_axis = np.arange(T)
    K = params.K

    # Create figure for factor comparison
    plt.figure(figsize=(12, 4 * K))

    for k in range(K):
        plt.subplot(K, 1, k + 1)
        plt.plot(time_axis, true_factors[:, k], 'k-', label='True', alpha=0.7)
        plt.plot(time_axis, results['bf']['states'][:, k], 'b-', label='BF', alpha=0.7)
        plt.plot(time_axis, results['bif']['states'][:, k], 'g-', label='BIF', alpha=0.7)
        plt.plot(time_axis, results['pf']['states'][:, k], 'r-', label='PF', alpha=0.7)
        plt.title(f'Factor {k+1} Comparison')
        plt.xlabel('Time')
        plt.ylabel('Factor Value')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Note: Log-volatility comparison removed as requested

    # Create figure for filter performance comparison
    plt.figure(figsize=(10, 6))

    # Prepare data for bar chart
    filter_names = ['BF', 'BIF', 'PF']
    times = [results['bf']['time'], results['bif']['time'], results['pf']['time']]
    lls = [float(results['bf']['ll']), float(results['bif']['ll']), float(results['pf']['ll'])]

    # Plot computation time
    plt.subplot(1, 2, 1)
    plt.bar(filter_names, times, color=['blue', 'green', 'red'])
    plt.title('Computation Time')
    plt.ylabel('Time (seconds)')
    plt.grid(True, axis='y')

    # Plot log-likelihood
    plt.subplot(1, 2, 2)
    plt.bar(filter_names, lls, color=['blue', 'green', 'red'])
    plt.title('Log-Likelihood')
    plt.ylabel('Log-Likelihood')
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.show()


def main():
    """Run the basic filtering example."""
    print("Basic Filtering Example for DFSV Models")
    print("======================================")

    # Create model parameters
    N, K = 3, 1  # 3 observed series, 1 factor
    params = create_simple_dfsv_model(N, K)
    print(f"Created DFSV model with {params.N} observed series and {params.K} factors")

    # Set simulation parameters
    T = 500  # Number of time periods (shorter for faster filtering)
    seed = 42  # Random seed for reproducibility

    # Run simulation
    print(f"Simulating DFSV model for T={T} time periods...")
    returns, factors, log_vols = simulate_DFSV(
        params=params,
        T=T,
        seed=seed
    )
    print("Simulation complete!")

    # Run filters and compare performance
    filter_results = run_filters(params, returns, factors, log_vols)

    # Plot filter comparison
    plot_filter_comparison(filter_results, factors, log_vols, params)

    return returns, factors, log_vols, params, filter_results


if __name__ == "__main__":
    main()
