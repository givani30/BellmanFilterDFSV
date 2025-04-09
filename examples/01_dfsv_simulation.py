#!/usr/bin/env python
"""
DFSV Model Simulation Example

This example demonstrates how to:
1. Create a Dynamic Factor Stochastic Volatility (DFSV) model
2. Simulate data from this model
3. Visualize the results

The DFSV model combines factor models with stochastic volatility, allowing
for time-varying volatility in the latent factors that drive returns.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV

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


def plot_simulation_results(returns, factors, log_vols, params):
    """
    Plot the simulated data from a DFSV model.

    Args:
        returns (np.ndarray): Simulated returns with shape (T, N)
        factors (np.ndarray): Simulated factors with shape (T, K)
        log_vols (np.ndarray): Simulated log-volatilities with shape (T, K)
        params (DFSVParamsDataclass): Model parameters
    """
    T = returns.shape[0]
    time_axis = np.arange(T)

    # Create a figure with subplots
    fig = plt.figure(figsize=(12, 10))

    # Plot factors
    ax1 = fig.add_subplot(3, 1, 1)
    for k in range(params.K):
        ax1.plot(time_axis, factors[:, k], label=f'Factor {k+1}')
    ax1.set_title('Latent Factors')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Factor Value')
    ax1.legend()
    ax1.grid(True)

    # Plot log-volatilities
    ax2 = fig.add_subplot(3, 1, 2)
    for k in range(params.K):
        ax2.plot(time_axis, log_vols[:, k], label=f'Log-Vol {k+1}')
    ax2.set_title('Log-Volatilities')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Log-Volatility')
    ax2.legend()
    ax2.grid(True)

    # Plot returns
    ax3 = fig.add_subplot(3, 1, 3)
    for n in range(min(params.N, 3)):  # Plot at most 3 return series to avoid clutter
        ax3.plot(time_axis, returns[:, n], label=f'Returns {n+1}', alpha=0.7)
    ax3.set_title('Observed Returns')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Return')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


def analyze_simulation(returns, factors, log_vols):
    """
    Analyze the simulated data from a DFSV model.

    Args:
        returns (np.ndarray): Simulated returns with shape (T, N)
        factors (np.ndarray): Simulated factors with shape (T, K)
        log_vols (np.ndarray): Simulated log-volatilities with shape (T, K)
    """
    # Calculate volatilities from log-volatilities
    volatilities = np.exp(log_vols / 2)

    # Calculate statistics
    print("\nSimulation Statistics:")
    print(f"Returns mean: {returns.mean(axis=0)}")
    print(f"Returns std: {returns.std(axis=0)}")
    print(f"Factors mean: {factors.mean(axis=0)}")
    print(f"Factors std: {factors.std(axis=0)}")
    print(f"Log-volatilities mean: {log_vols.mean(axis=0)}")
    print(f"Log-volatilities std: {log_vols.std(axis=0)}")
    print(f"Volatilities mean: {volatilities.mean(axis=0)}")
    print(f"Volatilities min: {volatilities.min(axis=0)}")
    print(f"Volatilities max: {volatilities.max(axis=0)}")

    # Calculate correlations between returns
    corr_returns = np.corrcoef(returns.T)
    print("\nReturn Correlations:")
    print(corr_returns)

    # Calculate autocorrelations for factors and volatilities
    def autocorr(x, lag=1):
        """Calculate autocorrelation at specified lag."""
        return np.corrcoef(x[lag:], x[:-lag])[0, 1]

    print("\nFactor Autocorrelations (lag=1):")
    for k in range(factors.shape[1]):
        print(f"Factor {k+1}: {autocorr(factors[:, k]):.4f}")

    print("\nLog-Volatility Autocorrelations (lag=1):")
    for k in range(log_vols.shape[1]):
        print(f"Log-Vol {k+1}: {autocorr(log_vols[:, k]):.4f}")


def main():
    """Run the DFSV simulation example."""
    print("DFSV Model Simulation Example")
    print("=============================")

    # Create model parameters
    N, K = 5, 2  # 5 observed series, 2 factors
    params = create_simple_dfsv_model(N, K)
    print(f"Created DFSV model with {params.N} observed series and {params.K} factors")

    # Set simulation parameters
    T = 1000  # Number of time periods
    seed = 42  # Random seed for reproducibility

    # Run simulation
    print(f"Simulating DFSV model for T={T} time periods...")
    returns, factors, log_vols = simulate_DFSV(
        params=params,
        T=T,
        seed=seed
    )
    print("Simulation complete!")

    # Analyze simulation results
    analyze_simulation(returns, factors, log_vols)

    # Plot simulation results
    plot_simulation_results(returns, factors, log_vols, params)

    return returns, factors, log_vols, params


if __name__ == "__main__":
    main()
