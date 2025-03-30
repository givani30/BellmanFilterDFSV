"""
Simple DFSV model example with K=1 and N=3.

This example demonstrates how to:
1. Create a simple DFSV model with one factor (K=1) and three observed series (N=3)
2. Simulate data from this model
3. Plot the results to visualize the relationship between factors and returns
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path so we can import from functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import DFSV model related functions
from models.dfsv import DFSV_params
from functions.simulation import simulate_DFSV


def create_simple_dfsv_model():
    """
    Create a simple DFSV model with K=1 factor and N=3 observed series.
    
    Returns:
        DFSV_params: Parameters for the simple model
    """
    # Set dimensions
    N = 3  # Number of observed series
    K = 1  # Number of latent factors
    
    # Define parameters
    # Factor loadings - how each observed series is affected by the factor
    lambda_r = np.array([[0.8], [0.5], [0.3]])  # Shape: (N, K) = (3, 1)
    
    # Factor persistence - how strongly the factor depends on its previous value
    Phi_f = np.array([[0.95]])  # Shape: (K, K) = (1, 1), high persistence
    
    # Volatility persistence - how strongly log-volatility depends on its previous value
    Phi_h = np.array([[0.97]])  # Shape: (K, K) = (1, 1), high persistence
    
    # Long-run mean of log-volatility
    mu = np.array([-1.0])  # Shape: (K,) = (1,)
    
    # Idiosyncratic variance - noise variance specific to each observed series
    sigma2 = np.array([0.2, 0.3, 0.1])  # Shape: (N,) = (3,)
    
    # Volatility of log-volatility - how much the volatility itself fluctuates
    Q_h = np.array([[0.1]])  # Shape: (K, K) = (1, 1)
    
    # Create and return parameter object
    return DFSV_params(
        N=N, 
        K=K,
        lambda_r=lambda_r,
        Phi_f=Phi_f,
        Phi_h=Phi_h,
        mu=mu,
        sigma2=sigma2,
        Q_h=Q_h
    )


def run_simulation_example():
    """
    Run a simulation example using a simple DFSV model.
    Simulates, visualizes and analyzes the results.
    """
    # Create model parameters
    params = create_simple_dfsv_model()
    
    # Set simulation parameters
    T = 1000  # Number of time periods
    seed = 42  # Random seed for reproducibility
    
    # Run simulation
    print(f"Simulating DFSV model with K={params.K} factors and N={params.N} observed series...")
    returns, factors, log_vols = simulate_DFSV(
        params=params,
        T=T,
        seed=seed
    )
    
    # Calculate volatilities (standard deviations) from log-volatilities
    volatilities = np.exp(log_vols / 2)
    
    # Plot results
    plot_simulation_results(returns, factors, volatilities, params)
    
    # Print some statistics
    print("\nSimulation Statistics:")
    print(f"Returns mean: {returns.mean(axis=0)}")
    print(f"Returns std: {returns.std(axis=0)}")
    print(f"Factor mean: {factors.mean(axis=0)[0]:.4f}, std: {factors.std(axis=0)[0]:.4f}")
    print(f"Volatility mean: {volatilities.mean():.4f}, min: {volatilities.min():.4f}, max: {volatilities.max():.4f}")
    
    return returns, factors, log_vols


def plot_simulation_results(returns, factors, volatilities, params):
    """
    Plot the results of the DFSV simulation.
    
    Args:
        returns: Simulated returns array with shape (T, N)
        factors: Simulated latent factors array with shape (T, K)
        volatilities: Calculated volatilities from log-volatilities with shape (T, K)
        params: The DFSV model parameters
    """
    T = returns.shape[0]
    time_axis = np.arange(T)
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Plot factor
    axes[0].plot(time_axis, factors[:, 0], 'b-', label='Latent Factor')
    axes[0].set_title('Latent Factor Process')
    axes[0].legend()
    
    # Plot factor volatility
    axes[1].plot(time_axis, volatilities[:, 0], 'r-', label='Factor Volatility')
    axes[1].set_title('Factor Volatility')
    axes[1].legend()
    
    # Plot returns for each series
    for i in range(params.N):
        axes[2].plot(time_axis, returns[:, i], label=f'Return Series {i+1}')
    axes[2].set_title('Observed Return Series')
    axes[2].legend()
    
    # Plot returns with scaling by loadings for comparison
    for i in range(params.N):
        # Scale return by its loading to show relationship with factor
        scaled_return = returns[:, i] / params.lambda_r[i, 0]
        axes[3].plot(time_axis, scaled_return, alpha=0.5, label=f'Scaled Return {i+1}')
    axes[3].plot(time_axis, factors[:, 0], 'k-', alpha=0.7, label='Factor')
    axes[3].set_title('Factor vs Scaled Returns')
    axes[3].legend()
    
    # Add x-label to bottom subplot
    axes[3].set_xlabel('Time')
    
    # Adjust layout and save figure
    plt.tight_layout()
    output_dir = Path('../outputs')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'simple_dfsv_example.png', dpi=300)
    print(f"Figure saved to {output_dir / 'simple_dfsv_example.png'}")
    plt.show()


if __name__ == "__main__":
    # Run the example
    returns, factors, log_vols = run_simulation_example()