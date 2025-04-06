"""
Simple DFSV Model Example with K=1 and N=3

This script demonstrates how to set up and simulate a Dynamic Factor Stochastic Volatility 
model with one factor (K=1) and three observed time series (N=3).
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp # Add JAX numpy import
# Updated imports
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass # Import the JAX dataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV

# Set random seed for reproducibility
np.random.seed(42)

def run_simple_dfsv_example():
    # Define parameters for a simple DFSV model with K=1 and N=3
    N = 3  # Number of observed time series
    K = 1  # Number of latent factors
    
    # Factor loadings (N x K)
    lambda_r = np.array([[0.8], [0.6], [0.4]])
    
    # Factor state transition matrix (K x K)
    Phi_f = np.array([[0.95]])
    
    # State transition matrix for log-volatilities (K x K)
    Phi_h = np.array([[0.98]])
    
    # Long-run mean for log-volatilities (K,)
    mu = np.array([0.0])
    
    # Idiosyncratic variance (N x N) - diagonal covariance matrix
    sigma2 = np.array([0.2, 0.3, 0.4]) # Keep as 1D array for dataclass
    
    # Noise covariance matrix for log-volatilities (K x K)
    Q_h = np.array([[0.1]])
    
    # Create DFSV parameters object
    # Create parameter dataclass object using JAX arrays
    params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=jnp.array(lambda_r),
        Phi_f=jnp.array(Phi_f),
        Phi_h=jnp.array(Phi_h),
        mu=jnp.array(mu),
        sigma2=jnp.array(sigma2), # Pass 1D array
        Q_h=jnp.array(Q_h)
    )
    
    # Number of time periods to simulate
    T = 500
    
    # Simulate DFSV model
    returns, factors, log_vols = simulate_DFSV(params, T=T, seed=42)
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Plot the latent factor
    axes[0].plot(factors, label='Latent Factor')
    axes[0].set_title('Latent Factor (K=1)')
    axes[0].legend()
    
    # Plot the log volatility
    axes[1].plot(log_vols, label='Log Volatility', color='orange')
    axes[1].set_title('Log Volatility')
    axes[1].legend()
    
    # Calculate and plot the volatility (exp(log_vol/2))
    volatility = np.exp(log_vols/2)
    axes[2].plot(volatility, label='Volatility', color='green')
    axes[2].set_title('Volatility')
    axes[2].legend()
    
    # Plot the returns for each time series
    for i in range(N):
        axes[3].plot(returns[:, i], label=f'Returns Series {i+1}', alpha=0.7)
    axes[3].set_title('Simulated Returns (N=3)')
    axes[3].legend()
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('outputs/simple_dfsv_example.png', dpi=300)
    plt.show()
    
    # Print some summary statistics
    print("Summary Statistics:")
    print(f"Returns shape: {returns.shape}")
    for i in range(N):
        print(f"Returns Series {i+1}: Mean = {returns[:, i].mean():.4f}, Std = {returns[:, i].std():.4f}")
    
    print(f"\nFactor shape: {factors.shape}")
    print(f"Factor: Mean = {factors.mean():.4f}, Std = {factors.std():.4f}")
    
    print(f"\nLog Volatility shape: {log_vols.shape}")
    print(f"Log Volatility: Mean = {log_vols.mean():.4f}, Std = {log_vols.std():.4f}")
    
    return returns, factors, log_vols

if __name__ == "__main__":
    returns, factors, log_vols = run_simple_dfsv_example()