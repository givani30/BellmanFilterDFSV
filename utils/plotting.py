"""
Plotting utilities for DFSV model analysis.
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
# Need to import the parameter class - adjust path as needed
# Assuming utils is at the same level as functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.jax_params import DFSVParamsDataclass

def calculate_return_variance(params: DFSVParamsDataclass, log_vols: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the variance of returns based on model parameters and log volatility.
    
    Parameters:
    -----------
    params : DFSVParamsDataclass
        Model parameters
    log_vols : np.ndarray
        Log volatility values
        
    Returns:
    --------
    np.ndarray
        Time series of covariance matrices
    """
    N, K = params.N, params.K
    lambda_r = params.lambda_r  # NxK
    sigma2 = params.sigma2      # NxN (should be diagonal)
    
    # Ensure log_vols is 2D
    if log_vols.ndim == 1:
        log_vols = log_vols.reshape(-1, 1)
    
    T = log_vols.shape[0]
    
    # Initialize result array to store covariance matrices
    result = jnp.zeros((T, N, N))
    
    # Convert log-volatility to volatility
    vol = jnp.exp(log_vols)  # TxK
    
    # Calculate covariances for each time point
    # Use vmap for potential speedup if needed, but loop is clearer for now
    for t in range(T):
        # Create the diagonal volatility matrix (KxK)
        vol_diag_t = jnp.diag(vol[t])
        
        # Calculate the factor-driven part of covariance: lambda_r * vol_diag_t * lambda_r.T
        factor_cov = lambda_r @ vol_diag_t @ lambda_r.T
        
        # Add idiosyncratic variance (should be diagonal matrix)
        total_cov = factor_cov + sigma2 # sigma2 is assumed NxN diagonal
        
        result = result.at[t].set(total_cov)
    
    return result

def plot_variance_comparison(true_params: DFSVParamsDataclass, 
                           standard_params: DFSVParamsDataclass, 
                           transformed_params: DFSVParamsDataclass, 
                           true_log_vols: jnp.ndarray, 
                           standard_log_vols: jnp.ndarray, 
                           transformed_log_vols: jnp.ndarray,
                           returns: jnp.ndarray, 
                           save_path: str = None):
    """
    Plot comparison of predicted vs true return variances.
    
    Parameters:
    -----------
    true_params, standard_params, transformed_params : DFSVParamsDataclass
        Parameter sets to compare
    true_log_vols, standard_log_vols, transformed_log_vols : np.ndarray
        Log volatility estimates from different methods
    returns : np.ndarray
        Observed returns (Used only for dimension reference N)
    save_path : str, optional
        Path to save the plot
    """
    
    # Calculate variances
    true_var = calculate_return_variance(true_params, true_log_vols)
    standard_var = calculate_return_variance(standard_params, standard_log_vols)
    transformed_var = calculate_return_variance(transformed_params, transformed_log_vols)
    
    # Extract diagonal elements (individual asset variances)
    true_diag_var = jnp.diagonal(true_var, axis1=1, axis2=2)
    standard_diag_var = jnp.diagonal(standard_var, axis1=1, axis2=2)
    transformed_diag_var = jnp.diagonal(transformed_var, axis1=1, axis2=2)
    
    # Plot the variance comparison
    N = true_params.N
    fig, axes = plt.subplots(N, 1, figsize=(15, 4*N), sharex=True)
    if N == 1:
        axes = [axes] # Ensure axes is always iterable

    for i in range(N):
        ax = axes[i]
        
        # Plot predicted variances
        ax.plot(true_diag_var[:, i], 'k-', alpha=0.7, label='True Model Variance')
        ax.plot(standard_diag_var[:, i], 'b--', label='Standard Opt. Variance')
        ax.plot(transformed_diag_var[:, i], 'r:', label='Transformed Opt. Variance')
        
        ax.set_title(f'Asset {i+1} Predicted Return Variance')
        ax.set_ylabel('Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.xlabel('Time')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved variance comparison plot to {save_path}")
    plt.show()
    
    # Calculate mean squared error between predicted and true variance
    mse_standard = jnp.mean((standard_diag_var - true_diag_var)**2)
    mse_transformed = jnp.mean((transformed_diag_var - true_diag_var)**2)
    
    print("\nMean squared error between predicted and true variance:")
    print(f"Standard optimization vs True: {mse_standard:.6f}")
    print(f"Transformed optimization vs True: {mse_transformed:.6f}")
    
    # Return calculated variances if needed elsewhere
    return true_var, standard_var, transformed_var
