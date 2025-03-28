#!/usr/bin/env python
"""
Script to examine the log-likelihood surface around current parameter values.

This helps us understand why the gradients are larger than expected.
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

from functions.simulation import DFSV_params, simulate_DFSV
from functions.jax_params import DFSVParamsDataclass, dfsv_params_to_dict
from functions.bellman_filter import DFSVBellmanFilter

def create_test_model():
    """Create a simple test model."""
    # Define model dimensions
    N = 3  # Number of observed series
    K = 1  # Number of factors

    # Factor loadings
    lambda_r = np.array([[0.9], [0.6], [0.3]])

    # Factor persistence
    Phi_f = np.array([[0.95]])

    # Log-volatility persistence
    Phi_h = np.array([[0.98]])

    # Long-run mean for log-volatilities
    mu = np.array([-1.0])
    
    # Idiosyncratic variance (diagonal)
    sigma2 = np.array([0.1, 0.1, 0.1])

    # Log-volatility noise covariance
    Q_h = np.array([[0.05]])

    # Create parameter object
    params = DFSV_params(
        N=N,
        K=K,
        lambda_r=lambda_r,
        Phi_f=Phi_f,
        Phi_h=Phi_h,
        mu=mu,
        sigma2=sigma2,
        Q_h=Q_h,
    )

    return params

def examine_likelihood_surface():
    """Examine the likelihood surface around current parameter values."""
    print("Examining log-likelihood surface...")
    
    # Create model and data
    params = create_test_model()
    T = 50  # Shorter time series for faster testing
    returns, factors, log_vols = simulate_DFSV(params, T=T, seed=42)
    
    # Create filter instance
    bf = DFSVBellmanFilter(params.N, params.K)
    
    # Convert to dictionary format
    jax_params = DFSVParamsDataclass.from_dfsv_params(params)
    param_dict, N_value, K_value = dfsv_params_to_dict(jax_params)
    
    # Define the objective function (negative log-likelihood)
    def objective(complete_params):
        return -bf.jit_log_likelihood_of_params(complete_params, returns)
    
    # Parameters to examine
    param_names = ['Phi_h', 'Phi_f', 'mu', 'Q_h']
    param_ranges = {
        'Phi_h': np.linspace(0.5, 0.999, 20),  # Range for Phi_h
        'Phi_f': np.linspace(0.5, 0.999, 20),  # Range for Phi_f
        'mu': np.linspace(-2.0, 0.0, 20),      # Range for mu
        'Q_h': np.linspace(0.01, 0.2, 20)      # Range for Q_h
    }
    
    # Base parameter set with N and K
    base_params = copy.deepcopy(param_dict)
    base_params['N'] = N_value
    base_params['K'] = K_value
    
    # Base objective value
    base_objective = objective(base_params)
    print(f"Base negative log-likelihood: {base_objective:.4f}")
    
    plt.figure(figsize=(20, 15))
    
    # For each parameter, plot likelihood surface
    for i, param_name in enumerate(param_names):
        print(f"Varying {param_name}...")
        values = param_ranges[param_name]
        base_value = base_params[param_name][0][0] if param_name != 'mu' else base_params[param_name][0]
        likelihoods = []
        
        for val in values:
            test_params = copy.deepcopy(base_params)
            # Handle different parameter shapes
            if param_name == 'mu':
                test_params[param_name] = np.array([val])
            else:
                test_params[param_name] = np.array([[val]])
            
            try:
                neg_ll = objective(test_params)
                likelihoods.append(neg_ll)
            except Exception as e:
                print(f"Error at {param_name}={val}: {e}")
                likelihoods.append(np.nan)
        
        plt.subplot(2, 2, i+1)
        
        # Plot negative log-likelihood
        plt.plot(values, likelihoods, 'b-', linewidth=2)
        plt.axvline(x=base_value, color='r', linestyle='--', 
                   label=f'Base value: {base_value:.4f}')
        
        # Mark minimum
        valid_idxs = ~np.isnan(likelihoods)
        if np.any(valid_idxs):
            min_idx = np.argmin(np.array(likelihoods)[valid_idxs])
            min_value = values[valid_idxs][min_idx]
            min_ll = np.min(np.array(likelihoods)[valid_idxs])
            plt.axvline(x=min_value, color='g', linestyle='-.',
                       label=f'Min at: {min_value:.4f}')
            plt.text(min_value, min_ll, f"Min: {min_ll:.2f}", 
                    fontsize=10, horizontalalignment='left')
        
        plt.grid(True, alpha=0.3)
        plt.xlabel(param_name)
        plt.ylabel('Negative Log-Likelihood')
        plt.title(f'Likelihood Surface for {param_name}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('likelihood_surface.png')
    print("Likelihood surface visualization saved as 'likelihood_surface.png'")

if __name__ == "__main__":
    examine_likelihood_surface()