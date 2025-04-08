#!/usr/bin/env python
"""
Test script to verify that all available optimizers work with the minimize_with_logging function.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import time
import pandas as pd

# Set random seed for reproducibility
jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(42)

# Import DFSV model and filter components
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.filters.objectives import bellman_objective
from bellman_filter_dfsv.utils.optimization import minimize_with_logging
from bellman_filter_dfsv.utils.solvers import create_optimizer, get_available_optimizers


def create_simple_model():
    """Create a simple DFSV model with one factor."""
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
    Q_h = np.array([[0.1]])

    # Create parameter object using the standard dataclass
    params = DFSVParamsDataclass(
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


def create_training_data(params, T=500, seed=None):
    """Generate training data from the model."""
    # Simulate data
    returns, factors, log_vols = simulate_DFSV(
        params=params,
        T=T,
        seed=seed
    )

    return returns, factors, log_vols


def create_initial_params(true_params, data_variance):
    """Create initial parameters for optimization."""
    N, K = true_params.N, true_params.K
    
    # Create initial parameters with some perturbation from true values
    initial_params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=0.5 * jnp.ones((N, K)),  # Moderate positive loadings
        Phi_f=0.8 * jnp.eye(K),  # Moderate persistence
        Phi_h=0.8 * jnp.eye(K),  # Moderate persistence
        mu=jnp.zeros(K),  # Zero mean for log volatility
        sigma2=0.5 * data_variance,  # Provide as 1D array (variances)
        Q_h=0.2 * jnp.eye(K)  # Moderate volatility of volatility
    )
    
    return initial_params


def test_all_optimizers():
    """Test all available optimizers with the minimize_with_logging function."""
    # Create output directory
    output_dir = Path("outputs/test_all_optimizers_with_logging")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple model
    true_params = create_simple_model()
    
    # Generate training data (very small dataset for quick testing)
    T = 100  # Use a very short time series for testing
    returns, _, _ = create_training_data(true_params, T=T, seed=42)
    
    # Create a BIF filter
    filter_instance = DFSVBellmanInformationFilter(true_params.N, true_params.K)
    
    # Create initial parameters
    data_variance = jnp.var(returns, axis=0)
    initial_params = create_initial_params(true_params, data_variance)
    
    # Define objective function
    def objective_fn(params, observations):
        return bellman_objective(params, observations, filter_instance), None
    
    # Get all available optimizers
    optimizer_names = list(get_available_optimizers().keys())
    
    # Results storage
    results = []
    
    # Test each optimizer
    for optimizer_name in optimizer_names:
        print(f"\nTesting optimizer: {optimizer_name}")
        
        try:
            # Create optimizer with very small step size and loose tolerances
            optimizer = create_optimizer(
                optimizer_name=optimizer_name,
                learning_rate=1e-5,  # Very small learning rate
                rtol=1e-2,           # Loose tolerance
                atol=1e-2,           # Loose tolerance
                verbose=True
            )
            
            # Run optimization with logging for just a few steps
            start_time = time.time()
            
            sol, param_history = minimize_with_logging(
                objective_fn=objective_fn,
                initial_params=initial_params,
                solver=optimizer,
                static_args=returns,
                max_steps=3,  # Just a few steps to test functionality
                log_interval=1,
                options={},
                throw=False  # Don't throw exceptions
            )
            
            end_time = time.time()
            
            # Record results
            results.append({
                'optimizer': optimizer_name,
                'success': True,
                'error': None,
                'time': end_time - start_time,
                'steps': len(param_history) - 1,
                'result': str(sol.result)
            })
            
            print(f"  Success: {True}")
            print(f"  Time: {end_time - start_time:.2f} seconds")
            print(f"  Steps: {len(param_history) - 1}")
            print(f"  Result: {sol.result}")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            results.append({
                'optimizer': optimizer_name,
                'success': False,
                'error': str(e),
                'time': None,
                'steps': None,
                'result': None
            })
    
    # Create a DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(output_dir / "optimizer_results.csv", index=False)
    
    # Print summary
    print("\nSummary:")
    print(f"Total optimizers tested: {len(optimizer_names)}")
    print(f"Successful: {results_df['success'].sum()}")
    print(f"Failed: {len(optimizer_names) - results_df['success'].sum()}")
    
    return results_df


if __name__ == "__main__":
    results_df = test_all_optimizers()
    print("\nResults DataFrame:")
    print(results_df)
