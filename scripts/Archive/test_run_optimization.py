#!/usr/bin/env python
"""
Test script for the run_optimization function.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import time
import pandas as pd


# Import DFSV model and filter components
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.utils.optimization import run_optimization, FilterType
from bellman_filter_dfsv.utils.transformations import apply_identification_constraint

# Set random seed for reproducibility
jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(42)

def create_simple_model():
    """Create a simple DFSV model with one factor."""
    # Define model dimensions
    N = 3  # Number of observed series
    K = 1  # Number of factors

    # Factor loadings
    lambda_r = np.array([[0.9], [0.6], [0.3]])

    # Factor persistence
    Phi_f = np.array([[0.15]])

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
    #apply_identification_constraint(params)
    # Ensure constraint is applied correctly
    params = apply_identification_constraint(params)
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


def plot_parameter_history(result, true_params, output_dir):
    """Plot the parameter history during optimization."""
    # Extract parameter values over iterations
    param_history = result.param_history
    loss_history = result.loss_history
    
    # Extract parameter values over iterations
    lambda_r_history = jnp.array([p.lambda_r[0, 0] for p in param_history])
    phi_f_history = jnp.array([p.Phi_f[0, 0] for p in param_history])
    phi_h_history = jnp.array([p.Phi_h[0, 0] for p in param_history])
    mu_history = jnp.array([p.mu[0] for p in param_history])
    sigma2_history = jnp.array([p.sigma2[0] for p in param_history])
    q_h_history = jnp.array([p.Q_h[0, 0] for p in param_history])
    
    # True parameter values
    true_lambda_r = true_params.lambda_r[0, 0]
    true_phi_f = true_params.Phi_f[0, 0]
    true_phi_h = true_params.Phi_h[0, 0]
    true_mu = true_params.mu[0]
    true_sigma2 = true_params.sigma2[0]
    true_q_h = true_params.Q_h[0, 0]
    
    # Create plots
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Lambda_r
    axes[0, 0].plot(lambda_r_history, label='Estimated')
    axes[0, 0].axhline(y=true_lambda_r, color='r', linestyle='--', label='True')
    axes[0, 0].set_title('Factor Loading (λ)')
    axes[0, 0].legend()
    
    # Phi_f
    axes[0, 1].plot(phi_f_history, label='Estimated')
    axes[0, 1].axhline(y=true_phi_f, color='r', linestyle='--', label='True')
    axes[0, 1].set_title('Factor Persistence (Φ_f)')
    axes[0, 1].legend()
    
    # Phi_h
    axes[1, 0].plot(phi_h_history, label='Estimated')
    axes[1, 0].axhline(y=true_phi_h, color='r', linestyle='--', label='True')
    axes[1, 0].set_title('Log-Volatility Persistence (Φ_h)')
    axes[1, 0].legend()
    
    # Mu
    axes[1, 1].plot(mu_history, label='Estimated')
    axes[1, 1].axhline(y=true_mu, color='r', linestyle='--', label='True')
    axes[1, 1].set_title('Long-run Mean (μ)')
    axes[1, 1].legend()
    
    # Sigma2
    axes[2, 0].plot(sigma2_history, label='Estimated')
    axes[2, 0].axhline(y=true_sigma2, color='r', linestyle='--', label='True')
    axes[2, 0].set_title('Idiosyncratic Variance (σ²)')
    axes[2, 0].legend()
    
    # Q_h
    axes[2, 1].plot(q_h_history, label='Estimated')
    axes[2, 1].axhline(y=true_q_h, color='r', linestyle='--', label='True')
    axes[2, 1].set_title('Log-Volatility Noise Variance (Q_h)')
    axes[2, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_history.png")
    plt.close()
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Loss History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    # plt.yscale('log')
    plt.grid(True)
    plt.savefig(output_dir / "loss_history.png")
    plt.close()


def test_run_optimization():
    """Test the run_optimization function with different configurations."""
    # Create output directory
    output_dir = Path("outputs/test_run_optimization")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple model
    true_params = create_simple_model()
    
    # Generate training data
    T = 1000  # Use a shorter time series for testing
    returns, _, _ = create_training_data(true_params, T=T, seed=42)
    
    # Define configurations to test
    configs = [
        {
            "name": "BIF_BFGS_transformed",
            "filter_type": FilterType.BIF,
            "optimizer_name": "BFGS",
            "use_transformations": True,
            "fix_mu": False,
            "max_steps": 20
        },
        {
            "name": "BIF_Adam_transformed",
            "filter_type": FilterType.BIF,
            "optimizer_name": "Adam",
            "use_transformations": True,
            "fix_mu": False,
            "max_steps": 20
        },
        {
            "name": "BF_BFGS_transformed",
            "filter_type": FilterType.BF,
            "optimizer_name": "BFGS",
            "use_transformations": True,
            "fix_mu": False,
            "max_steps": 20
        }
    ]
    
    # Results storage
    results = []
    
    # Test each configuration
    for config in configs:
        print(f"\nTesting configuration: {config['name']}")
        
        try:
            # Run optimization
            result = run_optimization(
                filter_type=config["filter_type"],
                returns=returns,
                true_params=true_params if config["fix_mu"] else None,
                use_transformations=config["use_transformations"],
                optimizer_name=config["optimizer_name"],
                max_steps=config["max_steps"],
                verbose=True
            )
            
            # Create config-specific output directory
            config_dir = output_dir / config["name"]
            os.makedirs(config_dir, exist_ok=True)
            
            # Plot parameter history
            plot_parameter_history(result, true_params, config_dir)
            
            # Record results
            results.append({
                "name": config["name"],
                "filter_type": str(result.filter_type),
                "optimizer_name": result.optimizer_name,
                "uses_transformations": result.uses_transformations,
                "fix_mu": result.fix_mu,
                "success": result.success,
                "final_loss": result.final_loss,
                "steps": result.steps,
                "time_taken": result.time_taken,
                "error_message": result.error_message,
                "lambda_r": result.final_params.lambda_r[0, 0],
                "phi_f": result.final_params.Phi_f[0, 0],
                "phi_h": result.final_params.Phi_h[0, 0],
                "mu": result.final_params.mu[0],
                "sigma2": result.final_params.sigma2[0],
                "q_h": result.final_params.Q_h[0, 0]
            })
            
            # Print results
            print(f"  Success: {result.success}")
            print(f"  Final loss: {result.final_loss:.4f}")
            print(f"  Steps: {result.steps}")
            print(f"  Time taken: {result.time_taken:.2f} seconds")
            if result.error_message:
                print(f"  Error: {result.error_message}")
            
            # Print parameter comparison
            print("\nParameter comparison:")
            print(f"  Lambda_r: True={true_params.lambda_r[0, 0]:.4f}, Estimated={result.final_params.lambda_r[0, 0]:.4f}")
            print(f"  Phi_f: True={true_params.Phi_f[0, 0]:.4f}, Estimated={result.final_params.Phi_f[0, 0]:.4f}")
            print(f"  Phi_h: True={true_params.Phi_h[0, 0]:.4f}, Estimated={result.final_params.Phi_h[0, 0]:.4f}")
            print(f"  Mu: True={true_params.mu[0]:.4f}, Estimated={result.final_params.mu[0]:.4f}")
            print(f"  Sigma2: True={true_params.sigma2[0]:.4f}, Estimated={result.final_params.sigma2[0]:.4f}")
            print(f"  Q_h: True={true_params.Q_h[0, 0]:.4f}, Estimated={result.final_params.Q_h[0, 0]:.4f}")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            results.append({
                "name": config["name"],
                "filter_type": str(config["filter_type"]),
                "optimizer_name": config["optimizer_name"],
                "uses_transformations": config["use_transformations"],
                "fix_mu": config["fix_mu"],
                "success": False,
                "final_loss": float('inf'),
                "steps": 0,
                "time_taken": 0,
                "error_message": str(e),
                "lambda_r": None,
                "phi_f": None,
                "phi_h": None,
                "mu": None,
                "sigma2": None,
                "q_h": None
            })
    
    # Create a DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(output_dir / "optimization_results.csv", index=False)
    
    # Print summary
    print("\nSummary:")
    print(f"Total configurations tested: {len(configs)}")
    print(f"Successful: {results_df['success'].sum()}")
    print(f"Failed: {len(configs) - results_df['success'].sum()}")
    
    return results_df


if __name__ == "__main__":
    results_df = test_run_optimization()
    print("\nResults DataFrame:")
    print(results_df[["name", "success", "final_loss", "steps", "time_taken"]])
