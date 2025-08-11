#!/usr/bin/env python
"""
Test script for the minimize_with_logging function with DFSV model and BIF filter.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import time

# Set random seed for reproducibility
jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(42)

# Import DFSV model and filter components
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.filters.objectives import bellman_objective
from bellman_filter_dfsv.utils.transformations import transform_params, untransform_params
from bellman_filter_dfsv.utils.optimization import minimize_with_logging
from bellman_filter_dfsv.utils.solvers import create_optimizer


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


def create_initial_params(true_params, data_variance, perturbation=0.2):
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


def plot_parameter_history(param_history, true_params, output_dir):
    """Plot the parameter history during optimization."""
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


def test_minimize_with_logging_dfsv():
    """Test the minimize_with_logging function with DFSV model and BIF filter."""
    # Create output directory
    output_dir = Path("outputs/test_minimize_with_logging_dfsv")
    os.makedirs(output_dir, exist_ok=True)

    # Create a simple model
    true_params = create_simple_model()

    # Generate training data
    T = 500  # Use a shorter time series for testing
    returns, factors, log_vols = create_training_data(true_params, T=T, seed=42)

    # Create a BIF filter
    filter_instance = DFSVBellmanInformationFilter(true_params.N, true_params.K)

    # Create initial parameters
    data_variance = jnp.var(returns, axis=0)
    initial_params = create_initial_params(true_params, data_variance)

    # Define objective function
    def objective_fn(params, observations):
        return bellman_objective(params, observations, filter_instance), None

    # Create optimizer
    optimizer = create_optimizer(
        optimizer_name="BFGS",
        learning_rate=1e-3,
        rtol=1e-4,
        atol=1e-4,
        verbose=True
    )

    # Print initial loss
    initial_loss = objective_fn(initial_params, returns)[0]
    print(f"Initial loss: {initial_loss:.4f}")

    # Run optimization with logging
    print("Starting optimization...")
    start_time = time.time()

    sol, param_history = minimize_with_logging(
        objective_fn=objective_fn,
        initial_params=initial_params,
        solver=optimizer,
        static_args=returns,
        max_steps=50,  # Use fewer steps for testing
        log_interval=1,
        options={}
    )

    end_time = time.time()

    # Print results
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    print(f"Final loss: {objective_fn(sol.value, returns)[0]:.4f}")
    print(f"Number of iterations: {len(param_history) - 1}")
    print(f"Result: {sol.result}")

    # Plot parameter history
    plot_parameter_history(param_history, true_params, output_dir)

    # Compare final parameters with true parameters
    final_params = sol.value

    print("\nParameter comparison:")
    print(f"Lambda_r: True={true_params.lambda_r[0, 0]:.4f}, Estimated={final_params.lambda_r[0, 0]:.4f}")
    print(f"Phi_f: True={true_params.Phi_f[0, 0]:.4f}, Estimated={final_params.Phi_f[0, 0]:.4f}")
    print(f"Phi_h: True={true_params.Phi_h[0, 0]:.4f}, Estimated={final_params.Phi_h[0, 0]:.4f}")
    print(f"Mu: True={true_params.mu[0]:.4f}, Estimated={final_params.mu[0]:.4f}")
    print(f"Sigma2: True={true_params.sigma2[0]:.4f}, Estimated={final_params.sigma2[0]:.4f}")
    print(f"Q_h: True={true_params.Q_h[0, 0]:.4f}, Estimated={final_params.Q_h[0, 0]:.4f}")

    return sol, param_history, true_params


if __name__ == "__main__":
    sol, param_history, true_params = test_minimize_with_logging_dfsv()
