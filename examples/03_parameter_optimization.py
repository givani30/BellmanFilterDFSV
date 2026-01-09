#!/usr/bin/env python
"""
Parameter Estimation Example for DFSV Models

This example demonstrates how to:
1. Create a DFSV model and simulate data
2. Estimate model parameters using fit_mle
3. Compare estimated parameters to true parameters
4. Visualize optimization results

v2.0.0 - Updated for new architecture using fit_mle.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from bellman_filter_dfsv import DFSVParams, fit_mle, simulate_dfsv


def create_simple_dfsv_model(N=3, K=1):
    """Create a simple DFSV model."""
    lambda_r = np.random.uniform(0.3, 0.9, size=(N, K))
    Phi_f = np.eye(K) * 0.7
    Phi_h = np.eye(K) * 0.95
    mu = np.ones(K) * -1.0
    sigma2 = np.ones(N) * 0.1
    Q_h = np.eye(K) * 0.05

    return DFSVParams(
        lambda_r=jnp.array(lambda_r),
        Phi_f=jnp.array(Phi_f),
        Phi_h=jnp.array(Phi_h),
        mu=jnp.array(mu),
        sigma2=jnp.array(sigma2),
        Q_h=jnp.array(Q_h),
    )


def perturb_parameters(params, scale=0.2):
    """Create perturbed parameters as initial guess."""
    np.random.seed(123)

    return DFSVParams(
        lambda_r=params.lambda_r + np.random.normal(0, scale, params.lambda_r.shape),
        Phi_f=params.Phi_f + np.random.normal(0, scale * 0.1, params.Phi_f.shape),
        Phi_h=params.Phi_h + np.random.normal(0, scale * 0.1, params.Phi_h.shape),
        mu=params.mu + np.random.normal(0, scale, params.mu.shape),
        sigma2=jnp.abs(params.sigma2 + np.random.normal(0, scale, params.sigma2.shape)),
        Q_h=params.Q_h + np.random.normal(0, scale * 0.1, params.Q_h.shape),
    )


def main():
    """Run parameter estimation example."""
    print("Parameter Estimation Example (v2.0.0)")
    print("=====================================")

    # Create true model
    N, K = 3, 1
    true_params = create_simple_dfsv_model(N, K)
    print(f"Created DFSV model with {N} observed series and {K} factors")

    # Simulate data
    T = 500
    print(f"Simulating {T} observations...")
    returns, _, _ = simulate_dfsv(true_params, T=T, key=42)
    print("Simulation complete!")

    # Create initial guess (perturbed parameters)
    init_params = perturb_parameters(true_params, scale=0.3)
    print("\nStarting parameter estimation...")

    # Run MLE estimation
    estimated_params, history = fit_mle(
        start_params=init_params,
        observations=returns,
        num_steps=50,
        learning_rate=0.01,
    )

    print("\nEstimation complete!")
    print(f"Initial negative log-likelihood: {history[0]:.2f}")
    print(f"Final negative log-likelihood: {history[-1]:.2f}")
    print(f"Improvement: {history[0] - history[-1]:.2f}")

    # Compare parameters
    print("\nParameter Comparison:")
    print("Lambda_r (factor loadings):")
    print(f"  True: {np.array(true_params.lambda_r).flatten()}")
    print(f"  Est:  {np.array(estimated_params.lambda_r).flatten()}")
    print(f"\nPhi_f (factor persistence):")
    print(f"  True: {np.diag(true_params.Phi_f)}")
    print(f"  Est:  {np.diag(estimated_params.Phi_f)}")
    print(f"\nPhi_h (volatility persistence):")
    print(f"  True: {np.diag(true_params.Phi_h)}")
    print(f"  Est:  {np.diag(estimated_params.Phi_h)}")

    # Plot optimization progress
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("MLE Optimization Progress")
    plt.grid(True)
    plt.show()

    return true_params, estimated_params, history


if __name__ == "__main__":
    main()
