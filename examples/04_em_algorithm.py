#!/usr/bin/env python
"""
EM Algorithm Example for DFSV Models

This example demonstrates the EM algorithm for parameter estimation.

v2.0.0 - Using fit_em with Rao-Blackwellized Particle Smoother.
"""

import jax.numpy as jnp
import numpy as np

from bellman_filter_dfsv import DFSVParams, fit_em, simulate_dfsv


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


def main():
    """Run EM algorithm example."""
    print("EM Algorithm Example (v2.0.0)")
    print("==============================")

    # Create true model
    N, K = 3, 1
    true_params = create_simple_dfsv_model(N, K)
    print(f"Created DFSV model with {N} observed series and {K} factors")

    # Simulate data
    T = 300
    print(f"Simulating {T} observations...")
    returns, _, _ = simulate_dfsv(true_params, T=T, key=42)
    print("Simulation complete!")

    # Create initial guess
    init_params = true_params._replace(
        lambda_r=true_params.lambda_r * 0.8,
        sigma2=true_params.sigma2 * 1.2,
    )

    print("\nStarting EM algorithm...")

    # Run EM estimation
    estimated_params = fit_em(
        start_params=init_params,
        observations=returns,
        num_em_steps=10,
        num_particles=500,
        num_trajectories=50,
    )

    print("\nEstimation complete!")

    # Compare parameters
    print("\nParameter Comparison:")
    print("Lambda_r (factor loadings):")
    print(f"  True: {np.array(true_params.lambda_r).flatten()}")
    print(f"  Est:  {np.array(estimated_params.lambda_r).flatten()}")
    print("\nSigma2 (idiosyncratic variances):")
    print(f"  True: {np.array(true_params.sigma2)}")
    print(f"  Est:  {np.array(estimated_params.sigma2)}")

    return true_params, estimated_params


if __name__ == "__main__":
    main()
