"""Simulation helpers for DFSV models.

This module provides functions for generating stable DFSV model parameters for simulation and testing.
"""

import jax
import jax.numpy as jnp
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.utils.transformations import apply_identification_constraint

def create_stable_dfsv_params(N: int = 3, K: int = 2) -> DFSVParamsDataclass:
    """Create a stable DFSV model parameter set.

    Ensures that persistence matrices Phi_f and Phi_h have eigenvalues strictly within the unit circle.

    Args:
        N: Number of observed series.
        K: Number of factors.

    Returns:
        DFSVParamsDataclass: Model parameters with stable Phi_f and Phi_h.
    """
    key = jax.random.PRNGKey(42)
    key, subkey1 = jax.random.split(key)

    # Factor loadings (lower triangular with diagonal fixed to 1)
    lambda_r_init = jax.random.normal(subkey1, (N, K)) * 0.5 + 0.5
    lambda_r = jnp.tril(lambda_r_init)
    diag_indices = jnp.diag_indices(n=min(N, K), ndim=2)
    lambda_r = lambda_r.at[diag_indices].set(1.0)

    # Factor persistence (diagonal-dominant, stable)
    key, subkey1 = jax.random.split(key)
    Phi_f = jax.random.uniform(subkey1, (K, K), minval=0.01, maxval=0.1)
    key, subkey1 = jax.random.split(key)
    diag_values = jax.random.uniform(subkey1, (K,), minval=0.15, maxval=0.35)
    Phi_f = Phi_f.at[jnp.diag_indices(K)].set(diag_values)
    # # Normalize to ensure spectral norm < 1 # 
    # Phi_f = Phi_f / jnp.linalg.norm(Phi_f, ord=2) * 0.97

    # Log-volatility persistence (diagonal-dominant, stable)
    key, subkey1 = jax.random.split(key)
    Phi_h = jax.random.uniform(subkey1, (K, K), minval=0.01, maxval=0.1)
    key, subkey1 = jax.random.split(key)
    diag_values = jax.random.uniform(subkey1, (K,), minval=0.9, maxval=0.99)
    Phi_h = Phi_h.at[jnp.diag_indices(K)].set(diag_values)
    Phi_h = Phi_h / jnp.linalg.norm(Phi_h, ord=2) * 0.97

    # Long-run mean for log-volatilities
    mu = jnp.array([-1.0, -0.5] if K == 2 else [-1.0] * K)

    # Idiosyncratic variance (diagonal)
    key, subkey1 = jax.random.split(key)
    sigma2 = jax.random.uniform(subkey1, (N,), minval=0.05, maxval=0.1)

    # Log-volatility noise covariance (diagonal)
    key, subkey1 = jax.random.split(key)
    Q_h_diag = jax.random.uniform(subkey1, (K,), minval=0.8, maxval=1.0)
    Q_h = jnp.diag(Q_h_diag)

    params = DFSVParamsDataclass(
        N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, Phi_h=Phi_h,
        mu=mu, sigma2=sigma2, Q_h=Q_h
    )
    params = apply_identification_constraint(params)
    return params