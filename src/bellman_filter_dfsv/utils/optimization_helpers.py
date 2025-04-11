"""Optimization helpers for DFSV models.

This module provides functions for generating stable initial parameters for optimization.
"""

import jax
import jax.numpy as jnp
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass

def create_stable_initial_params(N: int, K: int) -> DFSVParamsDataclass:
    """Create stable initial parameter values for optimization.

    Ensures that persistence matrices Phi_f and Phi_h have eigenvalues strictly within the unit circle.

    Args:
        N: Number of observed series.
        K: Number of factors.

    Returns:
        DFSVParamsDataclass: Initial parameter values with stable Phi_f and Phi_h.
    """
    # Lower triangular lambda_r with diagonal fixed to 1
    lambda_r_init = jnp.zeros((N, K))
    diag_indices = jnp.diag_indices(min(N, K))
    lambda_r_init = lambda_r_init.at[diag_indices].set(1.0)
    lambda_r_init = jnp.tril(lambda_r_init)

    # Factor persistence (diagonal-dominant, stable)
    phi_f_init = 0.3 * jnp.eye(K) + 0.05 * jnp.ones((K, K))
    phi_f_init = phi_f_init / jnp.linalg.norm(phi_f_init, ord=2) * 0.999

    # Log-volatility persistence (diagonal-dominant, stable)
    phi_h_init = 0.8 * jnp.eye(K) + 0.02 * jnp.ones((K, K))
    phi_h_init = phi_h_init / jnp.linalg.norm(phi_h_init, ord=2) * 0.999

    initial_params = DFSVParamsDataclass(
        N=N, K=K,
        lambda_r=lambda_r_init,
        Phi_f=phi_f_init,
        Phi_h=phi_h_init,
        mu=jnp.zeros(K),
        sigma2=0.1 * jnp.ones(N),
        Q_h=0.2 * jnp.eye(K)
    )

    return initial_params