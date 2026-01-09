"""Simulation utilities for Dynamic Factor Stochastic Volatility models.

This module provides utilities to simulate DFSV models using JAX for the v2 architecture.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from .types import DFSVParams


def simulate_dfsv(
    params: DFSVParams,
    T: int,
    *,
    key: PRNGKeyArray | int = 42,
    f0: Float[Array, "K"] | None = None,
    h0: Float[Array, "K"] | None = None,
) -> tuple[Float[Array, "T N"], Float[Array, "T K"], Float[Array, "T K"]]:
    """Simulate a Dynamic Factor Stochastic Volatility model.

    The DFSV model consists of:
        Observation:  r_t = λ_r f_t + e_t,  e_t ~ N(0, Σ)
        Factor:       f_t = Φ_f f_{t-1} + diag(exp(h_t/2)) ε_t,  ε_t ~ N(0, I_K)
        Log-vol:      h_t = μ + Φ_h (h_{t-1} - μ) + η_t,  η_t ~ N(0, Q_h)

    Args:
        params: DFSV model parameters.
        T: Number of time steps to simulate.
        key: JAX random key or integer seed. Default: 42.
        f0: Initial factors (K,). If None, defaults to zeros.
        h0: Initial log-volatilities (K,). If None, defaults to long-run mean μ.

    Returns:
        Tuple of (returns, factors, log_vols):
            - returns: Simulated returns (T, N)
            - factors: Simulated latent factors (T, K)
            - log_vols: Simulated log-volatilities (T, K)

    Example:
        >>> import jax.numpy as jnp
        >>> from bellman_filter_dfsv import DFSVParams, simulate_dfsv
        >>> params = DFSVParams(
        ...     lambda_r=jnp.array([[0.8], [0.7]]),
        ...     Phi_f=jnp.array([[0.7]]),
        ...     Phi_h=jnp.array([[0.95]]),
        ...     mu=jnp.array([-1.2]),
        ...     sigma2=jnp.array([0.3, 0.25]),
        ...     Q_h=jnp.array([[0.01]])
        ... )
        >>> returns, factors, log_vols = simulate_dfsv(params, T=100, key=42)
        >>> returns.shape
        (100, 2)
    """
    # Handle key
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    # Extract dimensions
    N, K = params.lambda_r.shape

    # Initialize states
    if f0 is None:
        f0 = jnp.zeros(K)
    if h0 is None:
        h0 = params.mu

    # Prepare covariance matrices
    Sigma = jnp.diag(params.sigma2)  # Idiosyncratic covariance (N, N)
    chol_Sigma = jnp.linalg.cholesky(Sigma)  # Lower triangular
    chol_Q_h = jnp.linalg.cholesky(params.Q_h)  # Lower triangular

    # Split key for random draws
    key_h, key_f, key_r = jr.split(key, 3)

    # Generate all random shocks at once
    eta = jr.normal(key_h, (T, K))  # Log-vol innovations
    eps = jr.normal(key_f, (T, K))  # Factor innovations
    e = jr.normal(key_r, (T, N))  # Observation noise

    def step(carry, t):
        """Single simulation step."""
        f_prev, h_prev = carry

        # Log-volatility transition: h_t = μ + Φ_h(h_{t-1} - μ) + η_t
        h_t = params.mu + params.Phi_h @ (h_prev - params.mu) + chol_Q_h @ eta[t]

        # Factor transition: f_t = Φ_f f_{t-1} + diag(exp(h_t/2)) ε_t
        vol_scale = jnp.exp(h_t / 2.0)
        f_t = params.Phi_f @ f_prev + vol_scale * eps[t]

        # Observation: r_t = λ_r f_t + e_t
        r_t = params.lambda_r @ f_t + chol_Sigma @ e[t]

        return (f_t, h_t), (r_t, f_t, h_t)

    # Run simulation
    _, (returns, factors, log_vols) = jax.lax.scan(step, (f0, h0), jnp.arange(T))

    return returns, factors, log_vols


# Alias for backwards compatibility with examples
simulate_DFSV = simulate_dfsv
