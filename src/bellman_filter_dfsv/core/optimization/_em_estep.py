"""E-step computations for EM algorithm on DFSV models."""

import jax.numpy as jnp
from jaxtyping import Array, Float


def compute_exp_neg_h(
    h_mean: Float[Array, "K"],
    h_var: Float[Array, "K"],
    max_var: float = 4.0,
) -> Float[Array, "K"]:
    """
    Compute E[exp(-h)] for h ~ N(h_mean, h_var).

    Uses log-normal moment formula: E[exp(a*X)] = exp(a*μ + a²σ²/2) for X ~ N(μ,σ²).
    With a = -1: E[exp(-h)] = exp(-μ + σ²/2).

    Args:
        h_mean: Posterior mean of h, shape (K,)
        h_var: Posterior variance of h (diagonal), shape (K,)
        max_var: Cap on variance to prevent numerical overflow

    Returns:
        E[exp(-h_k)] for each factor k, shape (K,)
    """
    h_var_capped = jnp.minimum(h_var, max_var)
    return jnp.exp(-h_mean + 0.5 * h_var_capped)
