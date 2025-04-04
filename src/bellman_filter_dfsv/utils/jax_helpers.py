"""
Helper functions for JAX operations, particularly for safe probability distributions.
"""
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.scipy.special import gammaln

# Epsilon for numerical stability
EPS = 1e-10

def safe_norm_logpdf(x: jnp.ndarray, loc: float | jnp.ndarray, scale: float | jnp.ndarray) -> jnp.ndarray:
    """
    Computes the log probability density function of the Normal distribution
    with safety checks for the scale parameter.

    Args:
        x: The value(s) at which to evaluate the log PDF.
        loc: The mean of the distribution.
        scale: The standard deviation of the distribution.

    Returns:
        The log PDF value(s), replacing NaN/Inf with a large negative number.
    """
    # Ensure scale is positive and finite
    safe_scale = jnp.maximum(scale, EPS) # Ensure positive
    safe_scale = jnp.where(jnp.isfinite(safe_scale), safe_scale, 1.0) # Replace non-finite with 1.0

    log_pdf = norm.logpdf(x, loc=loc, scale=safe_scale)

    # Replace NaN/Inf resulting from extreme inputs or calculations
    safe_log_pdf = jnp.where(jnp.isnan(log_pdf) | jnp.isinf(log_pdf), -1e10, log_pdf)
    return safe_log_pdf

def safe_inverse_gamma_logpdf(x: jnp.ndarray, alpha: float, beta: float) -> jnp.ndarray:
    """
    Computes the log probability density function of the Inverse-Gamma distribution
    with safety checks for the input x.

    Args:
        x: The value(s) at which to evaluate the log PDF (must be positive).
        alpha: The shape parameter (must be positive).
        beta: The scale parameter (must be positive).

    Returns:
        The log PDF value(s), replacing NaN/Inf with a large negative number.
    """
    # Ensure x is positive
    x_safe = jnp.maximum(x, EPS)

    # Ensure alpha and beta are positive (although typically static)
    alpha_safe = jnp.maximum(alpha, EPS)
    beta_safe = jnp.maximum(beta, EPS)

    # Calculate log PDF using safe inputs
    log_pdf = alpha_safe * jnp.log(beta_safe) - gammaln(alpha_safe) - (alpha_safe + 1) * jnp.log(x_safe) - beta_safe / x_safe

    # Replace NaN/Inf resulting from extreme inputs or calculations
    safe_log_pdf = jnp.where(jnp.isnan(log_pdf) | jnp.isinf(log_pdf), -1e10, log_pdf)
    return safe_log_pdf