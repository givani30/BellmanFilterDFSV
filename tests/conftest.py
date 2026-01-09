"""Pytest configuration and common fixtures for DFSV model testing - v2 architecture."""

import jax
import jax.numpy as jnp
import pytest
from hypothesis import strategies as st

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def simple_params():
    """Fixture providing simple DFSVParams for testing."""
    from bellman_filter_dfsv import DFSVParams

    return DFSVParams(
        lambda_r=jnp.array([[0.8], [0.7], [0.9]]),
        Phi_f=jnp.array([[0.7]]),
        Phi_h=jnp.array([[0.95]]),
        mu=jnp.array([-1.2]),
        sigma2=jnp.array([0.3, 0.25, 0.35]),
        Q_h=jnp.array([[0.01]]),
    )


@pytest.fixture(scope="session")
def multi_factor_params():
    """Fixture providing DFSVParams with K=2 factors for testing."""
    from bellman_filter_dfsv import DFSVParams

    return DFSVParams(
        lambda_r=jnp.array([[0.8, 0.3], [0.7, 0.4], [0.9, 0.2]]),
        Phi_f=jnp.array([[0.7, 0.0], [0.0, 0.6]]),
        Phi_h=jnp.array([[0.95, 0.0], [0.0, 0.90]]),
        mu=jnp.array([-1.2, -0.8]),
        sigma2=jnp.array([0.3, 0.25, 0.35]),
        Q_h=jnp.array([[0.01, 0.0], [0.0, 0.015]]),
    )


# === Hypothesis Strategies for Property-Based Testing ===


@st.composite
def dfsv_params_strategy(draw, N=None, K=None):
    """Generate valid DFSVParams for property-based testing.

    Args:
        draw: Hypothesis draw function.
        N: Number of observed series (if None, random between 2-5).
        K: Number of latent factors (if None, random between 1-3).

    Returns:
        DFSVParams with valid parameter values.
    """
    from bellman_filter_dfsv import DFSVParams

    # Dimension selection
    if N is None:
        N = draw(st.integers(min_value=2, max_value=5))
    if K is None:
        K = draw(st.integers(min_value=1, max_value=3))

    # Factor loadings: Unconstrained
    lambda_r = draw(
        st.lists(
            st.lists(st.floats(min_value=-2.0, max_value=2.0), min_size=K, max_size=K),
            min_size=N,
            max_size=N,
        )
    )
    lambda_r = jnp.array(lambda_r)

    # AR matrices: Stable dynamics (-0.98, 0.98)
    # For simplicity, use diagonal matrices
    phi_f_diag = draw(
        st.lists(st.floats(min_value=-0.98, max_value=0.98), min_size=K, max_size=K)
    )
    Phi_f = jnp.diag(jnp.array(phi_f_diag))

    # Volatility AR: Persistent (0.85, 0.99)
    phi_h_diag = draw(
        st.lists(st.floats(min_value=0.85, max_value=0.99), min_size=K, max_size=K)
    )
    Phi_h = jnp.diag(jnp.array(phi_h_diag))

    # Long-run mean: Unconstrained but reasonable
    mu = draw(
        st.lists(st.floats(min_value=-5.0, max_value=0.0), min_size=K, max_size=K)
    )
    mu = jnp.array(mu)

    # Idiosyncratic variances: Positive
    sigma2 = draw(
        st.lists(st.floats(min_value=0.01, max_value=1.0), min_size=N, max_size=N)
    )
    sigma2 = jnp.array(sigma2)

    # Log-vol innovation covariance: Positive definite diagonal
    q_h_diag = draw(
        st.lists(st.floats(min_value=0.001, max_value=0.1), min_size=K, max_size=K)
    )
    Q_h = jnp.diag(jnp.array(q_h_diag))

    return DFSVParams(lambda_r, Phi_f, Phi_h, mu, sigma2, Q_h)


@st.composite
def observations_strategy(draw, N=3, T_min=10, T_max=50):
    """Generate synthetic observations for testing.

    Args:
        draw: Hypothesis draw function.
        N: Number of observed series.
        T_min: Minimum time length.
        T_max: Maximum time length.

    Returns:
        Array of observations (T, N).
    """
    T = draw(st.integers(min_value=T_min, max_value=T_max))

    # Generate realistic financial returns: zero mean, moderate volatility
    observations = draw(
        st.lists(
            st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=N, max_size=N),
            min_size=T,
            max_size=T,
        )
    )
    return jnp.array(observations)
