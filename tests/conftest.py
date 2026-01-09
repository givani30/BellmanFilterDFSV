"""Pytest configuration and common fixtures for DFSV model testing - v2 architecture."""

import jax
import jax.numpy as jnp
import pytest

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
