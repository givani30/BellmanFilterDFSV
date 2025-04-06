"""Pytest configuration and common fixtures for DFSV model testing."""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray, Int, PyTree
from typing import Dict, Any, Callable

from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.core.simulation import simulate_DFSV
from bellman_filter_dfsv.core.filters.bellman import DFSVBellmanFilter
from bellman_filter_dfsv.core.filters.bellman_information import (
    DFSVBellmanInformationFilter,
)
from bellman_filter_dfsv.core.filters.particle import DFSVParticleFilter


@pytest.fixture(scope="session")
def params_fixture() -> Callable[..., DFSVParamsDataclass]:
    """
    Pytest fixture providing a factory function to create DFSV parameters.

    The factory function accepts N (default 4) and K (default 2).

    Returns:
        Callable[[int, int], DFSVParamsDataclass]: A function that generates
                                                   DFSV parameters consistent
                                                   with DFSVParamsDataclass.
    """

    def _create_params(N: int = 4, K: int = 2) -> DFSVParamsDataclass:
        """Generates DFSV parameters for given N and K."""
        key = jr.PRNGKey(0)  # Fixed key for reproducible test parameters
        key, subkey_phif = jr.split(key)
        key, subkey_phih = jr.split(key)
        key, subkey_qh = jr.split(key)
        key, subkey_lambda = jr.split(key)
        key, subkey_mu = jr.split(key)
        key, subkey_sigma2 = jr.split(key) # Added key for sigma2

        # Sensible default values matching DFSVParamsDataclass structure
        phi_f_diag = jnp.array([0.98] * K)
        phi_h_diag = jnp.array([0.95] * K)
        q_h_diag = jnp.array([0.25]*K )  # Variances for Q_h diagonal
        lambda_r_init = jr.normal(subkey_lambda, (N, K)) * 0.6
        mu_init = -jnp.abs(
            jr.normal(subkey_mu, (K,))
        ) * 0.5  # Ensure mu is negative
        sigma2_init = jnp.ones(N) * 0.05 # Default value for sigma2, shape N for idiosyncratic variances

        params = DFSVParamsDataclass(
            N=N,
            K=K,
            lambda_r=lambda_r_init,
            Phi_f=jnp.diag(phi_f_diag),
            Phi_h=jnp.diag(phi_h_diag),
            mu=mu_init,
            sigma2=sigma2_init,
            Q_h=jnp.diag(q_h_diag),
        )
        return params

    return _create_params


@pytest.fixture(scope="session")
def data_fixture() -> Callable[..., Dict[str, Any]]:
    """
    Pytest fixture providing a factory function to simulate DFSV data.

    The factory function accepts params (DFSVParamsDataclass), T (default 100),
    and seed (default 42).

    NOTE: Assumes simulate_DFSV is compatible with DFSVParamsDataclass.

    Returns:
        Callable[[DFSVParamsDataclass, int, int], Dict[str, Any]]:
            A function that simulates data and returns a dictionary containing
            'observations', 'true_factors', 'true_log_vols', 'T', and 'seed'.
    """

    def _simulate_data(
        params: DFSVParamsDataclass, T: int = 100, seed: int = 42
    ) -> Dict[str, Any]:
        """Simulates DFSV data for given parameters, T, and seed."""
        key = jr.PRNGKey(seed)
        # Assuming simulate_DFSV takes DFSVParamsDataclass directly
        observations_np, true_factors_np, true_log_vols_np = simulate_DFSV(
            params=params, T=T, seed=seed # Pass seed instead of key
        )
        # Convert to JAX arrays as expected by some filter methods
        return {
            "observations": jnp.asarray(observations_np),
            "true_factors": jnp.asarray(true_factors_np),
            "true_log_vols": jnp.asarray(true_log_vols_np),
            "T": T,
            "seed": seed,
        }

    return _simulate_data


@pytest.fixture(scope="session")
def filter_instances_fixture() -> Callable[..., Dict[str, PyTree]]:
    """
    Pytest fixture providing a factory function to create initialized DFSV filters.

    The factory function accepts params (DFSVParamsDataclass) and
    num_particles (default 1000).

    Returns:
        Callable[[DFSVParamsDataclass, int], Dict[str, PyTree]]:
            A function that returns a dictionary mapping filter names
            ('bellman', 'bellman_information', 'particle') to their
            initialized instances.
    """

    def _create_filters(
        params: DFSVParamsDataclass, num_particles: int = 1000
    ) -> Dict[str, PyTree]:
        """Initializes DFSV filters for given parameters."""
        # N, K are static fields in the dataclass
        N = params.N
        K = params.K

        bf = DFSVBellmanFilter(N=N, K=K)
        bif = DFSVBellmanInformationFilter(N=N, K=K)
        pf = DFSVParticleFilter(N=N, K=K, num_particles=num_particles)

        return {
            "bellman": bf,
            "bellman_information": bif,
            "particle": pf,
        }

    return _create_filters