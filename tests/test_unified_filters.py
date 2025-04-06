"""Unified tests for core functionalities of DFSV filters."""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray, Int, PyTree
from typing import Dict, Any, Callable

# Fixtures are automatically discovered by pytest from tests/conftest.py

from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.core.filters.base import DFSVFilter
from bellman_filter_dfsv.core.filters.bellman import DFSVBellmanFilter
from bellman_filter_dfsv.core.filters.bellman_information import (
    DFSVBellmanInformationFilter,
)
from bellman_filter_dfsv.core.filters.particle import DFSVParticleFilter

# Define filter types to parametrize over
FILTER_TYPES = ["bellman", "bellman_information", "particle"]


@pytest.mark.parametrize("filter_name", FILTER_TYPES)
def test_filter_stability(
    filter_name: str,
    params_fixture: Callable[..., DFSVParamsDataclass],
    data_fixture: Callable[..., Dict[str, Any]],
    filter_instances_fixture: Callable[..., Dict[str, PyTree]],
):
    """
    Tests if the filter runs without producing NaN or Inf outputs.

    Uses the primary `filter` method for each filter type.
    """
    # Arrange: Generate params, data, and filter instance
    # Use default N=4, K=2 for params
    params: DFSVParamsDataclass = params_fixture()
    # Use default T=100, seed=42 for data
    sim_data: Dict[str, Any] = data_fixture(params)
    observations: Float[Array, "T N"] = sim_data["observations"]
    # Use default num_particles=1000 for PF
    filters: Dict[str, PyTree] = filter_instances_fixture(params)
    filter_instance: DFSVFilter = filters[filter_name]

    # Act: Run the filter's primary filtering method
    # Assumes filter method signature: filter(self, params, observations) -> tuple
    # The exact return tuple structure might vary between filters.
    filtered_output = filter_instance.filter(params, observations)

    # Assert: Check finiteness of relevant outputs
    if isinstance(filter_instance, (DFSVBellmanFilter, DFSVBellmanInformationFilter)):
        # Expecting (filtered_states, filtered_covs_or_infos)
        assert len(filtered_output) >= 2, f"{filter_name} output tuple too short"
        filtered_states = filtered_output[0]
        filtered_covs_or_infos = filtered_output[1]
        assert jnp.all(
            jnp.isfinite(filtered_states)
        ), f"{filter_name} states not finite"
        assert jnp.all(
            jnp.isfinite(filtered_covs_or_infos)
        ), f"{filter_name} covs/infos not finite"
    elif isinstance(filter_instance, DFSVParticleFilter):
        # Expecting (filtered_states, filtered_weights, log_likelihoods) based on common PF patterns
        assert len(filtered_output) >= 3, f"{filter_name} output tuple too short"
        filtered_states = filtered_output[0]
        filtered_weights = filtered_output[1]
        log_likelihoods = filtered_output[2] # Per-step likelihoods usually
        assert jnp.all(
            jnp.isfinite(filtered_states)
        ), f"{filter_name} states not finite"
        assert jnp.all(
            jnp.isfinite(filtered_weights)
        ), f"{filter_name} weights not finite"
        assert jnp.all(
            jnp.isfinite(log_likelihoods)
        ), f"{filter_name} log likelihoods not finite"
    else:
        pytest.fail(f"Unknown filter type for stability assertion: {filter_name}")


@pytest.mark.parametrize("filter_name", FILTER_TYPES)
def test_log_likelihood_wrt_params(
    filter_name: str,
    params_fixture: Callable[..., DFSVParamsDataclass],
    data_fixture: Callable[..., Dict[str, Any]],
    filter_instances_fixture: Callable[..., Dict[str, PyTree]],
):
    """
    Tests if log_likelihood_wrt_params returns a finite scalar float.
    """
    # Arrange
    params: DFSVParamsDataclass = params_fixture()
    sim_data: Dict[str, Any] = data_fixture(params)
    observations: Float[Array, "T N"] = sim_data["observations"]
    filters: Dict[str, PyTree] = filter_instances_fixture(params)
    filter_instance: DFSVFilter = filters[filter_name]

    # Act
    log_likelihood = filter_instance.log_likelihood_wrt_params(params, observations)

    # Assert
    assert isinstance(
        log_likelihood, (float, jax.Array)
    ), f"{filter_name} loglik type mismatch: {type(log_likelihood)}"
    assert jnp.isscalar(log_likelihood), f"{filter_name} loglik is not scalar"
    assert jnp.isfinite(log_likelihood), f"{filter_name} loglik is not finite"