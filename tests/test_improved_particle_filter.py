"""Tests for the improved particle filter implementation."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Callable

from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from tests.custom_particle_filter import ImprovedParticleFilter, create_improved_filter

def test_improved_particle_filter(
    params_fixture: Callable[..., DFSVParamsDataclass],
    data_fixture: Callable[..., Dict[str, Any]],
):
    """
    Test the improved particle filter's ability to estimate states from simulated data.
    """
    # Arrange: Use fixtures
    params: DFSVParamsDataclass = params_fixture()  # Default N=4, K=2
    sim_data: Dict[str, Any] = data_fixture(params, T=200, seed=42)
    observations: jax.Array = sim_data["observations"]
    true_factors: np.ndarray = np.asarray(sim_data["true_factors"])
    true_log_vols: np.ndarray = np.asarray(sim_data["true_log_vols"])

    # Create improved particle filter
    pf = ImprovedParticleFilter(
        N=params.N,
        K=params.K,
        num_particles=2000,
        resample_threshold_frac=0.7,  # Higher threshold for more frequent resampling
        seed=42
    )

    # Act: Run filter
    _ = pf.filter(params=params, observations=observations)

    # Act: Extract factor and volatility estimates
    filtered_factors = pf.get_filtered_factors()
    filtered_log_vols = pf.get_filtered_volatilities()

    # Assert: Check correlation between true and filtered factors
    for k in range(params.K):
        ff_k = filtered_factors[:, k]
        tf_k = true_factors[:, k]
        valid_indices = np.isfinite(ff_k) & np.isfinite(tf_k)
        if np.sum(valid_indices) > 1 and np.std(ff_k[valid_indices]) > 1e-6 and np.std(tf_k[valid_indices]) > 1e-6:
            corr = np.corrcoef(tf_k[valid_indices], ff_k[valid_indices])[0, 1]
            print(f"Factor {k} correlation: {corr:.4f}")
            assert corr > 0.4, f"Factor {k} correlation too low: {corr:.4f}"
        else:
            print(f"Warning: Skipping Factor {k} correlation check due to insufficient valid/variant data.")

    # Assert: Check correlation between true and filtered log-volatilities
    for k in range(params.K):
        flv_k = filtered_log_vols[:, k]
        tlv_k = true_log_vols[:, k]
        valid_indices = np.isfinite(flv_k) & np.isfinite(tlv_k)
        if np.sum(valid_indices) > 1 and np.std(flv_k[valid_indices]) > 1e-6 and np.std(tlv_k[valid_indices]) > 1e-6:
            corr = np.corrcoef(tlv_k[valid_indices], flv_k[valid_indices])[0, 1]
            print(f"Log-volatility {k} correlation: {corr:.4f}")
            assert corr > 0.15, f"Log-volatility {k} correlation too low: {corr:.4f}"
        else:
            print(f"Warning: Skipping Log-Volatility {k} correlation check due to insufficient valid/variant data.")

    # Assert: Check the average estimation error is within reasonable bounds
    # For factors, we'll normalize before computing RMSE to account for scale differences
    valid_factors = np.isfinite(true_factors) & np.isfinite(filtered_factors)

    # Normalize factors for each dimension separately
    normalized_true_factors = np.zeros_like(true_factors)
    normalized_filtered_factors = np.zeros_like(filtered_factors)

    for k in range(params.K):
        # Extract valid data for this factor
        true_k = true_factors[:, k]
        filtered_k = filtered_factors[:, k]
        valid_k = np.isfinite(true_k) & np.isfinite(filtered_k)

        if np.sum(valid_k) > 1:
            # Normalize to zero mean and unit variance
            true_mean, true_std = np.mean(true_k[valid_k]), np.std(true_k[valid_k])
            filtered_mean, filtered_std = np.mean(filtered_k[valid_k]), np.std(filtered_k[valid_k])

            normalized_true_factors[:, k][valid_k] = (true_k[valid_k] - true_mean) / true_std
            normalized_filtered_factors[:, k][valid_k] = (filtered_k[valid_k] - filtered_mean) / filtered_std

    # Compute RMSE on normalized factors
    factor_rmse = np.sqrt(np.mean((normalized_true_factors[valid_factors] - normalized_filtered_factors[valid_factors]) ** 2))

    # For log volatilities, we can use the raw values
    valid_vols = np.isfinite(true_log_vols) & np.isfinite(filtered_log_vols)
    vol_rmse = np.sqrt(np.mean((true_log_vols[valid_vols] - filtered_log_vols[valid_vols]) ** 2))

    print(f"Normalized Factor RMSE: {factor_rmse:.4f}")
    print(f"Log-volatility RMSE: {vol_rmse:.4f}")

    assert factor_rmse < 1.5, f"Normalized Factor RMSE too high: {factor_rmse:.4f}"
    assert vol_rmse < 1.5, f"Log-volatility RMSE too high: {vol_rmse:.4f}"

def test_improved_particle_filter_smooth(
    params_fixture: Callable[..., DFSVParamsDataclass],
    data_fixture: Callable[..., Dict[str, Any]],
):
    """
    Test the improved particle filter's smoothing capability.
    """
    # Arrange
    params: DFSVParamsDataclass = params_fixture()  # Default N=4, K=2
    sim_data: Dict[str, Any] = data_fixture(params, T=100, seed=999)  # Shorter series
    observations: jax.Array = sim_data["observations"]

    # Create improved particle filter
    pf = ImprovedParticleFilter(
        N=params.N,
        K=params.K,
        num_particles=2000,
        resample_threshold_frac=0.7,
        seed=42
    )

    # Act: Run filter first
    _ = pf.filter(params=params, observations=observations)

    # Act: Run smoother
    try:
        smoothed_states, smoothed_covs = pf.smooth(params=params)

        # Assert: Check output shapes and types
        assert smoothed_states.shape == (sim_data["T"], params.K * 2)
        assert smoothed_covs.shape == (sim_data["T"], params.K * 2, params.K * 2)
        assert isinstance(smoothed_states, np.ndarray)
        assert isinstance(smoothed_covs, np.ndarray)

        # Assert: Check properties
        assert np.all(np.isfinite(smoothed_states)), "Smoothed states contain non-finite values"
        assert np.all(np.isfinite(smoothed_covs)), "Smoothed covs contain non-finite values"

        print("Smoothing successful!")
    except Exception as e:
        pytest.fail(f"Smoother raised an unexpected exception: {e}")
