"""Pytest-based test suite for the DFSVBellmanFilter implementation."""

import os
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Callable

# Assuming fixtures are available from conftest.py:
# params_fixture, data_fixture, filter_instances_fixture

from bellman_filter_dfsv.core.filters.bellman import DFSVBellmanFilter
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass


# --- Specific Tests Retained and Adapted ---

def test_bellman_single_step(
    params_fixture: Callable[..., DFSVParamsDataclass],
    filter_instances_fixture: Callable[..., Dict[str, Any]],
):
    """Tests a single prediction and update step of the filter."""
    # Arrange: Use specific params for this test if needed, or default fixture params
    N = 5
    K = 2
    key = jax.random.PRNGKey(42)
    key, subkey_lambda, subkey_y = jax.random.split(key, 3)

    # Create test parameters slightly different from fixture defaults if needed
    params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=jax.random.normal(subkey_lambda, (N, K)),
        Phi_f=jnp.array(0.9 * jnp.eye(K)),
        Phi_h=jnp.array(0.95 * jnp.eye(K)),
        mu=jnp.zeros(K),
        Q_h=jnp.array(0.1 * jnp.eye(K)),
        sigma2=jnp.array(0.1 * jnp.ones(N)),
    )

    # Get filter instance (can use fixture, but need specific N, K)
    # bf = filter_instances_fixture(params)["bellman"] # Fixture uses default N, K
    bf = DFSVBellmanFilter(N=N, K=K) # Instantiate directly for specific N, K

    # Create artificial data for one step
    T = 1
    y_np = np.random.RandomState(42).normal(0, 1, (T, N))
    y_jax = jnp.asarray(y_np) # Use JAX array

    # Initialize state using the filter's method
    initial_state, initial_cov = bf.initialize_state(params)

    # Act: Prediction step (using public API)
    predicted_state, predicted_cov = bf.predict(params, initial_state, initial_cov)

    # Act: Update step (using public API)
    # Note: update expects a single observation vector y[0]
    updated_state, updated_cov, log_likelihood = bf.update(
        params, predicted_state, predicted_cov, y_jax[0]
    )

    # Assert: Basic checks on shapes and finiteness
    state_dim = 2 * K
    assert initial_state.shape == (state_dim, 1), "Incorrect initial state shape"
    assert initial_cov.shape == (state_dim, state_dim), "Incorrect initial covariance shape"
    assert predicted_state.shape == (state_dim, 1), "Incorrect predicted state shape"
    assert predicted_cov.shape == (state_dim, state_dim), "Incorrect predicted covariance shape"
    assert updated_state.shape == (state_dim, 1), "Incorrect updated state shape"
    assert updated_cov.shape == (state_dim, state_dim), "Incorrect updated covariance shape"

    assert isinstance(log_likelihood, (jax.Array, float)), "Log-likelihood should be JAX array or float"
    assert jnp.isscalar(log_likelihood), "Log-likelihood should be scalar"

    assert jnp.all(jnp.isfinite(updated_state)), "Updated state contains non-finite values"
    assert jnp.all(jnp.isfinite(updated_cov)), "Updated covariance contains non-finite values"
    assert jnp.isfinite(log_likelihood), "Log-likelihood is non-finite"

    # Check symmetry of covariances
    np.testing.assert_allclose(initial_cov, initial_cov.T, atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(predicted_cov, predicted_cov.T, atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(updated_cov, updated_cov.T, atol=1e-8, rtol=1e-7)


def test_smooth(
    params_fixture: Callable[..., DFSVParamsDataclass],
    data_fixture: Callable[..., Dict[str, Any]],
    filter_instances_fixture: Callable[..., Dict[str, Any]],
):
    """Tests the smoother implementation for the Bellman Filter."""
    # Arrange: Use fixtures for params, data, filter
    params: DFSVParamsDataclass = params_fixture(N=3, K=1) # Use smaller model for speed
    sim_data: Dict[str, Any] = data_fixture(params, T=50, seed=555) # Shorter series
    observations: jax.Array = sim_data["observations"]
    bf: DFSVBellmanFilter = filter_instances_fixture(params)["bellman"]
    T = sim_data["T"]
    state_dim = params.K * 2

    # Act: Run filter first (using scan)
    _ = bf.filter_scan(params, observations) # Populates internal results needed for smooth

    # Act: Run smoother
    try:
        # Pass params to the smooth method
        smoothed_states, smoothed_covs = bf.smooth(params)
    except Exception as e:
        pytest.fail(f"Smoother raised an unexpected exception: {e}")

    # Assert: Check output shapes and types
    assert smoothed_states.shape == (T, state_dim), f"Expected smoothed states shape ({T}, {state_dim}), got {smoothed_states.shape}"
    assert smoothed_covs.shape == (T, state_dim, state_dim), f"Expected smoothed covs shape ({T}, {state_dim}, {state_dim}), got {smoothed_covs.shape}"
    assert isinstance(smoothed_states, np.ndarray), "Smoothed states should be NumPy array"
    assert isinstance(smoothed_covs, np.ndarray), "Smoothed covs should be NumPy array"
    assert smoothed_states.dtype == np.float64
    assert smoothed_covs.dtype == np.float64

    # Assert: Check properties
    assert np.all(np.isfinite(smoothed_states)), "Smoothed states contain non-finite values"
    assert np.all(np.isfinite(smoothed_covs)), "Smoothed covs contain non-finite values"

    # Assert: Check internal storage matches returned values
    np.testing.assert_array_equal(smoothed_states, bf.smoothed_states)
    np.testing.assert_array_equal(smoothed_covs, bf.smoothed_covs)

    # Assert: Check symmetry of smoothed covariances
    for i in range(T):
        matrix = smoothed_covs[i]
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-7, rtol=1e-6,
                                   err_msg=f"Smoothed covariance matrix at index {i} is not symmetric")


def test_bellman_filter_estimation(
    params_fixture: Callable[..., DFSVParamsDataclass],
    data_fixture: Callable[..., Dict[str, Any]],
    filter_instances_fixture: Callable[..., Dict[str, Any]],
):
    """Tests the filter's state estimation accuracy on simulated data."""
    # Arrange: Use fixtures
    params: DFSVParamsDataclass = params_fixture() # Default N=4, K=2
    sim_data: Dict[str, Any] = data_fixture(params, T=200, seed=42)
    observations: jax.Array = sim_data["observations"]
    true_factors: np.ndarray = np.asarray(sim_data["true_factors"]) # Ensure NumPy for comparison
    true_log_vols: np.ndarray = np.asarray(sim_data["true_log_vols"])
    bf: DFSVBellmanFilter = filter_instances_fixture(params)["bellman"]
    T = sim_data["T"]

    # Act: Run filter (using scan)
    _ = bf.filter_scan(params, observations)

    # Act: Extract factor and volatility estimates
    filtered_factors = bf.get_filtered_factors() # Returns NumPy array
    filtered_log_vols = bf.get_filtered_volatilities() # Returns NumPy array

    # Assert: Shapes and types
    assert filtered_factors.shape == (T, params.K)
    assert filtered_log_vols.shape == (T, params.K)
    assert isinstance(filtered_factors, np.ndarray)
    assert isinstance(filtered_log_vols, np.ndarray)

    # Assert: Check correlation between true and filtered factors
    for k in range(params.K):
        # Ensure finite values before calculating correlation
        ff_k = filtered_factors[:, k]
        tf_k = true_factors[:, k]
        valid_indices = np.isfinite(ff_k) & np.isfinite(tf_k)
        if np.sum(valid_indices) > 1 and np.std(ff_k[valid_indices]) > 1e-6 and np.std(tf_k[valid_indices]) > 1e-6:
             corr = np.corrcoef(tf_k[valid_indices], ff_k[valid_indices])[0, 1]
             assert corr > 0.7, f"Factor {k+1} correlation too low: {corr:.3f}"
        else:
             print(f"Warning: Skipping correlation check for Factor {k+1} due to insufficient valid data or zero variance.")


    # Assert: Check correlation between true and filtered log-volatilities
    for k in range(params.K):
        flv_k = filtered_log_vols[:, k]
        tlv_k = true_log_vols[:, k]
        valid_indices = np.isfinite(flv_k) & np.isfinite(tlv_k)
        if np.sum(valid_indices) > 1 and np.std(flv_k[valid_indices]) > 1e-6 and np.std(tlv_k[valid_indices]) > 1e-6:
            corr = np.corrcoef(tlv_k[valid_indices], flv_k[valid_indices])[0, 1]
            # Threshold kept as per original test, may need adjustment based on BIF impact
            assert corr > 0.4, f"Log-volatility {k+1} correlation too low: {corr:.3f}"
        else:
             print(f"Warning: Skipping correlation check for Log-Volatility {k+1} due to insufficient valid data or zero variance.")


    # Assert: Check the average estimation error is within reasonable bounds
    valid_factors = np.isfinite(true_factors) & np.isfinite(filtered_factors)
    factor_rmse = np.sqrt(np.mean((true_factors[valid_factors] - filtered_factors[valid_factors]) ** 2))

    valid_vols = np.isfinite(true_log_vols) & np.isfinite(filtered_log_vols)
    vol_rmse = np.sqrt(np.mean((true_log_vols[valid_vols] - filtered_log_vols[valid_vols]) ** 2))

    # Thresholds relaxed due to parameter changes (increased Q_h, decreased sigma2)
    assert factor_rmse < 0.9, f"Factor RMSE too high: {factor_rmse:.3f}"
    assert vol_rmse < 1.5, f"Log-volatility RMSE too high: {vol_rmse:.3f}"


# Helper function for visualization test
def _create_visual_comparison(
    params: DFSVParamsDataclass,
    bf: DFSVBellmanFilter,
    sim_data: Dict[str, Any],
    save_path: str = None
) -> plt.Figure:
    """Creates visual comparisons between true and filtered states."""
    # Extract data
    true_factors = np.asarray(sim_data["true_factors"])
    true_log_vols = np.asarray(sim_data["true_log_vols"])
    filtered_factors = bf.get_filtered_factors()
    filtered_log_vols = bf.get_filtered_volatilities()

    # Create figure
    fig, axs = plt.subplots(2, params.K, figsize=(7 * params.K, 10), sharex=True)
    if params.K == 1: # Handle case where axs is 1D
        axs = np.array([axs]).T # Make it 2x1

    # Plot factors
    for k in range(params.K):
        ax = axs[0, k]
        ax.plot(true_factors[:, k], "b-", label="True", alpha=0.8)
        ax.plot(filtered_factors[:, k], "r--", label="Filtered")
        ax.set_title(f"Factor {k+1}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    # Plot log-volatilities
    for k in range(params.K):
        ax = axs[1, k]
        ax.plot(true_log_vols[:, k], "b-", label="True", alpha=0.8)
        ax.plot(filtered_log_vols[:, k], "r--", label="Filtered")
        ax.set_title(f"Log-Volatility {k+1}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel("Time Step") # Add x-label to bottom row

    fig.suptitle(
        "Bellman Filter Performance: True vs. Filtered States", fontsize=16
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    # Save the figure if a path is provided
    if save_path:
        try:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Visual comparison saved to {save_path}")
        except Exception as e:
            print(f"Failed to save figure: {e}")
            pytest.fail(f"Failed to save figure: {e}") # Fail test if saving fails

    return fig


# Mark test to not run in CI environments if saving files is problematic
@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skipping file-saving test in CI")
def test_visualization(
    tmp_path: Path, # Use pytest's tmp_path fixture for temporary file saving
    params_fixture: Callable[..., DFSVParamsDataclass],
    data_fixture: Callable[..., Dict[str, Any]],
    filter_instances_fixture: Callable[..., Dict[str, Any]],
):
    """Generates and saves a visual comparison of filter results."""
    # Arrange
    params: DFSVParamsDataclass = params_fixture() # Default N=4, K=2
    sim_data: Dict[str, Any] = data_fixture(params, T=300, seed=456)
    observations: jax.Array = sim_data["observations"]
    bf: DFSVBellmanFilter = filter_instances_fixture(params)["bellman"]

    # Act: Run filter
    _ = bf.filter_scan(params, observations)

    # Define save path within the temporary directory
    save_path = tmp_path / "bf_visual_comparison.png"

    # Act: Generate visual comparison and save to file
    fig = _create_visual_comparison(params, bf, sim_data, save_path=str(save_path))

    # Assert: Basic check for figure creation and file existence
    assert isinstance(fig, plt.Figure)
    assert save_path.exists(), f"Figure was not saved to {save_path}"

    # Clean up the plot
    plt.close(fig)


# --- Redundant Tests Removed ---
# - test_bellman_full_filter (Covered by test_filter_stability and test_log_likelihood_wrt_params in test_unified_filters.py)
# - test_bellman_filter_numeric_stability (Covered by test_filter_stability in test_unified_filters.py)
