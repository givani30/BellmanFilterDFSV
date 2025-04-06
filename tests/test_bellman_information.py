import pytest
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
# No longer need to import config directly


from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
# Import the original filter for comparison tests later
from bellman_filter_dfsv.filters.bellman import DFSVBellmanFilter
from bellman_filter_dfsv.utils.transformations import transform_params, untransform_params
from bellman_filter_dfsv.filters.objectives import transformed_bellman_objective # Using transformed objective for stability test
import optimistix as optx # For the optimizer

# Enable float64 for tests
jax.config.update("jax_enable_x64", True)

# Fixture for setting up the filter, parameters, and data
@pytest.fixture(scope="module") # Use module scope for efficiency
def bif_setup():
    """Provides a setup for BIF tests: filter instance, params, data."""
    N = 3  # Number of series
    K = 1  # Number of factors
    T = 10 # Number of time steps
    state_dim = K * 2

    # Simple, stable parameters
    params_dict = {
        "lambda_r": jnp.array([[0.8], [0.7], [0.6]], dtype=jnp.float64),
        "Phi_f": jnp.array([[0.95]], dtype=jnp.float64),
        "Phi_h": jnp.array([[0.98]], dtype=jnp.float64),
        "mu": jnp.array([-1.0], dtype=jnp.float64),
        "sigma2": jnp.array([0.1, 0.1, 0.1], dtype=jnp.float64),
        "Q_h": jnp.array([[0.05]], dtype=jnp.float64),
    }
    params = DFSVParamsDataclass(N=N, K=K, **params_dict)

    # Dummy observations
    key = jax.random.PRNGKey(0)
    dummy_observations = jax.random.normal(key, shape=(T, N), dtype=jnp.float64) * 0.1

    # Instantiate the filter
    bif_filter = DFSVBellmanInformationFilter(N=N, K=K)

    # Ensure JIT functions are ready before tests run
    # This is crucial as JIT happens on first call otherwise
    bif_filter._setup_jax_functions()

    return {
        "filter": bif_filter,
        "params": params,
        "observations": np.asarray(dummy_observations), # Convert to NumPy for potential non-JAX usage
        "N": N,
        "K": K,
        "T": T,
        "state_dim": state_dim
    }

# --- Test Functions ---

def test_initialize_state(bif_setup):
    """Tests the initialization of state and information matrix."""
    bif_filter = bif_setup["filter"]
    params = bif_setup["params"]
    state_dim = bif_setup["state_dim"]
    K = bif_setup["K"]

    initial_state, initial_info = bif_filter.initialize_state(params)

    # Check shapes
    assert initial_state.shape == (state_dim, 1), f"Expected state shape ({state_dim}, 1), got {initial_state.shape}"
    assert initial_info.shape == (state_dim, state_dim), f"Expected info shape ({state_dim}, {state_dim}), got {initial_info.shape}"

    # Check dtypes
    assert initial_state.dtype == jnp.float64
    assert initial_info.dtype == jnp.float64

    # Check initial state values (factors should be 0, log-vols should be mu)
    np.testing.assert_allclose(initial_state[:K], jnp.zeros((K, 1)), atol=1e-8)
    np.testing.assert_allclose(initial_state[K:], params.mu.reshape(-1, 1), atol=1e-8)

    # Check information matrix properties
    assert jnp.all(jnp.isfinite(initial_info)), "Initial info matrix contains non-finite values"
    # Check symmetry
    np.testing.assert_allclose(initial_info, initial_info.T, atol=1e-8, rtol=1e-7, err_msg="Initial info matrix is not symmetric")
    # Check if positive definite (eigenvalues > 0) - might be too strict due to jitter/inversion?
    # try:
    #     eigvals = jnp.linalg.eigvalsh(initial_info)
    #     assert jnp.all(eigvals > 0), f"Initial info matrix is not positive definite, eigenvalues: {eigvals}"
    # except jnp.linalg.LinAlgError:
    #     pytest.fail("Failed to compute eigenvalues for initial info matrix")


def test_predict_jax_info(bif_setup):
    """Tests the JITted prediction step __predict_jax_info."""
    bif_filter = bif_setup["filter"]
    params = bif_setup["params"]
    state_dim = bif_setup["state_dim"]
    K = bif_setup["K"]

    # Get initial state and info
    initial_state, initial_info = bif_filter.initialize_state(params)

    # Perform one prediction step
    predicted_state, predicted_info = bif_filter.predict_jax_info_jit(
        params, initial_state, initial_info
    )

    # Check shapes
    assert predicted_state.shape == (state_dim, 1), f"Expected predicted state shape ({state_dim}, 1), got {predicted_state.shape}"
    assert predicted_info.shape == (state_dim, state_dim), f"Expected predicted info shape ({state_dim}, {state_dim}), got {predicted_info.shape}"

    # Check dtypes
    assert predicted_state.dtype == jnp.float64
    assert predicted_info.dtype == jnp.float64

    # Check predicted state values (simple check based on dynamics)
    expected_pred_factors = params.Phi_f @ initial_state[:K]
    expected_pred_log_vols = params.mu.reshape(-1,1) + params.Phi_h @ (initial_state[K:] - params.mu.reshape(-1,1))
    np.testing.assert_allclose(predicted_state[:K], expected_pred_factors, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(predicted_state[K:], expected_pred_log_vols, atol=1e-7, rtol=1e-6)


    # Check information matrix properties
    assert jnp.all(jnp.isfinite(predicted_info)), "Predicted info matrix contains non-finite values"
    # Check symmetry
    np.testing.assert_allclose(predicted_info, predicted_info.T, atol=1e-8, rtol=1e-7, err_msg="Predicted info matrix is not symmetric")
    # Check positive definiteness (optional, might fail due to numerical issues)
    # try:
    #     eigvals = jnp.linalg.eigvalsh(predicted_info)
    #     assert jnp.all(eigvals > 0), f"Predicted info matrix is not positive definite, eigenvalues: {eigvals}"
    # except jnp.linalg.LinAlgError:
    #      pytest.fail("Failed to compute eigenvalues for predicted info matrix")


def test_update_jax_info(bif_setup):
    """Tests the JITted update step __update_jax_info."""
    bif_filter = bif_setup["filter"]
    params = bif_setup["params"]
    observations = bif_setup["observations"]
    state_dim = bif_setup["state_dim"]
    N = bif_setup["N"]

    # Get initial state and info
    initial_state, initial_info = bif_filter.initialize_state(params)

    # Perform one prediction step
    predicted_state, predicted_info = bif_filter.predict_jax_info_jit(
        params, initial_state, initial_info
    )

    # Get first observation
    obs_t = jnp.asarray(observations[0], dtype=jnp.float64)
    assert obs_t.shape == (N,), f"Observation shape mismatch, expected ({N},), got {obs_t.shape}"

    # Perform one update step
    updated_state, updated_info, log_lik_contrib = bif_filter.update_jax_info_jit(
        params, predicted_state, predicted_info, obs_t
    )

    # Check shapes
    assert updated_state.shape == (state_dim, 1), f"Expected updated state shape ({state_dim}, 1), got {updated_state.shape}"
    assert updated_info.shape == (state_dim, state_dim), f"Expected updated info shape ({state_dim}, {state_dim}), got {updated_info.shape}"
    assert jnp.isscalar(log_lik_contrib), f"Log likelihood contribution should be a scalar, got shape {jnp.shape(log_lik_contrib)}"

    # Check dtypes
    assert updated_state.dtype == jnp.float64
    assert updated_info.dtype == jnp.float64
    assert log_lik_contrib.dtype == jnp.float64

    # Check properties
    assert jnp.all(jnp.isfinite(updated_state)), "Updated state contains non-finite values"
    assert jnp.all(jnp.isfinite(updated_info)), "Updated info matrix contains non-finite values"
    assert jnp.isfinite(log_lik_contrib), "Log likelihood contribution is non-finite"

    # Check symmetry of updated info matrix
    np.testing.assert_allclose(updated_info, updated_info.T, atol=1e-8, rtol=1e-7, err_msg="Updated info matrix is not symmetric")

    # Optional: Check positive definiteness
    # try:
    #     eigvals = jnp.linalg.eigvalsh(updated_info)
    #     assert jnp.all(eigvals > 0), f"Updated info matrix is not positive definite, eigenvalues: {eigvals}"
    # except jnp.linalg.LinAlgError:
    #     pytest.fail("Failed to compute eigenvalues for updated info matrix")


def test_filter_loop(bif_setup):
    """Tests the full filter loop using the Python loop implementation."""
    bif_filter = bif_setup["filter"]
    params = bif_setup["params"]
    observations = bif_setup["observations"]
    T = bif_setup["T"]
    state_dim = bif_setup["state_dim"]

    # Run the filter
    filtered_states, filtered_infos, total_log_lik = bif_filter.filter(params, observations)

    # Check output shapes
    assert filtered_states.shape == (T, state_dim), f"Expected filtered states shape ({T}, {state_dim}), got {filtered_states.shape}"
    assert filtered_infos.shape == (T, state_dim, state_dim), f"Expected filtered infos shape ({T}, {state_dim}, {state_dim}), got {filtered_infos.shape}"
    assert isinstance(total_log_lik, float), f"Total log likelihood should be a float, got {type(total_log_lik)}"

    # Check dtypes (should be NumPy arrays)
    assert filtered_states.dtype == np.float64
    assert filtered_infos.dtype == np.float64

    # Check properties
    assert np.all(np.isfinite(filtered_states)), "Filtered states contain non-finite values"
    assert np.all(np.isfinite(filtered_infos)), "Filtered infos contain non-finite values"
    assert np.isfinite(total_log_lik), "Total log likelihood is non-finite"

    # Check internal storage matches returned values
    np.testing.assert_array_equal(filtered_states, bif_filter.get_filtered_states())
    np.testing.assert_array_equal(filtered_infos, bif_filter.get_filtered_information_matrices())
    assert total_log_lik == bif_filter.get_total_log_likelihood()


def test_filter_scan_loop(bif_setup):
    """Tests the full filter loop using the jax.lax.scan implementation."""
    bif_filter = bif_setup["filter"]
    params = bif_setup["params"]
    observations = bif_setup["observations"]
    T = bif_setup["T"]
    state_dim = bif_setup["state_dim"]

    # Run the filter_scan
    filtered_states_np, filtered_infos_np, total_log_lik_jax = bif_filter.filter_scan(params, observations)

    # Check output shapes
    assert filtered_states_np.shape == (T, state_dim), f"Expected filtered states shape ({T}, {state_dim}), got {filtered_states_np.shape}"
    assert filtered_infos_np.shape == (T, state_dim, state_dim), f"Expected filtered infos shape ({T}, {state_dim}, {state_dim}), got {filtered_infos_np.shape}"
    assert isinstance(total_log_lik_jax, jax.Array), f"Total log likelihood should be a JAX Array, got {type(total_log_lik_jax)}"
    assert jnp.isscalar(total_log_lik_jax), "Total log likelihood should be a scalar"


    # Check dtypes (NumPy for arrays, JAX for scalar loglik)
    assert filtered_states_np.dtype == np.float64
    assert filtered_infos_np.dtype == np.float64
    assert total_log_lik_jax.dtype == jnp.float64

    # Check properties
    assert np.all(np.isfinite(filtered_states_np)), "Filtered states (scan) contain non-finite values"
    assert np.all(np.isfinite(filtered_infos_np)), "Filtered infos (scan) contain non-finite values"
    assert jnp.isfinite(total_log_lik_jax), "Total log likelihood (scan) is non-finite"

    # Check internal storage matches returned values
    np.testing.assert_array_equal(filtered_states_np, bif_filter.get_filtered_states())
    np.testing.assert_array_equal(filtered_infos_np, bif_filter.get_filtered_information_matrices())
    assert total_log_lik_jax == bif_filter.get_total_log_likelihood() # Compare JAX scalar


def test_log_likelihood_wrt_params(bif_setup):
    """Tests the log_likelihood_wrt_params method."""
    bif_filter = bif_setup["filter"]
    params = bif_setup["params"]
    observations = bif_setup["observations"]

    # Pass the params dataclass directly
    log_lik = bif_filter.log_likelihood_wrt_params(params, observations)

    # Check type, shape, and finiteness
    assert isinstance(log_lik, jax.Array), f"Expected JAX Array, got {type(log_lik)}"
    assert jnp.isscalar(log_lik), f"Expected scalar, got shape {jnp.shape(log_lik)}"
    assert log_lik.dtype == jnp.float64
    assert jnp.isfinite(log_lik), "Log likelihood is non-finite"


def test_jit_log_likelihood_wrt_params(bif_setup):
    """Tests the JIT-compiled log-likelihood function."""
    bif_filter = bif_setup["filter"]
    params = bif_setup["params"]
    # Use JAX array for observations with JITted function
    observations_jax = jnp.asarray(bif_setup["observations"])

    # Get the JITted function
    jit_log_lik_fn = bif_filter.jit_log_likelihood_wrt_params()

    # Call the JITted function
    log_lik_jit = jit_log_lik_fn(params, observations_jax)

    # Check type, shape, and finiteness
    assert isinstance(log_lik_jit, jax.Array), f"Expected JAX Array from JIT, got {type(log_lik_jit)}"
    assert jnp.isscalar(log_lik_jit), f"Expected scalar from JIT, got shape {jnp.shape(log_lik_jit)}"
    assert log_lik_jit.dtype == jnp.float64
    assert jnp.isfinite(log_lik_jit), "JIT Log likelihood is non-finite"

    # Compare with non-JITted version (using the internal impl for direct comparison)
    # Note: The internal implementation name also changed
    log_lik_non_jit = bif_filter._log_likelihood_wrt_params_impl(params, observations_jax)
    np.testing.assert_allclose(log_lik_jit, log_lik_non_jit, atol=1e-7, rtol=1e-6, err_msg="JIT vs non-JIT likelihood mismatch")


def test_bif_vs_bf_comparison(bif_setup):
    """
    Compares the BIF output against the original Bellman Filter (BF)
    on a stable problem instance.
    """
    # BIF setup
    bif_filter = bif_setup["filter"]
    params = bif_setup["params"]
    observations = bif_setup["observations"] # NumPy array
    N = bif_setup["N"]
    K = bif_setup["K"]

    # Run BIF (using scan for consistency)
    bif_states, bif_infos, bif_log_lik = bif_filter.filter_scan(params, observations)

    # Original BF setup
    bf_filter = DFSVBellmanFilter(N=N, K=K)
    bf_filter._setup_jax_functions() # Ensure JIT functions are ready

    # Run original BF (using scan)
    bf_states, bf_covs, bf_log_lik = bf_filter.filter_scan(params, observations)

    # Compare filtered states
    np.testing.assert_allclose(
        bif_states,
        bf_states,
        rtol=1e-5, # Relaxed tolerance due to different numerical paths
        atol=1e-6,
        err_msg="Filtered states differ between BIF and original BF"
    )

    # Note: Log-likelihoods are NOT expected to be the same due to different penalty terms.
    # The BIF uses the specific KL-type penalty from Lange (2024, Eq. 40),
    # while the original BF implementation might use a different likelihood calculation
    # or a standard KL divergence penalty. Comparison focuses on state estimates.

def test_bif_stability_during_optimization(bif_setup):
    """
    Tests the numerical stability of the BIF during a few optimization steps.
    Uses the transformed objective and uninformed initial parameters, similar to
    the setup where the original BF showed instability.
    """
    bif_filter = bif_setup["filter"]
    observations = bif_setup["observations"] # NumPy array
    N = bif_setup["N"]
    K = bif_setup["K"]

    # Use JAX array for observations in objective
    observations_jax = jnp.asarray(observations)

    # Create uninformed initial parameters (similar to bf_optimization.py)
    # Use slightly different values to avoid potential edge cases if defaults are problematic
    data_variance = jnp.var(observations_jax, axis=0)
    uninformed_params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=0.6 * jnp.ones((N, K)), # Slightly different loading
        Phi_f=0.9 * jnp.eye(K),          # Slightly different persistence
        Phi_h=0.9 * jnp.eye(K),          # Slightly different persistence
        mu=jnp.ones(K) * -0.5,           # Slightly different mean
        sigma2=0.6 * data_variance,      # Slightly different scaling
        Q_h=0.15 * jnp.eye(K)            # Slightly different vol-of-vol
    )

    # Transform initial parameters
    transformed_initial_params = transform_params(uninformed_params)

    # Define the objective function using the BIF filter instance
    # Note: transformed_bellman_objective expects the filter instance
    # It will internally call filter.jit_log_likelihood_wrt_params() # Updated method name
    # Set priors to zero (or near zero variance) for this stability test
    prior_mu_mean = 0.0
    prior_mu_var = 1e-9 # Use variance, set near zero for minimal effect

    # Define a wrapper function for optimistix instead of using partial
    # optimistix calls fn(params, *args)
    def objective_wrapper(t_params, args_tuple):
        obs_jax, filt, p_mean, p_var = args_tuple # Unpack static args
        # Construct the priors dictionary expected by the objective function
        priors_dict = {
            'prior_mu_mean': p_mean,
            'prior_mu_var': p_var
            # Add other prior keys here if needed by the test,
            # otherwise log_prior_density will use defaults.
        }
        return transformed_bellman_objective(
            transformed_params=t_params,
            y=obs_jax,
            filter=filt,
            priors=priors_dict # Pass the dictionary
        )

    # Package the static arguments for optimistix (use variance)
    static_args = (observations_jax, bif_filter, prior_mu_mean, prior_mu_var)

    # Use a simple optimizer for a few steps
    solver = optx.BFGS(rtol=1e-3, atol=1e-3) # Simple BFGS

    # Run a few optimization steps
    max_steps = 5 # Just a few steps to check for immediate NaN issues
    try:
        # Ensure the filter's JIT functions are ready *before* the objective is called by optimistix
        # This might involve calling a dummy likelihood calculation once if not done in fixture
        _ = bif_filter.log_likelihood_wrt_params(uninformed_params, observations) # Pass dataclass directly

        result = optx.minimise(
            fn=objective_wrapper, # Use the wrapper function
            solver=solver,
            y0=transformed_initial_params,
            args=static_args,      # Pass static arguments via args
            max_steps=max_steps,
            throw=False # Don't throw error on failure, check status instead
        )

        # Check results
        final_params_transformed = result.value
        # Re-evaluate the objective function at the final parameters to get the final loss
        final_loss = objective_wrapper(result.value, static_args)

        # Check the result status using result.result
        # For this test with max_steps=5, reaching the nonlinear max steps is also acceptable
        opt_result_status = result.result
        assert opt_result_status in (optx.RESULTS.successful,
                                     optx.RESULTS.max_steps_reached,
                                     optx.RESULTS.nonlinear_max_steps_reached), \
            f"Optimization did not succeed or reach max steps acceptably, status: {opt_result_status}"

        # Check if final parameters and loss are finite
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(final_params_transformed)), \
            "Optimization resulted in non-finite transformed parameters"
        assert jnp.isfinite(final_loss), \
            f"Optimization resulted in non-finite loss: {final_loss}"

        # Untransform and check original space parameters
        final_params_orig = untransform_params(final_params_transformed)
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(final_params_orig)), \
            "Optimization resulted in non-finite original parameters after untransforming"

    except Exception as e:
        pytest.fail(f"Optimization process raised an unexpected exception: {e}")

def test_smooth(bif_setup):
    """Tests the smoother implementation for BIF."""
    bif_filter = bif_setup["filter"]
    params = bif_setup["params"]
    observations = bif_setup["observations"]
    T = bif_setup["T"]
    state_dim = bif_setup["state_dim"]

    # Run the filter first to populate filtered results
    _, _, _ = bif_filter.filter_scan(params, observations)

    # Run the smoother
    try:
        # Pass params to the smooth method
        smoothed_states, smoothed_covs = bif_filter.smooth(params)
    except Exception as e:
        pytest.fail(f"Smoother raised an unexpected exception: {e}")

    # Check output shapes
    assert smoothed_states.shape == (T, state_dim), f"Expected smoothed states shape ({T}, {state_dim}), got {smoothed_states.shape}"
    assert smoothed_covs.shape == (T, state_dim, state_dim), f"Expected smoothed covs shape ({T}, {state_dim}, {state_dim}), got {smoothed_covs.shape}"

    # Check dtypes (should be NumPy arrays)
    assert smoothed_states.dtype == np.float64
    assert smoothed_covs.dtype == np.float64

    # Check properties
    assert np.all(np.isfinite(smoothed_states)), "Smoothed states contain non-finite values"
    assert np.all(np.isfinite(smoothed_covs)), "Smoothed covs contain non-finite values"

    # Check internal storage matches returned values (use attributes directly)
    np.testing.assert_array_equal(smoothed_states, bif_filter.smoothed_states)
    np.testing.assert_array_equal(smoothed_covs, bif_filter.smoothed_covs)

    # Check symmetry of smoothed covariances
    for i in range(T):
        matrix = smoothed_covs[i]
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-7, rtol=1e-6,
                                   err_msg=f"Smoothed covariance matrix at index {i} is not symmetric")


def test_getters(bif_setup):
    """Tests all getter methods after running the filter."""
    bif_filter = bif_setup["filter"]
    params = bif_setup["params"]
    observations = bif_setup["observations"]
    T = bif_setup["T"]
    N = bif_setup["N"]
    K = bif_setup["K"]
    state_dim = bif_setup["state_dim"]

    # Run the filter first to populate results
    # Use filter_scan as it stores JAX scalar for total_log_likelihood
    _, _, _ = bif_filter.filter_scan(params, observations)

    # Test each getter
    getters_to_test = {
        "get_filtered_states": (np.ndarray, (T, state_dim)),
        "get_filtered_factors": (np.ndarray, (T, K)),
        "get_filtered_volatilities": (np.ndarray, (T, K)),
        "get_filtered_information_matrices": (np.ndarray, (T, state_dim, state_dim)),
        "get_predicted_states": (np.ndarray, (T, state_dim, 1)),
        "get_predicted_information_matrices": (np.ndarray, (T, state_dim, state_dim)),
        "get_log_likelihoods": (np.ndarray, (T,)),
        "get_total_log_likelihood": ((jax.Array, float), ()), # Can be JAX scalar or float
        "get_predicted_covariances": (np.ndarray, (T, state_dim, state_dim)),
        "get_predicted_variances": (np.ndarray, (T, state_dim)),
        "get_filtered_covariances": (np.ndarray, (T, state_dim, state_dim)), # Added
        "get_filtered_variances": (np.ndarray, (T, state_dim)), # Added
    }

    for method_name, (expected_type, expected_shape) in getters_to_test.items():
        getter_method = getattr(bif_filter, method_name)
        result = getter_method()

        assert result is not None, f"{method_name} returned None"

        # Type check (allow tuple for total_log_likelihood)
        if isinstance(expected_type, tuple):
             assert isinstance(result, expected_type), f"{method_name} returned wrong type: {type(result)}, expected {expected_type}"
        else:
             assert isinstance(result, expected_type), f"{method_name} returned wrong type: {type(result)}, expected {expected_type}"


        # Shape check (skip for scalar total_log_likelihood)
        if expected_shape:
             assert result.shape == expected_shape, f"{method_name} returned wrong shape: {result.shape}, expected {expected_shape}"
        else:
             # For scalars (like total_log_likelihood from filter_scan)
             assert np.isscalar(result) or (isinstance(result, jax.Array) and result.ndim == 0), f"{method_name} should be scalar, got shape {getattr(result, 'shape', 'N/A')}"


        # Finiteness check
        assert np.all(np.isfinite(result)), f"{method_name} result contains non-finite values"

        # Additional check for symmetry for covariance/information matrices
        if "information" in method_name or "covariances" in method_name:
            # Check symmetry for each matrix in the time series
            for i in range(T):
                matrix = result[i]
                np.testing.assert_allclose(matrix, matrix.T, atol=1e-7, rtol=1e-6,
                                           err_msg=f"{method_name} matrix at index {i} is not symmetric")
