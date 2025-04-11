"""
Tests for the optimization utilities in bellman_filter_dfsv.utils.optimization.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optimistix as optx
from typing import Dict, Any, List, Tuple

from bellman_filter_dfsv.utils.optimization import (
    FilterType,
    OptimizerResult,
    create_filter,
    get_objective_function,
    minimize_with_logging,
    run_optimization
)
from bellman_filter_dfsv.utils.solvers import (
    create_optimizer,
    get_available_optimizers,
    get_optimizer_config
)
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.utils.transformations import transform_params, untransform_params
from bellman_filter_dfsv.utils.optimization_helpers import create_stable_initial_params # Added import

# Enable float64 for tests
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def simple_model_params():
    """Create a simple DFSV model with one factor."""
    # Define model dimensions
    N = 3  # Number of observed series
    K = 1  # Number of factors

    # Factor loadings
    lambda_r = jnp.array([[0.9], [0.6], [0.3]]) # Use jnp

    # Factor persistence
    Phi_f = jnp.array([[0.95]]) # Use jnp

    # Log-volatility persistence
    Phi_h = jnp.array([[0.98]]) # Use jnp

    # Long-run mean for log-volatilities
    mu = jnp.array([-1.0]) # Use jnp

    # Idiosyncratic variance (diagonal)
    sigma2 = jnp.array([0.1, 0.1, 0.1]) # Use jnp

    # Log-volatility noise covariance
    Q_h = jnp.array([[0.1]]) # Use jnp

    # Create parameter object using the standard dataclass
    params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=lambda_r,
        Phi_f=Phi_f,
        Phi_h=Phi_h,
        mu=mu,
        sigma2=sigma2,
        Q_h=Q_h,
    )

    return params


@pytest.fixture
def simulated_data(simple_model_params):
    """Generate simulated data for testing."""
    # Use a short time series for testing
    T = 100
    returns, factors, log_vols = simulate_DFSV(
        params=simple_model_params,
        T=T,
        seed=42
    )
    return returns, factors, log_vols


def test_get_available_optimizers():
    """Test that get_available_optimizers returns a dictionary of available optimizers."""
    optimizers = get_available_optimizers()

    # Check that the result is a dictionary
    assert isinstance(optimizers, dict)

    # Check that the dictionary contains the expected optimizers
    expected_optimizers = [
        "BFGS", "Adam", "AdamW", "RMSProp", "SGD", "Adagrad", "Adadelta",
        "DampedTrustRegionBFGS", "ArmijoBFGS", "DogLegBFGS"
    ]

    for optimizer in expected_optimizers:
        assert optimizer in optimizers
        assert isinstance(optimizers[optimizer], str)


def test_get_optimizer_config():
    """Test that get_optimizer_config returns valid configurations."""
    # Test with default parameters
    config = get_optimizer_config("BFGS")
    assert isinstance(config, dict)
    assert "learning_rate" in config
    assert "rtol" in config
    assert "atol" in config

    # Test with custom parameters
    custom_config = get_optimizer_config("Adam", learning_rate=0.01, rtol=1e-4, atol=1e-4)
    assert custom_config["learning_rate"] == 0.01
    assert custom_config["rtol"] == 1e-4
    assert custom_config["atol"] == 1e-4

    # Test with unknown optimizer
    with pytest.raises(ValueError):
        get_optimizer_config("UnknownOptimizer")


def test_create_optimizer():
    """Test that create_optimizer returns the correct optimizer instances."""
    # Test creating different optimizers
    optimizers = [
        "BFGS", "Adam", "AdamW", "RMSProp", "SGD", "Adagrad", "Adadelta",
        "DampedTrustRegionBFGS", "ArmijoBFGS", "DogLegBFGS"
    ]

    for optimizer_name in optimizers:
        optimizer = create_optimizer(optimizer_name)

        # Check that the optimizer is an instance of the correct class
        if optimizer_name in ["BFGS", "DampedTrustRegionBFGS", "ArmijoBFGS", "DogLegBFGS"]:
            assert isinstance(optimizer, optx.AbstractMinimiser)
        else:
            assert isinstance(optimizer, optx.OptaxMinimiser)

    # Test with unknown optimizer
    with pytest.raises(ValueError):
        create_optimizer("UnknownOptimizer")


def test_create_filter():
    """Test that create_filter returns the correct filter instances."""
    # Test creating different filters
    N, K = 3, 1

    # BIF filter
    bif_filter = create_filter(FilterType.BIF, N, K)
    assert bif_filter.N == N
    assert bif_filter.K == K

    # BF filter
    bf_filter = create_filter(FilterType.BF, N, K)
    assert bf_filter.N == N
    assert bf_filter.K == K

    # PF filter
    num_particles = 100
    pf_filter = create_filter(FilterType.PF, N, K, num_particles)
    assert pf_filter.N == N
    assert pf_filter.K == K
    assert pf_filter.num_particles == num_particles

    # Test with unknown filter type
    with pytest.raises(ValueError):
        create_filter("UnknownFilter", N, K)


def test_get_objective_function():
    """Test that get_objective_function returns a valid objective function."""
    # Create a filter
    N, K = 3, 1
    filter_instance = create_filter(FilterType.BIF, N, K)

    # Get objective function
    obj_fn = get_objective_function(
        filter_type=FilterType.BIF,
        filter_instance=filter_instance,
        stability_penalty_weight=1000.0,
        priors=None,
        is_transformed=False
    )

    # Check that the objective function has the correct signature
    assert callable(obj_fn)

    # Create a simple parameter object
    params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=jnp.ones((N, K)),
        Phi_f=0.8 * jnp.eye(K),
        Phi_h=0.8 * jnp.eye(K),
        mu=jnp.zeros(K),
        sigma2=0.1 * jnp.ones(N),
        Q_h=0.2 * jnp.eye(K)
    )

    # Create some dummy observations
    observations = jnp.ones((10, N))

    # Call the objective function
    loss, aux = obj_fn(params, observations)

    # Check that the loss is a scalar
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()

    # Check that aux is None
    assert aux is None

    # Test with unknown filter type
    with pytest.raises(ValueError):
        get_objective_function(
            filter_type="UnknownFilter",
            filter_instance=filter_instance
        )


def test_minimize_with_logging_quadratic():
    """Test minimize_with_logging with a simple quadratic function."""
    # Define a simple quadratic function
    def quadratic_fn(params, args=None):
        x, y = params
        return (x - 1.0) ** 2 + (y - 2.0) ** 2, None

    # Create optimizer
    optimizer = create_optimizer("BFGS")

    # Initial parameters
    initial_params = jnp.array([0.0, 0.0])

    # Run optimization with logging
    sol, param_history = minimize_with_logging(
        objective_fn=quadratic_fn,
        initial_params=initial_params,
        solver=optimizer,
        max_steps=20,
        log_interval=1,
        options={}
    )

    # Check that the solution is close to the true minimum
    assert jnp.allclose(sol.value, jnp.array([1.0, 2.0]), atol=1e-2)

    # Check that the parameter history has the correct length
    assert len(param_history) > 1

    # Check that the final loss is close to zero
    final_loss, _ = quadratic_fn(sol.value)
    assert jnp.allclose(final_loss, 0.0, atol=1e-4)


def test_minimize_with_logging_rosenbrock():
    """Test minimize_with_logging with the Rosenbrock function."""
    # Define the Rosenbrock function
    def rosenbrock_fn(params, args=None):
        x, y = params
        return 100.0 * (y - x**2)**2 + (1 - x)**2, None

    # Create optimizer
    optimizer = create_optimizer("BFGS")

    # Initial parameters
    initial_params = jnp.array([0.0, 0.0])

    # Run optimization with logging
    sol, param_history = minimize_with_logging(
        objective_fn=rosenbrock_fn,
        initial_params=initial_params,
        solver=optimizer,
        max_steps=50,
        log_interval=1,
        options={}
    )

    # Check that the solution is close to the true minimum
    assert jnp.allclose(sol.value, jnp.array([1.0, 1.0]), atol=1e-1)

    # Check that the parameter history has the correct length
    assert len(param_history) > 1

    # Check that the final loss is close to zero
    final_loss, _ = rosenbrock_fn(sol.value)
    assert final_loss < 1.0  # Rosenbrock can be challenging, so we use a looser tolerance


def test_minimize_with_logging_error_handling():
    """Test that minimize_with_logging handles errors correctly."""
    # Define a function that raises an error
    def error_fn(params, args=None):
        if params[0] > 0.5:
            raise ValueError("Test error")
        return jnp.sum(params**2), None

    # Create optimizer
    optimizer = create_optimizer("BFGS")

    # Initial parameters that will trigger the error immediately
    initial_params = jnp.array([1.0, 0.0])

    # Run optimization with logging, but don't throw errors
    sol, param_history = minimize_with_logging(
        objective_fn=error_fn,
        initial_params=initial_params,
        solver=optimizer,
        max_steps=20,
        log_interval=1,
        options={},
        throw=False
    )

    # Check that the solution is returned even though an error occurred
    assert sol is not None
    assert param_history is not None

    # Run optimization with logging, and throw errors
    with pytest.raises(ValueError):
        minimize_with_logging(
            objective_fn=error_fn,
            initial_params=initial_params,
            solver=optimizer,
            max_steps=20,
            log_interval=1,
            options={},
            throw=True
        )


def test_run_optimization_bif(simulated_data):
    """Test run_optimization with BIF filter."""
    returns, _, _ = simulated_data
    N, K = returns.shape[1], 1 # Assuming K=1 based on simple_model_params
    initial_params = create_stable_initial_params(N, K)

    # Run optimization with BIF filter
    result = run_optimization(
        filter_type=FilterType.BIF,
        returns=returns,
        initial_params=initial_params, # Added
        optimizer_name="BFGS",
        max_steps=5,  # Use a small number of steps for testing
        fix_mu=False, # Added: Avoid error when true_mu is None
        verbose=True
    )

    # Check that the result is an OptimizerResult
    assert isinstance(result, OptimizerResult)

    # Check that the result contains the expected fields
    assert result.filter_type == FilterType.BIF
    assert result.optimizer_name == "BFGS"
    assert isinstance(result.final_params, DFSVParamsDataclass)
    assert len(result.param_history) >= 1 # Changed > to >= for default log_params=False
    assert len(result.loss_history) >= 1 # Changed > 0 to >= 1 for consistency

    # Check that the final parameters have the correct shape
    N, K = returns.shape[1], 1
    assert result.final_params.lambda_r.shape == (N, K)
    assert result.final_params.Phi_f.shape == (K, K)
    assert result.final_params.Phi_h.shape == (K, K)
    assert result.final_params.mu.shape == (K,)
    assert result.final_params.sigma2.shape == (N,)
    assert result.final_params.Q_h.shape == (K, K)


def test_run_optimization_bf(simulated_data):
    """Test run_optimization with BF filter."""
    returns, _, _ = simulated_data
    N, K = returns.shape[1], 1 # Assuming K=1 based on simple_model_params
    initial_params = create_stable_initial_params(N, K)

    # Run optimization with BF filter
    result = run_optimization(
        filter_type=FilterType.BF,
        returns=returns,
        initial_params=initial_params, # Added
        optimizer_name="Adam",  # Use Adam for better stability
        max_steps=5,  # Use a small number of steps for testing
        fix_mu=False, # Added: Avoid error when true_mu is None
        verbose=True
    )

    # Check that the result is an OptimizerResult
    assert isinstance(result, OptimizerResult)

    # Check that the result contains the expected fields
    assert result.filter_type == FilterType.BF
    assert result.optimizer_name == "Adam"
    assert isinstance(result.final_params, DFSVParamsDataclass)

    # Check that the final parameters have the correct shape
    N, K = returns.shape[1], 1
    assert result.final_params.lambda_r.shape == (N, K)
    assert result.final_params.Phi_f.shape == (K, K)
    assert result.final_params.Phi_h.shape == (K, K)
    assert result.final_params.mu.shape == (K,)
    assert result.final_params.sigma2.shape == (N,)
    assert result.final_params.Q_h.shape == (K, K)


def test_run_optimization_pf(simulated_data):
    """Test run_optimization with PF filter."""
    returns, _, _ = simulated_data
    N, K = returns.shape[1], 1 # Assuming K=1 based on simple_model_params
    initial_params = create_stable_initial_params(N, K)

    # Run optimization with PF filter
    result = run_optimization(
        filter_type=FilterType.PF,
        returns=returns,
        initial_params=initial_params, # Added
        optimizer_name="Adam",  # Use Adam for better stability
        max_steps=5,  # Use a small number of steps for testing
        num_particles=100,  # Use a small number of particles for testing
        fix_mu=False, # Added: Avoid error when true_mu is None
        verbose=True
    )

    # Check that the result is an OptimizerResult
    assert isinstance(result, OptimizerResult)

    # Check that the result contains the expected fields
    assert result.filter_type == FilterType.PF
    assert result.optimizer_name == "Adam"
    assert isinstance(result.final_params, DFSVParamsDataclass)

    # Check that the final parameters have the correct shape
    N, K = returns.shape[1], 1
    assert result.final_params.lambda_r.shape == (N, K)
    assert result.final_params.Phi_f.shape == (K, K)
    assert result.final_params.Phi_h.shape == (K, K)
    assert result.final_params.mu.shape == (K,)
    assert result.final_params.sigma2.shape == (N,)
    assert result.final_params.Q_h.shape == (K, K)


def test_run_optimization_with_transformations(simulated_data):
    """Test run_optimization with parameter transformations."""
    returns, _, _ = simulated_data
    N, K = returns.shape[1], 1 # Assuming K=1 based on simple_model_params
    initial_params = create_stable_initial_params(N, K)

    # Run optimization with transformations
    result_transformed = run_optimization(
        filter_type=FilterType.BIF,
        returns=returns,
        initial_params=initial_params, # Added
        optimizer_name="BFGS",
        use_transformations=True,
        max_steps=5,  # Use a small number of steps for testing
        fix_mu=False, # Added: Avoid error when true_mu is None
        verbose=True
    )

    # Run optimization without transformations
    result_untransformed = run_optimization(
        filter_type=FilterType.BIF,
        returns=returns,
        initial_params=initial_params, # Added
        optimizer_name="BFGS",
        use_transformations=False,
        max_steps=5,  # Use a small number of steps for testing
        fix_mu=False, # Added: Avoid error when true_mu is None
        verbose=True
    )

    # Check that both results are OptimizerResults
    assert isinstance(result_transformed, OptimizerResult)
    assert isinstance(result_untransformed, OptimizerResult)

    # Check that the uses_transformations field is set correctly
    assert result_transformed.uses_transformations is True
    assert result_untransformed.uses_transformations is False

    # Check that the final parameters have the correct shape
    N, K = returns.shape[1], 1
    assert result_transformed.final_params.lambda_r.shape == (N, K)
    assert result_untransformed.final_params.lambda_r.shape == (N, K)


def test_run_optimization_with_true_params(simple_model_params, simulated_data):
    """Test run_optimization with true parameters."""
    returns, _, _ = simulated_data
    # Use true_params as initial_params for this test
    initial_params = simple_model_params

    # Run optimization with true parameters
    result = run_optimization(
        filter_type=FilterType.BIF,
        returns=returns,
        initial_params=initial_params, # Added
        true_params=simple_model_params,
        optimizer_name="BFGS",
        max_steps=5,  # Use a small number of steps for testing
        verbose=True
    )

    # Check that the result is an OptimizerResult
    assert isinstance(result, OptimizerResult)

    # Check that the fix_mu field is set correctly
    # When true_params is provided, fix_mu defaults to True in run_optimization
    assert result.fix_mu is True # Changed assertion

    # Check that the final parameters have the correct shape
    N, K = returns.shape[1], 1
    assert result.final_params.lambda_r.shape == (N, K)
    assert result.final_params.Phi_f.shape == (K, K)
    assert result.final_params.Phi_h.shape == (K, K)
    assert result.final_params.mu.shape == (K,)
    assert result.final_params.sigma2.shape == (N,)
    assert result.final_params.Q_h.shape == (K, K)


def test_run_optimization_with_priors(simulated_data):
    """Test run_optimization with priors."""
    returns, _, _ = simulated_data
    N, K = returns.shape[1], 1 # Assuming K=1 based on simple_model_params
    initial_params = create_stable_initial_params(N, K)

    # Define priors
    priors = {
        "mu_mean": jnp.array([-1.0]),
        "mu_std": jnp.array([0.5])
    }

    # Run optimization with priors
    result = run_optimization(
        filter_type=FilterType.BIF,
        returns=returns,
        initial_params=initial_params, # Added
        optimizer_name="BFGS",
        priors=priors,
        prior_config_name="Test Priors",
        max_steps=5,  # Use a small number of steps for testing
        fix_mu=False, # Added: Avoid error when true_mu is None
        verbose=True
    )

    # Check that the result is an OptimizerResult
    assert isinstance(result, OptimizerResult)

    # Check that the prior_config_name field is set correctly
    assert result.prior_config_name == "Test Priors"

    # Check that the final parameters have the correct shape
    N, K = returns.shape[1], 1
    assert result.final_params.lambda_r.shape == (N, K)
    assert result.final_params.Phi_f.shape == (K, K)
    assert result.final_params.Phi_h.shape == (K, K)
    assert result.final_params.mu.shape == (K,)
    assert result.final_params.sigma2.shape == (N,)
    assert result.final_params.Q_h.shape == (K, K)


def test_run_optimization_error_handling():
    """Test that run_optimization handles errors correctly."""
    # Create invalid returns data (NaN values)
    invalid_returns = jnp.ones((10, 3)) * jnp.nan  # Contains NaN values
    N, K = 3, 1 # Define N, K for initial params
    initial_params = create_stable_initial_params(N, K)

    # Run optimization with invalid data
    result = run_optimization(
        filter_type=FilterType.BIF,
        returns=invalid_returns,
        initial_params=initial_params, # Added
        optimizer_name="BFGS",
        max_steps=5,
        fix_mu=False, # Added: Avoid error when true_mu is None
        verbose=True
    )

    # Check that the result is an OptimizerResult
    assert isinstance(result, OptimizerResult)

    # Check that the success field is set to False
    assert result.success is False

    # Check that the error_message field is not None
    assert result.error_message is not None
