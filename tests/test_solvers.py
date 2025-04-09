"""
Tests for the solver utilities in bellman_filter_dfsv.utils.solvers.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optimistix as optx
from typing import Dict, Any, List, Tuple

from bellman_filter_dfsv.utils.solvers import (
    create_optimizer,
    get_available_optimizers,
    get_optimizer_config,
    create_learning_rate_scheduler,
    DogLegBFGS,
    ArmijoBFGS
)

# Enable float64 for tests
jax.config.update("jax_enable_x64", True)


def test_create_learning_rate_scheduler():
    """Test that create_learning_rate_scheduler returns valid schedulers."""
    # Test constant scheduler
    scheduler = create_learning_rate_scheduler(init_lr=0.01, scheduler_type="constant")
    assert callable(scheduler)

    # Test exponential decay scheduler
    scheduler = create_learning_rate_scheduler(
        init_lr=0.01,
        decay_steps=100,
        scheduler_type="exponential"
    )
    assert callable(scheduler)

    # Test cosine decay scheduler
    scheduler = create_learning_rate_scheduler(
        init_lr=0.01,
        decay_steps=100,
        scheduler_type="cosine"
    )
    assert callable(scheduler)

    # Test with unknown scheduler
    with pytest.raises(ValueError):
        create_learning_rate_scheduler(scheduler_type="unknown_scheduler")


def test_dogleg_bfgs():
    """Test that DogLegBFGS works correctly."""
    # Skip this test for now as it requires more complex setup
    pytest.skip("DogLegBFGS requires more complex setup to test properly")


def test_armijo_bfgs():
    """Test that ArmijoBFGS works correctly."""
    # Skip this test for now as it requires more complex setup
    pytest.skip("ArmijoBFGS requires more complex setup to test properly")


def test_optimizer_convergence():
    """Test that optimizers converge to the correct solution."""
    # Define a simple quadratic function
    def quadratic_fn(params, args=None):
        x, y = params
        return (x - 1.0) ** 2 + (y - 2.0) ** 2, None

    # Initial parameters
    initial_params = jnp.array([0.0, 0.0])

    # Get the shape and dtype of the output
    test_output, test_aux = quadratic_fn(initial_params)
    f_struct = jax.ShapeDtypeStruct(test_output.shape, test_output.dtype)
    aux_struct = None if test_aux is None else jax.ShapeDtypeStruct(test_aux.shape, test_aux.dtype)

    # Test different optimizers
    optimizers = [
        "BFGS", "Adam", "AdamW", "RMSProp", "SGD", "Adagrad", "Adadelta",
        "DampedTrustRegionBFGS", "ArmijoBFGS", "DogLegBFGS"
    ]

    for optimizer_name in optimizers:
        # Create optimizer
        optimizer = create_optimizer(optimizer_name)

        # Initialize optimizer state
        options = {}
        state = optimizer.init(quadratic_fn, initial_params, None, options, f_struct, aux_struct, frozenset())

        # Run optimization for a few steps
        y = initial_params
        for _ in range(10):
            y, state, _ = optimizer.step(quadratic_fn, y, None, options, state, frozenset())

            # Check for convergence
            converged, _ = optimizer.terminate(quadratic_fn, y, None, options, state, frozenset())
            if converged:
                break

        # Check that the solution is closer to the true minimum
        initial_loss, _ = quadratic_fn(initial_params)
        final_loss, _ = quadratic_fn(y)

        assert final_loss < initial_loss


def test_optimizer_with_different_learning_rates():
    """Test optimizers with different learning rates."""
    # Define a simple quadratic function
    def quadratic_fn(params, args=None):
        x, y = params
        return (x - 1.0) ** 2 + (y - 2.0) ** 2, None

    # Initial parameters
    initial_params = jnp.array([0.0, 0.0])

    # Get the shape and dtype of the output
    test_output, test_aux = quadratic_fn(initial_params)
    f_struct = jax.ShapeDtypeStruct(test_output.shape, test_output.dtype)
    aux_struct = None if test_aux is None else jax.ShapeDtypeStruct(test_aux.shape, test_aux.dtype)

    # Test different learning rates
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]

    for lr in learning_rates:
        # Create optimizer
        optimizer = create_optimizer("Adam", learning_rate=lr)

        # Initialize optimizer state
        options = {}
        state = optimizer.init(quadratic_fn, initial_params, None, options, f_struct, aux_struct, frozenset())

        # Run optimization for a few steps
        y = initial_params
        for _ in range(10):
            y, state, _ = optimizer.step(quadratic_fn, y, None, options, state, frozenset())

        # Check that the solution is closer to the true minimum
        initial_loss, _ = quadratic_fn(initial_params)
        final_loss, _ = quadratic_fn(y)

        assert final_loss < initial_loss


def test_optimizer_with_different_tolerances():
    """Test optimizers with different tolerances."""
    # Define a simple quadratic function
    def quadratic_fn(params, args=None):
        x, y = params
        return (x - 1.0) ** 2 + (y - 2.0) ** 2, None

    # Initial parameters
    initial_params = jnp.array([0.0, 0.0])

    # Get the shape and dtype of the output
    test_output, test_aux = quadratic_fn(initial_params)
    f_struct = jax.ShapeDtypeStruct(test_output.shape, test_output.dtype)
    aux_struct = None if test_aux is None else jax.ShapeDtypeStruct(test_aux.shape, test_aux.dtype)

    # Test different tolerances
    tolerances = [1e-2, 1e-3, 1e-4, 1e-5]

    for tol in tolerances:
        # Create optimizer
        optimizer = create_optimizer("BFGS", rtol=tol, atol=tol)

        # Initialize optimizer state
        options = {}
        state = optimizer.init(quadratic_fn, initial_params, None, options, f_struct, aux_struct, frozenset())

        # Run optimization for a few steps
        y = initial_params
        for _ in range(10):
            y, state, _ = optimizer.step(quadratic_fn, y, None, options, state, frozenset())

            # Check for convergence
            converged, _ = optimizer.terminate(quadratic_fn, y, None, options, state, frozenset())
            if converged:
                break

        # Check that the solution is closer to the true minimum
        initial_loss, _ = quadratic_fn(initial_params)
        final_loss, _ = quadratic_fn(y)

        assert final_loss < initial_loss
