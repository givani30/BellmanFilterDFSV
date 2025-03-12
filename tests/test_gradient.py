import numpy as np
import jax
import jax.numpy as jnp
import pytest
import sys
import os

# Add the parent directory to the path so we can import the functions module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.filters import DFSVBellmanFilter
from functions.simulation import DFSV_params, simulate_DFSV


def numerical_gradient(func, x, *args, h=1e-6):
    """
    Compute numerical gradient of func at x using central difference formula.

    Parameters:
    -----------
    func : callable
        Function for which to compute the gradient
    x : numpy.ndarray
        Point at which to compute the gradient
    args : tuple
        Additional arguments to pass to func
    h : float, optional
        Step size for finite difference, default is 1e-6

    Returns:
    --------
    numpy.ndarray
        Numerical gradient of func at x
    """
    grad = np.zeros_like(x)
    x_flat = x.flatten()

    for i in range(len(x_flat)):
        # Forward point
        x_forward = x_flat.copy()
        x_forward[i] += h
        f_forward = func(x_forward.reshape(x.shape), *args)

        # Backward point
        x_backward = x_flat.copy()
        x_backward[i] -= h
        f_backward = func(x_backward.reshape(x.shape), *args)

        # Central difference
        grad[i] = (f_forward - f_backward) / (2 * h)

    return grad


def setup_test_case(N=3, K=2, T=50, seed=42):
    """
    Set up a test case with simulated data and initialized Bellman filter.

    Returns:
    --------
    tuple
        (filter_instance, observation, predicted_state, I_pred)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Define model dimensions
    N = 3  # Number of observed series
    K = 2  # Number of factors

    # Factor loadings
    lambda_r = np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]])

    # Factor persistence
    Phi_f = np.array([[0.2, 0.005], [0.005, 0.2]])

    # Log-volatility persistence
    Phi_h = np.array([[0.95, 0.00], [0.00, 0.95]])

    # Log-volatility long-run mean
    mu = np.array([-1.0, -0.5])

    # Idiosyncratic variance (diagonal)
    sigma2 = np.array([0.1, 0.1, 0.1])

    # Log-volatility noise covariance
    Q_h = np.array([[0.1, 0.02], [0.02, 0.1]])

    # Create parameter object
    params = DFSV_params(
        N=N,
        K=K,
        lambda_r=lambda_r,
        Phi_f=Phi_f,
        Phi_h=Phi_h,
        mu=mu,
        sigma2=sigma2,
        Q_h=Q_h,
    )

    # Generate synthetic data
    y, _, _ = simulate_DFSV(params, T=T)

    # Create and initialize filter
    bellman_filter = DFSVBellmanFilter(params)
    state, cov = bellman_filter.initialize_state(y)

    # Run one prediction step
    predicted_state, predicted_cov = bellman_filter.predict(state, cov)

    # Convert to information form
    I_pred = np.linalg.inv(predicted_cov)

    # Get fifth observation
    observation = y[4:5, :]

    return bellman_filter, observation, predicted_state, I_pred


def test_gradient_accuracy():
    """
    Test the accuracy of the gradient computation by comparing the explicit gradient
    with numerical approximation.
    """
    # Set up test case
    bellman_filter, observation, predicted_state, I_pred = setup_test_case()

    # Convert numpy arrays to JAX arrays for the filter's functions
    jax_predicted_state = jnp.array(predicted_state)
    jax_observation = jnp.array(observation)
    jax_I_pred = jnp.array(I_pred)

    # Initial state to evaluate gradient at (use predicted state as starting point)
    x0 = np.array(predicted_state).flatten()

    # Define the objective function for numerical gradient computation
    def objective_fn(x, ps, ip, obs, lr, sig2):
        return float(bellman_filter.obj_and_grad_fn(x, ps, ip, obs, lr, sig2)[0])

    # Compute analytical gradient
    _, analytical_grad = bellman_filter.obj_and_grad_fn(
        x0,
        jax_predicted_state,
        jax_I_pred,
        jax_observation,
        bellman_filter.jax_lambda_r,
        bellman_filter.jax_sigma2,
    )

    # Compute numerical gradient
    numerical_grad = numerical_gradient(
        objective_fn,
        x0,
        jax_predicted_state,
        jax_I_pred,
        jax_observation,
        bellman_filter.jax_lambda_r,
        bellman_filter.jax_sigma2,
    )

    # Convert JAX arrays to numpy for comparison
    analytical_grad = np.array(analytical_grad)

    # Compute relative error
    abs_diff = np.abs(analytical_grad - numerical_grad)
    abs_max = np.maximum(np.abs(analytical_grad), np.abs(numerical_grad))
    # Avoid division by zero
    abs_max = np.where(abs_max < 1e-15, 1.0, abs_max)
    relative_error = abs_diff / abs_max

    # Print comparison for debugging
    print("\nGradient comparison:")
    print("Analytical gradient:", analytical_grad)
    print("Numerical gradient:", numerical_grad)
    print("Absolute difference:", abs_diff)
    print("Relative error:", relative_error)

    # Assert that the maximum relative error is small
    max_rel_error = np.max(relative_error)
    print(f"Maximum relative error: {max_rel_error:.6e}")

    # Use a relatively loose tolerance due to numerical issues
    assert max_rel_error < 0.01, f"Gradient error too large: {max_rel_error:.6e}"


def test_optimizer_with_gradient():
    """
    Test that the optimization converges properly using the analytical gradient.
    """
    import scipy.optimize

    # Set up test case
    bellman_filter, observation, predicted_state, I_pred = setup_test_case()

    # Convert numpy arrays to JAX arrays for the filter's functions
    jax_predicted_state = jnp.array(predicted_state)
    jax_observation = jnp.array(observation)
    jax_I_pred = jnp.array(I_pred)

    # Initial state to evaluate gradient at (use predicted state as starting point)
    x0 = np.array(predicted_state).flatten()

    # Define the objective function and gradient for scipy optimizer
    def objective_wrapper(x):
        obj_val, grad_val = bellman_filter.obj_and_grad_fn(
            x,
            jax_predicted_state,
            jax_I_pred,
            jax_observation,
            bellman_filter.jax_lambda_r,
            bellman_filter.jax_sigma2,
        )
        return float(obj_val), np.array(grad_val)

    # Optimize using analytical gradient
    result_with_grad = scipy.optimize.minimize(
        fun=lambda x: objective_wrapper(x)[0],
        x0=x0,
        jac=lambda x: objective_wrapper(x)[1],
        method="BFGS",
        options={"gtol": 1e-4, "disp": False},
    )

    # Optimize using numerical gradient
    result_numerical = scipy.optimize.minimize(
        fun=lambda x: objective_wrapper(x)[0],
        x0=x0,
        method="BFGS",
        options={"gtol": 1e-4, "disp": False},
    )

    print("\nOptimization Results:")
    print(
        f"With analytical gradient - success: {result_with_grad.success}, iterations: {result_with_grad.nit}, value: {result_with_grad.fun:.6e}"
    )
    print(
        f"With numerical gradient - success: {result_numerical.success}, iterations: {result_numerical.nit}, value: {result_numerical.fun:.6e}"
    )

    # Check that both optimizations succeeded
    assert result_with_grad.success, "Optimization with analytical gradient failed"
    assert result_numerical.success, "Optimization with numerical gradient failed"

    # Check that they reached similar function values
    rel_diff = abs(result_with_grad.fun - result_numerical.fun) / max(
        abs(result_with_grad.fun), abs(result_numerical.fun)
    )
    print(f"Relative difference in function values: {rel_diff:.6e}")
    assert rel_diff < 0.01, f"Optimizations reached different values: {rel_diff:.6e}"


if __name__ == "__main__":
    # Enable 64-bit precision for JAX
    jax.config.update("jax_enable_x64", True)

    # Run the tests
    test_gradient_accuracy()
    test_optimizer_with_gradient()
    print("All tests passed!")
