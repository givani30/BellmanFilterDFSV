"""
Example demonstrating parameter optimization for DFSV models.

This script shows how to optimize model parameters using the JAX-compatible
parameter classes and automatic differentiation through the Bellman filter.
"""

import sys
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import parameter classes
from functions.simulation import DFSV_params, simulate_DFSV
from functions.jax_params import DFSVParamsPytree
from functions.bellman_filter import DFSVBellmanFilter

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)


def create_simple_model():
    """Create a simple DFSV model with one factor."""
    # Define model dimensions
    N = 3  # Number of observed series
    K = 1  # Number of factors

    # Factor loadings
    lambda_r = np.array([[0.9], [0.6], [0.3]])

    # Factor persistence
    Phi_f = np.array([[0.95]])

    # Log-volatility persistence
    Phi_h = np.array([[0.98]])

    # Long-run mean for log-volatilities
    mu = np.array([-1.0])

    # Idiosyncratic variance (diagonal)
    sigma2 = np.array([0.1, 0.1, 0.1])

    # Log-volatility noise covariance
    Q_h = np.array([[0.05]])

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

    return params


def create_training_data(params, T=100, seed=42):
    """Generate simulated data for training."""
    returns, factors, log_vols = simulate_DFSV(params, T=T, seed=seed)
    return returns, factors, log_vols


def objective_function(param_values, param_structure, observations, bf):
    """
    Define the objective function for parameter optimization.

    Args:
        param_values: Flattened parameter values for optimization
        param_structure: Original parameter structure to know where values go
        observations: Observation data
        bf: Bellman filter instance

    Returns:
        float: Negative log-likelihood (for minimization)
    """
    # Reconstruct parameter object from flattened values
    new_params = reconstruct_params(param_values, param_structure)

    # Calculate log-likelihood using the static JIT function
    # This avoids issues with trying to JIT-compile the filter instance
    _, _, log_lik = bf.filter_scan(new_params, observations)

    # Return negative log-likelihood for minimization
    return -log_lik


def flatten_params(
    params,
    optimize_lambda=True,
    optimize_phi=True,
    optimize_mu=True,
    optimize_sigma2=True,
    optimize_qh=True,
):
    """
    Flatten parameters to a vector for optimization.

    Args:
        params: DFSVParamsPytree object
        optimize_*: Flags to determine which parameters to optimize

    Returns:
        tuple: Flattened parameter vector and parameter structure dictionary
    """
    param_vector = []
    param_structure = {
        "N": params.N,
        "K": params.K,
        "to_optimize": {
            "lambda_r": optimize_lambda,
            "Phi_f": optimize_phi,
            "Phi_h": optimize_phi,
            "mu": optimize_mu,
            "sigma2": optimize_sigma2,
            "Q_h": optimize_qh,
        },
    }

    # For each parameter, add to vector if optimized
    if optimize_lambda:
        param_structure["lambda_r_shape"] = params.lambda_r.shape
        param_vector.extend(params.lambda_r.flatten())
    else:
        param_structure["lambda_r"] = params.lambda_r

    if optimize_phi:
        param_structure["Phi_f_shape"] = params.Phi_f.shape
        param_structure["Phi_h_shape"] = params.Phi_h.shape
        param_vector.extend(params.Phi_f.flatten())
        param_vector.extend(params.Phi_h.flatten())
    else:
        param_structure["Phi_f"] = params.Phi_f
        param_structure["Phi_h"] = params.Phi_h

    if optimize_mu:
        param_structure["mu_shape"] = params.mu.shape
        param_vector.extend(params.mu.flatten())
    else:
        param_structure["mu"] = params.mu

    if optimize_sigma2:
        param_structure["sigma2_shape"] = params.sigma2.shape
        param_vector.extend(params.sigma2.flatten())
    else:
        param_structure["sigma2"] = params.sigma2

    if optimize_qh:
        param_structure["Q_h_shape"] = params.Q_h.shape
        param_vector.extend(params.Q_h.flatten())
    else:
        param_structure["Q_h"] = params.Q_h

    return jnp.array(param_vector), param_structure


def reconstruct_params(param_values, param_structure):
    """
    Reconstruct parameter object from flattened values.

    Args:
        param_values: Flattened parameter vector
        param_structure: Parameter structure dictionary

    Returns:
        DFSVParamsPytree: Reconstructed parameter object
    """
    # Extract N and K from structure
    N = param_structure["N"]
    K = param_structure["K"]
    to_optimize = param_structure["to_optimize"]

    # Initialize parameter dictionaries
    param_dict = {}

    # Index to keep track of where we are in the param_values vector
    idx = 0

    # Reconstruct lambda_r
    if to_optimize["lambda_r"]:
        shape = param_structure["lambda_r_shape"]
        size = np.prod(shape)
        param_dict["lambda_r"] = param_values[idx : idx + size].reshape(shape)
        idx += size
    else:
        param_dict["lambda_r"] = param_structure["lambda_r"]

    # Reconstruct Phi_f and Phi_h
    if to_optimize["Phi_f"]:
        shape = param_structure["Phi_f_shape"]
        size = np.prod(shape)
        param_dict["Phi_f"] = param_values[idx : idx + size].reshape(shape)
        idx += size

        shape = param_structure["Phi_h_shape"]
        size = np.prod(shape)
        param_dict["Phi_h"] = param_values[idx : idx + size].reshape(shape)
        idx += size
    else:
        param_dict["Phi_f"] = param_structure["Phi_f"]
        param_dict["Phi_h"] = param_structure["Phi_h"]

    # Reconstruct mu
    if to_optimize["mu"]:
        shape = param_structure["mu_shape"]
        size = np.prod(shape)
        param_dict["mu"] = param_values[idx : idx + size].reshape(shape)
        idx += size
    else:
        param_dict["mu"] = param_structure["mu"]

    # Reconstruct sigma2
    if to_optimize["sigma2"]:
        shape = param_structure["sigma2_shape"]
        size = np.prod(shape)
        param_dict["sigma2"] = param_values[idx : idx + size].reshape(shape)
        idx += size
    else:
        param_dict["sigma2"] = param_structure["sigma2"]

    # Reconstruct Q_h
    if to_optimize["Q_h"]:
        shape = param_structure["Q_h_shape"]
        size = np.prod(shape)
        param_dict["Q_h"] = param_values[idx : idx + size].reshape(shape)
    else:
        param_dict["Q_h"] = param_structure["Q_h"]

    # Create DFSVParamsPytree object
    return DFSVParamsPytree(
        N=N,
        K=K,
        lambda_r=param_dict["lambda_r"],
        Phi_f=param_dict["Phi_f"],
        Phi_h=param_dict["Phi_h"],
        mu=param_dict["mu"],
        sigma2=param_dict["sigma2"],
        Q_h=param_dict["Q_h"],
    )


def setup_constraints(param_structure):
    """
    Set up parameter constraints for optimization.

    Args:
        param_structure: Parameter structure dictionary

    Returns:
        tuple: Lower and upper bounds for parameters
    """
    to_optimize = param_structure["to_optimize"]
    lower_bounds = []
    upper_bounds = []

    # Lambda_r constraints (positive values make more sense economically)
    if to_optimize["lambda_r"]:
        shape = param_structure["lambda_r_shape"]
        size = np.prod(shape)
        lower_bounds.extend([0.01] * size)  # Keep loadings positive
        upper_bounds.extend([5.0] * size)  # Upper limit on loadings

    # Phi_f and Phi_h constraints (ensure stationarity)
    if to_optimize["Phi_f"]:
        # For Phi_f
        shape = param_structure["Phi_f_shape"]
        size = np.prod(shape)
        lower_bounds.extend([-0.99] * size)  # Allow some negative correlation
        upper_bounds.extend([0.99] * size)  # Ensure stationarity

        # For Phi_h
        shape = param_structure["Phi_h_shape"]
        size = np.prod(shape)
        lower_bounds.extend([0.5] * size)  # SV processes are usually persistent
        upper_bounds.extend([0.999] * size)  # Ensure stationarity

    # mu constraints
    if to_optimize["mu"]:
        shape = param_structure["mu_shape"]
        size = np.prod(shape)
        lower_bounds.extend([-10.0] * size)  # Lower limit on log-volatility mean
        upper_bounds.extend([2.0] * size)  # Upper limit on log-volatility mean

    # sigma2 constraints (ensure positive variance)
    if to_optimize["sigma2"]:
        shape = param_structure["sigma2_shape"]
        size = np.prod(shape)
        lower_bounds.extend([0.001] * size)  # Keep variances positive
        upper_bounds.extend([10.0] * size)  # Upper limit on variances

    # Q_h constraints (ensure positive variance)
    if to_optimize["Q_h"]:
        shape = param_structure["Q_h_shape"]
        size = np.prod(shape)
        lower_bounds.extend([0.001] * size)  # Keep variances positive
        upper_bounds.extend([1.0] * size)  # Upper limit on variances

    return jnp.array(lower_bounds), jnp.array(upper_bounds)


def optimize_params_gradient_descent(
    init_params, observations, bf, n_iterations=300, learning_rate=0.01, verbose=True
):
    """
    Optimize parameters using gradient descent with JAX.

    Args:
        init_params: Initial DFSVParamsPytree object
        observations: Observation data
        bf: Bellman filter instance
        n_iterations: Number of gradient descent iterations
        learning_rate: Learning rate for gradient descent
        verbose: Whether to print progress

    Returns:
        tuple: Optimized parameters and final log-likelihood
    """
    # Flatten parameters for optimization (optimize only lambda_r and mu)
    param_values, param_structure = flatten_params(
        init_params,
        optimize_lambda=True,
        optimize_phi=False,  # Keep transition matrices fixed
        optimize_mu=True,
        optimize_sigma2=True,
        optimize_qh=False,  # Keep volatility noise fixed
    )

    # Get constraints
    lower_bounds, upper_bounds = setup_constraints(param_structure)

    # Define objective function for this specific problem
    def objective(params):
        return objective_function(params, param_structure, observations, bf)

    # JIT-compile the objective function for better performance
    # The filter_instance is treated as static in the internal call
    objective_jit = jax.jit(objective)

    # Define gradient function
    grad_fn = jax.grad(objective_jit)

    # Project parameters to respect constraints
    @jax.jit
    def project_params(params):
        return jnp.clip(params, lower_bounds, upper_bounds)

    # Initialize
    params = param_values
    best_params = params
    best_loss = jnp.inf

    # Optimize using gradient descent
    for i in range(n_iterations):
        # Compute gradient
        grad = grad_fn(params)

        # Update parameters
        params = params - learning_rate * grad

        # Project to respect constraints
        params = project_params(params)

        # Evaluate loss
        loss = objective_jit(params)

        # Check if we improved
        if loss < best_loss:
            best_loss = loss
            best_params = params

        # Print progress
        if verbose and (i % 20 == 0 or i == n_iterations - 1):
            print(f"Iteration {i}: Loss = {loss:.6f}, Best loss = {best_loss:.6f}")

    # Reconstruct best parameters
    optimized_params = reconstruct_params(best_params, param_structure)

    # Return optimized parameters and final log-likelihood
    return optimized_params, -best_loss


def main():
    # Create true model parameters
    true_params = create_simple_model()
    print(f"Created true model with {true_params.N} series and {true_params.K} factor")

    # Generate training data
    T = 500
    returns, true_factors, true_log_vols = create_training_data(true_params, T=T)
    print(f"Generated {T} observations from true model")

    # Create JAX-compatible parameter object
    true_pytree_params = DFSVParamsPytree.from_dfsv_params(true_params)

    # Create perturbed initial parameters for optimization
    perturbed_params = true_pytree_params.replace(
        lambda_r=jnp.array([[0.7], [0.4], [0.2]]),  # Change factor loadings
        mu=jnp.array([-0.5]),  # Change mean log-volatility
        sigma2=jnp.array([0.2, 0.2, 0.2]),  # Change idiosyncratic variances
    )
    print("\nPerturbed parameters for optimization:")
    print(f"lambda_r:\n{perturbed_params.lambda_r}")
    print(f"mu: {perturbed_params.mu}")
    print(f"sigma2: {perturbed_params.sigma2}")

    # Create Bellman filter
    bf = DFSVBellmanFilter(true_params.N, true_params.K)
    print("\nInitialized Bellman filter")

    # Optimize parameters
    print("\nOptimizing parameters...")
    start_time = time.time()
    optimized_params, final_ll = optimize_params_gradient_descent(
        perturbed_params, jnp.array(returns), bf, n_iterations=150, learning_rate=0.02
    )
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")

    # Compare true and optimized parameters
    print("\nParameter comparison:")
    print("True lambda_r:")
    print(true_pytree_params.lambda_r)
    print("Optimized lambda_r:")
    print(optimized_params.lambda_r)
    print("\nTrue mu:", true_pytree_params.mu)
    print("Optimized mu:", optimized_params.mu)
    print("\nTrue sigma2:", true_pytree_params.sigma2)
    print("Optimized sigma2:", optimized_params.sigma2)

    # Test optimized parameters on data
    print("\nRunning filter with optimized parameters...")
    filtered_states, filtered_covs, ll = bf.filter(optimized_params, returns)

    # Calculate parameter recovery metrics
    rmse_lambda = jnp.sqrt(
        jnp.mean((optimized_params.lambda_r - true_pytree_params.lambda_r) ** 2)
    )
    rmse_mu = jnp.sqrt(jnp.mean((optimized_params.mu - true_pytree_params.mu) ** 2))
    rmse_sigma2 = jnp.sqrt(
        jnp.mean((optimized_params.sigma2 - true_pytree_params.sigma2) ** 2)
    )

    print("\nParameter recovery metrics:")
    print(f"RMSE lambda_r: {rmse_lambda:.4f}")
    print(f"RMSE mu: {rmse_mu:.4f}")
    print(f"RMSE sigma2: {rmse_sigma2:.4f}")
    print(f"Final log-likelihood: {final_ll:.4f}")

    # Extract filtered factors and log-volatilities
    filtered_factors = filtered_states[:, : true_params.K]
    filtered_log_vols = filtered_states[:, true_params.K :]

    # Calculate factor and volatility recovery metrics
    corr_factors = jnp.corrcoef(filtered_factors[:, 0], true_factors[:, 0])[0, 1]
    corr_vols = jnp.corrcoef(filtered_log_vols[:, 0], true_log_vols[:, 0])[0, 1]

    print("\nLatent state recovery metrics:")
    print(f"Factor correlation: {corr_factors:.4f}")
    print(f"Log-volatility correlation: {corr_vols:.4f}")


if __name__ == "__main__":
    main()
