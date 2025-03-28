#!/usr/bin/env python
"""
Parameter transformation optimization for DFSV models.

This script implements parameter transformations for DFSV model optimization
to improve numerical stability and convergence by mapping constrained parameters
to unconstrained space.
"""

import copy
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit
import jaxopt
# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import parameter classes
import optax
from jaxopt import OptaxSolver

from functions.bellman_filter import DFSVBellmanFilter
from functions.jax_params import DFSVParamsDataclass
from functions.simulation import DFSV_params, simulate_DFSV

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)


def transform_params(params):
    """
    Transform bounded parameters to unconstrained space for optimization.
    
    Parameters:
    -----------
    params : DFSVParamsDataclass
        Model parameters in their natural (constrained) space
        
    Returns:
    --------
    DFSVParamsDataclass
        Transformed parameters in unconstrained space
    """
    # Create a copy to avoid modifying the original
    result = copy.deepcopy(params)
    
    # Apply logit transform to persistence parameters (bounded in 0,1)
    # logit(p) = log(p/(1-p))
    eps = 1e-6  # Small epsilon to avoid numerical issues at boundaries
    
    # Transform Phi_f (factor persistence)
    phi_f_bounded = jnp.clip(params.Phi_f, eps, 1-eps)
    transformed_phi_f = jnp.log(phi_f_bounded / (1 - phi_f_bounded))
    
    # Transform Phi_h (log-volatility persistence)
    phi_h_bounded = jnp.clip(params.Phi_h, eps, 1-eps)
    transformed_phi_h = jnp.log(phi_h_bounded / (1 - phi_h_bounded))
    
    # Transform variance parameters (must be positive) using log transform
    transformed_sigma2 = jnp.log(jnp.maximum(params.sigma2, eps))
    transformed_q_h = jnp.log(jnp.maximum(params.Q_h, eps))
    
    # Return a new params object with transformed values
    return result.replace(
        Phi_f=transformed_phi_f,
        Phi_h=transformed_phi_h,
        sigma2=transformed_sigma2,
        Q_h=transformed_q_h
    )


def untransform_params(transformed_params):
    """
    Transform parameters back from unconstrained to constrained space.
    
    Parameters:
    -----------
    transformed_params : DFSVParamsDataclass
        Transformed parameters in unconstrained space
        
    Returns:
    --------
    DFSVParamsDataclass
        Parameters in their natural (constrained) space
    """
    # Apply sigmoid to transform back persistence parameters
    # sigmoid(x) = 1/(1+exp(-x))
    phi_f_original = 1.0 / (1.0 + jnp.exp(-transformed_params.Phi_f))
    phi_h_original = 1.0 / (1.0 + jnp.exp(-transformed_params.Phi_h))
    
    # Apply exp to transform back variance parameters
    sigma2_original = jnp.exp(transformed_params.sigma2)
    q_h_original = jnp.exp(transformed_params.Q_h)
    
    # Return a new params object with untransformed values
    return transformed_params.replace(
        Phi_f=phi_f_original,
        Phi_h=phi_h_original,
        sigma2=sigma2_original,
        Q_h=q_h_original
    )


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


@partial(jit, static_argnames=["filter"])
def bellman_objective(params, y, filter):
    """
    Compute the negative log-likelihood using the Bellman filter.
    
    Parameters:
    -----------
    params : DFSVParamsDataclass
        Model parameters
    y : jnp.ndarray
        Observed data
    filter : DFSVBellmanFilter
        Filter object
        
    Returns:
    --------
    float
        Negative log-likelihood value
    """
    # Run the bellman filter
    ll = filter.jit_log_likelihood_of_params(filter, params, y)
    return -ll


@partial(jit, static_argnames=["filter"])
def transformed_bellman_objective(transformed_params, y, filter):
    """
    Compute the objective function with transformed parameters.
    
    Parameters:
    -----------
    transformed_params : DFSVParamsDataclass
        Model parameters in transformed (unconstrained) space
    y : jnp.ndarray
        Observed data
    filter : DFSVBellmanFilter
        Filter object
        
    Returns:
    --------
    float
        Negative log-likelihood value
    """
    # Transform parameters back to original space
    original_params = untransform_params(transformed_params)
    
    # Run the bellman filter with original parameters
    return bellman_objective(original_params, y, filter)


def compare_gradients(params, returns, filter):
    """
    Compare gradients in original vs transformed parameter spaces.
    
    Parameters:
    -----------
    params : DFSVParamsDataclass
        Original model parameters
    returns : jnp.ndarray
        Observed data
    filter : DFSVBellmanFilter
        Filter object
    """
    print("\nComparing gradients in original vs transformed parameter spaces...")
    
    # Convert to JAX array for gradient computation
    jax_returns = jnp.array(returns)
    
    # Create gradient functions 
    grad_orig = grad(lambda p: bellman_objective(p, jax_returns, filter))
    grad_trans = grad(lambda p: transformed_bellman_objective(p, jax_returns, filter))
    
    # Compute gradients in original space
    start_time = time.time()
    original_grads = grad_orig(params)
    orig_time = time.time() - start_time
    
    # Transform parameters and compute gradients in transformed space
    transformed_params = transform_params(params)
    start_time = time.time()
    transformed_grads = grad_trans(transformed_params)
    trans_time = time.time() - start_time
    
    # Print results
    print(f"\nTime to compute original gradients: {orig_time:.4f} seconds")
    print(f"Time to compute transformed gradients: {trans_time:.4f} seconds")
    
    # Display magnitude of gradients for key parameters
    print("\nGradient magnitudes in original parameter space:")
    print("-----------------------------------------------")
    print(f"{'Parameter':<10} {'Value':<15} {'Gradient Magnitude':<20}")
    print("-----------------------------------------------")
    
    # Helper function to format gradient info
    def format_grad_info(name, value, grad):
        if hasattr(value, 'shape') and value.size > 1:
            val_str = f"array{value.shape}"
            grad_mag = float(jnp.mean(jnp.abs(grad)))
        else:
            if hasattr(value, 'shape') and value.ndim > 0:
                # Handle scalar arrays with explicit dimensions
                val_str = f"{jnp.asarray(value).item():.4f}"
                grad_mag = float(jnp.mean(jnp.abs(grad)))
            else:
                val_str = f"{float(value):.4f}"
                grad_mag = float(jnp.abs(grad))
        return f"{name:<10} {val_str:<15} {grad_mag:<20.4e}"
    
    # Print original gradients
    print(format_grad_info("Phi_f", params.Phi_f, original_grads.Phi_f))
    print(format_grad_info("Phi_h", params.Phi_h, original_grads.Phi_h))
    print(format_grad_info("sigma2", params.sigma2, original_grads.sigma2))
    print(format_grad_info("Q_h", params.Q_h, original_grads.Q_h))
    
    # Print transformed gradients
    print("\nGradient magnitudes in transformed parameter space:")
    print("--------------------------------------------------")
    print(f"{'Parameter':<10} {'Value':<15} {'Gradient Magnitude':<20}")
    print("--------------------------------------------------")
    print(format_grad_info("Phi_f", transformed_params.Phi_f, transformed_grads.Phi_f))
    print(format_grad_info("Phi_h", transformed_params.Phi_h, transformed_grads.Phi_h))
    print(format_grad_info("sigma2", transformed_params.sigma2, transformed_grads.sigma2))
    print(format_grad_info("Q_h", transformed_params.Q_h, transformed_grads.Q_h))


def optimize_with_transformations(params, returns, filter, T=1000, maxiter=100):
    """
    Optimize model parameters using parameter transformations.
    
    Parameters:
    -----------
    params : DFSV_params
        Initial model parameters
    returns : np.ndarray
        Observed data
    filter : DFSVBellmanFilter
        Filter object
    T : int
        Number of time steps
    maxiter : int
        Maximum number of optimization iterations
        
    Returns:
    --------
    tuple
        (original_params, optimized_params, original_loss, transformed_loss)
    """
    # Convert to JAX parameter class
    jax_params = DFSVParamsDataclass.from_dfsv_params(params)
    
    # Add small perturbations to parameters 
    key = jax.random.PRNGKey(42)
    perturbed_params = jax_params.replace(
        lambda_r=jax_params.lambda_r + 0.1 * jax.random.normal(key, jax_params.lambda_r.shape),
        Phi_f=jax_params.Phi_f + 0.02 * jax.random.normal(jax.random.fold_in(key, 0), jax_params.Phi_f.shape),
        Phi_h=jax_params.Phi_h + 0.02 * jax.random.normal(jax.random.fold_in(key, 1), jax_params.Phi_h.shape),
        mu=jax_params.mu + 0.1 * jax.random.normal(jax.random.fold_in(key, 2), jax_params.mu.shape),
        sigma2=jax_params.sigma2 + 0.05 * jax.random.normal(jax.random.fold_in(key, 3), jax_params.sigma2.shape),
        Q_h=jax_params.Q_h + 0.01 * jax.random.normal(jax.random.fold_in(key, 4), jax_params.Q_h.shape),
    )
    
    # Create optimizer configurations
    learning_rate = 0.01
    opt = optax.adam(learning_rate=learning_rate)
    
    # Define objective function for standard optimization
    def objective(params):
        return bellman_objective(params, returns, filter)
    
    # Define objective function for transformed optimization
    def objective_transformed(transformed_params):
        return transformed_bellman_objective(transformed_params, returns, filter)
    
    # Create solvers
    # solver_standard = OptaxSolver(opt=opt, fun=objective, maxiter=maxiter, tol=1e-6,verbose=True)
    # solver_transformed = OptaxSolver(opt=opt, fun=objective_transformed, maxiter=maxiter, tol=1e-6,verbose=True)
    solver_standard=jaxopt.LBFGS(fun=objective, maxiter=maxiter, tol=1e-6, verbose=True)
    solver_transformed=jaxopt.LBFGS(fun=objective_transformed, maxiter=maxiter, tol=1e-6, verbose=True)
    # Transform parameters for transformed optimization
    transformed_params = transform_params(perturbed_params)
    
    # Run standard optimization
    print("\nStarting standard optimization...")
    start_time = time.time()
    result_standard = solver_standard.run(perturbed_params)
    standard_time = time.time() - start_time
    print(f"Standard optimization took {standard_time:.2f} seconds")
    
    # Run transformed optimization
    print("\nStarting transformed optimization...")
    start_time = time.time()
    result_transformed_raw = solver_transformed.run(transformed_params)
    transformed_time = time.time() - start_time
    print(f"Transformed optimization took {transformed_time:.2f} seconds")
    
    # Transform parameters back to original space
    result_transformed = untransform_params(result_transformed_raw.params)
    
    return (result_standard.params, result_transformed, 
            result_standard.state.value, result_transformed_raw.state.value)


def main():
    """Run the parameter transformation optimization example."""
    print("Starting parameter transformation optimization example...")
    
    # Create a simple model
    params = create_simple_model()
    
    # Generate training data (longer time series for better optimization)
    T = 1000
    returns, factors, log_vols = create_training_data(params, T=T)
    
    # Create a Bellman filter object
    filter = DFSVBellmanFilter(params.N, params.K)
    
    # Convert to JAX parameter class
    jax_params = DFSVParamsDataclass.from_dfsv_params(params)
    
    # Compare gradients
    compare_gradients(jax_params, returns, filter)
    
    # Run optimization with both methods
    standard_params, transformed_params, standard_loss, transformed_loss = \
        optimize_with_transformations(params, returns, filter, T=T, maxiter=500)
    
    print("\nOptimization results:")
    print(f"Standard optimization final loss:      {standard_loss:.4f}")
    print(f"Transformed optimization final loss:   {transformed_loss:.4f}")
    
    # Run filters with optimized parameters to compare results
    standard_states, standard_cov, standard_ll = filter.filter(standard_params, returns)
    transformed_states, transformed_cov, transformed_ll = filter.filter(transformed_params, returns)
    
    # Extract state variables
    K = params.K
    standard_factors = standard_states[:, :K]
    standard_log_vols = standard_states[:, K:]
    transformed_factors = transformed_states[:, :K]
    transformed_log_vols = transformed_states[:, K:]
    
    # Calculate and print MSE for each set of parameters
    factor_mse_standard = jnp.mean((standard_factors - factors) ** 2)
    factor_mse_transformed = jnp.mean((transformed_factors - factors) ** 2)
    logvol_mse_standard = jnp.mean((standard_log_vols.squeeze() - log_vols.squeeze()) ** 2)
    logvol_mse_transformed = jnp.mean((transformed_log_vols.squeeze() - log_vols.squeeze()) ** 2)
    
    print("\nMean squared errors:")
    print(f"Factor MSE - Standard: {factor_mse_standard:.4f}, Transformed: {factor_mse_transformed:.4f}")
    print(f"Log-vol MSE - Standard: {logvol_mse_standard:.4f}, Transformed: {logvol_mse_transformed:.4f}")
    
    # Plot comparison
    import matplotlib.pyplot as plt
    
    # Plot factors
    plt.figure(figsize=(15, 10))
    
    # Factor states
    plt.subplot(2, 1, 1)
    plt.plot(factors, "k-", alpha=0.3, label="True")
    plt.plot(standard_factors, "b-", label="Standard optimization")
    plt.plot(transformed_factors, "r-", label="Transformed optimization")
    plt.title("Factor Estimates")
    plt.legend()
    
    # Log volatility states
    plt.subplot(2, 1, 2)
    plt.plot(log_vols.squeeze(), "k-", alpha=0.3, label="True")
    plt.plot(standard_log_vols.squeeze(), "b-", label="Standard optimization")
    plt.plot(transformed_log_vols.squeeze(), "r-", label="Transformed optimization")
    plt.title("Log Volatility Estimates")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("transform_optimization_comparison.png")
    plt.show()
    
    print(f"\nSaved comparison plot to transform_optimization_comparison.png")
    
    # Print parameter comparison
    print("\nParameter comparison:")
    print("-----------------------------------------------------")
    print(f"{'Parameter':<10} {'True':<15} {'Standard':<15} {'Transformed':<15}")
    print("-----------------------------------------------------")
    
    # Helper function to format parameter values
    def format_param(p, name):
        attr = getattr(p, name)
        if hasattr(attr, 'shape') and attr.size > 1:
            return f"array{attr.shape}"
        else:
            try:
                return f"{float(attr):.4f}"
            except:
                return str(attr)
    
    # Print key parameters
    for param_name in ['Phi_f', 'Phi_h', 'mu', 'Q_h']:
        true_val = format_param(jax_params, param_name)
        std_val = format_param(standard_params, param_name)
        trans_val = format_param(transformed_params, param_name)
        print(f"{param_name:<10} {true_val:<15} {std_val:<15} {trans_val:<15}")
    
    # For sigma2, which is often an array
    print("sigma2     ", end="")
    true_sigma = jax_params.sigma2
    std_sigma = standard_params.sigma2
    trans_sigma = transformed_params.sigma2
    
    if true_sigma.size <= 3:
        # Print individual values for small arrays
        print(f"{', '.join([f'{s:.4f}' for s in true_sigma]):<15}", end="")
        print(f"{', '.join([f'{s:.4f}' for s in std_sigma]):<15}", end="")
        print(f"{', '.join([f'{s:.4f}' for s in trans_sigma]):<15}")
    else:
        # Just print shape and mean for larger arrays
        print(f"mean={float(true_sigma.mean()):.4f} ", end="")
        print(f"mean={float(std_sigma.mean()):.4f} ", end="")
        print(f"mean={float(trans_sigma.mean()):.4f}")


if __name__ == "__main__":
    main()