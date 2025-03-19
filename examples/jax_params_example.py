"""
Example demonstrating the use of JAX-compatible parameter classes for DFSV models.

This script shows how to create and use the JAX-compatible parameter classes,
and demonstrates their benefits when using JAX transformations.
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
from functions.jax_params import DFSVParamsPytree, DFSVParamsDataclass


def create_test_parameters():
    """Create a set of test parameters for demonstration."""
    # Define model dimensions
    N = 5  # Number of observed series
    K = 2  # Number of factors
    
    # Factor loadings with some typical patterns
    lambda_r = np.array([
        [0.9, 0.1],  # First asset loads strongly on factor 1
        [0.8, 0.2],
        [0.5, 0.5],  # Middle asset loads equally
        [0.2, 0.8],
        [0.1, 0.9]   # Last asset loads strongly on factor 2
    ])
    
    # Factor persistence
    Phi_f = np.array([
        [0.95, 0.01],
        [0.01, 0.95]
    ])
    
    # Log-volatility persistence
    Phi_h = np.array([
        [0.98, 0.00],
        [0.00, 0.98]
    ])
    
    # Long-run mean for log-volatilities
    mu = np.array([-1.0, -1.2])
    
    # Idiosyncratic variance (diagonal)
    sigma2 = np.ones(N) * 0.1
    
    # Log-volatility noise covariance
    Q_h = np.array([
        [0.04, 0.01],
        [0.01, 0.04]
    ])
    
    # Create original parameter object
    params = DFSV_params(
        N=N, K=K,
        lambda_r=lambda_r,
        Phi_f=Phi_f,
        Phi_h=Phi_h,
        mu=mu,
        sigma2=sigma2,
        Q_h=Q_h
    )
    
    return params


# Define a function that makes heavy use of parameters and would benefit from JIT compilation
def log_likelihood_function(params, observation):
    """
    Calculate log-likelihood for a single observation.
    
    Args:
        params: Model parameters (either DFSVParamsPytree or DFSV_params)
        observation: Observation vector of shape (N,)
        
    Returns:
        float: Log-likelihood value
    """
    # Extract parameters
    lambda_r = params.lambda_r
    K = params.K
    
    # For DFSV_params, we need to convert to JAX arrays
    if isinstance(params, DFSV_params):
        lambda_r = jnp.array(lambda_r)
        if params.sigma2.ndim == 1:
            sigma2 = jnp.array(params.sigma2)
        else:
            sigma2 = jnp.array(params.sigma2)
    else:  # JAX parameters
        sigma2 = params.sigma2
    
    # For simplicity, let's use a zero state vector (would usually come from filtering)
    f = jnp.zeros(K)
    # Default log-volatilities to mean
    h = params.mu
    
    # Calculate predicted observation
    pred = lambda_r @ f
    
    # Calculate innovation
    innovation = observation - pred
    
    # Build observation covariance matrix
    if sigma2.ndim == 1:
        obs_cov = jnp.diag(sigma2)
    else:
        obs_cov = sigma2
    
    # Calculate log-likelihood
    N = len(innovation)
    sign, logdet = jnp.linalg.slogdet(obs_cov)
    quad_form = innovation @ jnp.linalg.solve(obs_cov, innovation)
    log_lik = -0.5 * (N * jnp.log(2.0 * jnp.pi) + logdet + quad_form)
    
    return log_lik


# Create a JIT-compiled version for JAX params
@jax.jit
def fast_log_likelihood(params_pytree, observation):
    """JIT-compiled log-likelihood function for JAX parameters."""
    return log_likelihood_function(params_pytree, observation)


def main():
    """Example program demonstrating JAX parameter classes."""
    print("Demonstrating JAX-compatible DFSV parameter classes\n")
    
    # Create standard parameters
    standard_params = create_test_parameters()
    print(f"Created standard parameters with {standard_params.N} series and {standard_params.K} factors")
    
    # Convert to JAX-compatible parameters
    pytree_params = DFSVParamsPytree.from_dfsv_params(standard_params)
    dataclass_params = DFSVParamsDataclass.from_dfsv_params(standard_params)
    print("Converted to JAX-compatible parameter classes")
    
    # Create a random observation
    observation = jnp.array(np.random.randn(standard_params.N))
    print(f"Created random observation with shape {observation.shape}")
    
    # Time standard parameters
    print("\nTiming log-likelihood calculations...")
    n_trials = 1000
    
    # Warmup runs
    _ = log_likelihood_function(standard_params, observation)
    _ = log_likelihood_function(pytree_params, observation)
    _ = fast_log_likelihood(pytree_params, observation)
    
    # Time standard parameters
    start = time.time()
    for _ in range(n_trials):
        ll_standard = log_likelihood_function(standard_params, observation)
    std_time = time.time() - start
    print(f"Standard parameters: {std_time:.4f} seconds for {n_trials} calls")
    
    # Time PyTree parameters without JIT
    start = time.time()
    for _ in range(n_trials):
        ll_pytree = log_likelihood_function(pytree_params, observation)
    pytree_time = time.time() - start
    print(f"PyTree parameters (no JIT): {pytree_time:.4f} seconds for {n_trials} calls")
    
    # Time PyTree parameters with JIT
    start = time.time()
    for _ in range(n_trials):
        ll_fast = fast_log_likelihood(pytree_params, observation)
    jit_time = time.time() - start
    print(f"PyTree parameters with JIT: {jit_time:.4f} seconds for {n_trials} calls")
    
    # Calculate speedup
    speedup = std_time / jit_time
    print(f"\nSpeedup from JIT compilation: {speedup:.2f}x")
    
    # Verify results match
    print("\nVerifying results match:")
    print(f"Standard: {float(ll_standard):.6f}")
    print(f"PyTree: {float(ll_pytree):.6f}")
    print(f"JIT: {float(ll_fast):.6f}")
    
    # Example of a more complex application: batched operations
    print("\nDemonstrating batched operations with vmap:")
    
    # Create a batch of 100 observations
    batch_size = 100
    observations = jnp.array(np.random.randn(batch_size, standard_params.N))
    
    # Create a batched log-likelihood function with vmap
    batched_log_lik = jax.vmap(fast_log_likelihood, in_axes=(None, 0))
    
    # Calculate log-likelihood for all observations
    start = time.time()
    batch_ll = batched_log_lik(pytree_params, observations)
    batch_time = time.time() - start
    
    print(f"Processed {batch_size} observations in {batch_time:.4f} seconds")
    print(f"Average per observation: {batch_time/batch_size*1000:.4f} ms")
    
    # Show parameter modification using replace
    print("\nExample of parameter modification:")
    new_params = pytree_params.replace(mu=jnp.array([-0.5, -0.8]))
    print(f"Original mu: {pytree_params.mu}")
    print(f"Modified mu: {new_params.mu}")
    
    # Example of differentiation
    print("\nExample of differentiation with JAX:")
    
    # Create a function to differentiate
    def objective_function(mu_values, params, obs):
        # Update parameters with new mu values
        new_params = params.replace(mu=mu_values)
        # Calculate log-likelihood
        return -fast_log_likelihood(new_params, obs)  # Negative for minimization
    
    # Get gradient function
    grad_fn = jax.grad(objective_function)
    
    # Calculate gradient
    gradient = grad_fn(pytree_params.mu, pytree_params, observation)
    print(f"Gradient of objective w.r.t. mu: {gradient}")


if __name__ == "__main__":
    # Enable 64-bit precision for JAX
    jax.config.update("jax_enable_x64", True)
    main()