"""
Test script for comparing the performance of standard optimization vs. lax.while_loop optimization.

This script runs parameter optimization using both the standard approach and the lax.while_loop
approach, and compares their performance.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from bellman_filter_dfsv.utils.optimization import run_optimization, FilterType
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV

# Set random seed for reproducibility
np.random.seed(42)
jax.config.update("jax_enable_x64", True)

# Generate synthetic data
print("Generating synthetic data...")
T = 200  # Very short time series for testing
N = 3    # Number of observed variables
K = 1    # Number of factors

# Create model parameters
lambda_r = jnp.array([[0.9], [0.6], [0.3]])  # Factor loadings
Phi_f = jnp.array([[0.95]])                  # Factor persistence
Phi_h = jnp.array([[0.98]])                  # Log-volatility persistence
mu = jnp.array([-1.0])                       # Long-run mean for log-volatilities
sigma2 = jnp.array([0.1, 0.1, 0.1])          # Idiosyncratic variance
Q_h = jnp.array([[0.1]])                     # Log-volatility noise covariance

# Create parameter object
true_params = DFSVParamsDataclass(
    N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, Phi_h=Phi_h,
    mu=mu, sigma2=sigma2, Q_h=Q_h
)

# Simulate data
returns, _, _ = simulate_DFSV(true_params, T=T, seed=42)
returns = jnp.asarray(returns)  # Convert to JAX array

# Define optimization parameters
max_steps = 20  # Use fewer steps for faster testing
optimizer_name = "DampedTrustRegionBFGS"

# Run optimization with standard approach
print("\nRunning optimization with standard approach...")
start_time = time.time()
result_standard = run_optimization(
    filter_type=FilterType.BF,  # Use Bellman Filter
    returns=returns,
    true_params=true_params,
    use_transformations=True,
    optimizer_name=optimizer_name,
    max_steps=max_steps,
    verbose=True,
    use_lax_while=False  # Use standard approach
)
standard_time = time.time() - start_time
print(f"Standard approach completed in {standard_time:.2f} seconds")
print(f"Final loss: {result_standard.final_loss}")
print(f"Steps taken: {result_standard.steps}")

# Run optimization with lax.while_loop approach
print("\nRunning optimization with lax.while_loop approach...")
start_time = time.time()
result_lax_while = run_optimization(
    filter_type=FilterType.BF,  # Use Bellman Filter
    returns=returns,
    true_params=true_params,
    use_transformations=True,
    optimizer_name=optimizer_name,
    max_steps=max_steps,
    verbose=True,
    use_lax_while=True  # Use lax.while_loop approach
)
lax_while_time = time.time() - start_time
print(f"lax.while_loop approach completed in {lax_while_time:.2f} seconds")
print(f"Final loss: {result_lax_while.final_loss}")
print(f"Steps taken: {result_lax_while.steps}")

# Compare results
print("\nPerformance comparison:")
print(f"Standard approach: {standard_time:.2f} seconds")
print(f"lax.while_loop approach: {lax_while_time:.2f} seconds")
print(f"Speedup: {standard_time / lax_while_time:.2f}x")

# Compare parameter estimates
print("\nParameter estimate comparison:")
print("Standard approach final parameters:")
print(f"lambda_r: {result_standard.final_params.lambda_r}")
print(f"Phi_f: {result_standard.final_params.Phi_f}")
print(f"Phi_h: {result_standard.final_params.Phi_h}")
print(f"mu: {result_standard.final_params.mu}")
print(f"Q_h: {result_standard.final_params.Q_h}")
print(f"sigma2: {result_standard.final_params.sigma2}")

print("\nlax.while_loop approach final parameters:")
print(f"lambda_r: {result_lax_while.final_params.lambda_r}")
print(f"Phi_f: {result_lax_while.final_params.Phi_f}")
print(f"Phi_h: {result_lax_while.final_params.Phi_h}")
print(f"mu: {result_lax_while.final_params.mu}")
print(f"Q_h: {result_lax_while.final_params.Q_h}")
print(f"sigma2: {result_lax_while.final_params.sigma2}")

# Compare parameter history
print("\nParameter history comparison:")
print(f"Standard approach parameter history length: {len(result_standard.param_history)}")
print(f"lax.while_loop approach parameter history length: {len(result_lax_while.param_history)}")

# Check if the parameter histories match
histories_match = True
if len(result_standard.param_history) == len(result_lax_while.param_history):
    for i, (p_std, p_lax) in enumerate(zip(result_standard.param_history, result_lax_while.param_history)):
        # Check if lambda_r values match (as a simple check)
        if not jnp.allclose(p_std.lambda_r, p_lax.lambda_r):
            histories_match = False
            print(f"Parameter histories differ at step {i}")
            break
else:
    histories_match = False
    print("Parameter histories have different lengths")

print(f"Parameter histories match: {histories_match}")

# Print the first and last entries of each parameter history
print("\nFirst entry in standard approach parameter history:")
print(f"lambda_r: {result_standard.param_history[0].lambda_r}")

print("\nLast entry in standard approach parameter history:")
print(f"lambda_r: {result_standard.param_history[-1].lambda_r}")

print("\nFirst entry in lax.while_loop approach parameter history:")
print(f"lambda_r: {result_lax_while.param_history[0].lambda_r}")

print("\nLast entry in lax.while_loop approach parameter history:")
print(f"lambda_r: {result_lax_while.param_history[-1].lambda_r}")
