#!/usr/bin/env python
"""Minimal script to debug BIF log-likelihood gradient calculation."""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import logging

# Project-specific imports
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.models.simulation_helpers import create_stable_dfsv_params
from bellman_filter_dfsv.utils.optimization_helpers import create_stable_initial_params
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.utils.transformations import transform_params, untransform_params, apply_identification_constraint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # --- Configuration ---
    N = 10
    K = 3
    T = 500
    replicate_seed = 1328822329
    fix_mu = True
    use_transformations = True # Match the setting in run_optimization

    logger.info(f"Configuration: N={N}, K={K}, T={T}, seed={replicate_seed}, fix_mu={fix_mu}, use_transformations={use_transformations}")

    # --- JAX Setup ---
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True) # Enable NaN checks
    # Consider setting EQX_ON_ERROR=breakpoint environment variable for interactive debugging
    logger.info("JAX configured for x64 precision and NaN debugging.")

    # --- Parameter Generation ---
    np.random.seed(replicate_seed) # Seed NumPy for parameter generation
    logger.info("Generating true parameters...")
    true_params = create_stable_dfsv_params(N=N, K=K)
    logger.info("Generating initial parameters...")
    initial_params_guess = create_stable_initial_params(N=N, K=K)

    # --- Data Simulation ---
    logger.info("Generating simulation data...")
    # Use a different seed for data simulation for reproducibility consistency
    returns, _, _ = simulate_DFSV(true_params, T=T, seed=replicate_seed + 1)
    returns = jnp.asarray(returns) # Convert to JAX array
    logger.info(f"Simulation data generated (T={T}).")

    # --- Filter Setup ---
    logger.info("Creating BIF filter instance...")
    filter_instance = DFSVBellmanInformationFilter(N=N, K=K)

    # --- Objective Function Definition (Negative Log-Likelihood) ---
    # This mirrors the 'likelihood_objective_for_grad' from optimization.py debug block
    def neg_log_likelihood_objective(params_opt, obs):
        # 1. Untransform parameters if they are passed in transformed space
        params_constrained = untransform_params(params_opt) if use_transformations else params_opt

        # 2. Fix mu if requested
        if fix_mu:
             # Ensure true_params.mu is available and correctly shaped
             true_mu_val = true_params.mu
             params_constrained = eqx.tree_at(lambda p: p.mu, params_constrained, true_mu_val)

        # 3. Apply identification constraint
        params_fixed_constrained = apply_identification_constraint(params_constrained)

        # 4. Calculate log likelihood using the filter instance
        # Use the public method which calls the JITted internal implementation
        log_lik = filter_instance.log_likelihood_wrt_params(params_fixed_constrained, obs)

        # Return negative log likelihood, handle NaNs robustly
        safe_neg_ll = jnp.nan_to_num(-log_lik, nan=jnp.inf, posinf=jnp.inf, neginf=-jnp.inf)
        # Add error check specifically for the output likelihood value
        safe_neg_ll = eqx.error_if(
            safe_neg_ll,
            jnp.isnan(safe_neg_ll) | jnp.isinf(safe_neg_ll),
            "NaN/Inf detected in calculated negative log-likelihood value"
        )
        return safe_neg_ll

    # --- Gradient Calculation ---
    logger.info("Preparing value and gradient function...")
    # Use Equinox's version for better error messages potentially
    value_and_grad_fn = eqx.filter_value_and_grad(neg_log_likelihood_objective, has_aux=False)

    # Prepare initial parameters for the gradient function
    initial_params_for_grad = initial_params_guess
    if use_transformations:
        logger.info("Applying transformations to initial parameters...")
        initial_params_for_grad = transform_params(initial_params_guess)
        # Add check after transformation
        initial_params_for_grad = eqx.error_if(
            initial_params_for_grad,
            jax.tree_util.tree_reduce(lambda x, y: x or jnp.any(jnp.isnan(y) | jnp.isinf(y)), initial_params_for_grad, initializer=False),
            "NaN/Inf detected in initial_params_for_grad *after* transformation"
        )

    logger.info("Attempting to calculate initial value only....")
    try:
        initial_value = neg_log_likelihood_objective(initial_params_for_grad, returns)
        logger.info(f"Initial Negative Log-Likelihood Value: {initial_value}")
    except Exception as e:
        logger.error(f"Error during value calculation: {e}", exc_info=True)
    
    logger.info("Attempting to calculate initial value and gradient...")
    try:
        # Calculate initial value and gradient
        initial_value, initial_grad = value_and_grad_fn(initial_params_for_grad, returns)

        logger.info(f"Initial Negative Log-Likelihood Value: {initial_value}")

        # Check gradient for NaNs/Infs
        grad_contains_nan = jax.tree_util.tree_reduce(
            lambda x, y: x or jnp.any(jnp.isnan(y)), initial_grad, initializer=False
        )
        grad_contains_inf = jax.tree_util.tree_reduce(
            lambda x, y: x or jnp.any(jnp.isinf(y)), initial_grad, initializer=False
        )

        logger.info(f"Gradient contains NaN? {grad_contains_nan}")
        logger.info(f"Gradient contains Inf? {grad_contains_inf}")

        if grad_contains_nan or grad_contains_inf:
            logger.warning("Gradient calculation succeeded but contains NaN/Inf values.")
            # Optionally print parts of the gradient here if needed for inspection
            # print("Gradient structure (first few elements):")
            # print(jax.tree_map(lambda x: x.flatten()[0] if hasattr(x, 'flatten') else x, initial_grad))
        else:
            logger.info("Gradient calculated successfully without NaN/Inf.")

    except Exception as e:
        logger.error(f"Error during value and gradient calculation: {e}", exc_info=True)
        logger.error("Gradient calculation failed.")

if __name__ == "__main__":
    main()