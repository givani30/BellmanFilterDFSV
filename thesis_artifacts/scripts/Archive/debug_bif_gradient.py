#!/usr/bin/env python
"""Script to debug BIF log-likelihood gradient calculation and run optimization."""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import logging
import time

# Project-specific imports
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.models.simulation_helpers import create_stable_dfsv_params
from bellman_filter_dfsv.utils.optimization_helpers import create_stable_initial_params
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.utils.transformations import transform_params, untransform_params, apply_identification_constraint
from bellman_filter_dfsv.utils.optimization import run_optimization, FilterType, create_filter
from bellman_filter_dfsv.utils.analysis import calculate_accuracy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _calculate_param_errors(true_params, est_params):
    """Calculate RMSE and Mean Error for each parameter field."""
    def safe_rmse(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            return float('nan')
        return float(np.sqrt(np.nanmean((a - b) ** 2)))

    def safe_mean_error(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            return float('nan')
        return float(np.nanmean(b - a)) # estimated - true

    errors = {}
    for field in DFSVParamsDataclass.__dataclass_fields__:
        if field == 'mu' and not hasattr(est_params, 'mu'): # Handle fixed mu case
            continue
        try:
            true_val = getattr(true_params, field)
            est_val = getattr(est_params, field)
            errors[field + '_rmse'] = safe_rmse(true_val, est_val)
            errors[field + '_mean_error'] = safe_mean_error(true_val, est_val)
        except Exception as e:
            logger.warning(f"Could not calculate error for field {field}: {e}")
            errors[field + '_rmse'] = float('nan')
            errors[field + '_mean_error'] = float('nan')
    return errors

def main():
    # --- Configuration ---
    N = 15
    K = 5
    T = 1000
    replicate_seed = 4114928869
    fix_mu = False
    use_transformations = True # Match the setting in run_optimization
    optimizer_name = "DampedTrustRegionBFGS" # Default optimizer for BIF
    max_steps = 5000

    logger.info(f"Configuration: N={N}, K={K}, T={T}, seed={replicate_seed}, fix_mu={fix_mu}, use_transformations={use_transformations}, optimizer={optimizer_name}, max_steps={max_steps}")

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

    # logger.info("Attempting to calculate initial value and gradient...")
    # try:
    #     # Calculate initial value and gradient
    #     initial_value, initial_grad = value_and_grad_fn(initial_params_for_grad, returns)

    #     logger.info(f"Initial Negative Log-Likelihood Value: {initial_value}")

    #     # Check gradient for NaNs/Infs
    #     grad_contains_nan = jax.tree_util.tree_reduce(
    #         lambda x, y: x or jnp.any(jnp.isnan(y)), initial_grad, initializer=False
    #     )
    #     grad_contains_inf = jax.tree_util.tree_reduce(
    #         lambda x, y: x or jnp.any(jnp.isinf(y)), initial_grad, initializer=False
    #     )
    #     logger.info(f"Initial gradient:{initial_grad} ")

    #     logger.info(f"Gradient contains NaN? {grad_contains_nan}")
    #     logger.info(f"Gradient contains Inf? {grad_contains_inf}")

    #     if grad_contains_nan or grad_contains_inf:
    #         logger.warning("Gradient calculation succeeded but contains NaN/Inf values.")
    #         # Optionally print parts of the gradient here if needed for inspection
    #         # print("Gradient structure (first few elements):")
    #         # print(jax.tree_map(lambda x: x.flatten()[0] if hasattr(x, 'flatten') else x, initial_grad))
    #     else:
    #         logger.info("Gradient calculated successfully without NaN/Inf.")

    except Exception as e:
        logger.error(f"Error during value and gradient calculation: {e}", exc_info=True)
        logger.error("Gradient calculation failed.")

    # --- Optimization ---
    logger.info("\n=== Starting Optimization ===\n")
    try:
        # Determine filter type
        filter_type = FilterType.BIF

        # Set up true parameters for optimization
        true_params_for_opt = true_params

        logger.info(f"Starting optimization with {optimizer_name}...")
        start_opt_time = time.time()

        # Run optimization
        result = run_optimization(
            filter_type=filter_type,
            returns=returns,
            initial_params=initial_params_guess,
            true_params=true_params_for_opt,
            fix_mu=fix_mu,
            use_transformations=use_transformations,
            optimizer_name=optimizer_name,
            priors=None,
            stability_penalty_weight=1e3, 
            max_steps=max_steps,
            verbose=True,
            rtol=1e-5,
            atol=1e-5
        )

        opt_duration = time.time() - start_opt_time
        logger.info(f"Optimization completed in {opt_duration:.2f} seconds.")

        # Log optimization results
        if result:
            logger.info("Optimization results:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Final loss: {result.final_loss:.4f}")
            logger.info(f"  Steps taken: {result.steps}")
            logger.info(f"  Result code: {result.result_code}")
            if not result.success:
                logger.warning(f"  Error message: {result.error_message}")

            # Calculate parameter errors if optimization succeeded
            if result.success and result.final_params is not None:
                logger.info("\n=== Parameter Estimation Analysis ===\n")
                param_errors = _calculate_param_errors(true_params, result.final_params)

                # Print parameter errors
                logger.info("Parameter estimation errors:")
                for param, error in param_errors.items():
                    if param.endswith('_rmse'):
                        param_name = param.replace('_rmse', '')
                        logger.info(f"  {param_name}: RMSE = {error:.4f}, Mean Error = {param_errors.get(param_name + '_mean_error', float('nan')):.4f}")

                # Print final parameter estimates
                logger.info("\n=== Final Parameter Estimates ===\n")
                final_params = result.final_params

                # Function to format matrix parameters nicely
                def format_matrix(matrix, precision=4):
                    matrix_np = np.asarray(matrix)
                    if matrix_np.ndim == 1:  # Vector
                        return np.array2string(matrix_np, precision=precision, separator=', ', suppress_small=True)
                    else:  # Matrix
                        rows = []
                        for i in range(matrix_np.shape[0]):
                            row = np.array2string(matrix_np[i], precision=precision, separator=', ', suppress_small=True)
                            rows.append(row)
                        return '\n'.join(rows)

                # Print comparison for each parameter
                param_descriptions = {
                    "lambda_r": "Factor Loadings (Lambda)",
                    "Phi_f": "Factor Transition Matrix (Phi_f)",
                    "Phi_h": "Log-Volatility Transition Matrix (Phi_h)",
                    "mu": "Log-Volatility Mean (mu)",
                    "sigma2": "Observation Noise Variance (sigma2)",
                    "Q_h": "Log-Volatility Noise Covariance (Q_h)"
                }

                for param_name in ["lambda_r", "Phi_f", "Phi_h", "mu", "sigma2", "Q_h"]:
                    # Get the true and estimated values
                    true_val = getattr(true_params, param_name)
                    est_val = getattr(final_params, param_name)

                    # Print parameter name with a more descriptive title
                    logger.info(f"\n{param_descriptions.get(param_name, param_name)}:")
                    logger.info("-" * 60)

                    # Print true value
                    logger.info("True Value:")
                    logger.info(format_matrix(true_val))

                    # Print estimated value
                    logger.info("\nEstimated Value:")
                    logger.info(format_matrix(est_val))

                # Calculate state estimation accuracy
                logger.info("\n=== State Estimation Analysis ===\n")
                try:
                    filter_instance = create_filter(filter_type, N, K)
                    filtered_states, _, _ = filter_instance.filter(result.final_params, returns)
                    filtered_factors = filtered_states[:, :K]
                    filtered_log_vols = filtered_states[:, K:]

                    # Get true states from simulation
                    _, true_factors, true_log_vols = simulate_DFSV(
                        true_params, T=T, seed=replicate_seed + 1
                    )

                    # Calculate accuracy metrics
                    factor_rmse, factor_corr = calculate_accuracy(true_factors, filtered_factors)
                    vol_rmse, vol_corr = calculate_accuracy(true_log_vols, filtered_log_vols)

                    logger.info("State estimation accuracy:")
                    logger.info(f"  Factor RMSE: {np.nanmean(factor_rmse):.4f}, Correlation: {np.nanmean(factor_corr):.4f}")
                    logger.info(f"  Volatility RMSE: {np.nanmean(vol_rmse):.4f}, Correlation: {np.nanmean(vol_corr):.4f}")
                except Exception as e:
                    logger.error(f"Error during state estimation analysis: {e}", exc_info=True)
        else:
            logger.error("Optimization failed to produce a result.")

    except Exception as e:
        logger.error(f"Error during optimization: {e}", exc_info=True)

if __name__ == "__main__":
    main()