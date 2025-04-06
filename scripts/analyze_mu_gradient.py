#!/usr/bin/env python
"""
Static Gradient Analysis for Mu Parameter in BIF.

This script analyzes the gradient of the Bellman Information Filter (BIF)
pseudo log-likelihood objective function with respect to the 'mu' parameter only.
All other parameters are held fixed at their true values. This helps diagnose
potential inherent biases in the likelihood surface related to 'mu' estimation.

Based on Phase 1.1 of memory-bank/plans/mu_identifiability_investigation_plan.md
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from collections import namedtuple
import dataclasses

# Project specific imports
from bellman_filter_dfsv.core.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.core.simulation import simulate_DFSV

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)
EPS = 1e-6 # Epsilon for numerical stability

# --- Model and Data Generation (Adapted from test_bif_identifiability_fix.py) ---

def create_constrained_simple_model(N=5, K=2, seed=42):
    """
    Create a simple DFSV model (K=2) with a lower-triangular lambda_r
    with diagonal fixed to 1.0 for identifiability.
    """
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed + 1) # JAX key for potential JAX random ops

    # Generate initial random lambda_r
    initial_lambda_r = jax.random.normal(key, (N, K)) * 0.5 + 0.5

    # Apply lower-triangular constraint
    lambda_r = jnp.tril(initial_lambda_r)

    # Ensure diagonal elements are fixed to 1.0
    diag_indices = jnp.diag_indices(n=min(N, K), ndim=2)
    lambda_r = lambda_r.at[diag_indices].set(1.0)

    # Other parameters (True values for K=2)
    Phi_f = jnp.diag(jnp.array([0.95, 0.90]))
    Phi_h = jnp.diag(jnp.array([0.98, 0.97]))
    mu = jnp.array([-1.0, -1.0]) # True mu
    sigma2 = jnp.array(np.random.uniform(0.05, 0.1, N)) # Keep random sigma2 for now
    Q_h = jnp.diag(jnp.array([0.1, 0.08]))

    params = DFSVParamsDataclass(
        N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, Phi_h=Phi_h, mu=mu, sigma2=sigma2, Q_h=Q_h
    )
    # Ensure parameters are JAX arrays and validated
    filter_instance = DFSVBellmanInformationFilter(N, K)
    params = filter_instance._process_params(params)
    return params

def create_training_data(params, T=1500, seed=123):
    """Generate simulated data for training."""
    returns, _, _ = simulate_DFSV(params, T=T, seed=seed)
    return jnp.asarray(returns) # Return as JAX array

# --- Objective Function for Mu Analysis ---

@eqx.filter_jit
def objective_fn_mu_only(mu_vector, static_args):
     """
     Calculates the negative BIF pseudo log-likelihood using a given mu_vector,
     while keeping all other parameters fixed at their true values provided
     in static_args.

     Args:
         mu_vector: The mu parameter vector to evaluate.
         static_args: A tuple containing (true_params_without_mu, returns_data, filter_instance).
                      true_params_without_mu should be a DFSVParamsDataclass instance
                      with all true parameters *except* mu (mu can be a placeholder).

     Returns:
         The negative pseudo log-likelihood value.
     """
     # Unpack arguments
     true_params_template, y, filter_instance = static_args

     # Create the parameter set for this evaluation, replacing mu
     current_params = true_params_template.replace(mu=mu_vector)

     # Calculate likelihood using the BIF instance
     # Note: jit_log_likelihood_wrt_params already includes the constraint application internally if needed
     # For this static analysis, we assume the true_params_template already has the correct lambda_r structure.
     log_lik = filter_instance.jit_log_likelihood_wrt_params()(current_params, y)

     # Use negative log-likelihood for minimization perspective (gradient points "downhill")
     # Handle non-finite values for robustness
     safe_neg_ll = jnp.nan_to_num(-log_lik, nan=1e10, posinf=1e10, neginf=1e10)
     return safe_neg_ll



@eqx.filter_jit
def objective_fit_only(mu_vector, static_args):
    """
    Calculates the negative of the 'fit' component of the BIF pseudo log-likelihood.

    Args:
        mu_vector: The mu parameter vector to evaluate.
        static_args: A tuple containing (true_params_without_mu, returns_data, filter_instance).

    Returns:
        The negative fit component sum (-fit_sum).
    """
    # Unpack arguments
    true_params_template, y, filter_instance = static_args

    # Create the parameter set for this evaluation, replacing mu
    current_params = true_params_template.replace(mu=mu_vector)

    # Calculate likelihood components
    # Returns (total_lik, fit_sum, penalty_sum)
    _, fit_sum, _ = filter_instance.jit_log_likelihood_wrt_params()(current_params, y, return_components=True)

    # Return negative fit sum (objective is minimizing negative log-likelihood)
    safe_neg_fit_sum = jnp.nan_to_num(-fit_sum, nan=1e10, posinf=1e10, neginf=1e10)
    return safe_neg_fit_sum


@eqx.filter_jit
def objective_penalty_only(mu_vector, static_args):
    """
    Calculates the 'penalty' component of the BIF pseudo log-likelihood.

    Args:
        mu_vector: The mu parameter vector to evaluate.
        static_args: A tuple containing (true_params_without_mu, returns_data, filter_instance).

    Returns:
        The penalty component sum (penalty_sum).
    """
    # Unpack arguments
    true_params_template, y, filter_instance = static_args

    # Create the parameter set for this evaluation, replacing mu
    current_params = true_params_template.replace(mu=mu_vector)

    # Calculate likelihood components
    # Returns (total_lik, fit_sum, penalty_sum)
    _, _, penalty_sum = filter_instance.jit_log_likelihood_wrt_params()(current_params, y, return_components=True)

    # Return penalty sum (objective is minimizing negative log-likelihood = -fit + penalty)
    safe_penalty_sum = jnp.nan_to_num(penalty_sum, nan=1e10, posinf=1e10, neginf=-1e10) # Note: neginf maps to -1e10
    return safe_penalty_sum

# --- Main Execution Logic ---

def main():
    """Run the static gradient analysis for mu, including decomposition."""
    print("Starting Static Mu Gradient Analysis with Decomposition...")

    # 1. Create Model Parameters (Constrained K=2)
    N_val, K_val = 5, 2
    true_params = create_constrained_simple_model(N=N_val, K=K_val, seed=42)
    print(f"Using true model N={true_params.N}, K={true_params.K}")
    print("True Parameters (Constrained lambda_r):")
    print(true_params)

    # 2. Generate Simulation Data
    T = 1500 # Time series length
    print(f"\nGenerating {T} time steps of simulation data...")
    returns = create_training_data(true_params, T=T, seed=123)
    print("Simulation data generated.")

    # 3. Instantiate Filter
    filter_instance = DFSVBellmanInformationFilter(N_val, K_val)

    # 4. Prepare for Gradient Calculation
    # Create a template of true parameters (mu will be replaced)
    true_params_template = true_params

    # Define the value-and-gradient functions using Equinox's filter version
    print("Defining filter_value_and_grad functions (Total, Fit, Penalty)...")
    value_and_grad_total_fn = eqx.filter_value_and_grad(objective_fn_mu_only)
    value_and_grad_fit_fn = eqx.filter_value_and_grad(objective_fit_only)
    value_and_grad_penalty_fn = eqx.filter_value_and_grad(objective_penalty_only)
    print("Functions defined.")

    # JIT the combined functions once before the loop
    print("JIT Compiling filter_value_and_grad functions...")
    value_and_grad_total_fn_jitted = eqx.filter_jit(value_and_grad_total_fn)
    value_and_grad_fit_fn_jitted = eqx.filter_jit(value_and_grad_fit_fn)
    value_and_grad_penalty_fn_jitted = eqx.filter_jit(value_and_grad_penalty_fn)
    print("JIT Compilation complete.")

    # Define mu points to test
    mu_points_to_test = {
        "True": jnp.array([-1.0, -1.0]),
        "Initial [0,0]": jnp.array([0.0, 0.0]),
        "Estimated [2.27, 3.40]": jnp.array([2.2689, 3.4034]), # From previous runs
        "Low [-2,-2]": jnp.array([-2.0, -2.0]),
        "Mid-Low [-0.5,-0.5]": jnp.array([-0.5, -0.5]),
        "Mid-High [1,1]": jnp.array([1.0, 1.0]),
        "High [3,3]": jnp.array([3.0, 3.0]),
    }

    print("\n--- Static Gradient Analysis Results (Objective = -LogLik) ---")
    print("-" * 100)
    header = f"{'Mu Point Description':<25} | {'Mu Value':<20} | {'Total Grad':<20} | {'Fit Grad':<20} | {'Penalty Grad':<20}"
    print(header)
    print("-" * len(header))

    # Static arguments for the objective function
    static_args = (true_params_template, returns, filter_instance)

    # 5. Evaluate Gradient at Each Point
    results = {}
    for name, mu_vec in mu_points_to_test.items():
        start_time = time.time()
        try:
            # Ensure mu_vec has the correct dtype
            mu_vec = jnp.asarray(mu_vec, dtype=true_params.mu.dtype)

            # Calculate total value and gradient
            _, grad_total = value_and_grad_total_fn_jitted(mu_vec, static_args)
            grad_total = np.asarray(grad_total)

            # Calculate fit value and gradient
            _, grad_fit = value_and_grad_fit_fn_jitted(mu_vec, static_args)
            grad_fit = np.asarray(grad_fit)

            # Calculate penalty value and gradient
            _, grad_penalty = value_and_grad_penalty_fn_jitted(mu_vec, static_args)
            grad_penalty = np.asarray(grad_penalty)

            results[name] = {
                'gradient_total': grad_total,
                'gradient_fit': grad_fit,
                'gradient_penalty': grad_penalty
            }

            # Format for printing
            mu_str = np.array2string(np.asarray(mu_vec), precision=4, suppress_small=True)
            grad_total_str = np.array2string(grad_total, precision=4, suppress_small=True)
            grad_fit_str = np.array2string(grad_fit, precision=4, suppress_small=True)
            grad_penalty_str = np.array2string(grad_penalty, precision=4, suppress_small=True)

            print(f"{name:<25} | {mu_str:<20} | {grad_total_str:<20} | {grad_fit_str:<20} | {grad_penalty_str:<20}")

        except Exception as e:
            print(f"{name:<25} | {'ERROR':<20} | {'N/A':<20} | {'N/A':<20} | {'N/A':<20} | Error: {e}")
            results[name] = {
                'gradient_total': np.array([np.nan]*K_val),
                'gradient_fit': np.array([np.nan]*K_val),
                'gradient_penalty': np.array([np.nan]*K_val)
            }
        finally:
            end_time = time.time()
            print(f"{'':<25} | {'':<20} | {'':<20} | {'':<20} | {'':<20} | (Took {end_time - start_time:.2f}s)")


    print("-" * len(header))

    # 6. Sanity Check: Sum of Component Gradients vs Total Gradient
    print("\n--- Gradient Decomposition Sanity Check ---")
    print(f"{'Mu Point Description':<25} | {'Total Grad':<20} | {'Fit Grad + Penalty Grad':<25} | {'Difference':<20} | {'Match?':<10}")
    print("-" * 105)
    all_match = True
    for name, res in results.items():
        grad_total = res['gradient_total']
        grad_fit = res['gradient_fit']
        grad_penalty = res['gradient_penalty']

        if np.any(np.isnan(grad_total)) or np.any(np.isnan(grad_fit)) or np.any(np.isnan(grad_penalty)):
            print(f"{name:<25} | {'NaN':<20} | {'NaN':<25} | {'NaN':<20} | {'N/A'}")
            all_match = False
            continue

        sum_components = grad_fit + grad_penalty
        diff = grad_total - sum_components
        is_close = np.allclose(grad_total, sum_components, atol=1e-5, rtol=1e-4) # Use appropriate tolerance
        match_str = "✅ Yes" if is_close else "❌ No"
        if not is_close:
            all_match = False

        grad_total_str = np.array2string(grad_total, precision=4, suppress_small=True)
        sum_comp_str = np.array2string(sum_components, precision=4, suppress_small=True)
        diff_str = np.array2string(diff, precision=4, suppress_small=True)

        print(f"{name:<25} | {grad_total_str:<20} | {sum_comp_str:<25} | {diff_str:<20} | {match_str}")

    print("-" * 105)
    if all_match:
        print("✅ Sanity check passed: Total gradient approximately equals the sum of fit and penalty gradients for all points.")
    else:
        print("❌ Sanity check failed: Discrepancies found between total gradient and sum of components.")


    print("\nStatic Mu Gradient Analysis with Decomposition finished.")


if __name__ == "__main__":
    main()