#!/usr/bin/env python
"""
Optimizer and Prior Comparison for DFSV Bellman Information Filter Hyperparameter Estimation.

This script compares the performance (stability, convergence, efficiency) of
various optimizers from Optimistix and Optax, under different prior configurations,
when minimizing the negative pseudo log-likelihood derived from the
DFSVBellmanInformationFilter.
"""

import time
import csv
import cloudpickle


import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import optax
from functools import partial
from collections import namedtuple
from typing import Dict, Any, Optional

# Project specific imports
from bellman_filter_dfsv.core.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.utils.transformations import transform_params, untransform_params
from bellman_filter_dfsv.core.simulation import simulate_DFSV
# Import the objective functions from likelihood.py
from bellman_filter_dfsv.core.likelihood import bellman_objective, transformed_bellman_objective

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)

# --- Model and Data Generation (Simplified from bf_optimization.py) ---

def create_simple_model(N=3, K=1):
    """Create a simple DFSV model."""
    # Factor loadings
    np.random.seed(42)
    lambda_r = np.array([[0.9], [0.6], [0.3]]) if K == 1 else np.random.randn(N, K) * 0.5 + 0.5
    # Factor persistence
    Phi_f = np.array([[0.95]]) if K == 1 else np.diag(np.random.uniform(0.8, 0.98, K))
    # Log-volatility persistence
    Phi_h = np.array([[0.98]]) if K == 1 else np.diag(np.random.uniform(0.9, 0.99, K))
    # Long-run mean for log-volatilities
    mu = np.array([-1.0]) if K == 1 else np.random.randn(K) * 0.5 - 1.0
    # Idiosyncratic variance (diagonal)
    sigma2 = np.random.uniform(0.05, 0.2, N)
    # Log-volatility noise covariance
    Q_h = np.array([[0.1]]) if K == 1 else np.diag(np.random.uniform(0.05, 0.15, K))

    params = DFSVParamsDataclass(
        N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, Phi_h=Phi_h, mu=mu, sigma2=sigma2, Q_h=Q_h
    )
    return params

def create_training_data(params, T=1000, seed=42):
    """Generate simulated data for training."""
    returns, _, _ = simulate_DFSV(params, T=T, seed=seed)
    return jnp.asarray(returns) # Return as JAX array

# --- Objective Functions (Now imported from core.likelihood) ---

# --- Main Comparison Logic ---

OptimizerResult = namedtuple("OptimizerResult", [
    "optimizer_name",
    "uses_transformations",
    "prior_config_name", # Added field
    "success",
    "final_loss",
    "steps",
    "time_taken",
    "error_message",
    "final_params" # Added field for estimated parameters
])

def run_comparison(true_params: DFSVParamsDataclass, returns: jnp.ndarray, max_steps: int = 100):
    """
    Runs the optimizer and prior comparison study.

    Args:
        true_params: The true parameters used for simulation (for N, K reference).
        returns: The simulated observation data.
        max_steps: Maximum iterations for each optimizer run.

    Returns:
        List[OptimizerResult]: A list containing results for each optimizer/prior run.
    """
    N, K = true_params.N, true_params.K
    filter_instance = DFSVBellmanInformationFilter(N, K)
    results = []

    # Define Optimizers to Compare
    rtol = 1e-3
    atol = 1e-5
    learning_rate = 1e-2 # Example LR for Optax optimizers

    optimizers_to_test = {
        # "BFGS": optx.BFGS(rtol=rtol, atol=atol, verbose=frozenset({"loss"})),
        # "NonlinearCG": optx.NonlinearCG(rtol=rtol, atol=atol),
        # "SGD": optx.OptaxMinimiser(optax.sgd(learning_rate=learning_rate), rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=frozenset({"loss"})),
        "Adam": optx.OptaxMinimiser(optax.adam(learning_rate=learning_rate), rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=frozenset({"loss"})),
        "AdamW": optx.OptaxMinimiser(optax.adamw(learning_rate=learning_rate), rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=frozenset({"loss"})),
    }

    # Define Prior Configurations to Test
    # Defined inside run_comparison to access K
    # Format matches the arguments of log_prior_density in likelihood.py
    prior_configurations = {
        "No Priors": None,
        
        #Not used for now
        # "Sigma2_Qh_Priors": {
        #     "prior_sigma2_alpha": 3.0,
        #     "prior_sigma2_beta": 0.1,
        #     "prior_q_h_alpha": 3.0,
        #     "prior_q_h_beta": 0.05,
        # },
        # "InvGamma_sigma2": {
        #     "prior_sigma2_alpha": 3.0,
        #     "prior_sigma2_beta": 0.1
        # },
        "Full_Priors": {
            # Mu Prior (Normal)
            "prior_mu_mean": jnp.zeros(K), # Use K here
            "prior_mu_var": jnp.ones(K),   # Variance, not stddev
            # Phi_h Prior (Normal - Diagonal only for simplicity here)
            # Note: log_prior_density expects means/vars for all elements.
            # This simplified prior only sets diagonal elements implicitly via log_prior_density logic.
            "prior_phi_h_diag_mean": 0.9, # Mean for diagonals
            "prior_phi_h_mean": 0.0,      # Mean for off-diagonals (default in log_prior_density)
            "prior_phi_h_var": 0.1**2,    # Variance (stddev^2)
            # Phi_f Prior (Normal - Diagonal only for simplicity here)
            "prior_phi_f_diag_mean": 0.9, # Mean for diagonals
            "prior_phi_f_mean": 0.0,      # Mean for off-diagonals (default in log_prior_density)
            "prior_phi_f_var": 0.1**2,    # Variance (stddev^2)
            # Sigma2 Prior (Inverse Gamma)
            "prior_sigma2_alpha": 3.0,
            "prior_sigma2_beta": 0.1,
            # Q_h Prior (Inverse Gamma - Diagonal)
            "prior_q_h_alpha": 3.0,
            "prior_q_h_beta": 0.05,
            # Note: Priors for lambda_r, Phi_f, could be added here following the
            # naming convention: prior_<param_name>_<hyperparam_name>
        }
    }


    # Create uninformed initial parameters
    data_variance = jnp.var(returns, axis=0)
    uninformed_params = DFSVParamsDataclass(
        N=N, K=K,
        lambda_r=0.5 * jnp.ones((N, K)),
        Phi_f=0.8 * jnp.eye(K),
        Phi_h=0.8 * jnp.eye(K),
        mu=jnp.zeros(K),
        sigma2=0.5 * data_variance,
        Q_h=0.2 * jnp.eye(K)
    )
    uninformed_params = filter_instance._process_params(uninformed_params) # Ensure JAX arrays

    # Loop through priors, optimizers, and transformation settings
    for prior_name, priors_dict in prior_configurations.items():
        for name, solver in optimizers_to_test.items():
            for use_transform in [True]: # Only run transformed version
                print(f"\n--- Running: Prior='{prior_name}' | Optimizer='{name}' | Transform={'Yes' if use_transform else 'No'} ---")

                if use_transform:
                    initial_y = transform_params(uninformed_params)
                    # Define wrapper for transformed objective
                    def transformed_objective_wrapper(t_params, args_tuple):
                        obs, filt, priors = args_tuple # Unpack static args including priors
                        # Use imported function name
                        loss = transformed_bellman_objective(t_params, obs, filt, priors)
                        return loss
                    fn_to_minimize = transformed_objective_wrapper
                    # Use imported function name for final loss calculation
                    objective_fn_for_loss_calc = bellman_objective
                else:
                    initial_y = uninformed_params
                    # Define wrapper for non-transformed objective
                    def objective_wrapper(params, args_tuple):
                        obs, filt, priors = args_tuple # Unpack static args including priors
                        # Use imported function name
                        loss = bellman_objective(params, obs, filt, priors)
                        return loss
                    fn_to_minimize = objective_wrapper
                    # Use imported function name for final loss calculation
                    objective_fn_for_loss_calc = bellman_objective

                # Package static arguments, now including priors
                static_args = (returns, filter_instance, priors_dict)

                start_time = time.time()
                final_loss = jnp.inf
                num_steps = -1
                success = False
                error_msg = "N/A"
                final_params_untransformed = None

                # --- Calculate Initial Objective and Gradient ---
                initial_loss = jnp.inf
                initial_grad = None
                try:
                    print("Calculating initial objective...")
                    initial_loss = fn_to_minimize(initial_y, static_args)
                    if not jnp.isfinite(initial_loss):
                         raise ValueError(f"Initial objective is non-finite: {initial_loss}")
                    print(f"Initial Objective Loss: {initial_loss:.4f}")

                    print("Calculating initial gradient...")
                    grad_fn = jax.grad(fn_to_minimize, argnums=0)
                    initial_grad = grad_fn(initial_y, static_args)

                    # Check if gradient is finite
                    grad_leaves = jax.tree_util.tree_leaves(initial_grad)
                    is_grad_finite = jnp.all(jnp.array([jnp.all(jnp.isfinite(leaf)) for leaf in grad_leaves]))
                    if not is_grad_finite:
                        raise ValueError("Initial gradient contains non-finite values.")
                    print(f"Initial Gradient Finite: {is_grad_finite}")
                    print("Initial gradient calculated.")

                except Exception as init_e:
                    print(f"ERROR during initial objective/gradient calculation: {init_e}")
                    error_msg = f"Initial obj/grad failed: {init_e}"
                    results.append(OptimizerResult(
                        optimizer_name=name, uses_transformations=use_transform, prior_config_name=prior_name,
                        success=False, final_loss=float(initial_loss), steps=-1, time_taken=0, error_message=error_msg
                    ))
                    continue # Skip to the next run
                # --- End Initial Calculation ---

                try:
                    # Run the minimization
                    sol = optx.minimise(
                        fn=fn_to_minimize,
                        solver=solver,
                        y0=initial_y,
                        args=static_args, # Pass static args here
                        max_steps=max_steps,
                        throw=False # Don't raise errors, check sol.result
                    )

                    end_time = time.time()
                    time_taken = end_time - start_time
                    num_steps = sol.stats.get('num_steps', -1) # Get steps if available

                    # Check solver success first
                    solver_success = (sol.result == optx.RESULTS.successful)

                    # --- Final Loss Calculation (Attempt even if solver failed) ---
                    final_loss_recalculated = jnp.inf
                    final_params_untransformed = None
                    validation_exception = None
                    try:
                        # Get final parameters (untransformed) - use sol.value which holds the final state
                        final_params_untransformed = untransform_params(sol.value) if use_transform else sol.value
                        # Recalculate loss using the non-transformed objective function
                        final_loss_recalculated = objective_fn_for_loss_calc(final_params_untransformed, returns, filter_instance, priors_dict)
                        if not jnp.isfinite(final_loss_recalculated):
                             # Raise an exception if the calculated loss is not finite
                             raise ValueError(f"Recalculated loss is non-finite: {final_loss_recalculated}")
                    except Exception as val_e:
                        validation_exception = val_e
                        final_loss_recalculated = jnp.inf # Ensure loss is Inf if calculation failed
                        print(f"Exception during final loss calculation: {validation_exception}")
                    # --- End Final Loss Calculation ---

                    # Determine success based *only* on solver status
                    success = solver_success # Overall success is just solver success

                    # Assign final loss based on recalculation result
                    if validation_exception is None and jnp.isfinite(final_loss_recalculated):
                         final_loss = final_loss_recalculated # Use recalculated loss if valid
                    else:
                         final_loss = jnp.inf # Otherwise, report Inf

                    # Determine error message based on solver status and validation status
                    if success:
                        error_msg = "N/A"
                        print(f"Success! Final Loss: {final_loss:.4f}, Steps: {num_steps}, Time: {time_taken:.2f}s")
                    else:
                        # Solver failed, report solver error
                        error_msg = f"Solver failed: {sol.result}"
                        # Add note if final loss calculation also failed
                        if validation_exception is not None:
                            error_msg += f"; Final loss calc failed: {validation_exception}"
                        elif not jnp.isfinite(final_loss_recalculated):
                             error_msg += f"; Final loss non-finite: {final_loss_recalculated}"
                        print(f"Failed! Status: {error_msg}, Final Loss: {final_loss:.4f}, Steps: {num_steps}, Time: {time_taken:.2f}s") # Print final_loss (potentially Inf)

                except Exception as e: # Catch exceptions during the main optx.minimise call
                    end_time = time.time()
                    time_taken = end_time - start_time
                    success = False
                    error_msg = f"Exception during minimize: {str(e)}"
                    final_loss = jnp.inf # Assign Inf on exception
                    print(f"Exception! Error: {e}, Time: {time_taken:.2f}s")

                # Store results
                results.append(OptimizerResult(
                    optimizer_name=name,
                    uses_transformations=use_transform,
                    prior_config_name=prior_name, # Added
                    success=success,
                    final_loss=float(final_loss), # Convert JAX scalar to float
                    steps=int(num_steps),
                    time_taken=time_taken,
                    error_message=error_msg,
                    final_params=final_params_untransformed # Store the final params Pytree (or None)
                ))

    return results

def print_results_table(results: list[OptimizerResult]):
    """Prints the comparison results in a formatted table."""
    print("\n\n--- Optimizer and Prior Comparison Results ---")
    # Header
    print(f"{'Prior Config':<20} | {'Optimizer':<15} | {'Transform':<10} | {'Success':<8} | {'Final Loss':<15} | {'Steps':<8} | {'Time (s)':<10} | {'Error Message'}")
    print("-" * 115) # Adjusted width
    # Rows - Sort by prior, then optimizer, then transform
    for res in sorted(results, key=lambda x: (x.prior_config_name, x.optimizer_name, x.uses_transformations)):
        success_str = "Yes" if res.success else "No"
        loss_str = f"{res.final_loss:.4e}" if np.isfinite(res.final_loss) else "Inf/NaN"
        steps_str = str(res.steps) if res.steps >= 0 else "N/A"
        time_str = f"{res.time_taken:.2f}"
        error_str = res.error_message if not res.success else "N/A"
        print(f"{res.prior_config_name:<20} | {res.optimizer_name:<15} | {'Yes' if res.uses_transformations else 'No':<10} | {success_str:<8} | {loss_str:<15} | {steps_str:<8} | {time_str:<10} | {error_str}")
    print("-" * 115) # Adjusted width


def save_results_to_csv(results: list[OptimizerResult], filename: str = "bif_prior_optimizer_results_rms.csv"):
    """Saves the comparison results to a CSV file."""
    if not results:
        print("No results to save.")
        return

    # Get headers from the namedtuple fields
    headers = OptimizerResult._fields

    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(headers)
            # Write data rows
            for result in results:
                # Convert JAX arrays/scalars if necessary (though they should be float/int by now)
                row = [float(item) if isinstance(item, (jnp.ndarray, jnp.generic)) else item for item in result]
                writer.writerow(row)
        print(f"Results successfully saved to {filename}")
    except IOError as e:
        print(f"Error saving results to CSV: {e}")



def main():
    """Run the optimizer and prior comparison study."""
    print("Starting BIF Optimizer and Prior Comparison Study...")

    # 1. Create Model Parameters
    true_params = create_simple_model(N=5, K=2) # Example: 5 series, 2 factors
    print(f"Using model N={true_params.N}, K={true_params.K}")

    # 2. Generate Simulation Data
    T = 500 # Shorter time series for faster testing
    print(f"Generating {T} time steps of simulation data...")
    returns = create_training_data(true_params, T=T, seed=123)
    print("Simulation data generated.")

    # 3. Run Comparison
    max_opt_steps = 500 # Keep max steps relatively low for comparison speed
    print(f"Running optimizer comparison (max_steps={max_opt_steps})...")
    results = run_comparison(true_params, returns, max_steps=max_opt_steps)

    # 4. Print Results
    print_results_table(results)

    # 5. Save Results to CSV
    save_results_to_csv(results)

    print("\nOptimizer and prior comparison study finished.")


if __name__ == "__main__":
    main()