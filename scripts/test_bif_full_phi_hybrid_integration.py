#!/usr/bin/env python
"""
Integration Test for Full Phi Hybrid Transformation in BIF Optimization.

This script tests the stability of the Bellman Information Filter (BIF)
hyperparameter estimation when using the hybrid transformation for full
Phi_f and Phi_h matrices (tanh on diagonal, unconstrained off-diagonal).
It runs the optimization for a fixed number of steps without an explicit
stability penalty and checks the eigenvalues of the final untransformed
Phi matrices.
"""

import time
import csv # Keep for potential future use, but not strictly needed now
import cloudpickle


import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import optax
from functools import partial
from collections import namedtuple
from typing import Dict, Any, Optional
import dataclasses # Added import
import equinox as eqx # Added for tree_at

# Project specific imports
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.utils.transformations import transform_params, untransform_params, safe_arctanh # Import safe_arctanh
from bellman_filter_dfsv.models.simulation import simulate_DFSV
# Import the objective functions from likelihood.py
from bellman_filter_dfsv.filters.objectives import bellman_objective, transformed_bellman_objective

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)

# --- Model and Data Generation (Simplified from bf_optimization.py) ---

def create_simple_model(N=3, K=2): # Default to K=2 for this experiment
    """Create a simple DFSV model."""
    # Factor loadings
    np.random.seed(42) # Keep seed for reproducibility
    # Generate K=2 parameters
    lambda_r_init = np.random.randn(N, K) * 0.5 + 0.5 # Generate initial random matrix (N x 2)
    # Apply lower-triangular constraint with diagonal fixed to 1
    lambda_r = jnp.tril(lambda_r_init)
    diag_indices = jnp.diag_indices(n=min(N, K), ndim=2) # n=2
    lambda_r = lambda_r.at[diag_indices].set(1.0) # Sets lambda_r[0,0] and lambda_r[1,1] to 1.0
    # Factor persistence (K=2 -> 2x2 matrix)
    # Generate full matrices for Phi_f and Phi_h with high values on diagonal and smaller on off-diagonal
    Phi_f = np.random.uniform(0.01, 0.1, (K, K))  # Off-diagonal elements
    Phi_f[np.diag_indices(K)] = np.random.uniform(0.8, 0.98, K)  # Diagonal elements
    Phi_h = np.random.uniform(0.01, 0.1, (K, K))  # Off-diagonal elements
    Phi_h[np.diag_indices(K)] = np.random.uniform(0.9, 0.99, K)  # Diagonal elements
    # Long-run mean for log-volatilities (K=2 -> vector of size 2)
    mu = np.array([-1.0, -0.5]) # Fix to true value for K=2
    # Idiosyncratic variance (diagonal)
    sigma2 = np.random.uniform(0.05, 0.1, N)
    # Log-volatility noise covariance (K=2 -> 2x2 diagonal matrix)
    Q_h = np.diag(np.random.uniform(0.1, 0.3, K)) # K=2 -> [[v1, 0], [0, v2]]

    params = DFSVParamsDataclass(
        N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, Phi_h=Phi_h, mu=mu, sigma2=sigma2, Q_h=Q_h
    )
    # Ensure constraint is applied correctly after creation
    params = apply_identification_constraint(params)
    return params

def create_training_data(params, T=1000, seed=42):
    """Generate simulated data for training."""
    returns, _, _ = simulate_DFSV(params, T=T, seed=seed)
    return jnp.asarray(returns) # Return as JAX array


# --- Constraint Helper Function (Copied from test_bif_identifiability_fix.py) ---

def apply_identification_constraint(params: DFSVParamsDataclass) -> DFSVParamsDataclass:
    """Applies lower-triangular constraint with diagonal fixed to 1 to lambda_r."""
    # Apply lower-triangular constraint first
    constrained_lambda_r = jnp.tril(params.lambda_r)
    # Set diagonal elements to 1.0
    diag_indices = jnp.diag_indices(n=min(params.N, params.K), ndim=2)
    constrained_lambda_r = constrained_lambda_r.at[diag_indices].set(1.0)
    return params.replace(lambda_r=constrained_lambda_r)


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

def run_comparison(true_params: DFSVParamsDataclass, returns: jnp.ndarray, true_mu: jnp.ndarray, max_steps: int = 500):
    """
    Runs the optimization with fixed mu and constrained lambda_r (K=2).
    """
    N, K = true_params.N, true_params.K
    filter_instance = DFSVBellmanInformationFilter(N, K)
    results = []

    # Define Optimizers to Compare
    rtol = 1e-3
    atol = 1e-5
    # Scheduler to use (Reduced LR - Attempt 2)
    initial_lr=1e-3 # Reduced from 1e-2
    peak_lr=1e-2    # Reduced from 1e-1
    end_lr=1e-6
    # Adjust decay steps based on max_steps
    print(f"Using reduced learning rate schedule: init={initial_lr}, peak={peak_lr}, end={end_lr}")
    scheduler=optax.warmup_cosine_decay_schedule(init_value=initial_lr, peak_value=peak_lr, end_value=end_lr, warmup_steps=int(max_steps*0.1), decay_steps=max_steps)
    optimizers_to_test = {
        # Only run AdamW for this specific analysis
        "AdamW": optx.OptaxMinimiser(optax.apply_if_finite(optax.adamw(learning_rate=scheduler),10), rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=frozenset({"loss"})),
    }

    # --- Fixed Mu + Constrained Lambda Configuration (K=2) ---
    config_name = f"K={K}, Fixed Mu, Constrained Lambda_f" # Updated config name
    # Priors are not used when mu is fixed
    priors_dict = None # No priors needed
    # --- End Configuration ---

    # Create uninformed initial parameters (K=2)
    # Generate initial lambda_r guess for K=2 with constraint (start with zeros)
    lambda_r_init_guess = jnp.zeros((N, K)) # Start with zeros
    diag_indices_init = jnp.diag_indices(n=min(N, K), ndim=2)
    lambda_r_init_guess = lambda_r_init_guess.at[diag_indices_init].set(1.0) # Set diagonal to 1.0
    lambda_r_init_guess = jnp.tril(lambda_r_init_guess) # Ensure lower triangular (redundant if starting with zeros, but safe)
    print(f"Initial lambda_r guess (constrained):\n{lambda_r_init_guess}") # Add print

    # Create initial guess for Phi matrices using hybrid approach
    # We need the *unconstrained* representation for the optimizer's initial state (y0)
    # Off-diagonals are unconstrained (start at 0)
    # Diagonals are arctanh of the desired initial stable value (e.g., 0.95)
    initial_stable_diag_val = 0.95
    unconstrained_diag_val = safe_arctanh(initial_stable_diag_val)

    phi_f_init = jnp.zeros((K, K))
    phi_f_init = phi_f_init.at[jnp.diag_indices(K)].set(unconstrained_diag_val)
    phi_h_init = jnp.zeros((K, K))
    phi_h_init = phi_h_init.at[jnp.diag_indices(K)].set(unconstrained_diag_val)

    # Define desired initial constrained diagonal value for Phi
    initial_stable_diag_val = 0.95

    uninformed_params = DFSVParamsDataclass(
        N=N, K=K,
        lambda_r=lambda_r_init_guess, # Use K=2 constrained guess
        Phi_f=initial_stable_diag_val * jnp.eye(K), # Use CONSTRAINED initial value
        Phi_h=initial_stable_diag_val * jnp.eye(K), # Use CONSTRAINED initial value
        mu=jnp.zeros(K), # Start mu at 0 (will be fixed)
        sigma2=jnp.ones(N) * 0.1, # Use smaller fixed initial sigma2
        Q_h=0.5 * jnp.eye(K) # Increased initial Q_h for stability
    )
    # Apply constraint to initial guess (important for transformation)
    uninformed_params_constrained = apply_identification_constraint(uninformed_params)
    uninformed_params_constrained = filter_instance._process_params(uninformed_params_constrained) # Ensure JAX arrays

    # Loop through optimizers (only AdamW defined above)
    for name, solver in optimizers_to_test.items():
        # Use transformations, objective fixes mu
        use_transform = True
        print(f"\n--- Running: Config='{config_name}' | Optimizer='{name}' | Transform=Yes ---")

        # Print the initial *constrained* parameter guess (before transformation)
        print("\nInitial Parameter Guess (Constrained, before transformation):")
        print(uninformed_params_constrained)

        # Transform the constrained initial guess
        initial_y = transform_params(uninformed_params_constrained)

        # Define wrapper for objective with fixed mu and stability penalty
        @eqx.filter_jit
        def fixed_mu_objective_wrapper(t_params, args_tuple):
            obs, filt, fixed_mu_val, penalty_weight = args_tuple # Unpack static args including penalty weight
            # 1. Untransform parameters being optimized (includes lambda_r)
            params_iter = untransform_params(t_params)
            # 2. Fix mu to its true value
            params_fixed_mu = eqx.tree_at(lambda p: p.mu, params_iter, fixed_mu_val)
            # 3. Apply identification constraint (ensures lambda_r has correct structure)
            #    This is crucial as lambda_r comes from the optimization state via untransform
            params_fixed_constrained = apply_identification_constraint(params_fixed_mu)
            # 4. Calculate loss using the original objective, passing the penalty weight
            loss = bellman_objective(
                params_fixed_constrained,
                obs,
                filt,
                priors=None, # Pass priors=None
                stability_penalty_weight=penalty_weight # Pass the penalty weight
            )
            return loss

        fn_to_minimize = fixed_mu_objective_wrapper # Use the updated wrapper

        # Define stability penalty weight
        stability_penalty_weight = 1000.0 # Increased penalty weight as requested
        print(f"Using stability_penalty_weight = {stability_penalty_weight}")

        # Package static arguments: observations, filter, fixed mu, penalty weight
        static_args = (returns, filter_instance, true_mu, stability_penalty_weight) # Pass penalty weight

        start_time = time.time()
        final_loss = jnp.inf
        num_steps = -1
        success = False
        error_msg = "N/A"
        final_params_untransformed = None

        # --- Calculate Initial Objective ---
        initial_loss = jnp.inf
        try:
            print("Calculating initial objective...")
            initial_loss = fn_to_minimize(initial_y, static_args)
            if not jnp.isfinite(initial_loss):
                 raise ValueError(f"Initial objective is non-finite: {initial_loss}")
            print(f"Initial Objective Loss: {initial_loss:.4f}")
        except Exception as init_e:
            print(f"ERROR during initial objective calculation: {init_e}")
            error_msg = f"Initial obj failed: {init_e}"
            results.append(OptimizerResult(
                optimizer_name=name, uses_transformations=use_transform, prior_config_name=config_name,
                success=False, final_loss=float(initial_loss), steps=-1, time_taken=0, error_message=error_msg, final_params=None
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
            final_params_untransformed = None # Initialize here
            validation_exception = None
            try:
                # Get final parameters (untransformed) and fix mu
                final_t_params = sol.value # Final transformed params
                # 1. Untransform first (gets optimized lambda_r, etc.)
                final_params_untransformed_temp = untransform_params(final_t_params)
                # 2. Fix mu to its true value in the final untransformed parameters
                final_params_fixed_mu = eqx.tree_at(lambda p: p.mu, final_params_untransformed_temp, true_mu)
                # 3. Apply constraint to ensure final lambda_r structure is correct
                final_params_untransformed = apply_identification_constraint(final_params_fixed_mu) # Final params with fixed mu and constrained lambda

                # Recalculate loss using the non-transformed objective function with the final parameters (fixed mu, constrained lambda, no priors)
                final_loss_recalculated = bellman_objective(final_params_untransformed, returns, filter_instance, priors=None) # Pass priors=None
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

        # --- Eigenvalue Check ---
        if final_params_untransformed is not None:
            try:
                phi_f_eigvals = jnp.abs(jnp.linalg.eigvals(final_params_untransformed.Phi_f))
                phi_h_eigvals = jnp.abs(jnp.linalg.eigvals(final_params_untransformed.Phi_h))
                print("\n--- Final Eigenvalue Magnitudes ---")
                print(f"Phi_f: {phi_f_eigvals}")
                print(f"Phi_h: {phi_h_eigvals}")
                if jnp.all(phi_f_eigvals < 1.0) and jnp.all(phi_h_eigvals < 1.0):
                    print("Stability Check: PASSED (All eigenvalue magnitudes < 1.0)")
                else:
                    print("Stability Check: FAILED (One or more eigenvalue magnitudes >= 1.0)")
                    # Potentially modify success status or error message if stability fails
                    if success: # If solver thought it succeeded but stability failed
                         error_msg += "; Stability Check FAILED"
                         success = False # Mark as overall failure if stability check fails
            except Exception as eig_e:
                print(f"ERROR calculating eigenvalues: {eig_e}")
                if success:
                    error_msg += f"; Eigenvalue calc failed: {eig_e}"
                    success = False # Mark as failure if eigenvalue check fails
        else:
            print("\n--- Final Eigenvalue Magnitudes ---")
            print("Skipping eigenvalue check (final_params is None).")
        # --- End Eigenvalue Check ---


        # Store results
        results.append(OptimizerResult(
            optimizer_name=name,
            uses_transformations=use_transform,
            prior_config_name=config_name, # Use config_name
            success=success, # Use potentially updated success status
            final_loss=float(final_loss), # Convert JAX scalar to float
            steps=int(num_steps),
            time_taken=time_taken,
            error_message=error_msg, # Use potentially updated error message
            final_params=final_params_untransformed # Store the final params Pytree (with fixed mu, constrained lambda)
        ))

    return results

def print_results_table(results: list[OptimizerResult]):
    """Prints the comparison results in a formatted table."""
    print("\n\n--- Optimizer and Prior Comparison Results ---")
    # Header
    print(f"{'Config Name':<35} | {'Optimizer':<15} | {'Transform':<10} | {'Success':<8} | {'Final Loss':<15} | {'Steps':<8} | {'Time (s)':<10} | {'Error Message'}") # Updated header
    print("-" * 130) # Adjusted width
    # Rows - Sort by config, then optimizer, then transform
    for res in sorted(results, key=lambda x: (x.prior_config_name, x.optimizer_name, x.uses_transformations)):
        success_str = "Yes" if res.success else "No"
        loss_str = f"{res.final_loss:.4e}" if np.isfinite(res.final_loss) else "Inf/NaN"
        steps_str = str(res.steps) if res.steps >= 0 else "N/A"
        time_str = f"{res.time_taken:.2f}"
        error_str = res.error_message if not res.success else "N/A"
        print(f"{res.prior_config_name:<35} | {res.optimizer_name:<15} | {'Yes' if res.uses_transformations else 'No':<10} | {success_str:<8} | {loss_str:<15} | {steps_str:<8} | {time_str:<10} | {error_str}") # Updated print format
    print("-" * 130) # Adjusted width


def save_results_to_csv(results: list[OptimizerResult], filename: str = "bif_strong_mu_prior_fixed_lambda_results.csv"): # Updated default filename
    """Saves the comparison results to a CSV file."""
    if not results:
        print("No results to save.")
        return

    # Get headers from the namedtuple fields
    # Exclude 'final_params' from CSV headers as it's a complex object
    headers = [field for field in OptimizerResult._fields if field != 'final_params']


    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(headers)
            # Write data rows
            for result in results:
                # Prepare row data, excluding final_params
                row_data = {field: getattr(result, field) for field in headers}
                # Convert JAX arrays/scalars if necessary
                row = [float(item) if isinstance(item, (jnp.ndarray, jnp.generic)) else item for item in row_data.values()]
                writer.writerow(row)
        print(f"Results successfully saved to {filename}")
    except IOError as e:
        print(f"Error saving results to CSV: {e}")

def print_parameter_comparison(results: list[OptimizerResult], true_params: DFSVParamsDataclass):
    """Prints a comparison between true and estimated parameters for successful runs."""
    print("\n\n--- Parameter Estimation Comparison ---")

    # Get parameter names from the dataclass, excluding N and K
    param_names = [f.name for f in dataclasses.fields(DFSVParamsDataclass) if f.name not in ['N', 'K']]

    # Set numpy print options for better readability
    np.set_printoptions(precision=4, suppress=True)

    for res in sorted(results, key=lambda x: (x.prior_config_name, x.optimizer_name, x.uses_transformations)):
        if res.final_params is not None:
            print(f"\n-- Run: Config='{res.prior_config_name}' | Optimizer='{res.optimizer_name}' | Success='{'Yes' if res.success else 'No'}' --") # Updated print
            print("-" * 80)
            print(f"{'Parameter':<10} | {'True Value':<35} | {'Estimated Value'}")
            print("-" * 80)

            for name in param_names:
                true_val = getattr(true_params, name)
                est_val = getattr(res.final_params, name)

                # Convert to numpy for consistent printing
                true_val_np = np.asarray(true_val)
                est_val_np = np.asarray(est_val)

                # Format for printing (handle multi-line arrays)
                true_str_lines = str(true_val_np).split('\n')
                est_str_lines = str(est_val_np).split('\n')

                # Print first line with parameter name
                print(f"{name:<10} | {true_str_lines[0]:<35} | {est_str_lines[0]}")

                # Print subsequent lines aligned
                max_lines = max(len(true_str_lines), len(est_str_lines))
                for i in range(1, max_lines):
                    true_line = true_str_lines[i] if i < len(true_str_lines) else ""
                    est_line = est_str_lines[i] if i < len(est_str_lines) else ""
                    print(f"{'':<10} | {true_line:<35} | {est_line}")
            print("-" * 80)
        else:
            print(f"\n-- Run: Config='{res.prior_config_name}' | Optimizer='{res.optimizer_name}' --") # Updated print
            print("  No final parameters available for comparison (likely failed early).")

    # Reset numpy print options to default if desired
    # np.set_printoptions()



def main():
    """Run the optimizer comparison study with fixed mu."""
    print(f"Starting BIF Fixed Mu + Constrained Lambda_f (K=2) Evaluation...") # Updated print

    # 1. Create Model Parameters (Use K=2)
    true_params = create_simple_model(N=3, K=2) # Set K=2 for this experiment
    print(f"Using model N={true_params.N}, K={true_params.K}")
    # Ensure true params are also JAX arrays for consistency if needed later
    true_params = DFSVBellmanInformationFilter(true_params.N, true_params.K)._process_params(true_params)
    true_mu = true_params.mu # Extract the true mu vector (K=2)
    # true_lambda_r is NOT fixed, it's constrained and optimized
    print("True Parameters (K=2, Constrained lambda_r):")
    print(true_params)
    print(f"True mu (fixed): {true_mu}") # Print fixed mu

    # 2. Generate Simulation Data
    T = 1500 # Set T
    print(f"\nGenerating {T} time steps of simulation data...")
    returns = create_training_data(true_params, T=T, seed=123)
    print("Simulation data generated.")

    # Priors are not used when mu is fixed
    priors_dict = None
    print(f"\nRunning with mu fixed to {true_mu}, no priors applied.")

    # Calculate objective at true parameters (with fixed mu, no priors) for reference
    print("\nCalculating objective function at true parameters (fixed mu, no priors)...")
    try:
        filter_instance_for_true = DFSVBellmanInformationFilter(true_params.N, true_params.K)
        # Need to transform true params to use the objective wrapper
        true_params_transformed = transform_params(true_params)
        # Define stability penalty weight (same as in run_comparison) for this calculation
        stability_penalty_weight_for_true = 10.0 # Restore original penalty weight

        # Define the objective wrapper locally for this calculation, including penalty weight
        @eqx.filter_jit # Use filter_jit
        def temp_objective_wrapper_fixed_mu(t_params, args_tuple):
            obs, filt, fixed_mu_val, penalty_weight = args_tuple # Unpack static args including penalty weight
            params_iter = untransform_params(t_params)
            # Fix mu, keep other params (including lambda_r) from optimization state
            params_fixed = eqx.tree_at(lambda p: p.mu, params_iter, fixed_mu_val)
            # Apply constraint to lambda_r (should already be correct for true params)
            params_fixed = apply_identification_constraint(params_fixed)
            # Calculate loss including the stability penalty
            loss = bellman_objective(
                params_fixed,
                obs,
                filt,
                priors=None, # No priors
                stability_penalty_weight=penalty_weight # Pass penalty weight
            )
            return loss

        true_objective_value = temp_objective_wrapper_fixed_mu(
            true_params_transformed,
            (returns, filter_instance_for_true, true_mu, stability_penalty_weight_for_true) # Pass penalty weight
        )

        if jnp.isfinite(true_objective_value):
            print(f"Objective function value at TRUE parameters (Fixed Mu, Constrained Lambda_f): {true_objective_value:.4f}") # Updated print
        else:
            print("Objective function value at TRUE parameters (Fixed Mu, Constrained Lambda_f) is non-finite.") # Updated print
    except Exception as e:
        print(f"ERROR calculating objective at true parameters: {e}")


    # 3. Run Comparison
    max_opt_steps = 500 # Increased max steps for final run
    print(f"\nRunning optimizer comparison (max_steps={max_opt_steps})...")
    results = run_comparison(
        true_params=true_params,
        returns=returns,
        true_mu=true_mu, # Pass true mu
        # No prior info needed
        max_steps=max_opt_steps # Pass updated max_steps
    )

    # 4. Print Results Table (includes success/failure and error messages)
    print_results_table(results)

    # 5. Save Final Estimated Parameters (Keep this for potential debugging)
    print("\nSaving final estimated parameters...")
    for res in results:
        if res.final_params is not None:
            status_str = 'Success' if res.success else 'Failure'
            opt_name = res.optimizer_name.replace(' ', '_')
            # Update config_name_safe to reflect "Fixed Mu, Constrained Lambda_f"
            config_name_safe = res.prior_config_name.replace(' ', '_').replace('(','').replace(')','').replace('+','_plus_').replace(',','').replace('=','_') # Make filename safe
            # Update filename for this specific test
            filename = f"outputs/estimated_params_hybrid_integration_{opt_name}_{config_name_safe}_{status_str}_K{true_params.K}.pkl" # Save in outputs/
            try:
                # Ensure outputs directory exists (optional, good practice)
                import os
                os.makedirs("outputs", exist_ok=True)
                with open(filename, 'wb') as f:
                    cloudpickle.dump(res.final_params, f)
                print(f"  Saved parameters to {filename}")
            except Exception as e:
                print(f"  ERROR saving parameters for {opt_name}/{config_name_safe} to {filename}: {e}")
        else:
            opt_name = res.optimizer_name.replace(' ', '_')
            config_name_safe = res.prior_config_name.replace(' ', '_').replace('(','').replace(')','').replace('+','_plus_').replace(',','').replace('=','_') # Make filename safe
            print(f"  Skipping parameter saving for {opt_name}/{config_name_safe} (final_params is None)")

    # 6. Print Parameter Comparison Table (Keep as requested)
    print_parameter_comparison(results, true_params)

    # 7. CSV saving removed for this specific test

    print(f"\nBIF Full Phi Hybrid Integration Test (K={true_params.K}) finished.") # Updated print


if __name__ == "__main__":
    main()