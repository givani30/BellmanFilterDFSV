#!/usr/bin/env python
"""
Test Script for DFSV Bellman Information Filter Identifiability Fix.

This script tests the estimation of DFSV parameters using the Bellman
Information Filter, specifically focusing on the identifiability fix
that enforces a lower-triangular constraint on the lambda_r matrix
with a positive diagonal. It uses a single optimizer and assumes
parameter transformations are applied.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import optax # Keep optax for the optimizer definition
import equinox as eqx
from collections import namedtuple
import dataclasses

# Project specific imports
from bellman_filter_dfsv.core.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.utils.transformations import transform_params, untransform_params
from bellman_filter_dfsv.core.simulation import simulate_DFSV

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)
EPS = 1e-6 # Epsilon for numerical stability

# --- Model and Data Generation with Identifiability Constraint ---

def create_constrained_simple_model(N=3, K=1, seed=42):
    """
    Create a simple DFSV model with a lower-triangular lambda_r
    with a strictly positive diagonal.
    """
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed + 1) # JAX key for potential JAX random ops

    # Generate initial random lambda_r
    initial_lambda_r = jax.random.normal(key, (N, K)) * 0.5 + 0.5

    # Apply lower-triangular constraint
    lambda_r = jnp.tril(initial_lambda_r)

    # Ensure diagonal elements are strictly positive
    diag_indices = jnp.diag_indices(n=min(N, K), ndim=2) # Correct way for (N, K) matrix
    # Set diagonal elements to 1.0 for identification
    lambda_r = lambda_r.at[diag_indices].set(1.0)

    # Other parameters (similar to the original script)
    Phi_f = jnp.diag(jnp.array([0.95] * K)) if K > 0 else jnp.empty((0,0))
    Phi_h = jnp.diag(jnp.array([0.98] * K)) if K > 0 else jnp.empty((0,0))
    mu = jnp.array([-1.0] * K) if K > 0 else jnp.empty((0,))
    sigma2 = jnp.array(np.random.uniform(0.05, 0.1, N))
    Q_h = jnp.diag(jnp.array([0.1] * K)) if K > 0 else jnp.empty((0,0))

    params = DFSVParamsDataclass(
        N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, Phi_h=Phi_h, mu=mu, sigma2=sigma2, Q_h=Q_h
    )
    # Ensure parameters are JAX arrays and validated
    filter_instance = DFSVBellmanInformationFilter(N, K)
    params = filter_instance._process_params(params)
    return params

def create_training_data(params, T=1000, seed=42):
    """Generate simulated data for training."""
    returns, _, _ = simulate_DFSV(params, T=T, seed=seed)
    return jnp.asarray(returns) # Return as JAX array

# --- Constraint Helper Function ---

def apply_identification_constraint(params: DFSVParamsDataclass) -> DFSVParamsDataclass:
    """Applies lower-triangular constraint with diagonal fixed to 1 to lambda_r."""
    # Apply lower-triangular constraint first
    constrained_lambda_r = jnp.tril(params.lambda_r)
    # Set diagonal elements to 1.0
    diag_indices = jnp.diag_indices(n=min(params.N, params.K), ndim=2)
    constrained_lambda_r = constrained_lambda_r.at[diag_indices].set(1.0)
    return params.replace(lambda_r=constrained_lambda_r)

# --- Objective Function Wrapper ---

@eqx.filter_jit
def constrained_transformed_objective(transformed_params, args):
     """
     Objective function wrapper for transformed parameters with constraints.
     Untransforms, applies lower-triangular constraint, then calculates likelihood.
     Matches optimistix fn(y, args) signature.
     """
     # Unpack arguments
     y, filter_instance = args

     # 1. Untransform (handles positive diagonal via modified untransform_params)
     params_positive_diag = untransform_params(transformed_params)
     # 2. Apply lower-triangular structure
     constrained_params = apply_identification_constraint(params_positive_diag)
     # 3. Calculate likelihood (no priors)
     # Ensure y (returns data) is passed correctly to the likelihood function
     log_lik = filter_instance.jit_log_likelihood_wrt_params()(constrained_params, y) # Pass unpacked y
     # Use negative log-likelihood for minimization, handle non-finite values
     safe_neg_ll = jnp.nan_to_num(-log_lik, nan=1e10, posinf=1e10, neginf=1e10)
     return safe_neg_ll

# --- Main Optimization Logic ---

OptimizationResult = namedtuple("OptimizationResult", [
    "optimizer_name",
    "success",
    "final_loss",
    "steps",
    "time_taken",
    "error_message",
    "final_params_constrained" # Store the final constrained parameters
])

def run_optimization(true_params: DFSVParamsDataclass, returns: jnp.ndarray, max_steps: int = 200):
    """
    Runs the optimization using a single optimizer with constraints.

    Args:
        true_params: The true parameters used for simulation.
        returns: The simulated observation data.
        max_steps: Maximum iterations for the optimizer run.

    Returns:
        OptimizationResult: The result of the optimization run.
    """
    N, K = true_params.N, true_params.K
    filter_instance = DFSVBellmanInformationFilter(N, K)

    # --- Optimizer Definition ---
    optimizer_name = "AdamW"
    rtol = 1e-3
    atol = 1e-5
    initial_lr=1e-2
    peak_lr=1e-1
    end_lr=1e-6
    scheduler=optax.warmup_cosine_decay_schedule(init_value=initial_lr, peak_value=peak_lr, end_value=end_lr, warmup_steps=20, decay_steps=max_steps) # Adjust decay steps
    # Use apply_if_finite wrapper for robustness
    solver = optx.OptaxMinimiser(
        optax.apply_if_finite(optax.adamw(learning_rate=scheduler), max_consecutive_errors=10),
        rtol=rtol,
        atol=atol,
        norm=optx.rms_norm,
        verbose=frozenset({"loss"})
    )
    print(f"\n--- Running: Optimizer='{optimizer_name}' (with constraints) ---")

    # --- Initial Guess Preparation ---
    # Create uninformed initial parameters
    data_variance = jnp.var(returns, axis=0)
    uninformed_params = DFSVParamsDataclass(
        N=N, K=K,
        lambda_r=0.5 * jnp.ones((N, K)), # Start with simple guess
        Phi_f=0.8 * jnp.eye(K),
        Phi_h=0.8 * jnp.eye(K),
        mu=jnp.zeros(K),
        sigma2=0.5 * data_variance, # Use data variance for sigma2 guess
        Q_h=0.2 * jnp.eye(K)
    )
    uninformed_params = filter_instance._process_params(uninformed_params) # Ensure JAX arrays

    # Apply lower-triangular constraint *before* transforming
    constrained_uninformed_params = apply_identification_constraint(uninformed_params)

    # Transform the constrained initial guess
    initial_y = transform_params(constrained_uninformed_params)
    print("Initial constrained guess created and transformed.")

    # --- Static Arguments for Objective ---
    static_args = (returns, filter_instance) # No priors needed

    # --- Run Optimization ---
    start_time = time.time()
    final_loss = jnp.inf
    num_steps = -1
    success = False
    error_msg = "N/A"
    final_params_constrained = None

    try:
        # Calculate initial objective (optional, for debugging)
        # Pass static_args directly as the 'args' tuple
        initial_loss = constrained_transformed_objective(initial_y, static_args)
        print(f"Initial Objective Loss (Constrained): {initial_loss:.4f}")
        if not jnp.isfinite(initial_loss):
            raise ValueError(f"Initial objective is non-finite: {initial_loss}")

        # Run the minimization
        sol = optx.minimise(
            fn=constrained_transformed_objective,
            solver=solver,
            y0=initial_y,
            args=static_args,
            max_steps=max_steps,
            throw=False # Don't raise errors, check sol.result
        )

        end_time = time.time()
        time_taken = end_time - start_time
        num_steps = sol.stats.get('num_steps', -1)
        solver_success = (sol.result == optx.RESULTS.successful)

        # --- Final Parameter Handling ---
        final_y = sol.value # Final transformed parameters
        validation_exception = None
        try:
            # 1. Untransform (handles positive diagonal)
            params_final_pos_diag = untransform_params(final_y)
            # 2. Apply lower-triangular constraint
            final_params_constrained = apply_identification_constraint(params_final_pos_diag)

            # Recalculate final loss using the *constrained* parameters and the *original* likelihood
            # (Objective function already includes constraint logic, so use it directly)
            # Pass static_args directly as the 'args' tuple
            final_loss_recalculated = constrained_transformed_objective(final_y, static_args)

            if not jnp.isfinite(final_loss_recalculated):
                 raise ValueError(f"Recalculated loss is non-finite: {final_loss_recalculated}")

        except Exception as val_e:
            validation_exception = val_e
            final_loss_recalculated = jnp.inf
            print(f"Exception during final parameter processing/loss calculation: {validation_exception}")
        # --- End Final Parameter Handling ---

        # Determine success based *only* on solver status
        success = solver_success

        # Assign final loss based on recalculation result
        if validation_exception is None and jnp.isfinite(final_loss_recalculated):
             final_loss = final_loss_recalculated
        else:
             final_loss = jnp.inf # Report Inf if validation/recalculation failed

        # Determine error message
        if success:
            error_msg = "N/A"
            print(f"Success! Final Loss: {final_loss:.4f}, Steps: {num_steps}, Time: {time_taken:.2f}s")
        else:
            error_msg = f"Solver failed: {sol.result}"
            if validation_exception is not None:
                error_msg += f"; Final param/loss processing failed: {validation_exception}"
            elif not jnp.isfinite(final_loss_recalculated):
                 error_msg += f"; Final loss non-finite: {final_loss_recalculated}"
            print(f"Failed! Status: {error_msg}, Final Loss: {final_loss:.4f}, Steps: {num_steps}, Time: {time_taken:.2f}s")

    except Exception as e: # Catch exceptions during the main optx.minimise call or initial calc
        end_time = time.time()
        time_taken = end_time - start_time
        success = False
        error_msg = f"Exception during optimization run: {str(e)}"
        final_loss = jnp.inf
        print(f"Exception! Error: {e}, Time: {time_taken:.2f}s")

    # Store results
    result = OptimizationResult(
        optimizer_name=optimizer_name,
        success=success,
        final_loss=float(final_loss), # Convert JAX scalar to float
        steps=int(num_steps),
        time_taken=time_taken,
        error_message=error_msg,
        final_params_constrained=final_params_constrained # Store the constrained params
    )

    return result

# --- Results Reporting ---

def print_result_summary(result: OptimizationResult):
    """Prints a summary of the optimization result."""
    print("\n\n--- Optimization Result Summary ---")
    print(f"{'Optimizer':<15} | {'Success':<8} | {'Final Loss':<15} | {'Steps':<8} | {'Time (s)':<10} | {'Error Message'}")
    print("-" * 80)
    success_str = "Yes" if result.success else "No"
    loss_str = f"{result.final_loss:.4e}" if np.isfinite(result.final_loss) else "Inf/NaN"
    steps_str = str(result.steps) if result.steps >= 0 else "N/A"
    time_str = f"{result.time_taken:.2f}"
    error_str = result.error_message if not result.success else "N/A"
    print(f"{result.optimizer_name:<15} | {success_str:<8} | {loss_str:<15} | {steps_str:<8} | {time_str:<10} | {error_str}")
    print("-" * 80)


def print_parameter_comparison(result: OptimizationResult, true_params: DFSVParamsDataclass):
    """Prints a comparison between true and estimated constrained parameters."""
    print("\n\n--- Parameter Estimation Comparison (Constrained) ---")

    if result.final_params_constrained is None:
        print(f"\n-- Run: Optimizer='{result.optimizer_name}' --")
        print("  No final constrained parameters available for comparison (optimization failed or errored).")
        return

    # Get parameter names from the dataclass, excluding N and K
    param_names = [f.name for f in dataclasses.fields(DFSVParamsDataclass) if f.name not in ['N', 'K']]

    # Set numpy print options for better readability
    np.set_printoptions(precision=4, suppress=True)

    print(f"\n-- Run: Optimizer='{result.optimizer_name}' | Success='{'Yes' if result.success else 'No'}' --")
    print("-" * 80)
    print(f"{'Parameter':<10} | {'True Value (Constrained)':<35} | {'Estimated Value (Constrained)'}")
    print("-" * 80)

    est_params = result.final_params_constrained
    for name in param_names:
        true_val = getattr(true_params, name)
        est_val = getattr(est_params, name)

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

    # Reset numpy print options to default if desired
    # np.set_printoptions()


# --- Main Execution ---

def main():
    """Run the constrained optimization test."""
    print("Starting BIF Constrained Optimization Test...")

    # 1. Create Model Parameters (Constrained)
    N_val, K_val = 5, 2 # Example: 5 series, 2 factors
    true_params = create_constrained_simple_model(N=N_val, K=K_val, seed=42)
    print(f"Using constrained model N={true_params.N}, K={true_params.K}")
    print("True Parameters (Constrained lambda_r):")
    print(true_params)


    # 2. Generate Simulation Data
    T = 1500 # Time series length
    print(f"\nGenerating {T} time steps of simulation data...")
    returns = create_training_data(true_params, T=T, seed=123)
    print("Simulation data generated.")

    # Calculate objective at true parameters for reference (using the constrained objective logic)
    print("\nCalculating objective function at true parameters...")
    try:
        filter_instance_for_true = DFSVBellmanInformationFilter(true_params.N, true_params.K)
        # Need to transform the true params to use the constrained_transformed_objective
        true_params_transformed = transform_params(true_params)
        # Pass returns and filter_instance as a tuple in the 'args' parameter
        true_objective_value = constrained_transformed_objective(true_params_transformed, (returns, filter_instance_for_true))

        if jnp.isfinite(true_objective_value):
            print(f"Objective function value at TRUE parameters (Constrained): {true_objective_value:.4f}")
        else:
            print("Objective function value at TRUE parameters is non-finite.")
    except Exception as e:
        print(f"ERROR calculating objective at true parameters: {e}")


    # 3. Run Optimization
    max_opt_steps = 300 # Increase steps slightly
    print(f"\nRunning constrained optimization (max_steps={max_opt_steps})...")
    result = run_optimization(true_params, returns, max_steps=max_opt_steps)

    # 4. Print Results
    print_result_summary(result)
    print_parameter_comparison(result, true_params)

    print("\nConstrained optimization test finished.")


if __name__ == "__main__":
    main()