#!/usr/bin/env python
"""
Parameter Recovery Check for DFSV Bellman Information Filter (BIF).

This script performs parameter recovery simulations for the BIF, focusing on
configurations involving full persistence matrices (Phi_f, Phi_h). It runs
multiple replicates for a specified experimental setup, optimizing the
pseudo log-likelihood objective function and storing the results.

Based on the plan: memory-bank/plans/parameter_recovery_check_plan_07-04-2025.md
"""

import argparse
import time
import os
import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import optax
import equinox as eqx
from collections import namedtuple
import dataclasses
from typing import Any, Dict, Tuple, Optional # Added Tuple, Optional

# Project specific imports
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.utils.transformations import transform_params, untransform_params
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.filters.objectives import bellman_objective # Only need base objective now

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)
# For debugging JIT errors
# eqx.error_if(..., EQX_ON_ERROR="breakpoint") # Uncomment for debugging

# --- Constants ---
N_DEFAULT = 5
K_DEFAULT = 2
T_DEFAULT = 2000
MAX_STEPS = 1000  # Max optimization steps
STABILITY_PENALTY_WEIGHT = 1000.0 # Penalty for full Phi matrices
DEFAULT_SEED = 20240407

# --- Result Structure ---
ReplicateResult = namedtuple("ReplicateResult", [
    "replicate_id",
    "success",
    "final_loss",
    "steps",
    "final_params_untransformed", # Store the final untransformed params Pytree
    "time_taken",
    "error_message"
])

# --- Helper Functions ---

def apply_identification_constraint(params: DFSVParamsDataclass) -> DFSVParamsDataclass:
    """Applies lower-triangular constraint with diagonal fixed to 1 to lambda_r."""
    # Apply lower-triangular constraint first
    constrained_lambda_r = jnp.tril(params.lambda_r)
    # Set diagonal elements to 1.0
    # Ensure we only set for min(N, K) diagonal elements
    diag_indices = jnp.diag_indices(n=min(params.N, params.K), ndim=2)
    constrained_lambda_r = constrained_lambda_r.at[diag_indices].set(1.0)
    return dataclasses.replace(params, lambda_r=constrained_lambda_r)

def create_true_params(N: int, K: int, seed: int = 42) -> DFSVParamsDataclass:
    """Creates true DFSV parameters for simulation with constraints."""
    print(f"Creating true parameters with N={N}, K={K}, seed={seed}")
    key = jax.random.PRNGKey(seed)
    key_lambda, key_phi_f, key_phi_h, key_mu, key_sigma, key_q = jax.random.split(key, 6)

    # Factor loadings (Lambda_r): Lower triangular, diag=1
    lambda_r_init = jax.random.normal(key_lambda, (N, K)) * 0.3 + 0.5 # Smaller variance init
    lambda_r = jnp.tril(lambda_r_init)
    diag_indices = jnp.diag_indices(n=min(N, K), ndim=2)
    lambda_r = lambda_r.at[diag_indices].set(1.0)

    # Factor persistence (Phi_f, Phi_h): Full matrices, stable
    def create_stable_matrix(key, size, max_eig=0.98, min_eig=0.8):
        matrix = jax.random.uniform(key, (size, size), minval=-0.2, maxval=0.2)
        diag_vals = jax.random.uniform(key, (size,), minval=min_eig, maxval=max_eig)
        matrix = matrix.at[jnp.diag_indices(size)].set(diag_vals)
        try:
            eigs = jnp.linalg.eigvals(matrix)
            max_abs_eig = jnp.max(jnp.abs(eigs))
            if max_abs_eig >= 1.0:
                print(f"Warning: Rescaling matrix, max eigenvalue was {max_abs_eig:.4f}")
                matrix = matrix / (max_abs_eig + 1e-4) # Rescale slightly below 1
        except jnp.linalg.LinAlgError:
             print("Warning: Eigendecomposition failed during stable matrix creation.")
             diag_vals = jax.random.uniform(key, (size,), minval=min_eig, maxval=max_eig)
             matrix = jnp.diag(diag_vals)
        return matrix

    Phi_f = create_stable_matrix(key_phi_f, K, max_eig=0.98, min_eig=0.85)
    Phi_h = create_stable_matrix(key_phi_h, K, max_eig=0.99, min_eig=0.9)

    # Long-run mean for log-volatilities (mu)
    mu = jax.random.normal(key_mu, (K,)) * 0.5 - 1.0 # Centered around -1

    # Idiosyncratic variance (sigma2) - positive
    sigma2 = jax.random.uniform(key_sigma, (N,), minval=0.05, maxval=0.2)

    # Log-volatility noise covariance (Q_h) - diagonal, positive
    q_diag = jax.random.uniform(key_q, (K,), minval=0.1, maxval=0.3)
    Q_h = jnp.diag(q_diag)

    params = DFSVParamsDataclass(
        N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, Phi_h=Phi_h, mu=mu, sigma2=sigma2, Q_h=Q_h
    )
    params = apply_identification_constraint(params)
    filter_instance = DFSVBellmanInformationFilter(N, K)
    params = filter_instance._process_params(params)
    return params

def create_initial_guess(N: int, K: int) -> DFSVParamsDataclass:
    """Creates a fixed, uninformed initial guess for DFSV parameters."""
    print("  Creating fixed, uninformed initial guess...")

    # Fixed values
    mu_guess = jnp.ones(K)*-1
    sigma2_guess = jnp.ones(N) * 0.1
    Q_h_guess = jnp.eye(K) * 0.1
    Phi_f_guess = jnp.eye(K) * 0.8
    Phi_h_guess = jnp.eye(K) * 0.9

    # Lambda_r: Lower triangular with 1s on the diagonal
    # Start with ones, make lower triangular, then set diagonal to 1
    lambda_r_guess = jnp.tril(jnp.ones((N, K))) # Create lower tri ones
    diag_indices = jnp.diag_indices(n=min(N, K), ndim=2)
    lambda_r_guess = lambda_r_guess.at[diag_indices].set(1.0)

    initial_guess_params = {
        'N': N,
        'K': K,
        'mu': mu_guess,
        'sigma2': sigma2_guess,
        'Q_h': Q_h_guess,
        'lambda_r': lambda_r_guess,
        'Phi_f': Phi_f_guess,
        'Phi_h': Phi_h_guess
    }

    initial_guess = DFSVParamsDataclass(**initial_guess_params)
    # Apply constraints and processing
    initial_guess = apply_identification_constraint(initial_guess)
    filter_instance = DFSVBellmanInformationFilter(N, K)
    initial_guess = filter_instance._process_params(initial_guess)
    return initial_guess


@eqx.filter_jit
def objective_wrapper(
    y: Any, # The parameters being optimized (transformed Pytree)
    args_tuple: Tuple[DFSVBellmanInformationFilter, jax.Array, jax.Array, bool, bool] # Static args
) -> jax.Array:
    """
    Wrapper for the objective function, compatible with optx.minimise(args=...).

    Handles untransforming parameters, applying constraints (lambda_r),
    fixing mu if required, calculating the base objective, and adding stability penalty.
    """
    # Unpack static arguments
    filter_instance, observations, true_mu, fix_mu_flag, phi_f_is_diagonal_flag = args_tuple

    # 1. Untransform parameters being optimized
    params_iter = untransform_params(y)

    # 2. Apply constraints and fixes
    # 2a. Apply lambda_r identification constraint
    params_constrained = apply_identification_constraint(params_iter)
    # 2b. Fix mu if needed
    if fix_mu_flag:
        params_constrained = eqx.tree_at(lambda p: p.mu, params_constrained, true_mu)
    # 2c. Ensure diagonal Phi_f if needed (should be handled by untransform, but double-check)
    if phi_f_is_diagonal_flag:
        params_constrained = dataclasses.replace(params_constrained, Phi_f=jnp.diag(jnp.diag(params_constrained.Phi_f)))

    # 3. Calculate the base objective using the constrained/fixed parameters
    base_loss = bellman_objective(
        params=params_constrained,
        y=observations,
        filter=filter_instance,
        priors=None # No priors for this experiment
    )

    # 4. Add stability penalty if Phi_f/Phi_h are full
    #    Penalty is applied *after* untransforming, based on the untransformed values.
    stability_penalty = 0.0
    if not phi_f_is_diagonal_flag: # If Phi_f is full (Phi_h is always full here)
        # Penalty based on eigenvalues of untransformed Phi_f and Phi_h
        try:
            eig_f = jnp.linalg.eigvals(params_constrained.Phi_f)
            penalty_f = jax.nn.relu(jnp.max(jnp.abs(eig_f)) - 1.0)
        except jnp.linalg.LinAlgError:
            print("Warning: Eigendecomposition failed for Phi_f penalty calculation.")
            penalty_f = 1e6 # Assign large penalty if decomp fails

        try:
            eig_h = jnp.linalg.eigvals(params_constrained.Phi_h)
            penalty_h = jax.nn.relu(jnp.max(jnp.abs(eig_h)) - 1.0)
        except jnp.linalg.LinAlgError:
            print("Warning: Eigendecomposition failed for Phi_h penalty calculation.")
            penalty_h = 1e6 # Assign large penalty if decomp fails

        stability_penalty = STABILITY_PENALTY_WEIGHT * (penalty_f + penalty_h)

    total_loss = base_loss + stability_penalty
    return total_loss


def run_single_replicate(
    replicate_id: int,
    num_replicates: int,
    true_params: DFSVParamsDataclass,
    observations: jax.Array,
    initial_guess: DFSVParamsDataclass, # Untransformed initial guess
    config: Dict[str, Any]
) -> ReplicateResult:
    """Runs the optimization for a single replicate."""
    print(f"--- Starting Replicate {replicate_id+1}/{num_replicates} ---")
    start_time = time.time()
    error_msg = "N/A"
    final_params_untransformed = None
    final_loss = jnp.inf
    steps = -1
    success = False

    # --- Print Initial Guess (Constrained) ---
    print("  Initial Guess (Constrained):")
    np.set_printoptions(precision=4, suppress=True)
    print(f"    mu: {np.asarray(initial_guess.mu)}")
    print(f"    sigma2: {np.asarray(initial_guess.sigma2)}")
    print(f"    Q_h (diag): {np.asarray(jnp.diag(initial_guess.Q_h))}")
    print(f"    lambda_r:\n{np.asarray(initial_guess.lambda_r)}")
    print(f"    Phi_f:\n{np.asarray(initial_guess.Phi_f)}")
    print(f"    Phi_h:\n{np.asarray(initial_guess.Phi_h)}")
    np.set_printoptions() # Reset default

    try:
        filter_instance = DFSVBellmanInformationFilter(true_params.N, true_params.K)

        # Package static arguments for the objective wrapper
        static_args = (
            filter_instance,
            observations,
            true_params.mu, # Pass true mu for potential fixing
            config['fix_mu'],
            config['phi_f_is_diagonal']
        )

        # --- Calculate Objective at True Parameters ---
        try:
            print("  Calculating objective at TRUE parameters...")
            true_params_transformed = transform_params(true_params)
            # Ensure true Phi_f is diagonal if config requires it (for consistent objective calc)
            if config['phi_f_is_diagonal']:
                 true_params_transformed = eqx.tree_at(
                     lambda p: p.Phi_f,
                     true_params_transformed,
                     transform_params(dataclasses.replace(true_params, Phi_f=jnp.diag(jnp.diag(true_params.Phi_f)))).Phi_f
                 )

            obj_at_true = objective_wrapper(true_params_transformed, static_args)
            if not jnp.isfinite(obj_at_true):
                 print(f"  WARNING: Objective at true parameters is non-finite: {obj_at_true}")
            else:
                 print(f"  Objective at True Params: {obj_at_true:.4f}")
        except Exception as true_obj_e:
            print(f"  ERROR calculating objective at true parameters: {true_obj_e}")


        # Transform the initial guess Pytree for optimization
        if config['phi_f_is_diagonal']:
             initial_guess = dataclasses.replace(initial_guess, Phi_f=jnp.diag(jnp.diag(initial_guess.Phi_f)))
        initial_y = transform_params(initial_guess)

        # Optimizer setup (AdamW with cosine decay schedule and RMS norm)
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=1e-3, peak_value=1e-1, end_value=1e-4,
            warmup_steps=int(MAX_STEPS * 0.1), decay_steps=MAX_STEPS # Adjust decay steps
        )
        optimizer = optax.apply_if_finite(optax.adamw(learning_rate=scheduler), max_consecutive_errors=10)
        solver = optx.OptaxMinimiser(optimizer, rtol=1e-3, atol=1e-5, norm=optx.rms_norm, verbose=frozenset({"loss"}))

        # --- Calculate Initial Objective (at Initial Guess) ---
        try:
            print("  Calculating initial objective (at initial guess)...")
            initial_loss_val = objective_wrapper(initial_y, static_args)
            if not jnp.isfinite(initial_loss_val):
                 raise ValueError(f"Initial objective is non-finite: {initial_loss_val}")
            print(f"  Initial Objective Loss (at guess): {initial_loss_val:.4f}")
        except Exception as init_e:
            error_msg = f"Initial obj failed: {init_e}"
            print(f"  ERROR during initial objective calculation: {init_e}")
            time_taken = time.time() - start_time
            return ReplicateResult(replicate_id, False, float(jnp.inf), 0, None, time_taken, error_msg)

        # --- Run Optimization ---
        print(f"  Running optimization (max_steps={MAX_STEPS})...")
        sol = optx.minimise(
            fn=objective_wrapper, # Pass the wrapper directly
            solver=solver,
            y0=initial_y,
            args=static_args, # Pass static args via 'args' parameter
            max_steps=MAX_STEPS,
            throw=False # Don't raise errors, check sol.result
        )

        # --- Process Results ---
        steps = sol.stats.get('num_steps', -1)
        solver_success = (sol.result == optx.RESULTS.successful)
        final_loss = jnp.inf # Initialize loss to Inf
        final_params_untransformed = None # Initialize params to None
        loss_finite = False
        success = False # Initialize overall success to False

        if solver_success:
            final_t_params = sol.value
            try:
                final_loss_recalc = objective_wrapper(final_t_params, static_args)
                loss_finite = jnp.isfinite(final_loss_recalc)

                if loss_finite:
                    final_loss = final_loss_recalc # Assign calculated loss
                    try:
                        final_params_untransformed_temp = untransform_params(final_t_params)
                        # Apply constraints and fixes
                        final_params_untransformed = apply_identification_constraint(final_params_untransformed_temp)
                        if config['fix_mu']:
                            final_params_untransformed = eqx.tree_at(lambda p: p.mu, final_params_untransformed, true_params.mu)
                        if config['phi_f_is_diagonal']:
                             final_params_untransformed = dataclasses.replace(final_params_untransformed, Phi_f=jnp.diag(jnp.diag(final_params_untransformed.Phi_f)))
                        success = True # Mark overall success
                        error_msg = "N/A"
                    except Exception as untransform_e:
                        print(f"  ERROR untransforming final parameters: {untransform_e}")
                        error_msg = f"Untransform failed: {untransform_e}"
                        success = False # Mark as failure
                        final_params_untransformed = None # Ensure params are None
                        final_loss = jnp.inf # Loss is invalid if untransform failed
                else:
                    error_msg = f"Final loss non-finite: {final_loss_recalc}"
                    success = False
                    final_loss = jnp.inf # Ensure loss is Inf

            except Exception as recalc_e:
                print(f"  ERROR recalculating final loss: {recalc_e}")
                error_msg = f"Final loss recalc failed: {recalc_e}"
                success = False
                final_loss = jnp.inf # Ensure loss is Inf
        else:
            error_msg = f"Solver failed: {optx.RESULTS[sol.result]}" # Use lookup for readable message
            success = False
            final_loss = jnp.inf # Ensure loss is Inf
            final_params_untransformed = None # Ensure params are None

    except Exception as e:
        error_msg = f"Exception during optimization setup/run: {str(e)}"
        print(f"  ERROR during optimization for replicate {replicate_id+1}: {e}")
        success = False
        final_loss = jnp.inf # Assign Inf on exception

    time_taken = time.time() - start_time
    print(f"  Replicate {replicate_id+1} {'Success' if success else 'Failed'}. Loss: {float(final_loss):.4f}, Steps: {steps}, Time: {time_taken:.2f}s")
    if not success:
        print(f"    Failure Reason: {error_msg}")

    return ReplicateResult(
        replicate_id=replicate_id,
        success=success,
        final_loss=float(final_loss), # Convert JAX scalar to float
        steps=int(steps),
        final_params_untransformed=final_params_untransformed, # Store the Pytree
        time_taken=time_taken,
        error_message=error_msg
    )


def print_param_comparison(true_p: DFSVParamsDataclass, est_p: Optional[DFSVParamsDataclass], replicate_id: int):
    """Prints a comparison table for true vs estimated parameters."""
    if est_p is None:
        print(f"  Replicate {replicate_id+1}: Estimation failed, cannot compare parameters.")
        return

    print(f"\n--- Parameter Comparison: Replicate {replicate_id+1} ---")
    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    print("  Parameter | True Value        | Estimated Value")
    print("-" * 50)

    # Mu
    print(f"  mu        | {np.asarray(true_p.mu)} | {np.asarray(est_p.mu)}")

    # Sigma2
    print(f"  sigma2    | {np.asarray(true_p.sigma2)} | {np.asarray(est_p.sigma2)}")

    # Q_h (diagonal)
    print(f"  Q_h (diag)| {np.asarray(jnp.diag(true_p.Q_h))} | {np.asarray(jnp.diag(est_p.Q_h))}")

    # Lambda_r
    print("  lambda_r:")
    true_lr_str = np.array2string(np.asarray(true_p.lambda_r), prefix="    True: ")
    est_lr_str = np.array2string(np.asarray(est_p.lambda_r), prefix="    Est:  ")
    print(f"    True:\n{true_lr_str}")
    print(f"    Est:\n{est_lr_str}")

    # Phi_f
    print("  Phi_f:")
    true_pf_str = np.array2string(np.asarray(true_p.Phi_f), prefix="    True: ")
    est_pf_str = np.array2string(np.asarray(est_p.Phi_f), prefix="    Est:  ")
    print(f"    True:\n{true_pf_str}")
    print(f"    Est:\n{est_pf_str}")

    # Phi_h
    print("  Phi_h:")
    true_ph_str = np.array2string(np.asarray(true_p.Phi_h), prefix="    True: ")
    est_ph_str = np.array2string(np.asarray(est_p.Phi_h), prefix="    Est:  ")
    print(f"    True:\n{true_ph_str}")
    print(f"    Est:\n{est_ph_str}")

    print("-" * 50)
    np.set_printoptions() # Reset default


def main(args):
    """Main function to run the parameter recovery experiment."""
    print(f"--- Starting Parameter Recovery Check ---")
    print(f"Experiment: {args.experiment}")
    print(f"N={args.N}, K={args.K}, T={args.T}, Replicates={args.replicates}, Seed={args.seed}")

    # Determine configuration based on experiment number
    if args.experiment == 1:
        config = {'fix_mu': False, 'phi_f_is_diagonal': False}
        config_desc = "Free mu, Full Phi_f/Phi_h"
    elif args.experiment == 11: # Use 11 for 1b
         config = {'fix_mu': True, 'phi_f_is_diagonal': False}
         config_desc = "Fixed mu, Full Phi_f/Phi_h"
    elif args.experiment == 2:
         config = {'fix_mu': True, 'phi_f_is_diagonal': True}
         config_desc = "Fixed mu, Diagonal Phi_f, Full Phi_h"
    else:
        raise ValueError(f"Invalid experiment number: {args.experiment}. Use 1, 11, or 2.")
    print(f"Configuration: {config_desc}")
    print(f"Stability Penalty Weight: {'N/A' if config['phi_f_is_diagonal'] else STABILITY_PENALTY_WEIGHT}")

    # Create true parameters (same for all replicates)
    true_params = create_true_params(args.N, args.K, seed=args.seed)
    print("\n--- True Parameters ---")
    np.set_printoptions(precision=4, suppress=True)
    print(f"  mu: {np.asarray(true_params.mu)}")
    print(f"  sigma2: {np.asarray(true_params.sigma2)}")
    print(f"  Q_h (diag): {np.asarray(jnp.diag(true_params.Q_h))}")
    print(f"  lambda_r:\n{np.asarray(true_params.lambda_r)}")
    print(f"  Phi_f:\n{np.asarray(true_params.Phi_f)}")
    print(f"  Phi_h:\n{np.asarray(true_params.Phi_h)}")
    np.set_printoptions() # Reset default

    # PRNG Key Management
    main_key = jax.random.PRNGKey(args.seed + 1)

    all_results = []
    for i in range(args.replicates):
        rep_key, main_key = jax.random.split(main_key)
        sim_key, init_key = jax.random.split(rep_key)

        # Simulate data for this replicate
        sim_seed_val = int(sim_key[0])
        print(f"\nGenerating simulation data for Replicate {i+1} (Sim Seed: {sim_seed_val})...")
        observations, _, _ = simulate_DFSV(true_params, T=args.T, seed=sim_seed_val)
        observations = jnp.asarray(observations)

        # Create initial guess for this replicate (using fixed values)
        initial_guess = create_initial_guess(args.N, args.K)

        # Run the optimization for this replicate
        result = run_single_replicate(i, args.replicates, true_params, observations, initial_guess, config)
        all_results.append(result)

    # --- Post-Processing ---
    print("\n--- All Replicates Completed ---")
    successful_runs = sum(1 for r in all_results if r.success)
    print(f"Total Successful Runs: {successful_runs}/{args.replicates}")

    # Print summary of results
    print("\n--- Results Summary ---")
    print(f"{'Rep':<4} | {'Success':<8} | {'Final Loss':<15} | {'Steps':<8} | {'Time (s)':<10} | {'Error'}")
    print("-" * 70) # Adjusted width
    for res in all_results:
        success_str = "Yes" if res.success else "No"
        loss_str = f"{res.final_loss:.4e}" if np.isfinite(res.final_loss) else "Inf/NaN"
        steps_str = str(res.steps) if res.steps >= 0 else "N/A"
        time_str = f"{res.time_taken:.2f}"
        error_str = res.error_message if not res.success else "N/A"
        print(f"{res.replicate_id+1:<4} | {success_str:<8} | {loss_str:<15} | {steps_str:<8} | {time_str:<10} | {error_str}")
    print("-" * 70) # Adjusted width

    # --- Parameter Comparison for Successful Runs ---
    print("\n--- Parameter Comparison (Successful Runs Only) ---")
    for res in all_results:
        if res.success:
            print_param_comparison(true_params, res.final_params_untransformed, res.replicate_id)
        # else: # Optional: print a message for failed runs
        #     print(f"\n--- Replicate {res.replicate_id+1}: Failed, skipping parameter comparison. ---")


    # --- Save Results ---
    # As per instructions, storing in memory is sufficient for this task.
    # Example saving code (commented out):
    # output_dir = "outputs/parameter_recovery_checks"
    # os.makedirs(output_dir, exist_ok=True)
    # output_filename = os.path.join(output_dir, f"exp{args.experiment}_N{args.N}K{args.K}T{args.T}_seed{args.seed}_results.pkl") # Added seed to filename
    # try:
    #     results_payload = {
    #         'args': vars(args),
    #         'config': config,
    #         'config_desc': config_desc,
    #         'true_params': true_params, # Store true params once
    #         'results': all_results # Contains ReplicateResult namedtuples
    #     }
    #     with open(output_filename, 'wb') as f:
    #         cloudpickle.dump(results_payload, f)
    #     print(f"\nFull results object saved to {output_filename}")
    # except Exception as e:
    #     print(f"\nERROR saving results to {output_filename}: {e}")

    print("\nScript finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BIF Parameter Recovery Check")
    parser.add_argument('--experiment', type=int, default=1, choices=[1, 11, 2],
                        help='Experiment number (1: Free mu/Full Phi_f, 11: Fixed mu/Full Phi_f, 2: Fixed mu/Diag Phi_f)')
    parser.add_argument('--N', type=int, default=N_DEFAULT, help='Number of assets')
    parser.add_argument('--K', type=int, default=K_DEFAULT, help='Number of factors')
    parser.add_argument('--T', type=int, default=T_DEFAULT, help='Number of time steps')
    parser.add_argument('--replicates', type=int, default=1, help='Number of simulation replicates')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Base PRNG seed for true params and simulation keys')

    args = parser.parse_args()
    main(args)