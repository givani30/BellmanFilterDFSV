#!/usr/bin/env python
"""
Multiprocessing-based Comprehensive Optimizer Comparison Script for BIF.

This script runs a comprehensive experiment comparing all available optimizers
for the Bellman Information Filter (BIF) on a DFSV model, parallelizing
the optimizer runs using Python's multiprocessing.Pool for CPU-bound tasks.
It tracks:
- Final loss value
- Number of steps taken
- Success status
- Computation time
- Final parameter estimates

Results are saved to CSV files and printed as tables for easy comparison.
"""

import os
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
import multiprocessing

# JAX imports (still needed for model/optimization logic)
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

# Project specific imports
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.utils.transformations import apply_identification_constraint
from bellman_filter_dfsv.utils.optimization import (
    run_optimization,
    FilterType,
    OptimizerResult
)
from bellman_filter_dfsv.utils.solvers import get_available_optimizers

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)
# Disable JAX preallocation to potentially save memory for multiprocessing
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # Uncomment if memory issues arise


# --- Model and Data Generation Functions (Copied from original) ---
# These need to be defined at the top level for multiprocessing pickling

def create_simple_model(N: int = 3, K: int = 2) -> DFSVParamsDataclass:
    """Create a simple DFSV model with reasonable parameters."""
    key = jax.random.PRNGKey(42)
    key, subkey1 = jax.random.split(key)
    lambda_r_init = jax.random.normal(subkey1, (N, K)) * 0.5 + 0.5
    lambda_r = jnp.tril(lambda_r_init)
    diag_indices = jnp.diag_indices(n=min(N, K), ndim=2)
    lambda_r = lambda_r.at[diag_indices].set(1.0)

    key, subkey1 = jax.random.split(key)
    Phi_f_off = jax.random.uniform(subkey1, (K, K), minval=-0.05, maxval=0.05)
    key, subkey1 = jax.random.split(key)
    Phi_f_diag = jax.random.uniform(subkey1, (K,), minval=0.3, maxval=0.5)
    Phi_f = Phi_f_off
    phi_diag_indices = jnp.diag_indices(n=K, ndim=2)
    Phi_f = Phi_f.at[phi_diag_indices].set(Phi_f_diag)

    key, subkey1 = jax.random.split(key)
    Phi_h_off = jax.random.uniform(subkey1, (K, K), minval=-0.1, maxval=0.1)
    key, subkey1 = jax.random.split(key)
    Phi_h_diag = jax.random.uniform(subkey1, (K,), minval=0.8, maxval=0.95)
    Phi_h = Phi_h_off
    Phi_h = Phi_h.at[phi_diag_indices].set(Phi_h_diag)

    mu = jnp.array([-1.0, -0.5] if K == 2 else [-1.0] * K)
    key, subkey1 = jax.random.split(key)
    sigma2 = jax.random.uniform(subkey1, (N,), minval=0.05, maxval=0.1)
    key, subkey1 = jax.random.split(key)
    Q_h_diag = jax.random.uniform(subkey1, (K,), minval=0.1, maxval=0.3)
    Q_h = jnp.diag(Q_h_diag)

    params = DFSVParamsDataclass(
        N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, Phi_h=Phi_h,
        mu=mu, sigma2=sigma2, Q_h=Q_h
    )
    params = apply_identification_constraint(params)
    return params


def create_training_data(params: DFSVParamsDataclass, T: int = 1000, seed: int = 123) -> jnp.ndarray:
    """Generate simulation data for training."""
    returns, _, _ = simulate_DFSV(params=params, T=T, seed=seed)
    return returns


def create_uninformed_parameters(true_params: DFSVParamsDataclass, returns: jnp.ndarray) -> DFSVParamsDataclass:
    """Create uninformed initial parameters for optimization."""
    N, K = true_params.N, true_params.K
    lambda_r_init = jnp.ones((N, K)) * 0.5
    lambda_r_init = jnp.tril(lambda_r_init)
    diag_indices = jnp.diag_indices(n=min(N, K), ndim=2)
    lambda_r_init = lambda_r_init.at[diag_indices].set(1.0)
    initial_params = DFSVParamsDataclass(
        N=N, K=K, lambda_r=lambda_r_init, Phi_f=jnp.eye(K) * 0.4,
        Phi_h=jnp.eye(K) * 0.8, mu=jnp.zeros(K), sigma2=0.1 * jnp.ones(N),
        Q_h=0.2 * jnp.eye(K)
    )
    return initial_params


# --- Multiprocessing Worker Function ---
# Must be defined at the top level for pickling

def run_single_optimizer_config_mp(
    optimizer_name: str,
    filter_type: FilterType,
    returns: jnp.ndarray,
    true_params: Optional[DFSVParamsDataclass],
    use_transformations: bool,
    priors: Optional[Dict[str, Any]],
    stability_penalty_weight: float,
    max_steps: int,
    verbose: bool,
    log_params: bool,
    log_interval: int,
    fix_mu: bool
) -> OptimizerResult:
    """
    Worker function for multiprocessing. Runs optimization for a single config.
    Handles potential errors and returns an OptimizerResult.
    """
    # Note: JAX might re-initialize on each process. This is generally fine.
    # Consider disabling GPU usage if only using CPU parallelism to avoid conflicts.
    # try:
    #     jax.config.update('jax_platform_name', 'cpu')
    # except ValueError:
    #     pass # Ignore if platform is already CPU or setting fails

    print(f"--- Starting MP: Optimizer={optimizer_name} (Process {os.getpid()}) ---")
    start_time = time.time()
    try:
        # Call the original run_optimization function
        result = run_optimization(
            filter_type=filter_type,
            returns=returns,
            true_params=true_params if fix_mu else None,
            use_transformations=use_transformations,
            optimizer_name=optimizer_name,
            priors=priors,
            stability_penalty_weight=stability_penalty_weight,
            max_steps=max_steps,
            verbose=verbose, # Keep verbose off for less clutter
            log_params=log_params,
            log_interval=log_interval
        )

        # Add fix_mu info to the result (assuming OptimizerResult is pickleable)
        try:
            result_dict = vars(result)
        except TypeError:
            result_dict = result.__dict__
        result_dict['fix_mu'] = fix_mu
        final_result = OptimizerResult(**result_dict)

        # Determine success string for immediate feedback
        if hasattr(final_result, 'result_code') and final_result.result_code is not None:
            if final_result.result_code == optx.RESULTS.successful: success_str = "converged"
            elif final_result.result_code == optx.RESULTS.max_steps_reached: success_str = "reached max steps"
            else: success_str = f"code {final_result.result_code}"
        else:
            success_str = "success=True" if final_result.success else "success=False"

        print(f"--- Finished MP: Optimizer={optimizer_name} ({success_str}) | Loss={final_result.final_loss:.4e} | Steps={final_result.steps} | Time={final_result.time_taken:.2f}s ---")
        return final_result

    except Exception as e:
        end_time = time.time()
        print(f"--- Error MP: Optimizer={optimizer_name} failed in process {os.getpid()} after {end_time - start_time:.2f}s: {e} ---")
        # Return a dummy result indicating failure
        dummy_params = DFSVParamsDataclass(
            N=true_params.N if true_params else 0, K=true_params.K if true_params else 0,
            lambda_r=jnp.empty((0,0)), Phi_f=jnp.empty((0,0)), Phi_h=jnp.empty((0,0)),
            mu=jnp.empty(0), sigma2=jnp.empty(0), Q_h=jnp.empty((0,0))
        )
        return OptimizerResult(
            optimizer_name=optimizer_name, uses_transformations=use_transformations,
            fix_mu=fix_mu, success=False, final_loss=jnp.inf,
            final_params=dummy_params, steps=-1, time_taken=end_time - start_time,
            error_message=str(e), loss_history=None, param_history=None, result_code=None
        )


# --- Results Analysis Functions (Copied from original, filenames changed) ---

def print_results_table(results: List[OptimizerResult], max_steps: int):
    """Prints optimization results table."""
    print("\n\n--- Optimization Results (Multiprocessing) ---")
    print(f"{'Optimizer':<20} | {'Transform':<10} | {'Fix_mu':<7} | {'Status':<20} | {'Final Loss':<15} | {'Steps':<8} | {'Time (s)':<10} | {'Error Message'}")
    print("-" * 160)
    for res in sorted(results, key=lambda x: (x.optimizer_name, x.uses_transformations, x.fix_mu)):
        if hasattr(res, 'result_code') and res.result_code is not None:
            if res.result_code == optx.RESULTS.successful: status_str = "Converged"
            elif res.result_code == optx.RESULTS.max_steps_reached: status_str = "Max steps reached"
            elif res.result_code == optx.RESULTS.nonlinear_max_steps_reached: status_str = "Nonlinear max steps"
            elif res.result_code == optx.RESULTS.nonlinear_divergence: status_str = "Diverged"
            elif res.result_code == optx.RESULTS.singular: status_str = "Singular matrix"
            elif res.result_code == optx.RESULTS.breakdown: status_str = "Iterative breakdown"
            elif res.result_code == optx.RESULTS.stagnation: status_str = "Stagnation"
            else: status_str = str(res.result_code)
        else:
            status_str = "Converged" if res.success else ("Max steps reached" if res.steps >= max_steps and not res.error_message else "Failed")
        fix_mu_str = "Yes" if res.fix_mu else "No"
        loss_str = f"{res.final_loss:.4e}" if jnp.isfinite(res.final_loss) else "Inf/NaN"
        steps_str = str(res.steps) if res.steps >= 0 else "N/A"
        time_str = f"{res.time_taken:.2f}"
        error_str = res.error_message if res.error_message else "N/A"
        print(f"{res.optimizer_name:<20} | {'Yes' if res.uses_transformations else 'No':<10} | {fix_mu_str:<7} | {status_str:<20} | {loss_str:<15} | {steps_str:<8} | {time_str:<10} | {error_str}")
    print("-" * 160)

def print_parameter_comparison(results: List[OptimizerResult], true_params: DFSVParamsDataclass):
    """Prints parameter comparison."""
    print("\n\n--- Parameter Comparison (Multiprocessing) ---")
    true_values = {
        "lambda_r": true_params.lambda_r.flatten(), "Phi_f": true_params.Phi_f.flatten(),
        "Phi_h": true_params.Phi_h.flatten(), "mu": true_params.mu.flatten(),
        "sigma2": true_params.sigma2.flatten(), "Q_h": true_params.Q_h.flatten()
    }
    for res in sorted(results, key=lambda x: (x.optimizer_name, x.uses_transformations, x.fix_mu)):
        if res.final_params is not None and res.success: # Only show successful runs
            print(f"\n-- Run: Optimizer='{res.optimizer_name}' | Fix_mu='{'Yes' if res.fix_mu else 'No'}' --")
            print("-" * 80)
            print(f"{'Parameter':<10} | {'True Value':<35} | {'Estimated Value'}")
            print("-" * 80)
            est_values = {
                "lambda_r": res.final_params.lambda_r.flatten(), "Phi_f": res.final_params.Phi_f.flatten(),
                "Phi_h": res.final_params.Phi_h.flatten(), "mu": res.final_params.mu.flatten(),
                "sigma2": res.final_params.sigma2.flatten(), "Q_h": res.final_params.Q_h.flatten()
            }
            for param_name, true_val in true_values.items():
                est_val = est_values.get(param_name, np.array([])) # Handle potential missing keys if dummy params used
                true_str = np.array2string(true_val, precision=4, separator=', ')
                est_str = np.array2string(est_val, precision=4, separator=', ')
                print(f"{param_name:<10} | {true_str:<35} | {est_str}")

def save_results_to_csv(results: List[OptimizerResult]):
    """Saves results to CSV."""
    print("\nSaving results to CSV...")
    os.makedirs("outputs", exist_ok=True)
    csv_path = os.path.join("outputs", "optimizer_comparison_mp_results.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Optimizer", "Transformations", "Fix_mu", "Success", "Final Loss", "Steps", "Time (s)", "Error Message"])
        for res in results:
            writer.writerow([
                res.optimizer_name, "Yes" if res.uses_transformations else "No",
                "Yes" if res.fix_mu else "No", "Yes" if res.success else "No",
                res.final_loss if jnp.isfinite(res.final_loss) else "Inf/NaN",
                res.steps if res.steps >= 0 else "N/A", f"{res.time_taken:.2f}",
                res.error_message if not res.success else "N/A"
            ])
    print(f"  Results saved to {csv_path}")

def save_parameter_errors_to_csv(results: List[OptimizerResult], true_params: DFSVParamsDataclass):
    """Saves parameter errors to CSV."""
    print("\nSaving parameter errors to CSV...")
    os.makedirs("outputs", exist_ok=True)
    true_values = {
        "lambda_r": true_params.lambda_r.flatten(), "Phi_f": true_params.Phi_f.flatten(),
        "Phi_h": true_params.Phi_h.flatten(), "mu": true_params.mu.flatten(),
        "sigma2": true_params.sigma2.flatten(), "Q_h": true_params.Q_h.flatten()
    }
    csv_path = os.path.join("outputs", "parameter_estimation_errors_mp.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ["Optimizer", "Transformations", "Fix_mu", "Success", "Final Loss"]
        header.extend([f"{p}_RMSE" for p in true_values] + [f"{p}_MAE" for p in true_values])
        writer.writerow(header)
        for res in results:
            if res.final_params is not None and res.success:
                row = [
                    res.optimizer_name, "Yes" if res.uses_transformations else "No",
                    "Yes" if res.fix_mu else "No", "Yes",
                    res.final_loss if jnp.isfinite(res.final_loss) else "Inf/NaN"
                ]
                errors_rmse = []
                errors_mae = []
                for param_name, true_val in true_values.items():
                    est_val = getattr(res.final_params, param_name, jnp.empty(0)).flatten()
                    if est_val.size == true_val.size:
                        rmse = jnp.sqrt(jnp.mean((true_val - est_val) ** 2))
                        mae = jnp.mean(jnp.abs(true_val - est_val))
                        errors_rmse.append(f"{rmse:.6f}")
                        errors_mae.append(f"{mae:.6f}")
                    else: # Handle size mismatch (e.g. from dummy params)
                        errors_rmse.append("N/A")
                        errors_mae.append("N/A")
                row.extend(errors_rmse)
                row.extend(errors_mae)
                writer.writerow(row)
    print(f"  Parameter errors saved to {csv_path}")

def plot_loss_history(results: List[OptimizerResult]):
    """Plots loss history."""
    print("\nGenerating loss history plots...")
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(12, 8))
    for res in results:
        if res.loss_history is not None and len(res.loss_history) > 0:
            label = f"{res.optimizer_name} ({'T' if res.uses_transformations else 'U'})"
            loss_hist_np = np.array(res.loss_history) # Ensure NumPy array
            plt.plot(loss_hist_np, label=label)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss History by Optimizer (Multiprocessing Run)')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plot_path = os.path.join("outputs", "loss_history_mp.png")
    plt.savefig(plot_path)
    print(f"  Loss history plot saved to {plot_path}")


# --- Main Execution ---

def main():
    """Main function to run the multiprocessing optimizer comparison."""
    print("Starting Multiprocessing Comprehensive Optimizer Comparison...")

    # 1. Create model parameters
    N, K = 5, 2
    true_params = create_simple_model(N=N, K=K)
    print(f"Created model with N={true_params.N}, K={true_params.K}")

    # 2. Generate simulation data
    T = 1500
    print(f"\nGenerating {T} time steps of simulation data...")
    returns = create_training_data(true_params, T=T, seed=123)
    print("Simulation data generated.")

    # 3. Get all available optimizers
    available_optimizers_dict = get_available_optimizers()
    optimizer_names = list(available_optimizers_dict.keys())
    print(f"\nFound {len(optimizer_names)} available optimizers to test:")
    # print(f"  {', '.join(optimizer_names)}") # Keep concise

    # 4. Define optimization configurations
    filter_type = FilterType.BIF
    use_transformations = True
    fix_mu = True
    max_steps = 500
    stability_penalty_weight = 1000.0
    verbose = False # Keep verbose off for parallel runs
    log_params = False
    log_interval = 1
    priors = None

    # 5. Prepare arguments for multiprocessing
    run_args = []
    for name in optimizer_names:
        run_args.append((
            name, filter_type, returns, true_params, use_transformations,
            priors, stability_penalty_weight, max_steps, verbose,
            log_params, log_interval, fix_mu
        ))

    # 6. Run optimizations using multiprocessing Pool
    num_processes = multiprocessing.cpu_count()
    print(f"\nRunning optimizations in parallel using {num_processes} processes...")
    overall_start_time = time.time()

    # Use try-finally to ensure pool cleanup
    pool = None
    try:
        # Create the pool
        # Consider 'spawn' context if 'fork' causes issues (esp. with JAX/GPU)
        # context = multiprocessing.get_context('spawn')
        # pool = context.Pool(processes=num_processes)
        pool = multiprocessing.Pool(processes=num_processes)

        # Run the tasks using starmap
        results = pool.starmap(run_single_optimizer_config_mp, run_args)

    except Exception as e:
        print(f"\nError during multiprocessing execution: {e}")
        results = [] # Ensure results is an empty list on error
    finally:
        if pool:
            pool.close() # Prevent new tasks
            pool.join()  # Wait for worker processes to exit

    overall_end_time = time.time()
    print(f"\nParallel execution finished in {overall_end_time - overall_start_time:.2f} seconds.")

    # 7. Process results
    valid_results = [res for res in results if res is not None and res.steps != -1]

    if valid_results:
        print(f"\nProcessing {len(valid_results)} valid results...")
        # 8. Print results table
        print_results_table(valid_results, max_steps)

        # 9. Print parameter comparison
        print_parameter_comparison(valid_results, true_params)

        # 10. Save results
        save_results_to_csv(valid_results)
        save_parameter_errors_to_csv(valid_results, true_params)
        plot_loss_history(valid_results)
    else:
        print("\nNo valid results to display. All optimizations may have failed or pool execution failed.")

    print("\nMultiprocessing Comprehensive Optimizer Comparison completed.")


# Crucial for multiprocessing: Protect the main execution
if __name__ == "__main__":
    # Optional: Set start method (useful on macOS/Windows or if 'fork' causes issues)
    # try:
    #     multiprocessing.set_start_method('spawn') # Alternatives: 'fork', 'forkserver'
    # except RuntimeError:
    #     pass # Ignore if already set or not applicable
    main()