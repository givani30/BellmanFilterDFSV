#!/usr/bin/env python
"""
Debug Script for DFSV Bellman Information Filter at True Parameters.

This script investigates why the bellman_objective function might return a
penalty value (indicating failure) when evaluated using the true parameters
that generated the simulation data. It runs the filter step-by-step for the
initial time points if the objective calculation fails, printing internal states.
"""

import jax
import jax.numpy as jnp
import numpy as np
import dataclasses

# Project specific imports (adjust paths if necessary)
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.filters.objectives import bellman_objective
# from bellman_filter_dfsv.utils.jax_helpers import safe_slogdet, safe_solve, safe_norm # Removed unused import causing error

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# --- Model and Data Generation (Copied from test_bif_priors_optimizers.py) ---

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
    # Idiosyncratic variance (diagonal) - Using values from the failing run
    sigma2 = np.random.uniform(0.05, 0.1, N)
    # Log-volatility noise covariance - Using values from the failing run
    Q_h = np.array([[0.1]]) if K == 1 else np.diag(np.random.uniform(0.1, 0.3, K))

    params = DFSVParamsDataclass(
        N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, Phi_h=Phi_h, mu=mu, sigma2=sigma2, Q_h=Q_h
    )
    return params

def create_training_data(params, T=1000, seed=42):
    """Generate simulated data for training."""
    returns, _, _ = simulate_DFSV(params, T=T, seed=seed)
    return jnp.asarray(returns) # Return as JAX array

# --- Debugging Logic ---

def check_matrix_stability(matrix, name="Matrix"):
    """Checks a matrix for NaNs/Infs and condition number."""
    if not jnp.all(jnp.isfinite(matrix)):
        print(f"  WARNING: {name} contains non-finite values!")
        return False
    try:
        cond_num = jnp.linalg.cond(matrix)
        print(f"  Condition Number ({name}): {cond_num:.4e}")
        if cond_num > 1e10: # Threshold for ill-conditioning
             print(f"  WARNING: {name} might be ill-conditioned!")
    except jnp.linalg.LinAlgError:
        print(f"  WARNING: Could not compute condition number for {name} (likely singular).")
    return True

def main():
    print("Starting BIF Debugging at True Parameters...")

    # --- Setup ---
    N_val = 3
    K_val = 1
    T_val = 1500 # Match the test script
    seed_val = 123

    true_params = create_simple_model(N=N_val, K=K_val)
    print(f"Using model N={true_params.N}, K={true_params.K}")
    print("True Parameters:")
    for field in dataclasses.fields(true_params):
        print(f"  {field.name}: {getattr(true_params, field.name)}")


    print(f"\nGenerating {T_val} time steps of simulation data (seed={seed_val})...")
    returns = create_training_data(true_params, T=T_val, seed=seed_val)
    print("Simulation data generated.")

    # Instantiate filter and process true params
    filter_instance = DFSVBellmanInformationFilter(true_params.N, true_params.K)
    true_params_jax = filter_instance._process_params(true_params)
    print("\nTrue Parameters processed for JAX.")

    # --- Run Step-by-Step Loop with NaN/Inf Checks ---
    print("\nRunning filter step-by-step with NaN/Inf checks...")

    # Get initial state from filter and unpack the tuple
    a_prev, Omega_prev = filter_instance.initialize_state(true_params_jax)
    log_lik_cumulative = 0.0
    nan_detected = False

    print(f"\nInitial State (a_0): {a_prev}")
    print("Initial Information (Omega_0):")
    print(Omega_prev)

    for t in range(T_val):

        print(f"\n--- Time Step t={t+1} ---")
        y_t = returns[t]
        # print(f"Observation y_t: {y_t}") # Can uncomment for more detail

        try:
            # 1. Prediction Step
            # print("Predicting...") # Can uncomment for more detail
            a_pred, Omega_pred = filter_instance.predict_jax_info_jit(true_params_jax, a_prev, Omega_prev)

            # Check for NaN/Inf after prediction
            if not jnp.all(jnp.isfinite(a_pred)):
                 print(f"  ERROR: NaN/Inf detected in predicted state a_{t+1|t} at step {t+1}")
                 print(a_pred)
                 nan_detected = True
                 break
            if not jnp.all(jnp.isfinite(Omega_pred)):
                 print(f"  ERROR: NaN/Inf detected in predicted info Omega_{t+1|t} at step {t+1}")
                 print(Omega_pred)
                 nan_detected = True
                 break
            # print(f"  Predicted State (a_{t+1|t}): {a_pred.flatten()}") # Can uncomment
            # print(f"  Predicted Information (Omega_{t+1|t}):") # Can uncomment
            # print(Omega_pred) # Can uncomment


            # 2. Update Step
            # print("Updating...") # Can uncomment for more detail
            a_updated, Omega_updated, step_lik = filter_instance.update_jax_info_jit(
                true_params_jax, a_pred, Omega_pred, y_t
            )

            # Check for NaN/Inf after update
            if not jnp.all(jnp.isfinite(a_updated)):
                 print(f"  ERROR: NaN/Inf detected in updated state a_{t+1|t+1} at step {t+1}")
                 print(a_updated)
                 nan_detected = True
                 break
            if not jnp.all(jnp.isfinite(Omega_updated)):
                 print(f"  ERROR: NaN/Inf detected in updated info Omega_{t+1|t+1} at step {t+1}")
                 print(Omega_updated)
                 nan_detected = True
                 break
            if not jnp.all(jnp.isfinite(step_lik)):
                 print(f"  ERROR: NaN/Inf detected in step likelihood at step {t+1}")
                 print(step_lik)
                 nan_detected = True
                 break
            # print(f"  Updated State (a_{t+1|t+1}): {a_updated.flatten()}") # Can uncomment
            # print(f"  Updated Information (Omega_{t+1|t+1}):") # Can uncomment
            # print(Omega_updated) # Can uncomment

            log_lik_cumulative += step_lik
            # print(f"  Step Pseudo-LogLik Contribution: {step_lik:.4f}") # Can uncomment
            # print(f"  Cumulative Pseudo-LogLik: {log_lik_cumulative:.4f}") # Can uncomment

            # Prepare for next iteration
            a_prev = a_updated
            Omega_prev = Omega_updated

        except Exception as step_e:
            print(f"ERROR during filter step t={t+1}: {step_e}")
            import traceback
            traceback.print_exc()
            nan_detected = True # Stop loop on exception
            break

    if not nan_detected:
        print(f"\nFilter completed {T_val} steps without detecting NaN/Inf.")
        print(f"Final Cumulative Log Likelihood: {log_lik_cumulative:.4f}")
    else:
        print(f"\nFilter stopped at step {t+1} due to NaN/Inf or error.")


    print("\nDebugging finished.")

if __name__ == "__main__":
    main()