#!/usr/bin/env python
"""
Optimizer Comparison for DFSV Bellman Information Filter Hyperparameter Estimation.

This script compares the performance (stability, convergence, efficiency) of
various optimizers from Optimistix and Optax when minimizing the negative
pseudo log-likelihood derived from the DFSVBellmanInformationFilter, using
parameter transformations.
"""

import time
from jax.experimental import checkify
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import optax
from functools import partial
from collections import namedtuple
import equinox as eqx

# Project specific imports
from bellman_filter_dfsv.core.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.utils.transformations import transform_params, untransform_params
from bellman_filter_dfsv.core.simulation import simulate_DFSV
# Import the new prior function
from bellman_filter_dfsv.core.likelihood import log_prior_density

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)

# --- Model and Data Generation (Simplified from bf_optimization.py) ---

def create_simple_model(N=3, K=1):
    """Create a simple DFSV model."""
    #set numpy seed
    np.random.seed(42)
    # Factor loadings
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

# --- Objective Functions ---

# @partial(jax.jit, static_argnames=["filter_instance"]) # Keep JIT off for now if debugging internals
def bif_objective(params: DFSVParamsDataclass, returns: jnp.ndarray, filter_instance: DFSVBellmanInformationFilter) -> jnp.ndarray:
    """
    Compute the negative pseudo log-posterior (likelihood + prior) using the BIF.

    Args:
        params: Model parameters (DFSVParamsDataclass with JAX arrays).
        returns: Observed data (T, N) as JAX array.
        filter_instance: An instance of DFSVBellmanInformationFilter.

    Returns:
        Negative pseudo log-posterior (JAX scalar). Returns Inf if errors occur.
    """

    # Calculate pseudo log-likelihood from filter
    # Use the non-JITted version if debugging internals, otherwise use JITted
    # total_log_lik = filter_instance.log_likelihood_of_params(params, returns) # Non-JIT version
    total_log_lik = filter_instance.jit_log_likelihood_of_params()(params, returns) # JIT version

    # Calculate log prior density (using default hyperparameters for now)
    # TODO: Pass prior hyperparameters if needed
    log_prior = log_prior_density(params)

    # Combine likelihood and prior
    log_posterior = total_log_lik + log_prior

    # Return negative log-posterior, replacing NaN/Inf with Inf for minimization
    neg_log_post = -log_posterior
    safe_neg_log_post = jnp.where(jnp.isnan(neg_log_post) | jnp.isinf(neg_log_post) , jnp.inf, neg_log_post)
    return safe_neg_log_post

@eqx.filter_jit
def transformed_bif_objective(transformed_params: DFSVParamsDataclass, returns: jnp.ndarray, filter_instance: DFSVBellmanInformationFilter) -> jnp.ndarray:
    """
    Compute the BIF objective function (Negative Log Posterior) with transformed parameters.

    Args:
        transformed_params: Model parameters in transformed (unconstrained) space.
        returns: Observed data (T, N) as JAX array.
        filter_instance: An instance of DFSVBellmanInformationFilter.

    Returns:
        Negative pseudo log-posterior (JAX scalar).
    """
    # Transform parameters back to original space
    original_params = untransform_params(transformed_params)
    #print params
    # jax.debug.print("transformed_bif_objective: original_params: {p}", p=original_params)
    # Call the standard objective function (which now includes priors)
    return bif_objective(original_params, returns, filter_instance)

# --- Main Comparison Logic ---

OptimizerResult = namedtuple("OptimizerResult", ["optimizer_name", "success", "final_loss", "steps", "time_taken", "error_message"])

def run_comparison(true_params: DFSVParamsDataclass, returns: jnp.ndarray, max_steps: int = 100):
    """
    Runs the optimizer comparison study using parameter transformations.

    Args:
        true_params: The true parameters used for simulation (for N, K reference).
        returns: The simulated observation data.
        max_steps: Maximum iterations for each optimizer run.

    Returns:
        List[OptimizerResult]: A list containing results for each optimizer run.
    """
    N, K = true_params.N, true_params.K
    filter_instance = DFSVBellmanInformationFilter(N, K)
    results = []

    # Define Optimizers to Compare
    # Set common tolerances
    rtol = 1e-5
    atol = 1e-5
    learning_rate = 1e-4 # Example LR for Optax optimizers

    optimizers_to_test = {
        "BFGS": optx.BFGS(rtol=rtol, atol=atol,norm=optx.rms_norm, verbose=frozenset({"loss"})), #, verbose=frozenset({"loss"})),
        # "NonlinearCG": optx.NonlinearCG(rtol=rtol, atol=atol),
        # "SGD": optx.OptaxMinimiser(optax.sgd(learning_rate=learning_rate), rtol=rtol, atol=atol, verbose=frozenset({"loss"})),
        "Adam": optx.OptaxMinimiser(optax.adam(learning_rate=learning_rate), rtol=rtol, atol=atol,norm=optx.rms_norm, verbose=frozenset({"loss"})),
        # "AdamW": optx.OptaxMinimiser(optax.adamw(learning_rate=learning_rate), rtol=rtol, atol=atol, verbose=frozenset({"loss"})),
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

    # Loop through optimizers (always using transformations)
    for name, solver in optimizers_to_test.items():
        print(f"\n--- Running Optimizer: {name} (Using Transformations) ---")

        initial_y = transform_params(uninformed_params)
        # Define wrapper for transformed objective
        # This wrapper is needed because the objective now returns only the loss
        def transformed_objective_wrapper(t_params, args_tuple):
            obs, filt = args_tuple # Unpack static args
            # --- Log Transformed Parameters ---
            # jax.debug.print("[Optimizer Step] Transformed Params: {p}", p=t_params) # Keep commented for now
            # --- End Log ---
            # --- DEBUG ---
            leaves_in = jax.tree_util.tree_leaves(t_params)
            is_params_finite = jnp.all(jnp.array([jnp.all(jnp.isfinite(leaf)) for leaf in leaves_in]))
            # jax.debug.print("transformed_objective_wrapper: Input t_params finite: {finite}", finite=is_params_finite) # Keep commented
            # --- END DEBUG ---

            # Call the objective function (which handles untransform and prior)
            loss = transformed_bif_objective(t_params, obs, filt)

            # --- DEBUG ---
            is_loss_finite = jnp.isfinite(loss)
            # jax.debug.print("transformed_objective_wrapper: Output loss: {loss}, finite: {finite}", loss=loss, finite=is_loss_finite) # Keep commented
            # --- END DEBUG ---
            return loss # Return only the loss to the optimizer

        fn_to_minimize = transformed_objective_wrapper
        objective_fn_for_loss_calc = bif_objective # Use non-transformed for final loss calculation

        # Package static arguments
        static_args = (returns, filter_instance)

        start_time = time.time()
        final_loss = jnp.inf
        num_steps = -1
        success = False
        error_msg = "N/A"
        final_params_untransformed = None

        # --- Calculate Initial Objective and Gradient ---
        initial_grad = None
        try:
            print("Calculating initial objective...")
            # Calculate initial loss using the wrapper
            initial_loss = fn_to_minimize(initial_y, static_args)
            print(f"Initial Objective Loss (Neg Log Posterior): {initial_loss:.4f}")

            print("Calculating initial gradient...")
            # Differentiate the wrapper function
            grad_fn = eqx.filter_grad(fn_to_minimize)
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
                optimizer_name=name, success=False,
                final_loss=jnp.inf, steps=-1, time_taken=0, error_message=error_msg
            ))
            continue # Skip to the next optimizer
        # --- End Initial Calculation ---

        # try:
        # Run the minimization
        sol = optx.minimise(
            fn=fn_to_minimize,
            solver=solver,
            y0=initial_y,
            args=static_args, # Pass static args here
            max_steps=max_steps,
            throw=False # Raise errors immediately
        )

        end_time = time.time()
        time_taken = end_time - start_time
        num_steps = sol.stats.get('num_steps', -1) # Get steps if available

        # Check solver success first
        solver_success = (sol.result == optx.RESULTS.successful)

        # --- Final Validation ---
        check_exception = None
        final_loss_recalculated = jnp.inf
        final_params_untransformed = None
        if solver_success:
            try:
                # Get final parameters (untransformed)
                final_params_untransformed = untransform_params(sol.value)
                # Recalculate final loss using the original objective (includes prior)
                final_loss_recalculated = objective_fn_for_loss_calc(final_params_untransformed, returns, filter_instance)
            except Exception as check_e:
                check_exception = check_e
                final_loss_recalculated = jnp.inf
                print(f"Exception during final loss recalculation: {check_exception}")
        # --- End Final Validation ---

        # Determine overall success
        success = solver_success and (check_exception is None) and jnp.isfinite(final_loss_recalculated)
        final_loss = final_loss_recalculated if success else jnp.inf # Use recalculated loss if successful

        # Determine error message
        if success:
            error_msg = "N/A"
            print(f"Success! Final Loss: {final_loss:.4f}, Steps: {num_steps}, Time: {time_taken:.2f}s")
        else:
            if not solver_success:
                error_msg = f"Solver failed: {sol.result}"
            elif check_exception is not None:
                error_msg = f"Exception in final recalc: {check_exception}"
            elif not jnp.isfinite(final_loss_recalculated):
                    error_msg = "Final loss calculation resulted in NaN/Inf"
            else: # Should only happen if solver failed but recalc worked
                error_msg = f"Solver failed ({sol.result}) - final recalc skipped or OK"
            print(f"Failed! Status: {error_msg}, Final Loss: {final_loss}, Steps: {num_steps}, Time: {time_taken:.2f}s")

        # except Exception as e: # Catch exceptions during the main optx.minimise call
        #     end_time = time.time()
        #     time_taken = end_time - start_time
        #     success = False
        #     error_msg = f"Exception during minimize: {str(e)}"
        #     final_loss = jnp.inf # Assign Inf on exception
        #     print(f"Exception! Error: {e}, Time: {time_taken:.2f}s")

        # Store results
        results.append(OptimizerResult(
            optimizer_name=name,
            success=success,
            final_loss=float(final_loss), # Convert JAX scalar to float
            steps=int(num_steps),
            time_taken=time_taken,
            error_message=error_msg
        ))

    return results

def print_results_table(results: list[OptimizerResult]):
    """Prints the comparison results in a formatted table."""
    print("\n\n--- Optimizer Comparison Results (Using Transformations) ---")
    # Header
    print(f"{'Optimizer':<15} | {'Success':<8} | {'Final Loss':<15} | {'Steps':<8} | {'Time (s)':<10} | {'Error Message'}")
    print("-" * 90) # Adjusted width
    # Rows
    for res in sorted(results, key=lambda x: x.optimizer_name): # Simplified sort key
        success_str = "Yes" if res.success else "No"
        loss_str = f"{res.final_loss:.4e}" if np.isfinite(res.final_loss) else "Inf/NaN"
        steps_str = str(res.steps) if res.steps >= 0 else "N/A"
        time_str = f"{res.time_taken:.2f}"
        error_str = res.error_message if not res.success else "N/A"
        print(f"{res.optimizer_name:<15} | {success_str:<8} | {loss_str:<15} | {steps_str:<8} | {time_str:<10} | {error_str}")
    print("-" * 90) # Adjusted width


def main():
    """Run the optimizer comparison study."""
    print("Starting BIF Optimizer Comparison Study (Using Transformations)...")

    # 1. Create Model Parameters
    true_params = create_simple_model(N=5, K=2) # Example: 5 series, 2 factors
    print(f"Using model N={true_params.N}, K={true_params.K}")

    # 2. Generate Simulation Data
    T = 2000 # Shorter time series for faster testing
    print(f"Generating {T} time steps of simulation data...")
    returns = create_training_data(true_params, T=T, seed=123)
    print("Simulation data generated.")

    # 3. Run Comparison
    max_opt_steps = 500 # Increased max steps again
    print(f"Running optimizer comparison (max_steps={max_opt_steps})...")
    results = run_comparison(true_params, returns, max_steps=max_opt_steps)

    # 4. Print Results
    print_results_table(results)

    print("\nOptimizer comparison study finished.")


if __name__ == "__main__":
    main()