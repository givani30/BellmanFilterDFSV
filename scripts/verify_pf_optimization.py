"""
Script to verify Particle Filter (PF) hyperparameter optimization using synthetic data.

This script sets up a basic optimization run mirroring patterns from BIF tests,
using Optax AdamW, a learning rate schedule, and the transformed objective
function with a stability penalty.
"""

import time
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optimistix as optx
import optax
import jax.debug # Add import for debug printing
from jax_dataclasses import pytree_dataclass

from bellman_filter_dfsv.filters.particle import DFSVParticleFilter
from bellman_filter_dfsv.filters.objectives import pf_objective # Only need pf_objective now
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.utils.transformations import safe_arctanh
# float64_decorator import removed (not implemented)
from optax import warmup_cosine_decay_schedule, apply_if_finite
from bellman_filter_dfsv.utils.transformations import (
    transform_params,
    untransform_params,
)
import numpy as np


def apply_identification_constraint(params):
    """
    Applies lower-triangular constraint with diagonal fixed to 1 to lambda_r.
    """
    lambda_r = params.lambda_r
    N, K = lambda_r.shape
    diag_indices = jnp.diag_indices(min(N, K))
    lambda_r = jnp.tril(lambda_r)
    lambda_r = lambda_r.at[diag_indices].set(1.0)
    return params.replace(lambda_r=lambda_r)


# Configure JAX for float64
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True) # Uncomment for debugging


@pytree_dataclass(frozen=True)
class StaticArgs:
    """Static arguments for the objective function."""

    observations: jax.Array
    filter_instance: DFSVParticleFilter
    stability_penalty_weight: float


def create_simple_model(N: int, K: int, seed: int = 42) -> DFSVParamsDataclass:
    """Create a simple DFSV model based on test_bif_full_phi_hybrid_integration.py."""
    np.random.seed(seed) # Keep seed for reproducibility

    # Factor loadings (N x K)
    lambda_r_init = np.random.randn(N, K) * 0.5 + 0.5
    lambda_r = jnp.tril(lambda_r_init)
    diag_indices = jnp.diag_indices(n=min(N, K), ndim=2)
    lambda_r = lambda_r.at[diag_indices].set(1.0)

    # Factor persistence (K x K)
    Phi_f = np.random.uniform(0.01, 0.1, (K, K))
    Phi_f[np.diag_indices(K)] = np.random.uniform(0.8, 0.98, K)
    Phi_h = np.random.uniform(0.01, 0.1, (K, K))
    Phi_h[np.diag_indices(K)] = np.random.uniform(0.9, 0.99, K)

    # Long-run mean for log-volatilities (K)
    mu = np.random.uniform(-1.5, -0.5, K)

    # Idiosyncratic variance (N)
    sigma2 = np.random.uniform(0.05, 0.1, N)

    # Log-volatility noise covariance (K x K diagonal)
    Q_h = np.diag(np.random.uniform(0.1, 0.3, K))

    params = DFSVParamsDataclass(
        N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, Phi_h=Phi_h, mu=mu, sigma2=sigma2, Q_h=Q_h
    )
    # Ensure constraint is applied correctly after creation
    params = apply_identification_constraint(params)
    return params


@eqx.filter_jit
def pf_objective_wrapper(
    transformed_params: DFSVParamsDataclass, static_args: StaticArgs
) -> jax.Array:
    """
    Wrapper for the PF objective function compatible with optimistix, with debugging.

    Untransforms parameters, logs key values, checks for NaNs/Infs, and returns
    only the total objective value for the optimizer.

    Args:
        transformed_params: Parameters in the unconstrained (transformed) space.
        static_args: Static arguments containing observations, filter instance,
                     and stability penalty weight.

    Returns:
        The total objective value (neg_loglik - log_prior + stability_penalty).
    """
    # Untransform parameters back to the constrained model space
    params = untransform_params(transformed_params)

    # Apply identification constraint (lower-triangular lambda_r with diag=1)
    params = apply_identification_constraint(params)

    # --- Debug Logging: Untransformed Parameters (AFTER constraint) ---
    jax.debug.print("pf_wrapper: mu={mu}", mu=params.mu)
    jax.debug.print("pf_wrapper: diag(Phi_f)={phif_diag}", phif_diag=jnp.diag(params.Phi_f))
    jax.debug.print("pf_wrapper: diag(Phi_h)={phih_diag}", phih_diag=jnp.diag(params.Phi_h))
    jax.debug.print("pf_wrapper: diag(Q_h)={qh_diag}", qh_diag=jnp.diag(params.Q_h))
    jax.debug.print("pf_wrapper: sigma2={sigma2}", sigma2=params.sigma2)
    # --- End Debug Logging ---

    # Call the core objective function - it now returns a tuple
    total_objective, neg_log_likelihood, stability_penalty = pf_objective(
        params=params,
        observations=static_args.observations,
        filter_instance=static_args.filter_instance,
        stability_penalty_weight=static_args.stability_penalty_weight,
    )

    # --- Debug Logging: Loss Components ---
    jax.debug.print("pf_wrapper: neg_log_likelihood={nll}", nll=neg_log_likelihood)
    jax.debug.print("pf_wrapper: stability_penalty={sp}", sp=stability_penalty)
    jax.debug.print("pf_wrapper: total_objective={to}", to=total_objective)
    # --- End Debug Logging ---

    # --- NaN/Inf Checks ---
    total_objective = eqx.error_if(
        total_objective,
        jnp.logical_not(jnp.isfinite(total_objective)),
        "Objective Error: Non-finite total_objective detected in pf_objective_wrapper",
    )
    neg_log_likelihood = eqx.error_if(
        neg_log_likelihood,
        jnp.logical_not(jnp.isfinite(neg_log_likelihood)),
        "Objective Error: Non-finite neg_log_likelihood detected in pf_objective_wrapper",
    )
    stability_penalty = eqx.error_if(
        stability_penalty,
        jnp.logical_not(jnp.isfinite(stability_penalty)),
        "Objective Error: Non-finite stability_penalty detected in pf_objective_wrapper",
    )
    # --- End NaN/Inf Checks ---

    # Return only the total objective to the optimizer
    return total_objective


def main(
    seed: int = 42,
    T: int = 250,
    K: int = 2,
    N: int = 5,
    N_particles: int = 5000,
    learning_rate: float = 1e-3,
    max_steps: int = 1000,
    warmup_steps: int = 100,
    stability_penalty_weight: float = 1000.0,
):
    """Main function to run the PF optimization verification."""
    print("--- Starting PF Optimization Verification ---")
    print(f"Config: seed={seed}, T={T}, K={K}, N_particles={N_particles}")
    print(f"Optimizer: AdamW, LR={learning_rate}, Steps={max_steps}, Warmup={warmup_steps}")
    print(f"Objective: Stability Penalty Weight={stability_penalty_weight}")

    key = jr.PRNGKey(seed)
    model_key, data_key, init_key, filter_key, optim_key = jr.split(key, 5)

    # 1. Create True Model and Data
    print("\n1. Creating true model and generating synthetic data...")
    # Use N=3 as default, similar to reference script
    true_params = create_simple_model(N=N, K=K, seed=seed)
    observations, _, _ = simulate_DFSV(params=true_params, T=T,seed=seed)
    print("   True Parameters:")
    print(true_params)
    print(f"   Generated {T} observations.")

    # 2. Instantiate Particle Filter
    print(f"\n2. Instantiating DFSV Particle Filter with N={N_particles} particles...")
    pf_filter = DFSVParticleFilter(N=true_params.N, K=true_params.K, num_particles=N_particles, seed=seed)
    print("   Filter instantiated.")

    # 3. Define Optimizer
    print("\n3. Setting up Optimizer (AdamW with Warmup Cosine Decay)...")
    schedule = warmup_cosine_decay_schedule(
        init_value=1e-4, # Start warmup from 0
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=max_steps - warmup_steps,
        end_value=learning_rate * 0.1, # Decay to 10% of peak LR
    )
    # Use apply_if_finite to prevent NaN/inf propagation
    optimizer = apply_if_finite(optax.adamw(learning_rate=schedule),max_consecutive_errors=5) # Reset state after 5 non-finite steps
    print("   Optimizer defined.")

    # 4. Initial Parameter Guess
    print("\n4. Creating initial parameter guess...")
    # Create an uninformative guess based on data dimensions

    lambda_r_init_guess = jnp.zeros((N, K))
    diag_indices_init = jnp.diag_indices(min(N, K))
    lambda_r_init_guess = lambda_r_init_guess.at[diag_indices_init].set(1.0)
    lambda_r_init_guess = jnp.tril(lambda_r_init_guess)

    initial_stable_diag_val = 0.95
    unconstrained_diag_val = safe_arctanh(initial_stable_diag_val)

    phi_f_init = jnp.eye(K)*unconstrained_diag_val
    phi_h_init = jnp.zeros((K, K))

    uninformed_params = DFSVParamsDataclass(
        N=N, K=K,
        lambda_r=lambda_r_init_guess,
        Phi_f=initial_stable_diag_val * jnp.eye(K),
        Phi_h=initial_stable_diag_val * jnp.eye(K),
        mu=jnp.zeros(K),
        sigma2=0.1 * jnp.ones(N),
        Q_h=0.5 * jnp.eye(K),
    )

    uninformed_params_constrained = apply_identification_constraint(uninformed_params)

    initial_guess_params = uninformed_params_constrained
    # Transform the initial guess to the unconstrained space for optimization
    transformed_initial_guess = transform_params(initial_guess_params)
    print("   Initial Guess (Untransformed):")
    print(initial_guess_params)
    print("   Initial Guess (Transformed):")
    print(transformed_initial_guess)


    # 5. Run Optimization
    print("\n5. Starting optimization...")
    static_args = StaticArgs(
        observations=observations,
        filter_instance=pf_filter,
        stability_penalty_weight=stability_penalty_weight,
    )

    # Define the solver
    # solver = optx.OptaxMinimiser(optimizer, rtol=1e-6, atol=1e-6,norm=optx.rms_norm,verbose=frozenset({"loss"}))
    solver=optx.BFGS(rtol=1e-6, atol=1e-6, norm=optx.rms_norm, verbose=frozenset({"loss"}))
    start_time = time.time()
    try:
        # Run the minimization
        solution = optx.minimise(
            fn=pf_objective_wrapper,
            solver=solver,
            y0=transformed_initial_guess,
            args=static_args,
            max_steps=max_steps,
            throw=False, # Throw errors instead of returning sentinel values
        )
        optim_result = solution.value
        # Recalculate final objective to get components if needed, but only print total
        # The wrapper now returns only the total objective, so this call is correct.
        final_objective_value = pf_objective_wrapper(optim_result, static_args)
        success = solution.result == optx.RESULTS.successful

    except Exception as e:
        print(f"   Optimization failed with error: {e}")
        optim_result = None
        final_objective_value = jnp.nan
        success = False

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"   Optimization finished in {elapsed_time:.2f} seconds.")
    print(f"   Success: {success}")
    print(f"   Final Objective Value: {final_objective_value}")


    # 6. Process and Compare Results
    print("\n6. Comparing True vs. Estimated Parameters...")
    if success and optim_result is not None:
        estimated_params_transformed = optim_result
        estimated_params = untransform_params(estimated_params_transformed)
        print("   Estimated Parameters (Untransformed):")
        print(estimated_params)

        # Basic comparison (can be expanded later)
        print("\n   Parameter Comparison:")
        print(f"   True Mu: {true_params.mu}")
        print(f"   Est. Mu: {estimated_params.mu}")
        print(f"   True Phi_f: \n{true_params.Phi_f}")
        print(f"   Est. Phi_f: \n{estimated_params.Phi_f}")
        print(f"   True Phi_h: \n{true_params.Phi_h}")
        print(f"   Est. Phi_h: \n{estimated_params.Phi_h}")
        print(f"   True sigma2: {true_params.sigma2}")
        print(f"   Est. sigma2: {estimated_params.sigma2}")
        print(f"   True Q_h: \n{true_params.Q_h}")
        print(f"   Est. Q_h: \n{estimated_params.Q_h}")
        # Lambda_r might be fixed or not estimated well initially
        print(f"   True Lambda_r: {true_params.lambda_r}")
        print(f"   Est. Lambda_r: {estimated_params.lambda_r}")

    else:
        print("   Skipping parameter comparison due to optimization failure.")

    print("\n--- PF Optimization Verification Complete ---")


if __name__ == "__main__":
    # Example usage:
    main(
        seed=42,
        T=500,
        K=2,
        N=5,
        N_particles=1000,
        learning_rate=1e-3,
        max_steps=1000, # Reduced steps for quicker test
        warmup_steps=100,
        stability_penalty_weight=10.0,
    )