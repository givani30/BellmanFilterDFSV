import time
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

# Enable double precision
jax.config.update("jax_enable_x64", True)

from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.utils.optimization import run_optimization, FilterType
from bellman_filter_dfsv.utils.optimization_helpers import (
    create_stable_initial_params,
)

def main():
    """Runs the baseline profiling for BIF optimization."""
    # --- Configuration ---
    N = 3
    K = 2
    T = 200
    seed = 42
    max_steps = 150
    stability_penalty_weight = 1e4
    optimizer_name = "DampedTrustRegionBFGS"

    print(f"Configuration: N={N}, K={K}, T={T}, Seed={seed}")
    print(f"Optimizer: {optimizer_name}, Max Steps: {max_steps}")
    print(f"Stability Penalty Weight: {stability_penalty_weight}")

    # --- Simulation ---
    print("\nSimulating data...")
    # 1. Generate true parameters to simulate from
    print("Generating true parameters for simulation...")
    true_params = create_stable_initial_params(N, K)
    print(f"Generated True mu: {true_params.mu}")

    # 2. Simulate data using the generated true parameters
    sim_key = jax.random.PRNGKey(seed + 666)
    observations, _, _ = simulate_DFSV(params=true_params, T=T, seed=seed + 666)
    print("Simulation complete.")

    # --- Initial Parameters ---
    print("Creating initial parameters...")
    initial_params_guess = create_stable_initial_params(N, K)
    print("Initial parameters created.")

    # --- Run Optimization ---
    print(f"\nRunning optimization for {max_steps} steps...")
    start_time = time.time()
    
    result = run_optimization(
        filter_type=FilterType.BIF,
        returns=observations,
        initial_params=initial_params_guess,
        fix_mu=True,
        true_params=true_params,
        use_transformations=True,
        optimizer_name=optimizer_name,
        stability_penalty_weight=stability_penalty_weight,
        max_steps=max_steps,
        log_params=False,  # Disable parameter logging
        verbose=True
    )

    end_time = time.time()
    execution_time = end_time - start_time

    # --- Results ---
    print("\n--- Optimization Results ---")
    print(f"Total Execution Time: {execution_time:.2f} seconds")
    print(f"Final Result: {result}")

if __name__ == "__main__":
    main()