"""
Benchmark script for the DFSVParticleFilter.

Measures the execution time of the particle filter for a given configuration,
discarding the first run to account for JIT compilation time.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp

# Ensure JAX uses float32 for this benchmark run
jax.config.update("jax_enable_x64", False)

# Project imports
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.filters.particle import DFSVParticleFilter
from simulation_study import create_sim_parameters # Import from sibling script

# --- Configuration ---
N_ASSETS = 25
N_FACTORS = 5
T_OBS = 500
N_PARTICLES = 10000
NUM_RUNS = 5 # Number of timed runs (after warmup)
SEED = 12345

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Particle Filter Benchmark ---")
    print(f"Config: N={N_ASSETS}, K={N_FACTORS}, T={T_OBS}, P={N_PARTICLES}")
    print(f"Precision: {'float64' if jax.config.jax_enable_x64 else 'float32'}") # Updated logic to reflect config
    print(f"Running {NUM_RUNS} timed iterations (plus 1 warmup run)...")

    # 1. Generate Parameters and Data
    print("Generating parameters and simulation data...")
    key = jax.random.PRNGKey(SEED)
    key, params_key, sim_key = jax.random.split(key, 3)

    # Use create_sim_parameters from simulation_study.py
    # Note: Ensure create_sim_parameters generates diagonal sigma2 as intended
    params_orig = create_sim_parameters(N=N_ASSETS, K=N_FACTORS, seed=SEED)
    # Convert parameters to float32
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32) if jnp.issubdtype(x.dtype, jnp.floating) else x, params_orig)


    # Simulate data using original (likely float64) params, then convert results
    returns_orig, _, _ = simulate_DFSV(
        params=params_orig, T=T_OBS, seed=SEED + 1
    )
    # Ensure observations are JAX array and float32
    observations_jax = jnp.asarray(returns_orig, dtype=jnp.float32)
    print("Data generation and conversion to float32 complete.")

    # 2. Instantiate Filter
    pf_instance = DFSVParticleFilter(
        N=N_ASSETS,
        K=N_FACTORS,
        num_particles=N_PARTICLES,
        seed=SEED + 2 # Use a different seed for the filter itself
    )
    print("Particle filter instantiated.")

    # 3. Run and Time Filter
    elapsed_times = []

    # Warmup run (includes JIT compilation)
    print("Performing warmup run (compilation)...")
    start_warmup = time.time()
    filtered_states_w, filtered_covs_w, log_likelihood_w = pf_instance.filter(
        params=params, observations=observations_jax
    )
    # Ensure computation finishes before stopping timer
    # NumPy conversion implicitly blocks, so block_until_ready is not needed on returned arrays
    # Keep block for log_likelihood as it might still be a JAX scalar internally
    jax.block_until_ready(log_likelihood_w)
    end_warmup = time.time()
    print(f"Warmup run finished in {end_warmup - start_warmup:.4f} seconds.")

    # Timed runs
    print(f"Starting {NUM_RUNS} timed runs...")
    for i in range(NUM_RUNS):
        print(f"  Run {i+1}/{NUM_RUNS}...")
        start_time = time.time()

        # Run the filter
        filtered_states, filtered_covs, log_likelihood = pf_instance.filter(
            params=params, observations=observations_jax
        )

        # Block until computations are complete before recording time
        # NumPy conversion implicitly blocks, so block_until_ready is not needed on returned arrays
        jax.block_until_ready(log_likelihood) # Ensure LL is computed

        end_time = time.time()
        elapsed = end_time - start_time
        elapsed_times.append(elapsed)
        print(f"  Run {i+1} finished in {elapsed:.4f} seconds.")

    # 4. Calculate and Print Results
    average_time = np.mean(elapsed_times)
    std_dev_time = np.std(elapsed_times)

    print("\n--- Benchmark Results ---")
    print(f"Average execution time over {NUM_RUNS} runs: {average_time:.4f} seconds")
    print(f"Standard deviation: {std_dev_time:.4f} seconds")
    print(f"Individual run times: {[f'{t:.4f}' for t in elapsed_times]}")
    print(f"Final Log Likelihood (last run): {log_likelihood:.4f}") # Print LL as a sanity check