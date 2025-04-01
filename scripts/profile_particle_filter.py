import jax
import jax.numpy as jnp
import time
import argparse
from functools import partial
import os
import sys
from typing import Tuple

# Add project root to path to allow importing package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from bellman_filter_dfsv.core.filters.particle import DFSVParticleFilter
    from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print(f"Make sure the package is installed or the project root is in PYTHONPATH.")
    print(f"Project root added to sys.path: {project_root}")
    sys.exit(1)

# Use float32 consistent with the particle filter implementation
DEFAULT_DTYPE = jnp.float32
# jax.config.update("jax_enable_x64", False) # Ensure default is not overridden

# --- Reusable Helper Functions (adapted from Bellman profiler) ---

def create_dummy_data(N: int, K: int, T: int, key: jax.random.PRNGKey) -> Tuple[DFSVParamsDataclass, jnp.ndarray]:
    """Generates dummy parameters and observations."""
    keys = jax.random.split(key, 10)
    lambda_r = jax.random.normal(keys[0], (N, K), dtype=DEFAULT_DTYPE) * 0.2
    Phi_f = jnp.eye(K, dtype=DEFAULT_DTYPE) * 0.95
    Phi_h = jnp.eye(K, dtype=DEFAULT_DTYPE) * 0.98
    mu = jnp.zeros(K, dtype=DEFAULT_DTYPE)
    sigma2 = jnp.abs(jax.random.normal(keys[1], (N,), dtype=DEFAULT_DTYPE) * 0.1) + 0.01
    # Ensure Q_h is positive definite for Cholesky
    q_noise = jax.random.normal(keys[3], (K, K), dtype=DEFAULT_DTYPE) * 0.1
    Q_h = q_noise @ q_noise.T + jnp.eye(K, dtype=DEFAULT_DTYPE) * 0.02

    params = DFSVParamsDataclass(
        N=N, K=K,
        lambda_r=lambda_r,
        Phi_f=Phi_f,
        Phi_h=Phi_h,
        mu=mu,
        sigma2=sigma2,
        Q_h=Q_h
    )

    # Generate dummy observations (simple random walk for factors/vol)
    key, subkey = jax.random.split(keys[2])
    initial_factors = jnp.zeros(K, dtype=DEFAULT_DTYPE)
    initial_log_vols = mu
    chol_Q_h = jax.scipy.linalg.cholesky(Q_h, lower=True)

    def scan_body(carry, key_t):
        f_prev, h_prev = carry
        key_f, key_h, key_obs = jax.random.split(key_t, 3)
        h_next = mu + Phi_h @ (h_prev - mu) + chol_Q_h @ jax.random.normal(key_h, (K,))
        f_next = Phi_f @ f_prev + jnp.exp(h_next / 2) * jax.random.normal(key_f, (K,))
        obs_noise = jnp.sqrt(sigma2) * jax.random.normal(key_obs, (N,))
        r_t = lambda_r @ f_next + obs_noise
        return (f_next, h_next), r_t

    keys_t = jax.random.split(subkey, T)
    _, observations = jax.lax.scan(scan_body, (initial_factors, initial_log_vols), keys_t)

    return params, observations

def block_on_results(results):
    """Calls block_until_ready() on JAX arrays within results (tuple or single)."""
    if isinstance(results, tuple):
        for item in results:
            if hasattr(item, 'block_until_ready'):
                item.block_until_ready()
    elif hasattr(results, 'block_until_ready'):
        results.block_until_ready()

def time_function(func, *args, repetitions=3):
    """Times a function, ensuring JAX compilation and execution."""
    # Warm-up run
    try:
        result_warmup = func(*args)
        block_on_results(result_warmup) # Block on results after function call
    except Exception as e:
        func_name = getattr(func, '__name__', repr(func))
        # Try getting name from partial if applicable
        if isinstance(func, partial):
            func_name = getattr(func.func, '__name__', func_name)
        print(f"  ERROR during warm-up of {func_name}: {e}")
        # import traceback
        # traceback.print_exc() # Optional: print full traceback
        return -1.0 # Indicate error

    # Timed runs
    times = []
    for _ in range(repetitions):
        start_time = time.perf_counter()
        try:
             result_timed = func(*args)
             block_on_results(result_timed) # Block on results after function call
             end_time = time.perf_counter()
             times.append(end_time - start_time)
        except Exception as e:
            func_name = getattr(func, '__name__', repr(func))
            if isinstance(func, partial):
                 func_name = getattr(func.func, '__name__', func_name)
            print(f"  ERROR during timed run of {func_name}: {e}")
            # import traceback
            # traceback.print_exc() # Optional: print full traceback
            return -1.0 # Indicate error

    return sum(times) / repetitions if times else -1.0

# --- Main Profiling Logic ---

def profile_particle_filter(N_vals, K_vals, P_vals, T, repetitions, key):
    """Runs the profiling for different N, K, and P values."""
    results = []
    for N in N_vals:
        for K in K_vals:
            for P in P_vals:
                print(f"\n--- Profiling N={N}, K={K}, P={P} ---")
                state_dim = 2 * K
                key, data_key, init_key, step_key = jax.random.split(key, 4)

                try:
                    # Generate data for this N, K
                    params, observations = create_dummy_data(N, K, T, data_key)
                    # Ensure params have JAX arrays (using float32)
                    params = params.replace(**{
                        f: jnp.asarray(getattr(params, f), dtype=DEFAULT_DTYPE)
                        for f in ["lambda_r", "Phi_f", "Phi_h", "mu", "sigma2", "Q_h"]
                    })
                    chol_Q_h = jax.scipy.linalg.cholesky(params.Q_h, lower=True)
                    obs_t = observations[0].reshape(-1, 1) # Use first observation (N, 1)
                    obs_noise_variances = params.sigma2 # Already 1D (N,)

                    # Instantiate filter
                    filt = DFSVParticleFilter(N=N, K=K, num_particles=P, seed=int(init_key[0]))

                    # --- Timing ---
                    print(f"  Timing with {repetitions} repetitions...")

                    # 1. Overall Filter
                    # Need to wrap the filter call for time_function
                    def filter_wrapper(p, o):
                        # Create a new instance each time to reset state? Or use the same?
                        # Using same instance might benefit from JIT cache but state persists.
                        # Let's use the same instance for now.
                        return filt.filter(p, o)

                    avg_time_filter = time_function(filter_wrapper, params, observations, repetitions=repetitions)
                    results.append({"N": N, "K": K, "P": P, "Function": "filter", "Avg Time (s)": avg_time_filter})
                    print(f"  filter: {avg_time_filter:.6f} s")

                    # 2. Initialization
                    # Need a fresh key each time for timing consistency
                    def init_wrapper(p):
                         k1, k2 = jax.random.split(step_key) # Use a consistent key for timing runs
                         return filt.initialize_particles(p, k2)
                    avg_time_init = time_function(init_wrapper, params, repetitions=repetitions)
                    results.append({"N": N, "K": K, "P": P, "Function": "initialize_particles", "Avg Time (s)": avg_time_init})
                    print(f"  initialize_particles: {avg_time_init:.6f} s")

                    # Get dummy particles/weights for component timing
                    _, dummy_particles, dummy_log_weights = filt.initialize_particles(params, init_key)

                    # 3. Predict Particles
                    # predict_particles is not JITted by default, JIT it here
                    predict_timed = jax.jit(filt.predict_particles)
                    def predict_wrapper(p, chol_q):
                        k1, k2 = jax.random.split(step_key)
                        return predict_timed(k2, dummy_particles, p, chol_q)
                    avg_time_predict = time_function(predict_wrapper, params, chol_Q_h, repetitions=repetitions)
                    results.append({"N": N, "K": K, "P": P, "Function": "predict_particles", "Avg Time (s)": avg_time_predict})
                    print(f"  predict_particles: {avg_time_predict:.6f} s")

                    # 4. Compute Log Likelihood
                    # compute_log_likelihood_particle is JITted with static K, N
                    # Need to wrap it to pass self_static correctly for timing
                    @partial(jax.jit, static_argnums=(0, 5, 6))
                    def likelihood_static_wrapper(self_static, particles_arg, obs_arg, factor_loadings_arg, obs_noise_vars_arg, K_arg, N_arg):
                         return self_static.compute_log_likelihood_particle(particles_arg, obs_arg, factor_loadings_arg, obs_noise_vars_arg, K_arg, N_arg)

                    def likelihood_wrapper(p):
                         # Pass filt instance statically
                         return likelihood_static_wrapper(filt, dummy_particles, obs_t, p.lambda_r, obs_noise_variances, K, N)
                    avg_time_likelihood = time_function(likelihood_wrapper, params, repetitions=repetitions)
                    results.append({"N": N, "K": K, "P": P, "Function": "compute_log_likelihood_particle", "Avg Time (s)": avg_time_likelihood})
                    print(f"  compute_log_likelihood_particle: {avg_time_likelihood:.6f} s")

                    # 5. Resample Particles
                    # resample_particles is JITted with static self
                    @partial(jax.jit, static_argnums=(0,))
                    def resample_static_wrapper(self_static, key_arg, particles_arg, log_weights_arg):
                        return self_static.resample_particles(key_arg, particles_arg, log_weights_arg)

                    # Create dummy unnormalized weights
                    dummy_unnorm_log_weights = dummy_log_weights + jax.random.normal(step_key, (P,))
                    def resample_wrapper():
                        k1, k2 = jax.random.split(step_key)
                        # Pass filt instance statically
                        return resample_static_wrapper(filt, k2, dummy_particles, dummy_unnorm_log_weights)
                    avg_time_resample = time_function(resample_wrapper, repetitions=repetitions)
                    results.append({"N": N, "K": K, "P": P, "Function": "resample_particles", "Avg Time (s)": avg_time_resample})
                    print(f"  resample_particles: {avg_time_resample:.6f} s")


                except Exception as e:
                    print(f"  ERROR profiling N={N}, K={K}, P={P}: {e}")
                    # Add error entry for this N, K, P combination
                    results.append({"N": N, "K": K, "P": P, "Function": "ERROR", "Avg Time (s)": -1})
                    func_names = ["filter", "initialize_particles", "predict_particles",
                                  "compute_log_likelihood_particle", "resample_particles"]
                    existing_funcs = {r['Function'] for r in results if r['N'] == N and r['K'] == K and r['P'] == P}
                    for fname in func_names:
                        if fname not in existing_funcs:
                             results.append({"N": N, "K": K, "P": P, "Function": fname, "Avg Time (s)": -1})

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile DFSV Particle Filter Components.")
    parser.add_argument('--N', type=int, nargs='+', default=[10, 30, 50], help='List of N values (number of series)')
    parser.add_argument('--K', type=int, nargs='+', default=[5, 10], help='List of K values (number of factors)')
    parser.add_argument('--P', type=int, nargs='+', default=[500, 1000, 2000], help='List of P values (number of particles)')
    parser.add_argument('--T', type=int, default=50, help='Number of time steps for dummy data generation')
    parser.add_argument('--reps', type=int, default=3, help='Number of repetitions for timing each function (use lower value for PF)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data generation')

    args = parser.parse_args()

    print("Starting Particle Filter Profiling...")
    print(f"N values: {args.N}")
    print(f"K values: {args.K}")
    print(f"P values: {args.P}")
    print(f"T: {args.T}")
    print(f"Repetitions: {args.reps}")
    print(f"Seed: {args.seed}")
    print(f"Using JAX default device: {jax.default_backend()}")
    print(f"Using dtype: {DEFAULT_DTYPE}")
    print("-" * 30)

    prng_key = jax.random.PRNGKey(args.seed)

    profiling_results = profile_particle_filter(args.N, args.K, args.P, args.T, args.reps, prng_key)

    print("\n--- Profiling Summary ---")
    try:
        import pandas as pd
        if profiling_results:
            df = pd.DataFrame(profiling_results)
            # Pivot for better readability
            df_pivot = df.pivot(index=['N', 'K', 'P'], columns='Function', values='Avg Time (s)')
            print(df_pivot.to_string(float_format="%.6f"))
        else:
            print("No results generated.")
    except ImportError:
        print("Install pandas for formatted table output ('pip install pandas' or 'uv pip install pandas').")
        header = ["N", "K", "P", "Function", "Avg Time (s)"]
        print("\t".join(header))
        print("-" * (sum(len(h) for h in header) + len(header)*3))
        for res in profiling_results:
            print(f"{res.get('N', '?')}\t{res.get('K', '?')}\t{res.get('P', '?')}\t{res.get('Function', 'N/A')}\t{res.get('Avg Time (s)', -1):.6f}")
    except Exception as e_pd:
         print(f"Error formatting results with pandas: {e_pd}")
         print("Raw results:")
         for res in profiling_results:
            print(res)

    print("\nProfiling Complete.")