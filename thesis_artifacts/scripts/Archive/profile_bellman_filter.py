import jax
import jax.numpy as jnp
import time
import argparse
from functools import partial
import os
from typing import Tuple # Added import
import sys

# Add project root to path to allow importing bellman_filter_dfsv
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Use standard package imports, assuming project root is in path
    from bellman_filter_dfsv.filters.bellman import DFSVBellmanFilter
    from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print(f"Make sure the package is installed or the project root is in PYTHONPATH.")
    print(f"Project root added to sys.path: {project_root}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


# Enable 64-bit precision consistent with the filter
jax.config.update("jax_enable_x64", True)
DEFAULT_DTYPE = jnp.float64

def create_dummy_data(N: int, K: int, T: int, key: jax.random.PRNGKey) -> Tuple[DFSVParamsDataclass, jnp.ndarray]:
    """Generates dummy parameters and observations."""
    keys = jax.random.split(key, 10)
    lambda_r = jax.random.normal(keys[0], (N, K), dtype=DEFAULT_DTYPE) * 0.2
    Phi_f = jnp.eye(K, dtype=DEFAULT_DTYPE) * 0.95
    Phi_h = jnp.eye(K, dtype=DEFAULT_DTYPE) * 0.98
    mu = jnp.zeros(K, dtype=DEFAULT_DTYPE)
    # Ensure sigma2 is 1D
    sigma2 = jnp.abs(jax.random.normal(keys[1], (N,), dtype=DEFAULT_DTYPE) * 0.1) + 0.01
    Q_h = jnp.eye(K, dtype=DEFAULT_DTYPE) * 0.05

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

    def scan_body(carry, key_t):
        f_prev, h_prev = carry
        key_f, key_h, key_obs = jax.random.split(key_t, 3)
        # Simplified dynamics for dummy data generation
        h_next = mu + Phi_h @ (h_prev - mu) + jax.random.multivariate_normal(key_h, jnp.zeros(K), Q_h)
        f_next = Phi_f @ f_prev + jax.random.multivariate_normal(key_f, jnp.zeros(K), jnp.diag(jnp.exp(h_next / 2)))
        obs_noise = jax.random.multivariate_normal(key_obs, jnp.zeros(N), jnp.diag(sigma2))
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
        print(f"  ERROR during warm-up of {getattr(func, '__name__', repr(func))}: {e}")
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
            print(f"  ERROR during timed run of {getattr(func, '__name__', repr(func))}: {e}")
            return -1.0 # Indicate error

    return sum(times) / repetitions if times else -1.0


def profile_filter(N_vals, K_vals, T, repetitions, key):
    """Runs the profiling for different N and K values."""
    results = []
    for N in N_vals:
        for K in K_vals:
            print(f"\n--- Profiling N={N}, K={K} ---")
            state_dim = 2 * K
            key, subkey = jax.random.split(key)

            try:
                # Generate data for this N, K
                params, observations = create_dummy_data(N, K, T, subkey)
                obs_t = observations[0] # Use first observation for update step timing

                # Instantiate filter
                filt = DFSVBellmanFilter(N, K)

                # Get initial state/covariance
                initial_state_jax, initial_cov_jax = filt.initialize_state(params)

                # --- Timing ---
                print(f"  Timing with {repetitions} repetitions...")

                # 1. Predict Jax
                # Ensure predict_jax is JITted if not already done internally
                predict_jax_timed = jax.jit(filt.predict_jax)
                avg_time_predict = time_function(predict_jax_timed, params, initial_state_jax, initial_cov_jax, repetitions=repetitions)
                results.append({"N": N, "K": K, "Function": "predict_jax", "Avg Time (s)": avg_time_predict})
                print(f"  predict_jax: {avg_time_predict:.6f} s")
                # Get outputs needed for update timing
                pred_state_jax, pred_cov_jax = predict_jax_timed(params, initial_state_jax, initial_cov_jax)

                # 2. Update Jax
                # Ensure update_jax is JITted
                update_jax_timed = jax.jit(filt.update_jax)
                avg_time_update = time_function(update_jax_timed, params, pred_state_jax, pred_cov_jax, obs_t, repetitions=repetitions)
                results.append({"N": N, "K": K, "Function": "update_jax", "Avg Time (s)": avg_time_update})
                print(f"  update_jax: {avg_time_update:.6f} s")

                # Prepare inputs for inner components (run update once to get intermediate values if needed)
                # Note: These inputs might not perfectly match internal state during a real filter run,
                # but are needed to call the JITted components for timing.
                try:
                    # Use pseudo-inverse for robustness in dummy data case
                    I_pred = jnp.linalg.pinv(pred_cov_jax + 1e-8 * jnp.eye(state_dim))
                except jnp.linalg.LinAlgError:
                     print("  WARN: Pseudo-inverse failed for I_pred, using identity.")
                     I_pred = jnp.eye(state_dim, dtype=DEFAULT_DTYPE)
                I_pred = (I_pred + I_pred.T) / 2 # Ensure symmetry

                alpha_init = pred_state_jax.flatten()

                # Need an updated alpha to time FIM and log_posterior reasonably
                # Run the block coordinate update once (un-timed) to get it
                try:
                    alpha_updated_uncompiled = filt._block_coordinate_update_impl(
                         params.lambda_r, params.sigma2, alpha_init, pred_state_jax.flatten(), I_pred, obs_t,
                         max_iters=10, # Use default max_iters
                         h_solver=filt.h_solver
                    )
                    alpha_updated = alpha_updated_uncompiled.block_until_ready() # Ensure computation finishes
                except Exception as e_bcu_run:
                    print(f"  WARN: Failed to run BCU for alpha_updated: {e_bcu_run}. Using alpha_init.")
                    alpha_updated = alpha_init


                # 3. Block Coordinate Update (JITted implementation)
                # Note: max_iters and h_solver are static args for block_coordinate_update_impl_jit
                # We need to pass them correctly to time_function or wrap the call.
                @partial(jax.jit, static_argnums=(6, 7))
                def bcu_impl_jit_wrapper(lambda_r, sigma2, alpha, pred_state, I_pred, observation, max_iters, h_solver):
                    # This wrapper is needed because time_function doesn't handle static args
                    return filt._block_coordinate_update_impl(lambda_r, sigma2, alpha, pred_state, I_pred, observation, max_iters, h_solver)

                avg_time_bcu = time_function(bcu_impl_jit_wrapper,
                                             params.lambda_r, params.sigma2, alpha_init, pred_state_jax.flatten(), I_pred, obs_t,
                                             10, filt.h_solver, # Pass static args here
                                             repetitions=repetitions)
                results.append({"N": N, "K": K, "Function": "block_coordinate_update_impl_jit", "Avg Time (s)": avg_time_bcu})
                print(f"  block_coordinate_update_impl_jit: {avg_time_bcu:.6f} s")


                # 4. Fisher Information (JITted)
                fisher_information_jit_timed = jax.jit(filt.fisher_information_jit)
                avg_time_fim = time_function(fisher_information_jit_timed, params.lambda_r, params.sigma2, alpha_updated, repetitions=repetitions)
                results.append({"N": N, "K": K, "Function": "fisher_information_jit", "Avg Time (s)": avg_time_fim})
                print(f"  fisher_information_jit: {avg_time_fim:.6f} s")

                # 5. Log Posterior (JITted)
                log_posterior_jit_timed = jax.jit(filt.log_posterior_jit)
                avg_time_logpost = time_function(log_posterior_jit_timed, params.lambda_r, params.sigma2, alpha_updated, obs_t, repetitions=repetitions)
                results.append({"N": N, "K": K, "Function": "log_posterior_jit", "Avg Time (s)": avg_time_logpost})
                print(f"  log_posterior_jit: {avg_time_logpost:.6f} s")

                # 6. Build Covariance (JITted)
                # Need a dummy exp_h based on alpha_updated
                exp_h_dummy = jnp.exp(alpha_updated[K:])
                build_covariance_jit_timed = jax.jit(filt.build_covariance_jit)
                avg_time_buildcov = time_function(build_covariance_jit_timed, params.lambda_r, exp_h_dummy, params.sigma2, repetitions=repetitions)
                results.append({"N": N, "K": K, "Function": "build_covariance_jit", "Avg Time (s)": avg_time_buildcov})
                print(f"  build_covariance_jit: {avg_time_buildcov:.6f} s")

            except Exception as e:
                print(f"  ERROR profiling N={N}, K={K}: {e}")
                # Add error entry for this N, K combination
                results.append({"N": N, "K": K, "Function": "ERROR", "Avg Time (s)": -1})
                # Add placeholders for other functions for this N,K if needed for table structure
                func_names = ["predict_jax", "update_jax", "block_coordinate_update_impl_jit",
                              "fisher_information_jit", "log_posterior_jit", "build_covariance_jit"]
                existing_funcs = {r['Function'] for r in results if r['N'] == N and r['K'] == K}
                for fname in func_names:
                    if fname not in existing_funcs:
                         results.append({"N": N, "K": K, "Function": fname, "Avg Time (s)": -1})


    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile DFSV Bellman Filter Components.")
    parser.add_argument('--N', type=int, nargs='+', default=[10, 30, 50], help='List of N values (number of series)')
    parser.add_argument('--K', type=int, nargs='+', default=[5, 10], help='List of K values (number of factors)')
    parser.add_argument('--T', type=int, default=50, help='Number of time steps for dummy data generation (only affects data gen)')
    parser.add_argument('--reps', type=int, default=5, help='Number of repetitions for timing each function')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data generation')

    args = parser.parse_args()

    print("Starting Bellman Filter Profiling...")
    print(f"N values: {args.N}")
    print(f"K values: {args.K}")
    print(f"T: {args.T}")
    print(f"Repetitions: {args.reps}")
    print(f"Seed: {args.seed}")
    print(f"Using JAX default device: {jax.default_backend()}")
    print("-" * 30)


    prng_key = jax.random.PRNGKey(args.seed)

    profiling_results = profile_filter(args.N, args.K, args.T, args.reps, prng_key)

    print("\n--- Profiling Summary ---")
    # Optional: Print results in a more structured way, e.g., using pandas
    try:
        import pandas as pd
        if profiling_results:
            df = pd.DataFrame(profiling_results)
            # Pivot for better readability
            df_pivot = df.pivot(index=['N', 'K'], columns='Function', values='Avg Time (s)')
            print(df_pivot.to_string(float_format="%.6f"))
        else:
            print("No results generated.")
    except ImportError:
        print("Install pandas for formatted table output ('pip install pandas' or 'uv pip install pandas').")
        # Basic printout if pandas is not available
        header = ["N", "K", "Function", "Avg Time (s)"]
        print("\t".join(header))
        print("-" * (sum(len(h) for h in header) + len(header)*3))
        for res in profiling_results:
            print(f"{res.get('N', '?')}\t{res.get('K', '?')}\t{res.get('Function', 'N/A')}\t{res.get('Avg Time (s)', -1):.6f}")
    except Exception as e_pd:
         print(f"Error formatting results with pandas: {e_pd}")
         print("Raw results:")
         for res in profiling_results:
            print(res)


    print("\nProfiling Complete.")