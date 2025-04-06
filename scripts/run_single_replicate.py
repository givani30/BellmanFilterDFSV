#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Runs a single replicate of the DFSV simulation study based on command-line arguments.
Designed to be called by a batch processing system (e.g., Google Cloud Batch).
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path
import json
import time
import typing
from typing import Dict, Any, Tuple, Optional, List

import jax
import numpy as np
import jax.numpy as jnp

# Assuming 'bellman_filter_dfsv' is installed and accessible in the environment
try:
    from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
    from bellman_filter_dfsv.models.simulation import simulate_DFSV
    from bellman_filter_dfsv.filters.bellman import DFSVBellmanFilter
    from bellman_filter_dfsv.filters.particle import DFSVParticleFilter
except ImportError as e:
    print(f"Error importing bellman_filter_dfsv package: {e}")
    print("Please ensure the package is installed correctly (e.g., 'pip install -e .')")
    sys.exit(1)

# --- Helper Functions (Copied/Adapted from simulation_study.py) ---

def save_replicate_results(run_path: Path, metrics: Dict[str, Any], raw_data_out: Dict[str, Any], save_format: str):
    """Saves metrics (JSON) and raw data (NPZ/Parquet) for a single replicate."""
    try:
        # Handle error case first
        if metrics.get('error'):
            print(f"      --- Skipping full save for {run_path.name} due to error: {metrics['error']} ---")
            metrics_path = run_path / "metrics_error.json"
            # Ensure basic info is present for error logging
            error_info = {k: metrics.get(k) for k in ['error', 'N', 'K', 'T', 'seed', 'filter_type', 'num_particles'] if metrics.get(k) is not None}
            with open(metrics_path, 'w') as f:
                json.dump(error_info, f, indent=4)
            print(f"      Saved error details to {metrics_path}")
            return # Stop saving process here

        # Proceed with normal saving if no error
        metrics_path = run_path / "metrics.json"
        # Convert numpy arrays/types in metrics to lists/basic types for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.ndarray, jnp.ndarray)): # Handle JAX arrays too
                 # Convert JAX array to numpy first if necessary
                if isinstance(value, jnp.ndarray):
                    value = np.array(value)
                # Handle potential NaN values before converting to list
                if np.isnan(value).any():
                    serializable_metrics[key] = [float('nan') if np.isnan(v) else v for v in value.tolist()]
                else:
                    serializable_metrics[key] = value.tolist()
            elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                serializable_metrics[key] = int(value)
            elif isinstance(value, (np.floating, np.float16, np.float32, np.float64)):
                # Handle potential NaN floats
                serializable_metrics[key] = float('nan') if np.isnan(value) else float(value)
            elif isinstance(value, (np.bool_)):
                serializable_metrics[key] = bool(value)
            else:
                serializable_metrics[key] = value # Assume other types are JSON serializable

        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)

        if save_format == "npz":
            raw_data_path = run_path / "raw_data.npz"
            # Filter out None values and convert JAX arrays before saving
            data_to_save = {}
            for k, v in raw_data_out.items():
                if v is not None:
                    if isinstance(v, jnp.ndarray):
                        data_to_save[k] = np.array(v)
                    else:
                        data_to_save[k] = v

            if data_to_save:
                np.savez_compressed(raw_data_path, **data_to_save)
            else:
                print(f"      Warning: No raw data to save for {run_path.name} (filtered results might be missing).")
        elif save_format == "parquet":
            # Placeholder for parquet saving logic if needed in the future
            print(f"      Warning: Parquet saving not implemented yet for {run_path.name}.")
            # Example using pyarrow (requires installation):
            # import pyarrow as pa
            # import pyarrow.parquet as pq
            # raw_data_path = run_path / "raw_data.parquet"
            # data_to_save = {k: v for k, v in raw_data_out.items() if v is not None}
            # if data_to_save:
            #     # Convert numpy arrays to pyarrow arrays/tables
            #     # This part needs careful handling of data types and structure
            #     try:
            #         # Convert JAX arrays to numpy first
            #         np_data_to_save = {k: (np.array(v) if isinstance(v, jnp.ndarray) else v) for k, v in data_to_save.items()}
            #         table = pa.Table.from_pydict(np_data_to_save) # Simplistic conversion
            #         pq.write_table(table, raw_data_path)
            #     except Exception as pq_e:
            #          print(f"      !!!!!! ERROR SAVING PARQUET for {run_path.name}: {pq_e}")
            # else:
            #      print(f"      Warning: No raw data to save for {run_path.name}.")
        else:
            print(f"      Warning: Unknown save_format '{save_format}' specified. Raw data not saved.")

        print(f"    Saved results for {run_path.name} to {run_path}")

    except Exception as e:
        print(f"    !!!!!! ERROR SAVING RESULTS for {run_path.name} to {run_path}: {e}")
        # Attempt to remove potentially corrupted files if save failed mid-way
        if 'metrics_path' in locals() and metrics_path.exists():
            try: metrics_path.unlink()
            except OSError: pass
        if 'raw_data_path' in locals() and raw_data_path.exists():
            try: raw_data_path.unlink()
            except OSError: pass

def create_sim_parameters(N, K, seed=None) -> DFSVParamsDataclass:
    """Generates valid DFSVParamsDataclass for given N and K, ensuring stationarity."""
    if seed is not None:
        np.random.seed(seed)

    # Generate parameters ensuring stationarity and positive definiteness
    lambda_r = np.random.normal(0, 1, size=(N, K))
    Phi_f_raw = np.random.normal(0, 0.5, size=(K, K))
    max_eig_f = np.max(np.abs(np.linalg.eigvals(Phi_f_raw)))
    Phi_f = 0.8 * Phi_f_raw / max_eig_f if max_eig_f > 0.8 else Phi_f_raw
    Phi_h_raw = np.random.normal(0, 0.5, size=(K, K))
    max_eig_h = np.max(np.abs(np.linalg.eigvals(Phi_h_raw)))
    Phi_h = 0.95 * Phi_h_raw / max_eig_h if max_eig_h > 0.95 else Phi_h_raw
    mu = np.random.normal(-1, 0.5, size=(K, 1))
    sigma2 = np.exp(np.random.normal(-1, 0.5, size=N))
    Q_h_raw = np.random.normal(0, 0.2, size=(K, K))
    Q_h = 1.0 * (Q_h_raw @ Q_h_raw.T) + np.eye(K) * 1e-4

    params = DFSVParamsDataclass(
        N=N, K=K,
        lambda_r=jnp.array(lambda_r),
        Phi_f=jnp.array(Phi_f),
        Phi_h=jnp.array(Phi_h),
        mu=jnp.array(mu.flatten()),
        sigma2=jnp.array(sigma2),
        Q_h=jnp.array(Q_h),
    )
    return params

def calculate_accuracy(true_values, estimated_values):
    """Calculates RMSE and Correlation."""
    # Convert JAX arrays to NumPy if necessary
    if isinstance(true_values, jnp.ndarray):
        true_values = np.array(true_values)
    if isinstance(estimated_values, jnp.ndarray):
        estimated_values = np.array(estimated_values)

    if true_values.shape != estimated_values.shape:
        # Try to handle T vs T+1 dimension mismatch if shapes are close
        min_T = min(true_values.shape[0], estimated_values.shape[0])
        if abs(true_values.shape[0] - estimated_values.shape[0]) <= 1 and true_values.shape[1:] == estimated_values.shape[1:]:
             print(f"Warning: Adjusting shapes for accuracy calculation (T={true_values.shape[0]} vs T={estimated_values.shape[0]}). Using first {min_T} steps.")
             true_values = true_values[:min_T]
             estimated_values = estimated_values[:min_T]
        else:
            raise ValueError(f"Shapes of true {true_values.shape} and estimated {estimated_values.shape} values must match or be reconcilable.")

    if true_values.ndim == 1:
        true_values = true_values.reshape(-1, 1)
        estimated_values = estimated_values.reshape(-1, 1)

    # Calculate RMSE, handling potential NaNs in estimates
    rmse = np.sqrt(np.nanmean((true_values - estimated_values) ** 2, axis=0))

    correlations = []
    for k in range(true_values.shape[1]):
        # Handle potential NaN values in correlation calculation
        valid_mask = ~np.isnan(true_values[:, k]) & ~np.isnan(estimated_values[:, k])
        if np.sum(valid_mask) < 2: # Need at least 2 points for correlation
             corr = np.nan
        else:
             # Check for zero variance
             if np.std(true_values[valid_mask, k]) < 1e-10 or np.std(estimated_values[valid_mask, k]) < 1e-10:
                 corr = np.nan # Or 1.0 if they are identical constants, but NaN is safer
             else:
                 corr_matrix = np.corrcoef(true_values[valid_mask, k], estimated_values[valid_mask, k])
                 if corr_matrix.shape == (2, 2):
                     corr = corr_matrix[0, 1]
                 else: # Handle cases where corrcoef returns scalar (e.g., perfect correlation)
                     corr = np.nan if np.isnan(corr_matrix) else float(corr_matrix) # Ensure float
        correlations.append(corr)
    return rmse, np.array(correlations)


# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description='Run a single DFSV simulation replicate.')
    parser.add_argument('--N', type=int, required=True, help='Number of assets.')
    parser.add_argument('--K', type=int, required=True, help='Number of factors.')
    parser.add_argument('--T', type=int, required=True, help='Time series length.')
    parser.add_argument('--filter_type', type=str, required=True, choices=['BF', 'PF'], help='Filter type (Bellman or Particle).')
    parser.add_argument('--num_particles', type=int, help='Number of particles (required for PF).')
    parser.add_argument('--rep', type=int, required=True, help='Replicate number (0-indexed).')
    parser.add_argument('--base_results_dir', type=str, required=True, help='Base directory where study folders are located.')
    parser.add_argument('--study_name', type=str, required=True, help='Name of the specific study sub-directory.')
    parser.add_argument('--save_format', type=str, default='npz', choices=['npz', 'parquet'], help='Format to save raw data.')

    args = parser.parse_args()

    # --- Validate Arguments ---
    if args.filter_type == 'PF' and args.num_particles is None:
        parser.error("--num_particles is required when --filter_type is PF.")
    if args.K > args.N:
        print(f"Skipping N={args.N}, K={args.K} as K > N.")
        sys.exit(0) # Exit gracefully, not an error for batch processing

    # --- Construct Paths ---
    base_results_path = Path(args.base_results_dir)
    study_path = base_results_path / args.study_name
    if args.filter_type == "BF":
        run_label = f"config_N{args.N}_K{args.K}_BF_rep{args.rep}"
    else: # PF
        run_label = f"config_N{args.N}_K{args.K}_PF{args.num_particles}_rep{args.rep}"
    run_path = study_path / run_label

    # --- Check if Already Completed ---
    metrics_path = run_path / "metrics.json"
    error_metrics_path = run_path / "metrics_error.json"
    if metrics_path.exists() or error_metrics_path.exists():
        print(f"Skipping completed or errored replicate: {run_label}")
        sys.exit(0) # Exit gracefully

    # --- Start Actual Work ---
    print(f"--- Starting {args.filter_type} Replicate {args.rep} ({run_label}) ---")
    try:
        run_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {run_path}: {e}")
        sys.exit(1)

    # --- Calculate Seed ---
    # Use a consistent seeding strategy based on parameters
    seed = args.N * 10000 + args.K * 1000 + (args.num_particles or 0) * 10 + args.rep

    # --- Initialize Metrics and Data ---
    metrics = {
        'N': args.N, 'K': args.K, 'T': args.T,
        'num_particles': args.num_particles if args.filter_type == 'PF' else None,
        'seed': seed, 'filter_type': args.filter_type,
        'param_gen_time': None, 'data_sim_time': None,
        'bf_instantiation_time': None, 'pf_instantiation_time': None,
        'bf_filter_time': None, 'pf_filter_time': None,
        'bf_rmse_f': None, 'bf_corr_f': None, 'bf_rmse_h': None, 'bf_corr_h': None,
        'pf_rmse_f': None, 'pf_corr_f': None, 'pf_rmse_h': None, 'pf_corr_h': None,
        'error': None
    }
    raw_data_out = {
        'returns': None, 'true_factors': None, 'true_log_vols': None,
        'filtered_factors_bf': None, 'filtered_log_vols_bf': None,
        'filtered_factors_pf': None, 'filtered_log_vols_pf': None
    }

    bf_instance = None
    pf_instance = None

    try:
        # 1. Create Parameters
        start_time = time.time()
        try:
            params_rep = create_sim_parameters(args.N, args.K, seed=seed)
            metrics['param_gen_time'] = time.time() - start_time
        except Exception as e:
             metrics['error'] = f"Parameter generation failed: {e}"
             print(f"      Error creating parameters: {e}. Skipping replicate.")
             save_replicate_results(run_path, metrics, raw_data_out, args.save_format)
             sys.exit(0) # Exit gracefully after logging error

        # 2. Simulate Data
        start_time = time.time()
        try:
            returns_rep, true_factors_rep, true_log_vols_rep = simulate_DFSV(params=params_rep, T=args.T, seed=seed+1) # Use different seed for simulation
            raw_data_out['returns'] = returns_rep
            raw_data_out['true_factors'] = true_factors_rep
            raw_data_out['true_log_vols'] = true_log_vols_rep
            metrics['data_sim_time'] = time.time() - start_time
        except Exception as e:
             metrics['error'] = f"Data simulation failed: {e}"
             print(f"      Error simulating data: {e}. Skipping replicate.")
             save_replicate_results(run_path, metrics, raw_data_out, args.save_format)
             sys.exit(0) # Exit gracefully

        # 3. Instantiate and Run Filter
        if args.filter_type == 'BF':
            start_time = time.time()
            try:
                bf_instance = DFSVBellmanFilter(args.N, args.K)
                metrics['bf_instantiation_time'] = time.time() - start_time
            except Exception as e:
                metrics['error'] = f"BF Instantiation failed: {e}"
                print(f"      Error instantiating Bellman Filter: {e}. Skipping filtering.")
                save_replicate_results(run_path, metrics, raw_data_out, args.save_format)
                sys.exit(0) # Exit gracefully

            start_time = time.time()
            bf_instance.filter_scan(params_rep, returns_rep)
            metrics['bf_filter_time'] = time.time() - start_time

            if hasattr(bf_instance, 'get_filtered_factors') and hasattr(bf_instance, 'get_filtered_volatilities'):
                filtered_factors_bf = bf_instance.get_filtered_factors()
                filtered_log_vols_bf = bf_instance.get_filtered_volatilities()
                raw_data_out['filtered_factors_bf'] = filtered_factors_bf
                raw_data_out['filtered_log_vols_bf'] = filtered_log_vols_bf
                metrics['bf_rmse_f'], metrics['bf_corr_f'] = calculate_accuracy(true_factors_rep, filtered_factors_bf)
                metrics['bf_rmse_h'], metrics['bf_corr_h'] = calculate_accuracy(true_log_vols_rep, filtered_log_vols_bf)
            else:
                 print("Warning: Bellman filter object missing get_filtered_factors or get_filtered_volatilities method.")

        elif args.filter_type == 'PF':
            start_time = time.time()
            try:
                # Use the main replicate seed for PF instantiation consistency
                pf_instance = DFSVParticleFilter(N=args.N, K=args.K, num_particles=args.num_particles, seed=seed)
                metrics['pf_instantiation_time'] = time.time() - start_time
            except Exception as e:
                metrics['error'] = f"PF Instantiation failed: {e}"
                print(f"      Error instantiating Particle Filter: {e}. Skipping filtering.")
                save_replicate_results(run_path, metrics, raw_data_out, args.save_format)
                sys.exit(0) # Exit gracefully

            start_time = time.time()
            _, _, _ = pf_instance.filter(params=params_rep, observations=returns_rep)
            metrics['pf_filter_time'] = time.time() - start_time

            if hasattr(pf_instance, 'get_filtered_factors') and hasattr(pf_instance, 'get_filtered_volatilities'):
                filtered_factors_pf = pf_instance.get_filtered_factors()
                filtered_log_vols_pf = pf_instance.get_filtered_volatilities()
                raw_data_out['filtered_factors_pf'] = filtered_factors_pf
                raw_data_out['filtered_log_vols_pf'] = filtered_log_vols_pf
                metrics['pf_rmse_f'], metrics['pf_corr_f'] = calculate_accuracy(true_factors_rep, filtered_factors_pf)
                metrics['pf_rmse_h'], metrics['pf_corr_h'] = calculate_accuracy(true_log_vols_rep, filtered_log_vols_pf)
            else:
                 print("Warning: Particle filter object missing get_filtered_factors or get_filtered_volatilities method.")

            # Clear JAX caches after PF run
            jax.clear_caches()
            # print(f"    Cleared JAX caches for Rep {args.rep} (PF P={args.num_particles})")


        print(f"Finished filtering: N={args.N}, K={args.K}, Filter={args.filter_type}, Seed={seed}. "
              f"{f'BF Time: {metrics['bf_filter_time']:.2f}s' if metrics['bf_filter_time'] is not None else ''}"
              f"{f', PF Time: {metrics['pf_filter_time']:.2f}s' if metrics['pf_filter_time'] is not None else ''}")

    except Exception as e:
        print(f"!!!!!! UNEXPECTED ERROR during execution of {run_label}: {e}")
        metrics['error'] = f"Outer execution error: {e}"
        # Attempt to save error metric if possible (save_replicate_results handles this)

    # 4. Save Results (handles errors internally)
    save_replicate_results(run_path, metrics, raw_data_out, args.save_format)

    print(f"--- Completed Replicate {args.rep} ({run_label}) ---")


if __name__ == "__main__":
    main()