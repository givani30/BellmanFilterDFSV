#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Runs all replicates for a single DFSV simulation configuration (N, K, Filter, Particles).
Designed to be called as a single task by a batch processing system (e.g., Google Cloud Batch),
amortizing JIT compilation costs.
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
import gcsfs # Added for GCS access
import io # For handling file streams with numpy

# Assuming 'bellman_filter_dfsv' is installed and accessible in the environment
try:
    from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
    from bellman_filter_dfsv.models.simulation import simulate_DFSV
    from bellman_filter_dfsv.filters.bellman import DFSVBellmanFilter
    from bellman_filter_dfsv.filters.particle import DFSVParticleFilter
    from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter # Added BIF
except ImportError as e:
    print(f"Error importing bellman_filter_dfsv package: {e}")
    print("Please ensure the package is installed correctly (e.g., 'pip install -e .')")
    sys.exit(1)

# --- Helper Functions (Copied/Adapted from simulation_study.py / run_single_replicate.py) ---

def save_replicate_results(run_path_str: str, metrics: Dict[str, Any], raw_data_out: Dict[str, Any], save_format: str):
    """
    Saves metrics (JSON) and raw data (NPZ/Parquet) for a single replicate.
    Handles both local and GCS paths.
    """
    is_gcs = run_path_str.startswith("gs://")
    fs = gcsfs.GCSFileSystem() if is_gcs else None
    run_path_obj = Path(run_path_str) # Keep Path object for naming if needed

    try:
        # Define paths as strings for gcsfs compatibility
        metrics_error_path_str = f"{run_path_str}/metrics_error.json"
        metrics_path_str = f"{run_path_str}/metrics.json"
        raw_data_path_str = f"{run_path_str}/raw_data.{save_format}" # Use correct extension

        # Handle error case first
        if metrics.get('error'):
            print(f"      --- Skipping full save for {run_path_obj.name} due to error: {metrics['error']} ---")
            # Ensure basic info is present for error logging
            error_info = {k: metrics.get(k) for k in ['error', 'N', 'K', 'T', 'seed', 'filter_type', 'num_particles', 'rep'] if metrics.get(k) is not None} # Added rep

            if is_gcs:
                with fs.open(metrics_error_path_str, 'w') as f:
                    json.dump(error_info, f, indent=4)
            else:
                # Ensure parent directory exists locally
                Path(metrics_error_path_str).parent.mkdir(parents=True, exist_ok=True)
                with open(metrics_error_path_str, 'w') as f:
                    json.dump(error_info, f, indent=4)
            print(f"      Saved error details to {metrics_error_path_str}")
            return # Stop saving process here

        # Proceed with normal saving if no error (using metrics_path_str)
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

        # Save metrics JSON
        if is_gcs:
            with fs.open(metrics_path_str, 'w') as f:
                json.dump(serializable_metrics, f, indent=4)
        else:
            Path(metrics_path_str).parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path_str, 'w') as f:
                json.dump(serializable_metrics, f, indent=4)

        # Save raw data (NPZ or Parquet)
        # Filter out None values and convert JAX arrays before saving
        data_to_save = {}
        for k, v in raw_data_out.items():
            if v is not None:
                if isinstance(v, jnp.ndarray):
                    data_to_save[k] = np.array(v)
                else:
                    data_to_save[k] = v

        if data_to_save:
            if save_format == "npz":
                if is_gcs:
                    # Use BytesIO buffer to save npz in memory then write to GCS
                    with io.BytesIO() as bio:
                        np.savez_compressed(bio, **data_to_save)
                        bio.seek(0) # Rewind buffer to the beginning
                        with fs.open(raw_data_path_str, 'wb') as f_gcs:
                            f_gcs.write(bio.read())
                else:
                    Path(raw_data_path_str).parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(raw_data_path_str, **data_to_save)
            elif save_format == "parquet":
                 print(f"      Warning: Parquet saving not implemented yet for {run_path_obj.name}.")
                 # Add GCS parquet saving logic here if needed later
            else:
                 print(f"      Warning: Unknown save_format '{save_format}' specified. Raw data not saved.")
        else:
            print(f"      Warning: No raw data to save for {run_path_obj.name} (filtered results might be missing).")


        # print(f"    Saved results for {run_path_obj.name} to {run_path_str}") # Reduce verbosity

    except Exception as e:
        print(f"    !!!!!! ERROR SAVING RESULTS for {run_path_obj.name} to {run_path_str}: {e}")
        # Attempt to remove potentially corrupted files if save failed mid-way
        # Use fs.exists and fs.rm for GCS, Path.exists/unlink for local
        try:
            if is_gcs:
                if fs.exists(metrics_path_str): fs.rm(metrics_path_str)
                if fs.exists(raw_data_path_str): fs.rm(raw_data_path_str)
            else:
                metrics_path_local = Path(metrics_path_str)
                raw_data_path_local = Path(raw_data_path_str)
                if metrics_path_local.exists(): metrics_path_local.unlink()
                if raw_data_path_local.exists(): raw_data_path_local.unlink()
        except Exception as cleanup_e:
             print(f"      !!!!!! FAILED TO CLEANUP potentially corrupted files for {run_path_obj.name}: {cleanup_e}")

def create_sim_parameters(N, K, seed=None) -> DFSVParamsDataclass:
    """Generates valid DFSVParamsDataclass for given N and K, ensuring stationarity."""
    if seed is not None:
        np.random.seed(seed)

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
    if isinstance(true_values, jnp.ndarray): true_values = np.array(true_values)
    if isinstance(estimated_values, jnp.ndarray): estimated_values = np.array(estimated_values)

    if true_values.shape != estimated_values.shape:
        min_T = min(true_values.shape[0], estimated_values.shape[0])
        if abs(true_values.shape[0] - estimated_values.shape[0]) <= 1 and true_values.shape[1:] == estimated_values.shape[1:]:
             # print(f"Warning: Adjusting shapes for accuracy calculation (T={true_values.shape[0]} vs T={estimated_values.shape[0]}). Using first {min_T} steps.")
             true_values = true_values[:min_T]
             estimated_values = estimated_values[:min_T]
        else:
            # Return NaNs if shapes are incompatible after trying adjustment
            num_cols = true_values.shape[1] if true_values.ndim > 1 else 1
            print(f"Error: Incompatible shapes for accuracy: {true_values.shape} vs {estimated_values.shape}")
            return np.full(num_cols, np.nan), np.full(num_cols, np.nan)


    if true_values.ndim == 1:
        true_values = true_values.reshape(-1, 1)
        estimated_values = estimated_values.reshape(-1, 1)

    rmse = np.sqrt(np.nanmean((true_values - estimated_values) ** 2, axis=0))
    correlations = []
    for k in range(true_values.shape[1]):
        valid_mask = ~np.isnan(true_values[:, k]) & ~np.isnan(estimated_values[:, k])
        if np.sum(valid_mask) < 2:
             corr = np.nan
        else:
             if np.std(true_values[valid_mask, k]) < 1e-10 or np.std(estimated_values[valid_mask, k]) < 1e-10:
                 corr = np.nan
             else:
                 try:
                     corr_matrix = np.corrcoef(true_values[valid_mask, k], estimated_values[valid_mask, k])
                     if corr_matrix.shape == (2, 2):
                         corr = corr_matrix[0, 1]
                     else:
                         corr = np.nan if np.isnan(corr_matrix) else float(corr_matrix)
                 except Exception: # Catch potential errors in corrcoef
                     corr = np.nan
        correlations.append(corr)
    return rmse, np.array(correlations)


# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description='Run all replicates for a single DFSV simulation configuration.')
    parser.add_argument('--N', type=int, required=True, help='Number of assets.')
    parser.add_argument('--K', type=int, required=True, help='Number of factors.')
    parser.add_argument('--T', type=int, required=True, help='Time series length.')
    parser.add_argument('--filter_type', type=str, required=True, choices=['BF', 'PF', 'BIF'], help='Filter type (Bellman, Particle, or Bellman Information).') # Added BIF
    parser.add_argument('--num_particles', type=int, help='Number of particles (required for PF).')
    parser.add_argument('--num_reps', type=int, required=True, help='Number of replicates to run for this configuration.')
    parser.add_argument('--base_results_dir', type=str, required=True, help='Base directory where study folders are located.')
    parser.add_argument('--study_name', type=str, required=True, help='Name of the specific study sub-directory.')
    parser.add_argument('--save_format', type=str, default='npz', choices=['npz', 'parquet'], help='Format to save raw data.')

    args = parser.parse_args()

    # --- Validate Arguments ---
    if args.filter_type == 'PF' and args.num_particles is None:
        parser.error("--num_particles is required when --filter_type is PF.")
    if args.K > args.N:
        print(f"Skipping N={args.N}, K={args.K} as K > N.")
        sys.exit(0) # Exit gracefully

    # --- Setup Paths & GCS ---
    base_results_dir_str = args.base_results_dir
    is_gcs = base_results_dir_str.startswith("gs://")
    fs = gcsfs.GCSFileSystem() if is_gcs else None

    study_path_str = f"{base_results_dir_str}/{args.study_name}"
    # Ensure study path exists
    if is_gcs:
        fs.mkdirs(study_path_str, exist_ok=True)
    else:
        Path(study_path_str).mkdir(parents=True, exist_ok=True)

    # --- Instantiate Filter ONCE ---
    bf_instance = None
    pf_instance = None
    bif_instance = None # Added BIF instance
    instantiation_time = None
    start_time_inst = time.time()
    try:
        if args.filter_type == 'BF':
            print(f"Instantiating Bellman Filter (N={args.N}, K={args.K})...")
            bf_instance = DFSVBellmanFilter(args.N, args.K)
            # Optional: Pre-compile JAX functions here if possible/needed
            # bf_instance.precompile_jax_functions() # Assuming such a method exists
        elif args.filter_type == 'PF':
            # Use a consistent seed for PF instantiation across identical configs
            pf_base_seed = args.N * 100 + args.K * 10 + args.num_particles
            print(f"Instantiating Particle Filter (N={args.N}, K={args.K}, P={args.num_particles}, seed={pf_base_seed})...")
            pf_instance = DFSVParticleFilter(N=args.N, K=args.K, num_particles=args.num_particles, seed=pf_base_seed)
            # Optional: Pre-compile JAX functions here
            # pf_instance.precompile_jax_functions() # Assuming such a method exists
        elif args.filter_type == 'BIF': # Added BIF instantiation
            print(f"Instantiating Bellman Information Filter (N={args.N}, K={args.K})...")
            bif_instance = DFSVBellmanInformationFilter(args.N, args.K)
            # Optional: Pre-compile JAX functions here
            # bif_instance.precompile_jax_functions() # Assuming such a method exists
        instantiation_time = time.time() - start_time_inst
        print(f"Filter instantiation took {instantiation_time:.2f}s")
    except Exception as e:
        # Updated error message to handle BIF (which doesn't use num_particles)
        filter_details = f"{args.filter_type}"
        if args.filter_type == 'PF':
            filter_details += f", P={args.num_particles}"
        print(f"!!!!!! FATAL ERROR: Filter instantiation failed for N={args.N}, K={args.K}, Filter={filter_details}: {e}")
        # Optionally write a config-level error file using string path and fs if needed
        error_file_suffix = f"{args.filter_type}"
        if args.filter_type == 'PF':
            error_file_suffix += f"{args.num_particles}"
        error_file_path_str = f"{study_path_str}/config_N{args.N}_K={args.K}_{error_file_suffix}_INSTANTIATION_ERROR.json"
        error_payload = {'error': f"Instantiation failed: {e}", **vars(args)}
        try:
            if is_gcs:
                with fs.open(error_file_path_str, 'w') as f:
                    json.dump(error_payload, f, indent=4)
            else:
                # Ensure parent directory exists locally
                Path(error_file_path_str).parent.mkdir(parents=True, exist_ok=True)
                with open(error_file_path_str, 'w') as f:
                    json.dump(error_payload, f, indent=4)
            print(f"      Saved instantiation error details to {error_file_path_str}")
        except Exception as save_e:
             print(f"      !!!!!! FAILED TO SAVE instantiation error details: {save_e}")
        sys.exit(1) # Exit with error code

    # --- Replicate Loop ---
    total_replicate_time = 0
    completed_reps = 0
    error_reps = 0

    # Updated log message to handle BIF
    filter_details_log = f"{args.filter_type}"
    if args.filter_type == 'PF':
        filter_details_log += f", P={args.num_particles}"
    print(f"\n--- Starting Batch for Config: N={args.N}, K={args.K}, Filter={filter_details_log}, Reps={args.num_reps} ---")
    start_time_batch = time.time()

    for rep in range(args.num_reps):
        start_time_rep = time.time()

        # --- Construct Run Label and Path ---
        if args.filter_type == "BF":
            run_label = f"config_N{args.N}_K{args.K}_BF_rep{rep}"
        elif args.filter_type == "PF":
            run_label = f"config_N{args.N}_K{args.K}_PF{args.num_particles}_rep{rep}"
        elif args.filter_type == "BIF": # Added BIF label
            run_label = f"config_N{args.N}_K{args.K}_BIF_rep{rep}"
        else: # Should not happen due to choices in argparse
            raise ValueError(f"Unsupported filter_type: {args.filter_type}")
        # Construct run path string
        run_path_str = f"{study_path_str}/{run_label}"

        # --- Check if Already Completed ---
        metrics_path_str = f"{run_path_str}/metrics.json"
        error_metrics_path_str = f"{run_path_str}/metrics_error.json"
        # Use fs.exists for GCS, Path.exists for local
        already_exists = False
        try:
            if is_gcs:
                already_exists = fs.exists(metrics_path_str) or fs.exists(error_metrics_path_str)
            else:
                already_exists = Path(metrics_path_str).exists() or Path(error_metrics_path_str).exists()
        except Exception as check_e:
             print(f"  Warning: Could not check existence for {run_label}: {check_e}")
             # Decide whether to proceed or skip if existence check fails (safer to skip)
             error_reps += 1
             continue

        if already_exists:
            # print(f"  Skipping completed/errored Replicate {rep}: {run_label}")
            completed_reps += 1 # Count skipped as completed for progress
            continue

        # --- Start Actual Work for Replicate ---
        # Directory creation is handled by save_replicate_results if needed
        # try:
        #     if is_gcs:
        #         fs.mkdirs(run_path_str, exist_ok=True) # mkdirs for GCS
        #     else:
        #         Path(run_path_str).mkdir(parents=True, exist_ok=True)
        # except Exception as e: # Catch broader exceptions for GCS
        #     print(f"  Error creating directory {run_path_str} for Rep {rep}: {e}")
        #     error_reps += 1
        #     continue # Skip to next replicate

        # --- Calculate Seed ---
        seed = args.N * 10000 + args.K * 1000 + (args.num_particles or 0) * 10 + rep

        # --- Initialize Metrics and Data ---
        metrics = {
            'N': args.N, 'K': args.K, 'T': args.T,
            'num_particles': args.num_particles if args.filter_type == 'PF' else None,
            'seed': seed, 'filter_type': args.filter_type, 'rep': rep, # Added rep
            'param_gen_time': None, 'data_sim_time': None,
            'bf_filter_time': None, 'pf_filter_time': None, 'bif_filter_time': None, # Added BIF time field
            'bf_rmse_f': None, 'bf_corr_f': None, 'bf_rmse_h': None, 'bf_corr_h': None,
            'pf_rmse_f': None, 'pf_corr_f': None, 'pf_rmse_h': None, 'pf_corr_h': None,
            'bif_rmse_f': None, 'bif_corr_f': None, 'bif_rmse_h': None, 'bif_corr_h': None, # Added BIF accuracy fields
            'error': None
        }
        raw_data_out = {
            'returns': None, 'true_factors': None, 'true_log_vols': None,
            'filtered_factors_bf': None, 'filtered_log_vols_bf': None,
            'filtered_factors_pf': None, 'filtered_log_vols_pf': None,
            'filtered_factors_bif': None, 'filtered_log_vols_bif': None # Added BIF raw data fields
        }

        try:
            # 1. Create Parameters
            start_time = time.time()
            params_rep = create_sim_parameters(args.N, args.K, seed=seed)
            metrics['param_gen_time'] = time.time() - start_time

            # 2. Simulate Data
            start_time = time.time()
            returns_rep, true_factors_rep, true_log_vols_rep = simulate_DFSV(params=params_rep, T=args.T, seed=seed+1)
            raw_data_out['returns'] = returns_rep
            raw_data_out['true_factors'] = true_factors_rep
            raw_data_out['true_log_vols'] = true_log_vols_rep
            metrics['data_sim_time'] = time.time() - start_time

            # 3. Run Filter (using pre-instantiated object)
            if args.filter_type == 'BF':
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
                else: print("Warning: BF object missing result methods.")

            elif args.filter_type == 'PF':
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
                else: print("Warning: PF object missing result methods.")

                # Clear JAX caches periodically if needed (might not be necessary if filter is reused)
                # if (rep + 1) % 10 == 0: jax.clear_caches()

            elif args.filter_type == 'BIF': # Added BIF execution
                start_time = time.time()
                # Assuming filter_scan exists and has a similar signature
                bif_instance.filter_scan(params_rep, returns_rep)
                metrics['bif_filter_time'] = time.time() - start_time # Store BIF time

                # Accuracy calculation and result storage for BIF
                if hasattr(bif_instance, 'get_filtered_factors') and hasattr(bif_instance, 'get_filtered_volatilities'):
                    filtered_factors_bif = bif_instance.get_filtered_factors()
                    filtered_log_vols_bif = bif_instance.get_filtered_volatilities()
                    raw_data_out['filtered_factors_bif'] = filtered_factors_bif
                    raw_data_out['filtered_log_vols_bif'] = filtered_log_vols_bif
                    metrics['bif_rmse_f'], metrics['bif_corr_f'] = calculate_accuracy(true_factors_rep, filtered_factors_bif)
                    metrics['bif_rmse_h'], metrics['bif_corr_h'] = calculate_accuracy(true_log_vols_rep, filtered_log_vols_bif)
                else: print("Warning: BIF object missing result methods.")

        except Exception as e:
            print(f"  !!!!!! ERROR during execution of Rep {rep} ({run_label}): {e}")
            metrics['error'] = f"Execution error: {e}"
            error_reps += 1
            # Attempt to save error metric if possible

        # 4. Save Results for this replicate (using string path)
        save_replicate_results(run_path_str, metrics, raw_data_out, args.save_format)

        if metrics['error'] is None:
            completed_reps += 1
        # else: error_reps already incremented

        rep_time = time.time() - start_time_rep
        total_replicate_time += rep_time
        # print(f"  Finished Rep {rep} in {rep_time:.2f}s") # Reduce verbosity

    # --- Batch Complete ---
    end_time_batch = time.time()
    total_batch_duration = end_time_batch - start_time_batch
    avg_rep_time = total_replicate_time / completed_reps if completed_reps > 0 else 0

    # Updated completion log message
    filter_details_log_end = f"{args.filter_type}"
    if args.filter_type == 'PF':
        filter_details_log_end += f", P={args.num_particles}"
    print(f"\n--- Completed Batch for Config: N={args.N}, K={args.K}, Filter={filter_details_log_end} ---")
    print(f"  Total Batch Time: {total_batch_duration:.2f}s")
    print(f"  Completed Replicates: {completed_reps}/{args.num_reps}")
    print(f"  Errored Replicates: {error_reps}")
    print(f"  Average Time per Successful Replicate (excluding skipped): {avg_rep_time:.3f}s")
    print(f"  Filter Instantiation Time: {instantiation_time:.2f}s")


if __name__ == "__main__":
    main()