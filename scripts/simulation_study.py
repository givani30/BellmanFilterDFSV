#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Check if we're in the correct environment
import sys
import argparse # Added import
from datetime import datetime
from pathlib import Path # Added import
import json # Added import
# import tracemalloc # REMOVING verification instrumentation


import jax # Import jax itself for clear_caches
import jax.profiler # Added for JAX device memory profiling

import numpy as np
import jax.numpy as jnp # Add JAX numpy import
import time

# Remove sys.path hack
# Imports should work if qf_thesis is installed editable

# Updated imports
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass # Import the JAX dataclass
from bellman_filter_dfsv.core.simulation import simulate_DFSV
from bellman_filter_dfsv.core.filters.bellman import DFSVBellmanFilter
from bellman_filter_dfsv.core.filters.particle import DFSVParticleFilter

# Added imports
from typing import Dict, Any, Tuple, Optional

# --- Configuration Handling ---
def load_simulation_config(config_path: Path) -> Dict[str, Any]:
    """Loads simulation configuration from a JSON file."""
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {config_path}")
        # Basic validation (can be expanded)
        required_keys = ["N_values", "K_values", "T", "num_particles_values", "num_reps", "base_results_dir", "save_format"]
        if not all(key in config for key in required_keys):
            print(f"Error: Configuration file {config_path} is missing required keys.")
            sys.exit(1)
        # Ensure base_results_dir is Path object
        config["base_results_dir"] = Path(config["base_results_dir"])
        return config
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        sys.exit(1)

# --- Study Environment Setup ---
def setup_study_environment(config: Dict[str, Any], resume_study_name: Optional[str]) -> Tuple[Path, Path, Dict[str, Any], Dict[str, Any]]: # Added config return
    """Sets up the study directory, handles resume state, and returns paths/indices/config."""
    base_results_path = config["base_results_dir"]
    start_indices = {"N": 0, "K": 0, "filter_type": "BF", "particles": 0, "rep": 0} # Default start

    if resume_study_name:
        study_path = base_results_path / resume_study_name
        print(f"Attempting to resume study: {study_path}")
        if not study_path.is_dir():
            print(f"Error: Resume directory not found: {study_path}")
            sys.exit(1)

        # Load config from the specific study directory (overrides initial config)
        config_load_path = study_path / "simulation_config.json"
        config = load_simulation_config(config_load_path) # Reload config from study dir

        # Try to load checkpoint
        checkpoint_path = study_path / "checkpoint.json"
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    loaded_indices = json.load(f)
                # Basic validation
                if all(k in loaded_indices for k in start_indices):
                    start_indices = loaded_indices
                    print(f"Loaded checkpoint. Resuming from: N_idx={start_indices['N']}, K_idx={start_indices['K']}, Filter={start_indices['filter_type']}, Particles_idx={start_indices['particles']}, Rep={start_indices['rep']}")
                else:
                    print("Warning: Checkpoint file format invalid. Starting from beginning of resume directory.")
            except Exception as e:
                print(f"Warning: Could not load checkpoint file {checkpoint_path}: {e}. Starting from beginning of resume directory.")
        else:
            print("No checkpoint file found. Starting from beginning of resume directory (will skip completed runs).")

    else:
        # Start a new study
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_dir_name = f"study_{timestamp}"
        study_path = base_results_path / study_dir_name
        study_path.mkdir(parents=True, exist_ok=True)
        print(f"Starting new study. Saving results to: {study_path}")

        # Save config to the new study directory
        config_save_path = study_path / "simulation_config.json"
        try:
            with open(config_save_path, 'w') as f:
                config_to_save = config.copy()
                # Store base_results_dir as string in JSON
                config_to_save["base_results_dir"] = str(config_to_save["base_results_dir"])
                json.dump(config_to_save, f, indent=4)
            print(f"Saved configuration to {config_save_path}")
        except Exception as e:
            print(f"Error saving configuration to {config_save_path}: {e}")
            sys.exit(1)
        # Define checkpoint path for the new study
        checkpoint_path = study_path / "checkpoint.json"

    return study_path, checkpoint_path, start_indices, config # Return potentially updated config

# --- Result Saving ---
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
            if isinstance(value, np.ndarray):
                # Handle potential NaN values before converting to list
                if np.isnan(value).any():
                    serializable_metrics[key] = [float('nan') if np.isnan(v) else v for v in value.tolist()] # Use tolist()
                else:
                    serializable_metrics[key] = value.tolist()
            elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                serializable_metrics[key] = int(value)
            elif isinstance(value, (np.floating, np.float16, np.float32, np.float64)):
                serializable_metrics[key] = float(value)
            elif isinstance(value, (np.bool_)):
                serializable_metrics[key] = bool(value)
            else:
                serializable_metrics[key] = value

        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)

        if save_format == "npz":
            raw_data_path = run_path / "raw_data.npz"
            # Filter out None values before saving
            data_to_save = {k: v for k, v in raw_data_out.items() if v is not None}
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
            #         table = pa.Table.from_pydict(data_to_save) # Simplistic conversion
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

# --- Checkpoint Calculation ---
def calculate_next_checkpoint(n_idx: int, k_idx: int, filter_type: str, p_idx: int, rep: int, config: Dict[str, Any]) -> Optional[Dict[str, Any]]: # Return Optional
    """Calculates the indices for the next checkpoint. Returns None if study is complete."""
    N_values = config["N_values"]
    K_values = config["K_values"]
    num_particles_values = config["num_particles_values"]
    num_reps = config["num_reps"]

    next_n_idx, next_k_idx, next_filter_type, next_particles_idx, next_rep = n_idx, k_idx, filter_type, p_idx, rep + 1

    if filter_type == "BF":
        if next_rep >= num_reps: # Finished all BF reps for this N/K
            if num_particles_values: # Check if there are any PF runs configured
                next_filter_type = "PF"
                next_particles_idx = 0
                next_rep = 0
            else: # No PF runs, move to next K/N directly
                next_filter_type = "BF" # Reset for next K/N
                next_particles_idx = 0 # Reset for safety
                next_rep = 0
                next_k_idx += 1
                if next_k_idx >= len(K_values):
                    next_k_idx = 0
                    next_n_idx += 1
                    # Outer loop handles K>N check and n_idx bounds
    elif filter_type == "PF":
        if next_rep >= num_reps: # Finished all PF reps for this particle count
            next_particles_idx += 1
            next_rep = 0
            if next_particles_idx >= len(num_particles_values): # Finished all particle counts
                next_filter_type = "BF" # Go back to BF for next K/N
                next_particles_idx = 0 # Reset for safety
                next_rep = 0
                next_k_idx += 1
                if next_k_idx >= len(K_values):
                    next_k_idx = 0
                    next_n_idx += 1
                    # Outer loop handles K>N check and n_idx bounds

    # Return None if we've finished all N values
    if next_n_idx >= len(N_values):
        return None

    return {
        "N": next_n_idx,
        "K": next_k_idx,
        "filter_type": next_filter_type,
        "particles": next_particles_idx,
        "rep": next_rep
    }


# --- Replicate Execution ---
def execute_single_replicate(
    config: Dict[str, Any],
    run_params: Dict[str, Any],
    study_path: Path,
    bf_instance: Optional[DFSVBellmanFilter] = None,
    pf_instance: Optional[DFSVParticleFilter] = None
) -> bool:
    """Executes a single simulation replicate, including data generation, filtering, and saving."""
    N = run_params['N']
    K = run_params['K']
    T = config['T'] # Get T from main config
    filter_type = run_params['filter_type']
    num_particles = run_params.get('num_particles') # Might be None for BF
    rep = run_params['rep']
    seed = run_params['seed']

    # Construct run label and path
    if filter_type == "BF":
        run_label = f"config_N{N}_K{K}_BF_rep{rep}"
    elif filter_type == "PF":
        run_label = f"config_N{N}_K{K}_PF{num_particles}_rep{rep}"
    else:
        print(f"Error: Unknown filter_type '{filter_type}' in run_params.")
        return False # Indicate failure
    run_path = study_path / run_label
    metrics_path = run_path / "metrics.json"
    error_metrics_path = run_path / "metrics_error.json"

    # Check if already completed (check both normal and error metrics)
    if metrics_path.exists() or error_metrics_path.exists():
        print(f"    Skipping completed Replicate {run_label}")
        return True # Indicate success (already done)

    # --- Start Actual Work ---
    try:
        run_path.mkdir(exist_ok=True) # Create dir only if not skipping
        print(f"    --- Starting {filter_type} Replicate {rep+1}/{config['num_reps']} ({run_label}) ---")

        # 1. Create parameters and simulate data for this replicate
        try:
            params_rep = create_sim_parameters(N, K, seed=seed)
        except Exception as e:
             print(f"      Error creating parameters for {run_label}: {e}. Skipping replicate.")
             # Save minimal error metric
             save_replicate_results(run_path, {'error': str(e), **run_params}, {}, config["save_format"])
             return True # Treat as "success" in terms of loop continuation, error is logged

        returns_rep, true_factors_rep, true_log_vols_rep = simulate_DFSV(params=params_rep, T=T, seed=seed+1) # Use different seed for simulation

        # 2. Run filtering using the appropriate instance
        metrics, raw_data_out = run_single_simulation(
            N=N, K=K, T=T, seed=seed,
            params=params_rep,
            returns=returns_rep,
            true_factors=true_factors_rep,
            true_log_vols=true_log_vols_rep,
            bf_instance=bf_instance if filter_type == "BF" else None,
            pf_instance=pf_instance if filter_type == "PF" else None,
            num_particles=num_particles # Pass through for metrics dict
        )
        # Add filter_type to metrics for clarity if not already present
        metrics['filter_type'] = filter_type

        # 3. Save results (handles errors internally)
        save_replicate_results(run_path, metrics, raw_data_out, config["save_format"])

        # 4. Clear JAX caches (especially important for PF)
        if filter_type == "PF":
            jax.clear_caches()
            # print(f"    Cleared JAX caches for Rep {rep} (PF P={num_particles})")

        return True # Indicate success

    except Exception as e:
        print(f"    !!!!!! UNEXPECTED ERROR during execution of {run_label}: {e}")
        # Attempt to save error metric if possible
        try:
            save_replicate_results(run_path, {'error': f"Outer execution error: {e}", **run_params}, {}, config["save_format"])
        except Exception as save_err:
            print(f"      !!!!!! FAILED TO SAVE UNEXPECTED ERROR DETAILS for {run_label}: {save_err}")
        return False # Indicate failure



def create_sim_parameters(N, K, seed=None) -> DFSVParamsDataclass: # Update return type hint
    """
    Generates valid DFSVParamsDataclass for given N and K, ensuring stationarity.

    Args:
        N (int): Number of assets.
        K (int): Number of factors.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        DFSVParamsDataclass: A valid DFSVParamsDataclass object with JAX arrays.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate parameters ensuring stationarity and positive definiteness
    # Factor loadings
    lambda_r = np.random.normal(0, 1, size=(N, K))

    # Factor transition matrix (ensure eigenvalues < 1)
    Phi_f_raw = np.random.normal(0, 0.5, size=(K, K))
    max_eig_f = np.max(np.abs(np.linalg.eigvals(Phi_f_raw)))
    Phi_f = 0.8 * Phi_f_raw / max_eig_f if max_eig_f > 0.8 else Phi_f_raw # Scale if needed

    # Volatility transition matrix (ensure eigenvalues < 1)
    Phi_h_raw = np.random.normal(0, 0.5, size=(K, K))
    max_eig_h = np.max(np.abs(np.linalg.eigvals(Phi_h_raw)))
    Phi_h = 0.95 * Phi_h_raw / max_eig_h if max_eig_h > 0.95 else Phi_h_raw # Scale if needed

    # Volatility mean
    mu = np.random.normal(-1, 0.5, size=(K, 1))

    # Idiosyncratic variance (ensure positive)
    sigma2 = np.exp(np.random.normal(-1, 0.5, size=N)) # Keep as 1D array

    # Volatility covariance matrix (ensure positive definite)
    Q_h_raw = np.random.normal(0, 0.2, size=(K, K))
    Q_h = 1.0 * (Q_h_raw @ Q_h_raw.T) + np.eye(K) * 1e-4 # Ensure positive definite (Increased scaling from 0.1)

    # Create parameter dataclass object using JAX arrays
    params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=jnp.array(lambda_r),
        Phi_f=jnp.array(Phi_f),
        Phi_h=jnp.array(Phi_h),
        mu=jnp.array(mu.flatten()), # Ensure mu is 1D
        sigma2=jnp.array(sigma2), # Pass 1D array
        Q_h=jnp.array(Q_h),
    )
    return params

def calculate_accuracy(true_values, estimated_values):
    """Calculates RMSE and Correlation."""
    if true_values.shape != estimated_values.shape:
        raise ValueError("Shapes of true and estimated values must match.")
    if true_values.ndim == 1:
        true_values = true_values.reshape(-1, 1)
        estimated_values = estimated_values.reshape(-1, 1)

    rmse = np.sqrt(np.mean((true_values - estimated_values) ** 2, axis=0))
    correlations = []
    for k in range(true_values.shape[1]):
        # Handle potential NaN values in correlation calculation
        valid_mask = ~np.isnan(true_values[:, k]) & ~np.isnan(estimated_values[:, k])
        if np.sum(valid_mask) < 2: # Need at least 2 points for correlation
             corr = np.nan
        else:
             corr_matrix = np.corrcoef(true_values[valid_mask, k], estimated_values[valid_mask, k])
             if corr_matrix.shape == (2, 2):
                 corr = corr_matrix[0, 1]
             else: # Handle cases where variance is zero -> corrcoef returns scalar
                 corr = np.nan if np.isnan(corr_matrix) else 1.0 # Or handle as appropriate
        correlations.append(corr)
    return rmse, np.array(correlations)

def run_single_simulation(
    N: int,
    K: int,
    T: int,
    seed: int,
    params: DFSVParamsDataclass,
    returns: jnp.ndarray,
    true_factors: jnp.ndarray,
    true_log_vols: jnp.ndarray,
    bf_instance: DFSVBellmanFilter | None = None,
    pf_instance: DFSVParticleFilter | None = None,
    num_particles: int | None = None # Still needed for metrics dict
    ):
    """
    Runs filtering for a single simulation replicate using pre-initialized filters
    and pre-generated data/parameters.

    Args:
        N (int): Number of assets.
        K (int): Number of factors.
        T (int): Time series length.
        seed (int): Random seed used for this replicate's data generation.
        params (DFSVParamsDataclass): Parameters used for this replicate.
        returns (jnp.ndarray): Simulated returns for this replicate.
        true_factors (jnp.ndarray): True factors for this replicate.
        true_log_vols (jnp.ndarray): True log-volatilities for this replicate.
        bf_instance (DFSVBellmanFilter, optional): Pre-initialized Bellman Filter.
        pf_instance (DFSVParticleFilter, optional): Pre-initialized Particle Filter.
        num_particles (int, optional): Number of particles (for PF metrics).

    Returns:
        tuple: (metrics_dict, raw_data_dict)
               metrics_dict contains configuration and scalar results for this run.
               raw_data_dict contains the raw time series arrays for this run.
    """
    filter_type_str = "BF" if bf_instance else "PF" if pf_instance else "Unknown"
    particles_info = f"Particles={num_particles}" if pf_instance else "Particles=N/A"
    print(f"Running filtering: N={N}, K={K}, T={T}, Filter={filter_type_str}, {particles_info}, Seed={seed}")

    # Initialize return structures
    metrics = {
        'N': N, 'K': K, 'T': T, 'num_particles': num_particles, 'seed': seed,
        'bf_time': None, 'pf_time': None,
        'bf_rmse_f': None, 'bf_corr_f': None, 'bf_rmse_h': None, 'bf_corr_h': None,
        'pf_rmse_f': None, 'pf_corr_f': None, 'pf_rmse_h': None, 'pf_corr_h': None,
        'error': None
    }
    # Raw data is now passed in, but we add placeholders for filtered results
    raw_data_out = {
        'returns': returns, # Passed in
        'true_factors': true_factors, # Passed in
        'true_log_vols': true_log_vols, # Passed in
        'filtered_factors_bf': None,
        'filtered_log_vols_bf': None,
        'filtered_factors_pf': None,
        'filtered_log_vols_pf': None
    }
    # Initialize filtered data placeholders to None
    filtered_factors_bf = None
    filtered_log_vols_bf = None
    filtered_factors_pf = None
    filtered_log_vols_pf = None

    try:
        # Data generation is now done outside this function

        # 2. Run Bellman Filter if instance provided
        if bf_instance is not None:
            start_time_bf = time.time()
            # Use the provided instance and replicate-specific params/returns
            bf_instance.filter_scan(params, returns)
            end_time_bf = time.time()
            metrics['bf_time'] = end_time_bf - start_time_bf
            # Check if methods exist before calling
            if hasattr(bf_instance, 'get_filtered_factors') and hasattr(bf_instance, 'get_filtered_volatilities'):
                filtered_factors_bf = bf_instance.get_filtered_factors()
                filtered_log_vols_bf = bf_instance.get_filtered_volatilities()
                raw_data_out['filtered_factors_bf'] = filtered_factors_bf
                raw_data_out['filtered_log_vols_bf'] = filtered_log_vols_bf
                metrics['bf_rmse_f'], metrics['bf_corr_f'] = calculate_accuracy(true_factors, filtered_factors_bf)
                metrics['bf_rmse_h'], metrics['bf_corr_h'] = calculate_accuracy(true_log_vols, filtered_log_vols_bf)
            else:
                 print("Warning: Bellman filter object missing get_filtered_factors or get_filtered_volatilities method.")


        # 3. Run Particle Filter if instance provided
        if pf_instance is not None:
            if num_particles is None or num_particles <= 0:
                 raise ValueError("num_particles must be provided and positive for Particle Filter.")
            # Use the provided instance and replicate-specific params/returns
            start_time_pf = time.time()
            # Call filter with external params
            _, _, _ = pf_instance.filter(params=params, observations=returns)
            end_time_pf = time.time()
            metrics['pf_time'] = end_time_pf - start_time_pf
            # Check if methods exist before calling
            if hasattr(pf_instance, 'get_filtered_factors') and hasattr(pf_instance, 'get_filtered_volatilities'):
                filtered_factors_pf = pf_instance.get_filtered_factors()
                filtered_log_vols_pf = pf_instance.get_filtered_volatilities()
                raw_data_out['filtered_factors_pf'] = filtered_factors_pf
                raw_data_out['filtered_log_vols_pf'] = filtered_log_vols_pf
                metrics['pf_rmse_f'], metrics['pf_corr_f'] = calculate_accuracy(true_factors, filtered_factors_pf)
                metrics['pf_rmse_h'], metrics['pf_corr_h'] = calculate_accuracy(true_log_vols, filtered_log_vols_pf)
            else:
                 print("Warning: Particle filter object missing get_filtered_factors or get_filtered_volatilities method.")


        print(f"Finished filtering: N={N}, K={K}, Filter={filter_type_str}, Seed={seed}. "
              f"{f'BF Time: {metrics["bf_time"]:.2f}s' if metrics["bf_time"] is not None else ''}"
              f"{f', PF Time: {metrics["pf_time"]:.2f}s' if metrics["pf_time"] is not None else ''}")

    except Exception as e:
        print(f"Error during filtering N={N}, K={K}, Filter={filter_type_str}, Seed={seed}: {e}")
        metrics['error'] = str(e)

    # Return metrics and the raw data dict (which now includes filtered results)
    return metrics, raw_data_out

def save_checkpoint(path: Path, indices: dict):
    """Saves the next indices to the checkpoint file."""
    try:
        with open(path, 'w') as f:
            json.dump(indices, f, indent=4)
        # print(f"Checkpoint saved: {indices}") # Optional: for debugging
    except Exception as e:
        print(f"Warning: Could not save checkpoint to {path}: {e}")



# tracemalloc.start() # REMOVING verification instrumentation

def main():
    parser = argparse.ArgumentParser(description='Run DFSV simulation study with optional resume.')
    parser.add_argument('--config', type=str, default='scripts/default_simulation_config.json', help='Path to the simulation configuration JSON file.')
    parser.add_argument('--resume_study', type=str, help='Name of the study directory (e.g., study_YYYYMMDD_HHMMSS) to resume.')
    args = parser.parse_args()

    # --- Load Configuration ---
    config_path = Path(args.config)
    config = load_simulation_config(config_path)

    # --- Setup Study Environment (Handles Resume) ---
    study_path, checkpoint_path, start_indices, config = setup_study_environment(config, args.resume_study) # Config might be updated

    # --- Get Config Values ---
    N_values = config["N_values"]
    K_values = config["K_values"]
    num_particles_values = config["num_particles_values"]
    num_reps = config["num_reps"]
    T = config["T"] # Get T once

    # --- Simulation Loop ---
    n_idx = start_indices["N"]
    k_idx = start_indices["K"]
    current_filter_type = start_indices["filter_type"]
    p_idx = start_indices["particles"]
    rep = start_indices["rep"]

    while n_idx < len(N_values):
        N = N_values[n_idx]

        # Handle K loop reset and K > N check
        if k_idx >= len(K_values):
            k_idx = 0
            n_idx += 1
            continue # Go to next N iteration

        K = K_values[k_idx]
        if K > N:
            print(f"Skipping N={N}, K={K} as K > N")
            k_idx += 1
            current_filter_type = "BF" # Reset for next K
            p_idx = 0
            rep = 0
            continue # Go to next K iteration

        # --- Filter Instantiation (Once per N/K or N/K/P) ---
        bf_instance = None
        pf_instance = None
        if current_filter_type == "BF":
            print(f"\n=== Processing Configuration N={N}, K={K}, Filter=BF ===")
            print(f"  Instantiating Bellman Filter (N={N}, K={K})...")
            try:
                bf_instance = DFSVBellmanFilter(N, K)
            except Exception as e:
                print(f"  Error instantiating Bellman Filter for N={N}, K={K}: {e}. Skipping BF runs for this config.")
                # Move to next K (or N if K loop is done)
                k_idx += 1
                current_filter_type = "BF" # Reset for next K
                p_idx = 0
                rep = 0
                continue
        elif current_filter_type == "PF":
            if p_idx >= len(num_particles_values): # Should not happen if logic is correct, but safety check
                 print(f"Warning: Invalid p_idx {p_idx} for N={N}, K={K}. Moving to next K.")
                 k_idx += 1
                 current_filter_type = "BF" # Reset for next K
                 p_idx = 0
                 rep = 0
                 continue

            num_particles = num_particles_values[p_idx]
            print(f"\n=== Processing Configuration N={N}, K={K}, Filter=PF, Particles={num_particles} ===")
            pf_base_seed = N * 100 + K * 10 + num_particles # Consistent base seed
            print(f"  Instantiating Particle Filter (N={N}, K={K}, P={num_particles}, seed={pf_base_seed})...")
            try:
                pf_instance = DFSVParticleFilter(N=N, K=K, num_particles=num_particles, seed=pf_base_seed)
            except Exception as e:
                print(f"  Error instantiating Particle Filter for N={N}, K={K}, P={num_particles}: {e}. Skipping PF runs for this particle count.")
                # Move to next particle count (or next K if done)
                p_idx += 1
                rep = 0 # Reset rep count for next particle size
                if p_idx >= len(num_particles_values): # Finished all particle counts for this K
                    k_idx += 1
                    current_filter_type = "BF" # Reset for next K
                    p_idx = 0
                # No need to reset current_filter_type here, it stays PF until p_idx loop finishes or error
                continue

        # --- Replicate Loop ---
        while rep < num_reps:
            # --- Calculate Next Checkpoint & Save ---
            # Pass the *current* state to calculate the *next* checkpoint state
            next_checkpoint_indices = calculate_next_checkpoint(n_idx, k_idx, current_filter_type, p_idx, rep, config)
            if next_checkpoint_indices:
                save_checkpoint(checkpoint_path, next_checkpoint_indices)
            # Else: Study is complete after this replicate, checkpoint will be deleted later

            # --- Prepare Run Parameters ---
            # Use a consistent seeding strategy based on the *current* state
            current_num_particles = num_particles_values[p_idx] if current_filter_type == "PF" else 0
            seed = N * 10000 + K * 1000 + current_num_particles * 10 + rep # Unique seed
            run_params = {
                'N': N, 'K': K, 'filter_type': current_filter_type,
                'num_particles': num_particles_values[p_idx] if current_filter_type == "PF" else None,
                'rep': rep, 'seed': seed
            }

            # --- Execute Single Replicate ---
            success = execute_single_replicate(
                config=config,
                run_params=run_params,
                study_path=study_path,
                bf_instance=bf_instance,
                pf_instance=pf_instance
            )
            # Note: execute_single_replicate handles skipping if already done and error logging/saving

            # --- Advance Replicate Counter ---
            rep += 1

        # --- Advance to Next State (Particle Count, Filter Type, K, N) ---
        rep = 0 # Reset replicate counter for the next state
        if current_filter_type == "BF":
            if num_particles_values: # Check if PF runs are configured
                current_filter_type = "PF"
                p_idx = 0
            else: # No PF runs, move directly to next K
                k_idx += 1
                # current_filter_type remains "BF" (reset happens at start of K loop or N loop)
        elif current_filter_type == "PF":
            p_idx += 1
            if p_idx >= len(num_particles_values): # Finished all particle counts for this K
                k_idx += 1
                current_filter_type = "BF" # Reset for next K
                p_idx = 0

    # --- Study Complete ---
    print(f"\nSimulation study complete. Results saved in: {study_path}")
    # --- Clean up checkpoint on successful completion ---
    try:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"Successfully deleted checkpoint file: {checkpoint_path}")
    except Exception as e:
        print(f"Warning: Could not delete checkpoint file {checkpoint_path}: {e}")


if __name__ == "__main__":
    main()
