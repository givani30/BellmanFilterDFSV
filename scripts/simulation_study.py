#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Check if we're in the correct environment
import sys
import argparse # Added import
from datetime import datetime
from pathlib import Path # Added import
import json # Added import


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



def main():
    parser = argparse.ArgumentParser(description='Run DFSV simulation study with optional resume.')
    parser.add_argument('--resume_study', type=str, help='Name of the study directory (e.g., study_YYYYMMDD_HHMMSS) to resume.')
    args = parser.parse_args()

    """Main function to run the simulation study."""

    # --- Configuration ---
    SIMULATION_CONFIG = {
        "N_values": [5, 10, 20,50,100,150], # Further expanded N range. Full set to be run is 5, 10, 20, 50, 100, 150
        "K_values": [2, 3, 5, 10, 15],       # Further expanded K range
        "T": 1500,                           # Increased time series length
        "num_particles_values": [1000,5000,10000,20000], # Further expanded particle counts. 1000, 5000, 10000, 20000
        "num_reps": 100,                     # Significantly increased number of replicates 100 in full study
        "base_results_dir": "simulation_results_raw",
        "save_format": "npz", # Options: "npz", "parquet" (requires pyarrow)
    }

    # --- Setup Output Directory & Load Config/Checkpoint ---
    base_results_path = Path(SIMULATION_CONFIG["base_results_dir"])
    start_indices = {"N": 0, "K": 0, "filter_type": "BF", "particles": 0, "rep": 0} # Default start indices

    if args.resume_study:
        study_dir_name = args.resume_study
        study_path = base_results_path / study_dir_name
        print(f"Attempting to resume study: {study_path}")
        if not study_path.is_dir():
            print(f"Error: Resume directory not found: {study_path}")
            sys.exit(1)

        # Load config from the specific study directory
        config_load_path = study_path / "simulation_config.json"
        if not config_load_path.exists():
             print(f"Error: simulation_config.json not found in resume directory: {config_load_path}")
             sys.exit(1)
        try:
            with open(config_load_path, 'r') as f:
                SIMULATION_CONFIG = json.load(f)
                # Ensure base_results_dir is Path object if needed later, though it's mainly for setup
                SIMULATION_CONFIG["base_results_dir"] = Path(SIMULATION_CONFIG["base_results_dir"])
            print(f"Loaded configuration from {config_load_path}")
        except Exception as e:
            print(f"Error loading configuration from {config_load_path}: {e}")
            sys.exit(1)

        # Try to load checkpoint
        checkpoint_path = study_path / "checkpoint.json"
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    loaded_indices = json.load(f)
                    # Basic validation (can be expanded)
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
                config_to_save = SIMULATION_CONFIG.copy()
                # Store base_results_dir as string in JSON
                config_to_save["base_results_dir"] = str(config_to_save["base_results_dir"])
                json.dump(config_to_save, f, indent=4)
            print(f"Saved configuration to {config_save_path}")
        except Exception as e:
            print(f"Error saving configuration to {config_save_path}: {e}")
            sys.exit(1)
        # Define checkpoint path for the new study
        checkpoint_path = study_path / "checkpoint.json"


    # --- Simulation Loop ---
    # Define checkpoint path (ensure it's defined in both resume/new cases)
    checkpoint_path = study_path / "checkpoint.json"
    T = SIMULATION_CONFIG["T"] # Get T once
    # Note: total_sims calculation might be slightly off if resuming, but it's just for progress display
    total_sims = len(SIMULATION_CONFIG["N_values"]) * len(SIMULATION_CONFIG["K_values"]) * (1 + len(SIMULATION_CONFIG["num_particles_values"])) * SIMULATION_CONFIG["num_reps"]
    current_sim = 0 # This counter also might be off when resuming, consider removing or adjusting if precise count needed

    # Get config lists once
    N_values = SIMULATION_CONFIG["N_values"]
    K_values = SIMULATION_CONFIG["K_values"]
    num_particles_values = SIMULATION_CONFIG["num_particles_values"]
    num_reps = SIMULATION_CONFIG["num_reps"]

    # --- Outer Loops with Resume Logic ---
    for n_idx, N in enumerate(N_values):
        if n_idx < start_indices["N"]:
            continue # Skip N values before the checkpoint

        for k_idx, K in enumerate(K_values):
            # Skip K values before the checkpoint only if we are at the starting N index
            if n_idx == start_indices["N"] and k_idx < start_indices["K"]:
                continue # Skip K values

            if K > N: # Skip cases where K > N
                print(f"Skipping N={N}, K={K} as K > N")
                continue

            print(f"\n=== Processing Configuration N={N}, K={K} ===")

            # --- Bellman Filter Runs ---
            run_bf = True
            # Skip BF runs if checkpoint starts at PF for this N/K
            if n_idx == start_indices["N"] and k_idx == start_indices["K"] and start_indices["filter_type"] == "PF":
                 print(f"Checkpoint starts at PF for N={N}, K={K}. Skipping BF runs.")
                 run_bf = False

            if run_bf:
                print(f"  Instantiating Bellman Filter (N={N}, K={K})...")
                try:
                    bf_instance = DFSVBellmanFilter(N, K)
                except Exception as e:
                    print(f"  Error instantiating Bellman Filter for N={N}, K={K}: {e}. Skipping BF runs for this config.")
                    continue # Skip to next K or N

                print(f"  Running {num_reps} replicates for Bellman Filter...")
                start_rep_bf = 0
                # Adjust starting replicate if resuming within BF for this N/K
                if n_idx == start_indices["N"] and k_idx == start_indices["K"] and start_indices["filter_type"] == "BF":
                    start_rep_bf = start_indices["rep"]

                for rep in range(start_rep_bf, num_reps):
                    # --- Checkpoint Update (Before starting work) ---
                    next_n_idx, next_k_idx, next_filter_type, next_particles_idx, next_rep = n_idx, k_idx, "BF", 0, rep + 1

                    if next_rep >= num_reps: # Finished all BF reps for this N/K
                        if num_particles_values: # Check if there are any PF runs configured
                            next_filter_type = "PF"
                            next_particles_idx = 0
                            next_rep = 0
                        else: # No PF runs, move to next K/N directly
                            next_filter_type = "BF"
                            next_rep = 0
                            next_k_idx += 1
                            if next_k_idx >= len(K_values):
                                next_k_idx = 0
                                next_n_idx += 1
                                # We don't need to check K>N here, outer loop handles it

                    # Prepare checkpoint data (only save if not finished all N)
                    if next_n_idx < len(N_values):
                        next_indices = {
                            "N": next_n_idx,
                            "K": next_k_idx,
                            "filter_type": next_filter_type,
                            "particles": next_particles_idx,
                            "rep": next_rep
                        }
                        save_checkpoint(checkpoint_path, next_indices)
                    # Else: If next_n_idx is out of bounds, we are done, checkpoint will be deleted later

                    # current_sim += 1 # Counter adjustment needed if resuming
                    seed = N * 1000 + K * 100 + rep # Unique seed per replicate
                    run_label = f"config_N{N}_K{K}_BF_rep{rep}"
                    run_path = study_path / run_label
                    metrics_path = run_path / "metrics.json" # Define metrics path

                    # Check if already completed
                    if metrics_path.exists():
                        print(f"    Skipping completed BF Replicate {run_label}")
                        continue

                    # --- Checkpoint Update (Before starting work) ---
                    # Calculate next state and save checkpoint HERE
                    # (Implementation deferred to next step)

                    # --- Start Actual Work ---
                    run_path.mkdir(exist_ok=True) # Create dir only if not skipping
                    print(f"    --- Starting BF Replicate {rep+1}/{num_reps} ({run_label}) ---")

                    # 1. Create parameters and simulate data for this replicate
                    try:
                        params_rep = create_sim_parameters(N, K, seed=seed)
                    except Exception as e:
                         print(f"      Error creating parameters for {run_label}: {e}. Skipping replicate.")
                         # Optionally log error to a file or metrics
                         continue # Skip to next replicate
                    # <<< Correctly indented block starts here >>>
                    returns_rep, true_factors_rep, true_log_vols_rep = simulate_DFSV(params=params_rep, T=T, seed=seed+1)

                    # 2. Run filtering using the single BF instance
                    metrics, raw_data_out = run_single_simulation(
                        N=N, K=K, T=T, seed=seed,
                        params=params_rep,
                        returns=returns_rep,
                        true_factors=true_factors_rep,
                        true_log_vols=true_log_vols_rep,
                        bf_instance=bf_instance, # Pass the instance
                        pf_instance=None,
                        num_particles=None
                    )

                    # 3. Check for errors during simulation before saving
                    if metrics.get('error'):
                        print(f"      --- Skipping save for {run_label} due to error during simulation: {metrics['error']} ---")
                        # Optionally save just the error metric
                        try:
                            metrics_path = run_path / "metrics_error.json"
                            serializable_metrics = {'error': metrics['error'], 'N': N, 'K': K, 'T': T, 'seed': seed, 'filter_type': 'BF'}
                            with open(metrics_path, 'w') as f:
                                json.dump(serializable_metrics, f, indent=4)
                            print(f"      Saved error details to {metrics_path}")
                        except Exception as save_err:
                            print(f"      !!!!!! FAILED TO SAVE ERROR DETAILS for {run_label}: {save_err}")
                    else:
                        # 4. Save results for this run (only if no error occurred during simulation)
                        try: # Add try block for saving
                            metrics_path = run_path / "metrics.json"
                            # Convert numpy arrays in metrics to lists for JSON serialization
                            serializable_metrics = {}
                            for key, value in metrics.items():
                                if isinstance(value, np.ndarray):
                                     # Handle potential NaN values before converting to list
                                     if np.isnan(value).any():
                                          serializable_metrics[key] = [float('nan') if np.isnan(v) else v for v in value]
                                     else:
                                          serializable_metrics[key] = value.tolist()
                                elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                                     serializable_metrics[key] = int(value) # Convert numpy int types
                                elif isinstance(value, (np.floating, np.float16, np.float32, np.float64)): # Use np.floating
                                     serializable_metrics[key] = float(value) # Convert numpy float types
                                elif isinstance(value, (np.bool_)):
                                     serializable_metrics[key] = bool(value) # Convert numpy bool types
                                else:
                                     serializable_metrics[key] = value
                            with open(metrics_path, 'w') as f:
                                json.dump(serializable_metrics, f, indent=4)

                            if SIMULATION_CONFIG["save_format"] == "npz":
                                raw_data_path = run_path / "raw_data.npz"
                                # Filter out None values before saving (use raw_data_out)
                                data_to_save = {k: v for k, v in raw_data_out.items() if v is not None}
                                if data_to_save: # Only save if there's actually data
                                    np.savez_compressed(raw_data_path, **data_to_save)
                                else:
                                    print(f"      Warning: No raw data to save for {run_label} (filtered results might be missing).")

                            # Add elif for parquet later if needed
                            # elif SIMULATION_CONFIG["save_format"] == "parquet":
                            #     # ... (parquet logic) ...

                            print(f"    Saved results for {run_label} to {run_path}")

                        except Exception as e:
                            print(f"    !!!!!! ERROR SAVING RESULTS for {run_label} to {run_path}: {e}")
                            # Attempt to remove potentially corrupted files if save failed mid-way
                            if 'metrics_path' in locals() and metrics_path.exists():
                                try: metrics_path.unlink()
                                except OSError: pass
                            if 'raw_data_path' in locals() and raw_data_path.exists():
                                try: raw_data_path.unlink()
                                except OSError: pass
                            # Continue to the next replicate instead of crashing
                    # <<< End of correctly indented block >>>


            # --- Particle Filter Runs ---
            for p_idx, num_particles in enumerate(num_particles_values):
                 # Skip particle counts before the checkpoint only if we are at the starting N/K and filter is PF
                if n_idx == start_indices["N"] and k_idx == start_indices["K"] and start_indices["filter_type"] == "PF" and p_idx < start_indices["particles"]:
                    continue # Skip particle counts

                # Instantiate PF once for this N, K, num_particles config
                pf_base_seed = N * 100 + K * 10 + num_particles # Consistent base seed
                print(f"  Instantiating Particle Filter (N={N}, K={K}, P={num_particles}, seed={pf_base_seed})...")
                try:
                    pf_instance = DFSVParticleFilter(N=N, K=K, num_particles=num_particles, seed=pf_base_seed)
                except Exception as e:
                    print(f"  Error instantiating Particle Filter for N={N}, K={K}, P={num_particles}: {e}. Skipping PF runs for this particle count.")
                    continue # Skip to next particle count

                print(f"  Running {num_reps} replicates for Particle Filter (P={num_particles})...")
                start_rep_pf = 0
                # Adjust starting replicate if resuming within PF for this N/K/Particles
                if n_idx == start_indices["N"] and k_idx == start_indices["K"] and start_indices["filter_type"] == "PF" and p_idx == start_indices["particles"]:
                    start_rep_pf = start_indices["rep"]

                for rep in range(start_rep_pf, num_reps):
                    # current_sim += 1 # Counter adjustment needed if resuming
                    # --- Checkpoint Update (Before starting work) ---
                    next_n_idx, next_k_idx, next_filter_type, next_particles_idx, next_rep = n_idx, k_idx, "PF", p_idx, rep + 1

                    if next_rep >= num_reps: # Finished all PF reps for this particle count
                        next_particles_idx += 1
                        next_rep = 0
                        if next_particles_idx >= len(num_particles_values): # Finished all particle counts
                            next_filter_type = "BF" # Go back to BF for next K/N
                            next_particles_idx = 0
                            next_rep = 0
                            next_k_idx += 1
                            if next_k_idx >= len(K_values):
                                next_k_idx = 0
                                next_n_idx += 1
                                # We don't need to check K>N here, outer loop handles it

                    # Prepare checkpoint data (only save if not finished all N)
                    if next_n_idx < len(N_values):
                        next_indices = {
                            "N": next_n_idx,
                            "K": next_k_idx,
                            "filter_type": next_filter_type,
                            "particles": next_particles_idx,
                            "rep": next_rep
                        }
                        save_checkpoint(checkpoint_path, next_indices)
                    # Else: If next_n_idx is out of bounds, we are done, checkpoint will be deleted later

                    # Unique seed per replicate, distinct from BF seeds and PF base seed
                    seed = N * 10000 + K * 1000 + num_particles * 10 + rep
                    run_label = f"config_N{N}_K{K}_PF{num_particles}_rep{rep}"
                    run_path = study_path / run_label
                    metrics_path = run_path / "metrics.json" # Define metrics path

                    # Check if already completed
                    if metrics_path.exists():
                        print(f"    Skipping completed PF Replicate {run_label}")
                        continue

                    # --- Checkpoint Update (Before starting work) ---
                    # Calculate next state and save checkpoint HERE
                    # (Implementation deferred to next step)

                    # --- Start Actual Work ---
                    run_path.mkdir(exist_ok=True) # Create dir only if not skipping
                    print(f"    --- Starting PF Replicate {rep+1}/{num_reps} ({run_label}) ---")

                    # 1. Create parameters and simulate data for this replicate
                    try:
                        params_rep = create_sim_parameters(N, K, seed=seed)
                    except Exception as e:
                         print(f"      Error creating parameters for {run_label}: {e}. Skipping replicate.")
                         # Optionally log error to a file or metrics
                         continue # Skip to next replicate
                    returns_rep, true_factors_rep, true_log_vols_rep = simulate_DFSV(params=params_rep, T=T, seed=seed+1)

                    # 2. Run filtering using the single PF instance
                    metrics, raw_data_out = run_single_simulation(
                        N=N, K=K, T=T, seed=seed,
                        params=params_rep,
                        returns=returns_rep,
                        true_factors=true_factors_rep,
                        true_log_vols=true_log_vols_rep,
                        bf_instance=None,
                        pf_instance=pf_instance, # Pass the instance
                        num_particles=num_particles # Pass num_particles for metrics
                    )

                    # 3. Check for errors during simulation before saving
                    if metrics.get('error'):
                        print(f"      --- Skipping save for {run_label} due to error during simulation: {metrics['error']} ---")
                        # Optionally save just the error metric
                        try:
                            metrics_path = run_path / "metrics_error.json"
                            serializable_metrics = {'error': metrics['error'], 'N': N, 'K': K, 'T': T, 'seed': seed, 'filter_type': 'PF', 'num_particles': num_particles}
                            with open(metrics_path, 'w') as f:
                                json.dump(serializable_metrics, f, indent=4)
                            print(f"      Saved error details to {metrics_path}")
                        except Exception as save_err:
                            print(f"      !!!!!! FAILED TO SAVE ERROR DETAILS for {run_label}: {save_err}")
                    else:
                        # 4. Save results for this run (only if no error occurred during simulation)
                        try: # Add try block for saving
                            metrics_path = run_path / "metrics.json"
                            # Convert numpy arrays in metrics to lists for JSON serialization
                            serializable_metrics = {}
                            for key, value in metrics.items():
                                 if isinstance(value, np.ndarray):
                                      # Handle potential NaN values before converting to list
                                      if np.isnan(value).any():
                                           serializable_metrics[key] = [float('nan') if np.isnan(v) else v for v in value]
                                      else:
                                           serializable_metrics[key] = value.tolist()
                                 elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                                      serializable_metrics[key] = int(value) # Convert numpy int types
                                 elif isinstance(value, (np.floating, np.float16, np.float32, np.float64)): # Use np.floating
                                      serializable_metrics[key] = float(value) # Convert numpy float types
                                 elif isinstance(value, (np.bool_)):
                                      serializable_metrics[key] = bool(value) # Convert numpy bool types
                                 else:
                                      serializable_metrics[key] = value
                            with open(metrics_path, 'w') as f:
                                json.dump(serializable_metrics, f, indent=4)

                            if SIMULATION_CONFIG["save_format"] == "npz":
                                raw_data_path = run_path / "raw_data.npz"
                                # Filter out None values before saving (use raw_data_out)
                                data_to_save = {k: v for k, v in raw_data_out.items() if v is not None}
                                if data_to_save: # Only save if there's actually data
                                    np.savez_compressed(raw_data_path, **data_to_save)
                                else:
                                     print(f"      Warning: No raw data to save for {run_label} (filtered results might be missing).")
                            # Add elif for parquet later if needed
                            # elif SIMULATION_CONFIG["save_format"] == "parquet":
                            #     # See comment above
                            #     print("Parquet saving not fully implemented yet.")

                            print(f"    Saved results for {run_label} to {run_path}")

                        except Exception as e:
                            print(f"    !!!!!! ERROR SAVING RESULTS for {run_label} to {run_path}: {e}")
                            # Attempt to remove potentially corrupted files if save failed mid-way
                            if 'metrics_path' in locals() and metrics_path.exists():
                                try: metrics_path.unlink()
                                except OSError: pass
                            if 'raw_data_path' in locals() and raw_data_path.exists():
                                try: raw_data_path.unlink()
                                except OSError: pass
                            # Continue to the next replicate instead of crashing
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
