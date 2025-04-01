#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Check if we're in the correct environment
import sys
import os
import subprocess
from datetime import datetime
from pathlib import Path # Added import
import json # Added import


import numpy as np
import jax.numpy as jnp # Add JAX numpy import
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import sys

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
    Q_h = 1.0 * (Q_h_raw @ Q_h_raw.T) + np.eye(K) * 1e-4 # Ensure positive definite (Increased scaling factor)

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


def run_single_simulation(N, K, T, num_particles, seed, filter_type='both'):
    """
    Runs a single simulation instance for given parameters.

    Args:
        N (int): Number of assets.
        K (int): Number of factors.
        T (int): Time series length.
        num_particles (int): Number of particles for the Particle Filter (only used if filter_type='pf').
        seed (int): Random seed.
        filter_type (str): Which filter to run ('bf', 'pf', or 'both').

    Returns:
        tuple: (metrics_dict, raw_data_dict)
               metrics_dict contains configuration and scalar results.
               raw_data_dict contains the raw time series arrays.
    """
    print(f"Running simulation: N={N}, K={K}, T={T}, Filter={filter_type}, Particles={num_particles if filter_type == 'pf' else 'N/A'}, Seed={seed}")
    # Initialize return structures
    metrics = {
        'N': N, 'K': K, 'T': T, 'num_particles': num_particles if filter_type == 'pf' else None, 'seed': seed,
        'bf_time': None, 'pf_time': None,
        'bf_rmse_f': None, 'bf_corr_f': None, 'bf_rmse_h': None, 'bf_corr_h': None,
        'pf_rmse_f': None, 'pf_corr_f': None, 'pf_rmse_h': None, 'pf_corr_h': None,
        'error': None
    }
    raw_data = {
        'returns': None, 'true_factors': None, 'true_log_vols': None,
        'filtered_factors_bf': None, 'filtered_log_vols_bf': None,
        'filtered_factors_pf': None, 'filtered_log_vols_pf': None
    }
    # Initialize filtered data placeholders to None
    filtered_factors_bf = None
    filtered_log_vols_bf = None
    filtered_factors_pf = None
    filtered_log_vols_pf = None

    try:
        # 1. Create parameters and simulate data
        params = create_sim_parameters(N, K, seed=seed)
        returns, true_factors, true_log_vols = simulate_DFSV(params=params, T=T, seed=seed+1)
        raw_data['returns'] = returns
        raw_data['true_factors'] = true_factors
        raw_data['true_log_vols'] = true_log_vols

        # 2. Run Bellman Filter if needed
        if filter_type in ['bf', 'both']:
            bf = DFSVBellmanFilter(N, K)
            start_time_bf = time.time()
            # Assuming filter_scan returns filtered state and cov, or filter object has methods
            # Let's assume the filter object stores results internally for now
            bf.filter_scan(params, returns)
            end_time_bf = time.time()
            metrics['bf_time'] = end_time_bf - start_time_bf
            # Check if methods exist before calling
            if hasattr(bf, 'get_filtered_factors') and hasattr(bf, 'get_filtered_volatilities'):
                filtered_factors_bf = bf.get_filtered_factors()
                filtered_log_vols_bf = bf.get_filtered_volatilities()
                raw_data['filtered_factors_bf'] = filtered_factors_bf
                raw_data['filtered_log_vols_bf'] = filtered_log_vols_bf
                metrics['bf_rmse_f'], metrics['bf_corr_f'] = calculate_accuracy(true_factors, filtered_factors_bf)
                metrics['bf_rmse_h'], metrics['bf_corr_h'] = calculate_accuracy(true_log_vols, filtered_log_vols_bf)
            else:
                 print("Warning: Bellman filter object missing get_filtered_factors or get_filtered_volatilities method.")


        # 3. Run Particle Filter if needed
        if filter_type in ['pf', 'both']:
            # Ensure num_particles is valid
            if num_particles is None or num_particles <= 0:
                 raise ValueError("num_particles must be a positive integer for Particle Filter.")
            pf = DFSVParticleFilter(params, num_particles=num_particles)
            start_time_pf = time.time()
            # Assuming filter method returns results or stores them
            # Let's assume filter method returns tuple: (filtered_states, filtered_covs, log_likelihood)
            # And the object has methods to get factors/vols separately
            _, _, _ = pf.filter(observations=returns) # Use observations keyword arg
            end_time_pf = time.time()
            metrics['pf_time'] = end_time_pf - start_time_pf
            # Check if methods exist before calling
            if hasattr(pf, 'get_filtered_factors') and hasattr(pf, 'get_filtered_volatilities'):
                filtered_factors_pf = pf.get_filtered_factors()
                filtered_log_vols_pf = pf.get_filtered_volatilities()
                raw_data['filtered_factors_pf'] = filtered_factors_pf
                raw_data['filtered_log_vols_pf'] = filtered_log_vols_pf
                metrics['pf_rmse_f'], metrics['pf_corr_f'] = calculate_accuracy(true_factors, filtered_factors_pf)
                metrics['pf_rmse_h'], metrics['pf_corr_h'] = calculate_accuracy(true_log_vols, filtered_log_vols_pf)
            else:
                 print("Warning: Particle filter object missing get_filtered_factors or get_filtered_volatilities method.")


        print(f"Finished simulation: N={N}, K={K}, Filter={filter_type}, Seed={seed}. "
              f"{f'BF Time: {metrics["bf_time"]:.2f}s' if metrics["bf_time"] is not None else ''}"
              f"{f', PF Time: {metrics["pf_time"]:.2f}s' if metrics["pf_time"] is not None else ''}")

    except Exception as e:
        print(f"Error during simulation N={N}, K={K}, Filter={filter_type}, Seed={seed}: {e}")
        metrics['error'] = str(e)

    # Return both metrics and raw data
    return metrics, raw_data

def main():
    """Main function to run the simulation study."""

    # --- Configuration ---
    SIMULATION_CONFIG = {
        "N_values": [5, 10], # Reduced for faster testing initially
        "K_values": [2, 3],   # Reduced for faster testing initially
        "T": 500,             # Reduced for faster testing initially
        "num_particles_values": [1000], # Reduced for faster testing initially
        "num_reps": 2,        # Reduced for faster testing initially
        "base_results_dir": "simulation_results_raw",
        "save_format": "npz", # Options: "npz", "parquet" (requires pyarrow)
    }

    # --- Setup Output Directory ---
    base_results_path = Path(SIMULATION_CONFIG["base_results_dir"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir_name = f"study_{timestamp}"
    study_path = base_results_path / study_dir_name
    study_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {study_path}")

    # Save config to study directory
    config_save_path = study_path / "simulation_config.json"
    with open(config_save_path, 'w') as f:
        # Convert Path object to string for JSON serialization
        config_to_save = SIMULATION_CONFIG.copy()
        config_to_save["base_results_dir"] = str(config_to_save["base_results_dir"])
        json.dump(config_to_save, f, indent=4)
    print(f"Saved configuration to {config_save_path}")

    # --- Simulation Loop ---
    total_sims = len(SIMULATION_CONFIG["N_values"]) * len(SIMULATION_CONFIG["K_values"]) * (1 + len(SIMULATION_CONFIG["num_particles_values"])) * SIMULATION_CONFIG["num_reps"]
    current_sim = 0

    for N in SIMULATION_CONFIG["N_values"]:
        for K in SIMULATION_CONFIG["K_values"]:
            if K > N: # Skip cases where K > N if not meaningful for the model
                print(f"Skipping N={N}, K={K} as K > N")
                total_sims -= (1 + len(SIMULATION_CONFIG["num_particles_values"])) * SIMULATION_CONFIG["num_reps"] # Adjust total count
                continue

            # Run Bellman Filter once for each N,K configuration
            for rep in range(SIMULATION_CONFIG["num_reps"]):
                current_sim += 1
                seed = N * 1000 + K * 100 + rep
                run_label = f"config_N{N}_K{K}_BF_rep{rep}"
                run_path = study_path / run_label
                run_path.mkdir(exist_ok=True)

                print(f"\n--- Starting Bellman Filter Simulation {current_sim}/{total_sims} ({run_label}) ---")
                metrics, raw_data = run_single_simulation(N, K, SIMULATION_CONFIG["T"], None, seed, filter_type='bf')

                # Save results for this run
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
                    # Filter out None values before saving
                    data_to_save = {k: v for k, v in raw_data.items() if v is not None}
                    np.savez_compressed(raw_data_path, **data_to_save)
                # Add elif for parquet later if needed
                # elif SIMULATION_CONFIG["save_format"] == "parquet":
                #     try:
                #         import pyarrow as pa
                #         import pyarrow.parquet as pq
                #         # Convert arrays to pandas DFs or directly to pyarrow tables
                #         # Example: pd.DataFrame(raw_data['returns']).to_parquet(run_path / 'returns.parquet')
                #         print("Parquet saving not fully implemented yet.")
                #     except ImportError:
                #         print("Error: pyarrow is required to save in parquet format.")

                print(f"Saved results for {run_label} to {run_path}")


            # Run Particle Filter with different particle counts
            for num_particles in SIMULATION_CONFIG["num_particles_values"]:
                for rep in range(SIMULATION_CONFIG["num_reps"]):
                    current_sim += 1
                    seed = N * 1000 + K * 100 + num_particles + rep
                    run_label = f"config_N{N}_K{K}_PF{num_particles}_rep{rep}"
                    run_path = study_path / run_label
                    run_path.mkdir(exist_ok=True)

                    print(f"\n--- Starting Particle Filter Simulation {current_sim}/{total_sims} ({run_label}) ---")
                    metrics, raw_data = run_single_simulation(N, K, SIMULATION_CONFIG["T"], num_particles, seed, filter_type='pf')

                    # Save results for this run
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
                        # Filter out None values before saving
                        data_to_save = {k: v for k, v in raw_data.items() if v is not None}
                        np.savez_compressed(raw_data_path, **data_to_save)
                    # Add elif for parquet later if needed
                    # elif SIMULATION_CONFIG["save_format"] == "parquet":
                    #     # See comment above
                    #     print("Parquet saving not fully implemented yet.")

                    print(f"Saved results for {run_label} to {run_path}")

    print(f"\nSimulation study complete. Results saved in: {study_path}")


if __name__ == "__main__":
    main()
