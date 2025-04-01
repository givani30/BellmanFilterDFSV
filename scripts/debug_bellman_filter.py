#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import jax.numpy as jnp
import sys
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Assuming the script is run from the project root
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.core.simulation import simulate_DFSV
from bellman_filter_dfsv.core.filters.bellman import DFSVBellmanFilter
# Import helper functions from simulation_study (or redefine if needed)
#go up one directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.simulation_study import create_sim_parameters, calculate_accuracy

def run_bellman_debug(N, K, T, seed):
    """
    Runs a single simulation and Bellman filter for debugging.

    Args:
        N (int): Number of assets.
        K (int): Number of factors.
        T (int): Time series length.
        seed (int): Random seed.

    Returns:
        tuple: (metrics, true_states, filtered_states)
               metrics: Dictionary with RMSE, Correlation, Time.
               true_states: Dictionary with 'factors', 'log_vols'.
               filtered_states: Dictionary with 'factors', 'log_vols'.
    """
    print(f"--- Debugging Bellman Filter ---")
    print(f"Configuration: N={N}, K={K}, T={T}, Seed={seed}")

    metrics = {
        'N': N, 'K': K, 'T': T, 'seed': seed,
        'bf_time': None,
        'bf_rmse_f': None, 'bf_corr_f': None,
        'bf_rmse_h': None, 'bf_corr_h': None,
        'error': None
    }
    true_states = {'factors': None, 'log_vols': None}
    filtered_states = {'factors': None, 'log_vols': None}

    try:
        # 1. Create parameters and simulate data
        params = create_sim_parameters(N, K, seed=seed)
        returns, true_factors, true_log_vols = simulate_DFSV(params=params, T=T, seed=seed+1)
        true_states['factors'] = true_factors
        true_states['log_vols'] = true_log_vols

        # 2. Run Bellman Filter
        bf = DFSVBellmanFilter(N, K)
        start_time_bf = time.time()
        bf.filter(params, returns) # Use non-scan version for debugging prints
        end_time_bf = time.time()
        metrics['bf_time'] = end_time_bf - start_time_bf

        # 3. Get filtered results
        if hasattr(bf, 'get_filtered_factors') and hasattr(bf, 'get_filtered_volatilities'):
            filtered_factors = bf.get_filtered_factors()
            filtered_log_vols = bf.get_filtered_volatilities()
            filtered_states['factors'] = filtered_factors
            filtered_states['log_vols'] = filtered_log_vols

            # 4. Calculate Accuracy
            metrics['bf_rmse_f'], metrics['bf_corr_f'] = calculate_accuracy(true_factors, filtered_factors)
            metrics['bf_rmse_h'], metrics['bf_corr_h'] = calculate_accuracy(true_log_vols, filtered_log_vols)
            print(f"Finished Filtering. Time: {metrics['bf_time']:.2f}s")
        else:
            metrics['error'] = "Filter object missing get_filtered_* methods."
            print(f"Error: {metrics['error']}")

    except Exception as e:
        print(f"Error during debug run: {e}")
        metrics['error'] = str(e)

    return metrics, true_states, filtered_states

def plot_debug_results(metrics, true_states, filtered_states, output_dir="outputs"):
    """Generates plots comparing true and filtered states."""
    N = metrics['N']
    K = metrics['K']
    T = metrics['T']
    seed = metrics['seed']

    true_f = true_states['factors']
    true_h = true_states['log_vols']
    filt_f = filtered_states['factors']
    filt_h = filtered_states['log_vols']

    if true_f is None or true_h is None or filt_f is None or filt_h is None:
        print("Skipping plotting due to missing data (likely filter error).")
        return

    num_plots = K * 2 # Factors + Log-Vols
    num_cols = 2
    num_rows = K

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 3), sharex=True)
    fig.suptitle(f'Bellman Filter Debug: N={N}, K={K}, T={T}, Seed={seed}', fontsize=16)

    time_axis = np.arange(T)

    for k in range(K):
        # Plot Factors
        ax_f = axes[k, 0] if num_rows > 1 else axes[0]
        ax_f.plot(time_axis, true_f[:, k], label='True', color='black', linestyle=':')
        ax_f.plot(time_axis, filt_f[:, k], label='Filtered (BF)', color='blue', alpha=0.8)
        ax_f.set_ylabel(f'Factor {k+1}')
        ax_f.legend()
        ax_f.grid(True, linestyle='--', alpha=0.6)
        rmse_f_k = metrics['bf_rmse_f'][k] if metrics['bf_rmse_f'] is not None else np.nan
        corr_f_k = metrics['bf_corr_f'][k] if metrics['bf_corr_f'] is not None else np.nan
        ax_f.set_title(f'Factor {k+1} (RMSE: {rmse_f_k:.3f}, Corr: {corr_f_k:.3f})')


        # Plot Log-Vols
        ax_h = axes[k, 1] if num_rows > 1 else axes[1]
        ax_h.plot(time_axis, true_h[:, k], label='True', color='black', linestyle=':')
        ax_h.plot(time_axis, filt_h[:, k], label='Filtered (BF)', color='red', alpha=0.8)
        ax_h.set_ylabel(f'Log-Vol {k+1}')
        ax_h.legend()
        ax_h.grid(True, linestyle='--', alpha=0.6)
        rmse_h_k = metrics['bf_rmse_h'][k] if metrics['bf_rmse_h'] is not None else np.nan
        corr_h_k = metrics['bf_corr_h'][k] if metrics['bf_corr_h'] is not None else np.nan
        ax_h.set_title(f'Log-Vol {k+1} (RMSE: {rmse_h_k:.3f}, Corr: {corr_h_k:.3f})')


    # Add X axis label to bottom plots
    if num_rows > 1:
        axes[num_rows-1, 0].set_xlabel('Time')
        axes[num_rows-1, 1].set_xlabel('Time')
    else:
         axes[0].set_xlabel('Time')
         axes[1].set_xlabel('Time')


    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap

    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_filename = output_path / f"debug_bellman_N{N}_K{K}_T{T}_seed{seed}.png"
    plt.savefig(plot_filename)
    print(f"Saved debug plot to {plot_filename}")
    plt.close(fig) # Close the figure to free memory


def main():
    # --- Configuration ---
    N = 5
    K = 2
    T = 200 # Shorter time series for quick debug
    SEED = 42
    OUTPUT_DIR = "outputs/debug_runs"

    # --- Run Debug ---
    metrics, true_states, filtered_states = run_bellman_debug(N, K, T, SEED)

    # --- Print Metrics ---
    print("\n--- Metrics ---")
    if metrics['error']:
        print(f"Error: {metrics['error']}")
    else:
        print(f"Time: {metrics['bf_time']:.3f} s")
        print(f"Factor RMSE: {metrics['bf_rmse_f']}")
        print(f"Factor Corr: {metrics['bf_corr_f']}")
        print(f"Log-Vol RMSE: {metrics['bf_rmse_h']}")
        print(f"Log-Vol Corr: {metrics['bf_corr_h']}") # This is the value of interest

    # --- Plot Results ---
    plot_debug_results(metrics, true_states, filtered_states, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()