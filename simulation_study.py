#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Check if we're in the correct environment
import sys
import os
import subprocess
from datetime import datetime # Added import


import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dfsv import DFSV_params
from functions.simulation import simulate_DFSV
from functions.bellman_filter import DFSVBellmanFilter
from functions.filters import DFSVParticleFilter # Assuming Particle Filter is in filters.py

def create_sim_parameters(N, K, seed=None):
    """
    Generates valid DFSV_params for given N and K, ensuring stationarity.

    Args:
        N (int): Number of assets.
        K (int): Number of factors.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        DFSV_params: A valid DFSV_params object.
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
    sigma2 = np.exp(np.random.normal(-1, 0.5, size=N)) # Use diagonal matrix or vector

    # Volatility covariance matrix (ensure positive definite)
    Q_h_raw = np.random.normal(0, 0.2, size=(K, K))
    Q_h = 0.1 * (Q_h_raw @ Q_h_raw.T) + np.eye(K) * 1e-4 # Ensure positive definite

    params = DFSV_params(
        N=N,
        K=K,
        lambda_r=lambda_r,
        Phi_f=Phi_f,
        Phi_h=Phi_h,
        mu=mu,
        sigma2=sigma2, # Pass as vector or diag matrix based on DFSV_params implementation
        Q_h=Q_h,
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
        corr = np.corrcoef(true_values[:, k], estimated_values[:, k])[0, 1]
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
        dict: Dictionary containing configuration and results.
    """
    print(f"Running simulation: N={N}, K={K}, T={T}, Filter={filter_type}, Particles={num_particles if filter_type == 'pf' else 'N/A'}, Seed={seed}")
    results = {
        'N': N, 'K': K, 'T': T, 'num_particles': num_particles if filter_type == 'pf' else None, 'seed': seed,
        'bf_time': None, 'pf_time': None,
        'bf_rmse_f': None, 'bf_corr_f': None, 'bf_rmse_h': None, 'bf_corr_h': None,
        'pf_rmse_f': None, 'pf_corr_f': None, 'pf_rmse_h': None, 'pf_corr_h': None,
        'error': None
    }

    try:
        # 1. Create parameters and simulate data
        params = create_sim_parameters(N, K, seed=seed)
        returns, true_factors, true_log_vols = simulate_DFSV(params=params, T=T, seed=seed+1)

        # 2. Run Bellman Filter if needed
        if filter_type in ['bf', 'both']:
            bf = DFSVBellmanFilter(N, K)
            start_time_bf = time.time()
            bf.filter_scan(params, returns)
            end_time_bf = time.time()
            results['bf_time'] = end_time_bf - start_time_bf
            filtered_factors_bf = bf.get_filtered_factors()
            filtered_log_vols_bf = bf.get_filtered_volatilities()
            results['bf_rmse_f'], results['bf_corr_f'] = calculate_accuracy(true_factors, filtered_factors_bf)
            results['bf_rmse_h'], results['bf_corr_h'] = calculate_accuracy(true_log_vols, filtered_log_vols_bf)

        # 3. Run Particle Filter if needed
        if filter_type in ['pf', 'both']:
            pf = DFSVParticleFilter(params, num_particles=num_particles)
            start_time_pf = time.time()
            pf.filter(params, returns)
            end_time_pf = time.time()
            results['pf_time'] = end_time_pf - start_time_pf
            filtered_factors_pf = pf.get_filtered_factors()
            filtered_log_vols_pf = pf.get_filtered_volatilities()
            results['pf_rmse_f'], results['pf_corr_f'] = calculate_accuracy(true_factors, filtered_factors_pf)
            results['pf_rmse_h'], results['pf_corr_h'] = calculate_accuracy(true_log_vols, filtered_log_vols_pf)

        print(f"Finished simulation: N={N}, K={K}, Filter={filter_type}, Seed={seed}. "
              f"{f'BF Time: {results["bf_time"]:.2f}s' if results["bf_time"] is not None else ''}"
              f"{f', PF Time: {results["pf_time"]:.2f}s' if results["pf_time"] is not None else ''}")

    except Exception as e:
        print(f"Error during simulation N={N}, K={K}, Filter={filter_type}, Seed={seed}: {e}")
        results['error'] = str(e)

    return results

def main():
    """Main function to run the simulation study."""
    # --- Simulation Configuration ---
    N_values = [ 5, 10, 50]  # Example values for N
    K_values = [2, 3, 5,10]   # Example values for K
    T = 1000               # Time series length
    num_particles_values = [1000, 10000]  # Number of particles for PF
    num_reps = 3          # Number of repetitions for each configuration
    # --- Output Configuration --- # Added section
    results_dir = "simulation_results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_file = os.path.join(results_dir, f"{timestamp}_simulation_results.csv")
    output_plot_file = os.path.join(results_dir, f"{timestamp}_simulation_summary_plots.html")

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    # --------------------------------

    all_results = []
    total_sims = len(N_values) * len(K_values) * (1 + len(num_particles_values)) * num_reps
    current_sim = 0

    for N in N_values:
        for K in K_values:
            if K > N: # Skip cases where K > N if not meaningful for the model
                print(f"Skipping N={N}, K={K} as K > N")
                total_sims -= (1 + len(num_particles_values)) * num_reps # Adjust total count
                continue
            
            # Run Bellman Filter once for each N,K configuration
            for rep in range(num_reps):
                current_sim += 1
                seed = N * 1000 + K * 100 + rep
                print(f"\n--- Starting Bellman Filter Simulation {current_sim}/{total_sims} ---")
                print(f"Configuration: N={N}, K={K}, Rep={rep+1}")
                sim_result = run_single_simulation(N, K, T, None, seed, filter_type='bf')
                all_results.append(sim_result)

            # Run Particle Filter with different particle counts
            for num_particles in num_particles_values:
                for rep in range(num_reps):
                    current_sim += 1
                    seed = N * 1000 + K * 100 + num_particles + rep
                    print(f"\n--- Starting Particle Filter Simulation {current_sim}/{total_sims} ---")
                    print(f"Configuration: N={N}, K={K}, Particles={num_particles}, Rep={rep+1}")
                    sim_result = run_single_simulation(N, K, T, num_particles, seed, filter_type='pf')
                    all_results.append(sim_result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Process array results (RMSE, Corr) for easier analysis - e.g., average over factors
    for filt in ['bf', 'pf']:
        for metric in ['rmse', 'corr']:
            for state in ['f', 'h']:
                col_name = f'{filt}_{metric}_{state}'
                # Calculate mean across factors/volatilities for summary
                results_df[f'{col_name}_mean'] = results_df[col_name].apply(lambda x: np.mean(x) if isinstance(x, np.ndarray) else x)
                # Keep the original array too if needed, or drop it
                # results_df = results_df.drop(columns=[col_name])


    # Save results
    results_df.to_csv(output_csv_file, index=False)
    print(f"\nSimulation study complete. Results saved to {output_csv_file}")

    # --- Plotting with Plotly ---
    if not results_df.empty:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.io as pio

            # Aggregate results across replications
            agg_results = results_df.groupby(['N', 'K', 'num_particles']).agg({
                'bf_time': 'mean',
                'pf_time': 'mean',
                'bf_corr_f_mean': 'mean',
                'pf_corr_f_mean': 'mean',
                'bf_corr_h_mean': 'mean',
                'pf_corr_h_mean': 'mean'
            }).reset_index()

            # Set the default template to a clean, modern style
            pio.templates.default = "plotly_white"

            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Computation Time vs K',
                    'Factor Estimation Accuracy',
                    'Log-Volatility Estimation Accuracy',
                    'Computation Time vs N'
                )
            )

            # Time vs K for different N
            for n_val in agg_results['N'].unique():
                # Bellman Filter
                bf_subset = agg_results[(agg_results['N'] == n_val) & (agg_results['num_particles'].isna())]
                fig.add_trace(
                    go.Scatter(
                        x=bf_subset['K'],
                        y=bf_subset['bf_time'],
                        name=f'BF (N={n_val})',
                        mode='lines+markers',
                        line=dict(width=2),
                        marker=dict(size=8)
                    ),
                    row=1, col=1
                )

                # Particle Filter with different particle counts
                for num_particles in num_particles_values:
                    pf_subset = agg_results[(agg_results['N'] == n_val) & (agg_results['num_particles'] == num_particles)]
                    fig.add_trace(
                        go.Scatter(
                            x=pf_subset['K'],
                            y=pf_subset['pf_time'],
                            name=f'PF (N={n_val}, {num_particles} particles)',
                            mode='lines+markers',
                            line=dict(width=2, dash='dash'),
                            marker=dict(size=8)
                        ),
                        row=1, col=1
                    )

            # Factor Correlation vs K
            for n_val in agg_results['N'].unique():
                # Bellman Filter
                bf_subset = agg_results[(agg_results['N'] == n_val) & (agg_results['num_particles'].isna())]
                fig.add_trace(
                    go.Scatter(
                        x=bf_subset['K'],
                        y=bf_subset['bf_corr_f_mean'],
                        name=f'BF (N={n_val})',
                        mode='lines+markers',
                        line=dict(width=2),
                        marker=dict(size=8),
                        showlegend=False
                    ),
                    row=1, col=2
                )

                # Particle Filter with different particle counts
                for num_particles in num_particles_values:
                    pf_subset = agg_results[(agg_results['N'] == n_val) & (agg_results['num_particles'] == num_particles)]
                    fig.add_trace(
                        go.Scatter(
                            x=pf_subset['K'],
                            y=pf_subset['pf_corr_f_mean'],
                            name=f'PF (N={n_val}, {num_particles} particles)',
                            mode='lines+markers',
                            line=dict(width=2, dash='dash'),
                            marker=dict(size=8),
                            showlegend=False
                        ),
                        row=1, col=2
                    )

            # Log-Volatility Correlation vs K
            for n_val in agg_results['N'].unique():
                # Bellman Filter
                bf_subset = agg_results[(agg_results['N'] == n_val) & (agg_results['num_particles'].isna())]
                fig.add_trace(
                    go.Scatter(
                        x=bf_subset['K'],
                        y=bf_subset['bf_corr_h_mean'],
                        name=f'BF (N={n_val})',
                        mode='lines+markers',
                        line=dict(width=2),
                        marker=dict(size=8),
                        showlegend=False
                    ),
                    row=2, col=1
                )

                # Particle Filter with different particle counts
                for num_particles in num_particles_values:
                    pf_subset = agg_results[(agg_results['N'] == n_val) & (agg_results['num_particles'] == num_particles)]
                    fig.add_trace(
                        go.Scatter(
                            x=pf_subset['K'],
                            y=pf_subset['pf_corr_h_mean'],
                            name=f'PF (N={n_val}, {num_particles} particles)',
                            mode='lines+markers',
                            line=dict(width=2, dash='dash'),
                            marker=dict(size=8),
                            showlegend=False
                        ),
                        row=2, col=1
                    )

            # Time vs N for different K
            for k_val in agg_results['K'].unique():
                # Bellman Filter
                bf_subset = agg_results[(agg_results['K'] == k_val) & (agg_results['num_particles'].isna())]
                fig.add_trace(
                    go.Scatter(
                        x=bf_subset['N'],
                        y=bf_subset['bf_time'],
                        name=f'BF (K={k_val})',
                        mode='lines+markers',
                        line=dict(width=2),
                        marker=dict(size=8),
                        showlegend=False
                    ),
                    row=2, col=2
                )

                # Particle Filter with different particle counts
                for num_particles in num_particles_values:
                    pf_subset = agg_results[(agg_results['K'] == k_val) & (agg_results['num_particles'] == num_particles)]
                    fig.add_trace(
                        go.Scatter(
                            x=pf_subset['N'],
                            y=pf_subset['pf_time'],
                            name=f'PF (K={k_val}, {num_particles} particles)',
                            mode='lines+markers',
                            line=dict(width=2, dash='dash'),
                            marker=dict(size=8),
                            showlegend=False
                        ),
                        row=2, col=2
                    )

            # Update layout
            fig.update_layout(
                height=1000,
                width=1200,
                title_text="Simulation Study Results (Averaged over Replications)",
                title_x=0.5,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05
                ),
                template="plotly_white"
            )

            # Update axes labels
            fig.update_xaxes(title_text="K (Number of Factors)", row=1, col=1)
            fig.update_xaxes(title_text="K (Number of Factors)", row=1, col=2)
            fig.update_xaxes(title_text="K (Number of Factors)", row=2, col=1)
            fig.update_xaxes(title_text="N (Number of Assets)", row=2, col=2)

            fig.update_yaxes(title_text="Average Computation Time (s)", row=1, col=1)
            fig.update_yaxes(title_text="Average Factor Correlation", row=1, col=2)
            fig.update_yaxes(title_text="Average Log-Volatility Correlation", row=2, col=1)
            fig.update_yaxes(title_text="Average Computation Time (s)", row=2, col=2)

            # Set y-axis ranges for correlation plots
            fig.update_yaxes(range=[0, 1], row=1, col=2)
            fig.update_yaxes(range=[0, 1], row=2, col=1)

            # Save the plot
            fig.write_html(output_plot_file)
            print(f"Summary plots saved to {output_plot_file}")

        except Exception as plot_err:
            print(f"Could not generate plots: {plot_err}")


if __name__ == "__main__":
    main()
