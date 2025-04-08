"""Visual comparison of Bellman Information Filter (BIF) filtered vs. smoothed state estimates on simulated DFSV data.

This script simulates data from a DFSV model, runs the BIF filter and smoother, and:
- Plots true latent states, filtered BIF estimates, and smoothed BIF estimates.
- Computes RMSE and correlation metrics for both filtered and smoothed estimates.
- Saves a Markdown table summarizing these metrics.

Author: Auto-generated adaptation based on compare_filters.py
Date: 08-04-2025
"""

import os
import time
import jax
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from sklearn.metrics import mean_squared_error

from bellman_filter_dfsv.filters.bellman import DFSVBellmanFilter
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV

def create_test_parameters() -> DFSVParamsDataclass:
    N, K = 3, 2
    lambda_r = np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]])
    Phi_f = np.array([[0.8, 0.005], [0.005, 0.8]])
    Phi_h = np.array([[0.95, 0.003], [0.002, 0.95]])
    mu = np.array([-1.0, -0.5])
    sigma2 = np.array([0.1, 0.1, 0.1])
    Q_h = np.array([[0.1, 0.02], [0.02, 0.1]]) * 1.5
    return DFSVParamsDataclass(
        N=N, K=K,
        lambda_r=jnp.array(lambda_r),
        Phi_f=jnp.array(Phi_f),
        Phi_h=jnp.array(Phi_h),
        mu=jnp.array(mu),
        sigma2=jnp.array(sigma2),
        Q_h=jnp.array(Q_h)
    )

def calculate_rmse(true_vals, est_vals):
    true_np = np.asarray(true_vals)
    est_np = np.asarray(est_vals)
    return [np.sqrt(mean_squared_error(true_np[:, k], est_np[:, k])) for k in range(true_np.shape[1])]

def calculate_corr(true_vals, est_vals):
    true_np = np.asarray(true_vals)
    est_np = np.asarray(est_vals)
    return [np.corrcoef(true_np[:, k], est_np[:, k])[0,1] for k in range(true_np.shape[1])]

def run_bif_filter_and_smoother(params, returns):
    bif = DFSVBellmanFilter(params.N, params.K)
    print("Running BIF filter...")
    t0 = time.perf_counter()
    filtered_states, filtered_infos, _ = bif.filter_scan(params, returns)
    t1 = time.perf_counter()
    print(f"BIF filtering completed in {t1 - t0:.2f} seconds.")

    print("Running BIF smoother...")
    t2 = time.perf_counter()
    smoothed_states, smoothed_covs = bif.smooth(params)
    t3 = time.perf_counter()
    print(f"BIF smoothing completed in {t3 - t2:.2f} seconds.")

    return filtered_states, smoothed_states

def main():
    np.random.seed(456)
    key = jax.random.PRNGKey(456)

    params = create_test_parameters()
    T = 500
    returns, true_factors, true_log_vols = simulate_DFSV(params, T=T, seed=456)

    filtered_means, smoothed_means = run_bif_filter_and_smoother(params, returns)

    K = params.K
    filt_f = np.array(filtered_means[:, :K])
    filt_h = np.array(filtered_means[:, K:])
    smooth_f = np.array(smoothed_means[:, :K])
    smooth_h = np.array(smoothed_means[:, K:])

    # Compute metrics
    filt_f_rmse = calculate_rmse(true_factors, filt_f)
    filt_h_rmse = calculate_rmse(true_log_vols, filt_h)
    smooth_f_rmse = calculate_rmse(true_factors, smooth_f)
    smooth_h_rmse = calculate_rmse(true_log_vols, smooth_h)

    filt_f_corr = calculate_corr(true_factors, filt_f)
    filt_h_corr = calculate_corr(true_log_vols, filt_h)
    smooth_f_corr = calculate_corr(true_factors, smooth_f)
    smooth_h_corr = calculate_corr(true_log_vols, smooth_h)

    print("\n=== BIF Filtered vs. Smoothed Metrics ===")
    print(f"{'':<12} | {'Filtered RMSE':>14} | {'Smoothed RMSE':>14} | {'Filtered Corr':>14} | {'Smoothed Corr':>14}")
    print("-"*70)
    for k in range(K):
        print(f"Factor {k+1:<5} | {filt_f_rmse[k]:14.4f} | {smooth_f_rmse[k]:14.4f} | {filt_f_corr[k]:14.4f} | {smooth_f_corr[k]:14.4f}")
    for k in range(K):
        print(f"LogVol {k+1:<5} | {filt_h_rmse[k]:14.4f} | {smooth_h_rmse[k]:14.4f} | {filt_h_corr[k]:14.4f} | {smooth_h_corr[k]:14.4f}")

    # Save Markdown table
    thesis_assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../thesis_assets')
    os.makedirs(thesis_assets_dir, exist_ok=True)
    md_path = os.path.join(thesis_assets_dir, 'bif_filter_vs_smoother_metrics.md')
    with open(md_path, 'w') as f:
        f.write("| Component | Filtered RMSE | Smoothed RMSE | Filtered Corr | Smoothed Corr |\n")
        f.write("|-----------|---------------|---------------|---------------|---------------|\n")
        for k in range(K):
            f.write(f"| Factor {k+1} | {filt_f_rmse[k]:.4f} | {smooth_f_rmse[k]:.4f} | {filt_f_corr[k]:.4f} | {smooth_f_corr[k]:.4f} |\n")
        for k in range(K):
            f.write(f"| LogVol {k+1} | {filt_h_rmse[k]:.4f} | {smooth_h_rmse[k]:.4f} | {filt_h_corr[k]:.4f} | {smooth_h_corr[k]:.4f} |\n")
    print(f"Saved metrics Markdown table to {md_path}")

    # Plotting
    fig, axes = plt.subplots(2, K, figsize=(15, 8))
    for k in range(K):
        ax = axes[0, k]
        ax.plot(true_factors[:, k], 'k-', label='True')
        ax.plot(filt_f[:, k], 'r--', label='Filtered')
        ax.plot(smooth_f[:, k], 'b-.', label='Smoothed')
        ax.set_title(f'Factor {k+1}')
        ax.legend()
        ax.grid(True)

        ax = axes[1, k]
        ax.plot(true_log_vols[:, k], 'k-', label='True')
        ax.plot(filt_h[:, k], 'r--', label='Filtered')
        ax.plot(smooth_h[:, k], 'b-.', label='Smoothed')
        ax.set_title(f'Log-Volatility {k+1}')
        ax.legend()
        ax.grid(True)

    plt.suptitle('BIF Filtered vs. Smoothed State Estimates (N=3, K=2, T=500)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(thesis_assets_dir, 'bif_filtered_vs_smoothed_N3_K2_T500.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {save_path}")

    plt.show()

if __name__ == "__main__":
    main()