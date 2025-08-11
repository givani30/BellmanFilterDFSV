"""
Visual comparison of Bellman, Bellman Information, and Particle filters for DFSV models.

This script compares the performance of Bellman, Bellman Information, and Particle filters
on simulated data from a DFSV model with N=3 observed series and K=2 factors.
It calculates RMSE, Correlation, (Pseudo) Log-Likelihood, and Execution Time.
"""

import os
import time
import jax # Add JAX import for key generation
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp # Add JAX numpy import
from sklearn.metrics import mean_squared_error

# Updated imports
from bellman_filter_dfsv.filters.bellman import DFSVBellmanFilter
from bellman_filter_dfsv.filters.particle import DFSVParticleFilter
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter # Import BIF
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass # Import the JAX dataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV


def create_test_parameters() -> DFSVParamsDataclass: # Update return type hint
    """
    Create DFSV model parameters (JAX dataclass) with N=3, K=2.

    Returns
    -------
    DFSVParamsDataclass
        Model parameters as a JAX dataclass.
    """
    # Define model dimensions
    N = 3  # Number of observed series
    K = 2  # Number of factors

    # Factor loadings
    lambda_r = np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]])

    # Factor persistence
    Phi_f = np.array([[0.2, 0.005], [0.005, 0.2]])

    # Log-volatility persistence
    Phi_h = np.array([[0.95, 0.00], [0.00, 0.95]])

    # Log-volatility long-run mean
    mu = np.array([-1.0, -0.5])

    # Idiosyncratic variance (diagonal)
    sigma2 = np.array([0.1, 0.1, 0.1])

    # Log-volatility noise covariance
    # Note: Using larger Q_h based on Decision [2025-04-01 02:47:00] for BF stability
    Q_h = np.array([[0.1, 0.02], [0.02, 0.1]]) * 1.0 # Increased scaling

    # Create parameter object
    # Create parameter dataclass object using JAX arrays
    params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=jnp.array(lambda_r),
        Phi_f=jnp.array(Phi_f),
        Phi_h=jnp.array(Phi_h),
        mu=jnp.array(mu),
        sigma2=jnp.array(sigma2), # Keep as 1D for dataclass
        Q_h=jnp.array(Q_h),
    )
    return params

def calculate_rmse(true_values, estimated_values):
    """Calculate RMSE for each column."""
    # Ensure inputs are numpy arrays for sklearn
    true_np = np.asarray(true_values)
    est_np = np.asarray(estimated_values)
    return [np.sqrt(mean_squared_error(true_np[:, k], est_np[:, k])) for k in range(true_np.shape[1])]

def create_visual_comparison(save_path=None):
    """
    Create visual comparison between Bellman, Bellman Information, and Particle filters.

    Parameters
    ----------
    save_path : str, optional
        Path to save the figure. If None, the figure will be displayed.

    Returns
    -------
    plt.Figure
        The created figure
    """
    # Set random seed for reproducibility
    np.random.seed(456)
    key = jax.random.PRNGKey(456) # JAX key for PF

    # Create test parameters
    params = create_test_parameters()

    # Simulate data
    T = 500  # Length of time series
    returns, true_factors, true_log_vols = simulate_DFSV(params, T=T, seed=456)

    # --- Run Bellman Filter (BF) ---
    print("Running Bellman filter (BF)...")
    bf = DFSVBellmanFilter(params.N, params.K)
    start_time_bf = time.perf_counter()
    # Use filter method due to memory leak with filter_scan (Decision [2025-04-02 00:17:00])
    bf_filtered_states, bf_filtered_covs, bf_pseudo_log_likelihood = bf.filter(
        params, returns
    )
    end_time_bf = time.perf_counter()
    bf_time = end_time_bf - start_time_bf
    print(f"BF finished in {bf_time:.2f} seconds.")

    # Extract Bellman filter results
    bf_filtered_factors = bf.get_filtered_factors()
    bf_filtered_log_vols = bf.get_filtered_volatilities()

    # --- Run Particle Filter (PF) ---
    print("Running particle filter (PF)...")
    pf = DFSVParticleFilter(N=params.N, K=params.K, num_particles=8000) # Instantiate with N, K
    start_time_pf = time.perf_counter()
    pf_filtered_states, pf_filtered_covs, pf_log_likelihood = pf.filter(params=params, observations=returns) # Pass params
    end_time_pf = time.perf_counter()
    pf_time = end_time_pf - start_time_pf
    print(f"PF finished in {pf_time:.2f} seconds.")

    # Extract particle filter results
    pf_filtered_factors = pf.get_filtered_factors()
    pf_filtered_log_vols = pf.get_filtered_volatilities()

    # --- Run Bellman Information Filter (BIF) ---
    print("Running Bellman Information filter (BIF)...")
    bif = DFSVBellmanInformationFilter(params.N, params.K)
    start_time_bif = time.perf_counter()
    # Use filter method
    bif_filtered_states, bif_filtered_infos, bif_pseudo_log_likelihood_lange = bif.filter(
        params, returns
    )
    end_time_bif = time.perf_counter()
    bif_time = end_time_bif - start_time_bif
    print(f"BIF finished in {bif_time:.2f} seconds.")

    # Extract Bellman Information filter results
    bif_filtered_factors = bif.get_filtered_factors()
    bif_filtered_log_vols = bif.get_filtered_volatilities()


    # --- Calculate Performance Metrics ---
    # Correlations
    bf_factor_corr = [np.corrcoef(true_factors[:, k], bf_filtered_factors[:, k])[0, 1] for k in range(params.K)]
    bf_vol_corr = [np.corrcoef(true_log_vols[:, k], bf_filtered_log_vols[:, k])[0, 1] for k in range(params.K)]
    pf_factor_corr = [np.corrcoef(true_factors[:, k], pf_filtered_factors[:, k])[0, 1] for k in range(params.K)]
    pf_vol_corr = [np.corrcoef(true_log_vols[:, k], pf_filtered_log_vols[:, k])[0, 1] for k in range(params.K)]
    bif_factor_corr = [np.corrcoef(true_factors[:, k], bif_filtered_factors[:, k])[0, 1] for k in range(params.K)]
    bif_vol_corr = [np.corrcoef(true_log_vols[:, k], bif_filtered_log_vols[:, k])[0, 1] for k in range(params.K)]

    # RMSE
    bf_factor_rmse = calculate_rmse(true_factors, bf_filtered_factors)
    bf_vol_rmse = calculate_rmse(true_log_vols, bf_filtered_log_vols)
    pf_factor_rmse = calculate_rmse(true_factors, pf_filtered_factors)
    pf_vol_rmse = calculate_rmse(true_log_vols, pf_filtered_log_vols)
    bif_factor_rmse = calculate_rmse(true_factors, bif_filtered_factors)
    bif_vol_rmse = calculate_rmse(true_log_vols, bif_filtered_log_vols)

    # --- Print Summary Metrics ---
    print("\n--- Performance Summary ---")
    print(f"Metric                 | {'BF':<15} | {'PF (8k)':<15} | {'BIF':<15}")
    print("-" * 60)
    print(f"(Pseudo) LogLik        | {bf_pseudo_log_likelihood:<15.2f} | {pf_log_likelihood:<15.2f} | {bif_pseudo_log_likelihood_lange:<15.2f}*")
    print(f"Time (s)               | {bf_time:<15.2f} | {pf_time:<15.2f} | {bif_time:<15.2f}")
    print(f"Mean Factor Corr       | {np.mean(bf_factor_corr):<15.3f} | {np.mean(pf_factor_corr):<15.3f} | {np.mean(bif_factor_corr):<15.3f}")
    print(f"Mean LogVol Corr       | {np.mean(bf_vol_corr):<15.3f} | {np.mean(pf_vol_corr):<15.3f} | {np.mean(bif_vol_corr):<15.3f}")
    print(f"Mean Factor RMSE       | {np.mean(bf_factor_rmse):<15.3f} | {np.mean(pf_factor_rmse):<15.3f} | {np.mean(bif_factor_rmse):<15.3f}")
    print(f"Mean LogVol RMSE       | {np.mean(bf_vol_rmse):<15.3f} | {np.mean(pf_vol_rmse):<15.3f} | {np.mean(bif_vol_rmse):<15.3f}")
    print("-" * 60)
    print("* BIF Pseudo LogLik uses Lange (2024) Eq. 40 penalty term.")
    print("Factor Corrs:", [f"{c:.3f}" for c in bf_factor_corr], [f"{c:.3f}" for c in pf_factor_corr], [f"{c:.3f}" for c in bif_factor_corr])
    print("LogVol Corrs:", [f"{c:.3f}" for c in bf_vol_corr], [f"{c:.3f}" for c in pf_vol_corr], [f"{c:.3f}" for c in bif_vol_corr])
    print("Factor RMSEs:", [f"{r:.3f}" for r in bf_factor_rmse], [f"{r:.3f}" for r in pf_factor_rmse], [f"{r:.3f}" for r in bif_factor_rmse])
    print("LogVol RMSEs:", [f"{r:.3f}" for r in bf_vol_rmse], [f"{r:.3f}" for r in pf_vol_rmse], [f"{r:.3f}" for r in bif_vol_rmse])
    print("-" * 60)


    # --- Create Figure ---
    fig = plt.figure(figsize=(15, 18)) # Increased height for table

    # Define grid layout: 5 rows x 2 columns (added row for table)
    gs = fig.add_gridspec(5, 2, height_ratios=[3, 3, 2, 2, 1.5]) # Adjust ratios, give table more space

    # --- First row: Factors ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(true_factors[:, 0], "k-", label="True", alpha=0.8)
    ax1.plot(bf_filtered_factors[:, 0], "r--", label="BF")
    ax1.plot(pf_filtered_factors[:, 0], "b-.", label="PF")
    ax1.plot(bif_filtered_factors[:, 0], "g:", label="BIF") # Add BIF
    ax1.set_title(
        f"Factor 1 (Corr BF:{bf_factor_corr[0]:.2f}, PF:{pf_factor_corr[0]:.2f}, BIF:{bif_factor_corr[0]:.2f})\n"
        f"(RMSE BF:{bf_factor_rmse[0]:.2f}, PF:{pf_factor_rmse[0]:.2f}, BIF:{bif_factor_rmse[0]:.2f})"
    )
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(true_factors[:, 1], "k-", label="True", alpha=0.8)
    ax2.plot(bf_filtered_factors[:, 1], "r--", label="BF")
    ax2.plot(pf_filtered_factors[:, 1], "b-.", label="PF")
    ax2.plot(bif_filtered_factors[:, 1], "g:", label="BIF") # Add BIF
    ax2.set_title(
        f"Factor 2 (Corr BF:{bf_factor_corr[1]:.2f}, PF:{pf_factor_corr[1]:.2f}, BIF:{bif_factor_corr[1]:.2f})\n"
        f"(RMSE BF:{bf_factor_rmse[1]:.2f}, PF:{pf_factor_rmse[1]:.2f}, BIF:{bif_factor_rmse[1]:.2f})"
    )
    ax2.legend()
    ax2.grid(True)

    # --- Second row: Log-volatilities ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(true_log_vols[:, 0], "k-", label="True", alpha=0.8)
    ax3.plot(bf_filtered_log_vols[:, 0], "r--", label="BF")
    ax3.plot(pf_filtered_log_vols[:, 0], "b-.", label="PF")
    ax3.plot(bif_filtered_log_vols[:, 0], "g:", label="BIF") # Add BIF
    ax3.set_title(
        f"Log-Vol 1 (Corr BF:{bf_vol_corr[0]:.2f}, PF:{pf_vol_corr[0]:.2f}, BIF:{bif_vol_corr[0]:.2f})\n"
        f"(RMSE BF:{bf_vol_rmse[0]:.2f}, PF:{pf_vol_rmse[0]:.2f}, BIF:{bif_vol_rmse[0]:.2f})"
    )
    ax3.legend()
    ax3.grid(True)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(true_log_vols[:, 1], "k-", label="True", alpha=0.8)
    ax4.plot(bf_filtered_log_vols[:, 1], "r--", label="BF")
    ax4.plot(pf_filtered_log_vols[:, 1], "b-.", label="PF")
    ax4.plot(bif_filtered_log_vols[:, 1], "g:", label="BIF") # Add BIF
    ax4.set_title(
        f"Log-Vol 2 (Corr BF:{bf_vol_corr[1]:.2f}, PF:{pf_vol_corr[1]:.2f}, BIF:{bif_vol_corr[1]:.2f})\n"
        f"(RMSE BF:{bf_vol_rmse[1]:.2f}, PF:{pf_vol_rmse[1]:.2f}, BIF:{bif_vol_rmse[1]:.2f})"
    )
    ax4.legend()
    ax4.grid(True)

    # --- Third row: Factor errors ---
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(bf_filtered_factors[:, 0] - true_factors[:, 0], "r--", label="BF Error")
    ax5.plot(pf_filtered_factors[:, 0] - true_factors[:, 0], "b-.", label="PF Error")
    ax5.plot(bif_filtered_factors[:, 0] - true_factors[:, 0], "g:", label="BIF Error") # Add BIF
    ax5.axhline(y=0, color="k", linestyle="--")
    ax5.set_title("Factor 1 Errors")
    ax5.legend()
    ax5.grid(True)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(bf_filtered_factors[:, 1] - true_factors[:, 1], "r--", label="BF Error")
    ax6.plot(pf_filtered_factors[:, 1] - true_factors[:, 1], "b-.", label="PF Error")
    ax6.plot(bif_filtered_factors[:, 1] - true_factors[:, 1], "g:", label="BIF Error") # Add BIF
    ax6.axhline(y=0, color="k", linestyle="--")
    ax6.set_title("Factor 2 Errors")
    ax6.legend()
    ax6.grid(True)

    # --- Fourth row: Log-volatility errors ---
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.plot(bf_filtered_log_vols[:, 0] - true_log_vols[:, 0], "r--", label="BF Error")
    ax7.plot(pf_filtered_log_vols[:, 0] - true_log_vols[:, 0], "b-.", label="PF Error")
    ax7.plot(bif_filtered_log_vols[:, 0] - true_log_vols[:, 0], "g:", label="BIF Error") # Add BIF
    ax7.axhline(y=0, color="k", linestyle="--")
    ax7.set_title("Log-Vol 1 Errors")
    ax7.legend()
    ax7.grid(True)

    ax8 = fig.add_subplot(gs[3, 1])
    ax8.plot(bf_filtered_log_vols[:, 1] - true_log_vols[:, 1], "r--", label="BF Error")
    ax8.plot(pf_filtered_log_vols[:, 1] - true_log_vols[:, 1], "b-.", label="PF Error")
    ax8.plot(bif_filtered_log_vols[:, 1] - true_log_vols[:, 1], "g:", label="BIF Error") # Add BIF
    ax8.axhline(y=0, color="k", linestyle="--")
    ax8.set_title("Log-Vol 2 Errors")
    ax8.legend()
    ax8.grid(True)

    # --- Fifth row: Summary Table ---
    ax_table = fig.add_subplot(gs[4, :]) # Span both columns
    ax_table.axis('tight')
    ax_table.axis('off')
    col_labels = ["Metric", "BF", "PF (8k)", "BIF"]
    table_data = [
        ["(Pseudo) LogLik", f"{bf_pseudo_log_likelihood:.2f}", f"{pf_log_likelihood:.2f}", f"{bif_pseudo_log_likelihood_lange:.2f}*"],
        ["Time (s)", f"{bf_time:.2f}", f"{pf_time:.2f}", f"{bif_time:.2f}"],
        ["Mean Factor Corr", f"{np.mean(bf_factor_corr):.3f}", f"{np.mean(pf_factor_corr):.3f}", f"{np.mean(bif_factor_corr):.3f}"],
        ["Mean LogVol Corr", f"{np.mean(bf_vol_corr):.3f}", f"{np.mean(pf_vol_corr):.3f}", f"{np.mean(bif_vol_corr):.3f}"],
        ["Mean Factor RMSE", f"{np.mean(bf_factor_rmse):.3f}", f"{np.mean(pf_factor_rmse):.3f}", f"{np.mean(bif_factor_rmse):.3f}"],
        ["Mean LogVol RMSE", f"{np.mean(bf_vol_rmse):.3f}", f"{np.mean(pf_vol_rmse):.3f}", f"{np.mean(bif_vol_rmse):.3f}"]
    ]
    table = ax_table.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5) # Adjust scale as needed

    # Add overall title
    fig.suptitle(
        "Comparison of Bellman (BF), Particle (PF), and Bellman Information (BIF) Filters (N=3, K=2, T=500)", fontsize=16
    )
    # Adjust layout to prevent overlap, especially with the table
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust rect to make space for suptitle and table note

    # Add note about BIF likelihood below the table
    fig.text(0.5, 0.01, "* BIF Pseudo LogLik uses Lange (2024) Eq. 40 penalty term.", ha='center', va='bottom', fontsize=9)


    # Make sure to save the figure if a path is provided
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure successfully saved to {save_path}")
        except Exception as e:
            print(f"Failed to save figure: {e}")

    return fig


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Create visual comparison and save to file
    # Update filename
    save_path = os.path.join(output_dir, "filter_comparison_BF_PF_BIF_N3_K2_T500.png")
    fig = create_visual_comparison(save_path=save_path)

    # Show plot (optional, can be commented out for headless environments)
    # plt.show() # Comment out for potentially headless execution
    print("\nPlot generated. Displaying plot window...")
    plt.show() # Keep show for interactive use unless specified otherwise
