"""
Visual comparison of Bellman and Particle filters for DFSV models.

This script compares the performance of Bellman and Particle filters
on simulated data from a DFSV model with N=3 observed series and K=2 factors.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from functions.simulation import DFSV_params, simulate_DFSV
from functions.filters import DFSVBellmanFilter, DFSVParticleFilter


def create_test_parameters():
    """
    Create DFSV model parameters with N=3 observed series and K=2 factors.

    Returns
    -------
    DFSV_params
        Model parameters
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
    Q_h = np.array([[0.1, 0.02], [0.02, 0.1]])

    # Create parameter object
    params = DFSV_params(
        N=N,
        K=K,
        lambda_r=lambda_r,
        Phi_f=Phi_f,
        Phi_h=Phi_h,
        mu=mu,
        sigma2=sigma2,
        Q_h=Q_h,
    )

    return params


def create_visual_comparison(save_path=None):
    """
    Create visual comparison between Bellman filter and particle filter.

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

    # Create test parameters
    params = create_test_parameters()

    # Simulate data
    T = 200  # Length of time series
    returns, true_factors, true_log_vols = simulate_DFSV(params, T=T, seed=456)

    # Create and run Bellman filter
    print("Running Bellman filter...")
    bf = DFSVBellmanFilter(params)
    bf_filtered_states, bf_filtered_covs, bf_log_likelihood = bf.filter(returns)

    # Extract Bellman filter results
    bf_filtered_factors = bf.get_filtered_factors()
    bf_filtered_log_vols = bf.get_filtered_volatilities()

    # Create and run particle filter
    print("Running particle filter...")
    pf = DFSVParticleFilter(params, num_particles=1000)
    pf_filtered_states, pf_filtered_covs, pf_log_likelihood = pf.filter(returns)

    # Extract particle filter results
    pf_filtered_factors = pf.get_filtered_factors()
    pf_filtered_log_vols = pf.get_filtered_volatilities()

    # Calculate performance metrics
    bf_factor_corr = [
        np.corrcoef(true_factors[:, k], bf_filtered_factors[:, k])[0, 1]
        for k in range(params.K)
    ]
    bf_vol_corr = [
        np.corrcoef(true_log_vols[:, k], bf_filtered_log_vols[:, k])[0, 1]
        for k in range(params.K)
    ]

    pf_factor_corr = [
        np.corrcoef(true_factors[:, k], pf_filtered_factors[:, k])[0, 1]
        for k in range(params.K)
    ]
    pf_vol_corr = [
        np.corrcoef(true_log_vols[:, k], pf_filtered_log_vols[:, k])[0, 1]
        for k in range(params.K)
    ]

    print(f"Bellman filter - log-likelihood: {bf_log_likelihood}")
    print(f"Particle filter - log-likelihood: {pf_log_likelihood}")
    print(f"Bellman filter - factor correlations: {bf_factor_corr}")
    print(f"Particle filter - factor correlations: {pf_factor_corr}")
    print(f"Bellman filter - volatility correlations: {bf_vol_corr}")
    print(f"Particle filter - volatility correlations: {pf_vol_corr}")

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))

    # Define grid layout: 4 rows x 2 columns
    gs = fig.add_gridspec(4, 2)

    # First row: Factors
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(true_factors[:, 0], "k-", label="True")
    ax1.plot(bf_filtered_factors[:, 0], "r--", label="Bellman")
    ax1.plot(pf_filtered_factors[:, 0], "b-.", label="Particle")
    ax1.set_title(
        f"Factor 1 (BF corr: {bf_factor_corr[0]:.3f}, PF corr: {pf_factor_corr[0]:.3f})"
    )
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(true_factors[:, 1], "k-", label="True")
    ax2.plot(bf_filtered_factors[:, 1], "r--", label="Bellman")
    ax2.plot(pf_filtered_factors[:, 1], "b-.", label="Particle")
    ax2.set_title(
        f"Factor 2 (BF corr: {bf_factor_corr[1]:.3f}, PF corr: {pf_factor_corr[1]:.3f})"
    )
    ax2.legend()
    ax2.grid(True)

    # Second row: Log-volatilities
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(true_log_vols[:, 0], "k-", label="True")
    ax3.plot(bf_filtered_log_vols[:, 0], "r--", label="Bellman")
    ax3.plot(pf_filtered_log_vols[:, 0], "b-.", label="Particle")
    ax3.set_title(
        f"Log-Vol 1 (BF corr: {bf_vol_corr[0]:.3f}, PF corr: {pf_vol_corr[0]:.3f})"
    )
    ax3.legend()
    ax3.grid(True)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(true_log_vols[:, 1], "k-", label="True")
    ax4.plot(bf_filtered_log_vols[:, 1], "r--", label="Bellman")
    ax4.plot(pf_filtered_log_vols[:, 1], "b-.", label="Particle")
    ax4.set_title(
        f"Log-Vol 2 (BF corr: {bf_vol_corr[1]:.3f}, PF corr: {pf_vol_corr[1]:.3f})"
    )
    ax4.legend()
    ax4.grid(True)

    # Third row: Factor errors
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(
        bf_filtered_factors[:, 0] - true_factors[:, 0], "r-", label="Bellman Error"
    )
    ax5.plot(
        pf_filtered_factors[:, 0] - true_factors[:, 0], "b-", label="Particle Error"
    )
    ax5.axhline(y=0, color="k", linestyle="--")
    ax5.set_title("Factor 1 Errors")
    ax5.legend()
    ax5.grid(True)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(
        bf_filtered_factors[:, 1] - true_factors[:, 1], "r-", label="Bellman Error"
    )
    ax6.plot(
        pf_filtered_factors[:, 1] - true_factors[:, 1], "b-", label="Particle Error"
    )
    ax6.axhline(y=0, color="k", linestyle="--")
    ax6.set_title("Factor 2 Errors")
    ax6.legend()
    ax6.grid(True)

    # Fourth row: Log-volatility errors
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.plot(
        bf_filtered_log_vols[:, 0] - true_log_vols[:, 0], "r-", label="Bellman Error"
    )
    ax7.plot(
        pf_filtered_log_vols[:, 0] - true_log_vols[:, 0], "b-", label="Particle Error"
    )
    ax7.axhline(y=0, color="k", linestyle="--")
    ax7.set_title("Log-Vol 1 Errors")
    ax7.legend()
    ax7.grid(True)

    ax8 = fig.add_subplot(gs[3, 1])
    ax8.plot(
        bf_filtered_log_vols[:, 1] - true_log_vols[:, 1], "r-", label="Bellman Error"
    )
    ax8.plot(
        pf_filtered_log_vols[:, 1] - true_log_vols[:, 1], "b-", label="Particle Error"
    )
    ax8.axhline(y=0, color="k", linestyle="--")
    ax8.set_title("Log-Vol 2 Errors")
    ax8.legend()
    ax8.grid(True)

    # Add overall title
    fig.suptitle(
        "Comparison of Bellman Filter vs Particle Filter (N=3, K=2)", fontsize=16
    )
    plt.tight_layout()

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
    save_path = os.path.join(output_dir, "bellman_particle_comparison_N3_K2.png")
    fig = create_visual_comparison(save_path=save_path)

    # Show plot (optional, can be commented out for headless environments)
    plt.show()
