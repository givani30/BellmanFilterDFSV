#!/usr/bin/env python
"""
Particle Cloud Visualization - "Uncertainty Collapse"

This example demonstrates the Rao-Blackwellized Particle Smoother (RBPS)
by visualizing how particle uncertainty collapses when data is informative.

Simulates a DFSV model with a volatility shock and shows how the particle
distribution (uncertainty) narrows after the shock provides information.

v2.0.0 - Showcasing the RBPS algorithm.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from bellman_filter_dfsv import DFSVParams, run_rbps, simulate_dfsv


def simulate_with_shock(params, T, shock_time, shock_magnitude, key):
    """Simulate DFSV with an injected volatility shock."""
    N, K = params.lambda_r.shape

    # Initialize
    h_curr = params.mu
    f_curr = jnp.zeros(K)

    # Cholesky decompositions
    L_Qh = jnp.linalg.cholesky(params.Q_h + 1e-6 * jnp.eye(K))
    L_Sigma = jnp.diag(jnp.sqrt(params.sigma2))

    def step(carry, t):
        f_prev, h_prev, rng_key = carry
        rng_key, k_h, k_f, k_r = jax.random.split(rng_key, 4)

        # Log-volatility dynamics with injected shock
        eta = L_Qh @ jax.random.normal(k_h, (K,))
        is_shock = t == shock_time
        eta = eta + jnp.where(is_shock, shock_magnitude, 0.0)
        h_curr = params.mu + params.Phi_h @ (h_prev - params.mu) + eta

        # Factor dynamics
        vol_scale = jnp.exp(h_curr / 2)
        eps = jax.random.normal(k_f, (K,))
        f_curr = params.Phi_f @ f_prev + vol_scale * eps

        # Observation
        e = L_Sigma @ jax.random.normal(k_r, (N,))
        r_curr = params.lambda_r @ f_curr + e

        return (f_curr, h_curr, rng_key), (r_curr, f_curr, h_curr)

    _, (returns, factors, log_vols) = jax.lax.scan(
        step, (f_curr, h_curr, key), jnp.arange(T)
    )

    return returns, factors, log_vols


def main():
    """Generate particle cloud visualization."""
    print("Particle Cloud Visualization - 'Uncertainty Collapse'")
    print("=" * 55)

    # Setup: Single factor, high persistence
    N, K = 5, 1
    params = DFSVParams(
        lambda_r=jnp.ones((N, K)) * 0.8,
        Phi_f=jnp.eye(K) * 0.6,
        Phi_h=jnp.eye(K) * 0.95,  # High persistence
        mu=jnp.array([-1.0]),
        sigma2=jnp.ones(N) * 0.2,
        Q_h=jnp.eye(K) * 0.1,
    )

    T = 200
    shock_time = 100
    shock_magnitude = 4.0  # Large volatility spike

    # Simulate with shock
    print(f"Simulating {T} observations with shock at t={shock_time}...")
    key = jax.random.PRNGKey(42)
    returns, true_factors, true_log_vols = simulate_with_shock(
        params, T, shock_time, shock_magnitude, key
    )
    print("Simulation complete!")

    # Run RBPS
    print("Running Rao-Blackwellized Particle Smoother...")
    rbps_result = run_rbps(
        params=params,
        observations=returns,
        num_particles=500,
        num_trajectories=100,
        seed=42,
    )
    print("RBPS complete!")

    # Extract particle trajectories
    h_particles = np.array(rbps_result.h_samples[..., 0])  # (M, T)
    true_h = np.array(true_log_vols[:, 0])  # (T,)
    h_mean = h_particles.mean(axis=0)  # (T,)

    # Visualization
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(12, 6))

    time_axis = np.arange(T)

    # Plot particle cloud (creates density effect with low alpha)
    for i in range(h_particles.shape[0]):
        ax.plot(
            time_axis,
            h_particles[i],
            color="#1f77b4",
            alpha=0.05,
            linewidth=1.0,
            zorder=1,
        )

    # Plot true log-volatility
    ax.plot(
        time_axis,
        true_h,
        color="black",
        linewidth=2.0,
        label="True Log-Volatility",
        linestyle="--",
        zorder=3,
    )

    # Plot RBPS mean estimate
    ax.plot(
        time_axis,
        h_mean,
        color="#d62728",
        linewidth=2.5,
        label="RBPS Smoothed Estimate",
        zorder=2,
    )

    # Mark shock location
    ax.axvline(x=shock_time, color="gray", linestyle=":", alpha=0.5)
    ax.text(
        shock_time + 2,
        max(true_h),
        "Volatility Shock",
        color="gray",
        fontsize=10,
    )

    # Annotate uncertainty regions
    quiet_t = 50  # Before shock: high uncertainty
    spread = np.std(h_particles[:, quiet_t]) * 4
    ax.annotate(
        "",
        xy=(quiet_t, h_mean[quiet_t] - spread / 2),
        xytext=(quiet_t, h_mean[quiet_t] + spread / 2),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
    )
    ax.text(
        quiet_t + 5,
        h_mean[quiet_t],
        "High Uncertainty\n(Wide Cloud)",
        verticalalignment="center",
    )

    tight_t = shock_time + 10  # After shock: collapsed uncertainty
    ax.text(
        tight_t + 5,
        h_mean[tight_t] + 1.0,
        "Uncertainty Collapse\n(Tight Cloud)",
        verticalalignment="bottom",
    )

    # Styling
    ax.set_title(
        "Uncertainty Collapse: Particle Cloud vs. True Volatility",
        fontsize=14,
        pad=15,
    )
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Log-Volatility ($h_t$)", fontsize=12)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = "particle_cloud_visualization.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to {output_path}")
    plt.show()

    return params, returns, rbps_result


if __name__ == "__main__":
    main()
