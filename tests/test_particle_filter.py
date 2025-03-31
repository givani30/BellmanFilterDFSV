"""
Unit tests for the Particle Filter implementation in DFSV models.

This module provides tests to validate the particle filter implementation,
including numerical accuracy tests and visualization tests.
"""

import sys
import os
import unittest
import numpy as np
import jax.numpy as jnp # Add this import
import matplotlib.pyplot as plt
from pathlib import Path

# Remove sys.path hack - imports should work if package is installed editable
# sys.path.insert(0, str(Path(__file__).parent.parent))

# Updated imports to use the new package structure
from qf_thesis.core.filters.particle import DFSVParticleFilter
from qf_thesis.core.simulation import simulate_DFSV
from qf_thesis.models.dfsv import DFSVParamsDataclass


class TestParticleFilter(unittest.TestCase):
    """Test suite for the DFSVParticleFilter class."""

    def create_test_parameters(self) -> DFSVParamsDataclass:
        """
        Create a set of test parameters for a small DFSV model as a JAX dataclass.

        Returns
        -------
        DFSVParamsDataclass
            Parameters for a 2-factor model with 4 observed series (JAX arrays).
        """
        # Define model dimensions
        N = 4  # Number of observed series
        K = 2  # Number of factors

        # Factor loadings
        lambda_r = jnp.array([[0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8]])

        # Factor persistence
        Phi_f = jnp.array([[0.9, 0.1], [0.1, 0.9]])

        # Log-volatility persistence
        Phi_h = jnp.array([[0.95, 0.0], [0.0, 0.95]])

        # Log-volatility long-run mean
        mu = jnp.array([-1.0, -0.5])

        # Idiosyncratic variance (diagonal)
        sigma2 = jnp.array([0.1, 0.1, 0.1, 0.1])

        # Log-volatility noise covariance
        Q_h = jnp.array([[0.1, 0.02], [0.02, 0.1]])

        # Create parameter dataclass object
        params = DFSVParamsDataclass(
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

    def test_particle_filter_estimation(self):
        """
        Test the particle filter's ability to estimate states from simulated data.

        This test:
        1. Generates synthetic time series data from a known DFSV model
        2. Applies the particle filter to estimate states
        3. Checks that the estimates are reasonably close to the true states
        """
        # Set random seed for reproducibility
        np.random.seed(42)

        # Create test parameters (now returns DFSVParamsDataclass)
        params_dataclass = self.create_test_parameters()

        # Simulate data using the JAX dataclass directly
        T = 200  # Time series length
        returns, true_factors, true_log_vols = simulate_DFSV(params_dataclass, T=T, seed=42)

        # Create and run particle filter using the DATACLASS
        pf = DFSVParticleFilter(params_dataclass, num_particles=500)
        filtered_states, filtered_covs, log_likelihood = pf.filter(params_dataclass, returns)

        # Extract factor and volatility estimates
        filtered_factors = pf.get_filtered_factors()
        filtered_log_vols = pf.get_filtered_volatilities()

        # Test: Check correlation between true and filtered factors
        for k in range(params_dataclass.K):
            corr = np.corrcoef(true_factors[:, k], filtered_factors[:, k])[0, 1]
            self.assertGreater(corr, 0.7, f"Factor {k} correlation too low: {corr}")

        # Test: Check correlation between true and filtered log-volatilities
        for k in range(params_dataclass.K):
            corr = np.corrcoef(true_log_vols[:, k], filtered_log_vols[:, k])[0, 1]
            self.assertGreater(
                corr, 0.15, f"Log-volatility {k} correlation too low: {corr}"
            )

        # Test: Check the average estimation error is within reasonable bounds
        factor_rmse = np.sqrt(np.mean((true_factors - filtered_factors) ** 2, axis=0))
        vol_rmse = np.sqrt(np.mean((true_log_vols - filtered_log_vols) ** 2, axis=0))

        # Check if all factor RMSEs are below the threshold
        self.assertTrue(np.all(factor_rmse < 1.0), f"Factor RMSE too high: {factor_rmse}")
        # Check if all log-volatility RMSEs are below the threshold
        self.assertTrue(np.all(vol_rmse < 1.5), f"Log-volatility RMSE too high: {vol_rmse}")

        # Print additional information (won't affect test outcome but helpful for analysis)
        print(
            f"Factor correlation: {np.corrcoef(true_factors.flatten(), filtered_factors.flatten())[0, 1]:.4f}"
        )
        print(
            f"Log-volatility correlation: {np.corrcoef(true_log_vols.flatten(), filtered_log_vols.flatten())[0, 1]:.4f}"
        )
        print(f"Factor RMSE: {factor_rmse}")
        print(f"Log-volatility RMSE: {vol_rmse}")

    def test_particle_filter_numeric_stability(self):
        """
        Test the particle filter's numerical stability with a longer time series.

        This test focuses on making sure the filter doesn't break down over long
        sequences and maintains reasonable estimation quality.
        """
        # Create test parameters (now returns DFSVParamsDataclass)
        params_dataclass = self.create_test_parameters()

        # Simulate data using the JAX dataclass directly
        T = 500
        returns, true_factors, true_log_vols = simulate_DFSV(params_dataclass, T=T, seed=123)

        # Create and run particle filter with more particles for stability, using DATACLASS
        pf = DFSVParticleFilter(params_dataclass, num_particles=1000)
        filtered_states, filtered_covs, log_likelihood = pf.filter(params_dataclass,returns)

        # Check that the filter completes without errors and returns valid estimates
        self.assertFalse(
            np.any(np.isnan(filtered_states)), "Filter produced NaN values"
        )
        self.assertFalse(
            np.any(np.isinf(filtered_states)), "Filter produced infinite values"
        )
        self.assertIsInstance(
            log_likelihood, float, "Log-likelihood is not a valid float"
        )

    def test_log_likelihood_calculation(self):
        """
        Test that the filter computes a finite log-likelihood value.
        """
        # Set random seed for reproducibility
        np.random.seed(789)

        # Create test parameters (now returns DFSVParamsDataclass)
        params_dataclass = self.create_test_parameters()

        # Simulate data using the JAX dataclass directly
        T = 1000  # Shorter series for this test
        returns, _, _ = simulate_DFSV(params_dataclass, T=T, seed=789)

        # Create and run particle filter using DATACLASS
        pf = DFSVParticleFilter(params_dataclass, num_particles=500)
        _, _, log_likelihood = pf.filter(params_dataclass, returns)

        # Test: Check that log-likelihood is a finite float
        self.assertIsInstance(
            log_likelihood, float, "Log-likelihood is not a float type."
        )
        self.assertTrue(
            np.isfinite(log_likelihood),
            f"Log-likelihood is not finite: {log_likelihood}",
        )
        print(f"Calculated Log-Likelihood: {log_likelihood}") # Print for info

    def test_visualization(self):
        """
        Test to generate and save visual comparison of particle filter results.
        """
        # Create an absolute path for the output file
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "outputs"
        )
        save_path = os.path.join(output_dir, "pf_visual_comparison.png")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Generate visual comparison and save to file
        fig = self.create_visual_comparison(save_path=save_path)

        # Basic assertion to ensure the figure was created
        self.assertIsInstance(fig, plt.Figure)

        # Verify the file was actually saved
        self.assertTrue(
            os.path.exists(save_path), f"Figure was not saved to {save_path}"
        )
        print(f"Figure successfully saved to {save_path}")

    def create_visual_comparison(self, save_path=None):
        """
        Create visual comparisons between true, filtered, and smoothed states.

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

        # Create test parameters (now returns DFSVParamsDataclass)
        params_dataclass = self.create_test_parameters()

        # Simulate data using the JAX dataclass directly
        T = 600
        returns, true_factors, true_log_vols = simulate_DFSV(params_dataclass, T=T, seed=456)

        # Create and run particle filter using DATACLASS
        pf = DFSVParticleFilter(params_dataclass, num_particles=1000)
        filtered_states, filtered_covs, log_likelihood = pf.filter(params_dataclass,returns)

        # Extract filtered factors and volatilities first
        filtered_factors = pf.get_filtered_factors()
        filtered_log_vols = pf.get_filtered_volatilities()

        try:
            # Try running smoother - wrap in try-except in case it fails
            smoothed_states, smoothed_covs = pf.smooth()
            smoothed_factors = pf.get_smoothed_factors()
            smoothed_log_vols = pf.get_smoothed_volatilities()
            include_smoothed = True
        except Exception as e:
            print(f"Smoothing failed with error: {e}")
            include_smoothed = False

        # Calculate return variances
        # True covvariances
        true_variances = np.zeros((T, params_dataclass.N, params_dataclass.N))
        filtered_variances = np.zeros((T, params_dataclass.N, params_dataclass.N))
        if include_smoothed:
            smoothed_variances = np.zeros((T, params_dataclass.N, params_dataclass.N))

        for t in range(T):
            # Calculate true variances
            true_vol_matrix = np.diag(np.exp(true_log_vols[t]))
            true_variances[t, :, :] = (
                np.diag(params_dataclass.lambda_r @ true_vol_matrix @ params_dataclass.lambda_r.T)
                + params_dataclass.sigma2
            )

            # Calculate filtered variances
            filtered_vol_matrix = np.diag(np.exp(filtered_log_vols[t]))
            filtered_variances[t, :, :] = (
                np.diag(params_dataclass.lambda_r @ filtered_vol_matrix @ params_dataclass.lambda_r.T)
                + params_dataclass.sigma2
            )

            if include_smoothed:
                # Calculate smoothed variances
                smoothed_vol_matrix = np.diag(np.exp(smoothed_log_vols[t]))
                smoothed_variances[t, :, :] = (
                    np.diag(params_dataclass.lambda_r @ smoothed_vol_matrix @ params_dataclass.lambda_r.T)
                    + params_dataclass.sigma2
                )

        # Create figure with subplots for factors, log-volatilities, errors and return variances
        fig, axs = plt.subplots(4, 2, figsize=(15, 16))

        # Plot factors
        for k in range(params_dataclass.K):
            ax = axs[0, k]
            ax.plot(true_factors[:, k], "b-", label="True")
            ax.plot(filtered_factors[:, k], "r--", label="Filtered")
            if include_smoothed:
                ax.plot(smoothed_factors[:, k], "g-.", label="Smoothed")
            ax.set_title(f"Factor {k + 1}")
            ax.legend()
            ax.grid(True)

        # Plot volatilities
        for k in range(params_dataclass.K):
            ax = axs[1, k]
            ax.plot(np.exp(true_log_vols[:, k] / 2), "b-", label="True")
            ax.plot(np.exp(filtered_log_vols[:, k] / 2), "r--", label="Filtered")
            if include_smoothed:
                ax.plot(np.exp(smoothed_log_vols[:, k] / 2), "g-.", label="Smoothed")
            ax.set_title(f"Volatility {k + 1}")
            ax.legend()
            ax.grid(True)

        # Plot log-volatility errors
        for k in range(params_dataclass.K):
            ax = axs[2, k]
            filtered_error = filtered_log_vols[:, k] - true_log_vols[:, k]
            ax.plot(filtered_error, "r-", label="Filtered Error")
            ax.axhline(y=0, color="k", linestyle="--")
            ax.set_title(f"Log-Volatility {k + 1} Error")
            ax.legend()
            ax.grid(True)

        # Plot return variances (show for first 2 return series)
        for n in range(min(2, params_dataclass.N)):
            ax = axs[3, n]
            ax.plot(true_variances[:, n, n], "b-", label="True")
            ax.plot(filtered_variances[:, n, n], "r--", label="Filtered")
            if include_smoothed:
                ax.plot(smoothed_variances[:, n, n], "g-,", label="Smoothed")
            ax.set_title(f"Return {n + 1} Variance")
            ax.legend()
            ax.grid(True)

        fig.suptitle(
            "Particle Filter Performance: True vs. Filtered vs. Smoothed States",
            fontsize=16,
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

    def test_log_likelihood_of_params_calculation(self):
        """
        Test the log_likelihood_of_params method for correctness and stability.
        """
        # Set random seed for reproducibility
        np.random.seed(111)
        jax_key_seed = 111 # Use a separate seed for JAX if needed within the method

        # Create test parameters (now returns DFSVParamsDataclass)
        params_dataclass = self.create_test_parameters()

        # Simulate data using the JAX dataclass directly
        T = 150  # Moderate time series length
        returns, _, _ = simulate_DFSV(params_dataclass, T=T, seed=111)

        # Convert returns to JAX array
        jax_returns = jnp.array(returns)

        # Create particle filter instance using DATACLASS
        # Use the same seed for the filter instance for consistency
        pf = DFSVParticleFilter(params_dataclass, num_particles=500, seed=jax_key_seed)

        # --- Test the log_likelihood_of_params method ---
        log_likelihood_opt = pf.log_likelihood_of_params(params_dataclass, jax_returns)

        # Assertions for the optimization-focused method
        self.assertIsInstance(
            log_likelihood_opt, float,
            "log_likelihood_of_params did not return a float."
        )
        self.assertTrue(
            np.isfinite(log_likelihood_opt),
            f"log_likelihood_of_params returned non-finite value: {log_likelihood_opt}"
        )
        print(f"\nLog-Likelihood from log_likelihood_of_params: {log_likelihood_opt:.4f}")

        # --- Optional: Compare with standard filter log-likelihood ---
        # Re-create filter with the same seed to ensure same particle initialization
        pf_filter = DFSVParticleFilter(params_dataclass, num_particles=500, seed=jax_key_seed)
        _, _, log_likelihood_filter = pf_filter.filter(params_dataclass, returns) # Use original numpy returns

        self.assertIsInstance(
            log_likelihood_filter, float,
            "Standard filter did not return a float log-likelihood."
        )
        self.assertTrue(
            np.isfinite(log_likelihood_filter),
            f"Standard filter returned non-finite log-likelihood: {log_likelihood_filter}"
        )
        print(f"Log-Likelihood from standard filter: {log_likelihood_filter:.4f}")

        # Compare the two log-likelihoods (adjust delta as needed)
        # Particle filters are stochastic, so allow for some difference
        self.assertAlmostEqual(
            log_likelihood_opt, log_likelihood_filter, delta=1.0, # Delta might need tuning
            msg=(f"Log-likelihoods differ significantly: "
                 f"Opt={log_likelihood_opt:.4f}, Filter={log_likelihood_filter:.4f}")
        )


if __name__ == "__main__":
    unittest.main()
