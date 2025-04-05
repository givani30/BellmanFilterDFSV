"""Unified test suite for the DFSVBellmanFilter implementation.

This module contains tests verifying the functionality, numerical stability,
and estimation accuracy of the covariance-based Bellman filter, which now
utilizes the BIF pseudo-likelihood calculation internally.
"""

import sys
import os
import unittest
import numpy as np
import jax.numpy as jnp # Add JAX numpy import
import matplotlib.pyplot as plt
from pathlib import Path

# Remove sys.path hack
# sys.path.insert(0, str(Path(__file__).parent.parent))

# Updated imports
from bellman_filter_dfsv.core.filters.bellman import DFSVBellmanFilter
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass # Import the JAX dataclass
from bellman_filter_dfsv.core.simulation import simulate_DFSV


class TestBasicBellmanFilter(unittest.TestCase):
    """Basic tests for the DFSVBellmanFilter implementation."""

    def test_bellman_single_step(self):
        """Tests a single prediction and update step of the filter."""
        # Set random seed for reproducibility
        np.random.seed(42)

        # Create test parameters
        N = 5  # Number of observed series
        K = 2  # Number of factors

        # Initialize model parameters using JAX dataclass
        params = DFSVParamsDataclass(
            N=N,
            K=K,
            lambda_r=jnp.array(np.random.normal(0, 1, (N, K))),  # Random factor loadings
            Phi_f=jnp.array(0.9 * np.eye(K)),  # AR(1) factor dynamics
            Phi_h=jnp.array(0.95 * np.eye(K)),  # Persistent volatility
            mu=jnp.array(np.zeros(K)),  # Mean log-volatility
            Q_h=jnp.array(0.1 * np.eye(K)),  # Volatility innovation variance
            sigma2=jnp.array(0.1 * np.ones(N)),  # Measurement noise variance
        )

        # Initialize filter
        bf = DFSVBellmanFilter(N,K)

        # Create some artificial data
        T = 1
        y = np.random.normal(0, 1, (T, N))

        # Initialize state
        initial_state, initial_cov = bf.initialize_state(params)

        # Prediction step
        predicted_state, predicted_cov = bf.predict(params,initial_state, initial_cov)

        # Update step
        # Note: y[0] because update expects a single observation vector
        updated_state, updated_cov, log_likelihood = bf.update(params,
            predicted_state, predicted_cov, y[0]
        )

        # Basic assertions
        self.assertEqual(initial_state.shape, (2 * K, 1), "Incorrect initial state shape")
        self.assertEqual(initial_cov.shape, (2 * K, 2 * K), "Incorrect initial covariance shape")
        self.assertEqual(predicted_state.shape, (2 * K, 1), "Incorrect predicted state shape")
        self.assertEqual(predicted_cov.shape, (2 * K, 2 * K), "Incorrect predicted covariance shape")
        self.assertEqual(updated_state.shape, (2 * K, 1), "Incorrect updated state shape")
        self.assertEqual(updated_cov.shape, (2 * K, 2 * K), "Incorrect updated covariance shape")
        self.assertIsInstance(log_likelihood, float, "Log-likelihood should be a float")
        self.assertFalse(np.any(np.isnan(updated_state)), "Updated state contains NaN values")
        self.assertFalse(np.any(np.isnan(updated_cov)), "Updated covariance contains NaN values")
        self.assertFalse(np.isnan(log_likelihood), "Log-likelihood is NaN")

    def test_bellman_full_filter(self):
        """Tests the full filter operation over multiple time steps."""
        # Create a small test model
        K = 1  # Number of factors (using a smaller model for speed)
        N = 3  # Number of observed series

        # Define model parameters
        lambda_r = np.random.normal(0, 1, size=(N, K))
        Phi_f = np.array([[0.9]])  # Autoregressive coefficients for factor
        Phi_h = np.array([[0.95]])  # Autoregressive coefficients for log-volatility
        mu = np.array([-5.0])  # Unconditional mean of log-volatility
        sigma2 = np.ones(N) * 0.1  # Measurement noise variance
        Q_h = np.array([[0.2]])  # Volatility of log-volatility process

        # Create parameter dataclass object using JAX arrays
        params = DFSVParamsDataclass(
            N=N,
            K=K,
            lambda_r=jnp.array(lambda_r),
            Phi_f=jnp.array(Phi_f),
            Phi_h=jnp.array(Phi_h),
            mu=jnp.array(mu),
            sigma2=jnp.array(sigma2),
            Q_h=jnp.array(Q_h),
        )

        # Generate a small sample of simulated data
        T = 20
        y, factors, log_vols = simulate_DFSV(params, T=T, seed=42)

        # Initialize the Bellman filter
        bf = DFSVBellmanFilter(N,K)

        # Run the filter (using the standard loop version for this test)
        filtered_states, filtered_covs, total_log_likelihood = bf.filter(params, y)

        # Basic assertions for filtered results
        self.assertEqual(
            filtered_states.shape,
            (T, 2 * K),
            f"Filtered states have wrong shape: {filtered_states.shape}",
        )
        self.assertEqual(
            filtered_covs.shape,
            (T, 2 * K, 2 * K),
             f"Filtered covariances have wrong shape: {filtered_covs.shape}"
        )
        self.assertIsInstance(total_log_likelihood, float, "Total log-likelihood should be a float")
        self.assertFalse(
            np.any(np.isnan(filtered_states)), "Filtered states contain NaN values"
        )
        self.assertFalse(
            np.any(np.isnan(filtered_covs)), "Filtered covariances contain NaN values"
        )
        self.assertFalse(np.isnan(total_log_likelihood), "Total log-likelihood is NaN")


class TestComprehensiveBellmanFilter(unittest.TestCase):
    """Comprehensive tests for the DFSVBellmanFilter class."""

    def create_test_parameters(self) -> DFSVParamsDataclass:
        """Creates a set of test parameters for a DFSV model.

        Returns:
            Parameters for a 2-factor, 4-observable model as a DFSVParamsDataclass
            with JAX arrays.
        """
        # Define model dimensions
        N = 4  # Number of observed series
        K = 2  # Number of factors

        # Factor loadings
        lambda_r = np.array([[0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8]])
        # Factor persistence
        Phi_f = np.array([[0.9, 0.1], [0.1, 0.9]])
        # Log-volatility persistence
        Phi_h = np.array([[0.95, 0.0], [0.0, 0.95]])
        # Log-volatility long-run mean
        mu = np.array([-1.0, -0.5])
        # Idiosyncratic variance (diagonal)
        sigma2 = np.array([0.1, 0.1, 0.1, 0.1])
        # Log-volatility noise covariance
        Q_h = np.array([[0.1, 0.02], [0.02, 0.1]])

        # Create parameter dataclass object using JAX arrays
        params = DFSVParamsDataclass(
            N=N,
            K=K,
            lambda_r=jnp.array(lambda_r),
            Phi_f=jnp.array(Phi_f),
            Phi_h=jnp.array(Phi_h),
            mu=jnp.array(mu),
            sigma2=jnp.array(sigma2),
            Q_h=jnp.array(Q_h),
        )
        return params

    def test_bellman_filter_estimation(self):
        """Tests the filter's state estimation accuracy on simulated data."""
        # Set random seed for reproducibility
        np.random.seed(42)

        # Create test parameters
        params = self.create_test_parameters()

        # Simulate data
        f0 = np.array([0.5, 0.1])  # Initial state for factors
        T = 200  # Time series length
        returns, true_factors, true_log_vols = simulate_DFSV(
            params, f0=f0, T=T, seed=42
        )

        # Create and run Bellman filter (using filter_scan for efficiency)
        bf = DFSVBellmanFilter(params.N,params.K)
        # Use filter_scan as it's generally preferred
        filtered_states_np, filtered_covs_np, log_likelihood_jax = bf.filter_scan(params, returns)

        # Extract factor and volatility estimates
        filtered_factors = bf.get_filtered_factors()
        filtered_log_vols = bf.get_filtered_volatilities()

        # Assertions on shapes and types
        self.assertEqual(filtered_factors.shape, (T, params.K))
        self.assertEqual(filtered_log_vols.shape, (T, params.K))

        # Test: Check correlation between true and filtered factors
        for k in range(params.K):
            corr = np.corrcoef(true_factors[:, k], filtered_factors[:, k])[0, 1]
            self.assertGreater(corr, 0.7, f"Factor {k+1} correlation too low: {corr:.3f}")

        # Test: Check correlation between true and filtered log-volatilities
        for k in range(params.K):
            corr = np.corrcoef(true_log_vols[:, k], filtered_log_vols[:, k])[0, 1]
            # Threshold kept as per original test, may need adjustment based on BIF impact
            self.assertGreater(
                corr, 0.4, f"Log-volatility {k+1} correlation too low: {corr:.3f}"
            )

        # Test: Check the average estimation error is within reasonable bounds
        factor_rmse = np.sqrt(np.mean((true_factors - filtered_factors) ** 2))
        vol_rmse = np.sqrt(np.mean((true_log_vols - filtered_log_vols) ** 2))

        # Thresholds kept as per original test
        self.assertLess(factor_rmse, 0.6, f"Factor RMSE too high: {factor_rmse:.3f}")
        self.assertLess(vol_rmse, 1.5, f"Log-volatility RMSE too high: {vol_rmse:.3f}")

    def test_bellman_filter_numeric_stability(self):
        """Tests the filter's numerical stability over a longer time series."""
        # Create test parameters
        params = self.create_test_parameters()

        # Simulate a longer series
        T = 500
        returns, true_factors, true_log_vols = simulate_DFSV(params, T=T, seed=123)

        # Create and run Bellman filter (using filter_scan)
        bf = DFSVBellmanFilter(params.N,params.K)
        filtered_states, filtered_covs, log_likelihood = bf.filter_scan(params, returns)

        # Check that the filter completes without errors and returns valid estimates
        self.assertFalse(
            np.any(np.isnan(filtered_states)), "Filter produced NaN states"
        )
        self.assertFalse(
            np.any(np.isinf(filtered_states)), "Filter produced infinite states"
        )
        self.assertFalse(
            np.any(np.isnan(filtered_covs)), "Filter produced NaN covariances"
        )
        self.assertFalse(
            np.any(np.isinf(filtered_covs)), "Filter produced infinite covariances"
        )
        self.assertTrue(
             jnp.isfinite(log_likelihood), "Log-likelihood is not finite"
        )

    def test_visualization(self):
        """Generates and saves a visual comparison of filter results."""
        # Create an absolute path for the output file
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "outputs"
        )
        save_path = os.path.join(output_dir, "bf_visual_comparison.png")

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
        # Close the plot to free memory
        plt.close(fig)

    def create_visual_comparison(self, save_path=None):
        """Creates visual comparisons between true and filtered states.

        Args:
            save_path (Optional[str]): Path to save the figure. If None, the
                figure is not saved.

        Returns:
            matplotlib.figure.Figure: The created figure object.
        """
        # Set random seed for reproducibility
        np.random.seed(456)

        # Create test parameters
        params = self.create_test_parameters()

        # Simulate data
        T = 300
        returns, true_factors, true_log_vols = simulate_DFSV(params, T=T, seed=456)

        # Create and run Bellman filter (using filter_scan)
        bf = DFSVBellmanFilter(params.N,params.K)
        filtered_states, filtered_covs, log_likelihood = bf.filter_scan(params, returns)

        # Extract filtered factors and volatilities
        filtered_factors = bf.get_filtered_factors()
        filtered_log_vols = bf.get_filtered_volatilities()

        # Create figure with subplots for factors and log-volatilities
        fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True) # Share x-axis

        # Plot factors
        for k in range(params.K):
            ax = axs[0, k]
            ax.plot(true_factors[:, k], "b-", label="True", alpha=0.8)
            ax.plot(filtered_factors[:, k], "r--", label="Filtered")
            ax.set_title(f"Factor {k+1}")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

        # Plot log-volatilities
        for k in range(params.K):
            ax = axs[1, k]
            ax.plot(true_log_vols[:, k], "b-", label="True", alpha=0.8)
            ax.plot(filtered_log_vols[:, k], "r--", label="Filtered")
            ax.set_title(f"Log-Volatility {k+1}")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlabel("Time Step") # Add x-label to bottom row

        fig.suptitle(
            "Bellman Filter Performance: True vs. Filtered States", fontsize=16
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

        # Save the figure if a path is provided
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Visual comparison saved to {save_path}")
            except Exception as e:
                print(f"Failed to save figure: {e}")

        return fig

if __name__ == "__main__":
    unittest.main()
