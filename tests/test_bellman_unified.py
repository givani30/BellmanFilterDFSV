"""
Unified test suite for all Bellman filter implementations.

This module combines tests for different implementations of the Bellman filter,
including the standard implementation and the JAX-based implementation.
"""

import sys
import os
import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from functions.bellman_filter import DFSVBellmanFilter
# Update import to use models.dfsv
from models.dfsv import DFSV_params
from functions.simulation import simulate_DFSV


class TestBasicBellmanFilter(unittest.TestCase):
    """
    Basic tests for the standard Bellman filter implementation.
    """

    def test_bellman_single_step(self):
        """Test a single step of prediction and update."""
        # Set random seed for reproducibility
        np.random.seed(42)

        # Create test parameters
        N = 5  # Number of observed series
        K = 2  # Number of factors

        # Initialize model parameters
        params = DFSV_params(
            N=N,
            K=K,
            lambda_r=np.random.normal(0, 1, (N, K)),  # Random factor loadings
            Phi_f=0.9 * np.eye(K),  # AR(1) factor dynamics
            Phi_h=0.95 * np.eye(K),  # Persistent volatility
            mu=np.zeros(K),  # Mean log-volatility
            Q_h=0.1 * np.eye(K),  # Volatility innovation variance
            sigma2=0.1 * np.ones(N),  # Measurement noise variance
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
        updated_state, updated_cov, log_likelihood = bf.update(params,
            predicted_state, predicted_cov, y
        )

        # Basic assertions
        self.assertEqual(initial_state.shape, (2 * K, 1), "Incorrect state shape")
        self.assertEqual(
            initial_cov.shape, (2 * K, 2 * K), "Incorrect covariance shape"
        )
        self.assertIsNotNone(log_likelihood, "Log-likelihood should not be None")
        self.assertFalse(np.any(np.isnan(updated_state)), "State contains NaN values")
        self.assertFalse(
            np.any(np.isnan(updated_cov)), "Covariance contains NaN values"
        )

    def test_bellman_full_filter(self):
        """
        Test the full filter operation on a small dataset.
        """
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

        # Generate a small sample of simulated data
        T = 20
        y, factors, log_vols = simulate_DFSV(params, T=T, seed=42)

        # Initialize the Bellman filter
        bf = DFSVBellmanFilter(N,K)

        # Run the filter
        filtered_states, filtered_covs, log_likelihood = bf.filter(params,y)

        # Basic assertions for filtered results
        self.assertEqual(
            filtered_states.shape,
            (T, 2 * K),
            f"Filtered states have wrong shape: {filtered_states.shape}",
        )
        self.assertEqual(len(filtered_covs), T, f"Should have {T} covariance matrices")
        self.assertIsInstance(log_likelihood, float, "Log-likelihood should be a float")
        self.assertFalse(
            np.any(np.isnan(filtered_states)), "Filtered states contain NaN values"
        )
        self.assertFalse(
            np.any([np.any(np.isnan(cov)) for cov in filtered_covs]),
            "Filtered covariances contain NaN values",
        )


class TestComprehensiveBellmanFilter(unittest.TestCase):
    """Test suite for comprehensive validation of the DFSVBellmanFilter class."""

    def create_test_parameters(self):
        """
        Create a set of test parameters for a small DFSV model.
        Returns
        -------
        DFSV_params
            Parameters for a 2-factor model with 4 observed series
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

    def test_bellman_filter_estimation(self):
        """
        Test the Bellman filter's ability to estimate states from simulated data.
        This test:
        1. Generates synthetic time series data from a known DFSV model
        2. Applies the Bellman filter to estimate states
        3. Checks that the estimates are reasonably close to the true states
        """
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

        # Create and run Bellman filter
        bf = DFSVBellmanFilter(params.N,params.K)
        filtered_states, filtered_covs, log_likelihood = bf.filter(params,returns)

        # Extract factor and volatility estimates
        filtered_factors = bf.get_filtered_factors()
        filtered_log_vols = bf.get_filtered_volatilities()

        # Test: Check correlation between true and filtered factors
        for k in range(params.K):
            corr = np.corrcoef(true_factors[:, k], filtered_factors[:, k])[0, 1]
            self.assertGreater(corr, 0.7, f"Factor {k} correlation too low: {corr}")

        # Test: Check correlation between true and filtered log-volatilities
        for k in range(params.K):
            corr = np.corrcoef(true_log_vols[:, k], filtered_log_vols[:, k])[0, 1]
            self.assertGreater(
                corr, 0.5, f"Log-volatility {k} correlation too low: {corr}"
            )

        # Test: Check the average estimation error is within reasonable bounds
        factor_rmse = np.sqrt(np.mean((true_factors - filtered_factors) ** 2))
        vol_rmse = np.sqrt(np.mean((true_log_vols - filtered_log_vols) ** 2))

        self.assertLess(factor_rmse, 0.6, f"Factor RMSE too high: {factor_rmse}")
        self.assertLess(vol_rmse, 1.5, f"Log-volatility RMSE too high: {vol_rmse}")

    def test_bellman_filter_numeric_stability(self):
        """
        Test the Bellman filter's numerical stability with a longer time series.
        This test focuses on making sure the filter doesn't break down over long
        sequences and maintains reasonable estimation quality.
        """
        # Create test parameters
        params = self.create_test_parameters()

        # Simulate a longer series
        T = 500
        returns, true_factors, true_log_vols = simulate_DFSV(params, T=T, seed=123)

        # Create and run Bellman filter
        bf = DFSVBellmanFilter(params.N,params.K)
        filtered_states, filtered_covs, log_likelihood = bf.filter(params,returns)

        # Check that the filter completes without errors and returns valid estimates
        self.assertFalse(
            np.any(np.isnan(filtered_states)), "Filter produced NaN values"
        )
        self.assertFalse(
            np.any(np.isinf(filtered_states)), "Filter produced infinite values"
        )
        self.assertIsInstance(
            log_likelihood, float, "Log-likelihood should be a valid float"
        )

    def test_visualization(self):
        """
        Test to generate and save visual comparison of Bellman filter results.
        """
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

    def create_visual_comparison(self, save_path=None):
        """
        Create visual comparisons between true and filtered states.
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
        params = self.create_test_parameters()

        # Simulate data
        T = 300
        returns, true_factors, true_log_vols = simulate_DFSV(params, T=T, seed=456)

        # Create and run Bellman filter
        bf = DFSVBellmanFilter(params.N,params.K)
        filtered_states, filtered_covs, log_likelihood = bf.filter(params,returns)

        # Extract filtered factors and volatilities
        filtered_factors = bf.get_filtered_factors()
        filtered_log_vols = bf.get_filtered_volatilities()

        # Create figure with subplots for factors and log-volatilities
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Plot factors
        for k in range(params.K):
            ax = axs[0, k]
            ax.plot(true_factors[:, k], "b-", label="True")
            ax.plot(filtered_factors[:, k], "r--", label="Filtered")
            ax.set_title(f"Factor {k+1}")
            ax.legend()
            ax.grid(True)

        # Plot log-volatilities
        for k in range(params.K):
            ax = axs[1, k]
            ax.plot(true_log_vols[:, k], "b-", label="True")
            ax.plot(filtered_log_vols[:, k], "r--", label="Filtered")
            ax.set_title(f"Log-Volatility {k+1}")
            ax.legend()
            ax.grid(True)

        fig.suptitle(
            "Bellman Filter Performance: True vs. Filtered States", fontsize=16
        )
        plt.tight_layout()

        # Save the figure if a path is provided
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            except Exception as e:
                print(f"Failed to save figure: {e}")

        return fig
