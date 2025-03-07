"""
Unit tests for the Particle Filter implementation in DFSV models.

This module provides tests to validate the particle filter implementation,
including numerical accuracy tests and visualization tests.
"""

import sys
import os
import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Updated imports to use functions directory
from functions.filters import DFSVParticleFilter
from functions.simulation import DFSV_params, simulate_DFSV


class TestParticleFilter(unittest.TestCase):
    """Test suite for the DFSVParticleFilter class."""
    
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
        lambda_r = np.array([
            [0.8, 0.2],
            [0.7, 0.3],
            [0.3, 0.7],
            [0.2, 0.8]
        ])
        
        # Factor persistence
        Phi_f = np.array([
            [0.9, 0.1],
            [0.1, 0.9]
        ])
        
        # Log-volatility persistence
        Phi_h = np.array([
            [0.95, 0.0],
            [0.0, 0.95]
        ])
        
        # Log-volatility long-run mean
        mu = np.array([-1.0, -0.5])
        
        # Idiosyncratic variance (diagonal)
        sigma2 = np.array([0.1, 0.1, 0.1, 0.1])
        
        # Log-volatility noise covariance
        Q_h = np.array([
            [0.1, 0.02],
            [0.02, 0.1]
        ])
        
        # Create parameter object
        params = DFSV_params(
            N=N,
            K=K,
            lambda_r=lambda_r,
            Phi_f=Phi_f,
            Phi_h=Phi_h,
            mu=mu,
            sigma2=sigma2,
            Q_h=Q_h
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
        
        # Create test parameters
        params = self.create_test_parameters()
        
        # Simulate data
        T = 200  # Time series length
        returns, true_factors, true_log_vols = simulate_DFSV(params, T=T, seed=42)
        
        # Create and run particle filter
        pf = DFSVParticleFilter(params, num_particles=500)
        filtered_states, filtered_covs, log_likelihood = pf.filter(returns)
        
        # Extract factor and volatility estimates
        filtered_factors = pf.get_filtered_factors()
        filtered_log_vols = pf.get_filtered_volatilities()
        
        # Test: Check correlation between true and filtered factors
        for k in range(params.K):
            corr = np.corrcoef(true_factors[:, k], filtered_factors[:, k])[0, 1]
            self.assertGreater(corr, 0.7, f"Factor {k} correlation too low: {corr}")
        
        # Test: Check correlation between true and filtered log-volatilities
        for k in range(params.K):
            corr = np.corrcoef(true_log_vols[:, k], filtered_log_vols[:, k])[0, 1]
            self.assertGreater(corr, 0.6, f"Log-volatility {k} correlation too low: {corr}")
        
        # Test: Check the average estimation error is within reasonable bounds
        factor_rmse = np.sqrt(np.mean((true_factors - filtered_factors)**2, axis=0))
        vol_rmse = np.sqrt(np.mean((true_log_vols - filtered_log_vols)**2, axis=0))
        
        self.assertLess(factor_rmse, 0.5, f"Factor RMSE too high: {factor_rmse}")
        self.assertLess(vol_rmse, 1.0, f"Log-volatility RMSE too high: {vol_rmse}")
        
        # Print additional information (won't affect test outcome but helpful for analysis)
        print(f"Factor correlation: {np.corrcoef(true_factors.flatten(), filtered_factors.flatten())[0, 1]:.4f}")
        print(f"Log-volatility correlation: {np.corrcoef(true_log_vols.flatten(), filtered_log_vols.flatten())[0, 1]:.4f}")
        print(f"Factor RMSE: {factor_rmse:.4f}")
        print(f"Log-volatility RMSE: {vol_rmse:.4f}")

    def test_particle_filter_numeric_stability(self):
        """
        Test the particle filter's numerical stability with a longer time series.
        
        This test focuses on making sure the filter doesn't break down over long
        sequences and maintains reasonable estimation quality.
        """
        # Create test parameters
        params = self.create_test_parameters()
        
        # Simulate a longer series
        T = 500
        returns, true_factors, true_log_vols = simulate_DFSV(params, T=T, seed=123)
        
        # Create and run particle filter with more particles for stability
        pf = DFSVParticleFilter(params, num_particles=1000)
        filtered_states, filtered_covs, log_likelihood = pf.filter(returns)
        
        # Check that the filter completes without errors and returns valid estimates
        self.assertFalse(np.any(np.isnan(filtered_states)), "Filter produced NaN values")
        self.assertFalse(np.any(np.isinf(filtered_states)), "Filter produced infinite values")
        self.assertIsInstance(log_likelihood, float, "Log-likelihood is not a valid float")

    def test_visualization(self):
        """
        Test to generate and save visual comparison of particle filter results.
        """
        # Create an absolute path for the output file
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
        save_path = os.path.join(output_dir, "pf_visual_comparison.png")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Generate visual comparison and save to file
        fig = self.create_visual_comparison(save_path=save_path)
        
        # Basic assertion to ensure the figure was created
        self.assertIsInstance(fig, plt.Figure)
        
        # Verify the file was actually saved
        self.assertTrue(os.path.exists(save_path), f"Figure was not saved to {save_path}")
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
        
        # Create test parameters
        params = self.create_test_parameters()
        
        # Simulate data
        T = 300
        returns, true_factors, true_log_vols = simulate_DFSV(params, T=T, seed=456)
        
        # Create and run particle filter
        pf = DFSVParticleFilter(params, num_particles=10000)
        filtered_states, filtered_covs, log_likelihood = pf.filter(returns)
        
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
        
        # Create figure with subplots for factors and log-volatilities
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot factors
        for k in range(params.K):
            ax = axs[0, k]
            ax.plot(true_factors[:, k], 'b-', label='True')
            ax.plot(filtered_factors[:, k], 'r--', label='Filtered')
            if include_smoothed:
                ax.plot(smoothed_factors[:, k], 'g-.', label='Smoothed')
            ax.set_title(f'Factor {k+1}')
            ax.legend()
            ax.grid(True)
        
        # Plot log-volatilities
        for k in range(params.K):
            ax = axs[1, k]
            ax.plot(true_log_vols[:, k], 'b-', label='True')
            ax.plot(filtered_log_vols[:, k], 'r--', label='Filtered')
            if include_smoothed:
                ax.plot(smoothed_log_vols[:, k], 'g-.', label='Smoothed')
            ax.set_title(f'Log-Volatility {k+1}')
            ax.legend()
            ax.grid(True)
        
        fig.suptitle('Particle Filter Performance: True vs. Filtered vs. Smoothed States', fontsize=16)
        plt.tight_layout()
        
        # Make sure to save the figure if a path is provided
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Figure successfully saved to {save_path}")
            except Exception as e:
                print(f"Failed to save figure: {e}")
        
        return fig


if __name__ == "__main__":
    unittest.main()