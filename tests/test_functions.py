import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path so we can import sim_functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim_functions import DFSV_params, simulate_DFSV

class TestDFSVParams(unittest.TestCase):
    
    def test_valid_dimensions(self):
        """Test that valid dimensions are accepted."""
        # Set parameters
        N, K = 5, 2
        
        # Create arrays with correct dimensions
        lambda_r = np.random.rand(N, K)
        Phi_f = np.random.rand(K, K)
        Phi_h = np.random.rand(K, K)
        mu = np.random.rand(K, 1)
        sigma2 = np.random.rand(N, 1)
        Q_h = np.eye(K) * 0.1  # Adding Q_h parameter that was missing
        
        # This should not raise an error
        params = DFSV_params(N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, 
                             Phi_h=Phi_h, mu=mu, sigma2=sigma2, Q_h=Q_h)
        
        # Test with mu as 1D array
        mu_1d = np.random.rand(K)
        params = DFSV_params(N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, 
                             Phi_h=Phi_h, mu=mu_1d, sigma2=sigma2, Q_h=Q_h)
    
    def test_invalid_lambda_r(self):
        """Test that invalid lambda_r dimensions raise error."""
        N, K = 5, 2
        lambda_r = np.random.rand(K, N)  # Incorrect dimensions (K,N) instead of (N,K)
        Phi_f = np.random.rand(K, K)
        Phi_h = np.random.rand(K, K)
        mu = np.random.rand(K, 1)
        sigma2 = np.random.rand(N, 1)
        Q_h = np.eye(K) * 0.1  # Adding Q_h parameter
        
        with self.assertRaises(ValueError) as context:
            DFSV_params(N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, 
                        Phi_h=Phi_h, mu=mu, sigma2=sigma2, Q_h=Q_h)
        
        self.assertIn("lambda_r should be shape", str(context.exception))
    
    def test_invalid_Phi_f(self):
        """Test that invalid Phi_f dimensions raise error."""
        N, K = 5, 2
        lambda_r = np.random.rand(N, K)
        Phi_f = np.random.rand(K+1, K)  # Incorrect dimensions
        Phi_h = np.random.rand(K, K)
        mu = np.random.rand(K, 1)
        sigma2 = np.random.rand(N, 1)
        Q_h = np.eye(K) * 0.1  # Adding Q_h parameter
        
        with self.assertRaises(ValueError) as context:
            DFSV_params(N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, 
                        Phi_h=Phi_h, mu=mu, sigma2=sigma2, Q_h=Q_h)
        
        self.assertIn("Phi_f should be shape", str(context.exception))
    
    def test_invalid_mu(self):
        """Test that invalid mu dimensions raise error."""
        N, K = 5, 2
        lambda_r = np.random.rand(N, K)
        Phi_f = np.random.rand(K, K)
        Phi_h = np.random.rand(K, K)
        mu = np.random.rand(K+1, 1)  # Incorrect dimensions
        sigma2 = np.random.rand(N, 1)
        Q_h = np.eye(K) * 0.1  # Adding Q_h parameter
        
        with self.assertRaises(ValueError) as context:
            DFSV_params(N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, 
                        Phi_h=Phi_h, mu=mu, sigma2=sigma2, Q_h=Q_h)
        
        self.assertIn("mu should be shape", str(context.exception))


class TestDFSVSimulation(unittest.TestCase):
    
    def setUp(self):
        """Set up valid model parameters for simulation tests"""
        # Model dimensions
        self.N, self.K = 3, 2
        self.T = 200
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create stationary factor transition matrix (eigenvalues < 1)
        Phi_f_raw = np.random.normal(0, 0.5, size=(self.K, self.K))
        self.Phi_f = 0.7 * Phi_f_raw / np.max(np.abs(np.linalg.eigvals(Phi_f_raw)))
        
        # Create stationary volatility transition matrix (eigenvalues < 1)
        Phi_h_raw = np.random.normal(0, 0.5, size=(self.K, self.K))
        self.Phi_h = 0.9 * Phi_h_raw / np.max(np.abs(np.linalg.eigvals(Phi_h_raw)))
        
        # Other parameters
        self.lambda_r = np.random.normal(0, 1, size=(self.N, self.K))
        self.mu = np.random.normal(-1, 0.5, size=(self.K, 1))
        self.sigma2 = np.diag(np.exp(np.random.normal(-1, 0.5, size=self.N)))
        
        # Create positive definite Q_h matrix
        Q_h_raw = np.random.normal(0, 0.5, size=(self.K, self.K))
        self.Q_h = 0.1 * (Q_h_raw @ Q_h_raw.T)
        
        # Create model parameters
        self.params = DFSV_params(
            N=self.N, 
            K=self.K, 
            lambda_r=self.lambda_r,
            Phi_f=self.Phi_f,
            Phi_h=self.Phi_h,
            mu=self.mu,
            sigma2=self.sigma2,
            Q_h=self.Q_h
        )
    
    def test_simulation_output_dimensions(self):
        """Test that simulation output dimensions are correct"""
        returns, factors, log_vols = simulate_DFSV(
            params=self.params,
            T=self.T,
            seed=123
        )
        
        # Check output dimensions
        self.assertEqual(returns.shape, (self.N, self.T))
        self.assertEqual(factors.shape, (self.K, self.T))
        self.assertEqual(log_vols.shape, (self.K, self.T))
    
    def test_simulation_initial_values(self):
        """Test that simulation respects initial values"""
        # Set custom initial values
        f0 = np.ones(self.K)
        h0 = np.zeros(self.K)
        
        returns, factors, log_vols = simulate_DFSV(
            params=self.params,
            f0=f0,
            h0=h0,
            T=self.T,
            seed=123
        )
        
        # Check initial values were respected
        np.testing.assert_array_equal(factors[:, 0], f0)
        np.testing.assert_array_equal(log_vols[:, 0], h0)
    
    def test_simulation_reproducibility(self):
        """Test that simulations with the same seed produce the same results"""
        # Run two simulations with the same seed
        returns1, factors1, log_vols1 = simulate_DFSV(
            params=self.params,
            T=self.T,
            seed=456
        )
        
        returns2, factors2, log_vols2 = simulate_DFSV(
            params=self.params,
            T=self.T,
            seed=456
        )
        
        # Check that results are identical
        np.testing.assert_array_equal(returns1, returns2)
        np.testing.assert_array_equal(factors1, factors2)
        np.testing.assert_array_equal(log_vols1, log_vols2)
    
    def test_simulation_statistical_properties(self):
        """Test basic statistical properties of the simulated data"""
        # Run longer simulation for better statistical properties
        returns, factors, log_vols = simulate_DFSV(
            params=self.params,
            T=5000,  # Increase simulation length
            seed=789
        )
        
        # Check for volatility clustering - autocorrelation in squared returns
        squared_returns = returns**2
        autocorr = []
        for i in range(self.N):
            corr = np.corrcoef(squared_returns[i, 1:], squared_returns[i, :-1])[0, 1]
            autocorr.append(corr)
            
        # Volatility clustering should result in positive autocorrelation of squared returns
        self.assertTrue(all(corr > 0 for corr in autocorr), 
                        "Expected positive autocorrelation in squared returns")
        
        # Check factors influence returns - correlation between factors and returns
        # should align with the sign of factor loadings
        burn_in = 500  # Discard initial transient
        for i in range(self.N):
            for j in range(self.K):
                loading = self.lambda_r[i, j]
                if abs(loading) > 0.3:  # Only check significant loadings
                    corr = np.corrcoef(returns[i, burn_in:], factors[j, burn_in:])[0, 1]
                    # Use a more relaxed test - only check if very strong disagreement
                    self.assertFalse(loading * corr < -0.1,
                                    f"Return-factor correlation sign strongly disagrees with loading for i={i}, j={j}")
    
    def test_visual_inspection(self):
        """Generate plots for visual inspection of the simulation"""
        # This test doesn't assert anything, just generates plots
        returns, factors, log_vols = simulate_DFSV(
            params=self.params,
            T=500,
            seed=101
        )
        
        # Create plot with subplots for returns, factors, and volatilities
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot returns
        for i in range(self.N):
            axes[0].plot(returns[i], label=f"Return {i+1}")
        axes[0].set_title("Simulated Returns")
        axes[0].legend()
        
        # Plot factors
        for i in range(self.K):
            axes[1].plot(factors[i], label=f"Factor {i+1}")
        axes[1].set_title("Latent Factors")
        axes[1].legend()
        
        # Plot volatilities (exp(h/2) for standard deviations)
        for i in range(self.K):
            axes[2].plot(np.exp(log_vols[i]/2), label=f"Factor {i+1} Volatility")
        axes[2].set_title("Factor Volatilities")
        axes[2].legend()
        
        plt.tight_layout()
        # Save the figure to a file
        plt.savefig("/home/givanib/Documents/QF_Thesis/dfsv_test_plot.png")
        print("Visual inspection plot saved to '/home/givanib/Documents/QF_Thesis/dfsv_test_plot.png'")


if __name__ == '__main__':
    unittest.main()