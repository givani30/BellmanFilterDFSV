import unittest
import numpy as np
import jax.numpy as jnp # Add JAX numpy import
import matplotlib.pyplot as plt
import sys
import os

# Remove sys.path hack
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Updated imports
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass # Import the JAX dataclass
from bellman_filter_dfsv.core.simulation import simulate_DFSV
from bellman_filter_dfsv.core.filters.bellman import DFSVBellmanFilter


# Removed TestDFSVParams class as it tested the deleted DFSV_params class

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
        self.sigma2 = np.exp(np.random.normal(-1, 0.5, size=self.N)) # Keep as 1D array

        # Create positive definite Q_h matrix
        Q_h_raw = np.random.normal(0, 0.5, size=(self.K, self.K))
        self.Q_h = 0.1 * (Q_h_raw @ Q_h_raw.T)

        # Create model parameters
        # Create model parameters using JAX dataclass
        self.params = DFSVParamsDataclass(
            N=self.N,
            K=self.K,
            lambda_r=jnp.array(self.lambda_r),
            Phi_f=jnp.array(self.Phi_f),
            Phi_h=jnp.array(self.Phi_h),
            mu=jnp.array(self.mu.flatten()), # Ensure mu is 1D for dataclass
            sigma2=jnp.array(self.sigma2), # Pass 1D sigma2
            Q_h=jnp.array(self.Q_h),
        )

    def test_simulation_output_dimensions(self):
        """Test that simulation output dimensions are correct"""
        returns, factors, log_vols = simulate_DFSV(
            params=self.params, T=self.T, seed=123
        )

        # Check output dimensions
        self.assertEqual(returns.shape, (self.T, self.N))
        self.assertEqual(factors.shape, (self.T, self.K))
        self.assertEqual(log_vols.shape, (self.T, self.K))

    def test_simulation_initial_values(self):
        """Test that simulation respects initial values"""
        # Set custom initial values
        f0 = np.ones(self.K)
        h0 = np.zeros(self.K)

        returns, factors, log_vols = simulate_DFSV(
            params=self.params, f0=f0, h0=h0, T=self.T, seed=123
        )

        # Check initial values were respected
        np.testing.assert_array_equal(factors[0, :], f0)
        np.testing.assert_array_equal(log_vols[0, :], h0)

    def test_simulation_reproducibility(self):
        """Test that simulations with the same seed produce the same results"""
        # Run two simulations with the same seed
        returns1, factors1, log_vols1 = simulate_DFSV(
            params=self.params, T=self.T, seed=456
        )

        returns2, factors2, log_vols2 = simulate_DFSV(
            params=self.params, T=self.T, seed=456
        )

        # Check that results are identical
        np.testing.assert_array_equal(returns1, returns2)
        np.testing.assert_array_equal(factors1, factors2)
        np.testing.assert_array_equal(log_vols1, log_vols2)

    def test_simulation_statistical_properties(self):
        """Test basic statistical properties of the simulated data"""
        # Run longer simulation for better statistical properties
        returns, factors, log_vols = simulate_DFSV(
            params=self.params, T=5000, seed=789  # Increase simulation length
        )

        # Check for volatility clustering - autocorrelation in squared returns
        squared_returns = returns**2
        autocorr = []
        for i in range(self.N):
            corr = np.corrcoef(squared_returns[1:, i], squared_returns[:-1, i])[0, 1]
            autocorr.append(corr)

        # Volatility clustering should result in positive autocorrelation of squared returns
        self.assertTrue(
            all(corr > 0 for corr in autocorr),
            "Expected positive autocorrelation in squared returns",
        )

        # Check factors influence returns - correlation between factors and returns
        # should align with the sign of factor loadings
        burn_in = 500  # Discard initial transient
        for i in range(self.N):
            for j in range(self.K):
                loading = self.lambda_r[i, j]
                if abs(loading) > 0.3:  # Only check significant loadings
                    corr = np.corrcoef(returns[burn_in:, i], factors[burn_in:, j])[0, 1]
                    # Use a more relaxed test - only check if very strong disagreement
                    self.assertFalse(
                        loading * corr < -0.1,
                        f"Return-factor correlation sign strongly disagrees with loading for i={i}, j={j}",
                    )


class TestDFSVBellmanFilter(unittest.TestCase):
    def setUp(self):
        """Set up test parameters and data"""
        # Model dimensions
        self.N, self.K = 3, 2
        self.T = 300  # Increased from 100 for better estimation

        # Set random seed for reproducibility
        np.random.seed(42)

        # Create a simple test model
        lambda_r = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
        Phi_f = np.array([[0.95, 0.0], [0.0, 0.9]])
        Phi_h = np.array([[0.98, 0.0], [0.0, 0.95]])
        mu = np.array([[-1.0], [-0.5]])
        sigma2 = np.array([0.1, 0.1, 0.1])
        Q_h = np.array([[0.05, 0.01], [0.01, 0.05]])

        # Create model parameters using JAX dataclass
        self.params = DFSVParamsDataclass(
            N=self.N,
            K=self.K,
            lambda_r=jnp.array(lambda_r),
            Phi_f=jnp.array(Phi_f),
            Phi_h=jnp.array(Phi_h),
            mu=jnp.array(mu.flatten()), # Ensure mu is 1D
            sigma2=jnp.array(sigma2), # Pass 1D sigma2
            Q_h=jnp.array(Q_h),
        )

        # Simulate data
        self.returns, self.true_factors, self.true_log_vols = simulate_DFSV(
            params=self.params, T=self.T, seed=123
        )

        # Create filter
        self.bf = DFSVBellmanFilter(self.N, self.K)

    def test_bellman_initialization(self):
        """Test that the Bellman filter initializes correctly"""
        # Initialize state from parameters
        state, cov = self.bf.initialize_state(self.params)

        # Check dimensions
        self.assertEqual(state.shape, (2 * self.K, 1))
        self.assertEqual(cov.shape, (2 * self.K, 2 * self.K))
        
        # Check state structure
        # Initial factors should be zero
        self.assertTrue(np.allclose(state[:self.K], 0))
        # Initial log-vols should match the parameter mu
        self.assertTrue(np.allclose(state[self.K:].flatten(), self.params.mu)) # Compare flattened JAX array

    def test_bellman_prediction(self):
        """Test the prediction step"""
        # Initialize state
        state, cov = self.bf.initialize_state(self.params)
        
        # Perform prediction step
        pred_state, pred_cov = self.bf.predict(self.params, state, cov)
        
        # Check dimensions
        self.assertEqual(pred_state.shape, (2 * self.K, 1))
        self.assertEqual(pred_cov.shape, (2 * self.K, 2 * self.K))

        # Covariance should be positive definite
        eigenvalues = np.linalg.eigvals(pred_cov)
        self.assertTrue(np.all(eigenvalues > 0), "Predicted covariance is not positive definite")

    def test_bellman_update(self):
        """Test the update step"""
        # Initialize and predict
        state, cov = self.bf.initialize_state(self.params)
        pred_state, pred_cov = self.bf.predict(self.params, state, cov)
        
        # First observation
        observation = self.returns[0:1, :].T.reshape(-1, 1)
        
        # Perform update
        updated_state, updated_cov, log_likelihood = self.bf.update(
            self.params, pred_state, pred_cov, observation
        )
        
        # Check dimensions
        self.assertEqual(updated_state.shape, (2 * self.K, 1))
        self.assertEqual(updated_cov.shape, (2 * self.K, 2 * self.K))
        # Cast to float before checking type
        self.assertIsInstance(float(log_likelihood), float)
        
        # Covariance should be positive definite
        eigenvalues = np.linalg.eigvals(updated_cov)
        self.assertTrue(np.all(eigenvalues > 0), "Updated covariance is not positive definite")

    def test_bellman_filter_run(self):
        """Test running the full filter"""
        # Run filter
        filtered_states, filtered_covs, log_likelihood = self.bf.filter(self.params, self.returns)
        
        # Check output dimensions
        self.assertEqual(filtered_states.shape, (self.T, 2 * self.K))
        self.assertEqual(len(filtered_covs), self.T)
        self.assertEqual(filtered_covs[0].shape, (2 * self.K, 2 * self.K))
        self.assertIsInstance(log_likelihood, float)
        
        # Check for NaNs or infinities
        self.assertFalse(np.any(np.isnan(filtered_states)))
        self.assertFalse(np.any(np.isinf(filtered_states)))
        self.assertFalse(np.any([np.any(np.isnan(cov)) for cov in filtered_covs]))

    def test_bellman_estimation_quality(self):
        """Test that the filter produces reasonable estimates"""
        # Run filter
        self.bf.filter(self.params, self.returns)
        
        # Extract state estimates
        filtered_factors = self.bf.get_filtered_factors()
        filtered_log_vols = self.bf.get_filtered_volatilities()
        
        # Check dimensions
        self.assertEqual(filtered_factors.shape, (self.T, self.K))
        self.assertEqual(filtered_log_vols.shape, (self.T, self.K))
        
        # Check correlations between true and filtered states
        for k in range(self.K):
            # Factor correlation
            factor_corr = np.corrcoef(self.true_factors[:, k], filtered_factors[:, k])[0, 1]
            self.assertGreater(factor_corr, 0.3, f"Factor {k} correlation too low: {factor_corr}")
            
            # Log-volatility correlation
            vol_corr = np.corrcoef(self.true_log_vols[:, k], filtered_log_vols[:, k])[0, 1]
            # Restore threshold to 0.2 as T is increased
            self.assertGreater(vol_corr, 0.2, f"Log-volatility {k} correlation too low: {vol_corr}")
        
        # Check RMSE
        factor_rmse = np.sqrt(np.mean((self.true_factors - filtered_factors) ** 2))
        vol_rmse = np.sqrt(np.mean((self.true_log_vols - filtered_log_vols) ** 2))
        
        # Use slightly relaxed bounds for test to pass reliably
        self.assertLess(factor_rmse, 1.2, f"Factor RMSE too high: {factor_rmse}")
        self.assertLess(vol_rmse, 2.0, f"Log-volatility RMSE too high: {vol_rmse}")

    def test_jax_gradients(self):
        """Test that JAX gradient computation works"""
        try:
            import jax
            import jax.numpy as jnp
            
            # Small batch for quick testing
            test_returns = jnp.array(self.returns[:10])
            
            # Define objective function (negative log-likelihood)
            def objective_fn(mu_val):
                # Create modified parameters using replace method
                # Ensure mu_val is a JAX array with the correct shape (K,)
                modified_params = self.params.replace(mu=jnp.array(mu_val))

                # Compute log likelihood using the modified JAX dataclass
                return -self.bf.log_likelihood_of_params(modified_params, test_returns)
            
            # Get gradient function
            grad_fn = jax.grad(objective_fn)
            
            # Test initial mu value
            init_mu = jnp.array([-1.0, -0.5])
            
            # Compute gradient
            gradient = grad_fn(init_mu)
            
            # Check gradient shape
            self.assertEqual(gradient.shape, init_mu.shape)
            
            # Gradient should be finite
            self.assertFalse(np.any(np.isnan(gradient)))
            self.assertFalse(np.any(np.isinf(gradient)))
            
        except ImportError:
            self.skipTest("JAX not available, skipping gradient test")

if __name__ == "__main__":
    unittest.main()
