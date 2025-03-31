"""
Tests for the DFSV parameter dataclass in the models.dfsv module.

This module contains tests for the JAX-compatible DFSVParamsDataclass.
"""

import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Remove sys.path hack
# sys.path.append(str(Path(__file__).parent.parent))

# Updated imports
from qf_thesis.models.dfsv import DFSVParamsDataclass, dfsv_params_to_dict # Removed DFSV_params


class TestDFSVModels(unittest.TestCase):
    """Tests for the DFSVParamsDataclass."""

    def setUp(self):
        """Set up test parameters with correct dimensions."""
        self.N, self.K = 3, 2  # Small dimensions for testing

        # Create arrays with correct dimensions
        self.lambda_r = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
        self.Phi_f = np.array([[0.95, 0.05], [0.05, 0.9]])
        self.Phi_h = np.array([[0.98, 0.0], [0.0, 0.95]])
        self.mu = np.array([[-1.0], [-0.5]])
        self.sigma2 = np.array([0.1, 0.1, 0.1])
        self.Q_h = np.array([[0.05, 0.01], [0.01, 0.05]])

    # Removed tests related to the old DFSV_params class:
    # - test_dfsv_params_initialization
    # - test_dfsv_params_validation
    # - test_dfsv_params_no_validation
    # - test_sigma2_handling
    def test_jax_dataclass_initialization(self):
        """Test initialization of the JAX-compatible dataclass."""
        try:
            import jax
            import jax.numpy as jnp
            
            # Removed creation of np_params and conversion via from_dfsv_params
            # Test direct initialization instead
            
            # Test direct initialization
            direct_jax_params = DFSVParamsDataclass(
                N=self.N,
                K=self.K,
                lambda_r=jnp.array(self.lambda_r),
                Phi_f=jnp.array(self.Phi_f),
                Phi_h=jnp.array(self.Phi_h),
                mu=jnp.array(self.mu),
                sigma2=jnp.array(self.sigma2), # Initialize with 1D array
                Q_h=jnp.array(self.Q_h)
            )
            
            self.assertEqual(direct_jax_params.N, self.N)
            self.assertEqual(direct_jax_params.K, self.K)
            
        except ImportError:
            self.skipTest("JAX not available, skipping JAX dataclass tests")

    # Removed test_jax_dataclass_to_original as to_dfsv_params method is deleted.
    def test_jax_replace(self):
        """Test the replace method of the JAX dataclass."""
        try:
            import jax
            import jax.numpy as jnp
            
            # Create JAX params
            jax_params = DFSVParamsDataclass(
                N=self.N,
                K=self.K,
                lambda_r=jnp.array(self.lambda_r),
                Phi_f=jnp.array(self.Phi_f),
                Phi_h=jnp.array(self.Phi_h),
                mu=jnp.array(self.mu),
                sigma2=jnp.array(self.sigma2), # Use 1D array
                Q_h=jnp.array(self.Q_h)
            )
            
            # Create new mu and Phi_f values
            new_mu = jnp.array([[-0.5], [-0.2]])
            new_Phi_f = jnp.array([[0.8, 0.1], [0.1, 0.85]])
            
            # Replace the values
            updated_params = jax_params.replace(mu=new_mu, Phi_f=new_Phi_f)
            
            # Check that only the specified parameters were updated
            np.testing.assert_allclose(np.array(updated_params.mu), np.array(new_mu))
            np.testing.assert_allclose(np.array(updated_params.Phi_f), np.array(new_Phi_f))
            
            # Check that other parameters remain unchanged
            np.testing.assert_allclose(np.array(updated_params.lambda_r), self.lambda_r)
            np.testing.assert_allclose(np.array(updated_params.Phi_h), self.Phi_h)
            np.testing.assert_allclose(np.array(updated_params.sigma2), self.sigma2)
            np.testing.assert_allclose(np.array(updated_params.Q_h), self.Q_h)
            
            # Check that the original object is not modified
            np.testing.assert_allclose(np.array(jax_params.mu), self.mu)
            np.testing.assert_allclose(np.array(jax_params.Phi_f), self.Phi_f)
            
        except ImportError:
            self.skipTest("JAX not available, skipping JAX dataclass tests")

    def test_to_from_dict(self):
        """Test conversion to and from dictionary representation."""
        try:
            import jax
            import jax.numpy as jnp
            
            # Create JAX params
            jax_params = DFSVParamsDataclass(
                N=self.N,
                K=self.K,
                lambda_r=jnp.array(self.lambda_r),
                Phi_f=jnp.array(self.Phi_f),
                Phi_h=jnp.array(self.Phi_h),
                mu=jnp.array(self.mu),
                sigma2=jnp.array(self.sigma2), # Use 1D array
                Q_h=jnp.array(self.Q_h)
            )
            
            # Convert to dict
            param_dict, N, K = jax_params.to_dict()
            
            # Check that N and K are correct
            self.assertEqual(N, self.N)
            self.assertEqual(K, self.K)
            
            # Check that all parameters are in the dictionary
            self.assertIn("lambda_r", param_dict)
            self.assertIn("Phi_f", param_dict)
            self.assertIn("Phi_h", param_dict)
            self.assertIn("mu", param_dict)
            self.assertIn("sigma2", param_dict)
            self.assertIn("Q_h", param_dict)
            
            # Convert back to DFSVParamsDataclass
            recreated_params = DFSVParamsDataclass.from_dict(param_dict, N, K)
            
            # Check that dimensions and values are preserved
            self.assertEqual(recreated_params.N, self.N)
            self.assertEqual(recreated_params.K, self.K)
            np.testing.assert_allclose(np.array(recreated_params.lambda_r), self.lambda_r)
            np.testing.assert_allclose(np.array(recreated_params.mu), self.mu)
            
        except ImportError:
            self.skipTest("JAX not available, skipping JAX dataclass tests")

    def test_dfsv_params_to_dict_function(self):
        """Test the utility function for converting to dictionary."""
        try:
            import jax
            import jax.numpy as jnp
            
            # Create JAX params
            jax_params = DFSVParamsDataclass(
                N=self.N,
                K=self.K,
                lambda_r=jnp.array(self.lambda_r),
                Phi_f=jnp.array(self.Phi_f),
                Phi_h=jnp.array(self.Phi_h),
                mu=jnp.array(self.mu),
                sigma2=jnp.array(self.sigma2), # Use 1D array
                Q_h=jnp.array(self.Q_h)
            )
            
            # Test with DFSVParamsDataclass
            param_dict, N, K = dfsv_params_to_dict(jax_params)
            self.assertEqual(N, self.N)
            self.assertEqual(K, self.K)
            self.assertIn("lambda_r", param_dict)
            
            # Test with dictionary
            test_dict = {
                "N": self.N,
                "K": self.K,
                "lambda_r": self.lambda_r,
                "Phi_f": self.Phi_f,
                "Phi_h": self.Phi_h,
                "mu": self.mu,
                "sigma2": self.sigma2,
                "Q_h": self.Q_h
            }
            
            param_dict, N, K = dfsv_params_to_dict(test_dict)
            self.assertEqual(N, self.N)
            self.assertEqual(K, self.K)
            self.assertIn("lambda_r", param_dict)
            self.assertNotIn("N", param_dict)
            self.assertNotIn("K", param_dict)
            
            # Test with unsupported type
            with self.assertRaises(TypeError):
                dfsv_params_to_dict(123)
                
        except ImportError:
            self.skipTest("JAX not available, skipping JAX dataclass tests")


if __name__ == "__main__":
    unittest.main()