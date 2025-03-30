"""
Tests for the DFSV parameter classes in the models.dfsv module.

This module contains comprehensive tests for both the NumPy-based DFSV_params class
and the JAX-compatible DFSVParamsDataclass.
"""

import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path to import from our modules
sys.path.append(str(Path(__file__).parent.parent))

from models.dfsv import DFSV_params, DFSVParamsDataclass, dfsv_params_to_dict


class TestDFSVModels(unittest.TestCase):
    """Tests for the DFSV parameter classes."""

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

    def test_dfsv_params_initialization(self):
        """Test initialization of DFSV_params with valid parameters."""
        params = DFSV_params(
            N=self.N,
            K=self.K,
            lambda_r=self.lambda_r,
            Phi_f=self.Phi_f,
            Phi_h=self.Phi_h,
            mu=self.mu,
            sigma2=self.sigma2,
            Q_h=self.Q_h
        )

        # Check that parameters were stored correctly
        np.testing.assert_array_equal(params.lambda_r, self.lambda_r)
        np.testing.assert_array_equal(params.Phi_f, self.Phi_f)
        np.testing.assert_array_equal(params.Phi_h, self.Phi_h)
        np.testing.assert_array_equal(params.mu, self.mu)
        # sigma2 is converted to diagonal matrix if 1D
        np.testing.assert_array_equal(params.sigma2, np.diag(self.sigma2))
        np.testing.assert_array_equal(params.Q_h, self.Q_h)

    def test_dfsv_params_validation(self):
        """Test that validation catches dimension mismatches."""
        # Test lambda_r with wrong dimensions
        lambda_r_wrong = np.random.rand(self.K, self.N)  # Transposed dimensions
        with self.assertRaises(ValueError):
            DFSV_params(
                N=self.N,
                K=self.K,
                lambda_r=lambda_r_wrong,  # Wrong dimensions
                Phi_f=self.Phi_f,
                Phi_h=self.Phi_h,
                mu=self.mu,
                sigma2=self.sigma2,
                Q_h=self.Q_h
            )

        # Test Phi_f with wrong dimensions
        Phi_f_wrong = np.random.rand(self.K, self.K + 1)  # Wrong columns
        with self.assertRaises(ValueError):
            DFSV_params(
                N=self.N,
                K=self.K,
                lambda_r=self.lambda_r,
                Phi_f=Phi_f_wrong,  # Wrong dimensions
                Phi_h=self.Phi_h,
                mu=self.mu,
                sigma2=self.sigma2,
                Q_h=self.Q_h
            )

    def test_dfsv_params_no_validation(self):
        """Test initialization without validation."""
        # Create parameters with incorrect dimensions but validation=False
        lambda_r_wrong = np.random.rand(self.K, self.N)  # Transposed dimensions
        
        # This should not raise an error since validation is disabled
        params = DFSV_params(
            N=self.N,
            K=self.K,
            lambda_r=lambda_r_wrong,
            Phi_f=self.Phi_f,
            Phi_h=self.Phi_h,
            mu=self.mu,
            sigma2=self.sigma2,
            Q_h=self.Q_h,
            validate=False
        )
        
        # Check that parameters were stored as-is without validation
        np.testing.assert_array_equal(params.lambda_r, lambda_r_wrong)

    def test_sigma2_handling(self):
        """Test different forms of sigma2 parameter."""
        # Test with 1D array
        sigma2_1d = np.array([0.1, 0.2, 0.3])
        params = DFSV_params(
            N=self.N,
            K=self.K,
            lambda_r=self.lambda_r,
            Phi_f=self.Phi_f,
            Phi_h=self.Phi_h,
            mu=self.mu,
            sigma2=sigma2_1d,
            Q_h=self.Q_h
        )
        np.testing.assert_array_equal(params.sigma2, np.diag(sigma2_1d))
        
        # Test with 2D diagonal matrix
        sigma2_diag = np.diag([0.1, 0.2, 0.3])
        params = DFSV_params(
            N=self.N,
            K=self.K,
            lambda_r=self.lambda_r,
            Phi_f=self.Phi_f,
            Phi_h=self.Phi_h,
            mu=self.mu,
            sigma2=sigma2_diag,
            Q_h=self.Q_h
        )
        np.testing.assert_array_equal(params.sigma2, sigma2_diag)
        
        # Test with full 2D matrix
        sigma2_full = np.array([[0.1, 0.01, 0.01], [0.01, 0.2, 0.01], [0.01, 0.01, 0.3]])
        params = DFSV_params(
            N=self.N,
            K=self.K,
            lambda_r=self.lambda_r,
            Phi_f=self.Phi_f,
            Phi_h=self.Phi_h,
            mu=self.mu,
            sigma2=sigma2_full,
            Q_h=self.Q_h
        )
        np.testing.assert_array_equal(params.sigma2, sigma2_full)

    def test_jax_dataclass_initialization(self):
        """Test initialization of the JAX-compatible dataclass."""
        try:
            import jax
            import jax.numpy as jnp
            
            # Create numpy-based params first
            np_params = DFSV_params(
                N=self.N,
                K=self.K,
                lambda_r=self.lambda_r,
                Phi_f=self.Phi_f,
                Phi_h=self.Phi_h,
                mu=self.mu,
                sigma2=self.sigma2,
                Q_h=self.Q_h
            )
            
            # Convert to JAX dataclass
            jax_params = DFSVParamsDataclass.from_dfsv_params(np_params)
            
            # Check that dimensions and values are preserved
            self.assertEqual(jax_params.N, self.N)
            self.assertEqual(jax_params.K, self.K)
            np.testing.assert_allclose(np.array(jax_params.lambda_r), self.lambda_r)
            np.testing.assert_allclose(np.array(jax_params.Phi_f), self.Phi_f)
            np.testing.assert_allclose(np.array(jax_params.Phi_h), self.Phi_h)
            np.testing.assert_allclose(np.array(jax_params.mu), self.mu)
            np.testing.assert_allclose(np.array(jax_params.sigma2), np.diag(self.sigma2))
            np.testing.assert_allclose(np.array(jax_params.Q_h), self.Q_h)
            
            # Check that arrays are actually JAX arrays
            self.assertIsInstance(jax_params.lambda_r, jnp.ndarray)
            self.assertIsInstance(jax_params.Phi_f, jnp.ndarray)
            
            # Test direct initialization
            direct_jax_params = DFSVParamsDataclass(
                N=self.N,
                K=self.K,
                lambda_r=jnp.array(self.lambda_r),
                Phi_f=jnp.array(self.Phi_f),
                Phi_h=jnp.array(self.Phi_h),
                mu=jnp.array(self.mu),
                sigma2=jnp.array(np.diag(self.sigma2)),
                Q_h=jnp.array(self.Q_h)
            )
            
            self.assertEqual(direct_jax_params.N, self.N)
            self.assertEqual(direct_jax_params.K, self.K)
            
        except ImportError:
            self.skipTest("JAX not available, skipping JAX dataclass tests")

    def test_jax_dataclass_to_original(self):
        """Test conversion from JAX dataclass back to original DFSV_params."""
        try:
            import jax
            import jax.numpy as jnp
            
            # Create numpy-based params first
            np_params = DFSV_params(
                N=self.N,
                K=self.K,
                lambda_r=self.lambda_r,
                Phi_f=self.Phi_f,
                Phi_h=self.Phi_h,
                mu=self.mu,
                sigma2=self.sigma2,
                Q_h=self.Q_h
            )
            
            # Convert to JAX dataclass and back
            jax_params = DFSVParamsDataclass.from_dfsv_params(np_params)
            converted_back = jax_params.to_dfsv_params()
            
            # Check that dimensions and values are preserved
            self.assertEqual(converted_back.N, self.N)
            self.assertEqual(converted_back.K, self.K)
            np.testing.assert_allclose(converted_back.lambda_r, self.lambda_r)
            np.testing.assert_allclose(converted_back.Phi_f, self.Phi_f)
            np.testing.assert_allclose(converted_back.Phi_h, self.Phi_h)
            np.testing.assert_allclose(converted_back.mu, self.mu)
            
            # Check that arrays are now NumPy arrays
            self.assertIsInstance(converted_back.lambda_r, np.ndarray)
            self.assertIsInstance(converted_back.Phi_f, np.ndarray)
            
        except ImportError:
            self.skipTest("JAX not available, skipping JAX dataclass tests")

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
                sigma2=jnp.array(np.diag(self.sigma2)),
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
            np.testing.assert_allclose(np.array(updated_params.sigma2), np.diag(self.sigma2))
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
                sigma2=jnp.array(np.diag(self.sigma2)),
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
                sigma2=jnp.array(np.diag(self.sigma2)),
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