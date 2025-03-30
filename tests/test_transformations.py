"""
Tests for parameter transformation functions used in DFSV model optimization.

This module tests the parameter transformation functionality that maps constrained 
parameters (e.g., variances > 0, correlations in [-1, 1]) to unconstrained space 
for optimization, and back.
"""

import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path to import from our modules
sys.path.append(str(Path(__file__).parent.parent))

from models.dfsv import DFSV_params, DFSVParamsDataclass
from functions.transformations import transform_params, untransform_params

class TestParameterTransformations(unittest.TestCase):
    """Tests for parameter transformation functions."""

    def setUp(self):
        """Set up test parameters with correct dimensions."""
        try:
            import jax
            import jax.numpy as jnp
            
            # Enable double precision for more accurate tests
            jax.config.update("jax_enable_x64", True)
            
            self.N, self.K = 3, 2  # Small dimensions for testing

            # Create arrays with correct dimensions
            self.lambda_r = jnp.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
            
            # Persistence parameters (between 0 and 1)
            self.Phi_f = jnp.array([[0.95, 0.0], [0.0, 0.9]])  # Diagonal for simplicity
            self.Phi_h = jnp.array([[0.98, 0.0], [0.0, 0.95]])  # Diagonal for simplicity
            
            # Unconstrained parameters
            self.mu = jnp.array([[-1.0], [-0.5]])
            
            # Variance parameters (must be positive)
            self.sigma2 = jnp.diag(jnp.array([0.1, 0.2, 0.3]))
            self.Q_h = jnp.array([[0.05, 0.0], [0.0, 0.08]])  # Diagonal for simplicity
            
            # Create parameter object
            self.params = DFSVParamsDataclass(
                N=self.N,
                K=self.K,
                lambda_r=self.lambda_r,
                Phi_f=self.Phi_f,
                Phi_h=self.Phi_h,
                mu=self.mu,
                sigma2=self.sigma2,
                Q_h=self.Q_h
            )
        except ImportError:
            self.skipTest("JAX not available, skipping transformation tests")

    def test_transform_persistence_params(self):
        """Test transformation of persistence parameters (Phi_f, Phi_h)."""
        try:
            import jax.numpy as jnp
            
            # Transform parameters
            transformed = transform_params(self.params)
            
            # Check that Phi_f and Phi_h are transformed
            # The transformation is logit(p) = log(p/(1-p))
            
            # For Phi_f
            expected_phi_f_diag = jnp.log(jnp.diag(self.Phi_f) / (1 - jnp.diag(self.Phi_f)))
            np.testing.assert_allclose(
                jnp.diag(transformed.Phi_f), 
                expected_phi_f_diag, 
                rtol=1e-5
            )
            
            # For Phi_h
            expected_phi_h_diag = jnp.log(jnp.diag(self.Phi_h) / (1 - jnp.diag(self.Phi_h)))
            np.testing.assert_allclose(
                jnp.diag(transformed.Phi_h), 
                expected_phi_h_diag, 
                rtol=1e-5
            )
            
            # Check that the transformation preserves structure (zeros off-diagonal)
            self.assertAlmostEqual(transformed.Phi_f[0, 1], 0.0)
            self.assertAlmostEqual(transformed.Phi_f[1, 0], 0.0)
            self.assertAlmostEqual(transformed.Phi_h[0, 1], 0.0)
            self.assertAlmostEqual(transformed.Phi_h[1, 0], 0.0)
            
        except ImportError:
            self.skipTest("JAX not available, skipping transformation tests")

    def test_transform_variance_params(self):
        """Test transformation of variance parameters (sigma2, Q_h)."""
        try:
            import jax.numpy as jnp
            
            # Transform parameters
            transformed = transform_params(self.params)
            
            # Check that sigma2 and Q_h are transformed
            # The transformation is log(p)
            
            # For sigma2
            expected_sigma2_diag = jnp.log(jnp.array([0.1, 0.2, 0.3]))
            np.testing.assert_allclose(
                jnp.diag(transformed.sigma2), 
                expected_sigma2_diag, 
                rtol=1e-5
            )
            
            # For Q_h
            expected_q_h_diag = jnp.log(jnp.diag(self.Q_h))
            np.testing.assert_allclose(
                jnp.diag(transformed.Q_h), 
                expected_q_h_diag, 
                rtol=1e-5
            )
            
            # Check that the transformation preserves structure (zeros off-diagonal)
            self.assertAlmostEqual(transformed.Q_h[0, 1], 0.0)
            self.assertAlmostEqual(transformed.Q_h[1, 0], 0.0)
            
            # Check that off-diagonal elements of sigma2 are zero
            self.assertAlmostEqual(transformed.sigma2[0, 1], 0.0)
            self.assertAlmostEqual(transformed.sigma2[1, 0], 0.0)
            self.assertAlmostEqual(transformed.sigma2[0, 2], 0.0)
            
        except ImportError:
            self.skipTest("JAX not available, skipping transformation tests")

    def test_untransform_persistence_params(self):
        """Test untransformation of persistence parameters."""
        try:
            import jax.numpy as jnp
            
            # Transform parameters
            transformed = transform_params(self.params)
            
            # Untransform parameters
            untransformed = untransform_params(transformed)
            
            # Check that Phi_f and Phi_h are correctly untransformed back to original values
            np.testing.assert_allclose(
                jnp.diag(untransformed.Phi_f), 
                jnp.diag(self.Phi_f), 
                rtol=1e-5
            )
            
            np.testing.assert_allclose(
                jnp.diag(untransformed.Phi_h), 
                jnp.diag(self.Phi_h), 
                rtol=1e-5
            )
            
            # Check structure preservation
            self.assertAlmostEqual(untransformed.Phi_f[0, 1], 0.0)
            self.assertAlmostEqual(untransformed.Phi_h[0, 1], 0.0)
            
        except ImportError:
            self.skipTest("JAX not available, skipping transformation tests")

    def test_untransform_variance_params(self):
        """Test untransformation of variance parameters."""
        try:
            import jax.numpy as jnp
            
            # Transform parameters
            transformed = transform_params(self.params)
            
            # Untransform parameters
            untransformed = untransform_params(transformed)
            
            # Check that sigma2 and Q_h are correctly untransformed back to original values
            np.testing.assert_allclose(
                jnp.diag(untransformed.sigma2), 
                jnp.array([0.1, 0.2, 0.3]), 
                rtol=1e-5
            )
            
            np.testing.assert_allclose(
                jnp.diag(untransformed.Q_h), 
                jnp.diag(self.Q_h), 
                rtol=1e-5
            )
            
            # Check structure preservation
            self.assertAlmostEqual(untransformed.Q_h[0, 1], 0.0)
            self.assertAlmostEqual(untransformed.sigma2[0, 1], 0.0)
            
        except ImportError:
            self.skipTest("JAX not available, skipping transformation tests")

    def test_roundtrip_transformation(self):
        """Test round-trip transformation and untransformation preserves values."""
        try:
            import jax.numpy as jnp
            
            # Transform then untransform
            transformed = transform_params(self.params)
            roundtrip = untransform_params(transformed)
            
            # Check that all parameters are preserved after roundtrip
            np.testing.assert_allclose(roundtrip.lambda_r, self.lambda_r)
            np.testing.assert_allclose(roundtrip.Phi_f, self.Phi_f)
            np.testing.assert_allclose(roundtrip.Phi_h, self.Phi_h)
            np.testing.assert_allclose(roundtrip.mu, self.mu)
            np.testing.assert_allclose(roundtrip.sigma2, self.sigma2)
            np.testing.assert_allclose(roundtrip.Q_h, self.Q_h)
            
        except ImportError:
            self.skipTest("JAX not available, skipping transformation tests")

    def test_boundary_values(self):
        """Test transformation near boundary values with epsilon handling."""
        try:
            import jax
            import jax.numpy as jnp
            
            # Create parameters with extreme values
            extreme_phi_f = jnp.array([[0.9999, 0.0], [0.0, 0.0001]])  # Very close to boundaries
            extreme_sigma2 = jnp.diag(jnp.array([1e-10, 1.0, 1e+10]))  # Very small and large values
            
            # Create parameter object
            extreme_params = self.params.replace(Phi_f=extreme_phi_f, sigma2=extreme_sigma2)
            
            # Transform parameters
            transformed = transform_params(extreme_params)
            
            # Untransform parameters
            untransformed = untransform_params(transformed)
            
            # Check that values are clamped and preserved
            # The epsilon in transformations.py is 1e-6
            np.testing.assert_allclose(
                jnp.diag(untransformed.Phi_f)[0],
                jnp.array(0.9999),
                rtol=1e-3  # Slightly relaxed tolerance due to epsilon clamping
            )
            
            np.testing.assert_allclose(
                jnp.diag(untransformed.Phi_f)[1],
                jnp.array(0.0001),
                rtol=1e-3  # Slightly relaxed tolerance due to epsilon clamping
            )
            
        except ImportError:
            self.skipTest("JAX not available, skipping transformation tests")

    def test_gradient_compatibility(self):
        """Test that transformed parameters can be used with JAX gradients."""
        try:
            import jax
            import jax.numpy as jnp
            
            # Define a simple objective function
            def objective(params):
                # Sum of squares for simplicity
                total = 0.0
                total += jnp.sum(jnp.square(params.lambda_r))
                total += jnp.sum(jnp.square(params.Phi_f))
                total += jnp.sum(jnp.square(params.Phi_h))
                total += jnp.sum(jnp.square(params.mu))
                total += jnp.sum(jnp.square(params.sigma2))
                total += jnp.sum(jnp.square(params.Q_h))
                return total
            
            # Create gradient function for transformed parameters
            def transformed_objective(params):
                # Transform parameters first
                transformed = transform_params(params)
                return objective(transformed)
            
            # Get gradient function
            grad_fn = jax.grad(transformed_objective, has_aux=False)
            
            # Compute gradient
            gradient = grad_fn(self.params)
            
            # Check that gradient has correct structure and finite values
            self.assertEqual(gradient.lambda_r.shape, self.lambda_r.shape)
            self.assertEqual(gradient.Phi_f.shape, self.Phi_f.shape)
            self.assertEqual(gradient.Phi_h.shape, self.Phi_h.shape)
            self.assertEqual(gradient.mu.shape, self.mu.shape)
            self.assertEqual(gradient.sigma2.shape, self.sigma2.shape)
            self.assertEqual(gradient.Q_h.shape, self.Q_h.shape)
            
            # Check for finite values
            self.assertTrue(jnp.all(jnp.isfinite(gradient.lambda_r)))
            self.assertTrue(jnp.all(jnp.isfinite(gradient.Phi_f)))
            self.assertTrue(jnp.all(jnp.isfinite(gradient.Phi_h)))
            self.assertTrue(jnp.all(jnp.isfinite(gradient.mu)))
            self.assertTrue(jnp.all(jnp.isfinite(gradient.sigma2)))
            self.assertTrue(jnp.all(jnp.isfinite(gradient.Q_h)))
            
        except ImportError:
            self.skipTest("JAX not available, skipping transformation tests")


if __name__ == "__main__":
    unittest.main()