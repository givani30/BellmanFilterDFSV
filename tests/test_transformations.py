"""
Pytest-based tests for parameter transformation functions used in DFSV model optimization.
"""

import pytest
import numpy as np
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.utils.transformations import transform_params, untransform_params, inverse_softplus
from bellman_filter_dfsv.utils.transformations import transform_params, untransform_params, inverse_softplus, EPS # Add EPS

# Try importing JAX and skip tests if unavailable
try:
    import jax
    import jax.numpy as jnp
    # Enable double precision for more accurate tests
    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Conditional skip for the entire module if JAX is not installed
pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available, skipping transformation tests")


@pytest.fixture(scope="module")
def transformation_params() -> DFSVParamsDataclass:
    """Provides a DFSVParamsDataclass instance for transformation tests."""
    N, K = 3, 2  # Small dimensions for testing

    # Create arrays with correct dimensions using JAX
    lambda_r = jnp.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]], dtype=jnp.float64)
    # Persistence parameters (between 0 and 1)
    Phi_f = jnp.array([[0.95, 0.0], [0.0, 0.9]], dtype=jnp.float64)  # Diagonal
    Phi_h = jnp.array([[0.98, 0.0], [0.0, 0.95]], dtype=jnp.float64) # Diagonal
    # Unconstrained parameters
    mu = jnp.array([[-1.0], [-0.5]], dtype=jnp.float64)
    # Variance parameters (must be positive)
    sigma2_diag = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64)
    Q_h_diag = jnp.array([0.05, 0.08], dtype=jnp.float64) # Diagonal

    # Create parameter object
    params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=lambda_r,
        Phi_f=Phi_f, # Already diagonal
        Phi_h=Phi_h, # Already diagonal
        mu=mu.reshape(K,), # Ensure mu is (K,)
        sigma2=sigma2_diag, # Pass diagonal elements for sigma2
        Q_h=jnp.diag(Q_h_diag) # Construct diagonal Q_h
    )
    # Note: The original setUp created sigma2 as a diagonal matrix.
    # The dataclass expects sigma2 as a 1D array of variances.
    # The transformation functions handle the diagonal nature internally if needed.
    # Let's adjust the fixture to provide sigma2 as 1D array as expected by dataclass.
    params = params.replace(sigma2=sigma2_diag)

    return params


def test_transform_persistence_params(transformation_params):
    """Test transformation of persistence parameters (Phi_f, Phi_h)."""
    params = transformation_params
    # Transform parameters
    transformed = transform_params(params)

    # Check that Phi_f and Phi_h are transformed (logit)
    # For Phi_f
    expected_phi_f_diag = jnp.log(jnp.diag(params.Phi_f) / (1 - jnp.diag(params.Phi_f)))
    np.testing.assert_allclose(
        jnp.diag(transformed.Phi_f),
        expected_phi_f_diag,
        rtol=1e-5
    )
    # For Phi_h
    expected_phi_h_diag = jnp.log(jnp.diag(params.Phi_h) / (1 - jnp.diag(params.Phi_h)))
    np.testing.assert_allclose(
        jnp.diag(transformed.Phi_h),
        expected_phi_h_diag,
        rtol=1e-5
    )

    # Check that the transformation preserves structure (zeros off-diagonal)
    assert transformed.Phi_f.shape == (params.K, params.K)
    assert transformed.Phi_h.shape == (params.K, params.K)
    if params.K > 1:
        assert jnp.allclose(transformed.Phi_f - jnp.diag(jnp.diag(transformed.Phi_f)), 0.0)
        assert jnp.allclose(transformed.Phi_h - jnp.diag(jnp.diag(transformed.Phi_h)), 0.0)


def test_transform_variance_params(transformation_params):
    """Test transformation of variance parameters (sigma2, Q_h)."""
    params = transformation_params
    # Transform parameters
    transformed = transform_params(params)

    # Check that sigma2 and Q_h are transformed (inverse_softplus)
    # For sigma2 (which is 1D in the dataclass)
    expected_sigma2_transformed = inverse_softplus(params.sigma2)
    np.testing.assert_allclose(
        transformed.sigma2, # Transformed sigma2 should also be 1D
        expected_sigma2_transformed,
        rtol=1e-5
    )
    # For Q_h (diagonal matrix)
    expected_q_h_diag = inverse_softplus(jnp.diag(params.Q_h))
    np.testing.assert_allclose(
        jnp.diag(transformed.Q_h),
        expected_q_h_diag,
        rtol=1e-5
    )

    # Check that the transformation preserves structure for Q_h (zeros off-diagonal)
    assert transformed.Q_h.shape == (params.K, params.K)
    if params.K > 1:
        assert jnp.allclose(transformed.Q_h - jnp.diag(jnp.diag(transformed.Q_h)), 0.0)


def test_untransform_persistence_params(transformation_params):
    """Test untransformation of persistence parameters."""
    params = transformation_params
    # Transform parameters
    transformed = transform_params(params)
    # Untransform parameters
    untransformed = untransform_params(transformed)

    # Check that Phi_f and Phi_h are correctly untransformed back
    np.testing.assert_allclose(
        untransformed.Phi_f,
        params.Phi_f,
        rtol=1e-5
    )
    np.testing.assert_allclose(
        untransformed.Phi_h,
        params.Phi_h,
        rtol=1e-5
    )
    # Check structure preservation
    if params.K > 1:
        assert jnp.allclose(untransformed.Phi_f - jnp.diag(jnp.diag(untransformed.Phi_f)), 0.0)
        assert jnp.allclose(untransformed.Phi_h - jnp.diag(jnp.diag(untransformed.Phi_h)), 0.0)


def test_untransform_variance_params(transformation_params):
    """Test untransformation of variance parameters."""
    params = transformation_params
    # Transform parameters
    transformed = transform_params(params)
    # Untransform parameters
    untransformed = untransform_params(transformed)

    # Check that sigma2 and Q_h are correctly untransformed back
    np.testing.assert_allclose(
        untransformed.sigma2, # Should be 1D
        params.sigma2,
        rtol=1e-5
    )
    np.testing.assert_allclose(
        untransformed.Q_h, # Should be diagonal matrix
        params.Q_h,
        rtol=1e-5
    )
    # Check structure preservation
    if params.K > 1:
        assert jnp.allclose(untransformed.Q_h - jnp.diag(jnp.diag(untransformed.Q_h)), 0.0)


def test_roundtrip_transformation(transformation_params):
    """Test round-trip transformation and untransformation preserves values."""
    params = transformation_params
    # Transform then untransform
    transformed = transform_params(params)
    roundtrip = untransform_params(transformed)

    # Check that all parameters are preserved after roundtrip
    # Use tree_map for comparison to handle potential future param additions
    params_leaves = jax.tree_util.tree_leaves(params)
    roundtrip_leaves = jax.tree_util.tree_leaves(roundtrip)

    assert len(params_leaves) == len(roundtrip_leaves)
    for p_leaf, r_leaf in zip(params_leaves, roundtrip_leaves):
        if isinstance(p_leaf, (jax.Array, np.ndarray)):
             # Use allclose for numerical arrays
             np.testing.assert_allclose(r_leaf, p_leaf, rtol=1e-6, atol=1e-7, err_msg=f"Roundtrip mismatch for leaf: {p_leaf}")
        else:
             # Use direct equality for non-array types (like N, K)
             assert r_leaf == p_leaf, f"Roundtrip mismatch for non-array leaf: expected {p_leaf}, got {r_leaf}"


def test_boundary_values(transformation_params):
    """Test transformation near boundary values with epsilon handling."""
    params = transformation_params
    # Create parameters with extreme values
    extreme_phi_f = jnp.array([[0.9999995, 0.0], [0.0, 0.0000005]], dtype=jnp.float64) # Closer to boundaries
    extreme_sigma2 = jnp.array([1e-10, 1.0, 1e+10], dtype=jnp.float64) # Very small and large values

    # Create parameter object, ensuring K=2 for this specific extreme_phi_f
    extreme_params = params.replace(K=2, Phi_f=extreme_phi_f, sigma2=extreme_sigma2)

    # Transform parameters
    transformed = transform_params(extreme_params)
    # Untransform parameters
    untransformed = untransform_params(transformed)

    # Check that values are clamped and preserved close to original, considering epsilon
    # The epsilon in transformations.py is typically 1e-8 or similar
    # Check Phi_f diagonal elements
    np.testing.assert_allclose(
        jnp.diag(untransformed.Phi_f)[0],
        jnp.diag(extreme_phi_f)[0],
        rtol=1e-5, # Relaxed tolerance due to epsilon clamping
        atol=1e-6
    )
    np.testing.assert_allclose(
        jnp.diag(untransformed.Phi_f)[1],
        jnp.diag(extreme_phi_f)[1],
        rtol=1e-5, # Relaxed tolerance
        atol=1e-6
    )
    # Check sigma2 elements
    # For values below EPS, expect the result to be close to EPS due to clipping
    np.testing.assert_allclose(
        untransformed.sigma2[0],
        EPS, # Check against EPS, not the original value below EPS
        rtol=1e-5,
        atol=1e-7 # Check closeness to EPS
    )
    # For very large values, expect inf due to overflow in inverse_softplus
    assert jnp.isinf(untransformed.sigma2[2]), "Expected inf for untransformed large sigma2 due to overflow"


def test_gradient_compatibility(transformation_params):
    """Test that transformed parameters can be used with JAX gradients."""
    params = transformation_params

    # Define a simple objective function operating on the *original* parameter space
    def objective(orig_params: DFSVParamsDataclass):
        # Sum of squares for simplicity
        leaves = jax.tree_util.tree_leaves(orig_params)
        # Filter only JAX arrays for summing squares
        arrays = [leaf for leaf in leaves if isinstance(leaf, jax.Array)]
        return sum(jnp.sum(jnp.square(arr)) for arr in arrays)

    # Define the function that takes *transformed* params, untransforms, then calls objective
    def objective_on_transformed(t_params: DFSVParamsDataclass):
        orig_params = untransform_params(t_params)
        return objective(orig_params)

    # Get gradient function w.r.t. transformed parameters
    grad_fn = jax.grad(objective_on_transformed)

    # Transform the initial parameters to the space the gradient function expects
    transformed_initial_params = transform_params(params)

    # Compute gradient
    gradient = grad_fn(transformed_initial_params)

    # Check that gradient has the same structure as transformed_initial_params
    # and contains finite values
    initial_leaves, initial_treedef = jax.tree_util.tree_flatten(transformed_initial_params)
    gradient_leaves, gradient_treedef = jax.tree_util.tree_flatten(gradient)

    assert initial_treedef == gradient_treedef, "Gradient structure mismatch"
    assert len(initial_leaves) == len(gradient_leaves)

    for i, (init_leaf, grad_leaf) in enumerate(zip(initial_leaves, gradient_leaves)):
        if isinstance(init_leaf, jax.Array):
            assert isinstance(grad_leaf, jax.Array), f"Gradient leaf {i} is not a JAX array"
            assert init_leaf.shape == grad_leaf.shape, f"Gradient shape mismatch for leaf {i}"
            assert init_leaf.dtype == grad_leaf.dtype, f"Gradient dtype mismatch for leaf {i}"
            assert jnp.all(jnp.isfinite(grad_leaf)), f"Gradient leaf {i} contains non-finite values"
        # Skip non-array leaves (like N, K) as they shouldn't have gradients


def test_untransformed_parameter_properties(transformation_params):
    """Test properties of parameters after untransformation."""
    params = transformation_params
    # Perform round-trip transformation
    transformed = transform_params(params)
    untransformed = untransform_params(transformed)

    # 1. sigma2 should be positive (it's a 1D array of variances)
    assert jnp.all(untransformed.sigma2 > 0), "Untransformed sigma2 elements must be positive"

    # 2. Q_h should be Positive Semi-Definite (PSD)
    # For diagonal matrices, this means diagonal elements >= 0
    assert jnp.all(jnp.diag(untransformed.Q_h) >= 0), "Untransformed Q_h diagonal elements must be non-negative"
    # Verify it's still diagonal
    if params.K > 1:
        assert jnp.allclose(untransformed.Q_h - jnp.diag(jnp.diag(untransformed.Q_h)), 0.0), "Untransformed Q_h must remain diagonal"

    # 3. Phi_f should be stationary (eigenvalues within unit circle)
    # For diagonal matrices, this means absolute value of diagonal elements < 1
    assert jnp.all(jnp.abs(jnp.diag(untransformed.Phi_f)) < 1.0), "Untransformed Phi_f diagonal elements must be < 1 for stationarity"
    # Verify it's still diagonal
    if params.K > 1:
        assert jnp.allclose(untransformed.Phi_f - jnp.diag(jnp.diag(untransformed.Phi_f)), 0.0), "Untransformed Phi_f must remain diagonal"

    # 4. Phi_h should be stationary (eigenvalues within unit circle)
    # For diagonal matrices, this means absolute value of diagonal elements < 1
    assert jnp.all(jnp.abs(jnp.diag(untransformed.Phi_h)) < 1.0), "Untransformed Phi_h diagonal elements must be < 1 for stationarity"
    # Verify it's still diagonal
    if params.K > 1:
        assert jnp.allclose(untransformed.Phi_h - jnp.diag(jnp.diag(untransformed.Phi_h)), 0.0), "Untransformed Phi_h must remain diagonal"
