import pytest
import numpy as np
import copy
import jax.random as jr # Add jr
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
# Import new functions and EPS
from bellman_filter_dfsv.utils.transformations import (
    transform_params,
    untransform_params,
    inverse_softplus,
    safe_arctanh, # Added
    EPS
)
# Removed stabilize_matrix, get_unconstrained_matrix

# Try importing JAX and skip tests if unavailable
try:
    import jax
    import jax.numpy as jnp
    from jax.nn import softplus # Import softplus for boundary test check
    # Enable double precision for more accurate tests
    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Conditional skip for the entire module if JAX is not installed
pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available, skipping transformation tests")


@pytest.fixture(scope="module")
def base_params() -> DFSVParamsDataclass:
    """Provides a base DFSVParamsDataclass instance for tests."""
    N, K = 3, 2  # Small dimensions for testing
    key = jr.PRNGKey(0)
    key, subkey1, subkey2, subkey3, subkey4, subkey5 = jr.split(key, 6)

    # Create arrays with correct dimensions using JAX
    lambda_r = jr.uniform(subkey1, (N, K), dtype=jnp.float64)
    lambda_r = lambda_r / jnp.sum(lambda_r, axis=1, keepdims=True) # Normalize rows

    # Persistence parameters (diagonal between -1 and 1, off-diagonal anything)
    diag_f = jr.uniform(subkey2, (K,), minval=-0.99, maxval=0.99, dtype=jnp.float64)
    off_diag_f = jr.normal(subkey3, (K, K), dtype=jnp.float64) * 0.1
    Phi_f = jnp.diag(diag_f) + (off_diag_f - jnp.diag(jnp.diag(off_diag_f))) # Ensure diagonal is from diag_f

    diag_h = jr.uniform(subkey4, (K,), minval=-0.99, maxval=0.99, dtype=jnp.float64)
    off_diag_h = jr.normal(subkey5, (K, K), dtype=jnp.float64) * 0.1
    Phi_h = jnp.diag(diag_h) + (off_diag_h - jnp.diag(jnp.diag(off_diag_h))) # Ensure diagonal is from diag_h

    # Unconstrained parameters
    mu = jr.normal(key, (K,), dtype=jnp.float64) # Shape (K,)

    # Variance parameters (must be positive)
    sigma2_diag = jr.uniform(key, (N,), minval=0.01, maxval=1.0, dtype=jnp.float64)
    Q_h_diag = jr.uniform(key, (K,), minval=0.01, maxval=0.5, dtype=jnp.float64)

    # Create parameter object
    params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=lambda_r,
        Phi_f=Phi_f,
        Phi_h=Phi_h,
        mu=mu,
        sigma2=jnp.diag(sigma2_diag), # Pass diagonal matrix for sigma2
        Q_h=jnp.diag(Q_h_diag) # Construct diagonal Q_h
    )
    return params


# --- Tests for safe_arctanh ---

def test_safe_arctanh_normal_values():
    """Test safe_arctanh with values strictly between -1 and 1."""
    x = jnp.array([-0.9, -0.5, 0.0, 0.5, 0.9], dtype=jnp.float64)
    expected = jnp.arctanh(x)
    result = safe_arctanh(x)
    np.testing.assert_allclose(result, expected, rtol=1e-7)

def test_safe_arctanh_boundary_values():
    """Test safe_arctanh with values exactly at -1 and 1 (should clip)."""
    x = jnp.array([-1.0, 1.0], dtype=jnp.float64)
    # Expect arctanh of clipped values
    expected = jnp.arctanh(jnp.array([-1.0 + EPS, 1.0 - EPS], dtype=jnp.float64))
    result = safe_arctanh(x)
    np.testing.assert_allclose(result, expected, rtol=1e-7)

def test_safe_arctanh_near_boundary_values():
    """Test safe_arctanh with values very close to -1 and 1."""
    x = jnp.array([-1.0 + EPS/2, 1.0 - EPS/2], dtype=jnp.float64)
    # Expect arctanh of clipped values (since input is within EPS of boundary)
    expected = jnp.arctanh(jnp.array([-1.0 + EPS, 1.0 - EPS], dtype=jnp.float64))
    result = safe_arctanh(x)
    np.testing.assert_allclose(result, expected, rtol=1e-7)

def test_safe_arctanh_outside_boundary():
    """Test safe_arctanh with values outside [-1, 1] (should clip)."""
    x = jnp.array([-2.0, -1.1, 1.1, 2.0], dtype=jnp.float64)
    # Expect arctanh of clipped values
    expected = jnp.arctanh(jnp.array([-1.0 + EPS, -1.0 + EPS, 1.0 - EPS, 1.0 - EPS], dtype=jnp.float64))
    result = safe_arctanh(x)
    np.testing.assert_allclose(result, expected, rtol=1e-7)

def test_safe_arctanh_nan_inf():
    """Test safe_arctanh with NaN and Inf inputs."""
    x = jnp.array([jnp.nan, jnp.inf, -jnp.inf], dtype=jnp.float64)
    result = safe_arctanh(x)
    # Clipping NaN results in NaN. Clipping Inf results in 1-EPS. Clipping -Inf results in -1+EPS.
    # Then arctanh is applied.
    expected_nan = jnp.nan
    expected_inf = jnp.arctanh(1.0 - EPS)
    expected_neg_inf = jnp.arctanh(-1.0 + EPS)
    assert jnp.isnan(result[0])
    np.testing.assert_allclose(result[1], expected_inf, rtol=1e-7)
    np.testing.assert_allclose(result[2], expected_neg_inf, rtol=1e-7)


# --- REMOVED ALL TESTS for stabilize_matrix and get_unconstrained_matrix ---
# (Original lines 100-277 deleted)


# --- Updated Tests for Hybrid Transformations ---

def test_transform_persistence_params_hybrid(base_params):
    """Test transformation of persistence parameters (Phi_f, Phi_h) with hybrid approach."""
    params = base_params
    # Transform parameters
    transformed = transform_params(params)

    # Check diagonal elements are transformed via inverse_softplus
    expected_diag_f = inverse_softplus(jnp.diag(params.Phi_f))
    np.testing.assert_allclose(jnp.diag(transformed.Phi_f), expected_diag_f, rtol=1e-6)

    expected_diag_h = inverse_softplus(jnp.diag(params.Phi_h))
    np.testing.assert_allclose(jnp.diag(transformed.Phi_h), expected_diag_h, rtol=1e-6)

    # Check off-diagonal elements remain unchanged
    off_diag_mask = ~jnp.eye(params.K, dtype=bool)
    np.testing.assert_allclose(transformed.Phi_f[off_diag_mask], params.Phi_f[off_diag_mask], atol=1e-9)
    np.testing.assert_allclose(transformed.Phi_h[off_diag_mask], params.Phi_h[off_diag_mask], atol=1e-9)

    # Check shapes
    assert transformed.Phi_f.shape == (params.K, params.K)
    assert transformed.Phi_h.shape == (params.K, params.K)


def test_transform_variance_params(base_params):
    """Test transformation of variance parameters (sigma2, Q_h)."""
    # This test remains largely the same as variances are handled identically.
    params = base_params
    # Transform parameters
    transformed = transform_params(params)

    # Check sigma2 (diagonal matrix)
    expected_sigma2_diag_transformed = inverse_softplus(jnp.diag(params.sigma2))
    np.testing.assert_allclose(
        jnp.diag(transformed.sigma2),
        expected_sigma2_diag_transformed,
        rtol=1e-5
    )
    assert transformed.sigma2.shape == (params.N, params.N) # Check shape
    if params.N > 1:
        assert jnp.allclose(transformed.sigma2 - jnp.diag(jnp.diag(transformed.sigma2)), 0.0)

    # Check Q_h (diagonal matrix)
    expected_q_h_diag = inverse_softplus(jnp.diag(params.Q_h))
    np.testing.assert_allclose(
        jnp.diag(transformed.Q_h), # Check diagonal of transformed Q_h
        expected_q_h_diag,
        rtol=1e-5
    )
    # Check that the transformation preserves structure for Q_h (zeros off-diagonal)
    assert transformed.Q_h.shape == (params.K, params.K)
    if params.K > 1:
        assert jnp.allclose(transformed.Q_h - jnp.diag(jnp.diag(transformed.Q_h)), 0.0)


def test_untransform_persistence_params_hybrid(base_params):
    """Test untransformation of persistence parameters with hybrid approach."""
    params = base_params
    # Transform parameters
    transformed = transform_params(params)
    # Untransform parameters
    untransformed = untransform_params(transformed)

    # Check diagonal elements are untransformed via softplus
    expected_diag_f = softplus(jnp.diag(transformed.Phi_f))
    np.testing.assert_allclose(jnp.diag(untransformed.Phi_f), expected_diag_f, rtol=1e-6)

    expected_diag_h = softplus(jnp.diag(transformed.Phi_h))
    np.testing.assert_allclose(jnp.diag(untransformed.Phi_h), expected_diag_h, rtol=1e-6)

    # Check off-diagonal elements remain unchanged (same as transformed off-diagonals)
    off_diag_mask = ~jnp.eye(params.K, dtype=bool)
    np.testing.assert_allclose(untransformed.Phi_f[off_diag_mask], transformed.Phi_f[off_diag_mask], atol=1e-9)
    np.testing.assert_allclose(untransformed.Phi_h[off_diag_mask], transformed.Phi_h[off_diag_mask], atol=1e-9)

    # Check shapes
    assert untransformed.Phi_f.shape == (params.K, params.K)
    assert untransformed.Phi_h.shape == (params.K, params.K)


def test_untransform_variance_params(base_params):
    """Test untransformation of variance parameters."""
    # This test remains largely the same.
    params = base_params
    # Transform parameters
    transformed = transform_params(params)
    # Untransform parameters
    untransformed = untransform_params(transformed)

    # Check that sigma2 and Q_h are correctly untransformed back
    np.testing.assert_allclose(
        untransformed.sigma2, # Should be diagonal matrix
        params.sigma2,
        rtol=1e-5
    )
    np.testing.assert_allclose(
        untransformed.Q_h, # Should be diagonal matrix
        params.Q_h,
        rtol=1e-5
    )
    # Check structure preservation
    if params.N > 1:
         assert jnp.allclose(untransformed.sigma2 - jnp.diag(jnp.diag(untransformed.sigma2)), 0.0, atol=1e-9)
    if params.K > 1:
        assert jnp.allclose(untransformed.Q_h - jnp.diag(jnp.diag(untransformed.Q_h)), 0.0)


def test_roundtrip_transformation_hybrid(base_params):
    """Test round-trip transformation preserves values with hybrid approach."""
    params = base_params
    # Transform then untransform
    transformed = transform_params(params)
    roundtrip = untransform_params(transformed)

    # Check that all parameters are preserved after roundtrip
    # Use tree_map for comparison to handle potential future param additions
    params_leaves, params_treedef = jax.tree_util.tree_flatten(params)
    roundtrip_leaves, roundtrip_treedef = jax.tree_util.tree_flatten(roundtrip)

    assert params_treedef == roundtrip_treedef
    assert len(params_leaves) == len(roundtrip_leaves)
    for p_leaf, r_leaf in zip(params_leaves, roundtrip_leaves):
        if isinstance(p_leaf, (jax.Array, np.ndarray)):
             # Use allclose for numerical arrays
             np.testing.assert_allclose(
                 r_leaf,
                 p_leaf,
                 rtol=1e-6, # Use appropriate tolerance
                 atol=1e-7,
                 err_msg=f"Roundtrip mismatch for leaf"
             )
        else:
             # Use direct equality for non-array types (like N, K)
              assert r_leaf == p_leaf, f"Roundtrip mismatch for non-array leaf: expected {p_leaf}, got {r_leaf}"

    # Explicitly check off-diagonals of Phi_f and Phi_h are identical
    off_diag_mask = ~jnp.eye(params.K, dtype=bool)
    np.testing.assert_allclose(roundtrip.Phi_f[off_diag_mask], params.Phi_f[off_diag_mask], atol=1e-9, err_msg="Phi_f off-diagonal mismatch in roundtrip")
    np.testing.assert_allclose(roundtrip.Phi_h[off_diag_mask], params.Phi_h[off_diag_mask], atol=1e-9, err_msg="Phi_h off-diagonal mismatch in roundtrip")


def test_boundary_values_hybrid(base_params):
    """Test transformation near boundary values with hybrid approach."""
    params = base_params
    K = params.K
    N = params.N

    # Create parameters with extreme values
    extreme_diag_f = jnp.array([1.0 - EPS/2, -1.0 + EPS/2], dtype=jnp.float64) # Near boundaries
    extreme_diag_h = jnp.array([0.9999999, -0.9999999], dtype=jnp.float64) # Very near boundaries
    # Keep off-diagonals from base_params
    extreme_phi_f = jnp.diag(extreme_diag_f) + (params.Phi_f - jnp.diag(jnp.diag(params.Phi_f)))
    extreme_phi_h = jnp.diag(extreme_diag_h) + (params.Phi_h - jnp.diag(jnp.diag(params.Phi_h)))

    extreme_sigma2_diag = jnp.array([1e-10, 1.0, 1e10], dtype=jnp.float64) # Very small and large values
    extreme_q_h_diag = jnp.array([1e-12, 1e8], dtype=jnp.float64)

    # Create parameter object, ensuring K=2 for this specific extreme_phi_f
    # and diagonalizing sigma2
    extreme_params = params.replace(
        Phi_f=extreme_phi_f,
        Phi_h=extreme_phi_h,
        sigma2=jnp.diag(extreme_sigma2_diag),
        Q_h=jnp.diag(extreme_q_h_diag)
    )

    # Transform parameters
    transformed = transform_params(extreme_params)
    # Untransform parameters
    untransformed = untransform_params(transformed)

    # Check Phi diagonal elements are close to softplus(inverse_softplus(original diagonals))
    expected_diag_f = softplus(inverse_softplus(jnp.diag(extreme_params.Phi_f)))
    expected_diag_h = softplus(inverse_softplus(jnp.diag(extreme_params.Phi_h)))
    np.testing.assert_allclose(
        jnp.diag(untransformed.Phi_f),
        expected_diag_f,
        rtol=1e-5,
        atol=1e-8
    )
    np.testing.assert_allclose(
        jnp.diag(untransformed.Phi_h),
        expected_diag_h,
        rtol=1e-5,
        atol=1e-8
    )
    # Check Phi off-diagonal elements are preserved exactly
    off_diag_mask = ~jnp.eye(K, dtype=bool)
    np.testing.assert_allclose(untransformed.Phi_f[off_diag_mask], extreme_params.Phi_f[off_diag_mask], atol=1e-9)
    np.testing.assert_allclose(untransformed.Phi_h[off_diag_mask], extreme_params.Phi_h[off_diag_mask], atol=1e-9)


    # Check sigma2 elements (diagonal)
    # For values below EPS, inverse_softplus -> softplus should recover something close to original
    # (inverse_softplus clips to EPS, log(EPS) is large negative, softplus(large_neg) is near 0)
    # Let's check against the original value with tolerance
    np.testing.assert_allclose(
        jnp.diag(untransformed.sigma2)[0],
        jnp.diag(extreme_params.sigma2)[0],
        atol=EPS # Expect it to be very close to the original small value or 0
    )
    # For very large values, inverse_softplus -> softplus can result in inf due to overflow
    assert jnp.isinf(jnp.diag(untransformed.sigma2)[2]), "Expected inf for untransformed large sigma2 due to overflow"

    # Check Q_h elements (diagonal)
    # Small value check
    np.testing.assert_allclose(
        jnp.diag(untransformed.Q_h)[0],
        jnp.diag(extreme_params.Q_h)[0],
        atol=EPS
    )
    # Large value check (expect inf)
    assert jnp.isinf(jnp.diag(untransformed.Q_h)[1]), "Expected inf for untransformed large Q_h due to overflow"


# Remove xfail marker - hybrid transformation should be differentiable
# @pytest.mark.xfail(reason="JAX cannot compute gradients through jnp.linalg.eig used in stabilize_matrix")
def test_gradient_compatibility(base_params):
    """Test that transformed parameters can be used with JAX gradients (hybrid)."""
    params = base_params

    # Define a simple objective function operating on the *original* parameter space
    def objective(orig_params: DFSVParamsDataclass):
        # Sum of squares for simplicity, focusing on transformed parts
        diag_f = jnp.diag(orig_params.Phi_f)
        off_diag_f = orig_params.Phi_f[~jnp.eye(orig_params.K, dtype=bool)]
        diag_h = jnp.diag(orig_params.Phi_h)
        off_diag_h = orig_params.Phi_h[~jnp.eye(orig_params.K, dtype=bool)]
        diag_s = jnp.diag(orig_params.sigma2)
        diag_q = jnp.diag(orig_params.Q_h)
        return (jnp.sum(jnp.square(diag_f)) + jnp.sum(jnp.square(off_diag_f)) +
                jnp.sum(jnp.square(diag_h)) + jnp.sum(jnp.square(off_diag_h)) +
                jnp.sum(jnp.square(diag_s)) + jnp.sum(jnp.square(diag_q)) +
                jnp.sum(jnp.square(orig_params.mu)) + jnp.sum(jnp.square(orig_params.lambda_r)))


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
            assert jnp.all(jnp.isfinite(grad_leaf)), f"Gradient leaf {i} contains non-finite values: {grad_leaf}"
        # Skip non-array leaves (like N, K) as they shouldn't have gradients


def test_untransformed_parameter_properties_hybrid(base_params):
    """Test properties of parameters after untransformation (hybrid approach)."""
    params = base_params
    # Perform round-trip transformation
    transformed = transform_params(params)
    untransformed = untransform_params(transformed)

    # 1. sigma2 diagonal elements > 0
    assert jnp.all(jnp.diag(untransformed.sigma2) > 0), "Untransformed sigma2 diagonal elements must be positive"
    if params.N > 1:
         assert jnp.allclose(untransformed.sigma2 - jnp.diag(jnp.diag(untransformed.sigma2)), 0.0, atol=1e-9), "Untransformed sigma2 must remain diagonal"

    # 2. Q_h diagonal elements >= 0
    assert jnp.all(jnp.diag(untransformed.Q_h) >= 0), "Untransformed Q_h diagonal elements must be non-negative"
    if params.K > 1:
        assert jnp.allclose(untransformed.Q_h - jnp.diag(jnp.diag(untransformed.Q_h)), 0.0, atol=1e-9), "Untransformed Q_h must remain diagonal"

    # 3. Phi_f diagonal elements should be within (-1, 1)
    assert jnp.all(jnp.abs(jnp.diag(untransformed.Phi_f)) < 1.0), f"Untransformed Phi_f diagonal elements must be < 1. Got: {jnp.diag(untransformed.Phi_f)}"

    # 4. Phi_h diagonal elements should be within (-1, 1)
    assert jnp.all(jnp.abs(jnp.diag(untransformed.Phi_h)) < 1.0), f"Untransformed Phi_h diagonal elements must be < 1. Got: {jnp.diag(untransformed.Phi_h)}"

    # Note: We no longer check stationarity of the full matrix via eigenvalues,
    # as the hybrid transformation only constrains the diagonal. Off-diagonals are unconstrained.

# --- Remove old full phi tests ---
# test_untransformed_parameter_properties (original)
# test_roundtrip_transformation_full_phi (original)
