"""
Parameter transformation functions for DFSV models.

Maps constrained parameters (e.g., variances > 0, correlations in [-1, 1])
to unconstrained space for optimization, and back.
"""
import copy
import jax
import jax.numpy as jnp
from jax.nn import softplus
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass

# Epsilon for numerical stability near boundaries (e.g., 0 or 1)
EPS = 1e-6

def inverse_softplus(x):
    """
    Compute the inverse of softplus: log(exp(x) - 1)
    With numerical stability safeguards.

    Parameters:
    -----------
    x : array_like
        Input values (must be positive)

    Returns:
    --------
    array_like
        Inverse softplus of x
    """
    # Ensure x is sufficiently positive to avoid numerical issues
    x_safe = jnp.maximum(x, EPS)
    # For very small x, softplus(y) ≈ exp(y), so inverse_softplus(x) ≈ log(x)
    # For larger x, use the standard formula log(exp(x) - 1)
    return jnp.where(x_safe < 1e-3,
                     jnp.log(x_safe),  # Approximation for small values
                     jnp.log(jnp.exp(x_safe) - 1.0))

def safe_arctanh(x):
    """
    Computes arctanh(x) with clipping to avoid +/- inf for x near +/- 1.

    Parameters:
    -----------
    x : array_like
        Input values, expected to be in [-1, 1].

    Returns:
    --------
    array_like
        arctanh(x) after clipping x to [-1 + EPS, 1 - EPS].
    """
    x_clipped = jnp.clip(x, -1.0 + EPS, 1.0 - EPS)
    return jnp.arctanh(x_clipped)

# """ REMOVED based on plan 'full_phi_hybrid_plan_07-04-2025.md'
# def stabilize_matrix(matrix_unc: jnp.ndarray) -> jnp.ndarray:
#     """Stabilizes a matrix by transforming its eigenvalues to have magnitude < 1.
#
#     Uses eigenvalue decomposition and maps eigenvalue magnitudes using tanh.
#
#     Args:
#         matrix_unc: An unconstrained KxK JAX array.
#
#     Returns:
#         A stabilized KxK JAX array (real-valued) with eigenvalues inside the
#         unit circle. Returns 0.95 * I if input contains NaN/Inf.
#     """
#     K = matrix_unc.shape[0]
#     default_stable_matrix = jnp.eye(K) * 0.95
#
#     def _stabilize(mat):
#         eigvals, eigvecs = jnp.linalg.eig(mat)
#         magnitudes = jnp.abs(eigvals)
#         transformed_magnitudes = jnp.tanh(magnitudes)
#         # Handle potential division by zero for zero eigenvalues
#         phases = jnp.where(magnitudes == 0, 1.0, eigvals / magnitudes)
#         new_eigvals = transformed_magnitudes * phases
#         stable_matrix = (eigvecs @ jnp.diag(new_eigvals) @ jnp.linalg.inv(eigvecs)).real
#         return jnp.nan_to_num(stable_matrix, nan=0.0, posinf=0.0, neginf=0.0) # Use nan_to_num on output
#
#     # Check for NaN/Inf in the input matrix
#     has_invalid_values = jnp.any(jnp.isnan(matrix_unc) | jnp.isinf(matrix_unc))
#
#     # Conditionally apply stabilization or return default
#     stable_matrix = jax.lax.cond(
#         has_invalid_values,
#         lambda _: default_stable_matrix,
#         _stabilize,
#         matrix_unc
#     )
#     return stable_matrix
# """

# """ REMOVED based on plan 'full_phi_hybrid_plan_07-04-2025.md'
# def get_unconstrained_matrix(stable_matrix: jnp.ndarray) -> jnp.ndarray:
#     """Transforms a stable matrix (eigenvalues < 1) back to unconstrained space.
#
#     Uses eigenvalue decomposition and maps eigenvalue magnitudes using arctanh.
#
#     Args:
#         stable_matrix: A stable KxK JAX array (eigenvalues inside unit circle).
#
#     Returns:
#         An unconstrained KxK JAX array (real-valued). Returns a zero matrix
#         if input contains NaN/Inf.
#     """
#     K = stable_matrix.shape[0]
#     default_unc_matrix = jnp.zeros((K, K)) # Default to zero matrix on invalid input
#
#     def _get_unconstrained(mat):
#         eigvals, eigvecs = jnp.linalg.eig(mat)
#         magnitudes = jnp.abs(eigvals)
#         # Clip magnitudes slightly away from 1 for numerical stability of arctanh
#         magnitudes_clipped = jnp.clip(magnitudes, 0, 1.0 - EPS)
#         unconstrained_magnitudes = jnp.arctanh(magnitudes_clipped)
#         # Handle potential division by zero for zero eigenvalues
#         phases = jnp.where(magnitudes == 0, 1.0, eigvals / magnitudes)
#         unc_eigvals = unconstrained_magnitudes * phases
#         unc_matrix = (eigvecs @ jnp.diag(unc_eigvals) @ jnp.linalg.inv(eigvecs)).real
#         return jnp.nan_to_num(unc_matrix, nan=0.0, posinf=0.0, neginf=0.0) # Use nan_to_num on output
#
#     # Check for NaN/Inf in the input matrix
#     has_invalid_values = jnp.any(jnp.isnan(stable_matrix) | jnp.isinf(stable_matrix))
#
#     # Conditionally apply inverse transformation or return default
#     unc_matrix = jax.lax.cond(
#         has_invalid_values,
#         lambda _: default_unc_matrix,
#         _get_unconstrained,
#         stable_matrix
#     )
#     return unc_matrix
# """


def transform_params(params: DFSVParamsDataclass) -> DFSVParamsDataclass:
    """
    Transform bounded parameters to unconstrained space for optimization.

    - Applies `inverse_softplus` to diagonal elements of `Phi_f` and `Phi_h`.
      Off-diagonals remain unconstrained.
    - Applies `inverse_softplus` to diagonal elements of `sigma2` and `Q_h`.
    - Leaves `mu` and `lambda_r` unchanged.

    Parameters:
    -----------
    params : DFSVParamsDataclass
        Model parameters in their natural (constrained) space.

    Returns:
    --------
    DFSVParamsDataclass
        Transformed parameters in unconstrained space.
    """
    # Create a copy to avoid modifying the original
    result = copy.deepcopy(params)

    # --- Phi_f and Phi_h Transformation (Diagonal inverse_softplus) ---
    # Ensures diagonal elements of Phi_f/Phi_h are positive after untransformation.
    # Off-diagonals are left unconstrained.
    diag_indices_phi_f = jnp.diag_indices_from(params.Phi_f)
    transformed_phi_f = params.Phi_f.at[diag_indices_phi_f].set(
        inverse_softplus(jnp.diag(params.Phi_f))
    )
    diag_indices_phi_h = jnp.diag_indices_from(params.Phi_h)
    transformed_phi_h = params.Phi_h.at[diag_indices_phi_h].set(
        inverse_softplus(jnp.diag(params.Phi_h))
    )

    # --- Variance/Covariance Transformations ---
    # Transform variance parameters (must be positive) using inverse softplus
    # Only transform diagonal elements of sigma2 (force off-diagonals to zero)
    if params.sigma2.ndim > 1:  # Handle matrix case (already diagonal from validation)
        diag_indices_sigma = jnp.diag_indices_from(params.sigma2)
        transformed_sigma2 = jnp.zeros_like(params.sigma2)
        # Apply inverse softplus to get unconstrained values
        transformed_sigma2 = transformed_sigma2.at[diag_indices_sigma].set(
            inverse_softplus(jnp.diag(params.sigma2))
        )
    else: # Should not happen if validation runs, but handle just in case
        transformed_sigma2 = inverse_softplus(params.sigma2)

    # Transform Q_h (log-volatility noise covariance) - Assuming diagonal
    # Using inverse softplus
    diag_q_h = jnp.diag(params.Q_h)
    transformed_diag_q_h = inverse_softplus(diag_q_h)
    transformed_q_h = jnp.diag(transformed_diag_q_h) # Keep diagonal structure

    # Note: mu is typically unconstrained.
    # lambda_r diagonal is fixed to 1, off-diagonals are unconstrained.
    # No transformation needed for lambda_r itself.

    # Return a new params object with transformed values
    # lambda_r is NOT transformed as its diagonal is fixed and off-diagonals are unconstrained.
    return result.replace(
        Phi_f=transformed_phi_f,
        Phi_h=transformed_phi_h,
        sigma2=transformed_sigma2,
        Q_h=transformed_q_h
        # lambda_r is intentionally omitted from replace()
    )

def untransform_params(transformed_params: DFSVParamsDataclass) -> DFSVParamsDataclass:
    """
    Transform parameters back from unconstrained to constrained space.

    - Applies `softplus` to diagonal elements of `Phi_f` and `Phi_h`.
      Off-diagonals remain unconstrained.
    - Applies `softplus` to diagonal elements of `sigma2` and `Q_h`.
    - Leaves `mu` and `lambda_r` unchanged.

    Parameters:
    -----------
    transformed_params : DFSVParamsDataclass
        Transformed parameters in unconstrained space.

    Returns:
    --------
    DFSVParamsDataclass
        Parameters in their natural (constrained) space.
    """
    # --- Phi_f and Phi_h Untransformation (Diagonal softplus) ---
    # Ensures diagonal elements of Phi_f/Phi_h are positive.
    # Off-diagonals are left unconstrained.
    diag_indices_phi_f = jnp.diag_indices_from(transformed_params.Phi_f)
    phi_f_original = transformed_params.Phi_f.at[diag_indices_phi_f].set(
        softplus(jnp.diag(transformed_params.Phi_f))
    )
    diag_indices_phi_h = jnp.diag_indices_from(transformed_params.Phi_h)
    phi_h_original = transformed_params.Phi_h.at[diag_indices_phi_h].set(
        softplus(jnp.diag(transformed_params.Phi_h))
    )

    # --- Variance/Covariance Untransformations ---
    # Apply softplus to transform back variance parameters
    # Handle matrix vs vector case for sigma2 (expecting diagonal matrix)
    if transformed_params.sigma2.ndim > 1:
        # For matrix case, only untransform diagonal elements
        diag_indices_sigma = jnp.diag_indices_from(transformed_params.sigma2)
        sigma2_original = jnp.zeros_like(transformed_params.sigma2)  # Start with zeros
        # Apply softplus to transform from unconstrained to positive values
        sigma2_original = sigma2_original.at[diag_indices_sigma].set(
            softplus(transformed_params.sigma2[diag_indices_sigma])
        )
    else: # Should not happen
        sigma2_original = softplus(transformed_params.sigma2)

    # Untransform Q_h (diagonal) using softplus
    diag_q_h_orig = softplus(jnp.diag(transformed_params.Q_h))
    q_h_original = jnp.diag(diag_q_h_orig)

    # lambda_r diagonal is fixed to 1 and was not transformed.
    # Off-diagonals remained unconstrained.
    # No untransformation needed for lambda_r itself.

    # Return a new params object with untransformed values
    # lambda_r is intentionally omitted as it wasn't transformed.
    return transformed_params.replace(
        Phi_f=phi_f_original,
        Phi_h=phi_h_original,
        sigma2=sigma2_original,
        Q_h=q_h_original
        # lambda_r is intentionally omitted from replace()
    )
def apply_identification_constraint(params: DFSVParamsDataclass) -> DFSVParamsDataclass:
    """Applies lower-triangular constraint with diagonal fixed to 1 to lambda_r."""
    # Apply lower-triangular constraint first
    constrained_lambda_r = jnp.tril(params.lambda_r)
    # Set diagonal elements to 1.0
    diag_indices = jnp.diag_indices(n=min(params.N, params.K), ndim=2)
    constrained_lambda_r = constrained_lambda_r.at[diag_indices].set(1.0)
    return params.replace(lambda_r=constrained_lambda_r)
