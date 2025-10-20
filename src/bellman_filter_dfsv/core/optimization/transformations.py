"""
Parameter transformation functions for DFSV models.

Maps constrained parameters (e.g., variances > 0, correlations in [-1, 1])
to unconstrained space for optimization, and back.
"""

import jax.numpy as jnp
from jax.nn import softplus

from bellman_filter_dfsv.core.models.dfsv import DFSVParamsDataclass

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
    return jnp.where(
        x_safe < 1e-3,
        jnp.log(x_safe),  # Approximation for small values
        jnp.log(jnp.exp(x_safe) - 1.0),
    )


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


def transform_params(params: DFSVParamsDataclass) -> DFSVParamsDataclass:
    """
    Transform bounded parameters to unconstrained space for optimization.

    - Applies `safe_arctanh` to diagonal elements of `Phi_f` and `Phi_h`.
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
    # --- Phi_f and Phi_h Transformation (Diagonal safe_arctanh) ---
    diag_phi_f = jnp.diag(params.Phi_f)
    transformed_diag_phi_f = safe_arctanh(diag_phi_f)
    transformed_phi_f = params.Phi_f.at[jnp.diag_indices_from(params.Phi_f)].set(transformed_diag_phi_f)

    diag_phi_h = jnp.diag(params.Phi_h)
    transformed_diag_phi_h = safe_arctanh(diag_phi_h)
    transformed_phi_h = params.Phi_h.at[jnp.diag_indices_from(params.Phi_h)].set(transformed_diag_phi_h)

    # --- Variance/Covariance Transformations ---
    if params.sigma2.ndim > 1:
        diag_sigma = jnp.diag(params.sigma2)
        transformed_sigma2 = jnp.diag(inverse_softplus(diag_sigma))
    else:
        transformed_sigma2 = inverse_softplus(params.sigma2)

    diag_q_h = jnp.diag(params.Q_h)
    transformed_q_h = jnp.diag(inverse_softplus(diag_q_h))

    return params.replace(
        Phi_f=transformed_phi_f,
        Phi_h=transformed_phi_h,
        sigma2=transformed_sigma2,
        Q_h=transformed_q_h,
    )


def untransform_params(transformed_params: DFSVParamsDataclass) -> DFSVParamsDataclass:
    """
    Transform parameters back from unconstrained to constrained space.

    - Applies `tanh` to diagonal elements of `Phi_f` and `Phi_h`.
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
    # --- Phi_f and Phi_h Untransformation (Diagonal tanh) ---
    diag_phi_f = jnp.diag(transformed_params.Phi_f)
    phi_f_original = transformed_params.Phi_f.at[jnp.diag_indices_from(transformed_params.Phi_f)].set(jnp.tanh(diag_phi_f))

    diag_phi_h = jnp.diag(transformed_params.Phi_h)
    phi_h_original = transformed_params.Phi_h.at[jnp.diag_indices_from(transformed_params.Phi_h)].set(jnp.tanh(diag_phi_h))

    # --- Variance/Covariance Untransformations ---
    if transformed_params.sigma2.ndim > 1:
        diag_sigma = jnp.diag(transformed_params.sigma2)
        sigma2_original = jnp.diag(softplus(diag_sigma))
    else:
        sigma2_original = softplus(transformed_params.sigma2)

    diag_q_h = jnp.diag(transformed_params.Q_h)
    q_h_original = jnp.diag(softplus(diag_q_h))

    return transformed_params.replace(
        Phi_f=phi_f_original,
        Phi_h=phi_h_original,
        sigma2=sigma2_original,
        Q_h=q_h_original,
    )

def apply_identification_constraint(params: DFSVParamsDataclass) -> DFSVParamsDataclass:
    """Applies lower-triangular constraint with diagonal fixed to 1 to lambda_r.

    For the factor loading matrix lambda_r with shape (N, K):
    1. Makes it lower triangular (zeros above the main diagonal)
    2. Sets the first K diagonal elements to 1.0
    3. For N > K, only the first K columns have the constraint applied
    """
    K = params.K  # K should be a static attribute
    lambda_r = params.lambda_r

    # 1. Zero out elements above the diagonal for the whole matrix
    tril_lambda = jnp.tril(lambda_r, k=0)

    # 2. Set the first K diagonal elements to 1.0
    #    Create indices for the diagonal elements up to K.
    #    .at[] handles out-of-bounds indices gracefully (ignores them),
    #    so we don't need explicit clipping by N if K is static.
    diag_indices_k = jnp.arange(K)  # K must be static for this to work under JIT
    constrained_lambda_r = tril_lambda.at[diag_indices_k, diag_indices_k].set(1.0)

    return params.replace(lambda_r=constrained_lambda_r)
