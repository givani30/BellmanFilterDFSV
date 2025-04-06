# src/bellman_filter_dfsv/core/filters/_bellman_impl.py
"""Static JAX implementations of core Bellman filter calculations.

This module provides standalone, JAX-based functions for computations
used within the Bellman filters (both covariance and information forms),
such as building covariance matrices, calculating Fisher information,
log posteriors, and likelihood penalties. These functions are designed
to be JIT-compiled by the filter classes that use them.
"""
from functools import partial
from typing import Callable
from jax.experimental import checkify
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg

# Type hint for build_covariance function signature
BuildCovarianceFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]


# Note: No JIT decorators here; JITting happens in the main class setup

def build_covariance_impl(
    lambda_r: jnp.ndarray, exp_h: jnp.ndarray, sigma2: jnp.ndarray
) -> jnp.ndarray:
    """Builds the observation covariance matrix Sigma_t = Lambda * Sigma_f * Lambda^T + Sigma_e.

    Args:
        lambda_r: Factor loading matrix Lambda (N, K).
        exp_h: Exponentiated log-volatilities diag(Sigma_f) = exp(h_t) (K,).
        sigma2: Idiosyncratic variances diag(Sigma_e) (N,).

    Returns:
        The observation covariance matrix Sigma_t (N, N).
    """
    K = lambda_r.shape[1]
    N = lambda_r.shape[0]

    # Ensure sigma2 is a 1D array for diagonal construction
    sigma2_1d = sigma2.flatten() if sigma2.ndim > 1 else sigma2
    Sigma_e = jnp.diag(sigma2_1d)

    Sigma_f = jnp.diag(exp_h.flatten()) # Ensure exp_h is 1D
    lambda_r = lambda_r.reshape(N, K) # Ensure correct shape

    # Calculate Sigma_t = Lambda * Sigma_f * Lambda^T + Sigma_e
    A = lambda_r @ Sigma_f @ lambda_r.T + Sigma_e + 1e-6 * jnp.eye(N) # Add jitter
    A = 0.5 * (A + A.T) # Ensure symmetry
    return A


# def fisher_information_impl(
#     lambda_r: jnp.ndarray,
#     sigma2: jnp.ndarray,
#     alpha: jnp.ndarray,
#     observation: jnp.ndarray,
#     K: int, # Pass K explicitly
#     build_covariance_fn: BuildCovarianceFn # Pass build_covariance dependency
# ) -> jnp.ndarray:
#     """
#     Static implementation of fisher_information. (Note: This calculates an approximation, not the full Observed FIM)
#
#     Args:
#         lambda_r (jnp.ndarray): Factor loading matrix (N, K).
#         sigma2 (jnp.ndarray): Idiosyncratic variances (N, N) or (N,).
#         alpha (jnp.ndarray): State vector [f, h] (2K,).
#         K (int): Number of factors.
#         build_covariance_fn (Callable): Function to build the covariance matrix.
#
#     Returns:
#         jnp.ndarray: Fisher information matrix (2K, 2K).
#     """
#     # ... implementation commented out ...

# This is the old fisher_information_impl - keeping it commented out
# def fisher_information_impl(
#     lambda_r: jnp.ndarray,
#     sigma2: jnp.ndarray,
#     alpha: jnp.ndarray,
#     observation: jnp.ndarray,
#     K: int, # Pass K explicitly
# ) -> jnp.ndarray:
#     """
#     Static implementation of the Observed Fisher Information (Negative Hessian).
#
#     Computes J = - d^2(log_lik) / d(alpha) d(alpha)^T using efficient Woodbury identity approach.
#
#     Args:
#         lambda_r (jnp.ndarray): Factor loading matrix (N, K).
#         sigma2 (jnp.ndarray): Idiosyncratic variances (N, N) or (N,).
#         alpha (jnp.ndarray): State vector [f, h] (2K,).
#         observation (jnp.ndarray): Observation vector (N,).
#         K (int): Number of factors.
#
#     Returns:
#         jnp.ndarray: Observed Fisher Information matrix (Negative Hessian) (2K, 2K).
#     """
    # ... implementation commented out ...

def observed_fim_impl(
    lambda_r: jnp.ndarray,
    sigma2: jnp.ndarray,
    alpha: jnp.ndarray,
    observation: jnp.ndarray,
    K: int, # Pass K explicitly
) -> jnp.ndarray:
    """Calculates the Observed Fisher Information (Negative Hessian).

    Computes J = - d^2(log p(y_t|alpha_t)) / d(alpha_t) d(alpha_t)^T using an
    efficient approach based on the Woodbury matrix identity.

    Args:
        lambda_r: Factor loading matrix Lambda (N, K).
        sigma2: Idiosyncratic variances diag(Sigma_e) (N,).
        alpha: State vector alpha_t = [f_t, h_t] (state_dim,).
        observation: Observation vector y_t (N,).
        K: Number of factors.

    Returns:
        The Observed Fisher Information matrix J_observed (state_dim, state_dim).
    """
    N = lambda_r.shape[0]

    alpha = alpha.flatten()
    observation = observation.flatten()

    f = alpha[:K]
    h = alpha[K:]
    exp_h = jnp.exp(h)

    r = observation - lambda_r @ f # Residuals

    # Ensure sigma2 is 1D for diagonal matrix operations
    sigma2_1d = sigma2.flatten() if sigma2.ndim > 1 else sigma2
    jitter = 1e-8
    Dinv_diag = 1.0 / (sigma2_1d + jitter) # Inverse of Sigma_e (diagonal)
    Cinv_diag = 1.0 / (exp_h + jitter)     # Inverse of Sigma_f (diagonal)

    Dinv_lambda_r = lambda_r * Dinv_diag[:, None] # Sigma_e^-1 @ Lambda
    Dinv_r = r * Dinv_diag                       # Sigma_e^-1 @ r

    # M = Cinv + U.T @ Dinv @ U = Sigma_f^-1 + Lambda^T @ Sigma_e^-1 @ Lambda
    M = jnp.diag(Cinv_diag) + lambda_r.T @ Dinv_lambda_r

    # Cholesky decomposition for stable inversion of M
    # Add jitter for numerical stability before Cholesky
    M_jittered = M + 1e-6 * jnp.eye(K)
    L_M = jax.scipy.linalg.cholesky(M_jittered, lower=True)

    # Calculate I_ff = Lambda^T @ Sigma_t^-1 @ Lambda using Woodbury
    # I_ff = Lambda^T @ (Dinv - Dinv U Minv U^T Dinv) @ Lambda
    # I_ff = Lambda^T@Dinv@Lambda - (Lambda^T@Dinv@U) @ Minv @ (U^T@Dinv@Lambda)
    # I_ff = (M - Cinv) - (M - Cinv) @ Minv @ (M - Cinv)
    V = M - jnp.diag(Cinv_diag) # V = Lambda^T @ Sigma_e^-1 @ Lambda
    Z = jax.scipy.linalg.cho_solve((L_M, True), V) # Z = M^-1 @ V
    I_ff = V - V @ Z # = V - V M^-1 V

    # Calculate P = Lambda^T @ Sigma_t^-1 @ r
    v = lambda_r.T @ Dinv_r # v = Lambda^T @ Sigma_e^-1 @ r
    z_p = jax.scipy.linalg.cho_solve((L_M, True), v) # z_p = M^-1 @ v
    Ainv_r = Dinv_r - Dinv_lambda_r @ z_p # Ainv_r = Sigma_t^-1 @ r
    P = lambda_r.T @ Ainv_r # P = Lambda^T @ Sigma_t^-1 @ r


    # Calculate blocks of the Hessian J = - d^2(log_lik) / d(alpha) d(alpha)^T
    J_ff = I_ff
    # J_fh[l, k] = exp(h_k) * I_ff[l, k] * P[k] (derived from Hessian calculation)
    J_fh = I_ff * P[None, :] * exp_h[None, :]

    # J_hh calculation (more complex derivation)
    exp_h_outer = jnp.outer(exp_h, exp_h)
    P_outer = jnp.outer(P, P)
    term1_diag = 0.5 * exp_h * (jnp.diag(I_ff) - P**2)
    term2 = -0.5 * exp_h_outer * I_ff * (I_ff - 2 * P_outer)
    J_hh = jnp.diag(term1_diag) + term2

    # Assemble the full Hessian matrix
    J = jnp.block([[J_ff, J_fh], [J_fh.T, J_hh]])
    J = 0.5 * (J + J.T) # Ensure symmetry

    return J


def log_posterior_impl(
    lambda_r: jnp.ndarray,
    sigma2: jnp.ndarray,
    alpha: jnp.ndarray,
    observation: jnp.ndarray,
    K: int, # Pass K explicitly
    build_covariance_fn: BuildCovarianceFn # Pass build_covariance dependency
) -> float:
    """Calculates the log posterior log p(y_t | alpha_t).

    Uses the Woodbury matrix identity and Matrix Determinant Lemma for efficient
    calculation of the log-likelihood of the observation given the current state.

    Args:
        lambda_r: Factor loading matrix Lambda (N, K).
        sigma2: Idiosyncratic variances diag(Sigma_e) (N,).
        alpha: State vector alpha_t = [f_t, h_t] (state_dim,).
        observation: Observation vector y_t (N,).
        K: Number of factors.
        build_covariance_fn: Function to build the observation covariance matrix
            Sigma_t = Lambda Sigma_f Lambda^T + Sigma_e. (Note: This dependency
            might be removable if calculation is done via Woodbury as below).

    Returns:
        The log posterior value log p(y_t | alpha_t) (scalar float).
    """
    N = lambda_r.shape[0]
    alpha = alpha.flatten()
    observation = observation.flatten()

    f = alpha[:K]
    log_vols = alpha[K:]
    exp_log_vols = jnp.exp(log_vols)

    pred_obs = lambda_r @ f
    innovation = observation - pred_obs

    # --- Use Woodbury Identity & Matrix Determinant Lemma ---
    # Sigma_t = D + U C U.T where D=Sigma_e, U=Lambda, C=Sigma_f
    sigma2_1d = sigma2.flatten() if sigma2.ndim > 1 else sigma2
    jitter = 1e-8
    Dinv_diag = 1.0 / (sigma2_1d + jitter)
    Cinv_diag = 1.0 / (exp_log_vols + jitter)

    # Precompute terms
    Dinv_lambda_r = lambda_r * Dinv_diag[:, None] # D^-1 @ U
    Dinv_innovation = innovation * Dinv_diag      # D^-1 @ innovation

    # Compute M = Cinv + U.T @ Dinv @ U
    M = jnp.diag(Cinv_diag) + lambda_r.T @ Dinv_lambda_r

    # Cholesky decomposition of M for stable inversion and logdet
    # Add jitter for numerical stability before Cholesky
    M_jittered = M + 1e-6 * jnp.eye(K)
    L_M = jax.scipy.linalg.cholesky(M_jittered, lower=True)

    # Calculate log determinant of Sigma_t using Matrix Determinant Lemma
    # logdet(Sigma_t) = logdet(M) + logdet(C) + logdet(D)
    logdet_M = 2.0 * jnp.sum(jnp.log(jnp.maximum(jnp.diag(L_M), 1e-10)))
    logdet_C = jnp.sum(log_vols) # logdet(diag(exp(h))) = sum(h)
    logdet_D = jnp.sum(jnp.log(jnp.maximum(sigma2_1d, 1e-10)))
    logdet_Sigma_t = logdet_M + logdet_C + logdet_D

    # Calculate quadratic form: innovation.T @ Sigma_t^-1 @ innovation
    # Sigma_t^-1 = Dinv - Dinv U Minv U.T Dinv
    # quad_form = innovation.T @ Dinv @ innovation - (innovation.T @ Dinv @ U) @ Minv @ (U.T @ Dinv @ innovation)
    term1 = jnp.dot(innovation, Dinv_innovation) # innovation.T @ Dinv @ innovation
    v = lambda_r.T @ Dinv_innovation             # v = U.T @ Dinv @ innovation
    z = jax.scipy.linalg.cho_solve((L_M, True), v) # z = Minv @ v
    term2 = jnp.dot(v, z)                         # term2 = v.T @ Minv @ v
    quad_form = term1 - term2

    # Calculate log likelihood: -0.5 * (N*log(2pi) + logdet(Sigma_t) + quad_form)
    # Constant term -0.5 * N * log(2pi) is often omitted for optimization
    log_lik = -0.5 * (logdet_Sigma_t + quad_form)

    return log_lik



def bif_likelihood_penalty_impl(
    a_pred: jnp.ndarray,      # Predicted state alpha_{t|t-1} (flattened)
    a_updated: jnp.ndarray,   # Updated state alpha_{t|t} (flattened)
    Omega_pred: jnp.ndarray,  # Predicted information Omega_{t|t-1}
    Omega_post: jnp.ndarray   # Updated information Omega_{t|t}
) -> jnp.ndarray:
    """Calculates the BIF pseudo-likelihood penalty term.

    This term approximates the KL divergence between the posterior and prior
    predictive distributions, used in the augmented log-likelihood calculation
    (Lange et al., 2024, Eq. 40).

    Formula:
    penalty = 0.5 * (log_det(Omega_post) - log_det(Omega_pred)
                     + diff^T @ Omega_pred @ diff)
    where diff = a_updated - a_pred.

    Args:
        a_pred: Predicted state mean alpha_{t|t-1} (state_dim,).
        a_updated: Updated state mean alpha_{t|t} (state_dim,).
        Omega_pred: Predicted information matrix Omega_{t|t-1}
                    (state_dim, state_dim).
        Omega_post: Updated information matrix Omega_{t|t}
                    (state_dim, state_dim).

    Returns:
        The calculated penalty term (JAX scalar).
    """
    a_pred_flat = a_pred.flatten()
    a_updated_flat = a_updated.flatten()

    # Calculate log-determinants using stable method (slogdet)
    jitter = 1e-8
    sign_pred, log_det_Omega_pred = jnp.linalg.slogdet(Omega_pred + jitter * jnp.eye(Omega_pred.shape[0]))
    sign_post, log_det_Omega_post = jnp.linalg.slogdet(Omega_post + jitter * jnp.eye(Omega_post.shape[0]))

    # Calculate quadratic term: diff^T @ Omega_pred @ diff
    diff = a_updated_flat - a_pred_flat
    quad_term = diff.T @ Omega_pred @ diff

    # Compute penalty
    penalty = 0.5 * (log_det_Omega_post - log_det_Omega_pred + quad_term)

    # Ensure the result is a scalar
    return jnp.asarray(penalty, dtype=jnp.float64)

# Removed kl_penalty_impl as it's no longer used by bellman.py
# def kl_penalty_impl(
#     a_pred: jnp.ndarray,
#     a_updated: jnp.ndarray,
#     I_pred: jnp.ndarray,
#     I_updated: jnp.ndarray,
# ) -> float:
#     """
#     Static implementation of the KL penalty term.
#
#     Args:
#         a_pred: Predicted state mean.
#         a_updated: Updated state mean.
#         I_pred: Predicted state precision.
#         I_updated: Updated state precision.
#
#     Returns:
#         KL divergence penalty value.
#     """
    # ... implementation commented out ...