# src/bellman_filter_dfsv/core/filters/_bellman_impl.py
"""
Static JAX implementations of core Bellman filter calculations.
"""
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy.linalg

# Type hint for build_covariance function signature
BuildCovarianceFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]


# Note: No JIT decorators here; JITting happens in the main class setup

def build_covariance_impl(
    lambda_r: jnp.ndarray, exp_h: jnp.ndarray, sigma2: jnp.ndarray
) -> jnp.ndarray:
    """
    Static implementation of build_covariance.

    Args:
        lambda_r (jnp.ndarray): Factor loading matrix (N, K).
        exp_h (jnp.ndarray): Exponentiated log-volatilities (K,).
        sigma2 (jnp.ndarray): Idiosyncratic variances (N, N) or (N,).

    Returns:
        jnp.ndarray: Covariance matrix (N, N).
    """
    K = lambda_r.shape[1]
    N = lambda_r.shape[0]

    # Ensure sigma2 is a 2D diagonal matrix if 1D
    sigma2_mat = jnp.diag(sigma2) if sigma2.ndim == 1 else sigma2

    Sigma_f = jnp.diag(exp_h.flatten()) # Ensure exp_h is 1D
    lambda_r = lambda_r.reshape(N, K) # Ensure correct shape
    A = lambda_r @ Sigma_f @ lambda_r.T
    # Use diag(diag(sigma2)) to ensure only diagonal elements are added
    A += jnp.diag(jnp.diag(sigma2_mat)) + 1e-6 * jnp.eye(N)
    A = 0.5 * (A + A.T) # Ensure symmetry
    return A


def fisher_information_impl(
    lambda_r: jnp.ndarray,
    sigma2: jnp.ndarray,
    alpha: jnp.ndarray,
    K: int, # Pass K explicitly
    build_covariance_fn: BuildCovarianceFn # Pass build_covariance dependency
) -> jnp.ndarray:
    """
    Static implementation of fisher_information. (Note: This calculates an approximation, not the full Observed FIM)

    Args:
        lambda_r (jnp.ndarray): Factor loading matrix (N, K).
        sigma2 (jnp.ndarray): Idiosyncratic variances (N, N) or (N,).
        alpha (jnp.ndarray): State vector [f, h] (2K,).
        K (int): Number of factors.
        build_covariance_fn (Callable): Function to build the covariance matrix.

    Returns:
        jnp.ndarray: Fisher information matrix (2K, 2K).
    """
    N = lambda_r.shape[0]
    h = alpha[K:]
    exp_h = jnp.exp(h)

    # --- Optimized FIM using Woodbury Identity & Rank-1 Reformulation ---
    # A = D + U C U.T where D=diag(sigma2), U=lambda_r, C=diag(exp(h))
    # Ainv = Dinv - Dinv U (Cinv + U.T Dinv U)^-1 U.T Dinv
    # Let M = Cinv + U.T Dinv U

    # Ensure sigma2 is 1D for easy diagonal inversion
    sigma2_1d = sigma2 if sigma2.ndim == 1 else jnp.diag(sigma2)
    # Add small jitter for numerical stability before inversion
    Dinv_diag = 1.0 / (sigma2_1d + 1e-8)
    Cinv_diag = 1.0 / (exp_h + 1e-8)

    # Precompute Dinv @ U
    Dinv_lambda_r = lambda_r * Dinv_diag[:, None] # (N, K)

    # Compute M = Cinv + U.T @ Dinv @ U (K x K matrix)
    M = jnp.diag(Cinv_diag) + lambda_r.T @ Dinv_lambda_r # (K,K)

    # Cholesky decomposition of M for stable inversion
    try:
        # Use cho_factor for consistency if needed, but cholesky is fine
        L_M = jax.scipy.linalg.cholesky(M, lower=True)
    except jnp.linalg.LinAlgError:
        # Fallback if M is not positive definite
        M_jittered = M + 1e-6 * jnp.eye(K)
        L_M = jax.scipy.linalg.cholesky(M_jittered, lower=True)

    # --- Compute U_matrix = A_inv @ lambda_r ---
    # We need to compute Ainv @ lambda_r using Woodbury:
    # Ainv @ U = Dinv @ U - Dinv @ U @ (Minv @ (U.T @ Dinv @ U))
    # Let V = U.T @ Dinv @ U = lambda_r.T @ Dinv_lambda_r (K x K matrix)
    V = M - jnp.diag(Cinv_diag) # (K, K)

    # Solve M @ Z = V for Z using Cholesky: L_M L_M.T Z = V
    # Z = Minv @ V
    Z = jax.scipy.linalg.cho_solve((L_M, True), V) # (K, K)

    # U_matrix = Dinv @ U - (Dinv @ U) @ Z
    U_matrix = Dinv_lambda_r - Dinv_lambda_r @ Z # (N, K)

    # --- Compute I_ff (mean_part) ---
    # I_ff = lambda_r.T @ A_inv @ lambda_r = lambda_r.T @ U_matrix
    I_ff = lambda_r.T @ U_matrix # (K, K)

    # --- Compute I_hh (cov_part_block) using Rank-1 Reformulation ---
    # I_hh[k, l] = 0.5 * exp(h_k + h_l) * (lambda_r_k.T @ A_inv @ lambda_r_l)^2
    # Note that lambda_r_k.T @ A_inv @ lambda_r_l is exactly I_ff[k, l]
    # Use outer product for exp(h_k + h_l)
    exp_h_outer = jnp.outer(exp_h, exp_h) # (K, K)
    I_hh = 0.5 * exp_h_outer * (I_ff ** 2) # (K, K)

    # --- Assemble ---
    I_fh = jnp.zeros((K, K)) # Block diagonal assumption
    I_fisher = jnp.block([[I_ff, I_fh], [I_fh.T, I_hh]])

    # Ensure symmetry (numerical precision might cause small asymmetries)
    I_fisher = 0.5 * (I_fisher + I_fisher.T)

    return I_fisher



def observed_fim_impl(
    lambda_r: jnp.ndarray,
    sigma2: jnp.ndarray,
    alpha: jnp.ndarray,
    observation: jnp.ndarray,
    K: int, # Pass K explicitly
) -> jnp.ndarray:
    """
    Static implementation of the Observed Fisher Information (Negative Hessian).

    Computes J = - d^2(log_lik) / d(alpha) d(alpha)^T using efficient Woodbury identity approach.

    Args:
        lambda_r (jnp.ndarray): Factor loading matrix (N, K).
        sigma2 (jnp.ndarray): Idiosyncratic variances (N, N) or (N,).
        alpha (jnp.ndarray): State vector [f, h] (2K,).
        observation (jnp.ndarray): Observation vector (N,).
        K (int): Number of factors.

    Returns:
        jnp.ndarray: Observed Fisher Information matrix (Negative Hessian) (2K, 2K).
    """
    N = lambda_r.shape[0]

    alpha = alpha.flatten()
    observation = observation.flatten()

    f = alpha[:K]
    h = alpha[K:]
    exp_h = jnp.exp(h)

    r = observation - lambda_r @ f


    sigma2_1d = sigma2 if sigma2.ndim == 1 else jnp.diag(sigma2)
    jitter = 1e-8
    Dinv_diag = 1.0 / (sigma2_1d + jitter)
    Cinv_diag = 1.0 / (exp_h + jitter)

    Dinv_lambda_r = lambda_r * Dinv_diag[:, None]
    Dinv_r = r * Dinv_diag

    M = jnp.diag(Cinv_diag) + lambda_r.T @ Dinv_lambda_r

    try:
        L_M = jax.scipy.linalg.cholesky(M, lower=True)
    except jnp.linalg.LinAlgError:
        M_jittered = M + 1e-6 * jnp.eye(K)
        L_M = jax.scipy.linalg.cholesky(M_jittered, lower=True)

    V = M - jnp.diag(Cinv_diag)
    Z = jax.scipy.linalg.cho_solve((L_M, True), V)
    I_ff = V - V @ Z


    v = lambda_r.T @ Dinv_r
    z_p = jax.scipy.linalg.cho_solve((L_M, True), v)
    Ainv_r = Dinv_r - Dinv_lambda_r @ z_p
    P = lambda_r.T @ Ainv_r


    J_ff = I_ff
    J_fh = exp_h[:, None] * I_ff * P[None, :]


    exp_h_outer = jnp.outer(exp_h, exp_h)
    P_outer = jnp.outer(P, P)
    term1_diag = 0.5 * exp_h * (jnp.diag(I_ff) - P**2)
    term2 = -0.5 * exp_h_outer * I_ff * (I_ff - 2 * P_outer)
    J_hh = jnp.diag(term1_diag)
    J_hh += jnp.where(jnp.eye(K, dtype=bool), 0.0, term2)

    J = jnp.block([[J_ff, J_fh], [J_fh.T, J_hh]])
    J = 0.5 * (J + J.T)


    return J


def log_posterior_impl(
    lambda_r: jnp.ndarray,
    sigma2: jnp.ndarray,
    alpha: jnp.ndarray,
    observation: jnp.ndarray,
    K: int, # Pass K explicitly
    build_covariance_fn: BuildCovarianceFn # Pass build_covariance dependency
) -> float:
    """
    Static implementation of log_posterior. This is the likelihood of the observation given the state.

    Args:
        lambda_r (jnp.ndarray): Factor loading matrix (N, K).
        sigma2 (jnp.ndarray): Idiosyncratic variances (N, N) or (N,).
        alpha (jnp.ndarray): State vector [f, h] (2K,).
        observation (jnp.ndarray): Observation vector (N,).
        K (int): Number of factors.
        build_covariance_fn (Callable): Function to build the covariance matrix.

    Returns:
        float: Log posterior value.
    """
    N = lambda_r.shape[0]
    alpha = alpha.flatten()
    observation = observation.flatten()

    f = alpha[:K]
    log_vols = alpha[K:]

    pred_obs = lambda_r @ f
    innovation = observation - pred_obs
    exp_log_vols = jnp.exp(log_vols)

    # --- Use Woodbury Identity & Matrix Determinant Lemma ---
    # A = D + U C U.T
    # D = diag(sigma2) -> Dinv = diag(1/sigma2)
    # U = lambda_r
    # C = diag(exp(h)) -> Cinv = diag(1/exp(h))

    # Ensure sigma2 is 1D for easy diagonal inversion
    sigma2_1d = sigma2 if sigma2.ndim == 1 else jnp.diag(sigma2)
    # Add small jitter for numerical stability before inversion
    Dinv_diag = 1.0 / (sigma2_1d + 1e-8)
    Cinv_diag = 1.0 / (exp_log_vols + 1e-8)

    # Precompute Dinv @ U and Dinv @ innovation
    Dinv_lambda_r = lambda_r * Dinv_diag[:, None] # Equivalent to diag(Dinv) @ lambda_r
    Dinv_innovation = innovation * Dinv_diag

    # Compute M = Cinv + U.T @ Dinv @ U (K x K matrix)
    M = jnp.diag(Cinv_diag) + lambda_r.T @ Dinv_lambda_r

    # Cholesky decomposition of M for stable inversion and logdet
    try:
        L_M = jax.scipy.linalg.cholesky(M, lower=True)
    except jnp.linalg.LinAlgError:
        # Fallback if M is not positive definite (should not happen in theory)
        # Add jitter and retry
        M_jittered = M + 1e-6 * jnp.eye(K)
        L_M = jax.scipy.linalg.cholesky(M_jittered, lower=True)

    # Calculate log determinant of A using Matrix Determinant Lemma
    # logdet(A) = logdet(M) + logdet(C) + logdet(D)
    # Note: logdet(M) = 2 * sum(log(diag(L_M)))
    #       logdet(C) = sum(log(exp(h))) = sum(h)
    #       logdet(D) = sum(log(sigma2))
    logdet_M = 2.0 * jnp.sum(jnp.log(jnp.maximum(jnp.diag(L_M), 1e-10)))
    logdet_C = jnp.sum(log_vols)
    logdet_D = jnp.sum(jnp.log(jnp.maximum(sigma2_1d, 1e-10)))
    logdet_A = logdet_M + logdet_C + logdet_D

    # Calculate quadratic form: innovation.T @ Ainv @ innovation
    # Ainv = Dinv - Dinv @ U @ Minv @ U.T @ Dinv
    # quad_form = innovation.T @ Dinv @ innovation - (innovation.T @ Dinv @ U) @ Minv @ (U.T @ Dinv @ innovation)

    # Term 1: innovation.T @ Dinv @ innovation
    term1 = jnp.dot(innovation, Dinv_innovation)

    # Term 2: (innovation.T @ Dinv @ U) @ Minv @ (U.T @ Dinv @ innovation)
    # Let v = U.T @ Dinv @ innovation (K-dim vector)
    v = lambda_r.T @ Dinv_innovation
    # Solve M @ z = v for z using Cholesky: L_M L_M.T z = v
    z = jax.scipy.linalg.cho_solve((L_M, True), v) # Solves M x = v -> z = Minv @ v
    term2 = jnp.dot(v, z) # v.T @ Minv @ v

    quad_form = term1 - term2
    #Calculate constant term (commented out because it's not needed for parameter optimization)
    # constant= -0.5 * N * jnp.log(2.0 * jnp.pi) 
    # Calculate log likelihood
    log_lik = -0.5 * ( logdet_A + quad_form)

    return log_lik


def kl_penalty_impl(
    a_pred: jnp.ndarray,
    a_updated: jnp.ndarray,
    I_pred: jnp.ndarray,
    I_updated: jnp.ndarray,
) -> float:
    """
    Static implementation of the KL penalty term.

    Args:
        a_pred: Predicted state mean.
        a_updated: Updated state mean.
        I_pred: Predicted state precision.
        I_updated: Updated state precision.

    Returns:
        KL divergence penalty value.
    """
    # Ensure inputs are flattened
    a_pred = a_pred.flatten()
    a_updated = a_updated.flatten()

    # Compute Cholesky decompositions for log determinants
    L_pred = jax.scipy.linalg.cholesky(I_pred, lower=True)
    L_updated = jax.scipy.linalg.cholesky(I_updated, lower=True)

    # Log determinants (using safe log)
    log_det_Ipred = 2.0 * jnp.sum(jnp.log(jnp.maximum(jnp.diag(L_pred), 1e-10)))
    log_det_Iupdated = 2.0 * jnp.sum(jnp.log(jnp.maximum(jnp.diag(L_updated), 1e-10)))

    # Compute inverse of I_updated (updated covariance) using Cholesky
    P_updated = jax.scipy.linalg.cho_solve((L_updated, True), jnp.eye(I_updated.shape[0]))

    # Trace term: trace(I_pred @ P_updated)
    trace_term = jnp.trace(I_pred @ P_updated)

    # Quadratic term: (a_updated - a_pred).T @ I_pred @ (a_updated - a_pred)
    diff = a_updated - a_pred
    quad_term = diff.T @ I_pred @ diff

    # KL divergence: 0.5 * (log|I_pred| - log|I_updated| + trace(I_pred @ P_updated) + quad_term - k)
    k = a_pred.shape[0] # Dimension of the state vector
    kl_div = 0.5 * (log_det_Ipred - log_det_Iupdated + trace_term + quad_term - k)

    return kl_div