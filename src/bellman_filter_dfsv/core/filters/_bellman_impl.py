# src/bellman_filter_dfsv/core/filters/_bellman_impl.py
"""Core mathematical implementations for the Bellman Information Filter (BIF).

This module provides the fundamental mathematical operations required by the BIF,
implemented as standalone, JAX-based functions for efficiency. These implementations
follow the methodology of Lange (2024) adapted for the DFSV model context of
Boekestijn (2025).

Key Components:
    - Observed Fisher Information calculation (negative Hessian of log-likelihood)
    - Log posterior evaluation using efficient matrix operations
    - BIF pseudo-likelihood penalty term (KL divergence approximation)

Mathematical Framework:
    The implementations handle the DFSV model structure:
    y_t = Λf_t + ε_t                            (Observation)
    f_{t+1} = Φ_f f_t + ν_{t+1}               (Factor Evolution)
    h_{t+1} = μ + Φ_h(h_t - μ) + η_{t+1}      (Log-Volatility Evolution)

    where:
    - f_t: K-dimensional factor vector
    - h_t: K-dimensional log-volatility vector
    - α_t = [f_t', h_t']': Complete state vector
    - ν_{t+1} ~ N(0, diag(exp(h_{t+1})))
    - η_{t+1} ~ N(0, Q_h)
    - ε_t ~ N(0, Σ_ε), Σ_ε = diag(σ²)

Implementation Notes:
    - Functions are designed to be JIT-compiled by the filter classes
    - Matrix operations use Woodbury identity and MDL for O(NK²) complexity
    - Numerical stability measures (jitter, regularization) are included
    - All functions expect and return JAX arrays

References:
    - Lange (2024): Bellman Filter methodology
    - Boekestijn (2025): DFSV model specification and implementation
"""
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import equinox as eqx

# Type hint for build_covariance function signature
BuildCovarianceFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]


# Note: No JIT decorators here; JITting happens in the main class setup

def build_covariance_impl(
    lambda_r: jnp.ndarray, exp_h: jnp.ndarray, sigma2: jnp.ndarray
) -> jnp.ndarray:
    """Builds the observation covariance matrix A_t for the DFSV model.

    Constructs A_t = ΛΣ_f(h_t)Λ' + Σ_ε, which represents the conditional covariance
    matrix of the observation y_t given the current state. This matrix is central to
    the likelihood calculations in both observation and state estimation.

    Mathematical Details:
        1. Model Structure:
           y_t = Λf_t + ε_t
           where ε_t ~ N(0, Σ_ε)

        2. Components:
           - Σ_f(h_t) = diag(exp(h_t)): State-dependent factor covariance
           - Σ_ε = diag(σ²): Idiosyncratic error covariance

        3. Implementation includes jitter (1e-6) for numerical stability
           and enforces matrix symmetry.

    Args:
        lambda_r: Factor loading matrix Λ (N, K).
        exp_h: Exponentiated log-volatilities diag(Σ_f) = exp(h_t) (K,).
        sigma2: Idiosyncratic variances diag(Σ_ε) (N,).

    Returns:
        The observation covariance matrix A_t (N, N).

    References:
        Boekestijn (2025), Eq. 3.1: Observation equation structure.
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


def observed_fim_impl(
    lambda_r: jnp.ndarray,
    sigma2: jnp.ndarray,
    alpha: jnp.ndarray,
    observation: jnp.ndarray,
    K: int, # Pass K explicitly
) -> jnp.ndarray:
    """Calculates the Observed Fisher Information Matrix (Negative Hessian).

    Computes J = -∇²ℓ(y_t|α_t) where ℓ(y_t|α_t) is the log-likelihood of observation y_t
    given state α_t = [f_t', h_t']'. Implements an efficient calculation using the
    Woodbury matrix identity to handle the observation covariance A_t = ΛΣ_f(h_t)Λ' + Σ_ε.

    Mathematical Derivation:
        1. Log-likelihood: ℓ(y_t|α_t) = -1/2[log|A_t| + r_t'A_t^{-1}r_t]
           where r_t = y_t - Λf_t, A_t = ΛΣ_f(h_t)Λ' + Σ_ε
           
        2. Block Structure:
           J = [J_ff  J_fh]
              [J_fh' J_hh]
           where:
           - J_ff = Λ'A_t^{-1}Λ
           - J_fh[l,k] = exp(h_k)·I_ff[l,k]·P[k]
           - J_hh[k,l] = 1/2·exp(h_k + h_l)·[tr(I_ff[k,l]) - P[k]P[l]]
           with P = Λ'A_t^{-1}r_t

        3. Implementation uses Woodbury identity for A_t^{-1} applications:
           A_t^{-1} = D^{-1} - D^{-1}Λ(Σ_f^{-1} + Λ'D^{-1}Λ)^{-1}Λ'D^{-1}
           where D = Σ_ε (diagonal) for efficiency

    Args:
        lambda_r: Factor loading matrix Λ (N, K).
        sigma2: Idiosyncratic variances diag(Σ_ε) (N,).
        alpha: State vector α_t = [f_t', h_t']' (state_dim,).
        observation: Observation vector y_t (N,).
        K: Number of factors.

    Returns:
        The Observed Fisher Information matrix J (state_dim, state_dim), negative
        Hessian of the log-likelihood. Used in the BIF update step:
        Ω_{t|t} = Ω_{t|t-1} + J.
        
    References:
        Lange (2024), Eq. 11: Information matrix update.
        Boekestijn (2025), Sec 3.2: DFSV model specification.
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

def expected_fim_impl(
    lambda_r: jnp.ndarray,
    sigma2: jnp.ndarray,
    alpha: jnp.ndarray,
    observation: jnp.ndarray,
    K: int, # Pass K explicitly
) -> jnp.ndarray:
    """Calculates the Expected Fisher Information Matrix (Expectation of the Negative Hessian).

    Computes J = E[-∇²ℓ(y_t|α_t)] where ℓ(y_t|α_t) is the log-likelihood of observation y_t
    given state α_t = [f_t', h_t']'. Implements an efficient calculation using the
    Woodbury matrix identity to handle the observation covariance A_t = ΛΣ_f(h_t)Λ' + Σ_ε.

    Mathematical Derivation:
        1. Log-likelihood: ℓ(y_t|α_t) = -1/2[log|A_t| + r_t'A_t^{-1}r_t]
           where r_t = y_t - Λf_t, A_t = ΛΣ_f(h_t)Λ' + Σ_ε

        2. Fisher Information Matrix:
           J = E[-∇²ℓ(y_t|α_t)] = E[-d^2ℓ/dαdα']

    Args:
        lambda_r: Factor loading matrix Λ (N, K).
        sigma2: Idiosyncratic variances diag(Σ_ε) (N,).
        alpha: State vector α_t = [f_t', h_t']' (state_dim,).
        observation: Observation vector y_t (N,). (not used in EFIM)
        K: Number of factors.
        build_covariance_fn: Function to build the observation covariance matrix
            Sigma_t = Lambda Sigma_f Lambda^T + Sigma_e. (Note: This dependency
            might be removable if calculation is done via Woodbury as below).

    Returns:
        The Expected Fisher Information matrix J (state_dim, state_dim), which is
        the expectation of the negative Hessian of the log-likelihood.
    """
    N = lambda_r.shape[0]
    state_dim = 2 * K

    # (Checks for alpha, h, exp_h remain the same)
    # alpha = eqx.error_if(alpha, jnp.any(jnp.isnan(alpha) | jnp.isinf(alpha)), "NaN/Inf input alpha")
    alpha = alpha.flatten()
    h = alpha[K:]
    # h = eqx.error_if(h, jnp.any(jnp.isnan(h) | jnp.isinf(h)), "NaN/Inf h")
    exp_h = jnp.exp(h)
    # exp_h = eqx.error_if(exp_h, jnp.any(jnp.isnan(exp_h) | jnp.isinf(exp_h)), "NaN/Inf exp_h")

    # --- Calculate components needed for I_ff ---
    sigma2_1d = sigma2.flatten() if sigma2.ndim > 1 else sigma2
    jitter_inv = 1e-8
    Dinv_diag = 1.0 / (sigma2_1d + jitter_inv)
    Cinv_diag = 1.0 / (exp_h + jitter_inv)
    # Dinv_diag = eqx.error_if(Dinv_diag, jnp.any(jnp.isnan(Dinv_diag) | jnp.isinf(Dinv_diag)), "NaN/Inf Dinv_diag")
    # Cinv_diag = eqx.error_if(Cinv_diag, jnp.any(jnp.isnan(Cinv_diag) | jnp.isinf(Cinv_diag)), "NaN/Inf Cinv_diag")

    Dinv_lambda_r = lambda_r * Dinv_diag[:, None]
    M = jnp.diag(Cinv_diag) + lambda_r.T @ Dinv_lambda_r
    # M = eqx.error_if(M, jnp.any(jnp.isnan(M) | jnp.isinf(M)), "NaN/Inf M")

    internal_jitter_M = 1e-6
    M_jittered = M + internal_jitter_M * jnp.eye(K)

    # --- JAX-compatible Cholesky with pinv fallback ---
    L_M = jax.scipy.linalg.cholesky(M_jittered, lower=True)
    cholesky_failed = jnp.any(jnp.isnan(L_M)) # Check if Cholesky produced NaNs

    V = M - jnp.diag(Cinv_diag) # V = Lambda^T D^-1 Lambda

    # Define branches for jax.lax.cond - they now unpack the combined operand
    def solve_with_pinv(operand_tuple):
        M_j, L_M_ignore, V_op = operand_tuple # Unpack all operands
        M_pinv = jnp.linalg.pinv(M_j)
        Z_op = M_pinv @ V_op
        return Z_op

    def solve_with_cho(operand_tuple):
        M_j_ignore, L_M_op, V_op = operand_tuple # Unpack all operands
        Z_op = jax.scipy.linalg.cho_solve((L_M_op, True), V_op)
        return Z_op

    # Pass ALL potential operands needed by either branch
    operands = (M_jittered, L_M, V)

    # Conditionally compute Z = M_inv @ V
    Z = jax.lax.cond(
        cholesky_failed,
        solve_with_pinv,    # function for True (failed) case
        solve_with_cho,     # function for False (succeeded) case
        operands            # The combined tuple of operands
    )
    # --- End JAX-compatible fallback ---

    I_ff = V - V @ Z # I_ff = Lambda^T A_t^{-1} Lambda
    # Check I_ff
    # I_ff = eqx.error_if(I_ff, jnp.any(jnp.isnan(I_ff) | jnp.isinf(I_ff)), "NaN/Inf in I_ff")

    # --- Calculate I_hh ---
    exp_h_outer_sum = jnp.exp(h[:, None] + h[None, :])
    I_ff_squared = I_ff**2
    I_hh = 0.5 * exp_h_outer_sum * I_ff_squared
    # Check I_hh
    # I_hh = eqx.error_if(I_hh, jnp.any(jnp.isnan(I_hh) | jnp.isinf(I_hh)), "NaN/Inf in I_hh")

    # --- Assemble the block-diagonal E-FIM matrix ---
    EFIM = jnp.zeros((state_dim, state_dim), dtype=I_ff.dtype)
    EFIM = EFIM.at[:K, :K].set(I_ff)
    EFIM = EFIM.at[K:, K:].set(I_hh)

    EFIM = (EFIM + EFIM.T) / 2

    return EFIM

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