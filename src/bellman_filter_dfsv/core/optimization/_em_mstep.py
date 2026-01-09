"""M-step closed-form updates for EM algorithm on DFSV models."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from ._em_suffstats import EMSufficientStats

JITTER = 1e-6
MIN_VARIANCE = 1e-8


def update_lambda_r(stats: EMSufficientStats) -> Float[Array, "N K"]:
    """
    M-step for factor loadings: λ_r = sum_r_f @ inv(sum_f_f).

    Derivation: OLS solution from r_t = λ_r f_t + e_t.
    """
    sum_f_f_reg = stats.sum_f_f + jnp.eye(stats.sum_f_f.shape[0]) * JITTER
    return stats.sum_r_f @ jnp.linalg.inv(sum_f_f_reg)


def update_sigma2(
    stats: EMSufficientStats, lambda_r: Float[Array, "N K"]
) -> Float[Array, "N"]:
    """
    M-step for idiosyncratic variances: σ²_n = (1/T) Σ E[(r_n - λ_n f)²].

    Expands to: (1/T) [sum_r_r - 2 λ @ sum_r_f.T + λ @ sum_f_f @ λ.T]
    """
    T = stats.T
    quad_term = jnp.sum(lambda_r * (lambda_r @ stats.sum_f_f), axis=1)
    cross_term = jnp.sum(lambda_r * stats.sum_r_f, axis=1)
    sigma2 = (stats.sum_r_r_diag - 2 * cross_term + quad_term) / T
    return jnp.maximum(sigma2, MIN_VARIANCE)


def update_Phi_h(
    stats: EMSufficientStats, mu: Float[Array, "K"]
) -> Float[Array, "K K"]:
    """
    M-step for log-vol AR (diagonal): φ_h,k = sum_hh_cross_k / sum_hprev_sq_k.

    For centered variables a_t = h_t - μ, b_t = h_{t-1} - μ:
    φ_h = Σ E[a_t b_t] / Σ E[b_t²]
    """
    T_minus_1 = stats.T - 1
    K = mu.shape[0]

    sum_h_centered = stats.sum_h - T_minus_1 * mu
    sum_hprev_centered = stats.sum_hprev - T_minus_1 * mu

    sum_a_b = (
        jnp.diag(stats.sum_h_hprev)
        - mu * stats.sum_hprev
        - mu * stats.sum_h
        + T_minus_1 * mu**2
    )
    sum_b_sq = (
        jnp.diag(stats.sum_hprev_hprev) - 2 * mu * stats.sum_hprev + T_minus_1 * mu**2
    )

    phi_h_diag = sum_a_b / jnp.maximum(sum_b_sq, JITTER)
    phi_h_diag = jnp.clip(phi_h_diag, -0.999, 0.999)

    return jnp.diag(phi_h_diag)


def update_mu(
    stats: EMSufficientStats, Phi_h: Float[Array, "K K"]
) -> Float[Array, "K"]:
    """
    M-step for long-run mean: μ = (sum_h - φ_h sum_hprev) / ((T-1)(1 - φ_h)).

    For diagonal Φ_h, computed element-wise.
    """
    T_minus_1 = stats.T - 1
    phi_h_diag = jnp.diag(Phi_h)
    numerator = stats.sum_h - phi_h_diag * stats.sum_hprev
    denominator = T_minus_1 * (1.0 - phi_h_diag)
    return (
        numerator
        / jnp.maximum(jnp.abs(denominator), JITTER)
        * jnp.sign(denominator + 1e-10)
    )


def update_Q_h(
    stats: EMSufficientStats, mu: Float[Array, "K"], Phi_h: Float[Array, "K K"]
) -> Float[Array, "K K"]:
    """
    M-step for log-vol innovation variance (diagonal): q_h = (1/(T-1)) Σ E[(h_t - μ - φ_h(h_{t-1} - μ))²].

    Expands using centered variables.
    """
    T_minus_1 = stats.T - 1
    phi_h_diag = jnp.diag(Phi_h)

    sum_a_sq = jnp.diag(stats.sum_h_h) - 2 * mu * stats.sum_h + T_minus_1 * mu**2
    sum_b_sq = (
        jnp.diag(stats.sum_hprev_hprev) - 2 * mu * stats.sum_hprev + T_minus_1 * mu**2
    )
    sum_a_b = (
        jnp.diag(stats.sum_h_hprev)
        - mu * (stats.sum_h + stats.sum_hprev)
        + T_minus_1 * mu**2
    )

    S_eta = sum_a_sq - 2 * phi_h_diag * sum_a_b + phi_h_diag**2 * sum_b_sq
    q_h_diag = S_eta / T_minus_1
    q_h_diag = jnp.maximum(q_h_diag, MIN_VARIANCE)

    return jnp.diag(q_h_diag)


def update_Phi_f(stats: EMSufficientStats) -> Float[Array, "K K"]:
    """
    M-step for factor AR (diagonal, weighted LS): φ_f,k = Σ w_t E[f_kt f_{k,t-1}] / Σ w_t E[f_{k,t-1}²].

    Uses independence approximation: E[exp(-h) f f'] ≈ E[exp(-h)] E[ff'].
    Weights w_t = E[exp(-h_t)].
    """
    phi_f_diag = stats.sum_exp_neg_h_f_fprev_diag / jnp.maximum(
        stats.sum_exp_neg_h_fprev_sq, JITTER
    )
    phi_f_diag = jnp.clip(phi_f_diag, -0.999, 0.999)
    return jnp.diag(phi_f_diag)
