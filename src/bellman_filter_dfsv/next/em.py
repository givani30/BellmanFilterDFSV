import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import Tuple

from .types import DFSVParams, EMSufficientStats
from .rbps import run_rbps, RBPSResult


def rbps_to_suffstats(
    rbps_result: RBPSResult, observations: Float[Array, "T N"], K: int
) -> EMSufficientStats:
    """
    Convert RBPS samples to EM sufficient statistics by averaging over trajectories.
    """
    M, T, K_dim = rbps_result.h_samples.shape
    assert K == K_dim

    E_f = jnp.mean(rbps_result.f_smooth_means, axis=0)

    m_mT = (
        rbps_result.f_smooth_means[..., None] * rbps_result.f_smooth_means[..., None, :]
    )
    E_ff_T = jnp.mean(rbps_result.f_smooth_covs + m_mT, axis=0)

    m_t = rbps_result.f_smooth_means[:, 1:, :, None]
    m_tm1 = rbps_result.f_smooth_means[:, :-1, None, :]
    m_lag_mT = m_t * m_tm1

    E_f_lag_f_T = jnp.mean(rbps_result.f_smooth_lag1_covs + m_lag_mT, axis=0)

    E_h = jnp.mean(rbps_result.h_samples, axis=0)

    h_hT = rbps_result.h_samples[..., None] * rbps_result.h_samples[..., None, :]
    E_hh_T = jnp.mean(h_hT, axis=0)

    h_t = rbps_result.h_samples[:, 1:, :, None]
    h_tm1 = rbps_result.h_samples[:, :-1, None, :]
    E_h_lag_h_T = jnp.mean(h_t * h_tm1, axis=0)

    h_t2T = rbps_result.h_samples[:, 1:, :]
    exp_neg_h = jnp.exp(-h_t2T)
    E_exp_neg_h = jnp.mean(exp_neg_h, axis=0)

    f_lag_f_T_m = rbps_result.f_smooth_lag1_covs + m_lag_mT
    f_lag_f_diag = jnp.diagonal(f_lag_f_T_m, axis1=2, axis2=3)

    weighted_cross = exp_neg_h * f_lag_f_diag
    E_exp_neg_h_f_fprev_diag = jnp.mean(weighted_cross, axis=0)

    ff_T_m = rbps_result.f_smooth_covs + m_mT
    ff_T_prev_m = ff_T_m[:, :-1, :, :]
    ff_prev_diag = jnp.diagonal(ff_T_prev_m, axis1=2, axis2=3)

    weighted_prev_sq = exp_neg_h * ff_prev_diag
    E_exp_neg_h_fprev_sq = jnp.mean(weighted_prev_sq, axis=0)

    cross_y_f = observations.T @ E_f
    obs_sq_sum = jnp.sum(observations**2, axis=0)

    return EMSufficientStats(
        sum_r_f=cross_y_f,
        sum_f_f=jnp.sum(E_ff_T, axis=0),
        sum_r_r_diag=obs_sq_sum,
        sum_f_fprev=jnp.sum(E_f_lag_f_T, axis=0),
        sum_fprev_fprev=jnp.sum(E_ff_T[:-1], axis=0),
        sum_exp_neg_h=jnp.sum(E_exp_neg_h, axis=0),
        sum_exp_neg_h_f_fprev_diag=jnp.sum(E_exp_neg_h_f_fprev_diag, axis=0),
        sum_exp_neg_h_fprev_sq=jnp.sum(E_exp_neg_h_fprev_sq, axis=0),
        sum_h=jnp.sum(E_h[1:], axis=0),
        sum_hprev=jnp.sum(E_h[:-1], axis=0),
        sum_h_h=jnp.sum(E_hh_T[1:], axis=0),
        sum_h_hprev=jnp.sum(E_h_lag_h_T, axis=0),
        sum_hprev_hprev=jnp.sum(E_hh_T[:-1], axis=0),
        T=T,
    )


JITTER = 1e-6
MIN_VARIANCE = 1e-6


def update_lambda_r(stats: EMSufficientStats) -> Float[Array, "N K"]:
    sum_f_f_reg = stats.sum_f_f + jnp.eye(stats.sum_f_f.shape[0]) * JITTER
    return stats.sum_r_f @ jnp.linalg.inv(sum_f_f_reg)


def update_sigma2(
    stats: EMSufficientStats, lambda_r: Float[Array, "N K"]
) -> Float[Array, "N"]:
    T = stats.T
    quad_term = jnp.sum(lambda_r * (lambda_r @ stats.sum_f_f), axis=1)
    cross_term = jnp.sum(lambda_r * stats.sum_r_f, axis=1)
    sigma2 = (stats.sum_r_r_diag - 2 * cross_term + quad_term) / T
    return jnp.maximum(sigma2, MIN_VARIANCE)


def update_Phi_h(
    stats: EMSufficientStats, mu: Float[Array, "K"]
) -> Float[Array, "K K"]:
    T_minus_1 = stats.T - 1
    K = mu.shape[0]

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
    phi_f_diag = stats.sum_exp_neg_h_f_fprev_diag / jnp.maximum(
        stats.sum_exp_neg_h_fprev_sq, JITTER
    )
    phi_f_diag = jnp.clip(phi_f_diag, -0.999, 0.999)
    return jnp.diag(phi_f_diag)


def m_step(
    stats: EMSufficientStats, n_mu_phi_iters: int = 3
) -> Tuple[Float[Array, "N K"], ...]:
    lambda_r = update_lambda_r(stats)
    sigma2 = update_sigma2(stats, lambda_r)
    Phi_f = update_Phi_f(stats)

    K = stats.sum_h.shape[0]
    mu = stats.sum_h / (stats.T - 1)
    Phi_h = jnp.eye(K) * 0.9

    for _ in range(n_mu_phi_iters):
        Phi_h = update_Phi_h(stats, mu)
        mu = update_mu(stats, Phi_h)

    Q_h = update_Q_h(stats, mu, Phi_h)

    return lambda_r, sigma2, Phi_f, mu, Phi_h, Q_h


def fit_em(
    observations: Float[Array, "T N"],
    init_params: DFSVParams,
    num_particles: int = 200,
    num_trajectories: int = 20,
    max_iters: int = 50,
    verbose: bool = True,
) -> Tuple[DFSVParams, list[DFSVParams]]:
    """
    Fit DFSV model using Expectation-Maximization with RBPS.
    """
    current_params = init_params
    history = []
    K = init_params.lambda_r.shape[1]

    jit_rbps = jax.jit(
        lambda p: run_rbps(p, observations, num_particles, num_trajectories)
    )

    if verbose:
        print(
            f"Starting EM (RBPS) | Particles: {num_particles} | Trajs: {num_trajectories}"
        )

    for i in range(max_iters):
        rbps_res = jit_rbps(current_params)
        stats = rbps_to_suffstats(rbps_res, observations, K)

        lambda_r, sigma2, Phi_f, mu, Phi_h, Q_h = m_step(stats)

        current_params = DFSVParams(
            lambda_r=lambda_r,
            Phi_f=Phi_f,
            Phi_h=Phi_h,
            mu=mu,
            sigma2=sigma2,
            Q_h=Q_h,
        )

        history.append(current_params)

        if verbose:
            print(
                f"Iter {i:3d} | "
                f"Lambda[0,0]: {lambda_r[0, 0]:.4f} | "
                f"Sigma2[0]: {sigma2[0]:.4f}"
            )

    return current_params, history
