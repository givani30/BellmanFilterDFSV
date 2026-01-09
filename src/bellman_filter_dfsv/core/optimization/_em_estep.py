"""E-step computations for EM algorithm on DFSV models."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._em_suffstats import EMSufficientStats


def compute_exp_neg_h(
    h_mean: Float[Array, "K"],
    h_var: Float[Array, "K"],
    max_var: float = 4.0,
) -> Float[Array, "K"]:
    """
    Compute E[exp(-h)] for h ~ N(h_mean, h_var).

    Uses log-normal moment formula: E[exp(a*X)] = exp(a*μ + a²σ²/2) for X ~ N(μ,σ²).
    With a = -1: E[exp(-h)] = exp(-μ + σ²/2).

    Args:
        h_mean: Posterior mean of h, shape (K,)
        h_var: Posterior variance of h (diagonal), shape (K,)
        max_var: Cap on variance to prevent numerical overflow

    Returns:
        E[exp(-h_k)] for each factor k, shape (K,)
    """
    h_var_capped = jnp.minimum(h_var, max_var)
    return jnp.exp(-h_mean + 0.5 * h_var_capped)


def accumulate_sufficient_stats(
    observations: Float[Array, "T N"],
    smoothed_states: Float[Array, "T state_dim"],
    smoothed_covs: Float[Array, "T state_dim state_dim"],
    smoothed_lag1_covs: Float[Array, "T state_dim state_dim"],
    K: int,
) -> EMSufficientStats:
    """
    Accumulate sufficient statistics from smoother output for EM M-step.

    Takes the output of BIF filter + RTS smoother and computes all statistics
    needed for closed-form parameter updates.

    The joint state is x_t = [f_t', h_t']' with dimension 2K.
    - f_t = x_t[:K] (factors)
    - h_t = x_t[K:] (log-volatilities)

    Args:
        observations: Observed returns r_t, shape (T, N)
        smoothed_states: E[x_t | r_{1:T}], shape (T, 2K)
        smoothed_covs: Cov[x_t | r_{1:T}], shape (T, 2K, 2K)
        smoothed_lag1_covs: Cov[x_{t+1}, x_t | r_{1:T}], shape (T, 2K, 2K)
            Index t holds P_{t+1,t|T}. Last entry (T-1) is unused.
        K: Number of latent factors

    Returns:
        EMSufficientStats with all accumulated statistics
    """
    T, N = observations.shape

    f_mean = smoothed_states[:, :K]
    h_mean = smoothed_states[:, K:]

    P_ff = smoothed_covs[:, :K, :K]
    P_hh = smoothed_covs[:, K:, K:]

    P_ff_lag1 = smoothed_lag1_covs[:, :K, :K]
    P_hh_lag1 = smoothed_lag1_covs[:, K:, K:]

    def compute_obs_stats(carry, t):
        r_t = observations[t]
        f_t = f_mean[t]
        P_ff_t = P_ff[t]

        E_f_f = jnp.outer(f_t, f_t) + P_ff_t

        r_f = jnp.outer(r_t, f_t)
        r_r_diag = r_t * r_t

        new_carry = (
            carry[0] + r_f,
            carry[1] + E_f_f,
            carry[2] + r_r_diag,
        )
        return new_carry, None

    init_obs = (jnp.zeros((N, K)), jnp.zeros((K, K)), jnp.zeros(N))
    (sum_r_f, sum_f_f, sum_r_r_diag), _ = jax.lax.scan(
        compute_obs_stats, init_obs, jnp.arange(T)
    )

    def compute_dynamics_stats(carry, t):
        f_t = f_mean[t]
        f_prev = f_mean[t - 1]
        h_t = h_mean[t]
        h_prev = h_mean[t - 1]

        P_ff_t = P_ff[t]
        P_ff_prev = P_ff[t - 1]
        P_hh_t = P_hh[t]
        P_hh_prev = P_hh[t - 1]

        P_ff_cross = P_ff_lag1[t - 1]
        P_hh_cross = P_hh_lag1[t - 1]

        E_f_f = jnp.outer(f_t, f_t) + P_ff_t
        E_f_fprev = jnp.outer(f_t, f_prev) + P_ff_cross
        E_fprev_fprev = jnp.outer(f_prev, f_prev) + P_ff_prev

        h_var_t = jnp.diag(P_hh_t)
        exp_neg_h_t = compute_exp_neg_h(h_t, h_var_t)

        E_f_fprev_diag = jnp.diag(E_f_fprev)
        E_fprev_sq = jnp.diag(E_fprev_fprev)

        E_h_h = jnp.outer(h_t, h_t) + P_hh_t
        E_h_hprev = jnp.outer(h_t, h_prev) + P_hh_cross
        E_hprev_hprev = jnp.outer(h_prev, h_prev) + P_hh_prev

        new_carry = (
            carry[0] + E_f_fprev,
            carry[1] + E_fprev_fprev,
            carry[2] + exp_neg_h_t,
            carry[3] + exp_neg_h_t * E_f_fprev_diag,
            carry[4] + exp_neg_h_t * E_fprev_sq,
            carry[5] + h_t,
            carry[6] + h_prev,
            carry[7] + E_h_h,
            carry[8] + E_h_hprev,
            carry[9] + E_hprev_hprev,
        )
        return new_carry, None

    init_dyn = (
        jnp.zeros((K, K)),
        jnp.zeros((K, K)),
        jnp.zeros(K),
        jnp.zeros(K),
        jnp.zeros(K),
        jnp.zeros(K),
        jnp.zeros(K),
        jnp.zeros((K, K)),
        jnp.zeros((K, K)),
        jnp.zeros((K, K)),
    )

    (
        (
            sum_f_fprev,
            sum_fprev_fprev,
            sum_exp_neg_h,
            sum_exp_neg_h_f_fprev_diag,
            sum_exp_neg_h_fprev_sq,
            sum_h,
            sum_hprev,
            sum_h_h,
            sum_h_hprev,
            sum_hprev_hprev,
        ),
        _,
    ) = jax.lax.scan(compute_dynamics_stats, init_dyn, jnp.arange(1, T))

    return EMSufficientStats(
        sum_r_f=sum_r_f,
        sum_f_f=sum_f_f,
        sum_r_r_diag=sum_r_r_diag,
        sum_f_fprev=sum_f_fprev,
        sum_fprev_fprev=sum_fprev_fprev,
        sum_exp_neg_h=sum_exp_neg_h,
        sum_exp_neg_h_f_fprev_diag=sum_exp_neg_h_f_fprev_diag,
        sum_exp_neg_h_fprev_sq=sum_exp_neg_h_fprev_sq,
        sum_h=sum_h,
        sum_hprev=sum_hprev,
        sum_h_h=sum_h_h,
        sum_h_hprev=sum_h_hprev,
        sum_hprev_hprev=sum_hprev_hprev,
        T=T,
    )
