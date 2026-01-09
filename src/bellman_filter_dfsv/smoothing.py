import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from ._bellman_math import invert_info_matrix, predict_info_step
from .types import BIFState, DFSVParams, RBParticleState, RBPSResult


class SmootherResult(eqx.Module):
    """Result container for RTS smoother."""

    smoothed_means: Float[Array, "T 2K"]
    smoothed_covs: Float[Array, "T 2K 2K"]
    smoothed_lag1_covs: Float[Array, "T 2K 2K"]


def rts_smoother(
    params: DFSVParams,
    filter_means: Float[Array, "T 2K"],
    filter_infos: Float[Array, "T 2K 2K"],
) -> SmootherResult:
    """Runs the Rauch-Tung-Striebel (RTS) smoother adapted for information filter results."""
    T, state_dim = filter_means.shape
    K = params.lambda_r.shape[1]

    # Pre-calculate covariances from information matrices
    vmap_invert = jax.vmap(invert_info_matrix)
    filter_covs = vmap_invert(filter_infos)

    # Re-run prediction step to get predicted statistics
    # Note: We could cache this from the forward pass if memory allows,
    # but re-computing is often cheaper than storing.
    filtered_states_bif = BIFState(mean=filter_means, info=filter_infos)

    predicted_states = jax.vmap(lambda s: predict_info_step(params, s))(
        filtered_states_bif
    )

    pred_means = predicted_states.mean
    pred_infos = predicted_states.info
    pred_covs = vmap_invert(pred_infos)

    F = jnp.block(
        [[params.Phi_f, jnp.zeros((K, K))], [jnp.zeros((K, K)), params.Phi_h]]
    )

    # Initialize backward pass with the final state
    init_carry = (filter_means[-1], filter_covs[-1])

    # Inputs for the backward scan (reversed in time implicitly by scan behavior or indices)
    # We need t=0...T-2 for the backward steps.
    xs = (
        filter_means[:-1],
        filter_covs[:-1],
        pred_means[:-1],
        pred_infos[:-1],
        pred_covs[:-1],
    )

    def backward_step(carry, x):
        smooth_mean_tp1, smooth_cov_tp1 = carry
        filt_mean_t, filt_cov_t, pred_mean_tp1, pred_info_tp1, pred_cov_tp1 = x

        # RTS Gain: J_t = P_{t|t} F^T P_{t+1|t}^-1
        # In info form: J_t = P_{t|t} F^T \Omega_{t+1|t} (approx, but we have P and Info)
        # Actually J_t = P_{t|t} F^T P_{t+1|t}^-1
        # P_{t+1|t}^-1 is exactly pred_info_tp1.
        J_t = filt_cov_t @ F.T @ pred_info_tp1

        smooth_mean_t = filt_mean_t + J_t @ (smooth_mean_tp1 - pred_mean_tp1)

        cov_diff = smooth_cov_tp1 - pred_cov_tp1
        smooth_cov_t = filt_cov_t + J_t @ cov_diff @ J_t.T
        smooth_cov_t = 0.5 * (smooth_cov_t + smooth_cov_t.T)

        lag1_cov = smooth_cov_tp1 @ J_t.T

        return (smooth_mean_t, smooth_cov_t), (smooth_mean_t, smooth_cov_t, lag1_cov)

    _, (means_rev, covs_rev, lag1_rev) = jax.lax.scan(
        backward_step, init_carry, xs, reverse=True
    )

    full_means = jnp.concatenate([means_rev, init_carry[0][None, :]], axis=0)
    full_covs = jnp.concatenate([covs_rev, init_carry[1][None, :, :]], axis=0)

    # Lag-1 covariances are for t=0...T-1.
    # The last element is somewhat undefined or can be padded.
    pad_lag1 = jnp.zeros((1, state_dim, state_dim))
    full_lag1 = jnp.concatenate([lag1_rev, pad_lag1], axis=0)

    return SmootherResult(
        smoothed_means=full_means,
        smoothed_covs=full_covs,
        smoothed_lag1_covs=full_lag1,
    )


# --- RBPS Logic ---


def _predict_rbpf(
    params: DFSVParams, state: RBParticleState, key: jax.Array
) -> RBParticleState:
    """Prediction step for RBPF."""
    K = params.lambda_r.shape[1]
    P = state.h_particles.shape[1]

    # 1. Propagate h particles
    # h_{t+1} = mu + Phi_h(h_t - mu) + eta_t
    mu_col = params.mu[:, None]
    h_dev = state.h_particles - mu_col
    h_pred_mean = mu_col + params.Phi_h @ h_dev

    # Sample noise
    L_Qh = jnp.linalg.cholesky(params.Q_h + 1e-6 * jnp.eye(K))
    noise = jax.random.normal(key, shape=(K, P))
    h_next = h_pred_mean + L_Qh @ noise

    # 2. Propagate f statistics (Kalman Predict)
    # f_{t+1} = Phi_f f_t + eps_t
    # E[f_{t+1}] = Phi_f E[f_t]
    # Var(f_{t+1}) = Phi_f Var(f_t) Phi_f' + Q_f(h_{t+1})

    # Batched matrix multiplication for means
    f_pred_means = params.Phi_f @ state.f_means

    # Batched covariance update
    # Need to handle (K, K, P) batching manually or via vmap
    def predict_cov(cov_p, h_next_p):
        Q_f = jnp.diag(jnp.exp(h_next_p))
        return params.Phi_f @ cov_p @ params.Phi_f.T + Q_f

    f_pred_covs = jax.vmap(predict_cov, in_axes=(2, 1), out_axes=2)(
        state.f_covs, h_next
    )

    return RBParticleState(
        h_particles=h_next,
        f_means=f_pred_means,
        f_covs=f_pred_covs,
        log_weights=state.log_weights,
    )


def _update_rbpf(
    params: DFSVParams, state: RBParticleState, observation: Float[Array, "N"]
) -> tuple[RBParticleState, Float[Array, "P"]]:
    """Update step for RBPF."""
    # y_t = Lambda f_t + e_t
    # e_t ~ N(0, Sigma)

    # This is a Kalman Update step for each particle
    # y_t | h_{1:t}, y_{1:t-1} ~ N(Lambda f_{t|t-1}, Lambda P_{t|t-1} Lambda' + R)

    N = params.lambda_r.shape[0]
    K = params.lambda_r.shape[1]
    R = jnp.diag(params.sigma2)

    def kalman_update(f_mean, f_cov):
        # Predicted observation mean
        y_pred = params.lambda_r @ f_mean

        # Innovation
        v = observation - y_pred

        # Innovation covariance
        S = params.lambda_r @ f_cov @ params.lambda_r.T + R

        # Kalman Gain
        # K = P H' S^-1
        # Use Cholesky solve for stability
        L_S = jnp.linalg.cholesky(S + 1e-6 * jnp.eye(N))

        # Calculate likelihood of v given S
        # log N(v; 0, S)
        log_det_S = 2 * jnp.sum(jnp.log(jnp.diag(L_S)))
        v_scaled = jax.scipy.linalg.solve_triangular(L_S, v, lower=True)
        mahalanobis = jnp.sum(v_scaled**2)
        log_lik = -0.5 * (N * jnp.log(2 * jnp.pi) + log_det_S + mahalanobis)

        # State update
        # K = P H' S^-1
        S_inv_H = jax.scipy.linalg.cho_solve((L_S, True), params.lambda_r)  # (N, K)
        K_gain = f_cov @ S_inv_H.T  # (K, N)

        f_new_mean = f_mean + K_gain @ v

        # Joseph form update for covariance stability
        # P = (I - KH) P (I - KH)' + KRK'
        I_K = jnp.eye(K)
        bracket = I_K - K_gain @ params.lambda_r
        f_new_cov = bracket @ f_cov @ bracket.T + K_gain @ R @ K_gain.T

        return f_new_mean, f_new_cov, log_lik

    # Vmap over particles
    f_means_new, f_covs_new, incremental_log_liks = jax.vmap(
        kalman_update, in_axes=(1, 2), out_axes=(1, 2, 0)
    )(state.f_means, state.f_covs)

    # Update weights
    new_log_weights = state.log_weights + incremental_log_liks

    return RBParticleState(
        h_particles=state.h_particles,
        f_means=f_means_new,
        f_covs=f_covs_new,
        log_weights=new_log_weights,
    ), incremental_log_liks


def systematic_resample_indices(key, log_weights, num_particles):
    # Stabilize weights
    max_log_w = jnp.max(log_weights)
    lw_norm = (
        log_weights - max_log_w - jax.scipy.special.logsumexp(log_weights - max_log_w)
    )

    w = jnp.exp(lw_norm)
    u = jax.random.uniform(key, (), minval=0.0, maxval=1.0 / num_particles)
    points = u + jnp.arange(num_particles) / num_particles
    cum_w = jnp.cumsum(w)
    indices = jnp.searchsorted(cum_w, points)
    return jnp.clip(indices, 0, num_particles - 1)


def run_rbps(
    params: DFSVParams,
    observations: Float[Array, "T N"],
    num_particles: int = 100,
    num_trajectories: int = 20,
    seed: int = 42,
) -> RBPSResult:
    T, N = observations.shape
    K = params.lambda_r.shape[1]
    key = jax.random.PRNGKey(seed)

    # --- Forward RBPF ---
    key, init_key = jax.random.split(key)

    # Init h
    kron_prod = jnp.kron(params.Phi_h, params.Phi_h)
    vec_Qh = params.Q_h.flatten()
    vec_Ph = jnp.linalg.solve(jnp.eye(K * K) - kron_prod, vec_Qh)
    P_h = vec_Ph.reshape(K, K)
    L_h = jnp.linalg.cholesky(P_h + 1e-6 * jnp.eye(K))
    h0 = params.mu[:, None] + L_h @ jax.random.normal(init_key, (K, num_particles))

    # Init f (unconditional)
    f0_means = jnp.zeros((K, num_particles))
    f0_covs = jnp.tile(jnp.eye(K)[:, :, None], (1, 1, num_particles))

    init_state = RBParticleState(
        h0, f0_means, f0_covs, jnp.full((num_particles,), -jnp.log(num_particles))
    )

    def scan_step(carry, obs_t):
        state_prev, key_curr = carry
        key_curr, pred_key, res_key = jax.random.split(key_curr, 3)

        # Predict & Update
        state_pred = _predict_rbpf(params, state_prev, pred_key)
        state_upd, _ = _update_rbpf(params, state_pred, obs_t)

        # Save state before resampling for backward pass
        saved_state = state_upd

        # Always resample for now (simplifies code, standard PF)
        idx = systematic_resample_indices(res_key, state_upd.log_weights, num_particles)

        h_new = state_upd.h_particles[:, idx]
        f_m_new = state_upd.f_means[:, idx]
        f_c_new = state_upd.f_covs[:, :, idx]
        lw_new = jnp.full((num_particles,), -jnp.log(num_particles))

        state_resampled = RBParticleState(h_new, f_m_new, f_c_new, lw_new)
        return (state_resampled, key_curr), saved_state

    _, history = jax.lax.scan(scan_step, (init_state, key), observations)

    # --- Backward FFBS ---
    final_weights = history.log_weights[-1]
    key, idx_key = jax.random.split(key)
    final_indices = jax.random.categorical(
        idx_key, final_weights, shape=(num_trajectories,)
    )

    # Use two-step indexing to avoid JAX/NumPy advanced indexing behavior
    h_T = history.h_particles[-1][:, final_indices]  # (K, M)

    def backward_step(h_next_batch, t):
        particles_t = history.h_particles[t]  # (K, P)
        log_w_t = history.log_weights[t]  # (P)

        # Transition density log p(h_{t+1}|h_t)
        # h_{t+1} ~ N(mu + Phi(h_t - mu), Q)
        h_n = h_next_batch[:, None, :]  # (K, 1, M)
        p_t = particles_t[:, :, None]  # (K, P, 1)

        mu_col = params.mu[:, None, None]
        dev = p_t - mu_col
        mean = mu_col + jnp.tensordot(params.Phi_h, dev, axes=(1, 0))  # (K, P, 1)

        res = h_n - mean  # (K, P, M)
        q_diag = jnp.diag(params.Q_h)[:, None, None]
        log_trans = -0.5 * jnp.sum((res**2) / (q_diag + 1e-8), axis=0)  # (P, M)

        log_wb = log_w_t[:, None] + log_trans

        # Stabilize weights
        max_log_wb = jnp.max(log_wb, axis=0, keepdims=True)
        log_wb = log_wb - max_log_wb

        # Sample indices
        rngs = jax.random.split(jax.random.fold_in(key, t), num_trajectories)
        indices = jax.vmap(lambda lw, k: jax.random.categorical(k, lw), in_axes=(1, 0))(
            log_wb, rngs
        )

        h_t_sampled = particles_t[:, indices]
        return h_t_sampled, h_t_sampled

    # JAX scan with reverse=True returns stacked outputs in input index order.
    # Explicit reversal is required to restore time-ascending order (0 to T-2).
    _, h_samples_rev = jax.lax.scan(backward_step, h_T, jnp.arange(T - 1)[::-1])
    h_samples = jnp.concatenate([h_samples_rev[::-1], h_T[None, ...]], axis=0)
    h_samples = jnp.transpose(h_samples, (2, 0, 1))  # (M, T, K)

    # --- Conditional Smoother ---
    def conditional_smooth(h_traj):
        return _conditional_rts_smoother(params, observations, h_traj)

    f_means, f_covs, f_lag1_covs = jax.vmap(conditional_smooth)(h_samples)

    return RBPSResult(h_samples, f_means, f_covs, f_lag1_covs)


def _conditional_rts_smoother(params, observations, h_traj):
    T, N = observations.shape
    K = params.lambda_r.shape[1]

    # Forward Filter
    def filter_step(carry, inp):
        m, P = carry
        y, h = inp

        Q = jnp.diag(jnp.exp(h))
        m_pred = params.Phi_f @ m
        P_pred = params.Phi_f @ P @ params.Phi_f.T + Q

        v = y - params.lambda_r @ m_pred
        S = params.lambda_r @ P_pred @ params.lambda_r.T + jnp.diag(params.sigma2)

        L_S = jnp.linalg.cholesky(S + 1e-6 * jnp.eye(N))
        K_gain = (
            P_pred
            @ params.lambda_r.T
            @ jax.scipy.linalg.cho_solve((L_S, True), jnp.eye(N))
        )

        m_upd = m_pred + K_gain @ v
        I_K = jnp.eye(K)
        P_upd = (I_K - K_gain @ params.lambda_r) @ P_pred
        P_upd = 0.5 * (P_upd + P_upd.T)

        return (m_upd, P_upd), (m_pred, P_pred, m_upd, P_upd)

    m0 = jnp.zeros(K)
    P0 = jnp.eye(K)
    _, (m_preds, P_preds, m_filts, P_filts) = jax.lax.scan(
        filter_step, (m0, P0), (observations, h_traj)
    )

    # Backward Smoother
    def smooth_step(carry, inp):
        m_next, P_next = carry
        m_p, P_p, m_f, P_f = inp

        # J = P_f Phi' P_p^-1
        L_P = jnp.linalg.cholesky(P_p + 1e-6 * jnp.eye(K))
        # solve P_p J' = Phi P_f
        J_T = jax.scipy.linalg.cho_solve((L_P, True), params.Phi_f @ P_f)
        J = J_T.T

        m_smooth = m_f + J @ (m_next - m_p)
        P_smooth = P_f + J @ (P_next - P_p) @ J.T
        cov_lag1 = P_next @ J.T  # P_{t+1, t | T}

        # Symmetrize
        P_smooth = 0.5 * (P_smooth + P_smooth.T)

        return (m_smooth, P_smooth), (m_smooth, P_smooth, cov_lag1)

    last_m = m_filts[-1]
    last_P = P_filts[-1]

    xs = (m_preds[1:], P_preds[1:], m_filts[:-1], P_filts[:-1])
    # JAX scan(reverse=True) returns outputs in ascending input index order.
    # No reversal is needed as the inputs are already aligned.
    _, (ms, Ps, Ps_lag1) = jax.lax.scan(smooth_step, (last_m, last_P), xs, reverse=True)

    f_smooth_means = jnp.concatenate([ms, last_m[None, :]], axis=0)
    f_smooth_covs = jnp.concatenate([Ps, last_P[None, :, :]], axis=0)
    f_smooth_lag1_covs = Ps_lag1

    return f_smooth_means, f_smooth_covs, f_smooth_lag1_covs
