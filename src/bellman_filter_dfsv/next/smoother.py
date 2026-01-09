import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float
import equinox as eqx

from .types import DFSVParams, BIFState
from .kernels import predict_info_step, invert_info_matrix


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

    def predict_scan(state_prev_tuple, _):
        mean, info = state_prev_tuple
        state_prev = BIFState(mean=mean, info=info)

        state_pred = predict_info_step(params, state_prev)

        return (state_pred.mean, state_pred.info), (state_pred.mean, state_pred.info)

    vmap_invert = jax.vmap(invert_info_matrix)
    filter_covs = vmap_invert(filter_infos)

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

    init_carry = (filter_means[-1], filter_covs[-1])

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

    pad_lag1 = jnp.zeros((1, state_dim, state_dim))
    full_lag1 = jnp.concatenate([lag1_rev, pad_lag1], axis=0)

    return SmootherResult(
        smoothed_means=full_means, smoothed_covs=full_covs, smoothed_lag1_covs=full_lag1
    )
