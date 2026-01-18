import jax
import jax.numpy as jnp
import jax.scipy.linalg
import optimistix as optx
from jax import Array
from jaxtyping import Float

from .types import BIFState, DFSVParams

OBS_COV_JITTER = 1e-6
WOODBURY_DIAG_JITTER = 1e-8
K_CHOL_JITTER = 1e-10
LOG_DIAG_FLOOR = 1e-10


def _symmetrize(a: Float[Array, "D D"]) -> Float[Array, "D D"]:
    return 0.5 * (a + a.T)


def _woodbury_setup(
    lambda_r: Float[Array, "N K"],
    sigma2: Float[Array, "N"],
    factor_cov: Float[Array, "K K"],
) -> tuple[
    Float[Array, "K K"], Float[Array, "K K"], Float[Array, "K"], Float[Array, "K K"]
]:
    K = lambda_r.shape[1]

    inv_sigma2 = 1.0 / (sigma2 + WOODBURY_DIAG_JITTER)
    lambda_scaled = lambda_r * inv_sigma2[:, None]
    B = lambda_r.T @ lambda_scaled

    chol_C = jax.scipy.linalg.cholesky(
        _symmetrize(factor_cov) + K_CHOL_JITTER * jnp.eye(K), lower=True
    )
    C_inv = jax.scipy.linalg.cho_solve((chol_C, True), jnp.eye(K))

    M = _symmetrize(C_inv + B)
    chol_M = jax.scipy.linalg.cholesky(M + K_CHOL_JITTER * jnp.eye(K), lower=True)

    return chol_C, chol_M, inv_sigma2, B


def build_covariance(
    lambda_r: Float[Array, "N K"], exp_h: Float[Array, "K"], sigma2: Float[Array, "N"]
) -> Float[Array, "N N"]:
    """Builds the observation covariance matrix A_t = ΛΣ_f(h_t)Λ' + Σ_ε."""
    N, K = lambda_r.shape

    Sigma_e = jnp.diag(sigma2)
    Sigma_f = jnp.diag(exp_h)

    A = lambda_r @ Sigma_f @ lambda_r.T + Sigma_e + OBS_COV_JITTER * jnp.eye(N)
    return 0.5 * (A + A.T)


def observed_fim(
    lambda_r: Float[Array, "N K"],
    sigma2: Float[Array, "N"],
    alpha: Float[Array, "2K"],
    observation: Float[Array, "N"],
) -> Float[Array, "2K 2K"]:
    """Observed Fisher Information (negative Hessian of log-likelihood)."""
    K = lambda_r.shape[1]

    f = alpha[:K]
    h = alpha[K:]
    exp_h = jnp.exp(h)

    r = observation - lambda_r @ f

    Dinv_diag = 1.0 / (sigma2 + WOODBURY_DIAG_JITTER)
    Cinv_diag = 1.0 / (exp_h + WOODBURY_DIAG_JITTER)

    Dinv_lambda_r = lambda_r * Dinv_diag[:, None]
    Dinv_r = r * Dinv_diag

    M = jnp.diag(Cinv_diag) + lambda_r.T @ Dinv_lambda_r

    M_jittered = M + K_CHOL_JITTER * jnp.eye(K)
    L_M = jax.scipy.linalg.cholesky(M_jittered, lower=True)

    V = M - jnp.diag(Cinv_diag)
    Z = jax.scipy.linalg.cho_solve((L_M, True), V)
    I_ff = V - V @ Z

    v = lambda_r.T @ Dinv_r
    z_p = jax.scipy.linalg.cho_solve((L_M, True), v)
    Ainv_r = Dinv_r - Dinv_lambda_r @ z_p
    P = lambda_r.T @ Ainv_r

    J_ff = I_ff
    J_fh = I_ff * P[None, :] * exp_h[None, :]

    exp_h_outer = jnp.outer(exp_h, exp_h)
    P_outer = jnp.outer(P, P)
    term1_diag = 0.5 * exp_h * (jnp.diag(I_ff) - P**2)
    term2 = -0.5 * exp_h_outer * I_ff * (I_ff - 2 * P_outer)
    J_hh = jnp.diag(term1_diag) + term2

    J = jnp.block([[J_ff, J_fh], [J_fh.T, J_hh]])
    return 0.5 * (J + J.T)


def log_posterior(
    lambda_r: Float[Array, "N K"],
    sigma2: Float[Array, "N"],
    alpha: Float[Array, "2K"],
    observation: Float[Array, "N"],
) -> Float[Array, ""]:
    """Calculates log p(y_t | alpha_t) using Woodbury identity."""
    K = lambda_r.shape[1]

    f = alpha[:K]
    h = alpha[K:]

    pred_obs = lambda_r @ f
    innovation = observation - pred_obs

    Dinv_diag = 1.0 / (sigma2 + WOODBURY_DIAG_JITTER)
    Cinv_diag = 1.0 / (jnp.exp(h) + WOODBURY_DIAG_JITTER)

    Dinv_lambda_r = lambda_r * Dinv_diag[:, None]
    Dinv_innovation = innovation * Dinv_diag

    M = jnp.diag(Cinv_diag) + lambda_r.T @ Dinv_lambda_r
    M_jittered = M + K_CHOL_JITTER * jnp.eye(K)
    L_M = jax.scipy.linalg.cholesky(M_jittered, lower=True)

    logdet_M = 2.0 * jnp.sum(jnp.log(jnp.maximum(jnp.diag(L_M), LOG_DIAG_FLOOR)))
    logdet_C = jnp.sum(h)
    logdet_D = jnp.sum(jnp.log(jnp.maximum(sigma2, LOG_DIAG_FLOOR)))
    logdet_Sigma_t = logdet_M + logdet_C + logdet_D

    term1 = jnp.dot(innovation, Dinv_innovation)
    v = lambda_r.T @ Dinv_innovation
    z = jax.scipy.linalg.cho_solve((L_M, True), v)
    term2 = jnp.dot(v, z)
    quad_form = term1 - term2

    return -0.5 * (logdet_Sigma_t + quad_form)


def bif_penalty(
    a_pred: Float[Array, "2K"],
    a_updated: Float[Array, "2K"],
    Omega_pred: Float[Array, "2K 2K"],
    Omega_post: Float[Array, "2K 2K"],
) -> Float[Array, ""]:
    """Calculates BIF pseudo-likelihood penalty."""
    K2 = Omega_pred.shape[0]

    _, log_det_pred = jnp.linalg.slogdet(Omega_pred + K_CHOL_JITTER * jnp.eye(K2))
    _, log_det_post = jnp.linalg.slogdet(Omega_post + K_CHOL_JITTER * jnp.eye(K2))

    diff = a_updated - a_pred
    quad_term = diff.T @ Omega_pred @ diff

    return 0.5 * (log_det_post - log_det_pred + quad_term)


def invert_info_matrix(info_matrix: Float[Array, "D D"]) -> Float[Array, "D D"]:
    """Stable inversion of information matrix to covariance matrix via Cholesky."""
    D = info_matrix.shape[0]
    info_jittered = info_matrix + OBS_COV_JITTER * jnp.eye(D)

    L_info = jax.scipy.linalg.cholesky(info_jittered, lower=True)
    cov = jax.scipy.linalg.cho_solve((L_info, True), jnp.eye(D))
    return 0.5 * (cov + cov.T)


def predict_info_step(params: DFSVParams, state_post: BIFState) -> BIFState:
    """Predicts next state and information matrix."""
    K = params.lambda_r.shape[1]

    alpha_post = state_post.mean
    Omega_post = state_post.info

    f_post = alpha_post[:K]
    h_post = alpha_post[K:]

    f_pred = params.Phi_f @ f_post
    h_pred = params.mu + params.Phi_h @ (h_post - params.mu)
    alpha_pred = jnp.concatenate([f_pred, h_pred])

    F_t = jnp.block(
        [[params.Phi_f, jnp.zeros((K, K))], [jnp.zeros((K, K)), params.Phi_h]]
    )

    Q_f_inv = jnp.diag(jnp.exp(-h_pred))

    Q_h_jittered = params.Q_h + K_CHOL_JITTER * jnp.eye(K)
    L_Qh = jax.scipy.linalg.cholesky(Q_h_jittered, lower=True)
    Q_h_inv = jax.scipy.linalg.cho_solve((L_Qh, True), jnp.eye(K))

    Q_t_inv = jnp.block([[Q_f_inv, jnp.zeros((K, K))], [jnp.zeros((K, K)), Q_h_inv]])

    term = F_t.T @ Q_t_inv @ F_t
    M = Omega_post + term
    M_inv = invert_info_matrix(M)

    Q_inv_F = Q_t_inv @ F_t
    term2 = Q_inv_F @ M_inv @ Q_inv_F.T

    Omega_pred = Q_t_inv - term2
    Omega_pred = 0.5 * (Omega_pred + Omega_pred.T)

    return BIFState(mean=alpha_pred, info=Omega_pred)


def neg_log_post_h(
    log_vols: Float[Array, "K"],
    factors: Float[Array, "K"],
    lambda_r: Float[Array, "N K"],
    sigma2: Float[Array, "N"],
    predicted_state: Float[Array, "2K"],
    I_pred: Float[Array, "2K 2K"],
    observation: Float[Array, "N"],
) -> Float[Array, ""]:
    """Calculates negative log posterior w.r.t. log-volatilities (h)."""
    K = factors.shape[0]
    alpha = jnp.concatenate([factors, log_vols])

    neg_log_lik = -log_posterior(lambda_r, sigma2, alpha, observation)

    h_pred = predicted_state[K:]
    h_diff = log_vols - h_pred
    I_pred_hh = I_pred[K:, K:]
    I_pred_fh = I_pred[:K, K:]

    prior_penalty = 0.5 * jnp.dot(h_diff, jnp.dot(I_pred_hh, h_diff))
    prior_penalty += jnp.dot(factors - predicted_state[:K], jnp.dot(I_pred_fh, h_diff))

    return neg_log_lik + prior_penalty


def update_factors(
    log_volatility: Float[Array, "K"],
    lambda_r: Float[Array, "N K"],
    sigma2: Float[Array, "N"],
    observation: Float[Array, "N"],
    factors_pred: Float[Array, "K"],
    log_vols_pred: Float[Array, "K"],
    I_f: Float[Array, "K K"],
    I_fh: Float[Array, "K K"],
) -> Float[Array, "K"]:
    """Updates factor values f by solving linear system (closed form)."""
    K = lambda_r.shape[1]

    C = jnp.diag(jnp.exp(log_volatility))
    _, chol_M, inv_sigma2, B = _woodbury_setup(lambda_r, sigma2, C)

    Minv_B = jax.scipy.linalg.cho_solve((chol_M, True), B)
    lam_Ainv_lam = _symmetrize(B - B @ Minv_B)

    Dinv_y = observation * inv_sigma2
    u = lambda_r.T @ Dinv_y
    Minv_u = jax.scipy.linalg.cho_solve((chol_M, True), u)
    lam_Ainv_y = u - B @ Minv_u

    lhs_mat = lam_Ainv_lam + I_f + K_CHOL_JITTER * jnp.eye(K)

    rhs_vec = (
        lam_Ainv_y + (I_f @ factors_pred) + (I_fh @ (log_volatility - log_vols_pred))
    )

    return jnp.linalg.solve(lhs_mat, rhs_vec)


def update_h_bfgs(
    h_init: Float[Array, "K"],
    factors: Float[Array, "K"],
    lambda_r: Float[Array, "N K"],
    sigma2: Float[Array, "N"],
    pred_state: Float[Array, "2K"],
    I_pred: Float[Array, "2K 2K"],
    observation: Float[Array, "N"],
    solver: optx.AbstractMinimiser,
) -> tuple[Float[Array, "K"], Float[Array, ""]]:
    """Updates log-volatilities h using BFGS."""
    K = h_init.shape[0]

    def objective_fn(h, _):
        return neg_log_post_h(
            log_vols=h,
            factors=factors,
            lambda_r=lambda_r,
            sigma2=sigma2,
            predicted_state=pred_state,
            I_pred=I_pred,
            observation=observation,
        )

    sol = optx.minimise(
        fn=objective_fn, solver=solver, y0=h_init, args=None, max_steps=100, throw=False
    )

    successful = sol.result == optx.RESULTS.successful
    return jnp.where(successful, sol.value, h_init), successful


def block_coordinate_update(
    alpha_init: Float[Array, "2K"],
    params: DFSVParams,
    pred_state: BIFState,
    observation: Float[Array, "N"],
    max_iters: int = 10,
    h_solver: optx.AbstractMinimiser = optx.BFGS(rtol=1e-4, atol=1e-6),
) -> Float[Array, "2K"]:
    """Optimizes the state vector alpha_t via alternating minimization."""
    K = params.lambda_r.shape[1]

    alpha_current = alpha_init
    alpha_pred = pred_state.mean
    I_pred = pred_state.info

    factors_current = alpha_current[:K]
    log_vols_current = alpha_current[K:]

    factors_pred = alpha_pred[:K]
    log_vols_pred = alpha_pred[K:]

    I_f = I_pred[:K, :K]
    I_fh = I_pred[:K, K:]

    def body_fn(i, carry):
        f_curr, h_curr = carry

        f_new = update_factors(
            log_volatility=h_curr,
            lambda_r=params.lambda_r,
            sigma2=params.sigma2,
            observation=observation,
            factors_pred=factors_pred,
            log_vols_pred=log_vols_pred,
            I_f=I_f,
            I_fh=I_fh,
        )

        h_new, _ = update_h_bfgs(
            h_init=h_curr,
            factors=f_new,
            lambda_r=params.lambda_r,
            sigma2=params.sigma2,
            pred_state=alpha_pred,
            I_pred=I_pred,
            observation=observation,
            solver=h_solver,
        )
        return (f_new, h_new)

    final_f, final_h = jax.lax.fori_loop(
        0, max_iters, body_fn, (factors_current, log_vols_current)
    )
    return jnp.concatenate([final_f, final_h])


def update_info_step(
    params: DFSVParams,
    state_pred: BIFState,
    observation: Float[Array, "N"],
    max_iters: int = 10,
) -> tuple[BIFState, Float[Array, ""]]:
    """Performs the BIF update step: posterior mode finding + FIM update."""
    K = params.lambda_r.shape[1]

    alpha_pred = state_pred.mean
    Omega_pred = state_pred.info

    alpha_updated = block_coordinate_update(
        alpha_init=alpha_pred,
        params=params,
        pred_state=state_pred,
        observation=observation,
        max_iters=max_iters,
    )

    J_obs = observed_fim(
        lambda_r=params.lambda_r,
        sigma2=params.sigma2,
        alpha=alpha_updated,
        observation=observation,
    )

    Omega_post = Omega_pred + J_obs + OBS_COV_JITTER * jnp.eye(2 * K)
    Omega_post = 0.5 * (Omega_post + Omega_post.T)

    log_lik_fit = log_posterior(
        lambda_r=params.lambda_r,
        sigma2=params.sigma2,
        alpha=alpha_updated,
        observation=observation,
    )

    penalty = bif_penalty(
        a_pred=alpha_pred,
        a_updated=alpha_updated,
        Omega_pred=Omega_pred,
        Omega_post=Omega_post,
    )

    log_lik_contrib = log_lik_fit - penalty

    return BIFState(mean=alpha_updated, info=Omega_post), log_lik_contrib
