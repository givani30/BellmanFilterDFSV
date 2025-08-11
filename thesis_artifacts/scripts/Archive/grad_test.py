import jax
import jax.numpy as jnp
import jax.scipy.linalg


def build_covariance(lambda_r, exp_h, sigma2):
    """
    Build A = Lambda diag(exp_h) Lambda^T + diag(sigma2).
    lambda_r: (N,K)
    exp_h: (K,)
    sigma2: (N,) or (N,)-shaped diagonal
    """
    # diag(exp_h) in JAX can be built as:
    Sigma_f = jnp.diag(exp_h)  # (K,K)
    # Then A = lambda_r @ Sigma_f @ lambda_r.T
    A = lambda_r @ Sigma_f @ lambda_r.T  # (N,N)
    # Add diagonal noise and tiny regularization
    A += jnp.diag(sigma2) + 1e-8 * jnp.eye(A.shape[0])
    # Symmetrize just in case of minor FP asymmetry
    A = 0.5 * (A + A.T)
    return A


@jax.jit
def jax_bellman_objective(
    alpha, predicted_state, I_pred, observation, lambda_r, sigma2
):
    """
    JAX implementation of the Bellman objective function, negative log-posterior.
    Minimizes: -(log_likelihood - penalty).

    alpha: (2K,)  => [f, log_vols]
    predicted_state: (2K,) => [f_pred, h_pred]
    I_pred: (2K, 2K)
    observation: (N,)
    lambda_r: (N,K)
    sigma2: (N,) diagonal noise
    """
    # Unpack
    K = lambda_r.shape[1]
    N = lambda_r.shape[0]

    alpha = alpha.flatten()
    predicted_state = predicted_state.flatten()
    observation = observation.flatten()

    f = alpha[:K]
    log_vols = alpha[K:]

    # Innovation
    pred_obs = lambda_r @ f
    innovation = observation - pred_obs

    # Build covariance A
    exp_log_vols = jnp.exp(log_vols)
    A = build_covariance(lambda_r, exp_log_vols, sigma2)

    # Cholesky factor
    L = jnp.linalg.cholesky(A)

    # quad_form = innovation^T A^{-1} innovation, done via triangular solves
    alpha_vec = jax.scipy.linalg.solve_triangular(L, innovation, lower=True)
    quad_form = jnp.sum(alpha_vec**2)

    # logdet(A) = 2.0 * sum(log(diag(L)))
    logdet_A = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

    # Negative log-likelihood
    N_ = observation.shape[0]
    neg_log_lik = 0.5 * (N_ * jnp.log(2.0 * jnp.pi) + logdet_A + quad_form)

    # Prior penalty
    state_diff = alpha - predicted_state
    penalty = 0.5 * (state_diff @ (I_pred @ state_diff))

    # Full negative log-posterior
    # i.e. we want to MINIMIZE this quantity
    return neg_log_lik + penalty


@jax.jit
def explicit_grad_bellman(
    alpha, predicted_state, I_pred, observation, lambda_r, sigma2
):
    """
    Hand-coded gradient of the same negative log-posterior.

    alpha: (2K,) => [f, h]
    """
    K = lambda_r.shape[1]
    N_ = lambda_r.shape[0]

    # Flatten vectors
    alpha = alpha.flatten()
    predicted_state = predicted_state.flatten()
    observation = observation.flatten()

    # Split alpha
    f = alpha[:K]
    h = alpha[K:]
    exp_h = jnp.exp(h)

    # Innovation
    innovation = observation - (lambda_r @ f)

    # Build covariance
    A = build_covariance(lambda_r, exp_h, sigma2)

    # Cholesky
    L = jnp.linalg.cholesky(A)

    def A_inv_matmul(x):
        return jax.scipy.linalg.cho_solve((L, True), x)

    # A^{-1} * lambda_r, A^{-1} * innovation
    A_inv_lambda = A_inv_matmul(lambda_r)  # shape (N_, K)
    A_inv_innov = A_inv_matmul(innovation)  # shape (N_,)

    # -- Gradient w.r.t f:
    #    d/d f of 0.5 * eps^T A^{-1} eps = -(lambda_r^T A^{-1} eps)
    grad_f = -lambda_r.T @ A_inv_innov

    # -- Gradient w.r.t h:
    #    0.5 * exp(h) [ (lam_k^T A^{-1} lam_k) - (lam_k^T A^{-1} eps)^2 ]
    term1 = jnp.diag(lambda_r.T @ A_inv_lambda)  # (K,)
    proj = lambda_r.T @ A_inv_innov  # (K,)
    term2 = proj**2  # (K,)
    grad_h = 0.5 * exp_h * (term1 - term2)

    # -- Add prior penalty gradient
    state_diff = alpha - predicted_state
    penalty_grad = I_pred @ state_diff  # shape (2K,)

    # Concatenate final gradient
    grad_likelihood = jnp.concatenate([grad_f, grad_h])
    grad_total = grad_likelihood + penalty_grad

    return grad_total


if __name__ == "__main__":
    # Example dimensions
    K = 2  # number of factors
    N = 3  # observation dimension

    key = jax.random.PRNGKey(0)
    # Random state vector alpha of length 2K (first K are f, next K are log_vols)
    alpha = jax.random.normal(key, (2 * K,))
    # Predicted state and information matrix for the penalty term
    predicted_state = jax.random.normal(key, (2 * K,))
    I_pred = jnp.eye(2 * K)

    # Random observation vector and observation matrix lambda_r
    lambda_r = jax.random.normal(key, (N, K))
    observation = jax.random.normal(key, (N,))
    sigma2 = jnp.ones((N,))  # diagonal measurement noise

    # Automatic gradient using JAX
    auto_grad = jax.grad(jax_bellman_objective)(
        alpha, predicted_state, I_pred, observation, lambda_r, sigma2
    )

    # Explicit gradient computed from our derivation
    expl_grad = explicit_grad_bellman(
        alpha, predicted_state, I_pred, observation, lambda_r, sigma2
    )

    print("Automatic gradient from jax.grad:")
    print(auto_grad)
    print("\nExplicit gradient from our derivation:")
    print(expl_grad)
    print("\nDifference (auto_grad - expl_grad):")
    print(auto_grad - expl_grad)
