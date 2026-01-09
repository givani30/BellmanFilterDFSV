import jax
import jax.numpy as jnp
import jax.scipy.special
from jax import Array
from jaxtyping import Float

from .types import DFSVParams, ParticleState


def initialize_particles(
    params: DFSVParams, num_particles: int, key: jax.Array
) -> ParticleState:
    """Initialize particles from the model's initial distribution.

    Assumes factors f_0 ~ N(0, I) and log-vols h_0 ~ N(μ, P_h).
    """
    K = params.lambda_r.shape[1]
    key, subkey = jax.random.split(key)

    # 1. Initialize means
    f0 = jnp.zeros(K)
    h0 = params.mu
    initial_mean = jnp.concatenate([f0, h0])

    # 2. Initialize covariances
    P_f = jnp.eye(K)

    # Solve for P_h (Lyapunov equation for AR(1))
    # P_h = Phi_h @ P_h @ Phi_h.T + Q_h
    # vec(P_h) = (I - Phi_h kron Phi_h)^-1 vec(Q_h)
    kron_prod = jnp.kron(params.Phi_h, params.Phi_h)
    I_k2 = jnp.eye(K * K)
    vec_Qh = params.Q_h.flatten()
    vec_Ph = jnp.linalg.solve(I_k2 - kron_prod, vec_Qh)
    P_h = vec_Ph.reshape(K, K)

    # Block diagonal covariance
    P_0 = jnp.block([[P_f, jnp.zeros((K, K))], [jnp.zeros((K, K)), P_h]])

    # 3. Sample particles
    # shape: (2K, P)
    L_0 = jnp.linalg.cholesky(P_0 + 1e-6 * jnp.eye(2 * K))
    noise = jax.random.normal(subkey, shape=(2 * K, num_particles))
    particles = initial_mean[:, None] + L_0 @ noise

    # 4. Initialize weights (uniform)
    log_weights = jnp.full((num_particles,), -jnp.log(num_particles))

    return ParticleState(particles=particles, log_weights=log_weights)


def predict_particles(
    params: DFSVParams, state: ParticleState, key: jax.Array
) -> tuple[ParticleState, jax.Array]:
    """Propagate particles one step forward."""
    K = params.lambda_r.shape[1]
    num_particles = state.particles.shape[1]

    key, key_f, key_h = jax.random.split(key, 3)

    # Extract components
    f_t = state.particles[:K, :]
    h_t = state.particles[K:, :]

    # 1. Predict log-volatilities: h_{t+1} = mu + Phi_h(h_t - mu) + eta_t
    mu_col = params.mu[:, None]
    h_dev = h_t - mu_col
    h_pred_mean = mu_col + params.Phi_h @ h_dev

    # Sample eta_t ~ N(0, Q_h)
    L_Qh = jnp.linalg.cholesky(params.Q_h + 1e-6 * jnp.eye(K))
    noise_h = L_Qh @ jax.random.normal(key_h, shape=(K, num_particles))
    h_tp1 = h_pred_mean + noise_h

    # 2. Predict factors: f_{t+1} = Phi_f f_t + diag(exp(h_{t+1}/2)) eps_t
    f_pred_mean = params.Phi_f @ f_t

    # Sample eps_t ~ N(0, I)
    noise_f = jax.random.normal(key_f, shape=(K, num_particles))

    # Stochastic volatility scaling using NEW h_{t+1}
    vol_scale = jnp.exp(h_tp1 / 2.0)
    f_tp1 = f_pred_mean + vol_scale * noise_f

    # Combine
    particles_tp1 = jnp.vstack([f_tp1, h_tp1])

    # Weights carry over until update step
    return ParticleState(particles=particles_tp1, log_weights=state.log_weights), key


def compute_log_likelihood_particles(
    params: DFSVParams, particles: Float[Array, "2K P"], observation: Float[Array, "N"]
) -> Float[Array, "P"]:
    """Compute log p(y_t | x^{(i)}_t) for each particle."""
    K = params.lambda_r.shape[1]
    N = params.lambda_r.shape[0]

    factors = particles[:K, :]

    # Predicted observation: y = Λ f
    y_pred = params.lambda_r @ factors

    # Residuals
    # observation is (N,) -> (N, 1) for broadcasting
    residuals = observation[:, None] - y_pred

    # Observation noise covariance is diagonal sigma2
    # log p(y|x) = -0.5 * (N log(2pi) + sum(log(sigma2)) + sum(res^2 / sigma2))

    # Precompute constants
    log_2pi = jnp.log(2 * jnp.pi)
    safe_sigma2 = jnp.maximum(params.sigma2, 1e-10)
    log_det_R = jnp.sum(jnp.log(safe_sigma2))

    # Weighted squared residuals
    # (N, P) / (N, 1) -> (N, P) -> sum over N -> (P,)
    weighted_sq_res = jnp.sum(residuals**2 / safe_sigma2[:, None], axis=0)

    log_liks = -0.5 * (N * log_2pi + log_det_R + weighted_sq_res)

    return log_liks


def systematic_resample(
    key: jax.Array,
    log_weights: Float[Array, "P"],
    particles: Float[Array, "2K P"],
) -> tuple[Float[Array, "2K P"], Float[Array, "P"]]:
    """Performs systematic resampling of particles."""
    num_particles = log_weights.shape[0]

    # Convert log weights to linear weights
    # Normalize first for stability: w = exp(log_w - max_log_w)
    log_weights_norm = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights = jnp.exp(log_weights_norm)

    # Generate uniform spacers
    # u ~ U[0, 1/P]
    u = jax.random.uniform(key, (), minval=0.0, maxval=1.0 / num_particles)
    # points = u + [0, 1/P, 2/P, ...]
    points = u + jnp.arange(num_particles) / num_particles

    # Cumulative weights
    cum_weights = jnp.cumsum(weights)

    # Find indices where cum_weights >= points
    # searchsorted returns the first index where cum_weights[i] >= point
    indices = jnp.searchsorted(cum_weights, points)

    # Clip indices to be safe (though searchsorted should be fine with correct weights)
    indices = jnp.clip(indices, 0, num_particles - 1)

    # Resample
    new_particles = particles[:, indices]

    # Reset weights to uniform
    new_log_weights = jnp.full((num_particles,), -jnp.log(num_particles))

    return new_particles, new_log_weights


def calculate_ess(log_weights: Float[Array, "P"]) -> Float[Array, ""]:
    """Calculate Effective Sample Size (ESS) from log weights."""
    # Normalize log weights
    log_w_norm = log_weights - jax.scipy.special.logsumexp(log_weights)
    w_norm = jnp.exp(log_w_norm)

    # ESS = 1 / sum(w^2)
    ess = 1.0 / jnp.sum(w_norm**2)
    return ess
