
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from ._bellman_math import invert_info_matrix, predict_info_step, update_info_step
from ._particle_math import (
    calculate_ess,
    compute_log_likelihood_particles,
    initialize_particles,
    predict_particles,
    systematic_resample,
)
from .types import (
    BIFState,
    DFSVParams,
    FilterResult,
    ParticleFilterResult,
    ParticleState,
)


class BellmanFilter(eqx.Module):
    """Bellman Information Filter for DFSV models."""

    params: DFSVParams

    def __init__(self, params: DFSVParams):
        """Initializes the filter with model parameters."""
        self.params = params

    def initialize(self) -> BIFState:
        """Initializes the filter state from model parameters."""
        K = self.params.lambda_r.shape[1]

        f0 = jnp.zeros(K)
        h0 = self.params.mu
        mean = jnp.concatenate([f0, h0])

        P_f = jnp.eye(K)

        kron_prod = jnp.kron(self.params.Phi_h, self.params.Phi_h)
        I_k2 = jnp.eye(K * K)
        vec_Qh = self.params.Q_h.flatten()
        vec_Ph = jnp.linalg.solve(I_k2 - kron_prod, vec_Qh)
        P_h = vec_Ph.reshape(K, K)

        P_0 = jnp.block([[P_f, jnp.zeros((K, K))], [jnp.zeros((K, K)), P_h]])

        Omega_0 = jnp.linalg.inv(P_0)

        return BIFState(mean=mean, info=Omega_0)

    def filter(self, observations: Float[Array, "T N"]) -> FilterResult:
        """Runs the filter over a sequence of observations."""
        init_state = self.initialize()

        # Initialize accumulator as JAX array to ensure type consistency
        init_carry = (init_state, jnp.array(0.0))

        def scan_step(carry, obs_t):
            state_prev, log_lik_prev = carry

            state_pred = predict_info_step(self.params, state_prev)

            state_updated, log_lik_contrib = update_info_step(
                self.params, state_pred, obs_t
            )

            new_log_lik = log_lik_prev + log_lik_contrib
            new_carry = (state_updated, new_log_lik)

            return new_carry, (state_updated.mean, state_updated.info)

        final_carry, (means, infos) = jax.lax.scan(scan_step, init_carry, observations)

        total_log_lik = final_carry[1]

        return FilterResult(means=means, infos=infos, log_likelihood=total_log_lik)

    def smooth_state(
        self, mean: Float[Array, "2K"], info: Float[Array, "2K 2K"]
    ) -> tuple[Float[Array, "2K"], Float[Array, "2K 2K"]]:
        """Converts information state (α, Ω) to covariance state (α, P)."""
        cov = invert_info_matrix(info)
        return mean, cov


class ParticleFilter(eqx.Module):
    """Particle Filter (SISR) for DFSV models."""

    params: DFSVParams
    num_particles: int
    resample_threshold_frac: float
    seed: int

    def __init__(
        self,
        params: DFSVParams,
        num_particles: int = 1000,
        resample_threshold_frac: float = 0.5,
        seed: int = 42,
    ):
        """Initializes the particle filter."""
        self.params = params
        self.num_particles = num_particles
        self.resample_threshold_frac = resample_threshold_frac
        self.seed = seed

    def filter(self, observations: Float[Array, "T N"]) -> ParticleFilterResult:
        """Runs the particle filter over a sequence of observations."""
        key = jax.random.PRNGKey(self.seed)

        # Initialize
        init_state = initialize_particles(self.params, self.num_particles, key)
        init_ll = jnp.array(0.0)

        # Scan carry: (state, key, log_lik_accum)
        init_carry = (init_state, key, init_ll)

        def scan_step(carry, obs_t):
            state_prev, key_curr, ll_accum = carry

            # 1. Predict
            state_pred, key_next = predict_particles(self.params, state_prev, key_curr)

            # 2. Update weights
            log_liks = compute_log_likelihood_particles(
                self.params, state_pred.particles, obs_t
            )
            unnorm_log_weights = state_pred.log_weights + log_liks

            # Update log-likelihood
            ll_step = jax.scipy.special.logsumexp(unnorm_log_weights)
            ll_next = ll_accum + ll_step

            # 3. Resample (if needed)
            # Normalize for ESS calculation
            ess = calculate_ess(unnorm_log_weights)
            threshold = self.num_particles * self.resample_threshold_frac

            def do_resample(args):
                k, lw, p = args
                p_new, lw_new = systematic_resample(k, lw, p)
                return p_new, lw_new

            def no_resample(args):
                _, lw, p = args
                # Just normalize weights for next step consistency if desired,
                # but standard PF carries unnormalized until resampling.
                # However, for stability we usually work with normalized weights or
                # carry log-weights. Here we keep log-weights.
                # Wait, systematic_resample returns uniform weights.
                # If we don't resample, we should keep the current weights.
                return p, lw

            # We need to pass unnorm_log_weights to the next step,
            # BUT systematic_resample resets them to uniform (-log N).
            # So we need to handle the weight transition carefully.

            particles_next, log_weights_next = jax.lax.cond(
                ess < threshold,
                do_resample,
                no_resample,
                (key_next, unnorm_log_weights, state_pred.particles),
            )

            state_next = ParticleState(
                particles=particles_next, log_weights=log_weights_next
            )

            # Calculate estimates for output (using weights BEFORE resampling, or AFTER?
            # Usually after update step but before resampling is 'filtered' estimate)
            # Let's use the weights derived from observation t

            # Normalize for estimation
            lw_norm = unnorm_log_weights - jax.scipy.special.logsumexp(
                unnorm_log_weights
            )
            w_norm = jnp.exp(lw_norm)

            mean_est = jnp.sum(state_pred.particles * w_norm[None, :], axis=1)

            # Covariance
            diff = state_pred.particles - mean_est[:, None]
            cov_est = (diff * w_norm[None, :]) @ diff.T

            output = (mean_est, cov_est)
            new_carry = (state_next, key_next, ll_next)

            return new_carry, output

        final_carry, (means, covs) = jax.lax.scan(scan_step, init_carry, observations)

        return ParticleFilterResult(
            means=means, covs=covs, log_likelihood=final_carry[2]
        )
