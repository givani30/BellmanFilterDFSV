"""Custom Particle Filter implementation for testing."""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from typing import Tuple, Dict, Any

from bellman_filter_dfsv.filters.particle import DFSVParticleFilter
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass

class ImprovedParticleFilter(DFSVParticleFilter):
    """
    Improved Particle Filter with better resampling strategy and initialization.
    """

    def __init__(
        self,
        N: int,
        K: int,
        num_particles: int = 2000,
        resample_threshold_frac: float = 0.7,  # Higher threshold for more frequent resampling
        seed: int = 42,
    ):
        """
        Initialize the Improved Particle Filter.

        Args:
            N: Number of assets.
            K: Number of factors.
            num_particles: Number of particles.
            resample_threshold_frac: Fraction of `num_particles` below which
                resampling is triggered based on Effective Sample Size (ESS).
            seed: Seed for JAX's random number generator.
        """
        super().__init__(N, K, num_particles, resample_threshold_frac, seed)

    def _initialize_particles(
        self, params: DFSVParamsDataclass, rng_key: jax.random.PRNGKey
    ) -> Tuple[jax.random.PRNGKey, jnp.ndarray, jnp.ndarray]:
        """
        Initialize particle states with a wider distribution to better explore the state space.

        Args:
            params: Model parameters (JAX dataclass) for this specific initialization.
            rng_key: JAX random key.

        Returns:
            Tuple containing:
                - Updated JAX random key.
                - Initialized particles (state_dim, num_particles).
                - Initial uniform normalized log weights (num_particles,).
        """
        # Get initial state mean and covariance from base class method
        initial_state_mean, initial_cov = super().initialize_state(params)

        # Split key for sampling
        rng_key, sample_key = jax.random.split(rng_key)

        # Use a wider covariance for initialization to better explore the state space
        # Scale the covariance matrix by a factor to increase exploration
        exploration_factor = 1.5
        wider_cov = initial_cov * exploration_factor

        try:
            # Use Cholesky for stable sampling
            L = jax.scipy.linalg.cholesky(wider_cov, lower=True)
            # Standard normal noise
            noise = jax.random.normal(
                sample_key, shape=(self.state_dim, self.num_particles)
            )
            # Affine transformation: mean + L @ noise
            particles = initial_state_mean.reshape(-1,1) + L @ noise
        except np.linalg.LinAlgError as e:
            # Fallback to diagonal covariance if Cholesky fails
            diag_cov = jnp.diag(jnp.diag(wider_cov))
            L = jnp.sqrt(diag_cov)
            noise = jax.random.normal(
                sample_key, shape=(self.state_dim, self.num_particles)
            )
            particles = initial_state_mean.reshape(-1,1) + L @ noise

        # Initialize weights to uniform normalized log weights (using float32)
        log_weights_normalized = jnp.full(
            self.num_particles, -jnp.log(self.num_particles), dtype=jnp.float32
        )

        return rng_key, particles, log_weights_normalized

    @eqx.filter_jit # self is static
    def resample_particles(
        self,
        rng_key: jax.random.PRNGKey,
        particles: jnp.ndarray,             # (state_dim, P)
        unnormalized_log_weights: jnp.ndarray # (P,)
    ) -> Tuple[jax.random.PRNGKey, jnp.ndarray, jnp.ndarray, float]:
        """
        Perform systematic resampling if ESS is below threshold.
        Also adds a small jitter to resampled particles to prevent degeneracy.

        Args:
            rng_key: JAX random key.
            particles: Predicted particles (state_dim, num_particles).
            unnormalized_log_weights: Log weights before normalization (num_particles,).

        Returns:
            Tuple containing:
                - Updated JAX random key.
                - Next particles (resampled or predicted) (state_dim, num_particles).
                - Corresponding normalized log weights (num_particles,).
                - Effective Sample Size (ESS) calculated before resampling.
        """
        # Call the parent class method first
        rng_key, resampled_particles, normalized_log_weights, ess = super().resample_particles(
            rng_key, particles, unnormalized_log_weights
        )

        # Add a small jitter to the resampled particles to prevent degeneracy
        # Use JAX's where to conditionally apply jitter based on ESS
        # Split the key for jittering
        rng_key, jitter_key = jax.random.split(rng_key)

        # Add small Gaussian noise to the particles
        jitter_scale = 0.01  # Small scale factor for the jitter
        jitter = jitter_scale * jax.random.normal(
            jitter_key, shape=resampled_particles.shape
        )

        # Apply jitter to the particles conditionally using JAX's where
        # This avoids the TracerBoolConversionError
        needs_jitter = ess < self.resample_threshold_ess
        jittered_particles = jnp.where(
            needs_jitter,
            resampled_particles + jitter,
            resampled_particles
        )

        return rng_key, jittered_particles, normalized_log_weights, ess

def create_improved_filter(params: DFSVParamsDataclass, num_particles: int = 2000) -> Dict[str, Any]:
    """
    Create an improved particle filter instance for testing.

    Args:
        params: DFSV model parameters.
        num_particles: Number of particles to use.

    Returns:
        Dictionary with the filter instance.
    """
    return {
        "improved_particle": ImprovedParticleFilter(
            N=params.N, K=params.K, num_particles=num_particles
        )
    }
