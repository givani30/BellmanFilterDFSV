"""
Nonlinear Filter implementations for Dynamic Factor Stochastic Volatility models.

This module provides filter classes for state estimation in DFSV models,
including Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF),
Particle Filters (PF) and Bellman Filters (BF) to handle the nonlinearities
introduced by stochastic volatility.
"""

import warnings
from typing import Tuple, Union, NamedTuple
from functools import partial
from dataclasses import asdict

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import jax.scipy.special
import numpy as np
from jax import jit

# Local imports
# Removed import of DFSV_params from simulation
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass # Only import the dataclass
from .base import DFSVFilter # Import base class from sibling module

# Try importing tqdm for progress bars, provide a fallback
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        """Fallback tqdm iterator if tqdm is not installed."""
        warnings.warn("tqdm not installed. Progress bars will not be shown.")
        return iterable

# Enable 64-bit precision for JAX if needed (can be set globally)
# jax.config.update("jax_enable_x64", True)


# Base class removed, will be imported from .base

# --- Particle Filter Implementation ---

# Define a structure for the state carried through lax.scan
class PFScanState(NamedTuple):
    """State carried through the particle filter's scan loop."""
    rng_key: jax.random.PRNGKey
    particles: jnp.ndarray  # Shape (state_dim, num_particles)
    normalized_log_weights: jnp.ndarray  # Shape (num_particles,)
    log_likelihood_accum: float


class DFSVParticleFilter(DFSVFilter):
    """
    Particle Filter (Bootstrap Filter) for DFSV models using JAX.

    Implements Sequential Importance Sampling with Resampling (SISR) for
    state estimation in DFSV models. Uses JAX for efficient computation,
    especially leveraging `jax.lax.scan` for the filtering loop.

    Inherits from `DFSVFilter` but overrides the `filter` method and provides
    particle-specific methods. Smoothing uses the base class RTS smoother
    with NumPy approximations.

    Attributes:
        num_particles (int): Number of particles used.
        resample_threshold_ess (float): ESS threshold for resampling (absolute value).
        seed (int): Seed for JAX PRNG initialization.
        params (DFSVParamsDataclass): Model parameters converted to JAX arrays.
        rng_key (jax.random.PRNGKey): Current state of the JAX random number generator.
        chol_Q_h (jnp.ndarray): Precomputed Cholesky decomposition of Q_h.
        particles (Optional[jnp.ndarray]): Current particles (state_dim, num_particles).
        weights (Optional[jnp.ndarray]): Current normalized log weights (num_particles,).
        effective_sample_size (Optional[np.ndarray]): History of ESS (T,).
    """

    def __init__(
        self,
        params: DFSVParamsDataclass,
        num_particles: int = 1000,
        resample_threshold_frac: float = 0.5,
        seed: int = 42,
    ):
        """
        Initialize the Particle Filter.

        Args:
            params: Parameters of the DFSV model (must be DFSVParamsDataclass).
                    Internal arrays will be ensured to be JAX arrays.
            num_particles: Number of particles.
            resample_threshold_frac: Fraction of `num_particles` below which
                resampling is triggered based on Effective Sample Size (ESS).
            seed: Seed for JAX's random number generator.
        """
        # Extract N, K and initialize base class
        N = params.N
        K = params.K
        super().__init__(N, K)

        if not isinstance(num_particles, int) or num_particles <= 0:
            raise ValueError("num_particles must be a positive integer.")
        if not 0.0 <= resample_threshold_frac <= 1.0:
            raise ValueError("resample_threshold_frac must be between 0 and 1.")

        self.num_particles: int = num_particles
        self.resample_threshold_ess: float = resample_threshold_frac * num_particles
        self.seed: int = seed

        # Convert input params to JAX DATACLASS and store
        self.params: DFSVParamsDataclass = self._params_to_jax(params)

        # Initialize JAX random key
        self.rng_key: jax.random.PRNGKey = jax.random.PRNGKey(self.seed)

        # Precompute Cholesky decomposition of Q_h (assumed constant)
        try:
            self.chol_Q_h: jnp.ndarray = jax.scipy.linalg.cholesky(
                self.params.Q_h, lower=True
            )
        except jnp.linalg.LinAlgError as e:
            raise ValueError(
                "Q_h matrix must be positive definite for Cholesky decomposition."
            ) from e

        # Storage for final JAX state (optional, set after filtering)
        self.particles: jnp.ndarray | None = None
        self.weights: jnp.ndarray | None = None # Stores normalized log weights
        self.effective_sample_size: np.ndarray | None = None # Stored as NumPy array

    def _params_to_jax(self, params: DFSVParamsDataclass) -> DFSVParamsDataclass:
        """
        Ensure the provided DFSVParamsDataclass contains JAX arrays.

        Args:
            params: Input DFSVParamsDataclass instance.

        Returns:
            DFSVParamsDataclass instance with JAX arrays.

        Raises:
            TypeError: If the input is not a DFSVParamsDataclass.
            ValueError: If conversion to JAX array fails for a parameter.
        """
        if not isinstance(params, DFSVParamsDataclass):
            raise TypeError(f"Input must be a DFSVParamsDataclass, got {type(params)}")

        # Convert relevant fields to JAX arrays, ensuring correct dtype (e.g., float64)
        default_dtype = jnp.float64 # Explicitly use float64
        updates = {}
        changed = False
        for field_name in ["lambda_r", "Phi_f", "Phi_h", "mu", "sigma2", "Q_h"]:
            current_value = getattr(params, field_name)
            # Check if it's already a JAX array of the correct type to avoid unnecessary conversion
            if not isinstance(current_value, jnp.ndarray) or current_value.dtype != default_dtype:
                try:
                    updates[field_name] = jnp.asarray(current_value, dtype=default_dtype)
                    changed = True
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Could not convert parameter '{field_name}' to JAX array: {e}")

        # If any arrays were converted, create a new instance with the JAX arrays
        if changed:
            return params.replace(**updates)
        else:
            # If no changes needed, return the original instance
            return params

    def initialize_particles(
        self, params: DFSVParamsDataclass, rng_key: jax.random.PRNGKey
    ) -> Tuple[jax.random.PRNGKey, jnp.ndarray, jnp.ndarray]:
        """
        Initialize particle states and weights using the prior distribution.

        Args:
            params: Model parameters (JAX dataclass).
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

        # Sample initial particles from N(initial_state_mean, initial_cov)
        try:
            # Use Cholesky for stable sampling
            L = jax.scipy.linalg.cholesky(initial_cov, lower=True)
            # Standard normal noise
            noise = jax.random.normal(
                sample_key, shape=(self.state_dim, self.num_particles)
            )
            # Affine transformation: mean + L @ noise
            particles = initial_state_mean + L @ noise
        except jnp.linalg.LinAlgError as e:
            raise ValueError(
                "Initial covariance matrix must be positive definite for sampling."
            ) from e

        # Initialize weights to uniform normalized log weights
        log_weights_normalized = jnp.full(
            self.num_particles, -jnp.log(self.num_particles)
        )

        return rng_key, particles, log_weights_normalized

    @partial(jit, static_argnums=(0,))
    def predict_particles(
        self, rng_key: jax.random.PRNGKey, particles: jnp.ndarray
    ) -> Tuple[jax.random.PRNGKey, jnp.ndarray]:
        """
        Propagate particles one step forward using the state transition dynamics.

        Args:
            rng_key: JAX random key.
            particles: Current particles (state_dim, num_particles).

        Returns:
            Tuple containing:
                - Updated JAX random key.
                - Predicted particles (state_dim, num_particles).
        """
        K = self.K
        params = self.params # Use stored JAX params

        # Split key for factor and volatility noise
        rng_key, key_h, key_f = jax.random.split(rng_key, 3)

        # Extract state components
        factors_t = particles[:K, :]
        log_vols_t = particles[K:, :]

        # Ensure mu is a column vector for broadcasting
        mu_col = params.mu.reshape(-1, 1)

        # 1. Predict log-volatilities: h_{t+1} = mu + Phi_h * (h_t - mu) + eta_t
        h_deviation = log_vols_t - mu_col
        h_mean_pred = mu_col + params.Phi_h @ h_deviation
        # Sample volatility noise: eta_t ~ N(0, Q_h) using precomputed Cholesky
        noise_h = self.chol_Q_h @ jax.random.normal(
            key_h, shape=(K, self.num_particles)
        )
        log_vols_tp1 = h_mean_pred + noise_h

        # 2. Predict factors: f_{t+1} = Phi_f * f_t + diag(exp(h_{t+1}/2)) * eps_t
        f_mean_pred = params.Phi_f @ factors_t
        # Sample factor noise: eps_t ~ N(0, I), then scale by predicted volatility
        std_noise_f = jax.random.normal(key_f, shape=(K, self.num_particles))
        # Use predicted log_vols_tp1 for the scaling factor
        vol_scale = jnp.exp(log_vols_tp1 / 2.0)
        noise_f = vol_scale * std_noise_f
        factors_tp1 = f_mean_pred + noise_f

        # Combine into predicted particles
        predicted_particles = jnp.vstack([factors_tp1, log_vols_tp1])

        return rng_key, predicted_particles

    @partial(jit, static_argnums=(0, 5, 6)) # self, K, N are static
    def compute_log_likelihood_particle(
        self,
        particles: jnp.ndarray,          # (state_dim, P)
        observation: jnp.ndarray,        # (N, 1)
        factor_loadings: jnp.ndarray,    # (N, K)
        obs_noise_cov: jnp.ndarray,      # (N, N) - Idiosyncratic variance matrix R_t
        K: int,
        N: int,
    ) -> jnp.ndarray:                     # (P,)
        """
        Compute the log-likelihood log p(y_t | x_t) for each particle x_t.

        Uses the observation equation: y_t = lambda_r @ f_t + epsilon_t
        where epsilon_t ~ N(0, R_t) and R_t = diag(sigma2).

        Args:
            particles: Predicted particles x_t (state_dim, num_particles).
            observation: Current observation y_t (N, 1).
            factor_loadings: Factor loading matrix lambda_r (N, K).
            obs_noise_cov: Observation noise covariance matrix R_t (N, N).
                           Typically diag(sigma2).
            K: Number of factors (static).
            N: Number of observations (static).

        Returns:
            Log-likelihood values for each particle (num_particles,).
        """
        # Extract factors from particles
        factors = particles[:K, :]  # Shape (K, P)

        # Calculate expected observation for each particle: lambda_r @ f_t
        expected_observation = factor_loadings @ factors  # Shape (N, P)

        # Calculate observation error (innovation) for each particle: y_t - E[y_t|x_t]
        # observation is (N, 1), expected_observation is (N, P) -> broadcasting
        observation_error = observation - expected_observation  # Shape (N, P)

        # Compute log probability density of N(observation_error | 0, obs_noise_cov)
        # We can use jax.scipy.stats.multivariate_normal.logpdf, but it expects
        # deviations as (P, N). Let's transpose the error.
        # Alternatively, use the manual Cholesky approach for potentially better
        # stability and control, especially if obs_noise_cov might be singular.

        # Use vmap with the static helper method for Cholesky approach
        # _compute_logprob_cholesky expects error (N,), cov (N, N), N (static)
        log_likelihoods = jax.vmap(
            DFSVParticleFilter._compute_logprob_cholesky,
            in_axes=(1, None, None),  # Map over axis 1 of error, broadcast cov and N
            out_axes=0,
        )(observation_error, obs_noise_cov, N)  # Result shape (P,)

        return log_likelihoods

    @staticmethod
    @partial(jit, static_argnums=(2,)) # N is static
    def _compute_logprob_cholesky(
        observation_error: jnp.ndarray, # (N,)
        covariance: jnp.ndarray,        # (N, N)
        N: int
    ) -> float:
        """
        Compute log N(observation_error | 0, covariance) using Cholesky.

        Args:
            observation_error: Difference y - E[y|x] for a single particle (N,).
            covariance: Covariance matrix R_t (N, N).
            N: Dimension of observation (static).

        Returns:
            Log-likelihood value for the particle.
        """
        try:
            # L @ L.T = covariance
            L = jax.scipy.linalg.cholesky(covariance, lower=True)
            # Solve L @ x = observation_error for x using forward substitution
            x = jax.scipy.linalg.solve_triangular(L, observation_error, lower=True)
            # Quadratic form: error^T @ Sigma^{-1} @ error = x^T @ x
            quad_form = jnp.sum(x**2)
            # Log determinant: log|Sigma| = 2 * sum(log(diag(L)))
            # Add epsilon for numerical stability if diagonal elements are near zero
            safe_diag_L = jnp.maximum(jnp.diag(L), 1e-10) # Epsilon
            log_det_sigma = 2 * jnp.sum(jnp.log(safe_diag_L))

            # Log likelihood: -0.5 * (N*log(2pi) + log_det_sigma + quad_form)
            log_prob = -0.5 * (N * jnp.log(2 * jnp.pi) + log_det_sigma + quad_form)

        except jnp.linalg.LinAlgError:
            # Handle cases where Cholesky fails (matrix not positive definite)
            # This might happen during optimization if sigma2 goes negative/zero
            log_prob = -jnp.inf # Assign very low likelihood

        return log_prob


    @partial(jit, static_argnums=(0,)) # self is static
    def resample_particles(
        self,
        rng_key: jax.random.PRNGKey,
        particles: jnp.ndarray,             # (state_dim, P)
        unnormalized_log_weights: jnp.ndarray # (P,)
    ) -> Tuple[jax.random.PRNGKey, jnp.ndarray, jnp.ndarray, float]:
        """
        Perform systematic resampling if ESS is below threshold.

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
        num_particles = particles.shape[1]

        # 1. Normalize log weights
        log_sum_weights = jax.scipy.special.logsumexp(unnormalized_log_weights)
        normalized_log_weights = unnormalized_log_weights - log_sum_weights
        # Also compute linear weights for ESS and resampling
        normalized_weights_linear = jnp.exp(normalized_log_weights)

        # 2. Calculate Effective Sample Size (ESS)
        # ESS = 1 / sum(w_norm^2)
        ess = 1.0 / jnp.sum(normalized_weights_linear**2)

        # 3. Resampling condition
        needs_resampling = ess < self.resample_threshold_ess

        # --- Define Resampling Logic (Systematic Resampling) ---
        def _systematic_resample(key_resample, particles_resample, weights_linear_resample):
            """Performs systematic resampling."""
            key_resample, subkey = jax.random.split(key_resample)
            n = weights_linear_resample.shape[0]
            # Generate starting point in [0, 1/n)
            u0 = jax.random.uniform(subkey) / n
            # Generate uniform strata points
            positions = u0 + jnp.arange(n) / n
            # Compute cumulative sum of weights
            cumulative_weights = jnp.cumsum(weights_linear_resample)
            # Find indices using searchsorted
            indices = jnp.searchsorted(cumulative_weights, positions)
            # Ensure indices are within bounds (can happen with numerical precision)
            indices = jnp.clip(indices, 0, n - 1)
            # Select particles based on indices
            resampled_particles = particles_resample[:, indices]
            # Reset weights to uniform normalized log weights after resampling
            resampled_log_weights = jnp.full(n, -jnp.log(n))
            return key_resample, resampled_particles, resampled_log_weights

        # --- Define Conditional Branches ---
        def _resample_branch(op):
            """Branch executed if resampling is needed."""
            key_in, particles_in, weights_lin_in, _ = op # Unpack, ignore log weights
            key_out, particles_out, log_weights_out = _systematic_resample(
                key_in, particles_in, weights_lin_in
            )
            return key_out, particles_out, log_weights_out

        def _no_resample_branch(op):
            """Branch executed if resampling is not needed."""
            key_in, particles_in, _, log_weights_in = op # Unpack, ignore linear weights
            # Return original key, predicted particles, and normalized log weights
            return key_in, particles_in, log_weights_in

        # 4. Use lax.cond to select branch
        # Operands must have the same structure for both branches
        operand = (rng_key, particles, normalized_weights_linear, normalized_log_weights)
        rng_key, next_particles, next_normalized_log_weights = jax.lax.cond(
            needs_resampling,
            _resample_branch,
            _no_resample_branch,
            operand
        )

        return rng_key, next_particles, next_normalized_log_weights, ess


    def filter(
        self, _params_arg: None = None, observations: Union[np.ndarray, jnp.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the Particle Filter using JAX scan.

        Args:
            _params_arg: This argument is ignored; the parameters stored in
                         `self.params` during initialization are used. It exists
                         for potential API compatibility but is not used internally.
            observations: Observed returns with shape (T, N) or (N, T).

        Returns:
            Tuple containing:
                - Filtered states (weighted mean estimate) (T, state_dim) as NumPy array.
                - Filtered state covariances (weighted estimate) (T, state_dim, state_dim) as NumPy array.
                - Total log-likelihood (float).

        Raises:
            ValueError: If observations are not provided or have incorrect shape.
        """
        if observations is None:
            raise ValueError("Observations must be provided.")

        # Ensure observations are JAX array and in (T, N) format
        obs_jax = jnp.asarray(observations)
        if obs_jax.ndim != 2:
             raise ValueError(f"Observations must be a 2D array, but got shape {obs_jax.shape}")
        if obs_jax.shape[1] != self.N:
             if obs_jax.shape[0] == self.N:
                 obs_jax = obs_jax.T # Transpose if (N, T)
             else:
                 raise ValueError(f"Observations dimension mismatch: expected {self.N} columns/rows, got {obs_jax.shape}")
        T = obs_jax.shape[0]

        # --- Prepare static parameters for scan body ---
        current_params = self.params
        K = self.K
        N = self.N
        # Precompute observation noise covariance matrix R (assuming diagonal sigma2)
        # Handle both 1D (variances) and 2D (full matrix) sigma2
        sigma2_curr = current_params.sigma2
        if sigma2_curr.ndim == 1:
            if sigma2_curr.shape[0] != N:
                 raise ValueError(f"sigma2 (1D) length {sigma2_curr.shape[0]} != N ({N})")
            obs_noise_cov = jnp.diag(sigma2_curr)
        elif sigma2_curr.ndim == 2:
            if sigma2_curr.shape != (N, N):
                 raise ValueError(f"sigma2 (2D) shape {sigma2_curr.shape} != ({N}, {N})")
            obs_noise_cov = sigma2_curr
        else:
            raise ValueError(f"sigma2 has invalid shape {sigma2_curr.shape}")


        # --- Initialize state for scan ---
        init_rng_key, init_particles, init_log_weights = self.initialize_particles(
            current_params, self.rng_key # Use the instance's current rng_key
        )
        initial_scan_state = PFScanState(
            rng_key=init_rng_key,
            particles=init_particles,
            normalized_log_weights=init_log_weights,
            log_likelihood_accum=0.0,
        )

        # --- Define the scan body function ---
        # @jit # JIT the scan body for performance
        def scan_body(state: PFScanState, obs_t: jnp.ndarray):
            """Body function for jax.lax.scan, performs one filter step."""
            key, current_particles, current_norm_log_weights, ll_accum = state
            observation = obs_t.reshape(-1, 1) # Ensure (N, 1)

            # 1. Predict
            key, predicted_particles = self.predict_particles(key, current_particles)

            # 2. Weight
            # Use precomputed obs_noise_cov
            log_likelihood_terms = self.compute_log_likelihood_particle(
                predicted_particles,
                observation,
                current_params.lambda_r,
                obs_noise_cov, # Pass precomputed R
                K,
                N,
            )
            # Calculate unnormalized weights for resampling and LL contribution
            unnormalized_log_weights = current_norm_log_weights + log_likelihood_terms

            # 3. Calculate Log-Likelihood Contribution for this step
            # log[ sum(w_{t-1} * p(y_t|x_t)) ] = logsumexp(log(w_{t-1}) + log p(y_t|x_t))
            ll_increment = jax.scipy.special.logsumexp(unnormalized_log_weights)
            ll_accum_next = ll_accum + ll_increment

            # 4. Resample
            key, particles_next, next_norm_log_weights, ess = self.resample_particles(
                key, predicted_particles, unnormalized_log_weights
            )

            # --- Calculate outputs for storage ---
            # Weighted mean estimate of state x_t
            weights_linear = jnp.exp(next_norm_log_weights)
            filtered_mean = jnp.sum(particles_next * weights_linear, axis=1)

            # Weighted covariance estimate of state x_t
            diff = particles_next - filtered_mean.reshape(-1, 1)
            # einsum: weights(p) * diff(j,p) * diff(k,p) -> sum over p -> cov(j,k)
            filtered_cov = jnp.einsum('p,jp,kp->jk', weights_linear, diff, diff)
            # Ensure symmetry
            filtered_cov = (filtered_cov + filtered_cov.T) / 2.0

            # Prepare next state and outputs
            next_scan_state = PFScanState(
                rng_key=key,
                particles=particles_next,
                normalized_log_weights=next_norm_log_weights,
                log_likelihood_accum=ll_accum_next,
            )
            scan_output = (filtered_mean, filtered_cov, ess)

            return next_scan_state, scan_output

        # --- Run the scan ---
        final_state, scan_outputs = jax.lax.scan(
            scan_body, initial_scan_state, obs_jax
        )

        # Unpack results
        filtered_states_means, filtered_states_covs, ess_history = scan_outputs
        final_log_likelihood = float(final_state.log_likelihood_accum)

        # Update instance state
        self.rng_key = final_state.rng_key # Store final key state
        self.particles = final_state.particles # Store final JAX particles
        self.weights = final_state.normalized_log_weights # Store final JAX log weights

        # Store results as NumPy arrays in the instance
        self.filtered_states = np.array(filtered_states_means)
        self.filtered_covs = np.array(filtered_states_covs)
        self.log_likelihood = final_log_likelihood
        self.effective_sample_size = np.array(ess_history)
        self.is_filtered = True
        self.is_smoothed = False # Reset smoothed flag

        return self.filtered_states, self.filtered_covs, self.log_likelihood


    # --- Methods for RTS Smoothing (using base class implementation) ---

    def _get_transition_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        Get the linearized state transition matrix F_t (NumPy version for smoother).

        This provides the approximation needed by the base class RTS smoother.
        It uses the stored `self.params` (expected to be JAX arrays) and
        converts them temporarily to NumPy.

        Args:
            state: Current state estimate (state_dim, 1) as NumPy array.
                   Note: This argument is not used in the standard DFSV linearization
                   where F_t only depends on parameters, but included for API consistency.

        Returns:
            Linearized transition matrix F_t (state_dim, state_dim) as NumPy array.
        """
        K = self.K
        # Convert necessary params from JAX to NumPy for the smoother
        phi_f_np = np.array(self.params.Phi_f)
        phi_h_np = np.array(self.params.Phi_h)

        F_t = np.zeros((self.state_dim, self.state_dim))
        F_t[:K, :K] = phi_f_np
        F_t[K:, K:] = phi_h_np
        return F_t

    # _predict_with_matrix is inherited from the base class and should work
    # as long as self.params is set and _get_transition_matrix is implemented.

    # smooth method is inherited and calls the base implementation.


    # --- Log-Likelihood for Parameter Optimization ---

    def log_likelihood_of_params(
        self,
        params: DFSVParamsDataclass, # Expecting JAX compatible Pytree
        observations: jnp.ndarray,
    ) -> float:
        """
        Calculate the log-likelihood for given parameters and observations.

        This method is designed for use within optimization routines (e.g., MLE).
        It leverages a JIT-compiled static helper function (`_jit_filter_scan`)
        that runs the core particle filter steps using `jax.lax.scan`.

        Args:
            params: DFSV parameters as a JAX-compatible PyTree (DFSVParamsDataclass).
                    These parameters will be used for the likelihood calculation,
                    overriding the instance's stored `self.params` for this call.
            observations: Observation data (T, N) as a JAX array.

        Returns:
            Total log-likelihood value (float).

        Raises:
            ValueError: If observations have incorrect shape or parameters are invalid.
        """
        # Input validation
        if not isinstance(params, DFSVParamsDataclass):
             raise TypeError("params must be a DFSVParamsDataclass instance.")
        if not isinstance(observations, jnp.ndarray):
             raise TypeError("observations must be a JAX array.")
        if observations.ndim != 2 or observations.shape[1] != params.N:
             raise ValueError(f"Observations shape {observations.shape} incompatible with N={params.N}")

        # Use the static, jitted helper function for the core computation
        # Pass the instance (`self`) as a static argument to access methods/attributes
        # like num_particles, seed, resample_threshold_ess etc.
        jax_log_likelihood = DFSVParticleFilter._jit_filter_scan(
            self, params, observations
        )

        # Cast the JAX scalar result to a Python float
        return float(jax_log_likelihood)

    @staticmethod
    @partial(jit, static_argnums=(0,)) # JIT compile, self_static is static
    def _jit_filter_scan(
        self_static, # Pass the instance statically
        params: DFSVParamsDataclass,
        observations: jnp.ndarray
    ) -> jnp.ndarray: # Return JAX scalar
        """
        Static, JIT-compiled function to run the particle filter scan for likelihood calculation.

        Args:
            self_static: The DFSVParticleFilter instance (passed statically).
            params: The specific DFSV parameters to use for this calculation.
            observations: The observation data (T, N).

        Returns:
            Total log-likelihood as a JAX scalar array.
        """
        T = observations.shape[0]
        N = params.N
        K = params.K

        # --- Prepare static parameters for scan body ---
        # Precompute observation noise covariance matrix R
        sigma2_curr = params.sigma2
        if sigma2_curr.ndim == 1:
            obs_noise_cov = jnp.diag(sigma2_curr)
        else: # Assume 2D
            obs_noise_cov = sigma2_curr
        # Precompute Cholesky of Q_h for this parameter set
        try:
            chol_Q_h_local = jax.scipy.linalg.cholesky(params.Q_h, lower=True)
        except jnp.linalg.LinAlgError:
             # If Q_h is not valid for these params, return -inf likelihood
             return jnp.array(-jnp.inf)


        # --- Initialize state for scan ---
        # Use instance's methods but with the provided `params` and a fresh key from seed
        rng_key = jax.random.PRNGKey(self_static.seed)
        init_rng_key, init_particles, init_log_weights = self_static.initialize_particles(
            params, rng_key # Use input params here
        )
        initial_scan_state = PFScanState(
            rng_key=init_rng_key,
            particles=init_particles,
            normalized_log_weights=init_log_weights,
            log_likelihood_accum=0.0,
        )

        # --- Define the scan body function (similar to filter, but uses input params) ---
        def scan_body_opt(state: PFScanState, obs_t: jnp.ndarray):
            """Body function for optimization scan."""
            key, current_particles, current_norm_log_weights, ll_accum = state
            observation = obs_t.reshape(-1, 1)

            # 1. Predict (using input params and local chol_Q_h)
            key, key_h, key_f = jax.random.split(key, 3)
            factors_t = current_particles[:K, :]
            log_vols_t = current_particles[K:, :]
            mu_col = params.mu.reshape(-1, 1)
            h_deviation = log_vols_t - mu_col
            h_mean_pred = mu_col + params.Phi_h @ h_deviation
            noise_h = chol_Q_h_local @ jax.random.normal(key_h, shape=(K, self_static.num_particles))
            log_vols_tp1 = h_mean_pred + noise_h
            f_mean_pred = params.Phi_f @ factors_t
            std_noise_f = jax.random.normal(key_f, shape=(K, self_static.num_particles))
            vol_scale = jnp.exp(log_vols_tp1 / 2.0)
            noise_f = vol_scale * std_noise_f
            factors_tp1 = f_mean_pred + noise_f
            predicted_particles = jnp.vstack([factors_tp1, log_vols_tp1])
            # End Predict

            # 2. Weight (using input params and precomputed obs_noise_cov)
            log_likelihood_terms = self_static.compute_log_likelihood_particle(
                predicted_particles,
                observation,
                params.lambda_r,
                obs_noise_cov, # Use precomputed R
                K,
                N,
            )
            unnormalized_log_weights = current_norm_log_weights + log_likelihood_terms
            # End Weight

            # 3. Calculate LL Increment
            ll_increment = jax.scipy.special.logsumexp(unnormalized_log_weights)
            # Handle potential -inf from likelihood terms
            ll_increment = jnp.where(jnp.isinf(ll_increment), -jnp.inf, ll_increment)
            ll_accum_next = ll_accum + ll_increment
            # End LL Increment

            # 4. Resample (using instance's resample method)
            key, particles_next, next_norm_log_weights, _ = self_static.resample_particles(
                key, predicted_particles, unnormalized_log_weights
            )
            # End Resample

            next_scan_state = PFScanState(
                rng_key=key,
                particles=particles_next,
                normalized_log_weights=next_norm_log_weights,
                log_likelihood_accum=ll_accum_next,
            )
            # We only need the final likelihood, so output is None
            return next_scan_state, None

        # --- Run the scan ---
        final_state, _ = jax.lax.scan(
            scan_body_opt, initial_scan_state, observations
        )

        # Return the final accumulated log-likelihood (as JAX scalar)
        # Handle case where accumulation resulted in NaN (e.g., from -inf + inf)
        final_ll = final_state.log_likelihood_accum
        return jnp.where(jnp.isnan(final_ll), -jnp.inf, final_ll)
