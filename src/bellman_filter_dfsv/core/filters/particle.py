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

# Precision should be controlled by the calling script, not set globally here.
# jax.config.update("jax_enable_x64", True) # REMOVED


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

    This version accepts parameters externally during the `filter` call,
    allowing the filter instance to be reused across different parameter sets,
    which is beneficial for JIT compilation in simulation studies.

    Inherits from `DFSVFilter` but overrides the `filter` method and provides
    particle-specific methods. Smoothing uses the base class RTS smoother
    with NumPy approximations, relying on parameters stored from the last
    `filter` call.

    Attributes:
        N (int): Number of observed series.
        K (int): Number of factors.
        state_dim (int): Dimension of the state vector (2 * K).
        num_particles (int): Number of particles used.
        resample_threshold_ess (float): ESS threshold for resampling (absolute value).
        seed (int): Seed for JAX PRNG initialization.
        rng_key (jax.random.PRNGKey): Current state of the JAX random number generator.
        particles (Optional[jnp.ndarray]): Final particles after filtering (state_dim, num_particles).
        weights (Optional[jnp.ndarray]): Final normalized log weights after filtering (num_particles,).
        effective_sample_size (Optional[np.ndarray]): History of ESS during filtering (T,).
        last_filter_params (Optional[DFSVParamsDataclass]): Parameters used in the most recent `filter` call.
        is_filtered (bool): Flag indicating if the filter has been run.
        is_smoothed (bool): Flag indicating if the smoother has been run.
        filtered_states (Optional[np.ndarray]): Filtered state estimates (T, state_dim).
        filtered_covs (Optional[np.ndarray]): Filtered state covariances (T, state_dim, state_dim).
        smoothed_states (Optional[np.ndarray]): Smoothed state estimates (T, state_dim).
        smoothed_covs (Optional[np.ndarray]): Smoothed state covariances (T, state_dim, state_dim).
        log_likelihood (Optional[float]): Total log-likelihood from the filter pass.
    """

    def __init__(
        self,
        N: int, # Pass N directly
        K: int, # Pass K directly
        num_particles: int = 1000,
        resample_threshold_frac: float = 0.5,
        seed: int = 42,
    ):
        """
        Initialize the Particle Filter state without storing model parameters.

        Args:
            N: Number of assets.
            K: Number of factors.
            num_particles: Number of particles.
            resample_threshold_frac: Fraction of `num_particles` below which
                resampling is triggered based on Effective Sample Size (ESS).
            seed: Seed for JAX's random number generator.
        """
        # Initialize base class with N, K
        super().__init__(N, K)

        if not isinstance(num_particles, int) or num_particles <= 0:
            raise ValueError("num_particles must be a positive integer.")
        if not 0.0 <= resample_threshold_frac <= 1.0:
            raise ValueError("resample_threshold_frac must be between 0 and 1.")

        self.num_particles: int = num_particles
        self.resample_threshold_ess: float = resample_threshold_frac * num_particles
        self.seed: int = seed

        # Initialize JAX random key
        self.rng_key: jax.random.PRNGKey = jax.random.PRNGKey(self.seed)

        # Remove parameter storage and precomputation
        # self.params: DFSVParamsDataclass = ... # REMOVED
        # self.chol_Q_h: jnp.ndarray = ... # REMOVED

        # Storage for results (optional, set after filtering)
        self.particles: jnp.ndarray | None = None
        self.weights: jnp.ndarray | None = None # Stores normalized log weights
        self.effective_sample_size: np.ndarray | None = None # Stored as NumPy array
        self.last_filter_params: DFSVParamsDataclass | None = None # Store params used in last filter call

    def _ensure_params_are_jax(self, params: DFSVParamsDataclass) -> DFSVParamsDataclass:
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

        # Convert relevant fields to JAX arrays, ensuring correct dtype (e.g., float32)
        default_dtype = jnp.float32 # Use float32 for potential speedup
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
        Initialize particle states and weights using the prior distribution derived
        from the provided parameters.

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

        # Initialize weights to uniform normalized log weights (using float32)
        log_weights_normalized = jnp.full(
            self.num_particles, -jnp.log(self.num_particles), dtype=jnp.float32
        )

        return rng_key, particles, log_weights_normalized

    # No JIT here, relies on JIT of scan_body
    def predict_particles(
        self,
        rng_key: jax.random.PRNGKey,
        particles: jnp.ndarray,
        params: DFSVParamsDataclass, # Pass params
        chol_Q_h: jnp.ndarray        # Pass Cholesky of Q_h
    ) -> Tuple[jax.random.PRNGKey, jnp.ndarray]:
        """
        Propagate particles one step forward using the state transition dynamics
        defined by the provided parameters.

        Args:
            rng_key: JAX random key.
            particles: Current particles (state_dim, num_particles).
            params: DFSV model parameters for this step.
            chol_Q_h: Cholesky decomposition of Q_h from the provided params.

        Returns:
            Tuple containing:
                - Updated JAX random key.
                - Predicted particles (state_dim, num_particles).
        """
        K = self.K # K is from self (instance dimension)

        # Split key for factor and volatility noise
        rng_key, key_h, key_f = jax.random.split(rng_key, 3)

        # Extract state components
        factors_t = particles[:K, :]
        log_vols_t = particles[K:, :]

        # Ensure mu is a column vector for broadcasting (use passed params)
        mu_col = params.mu.reshape(-1, 1)

        # 1. Predict log-volatilities: h_{t+1} = mu + Phi_h * (h_t - mu) + eta_t
        h_deviation = log_vols_t - mu_col
        h_mean_pred = mu_col + params.Phi_h @ h_deviation
        # Sample volatility noise: eta_t ~ N(0, Q_h) using passed Cholesky
        noise_h = chol_Q_h @ jax.random.normal(
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
        obs_noise_variances: jnp.ndarray,# (N,) - Diagonal elements of R_t
        K: int,
        N: int,
    ) -> jnp.ndarray:                     # (P,)
        """
        Compute the log-likelihood log p(y_t | x_t) for each particle x_t,
        assuming diagonal observation noise covariance R_t = diag(obs_noise_variances).

        Uses the observation equation: y_t = lambda_r @ f_t + epsilon_t
        where epsilon_t ~ N(0, R_t).

        Args:
            particles: Predicted particles x_t (state_dim, num_particles).
            observation: Current observation y_t (N, 1).
            factor_loadings: Factor loading matrix lambda_r (N, K).
            obs_noise_variances: Diagonal elements (variances) of the observation
                noise covariance matrix R_t (N,).
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

        # Pre-calculate log determinant of R_t = diag(variances)
        # Add epsilon for numerical stability if variances are near zero
        safe_variances = jnp.maximum(obs_noise_variances, 1e-10)
        log_det_R = jnp.sum(jnp.log(safe_variances)) # log|R| = sum(log(variances))

        # --- Vectorized Calculation ---
        # Quadratic form: sum_i (error_i^2 / variance_i) for each particle
        # observation_error is (N, P), safe_variances is (N,)
        # Use broadcasting: safe_variances[:, None] becomes (N, 1)
        quad_form = jnp.sum((observation_error ** 2) / safe_variances[:, None], axis=0) # Sum over N -> shape (P,)

        # Log likelihood: -0.5 * (N*log(2pi) + log_det_R + quad_form)
        log2pi = jnp.log(jnp.array(2 * jnp.pi, dtype=jnp.float32))
        log_likelihoods = -0.5 * (N * log2pi + log_det_R + quad_form)

        # Ensure correct dtype
        return log_likelihoods.astype(jnp.float32)

    # Removed _compute_logprob_cholesky static method as it's replaced by direct calculation

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
            # Reset weights to uniform normalized log weights after resampling (using float32)
            resampled_log_weights = jnp.full(n, -jnp.log(n), dtype=jnp.float32)
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
        operand = (rng_key, particles, normalized_weights_linear, normalized_log_weights)
        rng_key, next_particles, next_normalized_log_weights = jax.lax.cond(
            needs_resampling,
            _resample_branch,
            _no_resample_branch,
            operand
        )

        return rng_key, next_particles, next_normalized_log_weights, ess

    def filter(
        self, params: DFSVParamsDataclass, observations: Union[np.ndarray, jnp.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the Particle Filter using JAX scan with externally provided parameters.

        Args:
            params: DFSV model parameters for this specific filter run.
            observations: Observed returns with shape (T, N) or (N, T).

        Returns:
            Tuple containing:
                - Filtered states (weighted mean estimate) (T, state_dim) as NumPy array.
                - Filtered state covariances (weighted estimate) (T, state_dim, state_dim) as NumPy array.
                - Total log-likelihood (float).

        Raises:
            ValueError: If observations or params are not provided or have incorrect shape/type.
            TypeError: If params is not a DFSVParamsDataclass.
        """
        if observations is None:
            raise ValueError("Observations must be provided.")
        if not isinstance(params, DFSVParamsDataclass):
             raise TypeError(f"Input params must be a DFSVParamsDataclass, got {type(params)}")
        if params.N != self.N or params.K != self.K:
             raise ValueError(f"Parameter dimensions (N={params.N}, K={params.K}) do not match filter dimensions (N={self.N}, K={self.K})")


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

        # --- Prepare parameters and derived values for scan body ---
        jax_params = self._ensure_params_are_jax(params)
        K = self.K # From self
        N = self.N # From self

        # Compute Cholesky of Q_h from passed params
        try:
            chol_Q_h = jax.scipy.linalg.cholesky(jax_params.Q_h, lower=True)
        except jnp.linalg.LinAlgError as e:
            raise ValueError(
                "Q_h matrix from passed params must be positive definite."
            ) from e

        # Compute observation noise variances (diagonal of R) from passed params
        sigma2_curr = jax_params.sigma2
        if sigma2_curr.ndim == 1:
            if sigma2_curr.shape[0] != N:
                raise ValueError(f"sigma2 (1D) length {sigma2_curr.shape[0]} != N ({N})")
            obs_noise_variances = sigma2_curr # Assume it's already the variances
        elif sigma2_curr.ndim == 2:
            if sigma2_curr.shape != (N, N):
                raise ValueError(f"sigma2 (2D) shape {sigma2_curr.shape} != ({N}, {N})")
            # Ensure it's diagonal and extract variances
            if not jnp.allclose(sigma2_curr, jnp.diag(jnp.diag(sigma2_curr))):
                warnings.warn("sigma2 is 2D but not diagonal. Extracting diagonal for particle filter likelihood.")
            obs_noise_variances = jnp.diag(sigma2_curr)
        else:
            raise ValueError(f"sigma2 has invalid shape {sigma2_curr.shape}")

        # Extract factor loadings
        factor_loadings = jax_params.lambda_r # Shape (N, K)

        # --- Initialize state for scan ---
        # Use the instance's current rng_key, but pass params explicitly
        init_rng_key, init_particles, init_log_weights = self.initialize_particles(
            jax_params, self.rng_key
        )
        initial_scan_state = PFScanState(
            rng_key=init_rng_key,
            particles=init_particles,
            normalized_log_weights=init_log_weights,
            log_likelihood_accum=0.0,
        )

        # --- Define the scan body function (partially applied with params) ---
        # Note: JIT is applied to the partially applied scan_step below
        def scan_body(
            self_static, # Pass self statically for methods like resample
            params_static: DFSVParamsDataclass, # Pass params statically for this run
            chol_Q_h_static: jnp.ndarray,       # Pass chol_Q_h statically for this run
            obs_noise_variances: jnp.ndarray,   # Pass obs_noise_variances statically
            K_static: int,                      # Pass K statically
            N_static: int,                      # Pass N statically
            state: PFScanState,                 # The dynamic state
            obs_t: jnp.ndarray                  # The dynamic observation
            ):
            """Body function for jax.lax.scan, performs one filter step."""
            key, current_particles, current_norm_log_weights, ll_accum = state
            observation = obs_t.reshape(-1, 1) # Ensure (N, 1)

            # 1. Predict (pass params and chol_Q_h)
            key, predicted_particles = self_static.predict_particles(
                key, current_particles, params_static, chol_Q_h_static
            )

            # 2. Weight (pass factor_loadings and obs_noise_variances)
            log_likelihood_terms = self_static.compute_log_likelihood_particle(
                predicted_particles,
                observation,
                params_static.lambda_r, # Get factor loadings from params
                obs_noise_variances,    # Use passed obs_noise_variances (matches signature)
                K_static,
                N_static,
            )
            # Calculate unnormalized weights for resampling and LL contribution
            unnormalized_log_weights = current_norm_log_weights + log_likelihood_terms

            # 3. Calculate Log-Likelihood Contribution for this step
            ll_increment = jax.scipy.special.logsumexp(unnormalized_log_weights)
            ll_accum_next = ll_accum + ll_increment

            # 4. Resample (uses self_static for threshold etc.)
            key, particles_next, next_norm_log_weights, ess = self_static.resample_particles(
                key, predicted_particles, unnormalized_log_weights
            )

            # --- Calculate outputs for storage ---
            # Weighted mean estimate of state x_t (using next particles/weights)
            weights_linear = jnp.exp(next_norm_log_weights)
            filtered_mean = jnp.sum(particles_next * weights_linear, axis=1)

            # Weighted covariance estimate of state x_t
            diff = particles_next - filtered_mean.reshape(-1, 1) # Shape (state_dim, P)
            # Alternative calculation using matrix multiplication:
            # weighted_diff = diff * sqrt(weights_linear[None, :]) # Or just weights? Check derivation.
            # Let's use the direct weighting: diff * weights
            weighted_diff = diff * weights_linear[None, :] # Shape (state_dim, P)
            # filtered_cov = weighted_diff @ diff.T # (state_dim, P) @ (P, state_dim) -> (state_dim, state_dim)
            # This computes Sum_p [ w_p * diff_jp * diff_kp ] which is correct.
            filtered_cov = weighted_diff @ diff.T
            # Ensure symmetry (might still be needed due to precision)
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
        # --- End of scan_body definition ---

        # --- Create the function to be scanned ---
        # Partially apply the static arguments to scan_body and JIT the result
        scan_step = jit(partial(
            scan_body,
            self, # Pass self statically
            jax_params, # Pass params statically for this run
            chol_Q_h, # Pass chol_Q_h statically for this run
            obs_noise_variances, # Pass obs_noise_variances statically for this run
            K, # Pass K statically
            N, # Pass N statically
        ))

        # --- Run the scan ---
        final_state, scan_outputs = jax.lax.scan(
            scan_step, # Use the partially applied and JITted function
            initial_scan_state,
            obs_jax
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
        self.last_filter_params = jax_params # Store params used for this filter run

        return self.filtered_states, self.filtered_covs, self.log_likelihood


    # --- Methods for RTS Smoothing ---
    def _get_transition_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        Get the linearized state transition matrix F_t (NumPy version for smoother).

        Uses the parameters stored from the last filter run (`self.last_filter_params`).

        Args:
            state: Current state estimate (state_dim, 1) as NumPy array.
                   Note: This argument is not used in the standard DFSV linearization
                   where F_t only depends on parameters, but included for API consistency.

        Returns:
            Linearized transition matrix F_t (state_dim, state_dim) as NumPy array.

        Raises:
            RuntimeError: If filter has not been run or params were not stored.
        """
        if self.last_filter_params is None:
            raise RuntimeError("Filter must be run successfully before smoothing to store parameters.")

        K = self.K
        # Convert necessary params from JAX to NumPy for the smoother
        jax_params = self._ensure_params_are_jax(self.last_filter_params)
        phi_f_np = np.array(jax_params.Phi_f)
        phi_h_np = np.array(jax_params.Phi_h)

        F_t = np.zeros((self.state_dim, self.state_dim))
        F_t[:K, :K] = phi_f_np
        F_t[K:, K:] = phi_h_np
        return F_t

    def _predict_with_matrix(
        self, state: np.ndarray, cov: np.ndarray, transition_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict state and covariance using a given transition matrix (NumPy version).

        Overrides the base class method to use parameters stored from the last
        filter run (`self.last_filter_params`). Needed for the RTS smoother.

        Args:
            state: Current state estimate x_t (state_dim, 1) as NumPy array.
            cov: Current covariance P_t (state_dim, state_dim) as NumPy array.
            transition_matrix: Linearized state transition matrix F_t (NumPy array).

        Returns:
            Tuple containing:
                - Predicted state mean x_{t+1|t} (state_dim, 1) as NumPy array.
                - Predicted covariance P_{t+1|t} (state_dim, state_dim) as NumPy array.

        Raises:
            RuntimeError: If filter has not been run or params were not stored.
        """
        if self.last_filter_params is None:
            raise RuntimeError("Filter must be run successfully before smoothing to store parameters.")

        # Ensure params are JAX arrays first
        jax_params = self._ensure_params_are_jax(self.last_filter_params)

        K = self.K
        # Ensure state is a column vector
        state_col = state.reshape(-1, 1)

        # Use NumPy versions of parameters from stored params
        mu_np = np.array(jax_params.mu).reshape(-1, 1)
        q_h_np = np.array(jax_params.Q_h)

        # Predict state mean E[x_{t+1}|t] using the non-linear dynamics approximated by F_t
        # x_{t+1|t} = F_t @ x_t (This is often an approximation, the exact form depends on the model)
        # For DFSV:
        factors = state_col[:K, :]
        log_vols = state_col[K:, :]
        pred_factors_mean = transition_matrix[:K, :K] @ factors
        # h_{t+1} = mu + Phi_h @ (h_t - mu) + noise
        pred_log_vols_mean = mu_np + transition_matrix[K:, K:] @ (log_vols - mu_np)
        predicted_state_mean_col = np.vstack([pred_factors_mean, pred_log_vols_mean])

        # Predict covariance: P_{t+1|t} = F_t @ P_t @ F_t^T + Q_t
        # Process noise Q_t = Cov([factor_noise; h_noise])
        # Q_f = diag(exp(h_t)), Q_h = params.Q_h (state-dependent)
        Q_t = np.zeros((self.state_dim, self.state_dim))
        Q_t[K:, K:] = q_h_np
        # Approximate factor noise cov using current log_vols estimate h_t
        current_log_vols = state_col[K:, :].flatten()
        # Use state_col which is h_t
        Q_t[:K, :K] = np.diag(np.exp(current_log_vols))

        predicted_cov = transition_matrix @ cov @ transition_matrix.T + Q_t
        # Ensure symmetry
        predicted_cov = (predicted_cov + predicted_cov.T) / 2.0

        return predicted_state_mean_col, predicted_cov


    # --- Log-Likelihood for Parameter Optimization ---
    # NOTE: The log_likelihood_of_params and _jit_filter_scan methods below
    #       are designed for optimization where params are passed externally.
    #       They correctly use the passed params, not self.last_filter_params.

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
                    These parameters will be used for the likelihood calculation.
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

        # Return the JAX scalar array directly
        return jax_log_likelihood

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
        # Precompute observation noise variances (diagonal of R)
        sigma2_curr = params.sigma2
        if sigma2_curr.ndim == 1:
            if sigma2_curr.shape[0] != N:
                # This check might be redundant if params validation is done elsewhere,
                # but kept for safety within the JITted function context.
                return jnp.array(-jnp.inf) # Invalid params -> low likelihood
            obs_noise_variances = sigma2_curr
        elif sigma2_curr.ndim == 2:
            if sigma2_curr.shape != (N, N):
                return jnp.array(-jnp.inf) # Invalid params
            # Assuming diagonal, extract variances
            obs_noise_variances = jnp.diag(sigma2_curr)
            # Optional: Add check if it's truly diagonal? Might be slow inside JIT.
        else:
            return jnp.array(-jnp.inf) # Invalid params

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

            # 2. Weight (using input params and precomputed obs_noise_variances)
            log_likelihood_terms = self_static.compute_log_likelihood_particle(
                predicted_particles,
                observation,
                params.lambda_r,
                obs_noise_variances, # Use precomputed variances
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

    # --- Getters for Filtered/Smoothed Results ---
    # These should work as they read from stored results (self.filtered_states etc.)
    # which are set at the end of the filter method.

    def get_filtered_factors(self) -> np.ndarray:
        """Return the filtered latent factors."""
        if not self.is_filtered or self.filtered_states is None:
            raise RuntimeError("Must run filter first.")
        return self.filtered_states[:, : self.K]

    def get_filtered_volatilities(self) -> np.ndarray:
        """Return the filtered log-volatilities."""
        if not self.is_filtered or self.filtered_states is None:
            raise RuntimeError("Must run filter first.")
        return self.filtered_states[:, self.K :]

    def get_smoothed_factors(self) -> np.ndarray:
        """Return the smoothed latent factors."""
        if not self.is_smoothed or self.smoothed_states is None:
            raise RuntimeError("Must run smoother first.")
        return self.smoothed_states[:, : self.K]

    def get_smoothed_volatilities(self) -> np.ndarray:
        """Return the smoothed log-volatilities."""
        if not self.is_smoothed or self.smoothed_states is None:
            raise RuntimeError("Must run smoother first.")
        return self.smoothed_states[:, self.K :]
