"""
Nonlinear Filter implementations for Dynamic Factor Stochastic Volatility models.

This module provides filter classes for state estimation in DFSV models,
including Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), Particle Filters (PF) and Bellman Filters (BF)
to handle the nonlinearities introduced by stochastic volatility.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import jax.scipy.optimize
import numpy as np
from jax import jit
from functools import partial
import jax.debug

from functions.simulation import DFSV_params
from models.dfsv import DFSVParamsDataclass


class DFSVFilter:
    """
    Base class for filters applied to Dynamic Factor Stochastic Volatility models.

    This class provides the foundation for implementing various Kalman filter
    variants (EKF, UKF) for DFSV models, handling the joint filtering of
    latent factors and log-volatilities.
    """

    def __init__(self, N, K):
        """
        Initialize the Kalman filter with static parameters N and K.

        Parameters
        ----------
        N : int
            Number of observed series
        K : int
            Number of factors
        """
        self.N = N  # Number of observed series
        self.K = K  # Number of factors

        # State dimension is 2*K (K factors + K log-volatilities)
        self.state_dim = 2 * self.K

        # Flag to track if filter has been run
        self.is_filtered = False
        self.is_smoothed = False

        # Storage for filtered and smoothed states
        self.filtered_states = None
        self.filtered_covs = None
        self.smoothed_states = None
        self.smoothed_covs = None
        self.log_likelihood = None

    def initialize_state(self, params) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize state vector and covariance matrix.

        Parameters
        ----------
        params : DFSV_params
            Parameters of the DFSV model

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Initial state vector with shape (state_dim, 1) and
            initial state covariance matrix with shape (state_dim, state_dim)
        """
        # Initialize factors to zero
        K = self.K
        initial_factors = jnp.zeros((K, 1))

        # Initialize log-volatilities to the unconditional mean
        if params.mu.ndim == 1:
            initial_log_vols = params.mu.reshape(-1, 1)
        else:
            initial_log_vols = params.mu.copy()

        # Combine into state vector [f; h]
        initial_state = jnp.vstack([initial_factors, initial_log_vols])

        # Initialize factor covariance
        P_f = jnp.eye(K)

        # Solve discrete Lyapunov equation using JAX
        @jit
        def solve_discrete_lyapunov_jax(Phi, Q, num_iters=30):
            """Solve P = Phi @ P @ Phi.T + Q using iteration."""
            P = Q
            for _ in range(num_iters):
                P = Phi @ P @ Phi.T + Q
            # Ensure symmetry
            return (P + P.T) / 2.0

        # Ensure inputs are JAX arrays
        Phi_h_jax = jnp.asarray(params.Phi_h)
        Q_h_jax = jnp.asarray(params.Q_h)
        P_h = solve_discrete_lyapunov_jax(Phi_h_jax, Q_h_jax)

        # Construct block-diagonal covariance
        initial_cov = jnp.block([[P_f, jnp.zeros((K, K))], [jnp.zeros((K, K)), P_h]])

        return initial_state, initial_cov

    def filter(self, params, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the Kalman filter on the provided data.

        Parameters
        ----------
        params : DFSV_params
            Parameters of the DFSV model
        y : np.ndarray
            Observed returns with shape (T, N) or (N, T)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            Filtered states with shape (T, state_dim),
            filtered covariances with shape (T, state_dim, state_dim),
            and total log-likelihood
        """
        # Import tqdm for progress bar
        try:
            from tqdm import tqdm
        except ImportError:
            # If tqdm is not installed, create a simple pass-through iterator
            def tqdm(iterable, **kwargs):
                return iterable

            print("Warning: tqdm not installed. No progress bar will be shown.")
        # Convert y to jax array
        y = jnp.array(y)
        # Convert params to jax arrays
        if hasattr(params, "Phi_f"):
            params.Phi_f = jnp.array(params.Phi_f)
        if hasattr(params, "Phi_h"):
            params.Phi_h = jnp.array(params.Phi_h)
        if hasattr(params, "lambda_r"):
            params.lambda_r = jnp.array(params.lambda_r)
        if hasattr(params, "Q_h"):
            params.Q_h = jnp.array(params.Q_h)
        if hasattr(params, "mu"):
            params.mu = jnp.array(params.mu)
        if hasattr(params, "sigma2"):
            params.sigma2 = jnp.array(params.sigma2)
        # Ensure y is in (T, N) format
        if y.shape[0] < y.shape[1]:  # If (N, T) format
            y = y.T

        T = y.shape[0]

        # Storage for filtered states and covariances
        filtered_states = jnp.zeros((T, self.state_dim))
        filtered_covs = jnp.zeros((T, self.state_dim, self.state_dim))

        # Initialize
        state, cov = self.initialize_state(params)
        log_likelihood = 0.0

        # Forward pass
        for t in tqdm(range(T), desc="Filter Progress"):
            # Prediction step
            predicted_state, predicted_cov = self.predict(params, state, cov)

            # Update step - reshape observation to (N, 1)
            observation = y[t : t + 1, :].T.reshape(-1, 1)
            state, cov, ll_contrib = self.update(
                params, predicted_state, predicted_cov, observation
            )

            # Store results (convert state from (state_dim, 1) to row vector)
            filtered_states = filtered_states.at[t, :].set(state.flatten())
            filtered_covs = filtered_covs.at[t, :, :].set(cov)
            log_likelihood += ll_contrib

        # Store results in object
        self.filtered_states = filtered_states
        self.filtered_covs = filtered_covs
        self.log_likelihood = log_likelihood
        self.is_filtered = True

        # Convert filtered states and covariances to numpy arrays
        filtered_states = np.array(filtered_states)
        filtered_covs = np.array(filtered_covs)
        return filtered_states, filtered_covs, log_likelihood

    def predict(
        self, params, state: np.ndarray, cov: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the prediction step.

        Parameters
        ----------
        params : DFSV_params
            Parameters of the DFSV model
        state : np.ndarray
            Current state estimate with shape (state_dim, 1)
        cov : np.ndarray
            Current state covariance with shape (state_dim, state_dim)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Predicted state and covariance
        """
        raise NotImplementedError("Predict method must be implemented by subclasses")

    def update(
        self,
        params,
        predicted_state: np.ndarray,
        predicted_cov: np.ndarray,
        observation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform the update step.

        Parameters
        ----------
        params : DFSV_params
            Parameters of the DFSV model
        predicted_state : np.ndarray
            Predicted state with shape (state_dim, 1)
        predicted_cov : np.ndarray
            Predicted state covariance with shape (state_dim, state_dim)
        observation : np.ndarray
            Current observation with shape (N, 1)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            Updated state, updated covariance, and log-likelihood contribution
        """
        raise NotImplementedError("Update method must be implemented by subclasses")

    def smooth(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the smoother, must be called after filter().

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Smoothed states and covariances
        """
        if not self.is_filtered:
            raise RuntimeError("Must run filter before smoothing")

        T = self.filtered_states.shape[1]

        # Storage for smoothed states and covariances
        smoothed_states = np.zeros_like(self.filtered_states)
        smoothed_covs = np.zeros_like(self.filtered_covs)

        # Initialize with last filtered values
        last_t = T - 1
        smoothed_states[last_t : last_t + 1, :] = self.filtered_states[
            last_t : last_t + 1, :
        ]
        smoothed_covs[last_t, :, :] = self.filtered_covs[last_t, :, :]

        # Backward pass
        for t in range(T - 2, -1, -1):
            # Get filtered values at time t
            state_t = self.filtered_states[t : t + 1, :]
            cov_t = self.filtered_covs[t, :, :]

            # Get the transition matrix at time t (model specific)
            F_t = self._get_transition_matrix(state_t)

            # Predict from t to t+1
            pred_state, pred_cov = self._predict_with_matrix(state_t, cov_t, F_t)

            # Compute smoother gain
            K_t = cov_t @ F_t.T @ np.linalg.inv(pred_cov)

            # Update based on smoothed value at t+1
            smoothed_states[t : t + 1, :] = (
                state_t + (smoothed_states[t + 1 : t + 2, :] - pred_state) @ K_t
            )
            smoothed_covs[t, :, :] = (
                cov_t + K_t @ (smoothed_covs[t + 1, :, :] - pred_cov) @ K_t.T
            )

        self.smoothed_states = smoothed_states
        self.smoothed_covs = smoothed_covs
        self.is_smoothed = True

        return smoothed_states, smoothed_covs

    def _get_transition_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        Get the linearized state transition matrix at given state.
        Must be implemented by subclasses.

        Parameters
        ----------
        state : np.ndarray
            Current state

        Returns
        -------
        np.ndarray
            Transition matrix with shape (state_dim, state_dim)
        """
        raise NotImplementedError("Must be implemented by subclasses")

    def _predict_with_matrix(
        self, state: np.ndarray, cov: np.ndarray, transition_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make a prediction using the provided transition matrix.

        Parameters
        ----------
        state : np.ndarray
            Current state estimate x_t (NumPy array, shape (1, state_dim) expected from base smoother)
        cov : np.ndarray
            Current covariance P_t (NumPy array, shape (state_dim, state_dim))
        transition_matrix : np.ndarray
            Linearized state transition matrix F_t (NumPy array)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Predicted state mean x_{t+1|t} shape (1, state_dim) and covariance P_{t+1|t} shape (state_dim, state_dim)
        """
        K = self.K
        # Transpose input state (1, state_dim) to (state_dim, 1) for internal calculations
        state_col = state.T

        # Convert params back to NumPy temporarily
        mu_np = np.array(self.params.mu).reshape(-1, 1)
        q_h_np = np.array(self.params.Q_h)

        # Predict state mean E[x_{t+1}|t] using the non-linear dynamics
        factors = state_col[:K, :]    # Use column vector
        log_vols = state_col[K:, :]   # Use column vector

        pred_factors_mean = transition_matrix[:K, :K] @ factors # Linear part: Phi_f @ f_t
        pred_log_vols_mean = mu_np + transition_matrix[K:, K:] @ (log_vols - mu_np) # h_{t+1} = mu + Phi_h(h_t - mu)
        predicted_state_mean_col = np.vstack([pred_factors_mean, pred_log_vols_mean]) # Shape (state_dim, 1)

        # Predict covariance: P_{t+1|t} = F_t P_t F_t^T + Q_t
        # Process noise Q_t = Cov([f_noise; h_noise])
        # Q_f = diag(exp(h_t)), Q_h = params.Q_h
        Q_t = np.zeros((self.state_dim, self.state_dim))
        Q_t[K:, K:] = q_h_np
        # Approximate factor noise cov using current log_vols estimate h_t
        current_log_vols = state_col[K:, :].flatten() # Use column vector
        Q_t[:K, :K] = np.diag(np.exp(current_log_vols)) # Factor process noise variance depends on h_t

        predicted_cov = transition_matrix @ cov @ transition_matrix.T + Q_t

        # Return mean reshaped to (1, state_dim) for compatibility with base smoother
        return predicted_state_mean_col.T, predicted_cov

    def get_filtered_factors(self) -> np.ndarray:
        """
        Return the filtered latent factors.

        Returns
        -------
        np.ndarray
            Filtered factors with shape (T, K)
        """
        if not self.is_filtered:
            raise RuntimeError("Must run filter first")

        # First K columns are the factors
        return self.filtered_states[:, : self.K]

    def get_filtered_volatilities(self) -> np.ndarray:
        """
        Return the filtered log-volatilities.

        Returns
        -------
        np.ndarray
            Filtered log-volatilities with shape (KT K)
        """
        if not self.is_filtered:
            raise RuntimeError("Must run filter first")

        # Last K rows are the log-volatilities
        return self.filtered_states[:, self.K :]

    def get_smoothed_factors(self) -> np.ndarray:
        """
        Return the smoothed latent factors.

        Returns
        -------
        np.ndarray
            Smoothed factors with shape (T, K)
        """
        if not self.is_smoothed:
            raise RuntimeError("Must run smoother first")

        # First K rows are the factors
        return self.smoothed_states[:, : self.K]

    def get_smoothed_volatilities(self) -> np.ndarray:
        """
        Return the smoothed log-volatilities.

        Returns
        -------
        np.ndarray
            Smoothed log-volatilities with shape (T, K)
        """
        if not self.is_smoothed:
            raise RuntimeError("Must run smoother first")

        # Last K rows are the log-volatilities
        return self.smoothed_states[:, : self.K]


class DFSVParticleFilter(DFSVFilter):
    """
    Particle Filter implementation for DFSV models using JAX.

    This class implements a particle filter (bootstrap filter) for state
    estimation in DFSV models, using Sequential Importance Sampling with
    Resampling (SISR) to handle the nonlinearities in the DFSV model.
    Uses JAX for computations.
    """

    def __init__(
        self,
        params: DFSV_params | DFSVParamsDataclass, # Allow both types
        num_particles: int = 1000,
        resample_threshold: float = 0.5,
        seed: int = 42,  # Add seed for reproducibility
    ):
        """
        Initialize the Particle Filter with DFSV model parameters.

        Args:
            params: Parameters of the DFSV model (either old DFSV_params class
                    or new DFSVParamsDataclass). Will be converted to
                    DFSVParamsDataclass with JAX arrays internally.
            num_particles: Number of particles to use. Default is 1000.
            resample_threshold: Effective sample size threshold as a fraction of
                num_particles, below which resampling is triggered. Default is 0.5.
            seed: Seed for JAX's random number generator. Default is 42.
        """
        # Get N and K from input params regardless of type
        N = params.N
        K = params.K
        super().__init__(N, K)
        self.num_particles = num_particles
        self.resample_threshold = resample_threshold
        self._seed = seed # Store the seed

        # Enable 64-bit precision for JAX
        jax.config.update("jax_enable_x64", True)

        # Convert input params to JAX DATACLASS and store
        self.params: DFSVParamsDataclass = self._params_to_jax(params)

        # Storage for particles and weights (will be JAX arrays during filtering)
        self.particles = None
        self.weights = None # Stores normalized log weights
        self.effective_sample_size = None # Stored as NumPy array

        # Initialize JAX random key
        self.key = jax.random.PRNGKey(seed)

        # Initialize Cholesky decomposition of Q_h (as JAX array)
        # This assumes Q_h in initial params is fixed for the filter run
        self.chol_Q_h = jax.scipy.linalg.cholesky(self.params.Q_h, lower=True)


    def _params_to_jax(self, params: DFSV_params | DFSVParamsDataclass) -> DFSVParamsDataclass:
        """Converts NumPy arrays in params object to a JAX DFSVParamsDataclass.

        Args:
            params: DFSV parameters (either old class or new dataclass).

        Returns:
            DFSVParamsDataclass: DFSV parameters as a dataclass with JAX arrays.
        """
        # Extract attributes if it's the old class, else convert dataclass to dict
        if isinstance(params, DFSV_params) and not isinstance(params, DFSVParamsDataclass):
            # If it's the old class, extract attributes
            params_dict = {
                "N": params.N, "K": params.K, "lambda_r": params.lambda_r,
                "Phi_f": params.Phi_f, "Phi_h": params.Phi_h, "mu": params.mu,
                "sigma2": params.sigma2, "Q_h": params.Q_h
            }
        elif isinstance(params, DFSVParamsDataclass):
            # If it's already the dataclass, convert to dict
            from dataclasses import asdict
            params_dict = asdict(params)
        else:
            raise TypeError(f"Unsupported parameter type: {type(params)}")

        # Convert relevant fields to JAX arrays
        jax_params_dict = {k: jnp.array(v) if isinstance(v, (np.ndarray, list)) else v
                           for k, v in params_dict.items()
                           if k in ["lambda_r", "Phi_f", "Phi_h", "mu", "sigma2", "Q_h"]}

        # Create the JAX dataclass instance
        jax_dataclass_params = DFSVParamsDataclass(
            N=params_dict["N"],
            K=params_dict["K"],
            **jax_params_dict
        )
        return jax_dataclass_params

    def initialize_particles(
        self, params, key: jax.random.PRNGKey
    ) -> Tuple[jax.random.PRNGKey, jnp.ndarray, jnp.ndarray]:
        """
        Initialize particles and weights using JAX.

        Parameters
        ----------
        params : DFSV_params
            Parameters of the DFSV model (JAX arrays)
        key : jax.random.PRNGKey
            JAX random key.

        Returns
        -------
        Tuple[jax.random.PRNGKey, jnp.ndarray, jnp.ndarray]
            Updated key,
            Initialized particles with shape (state_dim, num_particles),
            and uniform normalized log weights with shape (num_particles,)
        """
        # Get initial state and covariance (using base class, returns JAX arrays)
        initial_state, initial_cov = super().initialize_state(params)

        # Split key
        key, subkey = jax.random.split(key)

        # Initialize particles by sampling from initial distribution N(initial_state, initial_cov)
        L = jax.scipy.linalg.cholesky(initial_cov, lower=True)

        # Generate num_particles samples
        noise = jax.random.normal(subkey, shape=(self.state_dim, self.num_particles))
        particles = initial_state.reshape(-1, 1) + L @ noise

        # Initialize weights to uniform normalized log weights
        log_weights = jnp.full(self.num_particles, -jnp.log(self.num_particles))

        return key, particles, log_weights

    @partial(jit, static_argnums=(0,))
    def predict_particles(self, key: jax.random.PRNGKey, particles: jnp.ndarray) -> Tuple[jax.random.PRNGKey, jnp.ndarray]:
        """
        Propagate particles through state transition equation using JAX.

        Parameters
        ----------
        key : jax.random.PRNGKey
            JAX random key.
        particles : jnp.ndarray
            Current particles with shape (state_dim, num_particles)

        Returns
        -------
        Tuple[jax.random.PRNGKey, jnp.ndarray]
            Updated key, Predicted particles with shape (state_dim, num_particles)
        """
        K = self.K
        params = self.params # Use stored JAX params

        # Split key for different noise sources
        key, key_h, key_f = jax.random.split(key, 3)

        # Extract state components
        factors = particles[:K, :]
        log_vols = particles[K:, :]

        # Make sure mu has the right shape for broadcasting
        mu = params.mu.reshape(-1, 1) # Already a JAX array

        # Predict log-volatilities: h_{t+1} = mu + Phi_h * (h_t - mu) + eta_t
        h_deviation = log_vols - mu
        h_evolution = mu + params.Phi_h @ h_deviation

        # Add noise: eta_t ~ N(0, Q_h)
        noise_h = self.chol_Q_h @ jax.random.normal(key_h, shape=(K, self.num_particles))
        predicted_log_vols = h_evolution + noise_h

        # Predict factors: f_{t+1} = Phi_f * f_t + diag(exp(h_{t+1}/2)) * eps_t
        # Use predicted log_vols for noise generation as per original logic
        f_evolution = params.Phi_f @ factors

        # Add state-dependent noise: eps_t ~ N(0, I) scaled by exp(h_{t+1}/2)
        std_noise = jax.random.normal(key_f, shape=(K, self.num_particles))
        # Use (potentially unclipped) predicted vols
        vol_scale = jnp.exp(predicted_log_vols / 2.0)
        factor_noise = vol_scale * std_noise
        predicted_factors = f_evolution + factor_noise

        # Combine into predicted particles
        # Use the (potentially unclipped) factors and log-vols
        predicted_particles = jnp.vstack([predicted_factors, predicted_log_vols])

        return key, predicted_particles

    @partial(jit, static_argnums=(0, 5, 6)) # Mark K (5) and N (6) as static
    def compute_log_likelihood_particle(
        self,
        particles: jnp.ndarray,
        observation: jnp.ndarray,
        lambda_r: jnp.ndarray,    # Expect lambda_r
        sigma2_matrix: jnp.ndarray, # Expect (N, N) matrix
        K: int,                   # Expect K
        N: int                    # Expect N
    ) -> jnp.ndarray:
        """Computes log p(y_t | x_t) for each particle x_t, using a unified Cholesky approach."""
        factors = particles[:K, :]
        pred_obs = lambda_r @ factors
        diff = observation - pred_obs # Shape (N, P)

        # Vmap the single-particle likelihood calculation over the particle dimension (axis=1) of diff
        # _compute_logprob_particle_cholesky expects diff (N,), sigma2_matrix (N, N), N (static)
        log_likelihoods = jax.vmap(
            DFSVParticleFilter._compute_logprob_particle_cholesky,
            in_axes=(1, None, None), # Map over axis 1 of diff, broadcast sigma2_matrix and N
            out_axes=0
        )(diff, sigma2_matrix, N) # Result shape (P,)

        return log_likelihoods

    # --- Static Helper Method for Unified Log-Likelihood Computation --- #

    @staticmethod
    @partial(jit, static_argnums=(2,)) # N is static
    def _compute_logprob_particle_cholesky(particle_diff, sigma2_matrix, N): # type: ignore
        """Computes log prob for a single particle's diff (N,) using Cholesky of sigma2_matrix (N,N)."""
        try:
            # Cholesky decomposition L of Sigma = L L^T
            L = jax.scipy.linalg.cholesky(sigma2_matrix, lower=True)
            # Solve L x = diff for x
            x = jax.scipy.linalg.solve_triangular(L, particle_diff, lower=True)
            # Quadratic form: diff^T Sigma^{-1} diff = x^T x
            quad_form = jnp.sum(x**2)
            # Log determinant: log|Sigma| = 2 * sum(log(diag(L)))
            safe_diag_L = jnp.maximum(jnp.diag(L), 1e-12) # Increased epsilon slightly
            log_det_sigma = 2 * jnp.sum(jnp.log(safe_diag_L))
            # Log likelihood: -0.5 * (quad_form + log_det + N*log(2pi))
            log_prob = -0.5 * (N * jnp.log(2 * jnp.pi) + log_det_sigma + quad_form)
        except jnp.linalg.LinAlgError:
            # If Cholesky fails (e.g., not positive definite during optimization), return -inf
            log_prob = -jnp.inf
        return log_prob

    # --- Smoothing ---
    # Smoothing relies on the base class implementation which uses NumPy
    # and linearization. We provide the necessary interface methods.
    # The base class `smooth` method will use the stored `self.filtered_states`
    # and `self.filtered_covs`, which we stored as NumPy arrays.

    def _get_transition_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        Get the linearized state transition matrix (NumPy version for smoother).

        Parameters
        ----------
        state : np.ndarray
            Current state (NumPy array expected by base smoother)

        Returns
        -------
        np.ndarray
            Transition matrix F_t with shape (state_dim, state_dim)
        """
        K = self.K
        # Convert params back to NumPy temporarily if needed, or use NumPy versions
        params_np = DFSV_params(
            N=self.params.N, K=self.params.K,
            lambda_r=np.array(self.params.lambda_r),
            Phi_f=np.array(self.params.Phi_f), Phi_h=np.array(self.params.Phi_h),
            mu=np.array(self.params.mu), sigma2=np.array(self.params.sigma2),
            Q_h=np.array(self.params.Q_h)
        )

        F_t = np.zeros((self.state_dim, self.state_dim))
        F_t[:K, :K] = params_np.Phi_f
        F_t[K:, K:] = params_np.Phi_h
        return F_t

    def _predict_with_matrix(
        self, state: np.ndarray, cov: np.ndarray, transition_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make a prediction using the provided transition matrix (NumPy for smoother).

        Parameters
        ----------
        state : np.ndarray
            Current state estimate x_t (NumPy array, shape (1, state_dim) expected from base smoother)
        cov : np.ndarray
            Current covariance P_t (NumPy array, shape (state_dim, state_dim))
        transition_matrix : np.ndarray
            Linearized state transition matrix F_t (NumPy array)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Predicted state mean x_{t+1|t} shape (1, state_dim) and covariance P_{t+1|t} shape (state_dim, state_dim)
        """
        K = self.K
        # Transpose input state (1, state_dim) to (state_dim, 1) for internal calculations
        state_col = state.T

        # Convert params back to NumPy temporarily
        mu_np = np.array(self.params.mu).reshape(-1, 1)
        q_h_np = np.array(self.params.Q_h)

        # Predict state mean E[x_{t+1}|t] using the non-linear dynamics
        factors = state_col[:K, :]    # Use column vector
        log_vols = state_col[K:, :]   # Use column vector

        pred_factors_mean = transition_matrix[:K, :K] @ factors # Linear part: Phi_f @ f_t
        pred_log_vols_mean = mu_np + transition_matrix[K:, K:] @ (log_vols - mu_np) # h_{t+1} = mu + Phi_h(h_t - mu)
        predicted_state_mean_col = np.vstack([pred_factors_mean, pred_log_vols_mean]) # Shape (state_dim, 1)

        # Predict covariance: P_{t+1|t} = F_t P_t F_t^T + Q_t
        # Process noise Q_t = Cov([f_noise; h_noise])
        # Q_f = diag(exp(h_t)), Q_h = params.Q_h
        Q_t = np.zeros((self.state_dim, self.state_dim))
        Q_t[K:, K:] = q_h_np
        # Approximate factor noise cov using current log_vols estimate h_t
        current_log_vols = state_col[K:, :].flatten() # Use column vector
        Q_t[:K, :K] = np.diag(np.exp(current_log_vols)) # Factor process noise variance depends on h_t

        predicted_cov = transition_matrix @ cov @ transition_matrix.T + Q_t

        # Return mean reshaped to (1, state_dim) for compatibility with base smoother
        return predicted_state_mean_col.T, predicted_cov

    def smooth(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the approximate forward-filtering backward-smoothing (RTS) smoother.
        Relies on the base class implementation using the stored NumPy filtered results.
        """
        if not self.is_filtered:
            raise RuntimeError("Must run filter before smoothing")

        # Call the base class smoother which uses the stored NumPy arrays
        # and the _get_transition_matrix and _predict_with_matrix implemented above.
        smoothed_states, smoothed_covs = super().smooth()

        self.smoothed_states = smoothed_states
        self.smoothed_covs = smoothed_covs
        self.is_smoothed = True

        return smoothed_states, smoothed_covs

    # --- Parameter Optimization --- #

    def log_likelihood_of_params(
        self,
        params_in: DFSVParamsDataclass, # Expecting JAX compatible Pytree
        observations: jnp.ndarray,
    ) -> float:
        """
        Calculate the log-likelihood of parameters given observations using scan.

        This method is designed for use in parameter optimization and uses
        JAX's scan for efficiency and differentiability.

        Args:
            params_in: Parameters as a JAX-compatible PyTree (DFSVParamsDataclass).
            observations: Observation data with shape (T, N) or (N, T).

        Returns:
            float: Total log-likelihood value.
        """
        # Use the static, jitted helper function
        # Pass self as a static argument to the jitted function
        # Cast the result (JAX array) to a Python float here
        jax_result = self._jit_log_likelihood_of_params(self, params_in, observations)
        return float(jax_result)

    @staticmethod
    @partial(jit, static_argnums=(0,)) # Temporarily disable JIT for debugging NaNs
    def _jit_log_likelihood_of_params(
        filter_instance, # The DFSVParticleFilter instance (static)
        params_in: DFSVParamsDataclass,
        observations: jnp.ndarray
    ) -> float:
        """
        Static, JIT-compiled helper for log-likelihood calculation.
        """
        if observations.shape[0] < observations.shape[1]:
            observations = observations.T
        T = observations.shape[0]
        N = observations.shape[1]

        current_params = params_in
        # dtype = current_params.sigma2.dtype # Not needed here

        # --- Sigma2 handling moved inside compute_log_likelihood_particle ---

        # --- Precompute Q_h Cholesky ---
        chol_Q_h_local = jax.scipy.linalg.cholesky(current_params.Q_h, lower=True)
        # --- End Precompute ---

        key = jax.random.PRNGKey(filter_instance._seed) # Use stored _seed
        key, initial_particles, initial_normalized_log_weights = filter_instance.initialize_particles(
            current_params, key
        )

        def scan_body_opt(carry, y_t):
            key, current_particles, current_normalized_log_weights, ll_accum = carry
            observation = y_t.reshape(-1, 1)

            # --- 1. Predict (Inline logic, using params_in etc.) ---
            K_opt = params_in.K
            key, key_h, key_f = jax.random.split(key, 3)
            factors_opt = current_particles[:K_opt, :]
            log_vols_opt = current_particles[K_opt:, :]
            mu_col_opt = params_in.mu.reshape(-1, 1)
            h_deviation_opt = log_vols_opt - mu_col_opt
            h_evolution_opt = mu_col_opt + params_in.Phi_h @ h_deviation_opt
            noise_h_opt = chol_Q_h_local @ jax.random.normal(key_h, shape=(K_opt, filter_instance.num_particles))
            predicted_log_vols_opt = h_evolution_opt + noise_h_opt
            f_evolution_opt = params_in.Phi_f @ factors_opt
            std_noise_opt = jax.random.normal(key_f, shape=(K_opt, filter_instance.num_particles))
            vol_scale_opt = jnp.exp(predicted_log_vols_opt / 2.0)
            factor_noise_opt = vol_scale_opt * std_noise_opt
            predicted_factors_opt = f_evolution_opt + factor_noise_opt
            predicted_particles = jnp.vstack([predicted_factors_opt, predicted_log_vols_opt])
            # --- End Predict ---

            # --- 2. Compute Weights (Inline logic) ---
            # Prepare sigma2_matrix (N, N) outside the JITted likelihood function
            sigma2_in = params_in.sigma2 # Use the sigma2 from params_in for this iteration
            sigma2_matrix_opt = jnp.where(
                sigma2_in.ndim == 1,
                jnp.diag(sigma2_in), # Convert 1D to 2D diag
                sigma2_in            # Assume already 2D
            )

            # Pass the prepared matrix to the likelihood function
            log_likelihood_terms = filter_instance.compute_log_likelihood_particle(
                predicted_particles,
                observation,
                params_in.lambda_r,
                sigma2_matrix_opt,  # Pass prepared (N, N) matrix
                params_in.K,
                params_in.N
            )
            new_unnormalized_log_weights = current_normalized_log_weights + log_likelihood_terms
            # --- End Compute Weights ---

            # --- 4. Calculate LL Contrib ---
            ll_contrib = jax.scipy.special.logsumexp(
                current_normalized_log_weights + log_likelihood_terms
            )
            ll_accum += ll_contrib
            # --- End LL Contrib ---

            # --- 5. Resample ---
            # Call filter_instance's resample_particles method directly
            key, particles_next, next_normalized_log_weights, ess_opt = filter_instance.resample_particles(
                key, predicted_particles, new_unnormalized_log_weights
            )
            # --- End Resample ---

            return (key, particles_next, next_normalized_log_weights, ll_accum), None
        initial_carry = (key, initial_particles, initial_normalized_log_weights, 0.0)
        final_carry, _ = jax.lax.scan(scan_body_opt, initial_carry, observations)
        key, _, _, total_log_likelihood = final_carry # Unpack the final key as well

        # Return the JAX array directly, casting happens in the wrapper
        return total_log_likelihood

    # --- Helper static methods for JIT compatibility --- #

    # --- Removed static helper methods ---
    # _resample_particles_static, _systematic_resample_static,
    # _static_compute_loglik_diag, _static_compute_loglik_full
    # are no longer needed as the logic is handled by instance methods
    # called correctly within the JITted context.

    def filter(self, params: DFSVParamsDataclass, y: jnp.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the Particle Filter on the provided data using JAX scan.

        Args:
            params: Parameters of the DFSV model (DFSVParamsDataclass with JAX arrays).
                    Note: This argument is for API consistency but self.params is used internally.
            y: Observed returns with shape (T, N) or (N, T).

        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                Filtered states (mean estimate) with shape (T, state_dim),
                Filtered covariances (approximate) with shape (T, state_dim, state_dim),
                Total log-likelihood (float).
        """
        # Ensure params match internal params if provided for consistency check (optional)
        # For now, we rely on self.params set during __init__
        current_params = self.params

        # Ensure y is JAX array and in (T, N) format
        y_jax = jnp.asarray(y)
        if y_jax.shape[0] < y_jax.shape[1]:
            y_jax = y_jax.T
        T = y_jax.shape[0]

        # Initialize particles and weights
        key, initial_particles, initial_normalized_log_weights = self.initialize_particles(
            current_params, self.key
        )

        # --- JAX Scan implementation ---
        def scan_body(carry, y_t):
            key, current_particles, current_normalized_log_weights, ll_accum = carry
            observation = y_t.reshape(-1, 1) # Ensure (N, 1)

            # --- 1. Predict ---
            # Use the JITted predict_particles method
            key, predicted_particles = self.predict_particles(key, current_particles)
            # --- End Predict ---

            # --- 2. Compute Weights ---
            # Prepare sigma2_matrix (N, N) outside the JITted likelihood function
            sigma2_curr = current_params.sigma2
            sigma2_matrix = jnp.where(
                sigma2_curr.ndim == 1,
                jnp.diag(sigma2_curr), # Convert 1D to 2D diag
                sigma2_curr            # Assume already 2D
            )
            # Pass the prepared matrix to the likelihood function
            log_likelihood_terms = self.compute_log_likelihood_particle(
                predicted_particles,
                observation,
                current_params.lambda_r,
                sigma2_matrix,  # Pass prepared (N, N) matrix
                current_params.K,
                current_params.N
            )
            new_unnormalized_log_weights = current_normalized_log_weights + log_likelihood_terms
            # --- End Compute Weights ---

            # --- 3. Calculate LL Contrib ---
            ll_contrib = jax.scipy.special.logsumexp(
                current_normalized_log_weights + log_likelihood_terms
            )
            ll_accum += ll_contrib
            # --- End LL Contrib ---

            # --- 4. Resample ---
            # Call the JITted resample_particles method
            key, particles_next, next_normalized_log_weights, ess = self.resample_particles(
                key, predicted_particles, new_unnormalized_log_weights
            )
            # --- End Resample ---

            # --- Calculate filtered state mean and covariance for storage ---
            # Weighted mean of particles
            weights_linear = jnp.exp(next_normalized_log_weights)
            filtered_state_mean = jnp.sum(particles_next * weights_linear, axis=1)
            # Weighted covariance (approximate)
            diff = particles_next - filtered_state_mean.reshape(-1, 1)
            # einsum: weights(p) * diff(j,p) * diff(k,p) -> sum over p -> cov(j,k)
            filtered_state_cov = jnp.einsum('p,jp,kp->jk', weights_linear, diff, diff)
            # Ensure covariance is symmetric
            filtered_state_cov = (filtered_state_cov + filtered_state_cov.T) / 2.0

            return (key, particles_next, next_normalized_log_weights, ll_accum), (filtered_state_mean, filtered_state_cov, ess)

        # Initialize carry for scan
        initial_carry = (key, initial_particles, initial_normalized_log_weights, 0.0)

        # Run scan
        final_carry, scan_outputs = jax.lax.scan(scan_body, initial_carry, y_jax)
        final_key, final_particles, final_weights, log_likelihood = final_carry
        filtered_states_means, filtered_states_covs, ess_history = scan_outputs

        # Update stored key, particles, weights (optional, maybe not needed after filter)
        self.key = final_key
        self.particles = final_particles # Store final JAX particles
        self.weights = final_weights     # Store final JAX log weights

        # Store results as NumPy arrays
        self.filtered_states = np.array(filtered_states_means)
        self.filtered_covs = np.array(filtered_states_covs)
        self.log_likelihood = float(log_likelihood)
        self.effective_sample_size = np.array(ess_history)
        self.is_filtered = True
        self.is_smoothed = False # Reset smoothed flag

        return self.filtered_states, self.filtered_covs, self.log_likelihood

    @partial(jit, static_argnums=(0,)) # Add JIT, self is static
    def resample_particles(
        self,
        key: jax.random.PRNGKey,
        particles: jnp.ndarray,
        unnormalized_log_weights: jnp.ndarray
    ) -> Tuple[jax.random.PRNGKey, jnp.ndarray, jnp.ndarray, float]:
        """
        Resample particles if ESS falls below threshold, using JAX.

        Args:
            key: JAX random key.
            particles: Predicted particles (state_dim, num_particles).
            unnormalized_log_weights: Unnormalized log weights (num_particles,).

        Returns:
            Tuple[jax.random.PRNGKey, jnp.ndarray, jnp.ndarray, float]:
                Updated key,
                Next particles (either resampled or predicted),
                Corresponding normalized log weights,
                Effective Sample Size (ESS).
        """
        # Normalize log weights
        log_sum_weights = jax.scipy.special.logsumexp(unnormalized_log_weights)
        normalized_log_weights = unnormalized_log_weights - log_sum_weights
        normalized_weights_linear = jnp.exp(normalized_log_weights)

        # Calculate ESS
        ess = 1.0 / jnp.sum(normalized_weights_linear**2)

        # Resampling threshold condition
        needs_resampling = ess < (self.resample_threshold * self.num_particles)

        # Define resampling function (Systematic Resampling)
        def do_resample(operand): # op = (key, particles, weights_linear)
            key_resample, particles_resample, weights_linear_resample = operand
            key_resample, subkey = jax.random.split(key_resample)
            n_particles = weights_linear_resample.shape[0]
            positions = (jax.random.uniform(subkey) + jnp.arange(n_particles)) / n_particles
            indices = jnp.searchsorted(jnp.cumsum(weights_linear_resample), positions)
            resampled_particles = particles_resample[:, indices]
            # Reset weights to uniform after resampling
            resampled_log_weights = jnp.full(n_particles, -jnp.log(n_particles))
            return key_resample, resampled_particles, resampled_log_weights

        def dont_resample(operand): # op = (key, particles, weights_log_normalized)
            key_no_resample, particles_no_resample, weights_log_normalized = operand
            # Return original key, predicted particles, and normalized log weights
            return key_no_resample, particles_no_resample, weights_log_normalized

        # Use lax.cond to choose whether to resample
        # Operands need to match expected inputs of branches
        operand_resample = (key, particles, normalized_weights_linear)
        operand_no_resample = (key, particles, normalized_log_weights)

        # Note: lax.cond requires operands to have same PyTree structure.
        # We need a way to pass the correct weights format to each branch.
        # A simpler approach is to have both branches return the same output structure:
        # (key, next_particles, next_normalized_log_weights)
        # We perform the weight reset inside do_resample.

        # Simplified operands (pass needed components)
        operand = (key, particles, normalized_weights_linear, normalized_log_weights)

        def _resample_branch(op):
            k, p, w_lin, _ = op # Ignore log weights
            k_new, p_new, lw_new = do_resample((k, p, w_lin))
            return k_new, p_new, lw_new

        def _no_resample_branch(op):
            k, p, _, w_log = op # Ignore linear weights
            k_new, p_new, lw_new = dont_resample((k, p, w_log))
            return k_new, p_new, lw_new

        # Perform conditional resampling
        key, next_particles, next_normalized_log_weights = jax.lax.cond(
            needs_resampling,
            _resample_branch,
            _no_resample_branch,
            operand
        )

        return key, next_particles, next_normalized_log_weights, ess
