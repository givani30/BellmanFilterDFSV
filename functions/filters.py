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

from functions.simulation import DFSV_params


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

        # Solve discrete Lyapunov equation: P_h = Phi_h * P_h * Phi_h' + Q_h for
        from scipy.linalg import solve_discrete_lyapunov

        P_h = solve_discrete_lyapunov(params.Phi_h, params.Q_h)

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
            Current state
        cov : np.ndarray
            Current covariance
        transition_matrix : np.ndarray
            State transition matrix

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Predicted state and covariance
        """
        raise NotImplementedError("Must be implemented by subclasses")

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
    Particle Filter implementation for DFSV models.

    This class implements a particle filter (bootstrap filter) for state
    estimation in DFSV models, using Sequential Importance Sampling with
    Resampling (SISR) to handle the nonlinearities in the DFSV model.
    """

    def __init__(
        self,
        params: DFSV_params,
        num_particles: int = 1000,
        resample_threshold: float = 0.5,
    ):
        """
        Initialize the Particle Filter with DFSV model parameters.

        Parameters
        ----------
        params : DFSV_params
            Parameters of the DFSV model
        num_particles : int, optional
            Number of particles to use. Default is 1000.
        resample_threshold : float, optional
            Effective sample size threshold as a fraction of num_particles,
            below which resampling is triggered. Default is 0.5.
        """
        super().__init__(params.N, params.K)
        self.num_particles = num_particles
        self.resample_threshold = resample_threshold

        # Store params
        self.params = params

        # Storage for particles and weights
        self.particles = None
        self.weights = None
        self.effective_sample_size = None

        # Random number generator
        self.rng = np.random.RandomState()

        # Initialize sampling matrices
        self._initialize_sampling_matrices(params)

    def _initialize_sampling_matrices(self, params):
        """Initialize matrices needed for sampling from distributions."""
        # Cholesky decomposition of Q_h for log-volatility noise
        self.chol_Q_h = np.linalg.cholesky(params.Q_h)

        # For idiosyncratic noise in observation equation
        if params.sigma2.ndim == 1:
            self.chol_sigma2 = np.sqrt(params.sigma2)
        else:
            self.chol_sigma2 = np.linalg.cholesky(params.sigma2)

    def initialize_particles(
        self, params, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize particles and weights.

        Parameters
        ----------
        params : DFSV_params
            Parameters of the DFSV model
        y : np.ndarray
            Observed data with shape (T, N)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Initialized particles with shape (state_dim, num_particles),
            and uniform weights with shape (num_particles,)
        """
        # Get initial state and covariance from base class
        initial_state, initial_cov = self.initialize_state(params)

        # Initialize particles by sampling from initial distribution
        L = np.linalg.cholesky(np.array(initial_cov))

        # Generate num_particles samples from N(initial_state, initial_cov)
        noise = self.rng.standard_normal(size=(self.state_dim, self.num_particles))
        particles = np.array(initial_state).reshape(-1, 1) + L @ noise

        # Initialize weights to uniform
        weights = np.ones(self.num_particles) / self.num_particles

        return particles, weights

    def predict_particles(self, params, particles: np.ndarray) -> np.ndarray:
        """
        Propagate particles through state transition equation.

        Parameters
        ----------
        params : DFSV_params
            Parameters of the DFSV model
        particles : np.ndarray
            Current particles with shape (state_dim, num_particles)

        Returns
        -------
        np.ndarray
            Predicted particles with shape (state_dim, num_particles)
        """
        K = self.K

        # Extract state components
        factors = particles[:K, :]
        log_vols = particles[K:, :]

        # Make sure mu has the right shape for broadcasting
        mu = params.mu.reshape(-1, 1) if params.mu.ndim == 1 else params.mu

        # Predict log-volatilities: h_{t+1} = mu + Phi_h * (h_t - mu) + eta_t
        h_deviation = log_vols - mu
        h_evolution = mu + params.Phi_h @ h_deviation

        # Add noise: eta_t ~ N(0, Q_h)
        noise_h = self.chol_Q_h @ self.rng.standard_normal(size=(K, self.num_particles))
        predicted_log_vols = h_evolution + noise_h

        # Predict factors: f_{t+1} = Phi_f * f_t + diag(exp(h_t/2)) * eps_t
        f_evolution = params.Phi_f @ factors

        # Add state-dependent noise: eps_t ~ N(0, diag(exp(h_t)))
        # Note: we use predicted_log_vols for more accurate simulation
        predicted_factors = np.zeros_like(factors)

        # Vectorize the computation
        std_noise = self.rng.standard_normal(size=(K, self.num_particles))
        vol_scale = np.exp(predicted_log_vols / 2)
        predicted_factors = f_evolution + vol_scale * std_noise

        # Combine into predicted particles
        predicted_particles = np.vstack([predicted_factors, predicted_log_vols])

        return predicted_particles

    def compute_weights(
        self,
        params,
        particles: np.ndarray,
        observation: np.ndarray,
        prev_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Compute importance weights based on observation likelihood.

        Parameters
        ----------
        params : DFSV_params
            Parameters of the DFSV model
        particles : np.ndarray
            Current particles with shape (state_dim, num_particles)
        observation : np.ndarray
            Current observation with shape (N, 1)
        prev_weights : np.ndarray
            Previous weights with shape (num_particles,)

        Returns
        -------
        np.ndarray
            Updated weights with shape (num_particles,)
        """
        K = self.K

        # Extract factors from particles
        factors = particles[:K, :]

        # Compute predicted observations for all particles
        pred_obs = params.lambda_r @ factors

        # Compute log-weights
        log_weights = np.zeros(self.num_particles)

        # If sigma2 is diagonal (most common case)
        if params.sigma2.ndim == 1 or np.all(
            np.diag(np.diag(params.sigma2)) == params.sigma2
        ):
            # Vectorized computation
            sigma_diag = (
                np.diag(params.sigma2) if params.sigma2.ndim == 2 else params.sigma2
            )
            diff = observation.flatten() - pred_obs.T  # Shape: (num_particles, N)
            log_likelihood = -0.5 * np.sum(diff * diff / sigma_diag, axis=1)
            log_weights = log_likelihood
        else:
            # Full covariance case
            Sigma_inv = np.linalg.inv(params.sigma2)
            for i in range(self.num_particles):
                diff = observation - pred_obs[:, i : i + 1]
                log_weights[i] = -0.5 * float(diff.T @ Sigma_inv @ diff)

        # Add log of previous weights (for sequential importance sampling)
        log_weights += np.log(prev_weights + 1e-10)

        # Subtract max for numerical stability
        log_weights -= np.max(log_weights)

        # Convert to weights and normalize
        weights = np.exp(log_weights)
        weights /= np.sum(weights)

        return weights

    def _systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        """
        Perform systematic resampling.

        Parameters
        ----------
        weights : np.ndarray
            Importance weights with shape (num_particles,)

        Returns
        -------
        np.ndarray
            Indices of selected particles
        """
        # Generate random offset
        u = self.rng.uniform() / self.num_particles

        # Create positions
        positions = (np.arange(self.num_particles) + u) / self.num_particles

        # Compute cumulative sum of weights
        cumsum = np.cumsum(weights)

        # Find indices using searchsorted
        indices = np.searchsorted(cumsum, positions)

        # Ensure indices are within bounds
        indices = np.clip(indices, 0, self.num_particles - 1)

        return indices

    def resample_particles(
        self, particles: np.ndarray, weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample particles based on weights.

        Parameters
        ----------
        particles : np.ndarray
            Current particles with shape (state_dim, num_particles)
        weights : np.ndarray
            Current weights with shape (num_particles,)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Resampled particles and equal weights
        """
        # Calculate effective sample size
        ess = 1.0 / np.sum(weights**2)
        self.effective_sample_size = ess

        # Check if resampling is needed
        if ess < self.resample_threshold * self.num_particles:
            # Systematic resampling
            indices = self._systematic_resample(weights)

            # Resample particles
            resampled_particles = particles[:, indices]

            # Reset weights to uniform
            resampled_weights = np.ones(self.num_particles) / self.num_particles

            return resampled_particles, resampled_weights
        else:
            # No resampling needed
            return particles, weights

    def predict(
        self, params, state: np.ndarray, cov: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method is not used in the particle filter implementation.
        Instead, particle propagation is handled directly in the filter method.

        Parameters
        ----------
        params : DFSV_params
            Parameters of the DFSV model
        state : np.ndarray
            Current state estimate
        cov : np.ndarray
            Current state covariance

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Placeholder to meet interface requirements
        """
        # This method is required by the abstract base class but not used
        # in particle filtering. It's implemented to fulfill the interface.
        return state, cov

    def update(
        self,
        params,
        predicted_state: np.ndarray,
        predicted_cov: np.ndarray,
        observation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        This method is not used in the particle filter implementation.
        Instead, weight update is handled directly in the filter method.

        Parameters
        ----------
        params : DFSV_params
            Parameters of the DFSV model
        predicted_state : np.ndarray
            Predicted state
        predicted_cov : np.ndarray
            Predicted state covariance
        observation : np.ndarray
            Current observation

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            Placeholder to meet interface requirements
        """
        # This method is required by the abstract base class but not used
        # in particle filtering. It's implemented to fulfill the interface.
        return predicted_state, predicted_cov, 0.0

    def filter(self, params, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run particle filter on the provided data.

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
        try:
            from tqdm import tqdm
        except ImportError:

            def tqdm(iterable, **kwargs):
                return iterable

            print("Warning: tqdm not installed. No progress bar will be shown.")

        # Ensure y is in (T, N) format
        if y.shape[0] < y.shape[1]:  # If (N, T) format
            y = y.T

        T = y.shape[0]

        # Storage for filtered states and covariances
        filtered_states = np.zeros((T, self.state_dim))
        filtered_covs = np.zeros((T, self.state_dim, self.state_dim))
        log_likelihood = 0.0

        # Initialize particles and weights
        particles, weights = self.initialize_particles(params, y)

        # Forward pass through observations
        for t in tqdm(range(T), desc="Particle Filter Progress"):
            # Prediction step
            particles = self.predict_particles(params, particles)

            # Update step
            observation = y[t : t + 1, :].T
            weights = self.compute_weights(params, particles, observation, weights)

            # Compute weighted mean for filtered state
            weighted_particles = particles * weights.reshape(1, -1)
            filtered_mean = np.sum(weighted_particles, axis=1)
            filtered_states[t, :] = filtered_mean

            # Compute weighted covariance
            centered_particles = particles - filtered_mean.reshape(-1, 1)
            weighted_centered = centered_particles * np.sqrt(weights).reshape(1, -1)
            filtered_covs[t] = weighted_centered @ weighted_centered.T

            # Compute log-likelihood contribution (skip first observation)
            if t > 0:
                log_likelihood += np.log(np.sum(weights))

            # Resample if necessary
            particles, weights = self.resample_particles(particles, weights)

        # Store results
        self.filtered_states = filtered_states
        self.filtered_covs = filtered_covs
        self.log_likelihood = log_likelihood
        self.is_filtered = True

        # Save final particles and weights for potential smoothing
        self.particles = particles
        self.weights = weights

        return filtered_states, filtered_covs, log_likelihood

    def _get_transition_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        Get the linearized state transition matrix at given state.
        This is used for smoothing but not for filtering in the particle filter.

        Parameters
        ----------
        state : np.ndarray
            Current state

        Returns
        -------
        np.ndarray
            Transition matrix with shape (state_dim, state_dim)
        """
        K = self.K
        # Create transition matrix
        F_t = np.zeros((self.state_dim, self.state_dim))

        # Top-left block: factor transition
        F_t[:K, :K] = self.params.Phi_f

        # Bottom-right block: log-volatility transition
        F_t[K:, K:] = self.params.Phi_h

        return F_t

    def _predict_with_matrix(
        self, state: np.ndarray, cov: np.ndarray, transition_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make a prediction using the provided transition matrix.
        Used for smoothing but not for filtering in the particle filter.

        Parameters
        ----------
        state : np.ndarray
            Current state
        cov : np.ndarray
            Current covariance
        transition_matrix : np.ndarray
            State transition matrix

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Predicted state and covariance
        """
        K = self.K
        mu = (
            self.params.mu.reshape(-1, 1)
            if self.params.mu.ndim == 1
            else self.params.mu
        )

        # Extract factors and log_vols
        factors = state.flatten()[:K]
        log_vols = state.flatten()[K:]

        # Predict factors: E[f_{t+1}|t] = Phi_f @ f_t
        predicted_factors = transition_matrix[:K, :K] @ factors

        # Predict log-volatilities: E[h_{t+1}|t] = mu + Phi_h @ (log_vols - mu)
        Phi_h = transition_matrix[K:, K:]
        predicted_log_vols = mu.flatten() + Phi_h @ (log_vols - mu.flatten())

        # Combine predicted state
        predicted_state = np.hstack([predicted_factors, predicted_log_vols]).reshape(
            -1, 1
        )

        # Process noise covariance
        Q_t = np.zeros((self.state_dim, self.state_dim))
        Q_t[K:, K:] = self.params.Q_h
        Q_t[:K, :K] = np.diag(np.exp(log_vols.flatten()))

        # Predict covariance
        predicted_cov = transition_matrix @ cov @ transition_matrix.T + Q_t

        return predicted_state, predicted_cov

    def smooth(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run forward-backward smoother on particle filter results.

        This implements a modified forward-filtering backward-smoothing approach
        that approximates the smoothed posterior based on the filtered particles.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Smoothed states and covariances
        """
        if not self.is_filtered:
            raise RuntimeError("Must run filter before smoothing")

        # Use the base class smoother as an approximation
        # The backward recursions are based on linearized dynamics
        smoothed_states, smoothed_covs = super().smooth()

        self.smoothed_states = smoothed_states
        self.smoothed_covs = smoothed_covs
        self.is_smoothed = True

        return smoothed_states, smoothed_covs
