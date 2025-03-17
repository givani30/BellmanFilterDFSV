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
from jax import hessian, jit

from functions.simulation import DFSV_params


class DFSVFilter:
    """
    Base class for filters applied to Dynamic Factor Stochastic Volatility models.

    This class provides the foundation for implementing various Kalman filter
    variants (EKF, UKF) for DFSV models, handling the joint filtering of
    latent factors and log-volatilities.
    """

    def __init__(self, params: DFSV_params):
        """
        Initialize the Kalman filter with DFSV model parameters.

        Parameters
        ----------
        params : DFSV_params
            Parameters of the DFSV model
        """
        self.params = params
        self.N = params.N  # Number of observed series
        self.K = params.K  # Number of factors

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

    def initialize_state(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize state vector and covariance matrix.

        Parameters
        ----------
        y : np.ndarray
            Observed data with shape (N, T)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Initial state vector with shape (state_dim, 1) and
            initial state covariance matrix with shape (state_dim, state_dim)
        """
        # Initialize factors to zero
        initial_factors = np.zeros((self.K, 1))

        # Initialize log-volatilities to the unconditional mean
        if self.params.mu.ndim == 1:
            initial_log_vols = self.params.mu.reshape(-1, 1)
        else:
            initial_log_vols = self.params.mu.copy()

        # Combine into state vector [f; h]
        initial_state = np.vstack([initial_factors, initial_log_vols])

        # Initialize state covariance
        # For factors, use identity or solve Lyapunov equation
        P_f = np.eye(self.K)

        # For log-volatilities, use unconditional covariance
        # Solve discrete Lyapunov equation: P_h = Phi_h * P_h * Phi_h' + Q_h
        from scipy.linalg import solve_discrete_lyapunov

        P_h = solve_discrete_lyapunov(self.params.Phi_h, self.params.Q_h)

        # Construct block-diagonal covariance
        initial_cov = np.block(
            [[P_f, np.zeros((self.K, self.K))], [np.zeros((self.K, self.K)), P_h]]
        )

        return initial_state, initial_cov

    def predict(
        self, state: np.ndarray, cov: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the prediction step.

        Parameters
        ----------
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
        predicted_state: np.ndarray,
        predicted_cov: np.ndarray,
        observation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform the update step.

        Parameters
        ----------
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

    def filter(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the Kalman filter on the provided data.

        Parameters
        ----------
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

        # Ensure y is in (T, N) format
        if y.shape[0] < y.shape[1]:  # If (N, T) format
            y = y.T

        T = y.shape[0]

        # Storage for filtered states and covariances
        filtered_states = np.zeros((T, self.state_dim))
        filtered_covs = np.zeros((T, self.state_dim, self.state_dim))

        # Initialize
        state, cov = self.initialize_state(y.T)  # Note: initialize_state expects (N, T)
        log_likelihood = 0.0

        # Forward pass
        for t in tqdm(range(T), desc="Filter Progress"):
            # Prediction step
            predicted_state, predicted_cov = self.predict(state, cov)

            # Update step - reshape observation to (N, 1)
            observation = y[t : t + 1, :].T.reshape(-1, 1)
            state, cov, ll_contrib = self.update(
                predicted_state, predicted_cov, observation
            )

            # Store results (convert state from (state_dim, 1) to row vector)
            filtered_states[t, :] = state.flatten()
            filtered_covs[t, :, :] = cov
            log_likelihood += ll_contrib

        # Store results in object
        self.filtered_states = filtered_states
        self.filtered_covs = filtered_covs
        self.log_likelihood = log_likelihood
        self.is_filtered = True

        return filtered_states, filtered_covs, log_likelihood

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
        super().__init__(params)
        self.num_particles = num_particles
        self.resample_threshold = resample_threshold

        # Storage for particles and weights
        self.particles = None
        self.weights = None
        self.effective_sample_size = None

        # Random number generator
        self.rng = np.random.RandomState()

        # Cholesky decompositions for sampling
        self._initialize_sampling_matrices()

    def _initialize_sampling_matrices(self):
        """Initialize matrices needed for sampling from distributions."""
        # Cholesky decomposition of Q_h for log-volatility noise
        self.chol_Q_h = np.linalg.cholesky(self.params.Q_h)

        # For idiosyncratic noise in observation equation
        if self.params.sigma2.ndim == 1:
            self.chol_sigma2 = np.sqrt(self.params.sigma2)
        else:
            self.chol_sigma2 = np.linalg.cholesky(self.params.sigma2)

    def initialize_particles(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize particles and weights.

        Parameters
        ----------
        y : np.ndarray
            Observed data with shape (N, T)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Initialized particles with shape (state_dim, num_particles),
            and uniform weights with shape (num_particles,)
        """
        # Get initial state and covariance from base class
        initial_state, initial_cov = self.initialize_state(y)

        # Initialize particles by sampling from initial distribution
        chol_initial_cov = np.linalg.cholesky(initial_cov)

        # Generate num_particles samples from N(initial_state, initial_cov)
        particles = initial_state + chol_initial_cov @ self.rng.standard_normal(
            size=(self.state_dim, self.num_particles)
        )

        # Initialize weights to uniform
        weights = np.ones(self.num_particles) / self.num_particles

        return particles, weights

    def predict_particles(self, particles: np.ndarray) -> np.ndarray:
        """
        Propagate particles through state transition equation.

        Parameters
        ----------
        particles : np.ndarray
            Current particles with shape (state_dim, num_particles)

        Returns
        -------
        np.ndarray
            Predicted particles with shape (state_dim, num_particles)
        """
        K = self.K
        Phi_f = self.params.Phi_f
        Phi_h = self.params.Phi_h
        mu = (
            self.params.mu.reshape(-1, 1)
            if self.params.mu.ndim == 1
            else self.params.mu
        )

        # Extract state components
        factors = particles[:K, :]
        log_vols = particles[K:, :]

        # Predicted particles storage
        predicted_particles = np.zeros_like(particles)

        # Predict log-volatilities: h_{t+1} = mu + Phi_h * (h_t - mu) + eta_t
        h_deviation = log_vols - mu  # Shape: (K, num_particles)
        h_evolution = mu + Phi_h @ h_deviation  # Shape: (K, num_particles)

        # Add noise: eta_t ~ N(0, Q_h)
        noise_h = self.chol_Q_h @ self.rng.standard_normal(size=(K, self.num_particles))
        predicted_log_vols = h_evolution + noise_h

        # Predict factors: f_{t+1} = Phi_f * f_t + diag(exp(h_t/2)) * eps_t
        f_evolution = Phi_f @ factors  # Shape: (K, num_particles)

        # Add state-dependent noise: eps_t ~ N(0, diag(exp(h_t)))
        # Note: we use predicted_log_vols for more accurate simulation
        for i in range(self.num_particles):
            vol_scale = np.exp(predicted_log_vols[:, i] / 2)
            vol_noise = np.diag(vol_scale) @ self.rng.standard_normal(size=(K, 1))
            predicted_particles[:K, i : i + 1] = f_evolution[:, i : i + 1] + vol_noise

        # Store predicted log-volatilities
        predicted_particles[K:, :] = predicted_log_vols

        return predicted_particles

    def compute_weights(
        self, particles: np.ndarray, observation: np.ndarray, prev_weights: np.ndarray
    ) -> np.ndarray:
        """
        Compute importance weights based on observed data.

        Parameters
        ----------
        particles : np.ndarray
            Predicted particles with shape (state_dim, num_particles)
        observation : np.ndarray
            Current observation with shape (N, 1)
        prev_weights : np.ndarray
            Previous weights with shape (num_particles,)

        Returns
        -------
        np.ndarray
            Updated normalized weights with shape (num_particles,)
        """
        K = self.K
        N = self.N
        lambda_r = self.params.lambda_r

        # Extract factors from particles
        factors = particles[:K, :]

        # Compute expected observations for each particle: y_hat = lambda_r * f
        pred_obs = lambda_r @ factors  # Shape: (N, num_particles)

        # Compute likelihood of observation for each particle
        log_weights = np.zeros(self.num_particles)

        # If sigma2 is diagonal
        if np.all(np.diag(np.diag(self.params.sigma2)) == self.params.sigma2):
            # Vectorized computation for diagonal covariance
            sigma_diag = np.diag(self.params.sigma2)
            for i in range(self.num_particles):
                # Calculate log-likelihood: -0.5 * (y - y_hat)' Sigma^-1 (y - y_hat)
                diff = observation.flatten() - pred_obs[:, i]
                log_likelihood = -0.5 * np.sum(diff**2 / sigma_diag)
                log_weights[i] = log_likelihood
        else:
            # General case for full covariance matrix
            Sigma_inv = np.linalg.inv(self.params.sigma2)
            for i in range(self.num_particles):
                diff = observation - pred_obs[:, i : i + 1]
                log_likelihood = -0.5 * (diff.T @ Sigma_inv @ diff)[0, 0]
                log_weights[i] = log_likelihood

        # Add log of previous weights (for sequential importance sampling)
        log_weights += np.log(
            prev_weights + 1e-10
        )  # Add small constant to avoid log(0)

        # Subtract max for numerical stability
        log_weights -= np.max(log_weights)

        # Convert to weights and normalize
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)

        return weights

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

    def _systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        """
        Perform systematic resampling.

        Parameters
        ----------
        weights : np.ndarray
            Importance weights

        Returns
        -------
        np.ndarray
            Indices of selected particles
        """
        positions = (
            np.arange(self.num_particles) + self.rng.uniform()
        ) / self.num_particles
        indices = np.zeros(self.num_particles, "i")
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < self.num_particles:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        return indices

    def predict(
        self, state: np.ndarray, cov: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method is not used in the particle filter implementation.
        Instead, particle propagation is handled directly in the filter method.

        Parameters
        ----------
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
        predicted_state: np.ndarray,
        predicted_cov: np.ndarray,
        observation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        This method is not used in the particle filter implementation.
        Instead, weight update is handled directly in the filter method.

        Parameters
        ----------
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

    def filter(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run particle filter on the provided data.

        Parameters
        ----------
        y : np.ndarray
            Observed returns with shape (T, N)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            Filtered states (mean of particles) with shape (T, state_dim),
            filtered covariances (covariance of particles) with shape (T, state_dim, state_dim),
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

        # Ensure y is in (T, N) format
        if y.shape[0] < y.shape[1]:  # If (N, T) format
            y = y.T

        T = y.shape[0]

        # Storage for filtered states and covariances
        filtered_states = np.zeros((T, self.state_dim))
        filtered_covs = np.zeros((T, self.state_dim, self.state_dim))
        log_likelihood = 0.0

        # Initialize particles and weights
        particles, weights = self.initialize_particles(y.T)

        # Forward pass through observations
        for t in tqdm(range(T), desc="Particle Filter Progress"):
            # Prediction step: propagate particles through state transition
            particles = self.predict_particles(particles)

            # Update step: compute importance weights based on observation
            weights = self.compute_weights(particles, y[t : t + 1, :].T, weights)

            # Store filtered state as weighted mean of particles
            filtered_states[t : t + 1, :] = np.sum(
                particles * weights, axis=1, keepdims=True
            ).T

            # Store filtered covariance as weighted covariance of particles
            centered_particles = particles - filtered_states[t : t + 1, :].T
            filtered_covs[t, :, :] = np.sum(
                weights
                * np.einsum("ij,kj->ikj", centered_particles, centered_particles),
                axis=2,
            )

            # Compute log-likelihood contribution
            if t > 0:  # Skip first observation as it depends on initialization
                # Approximate log-likelihood using effective sample size
                log_likelihood_t = np.log(np.sum(weights))
                log_likelihood += log_likelihood_t

            # Resample if necessary
            particles, weights = self.resample_particles(particles, weights)

        # Store results in object
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
        Phi_f = self.params.Phi_f
        Phi_h = self.params.Phi_h

        # Initialize transition matrix
        F_t = np.zeros((self.state_dim, self.state_dim))

        # Top-left block: factor transition
        F_t[:K, :K] = Phi_f

        # Bottom-right block: log-volatility transition
        F_t[K:, K:] = Phi_h

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
        # mu = self.params.mu.reshape(-1, 1) if self.params.mu.ndim == 1 else self.params.mu
        mu = self.params.mu
        # Extract state components
        factors = state.flatten()[:K]
        log_vols = state.flatten()[K:]

        # Predict factors: E[f_{t+1}|t] = Phi_f * f_t
        predicted_factors = transition_matrix[:K, :K] @ factors

        # Predict log-volatilities: E[h_{t+1}|t] = mu + Phi_h * (h_t - mu)
        Phi_h = transition_matrix[K:, K:]
        predicted_log_vols = mu + Phi_h.dot(log_vols - mu.T)

        # Combine predicted state
        predicted_state = np.hstack([predicted_factors, predicted_log_vols]).T

        # Process noise covariance
        Q_t = np.zeros((self.state_dim, self.state_dim))

        # Log-volatility noise (constant)
        Q_t[K:, K:] = self.params.Q_h

        # Factor noise (state-dependent)
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
