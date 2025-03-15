"""
Nonlinear Filter implementations for Dynamic Factor Stochastic Volatility models.

This module provides filter classes for state estimation in DFSV models,
including Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), Particle Filters (PF) and Bellman Filters (BF)
to handle the nonlinearities introduced by stochastic volatility.
"""

import jax.scipy.optimize
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List, Union

import scipy
from functions.simulation import DFSV_params
import jax
import jax.numpy as jnp
from jax import grad, hessian, jit
import jaxopt


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
        for t in tqdm(range(T), desc="Kalman Filter Progress"):
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


class DFSVBellmanFilter(DFSVFilter):
    """
    Bellman Filter implementation for DFSV models using JAX for automatic differentiation.

    This class implements a Bellman filter for state estimation in DFSV models,
    using dynamic programming principles to recursively compute the optimal
    state estimates and covariances. JAX is used for automatic differentiation,
    eliminating the need to manually compute gradients and Hessians.
    """

    def __init__(self, params: DFSV_params):
        """
        Initialize the JAX-based Bellman Filter with DFSV model parameters.

        Parameters
        ----------
        params : DFSV_params
            Parameters of the DFSV model
        """
        super().__init__(params)

        # Enable 64-bit precision for JAX
        jax.config.update("jax_enable_x64", True)

        # Convert model parameters to JAX arrays and setup computation functions
        self._setup_jax_params()
        self._compile_jax_functions()

    def _setup_jax_params(self):
        """
        Convert numpy parameters to JAX arrays for use in JAX functions.
        """
        # Convert model parameters to JAX arrays
        self.jax_lambda_r = jnp.array(self.params.lambda_r)

        # Handle sigma2 - ensure it's properly formatted
        if np.isscalar(self.params.sigma2) or (
            isinstance(self.params.sigma2, np.ndarray) and self.params.sigma2.ndim == 1
        ):
            self.jax_sigma2 = jnp.array(self.params.sigma2)
        else:
            self.jax_sigma2 = jnp.array(self.params.sigma2)

        # Handle mu - ensure it has the right shape
        if self.params.mu.ndim == 1:
            self.jax_mu = jnp.array(self.params.mu.reshape(-1, 1))
        else:
            self.jax_mu = jnp.array(self.params.mu)

        # Other model parameters
        self.jax_Phi_f = jnp.array(self.params.Phi_f)
        self.jax_Phi_h = jnp.array(self.params.Phi_h)
        self.jax_Q_h = jnp.array(self.params.Q_h)

    def _compile_jax_functions(self):
        """
        JIT-compile JAX functions for efficiency.
        """

        # Build covariance matrix function
        def build_covariance(lambda_r, exp_h, sigma2):
            """
            Build observation covariance A = Lambda diag(exp_h) Lambda^T + diag(sigma2).

            Parameters:
            lambda_r: (N,K) - factor loadings
            exp_h: (K,) - exponentiated log-volatilities
            sigma2: (N,) - idiosyncratic variances

            Returns:
            A: (N,N) - observation covariance matrix
            """
            Sigma_f = jnp.diag(exp_h)  # (K,K)
            A = lambda_r @ Sigma_f @ lambda_r.T  # (N,N)
            A += jnp.diag(sigma2) + 1e-8 * jnp.eye(
                lambda_r.shape[0]
            )  # Add regularization
            A = 0.5 * (A + A.T)  # Ensure symmetry
            return A

        # Bellman objective function (negative log-posterior)
        @jax.jit
        def jax_bellman_objective(
            alpha, predicted_state, I_pred, observation, lambda_r, sigma2
        ):
            """
            JAX implementation of the Bellman objective function (negative log-posterior).

            Parameters:
            alpha: (2K,) - current state estimate [f, log_vols]
            predicted_state: (2K,) - predicted state
            I_pred: (2K, 2K) - predicted information matrix
            observation: (N,) - observation vector
            lambda_r: (N,K) - factor loadings
            sigma2: (N,) - idiosyncratic variance

            Returns:
            Negative log posterior value (scalar)
            """
            # Unpack dimensions and flatten inputs
            K = lambda_r.shape[1]
            alpha = alpha.flatten()
            predicted_state = predicted_state.flatten()
            observation = observation.flatten()

            # Split state
            f = alpha[:K]
            log_vols = alpha[K:]

            # Innovation and covariance
            pred_obs = lambda_r @ f
            innovation = observation - pred_obs
            exp_log_vols = jnp.exp(log_vols)
            A = build_covariance(lambda_r, exp_log_vols, sigma2)

            # Compute negative log-likelihood with Cholesky decomposition
            try:
                L = jnp.linalg.cholesky(A)

                # Compute quadratic form via triangular solve
                alpha_vec = jax.scipy.linalg.solve_triangular(L, innovation, lower=True)
                quad_form = jnp.sum(alpha_vec**2)

                # Log determinant
                logdet_A = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

                # Negative log-likelihood
                N_ = observation.shape[0]
                neg_log_lik = 0.5 * (N_ * jnp.log(2.0 * jnp.pi) + logdet_A + quad_form)

                # Prior penalty
                state_diff = alpha - predicted_state
                penalty = 0.5 * (state_diff @ (I_pred @ state_diff))

                # Return negative log-posterior
                return neg_log_lik + penalty

            except:
                # In case of Cholesky failure, return a large value
                return jnp.array(1e10)

        # Hand-coded gradient of the negative log-posterior for better stability
        @jax.jit
        def explicit_grad_bellman(
            alpha, predicted_state, I_pred, observation, lambda_r, sigma2
        ):
            """
            Manually derived gradient of the negative log-posterior.

            Parameters:
            Same as jax_bellman_objective

            Returns:
            Gradient vector (2K,)
            """
            K = lambda_r.shape[1]
            N_ = lambda_r.shape[0]

            # Flatten vectors
            alpha = alpha.flatten()
            predicted_state = predicted_state.flatten()
            observation = observation.flatten()

            # Split alpha into factors and log-volatilities
            f = alpha[:K]
            h = alpha[K:]
            exp_h = jnp.exp(h)

            # Innovation and covariance
            innovation = observation - (lambda_r @ f)
            A = build_covariance(lambda_r, exp_h, sigma2)

            # Cholesky factor for efficient matrix operations
            L = jnp.linalg.cholesky(A)

            # Function to compute A^-1 @ x using Cholesky
            def A_inv_matmul(x):
                return jax.scipy.linalg.cho_solve((L, True), x)

            # Compute key terms for gradient
            A_inv_lambda = A_inv_matmul(lambda_r)  # shape (N_, K)
            A_inv_innov = A_inv_matmul(innovation)  # shape (N_,)

            # Gradient w.r.t factors
            grad_f = -lambda_r.T @ A_inv_innov

            # Gradient w.r.t log-volatilities
            term1 = jnp.diag(lambda_r.T @ A_inv_lambda)  # (K,)
            proj = lambda_r.T @ A_inv_innov  # (K,)
            term2 = proj**2  # (K,)
            grad_h = 0.5 * exp_h * (term1 - term2)

            # Add prior penalty gradient
            state_diff = alpha - predicted_state
            penalty_grad = I_pred @ state_diff  # shape (2K,)

            # Combine gradients
            grad_likelihood = jnp.concatenate([grad_f, grad_h])
            grad_total = grad_likelihood + penalty_grad

            return grad_total

        # log-posterior without the penalty term (used for the hessian)
        @jax.jit
        def log_posterior(alpha, observation):
            """
            JAX implementation of the Bellman objective function (negative log-posterior).

            Parameters:
            alpha: (2K,) - current state estimate [f, log_vols]
            observation: (N,) - observation vector
            Returns:
            Negative log posterior value (scalar)
            """
            # Unpack dimensions and flatten inputs
            lambda_r = self.jax_lambda_r
            sigma2 = self.jax_sigma2
            K = self.K
            alpha = alpha.flatten()
            observation = observation.flatten()

            # Split state
            f = alpha[:K]
            log_vols = alpha[K:]

            # Innovation and covariance
            pred_obs = lambda_r @ f
            innovation = observation - pred_obs
            exp_log_vols = jnp.exp(log_vols)
            A = build_covariance(lambda_r, exp_log_vols, sigma2)

            # Compute negative log-likelihood with Cholesky decomposition
            L = jnp.linalg.cholesky(A)

            # Compute quadratic form via triangular solve
            alpha_vec = jax.scipy.linalg.solve_triangular(L, innovation, lower=True)
            quad_form = jnp.sum(alpha_vec**2)

            # Log determinant
            logdet_A = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

            # log-likelihood
            N_ = observation.shape[0]
            log_lik = -0.5 * (N_ * jnp.log(2.0 * jnp.pi) + logdet_A + quad_form)

            # Return negative log-posterior
            return -log_lik

        # Combined objective and gradient function for optimization
        @jax.jit
        def obj_and_grad_fn(x, ps, ip, obs, lr, sig2):
            """Combined objective and gradient function for optimization."""
            x_reshaped = x.reshape(-1, 1)
            obj = jax_bellman_objective(x_reshaped, ps, ip, obs, lr, sig2)
            grad = explicit_grad_bellman(x_reshaped, ps, ip, obs, lr, sig2)
            return obj, grad.flatten()

        # Store the compiled functions
        self.obj_and_grad_fn = obj_and_grad_fn
        self.jax_bellman_objective = jit(jax_bellman_objective)
        self.jax_gradient = explicit_grad_bellman
        self.jax_hessian = jit(hessian(log_posterior, argnums=0))

    def initialize_state(self, y):
        """
        Initialize state and covariance by calling parent method.

        Parameters
        ----------
        y : np.ndarray
            Observed data

        Returns
        -------
        tuple
            Initial state and covariance
        """
        return super().initialize_state(y)

    def predict(
        self, state: np.ndarray, cov: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the Bellman prediction step using Schur complement for information matrix.

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
        # Extract parameters
        K = self.K
        Phi_f = self.params.Phi_f
        Phi_h = self.params.Phi_h
        mu = (
            self.params.mu.reshape(-1, 1)
            if self.params.mu.ndim == 1
            else self.params.mu
        )

        # Extract current state components
        factors = state[:K]
        log_vols = state[K:]

        # Convert to information form (precision matrix)
        I_curr = np.linalg.inv(cov)

        # Predict factors: E[f_{t+1}|t] = Phi_f @ f_t
        predicted_factors = Phi_f @ factors

        # Predict log-volatilities: E[h_{t+1}|t] = mu + Phi_h @ (log_vols - mu)
        predicted_log_vols = mu + Phi_h @ (log_vols - mu)

        # Combine predicted state
        predicted_state = np.vstack([predicted_factors, predicted_log_vols])

        # Get transition matrix
        F_t = self._get_transition_matrix(state)

        # Process noise covariance (using state-dependent factor noise)
        Q_t = np.zeros((self.state_dim, self.state_dim))
        Q_f = np.diag(np.exp(log_vols.flatten()))  # Factor process noise
        Q_h = self.params.Q_h  # Log-volatility process noise
        Q_t[:K, :K] = Q_f
        Q_t[K:, K:] = Q_h
        predicted_cov = F_t @ cov @ F_t.T + Q_t

        # Ensure the covariance is symmetric
        predicted_cov = (predicted_cov + predicted_cov.T) / 2

        return predicted_state, predicted_cov

    def update(
        self,
        predicted_state: np.ndarray,
        predicted_cov: np.ndarray,
        observation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform the Bellman update step using JAX-based optimization.

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
        # Convert inputs to JAX arrays
        jax_predicted_state = jnp.array(predicted_state)
        jax_observation = jnp.array(observation)

        # Compute information matrix (inverse of predicted covariance)
        try:
            # Use Cholesky for numerical stability
            jax_predicted_cov = jnp.array(predicted_cov)
            L = jax.scipy.linalg.cholesky(jax_predicted_cov, lower=True)
            jax_I_pred = jax.scipy.linalg.cho_solve((L, True), jnp.eye(self.state_dim))
        except:
            # Fallback to regularized pseudoinverse
            jax_I_pred = jnp.linalg.pinv(
                jnp.array(predicted_cov) + 1e-8 * jnp.eye(self.state_dim)
            )

        # Ensure symmetry
        jax_I_pred = (jax_I_pred + jax_I_pred.T) / 2.0

        # Initial guess is the predicted state
        jax_alpha_0 = jnp.array(predicted_state.flatten())

        # Optimize using LBFGS
        result = jax.scipy.optimize.minimize(
            fun=self.jax_bellman_objective,
            x0=jax_alpha_0,
            args=(
                jax_predicted_state,
                jax_I_pred,
                jax_observation,
                self.jax_lambda_r,
                self.jax_sigma2,
            ),
            method="BFGS",
            options={
                "gtol": 1e-4,
                # "disp": False,
                "maxiter": 100,
                # "hess_inv0": jax_I_pred,
            },
        )

        # Run optimization
        # result = solver.run(init_params=jax_alpha_0)

        # Extract optimized state
        optimal_params = result.x
        updated_state = np.array(optimal_params).reshape(-1, 1)

        # Compute the Hessian at the optimum for covariance estimation
        hessian = np.array(
            self.jax_hessian(
                jnp.array(updated_state),
                jax_observation,
            )
        ).reshape(self.state_dim, self.state_dim)
        updated_cov = np.linalg.inv(hessian + jax_I_pred)

        # Ensure symmetry
        updated_cov = (updated_cov + updated_cov.T) / 2.0

        # Compute log-likelihood contribution
        # val = self.jax_bellman_objective(
        #     jnp.array(updated_state),
        #     jax_predicted_state,
        #     jax_I_pred,
        #     jax_observation,
        #     self.jax_lambda_r,
        #     self.jax_sigma2,
        # )
        val = result.fun
        log_likelihood = -float(
            val
        )  # Negate because we minimize negative log-posterior

        return updated_state, updated_cov, log_likelihood

    def _get_transition_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        Get the state transition matrix for the DFSV model.

        Parameters
        ----------
        state : np.ndarray
            Current state vector with shape (state_dim, 1)

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

    def filter(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the Bellman filter on the provided data.

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
        for t in tqdm(range(T), desc="Bellman Filter Progress"):
            # Prediction step
            predicted_state, predicted_cov = self.predict(state, cov)

            # Update step - reshape observation to (N, 1)
            observation = y[t : t + 1, :].T.reshape(-1, 1)
            state, cov, ll_contrib = self.update(
                predicted_state, predicted_cov, observation
            )

            # Store results (convert state from column vector to row vector)
            filtered_states[t, :] = state.flatten()
            filtered_covs[t, :, :] = cov
            log_likelihood += ll_contrib

        # Store results in object
        self.filtered_states = filtered_states
        self.filtered_covs = filtered_covs
        self.log_likelihood = log_likelihood
        self.is_filtered = True

        return filtered_states, filtered_covs, log_likelihood
