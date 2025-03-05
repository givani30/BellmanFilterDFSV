"""
Nonlinear Filter implementations for Dynamic Factor Stochastic Volatility models.

This module provides filter classes for state estimation in DFSV models,
including Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), Particle Filters (PF) and Bellman Filters (BF)
to handle the nonlinearities introduced by stochastic volatility.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List, Union
from sim_functions import DFSV_params

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
        # Solve: P_h = Phi_h * P_h * Phi_h' + Q_h
        # Using a simple approximation here:
        I_minus_Phi = np.eye(self.K) - self.params.Phi_h @ self.params.Phi_h.T
        P_h = np.linalg.solve(I_minus_Phi, self.params.Q_h)
        
        # Construct block-diagonal covariance
        initial_cov = np.block([
            [P_f, np.zeros((self.K, self.K))],
            [np.zeros((self.K, self.K)), P_h]
        ])
        
        return initial_state, initial_cov
    
    def predict(self, state: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def update(self, 
              predicted_state: np.ndarray, 
              predicted_cov: np.ndarray, 
              observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
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
            Observed returns with shape (N, T)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            Filtered states with shape (state_dim, T),
            filtered covariances with shape (state_dim, state_dim, T),
            and total log-likelihood
        """
        T = y.shape[1]
        
        # Storage for filtered states and covariances
        filtered_states = np.zeros((self.state_dim, T))
        filtered_covs = np.zeros((self.state_dim, self.state_dim, T))
        
        # Initialize
        state, cov = self.initialize_state(y)
        log_likelihood = 0.0
        
        # Forward pass
        for t in range(T):
            # Prediction step
            predicted_state, predicted_cov = self.predict(state, cov)
            
            # Update step
            state, cov, ll_contrib = self.update(predicted_state, predicted_cov, y[:, t:t+1])
            
            # Store results
            filtered_states[:, t:t+1] = state
            filtered_covs[:, :, t] = cov
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
        smoothed_states[:, last_t:last_t+1] = self.filtered_states[:, last_t:last_t+1]
        smoothed_covs[:, :, last_t] = self.filtered_covs[:, :, last_t]
        
        # Backward pass
        for t in range(T-2, -1, -1):
            # Get filtered values at time t
            state_t = self.filtered_states[:, t:t+1]
            cov_t = self.filtered_covs[:, :, t]
            
            # Get the transition matrix at time t (model specific)
            F_t = self._get_transition_matrix(state_t)
            
            # Predict from t to t+1
            pred_state, pred_cov = self._predict_with_matrix(state_t, cov_t, F_t)
            
            # Compute smoother gain
            K_t = cov_t @ F_t.T @ np.linalg.inv(pred_cov)
            
            # Update based on smoothed value at t+1
            smoothed_states[:, t:t+1] = state_t + K_t @ (smoothed_states[:, t+1:t+2] - pred_state)
            smoothed_covs[:, :, t] = cov_t + K_t @ (smoothed_covs[:, :, t+1] - pred_cov) @ K_t.T
        
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
    
    def _predict_with_matrix(self, 
                            state: np.ndarray, 
                            cov: np.ndarray, 
                            transition_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
            Filtered factors with shape (K, T)
        """
        if not self.is_filtered:
            raise RuntimeError("Must run filter first")
        
        # First K rows are the factors
        return self.filtered_states[:self.K, :]
    
    def get_filtered_volatilities(self) -> np.ndarray:
        """
        Return the filtered log-volatilities.
        
        Returns
        -------
        np.ndarray
            Filtered log-volatilities with shape (K, T)
        """
        if not self.is_filtered:
            raise RuntimeError("Must run filter first")
        
        # Last K rows are the log-volatilities
        return self.filtered_states[self.K:, :]
    
    def get_smoothed_factors(self) -> np.ndarray:
        """
        Return the smoothed latent factors.
        
        Returns
        -------
        np.ndarray
            Smoothed factors with shape (K, T)
        """
        if not self.is_smoothed:
            raise RuntimeError("Must run smoother first")
        
        # First K rows are the factors
        return self.smoothed_states[:self.K, :]
    
    def get_smoothed_volatilities(self) -> np.ndarray:
        """
        Return the smoothed log-volatilities.
        
        Returns
        -------
        np.ndarray
            Smoothed log-volatilities with shape (K, T)
        """
        if not self.is_smoothed:
            raise RuntimeError("Must run smoother first")
        
        # Last K rows are the log-volatilities
        return self.smoothed_states[self.K:, :]


class DFSVExtendedKalmanFilter(DFSVFilter):
    """
    Extended Kalman Filter implementation for DFSV models.
    
    This class specializes the base Kalman filter by implementing
    linearization of the state transition and observation equations
    necessary for handling the nonlinearities in the DFSV model.
    """
    
    def predict(self, state: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the EKF prediction step.
        
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
        mu = self.params.mu.reshape(-1, 1) if self.params.mu.ndim == 1 else self.params.mu
        Q_h = self.params.Q_h
        
        # Extract current state components
        factors = state[:K]
        log_vols = state[K:]
        
        # Predict factors:
        # E[f_{t+1}|t] = Phi_f * f_t
        predicted_factors = Phi_f @ factors
        
        # Predict log-volatilities:
        # E[h_{t+1}|t] = mu + Phi_h * (h_t - mu)
        predicted_log_vols = mu + Phi_h @ (log_vols - mu)
        
        # Combine predicted state
        predicted_state = np.vstack([predicted_factors, predicted_log_vols])
        
        # Get linearized state transition matrix
        F_t = self._get_transition_matrix(state)
        
        # Process noise covariance
        Q_t = np.zeros((self.state_dim, self.state_dim))
        # Factor process noise (state-dependent)
        Q_f = np.diag(np.exp(log_vols.flatten()))
        # Log-volatility process noise (assumed constant)
        Q_t[:K, :K] = Q_f
        Q_t[K:, K:] = Q_h
        
        # Predict covariance
        predicted_cov = F_t @ cov @ F_t.T + Q_t
        
        return predicted_state, predicted_cov
    
    def update(self, 
              predicted_state: np.ndarray, 
              predicted_cov: np.ndarray, 
              observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform the EKF update step.
        
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
        # Extract parameters
        K = self.K
        N = self.N
        lambda_r = self.params.lambda_r
        sigma2 = self.params.sigma2
        
        # Extract predicted state components
        predicted_factors = predicted_state[:K]
        
        # Expected observation: E[r_t|t-1] = lambda_r * f_t|t-1
        predicted_obs = lambda_r @ predicted_factors
        
        # Innovation (measurement residual)
        innovation = observation - predicted_obs
        
        # Observation matrix (maps from state to measurement)
        H_t = np.zeros((N, self.state_dim))
        H_t[:, :K] = lambda_r
        
        # Innovation covariance
        S_t = H_t @ predicted_cov @ H_t.T + sigma2
        
        # Ensure S_t is positive definite
        S_t = (S_t + S_t.T) / 2
        
        # Kalman gain
        K_t = predicted_cov @ H_t.T @ np.linalg.inv(S_t)
        
        # Updated state estimate
        updated_state = predicted_state + K_t @ innovation
        
        # Updated covariance
        updated_cov = (np.eye(self.state_dim) - K_t @ H_t) @ predicted_cov
        
        # Ensure updated_cov is symmetric positive definite
        updated_cov = (updated_cov + updated_cov.T) / 2
        
        # Log-likelihood contribution
        log_likelihood = -0.5 * (
            N * np.log(2 * np.pi) + 
            np.log(np.linalg.det(S_t)) + 
            innovation.T @ np.linalg.inv(S_t) @ innovation
        )[0, 0]
        
        return updated_state, updated_cov, log_likelihood
    
    def _get_transition_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        Get the linearized state transition matrix at given state.
        
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
    
    def _predict_with_matrix(self, 
                            state: np.ndarray, 
                            cov: np.ndarray, 
                            transition_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        K = self.K
        mu = self.params.mu.reshape(-1, 1) if self.params.mu.ndim == 1 else self.params.mu
        
        # Extract state components
        factors = state[:K]
        log_vols = state[K:]
        
        # Predict factors: E[f_{t+1}|t] = Phi_f * f_t
        predicted_factors = transition_matrix[:K, :K] @ factors
        
        # Predict log-volatilities: E[h_{t+1}|t] = mu + Phi_h * (h_t - mu)
        Phi_h = transition_matrix[K:, K:]
        predicted_log_vols = mu + Phi_h @ (log_vols - mu)
        
        # Combine predicted state
        predicted_state = np.vstack([predicted_factors, predicted_log_vols])
        
        # Process noise covariance (simplified for this method)
        Q_t = np.zeros((self.state_dim, self.state_dim))
        Q_t[K:, K:] = self.params.Q_h
        
        # Predict covariance
        predicted_cov = transition_matrix @ cov @ transition_matrix.T + Q_t
        
        return predicted_state, predicted_cov


class DFSVUnscentedKalmanFilter(DFSVFilter):
    """
    Unscented Kalman Filter implementation for DFSV models.
    
    This class specializes the base Kalman filter using the unscented
    transform to handle nonlinearities without requiring explicit calculation
    of Jacobians, which can be useful for highly nonlinear DFSV variants.
    """
    
    def __init__(self, params: DFSV_params, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        """
        Initialize the UKF with DFSV model parameters.
        
        Parameters
        ----------
        params : DFSV_params
            Parameters of the DFSV model
        alpha : float, optional
            Spread of sigma points around mean
        beta : float, optional
            Prior knowledge of state distribution (2 is optimal for Gaussian)
        kappa : float, optional
            Secondary scaling parameter
        """
        super().__init__(params)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Calculate derived parameters
        self._setup_ukf_params()
    
    def _setup_ukf_params(self):
        """Set up UKF-specific parameters for sigma point calculations."""
        n = self.state_dim
        self.lambda_ukf = self.alpha**2 * (n + self.kappa) - n
        
        # Calculate weights
        # Weight for mean of state
        self.wm = np.zeros(2*n + 1)
        # Weight for covariance
        self.wc = np.zeros(2*n + 1)
        
        # Calculate weights
        self.wm[0] = self.lambda_ukf / (n + self.lambda_ukf)
        self.wc[0] = self.wm[0] + (1 - self.alpha**2 + self.beta)
        for i in range(1, 2*n + 1):
            self.wm[i] = 1 / (2 * (n + self.lambda_ukf))
            self.wc[i] = 1 / (2 * (n + self.lambda_ukf))
    
    def _generate_sigma_points(self, state: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Generate sigma points around the current state.
        
        Parameters
        ----------
        state : np.ndarray
            Current state with shape (state_dim, 1)
        cov : np.ndarray
            Current state covariance with shape (state_dim, state_dim)
            
        Returns
        -------
        np.ndarray
            Sigma points with shape (state_dim, 2*state_dim + 1)
        """
        n = self.state_dim
        sigma_points = np.zeros((n, 2*n + 1))
        
        # Ensure covariance is symmetric
        cov = (cov + cov.T) / 2
        
        # Compute square root of (n + lambda)*P using Cholesky decomposition
        try:
            L = np.linalg.cholesky((n + self.lambda_ukf) * cov)
        except np.linalg.LinAlgError:
            # If Cholesky fails, add a small regularization term
            L = np.linalg.cholesky((n + self.lambda_ukf) * (cov + 1e-8 * np.eye(n)))
        
        # First sigma point is the mean
        sigma_points[:, 0:1] = state
        
        # Generate remaining sigma points
        for i in range(n):
            sigma_points[:, i+1:i+2] = state + L[:, i:i+1]
            sigma_points[:, n+i+1:n+i+2] = state - L[:, i:i+1]
        
        return sigma_points
    
    def _predict_sigma_points(self, sigma_points: np.ndarray) -> np.ndarray:
        """
        Propagate sigma points through the nonlinear state transition function.
        
        Parameters
        ----------
        sigma_points : np.ndarray
            Sigma points with shape (state_dim, 2*state_dim + 1)
            
        Returns
        -------
        np.ndarray
            Propagated sigma points with shape (state_dim, 2*state_dim + 1)
        """
        K = self.K
        Phi_f = self.params.Phi_f
        Phi_h = self.params.Phi_h
        mu = self.params.mu.reshape(-1, 1) if self.params.mu.ndim == 1 else self.params.mu
        
        # Number of sigma points
        num_sigma_points = sigma_points.shape[1]
        
        # Container for transformed sigma points
        transformed_sigma_points = np.zeros_like(sigma_points)
        
        for i in range(num_sigma_points):
            # Extract state components from current sigma point
            factors = sigma_points[:K, i:i+1]
            log_vols = sigma_points[K:, i:i+1]
            
            # Predict factors: E[f_{t+1}|t] = Phi_f * f_t
            predicted_factors = Phi_f @ factors
            
            # Predict log-volatilities: E[h_{t+1}|t] = mu + Phi_h * (h_t - mu)
            predicted_log_vols = mu + Phi_h @ (log_vols - mu)
            
            # Combine predicted state
            transformed_sigma_points[:K, i:i+1] = predicted_factors
            transformed_sigma_points[K:, i:i+1] = predicted_log_vols
        
        return transformed_sigma_points
    
    def _observe_sigma_points(self, sigma_points: np.ndarray) -> np.ndarray:
        """
        Transform sigma points using the observation equation.
        
        Parameters
        ----------
        sigma_points : np.ndarray
            Sigma points with shape (state_dim, 2*state_dim + 1)
            
        Returns
        -------
        np.ndarray
            Observed sigma points with shape (N, 2*state_dim + 1)
        """
        K = self.K
        N = self.N
        lambda_r = self.params.lambda_r
        
        # Number of sigma points
        num_sigma_points = sigma_points.shape[1]
        
        # Container for observed sigma points
        observed_sigma_points = np.zeros((N, num_sigma_points))
        
        for i in range(num_sigma_points):
            # Extract factors from current sigma point
            factors = sigma_points[:K, i:i+1]
            
            # Observation equation: y = lambda_r * f
            observed_sigma_points[:, i:i+1] = lambda_r @ factors
        
        return observed_sigma_points
    
    def predict(self, state: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the UKF prediction step.
        
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
        # Generate sigma points
        sigma_points = self._generate_sigma_points(state, cov)
        
        # Propagate sigma points through state transition function
        transformed_sigma_points = self._predict_sigma_points(sigma_points)
        
        # Compute predicted mean
        predicted_state = np.sum([w * transformed_sigma_points[:, i:i+1] 
                                for i, w in enumerate(self.wm)], axis=0)
        
        # Compute predicted covariance
        predicted_cov = np.zeros((self.state_dim, self.state_dim))
        
        for i in range(len(self.wc)):
            diff = transformed_sigma_points[:, i:i+1] - predicted_state
            predicted_cov += self.wc[i] * diff @ diff.T
        
        # Add process noise covariance
        Q_t = np.zeros((self.state_dim, self.state_dim))
        K = self.K
        
        # Factor process noise (state-dependent)
        log_vols = state[K:]
        Q_f = np.diag(np.exp(log_vols.flatten()))
        
        # Log-volatility process noise
        Q_t[:K, :K] = Q_f
        Q_t[K:, K:] = self.params.Q_h
        
        predicted_cov = predicted_cov + Q_t
        
        return predicted_state, predicted_cov
    
    # def update(self, 
    #           predicted_state: np.ndarray, 
    #           predicted_cov: np.ndarray, 
    #           observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    #     """
    #     Perform the UKF update step