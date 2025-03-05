"""
Likelihood functions for Dynamic Factor Stochastic Volatility models.

This module provides functions to compute the log-likelihood for DFSV models,
with special attention to proper expansion of expressions containing logarithms
of exponential terms.
"""

import numpy as np
from sim_functions import DFSV_params


def log_likelihood_observation(y_t: np.ndarray, f_t: np.ndarray, 
                              params: DFSV_params) -> float:
    """
    Compute the log-likelihood of an observation given the factors.
    
    This function expands any logarithm of exponential terms properly.
    
    Parameters
    ----------
    y_t : np.ndarray
        Observation vector at time t with shape (N, 1)
    f_t : np.ndarray
        Factor vector at time t with shape (K, 1)
    params : DFSV_params
        Model parameters
        
    Returns
    -------
    float
        Log-likelihood value
    """
    # Extract parameters
    lambda_r = params.lambda_r
    sigma2 = params.sigma2
    N = params.N
    
    # Mean and residual
    mean_t = lambda_r @ f_t
    residual = y_t - mean_t
    
    # Log-likelihood without any exp inside log
    ll = -0.5 * (
        N * np.log(2 * np.pi) + 
        np.log(np.linalg.det(sigma2)) + 
        residual.T @ np.linalg.inv(sigma2) @ residual
    )[0, 0]
    
    return ll


def log_likelihood_factor_transition(f_t: np.ndarray, f_prev: np.ndarray, 
                                    h_t: np.ndarray, params: DFSV_params) -> float:
    """
    Compute the log-likelihood of factor transition with stochastic volatility.
    
    This function properly expands the log(exp) terms that appear in stochastic
    volatility models.
    
    Parameters
    ----------
    f_t : np.ndarray
        Current factor vector with shape (K, 1)
    f_prev : np.ndarray
        Previous factor vector with shape (K, 1)
    h_t : np.ndarray
        Current log-volatility vector with shape (K, 1)
    params : DFSV_params
        Model parameters
        
    Returns
    -------
    float
        Log-likelihood value
    """
    K = params.K
    Phi_f = params.Phi_f
    
    # Mean
    mean_f = Phi_f @ f_prev
    
    # Variance matrix with stochastic volatility
    # Instead of using exp(h_t) directly in a matrix and then taking log(det),
    # we expand the expression
    
    # The original expression would involve:
    # Sigma = diag(exp(h_t))
    # log(det(Sigma)) = log(prod(exp(h_t))) = sum(h_t)
    
    # The expanded form eliminates the log(exp) pattern
    sum_log_var = np.sum(h_t)
    
    # Standardized residuals
    # Instead of Sigma^(-1/2)*(f_t - mean_f) we use exp(-h_t/2)*(f_t - mean_f)
    std_residuals = (f_t - mean_f) * np.exp(-h_t/2)
    quad_form = np.sum(std_residuals**2)
    
    # Log-likelihood with expanded terms
    ll = -0.5 * (
        K * np.log(2 * np.pi) + 
        sum_log_var +  # This replaces log(det(Sigma))
        quad_form      # This replaces the quadratic form with matrix inversion
    )
    
    return ll[0] if hasattr(ll, '__len__') else ll


def log_likelihood_volatility_transition(h_t: np.ndarray, h_prev: np.ndarray, 
                                       params: DFSV_params) -> float:
    """
    Compute the log-likelihood of log-volatility transition.
    
    Parameters
    ----------
    h_t : np.ndarray
        Current log-volatility vector with shape (K, 1)
    h_prev : np.ndarray
        Previous log-volatility vector with shape (K, 1)
    params : DFSV_params
        Model parameters
        
    Returns
    -------
    float
        Log-likelihood value
    """
    K = params.K
    Phi_h = params.Phi_h
    mu = params.mu.reshape(-1, 1) if params.mu.ndim == 1 else params.mu
    Q_h = params.Q_h
    
    # Mean
    mean_h = mu + Phi_h @ (h_prev - mu)
    
    # Residual
    residual = h_t - mean_h
    
    # Log-likelihood (standard form, no exp inside log here)
    ll = -0.5 * (
        K * np.log(2 * np.pi) + 
        np.log(np.linalg.det(Q_h)) + 
        residual.T @ np.linalg.inv(Q_h) @ residual
    )[0, 0]
    
    return ll


def compute_joint_log_likelihood(y: np.ndarray, f: np.ndarray, h: np.ndarray, 
                              params: DFSV_params) -> float:
    """
    Compute the joint log-likelihood of observations, factors, and log-volatilities.
    
    Parameters
    ----------
    y : np.ndarray
        Observations with shape (N, T)
    f : np.ndarray
        Factors with shape (K, T)
    h : np.ndarray
        Log-volatilities with shape (K, T)
    params : DFSV_params
        Model parameters
        
    Returns
    -------
    float
        Total log-likelihood value
    """
    T = y.shape[1]
    
    # Initialize log-likelihood
    ll_total = 0.0
    
    # Compute log-likelihood for each time step
    for t in range(T):
        # For t=0, we only have the observation likelihood
        if t == 0:
            ll_total += log_likelihood_observation(y[:, t:t+1], f[:, t:t+1], params)
        else:
            # Observation likelihood
            ll_total += log_likelihood_observation(y[:, t:t+1], f[:, t:t+1], params)
            
            # Factor transition likelihood
            ll_total += log_likelihood_factor_transition(
                f[:, t:t+1], f[:, t-1:t], h[:, t:t+1], params
            )
            
            # Log-volatility transition likelihood
            ll_total += log_likelihood_volatility_transition(
                h[:, t:t+1], h[:, t-1:t], params
            )
    
    return ll_total


def kalman_filter_log_likelihood(y: np.ndarray, params: DFSV_params) -> float:
    """
    Compute the marginal log-likelihood of observations using the Kalman filter.
    
    This function integrates out the latent states (factors and log-volatilities)
    to compute the marginal likelihood of the observations.
    
    Parameters
    ----------
    y : np.ndarray
        Observations with shape (N, T)
    params : DFSV_params
        Model parameters
        
    Returns
    -------
    float
        Marginal log-likelihood value
    """
    # Import here to avoid circular imports
    from kalman_filter import DFSVExtendedKalmanFilter
    
    # Initialize Kalman filter
    kf = DFSVExtendedKalmanFilter(params)
    
    # Run filter and return log-likelihood
    _, _, log_likelihood = kf.filter(y)
    
    return log_likelihood
