"""
Simulation functions for Dynamic Factor Stochastic Volatility models.

This module provides classes and utilities to define and validate parameters
for DFSV (Dynamic Factor Stochastic Volatility) models in financial time series analysis.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class DFSV_params:

    """
    Parameters for a Dynamic Factor Stochastic Volatility model.
    
    This class stores and validates all necessary parameters for a DFSV model,
    ensuring proper dimensions and compatibility between parameters.
    
    Parameters
    ----------
    N : int
        Number of observed time series (dimensionality of returns)
    K : int
        Number of latent factors
    lambda_r : np.ndarray
        Factor loadings matrix with shape (N, K)
    Phi_f : np.ndarray
        Factor state transition matrix with shape (K, K)
    Phi_h : np.ndarray
        State transition matrix for log-volatilities with shape (K, K)
    mu : np.ndarray
        Long-run mean for log-volatilities with shape (K, 1) or (K,)
    sigma2 : np.ndarray
        Idiosyncratic variance with shape (N, 1), (N,) for diagonal or (N, N) for full covariance
    Q_h : np.ndarray
        Noise covariance matrix for log-volatilities with shape (K, K)
    validate : bool, optional
        Whether to validate parameter dimensions. Defaults to True.
    """

    # Model size
    N: int
    K: int
    # Factor loadings (N x K)
    lambda_r: np.ndarray
    # Factor state transition matrix (K x K)
    Phi_f: np.ndarray
    # State transition matrix for log-volatilities (K x K)
    Phi_h: np.ndarray
    # Long-run mean for log-volatilities (K x 1)
    mu: np.ndarray
    # Idiosyncratic variance (N x 1)
    sigma2: np.ndarray
    # Noise covariance matrix for log-volatilities (K x K)
    Q_h: np.ndarray
    # Optional validation flag - default to True for safety
    validate: bool = True

    def __post_init__(self):
        """
        Validate dimensions of all parameters after initialization.
        
        Raises
        ------
        ValueError
            If any parameter has incorrect dimensions and validate=True
        """
        # Skip validation if validate is False
        if not self.validate:
            # Still convert sigma2 to diagonal matrix if it's 1D for consistency
            if self.sigma2.ndim == 1:
                self.sigma2 = np.diag(self.sigma2)
            return
            
        # Check dimensions of arrays
        if self.lambda_r.shape != (self.N, self.K):
            raise ValueError(f"lambda_r should be shape ({self.N}, {self.K}), got {self.lambda_r.shape}")
            
        if self.Phi_f.shape != (self.K, self.K):
            raise ValueError(f"Phi_f should be shape ({self.K}, {self.K}), got {self.Phi_f.shape}")
            
        if self.Phi_h.shape != (self.K, self.K):
            raise ValueError(f"Phi_h should be shape ({self.K}, {self.K}), got {self.Phi_h.shape}")
            
        # Allow both (K,1) or (K,) for the mean vector
        if self.mu.shape not in [(self.K, 1), (self.K,)]:
            raise ValueError(f"mu should be shape ({self.K}, 1) or ({self.K},), got {self.mu.shape}")
            
        # Allow (N,N), (N,) or (N,1) for the idiosyncratic variance
        if self.sigma2.shape not in [(self.N, 1), (self.N,), (self.N, self.N)]:
            raise ValueError(f"sigma2 should be shape ({self.N}, 1) or ({self.N},) or ({self.N},{self.N}), got {self.sigma2.shape}")
        
        # Check dimensions for Q_h (noise covariance matrix for log-volatilities)
        if self.Q_h.shape != (self.K, self.K):
            raise ValueError(f"Q_h should be shape ({self.K}, {self.K}), got {self.Q_h.shape}")
        
        # If sigma2 is 1D, convert to diagonal matrix
        if self.sigma2.ndim == 1:
            self.sigma2 = np.diag(self.sigma2)
            

def simulate_DFSV(params: DFSV_params, f0: np.ndarray = None, h0: np.ndarray = None, T: int = 100, seed: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a Dynamic Factor Stochastic Volatility model.
    
    Parameters
    ----------
    params : DFSV_params
        Parameters for the DFSV model, should be modelled using the DFSV_params class.
    f0 : np.ndarray, optional
        Initial state for latent factors with shape (K,). If None, defaults to zero.
    h0 : np.ndarray, optional
        Initial state for log-volatilities with shape (K,). If None, defaults to the long-run mean.
    T : int, optional
        Number of time steps to simulate. Defaults to 100.
    seed : int, optional
        Seed for the random number generator. Defaults to None.
        
    Returns
    -------
    tuple
        Contains:
        returns : np.ndarray
            Simulated returns with shape (N, T)
        factors : np.ndarray
            Simulated latent factors with shape (K, T)
        log_vols : np.ndarray
            Simulated log-volatilities with shape (K, T)
    """
    # Unpack parameters
    N, K = params.N, params.K
    lambda_r, Phi_f, Phi_h = params.lambda_r, params.Phi_f, params.Phi_h
    mu, sigma2, Q_h = params.mu, params.sigma2, params.Q_h

    # Reshape mu to ensure it's a column vector (K,1) if it's flat
    if mu.ndim == 1:
        mu = mu.reshape(-1, 1)
    
    # Initialize arrays
    factors_t = np.zeros((K, T))
    log_vols_t = np.zeros((K, T))
    returns_t = np.zeros((N, T))
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Set initial states using pythonic defaults
    factors_t[:, 0] = f0 if f0 is not None else np.zeros(K)
    log_vols_t[:, 0] = h0 if h0 is not None else mu.flatten()
    
    # Prepare Cholesky decompositions for efficiency
    chol_Q_h = np.linalg.cholesky(Q_h)
    chol_sigma2 = np.linalg.cholesky(sigma2)
    
    # Simulate latent factors and log-volatilities
    for t in range(1, T):
        # Log-volatility transition with proper covariance
        # h_t = mu + Phi_h(h_{t-1} - mu) + eta_t, where eta_t ~ N(0, Q_h)
        h_deviation = log_vols_t[:, t-1:t] - mu  # Shape: (K,1)
        log_vols_t[:, t] = mu.flatten() + (Phi_h @ h_deviation).flatten() + (chol_Q_h @ np.random.normal(size=(K, 1))).flatten()
        
        # Factor transition with stochastic volatility
        # f_t = Phi_f*f_{t-1} + diag(exp(h_t/2))*eps_t, where eps_t ~ N(0, I_K)
        vol_scale = np.exp(log_vols_t[:, t-1] / 2)  # Shape: (K,)
        factors_t[:, t] = (Phi_f @ factors_t[:, t-1:t]).flatten() + vol_scale * np.random.normal(size=K)
        
        # Returns equation: r_t = lambda_r*f_t + e_t, where e_t ~ N(0, Sigma)
        returns_t[:, t] = (lambda_r @ factors_t[:, t:t+1]).flatten() + (chol_sigma2 @ np.random.normal(size=(N, 1))).flatten()
    
    return returns_t, factors_t, log_vols_t
