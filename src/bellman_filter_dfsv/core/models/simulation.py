"""
Simulation functions for Dynamic Factor Stochastic Volatility models.

This module provides utilities to simulate DFSV models using the parameter
classes defined in the models module.
"""
import numpy as np
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass # Use the JAX dataclass

def simulate_DFSV(params: DFSVParamsDataclass, f0: np.ndarray = None, h0: np.ndarray = None, T: int = 100, seed: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a Dynamic Factor Stochastic Volatility model.
    
    Parameters
    ----------
    params : DFSVParamsDataclass
        Parameters for the DFSV model, using the JAX-compatible dataclass.
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
            Simulated returns with shape (T, N)
        factors : np.ndarray
            Simulated latent factors with shape (T, K)
        log_vols : np.ndarray
            Simulated log-volatilities with shape (T, K)
    """
    # Unpack parameters
    N, K = params.N, params.K
    lambda_r, Phi_f, Phi_h = params.lambda_r, params.Phi_f, params.Phi_h
    mu, sigma2, Q_h = params.mu, params.sigma2, params.Q_h
    # Reshape mu to ensure it's a column vector (K,1) if it's flat
    if mu.ndim == 1:
        mu = mu.reshape(-1, 1)
    
    # Initialize arrays
    factors_t = np.zeros((T, K))
    log_vols_t = np.zeros((T, K))
    returns_t = np.zeros((T, N))
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Set initial states using pythonic defaults
    factors_t[0, :] = f0 if f0 is not None else np.zeros(K)
    #Initialize at long-run mean instead
    log_vols_t[0, :] = mu.flatten() if h0 is None else h0
    
    # Prepare Cholesky decompositions for efficiency
    # Convert JAX arrays to NumPy for np.linalg functions
    chol_Q_h = np.linalg.cholesky(np.asarray(Q_h))
    # Ensure sigma2 is 2D before Cholesky
    sigma2_np = np.asarray(sigma2)
    if sigma2_np.ndim == 1:
        sigma2_np = np.diag(sigma2_np)
    chol_sigma2 = np.linalg.cholesky(sigma2_np)
    
    # Simulate latent factors and log-volatilities
    for t in range(1, T):
        # Log-volatility transition with proper covariance
        # h_t = mu + Phi_h(h_{t-1} - mu) + eta_t, where eta_t ~ N(0, Q_h)
        h_deviation = log_vols_t[t-1, :] - mu.flatten()  # Shape: (K,)
        log_vols_t[t, :] = mu.flatten() + (Phi_h @ h_deviation).flatten() + (chol_Q_h @ np.random.normal(size=(K, 1))).flatten()
        
        # Factor transition with stochastic volatility
        # f_t = Phi_f*f_{t-1} + diag(exp(h_t/2))*eps_t, where eps_t ~ N(0, I_K)
        vol_scale = np.exp(log_vols_t[t, :] / 2)  # Shape: (K,)
        factors_t[t, :] = (Phi_f @ factors_t[t-1, :]).flatten() + vol_scale * np.random.normal(size=K)
        
        # Returns equation: r_t = lambda_r*f_t + e_t, where e_t ~ N(0, Sigma)
        returns_t[t, :] = (lambda_r @ factors_t[t, :]).flatten() + (chol_sigma2 @ np.random.normal(size=(N, 1))).flatten()
    
    return returns_t, factors_t, log_vols_t
