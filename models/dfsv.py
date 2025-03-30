"""
Parameter classes for Dynamic Factor Stochastic Volatility (DFSV) models.

This module provides both NumPy-based and JAX-compatible parameter classes for
DFSV models, supporting standard simulation and JAX-optimized computation.
"""

from dataclasses import dataclass, field
import numpy as np
import jax.numpy as jnp
import jax_dataclasses as jdc


@dataclass
class DFSV_params:
    """
    Parameters for a Dynamic Factor Stochastic Volatility model using NumPy arrays.
    
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


@jdc.pytree_dataclass
class DFSVParamsDataclass:
    """
    JAX-compatible pytree dataclass implementation for DFSV parameters.

    This class provides a JAX-compatible pytree for DFSV parameters,
    using Python dataclasses for cleaner syntax. It automatically
    works as a JAX pytree with N and K excluded from differentiation.
    """

    # Static parameters (excluded from PyTree)
    N: jdc.Static[int]
    K: jdc.Static[int]

    # Differentiable parameters
    lambda_r: jnp.ndarray
    Phi_f: jnp.ndarray
    Phi_h: jnp.ndarray
    mu: jnp.ndarray
    sigma2: jnp.ndarray
    Q_h: jnp.ndarray

    @classmethod
    def from_dfsv_params(cls, params: DFSV_params) -> "DFSVParamsDataclass":
        """
        Create a DFSVParamsDataclass from a DFSV_params instance.

        Args:
            params (DFSV_params): Original parameters object

        Returns:
            DFSVParamsDataclass: New JAX-compatible parameters
        """
        return cls(
            N=params.N,
            K=params.K,
            lambda_r=jnp.array(params.lambda_r),
            Phi_f=jnp.array(params.Phi_f),
            Phi_h=jnp.array(params.Phi_h),
            mu=jnp.array(params.mu),
            sigma2=jnp.array(params.sigma2),
            Q_h=jnp.array(params.Q_h),
        )

    def to_dfsv_params(self) -> DFSV_params:
        """
        Convert to a DFSV_params instance.

        Returns:
            DFSV_params: Original-style parameters object
        """
        return DFSV_params(
            N=self.N,
            K=self.K,
            lambda_r=np.array(self.lambda_r),
            Phi_f=np.array(self.Phi_f),
            Phi_h=np.array(self.Phi_h),
            mu=np.array(self.mu),
            sigma2=np.array(self.sigma2),
            Q_h=np.array(self.Q_h),
            validate=True,
        )

    def replace(self, **kwargs) -> "DFSVParamsDataclass":
        """
        Create a new parameters object with updated values.

        Args:
            **kwargs: New parameter values to update

        Returns:
            DFSVParamsDataclass: New parameters object with updates applied
        """
        # Create a dict of all current attributes
        param_dict = {
            "N": self.N,
            "K": self.K,
            "lambda_r": self.lambda_r,
            "Phi_f": self.Phi_f,
            "Phi_h": self.Phi_h,
            "mu": self.mu,
            "sigma2": self.sigma2,
            "Q_h": self.Q_h,
        }

        # Update with new values
        param_dict.update(kwargs)

        # Create a new instance
        return DFSVParamsDataclass(**param_dict)

    def to_dict(self) -> tuple:
        """
        Convert the parameter object to a dictionary.
        
        Returns:
            tuple: (Dictionary representation of parameters, N, K)
        """
        return {
            "lambda_r": self.lambda_r,
            "Phi_f": self.Phi_f, 
            "Phi_h": self.Phi_h,
            "mu": self.mu,
            "sigma2": self.sigma2,
            "Q_h": self.Q_h
        }, self.N, self.K
        
    @classmethod
    def from_dict(cls, param_dict: dict, N: int, K: int) -> "DFSVParamsDataclass":
        """
        Create a DFSVParamsDataclass from a dictionary.
        
        Args:
            param_dict (dict): Dictionary containing parameter values
            N (int): Number of observations
            K (int): Number of factors
            
        Returns:
            DFSVParamsDataclass: New parameters object
            
        Raises:
            KeyError: If required keys are missing from the dictionary
        """
        # Ensure N and K are included in the class
        param_dict["N"] = N
        param_dict["K"] = K
        # Check for required keys
        required_keys = ["N", "K", "lambda_r", "Phi_f", "Phi_h", "mu", "sigma2", "Q_h"]
        missing_keys = [key for key in required_keys if key not in param_dict]
        if missing_keys:
            raise KeyError(f"Missing required parameters: {missing_keys}")
            
        return cls(**param_dict)


def dfsv_params_to_dict(params) -> tuple:
    """
    Convert DFSVParamsDataclass to a dictionary.
    
    A convenience function that handles both DFSVParamsDataclass instances
    and regular dictionaries.
    
    Args:
        params: Parameter object or dictionary
        
    Returns:
        tuple: (Dictionary of parameters, N, K)
    """
    if isinstance(params, dict):
        # Check if the dictionary contains N and K, if so remove them from the dict
        N = params.pop("N", None)
        K = params.pop("K", None)
        return params, N, K
    elif isinstance(params, DFSVParamsDataclass):
        return params.to_dict()
    else:
        raise TypeError(f"Unsupported parameter type: {type(params)}")