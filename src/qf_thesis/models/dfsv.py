"""
Parameter class for Dynamic Factor Stochastic Volatility (DFSV) models.

This module provides a JAX-compatible pytree dataclass for DFSV model parameters,
suitable for use with JAX-based computations like automatic differentiation.
"""

from dataclasses import dataclass, field
import numpy as np
import jax.numpy as jnp
import jax_dataclasses as jdc


# Removed the old NumPy-based DFSV_params class.
# The DFSVParamsDataclass below is now the standard.


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

    # Removed from_dfsv_params method as DFSV_params class is deleted.

    # Removed to_dfsv_params method as DFSV_params class is deleted.

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