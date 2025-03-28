"""
JAX-compatible parameter structures for Dynamic Factor Stochastic Volatility models.

This module provides PyTree-compatible classes to store DFSV model parameters,
enabling efficient JAX transformations such as jit, grad, and vmap.
"""

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np

from functions.simulation import DFSV_params


@jdc.pytree_dataclass
class DFSVParamsDataclass:
    """
    Alternative JAX-compatible pytree implementation using dataclass for DFSV parameters.

    This class provides an alternative implementation of the JAX-compatible pytree
    for DFSV parameters, using Python dataclasses for cleaner syntax. It automatically
    works as a JAX pytree with N and K excluded from differentiation.

    Note: This class is provided as an alternative to DFSVParamsPytree. Both can
    be used interchangeably depending on your preference.
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
            lambda_r=params.lambda_r,
            Phi_f=params.Phi_f,
            Phi_h=params.Phi_h,
            mu=params.mu,
            sigma2=params.sigma2,
            Q_h=params.Q_h,
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

    def to_dict(self) -> dict:
        """
        Convert the parameter object to a dictionary.
        
        Returns:
            dict: Dictionary representation of parameters
        """
        return {
            "lambda_r": self.lambda_r,
            "Phi_f": self.Phi_f, 
            "Phi_h": self.Phi_h,
            "mu": self.mu,
            "sigma2": self.sigma2,
            "Q_h": self.Q_h
        },self.N,self.K
        
    @classmethod
    def from_dict(cls, param_dict: dict,N,K) -> "DFSVParamsDataclass":
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



def dfsv_params_to_dict(params: DFSVParamsDataclass) -> dict:
    """
    Convert DFSVParamsDataclass to a dictionary.
    
    A convenience function that handles both DFSVParamsDataclass instances
    and regular dictionaries.
    
    Args:
        params: Parameter object or dictionary
        
    Returns:
        dict: Dictionary of parameters
    """
    if isinstance(params, dict):
        return params
    elif isinstance(params, DFSVParamsDataclass):
        return params.to_dict()
    else:
        raise TypeError(f"Unsupported parameter type: {type(params)}")

