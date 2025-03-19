"""
JAX-compatible parameter structures for Dynamic Factor Stochastic Volatility models.

This module provides PyTree-compatible classes to store DFSV model parameters,
enabling efficient JAX transformations such as jit, grad, and vmap.
"""

from typing import ClassVar, Dict, Tuple, Any
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from functions.simulation import DFSV_params
import jax_dataclasses as jdc


class DFSVParamsPytree:
    """
    JAX-compatible pytree class for Dynamic Factor Stochastic Volatility model parameters.

    This class implements a custom PyTree for DFSV parameters, enabling efficient JAX
    transformations such as jit, grad, and vmap. Parameters are stored as JAX arrays
    with N and K as static parameters excluded from differentiation.

    Attributes:
        N (int): Number of observed time series (static)
        K (int): Number of latent factors (static)
        lambda_r (jnp.ndarray): Factor loadings matrix with shape (N, K)
        Phi_f (jnp.ndarray): Factor state transition matrix with shape (K, K)
        Phi_h (jnp.ndarray): Log-volatility state transition matrix with shape (K, K)
        mu (jnp.ndarray): Long-run mean for log-volatilities with shape (K,)
        sigma2 (jnp.ndarray): Idiosyncratic variance with shape (N,) for diagonal or (N, N) for full
        Q_h (jnp.ndarray): Noise covariance matrix for log-volatilities with shape (K, K)
    """

    # Class variables to specify which attributes are pytree nodes vs auxiliary data
    # N and K are auxiliary data (static parameters)
    _node_fields: ClassVar[Tuple[str, ...]] = (
        "lambda_r",
        "Phi_f",
        "Phi_h",
        "mu",
        "sigma2",
        "Q_h",
    )
    _aux_fields: ClassVar[Tuple[str, ...]] = ("N", "K")

    def __init__(
        self,
        N: int,
        K: int,
        lambda_r: jnp.ndarray,
        Phi_f: jnp.ndarray,
        Phi_h: jnp.ndarray,
        mu: jnp.ndarray,
        sigma2: jnp.ndarray,
        Q_h: jnp.ndarray,
    ):
        """
        Initialize the parameters and ensure arrays are JAX-compatible.

        Args:
            N (int): Number of observed time series.
            K (int): Number of latent factors.
            lambda_r (array-like): Factor loadings matrix, will be converted to jnp.ndarray.
            Phi_f (array-like): Factor transition matrix, will be converted to jnp.ndarray.
            Phi_h (array-like): Log-volatility transition matrix, will be converted to jnp.ndarray.
            mu (array-like): Long-run mean for log-volatilities, will be converted to jnp.ndarray.
            sigma2 (array-like): Idiosyncratic variance, will be converted to jnp.ndarray.
            Q_h (array-like): Noise covariance for log-volatilities, will be converted to jnp.ndarray.
        """
        # Store integer parameters as static values
        self.N = N
        self.K = K

        # Convert all arrays to JAX arrays
        self.lambda_r = lambda_r
        self.Phi_f = Phi_f
        self.Phi_h = Phi_h

        self.mu = mu

        self.sigma2 = sigma2

        self.Q_h = Q_h

    def get_sigma2_matrix(self) -> jnp.ndarray:
        """
        Get sigma2 as a full covariance matrix.

        Returns:
            jnp.ndarray: Full sigma2 covariance matrix with shape (N, N)
        """
        if self.sigma2.ndim == 1:
            return jnp.diag(self.sigma2)
        return self.sigma2

    @classmethod
    def from_dfsv_params(cls, params: DFSV_params) -> "DFSVParamsPytree":
        """
        Create a DFSVParamsPytree from a DFSV_params instance.

        Args:
            params (DFSV_params): Original parameters object

        Returns:
            DFSVParamsPytree: New JAX-compatible parameters
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
        # Convert JAX arrays to NumPy arrays
        return DFSVParamsDataclass(
            N=self.N,
            K=self.K,
            lambda_r=jnp.array(self.lambda_r),
            Phi_f=jnp.array(self.Phi_f),
            Phi_h=jnp.array(self.Phi_h),
            mu=jnp.array(self.mu),
            sigma2=jnp.array(self.sigma2),
            Q_h=jnp.array(self.Q_h),
        )

    def replace(self, **kwargs) -> "DFSVParamsPytree":
        """
        Create a new parameters object with updated values.

        Args:
            **kwargs: New parameter values to update

        Returns:
            DFSVParamsPytree: New parameters object with updates applied
        """
        # Create a dict of all current parameters
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
        return DFSVParamsPytree(**param_dict)

    # JAX PyTree methods
    def tree_flatten(self):
        """
        Flatten the pytree for JAX transformations.

        Returns:
            tuple: (children, aux_data)
        """
        # Only parameters included in _node_fields are differentiable
        children = tuple(getattr(self, field) for field in self._node_fields)
        # N and K are static (non-differentiable) parameters
        aux_data = {field: getattr(self, field) for field in self._aux_fields}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct the pytree from flattened data.

        Args:
            aux_data: Auxiliary data (N, K)
            children: Children nodes (differentiable parameters)

        Returns:
            DFSVParamsPytree: Reconstructed object
        """
        fields = {name: value for name, value in zip(cls._node_fields, children)}
        fields.update(aux_data)
        return cls(**fields)


# Register the custom pytree class with JAX
jax.tree_util.register_pytree_node_class(DFSVParamsPytree)


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
