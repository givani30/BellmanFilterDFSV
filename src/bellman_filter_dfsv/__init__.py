"""BellmanFilterDFSV: Dynamic Factor Stochastic Volatility Models with JAX

A high-performance Python package for filtering and parameter estimation in
Dynamic Factor Stochastic Volatility (DFSV) models using JAX.

v2.0.0 - Complete rewrite with functional architecture using Equinox.

Key Features:
- Functional Core + Equinox architecture for JAX
- Bellman Information Filter and Particle Filter
- RTS Smoother and Rao-Blackwellized Particle Smoother
- Maximum Likelihood and EM algorithm estimation
- Full type safety with jaxtyping
- JIT compilation compatible

Example:
    >>> import jax.numpy as jnp
    >>> from bellman_filter_dfsv import DFSVParams, BellmanFilter, fit_mle
    >>>
    >>> # Define model parameters
    >>> params = DFSVParams(
    ...     lambda_r=jnp.array([[0.8], [0.7], [0.9]]),
    ...     Phi_f=jnp.array([[0.7]]),
    ...     Phi_h=jnp.array([[0.95]]),
    ...     mu=jnp.array([-1.2]),
    ...     sigma2=jnp.array([0.3, 0.25, 0.35]),
    ...     Q_h=jnp.array([[0.01]])
    ... )
    >>>
    >>> # Create filter and run on data
    >>> filter = BellmanFilter(params)
    >>> result = filter(returns)
    >>>
    >>> # Estimate parameters from data
    >>> fitted_params, history = fit_mle(returns, initial_params)
"""

__version__ = "2.0.0"
__author__ = "Givani Boekestijn"
__email__ = "givaniboek@hotmail.com"

from .estimation import fit_em, fit_mle
from .filters import BellmanFilter, ParticleFilter
from .simulation import simulate_DFSV, simulate_dfsv
from .smoothing import SmootherResult, rts_smoother, run_rbps
from .types import (
    BIFState,
    DFSVParams,
    FilterResult,
    ParticleFilterResult,
    ParticleState,
    RBParticleState,
    RBPSResult,
)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "DFSVParams",
    "BIFState",
    "FilterResult",
    "ParticleState",
    "ParticleFilterResult",
    "RBParticleState",
    "RBPSResult",
    "BellmanFilter",
    "ParticleFilter",
    "rts_smoother",
    "SmootherResult",
    "run_rbps",
    "fit_mle",
    "fit_em",
    "simulate_dfsv",
    "simulate_DFSV",
]
