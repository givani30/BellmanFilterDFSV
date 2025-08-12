"""BellmanFilterDFSV: Dynamic Factor Stochastic Volatility Models with JAX

A high-performance Python package for filtering and parameter estimation in
Dynamic Factor Stochastic Volatility (DFSV) models using JAX.

Key Features:
- Multiple filtering algorithms (Bellman Information Filter, Bellman Filter, Particle Filter)
- JAX-based implementation for automatic differentiation and JIT compilation
- Comprehensive optimization framework for parameter estimation
- Clean, extensible API for research and applications

Example:
    >>> import bellman_filter_dfsv as bfdfsv
    >>> from bellman_filter_dfsv.core import DFSVParamsDataclass, simulate_DFSV
    >>>
    >>> # Define model parameters
    >>> params = DFSVParamsDataclass(N=3, K=1, ...)
    >>>
    >>> # Simulate data
    >>> returns, factors, log_vols = simulate_DFSV(params, T=500)
    >>>
    >>> # Create and run filter
    >>> filter = bfdfsv.DFSVBellmanInformationFilter(N=3, K=1)
    >>> states, covs, loglik = filter.filter(params, returns)
"""

# Version information
__version__ = "1.0.0"
__author__ = "Givani Boekestijn"
__email__ = "givaniboek@hotmail.com"

# Core imports for easy access
from .core import (
    DFSVBellmanFilter,
    DFSVBellmanInformationFilter,
    # Filters
    DFSVFilter,
    # Models
    DFSVParamsDataclass,
    DFSVParticleFilter,
    create_optimizer,
    # Optimization
    run_optimization,
    simulate_DFSV,
)

# Utilities
from .utils import analysis, jax_helpers

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Core components
    "DFSVFilter",
    "DFSVBellmanFilter",
    "DFSVBellmanInformationFilter",
    "DFSVParticleFilter",
    "DFSVParamsDataclass",
    "simulate_DFSV",
    "run_optimization",
    "create_optimizer",
    # Utilities
    "jax_helpers",
    "analysis",
]
