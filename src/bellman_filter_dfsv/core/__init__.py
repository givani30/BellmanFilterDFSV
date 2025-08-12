"""Core algorithms and models for Dynamic Factor Stochastic Volatility (DFSV) filtering.

This module contains the core implementations of filtering algorithms, model definitions,
and optimization utilities for DFSV models.
"""

# Import main components for easy access
from .filters import (
    DFSVBellmanFilter,
    DFSVBellmanInformationFilter,
    DFSVFilter,
    DFSVParticleFilter,
)
from .models import DFSVParamsDataclass, simulate_DFSV
from .optimization import create_optimizer, run_optimization

__all__ = [
    # Filters
    "DFSVFilter",
    "DFSVBellmanFilter",
    "DFSVBellmanInformationFilter",
    "DFSVParticleFilter",
    # Models
    "DFSVParamsDataclass",
    "simulate_DFSV",
    # Optimization
    "run_optimization",
    "create_optimizer",
]
