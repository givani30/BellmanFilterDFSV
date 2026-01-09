"""Optimization utilities for DFSV model parameter estimation.

This module provides:
- Optimization orchestration functions
- Objective function definitions
- Parameter transformations
- Solver configurations
- EM algorithm for parameter estimation
"""

from ._em_estep import accumulate_sufficient_stats, compute_exp_neg_h
from ._em_mstep import (
    m_step_full,
    update_lambda_r,
    update_mu,
    update_Phi_f,
    update_Phi_h,
    update_Q_h,
    update_sigma2,
)
from ._em_suffstats import EMSufficientStats, SmoothedLagMoments, SmoothedMoments
from .em import EMHistory, EMOptimizer
from .objectives import bellman_objective, pf_objective
from .optimization import run_optimization
from .solvers import create_optimizer
from .transformations import transform_params, untransform_params

__all__ = [
    "run_optimization",
    "create_optimizer",
    "transform_params",
    "untransform_params",
    "bellman_objective",
    "pf_objective",
    "EMOptimizer",
    "EMHistory",
    "EMSufficientStats",
    "SmoothedMoments",
    "SmoothedLagMoments",
    "accumulate_sufficient_stats",
    "compute_exp_neg_h",
    "m_step_full",
    "update_lambda_r",
    "update_sigma2",
    "update_Phi_f",
    "update_Phi_h",
    "update_mu",
    "update_Q_h",
]
