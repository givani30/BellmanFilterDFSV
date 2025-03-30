"""
JAX-compatible parameter structures for Dynamic Factor Stochastic Volatility models.

This module is now a re-export of the parameter classes from models.dfsv
for backwards compatibility.
"""

# Re-export the classes from models.dfsv
from models.dfsv import DFSVParamsDataclass, dfsv_params_to_dict, DFSV_params

# No additional code needed as all functionality has been moved to models.dfsv

