#!/usr/bin/env python
"""
Run a single optimization replicate for the batch optimization study.

This script parses command-line arguments for a single replicate run,
to be used in the batch optimization study framework.

Phase 1, Steps 1 & 2: Implements argument parsing and imports only.
"""

import os
import sys
import argparse
import cloudpickle
import json
import gcsfs

import jax
import jax.numpy as jnp
import numpy as np

# Project-specific imports (for future extensibility)
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.utils.optimization import run_optimization, FilterType
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.filters.particle import DFSVParticleFilter

# Refactored helpers
from bellman_filter_dfsv.models.simulation_helpers import create_stable_dfsv_params
from bellman_filter_dfsv.utils.optimization_helpers import create_stable_initial_params
from bellman_filter_dfsv.utils.analysis import calculate_accuracy
from bellman_filter_dfsv.models.simulation import simulate_DFSV

def parse_args():
    """Parse command-line arguments for a single optimization replicate."""
    parser = argparse.ArgumentParser(
        description="Run a single optimization replicate for the batch optimization study."
    )
    parser.add_argument('--N', type=int, required=True, help='Number of assets.')
    parser.add_argument('--K', type=int, required=True, help='Number of factors.')
    parser.add_argument('--T', type=int, required=True, help='Time series length.')
    parser.add_argument('--filter_type', type=str, required=True, choices=['BIF', 'PF'],
                        help="Filter type to use ('BIF' or 'PF').")
    parser.add_argument('--optimizer_name', type=str, required=True,
                        choices=['AdamW', 'DampedTrustRegionBFGS'],
                        help="Optimizer to use ('AdamW' or 'DampedTrustRegionBFGS').")
    parser.add_argument('--num_particles', type=int,
                        help="Number of particles (required if filter_type is 'PF').")
    parser.add_argument('--stability_penalty', type=float, default=1000.0,
                        help="Stability penalty weight (default: 1000.0).")
    parser.add_argument('--max_steps', type=int, default=1000,
                        help="Maximum number of optimization steps (default: 1000).")
    parser.add_argument('--replicate_seed', type=int, required=True,
                        help="Random seed for this replicate.")
    parser.add_argument('--fix_mu', action='store_true',
                        help="Fix the long-run mean parameter mu during optimization.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save outputs for this replicate.")
    parser.add_argument('--save_format', type=str, default='pkl', choices=['pkl', 'npz'],
                        help="Format to save results ('pkl' or 'npz').")

    args = parser.parse_args()

    # Conditional requirement: --num_particles is required if filter_type == 'PF'
    if args.filter_type == 'PF' and args.num_particles is None:
        parser.error("--num_particles is required when --filter_type is 'PF'.")

    return args



def create_training_data(params, T: int = 1000, seed: int = 42):
    """Generate simulated data for training.

    Args:
        params: Model parameters.
        T: Number of time steps.
        seed: Random seed.

    Returns:
        jnp.ndarray: Simulated returns with shape (T, N).
    """
    returns, _, _ = simulate_DFSV(params, T=T, seed=seed)
    return jnp.asarray(returns)

def _calculate_param_errors(true_params, est_params):
    """Calculate RMSE for each parameter field between true and estimated params."""
    def safe_rmse(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            return float('nan')
        return float(np.sqrt(np.nanmean((a - b) ** 2)))
    errors = {}
    for field in ['lambda_r', 'Phi_f', 'Phi_h', 'sigma2', 'Q_h']:
        try:
            true_val = getattr(true_params, field)
            est_val = getattr(est_params, field)
            errors[field + '_rmse'] = safe_rmse(true_val, est_val)
        except Exception:
            errors[field + '_rmse'] = float('nan')
    return errors

def _get_loss_at_true_params(true_params, returns, args):
    """Compute the loss at the true parameters using the same objective as optimization."""
    from bellman_filter_dfsv.utils.optimization import get_objective_function, FilterType, create_filter
    try:
        filter_type_enum = FilterType[args.filter_type]
        filter_instance = create_filter(
            filter_type_enum, args.N, args.K,
            args.num_particles if args.filter_type == "PF" else 5000
        )
        # Use the same settings as optimization
        objective_fn = get_objective_function(
            filter_type=filter_type_enum,
            filter_instance=filter_instance,
            stability_penalty_weight=args.stability_penalty,
            priors=None,
            is_transformed=False,
            fix_mu=args.fix_mu,
            true_mu=true_params.mu if args.fix_mu else None
        )
        loss, _ = objective_fn(true_params, returns)
        return float(loss)
    except Exception as e:
        print(f"Error calculating loss at true parameters: {e}")
        return float('nan')

def _serialize_for_json(obj):
    """Convert JAX/NumPy types to Python types for JSON serialization."""
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.float_, float)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.integer, int)):
        return int(obj)
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_json(x) for x in obj]
    return str(obj)

def save_metrics_json(metrics_dict, filepath, fs=None):
    """Save metrics dictionary to JSON, handling GCS and serialization."""
    data = _serialize_for_json(metrics_dict)
    if fs is not None:
        with fs.open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    else:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

def save_params_pkl(params_dict, filepath, fs=None):
    """Save parameters dictionary to PKL using cloudpickle, handling GCS."""
    if fs is not None:
        with fs.open(filepath, "wb") as f:
            cloudpickle.dump(params_dict, f)
    else:
        with open(filepath, "wb") as f:
            cloudpickle.dump(params_dict, f)

def main():
    """Main function for single replicate optimization script."""
    args = parse_args()

    # Set JAX precision
    jax.config.update("jax_enable_x64", True)

    # Print configuration
    print(f"Running optimization for N={args.N}, K={args.K}, T={args.T}, filter_type={args.filter_type}, "
          f"optimizer={args.optimizer_name}, num_particles={args.num_particles}, "
          f"stability_penalty={args.stability_penalty}, max_steps={args.max_steps}, "
          f"replicate_seed={args.replicate_seed}, fix_mu={args.fix_mu}")

    # Set global random seeds for reproducibility
    np.random.seed(args.replicate_seed)
    # JAX PRNG is handled in create_stable_dfsv_params, but if needed, could be passed as argument in future refactor

    # Generate true parameters
    true_params = create_stable_dfsv_params(N=args.N, K=args.K)

    # Generate simulation data (use a different seed for data)
    returns = create_training_data(true_params, T=args.T, seed=args.replicate_seed + 1)

    # Determine if mu should be fixed
    true_params_for_opt = true_params if args.fix_mu else None

    # Map filter_type string to FilterType enum
    filter_type_enum = FilterType[args.filter_type]

    # Only pass num_particles if filter_type is PF, else None
    num_particles = args.num_particles if args.filter_type == "PF" else None

    # Run optimization
    result = None
    try:
        result = run_optimization(
            filter_type=filter_type_enum,
            returns=returns,
            true_params=true_params_for_opt,
            use_transformations=True,
            optimizer_name=args.optimizer_name,
            priors=None,
            stability_penalty_weight=args.stability_penalty,
            max_steps=args.max_steps,
            num_particles=num_particles,
            verbose=True,
            prior_config_name="No Priors"
        )
    except Exception as e:
        print(f"Error during optimization: {e}")
        result = None
if __name__ == "__main__":
    main()