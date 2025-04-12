#!/usr/bin/env python
"""
Run a single optimization replicate for the batch optimization study.

This script parses command-line arguments for a single replicate run,
to be used in the batch optimization study framework.

Phase 1, Steps 1 & 2: Implements argument parsing and imports only.
"""

import os
import sys
import time
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
from bellman_filter_dfsv.utils.solvers import get_available_optimizers
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
    available_optimizers=get_available_optimizers()
    parser.add_argument('--N', type=int, required=True, help='Number of assets.')
    parser.add_argument('--K', type=int, required=True, help='Number of factors.')
    parser.add_argument('--T', type=int, required=True, help='Time series length.')
    parser.add_argument('--filter_type', type=str, required=True, choices=['BIF','BF', 'PF'],
                        help="Filter type to use ('BIF' or 'PF').")
    parser.add_argument('--optimizer_name', type=str, required=True,
                        choices=available_optimizers.keys(),
                        help=f"Optimizer to use ({', '.join(available_optimizers.keys())}).")
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



def _calculate_param_errors(true_params, est_params):
    """Calculate RMSE for each parameter field between true and estimated params."""
    def safe_rmse(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            return float('nan')
        return float(np.sqrt(np.nanmean((a - b) ** 2)))
    
    def safe_mean_error(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            return float('nan')
        return float(np.nanmean(b - a))  # estimated - true
        
    errors = {}
    for field in ['lambda_r', 'Phi_f', 'Phi_h', 'sigma2', 'Q_h']:
        try:
            true_val = getattr(true_params, field)
            est_val = getattr(est_params, field)
            errors[field + '_rmse'] = safe_rmse(true_val, est_val)
            errors[field + '_mean_error'] = safe_mean_error(true_val, est_val)
        except Exception:
            errors[field + '_rmse'] = float('nan')
            errors[field + '_mean_error'] = float('nan')
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
    if isinstance(obj, (np.float32, np.float64, float)):
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
    returns, true_factors, true_log_vols = simulate_DFSV(
        true_params, T=args.T, seed=args.replicate_seed + 1
    )
    # Convert to jax array for consistency
    returns = jnp.asarray(returns)

    # Determine if mu should be fixed
    true_params_for_opt = true_params if args.fix_mu else None

    # Map filter_type string to FilterType enum
    filter_type_enum = FilterType[args.filter_type]

    # Only pass num_particles if filter_type is PF, else None
    num_particles = args.num_particles if args.filter_type == "PF" else None

    # Create initial parameters for optimization
    initial_params_for_opt = create_stable_initial_params(N=args.N, K=args.K)

    # Run optimization
    result = None
    start_opt_call_time = time.time()
    optimization_error_msg = None # Initialize error message
    try:
        result = run_optimization(
            filter_type=filter_type_enum,
            returns=returns,
            initial_params=initial_params_for_opt,
            true_params=true_params_for_opt,
            fix_mu=args.fix_mu,
            optimizer_name=args.optimizer_name,
            priors=None,
            stability_penalty_weight=args.stability_penalty,
            max_steps=args.max_steps,
            num_particles=num_particles,
            verbose=True
        )
    except Exception as e:
        print(f"Error during optimization: {e}")
        result = None
        optimization_error_msg = str(e) # Capture the error message
    finally:
        end_opt_call_time = time.time()
        # Note: run_optimization_call_duration_s is calculated here but result.time_taken should be preferred for opt time.
        run_optimization_call_duration_s = end_opt_call_time - start_opt_call_time

    # Initialize accuracy metrics to NaN
    factor_rmse = float('nan')
    factor_corr = float('nan')
    vol_rmse = float('nan')
    vol_corr = float('nan')
    param_errors = {} # Initialize param_errors before the try block

    # Calculate accuracies if we have valid parameters, regardless of formal convergence
    if result is not None and result.final_params is not None:
        print("Starting state estimation accuracy calculation...")
        try:
            # Create appropriate filter instance
            if args.filter_type == "BIF":
                filter_instance = DFSVBellmanInformationFilter(N=args.N, K=args.K)
            else:  # PF
                filter_instance = DFSVParticleFilter(
                    N=args.N, K=args.K,
                    num_particles=args.num_particles,
                    seed=args.replicate_seed + 2  # Different seed than simulation
                )

            # Run filter with estimated parameters
            filtered_states, _, _ = filter_instance.filter(observations=returns, params=result.final_params)

            # Get filtered states
            filtered_factors = filtered_states[:, :args.K]
            filtered_log_vols = filtered_states[:, args.K:]

            # Calculate accuracies
            factor_rmse, factor_corr = calculate_accuracy(true_factors, filtered_factors)
            vol_rmse, vol_corr = calculate_accuracy(true_log_vols, filtered_log_vols)
            print(f"State estimation accuracy metrics calculated successfully:")
            # Print mean values using np.nanmean
            print(f"Factor Mean RMSE: {np.nanmean(factor_rmse):.4f}, Mean Correlation: {np.nanmean(factor_corr):.4f}")
            print(f"Log-Vol Mean RMSE: {np.nanmean(vol_rmse):.4f}, Mean Correlation: {np.nanmean(vol_corr):.4f}")

            # Calculate parameter estimation errors if optimization was successful
            param_errors = _calculate_param_errors(true_params, result.final_params)
            print("Parameter estimation errors calculated successfully:")
            for param, error in param_errors.items():
                 # Revert to float() as _calculate_param_errors returns scalars
                print(f"{param}: {float(error):.4f}")

            # Calculate loss at true parameters
            loss_at_true = _get_loss_at_true_params(true_params, returns, args)
            # Use .item() here too, just in case _get_loss_at_true_params returns an array
            print(f"Loss at true parameters: {loss_at_true:.4f}")


        except Exception as e:
            print(f"Error during accuracy calculation: {e}")
            # Metrics remain as NaN due to initialization above
    
    
    # Prepare and Save Results
    print("\nPreparing results for saving...")
    
    # Initialize filesystem
    fs = gcsfs.GCSFileSystem() if args.output_dir.startswith("gs://") else None
    if not fs and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Construct base filename from config details
    base_filename = f"replicate_{args.N}N_{args.K}K_{args.T}T_{args.filter_type}_{args.optimizer_name}"
    base_filename += f"_P{args.num_particles}" if args.num_particles else ""
    base_filename += f"_fixmu" if args.fix_mu else ""
    base_filename += f"_seed{args.replicate_seed}"
    
    # Assemble metrics dictionary
    metrics_dict = {
        "config": vars(args),
        "results": {
            "success": result.success if result else False,
            "final_loss": float(result.final_loss) if result else float('inf'), # Use result.final_loss directly
            "steps": result.steps if result else -1, # Use -1 if result is None
            "optimization_time_s": result.time_taken if result else -1.0, # Use time_taken from result object
            "error_message": result.error_message if result and result.error_message else optimization_error_msg, # Use captured error if result is None
            "result_code": str(result.result_code) if result and result.result_code else None, # Convert enum/object to string
            "loss_at_true_params": loss_at_true if 'loss_at_true' in locals() else float('nan') # Keep this field
        },
        "accuracy": {
            "state_estimation": {
                "factor_rmse": factor_rmse,
                "factor_correlation": factor_corr,
                "volatility_rmse": vol_rmse,
                "volatility_correlation": vol_corr
            },
            "parameter_estimation": param_errors if 'param_errors' in locals() else {}
        }
    }
    
    # Assemble parameters dictionary
    params_to_save = {
        'true_params': true_params,
        'estimated_params': result.final_params if result and result.final_params is not None else None,
        'optimization_status': {
            'success': result.success if result else False,
            'result_code': str(result.result_code) if result else "FAILED"
        }
    }
    
    # Construct full filepaths
    metrics_filepath = os.path.join(args.output_dir, f"{base_filename}_metrics.json")
    params_filepath = os.path.join(args.output_dir, f"{base_filename}_params.pkl")
    
    # Save results
    try:
        save_metrics_json(metrics_dict, metrics_filepath, fs)
        save_params_pkl(params_to_save, params_filepath, fs)
        print(f"Results saved successfully:\n  {metrics_filepath}\n  {params_filepath}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print("\nReplicate finished.")

if __name__ == "__main__":
    main()