#!/usr/bin/env python
"""Run a single optimization replicate with configurations loaded from GCS.

This script reads configuration from environment variables and GCS, then executes
a single optimization replicate for the batch optimization study framework.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, List

from google.cloud import storage
from google.api_core import retry
import cloudpickle
import gcsfs

import jax
import jax.numpy as jnp
import numpy as np

# Project-specific imports
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.utils.optimization import run_optimization, FilterType, get_objective_function, create_filter
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.filters.particle import DFSVParticleFilter
from bellman_filter_dfsv.models.simulation_helpers import create_stable_dfsv_params
from bellman_filter_dfsv.utils.optimization_helpers import create_stable_initial_params
from bellman_filter_dfsv.utils.analysis import calculate_accuracy
from bellman_filter_dfsv.models.simulation import simulate_DFSV

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config_from_gcs(config_uri: str, task_index: int) -> Dict[str, Any]:
    """Load configuration from GCS and select task-specific config.

    Args:
        config_uri: GCS URI (gs://bucket-name/path/to/config.json)
        task_index: Index of the configuration to select

    Returns:
        Dictionary containing the selected configuration

    Raises:
        ValueError: If URI is invalid or task_index is out of range
        Exception: For GCS access or JSON parsing errors
    """
    logger.info(f"Loading configuration from {config_uri} for task index {task_index}")
    try:
        # Check if it's a GCS URI or a local path
        if config_uri.startswith('gs://'):
            # GCS Path Logic
            bucket_name = config_uri.split('/')[2]
            blob_path = '/'.join(config_uri.split('/')[3:])

            # Initialize GCS client and get blob
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            # Download and parse JSON with retry
            @retry.Retry(predicate=retry.if_transient_error)
            def download_and_parse():
                logger.info(f"Attempting to download GCS blob: {blob_path}")
                json_str = blob.download_as_string(timeout=60) # Add timeout
                logger.info("GCS Blob downloaded successfully.")
                return json.loads(json_str)

            configs = download_and_parse()
            logger.info("GCS Configuration JSON parsed successfully.")
        else:
            # Local Path Logic
            logger.info(f"Assuming local path: {config_uri}")
            if not os.path.exists(config_uri):
                 raise FileNotFoundError(f"Local configuration file not found: {config_uri}")
            with open(config_uri, 'r') as f:
                configs = json.load(f)
            logger.info("Local configuration JSON parsed successfully.")

        # Validate and select configuration (common logic)
        if not isinstance(configs, list):
            raise ValueError("Config file must contain a list of configurations")
        if task_index >= len(configs):
            raise ValueError(f"Task index {task_index} exceeds config list length {len(configs)}")

        selected_config = configs[task_index]
        logger.info(f"Selected configuration for task {task_index}: {selected_config}")
        return selected_config

    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}", exc_info=True)
        raise

def determine_optimizer(filter_type: str) -> str:
    """Determine the appropriate optimizer based on filter type.

    Args:
        filter_type: Either 'PF' or 'BIF'

    Returns:
        Optimizer name string

    Raises:
        ValueError: If filter_type is not 'PF' or 'BIF'
    """
    if filter_type == 'PF':
        return 'ArmijoBFGS'
    elif filter_type == 'BIF':
        return 'DampedTrustRegionBFGS'
    else:
        raise ValueError(f"Unsupported filter_type for optimizer determination: {filter_type}")

def _calculate_param_errors(true_params, est_params):
    """Calculate RMSE and Mean Error for each parameter field."""
    def safe_rmse(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape: return float('nan')
        return float(np.sqrt(np.nanmean((a - b) ** 2)))

    def safe_mean_error(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape: return float('nan')
        return float(np.nanmean(b - a)) # estimated - true

    errors = {}
    for field in DFSVParamsDataclass.__dataclass_fields__:
        if field == 'mu' and not hasattr(est_params, 'mu'): # Handle fixed mu case
            continue
        try:
            true_val = getattr(true_params, field)
            est_val = getattr(est_params, field)
            errors[field + '_rmse'] = safe_rmse(true_val, est_val)
            errors[field + '_mean_error'] = safe_mean_error(true_val, est_val)
        except Exception as e:
            logger.warning(f"Could not calculate error for field {field}: {e}")
            errors[field + '_rmse'] = float('nan')
            errors[field + '_mean_error'] = float('nan')
    return errors

def _get_loss_at_true_params(true_params, returns, config):
    """Compute the loss at the true parameters using the same objective as optimization."""
    try:
        filter_type_enum = FilterType[config['filter_type']]
        filter_instance = create_filter(
            filter_type_enum, config['N'], config['K'],
            config.get('num_particles') # Use .get for optional num_particles
        )
        objective_fn = get_objective_function(
            filter_type=filter_type_enum,
            filter_instance=filter_instance,
            stability_penalty_weight=config.get('stability_penalty', 0.0), # Use default if missing
            priors=None,
            is_transformed=False,
            fix_mu=config.get('fix_mu', False),
            true_mu=true_params.mu if config.get('fix_mu', False) else None
        )
        loss, _ = objective_fn(true_params, returns)
        return float(loss)
    except Exception as e:
        logger.error(f"Error calculating loss at true parameters: {e}", exc_info=True)
        return float('nan')

def _serialize_for_json(obj):
    """Convert JAX/NumPy types to Python types for JSON serialization."""
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, jnp.float32, jnp.float64, float)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.integer, jnp.int32, jnp.int64, int)):
        return int(obj)
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_json(x) for x in obj]
    # Handle DFSVParamsDataclass specifically if needed, or rely on default str()
    if isinstance(obj, DFSVParamsDataclass):
         # Convert dataclass to dict first
         from dataclasses import asdict
         return _serialize_for_json(asdict(obj))
    return str(obj) # Fallback

def save_metrics_json(metrics_dict, filepath, fs=None):
    """Save metrics dictionary to JSON, handling GCS and serialization."""
    logger.info(f"Saving metrics to JSON: {filepath}")
    try:
        data = _serialize_for_json(metrics_dict)
        if fs is not None:
            with fs.open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        else:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        logger.info("Metrics saved successfully.")
    except Exception as e:
        logger.error(f"Error saving metrics JSON: {e}", exc_info=True)
        raise

def save_params_pkl(params_dict, filepath, fs=None):
    """Save parameters dictionary to PKL using cloudpickle, handling GCS."""
    logger.info(f"Saving parameters to PKL: {filepath}")
    try:
        if fs is not None:
            with fs.open(filepath, "wb") as f:
                cloudpickle.dump(params_dict, f)
        else:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                cloudpickle.dump(params_dict, f)
        logger.info("Parameters saved successfully.")
    except Exception as e:
        logger.error(f"Error saving parameters PKL: {e}", exc_info=True)
        raise

def main():
    """Main function for single replicate optimization script."""
    start_time = time.time()
    logger.info("Starting optimization replicate script.")

    # --- Configuration Loading ---
    try:
        task_index_str = os.environ.get('BATCH_TASK_INDEX')
        config_gcs_uri = os.environ.get('CONFIG_GCS_URI')
        base_output_dir_uri = os.environ.get('BASE_OUTPUT_DIR_URI')

        if not all([task_index_str, config_gcs_uri, base_output_dir_uri]):
            missing = [v for v, k in [('BATCH_TASK_INDEX', task_index_str),
                                      ('CONFIG_GCS_URI', config_gcs_uri),
                                      ('BASE_OUTPUT_DIR_URI', base_output_dir_uri)] if not k]
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        task_index = int(task_index_str)
        logger.info(f"Environment variables loaded: TASK_INDEX={task_index}, CONFIG_URI={config_gcs_uri}, OUTPUT_URI={base_output_dir_uri}")

        # Load the specific configuration for this task
        config = load_config_from_gcs(config_gcs_uri, task_index)

        # Determine optimizer and add to config
        config['optimizer_name'] = determine_optimizer(config['filter_type'])
        logger.info(f"Determined optimizer: {config['optimizer_name']} for filter type {config['filter_type']}")

        # Construct unique output directory URI
        output_dir_suffix = f"task_{task_index:05d}_{config['filter_type']}_N{config['N']}_K{config['K']}_seed{config['replicate_seed']}"
        config['output_dir'] = os.path.join(base_output_dir_uri, output_dir_suffix) # Use os.path.join for GCS paths too
        logger.info(f"Constructed output directory: {config['output_dir']}")

    except Exception as e:
        logger.error(f"Configuration failed: {e}", exc_info=True)
        sys.exit(1) # Exit if configuration fails

    # --- Setup ---
    jax.config.update("jax_enable_x64", True)
    logger.info("JAX configured for x64 precision.")

    # Print final configuration being used
    logger.info(f"Running with configuration: {json.dumps(config, indent=2)}")

    # Set global random seeds for reproducibility
    np.random.seed(config['replicate_seed'])
    # JAX PRNG key generation will be handled within functions needing randomness

    # --- Data Generation ---
    try:
        logger.info("Generating true parameters...")
        true_params = create_stable_dfsv_params(N=config['N'], K=config['K'])
        logger.info("Generating simulation data...")
        returns, true_factors, true_log_vols = simulate_DFSV(
            true_params, T=config['T'], seed=config['replicate_seed'] + 1 # Use different seed for data
        )
        returns = jnp.asarray(returns) # Convert to JAX array
        logger.info(f"Simulation data generated (T={config['T']}).")
    except Exception as e:
        logger.error(f"Data generation failed: {e}", exc_info=True)
        sys.exit(1)

    # --- Optimization ---
    try:
        logger.info("Creating initial parameters for optimization...")
        initial_params_for_opt = create_stable_initial_params(N=config['N'], K=config['K'])

        # Determine if mu should be fixed
        fix_mu = config.get('fix_mu', False)
        true_params_for_opt = true_params if fix_mu else None

        filter_type_enum = FilterType[config['filter_type']]
        num_particles = config.get('num_particles') if config['filter_type'] == "PF" else None

        logger.info(f"Starting optimization with {config['optimizer_name']}...")
        result = None
        start_opt_call_time = time.time()
        optimization_error_msg = None
        try:
            result = run_optimization(
                filter_type=filter_type_enum,
                returns=returns,
                initial_params=initial_params_for_opt,
                true_params=true_params_for_opt, # Pass true params only if fix_mu is True
                fix_mu=fix_mu,
                optimizer_name=config['optimizer_name'],
                priors=None, # Add priors from config if needed later
                stability_penalty_weight=config.get('stability_penalty', 0.0),
                max_steps=config.get('max_steps', 1000),
                num_particles=num_particles,
                verbose=True # Or control via config
            )
        except Exception as e:
            logger.error(f"Error during optimization call: {e}", exc_info=True)
            result = None
            optimization_error_msg = str(e)
        finally:
            end_opt_call_time = time.time()
            run_optimization_call_duration_s = end_opt_call_time - start_opt_call_time
            logger.info(f"run_optimization call finished in {run_optimization_call_duration_s:.2f} seconds.")

        if result:
            logger.info(f"Optimization finished. Success: {result.success}, Steps: {result.steps}, Final Loss: {result.final_loss:.4f}, Time: {result.time_taken:.2f}s")
            if not result.success:
                 logger.warning(f"Optimization did not converge successfully. Result code: {result.result_code}, Message: {result.error_message}")
        else:
            logger.error("Optimization failed to produce a result object.")

    except Exception as e:
        logger.error(f"Optimization setup failed: {e}", exc_info=True)
        sys.exit(1)

    # --- Analysis & Results ---
    factor_rmse, factor_corr = float('nan'), float('nan')
    vol_rmse, vol_corr = float('nan'), float('nan')
    param_errors = {}
    loss_at_true = float('nan')

    if result is not None and result.final_params is not None:
        logger.info("Calculating state estimation accuracy...")
        try:
            filter_instance = create_filter(
                filter_type_enum, config['N'], config['K'], num_particles
            )
            filtered_states, _, _ = filter_instance.filter(observations=returns, params=result.final_params)
            filtered_factors = filtered_states[:, :config['K']]
            filtered_log_vols = filtered_states[:, config['K']:]

            factor_rmse, factor_corr = calculate_accuracy(true_factors, filtered_factors)
            vol_rmse, vol_corr = calculate_accuracy(true_log_vols, filtered_log_vols)
            logger.info(f"State estimation accuracy: Factor RMSE={np.nanmean(factor_rmse):.4f}, Corr={np.nanmean(factor_corr):.4f}; "
                        f"Vol RMSE={np.nanmean(vol_rmse):.4f}, Corr={np.nanmean(vol_corr):.4f}")

            logger.info("Calculating parameter estimation errors...")
            param_errors = _calculate_param_errors(true_params, result.final_params)
            logger.info("Parameter errors calculated.")

            logger.info("Calculating loss at true parameters...")
            loss_at_true = _get_loss_at_true_params(true_params, returns, config)
            logger.info(f"Loss at true parameters: {loss_at_true:.4f}")

        except Exception as e:
            logger.error(f"Error during post-optimization analysis: {e}", exc_info=True)
    else:
        logger.warning("Skipping post-optimization analysis as final parameters are not available.")

    # --- Saving Results ---
    logger.info("Preparing results for saving...")
    fs = gcsfs.GCSFileSystem() if config['output_dir'].startswith("gs://") else None

    # Use the constructed output_dir from config
    base_filename = f"replicate_results" # Simpler base name, details are in the path/metrics

    metrics_dict = {
        "config": config, # Save the loaded and augmented config
        "results": {
            "success": result.success if result else False,
            "final_loss": float(result.final_loss) if result and result.final_loss is not None else float('inf'),
            "steps": result.steps if result else -1,
            "optimization_time_s": result.time_taken if result else -1.0,
            "error_message": result.error_message if result and result.error_message else optimization_error_msg,
            "result_code": str(result.result_code) if result and result.result_code else None,
            "loss_at_true_params": loss_at_true
        },
        "accuracy": {
            "state_estimation": {
                # Save mean values directly for easier aggregation later
                "factor_rmse_mean": float(np.nanmean(factor_rmse)),
                "factor_correlation_mean": float(np.nanmean(factor_corr)),
                "volatility_rmse_mean": float(np.nanmean(vol_rmse)),
                "volatility_correlation_mean": float(np.nanmean(vol_corr))
            },
            "parameter_estimation": param_errors # Already calculated
        },
        "timing": {
             "run_optimization_call_duration_s": run_optimization_call_duration_s if 'run_optimization_call_duration_s' in locals() else -1.0,
             "total_script_duration_s": time.time() - start_time
        }
    }

    params_to_save = {
        'true_params': true_params,
        'estimated_params': result.final_params if result and result.final_params is not None else None,
        'initial_params': initial_params_for_opt,
        'optimization_status': {
            'success': result.success if result else False,
            'result_code': str(result.result_code) if result else "OPTIMIZATION_FAILED_OR_NOT_RUN"
        }
    }

    metrics_filepath = os.path.join(config['output_dir'], f"{base_filename}_metrics.json")
    params_filepath = os.path.join(config['output_dir'], f"{base_filename}_params.pkl")

    try:
        save_metrics_json(metrics_dict, metrics_filepath, fs)
        save_params_pkl(params_to_save, params_filepath, fs)
    except Exception as e:
        # Error already logged in save functions, just note failure here
        logger.error("Failed to save one or more result files.")

    total_duration = time.time() - start_time
    logger.info(f"Replicate finished in {total_duration:.2f} seconds.")

if __name__ == "__main__":
    main()