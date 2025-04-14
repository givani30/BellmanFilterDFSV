#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregates results from a batch optimization study.

Scans a directory containing replicate outputs (subdirectories named 'task_*'),
parses metadata from folder names, reads metrics from JSON files, and reads
parameter objects and optimization status from PKL files.

Outputs two files:
1. A CSV file containing scalar metrics, configuration, metadata, and status
   for each replicate, linked by a unique integer ID.
2. An NPZ file containing dictionaries mapping the unique integer ID to the
   'true_params' and 'estimated_params' objects (likely DFSVParamsDataclass)
   loaded from the PKL files.

Handles errors gracefully (e.g., missing files, corrupted data, parsing issues)
and logs information about the process and any errors encountered.
"""

import os
import json
import cloudpickle  # Use cloudpickle for potentially complex objects
import pandas as pd
import numpy as np
import argparse
import logging
import datetime
from pathlib import Path
import re  # For potentially more robust parsing if needed

# Attempt to import DFSVParamsDataclass for type hints/safety, but don't fail if not found
try:
    # Adjust the import path based on your project structure
    from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
except ImportError:
    DFSVParamsDataclass = None  # Define as None if import fails
    logging.info("DFSVParamsDataclass not found. Proceeding without specific type hints for parameters.")


def setup_logging():
    """Configures basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_folder_name(folder_name: str) -> tuple[dict, bool]:
    """
    Parses metadata from the folder name.

    Expected format: task_{task_index:05d}_{filter_type}_N{N}_K{K}_seed{seed}

    Args:
        folder_name: The name of the subdirectory.

    Returns:
        A tuple containing:
        - A dictionary with parsed metadata ('task_index', 'filter_type', 'N', 'K', 'seed').
        - A boolean indicating if parsing was successful.
    """
    metadata = {
        'task_index': None,
        'filter_type': None,
        'N': None,
        'K': None,
        'seed': None
    }
    parse_error = False
    try:
        parts = folder_name.split('_')
        if len(parts) >= 6 and parts[0] == 'task':
            metadata['task_index'] = int(parts[1])
            metadata['filter_type'] = parts[2]
            # Extract numbers, handling potential errors if format deviates
            metadata['N'] = int(parts[3].replace('N', ''))
            metadata['K'] = int(parts[4].replace('K', ''))
            metadata['seed'] = int(parts[5].replace('seed', ''))
        else:
            raise ValueError("Folder name does not match expected format.")
    except (IndexError, ValueError, TypeError) as e:
        logging.warning(f"Error parsing metadata from folder '{folder_name}': {e}")
        parse_error = True
        # Reset metadata to None on error
        metadata = {key: None for key in metadata}

    return metadata, parse_error


def flatten_json_metrics(data: dict) -> dict:
    """
    Flattens the nested dictionary structure from replicate_results_metrics.json.

    Handles potential missing keys gracefully using .get().

    Args:
        data: The loaded JSON data as a dictionary.

    Returns:
        A flattened dictionary suitable for CSV columns.
    """
    flat_metrics = {}

    # Config section
    config_data = data.get('config', {})
    if isinstance(config_data, dict):
        for key, value in config_data.items():
            flat_metrics[f'config_{key}'] = value

    # Results section
    results_data = data.get('results', {})
    if isinstance(results_data, dict):
        for key, value in results_data.items():
             # Handle nested dicts like optimization_details if necessary
            if isinstance(value, dict):
                 for sub_key, sub_value in value.items():
                     flat_metrics[f'results_{key}_{sub_key}'] = sub_value
            else:
                flat_metrics[f'results_{key}'] = value


    # Accuracy section
    accuracy_data = data.get('accuracy', {})
    if isinstance(accuracy_data, dict):
        for key, value in accuracy_data.items():
            # Handle potentially nested structures if they exist
            if isinstance(value, dict):
                 for sub_key, sub_value in value.items():
                     flat_metrics[f'accuracy_{key}_{sub_key}'] = sub_value
            else:
                flat_metrics[f'accuracy_{key}'] = value


    # Timing section
    timing_data = data.get('timing', {})
    if isinstance(timing_data, dict):
        for key, value in timing_data.items():
            flat_metrics[f'timing_{key}'] = value

    return flat_metrics


def process_replicate(subdir_path: Path, unique_id: int) -> tuple[dict | None, dict | None, dict | None]:
    """
    Processes a single replicate subdirectory.

    Reads metrics (JSON) and parameters/status (PKL), parses metadata,
    and handles errors.

    Args:
        subdir_path: Path object for the replicate subdirectory.
        unique_id: The unique integer ID assigned to this replicate.

    Returns:
        A tuple containing:
        - csv_data: Dictionary for the CSV row (or None if fatal error).
        - true_params: The loaded true_params object (or None).
        - estimated_params: The loaded estimated_params object (or None).
    """
    folder_name = subdir_path.name
    logging.info(f"Processing ID {unique_id}: {folder_name}")

    metadata, metadata_parse_error = parse_folder_name(folder_name)

    json_read_error = False
    pkl_read_error = False
    metrics_data = {}
    status_data = {}
    true_params = None
    estimated_params = None

    # --- Read Metrics (JSON) ---
    metrics_path = subdir_path / 'replicate_results_metrics.json'
    try:
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
        with open(metrics_path, 'r') as f:
            loaded_json = json.load(f)
        metrics_data = flatten_json_metrics(loaded_json)
        logging.debug(f"Successfully read and flattened JSON for {folder_name}")
    except FileNotFoundError as e:
        logging.error(f"Error reading JSON for {folder_name}: {e}")
        json_read_error = True
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON for {folder_name} ({metrics_path}): {e}")
        json_read_error = True
    except Exception as e:
        logging.error(f"Unexpected error reading/flattening JSON for {folder_name} ({metrics_path}): {e}")
        json_read_error = True

    # --- Read Parameters & Status (PKL) ---
    params_path = subdir_path / 'replicate_results_params.pkl'
    try:
        if not params_path.is_file():
             raise FileNotFoundError(f"Parameters file not found: {params_path}")
        with open(params_path, 'rb') as f:
            # Use cloudpickle for loading
            loaded_pkl = cloudpickle.load(f)

        # Extract status safely using .get()
        opt_status = loaded_pkl.get('optimization_status', {})
        if isinstance(opt_status, dict):
             status_data['opt_success'] = opt_status.get('success')
             status_data['opt_status'] = opt_status.get('status') # e.g., scipy status code
             status_data['opt_message'] = opt_status.get('message')
             status_data['opt_nfev'] = opt_status.get('nfev')
             status_data['opt_nit'] = opt_status.get('nit')
        else:
            logging.warning(f"Optimization status in {folder_name} is not a dictionary.")
            status_data['opt_success'] = None # Indicate missing/malformed status

        # Extract parameters safely using .get()
        true_params = loaded_pkl.get('true_params')
        estimated_params = loaded_pkl.get('estimated_params')

        if true_params is None or estimated_params is None:
             logging.warning(f"True or estimated parameters missing in PKL for {folder_name}")
             # Don't set pkl_read_error=True just for missing params, but they won't be saved to NPZ

        logging.debug(f"Successfully loaded PKL for {folder_name}")

    except FileNotFoundError as e:
        logging.error(f"Error reading PKL for {folder_name}: {e}")
        pkl_read_error = True
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as e:
        # Catch common unpickling errors
        logging.error(f"Error unpickling PKL for {folder_name} ({params_path}): {e}")
        pkl_read_error = True
        true_params = None # Ensure params are None if unpickling fails
        estimated_params = None
    except Exception as e:
        logging.error(f"Unexpected error reading PKL for {folder_name} ({params_path}): {e}")
        pkl_read_error = True
        true_params = None
        estimated_params = None


    # --- Prepare CSV Data ---
    csv_data = {'unique_id': unique_id, 'folder_name': folder_name}
    csv_data.update(metadata)
    csv_data.update(metrics_data)
    csv_data.update(status_data)
    csv_data['json_read_error'] = json_read_error
    csv_data['pkl_read_error'] = pkl_read_error
    csv_data['metadata_parse_error'] = metadata_parse_error

    return csv_data, true_params, estimated_params


def main(input_dir: str, output_dir: str):
    """
    Main function to orchestrate the aggregation process.

    Args:
        input_dir: Path to the directory containing replicate subdirectories.
        output_dir: Path to the directory where output files will be saved.
    """
    setup_logging()
    logging.info("--- Starting Aggregation Script ---")
    logging.info(f"Input Directory: {input_dir}")
    logging.info(f"Output Directory: {output_dir}")

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Locate Result Files ---
    try:
        subdirs = sorted(list(input_path.rglob('task_*')))
        logging.info(f"Found {len(subdirs)} potential replicate directories.")
        if not subdirs:
            logging.warning("No directories matching 'task_*' found in input directory.")
            return
    except FileNotFoundError:
        logging.error(f"Input directory not found: {input_dir}")
        return
    except Exception as e:
        logging.error(f"Error accessing input directory {input_dir}: {e}")
        return


    # --- Initialization ---
    results_list = []
    all_true_params = {}
    all_estimated_params = {}
    unique_id_counter = 0
    successfully_processed_count = 0

    # --- Process Each Replicate ---
    for subdir_path in subdirs:
        if not subdir_path.is_dir():
            logging.warning(f"Skipping non-directory item: {subdir_path}")
            continue

        current_id = unique_id_counter
        csv_data, true_params, estimated_params = process_replicate(subdir_path, current_id)

        if csv_data:
            results_list.append(csv_data)
            # Check if PKL processing was successful for parameter saving
            if not csv_data['pkl_read_error']:
                 if true_params is not None and estimated_params is not None:
                     all_true_params[current_id] = true_params
                     all_estimated_params[current_id] = estimated_params
                     successfully_processed_count += 1 # Count as success if both JSON/PKL read ok
                 else:
                     logging.warning(f"Parameters for ID {current_id} ({subdir_path.name}) not saved to NPZ (missing in PKL or PKL read error).")
            else:
                 logging.warning(f"Parameters for ID {current_id} ({subdir_path.name}) not saved to NPZ due to PKL read error.")

        unique_id_counter += 1 # Increment regardless of processing success


    logging.info(f"Total replicates processed (attempted): {unique_id_counter}")
    logging.info(f"Replicates with successfully read PKL parameters: {len(all_true_params)}")


    # --- Aggregate & Save CSV ---
    if results_list:
        try:
            df = pd.DataFrame(results_list)
            # Reorder columns for better readability (optional)
            cols = ['unique_id', 'folder_name', 'task_index', 'filter_type', 'N', 'K', 'seed',
                    'metadata_parse_error', 'json_read_error', 'pkl_read_error', 'opt_success']
            # Add other columns dynamically, placing known ones first
            other_cols = [c for c in df.columns if c not in cols]
            df = df[cols + sorted(other_cols)] # Sort remaining columns alphabetically

            date_str = datetime.datetime.now().strftime('%d-%m-%Y')
            csv_filename = f'aggregated_optimization_metrics_{date_str}.csv'
            csv_output_path = output_path / csv_filename

            df.to_csv(csv_output_path, index=False)
            logging.info(f"Successfully saved aggregated metrics to: {csv_output_path}")
        except Exception as e:
            logging.error(f"Failed to create or save CSV file: {e}")
    else:
        logging.warning("No data processed. CSV file not created.")

    # --- Aggregate & Save NPZ ---
    if all_true_params or all_estimated_params: # Check if either dict has data
        try:
            date_str = datetime.datetime.now().strftime('%d-%m-%Y')
            npz_filename = f'aggregated_optimization_params_{date_str}.npz'
            npz_output_path = output_path / npz_filename

            # Use savez_compressed for potentially large parameter objects
            np.savez_compressed(
                npz_output_path,
                true_params=all_true_params,
                estimated_params=all_estimated_params
            )
            logging.info(f"Successfully saved aggregated parameters to: {npz_output_path}")
        except Exception as e:
            logging.error(f"Failed to save NPZ file: {e}")
    else:
        logging.warning("No parameter data successfully processed. NPZ file not created.")

    logging.info("--- Aggregation Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate results from batch optimization runs."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./batch_outputs",
        help="Directory containing the replicate subdirectories (default: ./batch_outputs)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save the aggregated CSV and NPZ files (default: ./outputs)"
    )
    # Add optional pickle import for safety
    try:
        import pickle
    except ImportError:
        pass # cloudpickle should handle it

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)