#!/usr/bin/env python
"""
Wrapper script to run the complete DCC-GARCH estimation process.
This script runs both stages of the DCC-GARCH estimation and tracks the total time.
"""
import os
import time
import json
import pathlib
import subprocess

# Get the script's directory
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
METADATA_FILE = os.path.join(DATA_DIR, "model_metadata.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def run_script(script_command):
    """Run a Python script and return its exit code.

    Args:
        script_command: Either a path to a script or a command with arguments
    """
    if ' ' in script_command:
        # Command with arguments
        cmd = ["python"] + script_command.split()
    else:
        # Just a script path
        cmd = ["python", script_command]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Error output: {result.stderr}")
    return result.returncode

# Start timing the entire estimation process
start_time = time.time()

print("=== Starting DCC-GARCH estimation ===")
print("Stage 1: Fitting univariate GARCH models...")

# Note: We're skipping the first stage (univariate GARCH) since we're now
# using the original returns directly in the DCC model
# The DCC model will perform its own univariate GARCH fitting internally
stage1_start = time.time()
stage1_time = 0.0
stage1_exit_code = 0
print("Skipping Stage 1 (univariate GARCH) - DCC model will handle this internally")
stage1_time = time.time() - stage1_start

if stage1_exit_code != 0:
    print(f"Error in Stage 1. Exit code: {stage1_exit_code}")
    exit(stage1_exit_code)

print(f"Stage 1 completed in {stage1_time:.2f} seconds")
print("Stage 2: Fitting DCC model...")

# Run the second stage (DCC estimation)
stage2_start = time.time()
stage2_script = os.path.join(SCRIPT_DIR, "02_dcc_fit.py")
stage2_exit_code = run_script(stage2_script)
stage2_time = time.time() - stage2_start

if stage2_exit_code != 0:
    print(f"Error in Stage 2. Exit code: {stage2_exit_code}")
    exit(stage2_exit_code)

print(f"Stage 2 completed in {stage2_time:.2f} seconds")

# Calculate total estimation time
total_time = time.time() - start_time
print(f"Total estimation time: {total_time:.2f} seconds")

# Update metadata with total estimation time
try:
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)

    # Update with stage times
    metadata["estimation_time"] = {
        "stage1_time": stage1_time,
        "stage2_time": stage2_time,
        "total_time": total_time
    }

    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)

    print("Updated metadata with total estimation time")
except Exception as e:
    print(f"Error updating metadata: {e}")

print("=== DCC-GARCH estimation completed ===")

# Run the in-sample metrics calculation
print("Calculating in-sample metrics...")
metrics_script = os.path.join(SCRIPT_DIR, "03_in_sample_metrics.py")
metrics_exit_code = run_script(metrics_script)

if metrics_exit_code != 0:
    print(f"Error in metrics calculation. Exit code: {metrics_exit_code}")
    exit(metrics_exit_code)

print("In-sample metrics calculation completed")

# Run the GMV forecast
print("Calculating GMV forecast...")
gmv_script = os.path.join(SCRIPT_DIR, "04_forecast_gmv.py")
gmv_exit_code = run_script(gmv_script)

if gmv_exit_code != 0:
    print(f"Error in GMV forecast. Exit code: {gmv_exit_code}")
    exit(gmv_exit_code)

print("GMV forecast completed")
print("=== All DCC-GARCH processing completed successfully ===")
