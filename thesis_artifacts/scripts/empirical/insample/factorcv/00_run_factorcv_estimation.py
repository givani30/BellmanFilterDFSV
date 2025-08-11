#!/usr/bin/env python3
"""
Wrapper script to run the entire Factor-CV estimation process.
This script tracks the total execution time and runs all the necessary steps.

Usage:
    python 00_run_factorcv_estimation.py [--reduced]

Options:
    --reduced   Use a reduced dataset for testing (200 time periods, 20 assets, 2 factors)
"""

import os
import time
import subprocess
import sys
import argparse

def run_script(script_path):
    """Run a Python script and return its exit code."""
    print(f"\n{'='*80}")
    print(f"Running {script_path}")
    print(f"{'='*80}\n")

    start_time = time.time()
    result = subprocess.run([sys.executable, script_path], check=False)
    end_time = time.time()

    print(f"\n{'='*80}")
    print(f"Finished {script_path}")
    print(f"Exit code: {result.returncode}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"{'='*80}\n")

    return result.returncode

def main():
    """Main function to run all Factor-CV estimation scripts."""
    print("Starting Factor-CV estimation process...")

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the scripts to run in order
    scripts = [
        os.path.join(script_dir, "02_factor_cv_fit.py"),
        os.path.join(script_dir, "03_factor_cv_metrics.py"),
    ]

    # Track total execution time
    total_start_time = time.time()

    # Run each script
    for script in scripts:
        exit_code = run_script(script)
        if exit_code != 0:
            print(f"Error running {script}. Stopping execution.")
            return exit_code

    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time

    print("\n" + "="*80)
    print("Factor-CV Estimation Process Complete")
    print(f"Total execution time: {total_execution_time:.2f} seconds")
    print("="*80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
