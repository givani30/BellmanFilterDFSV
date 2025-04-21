#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to merge the results from simulation1_data into one output CSV file.
This script uses the existing merge_simulation_results.py script.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Create the output directory if it doesn't exist
    output_dir = Path("outputs/simulation1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the output file path
    output_file = output_dir / "simulation1_merged_results.csv"
    
    # Define the input directory
    input_dir = Path("simulation1_data")
    
    # Check if the input directory exists
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Define the command to run the merge_simulation_results.py script
    cmd = [
        "python", 
        "scripts/simstudy_1/merge_simulation_results.py",
        "--input-dir", str(input_dir),
        "--output-file", str(output_file),
        "--study-pattern", "batch_study_*",  # Pattern to match all batch_study_* directories
        "--metrics-filename", "metrics.json"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print(f"\nSuccessfully merged simulation1 data into: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running merge_simulation_results.py: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
