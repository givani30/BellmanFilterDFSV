#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to aggregate the merged simulation1 results.
This script uses the existing aggregate_simulation_results.py script.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Define the input and output file paths
    input_file = Path("outputs/simulation1/simulation1_merged_results.csv")
    output_file = Path("outputs/simulation1/simulation1_aggregated_results.csv")
    expanded_output_file = Path("outputs/simulation1/simulation1_expanded_results.csv")
    
    # Check if the input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print("Please run the merge_simulation1_data.py script first.")
        sys.exit(1)
    
    # Define the command to run the aggregate_simulation_results.py script
    cmd = [
        "python", 
        "scripts/simstudy_1/aggregate_simulation_results.py",
        "--input-file", str(input_file),
        "--output-file", str(output_file),
        "--expanded-output", str(expanded_output_file)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print(f"\nSuccessfully aggregated simulation1 data:")
        print(f"1. Aggregated results: {output_file}")
        print(f"2. Expanded results: {expanded_output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running aggregate_simulation_results.py: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
