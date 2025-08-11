#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix BIF Average Correlation Script

This script fixes the average correlation calculation for the BIF model.
The original calculation incorrectly divided by the number of unique off-diagonal elements
while summing all off-diagonal elements, effectively doubling the average.

Author: Givani Boekestijn
"""

import numpy as np
import os
from pathlib import Path

# Define paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent.parent
BIF_DATA_DIR = BASE_DIR / "outputs" / "empirical" / "insample" / "bif" / "data"

print("Fixing BIF average correlation calculation...")

# Load correlation matrices
try:
    corr_matrices = np.load(BIF_DATA_DIR / "correlation_matrices.npy")
    T, N, _ = corr_matrices.shape
    
    print(f"Loaded correlation matrices with shape: {corr_matrices.shape}")
    print(f"Min value: {np.min(corr_matrices)}, Max value: {np.max(corr_matrices)}")
    
    # Calculate average correlation correctly
    avg_correlation_fixed = np.zeros(T)
    
    for t in range(T):
        # Method 1: Use total number of off-diagonal elements
        n_off_diag = N * (N - 1)  # Total number of off-diagonal elements
        avg_correlation_fixed[t] = (np.sum(corr_matrices[t]) - N) / n_off_diag
    
    print(f"Fixed average correlation range: {np.min(avg_correlation_fixed)} to {np.max(avg_correlation_fixed)}")
    print(f"Values > 1: {np.sum(avg_correlation_fixed > 1)}")
    
    # Save fixed average correlation
    avg_corr_path = BIF_DATA_DIR / "average_correlation.npy"
    np.save(avg_corr_path, avg_correlation_fixed)
    print(f"Fixed average correlation saved to {avg_corr_path}")
    
except Exception as e:
    print(f"Error fixing BIF average correlation: {e}")

print("Done.")
