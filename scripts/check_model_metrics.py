#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to check model metrics and diagnose issues with generalized variance.
"""

import numpy as np
import os
from pathlib import Path

# Define paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
OUTPUTS_DIR = BASE_DIR / "outputs" / "empirical" / "insample"
SCRIPTS_DIR = BASE_DIR / "scripts" / "empirical" / "insample"

# Model data paths
BIF_DATA_DIR = OUTPUTS_DIR / "bif" / "data"
PF_DATA_DIR = OUTPUTS_DIR / "pf" / "data"
FACTORCV_DATA_DIR = OUTPUTS_DIR / "factorcv" / "data"
DCC_DATA_DIR = SCRIPTS_DIR / "dcc" / "data"

def calculate_generalized_variance(covariance_matrices):
    """Calculate generalized variance (determinant) from covariance matrices."""
    T = covariance_matrices.shape[0]
    generalized_variance = np.zeros(T)
    log_generalized_variance = np.zeros(T)

    for t in range(T):
        # Use log determinant for numerical stability with large matrices
        sign, logdet = np.linalg.slogdet(covariance_matrices[t])
        # Store the log-determinant directly (more stable)
        log_generalized_variance[t] = logdet
        # Also calculate traditional generalized variance (may underflow)
        generalized_variance[t] = sign * np.exp(logdet) if logdet > -700 else 0

    return generalized_variance, log_generalized_variance

def calculate_average_correlation(covariance_matrices):
    """Calculate average correlation from covariance matrices."""
    T, N, _ = covariance_matrices.shape
    avg_correlation = np.zeros(T)

    for t in range(T):
        # Get the covariance matrix at time t
        cov_t = covariance_matrices[t]

        # Calculate the correlation matrix
        std_devs = np.sqrt(np.diag(cov_t))
        std_outer = np.outer(std_devs, std_devs)
        corr_t = cov_t / std_outer
        np.fill_diagonal(corr_t, 1.0)  # Ensure diagonal is exactly 1

        # Calculate average of off-diagonal elements
        n_off_diag = N * (N - 1) / 2  # Number of unique off-diagonal elements
        avg_correlation[t] = (np.sum(corr_t) - N) / n_off_diag  # Subtract diagonal elements (N ones)

    return avg_correlation

def check_bif_metrics():
    """Check BIF model metrics."""
    print("\n=== BIF Model Metrics ===")

    # Load generalized variance
    bif_gv = np.load(BIF_DATA_DIR / "generalized_variance.npy")
    print(f"Generalized Variance shape: {bif_gv.shape}")
    print(f"Generalized Variance range: {np.min(bif_gv)} to {np.max(bif_gv)}")

    # Calculate log-determinant from saved generalized variance
    # This is an approximation since we're working backwards
    bif_log_gv = np.log(np.maximum(bif_gv, 1e-320))
    print(f"Approximated Log-Determinant range: {np.min(bif_log_gv)} to {np.max(bif_log_gv)}")

    # Load average correlation
    bif_ac = np.load(BIF_DATA_DIR / "average_correlation.npy")
    print(f"Average Correlation shape: {bif_ac.shape}")
    print(f"Average Correlation range: {np.min(bif_ac)} to {np.max(bif_ac)}")

    # Check if any correlation values are invalid (>1 or <-1)
    invalid_corr = np.sum((bif_ac > 1) | (bif_ac < -1))
    if invalid_corr > 0:
        print(f"WARNING: {invalid_corr} invalid correlation values found!")

def check_pf_metrics():
    """Check PF model metrics."""
    print("\n=== PF Model Metrics ===")

    # Load generalized variance
    pf_gv = np.load(PF_DATA_DIR / "generalized_variance.npy")
    print(f"Generalized Variance shape: {pf_gv.shape}")
    print(f"Generalized Variance range: {np.min(pf_gv)} to {np.max(pf_gv)}")

    # Calculate log-determinant from saved generalized variance
    # This is an approximation since we're working backwards
    pf_log_gv = np.log(np.maximum(pf_gv, 1e-320))
    print(f"Approximated Log-Determinant range: {np.min(pf_log_gv)} to {np.max(pf_log_gv)}")

    # Load average correlation
    pf_ac = np.load(PF_DATA_DIR / "average_correlation.npy")
    print(f"Average Correlation shape: {pf_ac.shape}")
    print(f"Average Correlation range: {np.min(pf_ac)} to {np.max(pf_ac)}")

    # Check if any correlation values are invalid (>1 or <-1)
    invalid_corr = np.sum((pf_ac > 1) | (pf_ac < -1))
    if invalid_corr > 0:
        print(f"WARNING: {invalid_corr} invalid correlation values found!")

def check_dcc_metrics():
    """Check DCC-GARCH model metrics."""
    print("\n=== DCC-GARCH Model Metrics ===")

    # Load covariance matrices
    dcc_cov = np.load(DCC_DATA_DIR / "Ht.npy")
    print(f"Covariance Matrices shape: {dcc_cov.shape}")

    # Sample diagonal values
    print(f"Sample diagonal values (first time point): {np.diagonal(dcc_cov[0])[:5]}")

    # Calculate using slogdet for first time point
    sign, logdet = np.linalg.slogdet(dcc_cov[0])
    print(f"slogdet components - sign: {sign}, logdet: {logdet}")

    # Calculate generalized variance and log-determinant
    dcc_gv, dcc_log_gv = calculate_generalized_variance(dcc_cov)
    print(f"Generalized Variance range: {np.min(dcc_gv)} to {np.max(dcc_gv)}")
    print(f"Log-Determinant range: {np.min(dcc_log_gv)} to {np.max(dcc_log_gv)}")

    # Load correlation matrices directly (DCC model outputs these separately)
    try:
        dcc_corr = np.load(DCC_DATA_DIR / "Rt.npy")
        print(f"Correlation Matrices shape: {dcc_corr.shape}")

        # Calculate average correlation from the correlation matrices
        T, N, _ = dcc_corr.shape
        dcc_ac_direct = np.zeros(T)
        for t in range(T):
            # Calculate average of off-diagonal elements
            n_off_diag = N * (N - 1) / 2
            dcc_ac_direct[t] = (np.sum(dcc_corr[t]) - N) / n_off_diag

        print(f"Average Correlation range (from Rt.npy): {np.min(dcc_ac_direct)} to {np.max(dcc_ac_direct)}")

        # Check if any correlation values are invalid (>1 or <-1)
        invalid_corr_direct = np.sum((dcc_ac_direct > 1) | (dcc_ac_direct < -1))
        if invalid_corr_direct > 0:
            print(f"WARNING: {invalid_corr_direct} invalid correlation values found in direct calculation!")
    except Exception as e:
        print(f"Error loading correlation matrices: {e}")

    # Also calculate from covariance matrices for comparison
    dcc_ac = calculate_average_correlation(dcc_cov)
    print(f"Average Correlation range (calculated from Ht.npy): {np.min(dcc_ac)} to {np.max(dcc_ac)}")

    # Check if any correlation values are invalid (>1 or <-1)
    invalid_corr = np.sum((dcc_ac > 1) | (dcc_ac < -1))
    if invalid_corr > 0:
        print(f"WARNING: {invalid_corr} invalid correlation values found in calculation from covariance!")

    # Check for positive definiteness
    pd_count = 0
    for t in range(min(10, dcc_cov.shape[0])):  # Check first 10 matrices
        try:
            # Try Cholesky decomposition (only works for positive definite matrices)
            np.linalg.cholesky(dcc_cov[t])
            pd_count += 1
        except np.linalg.LinAlgError:
            pass
    print(f"Positive definite matrices in first 10: {pd_count}/10")

def check_factorcv_metrics():
    """Check Factor-CV model metrics."""
    print("\n=== Factor-CV Model Metrics ===")

    # Load covariance matrices
    fcv_cov = np.load(FACTORCV_DATA_DIR / "Ht.npy")
    print(f"Covariance Matrices shape: {fcv_cov.shape}")

    # Sample diagonal values
    print(f"Sample diagonal values (first time point): {np.diagonal(fcv_cov[0])[:5]}")

    # Calculate using slogdet for first time point
    sign, logdet = np.linalg.slogdet(fcv_cov[0])
    print(f"slogdet components - sign: {sign}, logdet: {logdet}")

    # Calculate generalized variance and log-determinant
    fcv_gv, fcv_log_gv = calculate_generalized_variance(fcv_cov)
    print(f"Generalized Variance range: {np.min(fcv_gv)} to {np.max(fcv_gv)}")
    print(f"Log-Determinant range: {np.min(fcv_log_gv)} to {np.max(fcv_log_gv)}")

    # Calculate average correlation
    fcv_ac = calculate_average_correlation(fcv_cov)
    print(f"Average Correlation range: {np.min(fcv_ac)} to {np.max(fcv_ac)}")

    # Check if any correlation values are invalid (>1 or <-1)
    invalid_corr = np.sum((fcv_ac > 1) | (fcv_ac < -1))
    if invalid_corr > 0:
        print(f"WARNING: {invalid_corr} invalid correlation values found!")

    # Check for positive definiteness
    pd_count = 0
    for t in range(min(10, fcv_cov.shape[0])):  # Check first 10 matrices
        try:
            # Try Cholesky decomposition (only works for positive definite matrices)
            np.linalg.cholesky(fcv_cov[t])
            pd_count += 1
        except np.linalg.LinAlgError:
            pass
    print(f"Positive definite matrices in first 10: {pd_count}/10")

def main():
    """Main function."""
    print("Checking model metrics...")

    try:
        check_bif_metrics()
    except Exception as e:
        print(f"Error checking BIF metrics: {e}")

    try:
        check_pf_metrics()
    except Exception as e:
        print(f"Error checking PF metrics: {e}")

    try:
        check_dcc_metrics()
    except Exception as e:
        print(f"Error checking DCC-GARCH metrics: {e}")

    try:
        check_factorcv_metrics()
    except Exception as e:
        print(f"Error checking Factor-CV metrics: {e}")

    print("\nMetrics check complete.")

if __name__ == "__main__":
    main()
