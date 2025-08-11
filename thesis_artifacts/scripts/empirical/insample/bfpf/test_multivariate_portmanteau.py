#!/usr/bin/env python
"""
Test script for multivariate Portmanteau tests implementation.

This script generates synthetic data with and without autocorrelation
and tests the multivariate Portmanteau tests implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from multivariate_portmanteau import multivariate_portmanteau_tests

# Set random seed for reproducibility
np.random.seed(42)

def generate_var1_process(T, N, Phi, sigma=1.0):
    """
    Generate a VAR(1) process: X_t = Phi * X_{t-1} + e_t
    
    Parameters:
        T (int): Number of time points
        N (int): Dimension of the process
        Phi (numpy.ndarray): Transition matrix of shape (N, N)
        sigma (float): Standard deviation of the noise
    
    Returns:
        numpy.ndarray: Generated time series of shape (T, N)
    """
    X = np.zeros((T, N))
    X[0] = np.random.randn(N)
    
    for t in range(1, T):
        X[t] = Phi @ X[t-1] + sigma * np.random.randn(N)
    
    return X

def run_tests():
    """Run tests on synthetic data."""
    # Parameters
    T = 500  # Number of time points
    N = 5    # Number of series
    lags = [5, 10, 15]
    
    print("=" * 80)
    print("Testing Multivariate Portmanteau Tests")
    print("=" * 80)
    
    # Test 1: White noise (no autocorrelation)
    print("\nTest 1: White Noise (No Autocorrelation)")
    print("-" * 50)
    white_noise = np.random.randn(T, N)
    wn_results = multivariate_portmanteau_tests(white_noise, lags=lags)
    
    for lag, result in wn_results.items():
        print(f"{lag}:")
        print(f"  Hosking: stat={result['hosking']['statistic']:.2f}, p-value={result['hosking']['p_value']:.4f}")
        print(f"  Li-McLeod: stat={result['li_mcleod']['statistic']:.2f}, p-value={result['li_mcleod']['p_value']:.4f}")
    
    # Test 2: VAR(1) process with moderate autocorrelation
    print("\nTest 2: VAR(1) Process with Moderate Autocorrelation")
    print("-" * 50)
    # Create a stable VAR(1) coefficient matrix with eigenvalues < 1
    Phi_moderate = np.array([
        [0.5, 0.1, 0.0, 0.0, 0.0],
        [0.1, 0.4, 0.1, 0.0, 0.0],
        [0.0, 0.1, 0.3, 0.1, 0.0],
        [0.0, 0.0, 0.1, 0.2, 0.1],
        [0.0, 0.0, 0.0, 0.1, 0.1]
    ])
    var1_moderate = generate_var1_process(T, N, Phi_moderate)
    var1_mod_results = multivariate_portmanteau_tests(var1_moderate, lags=lags, df_model=N**2)
    
    for lag, result in var1_mod_results.items():
        print(f"{lag}:")
        print(f"  Hosking: stat={result['hosking']['statistic']:.2f}, p-value={result['hosking']['p_value']:.4f}")
        print(f"  Li-McLeod: stat={result['li_mcleod']['statistic']:.2f}, p-value={result['li_mcleod']['p_value']:.4f}")
    
    # Test 3: VAR(1) process with strong autocorrelation
    print("\nTest 3: VAR(1) Process with Strong Autocorrelation")
    print("-" * 50)
    # Create a stable VAR(1) coefficient matrix with eigenvalues closer to 1
    Phi_strong = np.array([
        [0.8, 0.1, 0.0, 0.0, 0.0],
        [0.1, 0.7, 0.1, 0.0, 0.0],
        [0.0, 0.1, 0.6, 0.1, 0.0],
        [0.0, 0.0, 0.1, 0.5, 0.1],
        [0.0, 0.0, 0.0, 0.1, 0.4]
    ])
    var1_strong = generate_var1_process(T, N, Phi_strong)
    var1_strong_results = multivariate_portmanteau_tests(var1_strong, lags=lags, df_model=N**2)
    
    for lag, result in var1_strong_results.items():
        print(f"{lag}:")
        print(f"  Hosking: stat={result['hosking']['statistic']:.2f}, p-value={result['hosking']['p_value']:.4f}")
        print(f"  Li-McLeod: stat={result['li_mcleod']['statistic']:.2f}, p-value={result['li_mcleod']['p_value']:.4f}")
    
    # Test 4: High-dimensional case
    print("\nTest 4: High-Dimensional Case (N=50)")
    print("-" * 50)
    N_high = 50
    white_noise_high = np.random.randn(T, N_high)
    wn_high_results = multivariate_portmanteau_tests(white_noise_high, lags=[5], max_dimension=20)
    
    for lag, result in wn_high_results.items():
        print(f"{lag} (testing {result['N_tested']} of {N_high} series):")
        print(f"  Hosking: stat={result['hosking']['statistic']:.2f}, p-value={result['hosking']['p_value']:.4f}")
        print(f"  Li-McLeod: stat={result['li_mcleod']['statistic']:.2f}, p-value={result['li_mcleod']['p_value']:.4f}")
    
    # Visualize autocorrelation in the data
    plt.figure(figsize=(15, 10))
    
    # Plot white noise
    plt.subplot(3, 1, 1)
    plt.plot(white_noise[:100, 0])
    plt.title("White Noise")
    plt.xlabel("Time")
    plt.ylabel("Value")
    
    # Plot VAR(1) with moderate autocorrelation
    plt.subplot(3, 1, 2)
    plt.plot(var1_moderate[:100, 0])
    plt.title("VAR(1) with Moderate Autocorrelation")
    plt.xlabel("Time")
    plt.ylabel("Value")
    
    # Plot VAR(1) with strong autocorrelation
    plt.subplot(3, 1, 3)
    plt.plot(var1_strong[:100, 0])
    plt.title("VAR(1) with Strong Autocorrelation")
    plt.xlabel("Time")
    plt.ylabel("Value")
    
    plt.tight_layout()
    plt.savefig("multivariate_portmanteau_test_data.png")
    plt.close()
    
    print("\nTest data visualization saved to 'multivariate_portmanteau_test_data.png'")
    print("\nTest Summary:")
    print("- White Noise: Should have high p-values (fail to reject H0)")
    print("- VAR(1) with Moderate Autocorrelation: Should have lower p-values")
    print("- VAR(1) with Strong Autocorrelation: Should have very low p-values (reject H0)")
    print("- High-Dimensional Case: Tests the subset selection functionality")

if __name__ == "__main__":
    run_tests()
