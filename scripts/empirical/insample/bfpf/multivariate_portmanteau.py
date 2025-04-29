#!/usr/bin/env python
"""
Multivariate Portmanteau Tests for Residual Autocorrelation

This module implements the Hosking (1980) and Li-McLeod (1981) multivariate
Portmanteau tests for residual autocorrelation in multivariate time series.

These tests are multivariate extensions of the Ljung-Box test and are used to
detect autocorrelation in the residuals of a multivariate time series model.

References:
    - Hosking, J. R. M. (1980). The multivariate portmanteau statistic.
      Journal of the American Statistical Association, 75(371), 602-608.
    - Li, W. K., & McLeod, A. I. (1981). Distribution of the residual
      autocorrelations in multivariate ARMA time series models.
      Journal of the Royal Statistical Society: Series B, 43(2), 231-239.
"""

import numpy as np
from scipy import stats
import warnings


def hosking_test(residuals, lags, df_model=0):
    """
    Compute the Hosking (1980) multivariate portmanteau test statistic.
    
    Parameters:
        residuals (numpy.ndarray): Residuals matrix of shape (T, N) where T is the
                                  number of time points and N is the number of series.
        lags (int): Number of lags to include in the test.
        df_model (int): Degrees of freedom used by the model. For a VAR(p) model,
                       this would be p*N^2.
    
    Returns:
        tuple: (test_statistic, p_value, df)
    """
    T, N = residuals.shape
    
    # Center the residuals
    residuals = residuals - residuals.mean(axis=0)
    
    # Compute the covariance matrix of residuals
    C0 = np.cov(residuals, rowvar=False) * (T-1)/T
    C0_inv = np.linalg.inv(C0)
    
    # Compute the autocovariance matrices for each lag
    Q = 0
    for h in range(1, lags + 1):
        Ch = np.zeros((N, N))
        for t in range(h, T):
            Ch += np.outer(residuals[t], residuals[t-h])
        Ch = Ch / T
        
        # Compute the contribution to the test statistic
        Ch_C0inv = Ch @ C0_inv
        tr_term = np.trace(Ch_C0inv @ Ch_C0inv.T)
        Q += tr_term
    
    # Multiply by T to get the test statistic
    Q = T * Q
    
    # Degrees of freedom: N^2 * (lags - p) where p is the order of the model
    df = N**2 * (lags - df_model/N**2)
    
    # Compute p-value from chi-square distribution
    p_value = 1 - stats.chi2.cdf(Q, df)
    
    return Q, p_value, df


def li_mcleod_test(residuals, lags, df_model=0):
    """
    Compute the Li-McLeod (1981) multivariate portmanteau test statistic.
    
    Parameters:
        residuals (numpy.ndarray): Residuals matrix of shape (T, N) where T is the
                                  number of time points and N is the number of series.
        lags (int): Number of lags to include in the test.
        df_model (int): Degrees of freedom used by the model. For a VAR(p) model,
                       this would be p*N^2.
    
    Returns:
        tuple: (test_statistic, p_value, df)
    """
    T, N = residuals.shape
    
    # Center the residuals
    residuals = residuals - residuals.mean(axis=0)
    
    # Compute the covariance matrix of residuals
    C0 = np.cov(residuals, rowvar=False) * (T-1)/T
    C0_inv = np.linalg.inv(C0)
    
    # Compute the autocovariance matrices and the test statistic
    Q_lm = 0
    for h in range(1, lags + 1):
        Ch = np.zeros((N, N))
        for t in range(h, T):
            Ch += np.outer(residuals[t], residuals[t-h])
        Ch = Ch / T
        
        # Compute the contribution to the test statistic
        Ch_C0inv = Ch @ C0_inv
        tr_term = np.trace(Ch_C0inv @ Ch_C0inv.T)
        Q_lm += (1 / (T - h)) * tr_term
    
    # Multiply by T^2 to get the test statistic
    Q_lm = T**2 * Q_lm
    
    # Degrees of freedom: N^2 * (lags - p) where p is the order of the model
    df = N**2 * (lags - df_model/N**2)
    
    # Compute p-value from chi-square distribution
    p_value = 1 - stats.chi2.cdf(Q_lm, df)
    
    return Q_lm, p_value, df


def multivariate_portmanteau_tests(residuals, lags=[5, 10, 15], df_model=0, 
                                  max_dimension=30, random_state=None):
    """
    Perform both Hosking and Li-McLeod multivariate portmanteau tests.
    
    For high-dimensional data (N > max_dimension), a random subset of series
    is selected to make the computation feasible.
    
    Parameters:
        residuals (numpy.ndarray): Residuals matrix of shape (T, N) where T is the
                                  number of time points and N is the number of series.
        lags (list): List of lags to test.
        df_model (int): Degrees of freedom used by the model.
        max_dimension (int): Maximum number of series to include in the test.
        random_state (int): Random seed for reproducibility when selecting a subset.
    
    Returns:
        dict: Dictionary containing test results for each lag.
    """
    T, N = residuals.shape
    results = {}
    
    # For high-dimensional data, select a subset of series
    if N > max_dimension:
        warnings.warn(f"Number of series ({N}) exceeds max_dimension ({max_dimension}). "
                     f"Selecting a random subset of {max_dimension} series.")
        
        # Set random seed for reproducibility
        np.random.seed(random_state)
        
        # Select random subset of series
        subset_indices = np.random.choice(N, max_dimension, replace=False)
        subset_residuals = residuals[:, subset_indices]
        
        # Update N for the calculations
        N_test = max_dimension
        test_residuals = subset_residuals
    else:
        N_test = N
        test_residuals = residuals
    
    # Run tests for each lag
    for lag in lags:
        # Skip if lag is too large relative to sample size
        if lag >= T:
            results[f"lag_{lag}"] = {
                "error": f"Lag {lag} is too large for sample size {T}",
                "hosking": None,
                "li_mcleod": None
            }
            continue
            
        try:
            # Hosking test
            h_stat, h_pvalue, h_df = hosking_test(test_residuals, lag, df_model)
            
            # Li-McLeod test
            lm_stat, lm_pvalue, lm_df = li_mcleod_test(test_residuals, lag, df_model)
            
            results[f"lag_{lag}"] = {
                "N_tested": N_test,
                "hosking": {
                    "statistic": float(h_stat),
                    "p_value": float(h_pvalue),
                    "df": float(h_df),
                    "reject_null": bool(h_pvalue < 0.05)
                },
                "li_mcleod": {
                    "statistic": float(lm_stat),
                    "p_value": float(lm_pvalue),
                    "df": float(lm_df),
                    "reject_null": bool(lm_pvalue < 0.05)
                }
            }
        except Exception as e:
            results[f"lag_{lag}"] = {
                "error": str(e),
                "hosking": None,
                "li_mcleod": None
            }
    
    return results


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    T, N = 500, 5
    
    # Generate random residuals (white noise)
    white_noise = np.random.randn(T, N)
    
    # Generate autocorrelated residuals
    autocorrelated = np.zeros((T, N))
    autocorrelated[0] = np.random.randn(N)
    for t in range(1, T):
        autocorrelated[t] = 0.5 * autocorrelated[t-1] + np.random.randn(N)
    
    # Test white noise residuals
    print("Testing white noise residuals:")
    wn_results = multivariate_portmanteau_tests(white_noise, lags=[5, 10, 15])
    for lag, result in wn_results.items():
        print(f"{lag}:")
        print(f"  Hosking: stat={result['hosking']['statistic']:.2f}, p-value={result['hosking']['p_value']:.4f}")
        print(f"  Li-McLeod: stat={result['li_mcleod']['statistic']:.2f}, p-value={result['li_mcleod']['p_value']:.4f}")
    
    # Test autocorrelated residuals
    print("\nTesting autocorrelated residuals:")
    ac_results = multivariate_portmanteau_tests(autocorrelated, lags=[5, 10, 15])
    for lag, result in ac_results.items():
        print(f"{lag}:")
        print(f"  Hosking: stat={result['hosking']['statistic']:.2f}, p-value={result['hosking']['p_value']:.4f}")
        print(f"  Li-McLeod: stat={result['li_mcleod']['statistic']:.2f}, p-value={result['li_mcleod']['p_value']:.4f}")
