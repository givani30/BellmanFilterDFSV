#!/usr/bin/env python
"""
Real Data Application Example for DFSV Models

This example demonstrates how to:
1. Load and preprocess real financial data
2. Estimate DFSV model parameters using different filters
3. Filter latent states (factors and volatilities)
4. Analyze and visualize the results
5. Evaluate model performance

Note: This example requires pandas and yfinance packages for data handling.
Install them with: pip install pandas yfinance
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.filters.bellman import DFSVBellmanFilter
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.filters.particle import DFSVParticleFilter
from bellman_filter_dfsv.utils.optimization import (
    FilterType,
    run_optimization,
    OptimizerResult
)

# Set random seed for reproducibility
np.random.seed(42)


def load_financial_data(tickers, start_date, end_date):
    """
    Load financial data for a list of tickers.

    Args:
        tickers (list): List of ticker symbols
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: DataFrame containing adjusted close prices
    """
    print(f"Loading data for {len(tickers)} tickers from {start_date} to {end_date}...")

    # Download data
    data = yf.download(tickers, start=start_date, end=end_date)

    # Extract adjusted close prices
    prices = data['Adj Close']

    # Handle missing values
    prices = prices.dropna(axis=0, how='any')

    print(f"Loaded {len(prices)} days of data")

    return prices


def preprocess_financial_data(prices):
    """
    Preprocess financial data for DFSV modeling.

    Args:
        prices (pd.DataFrame): DataFrame containing adjusted close prices

    Returns:
        tuple: (returns, dates, tickers)
            returns (np.ndarray): Log returns with shape (T, N)
            dates (pd.DatetimeIndex): Dates corresponding to returns
            tickers (list): List of ticker symbols
    """
    print("Preprocessing financial data...")

    # Calculate log returns
    log_returns = np.log(prices).diff().dropna()

    # Extract dates and tickers
    dates = log_returns.index
    tickers = log_returns.columns.tolist()

    # Convert to numpy array
    returns = log_returns.values

    # Standardize returns (optional)
    # returns = (returns - returns.mean(axis=0)) / returns.std(axis=0)

    print(f"Preprocessed {returns.shape[0]} days of returns for {returns.shape[1]} assets")

    return returns, dates, tickers


def create_initial_parameters(returns, K=1):
    """
    Create initial parameters for DFSV model based on data characteristics.

    Args:
        returns (np.ndarray): Returns with shape (T, N)
        K (int): Number of factors

    Returns:
        DFSVParamsDataclass: Initial parameters
    """
    T, N = returns.shape

    # Calculate data variance for reasonable sigma2 initialization
    data_variance = np.var(returns, axis=0)

    # Create initial parameters
    initial_params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=jnp.ones((N, K)) * 0.5,  # Moderate positive loadings
        Phi_f=jnp.eye(K) * 0.7,  # Moderate persistence for factors
        Phi_h=jnp.eye(K) * 0.95,  # High persistence for volatilities
        mu=jnp.zeros(K),  # Zero mean for log volatility
        sigma2=jnp.array(data_variance * 0.5),  # Half of data variance for idiosyncratic variance
        Q_h=jnp.eye(K) * 0.1  # Moderate volatility of volatility
    )

    return initial_params


def estimate_parameters(returns, K=1, filter_type=FilterType.BIF, max_steps=200):
    """
    Estimate DFSV model parameters from returns data.

    Args:
        returns (np.ndarray): Returns with shape (T, N)
        K (int): Number of factors
        filter_type (FilterType): Type of filter to use
        max_steps (int): Maximum number of optimization steps

    Returns:
        tuple: (estimated_params, optimization_result)
    """
    # Convert returns to JAX array
    jax_returns = jnp.array(returns)

    # Create initial parameters
    initial_params = create_initial_parameters(returns, K)

    print(f"Estimating parameters with {filter_type.name} filter...")

    # Run optimization
    start_time = time.time()
    result = run_optimization(
        filter_type=filter_type,
        returns=jax_returns,
        use_transformations=True,
        optimizer_name="DampedTrustRegionBFGS",
        max_steps=max_steps,
        log_params=True,
        verbose=True
    )
    estimation_time = time.time() - start_time

    print(f"Parameter estimation completed in {estimation_time:.2f} seconds")
    print(f"Final log-likelihood: {-result.loss:.2f}")

    return result.params, result


def filter_states(params, returns, filter_type=FilterType.BIF):
    """
    Filter latent states using estimated parameters.

    Args:
        params (DFSVParamsDataclass): Model parameters
        returns (np.ndarray): Returns with shape (T, N)
        filter_type (FilterType): Type of filter to use

    Returns:
        tuple: (filtered_states, filtered_covs, log_likelihood)
    """
    # Convert returns to JAX array
    jax_returns = jnp.array(returns)

    # Create filter instance
    N, K = params.N, params.K

    if filter_type == FilterType.BF:
        filter_instance = DFSVBellmanFilter(N, K)
    elif filter_type == FilterType.BIF:
        filter_instance = DFSVBellmanInformationFilter(N, K)
    elif filter_type == FilterType.PF:
        filter_instance = DFSVParticleFilter(N, K, num_particles=1000)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    print(f"Filtering states with {filter_type.name} filter...")

    # Run filter
    start_time = time.time()
    filtered_states, filtered_covs, log_likelihood = filter_instance.filter(params, jax_returns)
    filtering_time = time.time() - start_time

    print(f"State filtering completed in {filtering_time:.2f} seconds")
    print(f"Log-likelihood: {log_likelihood:.2f}")

    return filtered_states, filtered_covs, log_likelihood


def analyze_filtered_states(filtered_states, dates, tickers, K):
    """
    Analyze filtered states from DFSV model.

    Args:
        filtered_states (np.ndarray): Filtered states with shape (T, state_dim)
        dates (pd.DatetimeIndex): Dates corresponding to returns
        tickers (list): List of ticker symbols
        K (int): Number of factors

    Returns:
        dict: Dictionary containing analysis results
    """
    T = filtered_states.shape[0]

    # Extract factors and log-volatilities
    factors = np.array(filtered_states[:, :K])
    log_vols = np.array(filtered_states[:, K:2*K])

    # Convert log-volatilities to volatilities
    volatilities = np.exp(log_vols / 2)

    # Calculate statistics
    factor_mean = factors.mean(axis=0)
    factor_std = factors.std(axis=0)
    vol_mean = volatilities.mean(axis=0)
    vol_min = volatilities.min(axis=0)
    vol_max = volatilities.max(axis=0)

    # Calculate autocorrelations
    def autocorr(x, lag=1):
        """Calculate autocorrelation at specified lag."""
        return np.corrcoef(x[lag:], x[:-lag])[0, 1]

    factor_autocorr = [autocorr(factors[:, k]) for k in range(K)]
    vol_autocorr = [autocorr(volatilities[:, k]) for k in range(K)]

    # Print statistics
    print("\nFiltered State Analysis:")
    print("=======================")

    print("\nFactor Statistics:")
    for k in range(K):
        print(f"Factor {k+1}:")
        print(f"  Mean: {factor_mean[k]:.4f}")
        print(f"  Std Dev: {factor_std[k]:.4f}")
        print(f"  Autocorrelation (lag=1): {factor_autocorr[k]:.4f}")

    print("\nVolatility Statistics:")
    for k in range(K):
        print(f"Volatility {k+1}:")
        print(f"  Mean: {vol_mean[k]:.4f}")
        print(f"  Min: {vol_min[k]:.4f}")
        print(f"  Max: {vol_max[k]:.4f}")
        print(f"  Autocorrelation (lag=1): {vol_autocorr[k]:.4f}")

    # Return analysis results
    analysis = {
        'factors': factors,
        'log_vols': log_vols,
        'volatilities': volatilities,
        'factor_mean': factor_mean,
        'factor_std': factor_std,
        'vol_mean': vol_mean,
        'vol_min': vol_min,
        'vol_max': vol_max,
        'factor_autocorr': factor_autocorr,
        'vol_autocorr': vol_autocorr
    }

    return analysis


def plot_filtered_states(analysis, dates, tickers, K):
    """
    Plot filtered states from DFSV model.

    Args:
        analysis (dict): Dictionary containing analysis results
        dates (pd.DatetimeIndex): Dates corresponding to returns
        tickers (list): List of ticker symbols
        K (int): Number of factors
    """
    # Extract data from analysis
    factors = analysis['factors']
    volatilities = analysis['volatilities']

    # Create time axis
    time_axis = np.arange(len(dates))

    # Plot factors
    plt.figure(figsize=(12, 4 * K))

    for k in range(K):
        plt.subplot(K, 1, k + 1)
        plt.plot(dates, factors[:, k])
        plt.title(f'Factor {k+1}')
        plt.xlabel('Date')
        plt.ylabel('Factor Value')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot volatilities
    plt.figure(figsize=(12, 4 * K))

    for k in range(K):
        plt.subplot(K, 1, k + 1)
        plt.plot(dates, volatilities[:, k])
        plt.title(f'Volatility {k+1}')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot volatility vs. market events (if K >= 1)
    if K >= 1:
        plt.figure(figsize=(12, 6))
        plt.plot(dates, volatilities[:, 0])
        plt.title('Market Volatility')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.grid(True)

        # Add annotations for major market events
        # These are example events - adjust based on your date range
        events = {
            '2008-09-15': 'Lehman Brothers Bankruptcy',
            '2010-05-06': 'Flash Crash',
            '2011-08-05': 'US Credit Downgrade',
            '2015-08-24': 'Black Monday 2015',
            '2016-06-24': 'Brexit Vote',
            '2018-02-05': 'Volatility Spike',
            '2020-03-16': 'COVID-19 Crash',
            '2022-02-24': 'Russia-Ukraine War'
        }

        # Add vertical lines and annotations for events within the date range
        for event_date, event_name in events.items():
            event_dt = pd.to_datetime(event_date)
            if event_dt in dates or (dates[0] <= event_dt <= dates[-1]):
                try:
                    idx = dates.get_indexer([event_dt], method='nearest')[0]
                    plt.axvline(x=dates[idx], color='r', linestyle='--', alpha=0.5)
                    plt.annotate(event_name, xy=(dates[idx], volatilities[idx, 0]),
                                xytext=(10, 30), textcoords='offset points',
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
                except:
                    pass  # Skip if event date is not in range

        plt.tight_layout()
        plt.show()


def evaluate_model_performance(returns, filtered_states, params, K):
    """
    Evaluate DFSV model performance.

    Args:
        returns (np.ndarray): Returns with shape (T, N)
        filtered_states (np.ndarray): Filtered states with shape (T, state_dim)
        params (DFSVParamsDataclass): Model parameters
        K (int): Number of factors

    Returns:
        dict: Dictionary containing performance metrics
    """
    T, N = returns.shape

    # Extract factors and log-volatilities
    factors = np.array(filtered_states[:, :K])
    log_vols = np.array(filtered_states[:, K:2*K])

    # Convert log-volatilities to volatilities
    volatilities = np.exp(log_vols / 2)

    # Calculate model-implied returns
    lambda_r = np.array(params.lambda_r)
    sigma2 = np.array(params.sigma2)

    # Calculate model-implied returns (mean)
    implied_returns_mean = np.zeros((T, N))
    for t in range(T):
        implied_returns_mean[t] = lambda_r @ factors[t]

    # Calculate residuals
    residuals = returns - implied_returns_mean

    # Calculate mean squared error
    mse = np.mean(residuals ** 2, axis=0)
    rmse = np.sqrt(mse)

    # Calculate R-squared
    tss = np.sum((returns - returns.mean(axis=0)) ** 2, axis=0)
    rss = np.sum(residuals ** 2, axis=0)
    r_squared = 1 - (rss / tss)

    # Print performance metrics
    print("\nModel Performance Metrics:")
    print("=========================")

    print("\nRoot Mean Squared Error (RMSE):")
    for n in range(N):
        print(f"  Asset {n+1}: {rmse[n]:.4f}")

    print("\nR-squared:")
    for n in range(N):
        print(f"  Asset {n+1}: {r_squared[n]:.4f}")

    print(f"\nAverage RMSE: {rmse.mean():.4f}")
    print(f"Average R-squared: {r_squared.mean():.4f}")

    # Return performance metrics
    performance = {
        'mse': mse,
        'rmse': rmse,
        'r_squared': r_squared,
        'implied_returns_mean': implied_returns_mean,
        'residuals': residuals
    }

    return performance


def main():
    """Run the real data application example."""
    print("Real Data Application Example for DFSV Models")
    print("============================================")

    # Define tickers (S&P 500 sectors ETFs)
    tickers = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE']

    # Define date range (5 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')

    # Load and preprocess data
    try:
        prices = load_financial_data(tickers, start_date, end_date)
        returns, dates, tickers = preprocess_financial_data(prices)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using simulated data instead...")

        # Create simulated data
        N = 10  # Number of assets
        T = 1000  # Number of time periods

        # Create random returns
        returns = np.random.normal(0, 0.01, size=(T, N))

        # Create dates
        end_date = datetime.now()
        dates = [end_date - timedelta(days=i) for i in range(T)]
        dates.reverse()
        dates = pd.DatetimeIndex(dates)

        # Create tickers
        tickers = [f"Asset{i+1}" for i in range(N)]

    # Set number of factors
    K = 1

    # Estimate parameters
    estimated_params, optimization_result = estimate_parameters(
        returns, K=K, filter_type=FilterType.BIF, max_steps=100
    )

    # Filter states
    filtered_states, filtered_covs, log_likelihood = filter_states(
        estimated_params, returns, filter_type=FilterType.BIF
    )

    # Analyze filtered states
    analysis = analyze_filtered_states(filtered_states, dates, tickers, K)

    # Plot filtered states
    plot_filtered_states(analysis, dates, tickers, K)

    # Evaluate model performance
    performance = evaluate_model_performance(returns, filtered_states, estimated_params, K)

    return returns, dates, tickers, estimated_params, filtered_states, analysis, performance


if __name__ == "__main__":
    main()
