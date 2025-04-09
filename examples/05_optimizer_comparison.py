#!/usr/bin/env python
"""
Optimizer Comparison Example for DFSV Models

This example demonstrates how to:
1. Create a DFSV model and simulate data
2. Compare different optimizers for parameter estimation:
   - AdamW
   - DampedTrustRegionBFGS (default)
3. Analyze optimizer performance in terms of:
   - Convergence speed
   - Final parameter accuracy
   - Numerical stability
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import time
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.utils.optimization import (
    FilterType,
    run_optimization
)
from bellman_filter_dfsv.utils.solvers import get_optimizer_config

# Set random seed for reproducibility
np.random.seed(42)


def create_simple_dfsv_model(N=3, K=1):
    """
    Create a simple DFSV model with K factors and N observed series.

    Args:
        N (int): Number of observed series
        K (int): Number of latent factors

    Returns:
        DFSVParamsDataclass: Parameters for the DFSV model
    """
    # Factor loadings - how each observed series is affected by the factors
    lambda_r = np.random.uniform(0.3, 0.9, size=(N, K))

    # Factor persistence - how strongly factors depend on their previous values
    Phi_f = np.eye(K) * 0.7  # Moderate persistence

    # Volatility persistence - how strongly log-volatilities depend on their previous values
    Phi_h = np.eye(K) * 0.97  # High persistence

    # Long-run mean of log-volatilities
    mu = np.ones(K) * -1.0  # Negative means lower volatility

    # Idiosyncratic variance of observed series
    sigma2 = np.ones(N) * 0.1  # Low idiosyncratic variance

    # Covariance matrix of log-volatility innovations
    Q_h = np.eye(K) * 0.05  # Low volatility of volatility

    # Create parameter object using JAX arrays
    params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=jnp.array(lambda_r),
        Phi_f=jnp.array(Phi_f),
        Phi_h=jnp.array(Phi_h),
        mu=jnp.array(mu),
        sigma2=jnp.array(sigma2),
        Q_h=jnp.array(Q_h)
    )

    return params


def compare_optimizers(true_params, returns, filter_type=FilterType.BIF, max_steps=100):
    """
    Compare different optimizers for parameter estimation.

    Args:
        true_params (DFSVParamsDataclass): True model parameters
        returns (np.ndarray): Simulated returns
        filter_type (FilterType): Type of filter to use
        max_steps (int): Maximum number of optimization steps

    Returns:
        dict: Dictionary containing optimization results for each optimizer
    """
    # Convert returns to JAX array
    jax_returns = jnp.array(returns)

    # Define optimizers to compare
    optimizers = [
        "AdamW",
        "DampedTrustRegionBFGS"  # Default optimizer
    ]

    # Run optimization with each optimizer
    results = {}

    for optimizer_name in optimizers:
        print(f"\nRunning optimization with {optimizer_name}...")

        # Get optimizer configuration
        config = get_optimizer_config(optimizer_name)
        print(f"Optimizer config: {config}")

        # Run optimization
        start_time = time.time()
        try:
            result = run_optimization(
                filter_type=filter_type,
                returns=jax_returns,
                true_params=true_params,  # For comparison only
                use_transformations=True,
                optimizer_name=optimizer_name,
                max_steps=max_steps,
                log_params=True,
                verbose=True
            )
            success = True
        except Exception as e:
            print(f"Optimization with {optimizer_name} failed: {e}")
            result = None
            success = False

        opt_time = time.time() - start_time
        print(f"{optimizer_name} optimization completed in {opt_time:.2f} seconds")

        # Store results
        results[optimizer_name] = {
            'result': result,
            'time': opt_time,
            'success': success
        }

    return results


def analyze_optimizer_results(true_params, optimizer_results):
    """
    Analyze optimizer results and compare performance.

    Args:
        true_params (DFSVParamsDataclass): True model parameters
        optimizer_results (dict): Dictionary containing optimization results

    Returns:
        dict: Dictionary containing performance metrics for each optimizer
    """
    # Initialize performance metrics
    performance = {}

    # Calculate parameter errors for each optimizer
    for optimizer_name, result_dict in optimizer_results.items():
        if not result_dict['success'] or result_dict['result'] is None:
            print(f"Skipping {optimizer_name} because optimization failed")
            performance[optimizer_name] = {
                'param_errors': None,
                'final_loss': float('inf'),
                'time': result_dict['time'],
                'success': False
            }
            continue

        # Extract result
        result = result_dict['result']

        # Calculate parameter errors
        def param_error(true, est):
            """Calculate relative error between true and estimated parameters."""
            return np.mean(np.abs((true - est) / (np.abs(true) + 1e-8)))

        # Calculate errors for each parameter
        param_errors = {
            'lambda_r': param_error(np.array(true_params.lambda_r), np.array(result.params.lambda_r)),
            'Phi_f': param_error(np.array(true_params.Phi_f), np.array(result.params.Phi_f)),
            'Phi_h': param_error(np.array(true_params.Phi_h), np.array(result.params.Phi_h)),
            'mu': param_error(np.array(true_params.mu), np.array(result.params.mu)),
            'sigma2': param_error(np.array(true_params.sigma2), np.array(result.params.sigma2)),
            'Q_h': param_error(np.array(true_params.Q_h), np.array(result.params.Q_h))
        }

        # Calculate average error
        avg_error = np.mean(list(param_errors.values()))

        # Store performance metrics
        performance[optimizer_name] = {
            'param_errors': param_errors,
            'avg_error': avg_error,
            'final_loss': float(result.loss),
            'time': result_dict['time'],
            'success': True,
            'num_iterations': len(result.loss_history)
        }

    # Print performance comparison
    print("\nOptimizer Performance Comparison:")
    print("================================")

    # Print average parameter error
    print("\nAverage Parameter Error:")
    for optimizer_name, metrics in performance.items():
        if metrics['success']:
            print(f"{optimizer_name}: {metrics['avg_error']:.4f}")
        else:
            print(f"{optimizer_name}: Failed")

    # Print final loss
    print("\nFinal Loss:")
    for optimizer_name, metrics in performance.items():
        if metrics['success']:
            print(f"{optimizer_name}: {metrics['final_loss']:.4f}")
        else:
            print(f"{optimizer_name}: Failed")

    # Print computation time
    print("\nComputation Time (seconds):")
    for optimizer_name, metrics in performance.items():
        print(f"{optimizer_name}: {metrics['time']:.2f}")

    # Print number of iterations
    print("\nNumber of Iterations:")
    for optimizer_name, metrics in performance.items():
        if metrics['success']:
            print(f"{optimizer_name}: {metrics['num_iterations']}")
        else:
            print(f"{optimizer_name}: Failed")

    return performance


def plot_optimizer_comparison(optimizer_results, performance_metrics):
    """
    Plot comparison of different optimizers.

    Args:
        optimizer_results (dict): Dictionary containing optimization results
        performance_metrics (dict): Dictionary containing performance metrics
    """
    # Plot loss histories
    plt.figure(figsize=(12, 6))

    for optimizer_name, result_dict in optimizer_results.items():
        if result_dict['success'] and result_dict['result'] is not None:
            loss_history = result_dict['result'].loss_history
            plt.plot(loss_history, label=optimizer_name)

    plt.title('Optimization Loss History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Log scale for better visualization
    plt.tight_layout()
    plt.show()

    # Plot computation time
    plt.figure(figsize=(10, 5))

    optimizer_names = list(optimizer_results.keys())
    times = [result_dict['time'] for result_dict in optimizer_results.values()]

    plt.bar(optimizer_names, times)
    plt.title('Optimization Time')
    plt.ylabel('Time (seconds)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

    # Plot average parameter error
    plt.figure(figsize=(10, 5))

    optimizer_names = []
    avg_errors = []

    for optimizer_name, metrics in performance_metrics.items():
        if metrics['success']:
            optimizer_names.append(optimizer_name)
            avg_errors.append(metrics['avg_error'])

    plt.bar(optimizer_names, avg_errors)
    plt.title('Average Parameter Error')
    plt.ylabel('Relative Error')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

    # Plot parameter errors by type
    plt.figure(figsize=(12, 8))

    param_names = ['lambda_r', 'Phi_f', 'Phi_h', 'mu', 'sigma2', 'Q_h']

    for i, param in enumerate(param_names):
        plt.subplot(2, 3, i + 1)

        optimizer_names = []
        errors = []

        for optimizer_name, metrics in performance_metrics.items():
            if metrics['success']:
                optimizer_names.append(optimizer_name)
                errors.append(metrics['param_errors'][param])

        plt.bar(optimizer_names, errors)
        plt.title(f'{param} Error')
        plt.ylabel('Relative Error')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')

    plt.tight_layout()
    plt.show()


def main():
    """Run the optimizer comparison example."""
    print("Optimizer Comparison Example for DFSV Models")
    print("===========================================")

    # Create model parameters
    N, K = 3, 1  # 3 observed series, 1 factor
    true_params = create_simple_dfsv_model(N, K)
    print(f"Created DFSV model with {true_params.N} observed series and {true_params.K} factors")

    # Set simulation parameters
    T = 500  # Number of time periods (shorter for faster optimization)
    seed = 42  # Random seed for reproducibility

    # Run simulation
    print(f"Simulating DFSV model for T={T} time periods...")
    returns, factors, log_vols = simulate_DFSV(
        params=true_params,
        T=T,
        seed=seed
    )
    print("Simulation complete!")

    # Compare optimizers
    optimizer_results = compare_optimizers(
        true_params, returns, filter_type=FilterType.BIF, max_steps=50
    )

    # Analyze optimizer results
    performance_metrics = analyze_optimizer_results(true_params, optimizer_results)

    # Plot optimizer comparison
    plot_optimizer_comparison(optimizer_results, performance_metrics)

    return true_params, returns, factors, log_vols, optimizer_results, performance_metrics


if __name__ == "__main__":
    main()
