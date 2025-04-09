#!/usr/bin/env python
"""
Parameter Transformation Example for DFSV Models

This example demonstrates how to:
1. Transform DFSV model parameters between constrained and unconstrained space
2. Compare optimization with and without parameter transformations
3. Visualize the benefits of parameter transformations for numerical stability

Parameter transformations map constrained parameters (e.g., positive definite matrices)
to unconstrained space, which can improve optimization stability and convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import equinox as eqx
import time
import optimistix as optx
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.utils.transformations import transform_params, untransform_params
from bellman_filter_dfsv.filters.objectives import bellman_objective
from bellman_filter_dfsv.utils.solvers import DampedTrustRegionBFGS

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


def create_uninformed_parameters(params):
    """
    Create uninformed initial parameters for optimization.

    Args:
        params (DFSVParamsDataclass): True model parameters (used only for dimensions)

    Returns:
        DFSVParamsDataclass: Uninformed parameters
    """
    N, K = params.N, params.K

    # Calculate data variance for reasonable sigma2 initialization
    data_variance = jnp.ones(N) * 0.5

    # Create uninformed parameters
    uninformed_params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=jnp.ones((N, K)) * 0.5,  # Moderate positive loadings
        Phi_f=jnp.eye(K) * 0.8,  # Moderate persistence
        Phi_h=jnp.eye(K) * 0.8,  # Moderate persistence
        mu=jnp.zeros(K),  # Zero mean for log volatility
        sigma2=data_variance,  # Moderate idiosyncratic variance
        Q_h=jnp.eye(K) * 0.2  # Moderate volatility of volatility
    )

    return uninformed_params


def demonstrate_parameter_transformations(params):
    """
    Demonstrate parameter transformations between constrained and unconstrained space.

    Args:
        params (DFSVParamsDataclass): Model parameters
    """
    print("Parameter Transformation Demonstration")
    print("======================================")

    # Transform parameters to unconstrained space
    transformed_params = transform_params(params)

    # Transform back to constrained space
    untransformed_params = untransform_params(transformed_params)

    # Print original and round-trip parameters
    print("\nOriginal Parameters:")
    print(f"lambda_r:\n{params.lambda_r}")
    print(f"Phi_f:\n{params.Phi_f}")
    print(f"Phi_h:\n{params.Phi_h}")
    print(f"mu:\n{params.mu}")
    print(f"sigma2:\n{params.sigma2}")
    print(f"Q_h:\n{params.Q_h}")

    print("\nTransformed Parameters (Unconstrained Space):")
    print(f"lambda_r:\n{transformed_params.lambda_r}")
    print(f"Phi_f:\n{transformed_params.Phi_f}")
    print(f"Phi_h:\n{transformed_params.Phi_h}")
    print(f"mu:\n{transformed_params.mu}")
    print(f"sigma2:\n{transformed_params.sigma2}")
    print(f"Q_h:\n{transformed_params.Q_h}")

    print("\nUntransformed Parameters (After Round-Trip):")
    print(f"lambda_r:\n{untransformed_params.lambda_r}")
    print(f"Phi_f:\n{untransformed_params.Phi_f}")
    print(f"Phi_h:\n{untransformed_params.Phi_h}")
    print(f"mu:\n{untransformed_params.mu}")
    print(f"sigma2:\n{untransformed_params.sigma2}")
    print(f"Q_h:\n{untransformed_params.Q_h}")

    # Check round-trip accuracy
    print("\nRound-Trip Accuracy:")
    # Compare original params with untransformed params
    print(f"lambda_r Mean Absolute Error: {np.mean(np.abs(np.array(params.lambda_r) - np.array(untransformed_params.lambda_r))):.6f}")
    print(f"Phi_f Mean Absolute Error: {np.mean(np.abs(np.array(params.Phi_f) - np.array(untransformed_params.Phi_f))):.6f}")
    print(f"Phi_h Mean Absolute Error: {np.mean(np.abs(np.array(params.Phi_h) - np.array(untransformed_params.Phi_h))):.6f}")
    print(f"mu Mean Absolute Error: {np.mean(np.abs(np.array(params.mu) - np.array(untransformed_params.mu))):.6f}")
    print(f"sigma2 Mean Absolute Error: {np.mean(np.abs(np.array(params.sigma2) - np.array(untransformed_params.sigma2))):.6f}")
    print(f"Q_h Mean Absolute Error: {np.mean(np.abs(np.array(params.Q_h) - np.array(untransformed_params.Q_h))):.6f}")


def optimize_with_and_without_transformations(params, returns, max_steps=100):
    """
    Compare optimization with and without parameter transformations.

    Args:
        params (DFSVParamsDataclass): True model parameters
        returns (np.ndarray): Simulated returns
        max_steps (int): Maximum number of optimization steps

    Returns:
        dict: Dictionary containing optimization results
    """
    print("\nComparing Optimization With and Without Transformations")
    print("======================================================")

    # Convert returns to JAX array
    jax_returns = jnp.array(returns)

    # Create filter instance
    N, K = params.N, params.K
    filter_instance = DFSVBellmanInformationFilter(N, K)

    # Create uninformed initial parameters
    uninformed_params = create_uninformed_parameters(params)

    # Create transformed parameters
    transformed_params = transform_params(uninformed_params)

    # Define objective functions with JIT compilation (IMPORTANT for speed)
    @eqx.filter_jit
    def standard_objective(p, args=None):
        """Standard objective function without transformations."""
        loss = bellman_objective(p, jax_returns, filter_instance)
        return loss  # Return just the loss for optimistix

    @eqx.filter_jit
    def transformed_objective(p_t, args=None):
        """Transformed objective function with parameter transformations."""
        # Untransform parameters
        p = untransform_params(p_t)
        # Calculate loss
        loss = bellman_objective(p, jax_returns, filter_instance)
        return loss  # Return just the loss for optimistix

    # Create optimizer
    solver = DampedTrustRegionBFGS(
        rtol=1e-5,
        atol=1e-5,
        norm=optx.rms_norm,
        verbose=frozenset({"step", "loss"})
    )

    # Run standard optimization
    print("\nStarting standard optimization (without transformations)...")
    start_time = time.time()
    try:
        result_standard = optx.minimise(
            fn=standard_objective,
            solver=solver,
            y0=uninformed_params,
            max_steps=max_steps,
            throw=False
        )
        standard_success = True
    except Exception as e:
        print(f"Standard optimization failed with error: {e}")
        result_standard = None
        standard_success = False
    standard_time = time.time() - start_time
    print(f"Standard optimization took {standard_time:.2f} seconds")

    # Run transformed optimization
    print("\nStarting transformed optimization...")
    start_time = time.time()
    try:
        result_transformed = optx.minimise(
            fn=transformed_objective,
            solver=solver,
            y0=transformed_params,
            max_steps=max_steps,
            throw=False
        )
        transformed_success = True
    except Exception as e:
        print(f"Transformed optimization failed with error: {e}")
        result_transformed = None
        transformed_success = False
    transformed_time = time.time() - start_time
    print(f"Transformed optimization took {transformed_time:.2f} seconds")

    # Compare results
    if standard_success and transformed_success:
        print("\nOptimization Results Comparison:")
        # Calculate final loss using the objective function directly
        standard_final_loss = standard_objective(result_standard.value)
        # Untransform the parameters for the transformed result
        untransformed_params = untransform_params(result_transformed.value)
        transformed_final_loss = bellman_objective(untransformed_params, jax_returns, filter_instance)
        print(f"Standard final loss: {standard_final_loss}")
        print(f"Transformed final loss: {transformed_final_loss}")

        # Print parameter comparison
        print("\nParameter Comparison:")

        # Helper function to format arrays for display
        def format_param(param):
            param_array = np.array(param)
            if param_array.size == 1:
                return f"{param_array.item():.4f}"
            elif param_array.size <= 4:  # Small arrays
                return str(param_array.round(4))
            else:  # Larger arrays
                return f"Shape: {param_array.shape}, Mean: {param_array.mean():.4f}"

        # Untransform the parameters from the transformed optimization
        # We already have untransformed_params from above

        # Compare each parameter
        param_names = ['lambda_r', 'Phi_f', 'Phi_h', 'mu', 'sigma2', 'Q_h']
        for param_name in param_names:
            print(f"\n{param_name}:")
            print(f"  True:        {format_param(getattr(params, param_name))}")
            print(f"  Standard:    {format_param(getattr(result_standard.value, param_name))}")
            print(f"  Transformed: {format_param(getattr(untransformed_params, param_name))}")


    # Return results
    return {
        'standard': {
            'result': result_standard,
            'time': standard_time,
            'success': standard_success
        },
        'transformed': {
            'result': result_transformed,
            'time': transformed_time,
            'success': transformed_success
        }
    }


def plot_optimization_comparison(optimization_results):
    """
    Plot comparison of optimization with and without transformations.

    Args:
        optimization_results (dict): Dictionary containing optimization results
    """
    # Check if both optimizations were successful
    standard_success = optimization_results['standard']['success']
    transformed_success = optimization_results['transformed']['success']

    if not (standard_success and transformed_success):
        print("Cannot plot comparison because one or both optimizations failed.")
        return


    # Plot computation time
    plt.figure(figsize=(8, 5))
    method_names = ['Standard', 'Transformed']
    times = [
        optimization_results['standard']['time'],
        optimization_results['transformed']['time']
    ]
    plt.bar(method_names, times, color=['blue', 'green'])
    plt.title('Optimization Time')
    plt.ylabel('Time (seconds)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


def run_filters_and_plot_states(params, result_standard, result_transformed, returns, factors, log_vols):
    """Run filters with optimized parameters and plot state estimates vs true values.

    Args:
        params (DFSVParamsDataclass): True model parameters
        result_standard (optx.Solution): Standard optimization result
        result_transformed (optx.Solution): Transformed optimization result
        returns (np.ndarray): Observed returns
        factors (np.ndarray): True factors
        log_vols (np.ndarray): True log-volatilities
    """
    print("\nRunning filters with optimized parameters...")

    # Create filter instance
    filter_instance = DFSVBellmanInformationFilter(params.N, params.K)

    # Convert returns to JAX array
    jax_returns = jnp.array(returns)

    # Run filter with true parameters
    print("Running filter with true parameters...")
    _, _, true_ll = filter_instance.filter_scan(params, jax_returns)
    true_filtered_states = np.array(filter_instance.filtered_states)

    # Run filter with standard optimization result
    print("Running filter with standard optimization result...")
    _, _, std_ll = filter_instance.filter_scan(result_standard.value, jax_returns)
    std_filtered_states = np.array(filter_instance.filtered_states)

    # Untransform the transformed parameters
    untransformed_params = untransform_params(result_transformed.value)

    # Run filter with transformed optimization result
    print("Running filter with transformed optimization result...")
    _, _, trans_ll = filter_instance.filter_scan(untransformed_params, jax_returns)
    trans_filtered_states = np.array(filter_instance.filtered_states)

    # Print log-likelihoods
    print(f"\nLog-likelihoods:")
    print(f"  True parameters: {float(true_ll):.4f}")
    print(f"  Standard optimization: {float(std_ll):.4f}")
    print(f"  Transformed optimization: {float(trans_ll):.4f}")

    # Plot factors
    T = returns.shape[0]
    time_index = np.arange(T)

    # Create figure with 2 rows (factors and log-vols) and K columns
    K = params.K
    fig, axes = plt.subplots(2, K, figsize=(15, 10), sharex=True)
    if K == 1:
        axes = axes.reshape(2, 1)

    # Plot factors
    for k in range(K):
        ax = axes[0, k]
        ax.plot(time_index, factors[:, k], 'k-', label='True')
        ax.plot(time_index, true_filtered_states[:, k], 'g--', label='True Params')
        ax.plot(time_index, std_filtered_states[:, k], 'b-.', label='Standard Opt')
        ax.plot(time_index, trans_filtered_states[:, k], 'r:', label='Transformed Opt')
        ax.set_title(f'Factor {k+1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    # Plot log-volatilities
    for k in range(K):
        ax = axes[1, k]
        ax.plot(time_index, log_vols[:, k], 'k-', label='True')
        ax.plot(time_index, true_filtered_states[:, k+K], 'g--', label='True Params')
        ax.plot(time_index, std_filtered_states[:, k+K], 'b-.', label='Standard Opt')
        ax.plot(time_index, trans_filtered_states[:, k+K], 'r:', label='Transformed Opt')
        ax.set_title(f'Log-Volatility {k+1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Calculate and plot state estimation errors
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Factor estimation errors
    factor_errors_true = np.mean(np.abs(true_filtered_states[:, :K] - factors), axis=0)
    factor_errors_std = np.mean(np.abs(std_filtered_states[:, :K] - factors), axis=0)
    factor_errors_trans = np.mean(np.abs(trans_filtered_states[:, :K] - factors), axis=0)

    # Log-vol estimation errors
    logvol_errors_true = np.mean(np.abs(true_filtered_states[:, K:2*K] - log_vols), axis=0)
    logvol_errors_std = np.mean(np.abs(std_filtered_states[:, K:2*K] - log_vols), axis=0)
    logvol_errors_trans = np.mean(np.abs(trans_filtered_states[:, K:2*K] - log_vols), axis=0)

    # Plot factor errors
    x = np.arange(K)
    width = 0.25
    axes[0].bar(x - width, factor_errors_true, width, label='True Params')
    axes[0].bar(x, factor_errors_std, width, label='Standard Opt')
    axes[0].bar(x + width, factor_errors_trans, width, label='Transformed Opt')
    axes[0].set_title('Mean Absolute Error in Factor Estimation')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'Factor {k+1}' for k in range(K)])
    axes[0].legend()
    axes[0].grid(True, axis='y')

    # Plot log-vol errors
    axes[1].bar(x - width, logvol_errors_true, width, label='True Params')
    axes[1].bar(x, logvol_errors_std, width, label='Standard Opt')
    axes[1].bar(x + width, logvol_errors_trans, width, label='Transformed Opt')
    axes[1].set_title('Mean Absolute Error in Log-Volatility Estimation')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'Log-Vol {k+1}' for k in range(K)])
    axes[1].legend()
    axes[1].grid(True, axis='y')

    plt.tight_layout()
    plt.show()


def main():
    """Run the parameter transformation example."""
    print("Parameter Transformation Example for DFSV Models")
    print("==============================================")

    # Create model parameters
    N, K = 3, 1  # 3 observed series, 1 factor
    params = create_simple_dfsv_model(N, K)
    print(f"Created DFSV model with {params.N} observed series and {params.K} factors")

    # Demonstrate parameter transformations
    demonstrate_parameter_transformations(params)

    # Set simulation parameters
    T = 500  # Number of time periods (shorter for faster optimization)
    seed = 42  # Random seed for reproducibility

    # Run simulation
    print(f"\nSimulating DFSV model for T={T} time periods...")
    returns, factors, log_vols = simulate_DFSV(
        params=params,
        T=T,
        seed=seed
    )
    print("Simulation complete!")

    # Compare optimization with and without transformations
    optimization_results = optimize_with_and_without_transformations(
        params, returns, max_steps=50
    )

    # Plot optimization comparison
    plot_optimization_comparison(optimization_results)

    # If both optimizations were successful, run filters and plot state estimates
    if optimization_results['standard']['success'] and optimization_results['transformed']['success']:
        result_standard = optimization_results['standard']['result']
        result_transformed = optimization_results['transformed']['result']
        run_filters_and_plot_states(params, result_standard, result_transformed, returns, factors, log_vols)

    return params, returns, factors, log_vols, optimization_results


if __name__ == "__main__":
    main()
