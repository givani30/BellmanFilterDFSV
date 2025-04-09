#!/usr/bin/env python
"""
Parameter Optimization Example for DFSV Models

This example demonstrates how to:
1. Create a DFSV model and simulate data
2. Estimate model parameters using different filters:
   - Bellman Filter (BF)
   - Bellman Information Filter (BIF)
   - Particle Filter (PF)
3. Compare estimated parameters to true parameters
4. Visualize optimization results
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import time
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.utils.optimization import (
    FilterType,
    run_optimization,
    OptimizerResult
)

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


def perturb_parameters(params, perturbation_scale=0.2):
    """
    Create perturbed parameters as initial guess for optimization.

    Args:
        params (DFSVParamsDataclass): True model parameters
        perturbation_scale (float): Scale of random perturbation

    Returns:
        DFSVParamsDataclass: Perturbed parameters
    """
    # Create random keys for each parameter
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 6)

    # Perturb each parameter
    perturbed_params = params.replace(
        lambda_r=params.lambda_r + perturbation_scale * jax.random.normal(keys[0], params.lambda_r.shape),
        Phi_f=params.Phi_f + perturbation_scale * jax.random.normal(keys[1], params.Phi_f.shape),
        Phi_h=params.Phi_h + perturbation_scale * jax.random.normal(keys[2], params.Phi_h.shape),
        mu=params.mu + perturbation_scale * jax.random.normal(keys[3], params.mu.shape),
        sigma2=params.sigma2 + perturbation_scale * jax.random.normal(keys[4], params.sigma2.shape),
        Q_h=params.Q_h + perturbation_scale * jax.random.normal(keys[5], params.Q_h.shape)
    )

    return perturbed_params


def create_uninformed_parameters(params):
    """
    Create uninformed initial parameters for optimization.

    Args:
        params (DFSVParamsDataclass): True model parameters (used only for dimensions)

    Returns:
        DFSVParamsDataclass: Uninformed parameters
    """
    N, K = params.N, params.K

    # Create uninformed parameters
    uninformed_params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=jnp.ones((N, K)) * 0.5,  # Moderate positive loadings
        Phi_f=jnp.eye(K) * 0.8,  # Moderate persistence
        Phi_h=jnp.eye(K) * 0.8,  # Moderate persistence
        mu=jnp.zeros(K),  # Zero mean for log volatility
        sigma2=jnp.ones(N) * 0.5,  # Moderate idiosyncratic variance
        Q_h=jnp.eye(K) * 0.2  # Moderate volatility of volatility
    )

    return uninformed_params


def run_parameter_optimization(true_params: DFSVParamsDataclass, returns:np.ndarray):
    """
    Run parameter optimization using different filters.

    Args:
        true_params (DFSVParamsDataclass): True model parameters
        returns (np.ndarray): Simulated returns with shape (T, N)

    Returns:
        dict: Dictionary containing optimization results
    """
    # Convert returns to JAX array
    jax_returns = jnp.array(returns)

    # Create uninformed initial parameters
    initial_params = create_uninformed_parameters(true_params)

    # Run optimization with Bellman Filter
    print("\nRunning optimization with Bellman Filter... (NOTE: This specific filter can be unstable during optimization)")
    start_time = time.time()
    bf_result = run_optimization(
        filter_type=FilterType.BF,
        returns=jax_returns,
        true_params=true_params,  # For comparison only
        use_transformations=True,
        optimizer_name="DampedTrustRegionBFGS",
        max_steps=100,
        log_params=True,
        verbose=True
    )
    bf_time = time.time() - start_time
    print(f"BF optimization completed in {bf_time:.2f} seconds")

    # Run optimization with Bellman Information Filter
    print("\nRunning optimization with Bellman Information Filter...")
    start_time = time.time()
    bif_result = run_optimization(
        filter_type=FilterType.BIF,
        returns=jax_returns,
        true_params=true_params,  # For comparison only
        use_transformations=True,
        optimizer_name="DampedTrustRegionBFGS",
        max_steps=100,
        log_params=True,
        verbose=True
    )
    bif_time = time.time() - start_time
    print(f"BIF optimization completed in {bif_time:.2f} seconds")

    # Run optimization with Particle Filter
    print("\nRunning optimization with Particle Filter...")
    start_time = time.time()
    pf_result = run_optimization(
        filter_type=FilterType.PF,
        returns=jax_returns,
        true_params=true_params,  # For comparison only
        use_transformations=True,
        optimizer_name="Adam",  # Adam works better for PF
        max_steps=100,
        num_particles=1000,
        log_params=True,
        verbose=True
    )
    pf_time = time.time() - start_time
    print(f"PF optimization completed in {pf_time:.2f} seconds")

    # Return results
    results = {
        'bf': {
            'result': bf_result,
            'time': bf_time
        },
        'bif': {
            'result': bif_result,
            'time': bif_time
        },
        'pf': {
            'result': pf_result,
            'time': pf_time
        }
    }

    return results


def compare_parameters(true_params, optimization_results):
    """
    Compare estimated parameters to true parameters.

    Args:
        true_params (DFSVParamsDataclass): True model parameters
        optimization_results (dict): Dictionary containing optimization results
    """
    # Extract estimated parameters
    bf_params = optimization_results['bf']['result'].final_params
    bif_params = optimization_results['bif']['result'].final_params
    pf_params = optimization_results['pf']['result'].final_params

    # Print parameter comparison
    print("\nParameter Comparison:")
    print("=====================")

    # Lambda_r comparison
    print("\nFactor Loadings (lambda_r):")
    print(f"True: \n{true_params.lambda_r}")
    print(f"BF Estimated: \n{bf_params.lambda_r}")
    print(f"BIF Estimated: \n{bif_params.lambda_r}")
    print(f"PF Estimated: \n{pf_params.lambda_r}")

    # Phi_f comparison
    print("\nFactor Persistence (Phi_f):")
    print(f"True: \n{true_params.Phi_f}")
    print(f"BF Estimated: \n{bf_params.Phi_f}")
    print(f"BIF Estimated: \n{bif_params.Phi_f}")
    print(f"PF Estimated: \n{pf_params.Phi_f}")

    # Phi_h comparison
    print("\nVolatility Persistence (Phi_h):")
    print(f"True: \n{true_params.Phi_h}")
    print(f"BF Estimated: \n{bf_params.Phi_h}")
    print(f"BIF Estimated: \n{bif_params.Phi_h}")
    print(f"PF Estimated: \n{pf_params.Phi_h}")

    # mu comparison
    print("\nLong-run Mean (mu):")
    print(f"True: \n{true_params.mu}")
    print(f"BF Estimated: \n{bf_params.mu}")
    print(f"BIF Estimated: \n{bif_params.mu}")
    print(f"PF Estimated: \n{pf_params.mu}")

    # sigma2 comparison
    print("\nIdiosyncratic Variance (sigma2):")
    print(f"True: \n{true_params.sigma2}")
    print(f"BF Estimated: \n{bf_params.sigma2}")
    print(f"BIF Estimated: \n{bif_params.sigma2}")
    print(f"PF Estimated: \n{pf_params.sigma2}")

    # Q_h comparison
    print("\nVolatility Covariance (Q_h):")
    print(f"True: \n{true_params.Q_h}")
    print(f"BF Estimated: \n{bf_params.Q_h}")
    print(f"BIF Estimated: \n{bif_params.Q_h}")
    print(f"PF Estimated: \n{pf_params.Q_h}")

    # Calculate parameter errors
    def param_error(true, est):
        """Calculate relative error between true and estimated parameters."""
        return np.mean(np.abs((true - est) / (np.abs(true) + 1e-8)))

    # Calculate errors for each parameter
    bf_errors = {
        'lambda_r': param_error(np.array(true_params.lambda_r), np.array(bf_params.lambda_r)),
        'Phi_f': param_error(np.array(true_params.Phi_f), np.array(bf_params.Phi_f)),
        'Phi_h': param_error(np.array(true_params.Phi_h), np.array(bf_params.Phi_h)),
        'mu': param_error(np.array(true_params.mu), np.array(bf_params.mu)),
        'sigma2': param_error(np.array(true_params.sigma2), np.array(bf_params.sigma2)),
        'Q_h': param_error(np.array(true_params.Q_h), np.array(bf_params.Q_h))
    }

    bif_errors = {
        'lambda_r': param_error(np.array(true_params.lambda_r), np.array(bif_params.lambda_r)),
        'Phi_f': param_error(np.array(true_params.Phi_f), np.array(bif_params.Phi_f)),
        'Phi_h': param_error(np.array(true_params.Phi_h), np.array(bif_params.Phi_h)),
        'mu': param_error(np.array(true_params.mu), np.array(bif_params.mu)),
        'sigma2': param_error(np.array(true_params.sigma2), np.array(bif_params.sigma2)),
        'Q_h': param_error(np.array(true_params.Q_h), np.array(bif_params.Q_h))
    }

    pf_errors = {
        'lambda_r': param_error(np.array(true_params.lambda_r), np.array(pf_params.lambda_r)),
        'Phi_f': param_error(np.array(true_params.Phi_f), np.array(pf_params.Phi_f)),
        'Phi_h': param_error(np.array(true_params.Phi_h), np.array(pf_params.Phi_h)),
        'mu': param_error(np.array(true_params.mu), np.array(pf_params.mu)),
        'sigma2': param_error(np.array(true_params.sigma2), np.array(pf_params.sigma2)),
        'Q_h': param_error(np.array(true_params.Q_h), np.array(pf_params.Q_h))
    }

    # Print parameter errors
    print("\nParameter Relative Errors:")
    print("=========================")

    param_names = ['lambda_r', 'Phi_f', 'Phi_h', 'mu', 'sigma2', 'Q_h']

    for param in param_names:
        print(f"\n{param}:")
        print(f"BF Error: {bf_errors[param]:.4f}")
        print(f"BIF Error: {bif_errors[param]:.4f}")
        print(f"PF Error: {pf_errors[param]:.4f}")

    # Return errors
    return {
        'bf': bf_errors,
        'bif': bif_errors,
        'pf': pf_errors
    }


def plot_optimization_results(optimization_results):
    """
    Plot optimization results.

    Args:
        optimization_results (dict): Dictionary containing optimization results
    """
    # Extract loss histories
    bf_loss_history = optimization_results['bf']['result'].loss_history
    bif_loss_history = optimization_results['bif']['result'].loss_history
    pf_loss_history = optimization_results['pf']['result'].loss_history

    # Plot loss histories
    plt.figure(figsize=(10, 6))
    plt.plot(bf_loss_history, 'b-', label='BF')
    plt.plot(bif_loss_history, 'g-', label='BIF')
    plt.plot(pf_loss_history, 'r-', label='PF')
    plt.title('Optimization Loss History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Log scale for better visualization
    plt.tight_layout()
    plt.show()

    # Plot computation time
    plt.figure(figsize=(8, 5))
    filter_names = ['BF', 'BIF', 'PF']
    times = [
        optimization_results['bf']['time'],
        optimization_results['bif']['time'],
        optimization_results['pf']['time']
    ]
    plt.bar(filter_names, times, color=['blue', 'green', 'red'])
    plt.title('Optimization Time')
    plt.ylabel('Time (seconds)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


def main():
    """Run the parameter optimization example."""
    print("Parameter Optimization Example for DFSV Models")
    print("=============================================")

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

    # Run parameter optimization
    optimization_results = run_parameter_optimization(true_params, returns)

    # Compare parameters
    param_errors = compare_parameters(true_params, optimization_results)

    # Plot optimization results
    plot_optimization_results(optimization_results)

    return true_params, returns, factors, log_vols, optimization_results, param_errors


if __name__ == "__main__":
    main()
