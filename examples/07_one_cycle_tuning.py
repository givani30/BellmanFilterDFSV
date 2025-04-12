"""
Example script to compare different one-cycle scheduler configurations with AdamW.

This script tests various configurations of the one-cycle learning rate scheduler
with the AdamW optimizer to find the optimal balance between stability and
optimization speed.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.utils.optimization import (
    FilterType,
    run_optimization,
    OptimizerResult
)
from bellman_filter_dfsv.utils.optimization_helpers import create_stable_initial_params
from bellman_filter_dfsv.utils.transformations import apply_identification_constraint

# Create output directory
output_dir = Path("outputs/one_cycle_tuning")
output_dir.mkdir(parents=True, exist_ok=True)

# Timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def create_test_data(N=3, K=2, T=300):
    """
    Create test data for optimization experiments.

    Args:
        N: Number of observed series
        K: Number of factors
        T: Number of time points

    Returns:
        Tuple of (true_params, returns)
    """
    # Create true parameters using jnp arrays to ensure compatibility
    true_params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=jnp.array([
            [1.0, 0.0],
            [0.8, 0.0],
            [0.6, 0.0]
        ]),
        Phi_f=jnp.array([
            [0.45, 0.02],
            [0.01, 0.44]
        ]),
        Phi_h=jnp.array([
            [0.98, 0.0],
            [0.0, 0.97]
        ]),
        mu=jnp.zeros(K),
        Q_h=jnp.array([
            [0.1, 0.0],
            [0.0, 0.1]
        ]),
        sigma2=jnp.array([0.1, 0.1, 0.1])
    )

    # Apply identification constraint
    true_params = apply_identification_constraint(true_params)

    # Simulate data
    returns, _, _ = simulate_DFSV(true_params, T)

    return true_params, returns


def run_one_cycle_experiment(
    true_params,
    returns,
    max_lr=1e-3,
    init_lr=5e-4,
    min_lr=1e-6,
    warmup_pct=0.1,
    div_factor=None,
    final_div_factor=None,
    cycle_shape="cosine",
    max_steps=400,
    fix_mu=True,
    use_transformations=True,
    verbose=True
) -> OptimizerResult:
    """
    Run an experiment with a specific one-cycle scheduler configuration.

    Args:
        true_params: True model parameters
        returns: Simulated returns data
        max_lr: Maximum learning rate during the cycle
        init_lr: Initial learning rate
        min_lr: Minimum learning rate at the end of the cycle
        warmup_pct: Percentage of steps for warmup phase
        div_factor: Division factor for initial learning rate (if None, calculated from init_lr and max_lr)
        final_div_factor: Division factor for final learning rate (if None, calculated from max_lr and min_lr)
        cycle_shape: Shape of the cycle ("cosine" or "linear")
        max_steps: Maximum number of optimization steps
        fix_mu: Whether to fix mu parameter to true value
        use_transformations: Whether to use parameter transformations
        verbose: Whether to print verbose output

    Returns:
        OptimizerResult: The result of the optimization run
    """
    # Create initial parameters
    N, K = true_params.N, true_params.K
    initial_params = create_stable_initial_params(N, K)

    # Calculate warmup steps
    warmup_steps = int(max_steps * warmup_pct)

    # These parameters are not used in the current implementation
    # but kept for future compatibility
    _ = div_factor
    _ = final_div_factor
    _ = cycle_shape

    # Print experiment configuration
    print("\n--- Running experiment with one_cycle scheduler ---")
    print(f"Max LR: {max_lr}, Init LR: {init_lr}, Min LR: {min_lr}")
    print(f"Warmup %: {warmup_pct} ({warmup_steps} steps)")

    # Run optimization
    result = run_optimization(
        filter_type=FilterType.BIF,
        returns=returns,
        initial_params=initial_params,
        true_params=true_params if fix_mu else None,
        use_transformations=use_transformations,
        optimizer_name="AdamW",
        priors=None,
        stability_penalty_weight=1000.0,
        max_steps=max_steps,
        verbose=verbose,
        log_params=True,  # Enable parameter logging
        log_interval=1,  # Log at every step
        rtol=1e-5,
        atol=1e-5,
        fix_mu=fix_mu,
        # One-cycle specific parameters
        scheduler_type="one_cycle",
        learning_rate=init_lr,
        max_learning_rate=max_lr,
        min_learning_rate=min_lr,
        warmup_steps=warmup_steps
    )
    return result


def compare_one_cycle_configs(true_params, returns, max_steps=400):
    """
    Compare different one-cycle scheduler configurations.

    Args:
        true_params: True model parameters
        returns: Simulated returns data
        max_steps: Maximum number of optimization steps

    Returns:
        dict: Dictionary containing optimization results for each configuration
    """
    # Define configurations to compare
    configs = {
        "default": {
            "max_lr": 1e-3,
            "init_lr": 5e-4,
            "min_lr": 1e-6,
            "warmup_pct": 0.1,
            "cycle_shape": "cosine"
        },
        "higher_max_lr": {
            "max_lr": 2e-3,
            "init_lr": 5e-4,
            "min_lr": 1e-6,
            "warmup_pct": 0.1,
            "cycle_shape": "cosine"
        },
        "lower_max_lr": {
            "max_lr": 5e-4,
            "init_lr": 2e-4,
            "min_lr": 1e-6,
            "warmup_pct": 0.1,
            "cycle_shape": "cosine"
        },
        "longer_warmup": {
            "max_lr": 1e-3,
            "init_lr": 5e-4,
            "min_lr": 1e-6,
            "warmup_pct": 0.2,
            "cycle_shape": "cosine"
        },
        "shorter_warmup": {
            "max_lr": 1e-3,
            "init_lr": 5e-4,
            "min_lr": 1e-6,
            "warmup_pct": 0.05,
            "cycle_shape": "cosine"
        },
        "linear_cycle": {
            "max_lr": 1e-3,
            "init_lr": 5e-4,
            "min_lr": 1e-6,
            "warmup_pct": 0.1,
            "cycle_shape": "linear"
        },
        "higher_min_lr": {
            "max_lr": 1e-3,
            "init_lr": 5e-4,
            "min_lr": 1e-5,
            "warmup_pct": 0.1,
            "cycle_shape": "cosine"
        }
    }

    # Run optimization with each configuration
    results = {}
    for name, config in configs.items():
        print(f"\n=== Testing {name} configuration ===")
        try:
            result = run_one_cycle_experiment(
                true_params=true_params,
                returns=returns,
                max_steps=max_steps,
                **config
            )
            results[name] = result
            print(f"Final loss: {result.final_loss:.6f}, Success: {result.success}, Steps: {result.steps}")
        except Exception as e:
            print(f"Error with {name} configuration: {str(e)}")
            results[name] = None

    return results


def plot_loss_curves(results, title, filename):
    """
    Plot loss curves for different optimization runs.

    Args:
        results: Dictionary of optimization results
        title: Plot title
        filename: Output filename
    """
    plt.figure(figsize=(12, 8))

    for name, result in results.items():
        if result is not None and hasattr(result, 'loss_history') and result.loss_history:
            # Get loss history
            loss_history = result.loss_history

            # Plot loss curve
            plt.plot(range(len(loss_history)), loss_history, label=f"{name}")

    plt.xlabel('Optimization Step')
    plt.ylabel('Loss')
    plt.yscale('log')  # Use log scale for better visualization
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig(output_dir / filename)
    plt.close()


def plot_learning_rate_schedules(configs, max_steps=400):
    """
    Plot learning rate schedules for different one-cycle configurations.

    Args:
        configs: Dictionary of configurations
        max_steps: Maximum number of steps
    """
    from bellman_filter_dfsv.utils.solvers import create_learning_rate_scheduler

    # Create figure
    plt.figure(figsize=(12, 8))

    # Generate step values
    steps = np.arange(max_steps)

    # Plot each scheduler
    for name, config in configs.items():
        # Extract parameters
        max_lr = config.get("max_lr", 1e-3)
        init_lr = config.get("init_lr", 5e-4)
        min_lr = config.get("min_lr", 1e-6)
        warmup_pct = config.get("warmup_pct", 0.1)
        # Note: cycle_shape is not used in the current implementation

        # Calculate warmup steps
        warmup_steps = int(max_steps * warmup_pct)

        # Create scheduler
        scheduler = create_learning_rate_scheduler(
            init_lr=init_lr,
            scheduler_type="one_cycle",
            peak_lr=max_lr,
            min_lr=min_lr,
            decay_steps=max_steps,
            warmup_steps=warmup_steps
        )

        # Get learning rates
        learning_rates = [float(scheduler(jnp.array(step))) for step in steps]

        # Plot learning rate curve
        plt.plot(steps, learning_rates, label=name)

    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.yscale('log')  # Use log scale for better visualization
    plt.title('One-Cycle Learning Rate Schedules')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig(output_dir / f'one_cycle_schedules_{timestamp}.png')
    plt.close()


def analyze_convergence_speed(results):
    """
    Analyze the convergence speed of different configurations.

    Args:
        results: Dictionary of optimization results

    Returns:
        DataFrame with convergence metrics
    """
    # Define loss thresholds for convergence analysis
    thresholds = [1e6, 5e5, 1e5, 5e4, 1e4]

    data = []
    for name, result in results.items():
        if result is not None and hasattr(result, 'loss_history') and result.loss_history:
            loss_history = result.loss_history

            # Find steps to reach each threshold
            steps_to_threshold = {}
            for threshold in thresholds:
                try:
                    step = next(i for i, loss in enumerate(loss_history) if loss < threshold)
                    steps_to_threshold[threshold] = step
                except StopIteration:
                    steps_to_threshold[threshold] = float('inf')

            # Calculate stability metrics
            loss_diffs = np.diff(loss_history)
            positive_jumps = np.sum(loss_diffs > 0)
            max_jump = np.max(loss_diffs) if len(loss_diffs) > 0 else 0

            row = {
                'Name': name,
                'Success': result.success,
                'Final Loss': result.final_loss if jnp.isfinite(result.final_loss) else float('inf'),
                'Steps': result.steps,
                'Time (s)': result.time_taken
            }

            # Add steps to thresholds
            for threshold in thresholds:
                row[f'Steps to {threshold:.0e}'] = steps_to_threshold[threshold]

            # Add stability metrics
            row['Positive Jumps'] = positive_jumps
            row['Max Jump'] = max_jump

            data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    return df


def save_results_to_csv(results, filename):
    """
    Save optimization results to a CSV file.

    Args:
        results: Dictionary of optimization results
        filename: Output filename
    """
    # Create data for CSV
    data = []

    # Process results
    for name, result in results.items():
        if result is not None:
            row = {
                'Name': name,
                'Success': result.success,
                'Final Loss': result.final_loss if jnp.isfinite(result.final_loss) else float('inf'),
                'Steps': result.steps,
                'Time (s)': result.time_taken
            }
            data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_dir / filename, index=False)
    print(f"Results saved to {output_dir / filename}")

    return df


def main():
    """Main function to run the one-cycle tuning experiment."""
    print("Starting one-cycle scheduler tuning experiment...")

    # Create test data
    print("Creating test data...")
    true_params, returns = create_test_data(N=3, K=2, T=300)

    # Define configurations to compare
    configs = {
        "default": {
            "max_lr": 1e-3,
            "init_lr": 5e-4,
            "min_lr": 1e-6,
            "warmup_pct": 0.1,
            "cycle_shape": "cosine"
        },
        "higher_max_lr": {
            "max_lr": 2e-3,
            "init_lr": 5e-4,
            "min_lr": 1e-6,
            "warmup_pct": 0.1,
            "cycle_shape": "cosine"
        },
        "lower_max_lr": {
            "max_lr": 5e-4,
            "init_lr": 2e-4,
            "min_lr": 1e-6,
            "warmup_pct": 0.1,
            "cycle_shape": "cosine"
        },
        "longer_warmup": {
            "max_lr": 1e-3,
            "init_lr": 5e-4,
            "min_lr": 1e-6,
            "warmup_pct": 0.2,
            "cycle_shape": "cosine"
        },
        "shorter_warmup": {
            "max_lr": 1e-3,
            "init_lr": 5e-4,
            "min_lr": 1e-6,
            "warmup_pct": 0.05,
            "cycle_shape": "cosine"
        },
        "linear_cycle": {
            "max_lr": 1e-3,
            "init_lr": 5e-4,
            "min_lr": 1e-6,
            "warmup_pct": 0.1,
            "cycle_shape": "linear"
        },
        "higher_min_lr": {
            "max_lr": 1e-3,
            "init_lr": 5e-4,
            "min_lr": 1e-5,
            "warmup_pct": 0.1,
            "cycle_shape": "cosine"
        }
    }

    # Plot learning rate schedules
    print("Plotting learning rate schedules...")
    plot_learning_rate_schedules(configs, max_steps=400)

    # Compare one-cycle configurations
    print("\nComparing one-cycle configurations...")
    results = compare_one_cycle_configs(
        true_params=true_params,
        returns=returns,
        max_steps=400
    )

    # Plot loss curves
    plot_loss_curves(
        results=results,
        title='Loss Curves for Different One-Cycle Configurations with AdamW',
        filename=f'one_cycle_loss_curves_{timestamp}.png'
    )

    # Analyze convergence speed
    print("\nAnalyzing convergence speed...")
    convergence_df = analyze_convergence_speed(results)
    convergence_df.to_csv(output_dir / f'one_cycle_convergence_analysis_{timestamp}.csv', index=False)
    print(f"Convergence analysis saved to {output_dir / f'one_cycle_convergence_analysis_{timestamp}.csv'}")

    # Save results to CSV
    basic_df = save_results_to_csv(
        results=results,
        filename=f'one_cycle_results_{timestamp}.csv'
    )

    print("\nOne-cycle scheduler tuning experiment completed!")

    # Print summary of results
    print("\nSummary of results:")
    print(basic_df.sort_values('Final Loss').to_string(index=False))


if __name__ == "__main__":
    main()
