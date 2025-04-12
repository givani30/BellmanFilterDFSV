#!/usr/bin/env python
"""
Simplified Learning Rate Scheduler Comparison Example for DFSV Models

This example demonstrates how to:
1. Create a DFSV model and simulate data
2. Compare different learning rate schedulers for parameter estimation
3. Analyze scheduler performance in terms of convergence speed
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import time
import pandas as pd
import os
from pathlib import Path
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.utils.optimization import (
    FilterType,
    run_optimization,
    OptimizerResult
)
from bellman_filter_dfsv.utils.optimization_helpers import create_stable_initial_params
from bellman_filter_dfsv.utils.transformations import apply_identification_constraint

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory if it doesn't exist
output_dir = Path("outputs/scheduler_comparison")
output_dir.mkdir(parents=True, exist_ok=True)


def create_test_data(N=3, K=2, T=300):
    """
    Create test data for optimization experiments.

    Args:
        N (int): Number of observed variables
        K (int): Number of factors
        T (int): Number of time points

    Returns:
        tuple: (true_params, returns) - True parameters and simulated returns
    """
    # Create true parameters
    true_params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=jnp.array([
            [1.0, 0.0],
            [0.8, 0.0],
            [0.6, 0.0]
        ]),
        Phi_f=jnp.array([
            [0.9, 0.0],
            [0.0, 0.8]
        ]),
        Phi_h=jnp.array([
            [0.95, 0.0],
            [0.0, 0.92]
        ]),
        mu=jnp.zeros(K),
        sigma2=jnp.ones(N) * 0.1,
        Q_h=jnp.array([
            [0.2, 0.0],
            [0.0, 0.15]
        ])
    )

    # Apply identification constraint to ensure parameters are properly identified
    true_params = apply_identification_constraint(true_params)

    # Simulate data
    returns, _, _ = simulate_DFSV(true_params, T)

    return true_params, returns


def run_scheduler_experiment(
    true_params,
    returns,
    optimizer_name="AdamW",
    scheduler_type="warmup_cosine",
    learning_rate=5e-4,  # Reduced learning rate for stability
    max_learning_rate=1e-3,  # Reduced max learning rate
    min_learning_rate=1e-6,
    decay_steps=400,  # Increased decay steps
    warmup_steps=40,  # Increased warmup steps
    cycle_period=80,  # Increased cycle period
    step_size_factor=0.5,
    step_interval=100,  # Increased step interval
    filter_type=FilterType.BIF,
    max_steps=400,  # Increased max steps
    fix_mu=True,
    use_transformations=True,
    verbose=True
) -> OptimizerResult:
    """
    Run an experiment with a specific scheduler configuration.

    Args:
        true_params: True model parameters
        returns: Simulated returns data
        optimizer_name: Name of the optimizer to use
        scheduler_type: Type of learning rate scheduler
        learning_rate: Initial learning rate
        max_learning_rate: Maximum learning rate (for schedulers with peaks)
        min_learning_rate: Minimum learning rate
        decay_steps: Number of steps for learning rate decay
        warmup_steps: Number of warmup steps
        cycle_period: Number of steps per cycle for cyclic schedulers
        step_size_factor: Factor to reduce learning rate at each step for step decay
        step_interval: Number of steps between learning rate reductions for step decay
        filter_type: Type of filter to use
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

    # Print experiment configuration
    print(f"\n--- Running experiment with {optimizer_name} and {scheduler_type} scheduler ---")
    print(f"Learning rate: {learning_rate}, Max LR: {max_learning_rate}, Min LR: {min_learning_rate}")
    print(f"Decay steps: {decay_steps}, Warmup steps: {warmup_steps}")
    if scheduler_type == "cyclic":
        print(f"Cycle period: {cycle_period}")
    elif scheduler_type == "step_decay":
        print(f"Step size factor: {step_size_factor}, Step interval: {step_interval}")

    # Run optimization
    result = run_optimization(
        filter_type=filter_type,
        returns=returns,
        initial_params=initial_params,
        true_params=true_params if fix_mu else None,
        use_transformations=use_transformations,
        optimizer_name=optimizer_name,
        priors=None,
        stability_penalty_weight=1000.0,
        max_steps=max_steps,
        verbose=verbose,
        log_params=True,  # Enable parameter logging
        log_interval=1,  # Log at every step
        learning_rate=learning_rate,
        rtol=1e-5,
        atol=1e-5,
        fix_mu=fix_mu,
        # Pass scheduler-specific parameters
        scheduler_type=scheduler_type,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_steps=warmup_steps,
        cycle_period=cycle_period,
        step_size_factor=step_size_factor,
        step_interval=step_interval
    )
    return result


def compare_schedulers(true_params, returns, optimizer_name="AdamW", filter_type=FilterType.BIF, max_steps=400):  # Increased max steps
    """
    Compare different learning rate schedulers for a specific optimizer.

    Args:
        true_params: True model parameters
        returns: Simulated returns data
        optimizer_name: Name of the optimizer to use
        filter_type: Type of filter to use
        max_steps: Maximum number of optimization steps

    Returns:
        dict: Dictionary containing optimization results for each scheduler
    """
    # Define schedulers to compare (simplified list)
    scheduler_configs = {
        "cosine": {
            "learning_rate": 5e-4,  # Reduced learning rate for stability
            "max_learning_rate": 1e-3,  # Reduced max learning rate
            "min_learning_rate": 1e-6,
            "decay_steps": max_steps,
            "warmup_steps": 0
        },
        "warmup_cosine": {
            "learning_rate": 5e-4,  # Reduced learning rate for stability
            "max_learning_rate": 1e-3,  # Reduced max learning rate
            "min_learning_rate": 1e-6,
            "decay_steps": max_steps,
            "warmup_steps": int(max_steps * 0.1)
        },
        "constant": {
            "learning_rate": 5e-4,  # Reduced learning rate for stability
            "max_learning_rate": 1e-3,  # Reduced max learning rate
            "min_learning_rate": 1e-6,
            "decay_steps": max_steps,
            "warmup_steps": 0
        }
    }

    # Run optimization with each scheduler
    results = {}
    for scheduler_type, config in scheduler_configs.items():
        print(f"\n=== Testing {scheduler_type} scheduler with {optimizer_name} ===")
        try:
            result = run_scheduler_experiment(
                true_params=true_params,
                returns=returns,
                optimizer_name=optimizer_name,
                scheduler_type=scheduler_type,
                filter_type=filter_type,
                max_steps=max_steps,
                **config
            )
            results[scheduler_type] = result
            print(f"Final loss: {result.final_loss:.6f}, Success: {result.success}, Steps: {result.steps}")
        except Exception as e:
            print(f"Error with {scheduler_type} scheduler: {str(e)}")
            results[scheduler_type] = None

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


def plot_learning_rate_schedules(max_steps=400):  # Increased max steps
    """
    Plot learning rate schedules for different scheduler types.

    Args:
        max_steps: Maximum number of steps
    """
    from bellman_filter_dfsv.utils.solvers import create_learning_rate_scheduler

    # Define scheduler types
    scheduler_types = [
        "cosine", "warmup_cosine", "constant"
    ]

    # Create figure
    plt.figure(figsize=(12, 8))

    # Generate step values
    steps = np.arange(max_steps)

    # Plot each scheduler
    for scheduler_type in scheduler_types:
        # Create scheduler
        if scheduler_type == "warmup_cosine":
            scheduler = create_learning_rate_scheduler(
                init_lr=1e-3,
                decay_steps=max_steps,
                min_lr=1e-6,
                warmup_steps=int(max_steps * 0.1),
                scheduler_type=scheduler_type,
                peak_lr=1e-2
            )
        else:
            scheduler = create_learning_rate_scheduler(
                init_lr=1e-3,
                decay_steps=max_steps,
                min_lr=1e-6,
                warmup_steps=0,
                scheduler_type=scheduler_type
            )

        # Get learning rates
        learning_rates = [float(scheduler(jnp.array(step))) for step in steps]

        # Plot learning rate curve
        plt.plot(steps, learning_rates, label=scheduler_type)

    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.yscale('log')  # Use log scale for better visualization
    plt.title('Learning Rate Schedules')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig(output_dir / 'learning_rate_schedules_simple.png')
    plt.close()


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


def main():
    """Main function to run the scheduler comparison example."""
    print("Starting simplified scheduler comparison example...")

    # Create test data
    print("Creating test data...")
    true_params, returns = create_test_data(N=3, K=2, T=300)  # Increased time points

    # Plot learning rate schedules
    print("Plotting learning rate schedules...")
    plot_learning_rate_schedules(max_steps=400)  # Increased max steps

    # Compare schedulers with AdamW (baseline)
    print("\nComparing schedulers with AdamW...")
    scheduler_results = compare_schedulers(
        true_params=true_params,
        returns=returns,
        optimizer_name="AdamW",
        filter_type=FilterType.BIF,
        max_steps=400  # Increased max steps
    )

    # Plot loss curves for scheduler comparison
    plot_loss_curves(
        results=scheduler_results,
        title='Loss Curves for Different Schedulers with AdamW',
        filename='scheduler_comparison_loss_curves_simple.png'
    )

    # Save scheduler comparison results to CSV
    save_results_to_csv(
        results=scheduler_results,
        filename='scheduler_comparison_results_simple.csv'
    )

    print("\nSimplified scheduler comparison example completed!")


if __name__ == "__main__":
    main()
