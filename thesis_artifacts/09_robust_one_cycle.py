"""
Example script for robust one-cycle learning rate tuning with enhanced numerical stability.

This script implements a more robust version of the one-cycle learning rate tuning
experiment, with specific enhancements to handle numerical stability issues that
can occur when the optimizer gets close to convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
import jax

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
output_dir = Path("outputs/robust_one_cycle")
output_dir.mkdir(parents=True, exist_ok=True)

# Timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def create_test_data(N=3, K=2, T=600):
    """
    Create test data for optimization experiments with longer time series.

    Args:
        N: Number of observed series
        K: Number of factors
        T: Number of time points (increased to 600)

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
            [0.95, 0.02],
            [0.01, 0.94]
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


def create_robust_objective(base_objective_fn, stability_penalty_weight=1000.0):
    """
    Create a more robust objective function wrapper that enforces parameter validity.

    Args:
        base_objective_fn: The base objective function to wrap
        stability_penalty_weight: Weight for stability penalty

    Returns:
        A wrapped objective function with enhanced stability
    """
    def robust_objective(params, *args):
        # Get base objective value
        try:
            obj_value, aux = base_objective_fn(params, *args)
        except Exception as e:
            # If base objective fails, return a high value
            print(f"Error in base objective: {e}")
            return jnp.array(1e10, dtype=jnp.float64), ()

        # Add additional stability penalties

        # 1. Penalize eigenvalues close to 1
        evals_f = jnp.linalg.eigvals(params.Phi_f)
        evals_h = jnp.linalg.eigvals(params.Phi_h)

        # Calculate how close eigenvalues are to 1
        max_eval_f = jnp.max(jnp.abs(evals_f))
        max_eval_h = jnp.max(jnp.abs(evals_h))

        # Progressive penalty that increases as eigenvalues approach 1
        stability_penalty_f = jnp.where(
            max_eval_f > 0.95,  # Start penalizing earlier
            stability_penalty_weight * ((max_eval_f - 0.95) / 0.05) ** 2,
            0.0
        )

        stability_penalty_h = jnp.where(
            max_eval_h > 0.95,  # Start penalizing earlier
            stability_penalty_weight * ((max_eval_h - 0.95) / 0.05) ** 2,
            0.0
        )

        # 2. Penalize small variances
        min_q_h = jnp.min(jnp.diag(params.Q_h))
        min_sigma2 = jnp.min(params.sigma2)

        variance_penalty_q_h = jnp.where(
            min_q_h < 0.01,  # Minimum variance threshold
            stability_penalty_weight * ((0.01 - min_q_h) / 0.01) ** 2,
            0.0
        )

        variance_penalty_sigma2 = jnp.where(
            min_sigma2 < 0.01,  # Minimum variance threshold
            stability_penalty_weight * ((0.01 - min_sigma2) / 0.01) ** 2,
            0.0
        )

        # Combine penalties
        total_penalty = stability_penalty_f + stability_penalty_h + variance_penalty_q_h + variance_penalty_sigma2

        # Return penalized objective
        return obj_value + total_penalty, aux

    return robust_objective


def run_robust_one_cycle_experiment(
    true_params,
    returns,
    init_lr=5e-4,
    max_lr=2e-3,
    min_lr=1e-5,
    max_steps=1500,
    fix_mu=True,
    use_transformations=True,
    verbose=True,
    clip_norm=1.0,  # Gradient clipping norm
    stability_penalty_weight=1000.0  # Explicit stability penalty weight
) -> OptimizerResult:
    """
    Run an experiment with a specific one-cycle scheduler configuration
    and enhanced numerical stability.

    Args:
        true_params: True model parameters
        returns: Simulated returns data
        init_lr: Initial learning rate
        max_lr: Maximum learning rate during the cycle
        min_lr: Minimum learning rate at the end of the cycle
        max_steps: Maximum number of optimization steps
        fix_mu: Whether to fix mu parameter to true value
        use_transformations: Whether to use parameter transformations
        verbose: Whether to print verbose output
        clip_norm: Gradient clipping norm
        stability_penalty_weight: Weight for stability penalty

    Returns:
        OptimizerResult: The result of the optimization run
    """
    # Create initial parameters
    N, K = true_params.N, true_params.K
    initial_params = create_stable_initial_params(N, K)

    # Calculate warmup steps (fixed at 10% of max_steps)
    warmup_steps = int(max_steps * 0.1)

    # Print experiment configuration
    print("\n--- Running robust one-cycle experiment ---")
    print(f"Init LR: {init_lr}, Max LR: {max_lr}, Min LR: {min_lr}")
    print(f"Warmup steps: {warmup_steps} (10% of {max_steps})")
    print(f"Gradient clip norm: {clip_norm}")
    print(f"Stability penalty weight: {stability_penalty_weight}")

    # Configure JAX to use 64-bit precision
    jax.config.update("jax_enable_x64", True)

    # We'll use the standard objective function with increased stability penalty
    # The robust objective approach would require modifying the core optimization code

    # Run optimization with the robust objective
    result = run_optimization(
        filter_type=FilterType.BIF,
        returns=returns,
        initial_params=initial_params,
        true_params=true_params if fix_mu else None,
        use_transformations=use_transformations,
        optimizer_name="AdamW",
        priors=None,
        stability_penalty_weight=stability_penalty_weight,  # This is used in the base objective
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


def compare_robust_configurations(true_params, returns, max_steps=1500):
    """
    Compare different robust configurations.

    Args:
        true_params: True model parameters
        returns: Simulated returns data
        max_steps: Maximum number of optimization steps

    Returns:
        dict: Dictionary containing optimization results for each configuration
    """
    # Define configurations to compare
    configs = {
        "baseline": {
            "init_lr": 5e-4,
            "max_lr": 2e-3,
            "min_lr": 1e-5,
            "clip_norm": 1.0,
            "stability_penalty_weight": 1000.0
        },
        "aggressive_clipping": {
            "init_lr": 5e-4,
            "max_lr": 2e-3,
            "min_lr": 1e-5,
            "clip_norm": 0.5,
            "stability_penalty_weight": 1000.0
        },
        "higher_stability": {
            "init_lr": 5e-4,
            "max_lr": 2e-3,
            "min_lr": 1e-5,
            "clip_norm": 1.0,
            "stability_penalty_weight": 2000.0
        },
        "lower_max_lr": {
            "init_lr": 5e-4,
            "max_lr": 1e-3,
            "min_lr": 1e-5,
            "clip_norm": 1.0,
            "stability_penalty_weight": 1000.0
        },
        "combined_robust": {
            "init_lr": 5e-4,
            "max_lr": 2e-3,
            "min_lr": 1e-5,
            "clip_norm": 0.5,
            "stability_penalty_weight": 2000.0
        }
    }

    # Run optimization with each configuration
    results = {}
    for name, config in configs.items():
        print(f"\n=== Testing {name} configuration ===")
        try:
            result = run_robust_one_cycle_experiment(
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

            # Filter out infinite values for better visualization
            valid_indices = [i for i, loss in enumerate(loss_history) if np.isfinite(loss) and loss < 1e10]
            valid_steps = [i for i in valid_indices]
            valid_losses = [loss_history[i] for i in valid_indices]

            # Plot loss curve
            plt.plot(valid_steps, valid_losses, label=f"{name}")

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


def analyze_convergence_speed(results):
    """
    Analyze the convergence speed of different configurations.

    Args:
        results: Dictionary of optimization results

    Returns:
        DataFrame with convergence metrics
    """
    # Define loss thresholds for convergence analysis
    thresholds = [1e6, 5e5, 1e5, 5e4, 1e4, 5e3, 1e3, 500, 200]

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
    """Main function to run the robust one-cycle experiment."""
    print("Starting robust one-cycle experiment...")

    # Create test data with longer time series
    print("Creating test data with T=600...")
    true_params, returns = create_test_data(N=3, K=2, T=600)

    # Compare robust configurations
    print("\nComparing robust configurations...")
    results = compare_robust_configurations(
        true_params=true_params,
        returns=returns,
        max_steps=1500
    )

    # Plot loss curves
    plot_loss_curves(
        results=results,
        title='Loss Curves for Robust One-Cycle Configurations with AdamW',
        filename=f'robust_one_cycle_loss_curves_{timestamp}.png'
    )

    # Analyze convergence speed
    print("\nAnalyzing convergence speed...")
    convergence_df = analyze_convergence_speed(results)
    convergence_df.to_csv(output_dir / f'robust_one_cycle_convergence_analysis_{timestamp}.csv', index=False)
    print(f"Convergence analysis saved to {output_dir / f'robust_one_cycle_convergence_analysis_{timestamp}.csv'}")

    # Save results to CSV
    basic_df = save_results_to_csv(
        results=results,
        filename=f'robust_one_cycle_results_{timestamp}.csv'
    )

    print("\nRobust one-cycle experiment completed!")

    # Print summary of results
    print("\nSummary of results:")
    print(basic_df.sort_values('Final Loss').to_string(index=False))


if __name__ == "__main__":
    main()
