"""
Example script demonstrating advanced stability techniques for optimization.

This script implements several advanced techniques from the optax library to
improve optimization stability, particularly for the BIF filter which can
encounter numerical issues during optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional, Callable

from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.utils.optimization import (
    FilterType,
    run_optimization,
    OptimizerResult,
    get_objective_function,
    create_filter
)
from bellman_filter_dfsv.utils.optimization_helpers import create_stable_initial_params
from bellman_filter_dfsv.utils.transformations import (
    apply_identification_constraint,
    transform_params,
    untransform_params
)

# Create output directory
output_dir = Path("outputs/advanced_stability")
output_dir.mkdir(parents=True, exist_ok=True)

# Timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def create_test_data(N=3, K=2, T=200):
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
            [1.0, 0.0],  # First factor loading fixed to 1.0 (identification constraint)
            [0.8, 0.5],  # Both factors load on second series
            [0.6, 0.7]   # Both factors load on third series
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


def create_error_handling_objective(base_objective_fn: Callable):
    """
    Create an objective function wrapper that handles errors gracefully.

    Args:
        base_objective_fn: The base objective function to wrap

    Returns:
        A wrapped objective function with error handling
    """
    def robust_objective(params, *args):
        # Get base objective value
        try:
            obj_value, aux = base_objective_fn(params, *args)

            # Check for non-finite values
            if not jnp.isfinite(obj_value):
                return jnp.array(1e10, dtype=jnp.float64), aux

            return obj_value, aux

        except Exception as e:
            # If base objective fails, return a high value
            jax.debug.print(f"Error in base objective: {e}")
            return jnp.array(1e10, dtype=jnp.float64), ()

    return robust_objective


def custom_optimization_with_recovery(
    objective_fn: Callable,
    initial_params: DFSVParamsDataclass,
    returns: jnp.ndarray,
    max_steps: int = 1500,
    learning_rate: float = 5e-4,
    max_learning_rate: float = 2e-3,
    min_learning_rate: float = 1e-5,
    warmup_steps: int = 150,
    clip_norm: float = 0.5,
    use_transformations: bool = True,
    verbose: bool = True
) -> Tuple[DFSVParamsDataclass, List[float], List[DFSVParamsDataclass]]:
    """
    Custom optimization loop with recovery from instability.

    This implementation includes several advanced stability techniques:
    1. Zero NaNs transformation to replace NaNs with zeros
    2. Aggressive gradient clipping to prevent large parameter updates
    3. Best parameter tracking with recovery from instability
    4. One-cycle learning rate schedule with cosine annealing
    5. Comprehensive error handling throughout the optimization process

    Args:
        objective_fn: The objective function to minimize
        initial_params: Initial parameters
        returns: Return data
        max_steps: Maximum number of optimization steps
        learning_rate: Initial learning rate
        max_learning_rate: Maximum learning rate for one-cycle
        min_learning_rate: Minimum learning rate for one-cycle
        warmup_steps: Number of warmup steps
        clip_norm: Gradient clipping norm
        use_transformations: Whether to use parameter transformations
        verbose: Whether to print verbose output

    Returns:
        Tuple of (final_params, loss_history, param_history)
    """
    start_time = time.time()

    # Transform initial parameters if needed
    params = transform_params(initial_params) if use_transformations else initial_params

    # Create one-cycle learning rate schedule
    total_steps = max_steps
    pct_start = warmup_steps / total_steps

    schedule_fn = optax.cosine_onecycle_schedule(
        transition_steps=total_steps,
        peak_value=max_learning_rate,
        pct_start=pct_start,
        div_factor=max_learning_rate/learning_rate,
        final_div_factor=learning_rate/min_learning_rate
    )

    # Create optimizer with advanced stability techniques
    optimizer = optax.chain(
        optax.zero_nans(),  # Replace NaNs with zeros
        optax.clip_by_global_norm(clip_norm),  # Clip gradients
        optax.scale_by_adam(eps=1e-5),  # Use Adam with larger epsilon for stability
        optax.scale_by_schedule(schedule_fn),  # Apply learning rate schedule
        optax.scale(-1.0)  # Convert maximization to minimization
    )

    # Initialize optimizer state
    opt_state = optimizer.init(params)

    # Initialize tracking variables
    best_params = params
    best_loss = float('inf')
    loss_history = []
    param_history = []
    plateau_counter = 0
    plateau_threshold = 50  # Number of steps with minimal improvement before reducing LR
    min_improvement = 1e-4  # Minimum improvement to reset plateau counter
    recovery_count = 0
    max_recoveries = 5  # Maximum number of recovery attempts before reducing learning rate

    # Run optimization loop
    for step in range(max_steps):
        # Compute loss and gradients
        try:
            loss_value, grads = jax.value_and_grad(lambda p: objective_fn(p, returns)[0])(params)

            # Check for NaN or Inf in loss or gradients
            has_nan_loss = not jnp.isfinite(loss_value)

            # Check gradients - need to handle pytrees properly
            grad_leaves = jax.tree_util.tree_leaves(grads)
            has_nan_grads = False
            for leaf in grad_leaves:
                if hasattr(leaf, 'shape'):  # Check if it's an array-like object
                    if not jnp.all(jnp.isfinite(leaf)):
                        has_nan_grads = True
                        break

            if has_nan_loss or has_nan_grads:
                recovery_count += 1
                if verbose:
                    print(f"Step {step}: Non-finite values detected (recovery #{recovery_count}), using best parameters")
                    if has_nan_loss:
                        print(f"  - Non-finite loss value: {loss_value}")
                    if has_nan_grads:
                        print("  - Non-finite gradients detected")

                # If we've had too many recoveries, reduce learning rate
                if recovery_count >= max_recoveries:
                    if verbose:
                        print(f"Too many recoveries ({recovery_count}), reducing learning rate")

                    # Create a new optimizer with reduced learning rate
                    new_max_lr = max_learning_rate * 0.5
                    new_schedule_fn = optax.cosine_onecycle_schedule(
                        transition_steps=total_steps - step,
                        peak_value=new_max_lr,
                        pct_start=0.3,  # Shorter warmup for the remainder
                        div_factor=new_max_lr/min_learning_rate,
                        final_div_factor=1.0
                    )

                    optimizer = optax.chain(
                        optax.zero_nans(),
                        optax.clip_by_global_norm(clip_norm * 0.5),  # More aggressive clipping
                        optax.scale_by_adam(eps=1e-4),  # Larger epsilon
                        optax.scale_by_schedule(new_schedule_fn),
                        optax.scale(-1.0)
                    )

                    opt_state = optimizer.init(best_params)
                    max_learning_rate = new_max_lr
                    recovery_count = 0

                params = best_params
                # Add current loss to history (using best loss)
                loss_history.append(float(best_loss))
                param_history.append(best_params)
                continue

            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            # Track loss
            loss_history.append(float(loss_value))
            param_history.append(params)

            # Update best parameters if loss improved
            if loss_value < best_loss:
                improvement = best_loss - loss_value
                best_loss = loss_value
                best_params = params

                # Reset plateau counter if significant improvement
                if improvement > min_improvement:
                    plateau_counter = 0
                else:
                    plateau_counter += 1
            else:
                plateau_counter += 1

            # Check for plateau
            if plateau_counter >= plateau_threshold:
                if verbose:
                    print(f"Step {step}: Optimization plateaued, reducing learning rate")

                # Create a new optimizer with reduced learning rate
                new_max_lr = max_learning_rate * 0.5
                new_schedule_fn = optax.cosine_onecycle_schedule(
                    transition_steps=total_steps - step,
                    peak_value=new_max_lr,
                    pct_start=0.3,  # Shorter warmup for the remainder
                    div_factor=new_max_lr/min_learning_rate,
                    final_div_factor=1.0
                )

                optimizer = optax.chain(
                    optax.zero_nans(),
                    optax.clip_by_global_norm(clip_norm),
                    optax.scale_by_adam(eps=1e-4),
                    optax.scale_by_schedule(new_schedule_fn),
                    optax.scale(-1.0)
                )

                opt_state = optimizer.init(params)
                max_learning_rate = new_max_lr
                plateau_counter = 0

            # Print progress
            if verbose and (step % 50 == 0 or step == max_steps - 1):
                print(f"Step {step}: Loss = {loss_value:.6f}, Best loss = {best_loss:.6f}")

        except Exception as e:
            if verbose:
                print(f"Error at step {step}: {e}")
                print("Continuing with best parameters so far")
            params = best_params
            # Add current loss to history (using best loss)
            loss_history.append(float(best_loss))
            param_history.append(best_params)

    # Use best parameters found during optimization
    final_params = best_params

    # Untransform parameters if needed
    if use_transformations:
        try:
            final_params = untransform_params(final_params)
            param_history = [untransform_params(p) for p in param_history]
        except Exception as e:
            if verbose:
                print(f"Error untransforming parameters: {e}")

    # Apply identification constraint
    try:
        final_params = apply_identification_constraint(final_params)
        param_history = [apply_identification_constraint(p) for p in param_history]
    except Exception as e:
        if verbose:
            print(f"Error applying identification constraint: {e}")

    if verbose:
        time_taken = time.time() - start_time
        print(f"Optimization completed in {time_taken:.2f} seconds")
        print(f"Final loss: {best_loss:.6f}")

    return final_params, loss_history, param_history


def run_advanced_stability_experiment(
    true_params,
    returns,
    max_steps=300,
    fix_mu=True,
    use_transformations=True,
    verbose=True
) -> Dict[str, Any]:
    """
    Run an experiment with advanced stability techniques.

    This experiment compares two approaches:
    1. Custom optimization with advanced stability techniques:
       - Error handling objective function
       - Zero NaNs transformation
       - Aggressive gradient clipping
       - Adaptive learning rate reduction
       - Recovery from instability
       - Plateau detection

    2. Standard optimization using run_optimization

    Args:
        true_params: True model parameters
        returns: Simulated returns data
        max_steps: Maximum number of optimization steps
        fix_mu: Whether to fix mu parameter to true value
        use_transformations: Whether to use parameter transformations
        verbose: Whether to print verbose output

    Returns:
        Dictionary containing experiment results
    """
    # Create initial parameters
    N, K = true_params.N, true_params.K
    initial_params = create_stable_initial_params(N, K)

    # Calculate warmup steps (fixed at 10% of max_steps)
    warmup_steps = int(max_steps * 0.1)

    # Print experiment configuration
    print("\n--- Running advanced stability experiment ---")
    print(f"Warmup steps: {warmup_steps} (10% of {max_steps})")
    print(f"Using parameter transformations: {use_transformations}")
    print(f"Fixing mu: {fix_mu}")
    print(f"Dataset size: {returns.shape}")

    # Configure JAX to use 64-bit precision
    jax.config.update("jax_enable_x64", True)

    # Create filter instance
    N, K = true_params.N, true_params.K
    filter_instance = create_filter(FilterType.BIF, N, K)

    # Get the base objective function
    base_objective_fn = get_objective_function(
        filter_type=FilterType.BIF,
        filter_instance=filter_instance,
        priors=None,
        stability_penalty_weight=1000.0,
        fix_mu=fix_mu,
        true_mu=true_params.mu if fix_mu else None,
        is_transformed=use_transformations
    )

    # Create a more robust objective function with error handling
    robust_objective_fn = create_error_handling_objective(base_objective_fn)

    # Run custom optimization with recovery
    print("\n--- Running custom optimization with recovery ---")
    final_params, loss_history, param_history = custom_optimization_with_recovery(
        objective_fn=robust_objective_fn,
        initial_params=initial_params,
        returns=returns,
        max_steps=max_steps,
        learning_rate=5e-4,
        max_learning_rate=1e-3,  # More conservative max learning rate
        min_learning_rate=1e-5,
        warmup_steps=warmup_steps,
        clip_norm=0.1,  # More aggressive gradient clipping
        use_transformations=use_transformations,
        verbose=verbose
    )

    # For comparison, also run standard optimization
    print("\n--- Running standard optimization ---")
    standard_result = run_optimization(
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
        log_params=True,
        log_interval=1,
        rtol=1e-5,
        atol=1e-5,
        fix_mu=fix_mu,
        scheduler_type="one_cycle",
        learning_rate=5e-4,
        max_learning_rate=2e-3,
        min_learning_rate=1e-5,
        warmup_steps=warmup_steps
    )

    # Create result dictionary
    result = {
        "custom_final_params": final_params,
        "custom_loss_history": loss_history,
        "custom_param_history": param_history,
        "standard_result": standard_result
    }

    return result


def plot_loss_comparison(custom_loss_history, standard_loss_history, title, filename):
    """
    Plot loss curves comparing custom and standard optimization.

    Args:
        custom_loss_history: Loss history from custom optimization
        standard_loss_history: Loss history from standard optimization
        title: Plot title
        filename: Output filename
    """
    plt.figure(figsize=(12, 8))

    # Filter out infinite values for better visualization
    valid_custom_indices = [i for i, loss in enumerate(custom_loss_history) if np.isfinite(loss) and loss < 1e10]
    valid_custom_steps = [i for i in valid_custom_indices]
    valid_custom_losses = [custom_loss_history[i] for i in valid_custom_indices]

    valid_standard_indices = [i for i, loss in enumerate(standard_loss_history) if np.isfinite(loss) and loss < 1e10]
    valid_standard_steps = [i for i in valid_standard_indices]
    valid_standard_losses = [standard_loss_history[i] for i in valid_standard_indices]

    # Plot loss curves
    plt.plot(valid_custom_steps, valid_custom_losses, label="Custom Optimization with Recovery")
    plt.plot(valid_standard_steps, valid_standard_losses, label="Standard Optimization")

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


def main():
    """Main function to run the advanced stability experiment."""
    print("Starting advanced stability experiment...")

    # Create test data with longer time series
    print("Creating test data with T=200...")
    true_params, returns = create_test_data(N=3, K=2, T=200)

    # Run experiment
    print("\nRunning experiment...")
    result = run_advanced_stability_experiment(
        true_params=true_params,
        returns=returns,
        max_steps=1500,
        fix_mu=True,
        use_transformations=True,
        verbose=True
    )

    # Plot loss comparison
    plot_loss_comparison(
        custom_loss_history=result["custom_loss_history"],
        standard_loss_history=result["standard_result"].loss_history,
        title='Loss Curves: Custom vs Standard Optimization',
        filename=f'advanced_stability_loss_comparison_{timestamp}.png'
    )

    # Save final parameters
    custom_final_loss = result["custom_loss_history"][-1] if result["custom_loss_history"] else float('inf')
    standard_final_loss = result["standard_result"].final_loss if result["standard_result"] else float('inf')

    print("\nAdvanced stability experiment completed!")
    print("\nSummary of results:")
    print(f"Custom optimization final loss: {custom_final_loss:.6f}")
    print(f"Standard optimization final loss: {standard_final_loss:.6f}")


if __name__ == "__main__":
    import time
    main()
