#!/usr/bin/env python
"""
Test script for the minimize_with_logging function.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

from bellman_filter_dfsv.utils.optimization import minimize_with_logging
from bellman_filter_dfsv.utils.solvers import create_optimizer


def quadratic_fn(params, args=None):
    """Simple quadratic function with minimum at [1.0, 2.0]."""
    x, y = params
    return (x - 1.0) ** 2 + (y - 2.0) ** 2, None


def rosenbrock_fn(params, args=None):
    """Rosenbrock function with minimum at [1.0, 1.0]."""
    x, y = params
    return 100.0 * (y - x**2)**2 + (1 - x)**2, None


def test_minimize_with_logging():
    """Test the minimize_with_logging function with different optimizers."""
    # Create output directory
    output_dir = Path("outputs/test_minimize_with_logging")
    os.makedirs(output_dir, exist_ok=True)

    # Test parameters
    initial_params = jnp.array([0.0, 0.0])
    max_steps = 100

    # Test different optimizers
    optimizers = {
        "BFGS": create_optimizer("BFGS"),
        "Adam": create_optimizer("Adam"),
        "AdamW": create_optimizer("AdamW"),
        "DampedTrustRegionBFGS": create_optimizer("DampedTrustRegionBFGS")
    }

    # Test different objective functions
    objective_fns = {
        "Quadratic": quadratic_fn,
        "Rosenbrock": rosenbrock_fn
    }

    # Run tests and collect results
    results = {}

    for obj_name, obj_fn in objective_fns.items():
        results[obj_name] = {}

        for opt_name, optimizer in optimizers.items():
            print(f"Testing {opt_name} on {obj_name} function...")

            # Run optimization with logging
            sol, param_history = minimize_with_logging(
                objective_fn=obj_fn,
                initial_params=initial_params,
                solver=optimizer,
                max_steps=max_steps,
                log_interval=1,
                options={}
            )

            # Store results
            results[obj_name][opt_name] = {
                "solution": sol,
                "param_history": param_history
            }

            # Print results
            print(f"  Result: {sol.result}")
            print(f"  Final value: {sol.value}")
            print(f"  Steps: {sol.stats.get('num_steps', -1)}")
            print(f"  Final loss: {obj_fn(sol.value)[0]}")
            print(f"  Parameter history length: {len(param_history)}")
            print()

            # Plot parameter trajectory
            param_history_np = np.array([p for p in param_history])

            plt.figure(figsize=(10, 6))
            plt.plot(param_history_np[:, 0], param_history_np[:, 1], 'b-', marker='o', alpha=0.5)
            plt.plot(param_history_np[0, 0], param_history_np[0, 1], 'go', markersize=10, label='Start')
            plt.plot(param_history_np[-1, 0], param_history_np[-1, 1], 'ro', markersize=10, label='End')

            if obj_name == "Quadratic":
                plt.plot(1.0, 2.0, 'kx', markersize=10, label='True Minimum')
                # Plot contours
                x = np.linspace(-1, 3, 100)
                y = np.linspace(0, 4, 100)
                X, Y = np.meshgrid(x, y)
                Z = (X - 1.0) ** 2 + (Y - 2.0) ** 2
                plt.contour(X, Y, Z, levels=20, alpha=0.5)
            else:  # Rosenbrock
                plt.plot(1.0, 1.0, 'kx', markersize=10, label='True Minimum')
                # Plot contours
                x = np.linspace(-1, 2, 100)
                y = np.linspace(-1, 2, 100)
                X, Y = np.meshgrid(x, y)
                Z = 100.0 * (Y - X**2)**2 + (1 - X)**2
                plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), alpha=0.5)

            plt.title(f"{opt_name} on {obj_name} Function")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.grid(True)

            # Save plot
            plt.savefig(output_dir / f"{obj_name}_{opt_name}_trajectory.png")
            plt.close()

    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    jax.config.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(42)

    # Run tests
    results = test_minimize_with_logging()

    print("All tests completed successfully!")
