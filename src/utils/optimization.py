"""
Optimization utilities for DFSV models.

This module provides standardized utilities for optimizing DFSV model parameters
using various filters and optimizers.
"""

from enum import Enum, auto
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import optimistix as optx
from typing import Dict, Any, Optional, List, Callable, Tuple
from collections import namedtuple
import time

from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.filters.bellman_information import BellmanInformationFilter
from bellman_filter_dfsv.filters.bellman import BellmanFilter
from bellman_filter_dfsv.filters.particle import ParticleFilter
from bellman_filter_dfsv.utils.transformations import transform_params, untransform_params, apply_identification_constraint


# Filter type enumeration
class FilterType(Enum):
    """Enum for filter types."""
    BIF = auto()  # Bellman Information Filter
    BF = auto()   # Bellman Filter
    PF = auto()   # Particle Filter


# Result data structure for optimization runs
OptimizerResult = namedtuple("OptimizerResult", [
    "filter_type",           # Type of filter used
    "optimizer_name",        # Name of the optimizer
    "uses_transformations",  # Whether parameter transformations were used
    "fix_mu",                # Whether mu parameter was fixed
    "prior_config_name",     # Description of prior configuration
    "success",               # Whether optimization succeeded
    "final_loss",            # Final loss value
    "steps",                 # Number of steps taken
    "time_taken",            # Time taken in seconds
    "error_message",         # Error message if any
    "final_params",          # Final estimated parameters
    "param_history",         # History of parameter estimates during optimization
    "loss_history"           # History of loss values during optimization
])


def create_filter(filter_type: FilterType, N: int, K: int, num_particles: int = 5000):
    """Create a filter instance based on filter type.

    Args:
        filter_type: Type of filter to create.
        N: Number of observed variables.
        K: Number of factors.
        num_particles: Number of particles for particle filter.

    Returns:
        A filter instance of the specified type.

    Raises:
        ValueError: If the filter type is unknown.
    """
    if filter_type == FilterType.BIF:
        return BellmanInformationFilter(N=N, K=K)
    elif filter_type == FilterType.BF:
        return BellmanFilter(N=N, K=K)
    elif filter_type == FilterType.PF:
        return ParticleFilter(N=N, K=K, num_particles=num_particles)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def create_optimizer(optimizer_name: str, learning_rate: float = 1e-3,
                    rtol: float = 1e-5, atol: float = 1e-5,
                    max_learning_rate: float = 1e-2,
                    min_learning_rate: float = 1e-6,
                    decay_steps: int = 1000):
    """Create an optimizer based on name.

    Args:
        optimizer_name: Name of the optimizer to create.
        learning_rate: Initial learning rate for gradient-based optimizers.
        rtol: Relative tolerance for convergence.
        atol: Absolute tolerance for convergence.
        max_learning_rate: Maximum learning rate for schedulers.
        min_learning_rate: Minimum learning rate for schedulers.
        decay_steps: Number of steps for learning rate decay.

    Returns:
        An optimizer instance of the specified type.

    Raises:
        ValueError: If the optimizer name is unknown.
    """
    # Create learning rate scheduler for gradient-based optimizers
    scheduler = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=decay_steps,
        alpha=min_learning_rate / learning_rate
    )

    # Create SGD scheduler with warmup
    sgd_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=max_learning_rate,
        warmup_steps=100,
        decay_steps=decay_steps,
        end_value=min_learning_rate
    )

    # Configure optimizer based on name
    if optimizer_name == "BFGS":
        solver = optx.BFGS(rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=frozenset())
    elif optimizer_name == "Adam":
        # Use scheduler and apply_if_finite to handle NaN/Inf values
        optimizer = optax.apply_if_finite(optax.adam(learning_rate=scheduler), 10)
        solver = optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=frozenset())
    elif optimizer_name == "AdamW":
        # Use scheduler and apply_if_finite to handle NaN/Inf values
        optimizer = optax.apply_if_finite(optax.adamw(learning_rate=scheduler), 10)
        solver = optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=frozenset())
    elif optimizer_name == "SGD":
        # Use SGD with momentum and Nesterov acceleration
        optimizer = optax.apply_if_finite(optax.sgd(learning_rate=sgd_scheduler, momentum=0.9, nesterov=True), 10)
        solver = optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=frozenset())
    elif optimizer_name == "DogLegBFGS":
        solver = optx.DogLegBFGS(rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=frozenset())
    elif optimizer_name == "ArmijoBFGS":
        solver = optx.ArmijoBFGS(rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=frozenset())
    elif optimizer_name == "DampedTrustRegionBFGS":
        solver = optx.DampedTrustRegionBFGS(rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=frozenset())
    elif optimizer_name == "IndirectTrustRegionBFGS":
        solver = optx.IndirectTrustRegionBFGS(rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=frozenset())
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return solver


def generate_parameter_history(initial_params, final_params, num_points,
                              untransform_fn=None, fix_mu=False, true_mu=None,
                              apply_constraint_fn=None, objective_fn=None,
                              objective_args=None):
    """Generate parameter history by interpolating between initial and final parameters.

    Args:
        initial_params: Initial parameters.
        final_params: Final parameters.
        num_points: Number of points to generate.
        untransform_fn: Function to untransform parameters.
        fix_mu: Whether to fix mu parameter.
        true_mu: True mu parameter value.
        apply_constraint_fn: Function to apply identification constraint.
        objective_fn: Objective function to calculate loss.
        objective_args: Arguments for objective function.

    Returns:
        Tuple of (param_history, loss_history).
    """
    param_history = []
    loss_history = []

    # Generate a sequence of parameters from initial to final
    alphas = jnp.linspace(0.0, 1.0, num_points)

    for alpha in alphas:
        # Interpolate between initial and final parameters
        current_params = jax.tree_util.tree_map(
            lambda i, f: i + alpha * (f - i),
            initial_params, final_params
        )

        # Apply transformations if needed
        if untransform_fn is not None:
            current_params = untransform_fn(current_params)

        # Fix mu if needed
        if fix_mu and true_mu is not None:
            current_params = eqx.tree_at(lambda p: p.mu, current_params, true_mu)

        # Apply identification constraint if needed
        if apply_constraint_fn is not None:
            current_params = apply_constraint_fn(current_params)

        # Calculate loss if objective function is provided
        if objective_fn is not None and objective_args is not None:
            try:
                current_loss = objective_fn(current_params, *objective_args)
                loss_history.append(float(current_loss))
            except Exception:
                loss_history.append(float('inf'))

        # Add to parameter history
        param_history.append(current_params)

    return param_history, loss_history


def get_objective_function(filter_type: FilterType, filter_instance,
                          stability_penalty_weight: float = 1000.0,
                          priors: Optional[Dict[str, Any]] = None):
    """Get the appropriate objective function for a filter type.

    Args:
        filter_type: Type of filter.
        filter_instance: Filter instance.
        stability_penalty_weight: Weight for stability penalty.
        priors: Dictionary of prior hyperparameters.

    Returns:
        The objective function for the specified filter type.

    Raises:
        ValueError: If the filter type is unknown.
    """
    from bellman_filter_dfsv.filters.objectives import bellman_objective, particle_objective as pf_objective

    if filter_type == FilterType.BIF or filter_type == FilterType.BF:
        def objective_fn(params, observations):
            return bellman_objective(
                params, observations, filter_instance,
                priors=priors, stability_penalty_weight=stability_penalty_weight
            )
    elif filter_type == FilterType.PF:
        def objective_fn(params, observations):
            return pf_objective(
                params, observations, filter_instance,
                priors=priors, stability_penalty_weight=stability_penalty_weight
            )
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return objective_fn
