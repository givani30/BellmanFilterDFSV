"""
Solver utilities for DFSV models.

This module provides standardized utilities for creating and configuring
different optimizers and solvers for DFSV model parameter estimation.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import optimistix as optx
from typing import Dict, Any, Optional, Union, Callable, NamedTuple, Tuple
from dataclasses import dataclass
from jaxtyping import Array, PyTree, Scalar


def create_learning_rate_scheduler(
    init_lr: float = 1e-3,
    decay_steps: int = 1000,
    min_lr: float = 1e-6,
    warmup_steps: int = 0,
    scheduler_type: str = "cosine"
) -> Callable:
    """Create a learning rate scheduler.

    Args:
        init_lr: Initial learning rate.
        decay_steps: Number of steps for learning rate decay.
        min_lr: Minimum learning rate.
        warmup_steps: Number of warmup steps (for schedulers that support warmup).
        scheduler_type: Type of scheduler ('cosine', 'exponential', 'linear', 'warmup_cosine', 'constant').

    Returns:
        A learning rate scheduler function.

    Raises:
        ValueError: If the scheduler type is unknown.
    """
    if scheduler_type == "cosine":
        return optax.cosine_decay_schedule(
            init_value=init_lr,
            decay_steps=decay_steps,
            alpha=min_lr / init_lr
        )
    elif scheduler_type == "exponential":
        return optax.exponential_decay(
            init_value=init_lr,
            transition_steps=decay_steps // 10,
            decay_rate=0.9,
            end_value=min_lr
        )
    elif scheduler_type == "linear":
        return optax.linear_schedule(
            init_value=init_lr,
            end_value=min_lr,
            transition_steps=decay_steps
        )
    elif scheduler_type == "warmup_cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=init_lr,
            peak_value=init_lr*10,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=min_lr
        )
    elif scheduler_type == "constant":
        # Return a constant learning rate scheduler
        return lambda count: jnp.ones_like(count, dtype=jnp.float32) * init_lr
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def create_optimizer(
    optimizer_name: str,
    learning_rate: float = 1e-3,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    max_learning_rate: float = 1e-2,
    min_learning_rate: float = 1e-6,
    decay_steps: int = 1000,
    warmup_steps: int = 100,
    scheduler_type: str = "warmup_cosine",
    verbose: bool = False
) -> Union[optx.AbstractMinimiser, optx.OptaxMinimiser]:
    """Create an optimizer based on name.

    Args:
        optimizer_name: Name of the optimizer to create.
        learning_rate: Initial learning rate for gradient-based optimizers.
        rtol: Relative tolerance for convergence.
        atol: Absolute tolerance for convergence.
        max_learning_rate: Maximum learning rate for schedulers.
        min_learning_rate: Minimum learning rate for schedulers.
        decay_steps: Number of steps for learning rate decay.
        warmup_steps: Number of warmup steps (for schedulers that support warmup).
        scheduler_type: Type of scheduler ('cosine', 'exponential', 'linear', 'warmup_cosine').
        verbose: Whether to enable verbose output from the optimizer.

    Returns:
        An optimizer instance of the specified type.

    Raises:
        ValueError: If the optimizer name is unknown.
    """
    # Configure verbosity
    verbose_set = frozenset({"step_size", "loss"}) if verbose else frozenset()

    # Create learning rate scheduler for gradient-based optimizers
    scheduler = create_learning_rate_scheduler(
        init_lr=learning_rate,
        decay_steps=decay_steps,
        min_lr=min_learning_rate,
        warmup_steps=warmup_steps,
        scheduler_type=scheduler_type
    )

    # Create SGD scheduler with warmup
    sgd_scheduler = create_learning_rate_scheduler(
        init_lr=max_learning_rate,
        decay_steps=decay_steps,
        min_lr=min_learning_rate,
        warmup_steps=warmup_steps,
        scheduler_type="warmup_cosine"
    )

    # Configure optimizer based on name
    if optimizer_name == "BFGS":
        return optx.BFGS(rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    # LBFGS is not available in Optimistix
    # elif optimizer_name == "LBFGS":
    #     return optx.LBFGS(rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "Adam":
        # Use scheduler and apply_if_finite to handle NaN/Inf values
        optimizer = optax.apply_if_finite(optax.adam(learning_rate=scheduler), 10)
        return optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "AdamW":
        # Use scheduler and apply_if_finite to handle NaN/Inf values
        optimizer = optax.apply_if_finite(optax.adamw(learning_rate=scheduler), 10)
        return optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "SGD":
        # Use SGD with momentum and Nesterov acceleration
        optimizer = optax.apply_if_finite(optax.sgd(learning_rate=sgd_scheduler, momentum=0.9, nesterov=True), 10)
        return optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "DogLegBFGS":
        return DogLegBFGS(rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "ArmijoBFGS":
        return ArmijoBFGS(rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "DampedTrustRegionBFGS":
        return DampedTrustRegionBFGS(rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "IndirectTrustRegionBFGS":
        return IndirectTrustRegionBFGS(rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "GradientDescent":
        optimizer = optax.apply_if_finite(optax.sgd(learning_rate=scheduler, momentum=0.0), 10)
        return optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "RMSProp":
        optimizer = optax.apply_if_finite(optax.rmsprop(learning_rate=scheduler), 10)
        return optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "Adagrad":
        optimizer = optax.apply_if_finite(optax.adagrad(learning_rate=scheduler), 10)
        return optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "Adadelta":
        optimizer = optax.apply_if_finite(optax.adadelta(learning_rate=scheduler), 10)
        return optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "Adafactor":
        optimizer = optax.apply_if_finite(optax.adafactor(learning_rate=scheduler), 10)
        return optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "Lion":
        # Lion optimizer (Facebook AI, 2023)
        optimizer = optax.apply_if_finite(optax.lion(learning_rate=scheduler), 10)
        return optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_available_optimizers() -> Dict[str, str]:
    """Get a dictionary of available optimizers with descriptions.

    Returns:
        A dictionary mapping optimizer names to descriptions.
    """
    return {
        "BFGS": "Broyden-Fletcher-Goldfarb-Shanno algorithm (quasi-Newton method)",
        # "LBFGS": "Limited-memory BFGS algorithm (memory-efficient quasi-Newton method)",
        "Adam": "Adaptive Moment Estimation (adaptive learning rates)",
        "AdamW": "Adam with weight decay (better regularization)",
        "SGD": "Stochastic Gradient Descent with momentum and Nesterov acceleration",
        "DogLegBFGS": "BFGS with dogleg trust region strategy",
        "ArmijoBFGS": "BFGS with Armijo line search",
        "DampedTrustRegionBFGS": "BFGS with damped trust region strategy",
        "IndirectTrustRegionBFGS": "BFGS with indirect trust region strategy",
        "GradientDescent": "Basic gradient descent without momentum",
        "RMSProp": "Root Mean Square Propagation (adaptive learning rates)",
        "Adagrad": "Adaptive Gradient Algorithm (per-parameter learning rates)",
        "Adadelta": "Extension of Adagrad with adaptive learning rates",
        "Adafactor": "Memory-efficient version of Adam",
        "Lion": "Evolved Sign Momentum (Facebook AI, 2023)"
    }


def get_optimizer_config(
    optimizer_name: str,
    learning_rate: float = 1e-3,
    rtol: float = 1e-5,
    atol: float = 1e-5
) -> Dict[str, Any]:
    """Get the configuration for a specific optimizer.

    Args:
        optimizer_name: Name of the optimizer.
        learning_rate: Learning rate for gradient-based optimizers.
        rtol: Relative tolerance for convergence.
        atol: Absolute tolerance for convergence.

    Returns:
        A dictionary containing the optimizer configuration.

    Raises:
        ValueError: If the optimizer is not available.
    """
    # Check if optimizer is available
    available_optimizers = get_available_optimizers()
    if optimizer_name not in available_optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Available optimizers: {list(available_optimizers.keys())}")

    # Create configuration dictionary
    config = {
        "learning_rate": learning_rate,
        "rtol": rtol,
        "atol": atol
    }

    # Add optimizer-specific configurations
    if optimizer_name == "BFGS":
        config["use_inverse"] = False
    elif optimizer_name == "Adam":
        config["b1"] = 0.9
        config["b2"] = 0.999
        config["eps"] = 1e-8
    elif optimizer_name == "AdamW":
        config["b1"] = 0.9
        config["b2"] = 0.999
        config["eps"] = 1e-8
        config["weight_decay"] = 1e-4
    elif optimizer_name == "SGD":
        config["momentum"] = 0.9
        config["nesterov"] = True
    elif optimizer_name == "RMSProp":
        config["decay"] = 0.9
        config["eps"] = 1e-8
    elif optimizer_name == "Adagrad":
        config["eps"] = 1e-8
    elif optimizer_name == "Adadelta":
        config["rho"] = 0.9
        config["eps"] = 1e-8

    return config


# Custom optimizer implementations

class DogLegBFGS(optx.AbstractBFGS):
    """DogLeg BFGS solver with specific configurations."""
    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: optx.AbstractDescent = optx.DoglegDescent()
    search: optx.AbstractSearch = optx.ClassicalTrustRegion()
    verbose: frozenset[str]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = optx.max_norm,
        use_inverse: bool = False,
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.use_inverse = use_inverse
        self.descent = optx.DoglegDescent()
        self.search = optx.ClassicalTrustRegion()
        self.verbose = verbose


class ArmijoBFGS(optx.AbstractBFGS):
    """BFGS solver with Backtracking Armijo line search."""
    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: optx.AbstractDescent
    search: optx.AbstractSearch
    verbose: frozenset[str]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = optx.max_norm,
        use_inverse: bool = False,
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.use_inverse = use_inverse
        self.descent = optx.DampedNewtonDescent()
        self.search = optx.BacktrackingArmijo(step_init=0.1)
        self.verbose = verbose


class DampedTrustRegionBFGS(optx.AbstractBFGS):
    """BFGS solver with Damped Newton descent and Classical Trust Region search."""
    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: optx.AbstractDescent = optx.DampedNewtonDescent()
    search: optx.AbstractSearch = optx.ClassicalTrustRegion()
    verbose: frozenset[str]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = optx.max_norm,
        use_inverse: bool = False,
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.use_inverse = use_inverse
        self.descent = optx.DampedNewtonDescent()
        self.search = optx.ClassicalTrustRegion()
        self.verbose = verbose


class IndirectTrustRegionBFGS(optx.AbstractBFGS):
    """BFGS solver with Indirect Damped Newton descent and Classical Trust Region search."""
    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: optx.AbstractDescent = optx.IndirectDampedNewtonDescent()
    search: optx.AbstractSearch = optx.ClassicalTrustRegion()
    verbose: frozenset[str]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = optx.max_norm,
        use_inverse: bool = False,
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.use_inverse = use_inverse
        self.descent = optx.IndirectDampedNewtonDescent()
        self.search = optx.ClassicalTrustRegion()
        self.verbose = verbose