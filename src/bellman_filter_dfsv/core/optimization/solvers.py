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
    scheduler_type: str = "cosine",
    cycle_period: int = 100,  # For cyclic schedulers
    step_size_factor: float = 0.5,  # For step decay
    step_interval: int = 100,  # For step decay
    peak_lr: float = None  # For one-cycle and warmup schedulers
) -> Callable:
    """Create a learning rate scheduler.

    Args:
        init_lr: Initial learning rate.
        decay_steps: Number of steps for learning rate decay.
        min_lr: Minimum learning rate.
        warmup_steps: Number of warmup steps (for schedulers that support warmup).
        scheduler_type: Type of scheduler ('cosine', 'exponential', 'linear', 'warmup_cosine',
                       'constant', 'cyclic', 'step_decay', 'one_cycle').
        cycle_period: Number of steps per cycle for cyclic schedulers.
        step_size_factor: Factor to reduce learning rate at each step for step decay.
        step_interval: Number of steps between learning rate reductions for step decay.
        peak_lr: Peak learning rate for one-cycle and warmup schedulers. Defaults to 10*init_lr if None.

    Returns:
        A learning rate scheduler function.

    Raises:
        ValueError: If the scheduler type is unknown.
    """
    # Set default peak_lr if not provided
    if peak_lr is None:
        peak_lr = init_lr * 10

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
            peak_value=peak_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=min_lr
        )
    elif scheduler_type == "constant":
        # Return a constant learning rate scheduler
        return lambda count: jnp.ones_like(count, dtype=jnp.float32) * init_lr
    elif scheduler_type == "cyclic":
        # Triangular cyclic learning rate scheduler
        def cyclic_schedule(count):
            # Convert to float32 for numerical stability
            count = count.astype(jnp.float32)
            cycle = jnp.floor(1 + count / (2 * cycle_period))
            x = jnp.abs(count / cycle_period - 2 * cycle + 1)
            lr_range = init_lr - min_lr
            return min_lr + lr_range * jnp.maximum(0, (1 - x))
        return cyclic_schedule
    elif scheduler_type == "step_decay":
        # Step decay learning rate scheduler
        def step_decay_schedule(count):
            # Number of steps completed
            steps = count.astype(jnp.float32)
            # Number of decay steps
            decay_steps = jnp.floor(steps / step_interval)
            # Calculate learning rate
            lr = init_lr * (step_size_factor ** decay_steps)
            # Ensure learning rate doesn't go below minimum
            return jnp.maximum(lr, min_lr)
        return step_decay_schedule
    elif scheduler_type == "one_cycle":
        # One-cycle learning rate scheduler (warmup then cooldown)
        # First half: linear warmup from init_lr to peak_lr
        # Second half: cosine annealing from peak_lr to min_lr
        half_cycle = decay_steps // 2
        warmup_schedule = optax.linear_schedule(
            init_value=init_lr,
            end_value=peak_lr,
            transition_steps=half_cycle
        )
        cooldown_schedule = optax.cosine_decay_schedule(
            init_value=peak_lr,
            decay_steps=half_cycle,
            alpha=min_lr / peak_lr
        )

        def one_cycle_schedule(count):
            is_warmup = count < half_cycle
            warmup_lr = warmup_schedule(count)
            cooldown_lr = cooldown_schedule(count - half_cycle)
            return jnp.where(is_warmup, warmup_lr, cooldown_lr)
        return one_cycle_schedule
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
    cycle_period: int = 100,  # For cyclic schedulers
    step_size_factor: float = 0.5,  # For step decay
    step_interval: int = 100,  # For step decay
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
        scheduler_type: Type of scheduler ('cosine', 'exponential', 'linear', 'warmup_cosine',
                       'constant', 'cyclic', 'step_decay', 'one_cycle').
        cycle_period: Number of steps per cycle for cyclic schedulers.
        step_size_factor: Factor to reduce learning rate at each step for step decay.
        step_interval: Number of steps between learning rate reductions for step decay.
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
        scheduler_type=scheduler_type,
        cycle_period=cycle_period,
        step_size_factor=step_size_factor,
        step_interval=step_interval,
        peak_lr=max_learning_rate
    )

    # Create SGD scheduler with warmup - SGD often needs a different schedule
    sgd_scheduler = create_learning_rate_scheduler(
        init_lr=learning_rate,
        decay_steps=decay_steps,
        min_lr=min_learning_rate,
        warmup_steps=warmup_steps,
        scheduler_type="warmup_cosine",
        peak_lr=max_learning_rate
    )

    # Configure optimizer based on name
    if optimizer_name == "BFGS":
        return optx.BFGS(rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)
    if optimizer_name == "NonlinearCG":
        return optx.NonlinearCG(rtol=rtol, atol=atol, norm=optx.rms_norm)

    # LBFGS is not available in Optimistix
    # elif optimizer_name == "LBFGS":
    #     return optx.LBFGS(rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "Adam":
        # Apply gradient clipping and handle NaN/Inf values
        # Create base optimizer with gradient clipping
        base_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=scheduler)
        )
        # Wrap with apply_if_finite to handle NaN/Inf values
        optimizer = optax.apply_if_finite(base_optimizer, 10)
        return optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "AdamW":
        # Apply gradient clipping and handle NaN/Inf values
        # Create base optimizer with gradient clipping
        base_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=scheduler)
        )
        # Wrap with apply_if_finite to handle NaN/Inf values
        optimizer = optax.apply_if_finite(base_optimizer, 10)
        return optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "SGD":
        # Apply gradient clipping with momentum and Nesterov acceleration
        # Create base optimizer with gradient clipping
        base_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.sgd(learning_rate=sgd_scheduler, momentum=0.9, nesterov=True)
        )
        # Wrap with apply_if_finite to handle NaN/Inf values
        optimizer = optax.apply_if_finite(base_optimizer, 10)
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
        # Create base optimizer with gradient clipping
        base_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.sgd(learning_rate=scheduler, momentum=0.0)
        )
        # Wrap with apply_if_finite to handle NaN/Inf values
        optimizer = optax.apply_if_finite(base_optimizer, 10)
        return optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "RMSProp":
        # Create base optimizer with gradient clipping
        base_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.rmsprop(learning_rate=scheduler)
        )
        # Wrap with apply_if_finite to handle NaN/Inf values
        optimizer = optax.apply_if_finite(base_optimizer, 10)
        return optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "Adagrad":
        # Create base optimizer with gradient clipping
        base_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adagrad(learning_rate=scheduler)
        )
        # Wrap with apply_if_finite to handle NaN/Inf values
        optimizer = optax.apply_if_finite(base_optimizer, 10)
        return optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "Adadelta":
        # Create base optimizer with gradient clipping
        base_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adadelta(learning_rate=scheduler)
        )
        # Wrap with apply_if_finite to handle NaN/Inf values
        optimizer = optax.apply_if_finite(base_optimizer, 10)
        return optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "Adafactor":
        # Create base optimizer with gradient clipping
        base_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adafactor(learning_rate=scheduler)
        )
        # Wrap with apply_if_finite to handle NaN/Inf values
        optimizer = optax.apply_if_finite(base_optimizer, 10)
        return optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=verbose_set)

    elif optimizer_name == "Lion":
        # Lion optimizer (Facebook AI, 2023)
        # Create base optimizer with gradient clipping
        base_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.lion(learning_rate=scheduler)
        )
        # Wrap with apply_if_finite to handle NaN/Inf values
        optimizer = optax.apply_if_finite(base_optimizer, 10)
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
        self.search = optx.ClassicalTrustRegion(high_cutoff=0.995, low_cutoff=0.1, high_constant=2.5, low_constant=0.2)
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