#!/usr/bin/env python
"""
Unified Filter Optimization Script for DFSV Models.

This script provides a unified interface for optimizing DFSV model parameters
using any of the three implemented filters:
1. Bellman Information Filter (BIF)
2. Bellman Filter (BF)
3. Particle Filter (PF)

It supports parameter transformations, various optimizers, priors, and stability
penalties, allowing for comprehensive comparison of filter performance in
parameter estimation.
"""

import time
import csv
import os
import cloudpickle
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Callable, Tuple, Union
from collections import namedtuple
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Scalar

# JAX imports
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import optax
import equinox as eqx

# Project specific imports
from bellman_filter_dfsv.filters.base import DFSVFilter
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.filters.bellman import DFSVBellmanFilter
from bellman_filter_dfsv.filters.particle import DFSVParticleFilter
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.utils.transformations import (
    transform_params,
    untransform_params,
    apply_identification_constraint
)
from bellman_filter_dfsv.filters.objectives import bellman_objective, pf_objective
from bellman_filter_dfsv.models.simulation import simulate_DFSV
from bellman_filter_dfsv.filters._bellman_optim import CustomBFGS

# Custom BFGS implementations
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

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)


# --- Enums and Data Structures ---

class FilterType(Enum):
    """Enum for the different filter types."""
    BIF = auto()  # Bellman Information Filter
    BF = auto()   # Bellman Filter
    PF = auto()   # Particle Filter


# Result data structure for optimization runs
OptimizerResult = namedtuple("OptimizerResult", [
    "filter_type",           # Type of filter used
    "optimizer_name",        # Name of the optimizer
    "uses_transformations",  # Whether parameter transformations were used
    "fix_mu",               # Whether mu parameter was fixed
    "prior_config_name",    # Description of prior configuration
    "success",              # Whether optimization succeeded
    "final_loss",           # Final loss value
    "steps",                # Number of steps taken
    "time_taken",           # Time taken in seconds
    "error_message",        # Error message if any
    "final_params"          # Final estimated parameters
])


# --- Model and Data Generation Functions ---

def create_simple_model(N: int = 3, K: int = 2) -> DFSVParamsDataclass:
    """Create a simple DFSV model with reasonable parameters.

    Args:
        N: Number of observed series.
        K: Number of factors.

    Returns:
        DFSVParamsDataclass: Model parameters.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Factor loadings (lower triangular with diagonal fixed to 1)
    lambda_r_init = np.random.randn(N, K) * 0.5 + 0.5
    lambda_r = jnp.tril(lambda_r_init)
    diag_indices = jnp.diag_indices(n=min(N, K), ndim=2)
    lambda_r = lambda_r.at[diag_indices].set(1.0)

    # Factor persistence (diagonal-dominant with eigenvalues < 1)
    Phi_f = np.random.uniform(0.01, 0.1, (K, K))  # Off-diagonal elements
    Phi_f[np.diag_indices(K)] = np.random.uniform(0.15, 0.35, K)  # Diagonal elements

    # Log-volatility persistence (diagonal-dominant with eigenvalues < 1)
    Phi_h = np.random.uniform(0.01, 0.1, (K, K))  # Off-diagonal elements
    Phi_h[np.diag_indices(K)] = np.random.uniform(0.9, 0.99, K)  # Diagonal elements

    # Long-run mean for log-volatilities
    mu = np.array([-1.0, -0.5] if K == 2 else [-1.0] * K)

    # Idiosyncratic variance (diagonal)
    sigma2 = np.random.uniform(0.05, 0.1, N)

    # Log-volatility noise covariance (diagonal)
    Q_h = np.diag(np.random.uniform(0.1, 0.3, K))

    # Create parameter object
    params = DFSVParamsDataclass(
        N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, Phi_h=Phi_h,
        mu=mu, sigma2=sigma2, Q_h=Q_h
    )

    # Ensure constraint is applied correctly
    params = apply_identification_constraint(params)
    return params


def create_training_data(params: DFSVParamsDataclass, T: int = 1000, seed: int = 42) -> jnp.ndarray:
    """Generate simulated data for training.

    Args:
        params: Model parameters.
        T: Number of time steps.
        seed: Random seed.

    Returns:
        jnp.ndarray: Simulated returns with shape (T, N).
    """
    returns, _, _ = simulate_DFSV(params, T=T, seed=seed)
    return jnp.asarray(returns)  # Convert to JAX array


# --- Filter Creation and Initialization ---

def create_filter(filter_type: FilterType, N: int, K: int, num_particles: int = 5000) -> DFSVFilter:
    """Create a filter instance based on the filter type.

    Args:
        filter_type: Type of filter to create.
        N: Number of observed series.
        K: Number of factors.
        num_particles: Number of particles for particle filter.

    Returns:
        DFSVFilter: Filter instance.
    """
    if filter_type == FilterType.BIF:
        return DFSVBellmanInformationFilter(N, K)
    elif filter_type == FilterType.BF:
        return DFSVBellmanFilter(N, K)
    elif filter_type == FilterType.PF:
        return DFSVParticleFilter(N, K, num_particles=num_particles)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def create_initial_params(N: int, K: int) -> DFSVParamsDataclass:
    """Create initial parameter values for optimization.

    Args:
        N: Number of observed series.
        K: Number of factors.
        data_variance: Variance of the data (for scaling sigma2).

    Returns:
        DFSVParamsDataclass: Initial parameter values.
    """
    # Create lower triangular lambda_r with diagonal fixed to 1
    lambda_r_init = jnp.zeros((N, K))
    diag_indices = jnp.diag_indices(min(N, K))
    lambda_r_init = lambda_r_init.at[diag_indices].set(1.0)
    lambda_r_init = jnp.tril(lambda_r_init)

    # Different initial values for factor and volatility persistence
    # Lower persistence for factors, higher for volatilities (following working script)
    # Initialize with small non-zero off-diagonal elements to encourage their estimation
    phi_f_init = 0.3 * jnp.eye(K) + 0.05 * jnp.ones((K, K))  # Add small off-diagonal values
    phi_h_init = 0.8 * jnp.eye(K) + 0.02 * jnp.ones((K, K))  # Add small off-diagonal values

    # Create initial parameters
    initial_params = DFSVParamsDataclass(
        N=N, K=K,
        lambda_r=lambda_r_init,
        Phi_f=phi_f_init,
        Phi_h=phi_h_init,
        mu=jnp.zeros(K),
        sigma2=0.1 * jnp.ones(N),
        Q_h=0.2 * jnp.eye(K)
    )

    return initial_params


# --- Optimization Functions ---

def run_optimization(
    filter_type: FilterType,
    returns: jnp.ndarray,
    true_params: Optional[DFSVParamsDataclass] = None,
    use_transformations: bool = True,
    optimizer_name: str = "BFGS",
    priors: Optional[Dict[str, Any]] = None,
    stability_penalty_weight: float = 1000.0,
    max_steps: int = 500,
    num_particles: int = 5000,
    prior_config_name: str = "No Priors"
) -> OptimizerResult:
    """Run optimization for a specific filter type and configuration.

    Args:
        filter_type: Type of filter to use.
        returns: Observed returns with shape (T, N).
        true_params: True parameters (for reference, not used in optimization).
        use_transformations: Whether to use parameter transformations.
        optimizer_name: Name of the optimizer to use.
        priors: Dictionary of prior hyperparameters.
        stability_penalty_weight: Weight for stability penalty.
        max_steps: Maximum number of optimization steps.
        num_particles: Number of particles for particle filter.
        # Note: learning_rate is set internally in the function
        prior_config_name: Description of prior configuration.

    Returns:
        OptimizerResult: Results of the optimization run.
    """
    # Extract dimensions
    _, N = returns.shape
    K = true_params.K if true_params is not None else 2

    # Create filter instance
    filter_instance = create_filter(filter_type, N, K, num_particles)

    # Create initial parameters
    initial_params = create_initial_params(N, K)

    # Apply transformations if requested
    if use_transformations:
        initial_y = transform_params(initial_params)
    else:
        initial_y = initial_params

    # Define optimizer
    rtol, atol = 1e-3, 1e-5

    # Create learning rate scheduler (similar to working script)
    initial_lr = 1e-3
    peak_lr = 1e-2
    end_lr = 1e-6
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=initial_lr,
        peak_value=peak_lr,
        end_value=end_lr,
        warmup_steps=int(max_steps*0.1),
        decay_steps=max_steps
    )

    if optimizer_name == "BFGS":
        solver = optx.BFGS(rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=frozenset())
    elif optimizer_name == "NonlinearCG":
        solver = optx.NonlinearCG(rtol=rtol, atol=atol, norm=optx.rms_norm)
    # NonlinearCG is already defined above
    elif optimizer_name == "Adam":
        # Use scheduler and apply_if_finite to handle NaN/Inf values
        optimizer = optax.apply_if_finite(optax.adam(learning_rate=scheduler), 10)
        solver = optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=frozenset())
    elif optimizer_name == "AdamW":
        # Use scheduler and apply_if_finite to handle NaN/Inf values
        optimizer = optax.apply_if_finite(optax.adamw(learning_rate=scheduler), 10)
        solver = optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=frozenset())
    elif optimizer_name == "SGD":
        # Use a more conservative learning rate schedule for SGD
        sgd_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=1e-4,  # Lower initial learning rate
            peak_value=1e-3,   # Lower peak learning rate
            end_value=1e-7,    # Lower end learning rate
            warmup_steps=int(max_steps*0.1),
            decay_steps=max_steps
        )
        # Use scheduler and apply_if_finite to handle NaN/Inf values
        optimizer = optax.apply_if_finite(optax.sgd(learning_rate=sgd_scheduler, momentum=0.9, nesterov=True), 10)
        solver = optx.OptaxMinimiser(optimizer, rtol=rtol, atol=atol, norm=optx.rms_norm, verbose=frozenset())
    elif optimizer_name == "CustomBFGS":
        solver = CustomBFGS(rtol=1e-5, atol=1e-5, norm=optx.rms_norm, verbose=frozenset())
    elif optimizer_name == "DogLegBFGS":
        solver = DogLegBFGS(rtol=1e-5, atol=1e-5, norm=optx.rms_norm, verbose=frozenset())
    elif optimizer_name == "ArmijoBFGS":
        solver = ArmijoBFGS(rtol=1e-5, atol=1e-5, norm=optx.rms_norm, verbose=frozenset())
    elif optimizer_name == "DampedTrustRegionBFGS":
        solver = DampedTrustRegionBFGS(rtol=1e-5, atol=1e-5, norm=optx.rms_norm, verbose=frozenset())
    elif optimizer_name == "IndirectTrustRegionBFGS":
        solver = IndirectTrustRegionBFGS(rtol=1e-5, atol=1e-5, norm=optx.rms_norm, verbose=frozenset())
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Define objective function based on filter type and transformation
    if filter_type == FilterType.BIF:
        # Bellman Information Filter
        # Only fix mu if true parameters are provided
        fix_mu = true_params is not None
        true_mu = true_params.mu if fix_mu else None
        if fix_mu:
            print(f"  Using mu={true_mu} for optimization (fixed parameter)")
        else:
            print("  Optimizing mu parameter (not fixed)")

        if use_transformations:
            @eqx.filter_jit
            def objective(t_params, args_tuple):
                obs, filt, priors_dict, penalty_weight = args_tuple
                # 1. Untransform parameters
                params_iter = untransform_params(t_params)
                # 2. Fix mu to true value if provided
                if fix_mu:
                    params_iter = eqx.tree_at(lambda p: p.mu, params_iter, true_mu)
                # 3. Apply identification constraint
                params_fixed_constrained = apply_identification_constraint(params_iter)
                # 4. Calculate loss
                return bellman_objective(
                    params_fixed_constrained, obs, filt,
                    priors=priors_dict, stability_penalty_weight=penalty_weight
                )
        else:
            @eqx.filter_jit
            def objective(params, args_tuple):
                obs, filt, priors_dict, penalty_weight = args_tuple
                # 1. Fix mu to true value if provided
                params_iter = params
                if fix_mu:
                    params_iter = eqx.tree_at(lambda p: p.mu, params_iter, true_mu)
                # 2. Apply identification constraint
                params_fixed_constrained = apply_identification_constraint(params_iter)
                # 3. Calculate loss
                return bellman_objective(
                    params_fixed_constrained, obs, filt,
                    priors=priors_dict, stability_penalty_weight=penalty_weight
                )
    elif filter_type == FilterType.BF:
        # Bellman Filter - with extra error handling for shape mismatches
        # Only fix mu if true parameters are provided
        fix_mu = true_params is not None
        true_mu = true_params.mu if fix_mu else None
        if fix_mu:
            print(f"  Using mu={true_mu} for optimization (fixed parameter)")
        else:
            print("  Optimizing mu parameter (not fixed)")

        if use_transformations:
            @eqx.filter_jit
            def objective(t_params, args_tuple):
                obs, filt, priors_dict, penalty_weight = args_tuple
                try:
                    # 1. Untransform parameters
                    params_iter = untransform_params(t_params)
                    # 2. Fix mu to true value if provided
                    if fix_mu:
                        params_iter = eqx.tree_at(lambda p: p.mu, params_iter, true_mu)
                    # 3. Apply identification constraint
                    params_fixed_constrained = apply_identification_constraint(params_iter)
                    # 4. Calculate loss
                    return bellman_objective(
                        params_fixed_constrained, obs, filt,
                        priors=priors_dict, stability_penalty_weight=penalty_weight
                    )
                except Exception as e:
                    print(f"Error in BF objective: {e}")
                    return jnp.inf
        else:
            @eqx.filter_jit
            def objective(params, args_tuple):
                obs, filt, priors_dict, penalty_weight = args_tuple
                try:
                    # 1. Fix mu to true value if provided
                    params_iter = params
                    if fix_mu:
                        params_iter = eqx.tree_at(lambda p: p.mu, params_iter, true_mu)
                    # 2. Apply identification constraint
                    params_fixed_constrained = apply_identification_constraint(params_iter)
                    # 3. Calculate loss
                    return bellman_objective(
                        params_fixed_constrained, obs, filt,
                        priors=priors_dict, stability_penalty_weight=penalty_weight
                    )
                except Exception as e:
                    print(f"Error in BF objective: {e}")
                    return jnp.inf
    else:
        # Particle Filter
        # Only fix mu if true parameters are provided
        fix_mu = true_params is not None
        true_mu = true_params.mu if fix_mu else None
        if fix_mu:
            print(f"  Using mu={true_mu} for optimization (fixed parameter)")
        else:
            print("  Optimizing mu parameter (not fixed)")

        if use_transformations:
            @eqx.filter_jit
            def objective(t_params, args_tuple):
                obs, filt, priors_dict, penalty_weight = args_tuple
                try:
                    # 1. Untransform parameters
                    params_iter = untransform_params(t_params)
                    # 2. Fix mu to true value if provided
                    if fix_mu:
                        params_iter = eqx.tree_at(lambda p: p.mu, params_iter, true_mu)
                    # 3. Apply identification constraint
                    params_fixed_constrained = apply_identification_constraint(params_iter)
                    # 4. Calculate loss
                    return pf_objective(
                        params_fixed_constrained, obs, filt,
                        priors=priors_dict, stability_penalty_weight=penalty_weight
                    )
                except Exception as e:
                    print(f"Error in PF objective: {e}")
                    return jnp.inf
        else:
            @eqx.filter_jit
            def objective(params, args_tuple):
                obs, filt, priors_dict, penalty_weight = args_tuple
                try:
                    # 1. Fix mu to true value if provided
                    params_iter = params
                    if fix_mu:
                        params_iter = eqx.tree_at(lambda p: p.mu, params_iter, true_mu)
                    # 2. Apply identification constraint
                    params_fixed_constrained = apply_identification_constraint(params_iter)
                    # 3. Calculate loss
                    return pf_objective(
                        params_fixed_constrained, obs, filt,
                        priors=priors_dict, stability_penalty_weight=penalty_weight
                    )
                except Exception as e:
                    print(f"Error in PF objective: {e}")
                    return jnp.inf

    # Package static arguments
    static_args = (returns, filter_instance, priors, stability_penalty_weight)

    # Run optimization
    start_time = time.time()
    final_loss = jnp.inf
    num_steps = -1
    success = False
    error_msg = "N/A"
    final_params_untransformed = None

    try:
        # Calculate objective with true parameters if available
        if true_params is not None:
            # Apply transformations if needed
            true_params_y = transform_params(true_params) if use_transformations else true_params
            true_params_loss = objective(true_params_y, static_args)
            print(f"Objective value with true parameters: {true_params_loss:.4f}")

        # Calculate initial objective
        initial_loss = objective(initial_y, static_args)
        print(f"Initial objective value: {initial_loss:.4f}")

        # Run the minimization
        sol = optx.minimise(
            fn=objective,
            solver=solver,
            y0=initial_y,
            args=static_args,
            max_steps=max_steps,
            throw=False
        )

        end_time = time.time()
        time_taken = end_time - start_time
        num_steps = sol.stats.get('num_steps', -1)

        # Check solver success
        success = (sol.result == optx.RESULTS.successful)

        # Get final parameters and loss
        if use_transformations:
            final_params_untransformed = untransform_params(sol.value)
        else:
            final_params_untransformed = sol.value

        # Fix mu in final parameters if it was fixed during optimization
        if fix_mu:
            final_params_untransformed = eqx.tree_at(lambda p: p.mu, final_params_untransformed, true_mu)

        # Apply identification constraint
        final_params_untransformed = apply_identification_constraint(final_params_untransformed)

        # Recalculate loss using the non-transformed objective function
        try:
            if filter_type == FilterType.BIF:
                final_loss = bellman_objective(
                    final_params_untransformed, returns, filter_instance,
                    priors=priors, stability_penalty_weight=stability_penalty_weight
                )
            elif filter_type == FilterType.BF:
                try:
                    final_loss = bellman_objective(
                        final_params_untransformed, returns, filter_instance,
                        priors=priors, stability_penalty_weight=stability_penalty_weight
                    )
                except Exception as bf_e:
                    print(f"Error recalculating BF loss: {bf_e}")
                    final_loss = jnp.inf
            else:
                final_loss = pf_objective(
                    final_params_untransformed, returns, filter_instance,
                    priors=priors, stability_penalty_weight=stability_penalty_weight
                )
        except Exception as e:
            print(f"Error recalculating final loss: {e}")
            final_loss = jnp.inf

        if not jnp.isfinite(final_loss):
            error_msg = "Final loss is non-finite"
            success = False

    except Exception as e:
        end_time = time.time()
        time_taken = end_time - start_time
        success = False
        error_msg = f"Exception: {str(e)}"
        print(f"Error during optimization: {e}")

    # Create result object
    result = OptimizerResult(
        filter_type=filter_type,
        optimizer_name=optimizer_name,
        uses_transformations=use_transformations,
        fix_mu=fix_mu,  # Add fix_mu parameter
        prior_config_name=prior_config_name,
        success=success,
        final_loss=float(final_loss),
        steps=int(num_steps),
        time_taken=time_taken,
        error_message=error_msg,
        final_params=final_params_untransformed
    )

    return result


# --- Results Analysis Functions ---

def print_results_table(results: List[OptimizerResult]):
    """Print a formatted table of optimization results.

    Args:
        results: List of optimization results.
    """
    print("\n\n--- Optimization Results ---")
    # Header
    print(f"{'Filter':<6} | {'Optimizer':<10} | {'Transform':<10} | {'Fix_mu':<7} | {'Prior Config':<20} | {'Success':<8} | {'Final Loss':<15} | {'Steps':<8} | {'Time (s)':<10} | {'Error Message'}")
    print("-" * 140)

    # Rows
    for res in sorted(results, key=lambda x: (x.filter_type.name, x.optimizer_name, x.uses_transformations, x.fix_mu)):
        filter_str = res.filter_type.name
        success_str = "Yes" if res.success else "No"
        fix_mu_str = "Yes" if res.fix_mu else "No"
        loss_str = f"{res.final_loss:.4e}" if jnp.isfinite(res.final_loss) else "Inf/NaN"
        steps_str = str(res.steps) if res.steps >= 0 else "N/A"
        time_str = f"{res.time_taken:.2f}"
        error_str = res.error_message if not res.success else "N/A"
        print(f"{filter_str:<6} | {res.optimizer_name:<10} | {'Yes' if res.uses_transformations else 'No':<10} | {fix_mu_str:<7} | {res.prior_config_name:<20} | {success_str:<8} | {loss_str:<15} | {steps_str:<8} | {time_str:<10} | {error_str}")

    print("-" * 140)


def print_parameter_comparison(results: List[OptimizerResult], true_params: DFSVParamsDataclass):
    """Print a comparison between true and estimated parameters.

    Args:
        results: List of optimization results.
        true_params: True model parameters.
    """
    import dataclasses

    print("\n\n--- Parameter Estimation Comparison ---")

    # Get parameter names from the dataclass, excluding N and K
    param_names = [f.name for f in dataclasses.fields(DFSVParamsDataclass) if f.name not in ['N', 'K']]

    # Set numpy print options for better readability
    np.set_printoptions(precision=4, suppress=True)

    for res in sorted(results, key=lambda x: (x.filter_type.name, x.optimizer_name, x.uses_transformations, x.fix_mu)):
        if res.final_params is not None:
            print(f"\n-- Run: Filter='{res.filter_type.name}' | Optimizer='{res.optimizer_name}' | Fix_mu='{'Yes' if res.fix_mu else 'No'}' | Success='{'Yes' if res.success else 'No'}' --")
            print("-" * 80)
            print(f"{'Parameter':<10} | {'True Value':<35} | {'Estimated Value'}")
            print("-" * 80)

            for name in param_names:
                true_val = getattr(true_params, name)
                est_val = getattr(res.final_params, name)

                # Convert to numpy for consistent printing
                true_val_np = np.asarray(true_val)
                est_val_np = np.asarray(est_val)

                # Format for printing (handle multi-line arrays)
                true_str_lines = str(true_val_np).split('\n')
                est_str_lines = str(est_val_np).split('\n')

                # Print first line with parameter name
                print(f"{name:<10} | {true_str_lines[0]:<35} | {est_str_lines[0]}")

                # Print subsequent lines aligned
                max_lines = max(len(true_str_lines), len(est_str_lines))
                for i in range(1, max_lines):
                    true_line = true_str_lines[i] if i < len(true_str_lines) else ""
                    est_line = est_str_lines[i] if i < len(est_str_lines) else ""
                    print(f"{'':<10} | {true_line:<35} | {est_line}")

            print("-" * 80)
        else:
            print(f"\n-- Run: Filter='{res.filter_type.name}' | Optimizer='{res.optimizer_name}' --")
            print("  No final parameters available for comparison (likely failed early).")


def save_results_to_csv(results: List[OptimizerResult], filename: str = "filter_optimization_results.csv"):
    """Save optimization results to a CSV file.

    Args:
        results: List of optimization results.
        filename: Name of the CSV file to save.
    """
    if not results:
        print("No results to save.")
        return

    # Get headers from the namedtuple fields, excluding final_params
    headers = [field for field in OptimizerResult._fields if field != 'final_params']

    try:
        # Ensure outputs directory exists
        os.makedirs("outputs", exist_ok=True)
        filepath = os.path.join("outputs", filename)

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(headers)
            # Write data rows
            for result in results:
                # Prepare row data, excluding final_params
                row_data = {field: getattr(result, field) for field in headers}
                # Convert filter_type enum to string
                if 'filter_type' in row_data:
                    row_data['filter_type'] = row_data['filter_type'].name
                # Convert JAX arrays/scalars if necessary
                row = [float(item) if isinstance(item, (jnp.ndarray, jnp.generic)) else item for item in row_data.values()]
                writer.writerow(row)
        print(f"Results successfully saved to {filepath}")
    except IOError as e:
        print(f"Error saving results to CSV: {e}")


def save_estimated_params(results: List[OptimizerResult], true_params: DFSVParamsDataclass):
    """Save estimated parameters to pickle files.

    Args:
        results: List of optimization results.
        true_params: True model parameters.
    """
    print("\nSaving estimated parameters...")

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Save true parameters for reference
    true_params_path = os.path.join("outputs", "true_params.pkl")
    with open(true_params_path, 'wb') as f:
        cloudpickle.dump(true_params, f)
    print(f"  Saved true parameters to {true_params_path}")

    # Save estimated parameters for each successful run
    for res in results:
        if res.final_params is not None:
            status_str = 'Success' if res.success else 'Failure'
            filter_str = res.filter_type.name
            opt_name = res.optimizer_name.replace(' ', '_')
            transform_str = 'Transformed' if res.uses_transformations else 'Untransformed'
            fix_mu_str = 'FixedMu' if res.fix_mu else 'FreeMu'

            filename = f"estimated_params_{filter_str}_{opt_name}_{transform_str}_{fix_mu_str}_{status_str}.pkl"
            filepath = os.path.join("outputs", filename)

            try:
                with open(filepath, 'wb') as f:
                    cloudpickle.dump(res.final_params, f)
                print(f"  Saved parameters to {filepath}")
            except Exception as e:
                print(f"  ERROR saving parameters to {filepath}: {e}")
        else:
            print(f"  Skipping parameter saving for {res.filter_type.name}/{res.optimizer_name} (final_params is None)")


# --- Main Function ---

def main():
    """Main function to run the unified filter optimization."""
    print("Starting Unified Filter Optimization...")

    # 1. Create model parameters
    N, K = 3, 2
    true_params = create_simple_model(N=N, K=K)
    print(f"Created model with N={true_params.N}, K={true_params.K}")
    print("True Parameters:")
    print(true_params)

    # 2. Generate simulation data
    T = 1500  # Full experiment with longer time series
    print(f"\nGenerating {T} time steps of simulation data...")
    returns = create_training_data(true_params, T=T, seed=123)
    print("Simulation data generated.")

    # 3. Define optimization configurations
    # Run all three filters
    filter_types = [FilterType.BIF, FilterType.PF]

    # Use both AdamW and DampedTrustRegionBFGS optimizers
    optimizers = ["AdamW", "DampedTrustRegionBFGS"]
    use_transformations_options = [True]  # Always use transformations for stability
    fix_mu_options = [True, False]  # Run with both fixed and non-fixed mu
    max_steps = 200  # Increased for better convergence
    stability_penalty_weight = 1000.0
    num_particles = 5000

    # 4. Run optimizations
    results = []

    print(f"\nRunning optimizations with max_steps={max_steps}, stability_penalty_weight={stability_penalty_weight}...")

    for filter_type in filter_types:
        for optimizer_name in optimizers:
            for use_transformations in use_transformations_options:
                for fix_mu in fix_mu_options:
                    print(f"\n--- Running: Filter={filter_type.name} | Optimizer={optimizer_name} | Transform={'Yes' if use_transformations else 'No'} | Fix_mu={'Yes' if fix_mu else 'No'} ---")

                    # Skip certain filter-optimizer combinations that are known to be problematic
                    # if (filter_type == FilterType.PF and optimizer_name == "BFGS") or \
                    #    (filter_type == FilterType.BF and optimizer_name == "AdamW"):
                    #     print(f"Skipping {filter_type.name} with {optimizer_name} (known compatibility issue)")
                    #     continue

                    try:
                        # Run optimization with error handling
                        result = run_optimization(
                            filter_type=filter_type,
                            returns=returns,
                            true_params=true_params if fix_mu else None,  # Only pass true_params if fix_mu is True
                            use_transformations=use_transformations,
                            optimizer_name=optimizer_name,
                            priors=None,
                            stability_penalty_weight=stability_penalty_weight,
                            max_steps=max_steps,
                            num_particles=num_particles,
                            prior_config_name="No Priors"
                        )

                        # Add result to list
                        results.append(result)
                    except Exception as e:
                        print(f"Error running optimization: {e}")
                        # Create a failure result
                        error_result = OptimizerResult(
                            filter_type=filter_type,
                            optimizer_name=optimizer_name,
                            uses_transformations=use_transformations,
                            fix_mu=fix_mu,  # Add fix_mu parameter
                            prior_config_name="No Priors",
                            success=False,
                            final_loss=float('inf'),
                            steps=-1,
                            time_taken=0.0,
                            error_message=f"Exception: {str(e)}",
                            final_params=None
                        )
                        results.append(error_result)

    # 5. Print results table
    if results:
        print_results_table(results)

        # 6. Print parameter comparison
        print_parameter_comparison(results, true_params)

        # 7. Save results
        save_results_to_csv(results)
        save_estimated_params(results, true_params)
    else:
        print("\nNo results to display. All optimizations failed.")

    print("\nUnified Filter Optimization completed.")


if __name__ == "__main__":
    main()

