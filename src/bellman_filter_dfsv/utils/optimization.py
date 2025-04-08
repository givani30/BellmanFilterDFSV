"""
Optimization utilities for DFSV models.

This module provides standardized utilities for optimizing DFSV model parameters
using various filters and optimizers.
"""

from enum import Enum, auto
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from typing import Dict, Any, Optional, List, Callable, Tuple, Union, Protocol
from collections import namedtuple
import time
from functools import partial

from bellman_filter_dfsv.utils.solvers import create_optimizer

from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.filters.bellman import DFSVBellmanFilter
from bellman_filter_dfsv.filters.particle import DFSVParticleFilter
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
        return DFSVBellmanInformationFilter(N=N, K=K)
    elif filter_type == FilterType.BF:
        return DFSVBellmanFilter(N=N, K=K)
    elif filter_type == FilterType.PF:
        return DFSVParticleFilter(N=N, K=K, num_particles=num_particles)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")




#TODO: Need to save all parameters instead of interpolation, this is wrong.
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
                          priors: Optional[Dict[str, Any]] = None,
                          is_transformed: bool = False):
    """Get the appropriate objective function for a filter type.

    This function returns the appropriate objective function based on the filter type
    and whether the parameters are in transformed space. It uses the existing objective
    functions from bellman_filter_dfsv.filters.objectives.

    Args:
        filter_type: Type of filter.
        filter_instance: Filter instance.
        stability_penalty_weight: Weight for stability penalty.
        priors: Dictionary of prior hyperparameters.
        is_transformed: Whether the parameters are in transformed space.

    Returns:
        The objective function for the specified filter type.

    Raises:
        ValueError: If the filter type is unknown.
    """
    from bellman_filter_dfsv.filters.objectives import (
        bellman_objective, transformed_bellman_objective,
        pf_objective, transformed_pf_objective
    )

    # Map filter types to their objective functions (both transformed and untransformed)
    objective_map = {
        FilterType.BIF: (bellman_objective, transformed_bellman_objective),
        FilterType.BF: (bellman_objective, transformed_bellman_objective),
        FilterType.PF: (pf_objective, transformed_pf_objective)
    }

    # Check if filter type is supported
    if filter_type not in objective_map:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Get the appropriate objective function based on whether parameters are transformed
    untransformed_obj, transformed_obj = objective_map[filter_type]
    selected_obj = transformed_obj if is_transformed else untransformed_obj

    # Create a closure with the fixed arguments
    def objective_fn(params, observations):
        loss = selected_obj(
            params, observations, filter_instance,
            priors=priors, stability_penalty_weight=stability_penalty_weight
        )
        return loss, None

    return objective_fn


def minimize_with_logging(objective_fn: Callable, initial_params: Any, solver: optx.AbstractMinimiser,
                         static_args: Any = None, max_steps: int = 100,
                         log_interval: int = 1, throw: bool = False,
                         options: Dict[str, Any] = None) -> Tuple[optx.Solution, List]:
    """Minimize an objective function with parameter logging.

    This function iteratively steps through a solver and returns both the optimization
    results and parameter history. It's useful for tracking parameter evolution during
    optimization.

    Args:
        objective_fn: The objective function to minimize.
        initial_params: Initial parameter values.
        solver: An optimistix solver instance.
        static_args: Static arguments to pass to the objective function.
        max_steps: Maximum number of optimization steps.
        log_interval: Interval at which to log parameters (1 = every step).
        throw: Whether to throw an exception if optimization fails.

    Returns:
        A tuple containing the optimization solution and parameter history.
    """
    # Initialize parameter history
    param_history = [initial_params]
    y=initial_params

    # Define a wrapper function that logs parameters
    def logging_step(param_history,params):
        # Save current parameters
        param_history.append(params)
        return param_history

    # Initialize solver state
    # Prepare options
    if options is None:
        options = {}

    # Get the shape and dtype of the output
    test_output, test_aux = objective_fn(initial_params, static_args)
    f_struct = jax.ShapeDtypeStruct(test_output.shape, test_output.dtype)
    aux_struct = None if test_aux is None else jax.ShapeDtypeStruct(test_aux.shape, test_aux.dtype)

    # Initialize solver state
    state = solver.init(objective_fn, initial_params, static_args, options, f_struct, aux_struct, frozenset())
    # Run optimization with logging
    for _ in range(max_steps):
        # Perform one step of optimization
        try:
            y, state, _ = solver.step(objective_fn, y, static_args, options, state, frozenset())
            # Log parameters
            param_history = logging_step(param_history, y)

            # Check for convergence
            converged, result = solver.terminate(objective_fn, y, static_args, options, state, frozenset())
            if converged:
                break
        except Exception as e:
            if throw:
                raise e
            else:
                # Create a failure result
                result = optx.RESULTS.failed
                break

    # Create solution object
    if 'result' not in locals():
        # If we reached max_steps without convergence or failure
        result = optx.RESULTS.max_steps_reached

    # Perform postprocessing
    try:
        # Get the auxiliary output (typically None for minimization)
        aux = None
        # Postprocess the result
        final_y, final_aux, stats = solver.postprocess(
            objective_fn, y, aux, static_args, options, state, frozenset(), result
        )
        # Create the solution object
        sol = optx.Solution(
            value=final_y,
            result=result,
            stats=stats,
            aux=final_aux,
            state=state
        )
    except Exception as e:
        if throw:
            raise e
        else:
            # Create a failure solution
            sol = optx.Solution(
                value=y,
                result=optx.RESULTS.nonlinear_divergence,
                stats={"error": str(e)},
                aux=None,
                state=state
            )

    return sol, param_history


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
    prior_config_name: str = "No Priors",
    log_params: bool = True,
    log_interval: int = 1,
    learning_rate: float = 1e-3,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    verbose: bool = False
) -> OptimizerResult:
    """Run optimization for a specific filter type and configuration.

    This function integrates all the components needed for optimizing DFSV model parameters
    using various filters and optimizers. It handles parameter transformations, objective
    function selection, optimization, and result processing.

    Args:
        filter_type: Type of filter to use (BIF, BF, or PF).
        returns: Observed returns data.
        true_params: True parameters (optional, used for fixing mu if needed).
        use_transformations: Whether to use parameter transformations.
        optimizer_name: Name of the optimizer to use.
        priors: Dictionary of prior hyperparameters.
        stability_penalty_weight: Weight for stability penalty in objective function.
        max_steps: Maximum number of optimization steps.
        num_particles: Number of particles for particle filter.
        prior_config_name: Description of prior configuration (for reporting).
        log_params: Whether to log parameters during optimization.
        log_interval: Interval at which to log parameters.
        learning_rate: Initial learning rate for gradient-based optimizers.
        rtol: Relative tolerance for convergence.
        atol: Absolute tolerance for convergence.
        verbose: Whether to enable verbose output from the optimizer.

    Returns:
        OptimizerResult: A namedtuple containing optimization results and metadata.
    """
    # Start timing
    start_time = time.time()

    # Create filter instance
    N, K = returns.shape[1], 1  # Assume K=1 if not provided in true_params
    if true_params is not None:
        N, K = true_params.N, true_params.K

    filter_instance = create_filter(filter_type, N, K, num_particles)

    # Create initial parameters - always use uninformed initial parameters
    # Use a small fixed value for sigma2 to ensure stability
    initial_params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=0.5 * jnp.ones((N, K)),  # Moderate positive loadings
        Phi_f=0.5 * jnp.eye(K)+0.05*jnp.ones((K,K)),  # Moderate persistence
        Phi_h=0.5 * jnp.eye(K)+0.05*jnp.ones((K,K)),  # Moderate persistence
        mu=jnp.zeros(K),  # Zero mean for log volatility
        sigma2=0.05 * jnp.ones(N),  # Small fixed value for stability
        Q_h=0.2 * jnp.eye(K)  # Moderate volatility of volatility
    )

    # If true_params is provided and we want to fix mu, use the true mu value
    if true_params is not None and use_transformations is False:  # Only fix mu if not using transformations
        initial_params = eqx.tree_at(lambda p: p.mu, initial_params, true_params.mu)

    # Apply identification constraint to initial parameters
    initial_params = apply_identification_constraint(initial_params)

    # Transform parameters if needed
    if use_transformations:
        initial_params = transform_params(initial_params)

    # Determine if we should fix mu
    fix_mu = true_params is not None and not use_transformations

    # Get objective function
    objective_fn = get_objective_function(
        filter_type=filter_type,
        filter_instance=filter_instance,
        stability_penalty_weight=stability_penalty_weight,
        priors=priors,
        is_transformed=use_transformations
    )

    # Create optimizer
    optimizer = create_optimizer(
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        rtol=rtol,
        atol=atol,
        verbose=verbose
    )

    # Print initial loss if verbose
    if verbose:
        try:
            initial_loss, _ = objective_fn(initial_params, returns)
            print(f"Initial loss: {initial_loss:.4f}")

            # If true_params is provided, also print loss at true parameters
            if true_params is not None:
                true_params_transformed = transform_params(true_params) if use_transformations else true_params
                true_loss, _ = objective_fn(true_params_transformed, returns)
                print(f"Loss at true parameters: {true_loss:.4f}")
        except Exception as e:
            print(f"Error calculating initial loss: {e}")

    # Run optimization with logging
    try:
        sol, param_history = minimize_with_logging(
            objective_fn=objective_fn,
            initial_params=initial_params,
            solver=optimizer,
            static_args=returns,
            max_steps=max_steps,
            log_interval=log_interval if log_params else max_steps + 1,  # Only log at the end if log_params is False
            options={},
            throw=False
        )

        # Calculate final loss
        try:
            final_loss, _ = objective_fn(sol.value, returns)
        except Exception:
            final_loss = float('inf')

        # Untransform parameters if needed
        final_params = sol.value
        if use_transformations:
            try:
                final_params = untransform_params(final_params)
            except Exception as e:
                if verbose:
                    print(f"Error untransforming parameters: {e}")

        # Apply identification constraint to final parameters
        try:
            final_params = apply_identification_constraint(final_params)
        except Exception as e:
            if verbose:
                print(f"Error applying identification constraint: {e}")

        # Untransform parameter history if needed
        if use_transformations and log_params:
            try:
                param_history = [untransform_params(p) for p in param_history]
                # Apply identification constraint to each parameter in history
                param_history = [apply_identification_constraint(p) for p in param_history]
            except Exception as e:
                if verbose:
                    print(f"Error processing parameter history: {e}")

        # Calculate loss history
        loss_history = []
        if log_params:
            for p in param_history:
                try:
                    p_transformed = transform_params(p) if use_transformations else p
                    loss, _ = objective_fn(p_transformed, returns)
                    loss_history.append(float(loss))
                except Exception:
                    loss_history.append(float('inf'))

        # Check if optimization was successful
        success = sol.result == optx.RESULTS.successful
        error_message = None
        steps = sol.stats.get('num_steps', len(param_history) - 1)

    except Exception as e:
        # Handle optimization failure
        success = False
        error_message = str(e)
        final_loss = float('inf')
        final_params = initial_params
        param_history = [initial_params]
        loss_history = [float('inf')]
        steps = 0

    # Calculate time taken
    time_taken = time.time() - start_time

    # Create and return result
    result = OptimizerResult(
        filter_type=filter_type,
        optimizer_name=optimizer_name,
        uses_transformations=use_transformations,
        fix_mu=fix_mu,
        prior_config_name=prior_config_name,
        success=success,
        final_loss=final_loss,
        steps=steps,
        time_taken=time_taken,
        error_message=error_message,
        final_params=final_params,
        param_history=param_history,
        loss_history=loss_history
    )

    return result
