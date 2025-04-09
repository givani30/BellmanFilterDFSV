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
from typing import Dict, Any, Optional, List, Callable, Tuple
from collections import namedtuple
import time
from jax import lax

from bellman_filter_dfsv.utils.solvers import create_optimizer

from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
from bellman_filter_dfsv.filters.bellman import DFSVBellmanFilter
from bellman_filter_dfsv.filters.particle import DFSVParticleFilter
from bellman_filter_dfsv.utils.transformations import transform_params, untransform_params, apply_identification_constraint


# Filter type enumeration
class FilterType(Enum):
    """Enumeration of available filter types.

    Attributes:
        BIF: Bellman Information Filter
        BF: Bellman Filter
        PF: Particle Filter
    """
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
    "result_code",           # The specific result code from the optimizer
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
                          is_transformed: bool = False,
                          fix_mu: bool = False,
                          true_mu: Optional[jnp.ndarray] = None):
    """Get the appropriate objective function wrapper for a filter type.

    This function returns a wrapper around the appropriate objective function.
    The wrapper handles parameter untransformation, fixing mu, applying
    identification constraints, and calling the underlying objective function.

    Args:
        filter_type: Type of filter.
        filter_instance: Filter instance.
        stability_penalty_weight: Weight for stability penalty.
        priors: Dictionary of prior hyperparameters.
        is_transformed: Whether the parameters passed to the wrapper are transformed.
        fix_mu: Whether to fix the mu parameter to true_mu.
        true_mu: The true value of mu to use if fix_mu is True.

    Returns:
        The objective function wrapper.

    Raises:
        ValueError: If the filter type is unknown or if fix_mu is True but true_mu is None.
    """
    if fix_mu and true_mu is None:
        raise ValueError("fix_mu is True, but true_mu was not provided.")

    from bellman_filter_dfsv.filters.objectives import bellman_objective, pf_objective

    # Map filter types to their underlying objective functions
    # Note: We select the *untransformed* objective here, as the wrapper handles transformations.
    objective_map = {
        FilterType.BIF: bellman_objective,
        FilterType.BF: bellman_objective,
        FilterType.PF: pf_objective
    }

    # Check if filter type is supported
    if filter_type not in objective_map:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Get the underlying objective function
    underlying_objective = objective_map[filter_type]

    # Create the objective function wrapper
    # This wrapper handles untransformation, fixing mu, and constraints
    # IMPORTANT: We use a separate JIT-compiled function for the actual computation
    # to ensure the JIT compilation is effective
    @eqx.filter_jit # JIT compile the computation for performance
    def _compute_objective(params, observations, is_transformed_flag, fix_mu_flag, true_mu_val):
        # 1. Untransform parameters if they are passed in transformed space
        params_iter = untransform_params(params) if is_transformed_flag else params

        # 2. Fix mu if requested
        if fix_mu_flag:
            # We already checked true_mu is not None if fix_mu is True
            params_iter = eqx.tree_at(lambda p: p.mu, params_iter, true_mu_val)

        # 3. Apply identification constraint
        params_fixed_constrained = apply_identification_constraint(params_iter)

        # 4. Calculate loss using the underlying objective function
        loss = underlying_objective(
            params_fixed_constrained, observations, filter_instance,
            priors=priors, stability_penalty_weight=stability_penalty_weight
        )
        return loss

    # Wrapper function that calls the JIT-compiled computation
    def objective_wrapper(params, observations):
        loss = _compute_objective(params, observations, is_transformed, fix_mu, true_mu)
        # Return loss and None for aux (as expected by optimistix)
        return loss, None

    return objective_wrapper


def minimize_with_logging(objective_fn: Callable, initial_params: Any, solver: optx.AbstractMinimiser,
                         static_args: Any = None, max_steps: int = 100,
                         log_interval: int = 1, throw: bool = False,
                         options: Dict[str, Any] = None, verbose: bool = False) -> Tuple[optx.Solution, List]:
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
        options: Optimizer options.
        verbose: Whether to print verbose output.

    Returns:
        A tuple containing the optimization solution and parameter history.
    """
    # Initialize parameter history
    param_history = [initial_params]
    y = initial_params

    # Prepare options
    if options is None:
        options = {}
    #prepare tags
    tags = frozenset()
    # Get the shape and dtype of the output
    try:
        test_output, test_aux = objective_fn(y, static_args)
        f_struct = jax.ShapeDtypeStruct(test_output.shape, test_output.dtype)
        aux_struct = None if test_aux is None else jax.ShapeDtypeStruct(test_aux.shape, test_aux.dtype)
    except Exception as e:
        if throw:
            raise e
        else:
            # Return a dummy solution with the initial parameters
            sol = optx.Solution(
                value=initial_params,
                result=optx.RESULTS.nonlinear_divergence,  # Use a valid result value
                aux=None,  # Add the aux parameter
                stats={"num_steps": 0},
                state=None
            )
            return sol, [initial_params]

    # Initialize solver state
    #also initialize termination
    #jit compile solver steps
    step=eqx.filter_jit(eqx.Partial(solver.step,fn=objective_fn,args=static_args,options=options,tags=tags))
    terminate=eqx.filter_jit(eqx.Partial(solver.terminate,fn=objective_fn,args=static_args,options=options,tags=tags))

    # Run optimization with logging
    step_count = 0

    # Initialize state and termination
    state = solver.init(objective_fn, y, static_args, options, f_struct, aux_struct, tags)
    converged,result=solver.terminate(objective_fn, y, static_args, options, state, tags)

    # Calculate initial loss for verbose output
    if verbose:
        initial_loss, _ = objective_fn(initial_params, static_args)
        print(f"Initial loss: {float(initial_loss):.4f}")

    while not converged and step_count < max_steps:
        # Perform one step of optimization
        try:
            # Take a step
            y, state, _ = step(y=y, state=state)
            step_count += 1


            # Log parameters at specified intervals
            if step_count % log_interval == 0:
                param_history.append(y)

            # Check for convergence
            converged, result = terminate(y=y, state=state)
            if converged:
                if verbose:
                    print(f"Converged after {step_count} steps")
                break
        except Exception as e:
            if throw:
                raise e
            else:
                # Create a failure result
                result = optx.RESULTS.failed
                break

    # Create solution object
    # Determine the final result code based on how the loop terminated
    if not converged and step_count >= max_steps:
        # Explicitly set max_steps_reached if the loop finished due to steps
        result = optx.RESULTS.max_steps_reached
    # If converged is True, 'result' holds the reason from terminate().
    # If an exception occurred, 'result' should be optx.RESULTS.failed (set in except block).

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

        # Make sure the final parameters are in the history
        if final_y is not None and (not param_history or param_history[-1] is not final_y):
            param_history.append(final_y)

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


def minimize_with_lax_while(objective_fn: Callable, initial_params: Any, solver: optx.AbstractMinimiser,
                          static_args: Any = None, max_steps: int = 100,
                          log_interval: int = 1, throw: bool = False,
                          options: Dict[str, Any] = None, verbose: bool = False) -> Tuple[optx.Solution, List]:
    """Minimize an objective function with parameter logging using lax.while_loop.

    This implementation uses JAX's lax.while_loop and pre-allocated arrays for parameter history,
    which allows for better compilation and potentially faster execution compared to the
    standard Python loop implementation. The function handles error cases gracefully and
    falls back to the standard implementation if the lax.while_loop approach fails.

    The implementation pre-allocates fixed-size arrays for parameter history to ensure
    compatibility with JAX's JIT compilation. It tracks parameters at specified intervals
    and ensures that both initial and final parameters are included in the history.

    Args:
        objective_fn: The objective function to minimize. Should return a tuple of
            (loss, aux), where aux can be None.
        initial_params: Initial parameter values as a PyTree.
        solver: An optimistix solver instance implementing the AbstractMinimiser interface.
        static_args: Static arguments to pass to the objective function. These arguments
            will not be differentiated with respect to.
        max_steps: Maximum number of optimization steps to perform.
        log_interval: Interval at which to log parameters. Set to 1 to log at every step,
            or to a larger value to reduce memory usage.
        throw: Whether to throw an exception if optimization fails. If False, returns a
            solution with a failure result code instead.
        options: Additional options to pass to the solver.

    Returns:
        A tuple containing:
            - The optimization solution as an optx.Solution object
            - A list of parameter estimates at each logged step

    Note:
        This function is significantly faster than the standard minimize_with_logging
        implementation due to better JIT compilation, but may not handle all edge cases
        as robustly. If an error occurs during the lax.while_loop execution, it falls
        back to the standard implementation.
    """
    if options is None:
        options = {}

    # Calculate the maximum number of history entries we'll need
    # If we log every log_interval steps, plus initial and final params
    max_history_size = (max_steps // log_interval) + 2  # +2 for initial and final params

    # Get the shape and dtype of the output
    try:
        test_output, test_aux = objective_fn(initial_params, static_args)
        f_struct = jax.ShapeDtypeStruct(test_output.shape, test_output.dtype)
        aux_struct = None if test_aux is None else jax.ShapeDtypeStruct(test_aux.shape, test_aux.dtype)
    except Exception as e:
        if throw:
            raise e
        else:
            # Return a dummy solution with the initial parameters
            sol = optx.Solution(
                value=initial_params,
                result=optx.RESULTS.nonlinear_divergence,
                aux=None,
                stats={"num_steps": 0},
                state=None
            )
            return sol, [initial_params]

    # Initialize solver state
    state = solver.init(objective_fn, initial_params, static_args, options, f_struct, aux_struct, frozenset())

    # Pre-allocate parameter history array
    # We'll use a PyTree with the same structure as initial_params, but with an additional
    # leading dimension for the history entries
    def create_history_array(leaf):
        # Create array with shape [max_history_size, *leaf.shape]
        return jnp.zeros((max_history_size,) + leaf.shape, dtype=leaf.dtype)

    # Create the parameter history arrays
    param_history_arrays = jax.tree_map(create_history_array, initial_params)

    # Store initial parameters in the first slot of history arrays
    def set_initial_params(history_leaf, param_leaf):
        return history_leaf.at[0].set(param_leaf)

    param_history_arrays = jax.tree_map(set_initial_params, param_history_arrays, initial_params)

    # Define the loop state (carry)
    # Initialize result with a valid RESULTS enum value to ensure type consistency
    init_result = optx.RESULTS.max_steps_reached  # Default value, will be updated during iterations

    init_carry = {
        'step': 0,
        'y': initial_params,
        'aux': None,
        'state': state,
        'converged': False,
        'result': init_result,
        'param_history': param_history_arrays,
        'history_idx': 1  # Next index to write to (0 already has initial params)
    }

    # Define condition function - continue until max steps or convergence
    def cond_fn(carry):
        return (carry['step'] < max_steps) & (~carry['converged'])

    # Define body function - one step of optimization
    def body_fn(carry):
        # Extract current state
        step = carry['step']
        y = carry['y']
        state = carry['state']
        param_history = carry['param_history']
        history_idx = carry['history_idx']

        # Perform one step of optimization
        try:
            new_y, new_state, aux = solver.step(
                objective_fn, y, static_args, options, state, frozenset()
            )

            # Check for convergence
            converged, result = solver.terminate(
                objective_fn, new_y, static_args, options, new_state, frozenset()
            )
        except Exception:
            # If there's an error, return a state that will terminate the loop
            # and indicate failure
            return {
                'step': max_steps,  # Force termination
                'y': y,
                'aux': None,
                'state': state,
                'converged': True,
                'result': optx.RESULTS.failed,
                'param_history': param_history,
                'history_idx': history_idx
            }

        # Determine if we should log this step
        should_log = ((step + 1) % log_interval == 0) | converged

        # Update parameter history if needed
        def update_history(history_leaf, param_leaf):
            return lax.cond(
                should_log & (history_idx < max_history_size),
                lambda: history_leaf.at[history_idx].set(param_leaf),
                lambda: history_leaf
            )

        new_param_history = jax.tree_map(
            update_history, param_history, new_y
        )

        # Update history index if we logged
        new_history_idx = lax.cond(
            should_log & (history_idx < max_history_size),
            lambda: history_idx + 1,
            lambda: history_idx
        )

        # Update carry
        new_carry = {
            'step': step + 1,
            'y': new_y,
            'aux': aux,
            'state': new_state,
            'converged': converged,
            'result': result,
            'param_history': new_param_history,
            'history_idx': new_history_idx
        }

        return new_carry

    # Run the loop
    try:
        final_carry = lax.while_loop(cond_fn, body_fn, init_carry)

        # Extract results
        final_y = final_carry['y']
        final_aux = final_carry['aux']
        final_state = final_carry['state']
        result = final_carry['result']
        param_history_arrays = final_carry['param_history']
        history_idx = final_carry['history_idx']

        # Check the termination reason explicitly
        final_step = final_carry['step']
        final_converged = final_carry['converged']
        final_result_from_terminate = final_carry['result'] # Result from the last terminate() call inside loop

        if not final_converged and final_step >= max_steps:
            # Override result if max_steps was the primary reason for stopping
            result = optx.RESULTS.max_steps_reached
        else:
            # Otherwise, use the result determined by terminate() or failure handling
            result = final_result_from_terminate

        # Perform postprocessing
        try:
            final_y, final_aux, stats = solver.postprocess(
                objective_fn, final_y, final_aux, static_args, options,
                final_state, frozenset(), result
            )

            # Create the solution object
            sol = optx.Solution(
                value=final_y,
                result=result,
                stats=stats,
                aux=final_aux,
                state=final_state
            )
        except Exception as e:
            if throw:
                raise e
            else:
                # Create a failure solution
                sol = optx.Solution(
                    value=final_y,
                    result=optx.RESULTS.nonlinear_divergence,
                    stats={"error": str(e)},
                    aux=None,
                    state=final_state
                )
    except Exception as e:
        if throw:
            raise e
        else:
            print(f"Error in lax.while_loop: {e}")
            # Fall back to standard implementation
            return minimize_with_logging(
                objective_fn=objective_fn,
                initial_params=initial_params,
                solver=solver,
                static_args=static_args,
                max_steps=max_steps,
                log_interval=log_interval,
                throw=throw,
                options=options,
                verbose=verbose
            )

    # We'll convert the parameter history arrays directly to a list of parameters

    # Convert the parameter history arrays back to a list of parameters
    param_history = []
    for i in range(history_idx):
        def get_params_at_idx(history_leaf):
            return history_leaf[i]
        param_at_idx = jax.tree_map(get_params_at_idx, param_history_arrays)
        param_history.append(param_at_idx)

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
    verbose: bool = False,
    use_lax_while: bool = True
) -> OptimizerResult:
    """Run optimization for a specific filter type and configuration.

    This function integrates all the components needed for optimizing DFSV model parameters
    using various filters and optimizers. It handles parameter transformations, objective
    function selection, optimization, and result processing.

    The function supports multiple filter types (BIF, BF, PF) and optimizers, and can
    optionally use parameter transformations to ensure constraints are satisfied during
    optimization. It also supports fixing the mu parameter to its true value if true_params
    is provided.

    Args:
        filter_type: Type of filter to use (BIF, BF, or PF).
        returns: Observed returns data with shape (T, N) where T is the number of time
            points and N is the number of observed variables.
        true_params: True parameters (optional). If provided, can be used for fixing mu
            and for reporting the true parameter log-likelihood.
        use_transformations: Whether to use parameter transformations to ensure constraints
            are satisfied during optimization.
        optimizer_name: Name of the optimizer to use (e.g., "BFGS", "Adam", "TrustRegion").
        priors: Dictionary of prior hyperparameters for regularization.
        stability_penalty_weight: Weight for stability penalty in objective function.
            Higher values enforce more stability in the estimated parameters.
        max_steps: Maximum number of optimization steps to perform.
        num_particles: Number of particles for particle filter (only used when
            filter_type is PF).
        prior_config_name: Description of prior configuration (for reporting).
        log_params: Whether to log parameters during optimization for tracking
            parameter evolution.
        log_interval: Interval at which to log parameters. Set to 1 to log at every step,
            or to a larger value to reduce memory usage.
        learning_rate: Initial learning rate for gradient-based optimizers.
        rtol: Relative tolerance for convergence criteria.
        atol: Absolute tolerance for convergence criteria.
        verbose: Whether to enable verbose output from the optimizer, showing
            progress at each step.
        use_lax_while: Whether to use lax.while_loop for optimization. This can
            provide significant speedups (3-4x) but may not handle all edge cases
            as robustly as the standard implementation. Note that if verbose=True,
            this parameter is ignored and the standard implementation is used to
            ensure proper verbose output.

    Returns:
        OptimizerResult: A namedtuple containing optimization results and metadata,
        including:
            - filter_type: The filter type used
            - optimizer_name: The optimizer used
            - uses_transformations: Whether parameter transformations were used
            - fix_mu: Whether mu parameter was fixed
            - prior_config_name: Description of prior configuration
            - success: Whether optimization succeeded
            - final_loss: Final loss value
            - steps: Number of steps taken
            - time_taken: Time taken in seconds
            - error_message: Error message if any
            - final_params: Final estimated parameters
            - param_history: History of parameter estimates during optimization
            - loss_history: History of loss values during optimization
    """
    # Start timing
    start_time = time.time()
    #TODO: add a warning that verbose=True will ignore use_lax_while and thus be slower

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

    # Determine if we should fix mu (only for BIF, requires true_params)
    fix_mu =  (true_params is not None)

    # Get objective function wrapper
    objective_fn = get_objective_function(
        filter_type=filter_type,
        filter_instance=filter_instance,
        stability_penalty_weight=stability_penalty_weight,
        priors=priors,
        is_transformed=use_transformations,
        fix_mu=fix_mu,  # Pass the flag
        true_mu=true_params.mu if fix_mu else None # Pass the true value if fixing
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
            # If true_params is provided, print loss at true parameters
            if true_params is not None:
                true_params_transformed = transform_params(true_params) if use_transformations else true_params
                true_loss, _ = objective_fn(true_params_transformed, returns)
                print(f"Loss at true parameters: {true_loss:.4f}")
        except Exception as e:
            print(f"Error calculating loss at true parameters: {e}")

    # Run optimization with logging
    try:
        # If verbose is True, always use standard implementation for better output
        # Otherwise, use lax.while_loop if specified
        if use_lax_while and not verbose:
            # Use lax.while_loop implementation for potentially better performance
            sol, param_history = minimize_with_lax_while(
                objective_fn=objective_fn,
                initial_params=initial_params,
                solver=optimizer,
                static_args=returns,
                max_steps=max_steps,
                log_interval=log_interval if log_params else max_steps + 1,  # Only log at the end if log_params is False
                options={},
                throw=False,
                verbose=verbose
            )
        else:
            # Use standard implementation
            sol, param_history = minimize_with_logging(
                objective_fn=objective_fn,
                initial_params=initial_params,
                solver=optimizer,
                static_args=returns,
                max_steps=max_steps,
                log_interval=log_interval if log_params else max_steps + 1,  # Only log at the end if log_params is False
                options={},
                throw=False,
                verbose=verbose
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

        # Extract loss history from optimizer state if available
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
        result_code=sol.result if 'sol' in locals() else None,
        final_loss=final_loss,
        steps=steps,
        time_taken=time_taken,
        error_message=error_message,
        final_params=final_params,
        param_history=param_history,
        loss_history=loss_history
    )

    return result
