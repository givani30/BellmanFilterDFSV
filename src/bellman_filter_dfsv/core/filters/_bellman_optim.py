# src/bellman_filter_dfsv/core/filters/_bellman_optim.py
"""State update optimization for the Bellman Information Filter.

This module implements the numerical optimization required for the BIF state
update step, which finds α_{t|t} by maximizing p(α_t|y_t,F_{t-1}) plus a penalty term.

The optimization problem minimizes the objective:
    J(α) = -ℓ(y_t|α_t) + 1/2||α_t - α_{t|t-1}||²_{Ω_{t|t-1}}

where:
- α_t = [f_t', h_t']': Complete state vector
- ℓ(y_t|α_t): Log-likelihood of observation
- Ω_{t|t-1}: Predicted information matrix
- α_{t|t-1}: Predicted state

Implementation Features:
- JAX automatic differentiation and JIT compilation
- BFGS optimization with trust region strategy
- Cholesky decomposition for stable matrix operations
- Numerical safeguards and fallback strategies

The implementation provides efficient functions for:
1. Log-posterior evaluation
2. Factor state updates
3. Log-volatility state updates
4. Complete state vector optimization
"""
from functools import partial
from typing import Callable, Tuple

import equinox as eqx  # Add equinox import
import jax
import jax.numpy as jnp
import jax.scipy.linalg  # Add linalg import
import numpy as np  # Add numpy import
import optimistix as optx
from jaxtyping import PyTree, Scalar

# Type hint for build_covariance function signature
BuildCovarianceFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
LogPosteriorFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], float]


@eqx.filter_jit
def neg_log_post_h(
    log_vols: jnp.ndarray,
    # --- Fixed arguments for this optimization step ---
    factors: jnp.ndarray,
    lambda_r: jnp.ndarray,
    sigma2: jnp.ndarray,
    predicted_state: jnp.ndarray,
    I_pred: jnp.ndarray,
    observation: jnp.ndarray,
    # --- Static arguments & Dependencies ---
    K: int,
    build_covariance_fn: BuildCovarianceFn,
    log_posterior_fn: LogPosteriorFn,
) -> float:
    """Calculates the negative log posterior w.r.t. log-volatilities (h).

    This function computes the objective minimized in the 'h' update step of
    the block coordinate descent. It includes the negative log-likelihood
    of the observation given the state and a prior penalty term derived from
    the predicted state distribution.

    Objective: J(h) = -log p(y|f,h) + 0.5 * (alpha - alpha_pred)^T I_pred (alpha - alpha_pred)
    where only the terms depending on h are relevant for this minimization.

    Args:
        log_vols: Current log-volatility values h (K,) (variable of optimization).
        factors: Current factor values f (K,) (fixed for this step).
        lambda_r: Factor loading matrix Lambda (N, K).
        sigma2: Idiosyncratic variances diag(Sigma_e) (N,).
        predicted_state: Predicted state vector alpha_{t|t-1} = [f_pred, h_pred]
                         (state_dim,).
        I_pred: Predicted precision matrix Omega_{t|t-1} (state_dim, state_dim).
        observation: Observation vector y_t (N,).
        K: Number of factors (static).
        build_covariance_fn: Function to build observation covariance Sigma_t.
        log_posterior_fn: Function to compute log p(y_t|alpha_t).

    Returns:
        The negative log posterior value for the given h (scalar float).
    """
    # Ensure inputs are flattened
    log_vols = log_vols.flatten()
    factors = factors.flatten()
    predicted_state = predicted_state.flatten()

    # Build current state vector alpha = [f, h]
    alpha = jnp.concatenate([factors, log_vols])

    # 1. Compute negative log-likelihood part using the passed function
    neg_log_lik = -log_posterior_fn(lambda_r, sigma2, alpha, observation)

    # 2. Compute prior penalty part involving h
    #    0.5 * (h - h_pred)^T I_hh (h - h_pred) + (f - f_pred)^T I_fh (h - h_pred)
    h_pred = predicted_state[K:]
    h_diff = log_vols - h_pred
    I_pred_hh = I_pred[K:, K:]
    I_pred_fh = I_pred[:K, K:]
    prior_penalty = 0.5 * jnp.dot(h_diff, jnp.dot(I_pred_hh, h_diff))
    prior_penalty += jnp.dot(factors - predicted_state[:K], jnp.dot(I_pred_fh, h_diff))

    # Total objective
    result = neg_log_lik + prior_penalty
    return result


def update_h_bfgs(
    h_init: jnp.ndarray,
    # --- Fixed arguments for this optimization step ---
    factors: jnp.ndarray,
    lambda_r: jnp.ndarray,
    sigma2: jnp.ndarray,
    pred_state: jnp.ndarray,
    I_pred: jnp.ndarray,
    observation: jnp.ndarray,
    # --- Static arguments & Dependencies ---
    K: int,
    build_covariance_fn: BuildCovarianceFn,
    log_posterior_fn: LogPosteriorFn,
    h_solver: optx.AbstractMinimiser, # Pass solver instance
    inner_max_steps: int = 100,
) -> Tuple[jnp.ndarray, bool]:
    """Updates log-volatilities h using trust-region BFGS optimization.

    This function minimizes J(h) = -ℓ(y_t|f,h) + Prior_Penalty(h) with respect to h,
    keeping factors f fixed. Uses a trust-region BFGS method with fallback strategies
    for robustness.

    Mathematical Details:
        1. Objective Function:
           J(h) = -ℓ(y_t|f,h) + 1/2·h'Ω_hh·h + h'Ω_fh'(f - f_pred)

        2. Gradient/Hessian:
           - Computed automatically via JAX AD
           - Trust region helps handle potential non-convexity
           - Uses CustomBFGS with modified DoglegDescent strategy

    Implementation Notes:
        - Employs Optimistix minimizer with trust region
        - Falls back to h_init if optimization fails
        - Uses JAX-compatible function wrapping
        - Maintains JIT compatibility throughout

    Args:
        h_init: Initial guess for log-volatilities h (K,).
        factors: Current factor values f (K,) (fixed).
        lambda_r: Factor loading matrix Λ (N, K).
        sigma2: Idiosyncratic variances diag(Σ_ε) (N,).
        pred_state: Predicted state vector α_{t|t-1} (state_dim,).
        I_pred: Predicted information matrix Ω_{t|t-1} (state_dim, state_dim).
        observation: Observation vector y_t (N,).
        K: Number of factors.
        build_covariance_fn: Function to build covariance Σ_t.
        log_posterior_fn: Function to compute ℓ(y_t|α_t).
        h_solver: Pre-configured Optimistix solver.
        inner_max_steps: Maximum BFGS iterations.

    Returns:
        A tuple containing:
            - h_new: Updated log-volatility values (K,)
            - success: Boolean indicating optimization success
    """
    # Objective function wrapper to match optimistix fn(y, args) signature
    def objective_fn_h(h, objective_args):
        (
            current_factors, current_lambda_r, current_sigma2,
            current_predicted_state, current_I_pred, current_observation,
            static_K, static_build_cov_fn, static_log_post_fn
        ) = objective_args
        # Call the standalone neg_log_post_h function
        return neg_log_post_h(
            log_vols=h,
            factors=current_factors,
            lambda_r=current_lambda_r,
            sigma2=current_sigma2,
            predicted_state=current_predicted_state,
            I_pred=current_I_pred,
            observation=current_observation,
            K=static_K,
            build_covariance_fn=static_build_cov_fn,
            log_posterior_fn=static_log_post_fn
        )

    # Prepare arguments tuple for the objective function
    objective_args = (
        factors, lambda_r, sigma2, pred_state, I_pred, observation,
        K, build_covariance_fn, log_posterior_fn
    )

    # Run the minimization using the passed solver instance
    sol = optx.minimise(
        fn=objective_fn_h,
        solver=h_solver,
        y0=h_init,
        args=objective_args,
        options={},
        max_steps=inner_max_steps,
        throw=False # Prevent errors from stopping JIT compilation
    )

    # Check if optimization was successful
    successful = sol.result == optx.RESULTS.successful

    # Use jnp.where to return h_init if optimization failed, sol.value if successful
    h_new = jnp.where(successful, sol.value, h_init)
    return h_new, successful


# Removed unused obj_and_grad_fn
# @eqx.filter_jit
# def obj_and_grad_fn(
#     x: jnp.ndarray,
#     # --- Fixed arguments ---
#     lambda_r: jnp.ndarray,
#     sigma2: jnp.ndarray,
#     predicted_state: jnp.ndarray,
#     I_pred: jnp.ndarray,
#     observation: jnp.ndarray,
#     # --- Static arguments & Dependencies ---
#     K: int,
#     build_covariance_fn: BuildCovarianceFn,
# ) -> Tuple[float, jnp.ndarray]:
#     """
#     Combined objective and gradient function for optimization (e.g., full state).
#
#     Objective J(x) = -log p(y|x) + 0.5 * (x - x_pred)^T I_pred (x - x_pred) + const.
#
#     Args:
#         x: Current state estimate [f, h].
#         lambda_r: Factor loading matrix.
#         sigma2: Idiosyncratic variances.
#         predicted_state: Predicted state [f_pred, h_pred].
#         I_pred: Predicted precision matrix.
#         observation: Observation vector.
#         K: Number of factors.
#         build_covariance_fn: Function to build covariance A(h).
#
#     Returns:
#         Tuple[float, jnp.ndarray]: Objective value and gradient w.r.t x.
#     """
#     # ... implementation removed ...


def update_factors(
    log_volatility: jnp.ndarray,
    # --- Fixed arguments ---
    lambda_r: jnp.ndarray,
    sigma2: jnp.ndarray,
    observation: jnp.ndarray,
    factors_pred: jnp.ndarray,
    log_vols_pred: jnp.ndarray,
    I_f: jnp.ndarray,
    I_fh: jnp.ndarray,
    # --- Dependencies ---
    build_covariance_fn: BuildCovarianceFn,
) -> jnp.ndarray:
    """Updates factor values f by solving the state update optimization.

    Given fixed log-volatilities h, this function finds the optimal factors f
    by solving the linear system that arises from setting ∂J/∂f = 0:

    Mathematical Details:
        ∂J/∂f = -∂ℓ(y_t|f,h)/∂f + Ω_ff(f - f_pred) + Ω_fh(h - h_pred) = 0

        This yields the linear system:
        (Λ'Σ_t^{-1}Λ + Ω_ff)f = Λ'Σ_t^{-1}y_t + Ω_ff·f_pred + Ω_fh(h - h_pred)

        where:
        - Σ_t = ΛΣ_f(h)Λ' + Σ_ε is the observation covariance
        - Σ_f(h) = diag(exp(h)) is the factor covariance
        - Ω_ff, Ω_fh are blocks of the predicted information matrix

    Implementation Notes:
        - Uses Cholesky decomposition for stable Σ_t^{-1} applications
        - Adds small jitter term for numerical stability
        - Employs JAX's linear algebra routines

    Args:
        log_volatility: Current log-volatility values h (K,).
        lambda_r: Factor loading matrix Λ (N, K).
        sigma2: Idiosyncratic variances diag(Σ_ε) (N,).
        observation: Observation vector y_t (N,).
        factors_pred: Predicted factor values f_{t|t-1} (K,).
        log_vols_pred: Predicted log-volatility values h_{t|t-1} (K,).
        I_f: Factor block Ω_ff of predicted information (K, K).
        I_fh: Factor-volatility block Ω_fh of predicted information (K, K).
        build_covariance_fn: Function to build observation covariance Σ_t.

    Returns:
        The optimal factor values f that minimize J(f,h) for fixed h (K,).
    """
    # Build Sigma_t = Lambda Sigma_f Lambda^T + Sigma_e
    A = build_covariance_fn(lambda_r, jnp.exp(log_volatility), sigma2)
    # Cholesky decomposition for stable inversion: Sigma_t = L L^T
    L = jax.scipy.linalg.cho_factor(A, lower=True)

    # Define function for Sigma_t^-1 @ x using Cholesky solver
    def A_inv(x):
        return jax.scipy.linalg.cho_solve(L, x)

    # Construct the linear system Ax = b for factors f
    # A = Lambda^T Sigma_t^-1 Lambda + Omega_ff
    lhs_mat = jnp.dot(lambda_r.T, A_inv(lambda_r)) + I_f + 1e-8 * jnp.eye(I_f.shape[0]) # Add jitter

    # b = Lambda^T Sigma_t^-1 y_t + Omega_ff f_pred + Omega_fh (h - h_pred)
    rhs_vec = (
        jnp.dot(lambda_r.T, A_inv(observation))
        + jnp.dot(I_f, factors_pred)
        + jnp.dot(I_fh, (log_volatility - log_vols_pred)) # Corrected term
    )

    # Solve the linear system A f = b
    updated_factors = jnp.linalg.solve(lhs_mat, rhs_vec)
    return updated_factors


# Removed duplicated code block from obj_and_grad_fn


def _block_coordinate_update_impl(
    lambda_r: jnp.ndarray,
    sigma2: jnp.ndarray, # Expect 1D JAX array
    alpha: jnp.ndarray,
    pred_state: jnp.ndarray,
    I_pred: jnp.ndarray, # Predicted Information Matrix (Omega_{t|t-1})
    observation: jnp.ndarray,
    K: int, # Pass K explicitly
    max_iters: int, # Static arg
    h_solver: optx.AbstractMinimiser, # Static arg
    build_covariance_fn: BuildCovarianceFn, # Pass JITted build_covariance
    log_posterior_fn: LogPosteriorFn # Pass JITted log_posterior
) -> jnp.ndarray:
    """Optimizes the state vector α_t in the BIF update step.

    This function finds α_{t|t} by minimizing:
    
    J(α) = -ℓ(y_t|α_t) + 1/2||α_t - α_{t|t-1}||²_{Ω_{t|t-1}}

    The optimization alternates between:
    1. Solving a linear system for factors f
    2. Using BFGS for log-volatilities h

    Implementation Notes:
        - Uses jax.lax.fori_loop for JIT compatibility
        - Employs trust region methods via h_solver
        - Includes fallback strategies for failed updates
        - Maintains strict numerical stability controls

    Args:
        lambda_r: Factor loading matrix Λ (N, K).
        sigma2: Idiosyncratic variances diag(Σ_ε) (N,).
        alpha: Initial guess for state vector α_{t|t} (state_dim,).
        pred_state: Predicted state vector α_{t|t-1} (state_dim,).
        I_pred: Predicted information matrix Ω_{t|t-1} (state_dim, state_dim).
        observation: Observation vector y_t (N,).
        K: Number of factors.
        max_iters: Maximum iterations for optimization.
        h_solver: Pre-configured Optimistix solver for h-update.
        build_covariance_fn: JIT-compiled function for Σ_t construction.
        log_posterior_fn: JIT-compiled function for ℓ(y_t|α_t).

    Returns:
        The optimized state vector α_{t|t} (state_dim,).
    """
    alpha = alpha.flatten()
    pred_state = pred_state.flatten()
    observation = observation.flatten()

    # Split states
    factors_guess = alpha[:K]
    log_vols_guess = alpha[K:]
    factors_pred = pred_state[:K]
    log_vols_pred = pred_state[K:]

    # Partition information matrix
    I_f = I_pred[:K, :K]
    I_fh = I_pred[:K, K:]

    # Define the loop body for coordinate descent iterations
    def body_fn(
        i, carry: Tuple[jnp.ndarray, jnp.ndarray] # Loop index, (f, h) carry
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ Single iteration of the block-coordinate update. `carry` is (f, h). """
        f_current, h_current = carry

        # 1. Update factors (f) holding h fixed
        f_new = update_factors(
            log_volatility=h_current,
            lambda_r=lambda_r,
            sigma2=sigma2,
            observation=observation,
            factors_pred=factors_pred,
            log_vols_pred=log_vols_pred,
            I_f=I_f,
            I_fh=I_fh,
            build_covariance_fn=build_covariance_fn
        )

        # 2. Update log-volatilities (h) holding f fixed (using the new f_new)
        h_new, h_update_success = update_h_bfgs(
            h_init=h_current, # Start from previous h
            factors=f_new,    # Use the newly updated factors
            lambda_r=lambda_r,
            sigma2=sigma2,
            pred_state=pred_state, # Pass full predicted state
            I_pred=I_pred,         # Pass full predicted precision
            observation=observation,
            K=K,
            build_covariance_fn=build_covariance_fn,
            log_posterior_fn=log_posterior_fn,
            h_solver=h_solver,
            inner_max_steps=100 # Max steps for BFGS
        )
        # Note: h_update_success is currently ignored, could be used for diagnostics
        return (f_new, h_new)

    # Run the block coordinate descent loop
    init_carry = (factors_guess, log_vols_guess)
    f_final, h_final = jax.lax.fori_loop(0, max_iters, body_fn, init_carry)

    # Return the final concatenated state vector
    return jnp.concatenate([f_final, h_final])

class CustomBFGS(optx.AbstractBFGS):
    """Custom BFGS solver optimized for BIF state updates.

    This class configures a BFGS solver with:
    - Trust region strategy for robust convergence
    - Dogleg-based descent method
    - Custom convergence criteria
    - Fallback mechanisms for numerical stability

    Mathematical Context:
        Used primarily for optimizing log-volatilities h in the BIF update step,
        where the objective function can be non-convex and numerically sensitive.

    Implementation Notes:
        - Uses classical trust region over line search
        - DoglegDescent handles potential indefinite Hessians
        - Default tolerances tuned for state estimation context
        - Maintains JAX-compatibility for JIT compilation
    """
    rtol:float
    atol:float
    norm: Callable[[PyTree], Scalar]
    use_inverse:bool
    descent:optx.AbstractDescent=optx.DoglegDescent()
    search:optx.AbstractSearch=optx.ClassicalTrustRegion()
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
        # self.search = optx.BacktrackingArmijo(decrease_factor=0.1,step_init=0.1)
        self.search=optx.ClassicalTrustRegion()
        self.verbose = verbose