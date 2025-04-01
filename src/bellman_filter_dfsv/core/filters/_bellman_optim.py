# src/bellman_filter_dfsv/core/filters/_bellman_optim.py
"""
Standalone JAX functions for Bellman filter optimization steps.
"""
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optimistix as optx
import numpy as np # Add numpy import
import jax.scipy.linalg # Add linalg import
# Type hint for build_covariance function signature
BuildCovarianceFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
LogPosteriorFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], float]


@partial(jax.jit, static_argnames=("K", "build_covariance_fn", "log_posterior_fn"))
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
    """
    Negative log-likelihood + prior penalty wrt h (with f fixed).
    J(h) = -log p(y|f,h) - log p(f,h|pred)
         = -log p(y|f,h) + 0.5 * (alpha - alpha_pred)^T I_pred (alpha - alpha_pred) + const.

    Args:
        log_vols (jnp.ndarray): Log-volatilities (variable of optimization).
        factors (jnp.ndarray): Factor values (fixed for this step).
        lambda_r (jnp.ndarray): Factor loading matrix.
        sigma2 (jnp.ndarray): Idiosyncratic variances.
        predicted_state (jnp.ndarray): Predicted state vector [f_pred, h_pred].
        I_pred (jnp.ndarray): Predicted precision matrix.
        observation (jnp.ndarray): Observation vector.
        K (int): Number of factors (static).
        build_covariance_fn (Callable): Function to build covariance A(h).
        log_posterior_fn (Callable): Function to compute log p(y|f,h).

    Returns:
        float: Negative log posterior value for h.
    """
    # Ensure inputs are flattened
    log_vols = log_vols.flatten()
    factors = factors.flatten()
    predicted_state = predicted_state.flatten()

    # Build current state vector alpha = [f, h]
    alpha = jnp.concatenate([factors, log_vols])

    # 1. Compute negative log-likelihood part using the passed function
    # Note: log_posterior_fn computes log p(y|alpha), so we negate it.
    neg_log_lik = -log_posterior_fn(lambda_r, sigma2, alpha, observation)

    # 2. Compute prior penalty part (using only h components for decoupling test)
    #    0.5 * (h - h_pred)^T I_pred_hh (h - h_pred)
    h_pred = predicted_state[K:]
    h_diff = log_vols - h_pred
    I_pred_hh = I_pred[K:, K:]
    prior_penalty = 0.5 * jnp.dot(h_diff, jnp.dot(I_pred_hh, h_diff))

    # Total objective
    return neg_log_lik + prior_penalty


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
    inner_max_steps: int = 100, # Increased significantly
) -> Tuple[jnp.ndarray, bool]: # Updated return type hint
    """
    Minimize neg_log_post_h w.r.t. h using the provided BFGS solver.

    Args:
        h_init: Initial log-volatility values.
        factors: Factor values (fixed).
        lambda_r: Factor loading matrix.
        sigma2: Idiosyncratic variances.
        pred_state: Predicted state vector [f_pred, h_pred].
        I_pred: Predicted precision matrix.
        observation: Observation vector.
        K: Number of factors.
        build_covariance_fn: Function to build covariance A(h).
        log_posterior_fn: Function to compute log p(y|f,h).
        h_solver: Pre-configured Optimistix solver instance.
        inner_max_steps: Max steps for the inner BFGS optimization.

    Returns:
        Tuple[jnp.ndarray, bool]: Updated log-volatility values and a boolean indicating success.
    """
    # Objective function wrapper to match optimistix fn(y, args) signature
    # It captures necessary variables from the outer scope (passed as args).
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

    # Conditionally return the optimized value or the initial guess, plus success status
    return jax.lax.cond(
        pred=successful,
        true_fun=lambda: (sol.value, True),  # Use the optimized value, return True
        false_fun=lambda: (sol.value, False) # Return optimizer's final value even on failure, return False
    ) # Returns (updated_h, success_status)


@partial(jax.jit, static_argnames=("K", "build_covariance_fn"))
def obj_and_grad_fn(
    x: jnp.ndarray,
    # --- Fixed arguments ---
    lambda_r: jnp.ndarray,
    sigma2: jnp.ndarray,
    predicted_state: jnp.ndarray,
    I_pred: jnp.ndarray,
    observation: jnp.ndarray,
    # --- Static arguments & Dependencies ---
    K: int,
    build_covariance_fn: BuildCovarianceFn,
) -> Tuple[float, jnp.ndarray]:
    """
    Combined objective and gradient function for optimization (e.g., full state).

    Objective J(x) = -log p(y|x) + 0.5 * (x - x_pred)^T I_pred (x - x_pred) + const.

    Args:
        x: Current state estimate [f, h].
        lambda_r: Factor loading matrix.
        sigma2: Idiosyncratic variances.
        predicted_state: Predicted state [f_pred, h_pred].
        I_pred: Predicted precision matrix.
        observation: Observation vector.
        K: Number of factors.
        build_covariance_fn: Function to build covariance A(h).

    Returns:
        Tuple[float, jnp.ndarray]: Objective value and gradient w.r.t x.
    """
    N = lambda_r.shape[0]
    x = x.flatten()
    predicted_state = predicted_state.flatten()
    observation = observation.flatten()
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
    """
    Solve for factors given log-volatility is fixed.

    Args:
        log_volatility: Current log-volatility values.
        lambda_r: Factor loading matrix.
        sigma2: Idiosyncratic variances.
        observation: Observation vector.
        factors_pred: Predicted factor values.
        log_vols_pred: Predicted log-volatility values.
        I_f: Factor block of predicted precision matrix.
        I_fh: Factor-volatility block of predicted precision matrix.
        build_covariance_fn: Function to build covariance A(h).

    Returns:
        Updated factor values.
    """
    A = build_covariance_fn(lambda_r, jnp.exp(log_volatility), sigma2)
    L = jax.scipy.linalg.cho_factor(A, lower=True)

    def A_inv(x):
        return jax.scipy.linalg.cho_solve(L, x)

    lhs_mat = jnp.dot(lambda_r.T, A_inv(lambda_r)) + I_f + 1e-8 * jnp.eye(I_f.shape[0])
    rhs_vec = (
        jnp.dot(lambda_r.T, A_inv(observation))
        + jnp.dot(I_f, factors_pred)
        + jnp.dot(I_fh, (-log_vols_pred + log_volatility))
    )

    # Use pseudoinverse for potentially better stability, although solve should be fine
    # return jnp.linalg.pinv(lhs_mat) @ rhs_vec
    return jnp.linalg.solve(lhs_mat, rhs_vec)

# Removed duplicated code block from obj_and_grad_fn