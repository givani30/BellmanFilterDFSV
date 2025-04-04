"""
Likelihood functions for Dynamic Factor Stochastic Volatility models.

This module provides functions to compute the log-likelihood for DFSV models,
with special attention to proper expansion of expressions containing logarithms
of exponential terms. Also includes objective functions for optimization.
"""
import jax # <-- Add import
from jax.scipy.special import gammaln
import equinox as eqx
import numpy as np
import jax.numpy as jnp
from jax import lax
from functools import partial
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass # Removed DFSV_params
from bellman_filter_dfsv.core.filters.bellman import DFSVBellmanFilter
from bellman_filter_dfsv.utils.transformations import untransform_params
from bellman_filter_dfsv.core.filters.particle import DFSVParticleFilter # Added for type hints
# Assuming safe versions exist based on prior diff attempt
from bellman_filter_dfsv.utils.jax_helpers import safe_norm_logpdf, safe_inverse_gamma_logpdf


def log_likelihood_observation(y_t: np.ndarray, f_t: np.ndarray,
                              params: DFSVParamsDataclass) -> float:
    """
    Compute the log-likelihood of an observation given the factors.

    This function expands any logarithm of exponential terms properly.

    Parameters
    ----------
    y_t : np.ndarray
        Observation vector at time t with shape (N, 1)
    f_t : np.ndarray
        Factor vector at time t with shape (K, 1)
    params : DFSV_params
        Model parameters

    Returns
    -------
    float
        Log-likelihood value
    """
    # Extract parameters
    lambda_r = params.lambda_r
    sigma2 = params.sigma2
    N = params.N

    # Mean and residual
    mean_t = lambda_r @ f_t
    residual = y_t - mean_t

    # Log-likelihood without any exp inside log
    ll = -0.5 * (
        N * np.log(2 * np.pi) +
        np.log(np.linalg.det(sigma2)) +
        residual.T @ np.linalg.inv(sigma2) @ residual
    )[0, 0]

    return ll


def log_likelihood_factor_transition(f_t: np.ndarray, f_prev: np.ndarray,
                                    h_t: np.ndarray, params: DFSVParamsDataclass) -> float:
    """
    Compute the log-likelihood of factor transition with stochastic volatility.

    This function properly expands the log(exp) terms that appear in stochastic
    volatility models.

    Parameters
    ----------
    f_t : np.ndarray
        Current factor vector with shape (K, 1)
    f_prev : np.ndarray
        Previous factor vector with shape (K, 1)
    h_t : np.ndarray
        Current log-volatility vector with shape (K, 1)
    params : DFSV_params
        Model parameters

    Returns
    -------
    float
        Log-likelihood value
    """
    K = params.K
    Phi_f = params.Phi_f

    # Mean
    mean_f = Phi_f @ f_prev

    # Variance matrix with stochastic volatility
    # Instead of using exp(h_t) directly in a matrix and then taking log(det),
    # we expand the expression

    # The original expression would involve:
    # Sigma = diag(exp(h_t))
    # log(det(Sigma)) = log(prod(exp(h_t))) = sum(h_t)

    # The expanded form eliminates the log(exp) pattern
    sum_log_var = np.sum(h_t)

    # Standardized residuals
    # Instead of Sigma^(-1/2)*(f_t - mean_f) we use exp(-h_t/2)*(f_t - mean_f)
    std_residuals = (f_t - mean_f) * np.exp(-h_t/2)
    quad_form = np.sum(std_residuals**2)

    # Log-likelihood with expanded terms
    ll = -0.5 * (
        K * np.log(2 * np.pi) +
        sum_log_var +  # This replaces log(det(Sigma))
        quad_form      # This replaces the quadratic form with matrix inversion
    )

    return ll[0] if hasattr(ll, '__len__') else ll


def log_likelihood_volatility_transition(h_t: np.ndarray, h_prev: np.ndarray,
                                       params: DFSVParamsDataclass) -> float:
    """
    Compute the log-likelihood of log-volatility transition.

    Parameters
    ----------
    h_t : np.ndarray
        Current log-volatility vector with shape (K, 1)
    h_prev : np.ndarray
        Previous log-volatility vector with shape (K, 1)
    params : DFSV_params
        Model parameters

    Returns
    -------
    float
        Log-likelihood value
    """
    K = params.K
    Phi_h = params.Phi_h
    mu = params.mu.reshape(-1, 1) if params.mu.ndim == 1 else params.mu
    Q_h = params.Q_h

    # Mean
    mean_h = mu + Phi_h @ (h_prev - mu)

    # Residual
    residual = h_t - mean_h

    # Log-likelihood (standard form, no exp inside log here)
    ll = -0.5 * (
        K * np.log(2 * np.pi) +
        np.log(np.linalg.det(Q_h)) +
        residual.T @ np.linalg.inv(Q_h) @ residual
    )[0, 0]

    return ll


def compute_joint_log_likelihood(y: np.ndarray, f: np.ndarray, h: np.ndarray,
                              params: DFSVParamsDataclass) -> float:
    """
    Compute the joint log-likelihood of observations, factors, and log-volatilities.

    Parameters
    ----------
    y : np.ndarray
        Observations with shape (N, T)
    f : np.ndarray
        Factors with shape (K, T)
    h : np.ndarray
        Log-volatilities with shape (K, T)
    params : DFSV_params
        Model parameters

    Returns
    -------
    float
        Total log-likelihood value
    """
    T = y.shape[1]

    # Initialize log-likelihood
    ll_total = 0.0

    # Compute log-likelihood for each time step
    for t in range(T):
        # For t=0, we only have the observation likelihood
        if t == 0:
            ll_total += log_likelihood_observation(y[:, t:t+1], f[:, t:t+1], params)
        else:
            # Observation likelihood
            ll_total += log_likelihood_observation(y[:, t:t+1], f[:, t:t+1], params)

            # Factor transition likelihood
            ll_total += log_likelihood_factor_transition(
                f[:, t:t+1], f[:, t-1:t], h[:, t:t+1], params
            )

            # Log-volatility transition likelihood
            ll_total += log_likelihood_volatility_transition(
                h[:, t:t+1], h[:, t-1:t], params
            )

    return ll_total



# Note: Using eqx.filter_jit which should handle non-JAX types like dicts as static by default.
# If JIT errors persist with priors, add static_argnames=['priors'] to the decorator.
@eqx.filter_jit
def bellman_objective(
    params: DFSVParamsDataclass,
    y: jnp.ndarray,
    filter: DFSVBellmanFilter,
    priors: dict | None = None
) -> float:
    """
    Compute the negative log-likelihood using the Bellman filter plus log-prior density.

    Parameters
    ----------
    params : DFSVParamsDataclass
        Model parameters in the original constrained space.
    y : jnp.ndarray
        Observed data.
    filter : DFSVBellmanFilter
        Filter object.
    priors : dict | None, optional
        A dictionary containing prior hyperparameters. Keys should match the
        arguments of `log_prior_density`. If None, prior density is not added.
        Defaults to None.

    Returns
    -------
    float
        Negative log-likelihood, potentially minus log-prior density.
    """
    # Original negative log-likelihood from the filter
    # Note: The filter's likelihood method does NOT receive priors.
    jit_ll_func = filter.jit_log_likelihood_of_params()
    log_lik = jit_ll_func(params, y)
    safe_neg_ll = jnp.nan_to_num(-log_lik, nan=1e10, posinf=1e10, neginf=1e10)

    # Calculate the log-prior density if priors are provided
    log_prior = 0.0
    if priors is not None:
        # Ensure priors is treated as static for JIT compilation if it contains arrays
        # However, eqx.filter_jit usually handles dicts passed as args correctly.
        log_prior = log_prior_density(params, **priors)
        log_prior = jnp.nan_to_num(log_prior, nan=-1e10, posinf=-1e10, neginf=-1e10) # Safety check

    # Total objective: Negative Log Likelihood - Log Prior
    # We subtract the log prior because we want to maximize posterior = likelihood * prior
    # Maximizing log(posterior) = log(likelihood) + log(prior)
    # Minimizing -log(posterior) = -log(likelihood) - log(prior) = neg_ll - log_prior
    total_objective = safe_neg_ll - log_prior
    return total_objective


# Note: Using eqx.filter_jit which should handle non-JAX types like dicts as static by default.
# If JIT errors persist with priors, add static_argnames=['priors'] to the decorator.
@eqx.filter_jit
def transformed_bellman_objective(
    transformed_params: DFSVParamsDataclass,
    y: jnp.ndarray,
    filter: DFSVBellmanFilter,
    priors: dict | None = None
) -> float:
    """
    Compute the objective function with transformed parameters.

    Untransforms parameters, then calls `bellman_objective` which computes
    negative log-likelihood minus log-prior density.

    Parameters
    ----------
    transformed_params : DFSVParamsDataclass
        Model parameters in transformed (unconstrained) space.
    y : jnp.ndarray
        Observed data.
    filter : DFSVBellmanFilter
        Filter object.
    priors : dict | None, optional
        A dictionary containing prior hyperparameters, passed to `bellman_objective`.
        If None, prior density is not added. Defaults to None.

    Returns
    -------
    float
        Negative log-likelihood, potentially minus log-prior density.
    """
    # Transform parameters back to original space
    original_params = untransform_params(transformed_params)
    # --- Add Debug Print for Untransformed Params ---
    # jax.debug.print("objective: untransformed params: {p}", p=original_params)
    # Check finiteness of all leaves in the original_params pytree
    # leaves = jax.tree_util.tree_leaves(original_params)
    # is_params_finite = jnp.all(jnp.array([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))
    # jax.debug.print("objective: untransformed params finite: {finite}", finite=is_params_finite)
    # --- End Debug Print ---

    # Run the bellman objective with original parameters and pass the priors dictionary
    return bellman_objective(
        original_params,
        y,
        filter,
        priors=priors # Pass the dictionary directly
    )


# -------------------------------------------------------------------------
# Particle Filter Objective Functions
# -------------------------------------------------------------------------

# Note: This function is NOT JITted by default, but calls a potentially JITted
# filter method. Priors are handled similarly to bellman_objective.
def pf_objective(
    params: DFSVParamsDataclass,
    observations: jnp.ndarray,
    filter_instance: DFSVParticleFilter, # Use imported class
    priors: dict | None = None
) -> float:
    """
    Objective function for standard parameter space using Particle Filter.

    Calculates the negative log-likelihood based on the particle filter's
    estimation, potentially minus the log-prior density.

    Args:
        params: Model parameters as a DFSVParamsDataclass Pytree.
        observations: Observation data.
        filter_instance: An instance of DFSVParticleFilter.
        priors : dict | None, optional
            A dictionary containing prior hyperparameters. Keys should match the
            arguments of `log_prior_density`. If None, prior density is not added.
            Defaults to None.

    Returns:
        float: Negative log-likelihood, potentially minus log-prior density.
    """
    # Calculate log-likelihood using the particle filter method
    # Note: The filter's likelihood method does NOT receive priors.
    log_lik = filter_instance.log_likelihood_of_params(params, observations)
    safe_neg_ll = jnp.nan_to_num(-log_lik, nan=1e10, posinf=1e10, neginf=1e10) # Add safety

    # Calculate the log-prior density if priors are provided
    log_prior = 0.0
    if priors is not None:
        log_prior = log_prior_density(params, **priors)
        log_prior = jnp.nan_to_num(log_prior, nan=-1e10, posinf=-1e10, neginf=-1e10) # Safety check

    # Return negative log-likelihood minus log prior
    return safe_neg_ll - log_prior

# Note: This function is NOT JITted by default.
def transformed_pf_objective(
    transformed_params: DFSVParamsDataclass,
    observations: jnp.ndarray,
    filter_instance: DFSVParticleFilter,
    priors: dict | None = None
) -> float:
    """
    Objective function for transformed parameter space using Particle Filter.

    Untransforms parameters, then calls `pf_objective` which computes
    negative log-likelihood, potentially minus log-prior density.

    Args:
        transformed_params: Model parameters in transformed space.
        observations: Observation data.
        filter_instance: An instance of DFSVParticleFilter.
        priors : dict | None, optional
            A dictionary containing prior hyperparameters, passed to `pf_objective`.
            If None, prior density is not added. Defaults to None.

    Returns:
        float: Negative log-likelihood, potentially minus log-prior density.
    """
    # Untransform parameters
    params_original = untransform_params(transformed_params)

    # Call the standard pf_objective with original parameters and pass the priors dictionary
    return pf_objective(
        params_original,
        observations,
        filter_instance,
        priors=priors # Pass the dictionary directly
    )


# -------------------------------------------------------------------------
# Prior Density Functions
# -------------------------------------------------------------------------

# Note: inverse_gamma_log_pdf is replaced by safe_inverse_gamma_logpdf from utils
# def inverse_gamma_log_pdf(x: jnp.ndarray, alpha: float, beta: float) -> jnp.ndarray:
#     """Computes the log probability density function of the Inverse-Gamma distribution."""
#     x_safe = jnp.maximum(x, 1e-10) # Small epsilon for stability
#     log_pdf = alpha * jnp.log(beta) - gammaln(alpha) - (alpha + 1) * jnp.log(x_safe) - beta / x_safe
#     return log_pdf

def log_prior_density(
    params: DFSVParamsDataclass,
    # Default Prior Hyperparameters (can be overridden by passing a dictionary)
    prior_mu_mean: float | jnp.ndarray = 0.0,
    prior_mu_var: float | jnp.ndarray = 1.0,
    prior_sigma2_alpha: float = 3.0,
    prior_sigma2_beta: float = 0.1,
    prior_q_h_alpha: float = 3.0,
    prior_q_h_beta: float = 0.05,
    prior_lambda_mean: float = 0.0,
    prior_lambda_var: float = 1.0,
    prior_phi_f_mean: float = 0.0, # Mean for off-diagonals
    prior_phi_f_diag_mean: float = 0.9, # Mean for diagonals
    prior_phi_f_var: float = 0.5,
    prior_phi_h_mean: float = 0.0, # Mean for off-diagonals
    prior_phi_h_diag_mean: float = 0.95, # Mean for diagonals
    prior_phi_h_var: float = 0.1,
    **kwargs # Allow accepting a dictionary via **priors
) -> jnp.ndarray:
    """
    Computes the total log-prior probability density for the model parameters.

    Uses default hyperparameters unless overridden by keyword arguments (e.g.,
    passed via a dictionary `**priors` from the calling objective function).

    Implements priors for:
    - mu: Normal(prior_mu_mean, prior_mu_var) for each element.
    - sigma2: Inverse-Gamma(prior_sigma2_alpha, prior_sigma2_beta) for each element.
    - Q_h: Inverse-Gamma(prior_q_h_alpha, prior_q_h_beta) for diagonal elements.
           Assumes Q_h is diagonal.
    - lambda_r: Normal(prior_lambda_mean, prior_lambda_var) for each element.
    - Phi_f: Normal on elements with different means for diagonal
             (prior_phi_f_diag_mean) and off-diagonal (prior_phi_f_mean) elements,
             and common variance (prior_phi_f_var).
             TODO: This is an interim prior and does NOT enforce stationarity.
                   Revise when matrix transformations are implemented.
    - Phi_h: Normal on elements similar to Phi_f, using prior_phi_h_* params.
             TODO: This is an interim prior and does NOT enforce stationarity.
                   Revise when matrix transformations are implemented.

    Args:
        params: The model parameters (DFSVParamsDataclass).
        prior_mu_mean: Mean for the Normal prior on mu elements.
        prior_mu_var: Variance for the Normal prior on mu elements.
        prior_sigma2_alpha: Shape parameter for the Inverse-Gamma prior on sigma2 elements.
        prior_sigma2_beta: Scale parameter for the Inverse-Gamma prior on sigma2 elements.
        prior_q_h_alpha: Shape parameter for the Inverse-Gamma prior on Q_h diagonal elements.
        prior_q_h_beta: Scale parameter for the Inverse-Gamma prior on Q_h diagonal elements.
        prior_lambda_mean: Mean for the Normal prior on lambda_r elements.
        prior_lambda_var: Variance for the Normal prior on lambda_r elements.
        prior_phi_f_mean: Mean for the Normal prior on Phi_f off-diagonal elements.
        prior_phi_f_diag_mean: Mean for the Normal prior on Phi_f diagonal elements.
        prior_phi_f_var: Variance for the Normal prior on Phi_f elements.
        prior_phi_h_mean: Mean for the Normal prior on Phi_h off-diagonal elements.
        prior_phi_h_diag_mean: Mean for the Normal prior on Phi_h diagonal elements.
        prior_phi_h_var: Variance for the Normal prior on Phi_h elements.
        **kwargs: Catches any unexpected prior arguments if passed via dictionary.

    Returns:
        The total log-prior density (JAX scalar).
    """
    # Use provided hyperparameters, falling back to defaults
    mu_mean = kwargs.get('prior_mu_mean', prior_mu_mean)
    mu_var = kwargs.get('prior_mu_var', prior_mu_var)
    sigma2_alpha = kwargs.get('prior_sigma2_alpha', prior_sigma2_alpha)
    sigma2_beta = kwargs.get('prior_sigma2_beta', prior_sigma2_beta)
    q_h_alpha = kwargs.get('prior_q_h_alpha', prior_q_h_alpha)
    q_h_beta = kwargs.get('prior_q_h_beta', prior_q_h_beta)
    lambda_mean = kwargs.get('prior_lambda_mean', prior_lambda_mean)
    lambda_var = kwargs.get('prior_lambda_var', prior_lambda_var)
    phi_f_mean = kwargs.get('prior_phi_f_mean', prior_phi_f_mean)
    phi_f_diag_mean = kwargs.get('prior_phi_f_diag_mean', prior_phi_f_diag_mean)
    phi_f_var = kwargs.get('prior_phi_f_var', prior_phi_f_var)
    phi_h_mean = kwargs.get('prior_phi_h_mean', prior_phi_h_mean)
    phi_h_diag_mean = kwargs.get('prior_phi_h_diag_mean', prior_phi_h_diag_mean)
    phi_h_var = kwargs.get('prior_phi_h_var', prior_phi_h_var)


    total_log_prior = jnp.array(0.0, dtype=jnp.result_type(float)) # Initialize with float dtype

    # --- Prior for mu (Long-run mean of log-volatility) ---
    # Normal prior for each element
    log_prior_mu = jnp.sum(safe_norm_logpdf(params.mu.flatten(), mu_mean, jnp.sqrt(mu_var)))
    total_log_prior += log_prior_mu

    # --- Prior for sigma2 (Idiosyncratic Variances) ---
    # Assuming independent Inverse-Gamma priors for each element
    sigma2_1d = params.sigma2.flatten()
    log_prior_sigma2 = jnp.sum(
        jax.vmap(safe_inverse_gamma_logpdf, in_axes=(0, None, None))(sigma2_1d, sigma2_alpha, sigma2_beta)
    )
    total_log_prior += log_prior_sigma2

    # --- Prior for Q_h (Volatility of log-volatility) ---
    # Assuming diagonal Q_h and independent Inverse-Gamma priors for diagonal elements
    q_h_diag = jnp.diag(params.Q_h)
    log_prior_q_h = jnp.sum(
        jax.vmap(safe_inverse_gamma_logpdf, in_axes=(0, None, None))(q_h_diag, q_h_alpha, q_h_beta)
    )
    total_log_prior += log_prior_q_h

    # --- Prior for lambda_r (Factor Loadings) ---
    # Normal prior for each element
    log_prior_lambda = jnp.sum(safe_norm_logpdf(params.lambda_r.flatten(), lambda_mean, jnp.sqrt(lambda_var)))
    total_log_prior += log_prior_lambda

    # --- Prior for Phi_f (Factor Transition Matrix) ---
    # TODO: Interim prior - does not enforce stationarity. Revise later.
    K = params.K
    diag_indices = jnp.diag_indices(K)
    # Construct prior mean matrix M_f
    M_f = jnp.full_like(params.Phi_f, phi_f_mean)
    M_f = M_f.at[diag_indices].set(phi_f_diag_mean)
    # Calculate log PDF
    log_prior_phi_f = jnp.sum(safe_norm_logpdf(params.Phi_f, M_f, jnp.sqrt(phi_f_var)))
    total_log_prior += log_prior_phi_f

    # --- Prior for Phi_h (Log-Volatility Transition Matrix) ---
    # TODO: Interim prior - does not enforce stationarity. Revise later.
    # Construct prior mean matrix M_h
    M_h = jnp.full_like(params.Phi_h, phi_h_mean)
    M_h = M_h.at[diag_indices].set(phi_h_diag_mean)
    # Calculate log PDF
    log_prior_phi_h = jnp.sum(safe_norm_logpdf(params.Phi_h, M_h, jnp.sqrt(phi_h_var)))
    total_log_prior += log_prior_phi_h

    # Return total log prior (JAX scalar)
    # Replace NaN/Inf with large negative number for stability
    return jnp.where(jnp.isnan(total_log_prior) | jnp.isinf(total_log_prior), -1e10, total_log_prior)

