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



@eqx.filter_jit
def bellman_objective(
    params: DFSVParamsDataclass,
    y: jnp.ndarray,
    filter: DFSVBellmanFilter,
    # Prior Hyperparameters (passed through to log_prior_density)
    prior_mu_mean: float | jnp.ndarray = 0.0,
    prior_mu_var: float | jnp.ndarray = 1.0,
    prior_sigma2_alpha: float = 3.0,
    prior_sigma2_beta: float = 0.1,
    prior_q_h_alpha: float = 3.0,
    prior_q_h_beta: float = 0.05,
    prior_lambda_mean: float = 0.0,
    prior_lambda_var: float = 1.0,
    prior_phi_f_mean: float = 0.0,
    prior_phi_f_diag_mean: float = 0.9,
    prior_phi_f_var: float = 0.5,
    prior_phi_h_mean: float = 0.0,
    prior_phi_h_diag_mean: float = 0.95,
    prior_phi_h_var: float = 0.1
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
    **kwargs :
        Prior hyperparameters passed directly to `log_prior_density`. See the
        docstring of `log_prior_density` for details. Floats/ints are static.

    Returns
    -------
    float
        Negative log-likelihood minus log-prior density.
    """
    # Original negative log-likelihood from the filter
    jit_ll_func = filter.jit_log_likelihood_of_params()
    neg_ll = -jit_ll_func(params, y)
    safe_neg_ll = jnp.nan_to_num(neg_ll, nan=1e10, posinf=1e10, neginf=1e10)

    # Calculate the log-prior density
    log_prior = log_prior_density(
        params,
        prior_mu_mean=prior_mu_mean,
        prior_mu_var=prior_mu_var,
        prior_sigma2_alpha=prior_sigma2_alpha,
        prior_sigma2_beta=prior_sigma2_beta,
        prior_q_h_alpha=prior_q_h_alpha,
        prior_q_h_beta=prior_q_h_beta,
        prior_lambda_mean=prior_lambda_mean,
        prior_lambda_var=prior_lambda_var,
        prior_phi_f_mean=prior_phi_f_mean,
        prior_phi_f_diag_mean=prior_phi_f_diag_mean,
        prior_phi_f_var=prior_phi_f_var,
        prior_phi_h_mean=prior_phi_h_mean,
        prior_phi_h_diag_mean=prior_phi_h_diag_mean,
        prior_phi_h_var=prior_phi_h_var
    )

    # Total objective: Negative Log Likelihood - Log Prior
    # We subtract the log prior because we want to maximize posterior = likelihood * prior
    # Maximizing log(posterior) = log(likelihood) + log(prior)
    # Minimizing -log(posterior) = -log(likelihood) - log(prior) = neg_ll - log_prior
    total_objective = safe_neg_ll - log_prior
    return total_objective


@eqx.filter_jit
def transformed_bellman_objective(
    transformed_params: DFSVParamsDataclass,
    y: jnp.ndarray,
    filter: DFSVBellmanFilter,
    # Prior Hyperparameters (passed through to bellman_objective -> log_prior_density)
    prior_mu_mean: float | jnp.ndarray = 0.0,
    prior_mu_var: float | jnp.ndarray = 1.0,
    prior_sigma2_alpha: float = 3.0,
    prior_sigma2_beta: float = 0.1,
    prior_q_h_alpha: float = 3.0,
    prior_q_h_beta: float = 0.05,
    prior_lambda_mean: float = 0.0,
    prior_lambda_var: float = 1.0,
    prior_phi_f_mean: float = 0.0,
    prior_phi_f_diag_mean: float = 0.9,
    prior_phi_f_var: float = 0.5,
    prior_phi_h_mean: float = 0.0,
    prior_phi_h_diag_mean: float = 0.95,
    prior_phi_h_var: float = 0.1
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
    **kwargs :
        Prior hyperparameters passed directly to `bellman_objective`. See the
        docstring of `log_prior_density` for details. Floats/ints are static.

    Returns
    -------
    float
        Negative log-likelihood minus log-prior density.
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

    # Run the bellman objective with original parameters and pass prior hyperparameters
    return bellman_objective(
        original_params,
        y,
        filter,
        prior_mu_mean=prior_mu_mean,
        prior_mu_var=prior_mu_var,
        prior_sigma2_alpha=prior_sigma2_alpha,
        prior_sigma2_beta=prior_sigma2_beta,
        prior_q_h_alpha=prior_q_h_alpha,
        prior_q_h_beta=prior_q_h_beta,
        prior_lambda_mean=prior_lambda_mean,
        prior_lambda_var=prior_lambda_var,
        prior_phi_f_mean=prior_phi_f_mean,
        prior_phi_f_diag_mean=prior_phi_f_diag_mean,
        prior_phi_f_var=prior_phi_f_var,
        prior_phi_h_mean=prior_phi_h_mean,
        prior_phi_h_diag_mean=prior_phi_h_diag_mean,
        prior_phi_h_var=prior_phi_h_var
    )


# -------------------------------------------------------------------------
# Particle Filter Objective Functions
# -------------------------------------------------------------------------

def pf_objective(
    params: DFSVParamsDataclass,
    observations: jnp.ndarray,
    filter_instance: DFSVParticleFilter, # Use imported class
    # Prior Hyperparameters (passed through to log_prior_density)
    prior_mu_mean: float | jnp.ndarray = 0.0,
    prior_mu_var: float | jnp.ndarray = 1.0,
    prior_sigma2_alpha: float = 3.0,
    prior_sigma2_beta: float = 0.1,
    prior_q_h_alpha: float = 3.0,
    prior_q_h_beta: float = 0.05,
    prior_lambda_mean: float = 0.0,
    prior_lambda_var: float = 1.0,
    prior_phi_f_mean: float = 0.0,
    prior_phi_f_diag_mean: float = 0.9,
    prior_phi_f_var: float = 0.5,
    prior_phi_h_mean: float = 0.0,
    prior_phi_h_diag_mean: float = 0.95,
    prior_phi_h_var: float = 0.1
) -> float:
    """
    Objective function for standard parameter space using Particle Filter.

    Calculates the negative log-likelihood based on the particle filter's
    estimation, minus the log-prior density.

    Args:
        params: Model parameters as a DFSVParamsDataclass Pytree.
        observations: Observation data.
        filter_instance: An instance of DFSVParticleFilter.
        **kwargs :
            Prior hyperparameters passed directly to `log_prior_density`. See the
            docstring of `log_prior_density` for details.

    Returns:
        float: Negative log-likelihood minus log-prior density.
    """
    # Calculate log-likelihood using the particle filter method
    log_lik = filter_instance.log_likelihood_of_params(params, observations)
    safe_neg_ll = jnp.nan_to_num(-log_lik, nan=1e10, posinf=1e10, neginf=1e10) # Add safety

    # Calculate the log-prior density
    log_prior = log_prior_density(
        params,
        prior_mu_mean=prior_mu_mean,
        prior_mu_var=prior_mu_var,
        prior_sigma2_alpha=prior_sigma2_alpha,
        prior_sigma2_beta=prior_sigma2_beta,
        prior_q_h_alpha=prior_q_h_alpha,
        prior_q_h_beta=prior_q_h_beta,
        prior_lambda_mean=prior_lambda_mean,
        prior_lambda_var=prior_lambda_var,
        prior_phi_f_mean=prior_phi_f_mean,
        prior_phi_f_diag_mean=prior_phi_f_diag_mean,
        prior_phi_f_var=prior_phi_f_var,
        prior_phi_h_mean=prior_phi_h_mean,
        prior_phi_h_diag_mean=prior_phi_h_diag_mean,
        prior_phi_h_var=prior_phi_h_var
    )

    # Return negative log-likelihood minus log prior
    return safe_neg_ll - log_prior

def transformed_pf_objective(
    transformed_params: DFSVParamsDataclass,
    observations: jnp.ndarray,
    filter_instance: DFSVParticleFilter,
    # Prior Hyperparameters (passed through to pf_objective -> log_prior_density)
    prior_mu_mean: float | jnp.ndarray = 0.0,
    prior_mu_var: float | jnp.ndarray = 1.0,
    prior_sigma2_alpha: float = 3.0,
    prior_sigma2_beta: float = 0.1,
    prior_q_h_alpha: float = 3.0,
    prior_q_h_beta: float = 0.05,
    prior_lambda_mean: float = 0.0,
    prior_lambda_var: float = 1.0,
    prior_phi_f_mean: float = 0.0,
    prior_phi_f_diag_mean: float = 0.9,
    prior_phi_f_var: float = 0.5,
    prior_phi_h_mean: float = 0.0,
    prior_phi_h_diag_mean: float = 0.95,
    prior_phi_h_var: float = 0.1
) -> float:
    """
    Objective function for transformed parameter space using Particle Filter.

    Untransforms parameters, then calls `pf_objective` which computes
    negative log-likelihood minus log-prior density.

    Args:
        transformed_params: Model parameters in transformed space.
        observations: Observation data.
        filter_instance: An instance of DFSVParticleFilter.
        **kwargs :
            Prior hyperparameters passed directly to `pf_objective`. See the
            docstring of `log_prior_density` for details.

    Returns:
        float: Negative log-likelihood minus log-prior density.
    """
    # Untransform parameters
    params_original = untransform_params(transformed_params)

    # Call the standard pf_objective with original parameters and pass prior hyperparameters
    return pf_objective(
        params_original,
        observations,
        filter_instance,
        prior_mu_mean=prior_mu_mean,
        prior_mu_var=prior_mu_var,
        prior_sigma2_alpha=prior_sigma2_alpha,
        prior_sigma2_beta=prior_sigma2_beta,
        prior_q_h_alpha=prior_q_h_alpha,
        prior_q_h_beta=prior_q_h_beta,
        prior_lambda_mean=prior_lambda_mean,
        prior_lambda_var=prior_lambda_var,
        prior_phi_f_mean=prior_phi_f_mean,
        prior_phi_f_diag_mean=prior_phi_f_diag_mean,
        prior_phi_f_var=prior_phi_f_var,
        prior_phi_h_mean=prior_phi_h_mean,
        prior_phi_h_diag_mean=prior_phi_h_diag_mean,
        prior_phi_h_var=prior_phi_h_var
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
    # Mu Prior Params (Normal)
    prior_mu_mean: float | jnp.ndarray = 0.0,
    prior_mu_var: float | jnp.ndarray = 1.0,
    # Sigma2 Prior Params (Inverse Gamma)
    prior_sigma2_alpha: float = 3.0,
    prior_sigma2_beta: float = 0.1,
    # Q_h Prior Params (Inverse Gamma for diagonal elements)
    prior_q_h_alpha: float = 3.0,
    prior_q_h_beta: float = 0.05,
    # Lambda_r Prior Params (Normal)
    prior_lambda_mean: float = 0.0,
    prior_lambda_var: float = 1.0,
    # Phi_f Prior Params (Normal on elements - interim)
    prior_phi_f_mean: float = 0.0, # Mean for off-diagonals
    prior_phi_f_diag_mean: float = 0.9, # Mean for diagonals
    prior_phi_f_var: float = 0.5,
    # Phi_h Prior Params (Normal on elements - interim)
    prior_phi_h_mean: float = 0.0, # Mean for off-diagonals
    prior_phi_h_diag_mean: float = 0.95, # Mean for diagonals
    prior_phi_h_var: float = 0.1
) -> jnp.ndarray:
    """
    Computes the total log-prior probability density for the model parameters.

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

    Returns:
        The total log-prior density (JAX scalar).
    """
    total_log_prior = jnp.array(0.0, dtype=jnp.result_type(float)) # Initialize with float dtype

    # --- Prior for mu (Long-run mean of log-volatility) ---
    # Normal prior for each element
    log_prior_mu = jnp.sum(safe_norm_logpdf(params.mu.flatten(), prior_mu_mean, jnp.sqrt(prior_mu_var)))
    total_log_prior += log_prior_mu

    # --- Prior for sigma2 (Idiosyncratic Variances) ---
    # Assuming independent Inverse-Gamma priors for each element
    sigma2_1d = params.sigma2.flatten()
    log_prior_sigma2 = jnp.sum(
        jax.vmap(safe_inverse_gamma_logpdf, in_axes=(0, None, None))(sigma2_1d, prior_sigma2_alpha, prior_sigma2_beta)
    )
    total_log_prior += log_prior_sigma2

    # --- Prior for Q_h (Volatility of log-volatility) ---
    # Assuming diagonal Q_h and independent Inverse-Gamma priors for diagonal elements
    q_h_diag = jnp.diag(params.Q_h)
    log_prior_q_h = jnp.sum(
        jax.vmap(safe_inverse_gamma_logpdf, in_axes=(0, None, None))(q_h_diag, prior_q_h_alpha, prior_q_h_beta)
    )
    total_log_prior += log_prior_q_h

    # --- Prior for lambda_r (Factor Loadings) ---
    # Normal prior for each element
    log_prior_lambda = jnp.sum(safe_norm_logpdf(params.lambda_r.flatten(), prior_lambda_mean, jnp.sqrt(prior_lambda_var)))
    total_log_prior += log_prior_lambda

    # --- Prior for Phi_f (Factor Transition Matrix) ---
    # TODO: Interim prior - does not enforce stationarity. Revise later.
    K = params.K
    diag_indices = jnp.diag_indices(K)
    # Construct prior mean matrix M_f
    M_f = jnp.full_like(params.Phi_f, prior_phi_f_mean)
    M_f = M_f.at[diag_indices].set(prior_phi_f_diag_mean)
    # Calculate log PDF
    log_prior_phi_f = jnp.sum(safe_norm_logpdf(params.Phi_f, M_f, jnp.sqrt(prior_phi_f_var)))
    total_log_prior += log_prior_phi_f

    # --- Prior for Phi_h (Log-Volatility Transition Matrix) ---
    # TODO: Interim prior - does not enforce stationarity. Revise later.
    # Construct prior mean matrix M_h
    M_h = jnp.full_like(params.Phi_h, prior_phi_h_mean)
    M_h = M_h.at[diag_indices].set(prior_phi_h_diag_mean)
    # Calculate log PDF
    log_prior_phi_h = jnp.sum(safe_norm_logpdf(params.Phi_h, M_h, jnp.sqrt(prior_phi_h_var)))
    total_log_prior += log_prior_phi_h

    # Return total log prior (JAX scalar)
    # Replace NaN/Inf with large negative number for stability
    return jnp.where(jnp.isnan(total_log_prior) | jnp.isinf(total_log_prior), -1e10, total_log_prior)

