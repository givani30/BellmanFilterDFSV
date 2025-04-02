"""
Likelihood functions for Dynamic Factor Stochastic Volatility models.

This module provides functions to compute the log-likelihood for DFSV models,
with special attention to proper expansion of expressions containing logarithms
of exponential terms. Also includes objective functions for optimization.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass # Removed DFSV_params
from bellman_filter_dfsv.core.filters.bellman import DFSVBellmanFilter
from bellman_filter_dfsv.utils.transformations import untransform_params
from bellman_filter_dfsv.core.filters.particle import DFSVParticleFilter # Added for type hints


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


def kalman_filter_log_likelihood(y: np.ndarray, params: DFSVParamsDataclass) -> float:
    """
    Compute the marginal log-likelihood of observations using the Kalman filter.
    
    This function integrates out the latent states (factors and log-volatilities)
    to compute the marginal likelihood of the observations.
    
    Parameters
    ----------
    y : np.ndarray
        Observations with shape (N, T)
    params : DFSV_params
        Model parameters
        
    Returns
    -------
    float
        Marginal log-likelihood value
    """
    # Import here to avoid circular imports
    from bellman_filter_dfsv.core.filters.kalman import DFSVExtendedKalmanFilter # Assuming future location
    
    # Initialize Kalman filter
    kf = DFSVExtendedKalmanFilter(params)
    
    # Run filter and return log-likelihood
    _, _, log_likelihood = kf.filter(y)
    
    return log_likelihood


@partial(jit, static_argnames=["filter"])
def bellman_objective(params: DFSVParamsDataclass, y: jnp.ndarray, filter: DFSVBellmanFilter, 
                      prior_mean: float, prior_std_dev: float) -> float:
    """
    Compute the negative log-likelihood using the Bellman filter, potentially with priors.
    
    Parameters
    ----------
    params : DFSVParamsDataclass
        Model parameters
    y : jnp.ndarray
        Observed data
    filter : DFSVBellmanFilter
        Filter object
    prior_mean : float
        Mean for the prior distribution (e.g., on mu)
    prior_std_dev : float
        Standard deviation for the prior distribution (e.g., on mu)
        
    Returns
    -------
    float
        Negative log-likelihood value + prior penalty
    """
    # Original negative log-likelihood
    # Get the JITted function and call it with (params, y)
    jit_ll_func = filter.jit_log_likelihood_of_params()
    neg_ll = -jit_ll_func(params, y)
    safe_neg_ll = jnp.nan_to_num(neg_ll, nan=1e10, posinf=1e10, neginf=1e10)
    
    # Define functions for the conditional penalty calculation
    # TODO: extend regularization framework to handle larger models
    def calculate_penalty(operands):
        p, mean, std_dev = operands
        # Assuming K=1 case here based on the original logic
        current_mu = p.mu.reshape(()) # Reshape (1,) array to scalar
        return 0.5 * jnp.square((current_mu - mean) / std_dev)

    def no_penalty(operands):
        # Return 0.0 with the same dtype as the penalty calculation
        return jnp.array(0.0, dtype=safe_neg_ll.dtype)

    # Condition for applying the penalty
    apply_penalty_cond = (params.K == 1) & (prior_std_dev > 0)
    
    # Operands for the conditional functions
    penalty_operands = (params, prior_mean, prior_std_dev)

    # Use lax.cond for conditional execution compatible with jit
    mu_penalty = lax.cond(
        apply_penalty_cond,
        calculate_penalty, # Function to call if True
        no_penalty,        # Function to call if False
        penalty_operands   # Operands passed to the chosen function
    )

    # Add penalty to the negative log-likelihood
    total_objective = safe_neg_ll + mu_penalty
    return total_objective


@partial(jit, static_argnames=["filter"])
def transformed_bellman_objective(transformed_params: DFSVParamsDataclass, y: jnp.ndarray, 
                                  filter: DFSVBellmanFilter, prior_mean: float, 
                                  prior_std_dev: float) -> float:
    """
    Compute the objective function with transformed parameters.
    
    Parameters
    ----------
    transformed_params : DFSVParamsDataclass
        Model parameters in transformed (unconstrained) space
    y : jnp.ndarray
        Observed data
    filter : DFSVBellmanFilter
        Filter object
    prior_mean : float
        Mean for the prior distribution (e.g., on mu)
    prior_std_dev : float
        Standard deviation for the prior distribution (e.g., on mu)
        
    Returns
    -------
    float
        Negative log-likelihood value + prior penalty
    """
    # Transform parameters back to original space
    original_params = untransform_params(transformed_params)
    
    # Run the bellman filter with original parameters
    return bellman_objective(original_params, y, filter, prior_mean, prior_std_dev)


# -------------------------------------------------------------------------
# Particle Filter Objective Functions
# -------------------------------------------------------------------------

def pf_objective(
    params: DFSVParamsDataclass,
    observations: jnp.ndarray,
    filter_instance: DFSVParticleFilter, # Use imported class
    prior_mu_mean: float = -1.0,       # Prior mean for mu
    prior_mu_std_dev: float = 0.5      # Prior std dev for mu
) -> float:
    """
    Objective function for standard parameter space using Particle Filter.

    Calculates the negative log-likelihood based on the particle filter's
    estimation, plus a prior penalty on the long-run mean 'mu'.

    Args:
        params: Model parameters as a DFSVParamsDataclass Pytree.
        observations: Observation data.
        filter_instance: An instance of DFSVParticleFilter.
        prior_mu_mean: Mean of the Gaussian prior for mu.
        prior_mu_std_dev: Standard deviation of the Gaussian prior for mu.

    Returns:
        float: Negative log-likelihood + prior penalty.
    """
    # Calculate log-likelihood using the particle filter method
    log_lik = filter_instance.log_likelihood_of_params(params, observations)

    # Add prior penalty for mu (assuming Gaussian prior)
    # Ensure params.mu is 1D
    mu_flat = params.mu.flatten()
    prior_penalty_mu = jnp.sum(0.5 * ((mu_flat - prior_mu_mean) / prior_mu_std_dev)**2)

    # Return negative log-likelihood + penalty
    return -log_lik + prior_penalty_mu

def transformed_pf_objective(
    transformed_params: DFSVParamsDataclass,
    observations: jnp.ndarray,
    filter_instance: DFSVParticleFilter,
    prior_mu_mean: float = -1.0,
    prior_mu_std_dev: float = 0.5
) -> float:
    """
    Objective function for transformed parameter space using Particle Filter.

    Transforms parameters back to the original space, calculates the negative
    log-likelihood using the particle filter, and adds a prior penalty.

    Args:
        transformed_params: Model parameters in transformed space.
        observations: Observation data.
        filter_instance: An instance of DFSVParticleFilter.
        prior_mu_mean: Mean of the Gaussian prior for mu.
        prior_mu_std_dev: Standard deviation of the Gaussian prior for mu.

    Returns:
        float: Negative log-likelihood + prior penalty.
    """
    # Untransform parameters
    params_original = untransform_params(transformed_params)

    # Calculate log-likelihood using the particle filter method
    log_lik = filter_instance.log_likelihood_of_params(params_original, observations)

    # Add prior penalty for mu (on the original scale)
    # Ensure params_original.mu is 1D
    mu_flat = params_original.mu.flatten()
    prior_penalty_mu = jnp.sum(0.5 * ((mu_flat - prior_mu_mean) / prior_mu_std_dev)**2)

    # Return negative log-likelihood + penalty
    return -log_lik + prior_penalty_mu
