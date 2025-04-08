"""
Objective functions for optimizing DFSV model parameters using different filters.
"""
import jax
import equinox as eqx
import jax.numpy as jnp
# No type imports needed
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.filters.bellman import DFSVBellmanFilter
from bellman_filter_dfsv.filters.particle import DFSVParticleFilter
from bellman_filter_dfsv.utils.transformations import untransform_params, apply_identification_constraint, EPS
from bellman_filter_dfsv.models.likelihoods import log_prior_density

def _compute_total_objective(
    params: DFSVParamsDataclass,
    y: jnp.ndarray,
    likelihood_fn,
    priors: dict | None,
    stability_penalty_weight: float
) -> float:
    """Shared helper: applies identification constraint, computes neg log likelihood, prior, penalty, fully JAX compatible."""
    # Enforce identification constraint
    params = apply_identification_constraint(params)

    # Compute log likelihood
    log_lik = likelihood_fn(params, y)
    safe_neg_ll = jnp.nan_to_num(-log_lik, nan=1e10, posinf=1e10, neginf=1e10)

    # Compute log prior or zero
    # Use a safer approach with jnp.where instead of jax.lax.cond for None check
    has_priors = jnp.array(priors is not None)

    # Only compute log_prior if priors is not None
    log_prior = jnp.where(
        has_priors,
        # This branch only executes if priors is not None
        jnp.nan_to_num(
            log_prior_density(params, **({} if priors is None else priors)),
            nan=-1e10, posinf=-1e10, neginf=-1e10
        ),
        # Default branch
        0.0
    )

    # Compute stability penalty or zero
    # Calculate penalty for all cases, but only use it if weight > 0
    mags_f = jnp.abs(jnp.linalg.eigvals(params.Phi_f))
    mags_h = jnp.abs(jnp.linalg.eigvals(params.Phi_h))
    penalty_f = jnp.sum(jax.nn.relu(mags_f - 1.0 + EPS))
    penalty_h = jnp.sum(jax.nn.relu(mags_h - 1.0 + EPS))
    penalty = penalty_f + penalty_h

    # Only apply penalty if weight > 0
    stability_penalty = jnp.where(
        stability_penalty_weight > 0,
        penalty,
        0.0
    )

    total = safe_neg_ll - log_prior + stability_penalty_weight * stability_penalty
    return total

@eqx.filter_jit
def bellman_objective(
    params: DFSVParamsDataclass,
    y: jnp.ndarray,
    filter: DFSVBellmanFilter,
    priors: dict | None = None,
    stability_penalty_weight: float = 0.0
) -> float:
    """Bellman filter objective, fully JAX compatible."""
    # Get the JITted function
    jit_ll_func = filter.jit_log_likelihood_wrt_params()
    # Define a function that matches the expected signature
    def ll_fn(p, y_):
        return jit_ll_func(p, y_)
    return _compute_total_objective(params, y, ll_fn, priors, stability_penalty_weight)

@eqx.filter_jit
def transformed_bellman_objective(
    transformed_params: DFSVParamsDataclass,
    y: jnp.ndarray,
    filter: DFSVBellmanFilter,
    priors: dict | None = None,
    stability_penalty_weight: float = 0.0
) -> float:
    """Bellman filter objective in transformed space, fully JAX compatible."""
    params = untransform_params(transformed_params)
    return bellman_objective(params, y, filter, priors, stability_penalty_weight)


# -------------------------------------------------------------------------
# Particle Filter Objective Functions
# -------------------------------------------------------------------------

@eqx.filter_jit
def pf_objective(
    params: DFSVParamsDataclass,
    observations: jnp.ndarray,
    filter_instance: DFSVParticleFilter,
    priors: dict | None = None,
    stability_penalty_weight: float = 0.0
) -> float:
    """
    Objective function for standard parameter space using Particle Filter, fully JAX compatible.

    Calculates the negative log-likelihood based on the particle filter's
    estimation, potentially minus the log-prior density, plus an optional
    stability penalty.

    Args:
        params: Model parameters as a DFSVParamsDataclass Pytree.
        observations: Observation data.
        filter_instance: An instance of DFSVParticleFilter.
        priors : dict | None, optional
            A dictionary containing prior hyperparameters. Keys should match the
            arguments of `log_prior_density`. If None, prior density is not added.
            Defaults to None.
        stability_penalty_weight : float, optional
            Weight for the stability penalty term applied to Phi_f and Phi_h.
            The penalty is calculated as sum of relu violations of eigenvalues > 1.
            Defaults to 0.0 (no penalty).

    Returns:
        float: Total objective value (negative log-likelihood - log-prior + stability penalty).
    """
    # Get the JITted function
    jit_ll_func = filter_instance.jit_log_likelihood_wrt_params()

    # Define a function that matches the expected signature for _compute_total_objective
    def ll_fn(p, y_):
        return jit_ll_func(p, y_)

    # Use the shared helper function for consistent implementation
    return _compute_total_objective(params, observations, ll_fn, priors, stability_penalty_weight)

@eqx.filter_jit
def transformed_pf_objective(
    transformed_params: DFSVParamsDataclass,
    observations: jnp.ndarray,
    filter_instance: DFSVParticleFilter,
    priors: dict | None = None,
    stability_penalty_weight: float = 0.0
) -> float:
    """
    Objective function for transformed parameter space using Particle Filter, fully JAX compatible.

    Untransforms parameters, then calls `pf_objective` which computes
    negative log-likelihood, potentially minus log-prior density, plus an
    optional stability penalty.

    Args:
        transformed_params: Model parameters in transformed space.
        observations: Observation data.
        filter_instance: An instance of DFSVParticleFilter.
        priors : dict | None, optional
            A dictionary containing prior hyperparameters, passed to `pf_objective`.
            If None, prior density is not added. Defaults to None.
        stability_penalty_weight : float, optional
            Weight for the stability penalty term, passed to `pf_objective`.
            Defaults to 0.0.

    Returns:
        float: Total objective value (negative log-likelihood - log-prior + stability penalty).
    """
    # Untransform parameters
    params_original = untransform_params(transformed_params)

    # Call the standard pf_objective with original parameters
    return pf_objective(
        params_original,
        observations,
        filter_instance,
        priors=priors,
        stability_penalty_weight=stability_penalty_weight
    )
