"""
Objective functions for optimizing DFSV model parameters using different filters.
"""
import jax
import equinox as eqx
import jax.numpy as jnp
# Removed incorrect import: import jax.linalg
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.filters.bellman import DFSVBellmanFilter # Updated import path
from bellman_filter_dfsv.filters.particle import DFSVParticleFilter # Updated import path
from bellman_filter_dfsv.utils.transformations import untransform_params, EPS
from bellman_filter_dfsv.models.likelihoods import log_prior_density # Updated import path


# Note: Using eqx.filter_jit which should handle non-JAX types like dicts as static by default.
# If JIT errors persist with priors, add static_argnames=['priors'] to the decorator.
@eqx.filter_jit
def bellman_objective(
    params: DFSVParamsDataclass,
    y: jnp.ndarray,
    filter: DFSVBellmanFilter,
    priors: dict | None = None,
    stability_penalty_weight: float = 0.0
) -> float:
    """
    Compute the negative log-likelihood using the Bellman filter plus log-prior density,
    optionally adding a stability penalty.

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
    stability_penalty_weight : float, optional
        Weight for the stability penalty term applied to Phi_f and Phi_h.
        The penalty is calculated as relu(max(|eigval|) - 1 + EPS)**2.
        Defaults to 0.0 (no penalty).

    Returns
    -------
    float
        Negative log-likelihood, potentially minus log-prior density, plus stability penalty.
    """
    # Original negative log-likelihood from the filter
    # Note: The filter's likelihood method does NOT receive priors or penalty weights.
    jit_ll_func = filter.jit_log_likelihood_wrt_params() # Corrected method name
    log_lik = jit_ll_func(params, y)
    safe_neg_ll = jnp.nan_to_num(-log_lik, nan=1e10, posinf=1e10, neginf=1e10)

    # Calculate the log-prior density if priors are provided
    log_prior = 0.0
    if priors is not None:
        # Ensure priors is treated as static for JIT compilation if it contains arrays
        # However, eqx.filter_jit usually handles dicts passed as args correctly.
        log_prior = log_prior_density(params, **priors)
        log_prior = jnp.nan_to_num(log_prior, nan=-1e10, posinf=-1e10, neginf=-1e10) # Safety check

    # Calculate stability penalty if weight > 0
    stability_penalty = 0.0
    if stability_penalty_weight > 0:
        # Calculate max eigenvalue magnitudes for Phi_f and Phi_h using jnp.linalg
        max_mag_f = jnp.max(jnp.abs(jnp.linalg.eigvals(params.Phi_f)))
        max_mag_h = jnp.max(jnp.abs(jnp.linalg.eigvals(params.Phi_h)))

        # Calculate penalty using relu
        penalty_f = jax.nn.relu(max_mag_f - 1.0 + EPS)**2
        penalty_h = jax.nn.relu(max_mag_h - 1.0 + EPS)**2
        stability_penalty = penalty_f + penalty_h

    # Total objective: Negative Log Likelihood - Log Prior + Weighted Stability Penalty
    # We subtract the log prior because we want to maximize posterior = likelihood * prior
    # Maximizing log(posterior) = log(likelihood) + log(prior)
    # Minimizing -log(posterior) = -log(likelihood) - log(prior) = neg_ll - log_prior
    # We add the stability penalty as it's a cost we want to minimize.
    total_objective = safe_neg_ll - log_prior + stability_penalty_weight * stability_penalty
    return total_objective


# Note: Using eqx.filter_jit which should handle non-JAX types like dicts as static by default.
# If JIT errors persist with priors, add static_argnames=['priors'] to the decorator.
@eqx.filter_jit
def transformed_bellman_objective(
    transformed_params: DFSVParamsDataclass,
    y: jnp.ndarray,
    filter: DFSVBellmanFilter,
    priors: dict | None = None,
    stability_penalty_weight: float = 0.0
) -> float:
    """
    Compute the objective function with transformed parameters.

    Untransforms parameters, then calls `bellman_objective` which computes
    negative log-likelihood minus log-prior density, plus an optional stability penalty.

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
    stability_penalty_weight : float, optional
        Weight for the stability penalty term, passed to `bellman_objective`.
        Defaults to 0.0.

    Returns
    -------
    float
        Negative log-likelihood, potentially minus log-prior density, plus stability penalty.
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
    # and the stability penalty weight
    return bellman_objective(
        original_params,
        y,
        filter,
        priors=priors, # Pass the dictionary directly
        stability_penalty_weight=stability_penalty_weight # Pass the weight
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