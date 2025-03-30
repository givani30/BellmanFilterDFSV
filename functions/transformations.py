"""
Parameter transformation functions for DFSV models.

Maps constrained parameters (e.g., variances > 0, correlations in [-1, 1])
to unconstrained space for optimization, and back.
"""
import copy
import jax.numpy as jnp
from models.dfsv import DFSVParamsDataclass

# Epsilon for numerical stability near boundaries (e.g., 0 or 1) #TODO: exten
EPS = 1e-6

def transform_params(params: DFSVParamsDataclass) -> DFSVParamsDataclass:
    """
    Transform bounded parameters to unconstrained space for optimization.
    
    Parameters:
    -----------
    params : DFSVParamsDataclass
        Model parameters in their natural (constrained) space
        
    Returns:
    --------
    DFSVParamsDataclass
        Transformed parameters in unconstrained space
    """
    # Create a copy to avoid modifying the original
    result = copy.deepcopy(params)
    
    # Apply logit transform to persistence parameters (bounded in 0,1)
    # logit(p) = log(p/(1-p))
    
    # Transform Phi_f (factor persistence) - Assuming diagonal for now
    diag_phi_f = jnp.diag(params.Phi_f)
    phi_f_bounded = jnp.clip(diag_phi_f, EPS, 1 - EPS)
    transformed_diag_phi_f = jnp.log(phi_f_bounded / (1 - phi_f_bounded))
    transformed_phi_f = jnp.diag(transformed_diag_phi_f) # Keep diagonal structure
    
    # Transform Phi_h (log-volatility persistence) - Assuming diagonal
    diag_phi_h = jnp.diag(params.Phi_h)
    phi_h_bounded = jnp.clip(diag_phi_h, EPS, 1 - EPS)
    transformed_diag_phi_h = jnp.log(phi_h_bounded / (1 - phi_h_bounded))
    transformed_phi_h = jnp.diag(transformed_diag_phi_h) # Keep diagonal structure
    
    # Transform variance parameters (must be positive) using log transform
    # Only transform diagonal elements of sigma2 (force off-diagonals to zero)
    if params.sigma2.ndim > 1:  # Handle matrix case (already diagonal from validation)
        diag_indices = jnp.diag_indices_from(params.sigma2)
        transformed_sigma2 = jnp.zeros_like(params.sigma2)
        transformed_sigma2 = transformed_sigma2.at[diag_indices].set(
            jnp.log(jnp.maximum(jnp.diag(params.sigma2), EPS))
        )
    else: # Should not happen if validation runs, but handle just in case
         transformed_sigma2 = jnp.log(jnp.maximum(params.sigma2, EPS))

    # Transform Q_h (log-volatility noise covariance) - Assuming diagonal
    diag_q_h = jnp.diag(params.Q_h)
    transformed_diag_q_h = jnp.log(jnp.maximum(diag_q_h, EPS))
    transformed_q_h = jnp.diag(transformed_diag_q_h) # Keep diagonal structure

    # Note: mu and lambda_r are typically unconstrained, so no transformation needed here.
    
    # Return a new params object with transformed values
    return result.replace(
        Phi_f=transformed_phi_f,
        Phi_h=transformed_phi_h,
        sigma2=transformed_sigma2,
        Q_h=transformed_q_h
    )

def untransform_params(transformed_params: DFSVParamsDataclass) -> DFSVParamsDataclass:
    """
    Transform parameters back from unconstrained to constrained space.
    
    Parameters:
    -----------
    transformed_params : DFSVParamsDataclass
        Transformed parameters in unconstrained space
        
    Returns:
    --------
    DFSVParamsDataclass
        Parameters in their natural (constrained) space
    """
    # Apply sigmoid to transform back persistence parameters (diagonal)
    # sigmoid(x) = 1/(1+exp(-x))
    diag_phi_f_orig = 1.0 / (1.0 + jnp.exp(-jnp.diag(transformed_params.Phi_f)))
    phi_f_original = jnp.diag(diag_phi_f_orig)

    diag_phi_h_orig = 1.0 / (1.0 + jnp.exp(-jnp.diag(transformed_params.Phi_h)))
    phi_h_original = jnp.diag(diag_phi_h_orig)
    
    # Apply exp to transform back variance parameters
    # Handle matrix vs vector case for sigma2 (expecting diagonal matrix)
    if transformed_params.sigma2.ndim > 1:
        # For matrix case, only untransform diagonal elements
        diag_indices = jnp.diag_indices_from(transformed_params.sigma2)
        sigma2_original = jnp.zeros_like(transformed_params.sigma2)  # Start with zeros
        # Only exponentiate and place values on the diagonal
        sigma2_original = sigma2_original.at[diag_indices].set(
            jnp.exp(transformed_params.sigma2[diag_indices])
        )
    else: # Should not happen
        sigma2_original = jnp.exp(transformed_params.sigma2)
    
    # Untransform Q_h (diagonal)
    diag_q_h_orig = jnp.exp(jnp.diag(transformed_params.Q_h))
    q_h_original = jnp.diag(diag_q_h_orig)
    
    # Return a new params object with untransformed values
    return transformed_params.replace(
        Phi_f=phi_f_original,
        Phi_h=phi_h_original,
        sigma2=sigma2_original,
        Q_h=q_h_original
    )
