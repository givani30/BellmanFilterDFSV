#!/usr/bin/env python
"""
Parameter transformation example for DFSV model optimization.

This example demonstrates how to transform DFSV model parameters from constrained 
to unconstrained space for optimization, addressing issues with large gradients.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import jax
import jax.numpy as jnp

# Add the parent directory to path so we can import from functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary functions from your codebase with updated imports
from functions.simulation import simulate_DFSV
from models.dfsv import DFSV_params, DFSVParamsDataclass, dfsv_params_to_dict
from functions.bellman_filter import DFSVBellmanFilter

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

def transform_params(param_dict):
    """
    Transform bounded parameters to unconstrained space for optimization.
    
    Parameters:
    -----------
    param_dict : dict
        Dictionary of model parameters in their natural (constrained) space
        
    Returns:
    --------
    dict
        Dictionary of transformed parameters in unconstrained space
    """
    transformed = {}
    
    # Extract N, K if they exist
    N = param_dict.get('N')
    K = param_dict.get('K')
    
    for key, value in param_dict.items():
        # Skip N and K, they're not optimization parameters
        if key in ['N', 'K']:
            continue
            
        # Handle different parameter types based on their constraints
        if key in ['Phi_f', 'Phi_h']:
            # Persistence parameters in (0,1) -> log-odds transformation to (-∞,∞)
            # Using a small epsilon to avoid numerical issues at boundaries
            eps = 1e-6
            # Use numpy for initial preprocessing (not part of JAX computation graph)
            bounded_value = np.clip(value, eps, 1-eps)
            transformed[key] = np.log(bounded_value / (1 - bounded_value))
        elif key in ['Q_h', 'sigma2']:
            # Variance parameters (must be positive) -> log transformation
            eps = 1e-10
            positive_value = np.maximum(value, eps)
            transformed[key] = np.log(positive_value)
        else:
            # Parameters without specific bounds (like mu, lambda_r) remain unchanged
            transformed[key] = value
    
    return transformed

def inverse_transform_params(transformed_dict):
    """
    Transform parameters back from unconstrained to constrained space.
    
    Parameters:
    -----------
    transformed_dict : dict
        Dictionary of transformed parameters in unconstrained space
        
    Returns:
    --------
    dict
        Dictionary of parameters in their natural (constrained) space
    """
    original = {}
    
    # Extract N, K if they exist in original params
    N = transformed_dict.get('N')
    K = transformed_dict.get('K')
    
    for key, value in transformed_dict.items():
        # Skip N and K, they're not optimization parameters
        if key in ['N', 'K']:
            continue
            
        # Handle different parameter types based on their constraints
        if key in ['Phi_f', 'Phi_h']:
            # Inverse of log-odds transformation: sigmoid function
            # Use jnp for operations that need to be differentiable
            original[key] = 1 / (1 + jnp.exp(-value))
        elif key in ['Q_h', 'sigma2']:
            # Inverse of log transformation: exponential
            original[key] = jnp.exp(value)
        else:
            # Parameters without specific bounds remain unchanged
            original[key] = value
    
    return original

def visualize_transformations():
    """
    Visualize the effect of parameter transformations on gradients.
    """
    print("Visualizing parameter transformations...")
    
    # Create parameter ranges
    phi_values = np.linspace(0.01, 0.99, 100)
    sigma_values = np.linspace(0.01, 0.5, 100)
    
    # Apply transformations
    transformed_phi = np.log(phi_values / (1 - phi_values))
    transformed_sigma = np.log(sigma_values)
    
    # Calculate more realistic example gradients
    # For phi (persistence parameter):
    # - Gradient increases dramatically as phi approaches 1
    orig_grad_phi = 100 * (phi_values - 0.9) / (1 - phi_values)  # Very large near 1
    
    # For sigma (variance parameter):
    # - Gradient is often inversely proportional to sigma^2 (common in variance parameters)
    # - This creates very large gradients for small sigma values
    orig_grad_sigma = 50 / (sigma_values**2)  # Large gradients for small sigma
    
    # Calculate transformed gradients
    # The chain rule gives us: grad_transform = grad_orig * d(orig)/d(transform)
    
    # For phi: using logit transform, derivative is phi*(1-phi)
    transformed_grad_phi = orig_grad_phi * phi_values * (1 - phi_values)
    
    # For sigma: using log transform, derivative is sigma
    transformed_grad_sigma = orig_grad_sigma * sigma_values
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 4, figure=fig)
    
    # Original parameter space
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(phi_values, orig_grad_phi, 'b-', linewidth=2, label='Original gradient')
    ax1.set_title('Original Phi gradient')
    ax1.set_xlabel('Phi')
    ax1.set_ylabel('Gradient')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Add some y-limits to avoid extreme values
    phi_grad_max = np.percentile(orig_grad_phi[~np.isinf(orig_grad_phi)], 95)
    ax1.set_ylim(-phi_grad_max*0.1, phi_grad_max*1.1)
    
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.plot(sigma_values, orig_grad_sigma, 'r-', linewidth=2, label='Original gradient')
    ax2.set_title('Original Sigma gradient')
    ax2.set_xlabel('Sigma')
    ax2.set_ylabel('Gradient')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Add some y-limits to avoid extreme values
    sigma_grad_max = np.percentile(orig_grad_sigma[~np.isinf(orig_grad_sigma)], 95)
    ax2.set_ylim(-sigma_grad_max*0.1, sigma_grad_max*1.1)
    
    # Transformed parameter space
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.plot(transformed_phi, transformed_grad_phi, 'b-', linewidth=2, label='Transformed gradient')
    ax3.set_title('Transformed Phi gradient')
    ax3.set_xlabel('log(Phi/(1-Phi))')
    ax3.set_ylabel('Gradient')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Add some y-limits for better visualization
    trans_phi_grad_max = np.percentile(transformed_grad_phi[~np.isinf(transformed_grad_phi)], 95)
    ax3.set_ylim(-trans_phi_grad_max*0.1, trans_phi_grad_max*1.1)
    
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.plot(transformed_sigma, transformed_grad_sigma, 'r-', linewidth=2, label='Transformed gradient')
    ax4.set_title('Transformed Sigma gradient')
    ax4.set_xlabel('log(Sigma)')
    ax4.set_ylabel('Gradient')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Add some y-limits for better visualization
    trans_sigma_grad_max = np.percentile(transformed_grad_sigma[~np.isinf(transformed_grad_sigma)], 95)
    ax4.set_ylim(-trans_sigma_grad_max*0.1, trans_sigma_grad_max*1.1)
    
    plt.tight_layout()
    plt.savefig('parameter_transformation.png')
    print("Saved visualization to parameter_transformation.png")

def demonstrate_optimizer_with_transformations():
    """
    Demonstrate optimizer using parameter transformations.
    """
    print("\nDemonstrating parameter transformations in optimization...")
    
    # Create a test model
    N, K = 3, 1
    lambda_r = np.array([[0.9], [0.6], [0.3]])
    Phi_f = np.array([[0.95]])
    Phi_h = np.array([[0.98]])
    mu = np.array([-1.0])
    sigma2 = np.array([0.1, 0.1, 0.1])
    Q_h = np.array([[0.05]])
    
    params = DFSV_params(
        N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, Phi_h=Phi_h,
        mu=mu, sigma2=sigma2, Q_h=Q_h
    )
    
    # Generate synthetic data
    T = 100  # Shorter time series for faster testing
    returns, factors, log_vols = simulate_DFSV(params, T=T, seed=42)
    
    # Create filter instance
    bf = DFSVBellmanFilter(params.N, params.K)
    
    # Convert parameters to dict format
    jax_params = DFSVParamsDataclass.from_dfsv_params(params)
    param_dict, N_value, K_value = dfsv_params_to_dict(jax_params)
    
    # Transform parameters to unconstrained space
    transformed_params = transform_params(param_dict)
    
    # Add N and K back for likelihood evaluation
    full_transformed_params = dict(transformed_params)
    full_transformed_params['N'] = N_value
    full_transformed_params['K'] = K_value
    
    # Define objective function using TRANSFORMED parameters
    def objective_transformed(transformed_p):
        # Convert transformed parameters back to original space
        original_p = inverse_transform_params(transformed_p)
        
        # Add N and K for filter evaluation
        original_p['N'] = N_value
        original_p['K'] = K_value
        
        # Calculate negative log-likelihood
        return -bf.log_likelihood_of_params(original_p, returns)
    
    # Define objective function using ORIGINAL parameters (for comparison)
    def objective_original(p):
        # Add N and K for filter evaluation
        p_with_dims = dict(p)
        p_with_dims['N'] = N_value
        p_with_dims['K'] = K_value
        
        # Calculate negative log-likelihood
        return -bf.log_likelihood_of_params(p_with_dims, returns)
    
    # Compute and compare gradients
    print("\nComputing gradients in original and transformed parameter spaces...")
    
    # Create JAX gradient functions
    grad_original = jax.grad(objective_original)
    grad_transformed = jax.grad(objective_transformed)
    
    # Compute gradients
    original_grads = grad_original(param_dict)
    transformed_grads = grad_transformed(transformed_params)
    
    # Display original gradients
    print("\nGradients in ORIGINAL parameter space:")
    print("-------------------------------------")
    print(f"{'Parameter':<10} {'Value':<15} {'Gradient':<15} {'Normalized':<15}")
    print("-------------------------------------")
    
    for key in param_dict:
        if key not in ['N', 'K']:
            param_value = param_dict[key]
            gradient = original_grads[key]
            
            if hasattr(param_value, 'shape') and param_value.size > 1:
                value_desc = f"array{param_value.shape}"
                mean_grad = np.mean(np.abs(gradient))
                mean_param = np.mean(np.abs(param_value))
                norm_grad = mean_grad * mean_param  # Normalized by param value
            else:
                value_desc = f"{float(param_value):.4f}"
                mean_grad = np.abs(gradient).item() if hasattr(gradient, 'item') else np.abs(gradient)
                norm_grad = mean_grad * float(param_value)
                
            print(f"{key:<10} {value_desc:<15} {mean_grad:<15.4e} {norm_grad:<15.4e}")
    
    # Display transformed gradients
    print("\nGradients in TRANSFORMED parameter space:")
    print("---------------------------------------")
    print(f"{'Parameter':<10} {'Value':<15} {'Gradient':<15} {'Original Value':<15}")
    print("---------------------------------------")
    
    for key in transformed_params:
        if key not in ['N', 'K']:
            trans_value = transformed_params[key]
            orig_value = param_dict[key]
            gradient = transformed_grads[key]
            
            if hasattr(trans_value, 'shape') and trans_value.size > 1:
                trans_desc = f"array{trans_value.shape}"
                orig_desc = f"array{orig_value.shape}"
                mean_grad = np.mean(np.abs(gradient))
            else:
                trans_desc = f"{float(trans_value):.4f}"
                orig_desc = f"{float(orig_value):.4f}"
                mean_grad = np.abs(gradient).item() if hasattr(gradient, 'item') else np.abs(gradient)
                
            print(f"{key:<10} {trans_desc:<15} {mean_grad:<15.4e} {orig_desc:<15}")

if __name__ == "__main__":
    # Visualize the transformation functions
    visualize_transformations()
    
    # Demonstrate optimization with transformations
    demonstrate_optimizer_with_transformations()