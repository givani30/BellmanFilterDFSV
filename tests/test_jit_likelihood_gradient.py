#!/usr/bin/env python
"""
Gradient testing script for the jit_log_likelihood_of_params function.

This script:
1. Creates a simple DFSV model
2. Computes analytical gradients using jit_log_likelihood_of_params
3. Computes numerical gradients using finite differences
4. Compares the gradients to validate correctness
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
import copy

# Import the necessary functions
from qf_thesis.core.simulation import simulate_DFSV
from qf_thesis.models.dfsv import DFSVParamsDataclass, dfsv_params_to_dict # Removed DFSV_params
from qf_thesis.core.filters.bellman import DFSVBellmanFilter

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

def create_test_model():
    """Create a simple test model (DFSVParamsDataclass) for gradient validation."""
    # Define model dimensions
    N = 3  # Number of observed series
    K = 1  # Number of factors

    # Factor loadings
    lambda_r = np.array([[0.9], [0.6], [0.3]])

    # Factor persistence
    Phi_f = np.array([[0.95]])

    # Log-volatility persistence
    Phi_h = np.array([[0.98]])

    # Long-run mean for log-volatilities
    mu = np.array([-1.0])
    
    # Idiosyncratic variance (diagonal)
    sigma2 = np.array([0.1, 0.1, 0.1])

    # Log-volatility noise covariance
    Q_h = np.array([[0.05]])

    # Create parameter object
    # Create parameter dataclass object using JAX arrays
    params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=jnp.array(lambda_r),
        Phi_f=jnp.array(Phi_f),
        Phi_h=jnp.array(Phi_h),
        mu=jnp.array(mu),
        sigma2=jnp.array(sigma2), # Keep as 1D for dataclass
        Q_h=jnp.array(Q_h),
    )
    return params

def compute_numerical_gradient(func, params_dict, eps=1e-6):
    """
    Compute numerical gradient using central differences.
    
    Args:
        func: Function to differentiate
        params_dict: Parameter dictionary 
        eps: Step size for finite differences
        
    Returns:
        Dictionary of numerical gradients
    """
    grad = {}
    base_value = func(params_dict)
    
    print("Computing numerical gradients for each parameter...")
    for key in params_dict:
        # Skip N and K parameters - they are dimensions and not part of gradient calculation
        if key in ['N', 'K']:
            continue
            
        value = params_dict[key]
        if np.isscalar(value):
            # Handle scalar parameters
            params_plus = copy.deepcopy(params_dict)
            params_plus[key] = value + eps
            params_minus = copy.deepcopy(params_dict)
            params_minus[key] = value - eps
            
            value_plus = func(params_plus)
            value_minus = func(params_minus)
            grad[key] = (value_plus - value_minus) / (2 * eps)
        else:
            # Handle array parameters
            flat_value = np.array(value).flatten()
            flat_grad = np.zeros_like(flat_value, dtype=np.float64)
            
            for i in range(len(flat_value)):
                params_plus = copy.deepcopy(params_dict)
                params_minus = copy.deepcopy(params_dict)
                
                # For arrays, we need to copy and modify just one element
                value_plus = np.array(value, dtype=np.float64).copy()
                value_minus = np.array(value, dtype=np.float64).copy()
                
                flat_plus = value_plus.flatten()
                flat_minus = value_minus.flatten()
                
                flat_plus[i] += eps
                flat_minus[i] -= eps
                
                # Reshape back to original shape
                params_plus[key] = flat_plus.reshape(value.shape)
                params_minus[key] = flat_minus.reshape(value.shape)
                
                value_plus = func(params_plus)
                value_minus = func(params_minus)
                flat_grad[i] = (value_plus - value_minus) / (2 * eps)
            
            grad[key] = flat_grad.reshape(value.shape)
            
        print(f"  Computed gradient for {key}")
    
    return grad

def test_jit_likelihood_gradient():
    """Test the gradient of jit_log_likelihood_of_params."""
    print("\n==== Testing jit_log_likelihood_of_params Gradient ====")
    
    # Create model and data
    jax_params = create_test_model() # Now returns DFSVParamsDataclass directly
    T = 50  # Shorter time series for faster testing
    returns, factors, log_vols = simulate_DFSV(jax_params, T=T, seed=42) # Use correct variable name
    
    # Create filter instance
    bf = DFSVBellmanFilter(jax_params.N, jax_params.K)
    
    # Convert to dictionary format
    # No conversion needed anymore
    param_dict, N_value, K_value = dfsv_params_to_dict(jax_params)
    
    # N and K should not be part of gradient calculation
    # but note they're already returned as separate values from dfsv_params_to_dict
    
    # Create a clean copy of the parameters without any integer values
    cleaned_params = {}
    for key, value in param_dict.items():
        if isinstance(value, (int, np.integer)):
            # Convert integers to floats
            cleaned_params[key] = float(value)
        elif isinstance(value, np.ndarray):
            # Convert arrays to float64
            if np.issubdtype(value.dtype, np.integer):
                cleaned_params[key] = value.astype(np.float64)
            else:
                cleaned_params[key] = value.astype(np.float64)
        else:
            cleaned_params[key] = value
    
    # Define the objective function (negative log-likelihood)
    # Pass N and K to the filter separately, not as part of the parameter dictionary
    def objective(p):
        # Create a complete parameter dictionary with N and K for the filter
        complete_params = dict(p)
        complete_params['N'] = N_value
        complete_params['K'] = K_value
        # Don't convert to float here - let JAX handle the value type during differentiation
        return -bf.log_likelihood_of_params(complete_params, returns)
    
    print("Parameter dictionary contents (excluding N and K for gradient):")
    for key, value in cleaned_params.items():
        if isinstance(value, (np.ndarray, jnp.ndarray)):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value} (type: {type(value)})")
    
    # Test the function
    print("\nTesting objective function with the parameter dictionary...")
    try:
        val = objective(cleaned_params)
        print(f"Objective value: {val}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nComputing analytical gradient...")
    start_time = time.time()
    
    # Use JAX to compute the gradient - N and K are not part of gradient calculation
    try:
        grad_fn = jax.grad(objective)
        analytical_grad = grad_fn(cleaned_params)
        analytical_time = time.time() - start_time
        print(f"Analytical gradient computed in {analytical_time:.4f} seconds")
    except Exception as e:
        print(f"Error computing analytical gradient: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nComputing numerical gradient...")
    start_time = time.time()
    try:
        numerical_grad = compute_numerical_gradient(objective, cleaned_params)
        numerical_time = time.time() - start_time
        print(f"Numerical gradient computed in {numerical_time:.4f} seconds")
    except Exception as e:
        print(f"Error computing numerical gradient: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compare gradients
    print("\nComparing analytical and numerical gradients:")
    print("---------------------------------------------")
    print(f"{'Parameter':<10} {'Max Abs Diff':<15} {'Max Rel Diff':<15} {'Mean Rel Diff':<15}")
    print("---------------------------------------------")
    
    max_diff = 0.0
    max_param = None
    
    for key in analytical_grad:
        # Calculate differences between analytical and numerical gradients
        abs_diff = np.abs(analytical_grad[key] - numerical_grad[key])
        denominator = np.maximum(np.abs(analytical_grad[key]), np.abs(numerical_grad[key]))
        denominator = np.where(denominator < 1e-10, 1.0, denominator)  # Avoid division by zero
        rel_diff = abs_diff / denominator
        
        max_abs_diff = np.max(abs_diff)
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)
        
        print(f"{key:<10} {max_abs_diff:<15.6e} {max_rel_diff:<15.6e} {mean_rel_diff:<15.6e}")
        
        if max_rel_diff > max_diff:
            max_diff = max_rel_diff
            max_param = key
    
    print("\nParameter values and normalized gradients:")
    print("------------------------------------------")
    print(f"{'Parameter':<10} {'Param Value':<20} {'Raw Grad Mean':<15} {'Scale-Corrected':<20}")
    print("------------------------------------------")
    
    # Log-likelihood is on a log scale, so changes of ~1 in input can cause large changes in output
    # We'll calculate a "scale-corrected" gradient that accounts for parameter scale
    for key in analytical_grad:
        param_value = cleaned_params[key]
        mean_grad = np.mean(np.abs(analytical_grad[key]))
        
        # For array parameters, show average value and use proper formatting
        if hasattr(param_value, 'shape') and param_value.size > 1:
            mean_param = np.mean(np.abs(param_value))
            param_desc = f"array({param_value.shape}), ~{mean_param:.4f}"
            
            # Scale-corrected gradient is gradient * parameter value            # This gives percent change in likelihood for percent change in parameter
            scale_corrected = mean_grad * mean_param
        else:
            # For scalar parameters (handle potential 0-dim or 1-element arrays)
            scalar_val = param_value.item() if hasattr(param_value, 'item') and param_value.size == 1 else param_value
            param_desc = f"{float(scalar_val):.4f}"
            scale_corrected = mean_grad * float(scalar_val)
            
        print(f"{key:<10} {param_desc:<20} {mean_grad:<15.6e} {scale_corrected:<20.6e}")
    
    print("\nOverall maximum relative difference:")
    print(f"  {max_diff:.6e} in parameter '{max_param}'")
    
    # Visualize the gradients
    print("\nCreating gradient visualization plot...")
    plt.figure(figsize=(15, 10))
    
    # Layout: a grid of subplots for each parameter
    n_params = len(analytical_grad)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    plot_idx = 1
    for key in analytical_grad:
        plt.subplot(n_rows, n_cols, plot_idx)
        
        # For scalar parameters or 1D arrays
        if np.isscalar(analytical_grad[key]) or analytical_grad[key].size == 1:
            # Convert JAX arrays to numpy arrays for plotting
            anal_val = np.array(analytical_grad[key]).item() if analytical_grad[key].size == 1 else analytical_grad[key]
            num_val = np.array(numerical_grad[key]).item() if numerical_grad[key].size == 1 else numerical_grad[key]
            
            plt.bar([1, 2], [anal_val, num_val], 
                   tick_label=['Analytical', 'Numerical'])
            plt.title(f"{key}")
        else:
            # For arrays, create scatter plot
            # Convert JAX arrays to numpy arrays for plotting
            flat_analytical = np.array(analytical_grad[key]).flatten()
            flat_numerical = np.array(numerical_grad[key]).flatten()
            
            plt.scatter(flat_analytical, flat_numerical, alpha=0.7)
            
            # Add a diagonal line
            min_val = min(flat_analytical.min(), flat_numerical.min())
            max_val = max(flat_analytical.max(), flat_numerical.max())
            pad = (max_val - min_val) * 0.1
            plt.plot([min_val-pad, max_val+pad], [min_val-pad, max_val+pad], 'r--')
            
            plt.xlabel('Analytical Gradient')
            plt.ylabel('Numerical Gradient')
            plt.title(f"Parameter: {key}")
            plt.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('gradient_validation.png')
    print("Plot saved as gradient_validation.png")
    
    if max_diff < 1e-5:
        print("\nValidation PASSED: Analytical and numerical gradients match closely!")
    elif max_diff < 1e-3:
        print("\nValidation WARNING: Small differences between analytical and numerical gradients.")
        print("This might be due to numerical precision issues or non-smooth functions.")
    else:
        print("\nValidation FAILED: Significant differences between analytical and numerical gradients.")
        print("The gradient computation may have errors.")

if __name__ == "__main__":
    test_jit_likelihood_gradient()