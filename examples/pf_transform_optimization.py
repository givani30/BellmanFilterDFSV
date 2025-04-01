#!/usr/bin/env python
"""
Parameter transformation optimization for DFSV models using Particle Filter.

This script implements parameter transformations for DFSV model optimization
using the Particle Filter to estimate the likelihood.
"""

from collections.abc import Callable
import copy
import os
import sys
import time
from functools import partial
import optimistix as optx
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit
import jaxopt
from jaxtyping import Array, Bool, Int, PyTree, Scalar
# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import parameter classes and PF
import optax
from jaxopt import OptaxSolver
from functions.filters import DFSVParticleFilter # Import Particle Filter
from models.dfsv import DFSV_params, DFSVParamsDataclass
from functions.transformations import transform_params, untransform_params
# Import PF specific objectives
from functions.likelihood_functions import pf_objective, transformed_pf_objective 
# from utils.plotting import plot_variance_comparison # Commented out plotting import
from functions.simulation import simulate_DFSV

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
PRIOR_MU_MEAN = -1.0
PRIOR_MU_STD_DEV = 0.5 # Tune this value! Start with something moderate.

def create_simple_model():
    """Create a simple DFSV model with one factor."""
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
    Q_h = np.array([[0.1]])

    # Create parameter object
    params = DFSV_params(
        N=N,
        K=K,
        lambda_r=lambda_r,
        Phi_f=Phi_f,
        Phi_h=Phi_h,
        mu=mu,
        sigma2=sigma2,
        Q_h=Q_h,
    )

    return params


def create_training_data(params, T=100, seed=42):
    """Generate simulated data for training."""
    returns, factors, log_vols = simulate_DFSV(params, T=T, seed=seed)
    return returns, factors, log_vols

def optimize_with_transformations_pf(params, returns, filter_instance, T=1000, maxiter=100):
    """
    Optimize model parameters using parameter transformations with Particle Filter.
    
    Parameters:
    -----------
    params : DFSV_params
        True model parameters (only used for dimensionality reference)
    returns : np.ndarray
        Observed data
    filter_instance : DFSVParticleFilter
        Particle Filter object
    T : int
        Number of time steps
    maxiter : int
        Maximum number of optimization iterations
        
    Returns:
    --------
    tuple
        (optimized_params_standard, optimized_params_transformed, final_loss_standard, final_loss_transformed)
    """
    # Convert to JAX parameter class for reference dimensions
    jax_params = DFSVParamsDataclass.from_dfsv_params(params)
    
    # Create uninformed initial parameters
    key = jax.random.PRNGKey(42)
    
    # Calculate data variance to inform initial sigma2
    data_variance = jnp.var(returns, axis=0)
    N, K = params.N, params.K
    
    uninformed_params = DFSVParamsDataclass(
        N=N,
        K=K,
        lambda_r=0.5 * jnp.ones((N, K)),  # Moderate positive loadings
        Phi_f=0.8 * jnp.eye(K),  # Moderate persistence
        Phi_h=0.8 * jnp.eye(K),  # Moderate persistence
        mu=jnp.zeros(K),  # Zero mean for log volatility
        sigma2=jnp.diag(0.5 * data_variance),  # Moderate portion of data variance
        Q_h=0.2 * jnp.eye(K)  # Moderate volatility of volatility
    )
    
    # Create optimizer configurations
    # PF optimization might require smaller learning rate and potentially more iterations
    learning_rate = 1e-4 # Further reduced learning rate for PF
    gradient_clipping_norm = 1.0 # Max gradient norm
    opt = optax.chain(
        optax.clip_by_global_norm(gradient_clipping_norm),
        optax.adam(learning_rate=learning_rate)
    )
    
    # Define objective function for standard optimization using PF
    # Pass prior info explicitly
    def objective(p, state=None):
        # Ensure parameters are the correct type for the filter - usually not needed
        # p_dataclass = DFSVParamsDataclass.from_pytree(p)
        return pf_objective(p, returns, filter_instance, PRIOR_MU_MEAN, PRIOR_MU_STD_DEV)
    
    # Define objective function for transformed optimization using PF
    # Pass prior info explicitly
    def objective_transformed(transformed_p, state=None):
        # Ensure transformed parameters are the correct type - usually not needed
        # transformed_p_dataclass = DFSVParamsDataclass.from_pytree(transformed_p)
        return transformed_pf_objective(transformed_p, returns, filter_instance, PRIOR_MU_MEAN, PRIOR_MU_STD_DEV)
        
    # Optimizer using OptaxSolver
    solver = optx.OptaxMinimiser(optim=opt, rtol=1e-4, atol=1e-4, norm=optx.rms_norm, 
                                verbose=frozenset({"step", "loss"})) # Adjusted tolerance
    
    # Transform parameters for transformed optimization
    transformed_params = transform_params(uninformed_params)
    
    # --- Standard Optimization --- #
    print("\nStarting standard optimization with Particle Filter...")
    start_time = time.time()
    result_standard = optx.minimise(fn=objective, solver=solver, y0=uninformed_params, max_steps=maxiter, throw=False)
    standard_time = time.time() - start_time
    print(f"Standard PF optimization took {standard_time:.2f} seconds")
    optimized_params_standard = result_standard.value # Assuming optimizer returns correct Pytree type
    final_loss_standard = result_standard.state.best_f # Get final loss from state
    
    # --- Transformed Optimization --- #
    print("\nStarting transformed optimization with Particle Filter...")
    start_time = time.time()
    result_transformed_raw = optx.minimise(fn=objective_transformed, solver=solver, y0=transformed_params, max_steps=maxiter, throw=False)
    transformed_time = time.time() - start_time
    print(f"Transformed PF optimization took {transformed_time:.2f} seconds")
    
    # Transform parameters back to original space
    optimized_params_transformed = untransform_params(result_transformed_raw.value)
    final_loss_transformed = result_transformed_raw.state.best_f # Get final loss from state
    
    return (optimized_params_standard, optimized_params_transformed, 
            final_loss_standard, final_loss_transformed)

def main():
    """Run the PF parameter transformation optimization example."""
    print("Starting Particle Filter parameter transformation optimization example...")
    
    # Create a simple model
    params = create_simple_model()
    
    # Generate training data (longer time series for better optimization)
    T = 200 # Reduced T for faster PF optimization during testing
    returns, factors, log_vols = create_training_data(params, T=T)
    
    # Create a Particle Filter object
    num_particles = 1000 # Increased particles for stability
    filter_pf = DFSVParticleFilter(N=params.N, K=params.K, num_particles=num_particles) # Instantiate with N, K

    # Convert true params to JAX parameter class for reference
    jax_params_true = DFSVParamsDataclass.from_dfsv_params(params)
    
    # Run optimization
    # Note: PF optimization can be very slow. maxiter might need adjustment.
    max_iterations = 5 # Reduced iterations for testing
    standard_params, transformed_params, standard_loss, transformed_loss = \
        optimize_with_transformations_pf(params, returns, filter_pf, T=T, maxiter=max_iterations)
    
    print("\nOptimization results (Particle Filter):")
    print(f"Standard optimization final loss:      {standard_loss:.4f}")
    print(f"Transformed optimization final loss:   {transformed_loss:.4f}")
    
    # --- Post-optimization Analysis --- #
    print("\nRunning filter with optimized parameters...")
    # Ensure parameters passed to filter are the correct type (e.g., DFSVParamsDataclass)
    standard_states, _, standard_ll = filter_pf.filter(standard_params, returns)
    transformed_states, _, transformed_ll = filter_pf.filter(transformed_params, returns)
    
    # Extract state variables
    K = params.K
    standard_factors = standard_states[:, :K]
    standard_log_vols = standard_states[:, K:]
    transformed_factors = transformed_states[:, :K]
    transformed_log_vols = transformed_states[:, K:]
    
    # Calculate and print MSE for each set of parameters against true values
    factor_mse_standard = jnp.mean((standard_factors - factors) ** 2)
    factor_mse_transformed = jnp.mean((transformed_factors - factors) ** 2)
    logvol_mse_standard = jnp.mean((standard_log_vols.squeeze() - log_vols.squeeze()) ** 2)
    logvol_mse_transformed = jnp.mean((transformed_log_vols.squeeze() - log_vols.squeeze()) ** 2)
    
    print("\nMean squared errors (vs True States):")
    print(f"Factor MSE - Standard Opt: {factor_mse_standard:.4f}, Transformed Opt: {factor_mse_transformed:.4f}")
    print(f"Log-vol MSE - Standard Opt: {logvol_mse_standard:.4f}, Transformed Opt: {logvol_mse_transformed:.4f}")
    
    # --- Plot Comparison --- #
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 10))
    
    # Factor states comparison
    plt.subplot(2, 1, 1)
    plt.plot(factors, "k-", alpha=0.3, label="True Factors")
    plt.plot(standard_factors, "b-", label="Factors (Standard Opt)")
    plt.plot(transformed_factors, "r-", label="Factors (Transformed Opt)")
    plt.title("Factor Estimates (Particle Filter Optimization)")
    plt.legend()
    
    # Log volatility states comparison
    plt.subplot(2, 1, 2)
    plt.plot(log_vols.squeeze(), "k-", alpha=0.3, label="True Log-Vols")
    plt.plot(standard_log_vols.squeeze(), "b-", label="Log-Vols (Standard Opt)")
    plt.plot(transformed_log_vols.squeeze(), "r-", label="Log-Vols (Transformed Opt)")
    plt.title("Log Volatility Estimates (Particle Filter Optimization)")
    plt.legend()
    
    plt.tight_layout()
    save_path_plot = "pf_transform_optimization_comparison.png"
    plt.savefig(save_path_plot)
    # plt.show() # Comment out plt.show() if running non-interactively
    print(f"\nSaved state comparison plot to {save_path_plot}")
    
    # --- Parameter Comparison --- #
    print("\nParameter comparison (Particle Filter Optimization):")
    print("-------------------------------------------------------------")
    print(f"{'Parameter':<10} {'True':<18} {'Standard Opt':<18} {'Transformed Opt':<18}")
    print("-------------------------------------------------------------")
    
    # Helper function to format parameter values
    def format_param(p, name):
        attr = getattr(p, name)
        if hasattr(attr, 'shape') and attr.size > 1:
            if name == 'lambda_r':
                if attr.ndim == 2:
                    rows = []
                    for i in range(attr.shape[0]):
                        rows.append('[' + ', '.join([f'{x:.4f}' for x in attr[i]]) + ']')
                    return '[' + ', '.join(rows) + ']'
                else:
                    return f"[{', '.join([f'{x:.4f}' for x in attr])}]"
            elif attr.ndim == 1 and attr.size <= 5:
                return f"[{', '.join([f'{x:.4f}' for x in attr])}]"
            elif attr.ndim == 2 and attr.shape[0] == 1 and attr.shape[1] == 1:
                return f"{attr[0, 0]:.4f}"
            elif attr.ndim == 2 and attr.size <= 6:
                # Format small matrices
                rows = []
                for i in range(attr.shape[0]):
                    rows.append('[' + ', '.join([f'{x:.4f}' for x in attr[i]]) + ']')
                return '[' + ', '.join(rows) + ']'
            else:
                return f"array{attr.shape}"
        else:
            try:
                if hasattr(attr, 'item'): return f"{attr.item():.4f}"
                else: return f"{float(attr):.4f}"
            except: return str(attr)
            
    # Print key parameters
    for param_name in ['lambda_r', 'Phi_f', 'Phi_h', 'mu', 'Q_h']:
        true_val = format_param(jax_params_true, param_name)
        std_val = format_param(standard_params, param_name)
        trans_val = format_param(transformed_params, param_name)
        print(f"{param_name:<10} {true_val:<18} {std_val:<18} {trans_val:<18}")
    
    # Special handling for sigma2 (can be matrix or vector)
    def format_sigma2(p):
        s2 = p.sigma2
        if s2.ndim == 2 and jnp.allclose(s2, jnp.diag(jnp.diag(s2))):
            s2_diag = jnp.diag(s2)
            if s2_diag.size <= 5:
                return f"diag([{', '.join([f'{s:.4f}' for s in s2_diag])}])"
            else:
                 return f"diag(mean={float(s2_diag.mean()):.4f})"
        elif s2.ndim == 1:
             if s2.size <= 5:
                 return f"[{', '.join([f'{s:.4f}' for s in s2])}]"
             else:
                 return f"vec(mean={float(s2.mean()):.4f})"
        else: # Full matrix
            if s2.size <= 6:
                rows = []
                for i in range(s2.shape[0]):
                    rows.append('[' + ', '.join([f'{x:.4f}' for x in s2[i]]) + ']')
                return '[' + ', '.join(rows) + ']'
            else:
                return f"matrix{s2.shape}"

    true_sigma_str = format_sigma2(jax_params_true)
    std_sigma_str = format_sigma2(standard_params)
    trans_sigma_str = format_sigma2(transformed_params)
    print(f"{'sigma2':<10} {true_sigma_str:<18} {std_sigma_str:<18} {trans_sigma_str:<18}")

    # --- Variance Comparison Plot (Commented Out) --- #
    # print("\nPlotting predicted vs realized return variance comparison...")
    # try:
    #     save_path_var = "/home/givanib/Documents/QF_Thesis/pf_variance_comparison.png"
    #     plot_variance_comparison(
    #         jax_params_true, standard_params, transformed_params, 
    #         log_vols, standard_log_vols, transformed_log_vols, 
    #         returns, save_path=save_path_var
    #     )
    #     print(f"Saved variance comparison plot to {save_path_var}")
    # except Exception as e:
    #     print(f"Could not generate variance comparison plot: {e}")


if __name__ == "__main__":
    main() 