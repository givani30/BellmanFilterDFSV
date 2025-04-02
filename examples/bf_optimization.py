#!/usr/bin/env python
"""
Parameter transformation optimization for DFSV models.

This script implements parameter transformations for DFSV model optimization
to improve numerical stability and convergence by mapping constrained parameters
to unconstrained space.
"""


from collections.abc import Callable
import copy
import os
from queue import PriorityQueue
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

# Import parameter classes
import optax
from jaxopt import OptaxSolver

from functions.bellman_filter import DFSVBellmanFilter
# Update imports to use models.dfsv directly
from models.dfsv import DFSV_params, DFSVParamsDataclass
from functions.transformations import transform_params, untransform_params
from functions.likelihood_functions import bellman_objective, transformed_bellman_objective
from utils.plotting import plot_variance_comparison
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


def compare_gradients(params, returns, filter):
    """
    Compare gradients in original vs transformed parameter spaces.
    
    Parameters:
    -----------
    params : DFSVParamsDataclass
        Original model parameters
    returns : jnp.ndarray
        Observed data
    filter : DFSVBellmanFilter
        Filter object
    """
    print("\nComparing gradients in original vs transformed parameter spaces...")
    
    # Convert to JAX array for gradient computation
    jax_returns = jnp.array(returns)
    
    # Create gradient functions - pass prior info explicitly
    grad_orig = grad(lambda p: bellman_objective(p, jax_returns, filter, PRIOR_MU_MEAN, PRIOR_MU_STD_DEV))
    grad_trans = grad(lambda p: transformed_bellman_objective(p, jax_returns, filter, PRIOR_MU_MEAN, PRIOR_MU_STD_DEV))
    
    # Compute gradients in original space
    start_time = time.time()
    original_grads = grad_orig(params)
    orig_time = time.time() - start_time
    
    # Transform parameters and compute gradients in transformed space
    transformed_params = transform_params(params)
    start_time = time.time()
    transformed_grads = grad_trans(transformed_params)
    trans_time = time.time() - start_time
    
    # Print results
    print(f"\nTime to compute original gradients: {orig_time:.4f} seconds")
    print(f"Time to compute transformed gradients: {trans_time:.4f} seconds")
    
    # Display magnitude of gradients for key parameters
    print("\nGradient magnitudes in original parameter space:")
    print("-----------------------------------------------")
    print(f"{'Parameter':<10} {'Value':<15} {'Gradient Magnitude':<20}")
    print("-----------------------------------------------")
    
    # Helper function to format gradient info
    def format_grad_info(name, value, grad):
        if hasattr(value, 'shape') and value.size > 1:
            val_str = f"array{value.shape}"
            grad_mag = float(jnp.mean(jnp.abs(grad)))
        else:
            if hasattr(value, 'shape') and value.ndim > 0:
                # Handle scalar arrays with explicit dimensions
                val_str = f"{jnp.asarray(value).item():.4f}"
                grad_mag = float(jnp.mean(jnp.abs(grad)))
            else:
                val_str = f"{float(value):.4f}"
                grad_mag = float(jnp.abs(grad))
        return f"{name:<10} {val_str:<15} {grad_mag:<20.4e}"
    
    # Print original gradients
    print(format_grad_info("Phi_f", params.Phi_f, original_grads.Phi_f))
    print(format_grad_info("Phi_h", params.Phi_h, original_grads.Phi_h))
    print(format_grad_info("sigma2", params.sigma2, original_grads.sigma2))
    print(format_grad_info("Q_h", params.Q_h, original_grads.Q_h))
    
    # Print transformed gradients
    print("\nGradient magnitudes in transformed parameter space:")
    print("--------------------------------------------------")
    print(f"{'Parameter':<10} {'Value':<15} {'Gradient Magnitude':<20}")
    print("--------------------------------------------------")
    print(format_grad_info("Phi_f", transformed_params.Phi_f, transformed_grads.Phi_f))
    print(format_grad_info("Phi_h", transformed_params.Phi_h, transformed_grads.Phi_h))
    print(format_grad_info("sigma2", transformed_params.sigma2, transformed_grads.sigma2))
    print(format_grad_info("Q_h", transformed_params.Q_h, transformed_grads.Q_h))


def optimize_with_transformations(params, returns, filter, T=1000, maxiter=100):
    """
    Optimize model parameters using parameter transformations.
    
    Parameters:
    -----------
    params : DFSV_params
        True model parameters (only used for dimensionality reference)
    returns : np.ndarray
        Observed data
    filter : DFSVBellmanFilter
        Filter object
    T : int
        Number of time steps
    maxiter : int
        Maximum number of optimization iterations
        
    Returns:
    --------
    tuple
        (original_params, optimized_params, original_loss, transformed_loss)
    """
    # Convert to JAX parameter class for reference dimensions
    jax_params = DFSVParamsDataclass.from_dfsv_params(params)
    
    # Create uninformed initial parameters
    key = jax.random.PRNGKey(42)
    
    # Set uninformed initial parameters
    # Use common default values for time series models:
    # - Persistence parameters near 0.8 (moderately persistent but not too close to non-stationarity)
    # - Volatility parameters based on data variance
    # - Factor loadings with modest values
    
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
    learning_rate = 0.01
    opt = optax.adam(learning_rate=learning_rate)
    
    # Define objective function for standard optimization - use imported function
    # Pass prior info explicitly
    def objective(params, state=None):
        return bellman_objective(params, returns, filter, PRIOR_MU_MEAN, PRIOR_MU_STD_DEV)
    
    # Define objective function for transformed optimization - use imported function
    # Pass prior info explicitly
    def objective_transformed(transformed_params, state=None):
        return transformed_bellman_objective(transformed_params, returns, filter, PRIOR_MU_MEAN, PRIOR_MU_STD_DEV)
    #Custom BFGS Solver
    class CustomBFGS(optx.AbstractBFGS):
        rtol:float
        atol:float
        norm: Callable[[PyTree], Scalar]
        use_inverse:bool
        descent:optx.AbstractDescent=optx.DoglegDescent()
        search:optx.AbstractSearch=optx.ClassicalTrustRegion()
        verbose: frozenset[str]
        def __init__(
            self,
            rtol: float,
            atol: float,
            norm: Callable[[PyTree], Scalar] = optx.max_norm,
            use_inverse: bool = False,
            verbose: frozenset[str] = frozenset(),
    ):
            self.rtol = rtol
            self.atol = atol
            self.norm = norm
            self.use_inverse = use_inverse
            self.descent = optx.DoglegDescent()
            # self.search = optx.BacktrackingArmijo(decrease_factor=0.1,step_init=0.1)
            self.search=optx.ClassicalTrustRegion()
            self.verbose = verbose
        
    # Transform parameters for transformed optimization
    solver = optx.OptaxMinimiser(optim=opt, rtol=1e-3, atol=1e-3, norm=optx.rms_norm, 
                                verbose=frozenset({"step", "loss"}))
    # solver=optx.BFGS(rtol=1e-3, atol=1e-3, norm=optx.max_norm,verbose=frozenset({"step_size", "loss"}))
    # solver=CustomBFGS(rtol=1e-5, atol=1e-5, norm=optx.max_norm, verbose=frozenset({"step_size", "loss"}))
    transformed_params = transform_params(uninformed_params)
    
    # Run standard optimization
    print("\nStarting standard optimization with uninformed parameters...")
    start_time = time.time()
    result_standard = optx.minimise(fn=objective, solver=solver, y0=uninformed_params, max_steps=maxiter, throw=False)
    standard_time = time.time() - start_time
    print(f"Standard optimization took {standard_time:.2f} seconds")
    
    # Run transformed optimization
    print("\nStarting transformed optimization with uninformed parameters...")
    start_time = time.time()
    result_transformed_raw = optx.minimise(fn=objective_transformed, solver=solver, y0=transformed_params, max_steps=maxiter, throw=False)
    transformed_time = time.time() - start_time
    print(f"Transformed optimization took {transformed_time:.2f} seconds")
    
    # Transform parameters back to original space
    result_transformed = untransform_params(result_transformed_raw.value)
    
    # Calculate objective values - pass prior info explicitly
    obj_standard = bellman_objective(result_standard.value, returns, filter, PRIOR_MU_MEAN, PRIOR_MU_STD_DEV)
    # For transformed, we need the raw transformed output and call the transformed objective
    obj_transformed = transformed_bellman_objective(result_transformed_raw.value, returns, filter, PRIOR_MU_MEAN, PRIOR_MU_STD_DEV)
    
    return (result_standard.value, result_transformed, 
            obj_standard, obj_transformed)


def main():
    """Run the parameter transformation optimization example."""
    print("Starting parameter transformation optimization example...")
    
    # Create a simple model
    params = create_simple_model()
    
    # Generate training data (longer time series for better optimization)
    T = 1000
    returns, factors, log_vols = create_training_data(params, T=T)
    
    # Create a Bellman filter object
    filter = DFSVBellmanFilter(params.N, params.K)
    
    # Convert to JAX parameter class
    jax_params = DFSVParamsDataclass.from_dfsv_params(params)
    
    # Compare gradients
    # compare_gradients(jax_params, returns, filter)
    
    # Run optimization with both methods
    standard_params, transformed_params, standard_loss, transformed_loss = \
        optimize_with_transformations(params, returns, filter, T=T, maxiter=30)
    
    print("\nOptimization results:")
    print(f"Standard optimization final loss:      {standard_loss:.4f}")
    print(f"Transformed optimization final loss:   {transformed_loss:.4f}")
    
    # Run filters with optimized parameters to compare results
    standard_states, standard_cov, standard_ll = filter.filter(standard_params, returns)
    transformed_states, transformed_cov, transformed_ll = filter.filter(transformed_params, returns)
    
    # Extract state variables
    K = params.K
    standard_factors = standard_states[:, :K]
    standard_log_vols = standard_states[:, K:]
    transformed_factors = transformed_states[:, :K]
    transformed_log_vols = transformed_states[:, K:]
    
    # Calculate and print MSE for each set of parameters
    factor_mse_standard = jnp.mean((standard_factors - factors) ** 2)
    factor_mse_transformed = jnp.mean((transformed_factors - factors) ** 2)
    logvol_mse_standard = jnp.mean((standard_log_vols.squeeze() - log_vols.squeeze()) ** 2)
    logvol_mse_transformed = jnp.mean((transformed_log_vols.squeeze() - log_vols.squeeze()) ** 2)
    
    print("\nMean squared errors:")
    print(f"Factor MSE - Standard: {factor_mse_standard:.4f}, Transformed: {factor_mse_transformed:.4f}")
    print(f"Log-vol MSE - Standard: {logvol_mse_standard:.4f}, Transformed: {logvol_mse_transformed:.4f}")
    
    # Plot comparison
    import matplotlib.pyplot as plt
    
    # Plot factors
    plt.figure(figsize=(15, 10))
    
    # Factor states
    plt.subplot(2, 1, 1)
    plt.plot(factors, "k-", alpha=0.3, label="True")
    plt.plot(standard_factors, "b-", label="Standard optimization")
    plt.plot(transformed_factors, "r-", label="Transformed optimization")
    plt.title("Factor Estimates")
    plt.legend()
    
    # Log volatility states
    plt.subplot(2, 1, 2)
    plt.plot(log_vols.squeeze(), "k-", alpha=0.3, label="True")
    plt.plot(standard_log_vols.squeeze(), "b-", label="Standard optimization")
    plt.plot(transformed_log_vols.squeeze(), "r-", label="Transformed optimization")
    plt.title("Log Volatility Estimates")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("transform_optimization_comparison.png")
    plt.show()
    
    print("\nSaved comparison plot to transform_optimization_comparison.png")
    
    # Print parameter comparison
    print("\nParameter comparison:")
    print("-----------------------------------------------------")
    print(f"{'Parameter':<10} {'True':<15} {'Standard':<15} {'Transformed':<15}")
    print("-----------------------------------------------------")
    
    # Helper function to format parameter values
    def format_param(p, name):
        attr = getattr(p, name)
        if hasattr(attr, 'shape') and attr.size > 1:
            if name == 'lambda_r':
                # Special handling for lambda_r to always show all elements
                if attr.ndim == 2:
                    rows = []
                    for i in range(attr.shape[0]):
                        rows.append('[' + ', '.join([f'{x:.4f}' for x in attr[i]]) + ']')
                    return '[' + ', '.join(rows) + ']'
                else:
                    return f"[{', '.join([f'{x:.4f}' for x in attr])}]"
            elif attr.ndim == 1 and attr.size <= 3:
                # For small 1D arrays, show values
                return f"[{', '.join([f'{x:.4f}' for x in attr])}]"
            elif attr.ndim == 2 and attr.shape[0] == 1 and attr.shape[1] == 1:
                # For 1x1 matrices, show as scalar
                return f"{attr[0, 0]:.4f}"
            else:
                # For larger arrays or matrices, just show shape
                return f"array{attr.shape}"
        else:
            # For scalars
            try:
                if hasattr(attr, 'item'):
                    # Safe way to extract scalar from array
                    return f"{attr.item():.4f}"
                else:
                    return f"{float(attr):.4f}"
            except:
                return str(attr)
    
    # Print key parameters
    for param_name in ['lambda_r', 'Phi_f', 'Phi_h', 'mu', 'Q_h']:
        true_val = format_param(jax_params, param_name)
        std_val = format_param(standard_params, param_name)
        trans_val = format_param(transformed_params, param_name)
        print(f"{param_name:<10} {true_val:<15} {std_val:<15} {trans_val:<15}")
    
    # For sigma2, which is often an array
    print("sigma2     ", end="")
    true_sigma = jax_params.sigma2
    std_sigma = standard_params.sigma2
    trans_sigma = transformed_params.sigma2
    
    if true_sigma.size <= 3:
        # Print individual values for small arrays
        print(f"{', '.join([f'{s:.4f}' for s in true_sigma]):<15}", end="")
        print(f"{', '.join([f'{s:.4f}' for s in std_sigma]):<15}", end="")
        print(f"{', '.join([f'{s:.4f}' for s in trans_sigma]):<15}")
    else:
        # Just print shape and mean for larger arrays
        print(f"mean={float(true_sigma.mean()):.4f} ", end="")
        print(f"mean={float(std_sigma.mean()):.4f} ", end="")
        print(f"mean={float(trans_sigma.mean()):.4f}")
    
    # Plot predicted vs real return variance comparison
    print("\nPlotting predicted vs realized return variance comparison...")
    true_var, standard_var, transformed_var = plot_variance_comparison(
        jax_params, standard_params, transformed_params, 
        log_vols, standard_log_vols, transformed_log_vols, 
        returns, save_path="/home/givanib/Documents/QF_Thesis/variance_comparison.png"
    )
    
    print("Saved variance comparison plot to /home/givanib/Documents/QF_Thesis/variance_comparison.png")


if __name__ == "__main__":
    main()