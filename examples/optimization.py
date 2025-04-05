import sys
import os
import time
import jaxopt
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from jax import jit
from sympy import true

# Add the parent directory to the path so we can import from functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import parameter classes
from functions.simulation import DFSV_params, simulate_DFSV
from functions.jax_params import DFSVParamsDataclass
from functions.bellman_filter import DFSVBellmanFilter
from jaxopt import OptaxSolver
import optax

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)


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
    Q_h = np.array([[0.05]])

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


@partial(jit, static_argnames=["filter"])
def bellman_objective(params, y, filter):
    """
    Compute the Bellman objective function for the DFSV model.

    Parameters
    ----------
    params : DFSV_params
        Model parameters.
    y : np.ndarray
        Observed data.
    filter : DFSVBellmanFilter
        Bellman filter object.

    Returns
    -------
    float
        The Bellman objective value.
    """
    # run the bellman filter
    ll = DFSVBellmanFilter.jit_log_likelihood_wrt_params(filter, params, y)
    return -ll


def main():
    # Create a simple model
    params = create_simple_model()
    # Generate training data
    returns, factors, log_vols = create_training_data(params, T=1000)
    # Create a Bellman filter object
    filter = DFSVBellmanFilter(params.N, params.K)
    # Create a JAX-compatible parameter object
    jax_params = DFSVParamsDataclass.from_dfsv_params(params)
    # Perturb the parameters
    jax_params = jax_params.replace(
        lambda_r=jax_params.lambda_r
        + 0.1 * jax.random.normal(jax.random.PRNGKey(0), jax_params.lambda_r.shape),
        Phi_f=jax_params.Phi_f
        + 0.1 * jax.random.normal(jax.random.PRNGKey(1), jax_params.Phi_f.shape),
        Phi_h=jax_params.Phi_h
        + 0.1 * jax.random.normal(jax.random.PRNGKey(2), jax_params.Phi_h.shape),
        mu=jax_params.mu
        + 0.1 * jax.random.normal(jax.random.PRNGKey(3), jax_params.mu.shape),
        sigma2=jax_params.sigma2
        + 0.1 * jax.random.normal(jax.random.PRNGKey(4), jax_params.sigma2.shape),
        Q_h=jax_params.Q_h
        + 0.1 * jax.random.normal(jax.random.PRNGKey(5), jax_params.Q_h.shape),
    )

    # Define objective function for this specific problem
    def objective(params):
        return bellman_objective(params, returns, filter)

    # solver = jaxopt.LBFGS(objective, maxiter=100, tol=1e-6, verbose=False)

    # Create an Optax solver
    opt = optax.adam(learning_rate=0.01)
    solver = OptaxSolver(opt=opt, fun=objective, maxiter=100, tol=1e-6, verbose=True)

    # Time the solver run
    print("Starting optimization...")
    start_time = time.time()
    result = solver.run(jax_params)
    end_time = time.time()
    print(f"Optimization took {end_time - start_time:.2f} seconds")
    # Print the optimized parameters
    print("Optimized parameters:")
    print(result.params)
    # Print the log-likelihood
    print("Log-likelihood:")
    print(result.state.value)

    # Compare original vs optimized parameters
    print("Comparing filter output with original vs optimized parameters")

    # Create filter objects for each set of parameters
    original_filter = DFSVBellmanFilter(params.N, params.K)
    optimized_filter = DFSVBellmanFilter(params.N, params.K)

    # Convert optimized parameters back to standard format if needed
    optimized_params = result.params
    # optimized_params = DFSVParamsDataclass(
    #     N=3,
    #     K=1,
    #     lambda_r=jnp.array([[1.8], [1.2], [0.58]]),
    #     Phi_f=jnp.array([[0.934]]),
    #     Phi_h=jnp.array([[0.967]]),
    #     mu=jnp.array([0.15]),
    #     sigma2=jnp.array([0.114, 0.09, 0.097]),
    #     Q_h=jnp.array([[0.023]]),
    # )

    # Run filters
    original_states, original_cov, original_ll = original_filter.filter(
        jax_params, returns
    )
    optimized_states, optimized_cov, optimized_ll = optimized_filter.filter(
        optimized_params, returns
    )
    K = params.K
    # Extract state variables
    original_factors = original_states[:, :K]
    original_log_vols = original_states[:, K:]
    optimized_factors = optimized_states[:, :K]
    optimized_log_vols = optimized_states[:, K:]

    # Plot comparison
    import matplotlib.pyplot as plt

    # Plot factors
    plt.figure(figsize=(15, 10))

    # Factor states
    plt.subplot(2, 1, 1)
    plt.plot(factors, "k-", alpha=0.3, label="True")
    plt.plot(original_factors, "b-", label="Original params")
    plt.plot(optimized_factors, "r-", label="Optimized params")
    plt.title("Factor Estimates")
    plt.legend()

    # Log volatility states
    plt.subplot(2, 1, 2)
    plt.plot(log_vols.squeeze(), "k-", alpha=0.3, label="True")
    plt.plot(original_log_vols.squeeze(), "b-", label="Original params")
    plt.plot(optimized_log_vols.squeeze(), "r-", label="Optimized params")
    plt.title("Log Volatility Estimates")
    plt.legend()

    plt.tight_layout()
    plt.savefig("filter_comparison.png")
    plt.show()

    # Calculate and print MSE for each set of parameters
    factor_mse_original = np.mean((original_factors - factors) ** 2)
    factor_mse_optimized = np.mean((optimized_factors - factors) ** 2)
    logvol_mse_original = np.mean(
        (original_log_vols.squeeze() - log_vols.squeeze()) ** 2
    )
    logvol_mse_optimized = np.mean(
        (optimized_log_vols.squeeze() - log_vols.squeeze()) ** 2
    )

    print(
        f"Factor MSE - Original: {factor_mse_original:.4f}, Optimized: {factor_mse_optimized:.4f}"
    )
    print(
        f"Log-vol MSE - Original: {logvol_mse_original:.4f}, Optimized: {logvol_mse_optimized:.4f}"
    )


if __name__ == "__main__":
    main()
