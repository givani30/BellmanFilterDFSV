import marimo

from functions.jax_params import dfsv_params_to_dict

__generated_with = "0.11.28"
app = marimo.App(width="medium")


@app.cell
def _(
    DFSVBellmanFilter,
    DFSVParamsDataclass,
    create_simple_model,
    create_training_data,
    jax,
):
    # Create a simple model
    params = create_simple_model()
    # Generate training data
    returns, factors, log_vols = create_training_data(params, T=200)
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
        # sigma2=jax_params.sigma2
        # + 0.1 * jax.random.normal(jax.random.PRNGKey(4), jax_params.sigma2.shape),
        # Q_h=jax_params.Q_h
        # + 0.1 * jax.random.normal(jax.random.PRNGKey(5), jax_params.Q_h.shape),
    )
    return factors, filter, jax_params, log_vols, params, returns


@app.cell
def optimizerloop(
    DFSVParamsMask,
    OptaxSolver,
    bellman_objective,
    filter,
    jax_params,
    optax,
    print,
    returns,
    time,
):
    # Define objective function for this specific problem
    def objective(params):
        return bellman_objective(params, returns, filter)

    # solver = jaxopt.LBFGS(objective, maxiter=50, tol=1e-6, verbose=True,linesearch="backtracking")


    # Create an Optax solver
    param_dict = dfsv_params_to_dict(jax_params)
    mask = {
        "lambda_r": True,
        "Phi_f": True,
        "Phi_h": True,
        "mu": True,
        "sigma2": False,
        "Q_h": False,
    }

    # Then we can create a masked optimizer
    # This means only fields with mask=True get updates
    masked_optimizer = optax.chain(optax.masked(optax.adam(learning_rate=1e-3), mask=mask))
    solver = OptaxSolver(opt=masked_optimizer, fun=objective, maxiter=50, tol=1e-6, verbose=True)

    # Time the solver run
    print("Starting optimization...")
    start_time = time.time()
    result = solver.run(param_dict)
    # final=result.params
    end_time = time.time()
    print(f"Optimization took {end_time - start_time:.2f} seconds")
    return (
        end_time,
        mask,
        masked_optimizer,
        objective,
        result,
        solver,
        start_time,
    )


@app.cell
def _(jnp, print, result):
    # Compare original vs optimized parameters
    print("Comparing filter output with original vs optimized parameters")
    def stablize_matrix(matrix):
        norm = jnp.linalg.norm(matrix, ord=2)
        return matrix / (1.0 + norm)
    final_dict= result.params
    # Convert optimized parameters back to standard format if needed
    optimized_params = DFSVParamsDataclass.from_dict(final_dict)
    optimized_params = optimized_params.replace(sigma2=jnp.exp(optimized_params.sigma2),
                                                Phi_f=jnp.tanh(optimized_params.Phi_f),
                                                Phi_h=jnp.tanh(optimized_params.Phi_h),
                                                Q_h=jnp.exp(optimized_params.Q_h))
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
    print(optimized_params)
    return optimized_params, stablize_matrix


@app.cell
def _(filter, jax_params, optimized_params, params, returns):
    # Run filters
    original_states, original_cov, original_ll = filter.filter(
        jax_params, returns
    )
    optimized_states, optimized_cov, optimized_ll = filter.filter(
        optimized_params, returns
    )
    K = params.K
    # Extract state variables
    original_factors = original_states[:, :K]
    original_log_vols = original_states[:, K:]
    optimized_factors = optimized_states[:, :K]  # Fixed index
    optimized_log_vols = optimized_states[:, K:]  # Fixed index
    optimized_states
    return (
        K,
        optimized_cov,
        optimized_factors,
        optimized_ll,
        optimized_log_vols,
        optimized_states,
        original_cov,
        original_factors,
        original_ll,
        original_log_vols,
        original_states,
    )


@app.cell
def _(optimized_log_vols):
    import pandas as pd
    import polars as pl
    df=pl.DataFrame(optimized_log_vols)
    df
    return df, pd, pl


@app.cell
def _(
    factors,
    log_vols,
    optimized_factors,
    optimized_log_vols,
    original_factors,
    original_log_vols,
    plt,
):
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
    return


@app.cell
def _(
    factors,
    log_vols,
    np,
    optimized_factors,
    optimized_log_vols,
    original_factors,
    original_log_vols,
    print,
):
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
    return (
        factor_mse_optimized,
        factor_mse_original,
        logvol_mse_optimized,
        logvol_mse_original,
    )


@app.cell
def _(DFSVBellmanFilter, DFSV_params, jit, jnp, np, partial, simulate_DFSV):
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
    def bellman_objective(params_unconstrained, y, filter,N,K):
        """
        Compute the Bellman objective function for the DFSV model.

        Parameters
        ----------
        params : DFSV_params
            Unconstrained Model parameters.
        y : np.ndarray
            Observed data.
        filter : DFSVBellmanFilter
            Bellman filter object.
        N : int
            Number of observed series.
        K : int
            Number of factors.

        Returns
        -------
        float
            The Bellman objective value.
        """
        #Create correct dataclass
        params_unconstrained = DFSVParamsDataclass.from_dict(params_unconstrained,N,K)
        #Transform parameters back to constrained space
        def stablize_matrix(matrix):
            norm = jnp.linalg.norm(matrix, ord=2)
            return matrix / (1.0 + norm)
        # 1. Build stable versions
        Phi_f_stable = stablize_matrix(params_unconstrained.Phi_f)
        Phi_h_stable = stablize_matrix(params_unconstrained.Phi_h)

        # 2. Exponential transform for sigma^2 and Q_h
        sigma2_pos = jnp.exp(params_unconstrained.sigma2)
        Q_h_pos = jnp.exp(params_unconstrained.Q_h)

        # 3. Create the new param object
        constrained_params = params_unconstrained.replace(
            Phi_f=Phi_f_stable,
            Phi_h=Phi_h_stable,
            sigma2=sigma2_pos,
            Q_h=Q_h_pos,
        )

        # run the bellman filter
        ll = DFSVBellmanFilter.jit_log_likelihood_of_params(filter, constrained_params, y)
        return -ll
    return bellman_objective, create_simple_model, create_training_data


@app.cell
def imports(__file__):
    import sys
    import os
    import time
    import jaxopt
    import numpy as np
    import jax
    import jax.numpy as jnp
    from rich import print
    from functools import partial
    from jax import jit
    from sympy import true
    # Plot comparison
    import matplotlib.pyplot as plt
    # Add parent directory to path to import our modules
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Import parameter classes
    import marimo as mo
    from functions.simulation import DFSV_params, simulate_DFSV
    from functions.jax_params import DFSVParamsDataclass, dfsv_params_to_dict
    from functions.bellman_filter import DFSVBellmanFilter
    from jaxopt import OptaxSolver
    import optax

    # Enable 64-bit precision for better numerical stability
    jax.config.update("jax_enable_x64", True)
    return (
        DFSVBellmanFilter,
        DFSVParamsDataclass,
        DFSV_params,
        OptaxSolver,
        jax,
        jaxopt,
        jit,
        jnp,
        mo,
        np,
        optax,
        os,
        partial,
        plt,
        print,
        simulate_DFSV,
        sys,
        time,
        true,
    )


if __name__ == "__main__":
    app.run()
