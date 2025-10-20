.. _examples:

Examples
========

This section provides complete, runnable examples demonstrating the key features of BellmanFilterDFSV.

Example 1: Basic DFSV Simulation
---------------------------------

This example shows how to simulate data from a DFSV model and examine its properties.

**File**: ``examples/01_dfsv_simulation.py``

.. code-block:: python

   """
   Basic DFSV Model Simulation Example
   
   This example demonstrates:
   1. Creating DFSV model parameters
   2. Simulating returns, factors, and log-volatilities
   3. Analyzing simulation statistics
   """
   
   import jax.numpy as jnp
   import jax.random as jr
   import matplotlib.pyplot as plt
   from bellman_filter_dfsv.core.models import DFSVParamsDataclass, simulate_DFSV
   
   # Set random seed for reproducibility
   key = jr.PRNGKey(42)
   
   # Define model parameters for 5 series, 2 factors
   params = DFSVParamsDataclass(
       N=5, K=2,
       lambda_r=jnp.array([
           [0.8, 0.2], [0.7, 0.3], [0.9, 0.1],
           [0.6, 0.4], [0.8, 0.2]
       ]),  # Factor loadings (N×K)
       Phi_f=jnp.array([[0.7, 0.1], [0.1, 0.6]]),  # Factor AR matrix (K×K)
       Phi_h=jnp.array([[0.95, 0.0], [0.0, 0.92]]),  # Log-vol AR matrix (K×K)
       mu=jnp.array([-1.2, -1.0]),  # Long-run mean of log-vols (K,)
       sigma2=jnp.array([0.3, 0.25, 0.35, 0.28, 0.32]),  # Idiosyncratic variances (N,)
       Q_h=jnp.array([[0.01, 0.0], [0.0, 0.0144]])  # Log-vol innovation cov (K×K)
   )
   
   # Simulate 1000 time periods
   returns, factors, log_vols = simulate_DFSV(params, T=1000, key=key)
   
   # Analyze results
   print("DFSV Model Simulation Results")
   print("=" * 40)
   print(f"Returns shape: {returns.shape}")
   print(f"Returns mean: {jnp.mean(returns, axis=0)}")
   print(f"Returns std: {jnp.std(returns, axis=0)}")
   print(f"Return correlations:\n{jnp.corrcoef(returns.T)}")

Example 2: Filter Comparison
----------------------------

This example compares the performance of different filtering algorithms.

**File**: ``examples/02_basic_filtering.py``

.. code-block:: python

   """
   Filter Comparison Example
   
   This example demonstrates:
   1. Using different filtering algorithms
   2. Comparing computational performance
   3. Analyzing filtering accuracy
   """
   
   import time
   import jax.numpy as jnp
   from bellman_filter_dfsv.core.models import DFSVParamsDataclass, simulate_DFSV
   from bellman_filter_dfsv.core.filters import (
       DFSVBellmanInformationFilter,
       DFSVBellmanFilter, 
       DFSVParticleFilter
   )
   
   # Simulate data (same as Example 1)
   params = DFSVParamsDataclass(N=3, K=1, ...)  # Simplified for speed
   returns, _, _ = simulate_DFSV(params, T=500, key=42)
   
   # Initialize filters
   filters = {
       "Bellman Information Filter": DFSVBellmanInformationFilter(N=3, K=1),
       "Bellman Filter": DFSVBellmanFilter(N=3, K=1),
       "Particle Filter": DFSVParticleFilter(N=3, K=1, num_particles=1000)
   }
   
   # Compare performance
   results = {}
   for name, filter_obj in filters.items():
       print(f"\nRunning {name}...")
       start_time = time.time()
       
       states, covs, loglik = filter_obj.filter(params, returns)
       
       elapsed = time.time() - start_time
       results[name] = {
           'loglik': loglik,
           'time': elapsed,
           'states': states
       }
       
       print(f"  Log-likelihood: {loglik:.2f}")
       print(f"  Time: {elapsed:.3f}s")
   
   # Find best performing filter
   best_filter = max(results.keys(), key=lambda k: results[k]['loglik'])
   print(f"\nBest filter by log-likelihood: {best_filter}")

Example 3: Parameter Estimation
-------------------------------

This example demonstrates maximum likelihood parameter estimation.

**File**: ``examples/03_parameter_optimization.py``

.. code-block:: python

   """
   Parameter Estimation Example
   
   This example demonstrates:
   1. Setting up parameter estimation
   2. Using different optimizers
   3. Analyzing convergence
   """
   
   import jax.numpy as jnp
   from bellman_filter_dfsv.core.models import DFSVParamsDataclass, simulate_DFSV
   from bellman_filter_dfsv.core.optimization import run_optimization, FilterType
   
   # Generate synthetic data with known parameters
   true_params = DFSVParamsDataclass(
       N=3, K=1,
       lambda_r=jnp.array([[0.8], [0.7], [0.9]]),  # Factor loadings (N×K)
       Phi_f=jnp.array([[0.7]]),  # Factor AR matrix (K×K)
       Phi_h=jnp.array([[0.95]]),  # Log-vol AR matrix (K×K)
       mu=jnp.array([-1.2]),  # Long-run mean of log-vols (K,)
       sigma2=jnp.array([0.3, 0.25, 0.35]),  # Idiosyncratic variances (N,)
       Q_h=jnp.array([[0.01]])  # Log-vol innovation cov (K×K)
   )
   
   returns, _, _ = simulate_DFSV(true_params, T=1000, key=42)
   
   # Create initial guess (perturbed true parameters)
   initial_params = DFSVParamsDataclass(
       N=3, K=1,
       lambda_r=jnp.array([[0.5], [0.5], [0.5]]),  # Factor loadings (N×K)
       Phi_f=jnp.array([[0.5]]),  # Factor AR matrix (K×K)
       Phi_h=jnp.array([[0.9]]),  # Log-vol AR matrix (K×K)
       mu=jnp.array([-1.0]),  # Long-run mean of log-vols (K,)
       sigma2=jnp.array([0.4, 0.4, 0.4]),  # Idiosyncratic variances (N,)
       Q_h=jnp.array([[0.0225]])  # Log-vol innovation cov (K×K)
   )
   
   # Run optimization
   result = run_optimization(
       filter_type=FilterType.BELLMAN_INFORMATION,
       returns=returns,
       initial_params=initial_params,
       fix_mu=True,
       use_transformations=True,
       optimizer_name="BFGS",
       max_steps=500,
       verbose=True
   )
   
   print(f"\nOptimization Results:")
   print(f"Converged: {result.converged}")
   print(f"Final log-likelihood: {result.final_loglik:.2f}")
   print(f"Iterations: {result.num_iterations}")
   
   # Compare estimated vs true parameters
   estimated_params = result.final_params
   print(f"\nParameter Comparison:")
   print(f"True lambda_r: {true_params.lambda_r.flatten()}")
   print(f"Est. lambda_r: {estimated_params.lambda_r.flatten()}")

Example 4: Real Data Application
--------------------------------

This example shows how to apply the filters to real financial data.

**File**: ``examples/04_real_data_application.py``

.. code-block:: python

   """
   Real Data Application Example
   
   This example demonstrates:
   1. Loading and preprocessing real financial data
   2. Estimating DFSV model parameters
   3. Analyzing results and factor loadings
   """
   
   import pandas as pd
   import jax.numpy as jnp
   import matplotlib.pyplot as plt
   from bellman_filter_dfsv.core.models import DFSVParamsDataclass
   from bellman_filter_dfsv.core.optimization import run_optimization, FilterType
   
   # Load real data (example with synthetic data)
   # In practice, load from CSV or financial data API
   dates = pd.date_range('2020-01-01', periods=500, freq='D')
   
   # Simulate realistic financial returns
   np.random.seed(42)
   returns_df = pd.DataFrame({
       'AAPL': np.random.normal(0.001, 0.02, 500),
       'GOOGL': np.random.normal(0.0008, 0.025, 500),
       'MSFT': np.random.normal(0.0012, 0.022, 500),
       'TSLA': np.random.normal(0.002, 0.04, 500),
   }, index=dates)
   
   # Convert to JAX arrays
   returns = jnp.array(returns_df.values)
   
   print(f"Data shape: {returns.shape}")
   print(f"Date range: {dates[0]} to {dates[-1]}")
   
   # Set up initial parameters for 4 series, 2 factors
   initial_params = DFSVParamsDataclass(
       N=4, K=2,
       lambda_r=jnp.ones((4, 2)) * 0.5,  # Factor loadings (N×K)
       Phi_f=jnp.eye(2) * 0.6,  # Factor AR matrix (K×K)
       Phi_h=jnp.eye(2) * jnp.array([0.95, 0.93]),  # Log-vol AR matrix (K×K)
       mu=jnp.array([-1.5, -1.3]),  # Long-run mean of log-vols (K,)
       sigma2=jnp.ones(4) * 0.3,  # Idiosyncratic variances (N,)
       Q_h=jnp.diag(jnp.array([0.01, 0.0144]))  # Log-vol innovation cov (K×K)
   )
   
   # Estimate parameters
   result = run_optimization(
       filter_type=FilterType.BELLMAN_INFORMATION,
       returns=returns,
       initial_params=initial_params,
       fix_mu=True,
       max_steps=300,
       verbose=True
   )
   
   # Analyze results
   if result.converged:
       print("\nEstimated Factor Loadings:")
       loadings = result.final_params.lambda_r
       for i, stock in enumerate(returns_df.columns):
           print(f"{stock}: Factor 1 = {loadings[i,0]:.3f}, Factor 2 = {loadings[i,1]:.3f}")
   
   # Plot factor loadings
   plt.figure(figsize=(10, 6))
   plt.bar(range(len(returns_df.columns)), loadings[:, 0], alpha=0.7, label='Factor 1')
   plt.bar(range(len(returns_df.columns)), loadings[:, 1], alpha=0.7, label='Factor 2')
   plt.xlabel('Stocks')
   plt.ylabel('Factor Loading')
   plt.title('Estimated Factor Loadings')
   plt.xticks(range(len(returns_df.columns)), returns_df.columns)
   plt.legend()
   plt.show()

Running the Examples
--------------------

All examples are located in the ``examples/`` directory and can be run directly:

.. code-block:: bash

   # Run basic simulation
   python examples/01_dfsv_simulation.py
   
   # Run filter comparison
   python examples/02_basic_filtering.py
   
   # Run parameter estimation
   python examples/03_parameter_optimization.py
   
   # Run real data application
   python examples/04_real_data_application.py

Or using uv:

.. code-block:: bash

   uv run python examples/01_dfsv_simulation.py

Each example includes detailed comments explaining the code and expected outputs.
