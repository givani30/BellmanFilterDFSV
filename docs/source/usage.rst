.. _usage:

Usage Guide
===========

This guide provides comprehensive examples of how to use BellmanFilterDFSV for filtering and parameter estimation in Dynamic Factor Stochastic Volatility models.

Basic Concepts
--------------

**Dynamic Factor Stochastic Volatility (DFSV) Model**

The DFSV model represents observed returns as:

.. math::

   y_t = \Lambda f_t + \epsilon_t

where:

* :math:`y_t` are observed returns (N×1)
* :math:`f_t` are latent factors (K×1)
* :math:`\Lambda` is the factor loading matrix (N×K)
* :math:`\epsilon_t` are idiosyncratic errors with stochastic volatility

**Filtering Algorithms**

The package provides three filtering algorithms:

1. **Bellman Information Filter (BIF)**: Information-form implementation for numerical stability
2. **Bellman Filter**: Traditional covariance-form implementation
3. **Particle Filter**: Bootstrap particle filter for non-linear/non-Gaussian cases

Basic Usage
-----------

**1. Model Definition and Simulation**

.. code-block:: python

   import jax.numpy as jnp
   from bellman_filter_dfsv.core.models import DFSVParamsDataclass, simulate_DFSV

   # Define model parameters
   params = DFSVParamsDataclass(
       N=5,  # Number of observed series
       K=2,  # Number of factors
       Lambda=jnp.array([[0.8, 0.2], [0.7, 0.3], [0.9, 0.1],
                        [0.6, 0.4], [0.8, 0.2]]),
       phi_f=jnp.array([[0.7, 0.1], [0.1, 0.6]]),  # Factor AR coefficients
       phi_h=jnp.array([0.95, 0.92]),  # Log-vol persistence
       sigma_f=jnp.array([1.0, 1.0]),  # Factor innovation std
       sigma_h=jnp.array([0.1, 0.12]),  # Log-vol innovation std
       sigma_eps=jnp.array([0.3, 0.25, 0.35, 0.28, 0.32]),  # Idiosyncratic std
       mu=jnp.array([-1.2, -1.0])  # Log-vol means
   )

   # Simulate data
   returns, factors, log_vols = simulate_DFSV(params, T=1000, key=42)
   print(f"Simulated {returns.shape[0]} time periods for {returns.shape[1]} series")

**2. Filtering with Bellman Information Filter**

.. code-block:: python

   from bellman_filter_dfsv.core.filters import DFSVBellmanInformationFilter

   # Create filter
   bif = DFSVBellmanInformationFilter(N=5, K=2)

   # Run filtering
   states, covs, loglik = bif.filter(params, returns)

   print(f"Log-likelihood: {loglik:.2f}")
   print(f"Filtered states shape: {states.shape}")  # (T, 2*K)
   print(f"Filtered covariances shape: {covs.shape}")  # (T, 2*K, 2*K)

**3. Comparing Filter Performance**

.. code-block:: python

   from bellman_filter_dfsv.core.filters import (
       DFSVBellmanInformationFilter,
       DFSVBellmanFilter,
       DFSVParticleFilter
   )
   import time

   # Initialize filters
   bif = DFSVBellmanInformationFilter(N=5, K=2)
   bf = DFSVBellmanFilter(N=5, K=2)
   pf = DFSVParticleFilter(N=5, K=2, num_particles=1000)

   filters = [("BIF", bif), ("Bellman", bf), ("Particle", pf)]

   # Compare performance
   for name, filter_obj in filters:
       start_time = time.time()
       states, covs, loglik = filter_obj.filter(params, returns)
       elapsed = time.time() - start_time

       print(f"{name} Filter:")
       print(f"  Log-likelihood: {loglik:.2f}")
       print(f"  Time: {elapsed:.3f}s")
       print()

Parameter Estimation
--------------------

**1. Maximum Likelihood Estimation**

.. code-block:: python

   from bellman_filter_dfsv.core.optimization import run_optimization, FilterType

   # Create initial parameter guess
   initial_params = DFSVParamsDataclass(
       N=5, K=2,
       Lambda=jnp.ones((5, 2)) * 0.5,
       phi_f=jnp.eye(2) * 0.5,
       phi_h=jnp.array([0.9, 0.9]),
       sigma_f=jnp.ones(2),
       sigma_h=jnp.array([0.15, 0.15]),
       sigma_eps=jnp.ones(5) * 0.3,
       mu=jnp.array([-1.0, -1.0])
   )

   # Run optimization using BIF
   result = run_optimization(
       filter_type=FilterType.BELLMAN_INFORMATION,
       returns=returns,
       initial_params=initial_params,
       fix_mu=True,  # Fix log-vol means for identification
       use_transformations=True,  # Use parameter transformations
       optimizer_name="BFGS",
       max_steps=500,
       verbose=True
   )

   print(f"Optimization converged: {result.converged}")
   print(f"Final log-likelihood: {result.final_loglik:.2f}")
   print(f"Number of iterations: {result.num_iterations}")

**2. Parameter Transformations**

.. code-block:: python

   from bellman_filter_dfsv.core.optimization.transformations import (
       transform_params, untransform_params
   )

   # Transform parameters to unconstrained space
   transformed_params = transform_params(params)

   # Untransform back to constrained space
   original_params = untransform_params(transformed_params)

   print("Parameter transformations ensure:")
   print("- Positive variances (log transformation)")
   print("- Stationary AR coefficients (tanh transformation)")
   print("- Proper identification constraints")

Advanced Usage
--------------

**1. Custom Optimization Configuration**

.. code-block:: python

   # Advanced optimization with custom settings
   result = run_optimization(
       filter_type=FilterType.BELLMAN_INFORMATION,
       returns=returns,
       initial_params=initial_params,
       optimizer_name="BFGS",
       learning_rate=1e-3,
       max_steps=1000,
       rtol=1e-6,
       atol=1e-6,
       scheduler_type="warmup_cosine",
       max_learning_rate=1e-2,
       min_learning_rate=1e-6,
       warmup_steps=100,
       log_params=True,  # Log parameter evolution
       verbose=True
   )

**2. Particle Filter Configuration**

.. code-block:: python

   # Particle filter with custom settings
   pf = DFSVParticleFilter(
       N=5, K=2,
       num_particles=5000,  # More particles for better accuracy
       resampling_threshold=0.5,  # ESS threshold for resampling
       seed=42  # Reproducible results
   )

   states, weights, loglik = pf.filter(params, returns)
   print(f"Effective sample size: {pf.effective_sample_size(weights[-1])}")

**3. Numerical Stability Features**

.. code-block:: python

   # BIF automatically handles numerical stability through:
   # - Information form propagation
   # - Joseph form covariance updates
   # - Regularization techniques
   # - Robust matrix operations

   # Access stability diagnostics
   bif = DFSVBellmanInformationFilter(N=5, K=2)
   states, covs, loglik = bif.filter(params, returns)

   # Check for numerical issues
   if hasattr(bif, 'stability_warnings'):
       print(f"Stability warnings: {bif.stability_warnings}")

Performance Tips
----------------

**1. JAX Compilation**

.. code-block:: python

   import jax

   # Enable 64-bit precision for numerical stability
   jax.config.update("jax_enable_x64", True)

   # Use CPU for development, GPU for production
   jax.config.update("jax_platform_name", "cpu")  # or "gpu"

**2. Memory Management**

.. code-block:: python

   # For large datasets, process in chunks
   chunk_size = 1000
   total_loglik = 0.0

   for i in range(0, len(returns), chunk_size):
       chunk = returns[i:i+chunk_size]
       _, _, chunk_loglik = bif.filter(params, chunk)
       total_loglik += chunk_loglik

**3. Batch Processing**

.. code-block:: python

   # Process multiple datasets efficiently
   datasets = [returns1, returns2, returns3]

   # Vectorized processing using JAX
   def batch_filter(params, datasets):
       return jax.vmap(lambda data: bif.filter(params, data))(datasets)

   # Note: Requires careful memory management for large batches