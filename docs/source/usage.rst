.. _usage:

Usage Guide
===========

This guide provides comprehensive examples of how to use BellmanFilterDFSV for filtering and parameter estimation in Dynamic Factor Stochastic Volatility models.

Basic Concepts
--------------

**Dynamic Factor Stochastic Volatility (DFSV) Model**

The DFSV model consists of three key equations:

**Observation Equation:**

.. math::

   r_t = \lambda_r f_t + e_t, \quad e_t \sim \mathcal{N}(0, \Sigma)

**Factor Dynamics:**

.. math::

   f_t = \Phi_f f_{t-1} + \text{diag}(\exp(h_t/2)) \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, I_K)

**Log-Volatility Dynamics:**

.. math::

   h_t = \mu + \Phi_h (h_{t-1} - \mu) + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q_h)

where:

* :math:`r_t` are observed returns (N×1)
* :math:`f_t` are latent factors with stochastic volatility (K×1)
* :math:`h_t` are log-volatilities of factors (K×1)
* :math:`\lambda_r` is the factor loading matrix (N×K)
* :math:`\Phi_f` is the factor autoregression matrix (K×K)
* :math:`\Phi_h` is the log-volatility autoregression matrix (K×K)
* :math:`\mu` is the long-run mean of log-volatilities (K×1)
* :math:`\Sigma` is the idiosyncratic error covariance (N×N, diagonal)
* :math:`Q_h` is the log-volatility innovation covariance (K×K)

**Filtering Algorithms**

The package provides two main filtering algorithms:

1. **Bellman Information Filter (BIF)**: Information-form implementation for numerical stability
2. **Particle Filter**: Bootstrap particle filter with systematic resampling

Basic Usage
-----------

**1. Model Definition and Simulation**

.. code-block:: python

   import jax.numpy as jnp
   from bellman_filter_dfsv import DFSVParams, simulate_dfsv

   # Define model parameters
   params = DFSVParams(
       lambda_r=jnp.array([[0.8], [0.7], [0.9]]),  # Factor loadings (N×K)
       Phi_f=jnp.array([[0.7]]),  # Factor AR matrix (K×K)
       Phi_h=jnp.array([[0.95]]),  # Log-vol AR matrix (K×K)
       mu=jnp.array([-1.2]),  # Long-run mean of log-vols (K,)
       sigma2=jnp.array([0.3, 0.25, 0.35]),  # Idiosyncratic variances (N,)
       Q_h=jnp.array([[0.01]])  # Log-vol innovation covariance (K×K)
   )

   # Simulate data (T=1000 time periods)
   returns, factors, log_vols = simulate_dfsv(params, T=1000, key=42)
   print(f"Simulated {returns.shape[0]} time periods for {returns.shape[1]} series")

**2. Filtering with Bellman Information Filter**

.. code-block:: python

   from bellman_filter_dfsv import BellmanFilter

   # Create filter (params embedded in filter)
   bf = BellmanFilter(params)

   # Run filtering
   result = bf.filter(returns)

   print(f"Log-likelihood: {result.log_likelihood:.2f}")
   print(f"Filtered states shape: {result.means.shape}")  # (T, 2*K)
   print(f"Information matrices shape: {result.infos.shape}")  # (T, 2*K, 2*K)

**3. Filtering with Particle Filter**

.. code-block:: python

   from bellman_filter_dfsv import ParticleFilter

   # Create particle filter with 1000 particles
   pf = ParticleFilter(params, num_particles=1000)

   # Run filtering
   result = pf.filter(returns)

   print(f"Log-likelihood: {result.log_likelihood:.2f}")
   print(f"Filtered means shape: {result.means.shape}")  # (T, 2*K)
   print(f"Filtered covariances shape: {result.covs.shape}")  # (T, 2*K, 2*K)

Parameter Estimation
--------------------

**1. Maximum Likelihood Estimation with Gradient Descent**

.. code-block:: python

   from bellman_filter_dfsv import fit_mle

   # Create initial parameter guess
   initial_guess = DFSVParams(
       lambda_r=jnp.ones((3, 1)) * 0.5,
       Phi_f=jnp.array([[0.5]]),
       Phi_h=jnp.array([[0.9]]),
       mu=jnp.array([-1.0]),
       sigma2=jnp.ones(3) * 0.3,
       Q_h=jnp.array([[0.02]])
   )

   # Run MLE optimization
   estimated_params, loss_history = fit_mle(
       start_params=initial_guess,
       observations=returns,
       num_steps=100,
       learning_rate=0.01,
   )

   print(f"Final negative log-likelihood: {loss_history[-1]:.2f}")
   print(f"Estimated factor loadings:\n{estimated_params.lambda_r}")

**2. Expectation-Maximization Algorithm**

.. code-block:: python

   from bellman_filter_dfsv import fit_em

   # Run EM algorithm (uses Rao-Blackwellized Particle Smoother)
   estimated_params = fit_em(
       start_params=initial_guess,
       observations=returns,
       num_em_steps=10,
       num_particles=500,
       num_trajectories=50,
   )

   print(f"EM estimated parameters:")
   print(f"  Lambda_r:\n{estimated_params.lambda_r}")
   print(f"  Phi_f: {estimated_params.Phi_f}")
   print(f"  Phi_h: {estimated_params.Phi_h}")

Advanced Usage
--------------

**1. Smoothing with RTS Smoother**

.. code-block:: python

   from bellman_filter_dfsv import BellmanFilter, rts_smoother

   # First run the forward filter
   bf = BellmanFilter(params)
   filter_result = bf.filter(returns)

   # Then run backward smoother
   smooth_means, smooth_infos = rts_smoother(
       filter_means=filter_result.means,
       filter_infos=filter_result.infos,
       params=params
   )

   print(f"Smoothed states shape: {smooth_means.shape}")

**2. Rao-Blackwellized Particle Smoother**

.. code-block:: python

   from bellman_filter_dfsv import run_rbps

   # Run RBPS (marginalizes out linear states analytically)
   rbps_result = run_rbps(
       params=params,
       observations=returns,
       num_particles=500,
       num_trajectories=100,
       seed=42
   )

   # Access sampled trajectories
   h_samples = rbps_result.h_samples  # (num_trajectories, T, K)
   f_means = rbps_result.f_smooth_means  # (num_trajectories, T, K)

   print(f"Sampled log-vol paths: {h_samples.shape}")
   print(f"Conditional factor means: {f_means.shape}")

**3. Custom Optimization with Optax**

.. code-block:: python

   import optax
   from bellman_filter_dfsv import fit_mle

   # Use custom optimizer (e.g., Adam with learning rate schedule)
   schedule = optax.exponential_decay(
       init_value=0.01,
       transition_steps=50,
       decay_rate=0.9
   )
   optimizer = optax.adam(learning_rate=schedule)

   # Run MLE with custom optimizer
   estimated_params, loss_history = fit_mle(
       start_params=initial_guess,
       observations=returns,
       num_steps=200,
       optimizer=optimizer
   )

Performance Tips
----------------

**1. JAX Configuration**

.. code-block:: python

   import jax
   
   # Enable 64-bit precision for numerical stability
   jax.config.update("jax_enable_x64", True)
   
   # Use CPU for development, GPU for large-scale work
   jax.config.update("jax_platform_name", "cpu")  # or "gpu"

**2. JIT Compilation**

All filtering and estimation functions are JIT-compatible:

.. code-block:: python

   import jax
   from bellman_filter_dfsv import BellmanFilter
   
   # JIT-compile the filter for speed
   bf = BellmanFilter(params)
   jitted_filter = jax.jit(bf.filter)
   
   # First call compiles, subsequent calls are fast
   result = jitted_filter(returns)

**3. Particle Filter Configuration**

.. code-block:: python

   from bellman_filter_dfsv import ParticleFilter
   
   # More particles = better accuracy but slower
   pf = ParticleFilter(
       params,
       num_particles=5000,  # Increase for better estimates
   )
   
   result = pf.filter(returns)

**4. Memory-Efficient Processing**

For very long time series, consider chunking:

.. code-block:: python

   chunk_size = 1000
   
   for i in range(0, len(returns), chunk_size):
       chunk = returns[i:i+chunk_size]
       result = bf.filter(chunk)
       # Process result...

Troubleshooting
---------------

**Numerical Stability Issues**

If you encounter numerical issues:

1. Enable 64-bit precision: ``jax.config.update("jax_enable_x64", True)``
2. Use BellmanFilter (information form) instead of ParticleFilter
3. Check parameter constraints (stationarity, positive variances)
4. Reduce learning rate in ``fit_mle()``

**Optimization Not Converging**

If MLE doesn't converge:

1. Try different initial parameter values
2. Increase ``num_steps`` in ``fit_mle()``
3. Reduce ``learning_rate``
4. Use EM algorithm (``fit_em()``) instead - more robust but slower
5. Check that data has sufficient variation

**Memory Issues with Particle Filter**

If running out of memory:

1. Reduce ``num_particles``
2. Process data in chunks
3. Use BellmanFilter instead (much more memory-efficient)

See Also
--------

* :ref:`examples` - Complete runnable examples
* :ref:`api_reference` - Full API documentation
* `ALGORITHMS.md <https://github.com/givani30/BellmanFilterDFSV/blob/main/ALGORITHMS.md>`_ - Mathematical specifications
