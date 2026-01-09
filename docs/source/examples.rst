.. _examples:

Examples
========

This section provides complete, runnable examples demonstrating the key features of BellmanFilterDFSV. All examples use the current API and are located in the ``examples/`` directory.

Example 1: DFSV Model Simulation
---------------------------------

Demonstrates how to create and simulate data from a DFSV model.

**File**: ``examples/01_dfsv_simulation.py``

**What it demonstrates:**

* Creating model parameters using ``DFSVParams``
* Simulating returns, factors, and log-volatilities with ``simulate_dfsv()``
* Visualizing time series properties

**Key Code:**

.. code-block:: python

   import jax.numpy as jnp
   from bellman_filter_dfsv import DFSVParams, simulate_dfsv

   # Create model parameters (3 series, 1 factor)
   params = DFSVParams(
       lambda_r=jnp.array([[0.8], [0.7], [0.9]]),  # Factor loadings
       Phi_f=jnp.array([[0.7]]),  # Factor persistence
       Phi_h=jnp.array([[0.95]]),  # Volatility persistence
       mu=jnp.array([-1.2]),  # Long-run log-vol mean
       sigma2=jnp.array([0.3, 0.25, 0.35]),  # Idiosyncratic variances
       Q_h=jnp.array([[0.01]])  # Log-vol innovation variance
   )

   # Simulate 1000 time periods
   returns, factors, log_vols = simulate_dfsv(params, T=1000, key=42)

**Running:**

.. code-block:: bash

   uv run python examples/01_dfsv_simulation.py

Example 2: Basic Filtering
---------------------------

Compares the Bellman Information Filter and Particle Filter.

**File**: ``examples/02_basic_filtering.py``

**What it demonstrates:**

* Running ``BellmanFilter`` for fast approximate inference
* Running ``ParticleFilter`` for more flexible inference
* Comparing log-likelihoods and computation times
* Evaluating filtering accuracy against true latent states

**Key Code:**

.. code-block:: python

   from bellman_filter_dfsv import BellmanFilter, ParticleFilter

   # Bellman Information Filter
   bf = BellmanFilter(params)
   bf_result = bf.filter(returns)
   print(f"BIF Log-likelihood: {bf_result.log_likelihood:.2f}")

   # Particle Filter (1000 particles)
   pf = ParticleFilter(params, num_particles=1000)
   pf_result = pf.filter(returns)
   print(f"PF Log-likelihood: {pf_result.log_likelihood:.2f}")

**Output:**

* Comparison table showing log-likelihood and time
* Plots comparing filtered estimates to true states
* Filter accuracy metrics (RMSE, correlation)

**Running:**

.. code-block:: bash

   uv run python examples/02_basic_filtering.py

Example 3: Parameter Estimation (MLE)
--------------------------------------

Maximum likelihood estimation using gradient-based optimization.

**File**: ``examples/03_parameter_optimization.py``

**What it demonstrates:**

* Creating initial parameter guesses
* Using ``fit_mle()`` for gradient-based MLE
* Tracking optimization progress
* Comparing estimated vs. true parameters

**Key Code:**

.. code-block:: python

   from bellman_filter_dfsv import fit_mle

   # Initial guess
   initial_guess = DFSVParams(
       lambda_r=jnp.ones((3, 1)) * 0.5,
       Phi_f=jnp.array([[0.5]]),
       Phi_h=jnp.array([[0.9]]),
       mu=jnp.array([-1.0]),
       sigma2=jnp.ones(3) * 0.3,
       Q_h=jnp.array([[0.02]])
   )

   # Run MLE (100 gradient steps)
   estimated_params, loss_history = fit_mle(
       start_params=initial_guess,
       observations=returns,
       num_steps=100,
       learning_rate=0.01,
   )

**Output:**

* Optimization convergence plot
* Parameter comparison table (true vs. estimated)
* Final log-likelihood

**Running:**

.. code-block:: bash

   uv run python examples/03_parameter_optimization.py

Example 4: EM Algorithm
-----------------------

Expectation-Maximization algorithm using Rao-Blackwellized Particle Smoother.

**File**: ``examples/04_em_algorithm.py``

**What it demonstrates:**

* Using ``fit_em()`` for EM-based parameter estimation
* Rao-Blackwellized Particle Smoother (RBPS) for E-step
* Comparing EM vs. MLE approaches
* Analyzing sufficient statistics

**Key Code:**

.. code-block:: python

   from bellman_filter_dfsv import fit_em

   # Run EM algorithm (10 iterations)
   estimated_params = fit_em(
       start_params=initial_guess,
       observations=returns,
       num_em_steps=10,
       num_particles=500,
       num_trajectories=50,
   )

**Output:**

* EM convergence trajectory
* Parameter evolution across iterations
* Comparison with MLE results

**Running:**

.. code-block:: bash

   uv run python examples/04_em_algorithm.py

**Note:** EM is slower but often more robust than gradient-based MLE, especially with poor initial guesses.

Example 5: Particle Cloud Visualization
----------------------------------------

Visualizes the "uncertainty collapse" phenomenon in particle smoothing.

**File**: ``examples/05_particle_cloud.py``

**What it demonstrates:**

* Running ``run_rbps()`` for Rao-Blackwellized Particle Smoother
* Visualizing particle distributions over time
* Showing how uncertainty collapses after informative observations
* Demonstrating the effect of volatility shocks

**Key Code:**

.. code-block:: python

   from bellman_filter_dfsv import run_rbps

   # Run RBPS to get sampled trajectories
   rbps_result = run_rbps(
       params=params,
       observations=returns,
       num_particles=500,
       num_trajectories=100,
       seed=42,
   )

   # Access sampled log-volatility paths
   h_samples = rbps_result.h_samples  # (num_trajectories, T, K)

**Output:**

* ``particle_cloud_visualization.png`` showing:
  * Particle spread over time
  * Uncertainty collapse after large shocks
  * Comparison with forward filter estimates

**Running:**

.. code-block:: bash

   uv run python examples/05_particle_cloud.py

Running All Examples
--------------------

Run all examples sequentially:

.. code-block:: bash

   cd examples/
   for script in *.py; do
       echo "Running $script..."
       uv run python "$script"
   done

Or individually:

.. code-block:: bash

   uv run python examples/01_dfsv_simulation.py
   uv run python examples/02_basic_filtering.py
   uv run python examples/03_parameter_optimization.py
   uv run python examples/04_em_algorithm.py
   uv run python examples/05_particle_cloud.py

Dependencies for Examples
--------------------------

Examples require additional visualization dependencies:

.. code-block:: bash

   uv pip install matplotlib numpy

Or install all optional dependencies:

.. code-block:: bash

   uv sync --extra examples

Example Output
--------------

Each example produces:

* **Console output**: Statistics, log-likelihoods, parameter estimates
* **Plots** (saved as PNG files): Visualizations of results
* **Timing information**: Performance benchmarks

Expected run times (on modern CPU):

* Example 1 (Simulation): ~1 second
* Example 2 (Filtering): ~5 seconds
* Example 3 (MLE): ~30 seconds
* Example 4 (EM): ~2 minutes
* Example 5 (RBPS): ~1 minute

Customizing Examples
--------------------

All examples are designed to be easily modified. Common modifications:

**Change model dimensions:**

.. code-block:: python

   # More series and factors
   params = DFSVParams(
       lambda_r=jnp.ones((10, 3)),  # 10 series, 3 factors
       Phi_f=jnp.eye(3) * 0.7,
       Phi_h=jnp.eye(3) * 0.95,
       mu=jnp.ones(3) * -1.0,
       sigma2=jnp.ones(10) * 0.2,
       Q_h=jnp.eye(3) * 0.01
   )

**Increase time series length:**

.. code-block:: python

   # Simulate longer series
   returns, _, _ = simulate_dfsv(params, T=5000, key=42)

**Adjust optimization settings:**

.. code-block:: python

   # More iterations, slower learning
   estimated_params, _ = fit_mle(
       start_params=initial_guess,
       observations=returns,
       num_steps=500,  # More steps
       learning_rate=0.005,  # Slower learning rate
   )

**More particles for better accuracy:**

.. code-block:: python

   pf = ParticleFilter(params, num_particles=10000)

See Also
--------

* :ref:`usage` - Detailed usage guide
* :ref:`api_reference` - Complete API documentation
* `README.md <https://github.com/givani30/BellmanFilterDFSV/blob/main/README.md>`_ - Quick start guide
* `ALGORITHMS.md <https://github.com/givani30/BellmanFilterDFSV/blob/main/ALGORITHMS.md>`_ - Mathematical specifications
