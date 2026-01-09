BellmanFilterDFSV Documentation
================================

**High-performance JAX-based filtering for Dynamic Factor Stochastic Volatility (DFSV) models**

BellmanFilterDFSV is a Python package that provides efficient implementations of filtering algorithms for Dynamic Factor Stochastic Volatility models using JAX for automatic differentiation and JIT compilation.

Key Features
------------

* **Functional Architecture**: Built with Equinox for clean, composable JAX code
* **Multiple Filtering Algorithms**: Bellman Information Filter and Particle Filter
* **Advanced Smoothing**: RTS Smoother and Rao-Blackwellized Particle Smoother
* **Parameter Estimation**: MLE (gradient descent) and EM algorithm
* **Full Type Safety**: Complete jaxtyping annotations (0 type errors)
* **93% Test Coverage**: Comprehensive test suite with property-based tests
* **JIT Compilation**: Fully compatible with JAX transformations

Quick Start
-----------

.. code-block:: python

   import jax.numpy as jnp
   from bellman_filter_dfsv import DFSVParams, BellmanFilter, simulate_dfsv

   # Define model parameters
   params = DFSVParams(
       lambda_r=jnp.array([[0.8], [0.7], [0.9]]),
       Phi_f=jnp.array([[0.7]]),
       Phi_h=jnp.array([[0.95]]),
       mu=jnp.array([-1.2]),
       sigma2=jnp.array([0.3, 0.25, 0.35]),
       Q_h=jnp.array([[0.01]])
   )

   # Simulate data
   returns, factors, log_vols = simulate_dfsv(params, T=500, key=42)

   # Create and run filter
   bf = BellmanFilter(params)
   result = bf.filter(returns)
   print(f"Log-likelihood: {result.log_likelihood:.2f}")

Installation
------------

.. code-block:: bash

   pip install bellman-filter-dfsv

Or with optional dependencies:

.. code-block:: bash

   pip install bellman-filter-dfsv[all]

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   usage
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development:

   contributing
   changelog

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
