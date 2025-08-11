BellmanFilterDFSV Documentation
================================

**High-performance JAX-based filtering for Dynamic Factor Stochastic Volatility (DFSV) models**

BellmanFilterDFSV is a Python package that provides efficient implementations of filtering algorithms for Dynamic Factor Stochastic Volatility models using JAX for automatic differentiation and JIT compilation.

Key Features
------------

* **Multiple Filtering Algorithms**: Bellman Information Filter (BIF), Bellman Filter, and Particle Filter
* **JAX-Powered Performance**: Automatic differentiation, JIT compilation, and vectorization
* **Numerical Stability**: Advanced techniques for robust parameter estimation
* **Clean API**: Intuitive interface for research and applications
* **Extensible Design**: Easy to adapt for other state-space models

Quick Start
-----------

.. code-block:: python

   import bellman_filter_dfsv as bfdfsv
   from bellman_filter_dfsv.core import DFSVParamsDataclass, simulate_DFSV

   # Define model parameters
   params = DFSVParamsDataclass(N=3, K=1, ...)

   # Simulate data
   returns, factors, log_vols = simulate_DFSV(params, T=500)

   # Create and run filter
   filter = bfdfsv.DFSVBellmanInformationFilter(N=3, K=1)
   states, covs, loglik = filter.filter(params, returns)

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
