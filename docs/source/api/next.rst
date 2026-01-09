.. _api_next:

New Architecture (v2)
=====================

This module contains the new **Functional Core + Equinox** architecture (v2). 
This is the recommended API for all new projects. It offers:

*   **Strict Type Safety**: Uses ``jaxtyping`` to ensure shape correctness.
*   **Zero-Cost Abstraction**: Re-creating filters inside JIT loops is free.
*   **Stable Smoother**: Implements the direct information form RTS smoother.
*   **Optimization Utility**: Easy ``fit_mle`` wrapper.

Filters
-------

.. automodule:: bellman_filter_dfsv.next.filters
   :members:
   :undoc-members:
   :show-inheritance:

Optimization
------------

.. automodule:: bellman_filter_dfsv.next.optimization
   :members:
   :undoc-members:
   :show-inheritance:

Smoother
--------

.. automodule:: bellman_filter_dfsv.next.smoother
   :members:
   :undoc-members:
   :show-inheritance:

Types
-----

.. automodule:: bellman_filter_dfsv.next.types
   :members:
   :undoc-members:
   :show-inheritance:

Kernels (Pure Functions)
------------------------

.. automodule:: bellman_filter_dfsv.next.kernels
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: bellman_filter_dfsv.next.particle_kernels
   :members:
   :undoc-members:
   :show-inheritance:
