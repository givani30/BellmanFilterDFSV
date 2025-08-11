.. _changelog:

Changelog
=========

All notable changes to BellmanFilterDFSV will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[1.0.0] - 2025-08-11
---------------------

**Major Release: Clean, Reusable Package**

This release represents a complete restructuring of the codebase to create a professional, reusable package suitable for both research and production use.

Added
^^^^^

**Core Package Structure**

* New ``bellman_filter_dfsv.core`` module with organized subpackages:

  * ``core.filters``: All filtering algorithms (BIF, Bellman, Particle)
  * ``core.models``: DFSV model definitions and simulation
  * ``core.optimization``: Parameter estimation and optimization utilities

* Clean public API with convenient imports
* Comprehensive type hints using ``jaxtyping``
* Professional package metadata and PyPI classifiers

**Documentation**

* Complete Sphinx documentation with:

  * Installation guide with optional dependencies
  * Comprehensive usage examples
  * Full API reference
  * Contributing guidelines
  * Mathematical background

* Four detailed examples demonstrating key functionality
* Google-style docstrings throughout codebase

**Dependency Management**

* Streamlined core dependencies (13 essential packages)
* Optional dependency groups for different use cases:

  * ``analysis``: Data analysis and visualization
  * ``cloud``: Cloud computing and batch processing  
  * ``notebooks``: Interactive development
  * ``econometrics``: Financial modeling extensions
  * ``dev``: Development and testing tools

**Testing Infrastructure**

* JAX CPU-only configuration for stable testing
* Pytest configuration excluding archived materials
* 76 comprehensive tests covering core functionality
* Automated test discovery and execution

Changed
^^^^^^^

**Package Structure**

* Moved from flat structure to organized ``core/`` hierarchy
* Separated reusable components from thesis-specific code
* Updated all import statements to use new structure
* Consolidated optimization utilities into single module

**Dependencies**

* Reduced from 46 to 13 core dependencies (72% reduction)
* Switched from ``jax[cuda12]`` to ``jax`` for better compatibility
* Removed thesis-specific packages (cloud, notebooks, econometrics)
* Organized optional dependencies into logical groups

**Configuration**

* Updated to version 1.0.0 for professional release
* Enhanced package metadata with proper author and keywords
* Removed redundant ``requirements.txt`` file
* Improved pytest configuration

**Performance**

* JAX array conversions return numpy arrays for better compatibility
* Improved numerical stability in Bellman Information Filter
* CPU-only JAX configuration for consistent behavior

Removed
^^^^^^^

**Thesis Artifacts**

* Moved all thesis-specific materials to ``thesis_artifacts/`` directory:

  * Research documents and PDFs
  * Analysis scripts and simulation studies
  * Batch processing configurations
  * Experimental notebooks and outputs
  * Memory bank with research notes

* Removed duplicate and deprecated code files
* Cleaned up build artifacts and cache files

**Dependencies**

* Removed 33 thesis-specific dependencies including:

  * Cloud computing: ``gcsfs``, ``google-cloud-batch``, ``cloudpickle``
  * Analysis: ``pandas``, ``seaborn``, ``plotly``, ``altair``, ``polars``
  * Notebooks: ``ipykernel``, ``marimo``, ``jupytext``, ``notebook``
  * Econometrics: ``statsmodels``, ``arch``, ``mgarch``, ``scikit-learn``
  * Development: ``python-lsp-server``, ``rich``, ``tqdm``

Fixed
^^^^^

**Import Issues**

* Resolved pytest discovery errors
* Fixed all import statements to use new package structure
* Corrected relative imports within core modules
* Updated examples and tests to use new imports

**Numerical Stability**

* Fixed JAX array type conversions in filter outputs
* Improved error handling in optimization routines
* Enhanced parameter transformation robustness
* Better handling of edge cases in filtering algorithms

**Testing**

* Configured JAX for CPU-only testing to avoid GPU memory issues
* Fixed test assertions to handle new return types
* Resolved import errors in test files
* Improved test isolation and reproducibility

Security
^^^^^^^^

* No security-related changes in this release

Migration Guide
^^^^^^^^^^^^^^^

**For Existing Users**

If you were using the previous version, update your imports:

.. code-block:: python

   # Old imports
   from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
   from bellman_filter_dfsv.filters.bellman_information import DFSVBellmanInformationFilter
   from bellman_filter_dfsv.utils.optimization import run_optimization
   
   # New imports  
   from bellman_filter_dfsv.core.models import DFSVParamsDataclass
   from bellman_filter_dfsv.core.filters import DFSVBellmanInformationFilter
   from bellman_filter_dfsv.core.optimization import run_optimization
   
   # Or use convenient top-level imports
   import bellman_filter_dfsv as bfdfsv
   filter = bfdfsv.DFSVBellmanInformationFilter(N=3, K=1)

**Installation Changes**

.. code-block:: bash

   # Install core package only
   pip install bellman-filter-dfsv
   
   # Install with optional dependencies
   pip install bellman-filter-dfsv[analysis,notebooks]

[0.1.0] - 2025-03-31
---------------------

**Initial Development Release**

Added
^^^^^

* Initial implementation of DFSV filtering algorithms
* Basic Sphinx documentation structure
* Core mathematical implementations
* Thesis-specific analysis scripts

Note
^^^^

This changelog starts from version 1.0.0, which represents the first clean, reusable release. Previous development was thesis-specific and has been archived.
