.. _changelog:

Changelog
=========

All notable changes to BellmanFilterDFSV will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[2.0.0] - 2026-01-09
---------------------

**Major Release: Functional Architecture Migration**

This release represents a complete rewrite of the package to adopt a functional architecture using JAX and Equinox. The codebase is now more modular, easier to test, and fully compatible with JAX transformations (JIT, grad, vmap).

Added
^^^^^

* **Functional Architecture**: Complete rewrite using Equinox modules for filters.
* **New EM Algorithm**: High-performance EM implementation using Rao-Blackwellized Particle Smoother.
* **Advanced Smoothing**: Added Rao-Blackwellized Particle Smoother (RBPS) for efficient state estimation.
* **Full Type Safety**: Integrated `jaxtyping` and `basedpyright` for static type checking across the entire codebase.
* **Streamlined API**: Flattened package structure for easier access to core components.

Changed
^^^^^^^

* **Package Structure**: Moved away from the `core/` subpackage hierarchy to a flatter, more direct structure.
* **Parameter Classes**: Replaced `DFSVParamsDataclass` with `DFSVParams` (NamedTuple) for better JAX compatibility.
* **Dependency Management**: Switched to `uv` for environment management and `ruff` for linting/formatting.
* **Testing**: Achieved 93% test coverage with 69 comprehensive tests, including property-based tests.

Removed
^^^^^^^

* Old `bellman_filter_dfsv.core` subpackage.
* Redundant optional dependency groups (`cloud`, `notebooks`, `econometrics`, `analysis`) in favor of a simpler set (`dev`, `docs`, `examples`, `all`).

[1.0.0] - 2025-08-11
---------------------

**Major Release: Clean, Reusable Package**

This release represented the first clean, reusable release of the package.

Added
^^^^^

* New ``bellman_filter_dfsv.core`` module with organized subpackages.
* Professional package metadata and PyPI classifiers.
* Complete Sphinx documentation.

[0.1.0] - 2025-03-31
---------------------

**Initial Development Release**

Added
^^^^^

* Initial implementation of DFSV filtering algorithms.
* Basic Sphinx documentation structure.
