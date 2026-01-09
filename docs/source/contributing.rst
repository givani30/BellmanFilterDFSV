.. _contributing:

Contributing
============

We welcome contributions to BellmanFilterDFSV! This guide will help you get started.

Development Setup
-----------------

**1. Clone the Repository**

.. code-block:: bash

   git clone https://github.com/givani30/BellmanFilterDFSV.git
   cd BellmanFilterDFSV

**2. Install Development Dependencies**

Using uv (recommended):

.. code-block:: bash

   uv sync
   uv run pytest  # Run tests to verify setup

Using pip:

.. code-block:: bash

   pip install -e .[dev,all]
   pytest

**3. Set up Pre-commit Hooks** (optional)

.. code-block:: bash

   pre-commit install

Code Style
----------

We follow these coding standards:

* **PEP 8** for Python code style
* **Google style** for docstrings
* **Functional programming** style using Equinox and JAX
* **Type safety** using jaxtyping and basedpyright

**Linting and Formatting Tools**

We use **ruff** for both linting and formatting, and **basedpyright** for type checking.

.. code-block:: bash

   # Format and lint code
   uv run ruff check --fix .
   uv run ruff format .
   
   # Type checking
   uv run basedpyright src/

Testing
-------

**Running Tests**

.. code-block:: bash

   # Run all tests
   uv run pytest
   
   # Run specific test file
   uv run pytest tests/test_filters.py
   
   # Run with coverage
   uv run pytest --cov=bellman_filter_dfsv

**Writing Tests**

* Place tests in the ``tests/`` directory
* Use descriptive test names: ``test_bellman_filter_convergence``
* Include docstrings explaining what the test validates
* Use fixtures from ``tests/conftest.py`` for common setup
* Use **Hypothesis** for property-based testing of mathematical identities

Documentation
-------------

**Building Documentation**

.. code-block:: bash

   cd docs/
   make html
   # Open docs/build/html/index.html

**Documentation Standards**

* Use **Google style docstrings** for all public functions
* Include mathematical notation using LaTeX: ``.. math::``
* Provide usage examples in docstrings
* Update API documentation when adding new modules

**Example Docstring**

.. code-block:: python

   def run_filter(filter_module: BellmanFilter, returns: jnp.ndarray) -> FilterResult:
       """Run Bellman filter on DFSV model.
       
       This function implements the Bellman filter for Dynamic Factor 
       Stochastic Volatility models using JAX for efficient computation.
       
       Args:
           filter_module: Initialized BellmanFilter Equinox module.
           returns: Observed return data of shape (T, N) where T is
               the number of time periods and N is the number of series.
               
       Returns:
           A FilterResult namedtuple containing filtered means, 
           information matrices, and the log-likelihood.
            
       Example:
           >>> params = DFSVParams(...)
           >>> bf = BellmanFilter(params)
           >>> result = bf.filter(returns)
           >>> print(f"Log-likelihood: {result.log_likelihood:.2f}")
       """

Contributing Guidelines
-----------------------

**1. Issue Reporting**

Before submitting a bug report or feature request:

* Check existing issues to avoid duplicates
* Provide minimal reproducible examples
* Include system information (Python version, JAX version, OS)

**2. Pull Requests**

* Fork the repository and create a feature branch
* Write tests for new functionality
* Ensure all tests pass
* Update documentation as needed
* Follow the code style guidelines

**Pull Request Process**

.. code-block:: bash

   # 1. Create feature branch
   git checkout -b feature/new-filter-algorithm
   
   # 2. Make changes and commit
   git add .
   git commit -m "feat: add new filter algorithm with tests"
   
   # 3. Run tests
   uv run pytest
   
   # 4. Push and create PR
   git push origin feature/new-filter-algorithm

**3. Code Review**

All contributions go through code review:

* Ensure code follows style guidelines (Ruff/Pyright)
* Verify tests provide adequate coverage
* Check documentation is complete and accurate
* Validate numerical correctness for mathematical code

Areas for Contribution
----------------------

**High Priority**

* Performance optimizations for high-dimensional factor models
* Real-world examples with financial datasets
* GPU acceleration benchmarks and improvements

**Medium Priority**

* Alternative optimization algorithms for the EM M-step
* Model diagnostics and validation tools (e.g., residual analysis)
* Visualization utilities for particle clouds

Getting Help
------------

* **GitHub Issues**: For bug reports and feature requests
* **Discussions**: For questions and general discussion
* **Email**: givaniboek@hotmail.com for direct contact

Recognition
-----------

Contributors will be acknowledged in the release notes for significant contributions.

Thank you for contributing to BellmanFilterDFSV!
