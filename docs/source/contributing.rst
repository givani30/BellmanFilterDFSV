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
* **Functional programming** style where possible
* **JAX best practices** for numerical computing

**Formatting Tools**

.. code-block:: bash

   # Format code
   black src/ tests/ examples/
   
   # Check style
   flake8 src/ tests/ examples/
   
   # Type checking
   mypy src/

Testing
-------

**Running Tests**

.. code-block:: bash

   # Run all tests
   uv run pytest
   
   # Run specific test file
   uv run pytest tests/test_dfsv_models.py
   
   # Run with coverage
   uv run pytest --cov=bellman_filter_dfsv

**Writing Tests**

* Place tests in the ``tests/`` directory
* Use descriptive test names: ``test_bellman_filter_convergence``
* Include docstrings explaining what the test validates
* Use fixtures from ``tests/conftest.py`` for common setup

**Test Categories**

* **Unit tests**: Test individual functions and classes
* **Integration tests**: Test component interactions
* **Performance tests**: Validate computational efficiency
* **Numerical tests**: Verify mathematical correctness

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

   def bellman_filter(params: DFSVParamsDataclass, returns: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
       """Run Bellman filter on DFSV model.
       
       This function implements the Bellman filter for Dynamic Factor 
       Stochastic Volatility models using JAX for efficient computation.
       
       Args:
           params: DFSV model parameters containing factor loadings,
               AR coefficients, and variance parameters.
           returns: Observed return data of shape (T, N) where T is
               the number of time periods and N is the number of series.
               
       Returns:
           A tuple containing:
           - states: Filtered state estimates of shape (T, 2*K)
           - covs: Filtered covariances of shape (T, 2*K, 2*K)  
           - loglik: Total log-likelihood value
           
       Example:
           >>> params = DFSVParamsDataclass(N=3, K=1, ...)
           >>> returns = jnp.array([[0.01, 0.02, -0.01], ...])
           >>> states, covs, loglik = bellman_filter(params, returns)
           >>> print(f"Log-likelihood: {loglik:.2f}")
           
       Note:
           This implementation uses the Joseph form for numerical stability.
           See Lange (2024) for mathematical details.
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
   git commit -m "Add new filter algorithm with tests"
   
   # 3. Run tests
   uv run pytest
   
   # 4. Push and create PR
   git push origin feature/new-filter-algorithm

**3. Code Review**

All contributions go through code review:

* Ensure code follows style guidelines
* Verify tests provide adequate coverage
* Check documentation is complete and accurate
* Validate numerical correctness for mathematical code

Areas for Contribution
----------------------

**High Priority**

* Additional filtering algorithms (Extended Kalman Filter, Unscented Kalman Filter)
* Performance optimizations and benchmarking
* Real-world examples with financial datasets
* GPU acceleration improvements

**Medium Priority**

* Alternative optimization algorithms
* Model diagnostics and validation tools
* Visualization utilities
* Integration with other econometric packages

**Documentation**

* Tutorial notebooks
* Mathematical background documentation
* Performance comparison studies
* Best practices guides

**Testing**

* Edge case testing
* Numerical stability tests
* Performance regression tests
* Cross-platform compatibility tests

Getting Help
------------

* **GitHub Issues**: For bug reports and feature requests
* **Discussions**: For questions and general discussion
* **Email**: givaniboek@hotmail.com for direct contact

**Before Asking for Help**

1. Check the documentation and examples
2. Search existing issues and discussions
3. Try to create a minimal reproducible example
4. Include relevant error messages and system information

Recognition
-----------

Contributors will be acknowledged in:

* The ``CONTRIBUTORS.md`` file
* Release notes for significant contributions
* Academic papers when appropriate

Thank you for contributing to BellmanFilterDFSV!
