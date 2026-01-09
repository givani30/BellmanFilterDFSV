.. _installation:

Installation
============

Requirements
------------

* Python 3.11 or higher
* JAX and JAXlib
* NumPy and SciPy
* Equinox (for functional architecture)

Basic Installation
------------------

Install the core package with minimal dependencies:

.. code-block:: bash

   pip install bellman-filter-dfsv

This installs only the essential dependencies needed for core filtering functionality.

Optional Dependencies
---------------------

For additional functionality, you can install optional dependency groups:

**Data Analysis and Visualization Examples**

.. code-block:: bash

   pip install bellman-filter-dfsv[examples]

Includes: matplotlib, seaborn, pandas, yfinance (used in example scripts)

**Development Tools**

.. code-block:: bash

   pip install bellman-filter-dfsv[dev]

Includes: pytest, pytest-cov, hypothesis, ruff, basedpyright

**Documentation Tools**

.. code-block:: bash

   pip install bellman-filter-dfsv[docs]

Includes: sphinx, sphinx-rtd-theme

**All Optional Dependencies**

.. code-block:: bash

   pip install bellman-filter-dfsv[all]

Development Installation
------------------------

For development, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/givani30/BellmanFilterDFSV.git
   cd BellmanFilterDFSV
   pip install -e .[dev,all]

Or using uv (recommended):

.. code-block:: bash

   git clone https://github.com/givani30/BellmanFilterDFSV.git
   cd BellmanFilterDFSV
   uv sync
   uv run pytest  # Run tests

Verification
------------

Verify your installation by running:

.. code-block:: python

   import bellman_filter_dfsv
   print(f"BellmanFilterDFSV version: {bellman_filter_dfsv.__version__}")

   # Test core functionality
   import jax.numpy as jnp
   from bellman_filter_dfsv import DFSVParams, simulate_dfsv
   
   params = DFSVParams(
       lambda_r=jnp.array([[0.8], [0.7]]),
       Phi_f=jnp.array([[0.7]]),
       Phi_h=jnp.array([[0.95]]),
       mu=jnp.array([-1.2]),
       sigma2=jnp.array([0.3, 0.25]),
       Q_h=jnp.array([[0.01]])
   )
   returns, factors, log_vols = simulate_dfsv(params, T=100)
   print("✅ Installation successful!")

Troubleshooting
---------------

**JAX Installation Issues**

If you encounter JAX-related errors, ensure you have the correct JAX version (>=0.4.35):

.. code-block:: bash

   pip install --upgrade jax jaxlib

**Import Errors**

If you get import errors, make sure you've installed the package correctly and are using a supported Python version (>=3.11):

.. code-block:: bash

   pip install --upgrade bellman-filter-dfsv

**GPU Support**

The package uses CPU-only JAX by default for better compatibility. For GPU support, install JAX with CUDA:

.. code-block:: bash

   pip install -U "jax[cuda12]"
