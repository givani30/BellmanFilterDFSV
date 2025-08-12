.. _installation:

Installation
============

Requirements
------------

* Python 3.12 or higher
* JAX and JAXlib
* NumPy and SciPy

Basic Installation
------------------

Install the core package with minimal dependencies:

.. code-block:: bash

   pip install bellman-filter-dfsv

This installs only the essential dependencies needed for core filtering functionality.

Optional Dependencies
---------------------

For additional functionality, you can install optional dependency groups:

**Data Analysis and Visualization**

.. code-block:: bash

   pip install bellman-filter-dfsv[analysis]

Includes: pandas, seaborn, plotly, altair, tabulate

**Cloud Computing and Batch Processing**

.. code-block:: bash

   pip install bellman-filter-dfsv[cloud]

Includes: gcsfs, google-cloud-batch, cloudpickle, pyarrow

**Notebook and Interactive Development**

.. code-block:: bash

   pip install bellman-filter-dfsv[notebooks]

Includes: jupyter, ipykernel, marimo, jupytext, notebook

**Econometrics and Financial Modeling**

.. code-block:: bash

   pip install bellman-filter-dfsv[econometrics]

Includes: statsmodels, arch, mgarch, scikit-learn

**Development Tools**

.. code-block:: bash

   pip install bellman-filter-dfsv[dev]

Includes: pytest, pytest-cov, black, flake8, mypy

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
   from bellman_filter_dfsv.core import DFSVParamsDataclass, simulate_DFSV
   params = DFSVParamsDataclass(N=2, K=1)
   returns, factors, log_vols = simulate_DFSV(params, T=100)
   print("✅ Installation successful!")

Troubleshooting
---------------

**JAX Installation Issues**

If you encounter JAX-related errors, ensure you have the correct JAX version:

.. code-block:: bash

   pip install --upgrade jax jaxlib

**Import Errors**

If you get import errors, make sure you've installed the package correctly:

.. code-block:: bash

   pip install --upgrade bellman-filter-dfsv

**GPU Support**

The package uses CPU-only JAX by default for better compatibility. For GPU support, install JAX with CUDA:

.. code-block:: bash

   pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html