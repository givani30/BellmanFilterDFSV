# BellmanFilterDFSV - Dynamic Factor Stochastic Volatility (DFSV) Models in JAX

This repository contains the Python code for a Quantitative Finance thesis project focused on implementing, simulating, and filtering Dynamic Factor Stochastic Volatility (DFSV) models using JAX for enhanced performance.

## Project Purpose

DFSV models are used in finance to model the time-varying volatility and correlation of multiple asset returns by relating them to a smaller set of unobserved latent factors, each with its own stochastic volatility process. This project provides tools to:

*   Define DFSV model parameters.
*   Simulate realistic financial time series data based on DFSV models.
*   Implement and compare filtering techniques (e.g., Particle Filter, Bellman Filter approximations) to estimate the latent states (factors and volatilities) from observed returns.
*   Leverage JAX for automatic differentiation, JIT compilation, and hardware acceleration (CPU/GPU/TPU), making simulations and potentially complex estimation procedures more efficient.

## Project Status

**Status:** Active Development

This project is currently under active development. Features and APIs might change.

## Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url> # Replace <repository-url> with the actual URL
    cd BellmanFilterDFSV
    ```

2.  **Create and activate a virtual environment:** (Recommended)
    ```bash
    # Using venv (standard library)
    python -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    # .\.venv\Scripts\activate  # On Windows

    # Or using conda
    # conda create -n bellman_filter_dfsv python=3.10 # Or your preferred Python version >= 3.10
    # conda activate qf_thesis
    ```

3.  **Install dependencies and the package:**
    This project uses `uv` for potentially faster dependency management, but `pip` works perfectly well. The dependencies are listed in `pyproject.toml`. Install the package in editable mode (`-e`) so changes in the `src/` directory are immediately reflected.
    ```bash
    # Using uv (if installed)
    uv pip install -e .

    # Using pip (standard)
    pip install -e .
    ```

## Usage Example: Simulating a Simple DFSV Model

The following example demonstrates how to define parameters for a DFSV model (with K=1 factor and N=3 observed series) and simulate data using the `bellman_filter_dfsv` package. This is based on `scripts/simple_dfsv_example.py`.

```python
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass
from bellman_filter_dfsv.core.simulation import simulate_DFSV

# 1. Define Model Parameters (K=1, N=3)
params = DFSVParamsDataclass(
    N=3,
    K=1,
    lambda_r=jnp.array([[0.8], [0.6], [0.4]]), # Factor loadings
    Phi_f=jnp.array([[0.95]]),                # Factor state transition
    Phi_h=jnp.array([[0.98]]),                # Log-vol state transition
    mu=jnp.array([0.0]),                      # Log-vol long-run mean
    sigma2=jnp.array([0.2, 0.3, 0.4]),        # Idiosyncratic variances
    Q_h=jnp.array([[0.1]])                    # Log-vol noise covariance
)

# 2. Simulation Settings
T = 500  # Number of time periods
seed = 42 # For reproducibility

# 3. Simulate Data
returns, factors, log_vols = simulate_DFSV(params, T=T, seed=seed)

# 4. (Optional) Plotting Results (as done in the script)
# This part generates plots similar to the one saved in outputs/simple_dfsv_example.png
print(f"Simulated returns shape: {returns.shape}") # Output: (500, 3)
print(f"Simulated factors shape: {factors.shape}") # Output: (500, 1)
print(f"Simulated log-vols shape: {log_vols.shape}") # Output: (500, 1)

# Example plot generation (requires matplotlib)
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(factors, label='Latent Factor')
axes[0].set_title('Latent Factor')
axes[0].legend()
axes[1].plot(log_vols, label='Log Volatility', color='orange')
axes[1].set_title('Log Volatility')
axes[1].legend()
axes[2].plot(returns[:, 0], label='Returns Series 1', alpha=0.7)
axes[2].plot(returns[:, 1], label='Returns Series 2', alpha=0.7)
axes[2].plot(returns[:, 2], label='Returns Series 3', alpha=0.7)
axes[2].set_title('Simulated Returns')
axes[2].legend()
plt.tight_layout()
plt.show() # Or plt.savefig('outputs/simple_dfsv_simulation.png')

```

Explore the `scripts/` and `examples/` directories for more detailed use cases, including filter implementations and parameter estimation examples.

## Project Structure

```python
BellmanFilterDFSV/
├── .clinerules           # Custom rules for Roo AI assistant
├── .coveragerc           # Coverage configuration
├── .gitignore            # Git ignore rules
├── pyproject.toml        # Build system, dependencies, project metadata
├── README.md             # This file
├── requirements.txt      # (Potentially redundant with pyproject.toml)
├── data/                 # Input data files (if any)
├── examples/             # Example usage scripts
├── notebooks/            # Jupyter notebooks for exploration/visualization
├── outputs/              # Generated outputs (plots, results)
├── scripts/              # Standalone scripts for running experiments, analysis
├── src/                  # Source code directory
│   └── bellman_filter_dfsv/ # Main package
│       ├── __init__.py
│       ├── core/           # Core algorithms (filters, simulation, likelihood)
│       │   └── filters/      # Filter implementations (Particle, Bellman, etc.)
│       ├── models/         # Model definitions (e.g., DFSV parameter dataclass)
│       └── utils/          # Utility functions (e.g., transformations)
├── tests/                # Unit and integration tests
└── .venv/                # Virtual environment directory (if created)
```

## Running Tests

Tests are implemented using `pytest` and are located in the `tests/` directory. Ensure your virtual environment is active and dependencies are installed.

```bash
pytest .
```

To run tests with coverage:

```bash
coverage run -m pytest .
coverage report -m
```

(Requires `coverage` package: `pip install coverage`)

## Contributing

Contributions are welcome! If you find a bug, have a suggestion, or want to contribute code, please follow these guidelines:

1.  **Issues:** Check the existing issues on the repository. If your issue isn't listed, please open a new one describing the bug or feature request.
2.  **Pull Requests:** For code contributions, please fork the repository, create a new branch for your feature or fix, and submit a pull request. Ensure your code adheres to the project's style and includes tests where appropriate.
3.  **Questions:** Feel free to open an issue for questions about the code or methodology.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file (if one is added later) or the standard MIT License text for details.