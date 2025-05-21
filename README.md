# BellmanFilterDFSV - Dynamic Factor Stochastic Volatility (DFSV) Models in JAX

This repository contains Python code for Dynamic Factor Stochastic Volatility (DFSV) models, leveraging JAX for enhanced performance.

## Project Purpose

This project addresses the modeling of time-varying volatility and correlation in financial asset returns using DFSV models. It offers tools for:
*   Simulating DFSV models.
*   Filtering latent states with a Bellman Information Filter (BIF) and a Particle Filter (PF).
*   Estimating model parameters using JAX-based optimization.
*   Utilizing JAX for high performance through automatic differentiation, JIT compilation, and hardware acceleration.

## Project Status

Post Thesis Archive, considering the following extensions:
- Implementing Expectation Maximization for parameter estimation
- Extending the models to also incorporate time-varying idiosyncratic variance, possibly grouped 

The core implementations of the DFSV model, Bellman Information Filter (BIF), and Particle Filter (PF) are stable and functional, validated by a comprehensive `pytest` test suite.

## Key Features

*   **DFSV Core:** Model definition (`DFSVParamsDataclass`), simulation, and JAX integration for performance.
*   **Filtering Algorithms:** Numerically stabilized Bellman Information Filter (BIF) and a benchmark Particle Filter (PF).
*   **Parameter Estimation:** Hyperparameter estimation framework using BIF pseudo log-likelihood and JAX optimization.
*   **Testing:** Comprehensive `pytest` suite.

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
    Project dependencies are managed with `pyproject.toml`. Install the package in editable mode (`-e`) so changes in the `src/` directory are immediately reflected.
    ```bash
    # Using pip (standard)
    pip install -e .
    ```

## Usage Example: Simulating a Simple DFSV Model

The following example demonstrates how to define parameters for a DFSV model (with K=1 factor and N=3 observed series) and simulate data using the `bellman_filter_dfsv` package. This is based on `scripts/simple_dfsv_example.py`.

To run this example:

1.  Ensure you have installed the package as described in the "Installation and Setup" section.
2.  Navigate to the `scripts/` directory: `cd scripts/`
3.  Run the script: `python simple_dfsv_example.py`

The script will simulate data from a simple DFSV model and generate plots of the simulated factors, log-volatilities, and returns.

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
# To embed this plot in a Markdown file, you would typically save it as an image and use Markdown image syntax: `![Simple DFSV Simulation](path/to/outputs/simple_dfsv_simulation.png)` (assuming the image is saved in the `outputs` directory and accessible where the README is displayed).
```

Explore the `scripts/` and `examples/` directories for more detailed use cases, including filter implementations and parameter estimation examples. To run other examples, navigate to the corresponding directory and execute the Python script. For instance, to run the basic filtering example:

1.  Navigate to the `examples/` directory: `cd examples/`
2.  Run the script: `python 02_basic_filtering.py`

## Project Structure

```
BellmanFilterDFSV/
├── .gitignore
├── Dockerfile
├── LICENSE                 # Recommended (Add a note if not present)
├── README.md
├── batch_job_bf.template.json
├── batch_job_pf.template.json
├── pyproject.toml
├── Data/
│   └── ... (various data files)
├── Figures/
│   └── ... (various figures)
├── docs/
│   ├── Makefile
│   ├── make.bat
│   └── source/
├── examples/
│   └── ... (example scripts)
├── notebooks/
│   └── ... (Jupyter notebooks)
├── outputs/
│   └── ... (generated outputs)
├── scripts/
│   ├── Archive/
│   └── ... (various scripts)
├── src/
│   └── bellman_filter_dfsv/
│       ├── __init__.py
│       ├── filters/
│       ├── models/
│       └── utils/
├── tests/
│   └── ... (test files)
└── .venv/                # Or similar virtualenv
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

This project is currently distributed without an explicit license. It is highly recommended to add a `LICENSE` file to the repository. If the intention is to follow the previous mention, the [MIT License](https://opensource.org/licenses/MIT) would be a suitable choice.

## Refactoring and Project Simplification Suggestions

1.  **Dependency Management:**
    *   Remove the `requirements.txt` file as it appears redundant with `pyproject.toml`, which should be considered the single source of truth for project dependencies. This simplifies dependency management and avoids potential conflicts.

2.  **Code Organization:**
    *   **`scripts/` Directory:** Conduct a thorough review of the `scripts/` directory, particularly the `Archive/` subfolder. Identify and remove any obsolete or experimental scripts that are no longer relevant to the project's core goals. For the remaining essential scripts, consider organizing them into more descriptive subdirectories (e.g., `simulation_studies`, `empirical_analysis`, `data_processing`, `utility_scripts`). Furthermore, any reusable utility functions currently within scripts should be migrated to the main `src/bellman_filter_dfsv/utils/` module to promote code reuse and maintainability.
    *   **`examples/` vs. `scripts/`:** Clarify the distinction between the `examples/` and `scripts/` directories. `examples/` should ideally contain minimal, focused code snippets demonstrating specific functionalities or how to use a particular module of the `bellman_filter_dfsv` package. `scripts/` can then be reserved for more complex workflows, such as running full simulation studies, batch processing, or manuscript-specific analyses.
    *   **`notebooks/` Directory:** Review the Jupyter notebooks in `notebooks/`. Archive or delete outdated experimental notebooks. If any notebooks contain code that has proven to be stable and useful for ongoing tasks or defines key methodologies, consider refactoring this code into Python scripts within `scripts/` or integrating it into the `src/` package for better version control and reusability.

3.  **Documentation:**
    *   **Sphinx Documentation:** Ensure that the Sphinx documentation located in the `docs/` directory is comprehensive, up-to-date with the current codebase, and easy to navigate. Include clear instructions on how to build the documentation locally. Add a prominent link to the generated documentation (e.g., on ReadTheDocs or a self-hosted site) from the main README.

4.  **Data Management:**
    *   **Data Directory (`Data/`):** Create a `README.md` file within the `Data/` directory (i.e., `Data/README.md`). This file should clearly describe the contents of the `Data/` directory, including the source of each dataset, its structure (e.g., columns, format), and how it is used by the project's scripts or examples. If data files are too large to be hosted directly in the repository, provide download links and instructions.

5.  **License File:**
    *   **Add `LICENSE` File:** Reiterate the recommendation from the main 'License' section: add a `LICENSE` file (e.g., containing the MIT License text) to the root of the repository to clearly define the terms under which the software can be used, modified, and distributed.
