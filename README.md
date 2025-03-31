# QF Thesis - DFSV Models

This repository contains the Python code for the Quantitative Finance thesis project focusing on Dynamic Factor Stochastic Volatility (DFSV) models.

## Project Structure

The codebase follows a standard Python package structure:

```
QF_Thesis/
├── .clinerules           # Custom rules for Roo
├── .coveragerc           # Coverage configuration
├── .gitignore            # Git ignore rules
├── pyproject.toml        # Build system, dependencies, project metadata
├── QF_Thesis_Proposal GBoekestijn.pdf # Original proposal
├── refactoring_plan.md   # Refactoring plan document
├── requirements.txt      # (Potentially redundant with pyproject.toml)
├── data/                 # Input data files
├── examples/             # Example usage scripts (may need updating)
├── notebooks/            # Jupyter notebooks for exploration/visualization
├── outputs/              # Generated outputs (plots, results CSVs)
├── scripts/              # Standalone scripts for running experiments, analysis
├── src/                  # Source code directory
│   └── qf_thesis/        # Main package
│       ├── __init__.py
│       ├── core/           # Core algorithms (filters, simulation, likelihood)
│       │   ├── __init__.py
│       │   ├── simulation.py
│       │   ├── likelihood.py
│       │   └── filters/      # Filter implementations
│       │       ├── __init__.py
│       │       ├── base.py
│       │       ├── bellman.py
│       │       ├── particle.py
│       │       ├── _bellman_impl.py # Bellman static implementations
│       │       └── _bellman_optim.py # Bellman optimization helpers
│       ├── models/         # Model definitions
│       │   ├── __init__.py
│       │   └── dfsv.py       # DFSV parameter dataclass
│       └── utils/          # Utility functions
│           ├── __init__.py
│           └── transformations.py
├── tests/                # Unit and integration tests
│   ├── __init__.py       # (Should exist if tests are importable)
│   ├── test_*.py
│   └── ...
└── ...                   # Other configuration files (.git, .venv, etc.)
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd QF_Thesis
    ```

2.  **Create and activate a virtual environment:** (Recommended)
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    # or .\.venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies and the package in editable mode:**
    This project uses `uv` for faster dependency management (as specified in previous steps). If you don't have `uv`, you can use `pip`.
    ```bash
    # Using uv (recommended)
    uv pip install -e .

    # Using pip
    # pip install -e .
    ```
    This command reads dependencies from `pyproject.toml` and installs the `qf_thesis` package so you can import it in scripts and notebooks.

## Running Tests

Tests are located in the `tests/` directory and use `pytest`.

```bash
# Ensure virtual environment is active
pytest .
```

## Usage

*   **Core Logic:** Import components from the `qf_thesis` package (e.g., `from qf_thesis.core.filters import DFSVParticleFilter`).
*   **Scripts:** Run Python scripts from the `scripts/` directory.
*   **Notebooks:** Use Jupyter notebooks in the `notebooks/` directory for interactive analysis.