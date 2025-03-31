# Refactoring Plan for QF_Thesis Codebase

This document outlines a plan to restructure the Python codebase for the econometrics thesis project, focusing on improving legibility, efficiency, and maintainability.

## Analysis Summary

*   **Inconsistent Structure:** Core logic (`functions/`, `models/`) resides outside the standard `src/` package layout. Numerous scripts and notebooks clutter the root directory.
*   **Technical Debt:** Coexistence of NumPy (`DFSV_params`) and JAX (`DFSVParamsDataclass`) parameter classes requires conversions, complicating code and tests. The base filter's smoother uses NumPy, potentially conflicting with JAX-based subclasses.
*   **Strengths:** Good use of JAX for computation, existing `unittest` suite provides a safety net, and an initial `src/` layout exists.
*   **Opportunity:** Consolidate code into a proper installable package, standardize on the JAX parameter class, clean up the root directory, and potentially refactor filter implementations for better modularity.

## Proposed Directory Structure

Adopt a standard Python package structure to make the codebase installable and imports cleaner.

```mermaid
graph TD
    A[QF_Thesis/] --> B(src/);
    A --> C(tests/);
    A --> D(scripts/);
    A --> E(notebooks/);
    A --> F(data/);
    A --> G(outputs/);
    A --> H(pyproject.toml);
    A --> I(README.md);
    A --> J(.gitignore);
    A --> K(.coveragerc);

    B --> B1(qf_thesis/);
    B1 --> B1a(__init__.py);
    B1 --> B1b(core/);
    B1 --> B1c(models/);
    B1 --> B1d(utils/);

    B1b --> B1b1(__init__.py);
    B1b --> B1b2(filters/);
    B1b --> B1b3(simulation.py);
    B1b --> B1b4(likelihood.py);
    B1b --> B1b5(optimization.py);
    B1b --> B1b6(...);

    B1b2 --> B1b2a(__init__.py);
    B1b2 --> B1b2b(base.py);
    B1b2 --> B1b2c(particle.py);
    B1b2 --> B1b2d(kalman.py); # If EKF/UKF added
    B1b2 --> B1b2e(...);

    B1c --> B1c1(__init__.py);
    B1c --> B1c2(dfsv.py); # Standardized JAX version
    B1c --> B1c3(...);

    B1d --> B1d1(__init__.py);
    B1d --> B1d2(transformations.py);
    B1d --> B1d3(parameter_utils.py);
    B1d --> B1d4(...);

    C --> C1(core/);
    C --> C2(models/);
    C --> C3(...);
    C1 --> C1a(filters/);
    C1a --> C1a1(test_particle.py);
    C1a --> C1a2(...);
```

**Explanation:**

*   **`src/qf_thesis/`**: Contains all the core library code. Makes the package installable (e.g., via `pip install -e .`).
    *   **`core/`**: Houses the main algorithms (filters, simulation, likelihood).
        *   **`filters/`**: Specific filter implementations (base class, particle filter, potentially others).
    *   **`models/`**: Definitions of models and parameters (standardized `DFSVParamsDataclass`).
    *   **`utils/`**: Helper functions (transformations, parameter handling).
*   **`tests/`**: Mirrors the `src/qf_thesis/` structure, containing unit/integration tests.
*   **`scripts/`**: Standalone Python scripts for running experiments, analyses, plotting (imports from the installed `qf_thesis` package).
*   **`notebooks/`**: Jupyter notebooks for exploration and visualization (imports from the installed `qf_thesis` package).
*   **`data/`, `outputs/`**: Remain as is for input data and generated results.
*   **`pyproject.toml`**: Defines build requirements, dependencies, and potentially tool configurations (like `pytest`, `black`, `flake8`).
*   **`README.md`**: Essential documentation.

## Incremental Refactoring Strategy (Test-Driven)

This strategy emphasizes small, verifiable steps using your existing tests.

1.  **Step 0: Setup & Baseline:**
    *   Ensure Git is clean.
    *   Run all tests (`pytest .` recommended) and confirm they pass. Record coverage.
    *   Create `pyproject.toml` (e.g., using `setuptools`).
2.  **Step 1: Consolidate Code into `src/qf_thesis`:**
    *   Move files from `functions/`, `models/`, etc., into the corresponding `src/qf_thesis/` subdirectories as outlined above.
    *   Update internal imports within moved files (use relative imports like `from . import ...` or `from ..models import ...`).
    *   Update test imports to point to the new locations (e.g., `from qf_thesis.core.filters import ...`) and remove `sys.path` hacks.
    *   **Run Tests:** Fix import errors and ensure all tests pass. Commit.
3.  **Step 2: Standardize Parameter Class:**
    *   Modify `src/qf_thesis/core/simulation.py` to accept `DFSVParamsDataclass`.
    *   Remove the old `DFSV_params` class definition and conversions from tests and `src/qf_thesis/models/dfsv.py`.
    *   **Run Tests:** Verify all tests pass. Commit.
4.  **Step 3: Refine Filter Implementation (Optional but Recommended):**
    *   Consider splitting large filter files (e.g., `base.py`, `particle.py`).
    *   Review the `smooth` method's reliance on NumPy vs. JAX. If possible, make it more consistent or move particle-specific smoothing into `particle.py`.
    *   **Run Tests:** Ensure filter/smoother tests pass. Commit.
5.  **Step 4: Clean Up Root Directory:**
    *   Move scripts (`.py`) to `scripts/`, notebooks (`.ipynb`) to `notebooks/`. Update their imports to use the installed package (requires `pip install -e .`).
    *   Move remaining relevant `.py` files (tests, experiments) to `tests/`, `scripts/`, or `notebooks/`.
    *   Move generated files (`.png`, `.csv`, etc.) to `outputs/`.
    *   Archive or delete clearly unused files.
    *   **Manual Verification:** Run key scripts/notebooks. Commit.
6.  **Step 5: Address Archives/Deprecated Code:**
    *   Review `Archive/` and `functions/deprecated/`. Delete if truly unused. Commit.
7.  **Step 6: Documentation & Final Touches:**
    *   Add `__init__.py` files to all package directories.
    *   Update/Create `README.md` (structure, setup, usage).
    *   Ensure docstrings meet the Google standard (as per `.clinerules`).
    *   Consider adding linters/formatters (`flake8`, `black`).
    *   **Run Tests:** Final check.