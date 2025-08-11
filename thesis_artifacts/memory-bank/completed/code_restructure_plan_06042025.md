# Codebase Restructuring Plan (06-04-2025)

This plan outlines the steps to restructure the `src/bellman_filter_dfsv` directory by moving filters, splitting likelihood/objective functions, moving simulation code, and removing the `core` subdirectory.

## 1. Prepare New Structure

*   Create the target directory: `src/bellman_filter_dfsv/filters/`
*   Ensure `src/bellman_filter_dfsv/models/` exists.

## 2. Move Existing Modules

*   Move all files from `src/bellman_filter_dfsv/core/filters/` to `src/bellman_filter_dfsv/filters/`.
*   Move `src/bellman_filter_dfsv/core/simulation.py` to `src/bellman_filter_dfsv/models/simulation.py`.

## 3. Split `likelihood.py`

*   **Create `src/bellman_filter_dfsv/models/likelihoods.py`:** Copy the following functions from the original `core/likelihood.py` into this new file:
    *   `log_likelihood_observation`
    *   `log_likelihood_factor_transition`
    *   `log_likelihood_volatility_transition`
    *   `compute_joint_log_likelihood`
    *   `log_prior_density`
    *   Ensure necessary imports (like `DFSVParamsDataclass`, `jax`, `jnp`, `safe_norm_logpdf`, etc.) are included in this new file.
*   **Create `src/bellman_filter_dfsv/filters/objectives.py`:** Copy the following functions from the original `core/likelihood.py` into this new file:
    *   `bellman_objective`
    *   `transformed_bellman_objective`
    *   `pf_objective`
    *   `transformed_pf_objective`
    *   Ensure necessary imports (like `DFSVParamsDataclass`, filter classes, `untransform_params`, `log_prior_density`, etc.) are included in this new file. Note that `log_prior_density` will now need to be imported from `bellman_filter_dfsv.models.likelihoods`.
*   **Delete Original:** Delete the file `src/bellman_filter_dfsv/core/likelihood.py`.

## 4. Remove `core` Directory

*   Delete the now empty directory `src/bellman_filter_dfsv/core/`.

## 5. Update Imports

*   Modify import statements in all affected files (identified 29 locations + any internal imports within moved/split files) to reflect the new structure. Examples:
    *   `from bellman_filter_dfsv.core.filters.some_filter import SomeFilter` becomes `from bellman_filter_dfsv.filters.some_filter import SomeFilter`
    *   `from bellman_filter_dfsv.core.likelihood import some_objective` becomes `from bellman_filter_dfsv.filters.objectives import some_objective`
    *   `from bellman_filter_dfsv.core.likelihood import some_likelihood_func` becomes `from bellman_filter_dfsv.models.likelihoods import some_likelihood_func`
    *   `from bellman_filter_dfsv.core.simulation import simulate_DFSV` becomes `from bellman_filter_dfsv.models.simulation import simulate_DFSV`

## 6. Verify

*   Run the test suite using `uv run pytest` to confirm that the restructuring is successful and no functionality is broken.

## Target Structure Diagram

```mermaid
graph TD
    A[src/bellman_filter_dfsv] --> B[__init__.py]
    A --> C[filters]
    A --> D[models]
    A --> E[utils]

    C --> C1[__init__.py]
    C --> C2[_bellman_impl.py]
    C --> C3[_bellman_optim.py]
    C --> C4[base.py]
    C --> C5[bellman_information.py]
    C --> C6[bellman.py]
    C --> C7[particle.py]
    C --> C8[objectives.py]

    D --> D1[__init__.py]
    D --> D2[dfsv.py]
    D --> D3[simulation.py]
    D --> D4[likelihoods.py]

    E --> E1[__init__.py]
    E --> E2[jax_helpers.py]
    E --> E3[transformations.py]

    style C fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#ccf,stroke:#333,stroke-width:2px
    style C8 fill:#f9d,stroke:#333,stroke-width:1px
    style D3 fill:#cce,stroke:#333,stroke-width:1px
    style D4 fill:#cce,stroke:#333,stroke-width:1px