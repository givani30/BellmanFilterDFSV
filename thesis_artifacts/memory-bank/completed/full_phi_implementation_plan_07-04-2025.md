# Plan: Implement Full, Stable Persistence Matrices (Phi_f, Phi_h)

**Date:** 07-04-2025

**Objective:** Modify the DFSV model implementation and parameter transformations to support full KxK persistence matrices (`Phi_f`, `Phi_h`) while ensuring stability (eigenvalue magnitudes < 1) using eigenvalue decomposition and transformation (Option A).

**Affected Files:**

*   `src/bellman_filter_dfsv/utils/transformations.py`: Main location for transformation logic changes.
*   `tests/test_transformations.py`: Add new unit tests for the stabilization logic and round-trip transformations.
*   `scripts/` (Potentially): May need minor adjustments to initialization in estimation scripts for testing.

**Detailed Plan:**

1.  **Implement `stabilize_matrix` Function:**
    *   **File:** `src/bellman_filter_dfsv/utils/transformations.py`
    *   **Input:** An unconstrained KxK JAX array (`matrix_unc`).
    *   **Logic:**
        *   Perform eigenvalue decomposition: `eigvals, eigvecs = jax.linalg.eig(matrix_unc)`.
        *   Calculate magnitudes: `magnitudes = jnp.abs(eigvals)`.
        *   Transform magnitudes using `tanh`: `transformed_magnitudes = jnp.tanh(magnitudes)`.
        *   Calculate phases (handle zero eigenvalues): `phases = jnp.where(magnitudes == 0, 1.0, eigvals / magnitudes)`.
        *   Compute new eigenvalues: `new_eigvals = transformed_magnitudes * phases`.
        *   Reconstruct the stable matrix: `stable_matrix = (eigvecs @ jnp.diag(new_eigvals) @ jax.linalg.inv(eigvecs)).real`.
        *   **JAX-Compatible Error Handling:** Check `jnp.any(jnp.isnan(matrix_unc) | jnp.isinf(matrix_unc))`. If true, return `jnp.eye(K) * 0.95`. Otherwise, proceed with eigendecomposition and use `jnp.nan_to_num` on the result `stable_matrix`.
    *   **Output:** A stabilized KxK JAX array (real-valued).

2.  **Implement Inverse Stabilization (`get_unconstrained_matrix`):**
    *   **File:** `src/bellman_filter_dfsv/utils/transformations.py`
    *   **Input:** A stable KxK JAX array (`stable_matrix`).
    *   **Logic:**
        *   Perform eigenvalue decomposition: `eigvals, eigvecs = jax.linalg.eig(stable_matrix)`.
        *   Get magnitudes: `magnitudes = jnp.abs(eigvals)`. Clip near 1: `magnitudes = jnp.clip(magnitudes, 0, 1.0 - EPS)`.
        *   Invert the `tanh` transformation: `unconstrained_magnitudes = jnp.arctanh(magnitudes)`.
        *   Calculate phases (handle zero eigenvalues): `phases = jnp.where(magnitudes == 0, 1.0, eigvals / magnitudes)`.
        *   Compute unconstrained eigenvalues: `unc_eigvals = unconstrained_magnitudes * phases`.
        *   Reconstruct the unconstrained matrix: `unc_matrix = (eigvecs @ jnp.diag(unc_eigvals) @ jax.linalg.inv(eigvecs)).real`.
        *   **JAX-Compatible Error Handling:** Apply similar input checks and output `nan_to_num` as in `stabilize_matrix`.
    *   **Output:** An unconstrained KxK JAX array (real-valued).

3.  **Update `untransform_params` Function:**
    *   **File:** `src/bellman_filter_dfsv/utils/transformations.py`
    *   Remove diagonal logic for `Phi_f`, `Phi_h`.
    *   Call `stabilize_matrix` on unconstrained `Phi_f`, `Phi_h`.
    *   Replace `Phi_f`, `Phi_h` fields in the returned object.

4.  **Update `transform_params` Function:**
    *   **File:** `src/bellman_filter_dfsv/utils/transformations.py`
    *   Remove diagonal logic for `Phi_f`, `Phi_h`.
    *   Call `get_unconstrained_matrix` on stable `Phi_f`, `Phi_h`.
    *   Replace `Phi_f`, `Phi_h` fields in the returned object.

5.  **Add Unit Tests:**
    *   **File:** `tests/test_transformations.py`
    *   Test `stabilize_matrix` and `get_unconstrained_matrix` (including NaN/Inf inputs).
    *   Test `transform_params` / `untransform_params` round trip for full `Phi` matrices.

6.  **Integration Test:**
    *   Create a copy of `scripts/test_bif_priors_optimizers.py`.
    *   Modify initialization for full KxK `Phi_f`, `Phi_h`.
    *   Ensure BIF is used and `mu` is fixed.
    *   Run optimization for ~250 steps.
    *   **Goal:** Verify code runs, transformations are applied, optimization proceeds without NaNs/Infs, and final `Phi` matrices are stable. Check convergence status and final parameter values.

7.  **Memory Bank Update:**
    *   Update `decisionLog.md` and `activeContext.md`.

**Mermaid Diagram (Transformation Flow):**

```mermaid
graph LR
    subgraph Optimization Loop
        Opt[Optimizer e.g., AdamW] -- Unconstrained Params --> ObjFn(Objective Function)
    end

    subgraph Objective Function
        ObjFn --> UT(untransform_params)
        UT -- Unconstrained Phi_unc --> SM(stabilize_matrix)
        SM -- Stable Phi --> Filter(BIF Filter Logic)
        Filter -- Pseudo-Likelihood --> ObjFn
        ObjFn -- Gradient --> Opt
    end

    subgraph Parameter Initialization / Loading
        Init(Initial Stable Params) --> TP(transform_params)
        TP -- Stable Phi --> GUM(get_unconstrained_matrix)
        GUM -- Unconstrained Phi_unc --> Opt(Set Initial Optimizer State)
    end

    style UT fill:#f9f,stroke:#333,stroke-width:2px
    style TP fill:#f9f,stroke:#333,stroke-width:2px
    style SM fill:#ccf,stroke:#333,stroke-width:2px
    style GUM fill:#ccf,stroke:#333,stroke-width:2px