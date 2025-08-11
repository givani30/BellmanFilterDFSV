# Plan: Improving Optimization Stability & Speed (11-04-2025)

**Goal:** Prevent expensive filter evaluations with invalid parameters and improve robustness of intermediate calculations to reduce NaN propagation and optimization slowdowns, while maintaining JAX/JIT compatibility.

**Core Principle:** Implement checks within the JIT-compiled objective function path to "fail fast" by returning a large penalty value immediately if invalid parameters are detected, bypassing the main filter computation. Enhance internal filter steps to handle potential numerical issues gracefully.

**Hypothesis:** Evaluating the full filter with parameters that lead to invalid model states (like unstable dynamics) is a likely cause of both numerical issues (NaNs/Infs) and the subsequent slowdowns due to optimizer line searches.

---

## Phase 1: Baseline Measurement & Early Stability Validation

1.  **Develop Baseline Performance Script:**
    *   **Goal:** Create a script to measure the performance of the current optimization setup on a small, controlled problem.
    *   **Action:** Create a new script: `scripts/profile_objective_stability.py`. Configure it to use a small dataset (e.g., N=3, K=2, T=200), BIF, BFGS, the current objective function, run for a fixed number of steps, and log time, evaluations, and loss spikes.
    *   **Rationale:** Establishes a clear benchmark *before* making changes.

2.  **Implement Early Stability Validation:**
    *   **Goal:** Modify the objective function to "fail fast" for unstable `Phi` matrices.
    *   **Location:** `src/bellman_filter_dfsv/filters/objectives.py`.
    *   **Action:** Before calling the filter's likelihood function, check `Phi_f`/`Phi_h` stability using eigenvalues. If unstable, use `jax.lax.cond` or `jnp.where` to return a large penalty directly, skipping the filter call. (Note: Positive variance checks deemed unnecessary due to transformations).
    *   **Rationale:** Avoids expensive filter runs for unstable dynamics, maintaining JIT compatibility.

3.  **Test & Evaluate Phase 1:**
    *   **Goal:** Verify the implementation and measure impact.
    *   **Action:** Run the baseline script with the modified objective. Compare time, evaluations, and loss spikes to the initial baseline. Run unit tests (`pytest tests/`). Optionally add a new test for the early penalty mechanism.
    *   **Rationale:** Quantifies improvement and ensures correctness.

---

## Phase 2: Further Robustness (If Necessary)

*   **Trigger:** If Phase 1 significantly improves performance but occasional NaNs or slowdowns still occur during the filter execution itself (even with valid initial parameters).
*   **Action:** Implement selected JIT-compatible robustness measures from the original plan's Step 2, focusing on the most likely sources:
    *   Modify `_bellman_optim.py::update_h_bfgs` to check `sol.result` and return `h_init` on failure using `jnp.where`.
    *   Consider adding `jnp.nan_to_num` around the output of `_bellman_optim.py::update_factors`.
*   **Rationale:** Address potential numerical issues within the filter's iterative updates.

---

## Phase 3: Optimizer Interaction (If Necessary)

*   **Trigger:** If the optimizer still takes excessively large steps or its state becomes corrupted despite objective function improvements.
*   **Action:** Implement Step 3 from the original plan: Ensure gradient clipping and `apply_if_finite` are correctly used in `utils/solvers.py`.
*   **Rationale:** Improve the optimizer's behavior when dealing with potentially difficult gradients.

---

## Implementation Flow Diagram (Phase 1 Focus)

```mermaid
graph TD
    A[Optimizer Proposes Transformed Params θ_t] --> B{Objective Function Call};
    B --> C{untransform_params(θ_t)};
    C --> D{Check Phi Stability (Eigenvalues)};
    D -- Unstable --> E[Return Large Penalty P];
    D -- Stable --> F{Call likelihood_fn(p, y)};
    F --> G{Filter Execution};
    G --> I[Calculate Log Likelihood L];
    I --> J{Apply nan_to_num to L};
    J --> K{Add Prior & Existing Penalty};
    K --> L[Return Final Objective Value O];
    E --> L;
    L --> M[Optimizer Receives O];
    M --> N{Optimizer State Update};
    N --> A;

    style E fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#ccf,stroke:#333,stroke-width:2px
```

---

## Affected Files

*   `src/bellman_filter_dfsv/filters/objectives.py` (Phase 1)
*   `src/bellman_filter_dfsv/filters/_bellman_optim.py` (Phase 2)
*   `src/bellman_filter_dfsv/filters/bellman_information.py` (Phase 2)
*   `src/bellman_filter_dfsv/utils/solvers.py` (Phase 3)
*   `scripts/profile_objective_stability.py` (New - Phase 1)
*   `tests/` (Phase 1 & potentially later)