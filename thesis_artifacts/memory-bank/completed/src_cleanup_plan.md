# Plan for `src` Directory Cleanup (Bellman Filter Update)

**Goal:** Update the original covariance-based Bellman filter (`bellman.py`) to use the BIF pseudo-likelihood, remove dead code, improve documentation (Google standard), consolidate duplicate logic, and ensure code consistency.

**Implementation Steps (to be performed by Code mode):**

1.  **Adapt Likelihood Calculation:**
    *   **Target:** Modify the filtering loop (`filter` or `filter_scan`) in `src/bellman_filter_dfsv/core/filters/bellman.py` and/or the objective function it uses (likely in `src/bellman_filter_dfsv/core/likelihood.py`).
    *   **Action:** Implement the BIF pseudo-likelihood calculation (Lange Eq. 40). This requires obtaining the predicted and posterior information matrices (`I_{t|t-1}` and `I_{t|t}`) by inverting the corresponding covariance matrices (`P_{t|t-1}` and `P_{t|t}`) available within `bellman.py`.
    *   **Stability:** Ensure robust inversion using techniques like adding jitter before inversion or using `jnp.linalg.pinv` if necessary, even if inversions are performed elsewhere in the filter.
2.  **Consolidate Helper Logic:**
    *   **Target:** `_bellman_impl.py`, `_bellman_optim.py`.
    *   **Action:** Identify logic shared between the original Bellman filter and the BIF (e.g., parts of FIM calculation, optimization steps). Refactor this shared logic into common utility functions (e.g., in `src/bellman_filter_dfsv/utils/jax_helpers.py` or potentially a new `src/bellman_filter_dfsv/core/filters/_common.py`). Update `bellman.py` and `bellman_information.py` to use these common functions.
3.  **Remove Dead Code:**
    *   **Target:** `bellman.py`, `_bellman_impl.py`, `_bellman_optim.py`.
    *   **Action:** After Steps 1 & 2, identify and remove any functions, methods, or code blocks that are no longer used or referenced.
4.  **Improve Documentation & Consistency:**
    *   **Target:** All modified files (`bellman.py`, `_bellman_impl.py`, `_bellman_optim.py`, `likelihood.py`, any new common files).
    *   **Action:** Review and update docstrings (module, class, function/method) and type hints to meet Google standards (as per `.clinerules`). Ensure consistent naming conventions and adherence to patterns documented in `memory-bank/systemPatterns.md`.
5.  **Testing:**
    *   **Target:** `tests/` directory, particularly tests involving `bellman.py` (e.g., `test_bellman_unified.py`).
    *   **Action:** Update existing unit tests to reflect the changes. Add specific tests to verify the stability and correctness of the updated likelihood calculation in the modified `bellman.py`. Tests comparing the old BF likelihood directly to the new BIF-style likelihood might need adjustment or removal.

**Illustrative Structure:**

```mermaid
graph TD
    subgraph core/filters
        BF[bellman.py (Updated BF)]
        BIF[bellman_information.py (BIF)]
        PF[particle.py (PF)]
        Base[base.py]
        subgraph Helpers
            Impl[_bellman_impl.py (Refactored)]
            Optim[_bellman_optim.py (Refactored)]
            Common[_common.py (New/Optional)]
        end
    end
    subgraph core
        Likelihood[likelihood.py (Updated)]
        Simulation[simulation.py]
    end
    subgraph models
        DFSV[dfsv.py]
    end
    subgraph utils
        Transforms[transformations.py]
        JaxHelpers[jax_helpers.py (Potentially Updated)]
    end

    BF -- Inherits --> Base
    BIF -- Inherits --> Base
    PF -- Inherits --> Base

    BF -- Uses --> Impl
    BF -- Uses --> Optim
    BIF -- Uses --> Impl
    BIF -- Uses --> Optim
    BF -- Calls --> Likelihood
    BIF -- Calls --> Likelihood

    Impl -- Uses --> Common
    Optim -- Uses --> Common
    Impl -- Uses --> JaxHelpers
    Optim -- Uses --> JaxHelpers

    Likelihood -- Uses --> Transforms
    Likelihood -- Uses --> DFSV

    BF -- Uses --> DFSV
    BIF -- Uses --> DFSV
    PF -- Uses --> DFSV