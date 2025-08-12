# Plan: Implement Full Persistence Matrices (Phi_f, Phi_h) via Hybrid Approach

**Date:** 07-04-2025

**Objective:** Modify the DFSV model implementation and parameter transformations to support full KxK persistence matrices (`Phi_f`, `Phi_h`) using a hybrid approach: transform diagonal elements with `tanh`/`arctanh` for stability and leave off-diagonals unconstrained. Enforce overall stability primarily through the diagonal transformation, with an optional penalty term added to the optimization objective if needed. This plan supersedes previous approaches.

**Rationale:**

* Avoids the non-differentiability of `jax.linalg.eig` in the core parameter transformation by only transforming diagonal elements.
* Directly ensures diagonal elements of `Phi` remain within the stable `(-1, 1)` range using `tanh`.
* Allows for non-zero off-diagonal elements, providing model flexibility.
* Relies on the diagonal transformation and the optimizer to find stable solutions. An optional penalty term using `jax.linalg.eigvals` can be added for stricter enforcement if required.

**Affected Files:**

* `src/bellman_filter_dfsv/utils/transformations.py`: Implement `safe_arctanh`, modify `transform_params` and `untransform_params` for the hybrid `Phi` transformation. Remove or comment out old `stabilize_matrix`/`get_unconstrained_matrix`.
* `src/bellman_filter_dfsv/filters/objectives.py`: (Potentially) Modify objective functions (`bellman_objective`, `pf_objective`, and callers) to include the optional penalty calculation based on `jax.linalg.eigvals` and accept a `penalty_weight`.
* `tests/test_transformations.py`: Update unit tests for `safe_arctanh` and the new hybrid `Phi` transformation logic (diagonal `tanh`/`arctanh`, off-diagonal identity).
* `scripts/` (e.g., `test_bif_priors_optimizers.py`): Adapt integration tests for correct parameter initialization (unconstrained off-diagonals, `arctanh`-transformed diagonals), potentially add penalty weight control, and verify stability post-optimization.

**Detailed Plan:**

1. **Update `transformations.py`:**
    * **Add `safe_arctanh`:** Implement the `safe_arctanh` function with clipping for numerical stability.
    * **Remove/Comment Out Old Stabilization:** Remove or comment out the `stabilize_matrix` and `get_unconstrained_matrix` functions based on `jax.linalg.eig`.
    * **Modify `transform_params`:** Apply `safe_arctanh` only to the diagonal elements of `Phi_f`/`Phi_h`. Leave off-diagonals unchanged.
    * **Modify `untransform_params`:** Apply `jnp.tanh` only to the diagonal elements of `Phi_f`/`Phi_h`. Leave off-diagonals unchanged.
    * **Update Docstrings:** Reflect the new hybrid transformation logic.

2. **Update Unit Tests (`tests/test_transformations.py`):**
    * Add tests for `safe_arctanh`.
    * Remove tests for old stabilization functions.
    * Implement round-trip tests verifying diagonal `tanh`/`arctanh` and off-diagonal identity for `Phi`.

3. **Update Integration Test (`scripts/` - Start *without* Penalty):**
    * Adapt an existing script.
    * **Initialization:** Initialize unconstrained `Phi` with unconstrained off-diagonals and `arctanh`-transformed diagonals.
    * **Run Optimization:** Use the `transformed_..._objective` function *without* the penalty term initially.
    * **Verification:** Check convergence, errors, and stability of final *untransformed* `Phi` matrices (`jnp.abs(jax.linalg.eigvals(Phi)) < 1.0`).

4. **Implement Optional Penalty (If Step 3 Shows Instability):**
    * **Modify `objectives.py`:** Add `penalty_weight` argument. Inside base objectives (after `untransform_params`), calculate eigenvalues (`jax.linalg.eigvals`) and penalty (`sum(relu(abs(eigvals) - 1 + EPS)**2)`). Add `penalty_weight * penalty` to the objective.
    * **Update Integration Test:** Pass non-zero `penalty_weight` and re-verify stability.

5. **Memory Bank Update:**
    * Log decision in `decisionLog.md`.
    * Update `activeContext.md`.

**Mermaid Diagram (Hybrid Transformation Flow):**

```mermaid
graph LR
    subgraph Optimization Loop
        Opt[Optimizer e.g., AdamW] -- Unconstrained Params --> ObjFnWrapper(Transformed Objective)
    end

    subgraph Objective Function Wrapper (e.g., transformed_bellman_objective)
        ObjFnWrapper -- Unconstrained Params --> UT(untransform_params)
        UT -- "Constrained" Params (tanh Diag Phi) --> BaseObjFn(Base Objective)
        BaseObjFn -- Objective Value (incl. Optional Penalty) --> ObjFnWrapper
        ObjFnWrapper -- Gradient --> Opt
    end

    subgraph untransform_params
        InputUnc["Unconstrained Params (Unc Diag Phi)"] --> ApplyTanh["Apply tanh to Phi Diagonals"]
        ApplyTanh --> OutputConst["Constrained Params (tanh Diag Phi)"]
    end

    subgraph Base Objective Function (e.g., bellman_objective)
        InputParams["Constrained Params (tanh Diag Phi)"] --> Filter(BIF/PF Filter Logic)
        Filter -- Pseudo-Likelihood --> CalcNegLL(Calculate -LL)
        InputParams --> CalcPrior(Calculate Log Prior)
        subgraph Optional Penalty Calculation
            InputParams -- Phi_f, Phi_h --> CalcEigvals(jax.linalg.eigvals)
            CalcEigvals --> CalcPenalty(Calculate Stability Penalty)
        end
        CalcNegLL & CalcPrior & CalcPenalty -- penalty_weight --> Combine(Combine: -LL - Prior + [scale*Penalty])
        Combine --> OutputValue(Final Objective Value)
    end

    subgraph Parameter Initialization / Loading
        Init(Initial "Stable" Params e.g., Diag=0.95) --> TP(transform_params)
        TP -- Unconstrained Params (Unc Diag Phi) --> Opt(Set Initial Optimizer State)
    end

    subgraph transform_params
        InputConst["Constrained Params (tanh Diag Phi)"] --> ApplyArcTanh["Apply safe_arctanh to Phi Diagonals"]
        ApplyArcTanh --> OutputUnc["Unconstrained Params (Unc Diag Phi)"]
    end

    style UT fill:#f9f,stroke:#333,stroke-width:2px
    style TP fill:#f9f,stroke:#333,stroke-width:2px
    style ApplyTanh fill:#ccf,stroke:#333,stroke-width:2px
    style ApplyArcTanh fill:#ccf,stroke:#333,stroke-width:2px
    style OptionalPenaltyCalculation fill:#eee,stroke:#999,stroke-dasharray: 5 5
