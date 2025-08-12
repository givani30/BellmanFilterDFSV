# Plan: Verifying and Ensuring Robustness of PF Hyperparameter Optimization (Inspired by BIF Scripts)

**Date:** 08-04-2025

**Goal:** Design a detailed, actionable plan to verify that Particle Filter (PF) hyperparameter optimization is functioning correctly, identify and fix any issues, and prepare the PF optimization pipeline for large-scale simulation studies, drawing inspiration from existing Bellman Information Filter (BIF) optimization and testing scripts.

---

## Phase 1: Preparation & Review (Inspired by BIF Setup)

- **1.1. Review PF implementation and optimization pipeline:**
  - Inspect `src/bellman_filter_dfsv/filters/particle.py`.
  - Review existing PF optimization scripts/notebooks (e.g., `examples/pf_transform_optimization.py`).
- **1.2. Analyze BIF optimization script structure:**
  - **Action:** Read `scripts/compare_bif_optimizers.py`, `scripts/bif_optimizer_stability.py`, and especially `scripts/test_bif_full_phi_hybrid_integration.py`.
  - **Goal:** Understand methodology for setting up optimizers (Optax AdamW, LR schedules), defining objectives (`transformed_bif_objective`), handling fixed parameters (`eqx.tree_at`), stability penalties, eigenvalue checks, and error handling.
- **1.3. Catalog PF parameters, constraints, and priors:**
  - Document PF hyperparameters, constraints (e.g., softplus), and priors.
  - Confirm use of `utils/transformations.py` and ensure consistency with BIF's hybrid transformation approach (e.g., `safe_arctanh` for diagonals).
- **1.4. Assess numerical stability measures:**
  - Check for eigenvalue clipping, precision settings (float32/float64), and use of `equinox.error_if`.

---

## Phase 2: Design & Implement Targeted Verification Tests (Mirroring BIF Test Patterns)

- **2.1. Parameter recovery on synthetic data:**
  - **Action:** Adapt `create_sim_parameters` / `create_training_data` functions to generate synthetic data for PF with known ground truth.
  - **Action:** Structure the optimization run similar to `run_optimization` or `run_comparison` in BIF scripts. Use **Optax AdamW** with **warmup cosine decay schedule** and `optax.apply_if_finite`, mirroring `test_bif_full_phi_hybrid_integration.py`.
  - **Action:** Evaluate recovery using metrics like RMSE/Correlation, adapting `calculate_accuracy` and `print_parameter_comparison` from BIF scripts.
  - **Action:** Run multiple replicates with different seeds to assess variability.
- **2.2. Convergence diagnostics:**
  - **Action:** Log optimizer loss, gradients, and parameter trajectories during the runs (similar to `OptimizerResult` in BIF scripts).
  - **Action:** Test multiple initializations (adapting patterns from `bif_optimizer_stability.py`) to check for local minima.
- **2.3. Precision sensitivity:**
  - **Action:** Run key recovery tests using both float32 and float64, comparing results for stability issues.
- **2.4. Objective Function Consistency:**
  - **Action:** Review `src/bellman_filter_dfsv/filters/objectives.py`. Ensure `pf_objective` and `transformed_pf_objective` handle priors consistently with BIF.
  - **Action:** **Add a stability penalty term** to `pf_objective` (and its transformed version) similar to `bellman_objective`, penalizing eigenvalue magnitudes > 1 for relevant state transition matrices. Use a configurable weight (`stability_penalty_weight`).

---

## Phase 3: Stress Testing & Robustness Checks (Adapting BIF Stability Approaches)

- **3.1. Challenging initializations:**
  - **Action:** Systematically test initial parameter guesses far from true values, inspired by stability checks in `bif_optimizer_stability.py`. Use hybrid transformations for initialization (e.g., `safe_arctanh`).
- **3.2. Noisy and misspecified data:**
  - **Action:** Introduce noise or simulate model misspecification in the synthetic data generation step. Observe optimizer performance.
- **3.3. Priors and constraints impact:**
  - **Action:** Vary prior strengths and constraint tightness, drawing inspiration from `test_bif_priors_optimizers.py`. Check stability.
- **3.4. Edge cases:**
  - **Action:** Test with varying particle counts (P), time series lengths (T), and dimensionality (N, K).
- **3.5. Fixed Parameter Handling:**
  - **Action:** Implement tests where specific parameters (e.g., `mu`, similar to BIF tests) are fixed during optimization using `eqx.tree_at` within the objective wrapper.

---

## Phase 4: Debugging, Refinement, and Improvements (Leveraging BIF Techniques)

- **4.1. Analyze failure modes:**
  - **Action:** Use `equinox.error_if` (`EQX_ON_ERROR=breakpoint`) for runtime debugging.
  - **Action:** Investigate gradient issues, parameter instability, or numerical errors (NaNs/Infs). Use verbose logging as seen in BIF scripts.
- **4.2. Refine priors and constraints:**
  - **Action:** Adjust priors/constraints based on stability test results, ensuring alignment with BIF practices.
- **4.3. Enhance numerical stability:**
  - **Action:** Implement/tune stabilizers (clipping, transforms). Ensure correct application of parameter transformations via `utils/transformations.py`.
  - **Action:** Perform **post-optimization eigenvalue checks** on relevant state transition matrices, similar to `test_bif_full_phi_hybrid_integration.py`, and flag failures.
- **4.4. Improve reproducibility:**
  - **Action:** Standardize random seed handling. Log optimizer states, final parameters (using `cloudpickle`), and diagnostics comprehensively.

---

## Phase 5: Prepare for Simulation Studies (Following BIF Workflow)

- **5.1. Automate verification tests:**
  - **Action:** Create `pytest` functions for key verification steps (parameter recovery, convergence, stability checks). Structure tests similar to BIF test scripts or single replicate runs.
  - **Action:** Utilize common fixtures from `tests/conftest.py`.
- **5.2. Document best practices:**
  - **Action:** Update `systemPatterns.md` or create a specific PF optimization guide summarizing findings (priors, initializations, optimizer settings, stability penalty weight, pitfalls).
- **5.3. Create simulation-ready configs:**
  - **Action:** Prepare/update `batch_job_pf.template.json` with robust default settings derived from verification (optimizer config, penalty weight, etc.).
- **5.4. Prepare for implementation:**
  - **Action:** Prepare to switch to `orchestrator` mode for decomposing tasks and implementation based on this plan.

---

## Mermaid Diagram: Revised PF Optimization Verification Workflow

```mermaid
graph TD
    subgraph Preparation
        A1[Review PF Code]
        A2[Analyze BIF Scripts (compare_bif_optimizers, test_bif_full_phi_hybrid)]
        A3[Catalog PF Params/Priors/Transforms]
        A4[Assess Stability Measures]
    end

    subgraph Verification Tests (Inspired by BIF)
        B1[Generate Synthetic Data]
        B2[Run Optimization (AdamW, LR Sched, Fixed Params)]
        B3[Evaluate Recovery (RMSE/Corr)]
        B4[Check Convergence & Precision]
        B5[Add Stability Penalty to PF Objective]
    end

    subgraph Stress Testing (Inspired by BIF)
        C1[Challenging Inits (Hybrid Transforms)]
        C2[Noisy/Misspecified Data]
        C3[Vary Priors/Constraints]
        C4[Edge Cases (P, T, N, K)]
    end

    subgraph Debugging & Refinement (Inspired by BIF)
        D1[Analyze Failures (use error_if, verbose logs)]
        D2[Refine Priors/Constraints]
        D3[Enhance Stability (Transforms)]
        D4[Post-Optimization Eigenvalue Checks]
        D5[Improve Reproducibility (Seeding, Logging)]
    end

    subgraph Simulation Readiness (Inspired by BIF)
        E1[Automate Tests (pytest)]
        E2[Document Best Practices (systemPatterns.md)]
        E3[Prepare Batch Configs (batch_job_pf.template.json)]
        E4[Prep Handoff to Orchestrator]
    end

    A1 & A2 & A3 & A4 --> B1 & B2 & B3 & B4 & B5
    B1 & B2 & B3 & B4 & B5 --> C1 & C2 & C3 & C4
    C1 & C2 & C3 & C4 --> D1 & D2 & D3 & D4 & D5
    D1 & D2 & D3 & D4 & D5 --> E1 & E2 & E3 & E4