# Parameter Recovery Check Plan (07-04-2025)

**Objective:** Perform a preliminary assessment of parameter recovery using the Bellman Information Filter (BIF) with the newly implemented full persistence matrices (`Phi_f`, `Phi_h`), focusing on optimizer convergence and basic parameter accuracy.

**Methodology:**

1.  **Simulation Setup:**
    *   **Parameters:** N=5 (observables), K=2 (factors), T=2000 (time steps).
    *   **True Values:** Use standard true parameter values consistent with previous tests (e.g., from `tests/conftest.py` or simulation scripts). Ensure `lambda_r` adheres to identifiability constraints (lower-triangular, diagonal=1.0).
    *   **Replicates:** Run 5 independent simulation replicates for each experimental configuration.
    *   **Filter:** Bellman Information Filter (BIF).
    *   **Optimizer:** Use AdamW (as successfully used in `scripts/test_bif_priors_optimizers.py`) or BFGS if preferred, with reasonable settings (e.g., learning rate, max steps).
    *   **Full Phi Handling:** Use the `softplus` element-wise transformation + stability penalty (`stability_penalty_weight=1000.0`) for full matrices, as per Decision [07-04-2025 02:05:00].

2.  **Experimental Configurations:**
    *   **Experiment 1: Estimate All (Free `mu`, Full `Phi_f`/`Phi_h`)**
        *   Estimate all parameters: Λ, Φ_f, Φ_h, μ, Q_h, Σ_ε.
        *   `Phi_f` and `Phi_h` are treated as full matrices using the `softplus`+penalty approach.
        *   `mu` is *not* fixed.
    *   **Experiment 1b (Contingency): Estimate All (Fixed `mu`, Full `Phi_f`/`Phi_h`)**
        *   *Run only if Experiment 1 shows poor `mu` recovery (as expected).*
        *   Estimate all parameters *except* `mu`.
        *   Fix `mu` to its true simulation value during optimization.
        *   `Phi_f` and `Phi_h` are treated as full matrices using the `softplus`+penalty approach.
    *   **Experiment 2: Estimate (Fixed `mu`, Diagonal `Phi_f`, Full `Phi_h`)**
        *   Estimate all parameters *except* `mu`.
        *   Fix `mu` to its true simulation value.
        *   Treat `Phi_f` as diagonal: Modify transformations/objective to only estimate/transform the diagonal elements (using `softplus`, no stability penalty needed for diagonal).
        *   Treat `Phi_h` as a full matrix using the `softplus`+penalty approach.

3.  **Implementation:**
    *   Create a new script: `scripts/check_full_phi_recovery.py`.
    *   Base the script on `scripts/test_bif_full_phi_hybrid_integration.py` or `scripts/test_bif_priors_optimizers.py`, adapting it for the different experimental configurations.
    *   Include functions for:
        *   Generating simulated data for N=5, K=2, T=2000.
        *   Defining the BIF objective function with flexible configurations (free/fixed `mu`, full/diagonal `Phi_f`).
        *   Running the optimization loop for a specified number of replicates.
        *   Saving convergence status and estimated parameters for each replicate.

4.  **Analysis & Output:**
    *   Aggregate results across replicates for each experiment.
    *   Generate simple tables summarizing:
        *   Optimizer convergence status (success/failure) per replicate.
        *   Mean/Median estimated parameters vs. True parameters.
    *   Save the tables to a markdown file in a new subdirectory: `outputs/parameter_recovery_checks/check_full_phi_recovery_results_DD-MM-YYYY.md`.

**Workflow Diagram:**

```mermaid
graph TD
    A[Start: Parameter Recovery Check] --> B(Define Base Config: N=5, K=2, T=2000, True Params);
    B --> C{Create Script: `scripts/check_full_phi_recovery.py`};
    C --> D[Experiment 1: Estimate All (Free Mu, Full Phi_f/h)];
    D --> E{Run 5 Replicates};
    E --> F{Analyze Mu Estimates};
    F -- Poor Mu Recovery --> G[Experiment 1b: Estimate All (Fixed Mu, Full Phi_f/h)];
    F -- Good Mu Recovery --> H[Proceed to Exp 2];
    G --> I{Run 5 Replicates};
    I --> H;
    H --> J[Experiment 2: Estimate (Fixed Mu, Diag Phi_f, Full Phi_h)];
    J --> K{Run 5 Replicates};
    K --> L[Aggregate Results: Convergence Status, Parameter Estimates];
    L --> M[Generate Output Tables];
    M --> N(Save Tables to `outputs/parameter_recovery_checks/`);
    N --> O[End Analysis];

    subgraph "Experiment 1 Config"
        direction LR
        D1[Objective: Free Mu]
        D2[Phi_f: Full (Softplus + Penalty 1k)]
        D3[Phi_h: Full (Softplus + Penalty 1k)]
    end

    subgraph "Experiment 1b Config"
        direction LR
        G1[Objective: Fixed Mu]
        G2[Phi_f: Full (Softplus + Penalty 1k)]
        G3[Phi_h: Full (Softplus + Penalty 1k)]
    end

    subgraph "Experiment 2 Config"
        direction LR
        J1[Objective: Fixed Mu]
        J2[Phi_f: Diagonal (Softplus, No Penalty)]
        J3[Phi_h: Full (Softplus + Penalty 1k)]
    end

    style D fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#f9f,stroke:#333,stroke-width:2px
    style J fill:#f9f,stroke:#333,stroke-width:2px