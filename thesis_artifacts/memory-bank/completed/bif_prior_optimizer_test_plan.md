# Execution Plan: Testing BIF Priors and Optimizers (Focus on Stability & RMS Norm)

**1. Goal:**
Execute the `scripts/test_bif_priors_optimizers.py` script to evaluate the impact of different prior configurations and optimizers (with Optax solvers using **RMS norm** for convergence) on the **stability** (reducing failures) of parameter estimation using the BIF pseudo log-likelihood. Secondary goals include assessing convergence/efficiency and saving results to CSV for analysis.

**2. Environment Setup:** (Skipped - Assumed complete)

**3. Script Modification (Requires Code Mode):**
*   **Determinism Check:** Confirmed OK. No changes needed.
*   **CSV Output:** Modify the `print_results_table` function (or add a new function) to save the `results` list to a CSV file (e.g., `bif_prior_optimizer_results_rms.csv`).
*   **Solver Modification (RMS Norm):** Modify the instantiation of `optx.OptaxMinimiser` for `SGD`, `Adam`, and `AdamW` (lines 99-101 in the script) to specify the use of RMS norm for evaluating the `rtol` and `atol` convergence criteria. The exact parameter name for this in `optimistix` needs to be confirmed and implemented by Code Mode (it might be `norm` or similar).

**4. Script Execution (Requires Code Mode):**
*   After modifications are implemented by Code mode, execute the updated script.
    *   Command: `python scripts/test_bif_priors_optimizers.py`
*   Monitor execution for errors.

**5. Result Analysis:**
*   **Primary Focus (Stability):** Analyze the generated CSV file (`bif_prior_optimizer_results_rms.csv`).
    *   Filter results by `Success == 'No'`. Examine `Error Message`.
    *   Compare frequency and types of errors across `Prior Config`, `Optimizer`, `Transform`. Identify configurations that reduce failures, especially noting the performance of the RMS-norm-based Optax solvers.
*   **Secondary Focus (Convergence/Efficiency):** For successful runs, analyze `Final Loss`, `Steps`, and `Time (s)` from the CSV.

**6. Documentation (Optional):**
*   Create a Markdown report summarizing the analysis, focusing on stability findings with the RMS norm solvers.
*   Include key tables/summaries derived from the CSV output.

**Execution Flow Diagram (Mermaid):**

```mermaid
graph TD
    A[Start] --> B{Plan Script Modifications};
    B --> B1[Plan CSV Output];
    B --> B2[Plan RMS Norm for Optax Solvers (Code Mode Task)];
    B --> B3[Confirm Determinism (No Change Needed)];
    B1 & B2 & B3 --> C{Confirm Plan with User};
    C -- User Confirms --> D{Switch to Code Mode};
    D --> E[Implement Modifications (Code Mode)];
    E --> F{Execute Modified Script (Code Mode)};
    F --> G[Generate CSV Results];
    G --> H{Analyze Results (Architect/User)};
    H --> H1[Focus on Stability (Errors, Success Rate)];
    H --> H2[Analyze Convergence/Efficiency];
    H1 & H2 --> I{Document Findings (Optional)};
    I --> J[End];
    F -- Error --> J;
    C -- User Requests Changes --> B;