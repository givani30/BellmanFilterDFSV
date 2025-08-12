# Plan: Thesis Update Summary (April 2025)

**Objective:** Draft a concise and informative summary highlighting the progress made on the "Bellman Filtering for Dynamic Factor SV models" thesis over the past month (approx. March 4th - April 6th, 2025).

**Information Sources:**
*   Thesis Proposal (`QF_Thesis_Proposal_GBoekestijn.pdf`) - As the baseline.
*   Memory Bank Files - For detailed progress, decisions, challenges, and code structure.
*   Project Files/Codebase (implicitly understood from Memory Bank).

**Proposed Summary Structure & Content:**

1.  **Introduction:**
    *   State the period covered by the update (approx. past month).
    *   Briefly reference the submitted proposal as the starting point.
    *   Mention the overall goal: Implementing and evaluating the Bellman filter for the proposed DFSV model.

2.  **Core Implementation Progress:**
    *   **Model & Filters:** Briefly state the successful implementation of:
        *   The Dynamic Factor Stochastic Volatility (DFSV) model structure (using JAX).
        *   The Bellman Information Filter (BIF) based on Lange (2024).
        *   A Particle Filter (PF) for benchmarking purposes.
        *   (Mention the initial covariance-based Bellman Filter implementation and the switch to BIF due to stability issues).
    *   **Simulation Framework:** Note the development of code to simulate data from the DFSV model for testing.

3.  **Debugging, Stabilization & Optimization:**
    *   **BIF Stability:** Highlight the significant effort in debugging the BIF, identifying numerical instability related to the Fisher Information Matrix calculation, and the successful implementation of a regularization fix (eigenvalue clipping). Emphasize that the BIF is now stable, even when run with true parameters.
    *   **Code Optimization:** Mention optimizations applied to filter implementations (e.g., using JAX features like `@equinox.filter_jit`, `scan`, and mathematical optimizations like Woodbury identity) leading to performance improvements.
    *   **Code Refinement:** Note the refactoring efforts for JIT compatibility, unifying filter APIs under a base class, and general code cleanup, leading to a more robust and maintainable codebase.

4.  **Hyperparameter Estimation:**
    *   **Framework:** Mention the setup of the framework for estimating static model parameters (Λ, Φ_f, Φ_h, μ, Q_h, Σ_ε) by maximizing the BIF's pseudo log-likelihood.
    *   **Experiments & Findings:** Summarize the initial experiments comparing optimizers (Adam/AdamW) and the impact of priors. Key findings to include:
        *   Optimization without priors is unstable (especially for BIF).
        *   Implementing a prior framework significantly improves stability but can slow convergence.
        *   Persistent challenges in accurately estimating certain parameters (notably `mu`, the long-run volatility mean), indicating potential identifiability issues even with structural constraints (tested on `lambda_r`).

5.  **Testing & Infrastructure:**
    *   **Testing:** Mention the unification and standardization of the test suite using `pytest`, resulting in a comprehensive set of passing tests (~49 tests) that verify filter functionality and stability.
    *   **(Optional) Simulation Infrastructure:** Briefly mention the setup for running larger simulation studies, including resume capabilities and the exploration of Google Cloud Batch.

6.  **Current Status & Next Steps:**
    *   State that the codebase is currently stable, well-tested, and key filtering components (BIF, PF) are functional.
    *   Outline high-level next steps, likely focusing on:
        *   Further investigation and refinement of the hyperparameter estimation process (tuning priors/optimizers, potentially exploring EM).
        *   Conducting simulation studies using the stabilized BIF.
        *   Analyzing simulation results.
        *   Preparing for application to real-world data as outlined in the proposal.

**Format:** The summary will be drafted as clear, concise text, suitable for an email or short report.



# MAIL

Subject: Thesis Progress Update: Bellman Filtering for DFSV Models (April 2025)

Dear Rutger-Jan,

This email summarizes the progress made on my MSc thesis, "Bellman Filtering for Dynamic Factor SV models," over the past month, following the submission of the proposal draft around March 4th. The primary focus has been on implementing and evaluating the Bellman filter methodology for the proposed DFSV model.

**1. Core Implementation:**

*   The Dynamic Factor Stochastic Volatility (DFSV) model, as specified in the proposal (VAR(1) dynamics for factors and log-volatilities), has been implemented using Python and the JAX library.
*   The Bellman filter *methodology* from your 2024 Journal of Econometrics paper has been successfully implemented, specifically using an information filter formulation (propagating the mode and precision matrix) within the `DFSVBellmanInformationFilter` class (`src/bellman_filter_dfsv/core/filters/bellman_information.py`).
*   A standard Particle Filter (Bootstrap/SISR) was also implemented (`src/bellman_filter_dfsv/core/filters/particle.py`) to serve as a benchmark.
*   A simulation framework (`src/bellman_filter_dfsv/core/simulation.py`) is in place to generate data from the DFSV model for testing and evaluation.
*   *Note:* An initial attempt to implement the Bellman filter using a covariance-based formulation encountered significant numerical stability challenges during hyperparameter optimization, leading to the adoption of the information filter approach.

**2. Debugging, Stabilization & Optimization:**

*   A major focus was placed on debugging and stabilizing the Bellman Information Filter (BIF) implementation. Numerical instability was traced to the calculation of the Expected Fisher Information Matrix (`J_observed`) within the filter's update step, which occasionally became non-positive semi-definite. This was resolved by implementing an eigenvalue clipping regularization fix, ensuring the matrix remains PSD. The BIF now runs stably, including tests using the true simulation parameters. This regularization was a necessary practical addition to ensure stability for this specific model.
*   Significant effort was invested in optimizing the filter implementations, leveraging JAX features (`@equinox.filter_jit`, `jax.lax.scan`) and applying mathematical optimizations (e.g., Woodbury Identity, Rank-1 FIM reformulation) which yielded substantial performance improvements.
*   The codebase underwent considerable refinement, including refactoring for better JIT compatibility (removing Python control flow from JIT paths), unifying the filter APIs under a common base class (`DFSVFilter`), and general code cleanup, resulting in a more robust and maintainable structure.

**3. Hyperparameter Estimation:**

*   The framework for estimating the static model hyperparameters (Λ, Φ_f, Φ_h, μ, Q_h, Σ_ε) by maximizing the BIF's pseudo log-likelihood (Eq. 40 in your paper) is established (`src/bellman_filter_dfsv/core/likelihood.py`).
*   Initial experiments were conducted comparing different optimizers (Adam, AdamW) and assessing the impact of Bayesian priors. Key findings indicate that optimization without priors is numerically unstable for the BIF, while adding priors significantly improves stability but can slow down convergence.
*   Persistent challenges were observed in accurately estimating certain parameters, particularly the long-run log-volatility mean (`mu`), even when applying structural identifiability constraints (e.g., fixing the diagonal of the factor loading matrix `lambda_r` to 1). This suggests potential identifiability issues within this specific model/filter combination that may require careful prior specification or alternative estimation strategies.

**4. Testing & Simulation Infrastructure:**

*   The testing framework has been unified and standardized using `pytest`. A comprehensive suite of tests (~49 tests) covering filter functionality, stability, transformations, and API consistency is in place, and all tests are currently passing.
*   Infrastructure for larger simulation studies was developed, including resume capabilities for long-running scripts and the setup for distributed execution using Google Cloud Batch.

**5. Current Status & Next Steps:**

*   The codebase is currently stable, the core filtering algorithms (BIF and PF) are functional and well-tested. The simulation experiments comparing the filters have been executed using the developed infrastructure.
*   The immediate next steps involve:
    *   Processing and analyzing the results from the completed simulation studies to compare the performance (accuracy, speed) of the stabilized BIF against the Particle Filter.
    *   Further investigation and refinement of the hyperparameter estimation process based on simulation findings and the identified challenges (tuning priors/optimizers, potentially exploring EM).
    *   Preparing for the application of the model and filters to real-world financial data as outlined in the proposal.

I believe significant technical progress has been made in establishing a robust implementation, executing the initial simulation comparison, and identifying key challenges related to estimation. I look forward to discussing these results and the next steps further.

Best regards,

Givani Boekestijn