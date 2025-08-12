# Comparison: Thesis Proposal Methodology vs. Current Implementation (07-04-2025)

This document outlines the key differences and missing elements when comparing the methodology described in the thesis proposal (Chapter 3) against the current project implementation, based on the analysis in `outputs/filter_implementation_analysis.md`. This serves as a guide for updating the thesis methodology section.

**Key Differences/Missing Elements:**

1.  **Primary Filter Specification:**
    *   **Proposal:** Describes the Bellman Filter (BF) generally.
    *   **Implementation:** Primarily uses the **Bellman Information Filter (BIF)** due to numerical stability advantages.
    *   **Thesis Update:** Explicitly state BIF is the core filter, detailing its information-form propagation.

2.  **BIF Implementation Details:**
    *   **Proposal:** Mentions standard prediction/update steps and pseudo-likelihood (Eqs. 3.8-3.11).
    *   **Implementation:** Uses specific stability techniques: **Joseph form prediction** and **eigenvalue clipping regularization** on the Observed Fisher Information (**_J_**$_{observed}$).
    *   **Thesis Update:** Detail these specific BIF techniques and their rationale. Clarify pseudo-likelihood calculation follows Lange (2024, Eq. 40).

3.  **State Update Optimization Algorithm:**
    *   **Proposal:** Defines the optimization problem (Eq. 3.10) but doesn't specify the algorithm.
    *   **Implementation:** Both BF and BIF use iterative **block coordinate descent** (`_block_coordinate_update_impl`) to find the posterior mode $\boldsymbol{\hat{\alpha}}_{t|t}$.
    *   **Thesis Update:** Document the use of **block coordinate descent** as the specific algorithm for the Bellman filter update step.

4.  **Handling of _μ_ Parameter:**
    *   **Proposal:** Does not mention specific challenges with estimating **_μ_**.
    *   **Implementation:** Identified bias in **_μ_** estimation using BIF pseudo-likelihood; adopted strategy is to **fix _μ_** during hyperparameter estimation.
    *   **Thesis Update:** Clearly document the strategy of fixing **_μ_**, the rationale (bias), and implications. This is a critical deviation.

5.  **Benchmark Filter:**
    *   **Proposal:** Mentions EKF, UKF, PF as alternatives.
    *   **Implementation:** Specifically implemented a **Particle Filter (Bootstrap SISR with Systematic Resampling)**.
    *   **Thesis Update:** Specify the PF (SISR) as the chosen benchmark and describe its features.

6.  **Parameter Transformations:**
    *   **Proposal:** Does not detail handling of parameter constraints during optimization.
    *   **Implementation:** Uses functions (`softplus`, `logit`/`tanh`, etc. in `utils/transformations.py`) for mapping between constrained and unconstrained spaces.
    *   **Thesis Update:** Describe the use of these transformations.

7.  **Full Persistence Matrices:**
    *   **Proposal:** Does not explicitly state if **_Φ_**$_f$, **_Φ_**$_h$ are diagonal or full.
    *   **Implementation:** Supports full matrices, stabilized via `softplus` transformation + penalty.
    *   **Thesis Update:** Clarify support for full matrices and the stabilization method.

8.  **Covariance-Based BF Status:**
    *   **Proposal:** Focuses on the general Bellman filter.
    *   **Implementation:** Original BF faced instability; later updated to use BIF pseudo-likelihood.
    *   **Thesis Update:** Briefly mention the history, limitations, and current status of the covariance BF.

**Conclusion:** The final methodology chapter needs significant updates to reflect the specific filter choices (BIF primary, PF benchmark), practical adaptations (fixing **_μ_**, stability enhancements), and key technical details (block coordinate descent, parameter transformations, full persistence matrices).