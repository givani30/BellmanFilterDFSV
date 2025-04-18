    # Product Context

    This file provides a high-level overview of the project and the expected product that will be created. Initially it is based upon projectBrief.md (if provided) and all other available project-related information in the working directory. This file is intended to be updated as the project evolves, and should be used to inform all other modes of the project's goals and context.

    *

    ## Project Goal

    *   Implement and evaluate the Bellman filter (Lange, 2024) for the specific Dynamic Factor Stochastic Volatility (DFSV) model with VAR(1) factor/volatility dynamics outlined in the thesis proposal (Boekestijn, 2025).
    *   Compare the Bellman filter's performance (accuracy, speed, stability) against a standard Particle Filter implementation.
    *   Estimate model hyperparameters (Λ, Φ_f, Φ_h, μ, Q_h) by maximizing the filter-implied pseudo log-likelihood derived from the Bellman filter.

    ## Key Features

    *   Implementation of the DFSV model with VAR(1) factor and log-volatility dynamics.
    *   Implementation of the Bellman filter based on Lange (2024).
    *   Implementation of a Particle Filter (Bootstrap/SISR) for benchmarking.
    *   Simulation capabilities for model testing and evaluation.

        *   Supports a wide range of optimizers (BFGS variants, AdamW, SGD, Lion, etc.) and learning rate schedules via centralized utilities (`solvers.py`, `optimization.py`).
        *   Includes parameter transformation, optional parameter fixing (e.g., `mu`), and detailed logging capabilities.
        *   Unified script (`unified_filter_optimization.py`) enables systematic comparison of filters and optimizers.
    *   Hyperparameter estimation framework using the Bellman filter's pseudo log-likelihood.

    ## Overall Architecture

    *   The project is structured as a Python package (`src/bellman_filter_dfsv/`) built heavily on the JAX library.
    *   **Main Components:**
        *   `filters/`: Implements state estimation algorithms (Bellman, Particle) inheriting from `base.py`. Includes optimization objective functions (`objectives.py`) and internal helpers (e.g., `_bellman_impl.py`).
            *   **API Consistency:** Filter classes adhere to a consistent API defined by `DFSVFilter` (in `base.py`), including standardized methods (`log_likelihood_wrt_params`, `jit_log_likelihood_wrt_params`, etc.).
        *   `models/`: Defines the model structure (`dfsv.py`), simulation logic (`simulation.py`), and model-specific likelihood/prior functions (`likelihoods.py`).
        *   `utils/`: Contains utility functions (e.g., `transformations.py`, `jax_helpers.py`).
    *   **Parameterization:** Uses `DFSVParamsDataclass` (a JAX Pytree) for consistent parameter handling across JAX-based components.

    ### DFSV Model Specification (Boekestijn, 2025, Ch. 3)

    **State Variables:**
    *   `f_t`: Latent factors (K-dimensional vector)
    *   `h_t`: Latent log-volatilities of factors (K-dimensional vector)
    *   State Vector: `α_t = [f_t', h_t']'` (2K-dimensional vector)

    **State Transition Equations:**
    1.  **Factor Evolution (VAR(1)):**
        `f_{t+1} = Φ_f f_t + ν_{t+1}`
        `ν_{t+1} ∼ N(0, diag(e^{h_{1,t+1}}, ..., e^{h_{K,t+1}}))` (Eq. 3.5)
        *   `Φ_f`: Factor transition matrix (K x K)
        *   Factor innovations `ν_{t+1}` have time-varying covariance dependent on *next period's* log-volatilities `h_{t+1}`.
    2.  **Log-Volatility Evolution (VAR(1)):**
        `h_{t+1} = μ + Φ_h (h_t - μ) + η_{t+1}`
        `η_{t+1} ∼ N(0, Q_h)` (Eq. 3.6)
        *   `μ`: Long-run mean vector of log-volatilities (K x 1)
        *   `Φ_h`: Log-volatility transition matrix (K x K)
        *   `Q_h`: Log-volatility process noise covariance (K x K, positive-definite)

    **Observation Equation:**

    * `r_t = Λ f_t + ε_t` (Eq. 3.1)
        * `r_t`: Observations (N-dimensional vector)
        * `Λ`: Factor loadings (N x K matrix, parameter `lambda_r`)
        * `f_t`: Latent factors (K-dimensional vector)
        * `ε_t`: Idiosyncratic errors, assumed `ε_t ∼ N(0, Σ_ε)`.

    * **Idiosyncratic Error Covariance `Σ_ε`:**
        * `Σ_ε = diag(sigma2)` represents the covariance of the idiosyncratic errors. In the current implementation, this component is assumed constant over time and its diagonal elements are parameters contained in `sigma2`.

    * **Effective Conditional Distribution `p(r_t | α_t)`:**
        * Given the full state `α_t = [f_t', h_t']'`, the filter's likelihood calculations (e.g., `log_posterior_impl` [cite: 18]) operate based on the effective conditional distribution for the observation `r_t`, which is Gaussian:
            `r_t | f_t, h_t ∼ N(Λ f_t, A_t)`

    * **Effective Observation Covariance `A_t`:**
        * The effective covariance matrix `A_t` used in the likelihood calculation incorporates variance from both the stochastic factors (dependent on `h_t`) and the idiosyncratic noise:
            `A_t = Λ Var(f_t | h_t) Λ^T + Σ_ε`
        * As implemented in `build_covariance_impl`[cite: 18], this covariance is calculated using the current log-volatility state `h_t` as:
            `A_t = Λ diag(exp(h_t)) Λ^T + Σ_ε`
        * **Key Point:** Because `A_t` includes the `exp(h_t)` term, the conditional distribution of `r_t` used within the filter *does* depend on the state `h_t` through this effective covariance[cite: 18]. This means the observation `r_t` provides information relevant for estimating `h_t` within the filter's update step.


    ### Estimation Approach

    *   **State Estimation:** Use the Bellman filter (Lange, 2024) to recursively estimate the posterior mode of the state vector `α_t`.
    *   **Hyperparameter Estimation:** Maximize the filter-implied pseudo log-likelihood (Eq. 40 in Lange, 2024 / Eq. 3.11 in Boekestijn, 2025) with respect to static hyperparameters Θ = {Λ, Φ_f, Φ_h, μ, Q_h, Σ_ε}. Optimization via gradient-based methods (e.g., BFGS), potentially using Automatic Differentiation for gradients. The pseudo-likelihood involves a fit term `ln p(r_t | α̂_{t|t})` penalized by a KL-divergence-like term based on predicted and filtered state estimates and their precisions.

    ### Likelihood Surface Details (Bellman Filter Pseudo Log-Likelihood)

    The optimization process targets the pseudo log-likelihood derived from the Bellman filter (Lange, 2024, Eq. 40; Boekestijn, 2025, Eq. 3.11). Key characteristics include:

    *   **Nature:** This is an *approximation* of the true marginal log-likelihood `p(y | Θ)`. It's based on the filter's recursive estimation of the posterior mode (`α̂_{t|t}`) and precision of the latent state vector `α_t`.
    *   **Components:** It implicitly combines:
        *   A measure of observation fit given the estimated state mode (`ln p(r_t | α̂_{t|t})`).
        *   A penalty term, analogous to a KL divergence, comparing the predicted state distribution (`p(α_t | y_{1:t-1})`) with the filtered state distribution (`p(α_t | y_{1:t})`).
    *   **Optimization Strategy:**
        *   The objective is to find hyperparameters Θ = {Λ, Φ_f, Φ_h, μ, Q_h, Σ_ε} that maximize this pseudo-likelihood (or minimize its negative).
        *   Gradient-based optimization (e.g., BFGS) is employed, utilizing JAX for automatic differentiation (`jax.grad`, `jax.value_and_grad`).
        *   Optimization often occurs in an *unconstrained* parameter space (using `transformed_bellman_objective`). The `untransform_params` function maps these values back to the constrained model space (e.g., ensuring positive variances, valid correlation matrices) before calculating the objective. This transformation aims to improve optimization geometry.
    *   **Implementation:**
        *   The objective function logic resides in `src/bellman_filter_dfsv/core/likelihood.py` (`bellman_objective`, `transformed_bellman_objective`).
        *   The core likelihood calculation is delegated to the `DFSVBellmanFilter` class (`jit_log_likelihood_of_params`).
        *   JAX (`@eqx.filter_jit`) is used for performance via JIT compilation.
    *   **Regularization:** Prior beliefs can be incorporated by adding log-prior density terms to the objective function (e.g., using `log_prior_density` for `sigma2` or custom penalties as seen for `mu` in `bellman_objective`). This modifies the surface to favor certain parameter regions.
    *   **Potential Characteristics:** As with many complex latent variable models, the surface may exhibit:
        *   Multi-modality (multiple local optima).
        *   Flat regions or sharp ridges, potentially hindering optimizer convergence.
        *   Numerical stability issues, addressed through techniques like `jnp.nan_to_num` and parameter transformations.
    **Known Issues with BIF Pseudo-Likelihood Estimation:**

    * **Historical Note on `mu` Estimation:** Initial testing of the BIF pseudo-likelihood (Eq. 40 in Lange, 2024 [cite: 1]) suggested significant bias in estimating the long-run mean log-volatility parameter `mu`. However, recent hyperparameter studies have shown that the impact of this bias is less severe than initially thought, and both fixed and unfixed `mu` approaches can be viable depending on the specific context.
    * **Numerical Stability in Information Update:** The Observed Fisher Information calculation has been replaced with Expected Fisher Information to improve numerical stability (see Decision [14-04-2025 02:30:00]). This modification, along with eigenvalue regularization in the BIF update step (`__update_jax_info` [cite: 21]), ensures robust filter operation.
    * **Current Approach:** The decision to fix or estimate `mu` is now made on a case-by-case basis, considering factors such as the specific filter used (BIF, BF, PF), the availability of prior information, and the goals of the analysis. When using BIF with unfixed `mu`, proper numerical stabilization techniques should be employed.

    ---

    *   [16-04-2025 02:55:16] - Updated documentation regarding `mu` fixing strategy based on recent hyperparameter studies. Clarified that both fixed and unfixed approaches are viable depending on context.
    **Update Log:**

    *   [01-04-2025 01:06:28] - Initial Memory Bank setup.
    *   [01-04-2025 01:12:01] - Added initial architecture summary based on `src/` directory analysis.
    *   [01-04-2025 01:13:06] - Added mathematical specification of the DFSV model under Overall Architecture based on simulation code.

    *   [04-06-2025 01:26:00] - Added note about standardized filter API consistency to Overall Architecture.
    *   [01-04-2025 02:47:00] - Added Model Specification and Estimation Approach details from Thesis Proposal & Lange (2024).
    *   [01-04-2025 21:57:48] - Consolidated architecture description, model specification, and moved log entries to footnotes during Memory Bank cleanup.