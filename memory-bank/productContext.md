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
*   Hyperparameter estimation framework using the Bellman filter's pseudo log-likelihood.

## Overall Architecture

*   The project is structured as a Python package (`src/bellman_filter_dfsv/`) built heavily on the JAX library.
*   **Core Components:**
    *   `core/`: Contains the main filtering logic.
        *   `filters/`: Implements different state estimation algorithms (Bellman, Particle) inheriting from a base class (`base.py`). Mathematical details are often separated into internal modules (e.g., `_bellman_impl.py`).
        *   `likelihood.py`: Handles likelihood calculations.
        *   `simulation.py`: Provides simulation capabilities.
    *   `models/`: Defines the model structure, primarily the Dynamic Factor Stochastic Volatility (DFSV) model parameters using a JAX-compatible dataclass (`dfsv.py`).
    *   `utils/`: Contains utility functions (e.g., `transformations.py`).
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
*   `r_t = Λ f_t + ε_t` (Eq. 3.1)
*   `ε_t ∼ N(0, Σ_ε)` where `Σ_ε = diag(e^{h^{(id)}_{1,t}}, ..., e^{h^{(id)}_{N,t}})` (Eq. 3.2, 3.4)
    *   `r_t`: Observations (N-dimensional vector)
    *   `Λ`: Factor loadings (N x K)
    *   `ε_t`: Idiosyncratic errors. Baseline assumes fixed idiosyncratic log-volatilities `h^{(id)}_{i,t} = h^{(id)}_{i}` over time.

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


---
**Update Log:**

*   [2025-04-01 01:06:28] - Initial Memory Bank setup.
*   [2025-04-01 01:12:01] - Added initial architecture summary based on `src/` directory analysis.
*   [2025-04-01 01:13:06] - Added mathematical specification of the DFSV model under Overall Architecture based on simulation code.
*   [2025-04-01 02:47:00] - Added Model Specification and Estimation Approach details from Thesis Proposal & Lange (2024).
*   [2025-04-01 21:57:48] - Consolidated architecture description, model specification, and moved log entries to footnotes during Memory Bank cleanup.