# Product Context

This file provides a high-level overview of the project and the expected product that will be created. Initially it is based upon projectBrief.md (if provided) and all other available project-related information in the working directory. This file is intended to be updated as the project evolves, and should be used to inform all other modes of the project's goals and context.

*

## Project Goal

*   Implement and evaluate the Bellman filter for Dynamic Factor Stochastic Volatility (DFSV) models, comparing its performance against standard Particle Filters.
*   Estimate model hyperparameters by maximizing the filter-implied pseudo log-likelihood.

## Key Features

*   Implementation of the DFSV model.
*   Implementation of the Bellman filter (Lange, 2024).
*   Implementation of a Particle Filter (Bootstrap/SISR).
*   Simulation capabilities for model testing.
*   Hyperparameter estimation framework.

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

### DFSV Model Specification

Based on the simulation code (`core/simulation.py`) and thesis proposal:

**State Variables:**
*   `f_t`: Latent factors (K-dimensional vector)
*   `h_t`: Log-volatilities (K-dimensional vector)
*   State Vector: `α_t = [f_t', h_t']'`

**State Transition Equations:**
1.  **Log-Volatility:** `h_{t+1} = μ + Φ_h (h_t - μ) + η_{t+1}`, where `η_{t+1} ~ N(0, Q_h)`
    *   `μ`: Long-run mean vector (K x 1)
    *   `Φ_h`: Transition matrix (K x K)
    *   `Q_h`: Process noise covariance (K x K)
2.  **Factor:** `f_{t+1} = Φ_f f_t + ν_{t+1}`
    *   `Φ_f`: Transition matrix (K x K)
    *   Factor Innovation Covariance: State-dependent: `ν_{t+1} ∼ N(0, diag(e^{h_{1,t+1}}, ..., e^{h_{K,t+1}}))`. Depends on `h` at time `t+1`. (Note: Simulation code uses `diag(exp(h_t / 2)) * ε_t` where `ε_t ~ N(0, I_K)`, implying covariance depends on `h_t`, check consistency).

**Observation Equation:**
*   `r_t = Λ f_t + ε_t`
    *   `r_t`: Observations (N-dimensional vector)
    *   `Λ`: Factor loadings (N x K, `lambda_r` in code)
    *   `ε_t`: Idiosyncratic errors with `Var(ε_{i,t}) = exp(h^{(id)}_{i,t})`. Baseline assumes constant `h^{(id)}` (often diagonal `diag(sigma2)` in code, represented by `Σ`).

### Estimation Approach

*   **Filter:** Bellman filter (Lange, 2024) used for state estimation (`α_t`). Based on dynamic programming, approximates posterior mode.
*   **Hyperparameter Estimation:** Maximize filter-implied pseudo log-likelihood (Eq. 40 in Lange, 2024) using gradient-based optimization (potentially via AD).

---
**Update Log:**

*   [2025-04-01 01:06:28] - Initial Memory Bank setup.
*   [2025-04-01 01:12:01] - Added initial architecture summary based on `src/` directory analysis.
*   [2025-04-01 01:13:06] - Added mathematical specification of the DFSV model under Overall Architecture based on simulation code.
*   [2025-04-01 02:47:00] - Added Model Specification and Estimation Approach details from Thesis Proposal & Lange (2024).
*   [2025-04-01 21:57:48] - Consolidated architecture description, model specification, and moved log entries to footnotes during Memory Bank cleanup.