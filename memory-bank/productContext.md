# Product Context

This file provides a high-level overview of the project and the expected product that will be created. Initially it is based upon projectBrief.md (if provided) and all other available project-related information in the working directory. This file is intended to be updated as the project evolves, and should be used to inform all other modes of the project's goals and context.
2025-04-01 01:06:28 - Log of updates made will be appended as footnotes to the end of this file.

*

## Project Goal

*   

## Key Features

*   

## Overall Architecture


[2025-04-01 01:12:01] - Added initial architecture summary based on `src/` directory analysis.


*   [2025-04-01 01:12:01] - The project is structured as a Python package (`src/bellman_filter_dfsv/`) built heavily on the JAX library.
*   **Core Components:**
    *   `core/`: Contains the main filtering logic.
        *   `filters/`: Implements different state estimation algorithms (Bellman, Particle) inheriting from a base class (`base.py`). Mathematical details are often separated into internal modules (e.g., `_bellman_impl.py`).
        *   `likelihood.py`: Handles likelihood calculations.
        *   `simulation.py`: Provides simulation capabilities.
    *   `models/`: Defines the model structure, primarily the Dynamic Factor Stochastic Volatility (DFSV) model parameters using a JAX-compatible dataclass (`dfsv.py`).
    *   `utils/`: Contains utility functions (e.g., `transformations.py`).

[2025-04-01 01:13:06] - Added mathematical specification of the DFSV model under Overall Architecture.


    ### DFSV Model Specification ([2025-04-01 01:13:06])

    Based on the simulation code (`core/simulation.py`), the model follows this state-space structure:

    **State Variables:**
    *   `f_t`: Latent factors (K-dimensional vector)
    *   `h_t`: Log-volatilities (K-dimensional vector)

    **State Transition Equations:**
    1.  **Log-Volatility:** `h_t = μ + Φ_h * (h_{t-1} - μ) + η_t`, where `η_t ~ N(0, Q_h)`
        *   `μ`: Long-run mean vector (K x 1)
        *   `Φ_h`: Transition matrix (K x K)
        *   `Q_h`: Process noise covariance (K x K)
    2.  **Factor:** `f_t = Φ_f * f_{t-1} + diag(exp(h_t / 2)) * ε_t`, where `ε_t ~ N(0, I_K)`
        *   `Φ_f`: Transition matrix (K x K)
        *   Process noise covariance depends on `h_t`: `diag(exp(h_t))`

    **Observation Equation:**
    *   `r_t = Λ * f_t + e_t`, where `e_t ~ N(0, Σ)`
        *   `r_t`: Observations (N-dimensional vector)
        *   `Λ`: Factor loadings (N x K, `lambda_r` in code)
        *   `Σ`: Observation noise covariance (N x N, often diagonal `diag(sigma2)` in code)
        *   `filters/`: Implements different state estimation algorithms (Bellman, Particle) inheriting from a base class (`base.py`). Mathematical details are often separated into internal modules (e.g., `_bellman_impl.py`).
        *   `likelihood.py`: Handles likelihood calculations.
        *   `simulation.py`: Provides simulation capabilities.
    *   `models/`: Defines the model structure, primarily the Dynamic Factor Stochastic Volatility (DFSV) model parameters using a JAX-compatible dataclass (`dfsv.py`).
    *   `utils/`: Contains utility functions (e.g., `transformations.py`).
*   **Parameterization:** Uses `DFSVParamsDataclass` (a JAX Pytree) for consistent parameter handling across JAX-based components.
*