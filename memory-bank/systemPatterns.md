# System Patterns

This file documents recurring patterns and standards used in the project.

*

## Coding Patterns

*   **JAX Ecosystem Integration:** Heavy reliance on JAX for performance-critical computations, especially within filtering algorithms. This includes:
    *   `@jit` for function compilation.
    *   `jax.lax.scan` for efficient temporal loops (e.g., in Particle Filter).
    *   `jax.vmap` for parallel computation over particles/data.
    *   `jax.lax.cond` for conditional execution (e.g., resampling).
    *   `jax.random` for stochastic operations.
    *   JAX optimization libraries (`jaxopt`, `optimistix`) for parameter updates (e.g., BFGS in Bellman Filter).
*   **JAX Pytree Parameters:** Use of `jax_dataclasses` (`@jdc.pytree_dataclass`) to define model parameters (e.g., `DFSVParamsDataclass` in `models/dfsv.py`). This allows parameter objects to be treated as JAX pytrees, enabling direct use with JAX transformations (`jit`, `grad`, etc.) and clear separation of static (`N`, `K`) vs. differentiable parameters.
*   **Helper Modules:** Encapsulation of complex mathematical implementations or optimization steps into internal helper modules (e.g., `_bellman_impl.py`, `_bellman_optim.py`) to keep primary class definitions cleaner.
*   **Parameter Passing:** Refactored `DFSVParticleFilter` to accept parameters externally via method arguments rather than storing them internally, facilitating JIT compilation across different parameter sets (Decision [2025-04-01 03:37:00]).

## Architectural Patterns

*   **Filter Base Class:** Filtering implementations (`DFSVBellmanFilter`, `DFSVParticleFilter`) inherit from a common base class (`core/filters/base.py`), suggesting a common interface for filters.
*   **Bellman Filter Strategy (Lange, 2024):**
    *   Approximates the posterior mode using dynamic programming.
    *   Update step involves maximizing `ℓ(y_t|α_t) - 1/2 ||α_t - a_{t|t-1}||^2_{I_{t|t-1}}`.
    *   Precision update uses Fisher Information: `I_{t|t} = I_{t|t-1} + (-E[d^2ℓ/dα^2 | α_t])`.
    *   Current implementation uses block coordinate descent for the state update optimization.
    *   Calculates Fisher Information assuming `I_fh = 0` (block-diagonal FIM).
    *   Uses an approximation for state-dependent process noise `Q_f` in prediction (Corrected [2025-04-01 02:27:00]).
    *   **Sensitivity:** Implementation is numerically sensitive to small process noise `Q_h`. When `Q_h` is small, the predicted precision `I_pred` dominates the likelihood information `I_fisher` in the update step, hindering log-volatility estimation. Using a larger `Q_h` in simulations mitigates this (Identified [2025-04-01 02:28:00], Confirmed [2025-04-01 02:48:00]).
*   **Particle Filter Strategy:** Implements a Bootstrap Filter (SISR) using `jax.lax.scan` for the main loop, with ESS-triggered Systematic Resampling to mitigate particle degeneracy.
*   **Optimization Pattern (Woodbury Identity & MDL):** Applied to `log_posterior_impl` (`_bellman_impl.py`) to avoid direct O(N^3) Cholesky decomposition/inversion of the N x N observation covariance matrix `A = D + UCU.T`. Replaced with operations involving the inverse of the K x K matrix `M = C^-1 + U.T D^-1 U`, reducing complexity significantly when N >> K (Implemented [2025-04-01 11:46:00]). Also applied to `fisher_information_impl` for `A_inv` applications (Implemented [2025-04-01 12:05:00]).
*   **Optimization Pattern (Rank-1 FIM Reformulation):** Utilized the rank-1 structure of `dSigma_k = exp(h_k) * lambda_k * lambda_k.T` to reformulate the calculation of the `I_hh` block in `fisher_information_impl`. Changed the O(K^2*N^2) `einsum` calculation `trace(B_k B_l)` to the equivalent `exp(h_k + h_l) * (lambda_k.T @ A_inv @ lambda_l)^2`, where the inner term is reused from the `I_ff` calculation. This avoids the expensive `einsum` (Implemented [2025-04-01 12:05:00]).
*   **Optimization Pattern (Diagonal Covariance):** Optimized particle filter likelihood calculation (`compute_log_likelihood_particle`) to exploit known diagonal structure of observation noise covariance `R_t`, avoiding redundant `vmap`ped decompositions (Implemented [2025-04-01 05:13:00]).

## Testing Patterns

*   **Unit Tests:** Use `pytest` framework for running tests located in the `tests/` directory.
*   **Filter Comparison:** Tests often compare outputs of different filter implementations or optimized vs. non-optimized versions (e.g., `tests/test_bellman_unified.py`).
*   **Profiling Scripts:** Dedicated scripts (`scripts/profile_*.py`, `scripts/benchmark_*.py`) used to measure performance and identify bottlenecks.

---
**Update Log:**

*   [2025-04-01 01:06:48] - Initial log entry.
*   [2025-04-01 01:10:14] - Added initial analysis of JAX usage, parameter handling, filter strategies, base class inheritance, and helper modules.
*   [2025-04-01 02:28:00] - Noted Bellman filter sensitivity to `Q_h`.
*   [2025-04-01 02:48:00] - Added details on Bellman filter implementation and confirmed `Q_h` sensitivity.
*   [2025-04-01 05:13:00] - Added particle filter diagonal covariance optimization pattern.
*   [2025-04-01 11:46:00] - Added Woodbury Identity optimization pattern for Bellman filter.
*   [2025-04-01 12:05:00] - Added Rank-1 FIM reformulation optimization pattern for Bellman filter.
*   [2025-04-01 12:22:00] - Refined descriptions of Woodbury and Rank-1 FIM optimization patterns.
*   [2025-04-01 21:58:34] - Merged duplicate sections, integrated log entries, and consolidated pattern descriptions during Memory Bank cleanup.