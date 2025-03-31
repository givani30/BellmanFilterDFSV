# System Patterns *Optional*

This file documents recurring patterns and standards used in the project.
It is optional, but recommended to be updated as the project evolves.
2025-04-01 01:06:48 - Log of updates made.

*

## Coding Patterns

*   

## Architectural Patterns

*   

## Testing Patterns


[2025-04-01 01:10:14] - Added initial analysis of JAX usage, parameter handling (DFSVParamsDataclass), filter strategies (Bellman, Particle), base class inheritance, and helper modules based on code review of `src/`.

*   **JAX Ecosystem Integration:** Heavy reliance on JAX for performance-critical computations, especially within filtering algorithms. This includes:
    *   `@jit` for function compilation.
    *   `jax.lax.scan` for efficient temporal loops (e.g., in Particle Filter).
    *   `jax.vmap` for parallel computation over particles/data.
    *   `jax.lax.cond` for conditional execution (e.g., resampling).
    *   `jax.random` for stochastic operations.
    *   JAX optimization libraries (`jaxopt`, `optimistix`) for parameter updates (e.g., BFGS in Bellman Filter).
*   **JAX Pytree Parameters:** Use of `jax_dataclasses` (`@jdc.pytree_dataclass`) to define model parameters (e.g., `DFSVParamsDataclass` in `models/dfsv.py`). This allows parameter objects to be treated as JAX pytrees, enabling direct use with JAX transformations (`jit`, `grad`, etc.) and clear separation of static (`N`, `K`) vs. differentiable parameters.
*   **Helper Modules:** Encapsulation of complex mathematical implementations or optimization steps into internal helper modules (e.g., `_bellman_impl.py`, `_bellman_optim.py`) to keep primary class definitions cleaner.
*   

## Architectural Patterns


*   **Filter Base Class:** Filtering implementations (`DFSVBellmanFilter`, `DFSVParticleFilter`) inherit from a common base class (`core/filters/base.py`), suggesting a common interface for filters.
*   **Bellman Filter Strategy:** Implements the update step using block coordinate descent, optimizing factors and log-volatilities iteratively.
*   **Particle Filter Strategy:** Implements a Bootstrap Filter (SISR) using `jax.lax.scan` for the main loop, with ESS-triggered Systematic Resampling to mitigate particle degeneracy.
*   

## Testing Patterns

*