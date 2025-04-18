# Active Context

This file tracks the project's current status, including recent changes, current goals, and open questions. Historical milestones are summarized in `progress.md`.

*


*   [14-04-2025 02:30:00] - BIF gradient NaN/Inf debugging task completed. Ready for next task.
## Current Focus

*   [11-04-2025 18:15:00] - Enhanced parameter logging in optimization process with improved `minimize_with_logging` function and better performance options.

*   [11-04-2025 17:30:00] - Aligned `mu` fixing logic in `optimization.py` with established strategy to always fix `mu` for BIF when true parameters are available.

*   [11-04-2025 16:23:11] - Vectorized `apply_identification_constraint` in `utils/transformations.py` and fixed related test failures.

*   [07-04-2025 23:28:15] - Debugging BIF (`bellman_information.py`) prediction/update steps to resolve inaccurate predicted covariances/information matrices. This is blocking the BIF EM implementation plan.

*   [04-06-2025 17:43:00] - Mu investigation complete. Initial strategy (for BIF): Fix `mu`. Focus shifts to simulation analysis & real data application using this strategy. This strategy has since been revised (see Decision [16-04-2025 02:52:23]).
*   [04-06-2025 17:10:00] - Completed Phase 2.2: Fixed *both* `mu` elements to `[-1.0, -1.0]`. Successful convergence, reasonable estimates for other params.
*   [NEW TASK] Test Particle Filter (PF) and base Bellman Filter (BF, covariance-based) performance in hyperparameter optimization (potentially fixing `mu` for BF as well, TBD).
*   [04-06-2025 03:45:00] - Test framework unification complete and all tests pass. Project is stable and awaiting the next task.

## Recent Changes

*   **[16-04-2025 02:53:19]** Updated Memory Bank documentation regarding `mu` fixing strategy:
    - Added new decision log entry documenting that fixing `mu` is no longer the default strategy.
    - Modified historical entry [04-06-2025 17:43:00] to clarify it was an initial BIF-specific strategy.
    - Reflects findings from recent hyperparameter studies showing both fixed and unfixed `mu` approaches are viable.


*   **[15-04-2025 17:28:00]** Completed overhaul of `scripts/analysis/analyze_optimization_results.py` as per plan `memory-bank/plans/results_analysis_refinement_plan_v8_15-04-2025.md`.
    - Key outputs generated and saved to `outputs/`: Summary tables (CSV), `analysis_full_data_<timestamp>.csv`, `analysis_agg_scalars_<timestamp>.csv`, `analysis_agg_param_errors_<timestamp>.csv`.

*   **[14-04-2025 02:30:00]** Debugged NaN/Inf error occurring during BIF gradient calculation (`value_and_grad` on likelihood).
    - Initial error pointed to linear solver failure during gradient computation.
    - Forward pass of likelihood calculation succeeded.
    - Error persisted despite bypassing internal BFGS optimizer (`update_h_bfgs`).
    - Final error trace (with update step non-JITted) pointed to `FloatingPointError` during differentiation of `jnp.linalg.eigh` applied to the Observed Fisher Information (OFIM) matrix (`J_observed`) in `__update_jax_info`.
    - **Resolution:** User implemented the Expected Fisher Information (EFIM) in `src/bellman_filter_dfsv/filters/_bellman_impl.py` instead of OFIM, which resolved the gradient instability.
- Removed temporary debugging checks (`eqx.error_if`) added during the session.

*   [11-04-2025 18:15:00] - Enhanced parameter logging in optimization process with improved `minimize_with_logging` function, added `BestSoFarMinimiser` wrapper, and optimized performance by using built-in Optimistix minimizer when detailed logging is not needed.

*   [11-04-2025 17:30:00] - Aligned `mu` fixing logic in `optimization.py` with established strategy to always fix `mu` for BIF when true parameters are available, regardless of transformation status.

*   [11-04-2025 16:23:11] - Vectorized `apply_identification_constraint` in `utils/transformations.py` (replacing Python loop with `jnp.tril`).
*   [11-04-2025 16:23:11] - Fixed multiple test failures in `tests/test_optimization.py` (imports, missing args, assertions, types, JIT tracing errors).

*   [09-04-2025 03:18:00] - Refactored optimization utilities: Centralized optimizer creation in `solvers.py`, standardized orchestration in `optimization.py`.
*   [09-04-2025 03:18:00] - Created `scripts/unified_filter_optimization.py` for comparative experiments.

*   [07-04-2025 23:28:15] - Attempted BIF EM Phase 1: Modified RTS smoother (`base.py`) for lag-1 covs, but smoother test (`test_smooth_state_accuracy` in `test_bellman_information.py`) failed due to inaccurate BIF predicted covariances.

*   [07-04-2025 02:05:00] - Completed implementation and testing of full persistence matrices (`Phi_f`, `Phi_h`) using element-wise `softplus` transformation and stability penalty in objective function. (See Decision [07-04-2025 02:05:00]).

*   [04-06-2025 00:01:00] - Completed JIT refactoring of filter implementations (`bellman_information.py`, `_bellman_impl.py`, `bellman.py`, `particle.py`) to remove Python control flow from JIT paths, improving performance and robustness. Tests pass (41/42). (See Decision [04-06-2025 00:01:00]).
*   [04-06-2025 14:20:00] - Completed execution of Bellman Information Filter (BIF) simulation runs as per thesis plan.

*   [04-06-2025 15:41:00] - Completed Phase 1 diagnostics (`mu` identifiability): Confirmed BIF penalty term causes significant upward gradient bias for `mu`. Dynamic analysis revealed mechanism (interaction of `mu`-biased state prediction difference `diff` with `Omega_pred`). Penalty ablation and strong prior tests showed bias is hard to overcome via direct optimization.
*   [04-06-2025 03:45:00] - Completed test framework unification: Standardized on `pytest`, created common fixtures (`tests/conftest.py`), implemented unified filter tests (`tests/test_unified_filters.py`), refactored existing tests (`test_bellman.py`, `test_particle_filter.py`, `test_transformations.py`), added constraint checks to transformation tests, and debugged resulting failures. All 49 tests pass.
*   [04-06-2025 16:55:00] - Completed Phase 2.3 variant: Fixed `mu[0]=-1.0`, estimated `mu[1]`. Successful convergence.

*   [11-04-2025 17:30:00] - `mu` fixing logic in `optimization.py` has been aligned with established strategy (Decision [04-06-2025 17:40:11]) and `unified_filter_optimization.py` implementation.


- (Resolved) [14-04-2025 02:30:00] - BIF gradient calculation produced NaN/Inf. Resolved by switching from OFIM to EFIM.
## Open Questions/Issues

*   [07-04-2025 23:28:15] - BIF filter (`bellman_information.py`) generates inaccurate predicted covariance/information matrices, causing downstream smoother tests (`test_smooth_state_accuracy`) to fail. Root cause investigation needed. BIF EM plan paused.

*   [07-04-2025 15:55:00] - Parameter recovery check (Full Phi, Fixed `mu`) revealed poor recovery for `Q_h`. This needs further investigation or consideration for model application reliability. (See Decision [07-04-2025 15:55:00])

## Recent Changes [11-04-2025 18:14:19]

* Completed optimization stability improvements task
* Major modifications to objective function calculation for early stability detection
* Enhanced BFGS update fallback behavior
* Implemented optimizer clip norms and finite checks
* Performance profiling confirms significant speedup for infeasible parameter combinations

## Current Focus

* Monitoring production runs with new stability improvements
* Collecting performance metrics on convergence speed
* Documenting optimization patterns for future reference
*   [04-06-2025 01:24:00] - Completed filter API alignment task: Refactored `DFSVFilter` base class and subclasses (`DFSVBellmanFilter`, `DFSVBellmanInformationFilter`, `DFSVParticleFilter`) for API consistency. Renamed likelihood methods, added abstract methods (`log_likelihood_wrt_params`, `jit_log_likelihood_wrt_params`), implemented `smooth` consistently, added/adjusted public `predict`/`update` methods. Updated tests and usage examples. All tests pass.
