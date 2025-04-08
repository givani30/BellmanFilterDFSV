# Active Context

This file tracks the project's current status, including recent changes, current goals, and open questions. Historical milestones are summarized in `progress.md`.

*

## Current Focus


*   [07-04-2025 23:28:15] - Debugging BIF (`bellman_information.py`) prediction/update steps to resolve inaccurate predicted covariances/information matrices. This is blocking the BIF EM implementation plan.

*   [04-06-2025 17:43:00] - Mu investigation complete. Final strategy: Fix `mu`. Focus shifts to simulation analysis & real data application using this strategy.
*   [04-06-2025 17:10:00] - Completed Phase 2.2: Fixed *both* `mu` elements to `[-1.0, -1.0]`. Successful convergence, reasonable estimates for other params.
*   [NEW TASK] Test Particle Filter (PF) and base Bellman Filter (BF, covariance-based) performance in hyperparameter optimization (potentially fixing `mu` for BF as well, TBD).
*   [04-06-2025 03:45:00] - Test framework unification complete and all tests pass. Project is stable and awaiting the next task.

## Recent Changes


*   [07-04-2025 23:28:15] - Attempted BIF EM Phase 1: Modified RTS smoother (`base.py`) for lag-1 covs, but smoother test (`test_smooth_state_accuracy` in `test_bellman_information.py`) failed due to inaccurate BIF predicted covariances.

*   [07-04-2025 02:05:00] - Completed implementation and testing of full persistence matrices (`Phi_f`, `Phi_h`) using element-wise `softplus` transformation and stability penalty in objective function. (See Decision [07-04-2025 02:05:00]).

*   [04-06-2025 00:01:00] - Completed JIT refactoring of filter implementations (`bellman_information.py`, `_bellman_impl.py`, `bellman.py`, `particle.py`) to remove Python control flow from JIT paths, improving performance and robustness. Tests pass (41/42). (See Decision [04-06-2025 00:01:00]).
*   [04-06-2025 14:20:00] - Completed execution of Bellman Information Filter (BIF) simulation runs as per thesis plan.

*   [04-06-2025 15:41:00] - Completed Phase 1 diagnostics (`mu` identifiability): Confirmed BIF penalty term causes significant upward gradient bias for `mu`. Dynamic analysis revealed mechanism (interaction of `mu`-biased state prediction difference `diff` with `Omega_pred`). Penalty ablation and strong prior tests showed bias is hard to overcome via direct optimization.
*   [04-06-2025 03:45:00] - Completed test framework unification: Standardized on `pytest`, created common fixtures (`tests/conftest.py`), implemented unified filter tests (`tests/test_unified_filters.py`), refactored existing tests (`test_bellman.py`, `test_particle_filter.py`, `test_transformations.py`), added constraint checks to transformation tests, and debugged resulting failures. All 49 tests pass.
*   [04-06-2025 16:55:00] - Completed Phase 2.3 variant: Fixed `mu[0]=-1.0`, estimated `mu[1]`. Successful convergence.

## Open Questions/Issues


*   [07-04-2025 23:28:15] - BIF filter (`bellman_information.py`) generates inaccurate predicted covariance/information matrices, causing downstream smoother tests (`test_smooth_state_accuracy`) to fail. Root cause investigation needed. BIF EM plan paused.

*   [07-04-2025 15:55:00] - Parameter recovery check (Full Phi, Fixed `mu`) revealed poor recovery for `Q_h`. This needs further investigation or consideration for model application reliability. (See Decision [07-04-2025 15:55:00])
*   [04-06-2025 01:24:00] - Completed filter API alignment task: Refactored `DFSVFilter` base class and subclasses (`DFSVBellmanFilter`, `DFSVBellmanInformationFilter`, `DFSVParticleFilter`) for API consistency. Renamed likelihood methods, added abstract methods (`log_likelihood_wrt_params`, `jit_log_likelihood_wrt_params`), implemented `smooth` consistently, added/adjusted public `predict`/`update` methods. Updated tests and usage examples. All tests pass.