# Active Context

This file tracks the project's current status, including recent changes, current goals, and open questions.

*

## Current Focus

*   [2025-04-01 21:58:18] - Analyze results from the expanded simulation study (launched [2025-04-01 04:21:00]).
*   [2025-04-01 21:58:18] - Investigate potential memory leak identified during simulation runs (see `memory_leak_investigation_plan.md`).

## Recent Changes

*   [2025-04-01 01:06:35] - Initialized Memory Bank.
*   [2025-04-01 01:11:31] - Documented initial architectural and coding patterns.
*   [2025-04-01 01:31:08] - Refactored `scripts/simulation_study.py` (Phase 1) for configuration flexibility and raw data output. Fixed NumPy 2.0 compatibility issue. Tested refactoring.
*   [2025-04-01 01:49:47] - Investigated Bellman filter issues: Created debug script (`debug_bellman_filter.py`), confirmed low log-volatility correlation, identified and fixed `jax.errors.ConcretizationTypeError` in `bellman.py`. Modified `_bellman_optim.py` to return success status.
*   [2025-04-01 02:47:00] - Completed in-depth Bellman filter debugging: Identified root cause of low log-vol correlation as numerical dominance of the prior term due to small process noise `Q_h` used in simulation setup. Corrected `Q_f` calculation in `_predict_jax` (`bellman.py`). Pragmatic fix applied by increasing `Q_h` scaling factor in `create_sim_parameters` (`simulation_study.py`) from 0.1 to 1.0. Filter performance significantly improved. Concluded debugging.
*   [2025-04-01 03:37:00] - Refactored `DFSVParticleFilter` class to accept parameters externally, enabling JIT compilation benefits for simulation study.
*   [2025-04-01 04:21:00] - Completed refactoring of `DFSVParticleFilter` and `scripts/simulation_study.py`. Verified with minimal test and launched expanded simulation study. Created `simulation_analysis_plan.md`.
*   [2025-04-01 11:28:32] - Added resume capability to `scripts/simulation_study.py` using command-line argument (`--resume_study`) and checkpointing (`checkpoint.json`). Documented plan in `simulation_resume_plan.md`.
*   [2025-04-01 12:05:00] - Optimized Bellman filter performance: Implemented Woodbury identity in `log_posterior_impl` and rank-1 reformulation + Woodbury in `fisher_information_impl` (`_bellman_impl.py`), removing O(N^3) and O(K^2*N^2) bottlenecks. Verified with profiling and tests.
*   [2025-04-01 12:20:00] - Profiled particle filter (`scripts/profile_particle_filter.py`). Confirmed scaling consistent with O(T*(NKP + PK^2)). Implemented and tested minor optimizations (vectorized likelihood, replaced covariance `einsum`), observing minimal impact.
*   [2025-04-01 21:58:18] - Consolidated and cleaned up Memory Bank files.
*   [2025-04-02 01:13:12] - Completed refactoring of `main` function in `scripts/simulation_study.py` using helper functions.

## Open Questions/Issues

*   [2025-04-01 01:49:47] - `write_to_file` tool seems unreliable when applying complex changes or when file content might change subtly between operations. (Monitor tool behavior).
*   [2025-04-01 21:58:18] - Potential memory leak during long simulation runs needs investigation.

---
**Update Log:**

*   [2025-04-01 01:06:35] - Initial log entry.
*   [2025-04-01 21:58:18] - Consolidated sections, updated status, and summarized recent activities during Memory Bank cleanup.