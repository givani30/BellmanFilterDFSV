# Progress

This file tracks the project's progress using a task list format.

*

## Completed Tasks

*   [2025-04-01 01:06:39] - Initialized Memory Bank.
*   [2025-04-01 01:11:45] - Analyzed code structure (`src/`, `core/filters/bellman.py`, `models/dfsv.py`, `core/filters/particle.py`).
*   [2025-04-01 01:11:45] - Documented initial system patterns in `systemPatterns.md`.
*   [2025-04-01 01:31:38] - Refactored `scripts/simulation_study.py` (Phase 1) for config/raw output & fixed NumPy 2.0 compatibility. Tested refactoring.
*   [2025-04-01 02:47:00] - Debugged Bellman filter: Fixed low log-volatility correlation (identified prior dominance due to small `Q_h`, increased `Q_h` scaling, corrected `Q_f` prediction), fixed `ConcretizationTypeError`.
*   [2025-04-01 03:37:00] - Refactored `DFSVParticleFilter` to accept parameters externally for JIT compatibility.
*   [2025-04-01 04:21:00] - Completed Particle Filter refactoring and launched expanded simulation study.
*   [2025-04-01 05:13:00] - Optimized particle filter likelihood calculation for diagonal `R_t`.
*   [2025-04-01 05:13:00] - Benchmarked particle filter with `float32` vs `float64`.
*   [2025-04-01 11:28:32] - Implemented resume capability for `scripts/simulation_study.py`.
*   [2025-04-01 12:05:00] - Optimized Bellman filter performance using Woodbury Identity and Rank-1 FIM reformulation. Verified with profiling and tests.
*   [2025-04-01 12:22:00] - Profiled particle filter performance and tested minor optimizations (vectorized likelihood, covariance calculation).
*   [2025-04-01 21:59:14] - Cleaned up and consolidated all Memory Bank files (`productContext.md`, `activeContext.md`, `systemPatterns.md`, `decisionLog.md`, `progress.md`).

## Current Tasks

*   [2025-04-01 21:59:14] - Analyze results from the expanded simulation study (Launched [2025-04-01 04:21:00], Plan: `simulation_analysis_plan.md`).
*   [2025-04-02 00:17:00] - **Completed:** Investigate potential memory leak identified during simulation runs (Plan: `memory_leak_investigation_plan.md`).
    *   [2025-04-01 22:15:39] - Used `memory_profiler` - confirmed leak originates within Bellman filter `jax.lax.scan`.
    *   [2025-04-01 22:43:36] - Tested disabling history accumulation in `lax.scan` - did not resolve leak.
    *   [2025-04-01 22:46:35] - Tested adding `jax.block_until_ready` in `lax.scan` - did not resolve leak.
    *   [2025-04-01 23:58:29] - Used JAX device profiler (`save_device_memory_profile`) and `pprof --diff-base` - confirmed leak is not persistent device memory growth between replicates, but likely host RAM accumulation/fragmentation during `lax.scan` execution.
    *   [2025-04-02 00:14:24] - Confirmed switching Bellman filter from `filter_scan` to `filter` (Python loop) resolves the observed RAM growth.
    *   [2025-04-02 00:17:00] - Implemented fix by changing `simulation_study.py` to use `bf_instance.filter()`.

## Next Steps

*   (Addressed) Address memory leak if confirmed by investigation.
*   Complete simulation analysis according to the plan.
*   Consider implementing hyperparameter estimation framework.
*   Potentially revisit Bellman filter `Q_h` sensitivity if required by real data characteristics.

---
**Update Log:**

*   [2025-04-01 01:06:39] - Initial log entry.
*   [2025-04-01 21:59:14] - Merged duplicate sections, updated task statuses based on cross-file review, and removed simple log entries during Memory Bank cleanup.