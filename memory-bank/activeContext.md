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



## Recent Changes

[2025-04-02 05:59:00] - Successfully configured and launched the simulation study on Google Cloud Batch.
  - Created separate job templates for CPU (BF) and GPU (PF) tasks (`batch_job_bf.template.json`, `batch_job_pf.template.json`).
  - Corrected Dockerfile for GPU support and build issues.
  - Adapted `run_config_batch.py` script for GCS output.
  - Established submission process using `envsubst` and `gcloud batch jobs submit`.
  - Jobs are currently running and saving results to GCS bucket `gs://dsfv-simulation-results-bucket`.


## Current Focus

[2025-04-02 05:59:00] - Monitoring the execution of the Cloud Batch jobs for the simulation study.


## Recent Changes

*   [2025-04-02 19:57:00] - Investigated NaN errors in `examples/bf_optimization.py`. Implemented correct Observed Fisher Information (negative Hessian) in Bellman filter update. NaN propagation persisted, traced to numerical instability in precision matrix update/inversion cycle (`I_updated = I_pred + J`, `P_updated = I_updated^-1`). Stabilization attempts were unsuccessful.

## Current Focus

*   [2025-04-02 19:57:00] - Debugging of current Bellman filter implementation stopped due to persistent numerical instability.
*   [2025-04-02 19:57:00] - Next step: Plan and potentially implement a Bellman Information Filter formulation.


## Recent Changes

*   [2025-04-02 20:26:30] - Completed planning phase for implementing the Bellman Information Filter (BIF) to address numerical instability in the covariance-based filter. Plan saved to `bif_implementation_plan.md`.


*   [2025-04-02 20:35:10] - Reviewed Lange (2024) paper and Boekestijn (2025) thesis proposal. Integrated relevant methodology, model specification, and estimation details into Memory Bank (`productContext.md`, `systemPatterns.md`, `decisionLog.md`).
## Current Focus

*   [2025-04-02 20:35:40] - Implement the Bellman Information Filter (`DFSVBellmanInformationFilter`) according to the specification in `bif_implementation_plan.md`.

## Recent Changes

*   [2025-04-01 01:06:35] - Initialized Memory Bank.
*   [2025-04-01 01:11:31] - Documented initial architectural and coding patterns.
*   [2025-04-01 01:31:08] - Refactored `scripts/simulation_study.py` (Phase 1) for config/output & NumPy 2.0 fix. Tested.
*   [2025-04-01 02:47:00] - **Debugged original Bellman Filter:** Fixed `ConcretizationTypeError`, corrected `Q_f` prediction logic. Identified numerical sensitivity to small `Q_h` (prior dominance); pragmatically increased `Q_h` scaling in simulations as a workaround. Filter performance improved but underlying sensitivity remained.
*   [2025-04-01 03:37:00] - Refactored `DFSVParticleFilter` for external parameter passing & JIT compatibility.
*   [2025-04-01 04:21:00] - Completed Particle Filter refactoring and launched expanded simulation study. Created `simulation_analysis_plan.md`.
*   [2025-04-01 11:28:32] - Added resume capability to `scripts/simulation_study.py`.
*   [2025-04-01 12:05:00] - Optimized original Bellman filter update step using Woodbury Identity and Rank-1 FIM reformulation.
*   [2025-04-01 12:20:00] - Profiled particle filter; implemented minor likelihood/covariance optimizations.
*   [2025-04-02 00:17:00] - **Resolved Memory Leak:** Identified leak associated with `jax.lax.scan` in Bellman filter. Switched simulation study to use Python loop (`filter` method) instead of `filter_scan`.
*   [2025-04-02 01:13:12] - Completed refactoring of `main` function in `scripts/simulation_study.py`.
*   [2025-04-02 05:59:00] - Configured and launched simulation study on Google Cloud Batch.
*   [2025-04-02 19:57:00] - **Stopped Debugging Covariance BF:** Halted efforts to fix NaNs in original Bellman filter due to persistent numerical instability in precision matrix update/inversion. Decided to pursue Bellman Information Filter (BIF).
*   [2025-04-02 20:26:30] - Completed planning phase for BIF implementation (`bif_implementation_plan.md`).
*   [2025-04-02 20:35:10] - Reviewed Lange (2024) paper and Boekestijn (2025) thesis proposal. Integrated relevant methodology, model specification, and estimation details into Memory Bank (`productContext.md`, `systemPatterns.md`, `decisionLog.md`).

## Open Questions/Issues

*   [2025-04-01 01:49:47] - `write_to_file` tool seems unreliable when applying complex changes or when file content might change subtly between operations. (Monitor tool behavior).

---
**Update Log:**


[2025-04-02 05:59:00] - Monitoring the execution of the Cloud Batch jobs for the simulation study.


*   [2025-04-02 19:57:00] - Debugging of current Bellman filter implementation stopped due to persistent numerical instability.
*   [2025-04-02 19:57:00] - Next step: Plan and potentially implement a Bellman Information Filter formulation.


*   [2025-04-02 20:26:30] - Completed planning phase for implementing the Bellman Information Filter (BIF) to address numerical instability in the covariance-based filter. Plan saved to `bif_implementation_plan.md`.


*   [2025-04-02 20:26:30] - Implement the Bellman Information Filter (`DFSVBellmanInformationFilter`) according to the specification in `bif_implementation_plan.md`.

*   [2025-04-02 23:26:45] - Implemented and tested `DFSVBellmanInformationFilter` (BIF) in `src/bellman_filter_dfsv/core/filters/bellman_information.py` and `tests/test_bellman_information.py`.

*   [2025-04-02 23:26:45] - Added covariance/variance getter methods to BIF.

*   [2025-04-02 23:26:45] - BIF implementation complete. Next steps involve using it in optimization examples and simulation study.
*   [2025-04-01 01:06:35] - Initial log entry.
*   [2025-04-01 21:58:18] - Consolidated sections, updated status, and summarized recent activities during Memory Bank cleanup.