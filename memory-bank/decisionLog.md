# Decision Log

This file records architectural and implementation decisions using a structured format.

*

---

**Decision [2025-04-01 01:32:03]:** Pause expansion of simulation study (`scripts/simulation_study.py`) and shift focus to investigating/fixing potential issues in the Bellman filter implementation.

**Rationale:** User identified potential problems with the Bellman filter that need addressing before proceeding with large-scale simulations.

**Implementation Details:** Next steps involve clarifying the specific Bellman filter issues and debugging the relevant code (`core/filters/bellman.py`, `_bellman_impl.py`, `_bellman_optim.py`). Debugging approach involved creating a dedicated script (`debug_bellman_filter.py`), identifying JAX tracer errors (`TracerArrayConversionError`, `ConcretizationTypeError`) related to NumPy conversions within JITted functions (`filter_scan` body), and fixing these errors in `bellman.py`.

---

**Decision [2025-04-01 02:27:00]:** Correct `Q_f` calculation in `src/bellman_filter_dfsv/core/filters/bellman.py::_predict_jax`.

**Rationale:** The original code used `h_{t-1|t-1}` instead of the predicted `h_{t|t-1}` to calculate the factor process noise `Q_f`. This was incorrect according to the model dynamics.

**Implementation Details:** Modified the `_predict_jax` function to use the predicted log-volatility `h_pred` when calculating `Q_f_pred`.

---

**Decision [2025-04-01 02:47:00]:** Accept Bellman filter sensitivity to small `Q_h` and proceed using a larger `Q_h` scaling factor (1.0) in `scripts/simulation_study.py::create_sim_parameters`.

**Rationale:** Extensive debugging confirmed numerical dominance of the prior term when `Q_h` is small, hindering `h` estimation. While the root cause of potential past success with small `Q_h` remains elusive, increasing `Q_h` provides the most stable performance for the current implementation. This is necessary for proceeding, but the sensitivity should be noted for future work. Initial attempts to improve convergence included changing the h-update optimizer (BFGS -> NonlinearCG -> BFGS) and increasing `inner_max_steps` (15 -> 100), but increasing `Q_h` scaling (0.1 -> 1.0) proved most effective.

**Implementation Details:** Updated the `create_sim_parameters` function in `scripts/simulation_study.py` to use `q_h_scale=1.0`. Reverted h-update optimizer to BFGS with `inner_max_steps=100` in `bellman.py`.

---

**Decision [2025-04-01 03:37:00]:** Refactor `DFSVParticleFilter` class to accept model parameters externally via method arguments instead of storing them internally.

**Rationale:** To enable the reuse of a single `DFSVParticleFilter` instance across multiple simulation replicates with varying parameters and to maximize JAX JIT compilation benefits for the particle filtering logic, improving performance in `scripts/simulation_study.py`.

**Implementation Details:** Modified `__init__`, `filter`, `scan_body`, `initialize_particles`, `predict_particles`, `compute_log_likelihood_particle` methods. Updated JIT decorators. Detailed plan saved in `refactoring_pf_jit_plan.md`.

---

**Decision [2025-04-01 05:13:00]:** Optimize particle filter likelihood calculation (`compute_log_likelihood_particle`) to exploit known diagonal structure of observation noise covariance `R_t`.

**Rationale:** The original implementation performed redundant Cholesky decompositions and log-determinant calculations inside a `vmap` over particles. Since `R_t` is diagonal and constant for all particles at time `t`, pre-calculating the log-determinant and using direct variance division for the quadratic form significantly reduces computation.

**Implementation Details:** Modified `compute_log_likelihood_particle` in `particle.py`. Achieved ~5.5x speedup in benchmark (`float64`).

---

**Decision [2025-04-01 05:13:00]:** Test particle filter implementation and benchmark using `float32` precision.

**Rationale:** Explore potential performance gains from lower precision, especially relevant for GPU execution.

**Implementation Details:** Ran benchmark script `scripts/benchmark_particle_filter.py` with `float32`. Achieved further ~1.34x speedup (total ~7.4x vs baseline). However, the final log-likelihood value changed significantly, indicating potential numerical precision issues. Removed global `jax.config.update("jax_enable_x64", True)` from `particle.py` to allow external precision control.

---

**Decision [2025-04-01 11:28:32]:** Implement resume capability for `scripts/simulation_study.py`.

**Rationale:** Long-running simulations are prone to interruption. A resume mechanism prevents loss of progress and avoids redundant computation.

**Implementation Details:** Use a command-line argument (`--resume_study <study_dir_name>`) to specify the target study. Within the study directory, use a `checkpoint.json` file to store the indices of the *next* replicate to run. Before starting a replicate, check if `metrics.json` exists (skip if yes), then update `checkpoint.json` with the *next* replicate's indices. Load configuration from `simulation_config.json` within the specified study directory when resuming. Delete `checkpoint.json` on successful completion. Plan documented in `simulation_resume_plan.md`.

---

**Decision [2025-04-01 12:05:00]:** Optimize Bellman filter update step (`_bellman_impl.py`) using Woodbury Identity and Rank-1 FIM Reformulation.

**Rationale:** Profiling identified O(N^3) Cholesky decomposition in `log_posterior_impl` and O(N^3) + O(K^2*N^2) operations in `fisher_information_impl` as major bottlenecks. Applying Woodbury Identity avoids the N^3 inversion/decomposition. Rank-1 reformulation avoids the expensive O(K^2*N^2) `einsum` in FIM calculation.

**Implementation Details:**
1.  Applied Woodbury Matrix Identity and Matrix Determinant Lemma in `log_posterior_impl` (Decision [2025-04-01 11:46:00]).
2.  Applied Woodbury Identity for `A_inv` applications and Rank-1 reformulation (`I_hh[k, l] = 0.5 * exp(h_k + h_l) * (I_ff[k, l])^2`) in `fisher_information_impl` (Decision [2025-04-01 12:05:00]).
Reduced complexity significantly, achieving large speedups verified by profiling and tests (`test_bellman_unified.py`).

---

**Decision [2025-04-01 12:22:00]:** Implement minor optimizations for Particle Filter (`particle.py`).

**Rationale:** Explore potential minor performance gains based on profiling insights and user feedback.

**Implementation Details:**
1.  Implemented vectorized likelihood calculation in `compute_log_likelihood_particle` (replacing `vmap` with broadcasting). Observed minor improvement.
2.  Replaced `einsum` with matrix multiplication (`@`) for weighted covariance calculation in scan loop. Observed no significant change. Tests passed for both.

---
**Update Log:**

*   [2025-04-01 01:06:44] - Initial log entry.
*   [2025-04-01 21:58:53] - Restructured file, merged duplicate sections, consolidated related decisions (e.g., Bellman filter debugging/optimization, `Q_h` changes), and enforced Decision/Rationale/Implementation format during Memory Bank cleanup.

---
**Decision [2025-04-02 00:17:00]:** Switch Bellman filter implementation in `scripts/simulation_study.py` from `filter_scan` (using `jax.lax.scan`) to `filter` (using Python loop).

**Rationale:** Debugging revealed a significant memory leak (RAM growth over replicates) when using `filter_scan`. Profiling (`memory_profiler`, JAX device profiler via `pprof`) indicated memory accumulation during the `lax.scan` execution, likely due to JAX/XLA retaining intermediate arrays from the complex update step across iterations. Explicit GC and `jax.remat` did not resolve the issue. Switching to the Python-loop based `filter` method was confirmed to prevent this memory leak, although it might be computationally slower. Modified `scripts/simulation_study.py` (line ~175) to call `bf_instance.filter(params, returns)`.

---

**Decision [2025-04-02 01:50:00]:** Use Google Cloud Batch for running the simulation study.

**Rationale:**
*   The study involves a large number (13,500) of independent replicates, making it suitable for parallel execution.
*   Cloud Batch allows for efficient scaling and management of compute resources.
*   Cost analysis suggests Batch is cost-effective (~$10-$20 compute estimate) compared to sequential execution, especially when amortizing JAX JIT compilation.
*   Allows using different hardware (CPU for BF, GPU for PF) for different task types, optimizing performance and cost.
*   **Implementation:** A new script `scripts/run_config_batch.py` was created. Each Batch task will execute this script for one full configuration (N, K, Filter, Particles), running all 100 replicates within that task to amortize JIT compilation. The application will be containerized using Docker and pushed to Google Artifact Registry. Input configuration and output results will be stored in Google Cloud Storage (GCS). The script `run_config_batch.py` needs modification to handle GCS paths. A Batch job definition file (JSON/YAML) will be created to define tasks, hardware, container image, and arguments.

---

**Decision [2025-04-02 19:13:00]:** Refine debugging strategy for NaN errors in Bellman filter update step based on detailed feedback.

**Rationale:** Tests failed with NaNs after implementing the Observed FIM (`J_observed`). Debug prints indicated NaN propagation starting from `I_pred`, likely due to unstable inversion of `I_updated = I_pred + J_observed` in the previous step. User feedback provided a comprehensive list of potential numerical instability sources within the Woodbury/Hessian calculations.
**Implementation Details:** The refined plan focuses on adding checks within `observed_fim_impl` and `__update_jax`:
1. Check inputs (`sigma2`, `h`) for extreme values that could cause division issues or `exp(h)` blowup.
2. Check the condition number of the intermediate matrix `M` used in Woodbury.
3. Check the magnitude of intermediate vector `P`.
4. Add `isnan`/`isinf` checks after key calculations.
5. Consider clamping extreme values or increasing/adapting jitter if necessary.
6. Postpone exploring a full Information Filter formulation (propagating `I` instead of `P`) until current stabilization attempts are exhausted.

---

**Decision [2025-04-02 19:57:00]:** Stop debugging the current covariance-based Bellman filter implementation (`bellman.py`, `_bellman_impl.py`) for the NaN issue observed in `bf_optimization.py`.

**Rationale:** Despite implementing the mathematically correct Observed Fisher Information (negative Hessian), persistent NaN propagation indicates deep numerical instability in the precision matrix update (`I_updated = I_pred + J`) and inversion (`updated_cov = I_updated^-1`) cycle. Further stabilization within the current framework seems unlikely to succeed robustly.

**Next Step:** Explore the implementation of a Bellman Information Filter, which directly propagates precision matrices and might be numerically more stable for this problem.

---

**Decision [2025-04-02 20:26:00]:** Proceed with implementing a Bellman Information Filter (BIF) as specified in `bif_implementation_plan.md`.

**Rationale:** The existing covariance-based Bellman filter (`DFSVBellmanFilter`) exhibits numerical instability (NaN propagation) during optimization runs, particularly in the precision matrix update/inversion cycle. The BIF formulation, which directly propagates information matrices, is expected to be more numerically robust. The BIF will be implemented in `src/bellman_filter_dfsv/core/filters/bellman_information.py`, inheriting from `DFSVFilter` and reusing components like `observed_fim_impl`, `log_posterior_impl`, and optimization helpers where possible. It will propagate the information state (`alpha`, `Omega`) and store filtered information matrices.

---

**Decision [2025-04-02 20:35:00]:** Formally adopt the Bellman filter methodology described in Lange (2024) as the core approach for state estimation and hyperparameter estimation for the specific DFSV model defined in the thesis proposal (Boekestijn, 2025).

**Rationale:** This project aims to implement and evaluate the Bellman filter for a specific DFSV model. Explicitly referencing both the source methodology (Lange, 2024) and the target model specification (Boekestijn, 2025) provides clear context and aligns the project's implementation with its stated goals. Memory Bank files (`productContext.md`, `systemPatterns.md`) were updated to reflect the specific model equations and estimation approach based on both documents. Future development should adhere to this chosen methodology.

---

**Decision [2025-04-02 20:26:00]:** Implement a specific KL-type penalty term for the BIF based on Eq. (40) from Lange (2024) paper image provided, instead of reusing the standard KL divergence function (`kl_penalty_impl`).



---

**Decision [2025-04-02 23:26:45]:** Use Joseph form / Woodbury identity for BIF prediction step (`__predict_jax_info`).

**Rationale:** Standard KF information prediction requires inverting the posterior information matrix, which can be unstable. The Joseph form avoids this inversion, improving numerical stability, although it requires inverting the process noise `Q_t` and an intermediate matrix `M`.

**Implementation Details:** Implemented in `src/bellman_filter_dfsv/core/filters/bellman_information.py::__predict_jax_info` using stable Cholesky-based inversions with `pinv` fallback for `Q_h` and `M`.


---

**Decision [2025-04-02 23:26:45]:** Remove `try/except` blocks around JAX operations (`vmap`, `_invert_info_matrix`) in BIF getter methods.

**Rationale:** Catching generic exceptions around JAX operations can mask underlying errors and is generally discouraged. Stable inversion logic is handled within `_invert_info_matrix`. Let JAX handle propagation of errors from `vmap` or inversion failures.

**Implementation Details:** Removed `try/except` blocks from `get_predicted_covariances` and `_invert_info_matrix` in `src/bellman_filter_dfsv/core/filters/bellman_information.py`.


---

**Decision [2025-04-02 23:26:45]:** Modify BIF comparison test (`test_bif_vs_bf_comparison`) to exclude log-likelihood comparison.

**Rationale:** The BIF uses a specific pseudo log-likelihood (Lange Eq. 40) involving a KL-type penalty, which differs from the likelihood calculation in the original BF implementation. Comparing them directly is invalid. The test should focus on comparing state estimates under stable conditions.

**Implementation Details:** Removed log-likelihood assertion from `test_bif_vs_bf_comparison` in `tests/test_bellman_information.py`.


---

**Decision [2025-04-02 23:26:45]:** Correct argument passing and status checking in BIF stability test (`test_bif_stability_during_optimization`).

**Rationale:** Initial test versions had `TypeError` due to incorrect keyword arguments (`prior_mu_mean` vs `prior_mean`, `returns` vs `y`) passed to the objective function via `partial`/`optimistix`, and `AttributeError` accessing `result.status` instead of `result.result`. Also, the expected success status needed to include `nonlinear_max_steps_reached` for short test runs.

**Implementation Details:** Switched from `partial` to a wrapper function for the objective. Corrected keyword arguments. Changed status check to use `result.result` and included `optx.RESULTS.nonlinear_max_steps_reached` in the assertion in `tests/test_bellman_information.py`.
**Rationale:** Eq. (40) represents the specific penalty used in the augmented likelihood for parameter estimation within the Bellman filter context. While it uses information matrices as input, its formula differs from the standard KL divergence (lacking trace and dimension terms). Using the correct formula is crucial for consistency with the paper's methodology. A new function, `_kl_penalty_pseudo_lik_impl`, will be created to compute `0.5 * (log_det(Omega_post) - log_det(Omega_pred) + (a_updated - a_pred).T @ Omega_pred @ (a_updated - a_pred))`. This will be called within the BIF's update step.
