# Execution Plan: EM Algorithm for Bellman Information Filter (BIF) - Revised

**Date:** 07-04-2025
**Objective:** Implement an Expectation-Maximization (EM) algorithm to estimate parameters **Θ = {lambda_r, Phi_f, Phi_h, sigma2, Q_h, mu}** of the Dynamic Factor Stochastic Volatility (DFSV) model, using the Bellman Information Filter (BIF) smoother for the E-step.

**Context & Constraints:**

* **Model:** DFSV with state `α_t = [f_t', h_t']'`.
* **Filter:** BIF (`src/bellman_filter_dfsv/filters/bellman_information.py`).
* **Smoother:** Base RTS smoother (`src/bellman_filter_dfsv/filters/base.py`) needs modification.
* **Parameters to Estimate:** `lambda_r`, `Phi_f`, `Phi_h`, `sigma2`, `Q_h`, `mu`.
* **Parameter Constraints:**
  * `lambda_r`: Lower-triangular with diagonal elements equal to 1. Apply using `apply_identification_constraint`.
  * `Q_h`: Diagonal and positive. Enforce in M-step.
  * `sigma2`: Diagonal. Enforce in M-step.
  * `Phi_f`, `Phi_h`: Consider stability projection post-update if needed.
* **Approximation:** Use `E[exp(-h_{k,t})] ≈ exp(-E[h_{k,t}] + 0.5 * Var(h_{k,t}))` (Method 2 from outline) for the `Phi_f` update calculation.
* **Codebase:** JAX/Equinox. Adhere to PEP 8, functional style, Google docstrings (`.clinerules`). Use `@equinox.filter_jit`.
* **Workflow:** **All core numerical computations (filtering, smoothing, E-step, M-step) must be implemented using JAX functions (`jax.numpy`, `jax.lax`, etc.) without intermediate conversions to NumPy.** NumPy may be used for loading initial data or final visualization/analysis outside the JAX-compiled functions.
* **File Locations:**
  * New EM logic: `src/bellman_filter_dfsv/estimation/em.py`
  * Smoother modification: `src/bellman_filter_dfsv/filters/base.py`
  * Potential BIF wrapper adjustment: `src/bellman_filter_dfsv/filters/bellman_information.py`
* **Testing:** Use `pytest` with simulated data (`tests/`). Specific tests for smoother output added.
* **Output:** Save plan to `memory-bank/plans/bif_em_implementation_plan_07-04-2025.md`. Test outputs to `outputs/`.

---

## Current Status & Blockers (as of 08-04-2025)

* **Phase 1 (Smoother Mod & Test):**
  * Base smoother (`base.py`) modified to compute/return lag-1 covariances.
  * Accuracy test (`test_smooth_state_accuracy`) created.
  * Test **FAILED**: Smoothed RMSE (1.4521) > threshold (0.5).
  * **Root Cause:** BIF filter (`bellman_information.py`) produces inaccurate predicted covariances, corrupting smoother inputs.
* **Plan Status:** **PAUSED**. Cannot proceed to EM implementation (Phases 2-6) until BIF filter accuracy is resolved.
* **Next Step:** Debug BIF filter prediction/update logic (`bellman_information.py`).

---
---

## Execution Phases & Steps

**Phase 1: Base Smoother Modification & Testing**

* **Goal:** Ensure the base RTS smoother calculates and stores the smoothed lag-1 covariance `P_{t,t-1|T}` using pure JAX operations, and verify its output.
* **File:** `src/bellman_filter_dfsv/filters/base.py`
* **Action (Implementation):**
    1. Modify the `smooth` method (or its internal JAX-compatible loop function like `_rts_smoother_step`) within the `DFSVFilter` base class.
    2. Inside the backward pass (likely a `jax.lax.scan`), calculate `P_{t,t-1|T}` using the standard RTS formula implemented with JAX functions. Ensure correct indexing.
    3. Store the computed `P_{t,t-1|T}` matrices (for t = 1 to T) in a new attribute within the JAX-compatible state/results pytree (e.g., `smoothed_lag1_covs`).
    4. Update the pytree definition for the smoother state/results to include this new attribute.
* **Action (Testing):**
    1. **File:** `tests/test_unified_filters.py` (or a dedicated new test file for smoothing).
    2. **Numerical Test:** Implement a test case using a small, known state-space model where `P_{t,t-1|T}` can be calculated analytically or compared against a trusted implementation (e.g., from another library, calculated beforehand). Verify that the JAX implementation produces matching results within numerical tolerance.
    3. **Visual Inspection Test (Optional but Recommended):** Create a script (e.g., in `scripts/` or `notebooks/`) that runs the filter and smoother on simulated data, extracts the smoothed states (`a_{t|T}`), covariances (`P_{t|T}`), and lag-1 covariances (`P_{t,t-1|T}`). Plot these quantities (converting to NumPy *only* for plotting). Visually inspect the plots for reasonableness (e.g., smooth trajectories, positive definite covariances, sensible cross-correlations). This helps catch potential implementation errors not caught by simple numerical checks.

**Phase 2: M-Step Derivation (Theoretical)**

* **Goal:** Derive the analytical update equations for **Θ = {lambda_r, Phi_f, Phi_h, sigma2, Q_h, mu}** by maximizing the expected complete data log-likelihood `Q(Θ | Θ^(k))`.
* **Action:**
    1. Write down the complete data log-likelihood `L_c(Θ)`, noting the volatility transition term now involves `mu`: `-1/2 * sum[ log|Q_h| + (h_t - mu - Phi_h(h_{t-1}-mu))' Q_h^{-1} (h_t - mu - Phi_h(h_{t-1}-mu)) ]`.
    2. Substitute latent variables with their conditional expectations `E[.] = E[. | Y, Θ^(k)]`.
    3. Isolate terms dependent on each parameter in Θ.
    4. For each parameter (`lambda_r`, `Phi_f`, `Phi_h`, `sigma2`, `Q_h`, **and `mu`**):
        * Take `∂Q / ∂(parameter)`.
        * Set to zero.
        * Solve for `parameter^(k+1)` in terms of the E-step sufficient statistics. Pay close attention to the derivation for `mu`, `Phi_h`, and `Q_h` due to their interdependence in the likelihood term. The update for `mu` will likely depend on `E[h_t]`, `E[h_{t-1}]`, `Phi_h^(k+1)`, and `Q_h^(k+1)`.
    5. **Documentation:** Document the derived update equations clearly (including the one for `mu`) in the `em.py` docstrings or a separate markdown file.

**Phase 3: E-Step Implementation**

* **Goal:** Implement the E-step function to compute expected sufficient statistics using the BIF smoother and pure JAX operations.
* **File:** `src/bellman_filter_dfsv/estimation/em.py`
* **Action:**
    1. Create the file `src/bellman_filter_dfsv/estimation/em.py`.
    2. Define the `e_step` function: `def e_step(params_k: DFSVParamsDataclass, observations: jnp.ndarray, filter_instance: DFSVBellmanInformationFilter) -> Dict:`.
    3. Inside `e_step`:
        * Instantiate or use the provided `filter_instance`.
        * Run the BIF filter: `filtered_results = filter_instance.filter_scan(params_k, observations)`.
        * Run the BIF smoother: `smoothed_results = filter_instance.smooth(params_k, filtered_results)`. *Dependency: Phase 1 must be complete and tested.*
        * Extract smoothed quantities (`a_smooth`, `P_smooth`, `P_lag1_smooth`) from the `smoothed_results` pytree using JAX operations.
        * Implement JAX helper functions for block extraction if needed.
        * Calculate all conditional expectations (`E[f_t]`, `E[h_t]`, `E[f_t f_t']`, etc.) using JAX functions and the formulas involving smoothed quantities.
        * Calculate `E_exp_neg_h_approx` using JAX functions.
        * Compute all required sufficient statistics (sums over time `t`) using `jax.numpy.sum`. Ensure all statistics needed for the M-step updates (including the one for `mu`, e.g., `S_h_t = sum(E[h_t])`, `S_h_tm1 = sum(E[h_{t-1}])`) are calculated.
        * Return the sufficient statistics dictionary (containing JAX arrays).

**Phase 4: M-Step Implementation**

* **Goal:** Implement the M-step function to update parameters (including `mu`) using the derived equations, sufficient statistics, and pure JAX operations.
* **File:** `src/bellman_filter_dfsv/estimation/em.py`
* **Action:**
    1. Define the `m_step` function: `def m_step(sufficient_stats: Dict, N: int, K: int, T: int, current_params_k: DFSVParamsDataclass) -> DFSVParamsDataclass:`.
    2. Implement the update equations derived in Phase 2 for `lambda_r`, `Phi_f`, `Phi_h`, `sigma2`, `Q_h`, **and `mu`** using JAX functions and the values from `sufficient_stats`. Note potential dependencies (e.g., needing `Phi_h_new` and `Q_h_new` to calculate `mu_new`). Handle this order appropriately.
    3. **Apply Constraints (using JAX):**
        * `lambda_r`: Apply `apply_identification_constraint`.
        * `sigma2`, `Q_h`: Enforce diagonal and positive constraints (e.g., `jnp.diag(jnp.maximum(jnp.diag(raw_update), EPS))`).
        * `Phi_f`, `Phi_h`: Apply stability projection if implemented.
    4. Construct a new `DFSVParamsDataclass` instance with all updated parameters (`lambda_r_new`, ..., `Q_h_new`, **`mu_new`**).
    5. Return the new parameter dataclass (pytree).

**Phase 5: EM Loop Implementation**

* **Goal:** Implement the main EM iteration loop using JAX.
* **File:** `src/bellman_filter_dfsv/estimation/em.py`
* **Action:**
    1. Define the main EM function: `def run_em(initial_params: DFSVParamsDataclass, observations: jnp.ndarray, filter_instance: DFSVBellmanInformationFilter, max_iter: int = 100, tolerance: float = 1e-4) -> Tuple[DFSVParamsDataclass, jnp.ndarray]:`.
    2. Initialize parameters `params_k = initial_params`.
    3. Initialize storage for convergence metrics (e.g., a JAX array using `jax.numpy.zeros`).
    4. Implement the loop (potentially using `jax.lax.while_loop` or `scan` for better JAX compatibility if computing likelihood within the loop for convergence check):
        * Call `e_step(params_k, observations, filter_instance)` to get `sufficient_stats`.
        * Call `m_step(sufficient_stats, N, K, T, params_k)` to get `params_kp1`.
        * Calculate change (e.g., norm of parameter difference `tree_leaves(params_kp1) - tree_leaves(params_k)`).
        * Store convergence metric.
        * Check convergence condition.
        * Update `params_k = params_kp1`.
    5. Return the final estimated parameters `params_k` and the convergence history (as JAX arrays/pytrees).
    6. Apply `@equinox.filter_jit` to the main EM step function or the entire `run_em` loop for performance.

**Phase 6: Testing and Validation**

* **Goal:** Thoroughly test the complete EM implementation using simulated data.
* **Files:** `tests/test_em.py` (new file).
* **Action:**
    1. Create `tests/test_em.py`.
    2. Generate simulated data using `simulate_dfsv` with known true parameters (`Θ_true`, **including `mu_true`**).
    3. Initialize the EM algorithm with parameters perturbed from the true values.
    4. Run the JIT-compiled `run_em` function.
    5. Compare the estimated parameters `Θ_est` (including `mu_est`) with `Θ_true`. Assess convergence and accuracy.
    6. Debug any issues using JAX debugging tools (`jax.debug.print`, `equinox.error_if` with `EQX_ON_ERROR=breakpoint`).
    7. Test edge cases.
    8. Run tests using `pytest`.
