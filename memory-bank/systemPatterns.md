# System Patterns

This file documents recurring patterns and standards used in the project.

*

## Coding Patterns

* **JAX Ecosystem Integration:** Heavy reliance on JAX for performance-critical computations, especially within filtering algorithms. This includes:
  * `@jit` for function compilation.
    * **Preference:** `@equinox.filter_jit` is preferred over `@jax.jit` for better handling of static arguments and general Equinox integration (as per `.clinerules`). [Added 04-05-2025 23:05:00]

  * `jax.lax.scan` for efficient temporal loops (e.g., in Particle Filter).
  * `jax.vmap` for parallel computation over particles/data.
  * `jax.lax.cond` for conditional execution (e.g., resampling).
  * `jax.random` for stochastic operations.
  * JAX optimization libraries (`jaxopt`, `optimistix`) for parameter updates (e.g., BFGS in Bellman Filter).
* **JAX Pytree Parameters:** Use of `jax_dataclasses` (`@jdc.pytree_dataclass`) to define model parameters (e.g., `DFSVParamsDataclass` in `models/dfsv.py`). This allows parameter objects to be treated as JAX pytrees, enabling direct use with JAX transformations (`jit`, `grad`, etc.) and clear separation of static (`N`, `K`) vs. differentiable parameters.
* **Helper Modules:** Encapsulation of complex mathematical implementations or optimization steps into internal helper modules (e.g., `_bellman_impl.py`, `_bellman_optim.py`) to keep primary class definitions cleaner.
* **Parameter Passing:** Refactored `DFSVParticleFilter` to accept parameters externally via method arguments rather than storing them internally, facilitating JIT compilation across different parameter sets (Decision [01-04-2025 03:37:00]).

* **Centralized Optimizer Creation (`solvers.py`):**
  * Provides `create_optimizer` function for instantiating various optimizers:
    * Custom BFGS variants (DogLeg, Armijo, DampedTrustRegion, IndirectTrustRegion).
    * Standard Optax optimizers (Adam, AdamW, SGD, RMSProp, Adagrad, Adadelta, Adafactor, Lion).
  * Includes `create_learning_rate_scheduler` for flexible LR scheduling (`cosine`, `exponential`, `linear`, `warmup_cosine`, `constant`).
  * Employs `optax.apply_if_finite` for numerical stability with gradient-based optimizers.
* **Optimization Orchestration (`optimization.py`):**
  * Standardizes the optimization workflow using `run_optimization`.
  * Selects and wraps objective functions (`get_objective_function`) to handle transformations, priors, and stability penalties.
  * Enables detailed parameter/loss history logging via `minimize_with_logging` wrapper around `optimistix` solvers.
  * Handles parameter transformations (`transform_params`, `untransform_params`) and identification constraints (`apply_identification_constraint`).
  * Provides option to log parameters during optimization with `log_params=True` flag.
  * Uses `BestSoFarMinimiser` to keep track of the best parameter values during optimization.
  * Defaults to using the built-in Optimistix minimizer when `log_params=False` for better performance.
  * **Note:** `mu` fixing logic previously aligned with BIF strategy (Decision [11-04-2025 17:30:00]). Now supports both fixed and unfixed `mu` approaches (Decision [16-04-2025 02:52:23]).
* **Unified Experiment Script (`unified_filter_optimization.py`):**
  * Provides a framework for comparative experiments across filters and optimizers.
  * Supports running multiple filter types (BIF, PF) with different optimizers (AdamW, DampedTrustRegionBFGS).
  * Includes comprehensive results analysis and visualization capabilities.
  * Saves estimated parameters to pickle files for further analysis.

## Architectural Patterns

* **Filter Base Class:** Filtering implementations (`DFSVBellmanFilter`, `DFSVParticleFilter`, `DFSVBellmanInformationFilter`) inherit from a common base class (`core/filters/base.py`), suggesting a common interface for filters.
* **Bellman Filter Strategy (Lange, 2024) for DFSV Model:**
  * **Core Idea:** Applies Bellman's dynamic programming principle to approximate the posterior mode of the state vector `α_t = [f_t', h_t']'` in the DFSV model. Avoids integration by focusing on optimization.
  * **Value Function Approximation:** Approximates the value function `V_t(α_t)` at each step `t` with a multivariate quadratic function, parameterized by its mode `a_{t|t}` and negative Hessian (information matrix) `I_{t|t}` (Lange, Eq. 8). This is exact for linear Gaussian models (Kalman filter is a special case).
  * **Recursive Update:** The core recursion involves solving `V_t(a_t) = ℓ(y_t|a_t) + max_{a_{t-1}} [ ℓ(a_t|a_{t-1}) + V_{t-1}(a_{t-1}) ]` (Lange, Eq. 5).
  * **Optimization Step:** The filter update requires solving a joint optimization problem at each step `t` (Lange, Eq. 7):
        `[a_{t|t}', a_{t-1|t}']' = argmax_{a_t, a_{t-1}} [ ℓ(y_t|a_t) + ℓ(a_t|a_{t-1}) + V_{t-1}(a_{t-1}) ]`
        For linear Gaussian state transitions (like the DFSV model), this simplifies to maximizing `ℓ(y_t|a_t) - 1/2 ||a_t - a_{t|t-1}||^2_{I_{t|t-1}}` w.r.t `a_t` (Lange, Eq. 16).
  * **Information Matrix Update:** The precision/information matrix is updated using derivatives of the log-likelihoods (Lange, Eq. 11). The Fisher Information version is often used: `I_{t|t} ≈ I_{t|t-1} + (-E[d^2ℓ(y_t|a_t)/da_t^2 | a_t])`.
  * **Original DFSV Implementation (Covariance-based):**
    * Used block coordinate descent (or BFGS) for the state update optimization within `_bellman_optim.py`.
    * Calculated Fisher Information assuming `I_fh = 0` (block-diagonal FIM between factors `f` and log-vols `h`).
    * Used the correct predicted log-volatility `h_{t|t-1}` for state-dependent process noise `Q_f` in prediction (Corrected [01-04-2025 02:27:00]).
    * **Sensitivity:** Was numerically sensitive to small process noise `Q_h`. When `Q_h` was small, the predicted precision `I_{t|t-1}` dominated the likelihood information `I_fisher` in the update step, hindering log-volatility estimation. Using a larger `Q_h` in simulations mitigated this (Identified [01-04-2025 02:28:00], Confirmed [01-04-2025 02:48:00]).
    * **Instability:** Showed persistent numerical instability (NaN propagation) during optimization runs, traced to the precision matrix update (`I_updated = I_pred + J`) and inversion (`P_updated = I_updated^-1`) cycle. Debugging efforts were halted (Decision [02-04-2025 19:57:00]).
  * **Current Approach (Bellman Information Filter - BIF):**
    * **Motivation:** Implemented to address numerical instability observed in the covariance-based Bellman filter, particularly during parameter optimization.
    * **Approach:** Directly propagates the information state (state `alpha`, information matrix `Omega = P^-1`). Follows Lange (2024) methodology.
    * **Prediction:** Uses Joseph form / Woodbury identity for stable information matrix prediction (`__predict_jax_info`).
    * **Update:** Reuses block coordinate descent (`_block_coordinate_update_impl`) for state update, passing predicted information `Omega_{t|t-1}`. Updates information matrix via `Omega_{t|t} = Omega_{t|t-1} + J_observed`.
    * **Likelihood:** Uses pseudo log-likelihood from Lange (2024, Eq. 40) involving a specific KL-type penalty (`_kl_penalty_pseudo_lik_impl`).

    * **Numerical Stability:** Debugging revealed that the calculated Expected FIM (`J_observed` in the update step) could become numerically non-positive semi-definite (non-PSD) for certain state values, leading to filter instability. To mitigate this, the implementation now includes eigenvalue clipping regularization: negative eigenvalues of `J_observed` are clipped to a small positive value (`1e-8`) before the matrix is used in the information update (`updated_info = predicted_info + J_observed_psd + jitter`). (See Decision [04-05-2025 16:00:00]).

    * **Identifiability Constraints:** While fixing `mu` was initially used to address estimation bias (especially in BIF), recent studies show both fixed and unfixed approaches are viable. Constraints on the factor loading matrix `lambda_r` (e.g., lower-triangular with fixed diagonal) remain important for resolving rotational/scale ambiguity in the latent factors and other parameters. (See Decisions [04-05-2025 21:14:00] and [16-04-2025 02:52:23])
    * **Location:** `src/bellman_filter_dfsv/core/filters/bellman_information.py`.
* **Bellman Smoother (Lange, 2024, Sec. 6):**
  * Extends the Bellman filter using forward *and* backward value functions (`V_t`, `W_t`) to compute smoothed state estimates `a_{t|n}`.
  * Relies on combining forward and backward value functions via the state transition density (Lange, Prop 3, Eq. 31-32).
  * For linear Gaussian state transitions (like DFSV), if value functions are approximated quadratically, the resulting smoother recursions are identical to the classic Rauch-Tung-Striebel (RTS) smoother (Lange, Prop 4, Eq. 33-34).
  * This allows using standard RTS formulas with the Bellman filter's output (`a_{t|t}`, `I_{t|t}`, `I_{t+1|t}`) to obtain approximate smoothed estimates for non-Gaussian observation models.
* **Parameter Estimation (Lange, 2024, Sec. 7):**
  * Estimates static hyperparameters (Θ) by maximizing an approximate pseudo log-likelihood derived from the Bellman filter's output.
  * Decomposes the exact log-likelihood contribution `ℓ(y_t|F_{t-1})` into a 'fit' term `ℓ(y_t|a_{t|t})` and a 'realized KL divergence' penalty term `ℓ(α_t|F_t) - ℓ(α_t|F_{t-1})` evaluated at `a_{t|t}` (Lange, Eq. 36).
  * Approximates the KL divergence term using the predicted and filtered state estimates and their information matrices (Lange, Eq. 39).
  * The final pseudo log-likelihood to maximize is (Lange, Eq. 40 / Boekestijn, Eq. 3.11):
        `L(Θ) ≈ Σ [ ℓ(y_t|a_{t|t}) - 0.5 * (log(det(I_{t|t})/det(I_{t|t-1})) + ||a_{t|t} - a_{t|t-1}||^2_{I_{t|t-1}}) ]`
  * This approach avoids sampling/integration and allows standard gradient-based optimization.
* **Particle Filter Strategy:** Implements a Bootstrap Filter (SISR) using `jax.lax.scan` for the main loop, with ESS-triggered Systematic Resampling to mitigate particle degeneracy.
* **Optimization Pattern (Woodbury Identity & MDL):** Applied to `log_posterior_impl` (`_bellman_impl.py`) to avoid direct O(N^3) Cholesky decomposition/inversion of the N x N observation covariance matrix `A = D + UCU.T`. Replaced with operations involving the inverse of the K x K matrix `M = C^-1 + U.T D^-1 U`, reducing complexity significantly when N >> K (Implemented [01-04-2025 11:46:00]). Also applied to `fisher_information_impl` for `A_inv` applications (Implemented [01-04-2025 12:05:00]).
* **Optimization Pattern (Rank-1 FIM Reformulation):** Utilized the rank-1 structure of `dSigma_k = exp(h_k) * lambda_k * lambda_k.T` to reformulate the calculation of the `Omega_hh` block in `fisher_information_impl`. Changed the O(K^2*N^2) `einsum` calculation `trace(B_k B_l)` to the equivalent `exp(h_k + h_l) * (lambda_k.T @ A_inv @ lambda_l)^2`, where the inner term is reused from the `Omega_ff` calculation. This avoids the expensive `einsum` (Implemented [01-04-2025 12:05:00]).
* **Optimization Pattern (Diagonal Covariance):** Optimized particle filter likelihood calculation (`compute_log_likelihood_particle`) to exploit known diagonal structure of observation noise covariance `R_t`, avoiding redundant `vmap`ped decompositions (Implemented [01-04-2025 05:13:00]).

## Testing Patterns

* **Unit Tests:** Use `pytest` framework for running tests located in the `tests/` directory.
* **Filter Comparison:** Tests often compare outputs of different filter implementations or optimized vs. non-optimized versions (e.g., `tests/test_bellman_unified.py`).
* **Profiling Scripts:** Dedicated scripts (`scripts/profile_*.py`, `scripts/benchmark_*.py`) used to measure performance and identify bottlenecks.
* **Runtime Error Debugging:** Use `equinox.error_if` to add runtime checks for conditions like NaN/Inf within JIT-compiled functions. Running the code with the environment variable `EQX_ON_ERROR=breakpoint` will then trigger the debugger precisely where the check fails, aiding in pinpointing errors that occur during gradient calculations or within JITted filter steps (Identified [04-04-2025 15:25:00]).


## Optimization Stability Patterns [11-04-2025 18:15:33]

### Early Stability Detection
* Check eigenvalues of Phi matrices before full likelihood calculation
* Use `jax.lax.cond` for efficient branching in JIT-compiled code
* Return large penalty values (1e10 + weighted violation) for unstable configurations

### BFGS Update Safety
* Implement fallback to initial values when optimization fails
* Use `jnp.where` for conditional updates in JIT context

### Gradient Management
* Apply `optax.clip_by_global_norm(1.0)` to prevent explosive gradients
* Use `optax.apply_if_finite` to handle NaN/Inf gracefully
* Maintain optimization state validity through bounded transformations


---
**[14-04-2025 02:30:00] - Numerical Stability: Differentiating Eigendecomposition**
- **Pattern:** Automatic differentiation (AD) through eigenvalue decomposition (`jnp.linalg.eigh`) can be numerically unstable, especially if the input matrix has near-degenerate (close) eigenvalues. The backward pass calculation involves terms like `1 / (λ_i - λ_j)`, which can lead to NaN/Inf if eigenvalues are very close.
- **Context:** Encountered during BIF gradient calculation (`value_and_grad`) when differentiating the regularization step involving `jnp.linalg.eigh(J_observed)` in `__update_jax_info`.
- **Solution/Mitigation:**
    - **Jittering:** Add a small multiple of the identity matrix (`jitter * jnp.eye(...)`) to the input matrix *before* calling `eigh`. This perturbs eigenvalues slightly, potentially separating near-degenerate ones and stabilizing the gradient. (Standard practice).
    - **Expected FIM:** Replace the Observed FIM (which requires differentiating the Hessian/`eigh`) with the Expected FIM if its analytical form is more stable under differentiation. (Implemented by user in this case).
    - **Custom VJP:** Define a custom vector-Jacobian product for the `eigh` operation or the surrounding regularization block, implementing a numerically stable backward pass manually. (More complex).
    - **Stop Gradient:** Use `jax.lax.stop_gradient` on the `eigh` outputs or the resulting regularized matrix if the gradient information through this specific step is not critical for optimization.
* **JIT Static vs Traced Values:** When using JAX transformations like `@eqx.filter_jit`, ensure that values used to define computation graph structure (e.g., loop bounds, shapes for `jnp.arange`, dimensions for indexing) are static (known at compile time) or marked as static arguments. Using traced values (derived from function arguments that are JAX arrays/pytrees) for these purposes can lead to `ConcretizationTypeError` or `NonConcreteBooleanIndexError`. Vectorized operations often avoid these issues compared to explicit loops or conditional logic based on traced values. (Noted [11-04-2025 16:23:11])
