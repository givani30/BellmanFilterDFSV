# Plan: Remove Python Control Flow from JAX JIT Paths

**Objective:** Enhance JAX JIT compilation efficiency and robustness by removing Python `try...except` blocks and dynamic `if` checks from JIT-compiled functions within the Bellman and Particle filter implementations.

**Problem:**
1.  The use of `try...except jnp.linalg.LinAlgError` blocks within JIT-compiled functions prevents proper JIT compilation and can mask numerical issues.
2.  The use of `if` statements checking dynamic properties like array dimensions (`ndim`) or shapes (`shape`) within JIT-compiled functions also breaks JIT compilation.

**Refactoring Strategy:**

1.  **Remove `try...except LinAlgError`:** Eliminate these blocks entirely from the identified JIT-compiled functions.
2.  **Standardize on Cholesky Inversion/Decomposition:** Use `jax.scipy.linalg.cholesky` (potentially followed by `jax.scipy.linalg.cho_solve` for inversion) as the standard method.
3.  **Ensure Jittering:** Verify that a small positive jitter (e.g., `1e-8 * jnp.eye(...)`) is consistently added to matrices *before* attempting Cholesky decomposition to improve numerical stability.
4.  **Remove `pinv` Fallbacks:** Delete the `jnp.linalg.pinv` calls currently located within the `except` blocks in JIT paths.
5.  **Move Dynamic `if` Checks:** Relocate `if` statements that check dynamic array properties (like `ndim`, `shape`) from JIT-compiled functions to their calling context (before the JIT call). The JITted function should receive pre-validated inputs or assume a specific structure.
6.  **Error Handling:** Allow JAX's `LinAlgError` to propagate naturally during runtime if Cholesky fails on a non-PSD matrix within the JIT path. Handle errors *outside* the JIT context. Use `jnp.where` or `jnp.nan_to_num` inside JIT paths if specific numerical results (like `-inf` for invalid likelihoods) are desired instead of errors.

**Affected Files and Functions:**

1.  **`src/bellman_filter_dfsv/core/filters/bellman_information.py`**
    *   `__predict_jax_info`: Remove `try/except` around `chol_Qh` and `chol_M`. Rely on jittered Cholesky.
    *   `_invert_info_matrix`: Remove `try/except` around `chol_info`. Rely on jittered Cholesky.
2.  **`src/bellman_filter_dfsv/core/filters/_bellman_impl.py`**
    *   `observed_fim_impl`: Remove `try/except` around `L_M`. Rely on jittered Cholesky.
    *   `log_posterior_impl`: Remove `try/except` around `L_M`. Rely on jittered Cholesky.
3.  **`src/bellman_filter_dfsv/core/filters/bellman.py`**
    *   `__update_jax`: Remove `try/except` around `chol_pred_cov` and `chol_Omega_post`. Rely on jittered Cholesky.
4.  **`src/bellman_filter_dfsv/core/filters/particle.py`**
    *   `_jit_filter_scan`:
        *   Remove `try/except` around `chol_Q_h_local`. Rely on jittered Cholesky. If Cholesky fails here, return `-jnp.inf` using `jnp.where` or allow error propagation.
        *   Remove `if/elif/else` block checking `sigma2_curr.ndim` and `sigma2_curr.shape`. This logic should be moved to the calling function (`log_likelihood_of_params`), and `_jit_filter_scan` should receive the pre-calculated `obs_noise_variances` directly.
    *   *(Optional Recommended)* `initialize_particles`: Remove `try/except` around `L`. Rely on jittered Cholesky.
    *   *(Optional Recommended)* `filter`: Remove `try/except` around `chol_Q_h`. Rely on jittered Cholesky.

**Verification:**

*   After applying the changes, run the full test suite (`uv run pytest`) to ensure no regressions are introduced.
*   Monitor performance and stability during subsequent optimization runs or simulations.

**Diagram (Conceptual Flow Change - remains the same as previous):**

```mermaid
graph TD
    subgraph Before Refactoring (Inside JIT Function)
        A[Add Jitter to Matrix M] --> B{Try Cholesky(M)};
        B -- Success --> C[Solve using Cholesky];
        B -- LinAlgError --> D{Except LinAlgError};
        D --> E[Use pinv(M)];
        C --> F[Result];
        E --> F;
    end

    subgraph After Refactoring (Inside JIT Function)
        G[Add Jitter to Matrix M] --> H[Cholesky(M)];
        H --> I[Solve using Cholesky];
        I --> J[Result];
        H -- LinAlgError during Runtime --> K(Error Propagates Out);
    end

    Start --> A;
    Start --> G;