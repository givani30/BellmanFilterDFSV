# Experimentation Plan: Improving BIF Optimization Stability via Transformations

**Date:** 2025-04-04

**Goal:** Identify parameter transformations that allow gradient-based optimizers (SGD, Adam) to converge reliably when minimizing the Bellman Information Filter (BIF) pseudo log-likelihood, especially when parameters are near their boundaries (variances near zero, persistences near one). This will be aided by logging of both transformed and untransformed parameters during optimization.

**Background:** Optimization using the BIF pseudo log-likelihood has shown instability, potentially due to numerical issues with the gradients of the current parameter transformations (`inverse_softplus` for variances, `logit` for persistence) near parameter boundaries.

**Metrics:**

*   **Primary:** Successful convergence reported by `optimistix` (Result `successful`) without numerical errors (NaN/Inf in loss or gradients during optimization).
*   **Secondary:** Number of steps taken (as reported by `optimistix`).
*   **Diagnostic:** Logged transformed and untransformed parameter values during optimization steps.

**Optimizers:**

*   `optimistix.OptaxMinimiser` with `optax.sgd`
*   `optimistix.OptaxMinimiser` with `optax.adam`
    *(Ensure Adam is uncommented in `scripts/bif_optimizer_stability.py`)*

**Baseline Transformation:**

*   Current transformations in `src/bellman_filter_dfsv/utils/transformations.py`:
    *   Variances (`sigma2`, `Q_h` diagonals): `softplus` / `inverse_softplus`
    *   Persistence (`Phi_f`, `Phi_h` diagonals): `sigmoid` / `logit`

**Experiment Steps:**

1.  **Modify Logging in Test Script:**
    *   **File:** `scripts/bif_optimizer_stability.py`
    *   **Location:** Inside the `transformed_objective_wrapper` function (around line 153).
    *   **Action:** Add `jax.debug.print` to log the `original_params` *after* they are untransformed within the wrapper, before the loss calculation. Ensure the loss calculation uses these `original_params`.
    *   **Example Code Snippet:**
        ```python
        # ... inside transformed_objective_wrapper(t_params, args_tuple) ...
        obs, filt = args_tuple # Unpack static args

        # Existing log for transformed params:
        jax.debug.print("[Optimizer Step] Transformed Params: {p}", p=t_params)

        # Untransform for objective calculation AND logging
        original_params = untransform_params(t_params)
        # --- ADD THIS ---
        jax.debug.print("[Optimizer Step] Original Params (Untransformed): {p}", p=original_params)
        # --- END ADD ---

        # Calculate loss using original_params
        # Note: Ensure bif_objective or equivalent is called with original_params
        loss = bif_objective(original_params, obs, filt) # Adjust if necessary based on script structure

        # Existing log for loss
        jax.debug.print("[Optimizer Step] Output loss: {loss}", loss=loss)
        return loss
        ```

2.  **Baseline Run:**
    *   Execute the *modified* `scripts/bif_optimizer_stability.py`.
    *   Record convergence results (Success/Fail, Steps) for SGD and Adam.
    *   Observe logged parameter values for initial behavior.

3.  **Experiment 1: Log Transform for Variances:**
    *   **Modify `transformations.py`:**
        *   `transform_params`: Replace `inverse_softplus` with `jnp.log(jnp.maximum(val, EPS))` for `sigma2`, `Q_h`.
        *   `untransform_params`: Replace `softplus` with `jnp.exp` for `sigma2`, `Q_h`.
    *   Run modified script & record results/observations.

4.  **Experiment 2: Sqrt Transform for Variances:**
    *   **Modify `transformations.py`:**
        *   `transform_params`: Replace `inverse_softplus` with `jnp.sqrt(jnp.maximum(val, EPS))` for `sigma2`, `Q_h`.
        *   `untransform_params`: Replace `softplus` with `lambda x: x**2` for `sigma2`, `Q_h`.
    *   Run modified script & record results/observations.

5.  **Analyze Variance Results:**
    *   Compare results from Baseline, Exp 1, and Exp 2.
    *   Identify which variance transformation (softplus, log, sqrt) yields the most reliable convergence and stable parameter behavior for SGD and Adam.

6.  **Experiment 3: Scaled `atanh` for Persistence:**
    *   **Choose Best Variance Transform:** Select the best-performing variance transformation from step 5.
    *   **Modify `transformations.py`:**
        *   Use the chosen variance transform/untransform.
        *   `transform_params`: Replace `logit` with `lambda p: jnp.arctanh(2 * jnp.clip(p, EPS, 1 - EPS) - 1)` for `Phi_f`, `Phi_h`.
        *   `untransform_params`: Replace `sigmoid` with `lambda t: (jnp.tanh(t) + 1) / 2` for `Phi_f`, `Phi_h`.
    *   Run modified script & record results/observations.

7.  **Final Analysis:**
    *   Compare the results of all successful transformation combinations.
    *   Determine which combination provides the most reliable convergence and stable parameter behavior for both SGD and Adam based on success metrics and logged values.
    *   Document the findings and the recommended transformation strategy.

**Workflow Diagram:**

```mermaid
graph TD
    A[Start: Baseline (Softplus/Logit)] --> B(1. Modify Logging in Script);
    B --> C(2. Run Baseline);
    C --> D{Converges Reliably?};
    D -- Yes --> Z[End: Baseline OK];
    D -- No --> E{Try Variance Alternatives};

    E --> F1[3. Exp 1: Log Variance];
    F1 --> G1(Modify Transformations & Run);
    G1 --> H1{Converges Reliably?};

    E --> F2[4. Exp 2: Sqrt Variance];
    F2 --> G2(Modify Transformations & Run);
    G2 --> H2{Converges Reliably?};

    H1 -- Yes --> I(5. Analyze Variance Results);
    H2 -- Yes --> I;
    H1 -- No --> J{Try Persistence Alternatives};
    H2 -- No --> J;

    I --> J;

    J --> K1[6. Exp 3: Atanh Persistence (with Best Variance Alt)];
    K1 --> L1(Modify Transformations & Run);
    L1 --> N1{Converges Reliably?};

    N1 -- Yes --> O(7. Final Analysis);
    N1 -- No --> P[Further Investigation Needed];

    O --> Q[Select Best Transformation];
    P --> Q;
    Q --> Z;
```

**Contingency:**

*   If no tested transformation provides reliable convergence, further investigation is needed (e.g., BIF implementation review, optimizer hyperparameter tuning, advanced constrained optimization techniques, non-diagonal Phi transformations).