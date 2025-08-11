# Plan: Implement Prior Regularization Framework

**Goal:** Enhance the `log_prior_density` function in `src/bellman_filter_dfsv/core/likelihood.py` to calculate and combine log-prior densities for all relevant model hyperparameters (`lambda_r`, `Phi_f`, `Phi_h`, `mu`, `sigma2`, `Q_h`), and integrate this into the optimization objective functions.

**Target File:** `src/bellman_filter_dfsv/core/likelihood.py`

**Primary Function to Modify:** `log_prior_density`

**Related Functions to Modify:**
*   `bellman_objective`
*   `transformed_bellman_objective`
*   `pf_objective`
*   `transformed_pf_objective`

**Detailed Implementation Plan:**

1.  **Refactor `log_prior_density` Signature:**
    *   Modify the function signature to accept the `params: DFSVParamsDataclass` and a comprehensive set of prior hyperparameters.
    *   Example Signature:
        ```python
        def log_prior_density(
            params: DFSVParamsDataclass,
            # Mu Prior Params
            prior_mu_mean: float | jnp.ndarray = 0.0,
            prior_mu_var: float | jnp.ndarray = 1.0,
            # Sigma2 Prior Params (Inverse Gamma)
            prior_sigma2_alpha: float = 3.0,
            prior_sigma2_beta: float = 0.1,
            # Q_h Prior Params (Inverse Gamma for diagonal elements)
            prior_q_h_alpha: float = 3.0,
            prior_q_h_beta: float = 0.05,
            # Lambda_r Prior Params (Normal)
            prior_lambda_mean: float = 0.0,
            prior_lambda_var: float = 1.0,
            # Phi_f Prior Params (Normal on elements - interim)
            prior_phi_f_mean: float = 0.0, # Mean for off-diagonals
            prior_phi_f_diag_mean: float = 0.9, # Mean for diagonals
            prior_phi_f_var: float = 0.5,
            # Phi_h Prior Params (Normal on elements - interim)
            prior_phi_h_mean: float = 0.0, # Mean for off-diagonals
            prior_phi_h_diag_mean: float = 0.95, # Mean for diagonals
            prior_phi_h_var: float = 0.1
        ) -> jnp.ndarray:
            # ... implementation ...
        ```
    *   **Justification:** Centralizes prior specification, making it easier to manage and modify. Allows different prior strengths for different parameter types.

2.  **Implement Prior for `mu` (Gaussian):**
    *   Calculate the sum of log-PDFs for a Normal distribution for each element of `params.mu`.
    *   `log_pdf_mu = jnp.sum(-0.5 * (jnp.log(2 * jnp.pi * prior_mu_var) + ((params.mu.flatten() - prior_mu_mean) / jnp.sqrt(prior_mu_var))**2))`
    *   Add this `log_pdf_mu` to the `total_log_prior`.

3.  **Implement Prior for `sigma2` (Inverse Gamma):**
    *   Retain the existing Inverse-Gamma prior calculation using `jax.vmap` for the diagonal elements of `params.sigma2`. Ensure it uses the new hyperparameters from the function signature (`prior_sigma2_alpha`, `prior_sigma2_beta`).
    *   Add the result to `total_log_prior`.

4.  **Implement Prior for `Q_h` (Inverse Gamma - Diagonal):**
    *   **Assumption:** `Q_h` remains diagonal for now (consistent with `transformations.py`). If it becomes non-diagonal later, an Inverse-Wishart prior would be needed.
    *   Implement an Inverse-Gamma prior for each diagonal element of `params.Q_h`, similar to `sigma2`, using `prior_q_h_alpha` and `prior_q_h_beta`.
    *   Use `jnp.diag(params.Q_h)` to get the diagonal elements.
    *   Use `jax.vmap(inverse_gamma_log_pdf, ...)` for calculation.
    *   Add the result to `total_log_prior`.

5.  **Implement Prior for `lambda_r` (Gaussian):**
    *   Calculate the sum of log-PDFs for a Normal distribution for each element of `params.lambda_r`.
    *   `log_pdf_lambda = jnp.sum(-0.5 * (jnp.log(2 * jnp.pi * prior_lambda_var) + ((params.lambda_r - prior_lambda_mean) / jnp.sqrt(prior_lambda_var))**2))`
    *   Add this `log_pdf_lambda` to the `total_log_prior`.

6.  **Implement Prior for `Phi_f` (Gaussian on Elements - Interim):**
    *   **Acknowledge Limitation:** This approach does *not* enforce stationarity (eigenvalues < 1) for the full matrix. This is an interim solution until matrix transformations are updated.
    *   Define separate means for diagonal (`prior_phi_f_diag_mean`) and off-diagonal (`prior_phi_f_mean`) elements.
    *   Construct a prior mean matrix `M_f` with `prior_phi_f_diag_mean` on the diagonal and `prior_phi_f_mean` elsewhere.
    *   Calculate the sum of log-PDFs for a Normal distribution for each element:
        `log_pdf_phi_f = jnp.sum(-0.5 * (jnp.log(2 * jnp.pi * prior_phi_f_var) + ((params.Phi_f - M_f) / jnp.sqrt(prior_phi_f_var))**2))`
    *   Add this `log_pdf_phi_f` to the `total_log_prior`.
    *   **Add Code Comment:** Include a prominent `TODO` comment explaining the stationarity limitation and the need to revise this prior when matrix transformations are implemented.

7.  **Implement Prior for `Phi_h` (Gaussian on Elements - Interim):**
    *   Apply the same logic as for `Phi_f`, using `prior_phi_h_diag_mean`, `prior_phi_h_mean`, and `prior_phi_h_var`.
    *   Construct the prior mean matrix `M_h`.
    *   Calculate `log_pdf_phi_h` similarly.
    *   Add this `log_pdf_phi_h` to the `total_log_prior`.
    *   **Add Code Comment:** Include the same `TODO` regarding stationarity.

8.  **Combine and Stabilize:**
    *   Ensure `total_log_prior` correctly sums all individual log-prior densities.
    *   Retain the final check: `jnp.where(jnp.isnan(total_log_prior) | jnp.isinf(total_log_prior), -1e10, total_log_prior)` for numerical stability.

9.  **Integrate into Objective Functions:**
    *   Modify `bellman_objective`, `transformed_bellman_objective`, `pf_objective`, and `transformed_pf_objective`.
    *   Remove the existing specific prior penalty calculation for `mu` within these functions.
    *   Add calls to the enhanced `log_prior_density` function.
        *   For `*_objective` (standard parameters): Call `log_prior_density(params, ...)` directly.
        *   For `transformed_*_objective`: Call `log_prior_density(original_params, ...)` after the `untransform_params` call.
    *   Subtract the returned `log_prior` value from the negative log-likelihood (since the objective is typically minimizing negative log-likelihood + negative log-prior). `total_objective = safe_neg_ll - log_prior`.
    *   Pass the necessary prior hyperparameters through the objective function signatures or configure them elsewhere (e.g., via partial application or a config object).

10. **Documentation:**
    *   Update the docstring for `log_prior_density` to detail all implemented priors, their hyperparameters, assumptions (like diagonal `Q_h`), and the limitations regarding `Phi_f`/`Phi_h` stationarity.
    *   Update docstrings for the objective functions to reflect that priors are now handled by `log_prior_density`.

11. **Testing:**
    *   Add unit tests for `log_prior_density` itself:
        *   Test with known parameter values and hyperparameters to check if the calculated log-prior density is correct for each component.
        *   Test edge cases (e.g., zero variance parameters if using IG).
        *   Test the combination of priors.
    *   Modify existing optimization tests (like `bif_optimizer_stability.py`) to use the new objective functions incorporating the full priors.

12. **Memory Bank Update:**
    *   Create a new entry in `decisionLog.md` detailing the decision to implement a comprehensive prior framework, the chosen prior distributions (including the interim approach for `Phi`), and the rationale.
    *   Optionally, update `systemPatterns.md` to reflect the standard prior choices.

**Diagram (Conceptual Flow):**

```mermaid
graph TD
    subgraph Optimization Objective (e.g., transformed_bellman_objective)
        A[Input: Transformed Params, Data, Prior Hyperparams] --> B{Untransform Params};
        B --> C[original_params];
        C --> D{Calculate Filter Likelihood};
        D --> E[neg_ll];
        C --> F{Call log_prior_density};
        F -- Prior Hyperparams --> G[log_prior];
        E & G --> H{Combine: neg_ll - log_prior};
        H --> I[Output: Total Objective Value];
    end