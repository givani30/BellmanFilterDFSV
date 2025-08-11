# Plan: Implement Lower-Triangular Lambda_r Constraint for BIF Estimation

**Goal:** Address the identifiability issue between `mu` and `lambda_r` in the DFSV BIF estimation by imposing a lower-triangular structure on `lambda_r` with a strictly positive diagonal. This will be tested in a new script without priors.

**Requirements:**

1.  `lambda_r` must be lower-triangular.
2.  The diagonal elements of `lambda_r` must be strictly positive (`> 0`).
3.  The constraints should be handled appropriately during optimization (using transformations where possible).
4.  The *true* parameters used for data simulation must also adhere to these constraints.
5.  Implementation should occur in a new, dedicated script (`scripts/test_bif_identifiability_fix.py`).
6.  The script should focus on a single optimizer run without priors.

**Plan Details:**

1.  **Modify Core Transformation Functions (`src/bellman_filter_dfsv/utils/transformations.py`):**
    *   **Integrate Positive Diagonal Constraint:**
        *   Update `transform_params`: Apply `inverse_softplus` only to the diagonal elements of `params.lambda_r`. Off-diagonal elements remain unchanged by this function.
        *   Update `untransform_params`: Apply `softplus` only to the diagonal elements of `transformed_params.lambda_r` to ensure positivity. Off-diagonal elements remain unchanged.
    *   *Rationale:* This cleanly maps the positive diagonal constraint to an unconstrained space, fitting the purpose of the transformation functions.

2.  **Create New Test Script (`scripts/test_bif_identifiability_fix.py`):**
    *   **Base Script:** Copy and simplify `scripts/test_bif_priors_optimizers.py`.
        *   Remove all prior-related code (configurations, objective modifications, reporting).
        *   Configure for a single optimizer (e.g., AdamW).
        *   Remove comparison loops; perform only one optimization run.
        *   Simplify results reporting and saving.
    *   **Modify Data Generation (`create_simple_model` or equivalent):**
        *   Generate an initial `lambda_r` matrix (e.g., random).
        *   Apply `lambda_r = jnp.tril(initial_lambda_r)` to make it lower-triangular.
        *   Ensure diagonal elements are positive: `diag_indices = jnp.diag_indices_from(lambda_r); lambda_r = lambda_r.at[diag_indices].set(jnp.abs(lambda_r[diag_indices]) + 1e-6)` (add epsilon for strict positivity).
        *   Use this constrained `lambda_r` when creating the `true_params` dataclass for simulation.
    *   **Define Lower-Triangular Constraint Helper:**
        ```python
        import jax.numpy as jnp
        from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass

        def apply_lower_triangular_constraint(params: DFSVParamsDataclass) -> DFSVParamsDataclass:
            """Applies the lower-triangular constraint to lambda_r."""
            constrained_lambda_r = jnp.tril(params.lambda_r)
            return params.replace(lambda_r=constrained_lambda_r)
        ```
    *   **Modify Initial Guess Handling:**
        *   Create `uninformed_params`.
        *   Apply `uninformed_params = apply_lower_triangular_constraint(uninformed_params)`.
        *   Transform using the *modified* `transform_params`: `initial_y = transform_params(uninformed_params)`.
    *   **Define New Objective Function Wrapper:**
        ```python
        # Inside the new script
        @eqx.filter_jit # Or appropriate JIT decorator
        def constrained_transformed_objective(transformed_params, y, filter_instance):
             # 1. Untransform (handles positive diagonal via modified untransform_params)
             params_positive_diag = untransform_params(transformed_params)
             # 2. Apply lower-triangular structure
             constrained_params = apply_lower_triangular_constraint(params_positive_diag)
             # 3. Calculate likelihood (no priors)
             log_lik = filter_instance.jit_log_likelihood_of_params()(constrained_params, y)
             safe_neg_ll = jnp.nan_to_num(-log_lik, nan=1e10, posinf=1e10, neginf=1e10)
             return safe_neg_ll
        ```
    *   **Use New Objective:** Pass `constrained_transformed_objective` to `optx.minimise`.
    *   **Modify Final Parameter Handling:**
        *   Get final transformed parameters: `final_y = sol.value`.
        *   Untransform using *modified* `untransform_params`: `params_final_pos_diag = untransform_params(final_y)`.
        *   Apply lower-triangular constraint: `final_params_constrained = apply_lower_triangular_constraint(params_final_pos_diag)`.
        *   Use `final_params_constrained` for printing comparison tables.

**Implementation Handoff:**

*   The Code mode will implement the changes to `src/bellman_filter_dfsv/utils/transformations.py` and create the new script `scripts/test_bif_identifiability_fix.py` according to this plan.