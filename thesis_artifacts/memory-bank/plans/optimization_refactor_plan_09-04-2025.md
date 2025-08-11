# Plan: Align `mu` Fixing Logic in Optimization Utilities

**Date:** 09-04-2025

**Goal:** Align the logic for fixing the `mu` parameter in `src/bellman_filter_dfsv/utils/optimization.py` with the established strategy (Decision [04-06-2025 17:40:11]) and the implementation in `scripts/unified_filter_optimization.py`.

**Rationale:** Fixing `mu` is critical for the stability and reliable estimation of other parameters when using the Bellman Information Filter (BIF). The current logic in `optimization.py` only fixes `mu` under specific conditions (`true_params` provided AND `use_transformations` is False), which is inconsistent and potentially unstable for BIF runs.

**Proposed Changes:**

1.  **Modify `fix_mu` Condition in `run_optimization`:**
    *   **File:** `src/bellman_filter_dfsv/utils/optimization.py`
    *   **Function:** `run_optimization`
    *   **Current Logic:** `fix_mu = true_params is not None and not use_transformations`
    *   **Proposed Logic:** `fix_mu = (filter_type == FilterType.BIF) and (true_params is not None)`
        *   This ensures `mu` is flagged for fixing specifically for BIF runs whenever `true_params` are available to provide the value, regardless of whether transformations are used.

2.  **Ensure Objective Function Wrapper Fixes `mu`:**
    *   **Location:** Within the objective function wrapper created inside or before `get_objective_function` in `src/bellman_filter_dfsv/utils/optimization.py`, OR potentially within the `bellman_objective`/`transformed_bellman_objective` functions themselves if deemed cleaner.
    *   **Logic:** The wrapper (or the objective function) must check the `fix_mu` flag (determined in `run_optimization`). If `fix_mu` is `True`, it must use `eqx.tree_at` to replace the `mu` parameter in the parameter pytree with the `true_mu` value *before* passing the parameters to the filter's likelihood calculation. This needs to happen correctly both when `is_transformed=True` (after `untransform_params`) and `is_transformed=False`.

**Example Snippet (Illustrative - for objective wrapper):**

```python
# Inside the objective function wrapper (e.g., within get_objective_function)
# Assume 'params' are the potentially transformed parameters passed by the optimizer
# Assume 'true_mu' is available if fix_mu is True

def objective_wrapper(params, args_tuple):
    observations, filter_instance, priors_dict, penalty_weight, fix_mu_flag, true_mu_val = args_tuple # Add fix_mu info to args

    if is_transformed:
        params_iter = untransform_params(params)
    else:
        params_iter = params

    if fix_mu_flag:
        # Ensure true_mu_val is not None before attempting to fix
        if true_mu_val is not None:
             params_iter = eqx.tree_at(lambda p: p.mu, params_iter, true_mu_val)
        else:
             # Handle error: fix_mu is True but true_mu_val is None
             # Option 1: Raise an error
             # Option 2: Log a warning and proceed without fixing (less safe)
             print("Warning: fix_mu is True but true_mu value is not available.")


    # Apply identification constraint AFTER fixing mu
    params_fixed_constrained = apply_identification_constraint(params_iter)

    # Calculate loss using the selected objective (bellman_objective or pf_objective)
    # Note: The base objective functions (bellman_objective, pf_objective)
    # should NOT perform the mu fixing themselves if handled by the wrapper.
    loss = selected_obj( # selected_obj determined earlier based on filter_type
        params_fixed_constrained, observations, filter_instance,
        priors=priors_dict, stability_penalty_weight=penalty_weight
    )
    return loss, None # Assuming objective returns (loss, aux)
```

**Testing:**

*   Modify existing tests or add new ones in `tests/test_optimization.py` to verify:
    *   `run_optimization` correctly sets `fix_mu` for BIF when `true_params` are given, regardless of `use_transformations`.
    *   The objective function wrapper correctly applies the `true_mu` value when `fix_mu` is True.
    *   Optimization runs for BIF with fixed `mu` converge stably.

**Next Steps:**

1.  Review and approve this plan.
2.  Switch to Code mode to implement the changes in `src/bellman_filter_dfsv/utils/optimization.py`.
3.  Update or add relevant tests.
4.  Run tests to confirm correctness and stability.