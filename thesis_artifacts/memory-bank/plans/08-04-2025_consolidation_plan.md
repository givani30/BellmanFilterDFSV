# Plan: Consolidate DFSV Objective Functions, Enforce Constraints, and Unify Filter APIs  
**Date:** 08-04-2025

---

## Goals

- Remove duplicate logic in Bellman and Particle filter objective functions.
- Enforce identification constraints consistently.
- Standardize stability penalty.
- Unify log-likelihood API across filters.
- Ensure full JAX compatibility (no Python control flow).

---

## 1. Consolidate Objective Logic

- Create `_compute_total_objective` helper:
  - Applies `apply_identification_constraint`.
  - Computes log-likelihood via passed-in closure.
  - Adds prior penalty if priors provided.
  - Adds standardized stability penalty (sum of eigenvalue violations).
  - Returns **only** total objective.
- Use `jax.lax.cond` for all control flow (priors, penalty weight).

---

## 2. Standardize Stability Penalty

- Penalize **sum of relu violations** of eigenvalues > 1 for `Phi_f` and `Phi_h`.
- Remove max-eigenvalue squared penalty.

---

## 3. Enforce Identification Constraint

- Always apply **after untransforming** parameters.
- Done inside the helper before likelihood/prior/penalty.

---

## 4. Unify Log-Likelihood API

- Refactor `DFSVParticleFilter.jit_log_likelihood_wrt_params()` to:
  - Accept **only** `(params, y)`.
  - Internally compute Cholesky of `Q_h` with jitter, using `jax.lax.cond`.
  - Internally extract observation noise variances.
  - Internally call `_jit_filter_scan_for_likelihood`.
  - Return scalar log-likelihood.
  - No Python exceptions or control flow.

---

## 5. Objective Functions

- `bellman_objective` and `pf_objective`:
  - Call `_compute_total_objective` with appropriate closure.
  - Decorate with `@eqx.filter_jit` where possible.
- `transformed_*`:
  - Untransform params.
  - Call respective base function.

---

## 6. JAX Compatibility

- Remove all Python `try/except` and `if` in objective code.
- Use `jax.lax.cond` for:
  - Priors present or not.
  - Penalty weight > 0 or not.
  - Cholesky success or failure.
- Guarantees **full JAX compatibility and JIT-ability**.

---

## Summary

- Clean, unified, constraint-enforcing, fully JAX-compatible objective functions.
- Consistent API for Bellman and Particle filters.
- Easier maintenance and extension.

---

## Next Step

Switch to implementation mode and refactor accordingly.