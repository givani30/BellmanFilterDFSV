# EM Algorithm Implementation Plan

**Branch**: `feature/em-algorithm`  
**Created**: 2026-01-09  
**Status**: Phase 0 - Planning  

---

## Overview

This document tracks the phased implementation of the EM algorithm for DFSV parameter estimation. The approach prioritizes **verification at every step** due to the mathematical complexity and the untested RTS smoother.

### Risk Assessment

| Component | Risk Level | Concern |
|:----------|:-----------|:--------|
| M-step formulas | LOW | SymPy verified, unit tested |
| E-step accumulation | MEDIUM | New code, needs careful indexing |
| RTS Smoother | HIGH | Built but never used in production; lag-1 covariance logic unverified |
| Full EM convergence | MEDIUM | Depends on all components working together |

---

## Phase 0: Verify RTS Smoother (BLOCKING)

**Goal**: Ensure the smoother returns mathematically correct results before building on top of it.

### Tasks

- [x] **0.1** Review `base.py` smoother implementation (lines 544-763)
- [x] **0.2** Verify lag-1 covariance formula: `P_{t+1,t|T} = P_{t+1|T} @ J_t.T`
- [x] **0.3** Create test: smoother on LINEAR Gaussian system (known analytical solution)
- [x] **0.4** Create test: smoother MSE < filter MSE (fundamental property)
- [x] **0.5** Create test: boundary conditions (t=0, t=T-1) are correct
- [x] **0.6** Create test: lag-1 covariances satisfy symmetry `P_{t+1,t}' = P_{t,t+1}`

### Acceptance Criteria
- All smoother tests pass
- Lag-1 covariances match analytical formulas on simple AR(1) system

### Files to Create/Modify
- `tests/test_rts_smoother_academic.py` (extend existing)

---

## Phase 1: E-Step - Sufficient Statistics Accumulation

**Goal**: Implement the bridge between smoother output and `EMSufficientStats`.

### Tasks

- [x] **1.1** Create `_em_estep.py::accumulate_sufficient_stats()` function
- [x] **1.2** Handle state decomposition: extract `f_t`, `h_t` from joint state
- [x] **1.3** Compute second moments: `E[f_t f_t'] = f_t f_t' + P_ff`
- [x] **1.4** Compute cross-lag moments: `E[f_t f_{t-1}'] = f_t f_{t-1}' + P_{t,t-1}^{ff}`
- [x] **1.5** Compute `E[exp(-h_t)]` using existing `compute_exp_neg_h()`
- [x] **1.6** Use `jax.lax.scan` for efficient accumulation over time

### Tests

- [ ] **T1.1** Given known smoothed states, verify `sum_r_f` computation
- [ ] **T1.2** Given known smoothed states, verify `sum_f_f` includes covariance term
- [ ] **T1.3** Verify `sum_h_hprev` uses lag-1 covariance correctly
- [ ] **T1.4** Test `E[exp(-h)]` matches Monte Carlo estimate (existing test)
- [ ] **T1.5** Edge case: T=2 (minimal dynamics case)

### Acceptance Criteria
- All sufficient statistics computed correctly from mock smoother output
- Shapes match `EMSufficientStats` specification

### Files to Create/Modify
- `src/.../optimization/_em_estep.py` (extend)
- `tests/test_em.py` (extend with E-step tests)

---

## Phase 2: M-Step - Parameter Update Integration

**Goal**: Verify M-step updates work correctly when called in sequence.

### Tasks

- [x] **2.1** Create `_em_mstep.py::m_step_full()` that calls all updates in correct order
- [x] **2.2** Handle μ/Φ_h coupling (iterate or use ECM)
- [x] **2.3** Enforce constraints: eigenvalues < 1, variances > 0
- [x] **2.4** Return updated `DFSVParamsDataclass`

### Tests

- [ ] **T2.1** Given synthetic stats from known params, M-step recovers params
- [ ] **T2.2** Constraint enforcement: Φ values clipped to (-0.999, 0.999)
- [ ] **T2.3** Constraint enforcement: σ², Q_h bounded away from 0
- [ ] **T2.4** μ/Φ_h coupling: joint update converges

### Acceptance Criteria
- Single M-step from known sufficient stats recovers true parameters (within tolerance)

### Files to Create/Modify
- `src/.../optimization/_em_mstep.py` (extend with `m_step_full`)
- `tests/test_em.py` (extend)

---

## Phase 3: EMOptimizer Class

**Goal**: Create the main orchestrator class.

### Tasks

- [x] **3.1** Create `em.py` with `EMOptimizer` class skeleton
- [x] **3.2** Implement `__init__(N, K, max_iters, tol, verbose)`
- [x] **3.3** Implement `e_step(params, observations)` → `EMSufficientStats`
- [x] **3.4** Implement `m_step(stats, current_params)` → `DFSVParamsDataclass`
- [x] **3.5** Implement `fit(observations, initial_params)` → `(params, history)`
- [x] **3.6** Create `EMHistory` dataclass for iteration tracking

### Tests

- [ ] **T3.1** Single E-step produces valid `EMSufficientStats`
- [ ] **T3.2** Single M-step produces valid `DFSVParamsDataclass`
- [ ] **T3.3** One full EM iteration increases (or maintains) log-likelihood
- [ ] **T3.4** `fit()` returns after convergence

### Acceptance Criteria
- `EMOptimizer` can run a single iteration without error
- Log-likelihood is computed and stored

### Files to Create/Modify
- `src/.../optimization/em.py` (new)
- `src/.../optimization/__init__.py` (add exports)
- `tests/test_em.py` (extend)

---

## Phase 4: Integration Testing

**Goal**: Verify full EM algorithm on simulated data.

### Tasks

- [x] **4.1** Create `test_em_integration.py` with controlled simulations
- [x] **4.2** Test: EM on data simulated from known params → recovers params
- [x] **4.3** Test: Log-likelihood monotonically increases (EM guarantee)
- [x] **4.4** Test: Convergence within max_iters for well-behaved data
- [x] **4.5** Test: Compare EM estimate vs direct MLE (should be similar)

### Tests

- [ ] **T4.1** Small system (N=3, K=1, T=200): parameter recovery within 10%
- [ ] **T4.2** Medium system (N=10, K=2, T=500): parameter recovery within 15%
- [ ] **T4.3** Monotonicity: `ll[k+1] >= ll[k] - epsilon` for all k
- [ ] **T4.4** Stress test: poorly initialized params still converge

### Acceptance Criteria
- EM converges on simulated data
- Parameters recovered within reasonable tolerance
- Log-likelihood never decreases (within numerical tolerance)

### Files to Create/Modify
- `tests/test_em_integration.py` (new)

---

## Phase 5: Documentation & Cleanup

**Goal**: Production-ready code.

### Tasks

- [ ] **5.1** Add docstrings to all public functions
- [ ] **5.2** Update `EM_ALGORITHM_DESIGN.md` with final implementation notes
- [ ] **5.3** Add usage example in `examples/` directory
- [ ] **5.4** Run full test suite, ensure no regressions
- [ ] **5.5** Update `__init__.py` exports

### Acceptance Criteria
- `uv run pytest` passes all tests
- Example script runs without error

---

## Test Coverage Requirements

| Component | Minimum Tests | Focus |
|:----------|:--------------|:------|
| Smoother lag-1 cov | 4 | Mathematical correctness |
| E-step accumulation | 5 | Indexing, shapes, edge cases |
| M-step sequence | 4 | Recovery, constraints |
| EMOptimizer | 4 | Single iter, convergence |
| Integration | 4 | Full loop, monotonicity |

**Total**: ~21 new tests minimum

---

## Implementation Order

```
Phase 0 (Smoother Verification)
    ↓
Phase 1 (E-step)
    ↓
Phase 2 (M-step integration)
    ↓
Phase 3 (EMOptimizer)
    ↓
Phase 4 (Integration tests)
    ↓
Phase 5 (Docs & cleanup)
```

**Critical Path**: Phase 0 MUST complete before Phase 1. The smoother is the foundation.

---

## Current Progress

| Phase | Status | Notes |
|:------|:-------|:------|
| Phase 0 | COMPLETED | Verified in `scripts/experiments/debug_kalman.py` |
| Phase 1 | COMPLETED | Implemented in `scripts/experiments/rbps_em_lib.py` |
| Phase 2 | COMPLETED | Verified in `scripts/experiments/exp08_rbps_em.py` |
| Phase 3 | PROTOTYPE | `run_particle_em` in `exp08_rbps_em.py` |
| Phase 4 | PROTOTYPE | `exp08_rbps_em.py` confirms convergence |
| Phase 5 | NOT STARTED | Need to port to src/ |

---

## Commands

```bash
# Run all EM tests
uv run pytest tests/test_em.py tests/test_rts_smoother_academic.py -v

# Run with coverage
uv run pytest tests/test_em.py --cov=bellman_filter_dfsv.core.optimization

# Run specific phase tests
uv run pytest tests/test_rts_smoother_academic.py -v  # Phase 0
uv run pytest tests/test_em.py -k "estep" -v          # Phase 1
uv run pytest tests/test_em.py -k "mstep" -v          # Phase 2
```

---

*Last updated: 2026-01-09*
