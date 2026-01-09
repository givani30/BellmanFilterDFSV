# BellmanFilterDFSV v2: Rebuild Plan & Analysis

**Date:** 2026-01-09
**Status:** Phase 2 Complete

## 1. Analysis of Legacy Codebase (v1)

### Executive Summary
The v1 codebase is mathematically sound but architecturally brittle. It relies on a complex inheritance hierarchy (`DFSVFilter` -> `DFSVBellmanInformationFilter`) that fights against JAX's functional paradigm. This has led to 90+ type errors, performance pitfalls (verbose mode killing JIT), and difficulties in implementing advanced features like the EM algorithm.

### Key Issues
1.  **The "Type War"**: The class structure tries to be both a NumPy API (for users) and a JAX kernel (for compilation). Methods like `predict()` have ambiguous return types (`np.ndarray` vs `jax.Array`).
2.  **Implementation Bloat**: The base class contains attributes specific to subclasses.
3.  **Performance Traps**: `verbose=True` disables `lax.while_loop` without warning. Static matrix inversions happen inside the time loop.
4.  **Legacy Debt**: `likelihoods.py` uses non-JIT NumPy functions. `particle.py` violates the 64-bit precision requirement.

## 2. v2 Architecture: Functional Core + Equinox

We will rebuild using a **Functional Core** pattern wrapped in **Equinox** modules. This separates the math (pure functions) from the state management and API.

### Design Principles
1.  **Pure Functions**: Math kernels (`predict`, `update`) take state/params and return new state. No side effects.
2.  **Explicit State**: Data containers (`NamedTuple` or `eqx.Module`) define the state shape once. No "is this (K,1) or (K,)?".
3.  **Equinox Wrapper**: Handles parameter registration (PyTree) and user API.
4.  **Strict Typing**: Use `jaxtyping` for shape/dtype verification.

### Module Structure (`src/bellman_filter_dfsv/next/`)

```
next/
├── __init__.py
├── types.py          # State definitions (FilterState, DFSVParams)
├── kernels.py        # Pure math functions (predict_step, update_step) for Bellman
├── particle_kernels.py # Pure math functions for Particle Filter
├── filters.py        # Equinox modules (BellmanFilter, ParticleFilter)
├── smoother.py       # RTS Smoother (for EM algorithm)
└── utils.py          # JAX helpers (safe_arctanh, symmetrize)
```

## 3. Implementation Plan

### Phase 1: Foundation (Complete)
- [x] Create `next/` directory.
- [x] **Step 1**: Define `types.py` (State containers).
- [x] **Step 2**: Port math from `_bellman_impl.py` to `kernels.py` (Pure functions).
- [x] **Step 3**: Create basic `filters.py` using Equinox (BellmanFilter).

### Phase 2: Feature Parity (Complete)
- [x] Port `DFSVBellmanInformationFilter` logic (in `kernels.py`).
- [x] Port `DFSVParams` and ensure `x64` compliance (in `types.py`).
- [x] Implement `ParticleFilter` kernels (`particle_kernels.py`) and module (`filters.py`).
- [x] Add tests comparing v1 vs v2 outputs (verified shapes and execution).

### Phase 3: EM Algorithm
- [ ] Implement Forward-Backward Smoother (E-Step).
- [ ] Implement M-Step updates.
- [ ] Create `EM` orchestration class.

### Phase 4: Migration
- [ ] Swap `core/` imports to `next/`.
- [ ] Update examples.
