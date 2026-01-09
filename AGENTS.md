# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-09
**Commit:** a40769f
**Branch:** main

## OVERVIEW

JAX-based filtering library for Dynamic Factor Stochastic Volatility (DFSV) models. Implements Bellman Information Filter (BIF), Bellman Filter, and Particle Filter with automatic differentiation for MLE parameter estimation.

## STRUCTURE

```
BellmanFilterDFSV/
├── src/bellman_filter_dfsv/
│   ├── next/                # [v2] NEW ARCHITECTURE (Use this!)
│   │   ├── filters.py       # Equinox modules for BIF and Particle Filter
│   │   ├── kernels.py       # Pure math kernels for BIF
│   │   ├── particle_kernels.py # Pure math kernels for PF
│   │   ├── optimization.py  # fit_mle utility
│   │   ├── smoother.py      # RTS Smoother (Direct Information Form)
│   │   └── types.py         # Strict NamedTuples for state/params
│   ├── core/                # [v1] LEGACY ARCHITECTURE (Deprecated)
│   │   ├── filters/         # Old class hierarchy (DFSVFilter -> DFSVBellmanInformationFilter)
│   │   ├── models/          # Old model definitions
│   │   └── optimization/    # Old MLE solvers
│   └── utils/               # Shared JAX helpers
├── tests/                   # pytest suite
├── examples/                # Numbered scripts 01-06
└── docs/                    # Sphinx documentation
```

## WHERE TO LOOK (v2 Architecture)

| Task | Location (`src/bellman_filter_dfsv/next/`) | Notes |
|------|--------------------------------------------|-------|
| Create filter | `filters.py` | Use `BellmanFilter(params)` or `ParticleFilter(params)` |
| Fit parameters | `optimization.py` | Use `fit_mle(start_params, data)` |
| Smooth states | `smoother.py` | Use `rts_smoother(params, filter_result)` |
| Define params | `types.py` | Use `DFSVParams(lambda_r, ...)` (NamedTuple) |
| Check math | `kernels.py` | Pure JAX functions (stateless) |

## ARCHITECTURE: v2 vs v1

We are migrating from a complex OOP inheritance hierarchy (v1) to a **Functional Core + Equinox** pattern (v2).

| Feature | v2 (New) | v1 (Legacy) |
|---------|----------|-------------|
| **Core Pattern** | Functional Core + Equinox Modules | Abstract Base Class Inheritance |
| **State** | Immutable `NamedTuple` / `eqx.Module` | Mutable class attributes (`self.x`) |
| **Type Safety** | Strict `jaxtyping` (0 errors) | `np.ndarray` vs `jax.Array` ambiguity |
| **Optimization** | `fit_mle` / `optax` integration | Ad-hoc solvers, fragile `run_optimization` |
| **Performance** | Fully JIT-compatible | `verbose=True` kills JIT (100x slowdown) |

## CONVENTIONS (v2)

### JAX & Equinox Patterns
- **Equinox Modules**: All high-level components are `eqx.Module`. Parameters are stored in `self.params`.
- **Pure Functions**: Math logic lives in `kernels.py`. It takes `(state, params)` and returns `new_state`.
- **Strict Typing**: Use `jaxtyping` for ALL array arguments.
  - Correct: `observations: Float[Array, "T N"]`
  - Incorrect: `observations: jnp.ndarray`
- **Zero-Cost Abstraction**: Re-creating `BellmanFilter(params)` inside a JIT loop is free and recommended.

### Numerical Stability
- **Direct Information Form**: The smoother uses the information form gain $J_t = P_{t|t} F^T \Omega_{t+1|t}$ to avoid double inversions.
- **Symmetrization**: Always `0.5 * (A + A.T)` after covariance updates.
- **Jitter**: `1e-6 * I` added before Cholesky decompositions.

## DFSV MODEL EQUATIONS
```
Observation:    r_t = λ_r f_t + e_t,  e_t ~ N(0, Σ)
Factor:         f_t = Φ_f f_{t-1} + diag(exp(h_t/2)) ε_t
Log-vol:        h_t = μ + Φ_h (h_{t-1} - μ) + η_t
```

## COMMANDS

```bash
# Environment
uv sync                              # Install dependencies

# Testing
uv run pytest                        # Run all tests (v1 + v2 parity checks)
uv run pytest tests/test_v2_parity.py # Check v2 architecture specifically

# Documentation
cd docs && make html                 # Build Sphinx docs
```

## NOTES

### Known Issues
- **Legacy Type Errors**: v1 code (`core/`) still has 90+ type errors. Ignore them; focusing on v2 migration.
- **Particle Filter Precision**: v1 `particle.py` violated x64 precision. v2 `particle_kernels.py` is fully x64 compliant.

### Planned Feature: EM Algorithm
The EM algorithm will be implemented on top of the v2 `rts_smoother` and `filters`.
- **E-Step**: `rts_smoother(params, filter_result)`
- **M-Step**: Closed-form updates using smoothed statistics.
