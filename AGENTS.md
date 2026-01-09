# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-08
**Commit:** 38d94d8
**Branch:** main

## OVERVIEW

JAX-based filtering library for Dynamic Factor Stochastic Volatility (DFSV) models. Implements Bellman Information Filter (BIF), Bellman Filter, and Particle Filter with automatic differentiation for MLE parameter estimation.

## STRUCTURE

```
BellmanFilterDFSV/
├── src/bellman_filter_dfsv/
│   ├── core/
│   │   ├── filters/         # BIF, Bellman, Particle filter implementations
│   │   ├── models/          # DFSV parameter definitions, simulation, likelihoods
│   │   └── optimization/    # MLE solvers, objective functions, transformations
│   └── utils/               # JAX helpers, analysis utilities
├── tests/                   # pytest suite with factory fixtures
├── examples/                # Numbered scripts 01-06: simulation → real data
└── docs/                    # Sphinx documentation
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Create/modify filter | `core/filters/` | Inherit `DFSVFilter`, implement `predict`/`update`/`filter` |
| Add model parameter | `core/models/dfsv.py` | Use `@jdc.pytree_dataclass`, mark dimensions as `Static` |
| Optimize parameters | `core/optimization/` | Use `run_optimization()`, ensure JIT-compatible objective |
| Simulate data | `core/models/simulation.py` | `simulate_DFSV(params, T, seed)` |
| Compute likelihood | `core/models/likelihoods.py` | JAX-compatible, uses Woodbury identity |
| Add test | `tests/` | Use factory fixtures from `conftest.py` |
| Run example | `examples/0N_*.py` | Numbered progression: simulation → optimization → real data |

## CODE MAP

### Core Filters (Inheritance Hierarchy)
```
DFSVFilter (base.py)                    # Abstract base: predict/update/smooth interface
├── DFSVBellmanFilter (bellman.py)      # Covariance-form, block coordinate descent
├── DFSVBellmanInformationFilter        # Information-form (precision matrix), more stable
└── DFSVParticleFilter (particle.py)    # Bootstrap SISR for non-Gaussian cases
```

### Key Implementation Files
| File | Purpose | Lines |
|------|---------|-------|
| `bellman_information.py` | BIF filter (recommended) | 1343 |
| `bellman.py` | Standard Bellman filter | 1250 |
| `particle.py` | Bootstrap particle filter | 1094 |
| `_bellman_impl.py` | FIM, Woodbury, BIF penalty math | 451 |
| `_bellman_optim.py` | Block coordinate descent impl | 487 |
| `optimization.py` | MLE wrapper, logging, history | 761 |
| `solvers.py` | Custom BFGS, Trust Region | 521 |

## CONVENTIONS

### JAX Patterns (MUST follow)
- **64-bit precision**: `jax.config.update("jax_enable_x64", True)` — always enabled
- **JIT via Equinox**: Use `@eqx.filter_jit`, NOT raw `jax.jit`
- **Time loops**: Use `jax.lax.scan` for filtering sequences
- **Iterations**: Use `jax.lax.fori_loop` or `while_loop` for JIT-compatible loops
- **Vectorization**: Use `jax.vmap` for batch operations
- **Parameters**: Wrap in `@jdc.pytree_dataclass`, mark `N`/`K` as `jdc.Static[int]`

### Numerical Stability (CRITICAL)
- **Symmetrize matrices**: Always `(P + P.T) / 2` after covariance/information updates
- **Jitter before inversion**: Add small diagonal (`1e-6`) before Cholesky
- **Fallback to pinv**: Use pseudo-inverse if singular
- **safe_arctanh**: Clip inputs to avoid ±Inf at boundaries

### Code Organization
- **API classes** in main files (`bellman.py`, `particle.py`)
- **Math kernels** in `_impl.py` or `_optim.py` files
- **Public exports** via `__init__.py` at each level

## ANTI-PATTERNS (THIS PROJECT)

| Pattern | Why Forbidden |
|---------|---------------|
| Capture loop vars in lambdas | Late-binding breaks JAX tracing |
| Non-JAX objects in JIT | Must use JAX arrays in `lax.scan` |
| NumPy in core filters | Use `jnp`, not `np` (legacy helpers deprecated) |
| Skip matrix symmetrization | Causes numerical drift in covariance |
| `verbose=True` in optimization | Disables `lax_while`, 10x slower |
| Bare `mu` without `fix_mu` check | Raises ValueError if `true_mu` not provided |

## UNIQUE STYLES

### Test Pattern: Factory Fixtures
```python
# conftest.py provides factories, not static fixtures
@pytest.fixture(scope="session")
def params_fixture() -> Callable[..., DFSVParamsDataclass]:
    def _create_params(N: int = 4, K: int = 2) -> DFSVParamsDataclass:
        ...
    return _create_params

# Usage in test:
def test_filter(params_fixture, data_fixture):
    params = params_fixture(N=3, K=1)  # Call the factory
```

### BIF Pseudo-Likelihood
The Bellman Information Filter adds a KL-divergence penalty to the likelihood for stability. See `_bellman_impl.py:bif_likelihood_penalty_impl`.

### DFSV Model Equations
```
Observation:    r_t = λ_r f_t + e_t,  e_t ~ N(0, Σ)
Factor:         f_t = Φ_f f_{t-1} + diag(exp(h_t/2)) ε_t
Log-vol:        h_t = μ + Φ_h (h_{t-1} - μ) + η_t
```

## COMMANDS

```bash
# Environment
uv sync                              # Install dependencies
uv sync --extra dev                  # Include dev tools

# Testing
uv run pytest                        # Run all tests
uv run pytest --cov=bellman_filter_dfsv  # With coverage
uv run pytest tests/test_bellman.py  # Specific file

# Linting
uv run ruff check .                  # Lint
uv run ruff format .                 # Format

# Documentation
cd docs && make html                 # Build Sphinx docs

# Examples
uv run python examples/01_dfsv_simulation.py
uv run python examples/02_basic_filtering.py
```

## NOTES

### Known Issues
- **mypy disabled**: 90+ pre-existing type errors (JAX arrays vs ndarray). CI skips type checking.
- **BF filter unstable**: Standard Bellman Filter less stable than BIF during optimization. Prefer `DFSVBellmanInformationFilter`.
- **Stationarity not enforced**: Prior on Φ_f/Φ_h doesn't enforce eigenvalues < 1.

### Performance Tips
- First JIT call is slow (XLA compilation). Precompile with dummy data if latency matters.
- `verbose=True` in optimization ignores `use_lax_while` — significantly slower.
- Particle filter scales with `num_particles` — use 1000+ for production, fewer for testing.

### Research Context
This library was developed for a quantitative finance thesis. Research artifacts (Monte Carlo studies, experimental code) are in a separate repository: [BellmanFilterDFSV-ThesisResearch](https://github.com/givani30/BellmanFilterDFSV-ThesisResearch).

### Planned Feature: EM Algorithm
Design document at `docs/design/EM_ALGORITHM_DESIGN.md` specifies an EM algorithm as an alternative to direct MLE. Uses BIF + RTS smoother for E-step with closed-form M-step updates.
