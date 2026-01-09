# Changelog

All notable changes to BellmanFilterDFSV will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-01-09

### 🚨 Breaking Changes

v2.0.0 is a **complete rewrite** of the package with a new architecture. The v1.x API is not compatible.

#### Migration Required

- **Parameters**: `DFSVParamsDataclass(N, K, ...)` → `DFSVParams(lambda_r, ...)`
- **Filters**: `DFSVBellmanInformationFilter(N, K)` → `BellmanFilter(params)`
- **Filter Call**: `filter.filter(params, data)` → `filter.filter(data)`
- **Simulation**: `simulate_DFSV(params, T, seed)` → `simulate_dfsv(params, T, key=seed)`
- **Optimization**: `run_optimization(...)` → `fit_mle(...)` or `fit_em(...)`

See `examples/README.md` for detailed migration guide.

### Added

#### Core Architecture
- **Functional Core + Equinox**: Complete rewrite using functional patterns with Equinox modules
- **Type Safety**: Full `jaxtyping` annotations on all array operations (0 type errors)
- **NamedTuples**: Immutable parameter and state containers (`DFSVParams`, `BIFState`, `FilterResult`, etc.)

#### New Algorithms
- **RTS Smoother**: Direct Information Form implementation (`rts_smoother`)
- **Rao-Blackwellized Particle Smoother**: Analytically marginalizes linear states (`run_rbps`)
- **EM Algorithm**: Complete EM implementation with RBPS E-step (`fit_em`)

#### New Modules
- `simulation.py`: JAX-based `simulate_dfsv()` function
- `estimation.py`: `fit_mle()` and `fit_em()` with Optax integration
- `smoothing.py`: `rts_smoother()` and `run_rbps()`
- `types.py`: Strict type definitions for all data structures

#### Examples
- 5 new comprehensive examples showcasing v2.0.0 features
- `05_particle_cloud.py`: Visualizes "uncertainty collapse" phenomenon
- All examples updated with v2 API

#### Testing
- **93% test coverage** (up from ~50% in v1.x)
- 69 tests (68 passing, 1 skipped)
- 7 property-based tests using Hypothesis
- Consolidated test suite from 16 files → 5 focused files

#### CI/CD
- Enabled basedpyright type checking in CI
- Added pre-commit hooks configuration
- All quality gates passing (ruff, basedpyright, pytest)

### Changed

#### Architecture
- **Bellman Filter**: Now uses information form exclusively (removed covariance form)
- **State Representation**: Information matrices (`Ω`) instead of covariances (`P`)
- **Parameter Storage**: NamedTuples replace dataclasses for immutability
- **Filter Interface**: Equinox modules replace OOP class hierarchy

#### Performance
- Full JIT compilation support (no `verbose` flag breaking JIT)
- Vectorized operations throughout
- Eliminated intermediate array allocations

#### Dependencies
- Updated to latest stable versions:
  - JAX/JAXlib: 0.4.35
  - Jaxtyping: 0.2.34
  - Equinox: 0.11.8
  - Optax: 0.2.4
  - NumPy: 1.26.0
  - SciPy: 1.14.0

### Removed

#### v1 Architecture
- Removed entire `src/bellman_filter_dfsv/core/` directory
- Removed OOP filter classes (`DFSVFilter`, `DFSVBellmanFilter`, etc.)
- Removed v1 parameter classes (`DFSVParamsDataclass`)
- Removed legacy optimization framework (`run_optimization`)

#### Examples
- Removed v1 examples (04-06) - replaced with v2 equivalents

### Fixed

- **Type Safety**: Eliminated all type errors (0 errors with basedpyright)
- **Numerical Stability**: Direct Information Form in RTS Smoother avoids double inversions
- **Particle Filter**: Fixed x64 precision violations

### Documentation

- Updated README.md with v2 Quick Start and examples
- Updated examples/README.md with comprehensive v2 usage guide
- Added migration guide (v1 → v2)
- Updated all docstrings to reflect v2 API

---

## [1.x] - Legacy

v1.x releases used OOP architecture with class-based filters. See git tags for historical versions.

### v1 → v2 Migration Summary

| Component | v1.x | v2.0.0 |
|-----------|------|--------|
| Architecture | OOP class hierarchy | Functional Core + Equinox |
| Type System | Partial annotations | Full jaxtyping (0 errors) |
| State Representation | Covariance form | Information form |
| Parameters | Mutable dataclass | Immutable NamedTuple |
| Optimization | Custom solvers | Optax integration |
| Test Coverage | ~50% | 93% |

For detailed migration instructions, see `examples/README.md`.

---

## Links

- [GitHub Repository](https://github.com/givani30/BellmanFilterDFSV)
- [Documentation](https://givani30.github.io/BellmanFilterDFSV/)
- [PyPI Package](https://pypi.org/project/bellman-filter-dfsv/) *(coming soon)*
