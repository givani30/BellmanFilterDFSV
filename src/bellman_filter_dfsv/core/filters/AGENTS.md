# FILTERS MODULE

## OVERVIEW
Filtering algorithms for DFSV state estimation. BIF recommended for optimization; Particle Filter for non-Gaussian.

## STRUCTURE

```
filters/
├── base.py                  # DFSVFilter ABC, SmootherResults, shared utilities
├── bellman.py               # DFSVBellmanFilter (covariance-form)
├── bellman_information.py   # DFSVBellmanInformationFilter (information-form) ← PREFERRED
├── particle.py              # DFSVParticleFilter (Bootstrap SISR)
├── _bellman_impl.py         # Math kernels: FIM, Woodbury, BIF penalty
└── _bellman_optim.py        # Block coordinate descent implementation
```

## WHERE TO LOOK

| Task | File | Method/Function |
|------|------|-----------------|
| Create new filter | `base.py` | Inherit `DFSVFilter`, implement abstract methods |
| Modify BIF update | `bellman_information.py` | `_update_joint_state()` |
| Change FIM calculation | `_bellman_impl.py` | `observed_fim_impl`, `expected_fim_impl` |
| Tune BCD iterations | `_bellman_optim.py` | `_block_coordinate_update_impl` |
| Add smoothing logic | `base.py` | `smooth()` uses RTS backward pass |

## CONVENTIONS

### Filter Interface (MUST implement)
```python
class MyFilter(DFSVFilter):
    def initialize_state(self, params) -> tuple[mean, cov]: ...
    def predict(self, ...) -> tuple[pred_mean, pred_cov]: ...
    def update(self, ...) -> tuple[post_mean, post_cov, log_lik]: ...
    def filter(self, params, y) -> tuple[states, covs, total_ll]: ...
    def jit_log_likelihood_wrt_params(self) -> Callable: ...
```

### File Naming
- **Public API**: `{name}.py` (e.g., `bellman.py`)
- **Internal kernels**: `_{name}_impl.py` or `_{name}_optim.py`

### State Representation
- **Bellman**: Covariance form `(x, P)` where `P` is covariance matrix
- **BIF**: Information form `(ξ, Ω)` where `Ω = P⁻¹` is precision matrix
- **Particle**: Weighted particles `(particles, log_weights)`

## ANTI-PATTERNS

| Pattern | Why Forbidden |
|---------|---------------|
| Skip `(P + P.T) / 2` | Covariance drift causes NaN |
| Direct matrix inverse | Use Cholesky + solve, or add jitter |
| Return NumPy from filter | Keep as JAX arrays until final getter |
| Non-static N/K in JIT | Mark dimensions as `jdc.Static` |

## COMPLEXITY NOTES

| File | Lines | Hotspots |
|------|-------|----------|
| `bellman_information.py` | 1343 | `_update_joint_state` (459 lines nested) |
| `bellman.py` | 1250 | Block coordinate update loop |
| `particle.py` | 1094 | Resampling + vmap over particles |

Refactoring opportunity: Extract nested JAX logic from `bellman_information.py` into `_bellman_impl.py`.
