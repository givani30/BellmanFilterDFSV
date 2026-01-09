# OPTIMIZATION MODULE

## OVERVIEW
MLE parameter estimation for DFSV models. Custom BFGS/Trust Region solvers + parameter transformations for constrained optimization.

## STRUCTURE

```
optimization/
├── optimization.py          # run_optimization(), minimize_with_logging
├── solvers.py               # DogLegBFGS, ArmijoBFGS, DampedTrustRegionBFGS
├── objectives.py            # Objective function wrappers
├── transformations.py       # Constrained ↔ unconstrained parameter transforms
└── optimization_helpers.py  # Logging, history tracking utilities
```

## WHERE TO LOOK

| Task | File | Function |
|------|------|----------|
| Run MLE optimization | `optimization.py` | `run_optimization()` |
| Add custom solver | `solvers.py` | Inherit from Optimistix base |
| Transform parameters | `transformations.py` | `transform_params()`, `untransform_params()` |
| Create objective | `objectives.py` | Wrap filter's `log_likelihood_wrt_params` |
| Track optimization history | `optimization_helpers.py` | History dataclass |

## CONVENTIONS

### Objective Functions
- **MUST be JIT-compatible**: No Python side effects in the objective
- **Return negative log-likelihood**: Optimizers minimize, likelihoods maximize
- **Use transformations**: Map constrained params to unconstrained space

### Parameter Transformations
```python
# Constrained → Unconstrained (for optimizer)
Phi (|eigenvalues| < 1)  →  arctanh of eigenvalues
sigma2 (> 0)             →  log(sigma2)
Q_h (PSD)                →  Cholesky factor (lower triangular)

# Use safe_arctanh to avoid ±Inf at boundaries
```

### Solver Selection
| Solver | Use When |
|--------|----------|
| `optx.BFGS` | Default choice, good convergence |
| `DogLegBFGS` | Trust region needed |
| `ArmijoBFGS` | Line search issues |

## ANTI-PATTERNS

| Pattern | Why Forbidden |
|---------|---------------|
| Capture loop vars in lambda | Late-binding breaks JAX tracing (see line 132) |
| `verbose=True` in production | Disables `lax_while`, 10x performance hit |
| Skip JIT on objective | Unacceptable performance for MLE |
| Use bare `mu` with `fix_mu` | Must provide `true_mu` or raises ValueError |

## NOTES

### Performance
- First optimization call compiles the objective (~30s)
- Subsequent calls are fast (~ms per iteration)
- Use `use_lax_while=True` for best performance

### TODO (from codebase)
- Line 96: "Need to save all parameters instead of interpolation, this is wrong"
