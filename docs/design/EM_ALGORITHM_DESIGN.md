# EM Algorithm Design for DFSV Parameter Estimation

**Status**: Design Document  
**Author**: AI-assisted design session  
**Date**: 2026-01-08  
**Related**: Thesis Section 6.5, Lange (2024), Shumway & Stoffer (1982)

## Next Steps (Resume Here)

When resuming implementation:

1. **SymPy Verification** (~3h) — Verify all 6 M-step closed-forms symbolically
   - Start with `λ_r` and `σ²` (easiest)
   - Then `μ`, `Φ_h`, `Q_h` (log-vol block)
   - Finally `Φ_f` (weighted regression with independence approximation)

2. **Phase 1: Observation Params** (~12h) — λ_r, σ² only
   - Create `_em_suffstats.py` with dataclass
   - Implement E-step using existing BIF + smoother
   - Implement M-step for observation equation
   - Test: EM should match direct MLE for λ_r, σ²

3. **Phase 2: Log-Vol Params** (~10h) — μ, Φ_h, Q_h
   - Extend sufficient stats
   - Add M-step updates (coupled μ/Φ_h needs care)

4. **Phase 3: Factor Dynamics** (~12h) — Φ_f (hardest)
   - Weighted least squares with E[exp(-h)]
   - Validate independence approximation

5. **Phase 4: Integration** (~8h) — EMOptimizer class, tests, docs

**Command to run existing smoother** (verify it works):
```python
from bellman_filter_dfsv.core.filters import DFSVBellmanInformationFilter
bif = DFSVBellmanInformationFilter(N=10, K=2)
states, covs, ll = bif.filter(params, observations)
smooth_states, smooth_covs, lag1_covs = bif.smooth(params)  # Check this returns lag-1 covs
```

---

## Overview

This document specifies the design for an Expectation-Maximization (EM) algorithm as an **alternative** to direct pseudo-likelihood maximization for DFSV parameter estimation.

### Motivation

From the thesis (Section 6.5, p.58):
> "The BIF, with its efficient filter scaling, could potentially be used effectively within the E-step to compute expected sufficient statistics, possibly offering computational advantages for parameter estimation compared to the direct optimization approach."

### Use Case: Market Engine Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Market Engine Pipeline                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Market Data (r_t)                                                      │
│         │                                                                │
│         ▼                                                                │
│   ┌─────────────────────┐                                                │
│   │  BellmanFilterDFSV  │  ← EM for robust parameter estimation          │
│   │  (This Library)     │                                                │
│   └─────────────────────┘                                                │
│         │                                                                │
│         ▼                                                                │
│   Latent States: f_t (factors), h_t (log-volatilities)                   │
│         │                                                                │
│         ▼                                                                │
│   ┌─────────────────────┐                                                │
│   │  Downstream Model   │  (e.g., ETF Agent, Portfolio Optimization)     │
│   └─────────────────────┘                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## DFSV Model Specification

### State-Space Representation

**Observation Equation:**
```
r_t = λ_r f_t + e_t,    e_t ~ N(0, Σ)
```
where Σ = diag(σ²) is diagonal with idiosyncratic variances.

**Factor Dynamics:**
```
f_t = Φ_f f_{t-1} + diag(exp(h_t/2)) ε_t,    ε_t ~ N(0, I_K)
```

**Log-Volatility Dynamics:**
```
h_t = μ + Φ_h (h_{t-1} - μ) + η_t,    η_t ~ N(0, Q_h)
```

### Parameters to Estimate

| Parameter | Shape | Description | Constraints |
|-----------|-------|-------------|-------------|
| λ_r | (N, K) | Factor loadings | First row positive (identification) |
| σ² | (N,) | Idiosyncratic variances | > 0 |
| Φ_f | (K, K) | Factor AR matrix | Diagonal, \|eigenvalues\| < 1 |
| Φ_h | (K, K) | Log-vol AR matrix | Diagonal, \|eigenvalues\| < 1 |
| μ | (K,) | Long-run log-vol mean | Unconstrained |
| Q_h | (K, K) | Log-vol innovation cov | Positive definite |

**Assumption**: Φ_f and Φ_h are **diagonal** matrices. This simplifies the M-step significantly and is standard in factor stochastic volatility models.

---

## Algorithm: BIF-EM (Gaussian Approximation EM)

### High-Level Flow

```
Algorithm: BIF-EM for DFSV
─────────────────────────────────────────────────────────────────────────

Input: observations r_{1:T}, initial parameters θ⁽⁰⁾, tolerance ε, max_iters
Output: estimated parameters θ̂, log-likelihood history

1. Initialize θ = θ⁽⁰⁾
2. For k = 1, 2, ... until convergence:

   ┌─ E-Step ──────────────────────────────────────────────────────────┐
   │ a) Run BIF forward filter: p(x_t | r_{1:t}) for t = 1,...,T       │
   │ b) Run RTS backward smoother: p(x_t | r_{1:T}) for t = T,...,1    │
   │ c) Compute sufficient statistics S from smoothed posterior        │
   │    - Uses Gaussian moments and log-normal moment for E[exp(-h)]   │
   └───────────────────────────────────────────────────────────────────┘

   ┌─ M-Step ──────────────────────────────────────────────────────────┐
   │ a) Update observation parameters: λ_r, σ²                         │
   │ b) Update factor dynamics: Φ_f (weighted regression)              │
   │ c) Update log-vol dynamics: μ, Φ_h, Q_h                           │
   │    - All updates have closed-form solutions                       │
   └───────────────────────────────────────────────────────────────────┘

   d) Compute log-likelihood ℓ⁽ᵏ⁾ = log p(r_{1:T} | θ⁽ᵏ⁾)
   e) Check convergence: |ℓ⁽ᵏ⁾ - ℓ⁽ᵏ⁻¹⁾| < ε

3. Return θ̂ = θ⁽ᵏ⁾
```

---

## E-Step: Sufficient Statistics

### State Decomposition

The joint state is x_t = [f_t', h_t']' with dimension 2K.

From the RTS smoother, we obtain:
- `x_{t|T}` = E[x_t | r_{1:T}]  — smoothed state mean
- `P_{t|T}` = Cov[x_t | r_{1:T}]  — smoothed state covariance  
- `P_{t,t-1|T}` = Cov[x_t, x_{t-1} | r_{1:T}]  — lag-1 cross-covariance

### Sufficient Statistics Dataclass

```python
@jdc.pytree_dataclass
class DFSVSufficientStats:
    """Sufficient statistics for DFSV EM algorithm.
    
    All expectations are conditional on the full observation sequence r_{1:T}.
    
    Notation:
        E[·] denotes E[· | r_{1:T}, θ_old]
        f_t ∈ R^K: factors
        h_t ∈ R^K: log-volatilities
        r_t ∈ R^N: observations
    """
    
    # === Observation Equation Statistics ===
    # For updating λ_r and σ²
    
    sum_r_f: Float[Array, "N K"]       # Σ_{t=1}^T r_t E[f_t]'
    sum_f_f: Float[Array, "K K"]       # Σ_{t=1}^T E[f_t f_t']
    sum_r_r_diag: Float[Array, "N"]    # Σ_{t=1}^T r_t ⊙ r_t (element-wise)
    
    # === Factor Dynamics Statistics ===
    # For updating Φ_f (diagonal)
    
    sum_f_fprev: Float[Array, "K K"]       # Σ_{t=2}^T E[f_t f_{t-1}']
    sum_fprev_fprev: Float[Array, "K K"]   # Σ_{t=2}^T E[f_{t-1} f_{t-1}']
    sum_exp_neg_h: Float[Array, "K"]       # Σ_{t=2}^T E[exp(-h_t)]
    sum_exp_neg_h_f_fprev: Float[Array, "K"]   # Σ_{t=2}^T E[exp(-h_t)] ⊙ E[f_t f_{t-1}] (diagonal)
    sum_exp_neg_h_fprev_sq: Float[Array, "K"]  # Σ_{t=2}^T E[exp(-h_t)] ⊙ E[f_{t-1}²] (diagonal)
    
    # === Log-Volatility Dynamics Statistics ===
    # For updating μ, Φ_h, Q_h
    
    sum_h: Float[Array, "K"]               # Σ_{t=1}^T E[h_t]
    sum_h_hprev: Float[Array, "K K"]       # Σ_{t=2}^T E[h_t h_{t-1}']
    sum_hprev_hprev: Float[Array, "K K"]   # Σ_{t=2}^T E[h_{t-1} h_{t-1}']
    sum_h_h: Float[Array, "K K"]           # Σ_{t=1}^T E[h_t h_t']
    sum_hprev: Float[Array, "K"]           # Σ_{t=2}^T E[h_{t-1}]
    
    # === Counts ===
    T: jdc.Static[int]  # Total number of time steps
```

### Computing E[f_t f_t'] from Smoothed Posterior

For Gaussian posterior:
```
E[f_t f_t' | r_{1:T}] = E[f_t] E[f_t]' + Cov[f_t | r_{1:T}]
                      = f_{t|T} f_{t|T}' + P_{t|T}^{ff}
```

where P_{t|T}^{ff} is the (K × K) upper-left block of P_{t|T}.

### Computing E[exp(-h_t)] from Smoothed Posterior

Since h_t | r_{1:T} ~ N(h_{t|T}, P_{t|T}^{hh}), and for a Gaussian random variable X ~ N(μ, σ²):
```
E[exp(aX)] = exp(aμ + a²σ²/2)
```

Therefore:
```
E[exp(-h_{kt})] = exp(-h_{kt|T} + ½ P_{t|T}^{hh}[k,k])
```

**Numerical stability**: Cap the variance term to prevent explosion:
```python
h_var_capped = jnp.minimum(P_hh_diag, 4.0)  # Cap at 4.0
E_exp_neg_h = jnp.exp(-h_mean + 0.5 * h_var_capped)
```

---

## M-Step: Closed-Form Updates

All derivations assume the E-step provides exact sufficient statistics. The updates maximize the expected complete-data log-likelihood Q(θ | θ_old).

### Complete-Data Log-Likelihood

```
log p(r_{1:T}, f_{1:T}, h_{1:T} | θ) = 
    Σ_t log p(r_t | f_t, λ_r, σ²)           [Observation]
  + Σ_t log p(f_t | f_{t-1}, h_t, Φ_f)      [Factor transition]
  + Σ_t log p(h_t | h_{t-1}, μ, Φ_h, Q_h)   [Log-vol transition]
  + log p(f_0, h_0)                          [Initial state]
```

### Update 1: Factor Loadings (λ_r)

**Objective**: Maximize w.r.t. λ_r:
```
Q_obs = -½ Σ_t E[(r_t - λ_r f_t)' Σ⁻¹ (r_t - λ_r f_t)]
```

**Closed-form solution** (matrix regression):
```
λ_r^new = (Σ_t r_t E[f_t]') (Σ_t E[f_t f_t'])⁻¹
        = sum_r_f @ inv(sum_f_f)
```

### Update 2: Idiosyncratic Variances (σ²)

**Closed-form solution** (diagonal elements):
```
σ²_n^new = (1/T) Σ_t E[(r_{nt} - λ_{r,n} f_t)²]
         = (1/T) [sum_r_r_diag[n] - 2 λ_r[n,:] @ sum_r_f[n,:].T 
                  + λ_r[n,:] @ sum_f_f @ λ_r[n,:].T]
```

### Update 3: Factor AR Coefficients (Φ_f) — Diagonal Case

For diagonal Φ_f with state-dependent volatility, the factor transition likelihood is:
```
log p(f_t | f_{t-1}, h_t, Φ_f) = -½ Σ_k [h_{kt} + (f_{kt} - φ_{f,k} f_{k,t-1})² exp(-h_{kt})]
```

**Weighted least squares** (for each factor k):
```
φ_{f,k}^new = [Σ_t E[exp(-h_{kt})] E[f_{kt} f_{k,t-1}]] / [Σ_t E[exp(-h_{kt})] E[f_{k,t-1}²]]
```

**Implementation note**: Due to the Gaussian approximation, we use:
```
E[exp(-h_{kt}) f_{kt} f_{k,t-1}] ≈ E[exp(-h_{kt})] × E[f_{kt} f_{k,t-1}]
```

This independence approximation is exact when factors and log-vols are independent in the posterior (which they're not, but the approximation works well in practice).

### Update 4: Log-Vol Long-Run Mean (μ)

**Closed-form solution**:
```
μ^new = (I - Φ_h)⁻¹ × (1/(T-1)) Σ_{t=2}^T E[h_t - Φ_h h_{t-1}]
      = (I - Φ_h)⁻¹ × (1/(T-1)) × (sum_h[2:T] - Φ_h @ sum_hprev)
```

For diagonal Φ_h, this simplifies to element-wise:
```
μ_k^new = (sum_h_k - φ_{h,k} × sum_hprev_k) / ((T-1) × (1 - φ_{h,k}))
```

**Note**: μ and Φ_h are coupled. Use iterative updates within M-step or ECM.

### Update 5: Log-Vol AR Coefficients (Φ_h) — Diagonal Case

**Closed-form solution** (standard VAR regression, centered at μ):
```
Φ_h^new = [Σ_t E[(h_t - μ)(h_{t-1} - μ)']  ] @ [Σ_t E[(h_{t-1} - μ)(h_{t-1} - μ)']]⁻¹
```

For diagonal Φ_h:
```
φ_{h,k}^new = [Σ_t E[(h_{kt} - μ_k)(h_{k,t-1} - μ_k)]] / [Σ_t E[(h_{k,t-1} - μ_k)²]]
```

### Update 6: Log-Vol Innovation Covariance (Q_h)

**Closed-form solution**:
```
Q_h^new = (1/(T-1)) Σ_{t=2}^T E[(h_t - μ - Φ_h(h_{t-1} - μ))(...)']
```

Expanding:
```
Q_h^new = (1/(T-1)) × [sum_h_h - 2μ×sum_h' + Tμμ' 
                       - Φ_h @ (sum_h_hprev' - μ×sum_hprev')
                       - (sum_h_hprev - sum_hprev×μ') @ Φ_h'
                       + Φ_h @ (sum_hprev_hprev - μ×sum_hprev' - sum_hprev×μ' + (T-1)μμ') @ Φ_h']
```

For diagonal Φ_h with diagonal Q_h:
```
q_{h,k}^new = (1/(T-1)) × [sum_h_h[k,k] - 2μ_k×sum_h[k] + (T-1)μ_k²
                           - 2φ_{h,k}×(sum_h_hprev[k,k] - μ_k×sum_hprev[k])
                           + φ_{h,k}²×(sum_hprev_hprev[k,k] - 2μ_k×sum_hprev[k] + (T-1)μ_k²)]
```

---

## Implementation Structure

### File Organization

```
src/bellman_filter_dfsv/core/optimization/
├── __init__.py              # Add: EMOptimizer, fit_em
├── em.py                    # NEW: Main EM algorithm
├── _em_suffstats.py         # NEW: Sufficient statistics computation
├── _em_mstep.py             # NEW: M-step update functions
├── optimization.py          # Existing: Direct MLE
└── ...
```

### Class Design

```python
class EMOptimizer:
    """EM algorithm for DFSV parameter estimation.
    
    Alternative to direct pseudo-likelihood maximization.
    Uses BIF filter + RTS smoother for E-step.
    
    Attributes:
        filter: DFSVBellmanInformationFilter instance
        max_iters: Maximum EM iterations
        tol: Convergence tolerance on log-likelihood
        verbose: Print progress
    
    Example:
        >>> em = EMOptimizer(N=10, K=2, max_iters=100)
        >>> params_hat, history = em.fit(observations, initial_params)
    """
    
    def __init__(self, N: int, K: int, max_iters: int = 100, 
                 tol: float = 1e-4, verbose: bool = True):
        ...
    
    def fit(self, observations: Array, initial_params: DFSVParamsDataclass
           ) -> tuple[DFSVParamsDataclass, EMHistory]:
        """Run EM algorithm to convergence."""
        ...
    
    def e_step(self, params: DFSVParamsDataclass, observations: Array
              ) -> DFSVSufficientStats:
        """Compute sufficient statistics via BIF + smoother."""
        ...
    
    def m_step(self, stats: DFSVSufficientStats, 
               current_params: DFSVParamsDataclass
              ) -> DFSVParamsDataclass:
        """Update all parameters given sufficient statistics."""
        ...
```

### EMHistory Dataclass

```python
@jdc.pytree_dataclass  
class EMHistory:
    """History of EM iterations for diagnostics."""
    log_likelihoods: Float[Array, "num_iters"]
    params_history: list[DFSVParamsDataclass]  # Optional, can be memory-intensive
    converged: bool
    num_iters: int
    final_params: DFSVParamsDataclass
```

---

## Numerical Stability

### Known Issues and Mitigations

| Issue | Mitigation |
|-------|------------|
| E[exp(-h)] explosion | Cap variance: `min(h_var, 4.0)` before computing log-normal moment |
| Matrix inversion in M-step | Add ridge regularization: `inv(A + εI)` with ε = 1e-6 |
| Φ_f, Φ_h outside unit circle | Project eigenvalues: `min(abs(φ), 0.999) * sign(φ)` |
| Q_h not positive definite | Symmetrize and add jitter: `(Q + Q')/2 + εI` |
| σ² ≤ 0 | Enforce minimum: `max(σ², 1e-6)` |
| Slow convergence | Use SQUAREM acceleration (optional) |

### Initialization Strategy

Good initialization is crucial for EM. Recommended approach:

1. **λ_r**: PCA on observations, take first K loadings
2. **σ²**: Residual variance from PCA
3. **Φ_f, Φ_h**: Start at 0.9 (high persistence)
4. **μ**: log(sample variance of PCA factors)
5. **Q_h**: 0.1 × I_K (small innovation variance)

---

## Comparison with Direct MLE

| Aspect | Direct MLE | EM |
|--------|------------|-----|
| Convergence rate | Superlinear (BFGS) | Linear |
| Iterations to converge | 50-200 | 100-500 |
| Per-iteration cost | O(T·N·K²) filter + gradient | O(T·N·K²) filter + smoother + M-step |
| Gradient computation | Automatic (JAX) | Not needed |
| Constraint handling | Transforms (log, tanh) | Natural projection |
| Monotonicity | Not guaranteed | Guaranteed ↑ |
| Local minima | Can oscillate | Stable approach |
| Implementation complexity | Low (auto-diff) | High (manual derivations) |

**When to use EM**:
- Poor initialization
- Direct MLE getting stuck
- Need guaranteed likelihood improvement per iteration
- Parameter constraints are complex

---

## Verification Plan

All M-step updates should be verified symbolically using SymPy before implementation.

```python
# Example verification for λ_r update
import sympy as sp

# Define symbols
N, K, T = sp.symbols('N K T', integer=True, positive=True)
lambda_r = sp.MatrixSymbol('lambda_r', N, K)
f_t = sp.MatrixSymbol('f_t', K, 1)
r_t = sp.MatrixSymbol('r_t', N, 1)
Sigma = sp.MatrixSymbol('Sigma', N, N)

# Observation log-likelihood (single time step)
residual = r_t - lambda_r @ f_t
log_lik = -sp.Rational(1,2) * (residual.T @ Sigma.inv() @ residual)[0,0]

# Differentiate w.r.t. lambda_r and set to zero
# ... (verify closed-form solution)
```

---

## References

1. **Lange, R.J. (2024)**. "Bellman filtering and smoothing for state-space models." *Journal of Econometrics*, 238(2), 105632.

2. **Shumway, R.H. & Stoffer, D.S. (1982)**. "An approach to time series smoothing and forecasting using the EM algorithm." *Journal of Time Series Analysis*, 3(4), 253-264.

3. **Durbin, J. & Koopman, S.J. (2012)**. *Time Series Analysis by State Space Methods*. Oxford University Press.

4. **Dynamax Library**. https://github.com/probml/dynamax — JAX implementation of EM for state-space models.

---

## Appendix: Derivation Sketches

### A.1 Observation Equation M-Step

Starting from:
```
Q_obs(λ_r) = E[-½ Σ_t (r_t - λ_r f_t)' Σ⁻¹ (r_t - λ_r f_t)]
```

Taking derivative w.r.t. λ_r and setting to zero:
```
∂Q/∂λ_r = Σ⁻¹ Σ_t E[(r_t - λ_r f_t) f_t'] = 0
Σ_t r_t E[f_t]' = λ_r Σ_t E[f_t f_t']
λ_r = (Σ_t r_t E[f_t]') (Σ_t E[f_t f_t'])⁻¹
```

### A.2 Factor Dynamics M-Step (Diagonal Φ_f)

For factor k, the relevant log-likelihood term is:
```
Q_f(φ_{f,k}) = E[-½ Σ_t (h_{kt} + (f_{kt} - φ_{f,k} f_{k,t-1})² exp(-h_{kt}))]
```

Taking derivative:
```
∂Q/∂φ_{f,k} = Σ_t E[exp(-h_{kt}) (f_{kt} - φ_{f,k} f_{k,t-1}) f_{k,t-1}] = 0
Σ_t E[exp(-h_{kt})] E[f_{kt} f_{k,t-1}] = φ_{f,k} Σ_t E[exp(-h_{kt})] E[f_{k,t-1}²]
```

(Using independence approximation for the Gaussian posterior.)

---

*End of Design Document*
