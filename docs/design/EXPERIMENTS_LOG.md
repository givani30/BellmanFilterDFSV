# EM Algorithm Experiments Log

**Created:** 2026-01-09
**Status:** Active
**Context:** Developing stable parameter estimation for DFSV models (High-dimensional State Space Models).

---

## 1. Motivation: The "Why" of EM

Initial attempts to use direct Maximum Likelihood Estimation (MLE) with gradient descent (L-BFGS-B, Adam) failed as we scaled the problem dimensions (N > 5, K > 1).

### The Problem with MLE
1.  **Gradient Instability**: Propagating gradients through hundreds of time steps ($T \approx 500$) in a recursive filter leads to exploding or vanishing gradients.
2.  **Parameter Coupling**: Parameters like $\Phi_f$ (factor dynamics) and $\Lambda$ (loadings) are highly coupled. Optimizers struggle to find the narrow valley of high likelihood.
3.  **Local Optima**: MLE often overfits the observation noise ($e_t$) rather than capturing the latent factor structure ($f_t$), leading to high likelihood scores but poor parameter recovery.

### The EM Solution
We turned to the **Expectation-Maximization (EM)** algorithm, specifically using a **Rao-Blackwellized Particle Smoother (RBPS)** for the E-step.
-   **E-Step**: Estimate latent states ($f_t, h_t$) given current params. (Smoothing)
-   **M-Step**: Analytically update params to maximize expected log-likelihood. (Regression)

**Why it works**: The M-step decomposes the complex global optimization into simple, convex local problems (Linear Regression for $\Lambda$, AR(1) fitting for $\Phi$).

---

## 2. Experiments & Findings

### Experiment 01-07: Early Failures (MLE)
-   **Method**: Direct autodiff on `BellmanFilter.log_likelihood`.
-   **Outcome**: Works for toy problems ($N=1, K=1$). Fails for ($N=10, K=3$) with `NaN` gradients or divergence.

### Experiment 08: RBPS-EM Prototype
-   **Goal**: Prove EM convergence using a python-loop heavy prototype.
-   **Challenge**: Initial implementation diverged.
-   **Bug**: The backward smoother (`jax.lax.scan` with `reverse=True`) was returning outputs in *input index order* ($0 \to T$), but we were treating them as *reversed time* ($T \to 0$) and applying `[::-1]`. This scrambled the time indices, matching $t$ with $T-t$.
-   **Fix**: Removed the incorrect `[::-1]` reversal on the scan outputs.
-   **Result**: Immediate convergence. Monotonic log-likelihood increase.

### Experiment 09: "Next" Architecture
-   **Goal**: Port the messy prototype to the clean `src/bellman_filter_dfsv/next/` architecture.
-   **Method**: Implemented `next/em.py`, `next/rbps.py`, using `NamedTuple` and strict typing.
-   **Result**: Verified correct parameter recovery on $N=3, K=1$ system.

### scaling_test: MLE vs EM Showdown
We ran a head-to-head comparison on a larger system ($T=500, N=10, K=3$).

| Metric | MLE (Adam) | EM (RBPS) | Winner |
|:-------|:-----------|:----------|:-------|
| **Speed** | 71.7s | **12.9s** | **EM (5.5x Faster)** |
| **Stability** | FAIL (No convergence) | **PASS** | **EM** |
| **$\Phi_f$ Error** | 0.1283 | **0.0028** | **EM (45x Better)** |
| **$\Lambda$ Error** | 0.1623 | **0.1421** | **EM** |
| **Likelihood** | **1090.8** | 1014.4 | MLE (Overfitting) |

**Conclusion**: EM is superior for structural recovery and speed, even if it yields a lower "raw" likelihood score (which is often due to MLE overfitting noise).

---

## 3. Current Architecture

We have established a new standard for this codebase in `src/bellman_filter_dfsv/next/`:

### Core Components
1.  **`em.py`**: The high-level orchestrator. Contains `fit_em`.
2.  **`rbps.py`**: The Rao-Blackwellized Particle Smoother. Handles the heavy lifting of the E-step using `jax.lax.scan` and `vmap`.
3.  **`types.py`**: Strict data structures (`DFSVParams`, `EMSufficientStats`) ensuring type safety.

### Key Implementation Details
-   **JIT Compilation**: The entire E-step is JIT-compiled (`jax.jit`), making it extremely fast despite the particle loop.
-   **Numerical Stability**:
    -   **Log-Sum-Exp**: Used for weight normalization.
    -   **Jitter**: Small diagonal constants (`1e-6`) added to covariances before Cholesky decomposition.
    -   **Symmetrization**: `0.5 * (P + P.T)` applied after every covariance update.

---

## 4. Usage Guide

To fit a model using the new stable EM algorithm:

```python
from bellman_filter_dfsv.next import fit_em, DFSVParams

# Initialize
params = DFSVParams(...)

# Fit
final_params, history = fit_em(
    observations=data,
    init_params=params,
    num_particles=200,    # Good balance for speed/accuracy
    num_trajectories=20,  # Sufficient for smoothing
    max_iters=50
)
```
