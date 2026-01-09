# EM Algorithm: Motivation, Challenges, and Status
**Date:** 2026-01-09

## 1. Motivation: Why EM instead of Direct MLE?

The primary motivation for implementing the Expectation-Maximization (EM) algorithm for the Dynamic Factor Stochastic Volatility (DFSV) model is the **numerical instability and poor scalability of direct Maximum Likelihood Estimation (MLE)**.

### 1.1. The Scalability Bottleneck
As we scale the model dimensions:
-   **N (Observations)**: Number of stock returns (e.g., 50-100+).
-   **K (Factors)**: Number of latent factors (e.g., 3-5+).

The log-likelihood surface becomes increasingly complex and multimodal. Direct optimization of the log-likelihood function $\mathcal{L}(\theta) = \log P(y_{1:T} | \theta)$ using gradient-based methods (L-BFGS-B, Adam, etc.) faces severe challenges:

1.  **Gradient Explosion/Vanishing**: The likelihood involves recursive filtering (Particle Filter or Bellman Filter). Propagating gradients through time ($T \approx 1000+$ steps) and through the resampling steps (in PF) or complex updates (in BIF) leads to unstable gradients.
2.  **Parameter Coupling**: In the DFSV model, parameters like factor loadings ($\Lambda$), transition matrices ($\Phi$), and stochastic volatility parameters ($\sigma_\eta$) are highly coupled in the likelihood function. Perturbing one often requires compensatory changes in others to maintain high likelihood, creating narrow "valleys" in the optimization landscape that are hard for generic optimizers to traverse.
3.  **Computational Cost**: Evaluating the gradient $\nabla_\theta \mathcal{L}$ requires a full forward pass of the filter (and potentially a backward pass for reverse-mode AD). Doing this for every line search step in an optimizer is prohibitively expensive.

### 1.2. The EM Advantage
The EM algorithm decouples the problem into two easier sub-problems:

*   **E-Step (Smoothing)**: Estimate the posterior distribution of the latent states ($f_t, h_t$) given the *current* parameters. This is handled by the **Rao-Blackwellized Particle Smoother (RBPS)**. This step does not require gradients with respect to $\theta$.
*   **M-Step (Maximization)**: Maximize the expected complete-data log-likelihood with respect to $\theta$, given the smoothed sufficient statistics.
    *   Many updates (e.g., $\Lambda$, $\Phi$, $\mu$) have **closed-form solutions** (like OLS or weighted regression).
    *   This avoids the need for a global optimizer to "search" blindly; the E-step guides the parameters directly toward the region of higher probability.
    *   It is numerically much more stable because we are optimizing "local" transition probabilities rather than the full path integral.

---

## 2. Current Implementation: RBPS-EM

We are implementing a **Rao-Blackwellized Particle Smoother (RBPS)** to perform the E-Step.

### 2.1. Why RBPS?
The DFSV model has a specific structure:
*   **Linear/Gaussian substructure**: The factor dynamics $f_t$ are linear-Gaussian *conditional* on the volatilities $h_t$.
*   **Non-linear substructure**: The log-volatilities $h_t$ are non-linear (or rather, they enter the observation variance non-linearly).

RBPS exploits this by:
1.  Using particles to sample the log-volatilities $h_t$.
2.  Using exact Kalman Smoothing (RTS Smoother) for the factors $f_t$ *conditional* on each particle's volatility path.

This reduces the variance of the estimator significantly compared to a standard Particle Smoother, as we analytically integrate out the factors.

### 2.2. Architecture
*   **E-Step**: `RBPS_E_Step`
    *   Run `RBPF` (Forward Filter) to get particles and weights.
    *   Run `Backward Simulation` (FFBS) or `Backward Smoothing` to get smoothed trajectories of $h_t$ and sufficient statistics for $f_t$.
*   **M-Step**: `M_Step`
    *   Update $\Lambda$ (Loadings): OLS on smoothed factors.
    *   Update $\Phi_f, \Sigma_f$ (Factor dynamics): VAR(1) on smoothed factors.
    *   Update parameters for $h_t$: AR(1) optimization on smoothed log-vols.

---

## 3. Session Rules & Protocol

To ensure stability and prevent regressions during this critical debugging phase:

1.  **Isolation**:
    *   Work **exclusively** in `scripts/experiments/` (e.g., `rbps_em_lib.py`, `exp08_rbps_em.py`).
    *   **DO NOT** modify the core library (`src/bellman_filter_dfsv/`) until the EM algorithm is proven to converge in the experiments.
    *   This prevents breaking the stable, tested v2 architecture while experimenting with volatile EM logic.

2.  **Reference Implementation**:
    *   When needing simulation or filtering components, look at `src/bellman_filter_dfsv/next/` (v2 architecture) as the reference.
    *   Re-implement or import these trusted components into the experiment folder rather than modifying them in place.

3.  **Verification Steps**:
    *   Every fix must be verified with `debug_kalman.py` (Linear Gaussian ground truth) first.
    *   Then verify with `verify_smoother_isolation.py` (DFSV model).
    *   Only run full EM (`exp08_rbps_em.py`) after the smoother is proven correct.

---

## 4. Current Status & Debugging (Session Log)

**Current Focus**: Debugging the RBPS Backward Step in `scripts/experiments/rbps_em_lib.py`.

### 4.1. The Issue
We observed **divergence** in the EM algorithm ($\lambda \to 0$, variance exploding).
Investigation revealed that the **Smoother is producing negatively correlated estimates** for the factors (Correlation $\approx -0.5$ between Ground Truth and Estimate, instead of $>0.9$).

### 4.2. The Root Cause
We identified a bug in how `jax.lax.scan` output was being handled.
*   We use `jax.lax.scan(..., reverse=True)` to run the backward pass.
*   **Discovery**: `scan(reverse=True)` iterates backwards ($T \to 0$) but returns the stacked outputs in the **original input order** ($0 \to T$), corresponding to the indices of the input arrays.
*   **The Bug**: Our code assumed the output was also reversed (time $T \to 0$) and applied `[::-1]` to "fix" it.
*   **Result**: This actually *scrambled* the time indices, matching time $t$ states with time $T-t$ observations/parameters, leading to garbage gradients and negative correlations.

### 4.3. Next Steps
1.  **Fix**: Remove the `[::-1]` reversal in `rbps_em_lib.py`.
2.  **Verify**: Run `debug_kalman.py` (Linear Gaussian case) and `verify_smoother_isolation.py` (DFSV case) to confirm high positive correlation.
3.  **Execute**: Run `exp08_rbps_em.py` to confirm EM convergence.
