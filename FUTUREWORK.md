# Future Work: From Research to Production

This document outlines the roadmap for elevating `BellmanFilterDFSV` from a high-quality research package to a robust, production-grade "Market Engine" capable of processing real-world financial data at scale.

The goal is to prepare this package to serve as the reliable upstream signal generator for the Physics-Informed Market Discovery (PHS) pipeline.

---

## 1. Native Handling of "Ragged" Market Universes (High Priority)

**Problem:**
Real-world equity indices (like the S&P 500) are dynamic. Over any significant period, companies are listed, delisted, halted, or merge. A naive fixed-size matrix input ($T \times N$) cannot represent this "churn" without dropping data or complex imputation.

**The Solution: Dynamic Masking**
Modify the filter updates to accept a binary validity mask.

*   **Implementation Strategy:**
    *   Update `BellmanFilter.filter` to accept an optional `mask` argument of shape $(T, N)$.
    *   In `update_info_step`, incorporate the mask into the Woodbury matrix identity.
    *   Effectively, masked assets should contribute **zero information** to the update, behaving as if they have infinite variance.
    *   Ensure the precision matrix construction handles these "missing" dimensions without singularity (using the "infinite variance" trick or explicit sub-indexing).

**Benefit:** allows processing the full, raw history of an index without manual survivorship bias management.

---

## 2. Automated "Cold Start" Initialization (PCA) (High Priority)

**Problem:**
EM algorithms are sensitive to initialization. Starting with random factor loadings ($\Lambda$) and factors ($f$) often leads to:
1.  **Slow Convergence:** Wasting GPU hours finding the "right" rotation.
2.  **Mode Collapse:** Converging to local optima where factors explain trivial variance or are essentially white noise.

**The Solution: PCA Heuristic Initialization**
Provide a helper to warm-start the model using linear Principal Component Analysis.

*   **Implementation Strategy:**
    *   Create `initialize_from_data(returns: Array, K: int) -> DFSVParams`.
    *   **Step 1:** Run standard PCA on the covariance of `returns`.
    *   **Step 2:** Initialize $\Lambda$ (loadings) using the first $K$ scaled eigenvectors.
    *   **Step 3:** Initialize factor paths $f_{0:T}$ using the Principal Components.
    *   **Step 4:** Initialize $\Phi_f$ and $\Sigma_e$ estimates from the PCA residuals.

**Benefit:** Drastically reduces training time and guarantees the model starts with the dominant linear signal drivers already captured.

---

## 3. "Production-Grade" Numerical Configuration (Medium Priority)

**Problem:**
Financial data contains "black swan" events (e.g., 2008, 2020) where volatility spikes by orders of magnitude (10-20$\sigma$). Hardcoded numerical stability terms (like `1e-6` jitter) may be insufficient for these regimes, causing Cholesky failures deep in long-running batch jobs.

**The Solution: Adaptive Numerical Context**
Decouple mathematical logic from numerical stability constants.

*   **Implementation Strategy:**
    *   Define a `NumericalConfig` dataclass:
        ```python
        @dataclass
        class NumericalConfig:
            cholesky_jitter: float = 1e-6
            optimization_tol: float = 1e-8
            min_volatility_floor: float = 1e-6
        ```
    *   Thread this config through `BellmanFilter`, `fit_mle`, and `fit_em`.
    *   **Advanced:** Implement a "Safe Cholesky" wrapper that catches decomposition failures, increases jitter automatically, and retries.

**Benefit:** Prevents crashes during expensive computations and allows adaptation to different asset classes (e.g., Crypto vs. Bonds).

---

## 4. Strict Output Schema & Export (Medium Priority)

**Problem:**
As the upstream "Market Engine," this package's output is the input for the PHS Discovery Engine. If internal changes alter the output format (e.g., shape, units, variable names), it breaks the downstream pipeline.

**The Solution: Explicit Interface Contract**
Formalize the output into a versioned schema.

*   **Implementation Strategy:**
    *   Define a `MarketSignal` dataclass in `types.py` containing:
        *   `factors`: $(T, K)$
        *   `log_volatilities`: $(T, K)$
        *   `volatilities`: $(T, K)$ (Pre-computed $\exp(h/2)$)
        *   `metadata`: Dictionary (tickers, dates, convergence stats)
    *   Implement `export_to_phs(result: MarketSignal, path: str)` to save as versioned `.npz` or `.h5`.

**Benefit:** Creates a rigid API boundary, allowing independent evolution of the Factor Extraction and Physics Discovery codebases.

---

## 5. Signal Verification (Standard Errors) (Low Priority)

**Problem:**
The PHS engine assumes the extracted factors are "truth." It is vital to know if a factor is statistically significant or just fitting noise. Point estimates (MLE) do not provide this confidence.

**The Solution: Hessian-Based Inference**
Quantify uncertainty in the estimated parameters.

*   **Implementation Strategy:**
    *   After EM convergence, run `jax.hessian` on the marginal log-likelihood w.r.t. parameters.
    *   Compute the inverse Hessian (Observed Fisher Information) to get standard errors.
    *   Generate a "Signal Quality Report" flagging:
        *   Non-persistent factors ($\Phi_f \approx 0$).
        *   Insignificant loadings ($\lambda_{i,k} \approx 0$).

**Benefit:** Provides a "Go/No-Go" gauge for factors before they are used in downstream hypothesis generation.

---

## Implementation Roadmap

1.  **Phase 1 (Data Readiness):** Implement **Ragged Inputs** and **PCA Initialization**. This enables running on real S&P 500 data.
2.  **Phase 2 (Reliability):** Implement **Numerical Config** and **Output Schema**. This ensures long-running jobs are stable and results are portable.
3.  **Phase 3 (Validation):** Implement **Standard Errors**. This adds scientific rigor to the thesis defense.
