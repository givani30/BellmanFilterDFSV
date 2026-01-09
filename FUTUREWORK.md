# Future Work: From Research to Production

This document outlines the roadmap for elevating `BellmanFilterDFSV` from a high-quality research package to a robust, production-grade "Market Engine" capable of processing real-world financial data at scale.

The goal is to prepare this package to serve as the reliable upstream signal generator for the Physics-Informed Market Discovery (PHS) pipeline.

---

## 1. Native Handling of "Ragged" Market Universes (High Priority)

**Problem:**
Real-world equity indices (like the S&P 500) are dynamic. Over any significant period, companies are listed, delisted, halted, or merge. A naive fixed-size matrix input ($T \times N$) cannot represent this "churn" without dropping data or complex imputation.

**The Solution: Dynamic Masking with Zero Precision Addition**
Modify the filter updates to accept a binary validity mask.

*   **Implementation Strategy:**
    *   Update `BellmanFilter.filter` to accept an optional `mask` argument of shape $(T, N)$.
    *   **Crucial Nuance:** Do *not* use the "infinite variance" trick (setting $\sigma^2 = \infty$), as `0 * inf` results in `NaN` gradients in JAX.
    *   **Correct Approach:** Use **Zero Precision Addition**.
        *   In the Information Filter update step, the precision (information) contribution of an observation is $\Lambda^T \Sigma^{-1} \Lambda$.
        *   Multiply this precision term by the mask: $\Omega_{update} = \text{mask}_t \odot (\Lambda^T \Sigma^{-1} \Lambda)$.
        *   Masked assets effectively contribute zero information to the posterior, preserving numerical stability without hazardous floating-point arithmetic.

**Benefit:** allows processing the full, raw history of an index without manual survivorship bias management or numerical instability.

---

## 2. Automated "Cold Start" Initialization (PCA) (High Priority)

**Problem:**
EM algorithms are sensitive to initialization. Starting with random factor loadings ($\Lambda$) and factors ($f$) often leads to slow convergence or mode collapse. Additionally, the DFSV model assumes unit variance for process noise, which creates a scale mismatch if not respected during init.

**The Solution: Standardized PCA Heuristic Initialization**
Provide a helper to warm-start the model using linear Principal Component Analysis, carefully aligned with model assumptions.

*   **Implementation Strategy:**
    *   Create `initialize_from_data(returns: Array, K: int) -> DFSVParams`.
    *   **Step 1: Standardization.** Z-score the `returns` matrix (subtract mean, divide by std) before PCA. DFSV models typically assume zero-mean innovations.
    *   **Step 2: Run PCA.** Extract eigenvectors and principal components.
    *   **Step 3: Normalize & Rotate.**
        *   Normalize the PCA factor paths ($f_{0:T}$) to have **unit variance**. This aligns with the DFSV state equation assumption ($f_t = \Phi f_{t-1} + \dots + \varepsilon_t$, where $\varepsilon_t \sim N(0, I)$).
        *   Push the magnitude scale into the loadings matrix ($\Lambda$).
    *   **Step 4:** Initialize $\Phi_f$ and $\Sigma_e$ estimates from the PCA residuals.

**Benefit:** Drastically reduces training time, prevents "scale shock" in the first EM iterations, and guarantees the model starts with the dominant linear signal drivers.

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
As the upstream "Market Engine," this package's output is the input for the PHS Discovery Engine. Misalignment in dates or shapes breaks the pipeline. Merging signal data with external holdings data (like ARKK) requires precise time alignment.

**The Solution: Explicit Interface Contract with Mandatory Dates**
Formalize the output into a versioned schema that enforces time-awareness.

*   **Implementation Strategy:**
    *   Define a `MarketSignal` dataclass in `types.py` containing:
        *   `factors`: $(T, K)$
        *   `log_volatilities`: $(T, K)$
        *   `volatilities`: $(T, K)$ (Pre-computed $\exp(h/2)$)
        *   **`dates`**: $(T,)$ array of Unix timestamps or integer dates. **Mandatory.**
        *   `metadata`: Dictionary (tickers, convergence stats).
    *   **Crucial Nuance:** Do not rely on implicit row indices (e.g., "row 0 is start date"). Explicit timestamps prevent off-by-one errors (T+2 settlement vs. trade date) when merging with other datasets.
    *   Implement `export_to_phs(result: MarketSignal, path: str)` to save as versioned `.npz` or `.h5`.

**Benefit:** Creates a rigid, date-aligned API boundary, ensuring safe integration with the Physics Discovery codebase.

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

1.  **Phase 1 (Data Readiness):** Implement **Ragged Inputs** (Zero Precision Masking) and **PCA Initialization** (Standardized). This enables running on real S&P 500 data.
2.  **Phase 2 (Reliability):** Implement **Numerical Config** and **Output Schema** (with Dates). This ensures long-running jobs are stable and results are portable.
3.  **Phase 3 (Validation):** Implement **Standard Errors**. This adds scientific rigor to the thesis defense.
