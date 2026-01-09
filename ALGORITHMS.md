# Mathematical Details of Filtering and Smoothing Algorithms

This document provides the mathematical specifications for the algorithms implemented in the `bellman_filter_dfsv` package.

## 1. Dynamic Factor Stochastic Volatility (DFSV) Model

The core model is a state-space model with hierarchical latent variables: Factors ($f_t$) and Log-Volatilities ($h_t$).

### 1.1 Model Equations

**1. Observation Equation:**
$$ r_t = \Lambda_r f_t + e_t, \quad e_t \sim \mathcal{N}(0, \Sigma_e) $$
*   $r_t \in \mathbb{R}^N$: Observed returns.
*   $f_t \in \mathbb{R}^K$: Latent factors.
*   $\Lambda_r \in \mathbb{R}^{N \times K}$: Factor loadings.
*   $\Sigma_e = \text{diag}(\sigma_1^2, \dots, \sigma_N^2)$: Idiosyncratic variances.

**2. Factor Dynamics:**
$$ f_t = \Phi_f f_{t-1} + \text{diag}(e^{h_t/2}) \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, I_K) $$
*   $\Phi_f \in \mathbb{R}^{K \times K}$: Factor autoregressive matrix (usually diagonal).
*   $h_t \in \mathbb{R}^K$: Log-volatilities of factors.
*   Note: The volatility of $f_t$ depends on $h_t$ (stochastic volatility).

**3. Log-Volatility Dynamics:**
$$ h_t = \mu + \Phi_h (h_{t-1} - \mu) + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q_h) $$
*   $\mu \in \mathbb{R}^K$: Long-run mean of log-volatilities.
*   $\Phi_h \in \mathbb{R}^{K \times K}$: Log-volatility autoregressive matrix.
*   $Q_h \in \mathbb{R}^{K \times K}$: Volatility innovation covariance.

### 1.2 State Vector
The joint state vector at time $t$ is denoted as $\alpha_t$:
$$ \alpha_t = \begin{bmatrix} f_t \\ h_t \end{bmatrix} \in \mathbb{R}^{2K} $$

---

## 2. Bellman Information Filter (BIF)

The Bellman Information Filter is an approximate Bayesian filter that operates in the **Information form** (Canonical form) of the Gaussian distribution. It maintains the information vector and information matrix instead of the mean and covariance.

**State Representation:**
*   Information Matrix: $\Omega_t = P_t^{-1}$
*   Information Vector: $\psi_t = P_t^{-1} \alpha_t$ (In implementation, we explicitly track mean $\alpha_t$ and information $\Omega_t$).

### 2.1 Prediction Step

Given posterior at $t-1$: $(\alpha_{t-1|t-1}, \Omega_{t-1|t-1})$.

1.  **State Prediction (Linear):**
    $$ f_{t|t-1} = \Phi_f f_{t-1|t-1} $$
    $$ h_{t|t-1} = \mu + \Phi_h (h_{t-1|t-1} - \mu) $$

2.  **Information Prediction:**
    The prediction covariance $P_{t|t-1}$ is implicitly calculated using the information form update.
    $$ P_{t|t-1} = F P_{t-1|t-1} F^T + Q_t $$
    where $F = \text{diag}(\Phi_f, \Phi_h)$ and $Q_t = \text{blockdiag}(Q_f(h_{t|t-1}), Q_h)$.
    
    The implementation computes $\Omega_{t|t-1}$ directly using the Woodbury matrix identity to avoid inverting large matrices if possible, or by inverting the predicted covariance block-wise.
    $$ \Omega_{t|t-1} = (F \Omega_{t-1|t-1}^{-1} F^T + Q_t)^{-1} $$

### 2.2 Update Step (Iterative Mode Finding)

Unlike the standard EKF which linearizes once, the BIF uses an optimization procedure to find the posterior mode, handling the non-Gaussianity of the observation density $p(r_t | f_t, h_t)$ more robustly.

**Problem:** Find $\alpha^* = \arg\max_{\alpha} \log p(\alpha | r_t, \alpha_{t|t-1})$.
$$ \log p(\alpha | r_t) \propto \log p(r_t | \alpha) + \log p(\alpha | \alpha_{t|t-1}) $$

**Algorithm (Block Coordinate Descent):**
Iterate until convergence:
1.  **Update Factors ($f$):** Closed-form Weighted Least Squares.
    $$ f^{(k+1)} = ( \Lambda_r^T \Sigma_e^{-1} \Lambda_r + \Omega_{ff} )^{-1} ( \Lambda_r^T \Sigma_e^{-1} r_t + \Omega_{ff} f_{pred} + \Omega_{fh} (h^{(k)} - h_{pred}) ) $$
2.  **Update Log-Vols ($h$):** Newton/BFGS optimization.
    $$ h^{(k+1)} = \arg\max_h \left[ \log p(r_t | f^{(k+1)}, h) - \frac{1}{2} (h - h_{pred})^T \Omega_{hh} (h - h_{pred}) - \dots \right] $$

**Information Update:**
Once the mode $\alpha_{t|t}$ is found, the information matrix is updated using the **Observed Fisher Information** (negative Hessian of log-posterior) at the mode.
$$ \Omega_{t|t} = \Omega_{t|t-1} + J_{obs}(\alpha_{t|t}) $$
$$ J_{obs} = -\nabla_{\alpha}^2 \log p(r_t | \alpha) \bigg|_{\alpha=\alpha_{t|t}} $$

---

## 3. Particle Filter (Standard)

A Sequential Importance Sampling with Resampling (SISR) filter.

**Particles:** $\{ x_t^{(i)}, w_t^{(i)} \}_{i=1}^P$, where $x_t^{(i)} = [f_t^{(i)}, h_t^{(i)}]^T$.

### 3.1 Proposal (Transition Prior)
We sample from the model dynamics:
1.  Sample $h_t^{(i)} \sim p(h_t | h_{t-1}^{(i)})$.
2.  Sample $f_t^{(i)} \sim p(f_t | f_{t-1}^{(i)}, h_t^{(i)})$.

### 3.2 Weight Update
Weights are updated by the observation likelihood:
$$ w_t^{(i)} \propto w_{t-1}^{(i)} \cdot p(r_t | f_t^{(i)}) $$
Since observation noise is Gaussian:
$$ \log p(r_t | f_t^{(i)}) = -\frac{1}{2} (r_t - \Lambda_r f_t^{(i)})^T \Sigma_e^{-1} (r_t - \Lambda_r f_t^{(i)}) + C $$

### 3.3 Resampling
**Systematic Resampling** is performed when the Effective Sample Size (ESS) falls below a threshold (e.g., $N/2$).
$$ ESS = \frac{1}{\sum (w_t^{(i)})^2} $$

---

## 4. Rao-Blackwellized Particle Filter (RBPF)

Exploits the substructure where $f_t$ is conditionally linear given $h_t$.
State is factored: $p(f_t, h_t | r_{1:t}) = p(f_t | h_t, r_{1:t}) p(h_t | r_{1:t})$.

**RBPF State:**
*   $h$-particles: $\{ h_t^{(i)} \}_{i=1}^P$
*   $f$-distributions: $\{ \mu_{f,t}^{(i)}, \Sigma_{f,t}^{(i)} \}_{i=1}^P$ (Kalman Filter states for each particle)

### 4.1 Step 1: Predict Non-Linear State ($h$)
Sample $h_t^{(i)} \sim p(h_t | h_{t-1}^{(i)})$.

### 4.2 Step 2: Kalman Predict ($f$)
For each particle $i$:
$$ \mu_{f, t|t-1}^{(i)} = \Phi_f \mu_{f, t-1}^{(i)} $$
$$ \Sigma_{f, t|t-1}^{(i)} = \Phi_f \Sigma_{f, t-1}^{(i)} \Phi_f^T + Q_f(h_t^{(i)}) $$
where $Q_f(h) = \text{diag}(e^h)$.

### 4.3 Step 3: Update Weights & Kalman Update ($f$)
Compute incremental likelihood for weight update:
$$ y_{pred}^{(i)} = \Lambda_r \mu_{f, t|t-1}^{(i)} $$
$$ S_t^{(i)} = \Lambda_r \Sigma_{f, t|t-1}^{(i)} \Lambda_r^T + \Sigma_e $$
$$ w_t^{(i)} \propto w_{t-1}^{(i)} \cdot \mathcal{N}(r_t; y_{pred}^{(i)}, S_t^{(i)}) $$

Update Kalman states for each particle:
$$ K_t^{(i)} = \Sigma_{f, t|t-1}^{(i)} \Lambda_r^T (S_t^{(i)})^{-1} $$
$$ \mu_{f, t}^{(i)} = \mu_{f, t|t-1}^{(i)} + K_t^{(i)} (r_t - y_{pred}^{(i)}) $$
$$ \Sigma_{f, t}^{(i)} = (I - K_t^{(i)} \Lambda_r) \Sigma_{f, t|t-1}^{(i)} $$

---

## 5. Smoothing Algorithms

### 5.1 RTS Smoother (Gaussian/Information Approximation)
A backward pass applied to the output of the Bellman Filter.
Runs from $t = T-1$ to $0$.

**Smoother Gain:**
$$ J_t = P_{t|t} \Phi^T P_{t+1|t}^{-1} $$
**Smoothed State:**
$$ \alpha_{t|T} = \alpha_{t|t} + J_t (\alpha_{t+1|T} - \alpha_{t+1|t}) $$
**Smoothed Covariance:**
$$ P_{t|T} = P_{t|t} + J_t (P_{t+1|T} - P_{t+1|t}) J_t^T $$

### 5.2 Rao-Blackwellized Particle Smoother (RBPS)
Combines **Forward Filtering Backward Sampling (FFBS)** for $h_t$ and **Conditional Kalman Smoothing** for $f_t$.

**Step 1: Backward Sampling of $h_{1:T}$**
Generate $M$ trajectories of log-volatilities.
Sample $h_T^{(j)} \sim \{ h_T^{(i)}, w_T^{(i)} \}$.
For $t = T-1$ to $0$:
Sample index $i$ with probability:
$$ w_{t|t+1}^{(i)} \propto w_t^{(i)} \cdot p(h_{t+1}^{(j)} | h_t^{(i)}) $$
Set $h_t^{(j)} = h_t^{(i)}$.

**Step 2: Conditional Smoothing of $f_{1:T}$**
For each sampled trajectory $h_{1:T}^{(j)}$, run a standard RTS smoother on $f_t$, treating $h_t$ (and thus $Q_{f,t}$) as known time-varying parameters.
The final smoothed estimate is the average over the $M$ trajectories.

---

## 6. Estimation (EM Algorithm)

The Expectation-Maximization (EM) algorithm finds Maximum Likelihood Estimates by iteratively maximizing the expected complete-data log-likelihood.

### 6.1 E-Step: Sufficient Statistics
We approximate expectations using the output of the RBPS.
Sufficient statistics needed for the M-step:
*   $\mathbb{E}[f_t]$
*   $\mathbb{E}[f_t f_t^T]$ and $\mathbb{E}[f_t f_{t-1}^T]$
*   $\mathbb{E}[h_t]$ and $\mathbb{E}[h_t h_t^T]$
*   $\mathbb{E}[h_t h_{t-1}^T]$
*   Specialized expectations for volatility-weighted terms (due to $\Phi_f$ update):
    *   $\mathbb{E}[e^{-h_{k,t}} f_{k,t} f_{k,t-1}]$
    *   $\mathbb{E}[e^{-h_{k,t}} f_{k,t-1}^2]$

These are computed by averaging over the $M$ sampled trajectories from the RBPS.

### 6.2 M-Step: Parameter Updates
Updates maximize $Q(\theta | \theta_{old})$.

**1. Factor Loadings ($\Lambda_r$):**
Standard OLS regression of $r_t$ on $f_t$.
$$ \hat{\Lambda}_r = \left( \sum_{t=1}^T r_t \mathbb{E}[f_t]^T \right) \left( \sum_{t=1}^T \mathbb{E}[f_t f_t^T] \right)^{-1} $$

**2. Idiosyncratic Variances ($\Sigma_e$):**
Residual variance of the observation equation.
$$ \hat{\sigma}_j^2 = \frac{1}{T} \sum_{t=1}^T \mathbb{E} \left[ (r_{j,t} - \hat{\lambda}_j^T f_t)^2 \right] $$
Implementation uses sufficient statistics expansion: $\mathbb{E}[r^2] - 2 \lambda \mathbb{E}[rf] + \lambda \mathbb{E}[ff^T] \lambda^T$.

**3. Factor Autoregression ($\Phi_f$):**
This is a **Weighted Least Squares** problem because the noise variance depends on $h_t$.
Assuming $\Phi_f$ is diagonal, for each factor $k$:
$$ \hat{\phi}_{f,k} = \frac{\sum_{t=2}^T \mathbb{E} \left[ e^{-h_{k,t}} f_{k,t} f_{k,t-1} \right]}{\sum_{t=2}^T \mathbb{E} \left[ e^{-h_{k,t}} f_{k,t-1}^2 \right]} $$
This is the optimal estimator derived from minimizing the volatility-standardized residuals.

**4. Log-Volatility Parameters ($\mu, \Phi_h$):**
These parameters are coupled in the AR(1) process: $h_t - \mu = \Phi_h (h_{t-1} - \mu) + \eta_t$.
We solve this iteratively (block coordinate descent) within each M-step:

*   **Update $\Phi_h$ given $\mu$:**
    Standard OLS of centered log-vols.
    $$ \hat{\Phi}_h = \left( \sum_{t=2}^T \mathbb{E}[(h_t - \mu)(h_{t-1} - \mu)^T] \right) \left( \sum_{t=2}^T \mathbb{E}[(h_{t-1} - \mu)(h_{t-1} - \mu)^T] \right)^{-1} $$
    (Restricted to diagonal in implementation).

*   **Update $\mu$ given $\Phi_h$:**
    $$ \hat{\mu} = (T-1)^{-1} (I - \Phi_h)^{-1} \sum_{t=2}^T \mathbb{E}[h_t - \Phi_h h_{t-1}] $$

**5. Log-Volatility Variance ($Q_h$):**
Residual variance of the $h_t$ equation.
$$ \hat{Q}_h = \frac{1}{T-1} \sum_{t=2}^T \mathbb{E} \left[ (h_t - \mu - \Phi_h(h_{t-1} - \mu)) (h_t - \mu - \Phi_h(h_{t-1} - \mu))^T \right] $$
