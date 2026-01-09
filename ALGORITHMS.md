# Mathematical Specifications: Filtering and Smoothing Algorithms

This document provides complete mathematical specifications for all algorithms implemented in `bellman_filter_dfsv`. Each section includes the theoretical foundation, algorithmic steps, and implementation notes.

---

## 1. Dynamic Factor Stochastic Volatility (DFSV) Model

The DFSV model is a hierarchical state-space model with two layers of latent variables: **factors** ($f_t$) with time-varying volatility, and **log-volatilities** ($h_t$) that govern the factor dynamics.

### 1.1 Model Equations

#### Observation Equation

$$
r_t = \Lambda_r f_t + e_t, \quad e_t \sim \mathcal{N}(0, \Sigma_e)
$$

**where:**
- $r_t \in \mathbb{R}^N$ — Observed returns (e.g., asset returns)
- $f_t \in \mathbb{R}^K$ — Latent common factors
- $\Lambda_r \in \mathbb{R}^{N \times K}$ — Factor loading matrix
- $\Sigma_e = \text{diag}(\sigma_1^2, \ldots, \sigma_N^2)$ — Idiosyncratic variance (diagonal)

The observation equation decomposes returns into a common factor component ($\Lambda_r f_t$) and asset-specific noise ($e_t$).

---

#### Factor Dynamics (Stochastic Volatility)

$$
f_t = \Phi_f f_{t-1} + \text{diag}(\exp(h_t/2)) \, \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, I_K)
$$

**where:**
- $\Phi_f \in \mathbb{R}^{K \times K}$ — Factor autoregression matrix (typically diagonal for parsimony)
- $h_t \in \mathbb{R}^K$ — Log-volatilities of factors (element-wise)
- $\varepsilon_t \sim \mathcal{N}(0, I_K)$ — Unit-variance shocks

**Key property:** The innovation covariance matrix is **time-varying** and depends on $h_t$:

$$
Q_{f,t} = \text{diag}(\exp(h_{1,t}), \ldots, \exp(h_{K,t}))
$$

This captures volatility clustering in factor dynamics — periods of high/low volatility persist.

---

#### Log-Volatility Dynamics (Ornstein-Uhlenbeck Process)

$$
h_t = \mu + \Phi_h (h_{t-1} - \mu) + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q_h)
$$

**where:**
- $\mu \in \mathbb{R}^K$ — Long-run mean of log-volatilities
- $\Phi_h \in \mathbb{R}^{K \times K}$ — Mean-reversion matrix (typically diagonal, $|\Phi_{h,kk}| < 1$ for stationarity)
- $Q_h \in \mathbb{R}^{K \times K}$ — Innovation covariance (log-volatility process noise)

The log-volatility follows a **Gaussian AR(1)** process, ensuring volatility itself $\exp(h_t/2)$ is always positive.

---

### 1.2 Joint State Vector

We define the augmented state vector as:

$$
\alpha_t = \begin{bmatrix} f_t \\ h_t \end{bmatrix} \in \mathbb{R}^{2K}
$$

The filtering and smoothing algorithms estimate the posterior distribution $p(\alpha_{0:T} | r_{1:T})$.

---

## 2. Bellman Information Filter (BIF)

The Bellman Information Filter operates in the **information form** (canonical parameterization) of the Gaussian distribution. Instead of tracking mean $\mu$ and covariance $P$, it maintains:

$$
\begin{aligned}
\Omega_t &= P_t^{-1} \quad \text{(precision/information matrix)} \\
\psi_t &= P_t^{-1} \mu_t \quad \text{(information vector)}
\end{aligned}
$$

**Advantages:**
- Numerically stable for high-dimensional systems
- Direct incorporation of information (no matrix inversions in update step)
- Natural for handling missing data (zero precision contribution)

**Our implementation:** We track mean $\alpha_t$ and information $\Omega_t$ explicitly (not $\psi_t$) for computational efficiency.

---

### 2.1 Prediction Step

**Input:** Posterior at time $t-1$: $(\alpha_{t-1|t-1}, \Omega_{t-1|t-1})$

#### Step 1: State Prediction (Deterministic)

$$
\begin{aligned}
f_{t|t-1} &= \Phi_f f_{t-1|t-1} \\
h_{t|t-1} &= \mu + \Phi_h (h_{t-1|t-1} - \mu)
\end{aligned}
$$

#### Step 2: Covariance Prediction

Define the block-diagonal transition matrix:

$$
F = \begin{bmatrix}
\Phi_f & 0 \\
0 & \Phi_h
\end{bmatrix}, \quad
Q_t = \begin{bmatrix}
Q_{f,t}(h_{t|t-1}) & 0 \\
0 & Q_h
\end{bmatrix}
$$

where $Q_{f,t}(h) = \text{diag}(\exp(h_1), \ldots, \exp(h_K))$ (note: full variance, not $\exp(h/2)$).

Standard Kalman prediction:

$$
P_{t|t-1} = F P_{t-1|t-1} F^\top + Q_t
$$

#### Step 3: Information Prediction

Convert to information form:

$$
\Omega_{t|t-1} = P_{t|t-1}^{-1}
$$

**Implementation note:** We use block-wise inversion or Woodbury matrix identity to avoid full $2K \times 2K$ inversion when possible.

---

### 2.2 Update Step (Iterative Mode Finding)

Unlike the Extended Kalman Filter (EKF), which linearizes the observation model once, the BIF uses **iterative optimization** to find the posterior mode. This handles the non-Gaussianity introduced by the stochastic volatility structure more robustly.

#### Problem Formulation

Find the Maximum A Posteriori (MAP) estimate:

$$
\alpha_{t|t}^* = \arg\max_{\alpha} \log p(\alpha | r_t, \alpha_{t|t-1})
$$

Decompose the log-posterior:

$$
\log p(\alpha | r_t) \propto \underbrace{\log p(r_t | \alpha)}_{\text{Observation likelihood}} + \underbrace{\log p(\alpha | \alpha_{t|t-1})}_{\text{Prior from prediction}}
$$

The prior term is Gaussian:

$$
\log p(\alpha | \alpha_{t|t-1}) = -\frac{1}{2} (\alpha - \alpha_{t|t-1})^\top \Omega_{t|t-1} (\alpha - \alpha_{t|t-1}) + \text{const}
$$

---

#### Algorithm: Block Coordinate Descent

We alternate between updating factors $f$ (which have closed-form solutions) and log-volatilities $h$ (which require numerical optimization).

**Iterate until convergence:**

1. **Update Factors $f$ (Closed-form)**

   Given current $h^{(k)}$, solve the quadratic optimization:

   $$
   f^{(k+1)} = \arg\max_f \left[ \log p(r_t | f, h^{(k)}) + \log p(f | f_{\text{pred}}, h^{(k)}) \right]
   $$

   This is a **weighted least squares** problem with solution:

   $$
   f^{(k+1)} = \left( \Lambda_r^\top \Sigma_e^{-1} \Lambda_r + \Omega_{ff} \right)^{-1} 
   \left( \Lambda_r^\top \Sigma_e^{-1} r_t + \Omega_{ff} f_{\text{pred}} + \Omega_{fh} (h^{(k)} - h_{\text{pred}}) \right)
   $$

   where $\Omega_{ff}, \Omega_{fh}$ are blocks of $\Omega_{t|t-1}$.

2. **Update Log-Volatilities $h$ (Numerical Optimization)**

   Given updated $f^{(k+1)}$, solve:

   $$
   h^{(k+1)} = \arg\max_h \left[ \log p(r_t | f^{(k+1)}, h) + \log p(h | h_{\text{pred}}) \right]
   $$

   **Implementation:** We use **L-BFGS-B** optimizer with automatic differentiation (JAX).

   The log-posterior includes:
   - Observation likelihood term: $\log p(r_t | f^{(k+1)}) = \text{const}$ (doesn't depend on $h$ given $f$)
   - Prior term: $-\frac{1}{2} (h - h_{\text{pred}})^\top \Omega_{hh} (h - h_{\text{pred}})$

3. **Convergence Check**

   Stop when $\| \alpha^{(k+1)} - \alpha^{(k)} \|_2 < \epsilon$ (typically $\epsilon = 10^{-5}$).

---

#### Information Matrix Update

Once the mode $\alpha_{t|t}$ is found, update the information matrix using the **observed Fisher information** (negative Hessian of log-likelihood):

$$
\begin{aligned}
J_{\text{obs}}(\alpha) &= -\nabla_\alpha^2 \log p(r_t | \alpha) \bigg|_{\alpha = \alpha_{t|t}} \\
\Omega_{t|t} &= \Omega_{t|t-1} + J_{\text{obs}}(\alpha_{t|t})
\end{aligned}
$$

**Closed-form for DFSV observation model:**

$$
J_{\text{obs}} = \begin{bmatrix}
\Lambda_r^\top \Sigma_e^{-1} \Lambda_r & 0 \\
0 & 0
\end{bmatrix}
$$

(The $h$ block is zero because the observation $r_t$ is independent of $h_t$ given $f_t$.)

---

## 3. Particle Filter (Bootstrap Filter)

A **Sequential Importance Sampling with Resampling (SISR)** filter using the transition prior as the proposal distribution.

### 3.1 Representation

At time $t$, we maintain a weighted particle cloud:

$$
\{ (x_t^{(i)}, w_t^{(i)}) \}_{i=1}^P, \quad x_t^{(i)} = \begin{bmatrix} f_t^{(i)} \\ h_t^{(i)} \end{bmatrix}
$$

where:
- $P$ = number of particles
- $w_t^{(i)}$ = normalized importance weights (sum to 1)

The posterior distribution is approximated as:

$$
p(\alpha_t | r_{1:t}) \approx \sum_{i=1}^P w_t^{(i)} \, \delta_{\alpha_t^{(i)}}(\alpha_t)
$$

---

### 3.2 Algorithm

For $t = 1, 2, \ldots, T$:

#### Step 1: Proposal Sampling (Transition Prior)

For each particle $i$:

1. Sample log-volatility:
   $$
   h_t^{(i)} \sim \mathcal{N}(\mu + \Phi_h (h_{t-1}^{(i)} - \mu), Q_h)
   $$

2. Sample factor (conditional on new $h_t^{(i)}$):
   $$
   f_t^{(i)} \sim \mathcal{N}(\Phi_f f_{t-1}^{(i)}, \text{diag}(\exp(h_t^{(i)})))
   $$

---

#### Step 2: Weight Update

Update importance weights using the observation likelihood:

$$
\tilde{w}_t^{(i)} = w_{t-1}^{(i)} \cdot p(r_t | f_t^{(i)})
$$

For the Gaussian observation model:

$$
\log p(r_t | f_t^{(i)}) = -\frac{1}{2} (r_t - \Lambda_r f_t^{(i)})^\top \Sigma_e^{-1} (r_t - \Lambda_r f_t^{(i)}) - \frac{N}{2} \log(2\pi) - \frac{1}{2} \log |\Sigma_e|
$$

**Normalize weights:**

$$
w_t^{(i)} = \frac{\tilde{w}_t^{(i)}}{\sum_{j=1}^P \tilde{w}_t^{(j)}}
$$

---

#### Step 3: Resampling (Adaptive)

To prevent weight degeneracy, resample when the **Effective Sample Size (ESS)** drops below a threshold (e.g., $P/2$):

$$
\text{ESS}_t = \frac{1}{\sum_{i=1}^P (w_t^{(i)})^2}
$$

**Systematic Resampling Algorithm:**
1. Draw uniform random start $u_0 \sim \text{Uniform}[0, 1/P]$
2. Generate resampling indices via stratified sampling: $u_i = u_0 + i/P$
3. Select particles deterministically based on cumulative weight distribution
4. Reset weights to $w_t^{(i)} = 1/P$

---

### 3.3 Posterior Estimates

**Mean:**
$$
\hat{\alpha}_t = \sum_{i=1}^P w_t^{(i)} \alpha_t^{(i)}
$$

**Covariance:**
$$
\hat{P}_t = \sum_{i=1}^P w_t^{(i)} (\alpha_t^{(i)} - \hat{\alpha}_t)(\alpha_t^{(i)} - \hat{\alpha}_t)^\top
$$

---

## 4. Rao-Blackwellized Particle Filter (RBPF)

Exploits the **conditional linearity** of the DFSV model: given $h_t$, the factors $f_t$ have Gaussian dynamics. We marginalize out $f_t$ analytically using a Kalman filter, reducing variance.

### 4.1 State Decomposition

Factorize the posterior:

$$
p(f_t, h_t | r_{1:t}) = \underbrace{p(f_t | h_t, r_{1:t})}_{\text{Kalman filter (Gaussian)}} \cdot \underbrace{p(h_t | r_{1:t})}_{\text{Particle filter (non-Gaussian)}}
$$

**RBPF State:** For each particle $i$:
- $h_t^{(i)}$ — Particle representing log-volatility trajectory
- $(\mu_{f,t}^{(i)}, \Sigma_{f,t}^{(i)})$ — Gaussian sufficient statistics for $f_t | h_t^{(i)}, r_{1:t}$

---

### 4.2 Algorithm

For $t = 1, 2, \ldots, T$:

#### Step 1: Sample $h_t$ (Particle Propagation)

For each particle $i$:

$$
h_t^{(i)} \sim \mathcal{N}(\mu + \Phi_h (h_{t-1}^{(i)} - \mu), Q_h)
$$

---

#### Step 2: Kalman Predict for $f_t$

Given $h_t^{(i)}$, propagate the Gaussian state for factors:

$$
\begin{aligned}
\mu_{f,t|t-1}^{(i)} &= \Phi_f \mu_{f,t-1}^{(i)} \\
\Sigma_{f,t|t-1}^{(i)} &= \Phi_f \Sigma_{f,t-1}^{(i)} \Phi_f^\top + Q_{f,t}(h_t^{(i)})
\end{aligned}
$$

where $Q_{f,t}(h) = \text{diag}(\exp(h_1), \ldots, \exp(h_K))$.

---

#### Step 3: Compute Marginal Likelihood (for Weights)

Prediction of observation:

$$
\begin{aligned}
\hat{r}_t^{(i)} &= \Lambda_r \mu_{f,t|t-1}^{(i)} \\
S_t^{(i)} &= \Lambda_r \Sigma_{f,t|t-1}^{(i)} \Lambda_r^\top + \Sigma_e
\end{aligned}
$$

Update weight using the **marginal likelihood** $p(r_t | h_t^{(i)}, r_{1:t-1})$:

$$
\tilde{w}_t^{(i)} = w_{t-1}^{(i)} \cdot \mathcal{N}(r_t; \hat{r}_t^{(i)}, S_t^{(i)})
$$

**Normalize:** $w_t^{(i)} = \tilde{w}_t^{(i)} / \sum_j \tilde{w}_t^{(j)}$

---

#### Step 4: Kalman Update for $f_t$

For each particle $i$:

$$
\begin{aligned}
K_t^{(i)} &= \Sigma_{f,t|t-1}^{(i)} \Lambda_r^\top (S_t^{(i)})^{-1} \\
\mu_{f,t}^{(i)} &= \mu_{f,t|t-1}^{(i)} + K_t^{(i)} (r_t - \hat{r}_t^{(i)}) \\
\Sigma_{f,t}^{(i)} &= (I - K_t^{(i)} \Lambda_r) \Sigma_{f,t|t-1}^{(i)}
\end{aligned}
$$

---

#### Step 5: Resampling (Adaptive)

Same ESS-based adaptive resampling as standard particle filter.

---

### 4.3 Posterior Estimates

**Mean of $f_t$:**
$$
\hat{f}_t = \sum_{i=1}^P w_t^{(i)} \mu_{f,t}^{(i)}
$$

**Covariance of $f_t$:**
$$
\hat{\Sigma}_{f,t} = \sum_{i=1}^P w_t^{(i)} \left( \Sigma_{f,t}^{(i)} + (\mu_{f,t}^{(i)} - \hat{f}_t)(\mu_{f,t}^{(i)} - \hat{f}_t)^\top \right)
$$

**Mean of $h_t$:**
$$
\hat{h}_t = \sum_{i=1}^P w_t^{(i)} h_t^{(i)}
$$

---

## 5. Smoothing Algorithms

Smoothers compute the **full-sample posterior** $p(\alpha_{0:T} | r_{1:T})$, refining filtered estimates using future observations.

---

## 5.1 Rauch-Tung-Striebel (RTS) Smoother

The RTS smoother is a **backward recursion** applied to the Bellman filter output. It assumes Gaussian state transitions (an approximation for DFSV).

### Algorithm

**Initialization:** Start from filtered estimate at $T$:
$$
\alpha_{T|T}, \quad P_{T|T}
$$

**For $t = T-1, T-2, \ldots, 0$:**

1. **Compute Smoother Gain**

   $$
   J_t = P_{t|t} F^\top P_{t+1|t}^{-1}
   $$

   where $F = \text{diag}(\Phi_f, \Phi_h)$.

2. **Backward State Correction**

   $$
   \alpha_{t|T} = \alpha_{t|t} + J_t (\alpha_{t+1|T} - \alpha_{t+1|t})
   $$

3. **Backward Covariance Correction**

   $$
   P_{t|T} = P_{t|t} + J_t (P_{t+1|T} - P_{t+1|t}) J_t^\top
   $$

---

### Implementation Notes

- We convert information form $(\alpha_{t|t}, \Omega_{t|t})$ to moment form $(P_{t|t} = \Omega_{t|t}^{-1})$ for the smoother.
- The smoothed covariance $P_{t|T}$ accounts for uncertainty reduction from future observations.
- **Output:** `SmootherResult(smoothed_means, smoothed_covs)`

---

## 5.2 Rao-Blackwellized Particle Smoother (RBPS)

Combines **Forward Filtering Backward Sampling (FFBS)** for log-volatilities with **conditional Kalman smoothing** for factors.

### Algorithm Overview

1. **Forward Pass:** Run RBPF to obtain $\{ (h_t^{(i)}, \mu_{f,t}^{(i)}, \Sigma_{f,t}^{(i)}, w_t^{(i)}) \}_{t=0}^T$
2. **Backward Sampling:** Generate $M$ trajectories of $h_{0:T}$ via FFBS
3. **Conditional Smoothing:** For each $h_{0:T}$ trajectory, run RTS smoother on $f_{0:T}$
4. **Averaging:** Compute smoothed estimates by averaging over trajectories

---

### Step 1: Backward Sampling of $h_{0:T}$

**Initialization:** Sample terminal particle index $j \sim \{ w_T^{(i)} \}$, set $h_T^{(m)} = h_T^{(j)}$.

**For $t = T-1, T-2, \ldots, 0$:**

Compute backward weights:

$$
w_{t|t+1}^{(i)} \propto w_t^{(i)} \cdot p(h_{t+1}^{(m)} | h_t^{(i)})
$$

where:

$$
p(h_{t+1} | h_t) = \mathcal{N}(h_{t+1}; \mu + \Phi_h(h_t - \mu), Q_h)
$$

**Sample index** $j \sim \{ w_{t|t+1}^{(i)} \}$ and set $h_t^{(m)} = h_t^{(j)}$.

Repeat $M$ times to generate $\{ h_{0:T}^{(m)} \}_{m=1}^M$.

---

### Step 2: Conditional Kalman Smoothing

For each sampled trajectory $h_{0:T}^{(m)}$:

1. **Forward pass:** Run Kalman filter treating $h_t^{(m)}$ as known (fixes $Q_{f,t}$)
2. **Backward pass:** Run standard RTS smoother to obtain $\{ \mu_{f,t|T}^{(m)}, \Sigma_{f,t|T}^{(m)} \}_{t=0}^T$

---

### Step 3: Compute Smoothed Estimates

**Mean:**
$$
\hat{\alpha}_{t|T} = \frac{1}{M} \sum_{m=1}^M \begin{bmatrix} \mu_{f,t|T}^{(m)} \\ h_t^{(m)} \end{bmatrix}
$$

**Covariance:**
$$
\hat{P}_{t|T} = \frac{1}{M} \sum_{m=1}^M \begin{bmatrix}
\Sigma_{f,t|T}^{(m)} + (\mu_{f,t|T}^{(m)} - \hat{f}_{t|T})(\mu_{f,t|T}^{(m)} - \hat{f}_{t|T})^\top & 0 \\
0 & (h_t^{(m)} - \hat{h}_{t|T})(h_t^{(m)} - \hat{h}_{t|T})^\top
\end{bmatrix}
$$

---

### Advantages over Standard Particle Smoother

- **Lower variance:** Marginalizing $f_t$ reduces Monte Carlo noise
- **Fewer trajectories:** Requires $M \ll P$ (e.g., $M = 50$ instead of $P = 500$)
- **Smooth state estimates:** Kalman smoothing provides continuous factor paths

---

## 6. Expectation-Maximization (EM) Algorithm

The EM algorithm finds **Maximum Likelihood Estimates (MLE)** of model parameters $\theta = \{ \Lambda_r, \Sigma_e, \Phi_f, \Phi_h, \mu, Q_h \}$ by iteratively maximizing the expected complete-data log-likelihood.

---

### 6.1 E-Step: Compute Sufficient Statistics

Run the **RBPS** to obtain smoothed trajectories $\{ \alpha_{0:T}^{(m)} \}_{m=1}^M$. Compute expectations (averaging over trajectories):

#### Basic Moments

$$
\begin{aligned}
\mathbb{E}[f_t] &= \frac{1}{M} \sum_{m=1}^M \mu_{f,t|T}^{(m)} \\
\mathbb{E}[f_t f_t^\top] &= \frac{1}{M} \sum_{m=1}^M \left( \Sigma_{f,t|T}^{(m)} + \mu_{f,t|T}^{(m)} (\mu_{f,t|T}^{(m)})^\top \right) \\
\mathbb{E}[f_t f_{t-1}^\top] &= \frac{1}{M} \sum_{m=1}^M \left( \Sigma_{f,t,t-1|T}^{(m)} + \mu_{f,t|T}^{(m)} (\mu_{f,t-1|T}^{(m)})^\top \right)
\end{aligned}
$$

where $\Sigma_{f,t,t-1|T}^{(m)}$ is the lag-1 smoothed covariance (computed via RTS smoother).

#### Log-Volatility Moments

$$
\begin{aligned}
\mathbb{E}[h_t] &= \frac{1}{M} \sum_{m=1}^M h_t^{(m)} \\
\mathbb{E}[h_t h_t^\top] &= \frac{1}{M} \sum_{m=1}^M h_t^{(m)} (h_t^{(m)})^\top \\
\mathbb{E}[h_t h_{t-1}^\top] &= \frac{1}{M} \sum_{m=1}^M h_t^{(m)} (h_{t-1}^{(m)})^\top
\end{aligned}
$$

#### Volatility-Weighted Moments (for $\Phi_f$ Update)

Since the factor equation has heteroskedastic noise, we need:

$$
\begin{aligned}
\mathbb{E}[\exp(-h_{k,t}) f_{k,t} f_{k,t-1}] &= \frac{1}{M} \sum_{m=1}^M \exp(-h_{k,t}^{(m)}) \mu_{f,k,t|T}^{(m)} \mu_{f,k,t-1|T}^{(m)} \\
\mathbb{E}[\exp(-h_{k,t}) f_{k,t-1}^2] &= \frac{1}{M} \sum_{m=1}^M \exp(-h_{k,t}^{(m)}) \left( (\Sigma_{f,t-1|T}^{(m)})_{kk} + (\mu_{f,k,t-1|T}^{(m)})^2 \right)
\end{aligned}
$$

These account for time-varying noise variance in weighted least squares estimation of $\Phi_f$.

---

### 6.2 M-Step: Parameter Updates

Maximize $Q(\theta | \theta_{\text{old}}) = \mathbb{E}[\log p(\alpha_{0:T}, r_{1:T} | \theta)]$ w.r.t. $\theta$.

---

#### 1. Factor Loadings $\Lambda_r$

Standard **Ordinary Least Squares (OLS)** regression:

$$
\hat{\Lambda}_r = \left( \sum_{t=1}^T r_t \mathbb{E}[f_t]^\top \right) \left( \sum_{t=1}^T \mathbb{E}[f_t f_t^\top] \right)^{-1}
$$

---

#### 2. Idiosyncratic Variances $\Sigma_e$

For each asset $j = 1, \ldots, N$:

$$
\hat{\sigma}_j^2 = \frac{1}{T} \sum_{t=1}^T \left( r_{j,t}^2 - 2 \hat{\lambda}_j^\top \mathbb{E}[f_t] r_{j,t} + \hat{\lambda}_j^\top \mathbb{E}[f_t f_t^\top] \hat{\lambda}_j \right)
$$

This expands the expectation $\mathbb{E}[(r_{j,t} - \lambda_j^\top f_t)^2]$ using sufficient statistics.

---

#### 3. Factor Autoregression $\Phi_f$

Due to time-varying noise variance $\exp(h_{k,t})$, this is a **Weighted Least Squares (WLS)** problem. Assuming $\Phi_f$ is diagonal, for each factor $k$:

$$
\hat{\phi}_{f,k} = \frac{\sum_{t=2}^T \mathbb{E}[\exp(-h_{k,t}) f_{k,t} f_{k,t-1}]}{\sum_{t=2}^T \mathbb{E}[\exp(-h_{k,t}) f_{k,t-1}^2]}
$$

**Derivation:** Minimizing the volatility-standardized residual:

$$
\sum_{t=2}^T \mathbb{E}\left[ \exp(-h_{k,t}) (f_{k,t} - \phi_{f,k} f_{k,t-1})^2 \right]
$$

---

#### 4. Log-Volatility Parameters $(\mu, \Phi_h)$

The AR(1) equation $h_t - \mu = \Phi_h (h_{t-1} - \mu) + \eta_t$ couples $\mu$ and $\Phi_h$. We solve via **block coordinate descent** within each M-step:

**Update $\Phi_h$ given $\mu$:**

Assuming $\Phi_h$ is diagonal:

$$
\hat{\phi}_{h,k} = \frac{\sum_{t=2}^T \mathbb{E}[(h_{k,t} - \mu_k)(h_{k,t-1} - \mu_k)]}{\sum_{t=2}^T \mathbb{E}[(h_{k,t-1} - \mu_k)^2]}
$$

**Update $\mu$ given $\Phi_h$:**

$$
\hat{\mu} = (T-1)^{-1} (I - \Phi_h)^{-1} \sum_{t=2}^T \mathbb{E}[h_t - \Phi_h h_{t-1}]
$$

**Iterate 3-5 times** until convergence.

---

#### 5. Log-Volatility Innovation Covariance $Q_h$

Residual variance:

$$
\hat{Q}_h = \frac{1}{T-1} \sum_{t=2}^T \mathbb{E}\left[ (h_t - \mu - \Phi_h(h_{t-1} - \mu))(h_t - \mu - \Phi_h(h_{t-1} - \mu))^\top \right]
$$

Expand using sufficient statistics:

$$
\hat{Q}_h = \frac{1}{T-1} \sum_{t=2}^T \left( \mathbb{E}[h_t h_t^\top] - (\mu + \Phi_h \mathbb{E}[h_t]) \mathbb{E}[h_t]^\top - \dots \right)
$$

---

### 6.3 EM Convergence

**Stopping criteria:**
1. Relative change in log-likelihood: $|\ell^{(k)} - \ell^{(k-1)}| / |\ell^{(k-1)}| < \epsilon$ (e.g., $\epsilon = 10^{-4}$)
2. Maximum iterations (e.g., 50)

**Implementation notes:**
- Each E-step requires running RBPS ($\mathcal{O}(T \cdot P \cdot M)$ cost)
- M-step updates are closed-form (fast)
- Typical convergence: 10-20 iterations for well-initialized parameters

---

## 7. Computational Complexity

| Algorithm | Complexity per Time Step | Memory | Notes |
|-----------|-------------------------|--------|-------|
| **Bellman Filter** | $\mathcal{O}(K^3 + N K^2)$ | $\mathcal{O}(K^2)$ | Dominated by $2K \times 2K$ matrix ops |
| **Particle Filter** | $\mathcal{O}(P \cdot N K)$ | $\mathcal{O}(P \cdot K)$ | Linear in $P$, $N$, $K$ |
| **RBPF** | $\mathcal{O}(P \cdot K^3)$ | $\mathcal{O}(P \cdot K^2)$ | Kalman filter per particle |
| **RTS Smoother** | $\mathcal{O}(K^3)$ | $\mathcal{O}(T \cdot K^2)$ | Store all $P_{t\|t}$ |
| **RBPS** | $\mathcal{O}(M \cdot T \cdot K^3)$ | $\mathcal{O}(M \cdot T \cdot K)$ | $M$ conditional smoothers |
| **EM (per iter)** | $\mathcal{O}(\text{RBPS cost})$ | $\mathcal{O}(T \cdot K)$ | E-step dominates |

**Scalability notes:**
- For large $N$ (e.g., S&P 500 with $N = 500$): Use RBPF to marginalize high-dimensional $f_t$
- For large $K$ (many factors): BIF becomes expensive due to $\mathcal{O}(K^3)$ matrix operations
- JAX JIT compilation reduces overhead by 10-100× compared to pure Python

---

## References

1. **Bellman Information Filter:**  
   Lange, R.-J. (2024). *Bayesian Estimation of Dynamic Factor Models with Stochastic Volatility*. Working Paper.

2. **Rao-Blackwellized Particle Methods:**  
   Doucet, A., et al. (2000). "On Sequential Monte Carlo Sampling Methods for Bayesian Filtering." *Statistics and Computing*, 10(3), 197-208.

3. **DFSV Models in Finance:**  
   Aguilar, O., & West, M. (2000). "Bayesian Dynamic Factor Models and Portfolio Allocation." *Journal of Business & Economic Statistics*, 18(3), 338-357.

4. **EM Algorithm:**  
   Dempster, A. P., et al. (1977). "Maximum Likelihood from Incomplete Data via the EM Algorithm." *Journal of the Royal Statistical Society: Series B*, 39(1), 1-22.

---

**Document Version:** 2.0 (2026-01-09)  
**Package Version:** `bellman_filter_dfsv` v2.0.0
