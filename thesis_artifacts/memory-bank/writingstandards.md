# Core Econometrics Notation Standards (Condensed)

Based on Abadir &amp; Magnus standards, adapted for project needs.

## I. General Principles

* **Consistency:** Use notation consistently.
* **Clarity:** Choose recognizable symbols, minimize ambiguity.
* **Efficiency:** Adhere to this standard to avoid redefining.

## II. Vectors and Matrices

* **Vectors:** Lowercase bold-italic (**_a_**, **_α_**). Column vector by default.
* **Matrices:** Uppercase bold-italic (**_A_**, **_Γ_**).
* **Transpose:** Prime symbol ('). E.g., **_a_**'.
* **Special:** Null (**0**, **O**), Sum (**i**$_n$), Unit (**e**$_i$), Identity (**I**, **I**$_n$).
* **Key Operations:**
  * Transpose: **_A_**'
  * Inverse: **_A_**$^{-1}$ (Moore-Penrose: **_A_**$^+$)
  * Diagonal Matrix: dg(**_A_**), diag($a_1$,...,$a_n$)
  * Vec Operator: vec(**_A_**) (stacks columns)
  * Rank: rk(**_A_**)
  * Eigenvalue: $\lambda_i$(**_A_**)
  * Trace: tr(**_A_**)
  * Determinant: |**_A_**| or det(**_A_**)
  * Vector Norm: $||$**_a_**$||$ (Euclidean)
  * Matrix Norm: $||$**_A_**$||$ (Frobenius)
  * Positive (Semi)definite: **_A_** > **_B_**, **_A_** $\ge$ **_B_**
  * Kronecker Product: **_A_** $\otimes$ **_B_**
  * Hadamard Product: **_A_** $\odot$ **_B_** (element-wise)

## III. Regression Models

* **Linear Model:** **_y_** = **_Xβ_** + **_ε_**.
* **Disturbances:** **_ε_** (spherical), **_u_** (non-spherical, e.g., var(**_u_**) = **_Ω_**).
* **Estimators:** Hats (**_β̂_**).
* **Fitted Values &amp; Residuals:** **_ŷ_** = **_Xβ̂_**, **_ε̂_** = **_y_** - **_ŷ_**.

## IV. Mathematical Symbols and Functions

* **Convergence:** $\xrightarrow{a.s.}$ (almost surely), $\xrightarrow{p}$ (in probability), $\xrightarrow{d}$ (in distribution).
* **Probabilistic Order:** $O_p(g(n))$, $o_p(g(n))$.
* **Standard Sets:** $\mathbb{R}$ (Reals), $\mathbb{N}$ (Naturals {1,2,...}).
* **Derivatives:**
  * Partial: $D_j \phi = \partial \phi / \partial x_j$.
  * Gradient (vector arg): $\nabla \phi(\mathbf{x}) = \partial\phi/\partial\mathbf{x}$ (column vector).
  * Jacobian (vector func): $D\mathbf{f}(\mathbf{x}) = \partial\mathbf{f}(\mathbf{x})/\partial\mathbf{x}'$ (row vector).
  * Hessian (scalar func): $H\phi(\mathbf{x}) = \partial^2\phi/\partial\mathbf{x}\partial\mathbf{x}'$.
* **Common Functions:** exp, log, $\Gamma(x)$ (Gamma).

## V. Statistical Notation

* **Distributed As:** $\sim$ (is distributed as), $\sim_a$ (asymptotically).
* **Moments:** E[$X$] (expectation), var($X$), cov($X,Y$).
* **Likelihood:** L($\theta$), $l(\theta) = \log L(\theta)$.
* **Score Vector:** S($\theta$) = $\partial l / \partial \theta$.
* **Hessian:** H($\theta$) = $\partial^2 l / \partial \theta \partial \theta'$.
* **Information Matrix:** $\mathcal{I}(\theta)$ = -E[H($\theta$)] (expected Fisher).
* **Filtration:** $\mathcal{F}_t$ (information up to time $t$).

## VI. Common Distributions

* **Normal (Multivariate):** N$_m$(**_μ_**,**_Ω_**) (mean **_μ_**, covariance **_Ω_**)
* **Standard Normal:** PDF $\phi(z)$, CDF $\Phi(z)$.
* **Chi-squared:** $\chi_n^2(\delta)$ (df $n$, non-centrality $\delta$).
* **Student's t:** $t_n(\delta)$ (df $n$, non-centrality $\delta$).
* **Gamma:** $\Gamma(\alpha,\lambda)$ (shape $\alpha$, rate $\lambda$).
* **Inverse Gamma:** IG($\alpha, \beta$) (shape $\alpha$, scale $\beta$). _Added as commonly used for variance priors._
