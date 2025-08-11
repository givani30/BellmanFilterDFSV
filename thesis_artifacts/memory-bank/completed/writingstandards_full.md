# Econometrics Notation Standards

This document outlines a proposed standard for notation in econometrics, aiming for consistency, clarity, and alignment with ISO regulations, using GitHub-flavored Markdown for mathematical expressions.

## I. General Principles

* **Consistency:** Use notation consistently throughout a document.
* **Clarity:** Choose symbols that are instantly recognizable and minimize ambiguity.
* **Efficiency:** A common standard reduces the need to define notation repeatedly.

## II. Vectors and Matrices

### A. Basic Notation

* **Vectors:** Lowercase bold-italic (e.g., **_a_**, **_α_**). Represents a column vector by default.
* **Matrices:** Uppercase bold-italic (e.g., **_A_**, **_Γ_**).
* **Elements:** $a_{ij}$ denotes the element in the $i$-th row and $j$-th column of matrix **_A_**.
* **Dimensions:** Prefer $m \ge n$ for an $m \times n$ matrix if a choice exists.
* **Rows & Columns:** Columns of **_A_**: **_a_**$_{.1}$, ..., **_a_**$_{.n}$. Rows of **_A_**: **_a_**$_{1.}$', ..., **_a_**$_{m.}$'.
* **Transpose:** Use a prime symbol ('). E.g., **_a_**' is the transpose (row vector) of **_a_** (column vector).

### B. Special Vectors & Matrices

* **Null Vector:** **0** or **0**$_n$.
* **Sum Vector:** **i**$_n$ (vector of ones).
* **Unit Vector:** **e**$_i$ ($i$-th column of the identity matrix).
* **Null Matrix:** **O** or **O**$_{mn}$.
* **Identity Matrix:** **I** or **I**$_n$.

### C. Spaces and Orthogonality

* **Conformability:** Matrices/vectors are conformable if their sum or product is mathematically defined.
* **Orthogonality:** Vectors **_a_** and **_b_** are orthogonal if **_a_**'**_b_** = 0 (denoted **_a_** $\perp$ **_b_**).
* **Column Space:** col(**_A_**) = {**_x_** : **_x_** = **_Ac_** for some **_c_** $\neq$ **0**}.
* **Null Space:** {**_x_** : **_Ax_** = **0**}.
* **Orthogonal Complement:** col$^\perp$(**_A_**) = {**_x_** : **_A_**'**_x_** = **0**}.

### D. Matrix/Vector Operations

| Operation                  | Symbol(s)                               | Notes                                                  |
| :------------------------- | :-------------------------------------- | :----------------------------------------------------- |
| Transpose                  | **_A_**'                                |                                                        |
| Inverse                    | **_A_**$^{-1}$                          |                                                        |
| Moore-Penrose Inverse      | **_A_**$^+$                             |                                                        |
| Generalized Inverse        | **_A_**$^-$                             |                                                        |
| Diagonal Matrix (from A)   | dg(**_A_**)                             | Contains diagonal elements of **_A_** |
| Diagonal Matrix (from list)| diag($a_1$,...,$a_n$)                   |                                                        |
| Block-Diagonal Matrix      | diag(**_A_**$_1$,...,**_A_**$_n$)       |                                                        |
| Matrix Power               | **_A_**$^p$                             | E.g., **_A_**$^2$ = **_AA_** |
| Matrix Square Root         | **_A_**$^{1/2}$                         | Unique for positive semidefinite matrix                  |
| Adjoint Matrix             | **_A_**$^\#$                            |                                                        |
| Complex Conjugate Transpose| **_A_**$^*$                             | Hermitian conjugate                                      |
| Principal Submatrix        | **_A_**$_k$                             | Top-left $k \times k$ submatrix                           |
| Partitioned Matrix         | (**_A_**, **_B_**) or (**_A_**:**_B_**) |                                                        |
| Vec Operator               | vec(**_A_**)                            | Stacks columns of **_A_** |
| Vech Operator              | vech(**_A_**)                           | Stacks lower triangular part (incl. diagonal)           |
| Rank                       | rk(**_A_**)                             |                                                        |
| Eigenvalue                 | $\lambda_i$(**_A_**)                    | Order $\lambda_1 \ge \dots \ge \lambda_n$ recommended |
| Trace                      | tr(**_A_**)                             | Sum of diagonal elements                                 |
| Exponential Trace          | etr(**_A_**)                            | $\exp(\text{tr } \mathbf{A})$                         |
| Determinant                | |**_A_**| or det(**_A_**)                 | Use $|\det \mathbf{A}|$ for absolute value              |
| Matrix Norm                | $||$**_A_**$||$                         | Frobenius norm: $\sqrt{\text{tr}( \mathbf{A}^* \mathbf{A} )}$ |
| Vector Norm                | $||$**_a_**$||$                         | Euclidean norm: $\sqrt{\mathbf{a}^* \mathbf{a}}$       |
| Positive Semidefinite      | **_A_** $\ge$ **_B_** | **_A_** - **_B_** is positive semidefinite               |
| Positive Definite          | **_A_** > **_B_** | **_A_** - **_B_** is positive definite                   |
| Kronecker Product          | **_A_** $\otimes$ **_B_** |                                                        |
| Hadamard Product           | **_A_** $\odot$ **_B_** | Element-wise product                                     |
| Commutation Matrix         | **_K_**$_{mn}$                          | vec(**_A_**') = **_K_**$_{mn}$ vec(**_A_**) for $m \times n$ **_A_** |
| Duplication Matrix         | **_D_**$_n$                             | vec(**_A_**) = **_D_**$_n$ vech(**_A_**) for symmetric **_A_** |
| Jordan Block               | **_J_**$_k$($\lambda$)                  | $k \times k$ block with $\lambda$ on diagonal, 1s above |

## III. Regression Models

### A. Linear Regression
* **Model:** **_y_** = **_Xβ_** + **_ε_** (vector form).
* **Scalar Form:** $y_i$ = **_x_**$_i$'**_β_** + $\epsilon_i$. Index $i$ for cross-section, $t$ for time-series.
* **Regressors:** Index $h=1,...,k$.
* **Constant Term:** If included, write $y_i = \beta_1 + \beta_2 x_{i2} + \dots + \beta_k x_{ik} + \epsilon_i$. Avoid using $\beta_0$.
* **Disturbances:** Use **_ε_** (or $\epsilon_i$) for spherical errors (i.i.d., mean 0, constant variance). Use **_u_** (or $u_i$) for non-spherical errors.
* **Estimators & Predictors:** Use hats (**_β̂_**, $\hat{y}$) or tildes (**_β̃_**, $\tilde{y}$).
* **Fitted Values & Residuals:** Use **_ŷ_** = **_Xβ̂_** and **_ε̂_** = **_y_** - **_ŷ_**. Avoid special symbols like **b** and **e** for OLS.
* **Variance Estimator:** Use $\hat{\sigma}^2$.
* **Goodness of Fit:** $R^2$ (coefficient of determination), $\bar{R}^2$ (adjusted $R^2$).
* **Projection Matrices:** **_P_**<sub>**_X_**</sub> = **_X_**(**_X_**'**_X_**)<sup>+</sup>**_X_**', **_M_**<sub>**_X_**</sub> = **_I_**$_n$ - **_P_**<sub>**_X_**</sub>.
* **De-meaning Matrix:** **_M_**<sub>**i**</sub> = **_I_**$_n$ - (1/$n$)**ii**'. **_M_**<sub>**i**</sub>**_a_** gives vector **_a_** in deviation from its mean.
* **Hypotheses:** Null $H_0$, Alternative $H_A$. State restrictions as **_R_**'**_β_** = **_c_**. Let $r$ be the number of restrictions (dimension of **_c_**).

### B. GLS Model
* **Model:** **_y_** = **_Xβ_** + **_u_**, with E[**_u_**] = **0** and var(**_u_**) = **_Ω_**. Use **_Ω_** (Omega), not $\Sigma$ (Sigma).

### C. Multivariate & Simultaneous Equations Models
* **Multivariate Linear Model:** **_Y_** = **_XB_** + **_U_**, or **_y_**$_{i.}$' = **_x_**$_i$'**_B_** + **_u_**$_{i.}$'.
* **Simultaneous Equations (Structural Form):** **_YΓ_** = **_XB_** + **_U_**.
* **Simultaneous Equations (Reduced Form):** **_Y_** = **_XΠ_** + **_V_**, where **_Π_** = **_BΓ_**$^{-1}$ and **_V_** = **_UΓ_**$^{-1}$ (if **_Γ_** is invertible).

## IV. Mathematical Symbols and Functions

### A. Logic, Sets, and Convergence
* **Equivalence/Definition:** $\equiv$ (identity), := (defines).
* **Implication:** $\implies$ (implies), $\iff$ (if and only if).
* **Mappings/Convergence:** $\to$ (converges to), $\mapsto$ (maps to).
* **Approximation:** $\approx$ (approximately equal), $\propto$ (proportional to).
* **Order Notation:** O($g(x)$) (at most order), o($g(x)$) (lesser order), $\sim$ (asymptotically equal).
* **Standard Sets:** $\mathbb{N}$ (Naturals {1,2,...}), $\mathbb{Z}$ (Integers), $\mathbb{Q}$ (Rationals), $\mathbb{R}$ (Reals), $\mathbb{C}$ (Complex). Use superscripts for dimension ($\mathbb{R}^n$) and subscripts for subsets ($\mathbb{R}_+$, $\mathbb{Z}_{0,+}$).
* **Set Operations:** $\in$ (belongs to), $\notin$ (does not belong to), {$x$ : P} (set builder), $\subseteq$ (subset), $\subset$ (proper subset), $\cup$ (union), $\cap$ (intersection), $\setminus$ (set difference), $\emptyset$ (empty set), A$^c$ (complement).
* **Intervals:** [$a$,$b$], ($a$,$b$), [$a$,$b$), ($a$,$b$]. Alternative for open: ]$a$,$b$[.
* **Topological:** interior(S), S' (derived set), $\bar{S}$ (closure), $\partial S$ (boundary).
* **Sequences:** {$Z_j$} or {$Z_j$}$_m^n$.

### B. Functions and Operators
* **Function Notation:** $f: S \to T$. Use $f, g, \phi, \psi, \vartheta$ for scalar-valued, **_f_**, **_g_** for vector-valued, **_F_**, **_G_** for matrix-valued.
* **Composition:** $g \circ f$.
* **Convolution:** $(g * f)(x) = \int g(y)f(x-y) dy$.
* **Differential:** d.
* **Derivatives:**
    * Partial: $D_j \phi(x) = \partial \phi / \partial x_j$. Second partial: $D_{kj}^2 \phi(x)$.
    * Total (scalar): $\phi'(x)$, $\phi''(x)$, $\phi^{(n)}(x)$.
    * Derivative/Gradient (vector argument): $D\phi(\mathbf{x}) = \partial\phi/\partial\mathbf{x}'$ (row vector), $\nabla \phi(\mathbf{x}) = \partial\phi/\partial\mathbf{x}$ (column vector/gradient).
    * Jacobian (vector function): $D\mathbf{f}(\mathbf{x}) = \partial\mathbf{f}(\mathbf{x})/\partial\mathbf{x}'$.
    * Hessian (scalar function): $H\phi(\mathbf{x}) = \partial^2\phi/\partial\mathbf{x}\partial\mathbf{x}'$.
* **Difference/Lag Operators:** L or B (backward shift: L$x_t = x_{t-1}$), $\nabla$ (backward difference: $\nabla x_t = x_t - x_{t-1}$), $\Delta$ (forward difference: $\Delta x_t = x_{t+1} - x_t$).
* **Integral Evaluation:** $[f(x)]_a^b = f(b) - f(a)$.
* **Transforms:** $\mathcal{F}$ (Fourier), $\mathcal{L}$ (Laplace), $\mathcal{M}$ (Mellin).

### C. Common Symbols & Functions
* **Constants:** i (imaginary unit), e (base of natural log).
* **Functions:** exp (exponential), log (natural log), $\log_a$ (log base $a$), ! (factorial), sgn($x$) (sign), $\lfloor x \rfloor$ (integer part/floor), |$x$| (absolute value/modulus), Re($x$) (real part), Im($x$) (imaginary part), $\Gamma(x)$ (Gamma function), B($x,y$) (Beta function).
* **Other:** $\delta_{ij}$ (Kronecker delta), $x^*$ (complex conjugate of scalar $x$), $1_K$ (indicator function: 1 if K true, 0 otherwise), B($c$;$r$) (neighborhood/ball center $c$, radius $r$).
* **Manifolds:** $\mathcal{V}^{n \times k}$ (Stiefel manifold: $n \times k$ matrices **_X_** s.t. **_X_**'**_X_**=**_I_**$_k$), $\mathcal{O}^n$ (Orthogonal group $\mathcal{V}^{n \times n}$), $\mathcal{S}^n$ (Unit sphere $\mathcal{V}^{n \times 1}$).

## V. Statistical Notation

### A. Distributions and Convergence
* **Distributed As:** $\sim$ (is distributed as), $\sim_a$ (is asymptotically distributed as).
* **Convergence:**
    * $\xrightarrow{a.s.}$ or $\to$ (almost surely).
    * $\xrightarrow{p}$ (in probability).
    * $\xrightarrow{d}$ (in distribution).
    * $\xrightarrow{w}$ (weakly).
    * plim (probability limit).
* **Probabilistic Order:** $O_p(g(n))$, $o_p(g(n))$.

### B. Moments and Probability
* **Probability:** Pr($A$).
* **Expectation:** E[$X$], E[$X$|$Y$]. Use 'expectation' for population parameter. Use 'average' (e.g., $\bar{x}$) for sample mean. Avoid 'mean'.
* **Variance/Covariance:** var($X$), cov($X,Y$), corr($X,Y$). Use these for population parameters. Use 'sample variance', etc., for sample statistics.
* **Standard Deviation vs. Error:** 'Standard deviation' is $\sqrt{var(X)}$. 'Standard error' is the estimate or realization of the standard deviation of an estimator.

### C. Likelihood and Inference
* **Likelihood:** L($\theta$), $l(\theta) = \log L(\theta)$ (log-likelihood).
* **Score Vector:** S($\theta$) = $\partial l / \partial \theta$.
* **Hessian:** H($\theta$) = $\partial^2 l / \partial \theta \partial \theta'$.
* **Information Matrix:** $\mathcal{I}(\theta)$ = -E[H($\theta$)] (expected Fisher information). -H($\theta$) is observed information.
* **Filtration:** $\mathcal{F}_t$ (information up to time $t$).
* **t-statistic:** Refers to the random variable; 't-value' is its realization.

### D. Common Distributions

| Distribution          | Notation                                  | Parameters                                      |
| :-------------------- | :---------------------------------------- | :---------------------------------------------- |
| Binomial              | bin($n,p$)                                | $n$ trials, $p$ success probability           |
| Poisson               | Po($\mu$)                                 | $\mu$ rate/mean                                 |
| Uniform               | U($a,b$)                                  | Interval [$a,b$]                                |
| Normal (Multivariate) | N$_m$(**_μ_**,**_Ω_**)                    | $m$ dimensions, mean **_μ_**, covariance **_Ω_** |
| Lognormal             | LN($\mu,\sigma^2$)                        | Parameters $\mu, \sigma^2$ of underlying normal |
| Standard Normal PDF   | $\phi(z)$                                 |                                                 |
| Standard Normal CDF   | $\Phi(z)$                                 |                                                 |
| Chi-squared           | $\chi_n^2(\delta)$                        | $n$ degrees of freedom, non-centrality $\delta$ |
| Student's t           | $t_n(\delta)$                             | $n$ degrees of freedom, non-centrality $\delta$ |
| Cauchy                | C($a,b$)                                  | Location $a$, scale $b$                         |
| Fisher's F            | $F_{m,n}(\delta)$                         | $m, n$ degrees of freedom, non-centrality $\delta$ |
| Gamma                 | $\Gamma(\alpha,\lambda)$                  | Shape $\alpha$, rate $\lambda$ (or scale $1/\lambda$) |
| Beta                  | B($a,b$)                                  | Shape parameters $a, b$                         |
| Wiener Process        | W($\tau$) or B($\tau$)                     | Standard Brownian motion on $\tau \in [0,1]$   |

## VI. Abbreviations and Acronyms

Common acronyms like OLS, GLS, MLE, IV, GMM, ARCH, ARMA, i.i.d., p.d.f., c.d.f., etc., are standard. Refer to the original paper or common texts for a comprehensive list.

---
*This markdown file is a restructured summary based on the standards proposed by Abadir & Magnus, using standard Markdown and inline LaTeX formatting via `$` delimiters as supported by GitHub.*