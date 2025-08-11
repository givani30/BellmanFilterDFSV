#!/usr/bin/env python3
# Script to extract key parameter estimates from BIF model for LaTeX table

import numpy as np
import cloudpickle
import os
from rich import print

# Load the BIF model results
print("Loading BIF model results...")
with open('scripts/empirical/insample/bfpf/bif_full_result_20250425_144625.pkl', 'rb') as f:
    result_bif = cloudpickle.load(f)

# Extract parameters
bif_params = result_bif.final_params
print("Parameters loaded successfully.")

# Extract key parameters
lambda_hat_bif = np.array(bif_params.lambda_r)
phi_f_hat_bif = np.array(bif_params.Phi_f)
phi_h_hat_bif = np.array(bif_params.Phi_h)
mu_hat_bif = np.array(bif_params.mu)
sigma_eps_hat_diag_bif = np.array(bif_params.sigma2)
Q_h_hat_bif = np.array(bif_params.Q_h)

# Calculate statistics
# 1. Max eigenvalue of Phi_f
phi_f_max_eig = np.max(np.abs(np.linalg.eigvals(phi_f_hat_bif)))

# 2. Max eigenvalue of Phi_h
phi_h_max_eig = np.max(np.abs(np.linalg.eigvals(phi_h_hat_bif)))

# 3. Mean of mu
mu_mean = np.mean(mu_hat_bif)

# 4. Range of diagonal elements of Sigma_epsilon
sigma_eps_min = np.min(sigma_eps_hat_diag_bif)
sigma_eps_max = np.max(sigma_eps_hat_diag_bif)
sigma_eps_range = f"{sigma_eps_min:.4f} - {sigma_eps_max:.4f}"

# 5. Mean of diagonal elements of Sigma_epsilon
sigma_eps_mean = np.mean(sigma_eps_hat_diag_bif)

# 6. Mean of diagonal elements of Q_h
Q_h_diag = np.diag(Q_h_hat_bif)
Q_h_mean = np.mean(Q_h_diag)

# Print results in a format suitable for the LaTeX table
print("\n=== Key Parameter Estimates (BIF) ===")
print(f"Max Eigenvalue(Phi_f): {phi_f_max_eig:.4f}")
print(f"Max Eigenvalue(Phi_h): {phi_h_max_eig:.4f}")
print(f"Mean(mu): {mu_mean:.4f}")
print(f"Range(diag(Sigma_epsilon)): {sigma_eps_range}")
print(f"Mean(diag(Sigma_epsilon)): {sigma_eps_mean:.4f}")
print(f"Mean(diag(Q_h)): {Q_h_mean:.4f}")

# Print in LaTeX format
print("\n=== LaTeX Table Format ===")
print(r"% Table 5.2: Summary of Key Parameter Estimates (BIF)")
print(r"\begin{table}[htbp]")
print(r"  \centering")
print(r"  \caption{Summary of Key Parameter Estimates (DFSV-BIF)}")
print(r"  \label{tab:empirical_key_params_bif}")
print(r"  \begin{tabular}{lc}")
print(r"    \toprule")
print(r"    Parameter Description & Value / Range \\")
print(r"    \midrule")
print(f"    Max Eigenvalue($\\hat{{\\mPhi}}_f$) & {phi_f_max_eig:.4f} \\\\")
print(f"    Max Eigenvalue($\\hat{{\\mPhi}}_h$) & {phi_h_max_eig:.4f} \\\\")
print(f"    Mean($\\hat{{\\vmu}}$) & {mu_mean:.4f} \\\\")
print(f"    Range(diag($\\hat{{\\mSigma}}_\\epsilon$)) & {sigma_eps_range} \\\\")
print(f"    Mean(diag($\\hat{{\\mSigma}}_\\epsilon$)) & {sigma_eps_mean:.4f} \\\\")
print(f"    Mean(diag($\\hat{{\\mQ}}_h$)) & {Q_h_mean:.4f} \\\\")
print(r"    \bottomrule")
print(r"  \end{tabular}")
print(r"  \par\medskip \footnotesize Note: Summary statistics for key estimated parameters of the DFSV-BIF model. Eigenvalues indicate stationarity. $\hat{\vmu}$ represents the unconditional mean of log-volatilities. $\hat{\mSigma}_\epsilon$ captures idiosyncratic variances, and $\hat{\mQ}_h$ captures the volatility of volatilities.")
print(r"\end{table}")
