#!/usr/bin/env python3
# Script to extract and compare key parameter estimates from BIF and PF models for LaTeX table

import numpy as np
import cloudpickle
import os
from rich import print

# Load the BIF model results
print("Loading BIF model results...")
with open('scripts/empirical/insample/bfpf/bif_full_result_20250425_144625.pkl', 'rb') as f:
    result_bif = cloudpickle.load(f)

# Extract BIF parameters
bif_params = result_bif.final_params
print("BIF parameters loaded successfully.")

# Load the PF model results
print("Loading PF model results...")
with open('scripts/empirical/insample/bfpf/pf_full_result_20250425_194312.pkl', 'rb') as f:
    result_pf = cloudpickle.load(f)

# Extract PF parameters
pf_params = result_pf.final_params
print("PF parameters loaded successfully.")

# Extract key parameters for BIF
lambda_hat_bif = np.array(bif_params.lambda_r)
phi_f_hat_bif = np.array(bif_params.Phi_f)
phi_h_hat_bif = np.array(bif_params.Phi_h)
mu_hat_bif = np.array(bif_params.mu)
sigma_eps_hat_diag_bif = np.array(bif_params.sigma2)
Q_h_hat_bif = np.array(bif_params.Q_h)

# Extract key parameters for PF
lambda_hat_pf = np.array(pf_params.lambda_r)
phi_f_hat_pf = np.array(pf_params.Phi_f)
phi_h_hat_pf = np.array(pf_params.Phi_h)
mu_hat_pf = np.array(pf_params.mu)
sigma_eps_hat_diag_pf = np.array(pf_params.sigma2)
Q_h_hat_pf = np.array(pf_params.Q_h)

# Calculate statistics for BIF
phi_f_max_eig_bif = np.max(np.abs(np.linalg.eigvals(phi_f_hat_bif)))
phi_h_max_eig_bif = np.max(np.abs(np.linalg.eigvals(phi_h_hat_bif)))
mu_mean_bif = np.mean(mu_hat_bif)
sigma_eps_min_bif = np.min(sigma_eps_hat_diag_bif)
sigma_eps_max_bif = np.max(sigma_eps_hat_diag_bif)
sigma_eps_range_bif = f"{sigma_eps_min_bif:.4f} - {sigma_eps_max_bif:.4f}"
sigma_eps_mean_bif = np.mean(sigma_eps_hat_diag_bif)
Q_h_diag_bif = np.diag(Q_h_hat_bif)
Q_h_mean_bif = np.mean(Q_h_diag_bif)

# Calculate statistics for PF
phi_f_max_eig_pf = np.max(np.abs(np.linalg.eigvals(phi_f_hat_pf)))
phi_h_max_eig_pf = np.max(np.abs(np.linalg.eigvals(phi_h_hat_pf)))
mu_mean_pf = np.mean(mu_hat_pf)
sigma_eps_min_pf = np.min(sigma_eps_hat_diag_pf)
sigma_eps_max_pf = np.max(sigma_eps_hat_diag_pf)
sigma_eps_range_pf = f"{sigma_eps_min_pf:.4f} - {sigma_eps_max_pf:.4f}"
sigma_eps_mean_pf = np.mean(sigma_eps_hat_diag_pf)
Q_h_diag_pf = np.diag(Q_h_hat_pf)
Q_h_mean_pf = np.mean(Q_h_diag_pf)

# Print results in a format suitable for the LaTeX table
print("\n=== Key Parameter Estimates Comparison ===")
print(f"Parameter                   | BIF Value      | PF Value")
print(f"------------------------------|----------------|----------------")
print(f"Max Eigenvalue(Phi_f)        | {phi_f_max_eig_bif:.4f}         | {phi_f_max_eig_pf:.4f}")
print(f"Max Eigenvalue(Phi_h)        | {phi_h_max_eig_bif:.4f}         | {phi_h_max_eig_pf:.4f}")
print(f"Mean(mu)                     | {mu_mean_bif:.4f}         | {mu_mean_pf:.4f}")
print(f"Range(diag(Sigma_epsilon))   | {sigma_eps_range_bif} | {sigma_eps_range_pf}")
print(f"Mean(diag(Sigma_epsilon))    | {sigma_eps_mean_bif:.4f}         | {sigma_eps_mean_pf:.4f}")
print(f"Mean(diag(Q_h))              | {Q_h_mean_bif:.4f}         | {Q_h_mean_pf:.4f}")

# Print in LaTeX format
print("\n=== LaTeX Table Format (Side-by-Side Comparison) ===")
print(r"% Table 5.4: Comparison of Key Parameter Estimates (BIF vs PF)")
print(r"\begin{table}[htbp]")
print(r"  \centering")
print(r"  \caption{Comparison of Key Parameter Estimates (DFSV-BIF vs DFSV-PF)}")
print(r"  \label{tab:empirical_key_params_comparison}")
print(r"  \begin{tabular}{lcc}")
print(r"    \toprule")
print(r"    Parameter Description & BIF Value & PF Value \\")
print(r"    \midrule")
print(f"    Max Eigenvalue($\\hat{{\\mPhi}}_f$) & {phi_f_max_eig_bif:.4f} & {phi_f_max_eig_pf:.4f} \\\\")
print(f"    Max Eigenvalue($\\hat{{\\mPhi}}_h$) & {phi_h_max_eig_bif:.4f} & {phi_h_max_eig_pf:.4f} \\\\")
print(f"    Mean($\\hat{{\\vmu}}$) & {mu_mean_bif:.4f} & {mu_mean_pf:.4f} \\\\")
print(f"    Range(diag($\\hat{{\\mSigma}}_\\epsilon$)) & {sigma_eps_range_bif} & {sigma_eps_range_pf} \\\\")
print(f"    Mean(diag($\\hat{{\\mSigma}}_\\epsilon$)) & {sigma_eps_mean_bif:.4f} & {sigma_eps_mean_pf:.4f} \\\\")
print(f"    Mean(diag($\\hat{{\\mQ}}_h$)) & {Q_h_mean_bif:.4f} & {Q_h_mean_pf:.4f} \\\\")
print(r"    \bottomrule")
print(r"  \end{tabular}")
print(r"  \par\medskip \footnotesize Note: Comparison of key parameter estimates between DFSV-BIF and DFSV-PF models. Eigenvalues indicate stationarity. $\hat{\vmu}$ represents the unconditional mean of log-volatilities. $\hat{\mSigma}_\epsilon$ captures idiosyncratic variances, and $\hat{\mQ}_h$ captures the volatility of volatilities.")
print(r"\end{table}")
