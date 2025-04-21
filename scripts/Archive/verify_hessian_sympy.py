# scripts/verify_hessian_sympy.py
import sympy
from sympy import symbols, MatrixSymbol, Matrix, Function, diag, Derivative, trace, KroneckerDelta, simplify, expand, init_printing, assuming, Q

# Use init_printing for better display in environments like Jupyter
# init_printing(use_unicode=True)

print("--- SymPy Verification Script for DFSV Hessian Blocks ---")

# --- Dimensions ---
# Using small concrete dimensions for potential evaluation, though derivation is symbolic
N_val, K_val = 3, 2
N = symbols('N', integer=True, positive=True)
K = symbols('K', integer=True, positive=True)


# --- Indices ---
i, j, k, l = symbols('i j k l', integer=True) # Using generic indices

# --- Parameters and Variables ---
# Use Matrix instead of MatrixSymbol for element access if needed for differentiation
f_vec = sympy.Matrix([symbols(f'f_{idx}') for idx in range(K_val)])
h_vec = sympy.Matrix([symbols(f'h_{idx}') for idx in range(K_val)])
y_vec = sympy.Matrix([symbols(f'y_{idx}') for idx in range(N_val)])

# --- Model Matrices ---
# Define symbolic matrices with fixed dimensions for this example
Lambda_mat = MatrixSymbol('Lambda', N_val, K_val)
Sigma_mat = MatrixSymbol('Sigma', N_val, N_val) # Assume symmetric: Sigma_mat.T = Sigma_mat

# Define D_h = diag(exp(h))
exp_h = Matrix([sympy.exp(hk) for hk in h_vec])
D_h = diag(*exp_h)

# Covariance Matrix A(h)
A = Lambda_mat * D_h * Lambda_mat.T + Sigma_mat
# Assume A is invertible and symmetric
# Represent A_inv as a symbolic inverse for derivation purposes
A_inv = A.inverse() # Sympy can handle symbolic inverse

# --- Log-Likelihood (Ignoring constant C) ---
# Use sympy.log(A.det()) for log determinant
logdet_A = sympy.log(A.det())
y_mf = y_vec - Lambda_mat * f_vec
# Ensure term2 is treated as a scalar for differentiation
term2_expr = (y_mf).T * A_inv * (y_mf)
# Extract the scalar element directly from the 1x1 matrix expression
term2 = term2_expr[0,0]

log_likelihood = -sympy.Rational(1, 2) * logdet_A - sympy.Rational(1, 2) * term2

print("\nLog-Likelihood Expression (Symbolic):")
# Use sympy.pretty for better console printing if init_printing is not active
print(sympy.pretty(log_likelihood))
print("-" * 60)

# --- Define Helper Variables Symbolically ---
# These represent the terms in the derived formulas
q_mat_sym = Lambda_mat.T * A_inv * Lambda_mat
p_vec_sym = Lambda_mat.T * A_inv * y_mf

print("\nHelper Variable Definitions (Symbolic):")
print("q = Lambda^T * A^-1 * Lambda")
print(sympy.pretty(q_mat_sym))
print("\np = Lambda^T * A^-1 * (y - Lambda*f)")
print(sympy.pretty(p_vec_sym))
print("-" * 60)


# --- Compute Derivatives (Element-wise) ---
# Note: Full symbolic matrix differentiation and simplification is computationally
# intensive and often requires manual guidance based on known matrix identities.
# This script focuses on setting up the expressions and stating the targets derived manually.

# Select specific elements/indices for demonstration (e.g., k=0, l=1, i=0)
k_idx, l_idx, i_idx = 0, 1, 0
f_i = f_vec[i_idx]
h_k = h_vec[k_idx]
h_l = h_vec[l_idx]

# 1. Derivative d(log_likelihood) / df_i
print(f"\nCalculating d(log_likelihood) / df_{i_idx}...")
dl_dfi = Derivative(log_likelihood, f_i).doit()
# Target: p_vec_sym[i_idx] (Manual verification needed for simplification)
print(f"Symbolic Result (needs simplification):")
print(sympy.pretty(dl_dfi))
print(f"Target: p[{i_idx}]")


# 2. Derivative d(log_likelihood) / dh_k
print(f"\nCalculating d(log_likelihood) / dh_{k_idx}...")
dl_dhk = Derivative(log_likelihood, h_k).doit()
# Target: 1/2 * exp(h_k) * (p_vec_sym[k_idx]^2 - q_mat_sym[k_idx, k_idx]) (Manual verification needed)
print(f"Symbolic Result (needs simplification):")
print(sympy.pretty(dl_dhk))
print(f"Target: 1/2 * exp(h_{k_idx}) * (p[{k_idx}]^2 - q[{k_idx},{k_idx}])")


# 3. Second Derivative d^2(log_likelihood) / (df_i dh_k) -> H_fh[i, k]
print(f"\nCalculating d^2(log_likelihood) / (df_{i_idx} dh_{k_idx})...")
# Use the calculated dl_dfi for efficiency if simplification was possible
d2l_dfidhk = Derivative(log_likelihood, f_i, h_k).doit()
print(f"Symbolic Result (needs simplification):")
print(sympy.pretty(d2l_dfidhk))
# Target H_fh[i, k]: -exp(h_k) * q_mat_sym[i, k] * p_vec_sym[k]
# Target J_fh[i, k]: +exp(h_k) * q_mat_sym[i, k] * p_vec_sym[k]
print(f"Target J_fh[{i_idx},{k_idx}]: exp(h_{k_idx}) * q[{i_idx},{k_idx}] * p[{k_idx}]")


# 4. Second Derivative d^2(log_likelihood) / (dh_k dh_l) -> H_hh[k, l]
print(f"\nCalculating d^2(log_likelihood) / (dh_{k_idx} dh_{l_idx})...")
# Use the calculated dl_dhk for efficiency if simplification was possible
d2l_dhkdhl = Derivative(log_likelihood, h_k, h_l).doit()
print(f"Symbolic Result (needs simplification):")
print(sympy.pretty(d2l_dhkdhl))
# Target J_hh[k, l]: 1/2*δ_kl*e^hk*(q_kk-p_k^2) - 1/2*e^hk*e^hl*q_kl*(q_kl-2*p_k*p_l)
print(f"Target J_hh[{k_idx},{l_idx}]: 1/2*δ({k_idx},{l_idx})*exp(h_{k_idx})*(q[{k_idx},{k_idx}]-p[{k_idx}]^2) - 1/2*exp(h_{k_idx})*exp(h_{l_idx})*q[{k_idx},{l_idx}]*(q[{k_idx},{l_idx}]-2*p[{k_idx}]*p[{l_idx}])")

print("-" * 60)
print("\nNOTE: This script demonstrates setting up the symbolic derivatives.")
print("      Directly simplifying the results from .doit() to match the target")
print("      formulas involving 'p' and 'q' requires manual application of")
print("      matrix derivative identities and substitutions, as shown in the")
print("      mathematical derivation.")
print("-" * 60)

# --- Explicit Target Formulas (using symbolic p's and q's for clarity) ---
q_ik_sym = symbols(f'q_{i_idx}{k_idx}')
q_kk_sym = symbols(f'q_{k_idx}{k_idx}')
q_kl_sym = symbols(f'q_{k_idx}{l_idx}')
p_k_sym = symbols(f'p_{k_idx}')
p_l_sym = symbols(f'p_{l_idx}')
exp_hk_sym = sympy.exp(h_k)
exp_hl_sym = sympy.exp(h_l)
delta_kl_sym = KroneckerDelta(k_idx, l_idx) # Evaluates to 0 since k_idx != l_idx

target_J_fh_ik = exp_hk_sym * q_ik_sym * p_k_sym
target_J_hh_kl = (
    sympy.Rational(1, 2) * delta_kl_sym * exp_hk_sym * (q_kk_sym - p_k_sym**2) -
    sympy.Rational(1, 2) * exp_hk_sym * exp_hl_sym * q_kl_sym * (q_kl_sym - 2 * p_k_sym * p_l_sym)
)

print("\nTarget Formulas (using symbolic p, q for comparison):")
print(f"Target J_fh[{i_idx},{k_idx}] = {sympy.pretty(target_J_fh_ik)}")
print(f"Target J_hh[{k_idx},{l_idx}] = {sympy.pretty(target_J_hh_kl)}")
print("-" * 60)