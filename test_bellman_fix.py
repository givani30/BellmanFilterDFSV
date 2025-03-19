import numpy as np
from functions.bellman_filter import DFSVBellmanFilter
from functions.simulation import DFSV_params, simulate_DFSV

# Create a small test model for testing
N = 3  # Number of observed series
K = 1  # Number of factors

# Define model parameters
np.random.seed(42)
lambda_r = np.random.normal(0, 1, size=(N, K))
Phi_f = np.array([[0.9]])  # Autoregressive coefficients for factor
Phi_h = np.array([[0.95]])  # Autoregressive coefficients for log-volatility
mu = np.array([-5.0])  # Unconditional mean of log-volatility
sigma2 = np.ones(N) * 0.1  # Measurement noise variance
Q_h = np.array([[0.2]])  # Volatility of log-volatility process

# Create parameter object
params = DFSV_params(
    N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, Phi_h=Phi_h, mu=mu, sigma2=sigma2, Q_h=Q_h
)

print("Model parameters created successfully")

# Generate a small amount of synthetic data
T = 20  # Just a few time points for testing
y, factors, log_vols = simulate_DFSV(params, T=T, seed=42)

print("Data generated:")
print(f"y shape: {y.shape}")
print(f"factors shape: {factors.shape}")
print(f"log_vols shape: {log_vols.shape}")

# Initialize the filter
print("Initializing filter...")
bf = DFSVBellmanFilter(N, K)
print("Filter initialized successfully")

# Run the filter
print("Running filter...")
filtered_states, filtered_covs, log_likelihood = bf.filter(params, y)

print("Filter completed successfully!")
print(f"Filtered states shape: {filtered_states.shape}")
print(f"Filtered covariances shape: {filtered_covs.shape}")
print(f"Log-likelihood: {log_likelihood:.4f}")
