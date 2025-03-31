import numpy as np
import jax.numpy as jnp # Add JAX numpy import
from bellman_filter_dfsv.core.filters.bellman import DFSVBellmanFilter
from bellman_filter_dfsv.core.simulation import simulate_DFSV
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass # Import the JAX dataclass

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
# Create parameter dataclass object using JAX arrays
params = DFSVParamsDataclass(
    N=N, K=K,
    lambda_r=jnp.array(lambda_r),
    Phi_f=jnp.array(Phi_f),
    Phi_h=jnp.array(Phi_h),
    mu=jnp.array(mu),
    sigma2=jnp.array(sigma2), # Keep as 1D for dataclass, filter handles conversion if needed
    Q_h=jnp.array(Q_h)
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
