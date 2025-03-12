import numpy as np
import jax
import jax.numpy as jnp
from functions.simulation import DFSV_params, simulate_DFSV
from functions.filters import DFSVBellmanFilter

# Set random seed for reproducibility
np.random.seed(42)

# Create a minimal test model
K = 1  # Number of factors
N = 2  # Number of observed series

# Define model parameters
lambda_r = np.array([[1.0], [0.5]])  # Simple factor loadings
Phi_f = np.array([[0.9]])  # Factor persistence
Phi_h = np.array([[0.95]])  # Volatility persistence
mu = np.array([-1.0])  # Log-volatility mean
sigma2 = np.ones(N) * 0.1  # Measurement noise
Q_h = np.array([[0.2]])  # Volatility of log-volatility

# Create parameter object
params = DFSV_params(
    N=N, K=K, lambda_r=lambda_r, Phi_f=Phi_f, Phi_h=Phi_h, mu=mu, sigma2=sigma2, Q_h=Q_h
)

print("Model parameters created successfully")
# Generate synthetic data
T = 20  # Just a few time points for testing
y, factors, log_vols = simulate_DFSV(params, T=T, seed=42)

print("Data generated:")
print(f"y shape: {y.shape}")
print(f"factors shape: {factors.shape}")
print(f"log_vols shape: {log_vols.shape}")
# Initialize the filter
try:
    bf = DFSVBellmanFilter(params)
    print("\nBellman filter initialized")
    print("\nJAX parameters:")
    print(f"lambda_r shape: {bf.jax_lambda_r.shape}")
    print(f"sigma2 shape: {bf.jax_sigma2.shape}")
    print(f"mu shape: {bf.jax_mu.shape}")
    print(f"Phi_f shape: {bf.jax_Phi_f.shape}")
    print(f"Phi_h shape: {bf.jax_Phi_h.shape}")
    print(f"Q_h shape: {bf.jax_Q_h.shape}")
except Exception as e:
    print(f"Error during initialization: {e}")
    # Initialize state and test prediction step
try:
    # Initialize state
    state0, cov0 = bf.initialize_state(y)
    print(f"Initial state shape: {state0.shape}")
    print(f"Initial covariance shape: {cov0.shape}")
    print(f"\nInitial state:\n{state0}")

    # Test prediction step
    predicted_state, predicted_cov = bf.predict(state0, cov0)
    print("\nPrediction step completed")
    print(f"Predicted state:\n{predicted_state}")

    # Verify covariance matrices
    try:
        np.linalg.cholesky(predicted_cov)
        print("\nPredicted covariance is positive definite ✓")
    except np.linalg.LinAlgError:
        print("\nWARNING: Predicted covariance is not positive definite!")
        print("Eigenvalues:", np.linalg.eigvals(predicted_cov))
except Exception as e:
    print(f"Error during prediction: {e}")
logdir = "./jax-trace"
jax.profiler.start_trace(log_dir=logdir)
observation = y[0:1, :].T.reshape(-1, 1)
print(f"Observation shape: {observation.shape}")
print(f"Observation values:\n{observation}")

# Check JAX objective function
print("\nTesting JAX objective function...")
alpha_test = predicted_state.copy()
alpha_test[0] = 0.7
# Convert inputs for JAX
jax_alpha = jnp.array(alpha_test)
jax_pred = jnp.array(predicted_state)
jax_I_pred = jnp.array(np.linalg.inv(predicted_cov))
jax_obs = jnp.array(observation)

# Test objective function with new parameter order (removed K, N params)
obj_val, grad_val = bf.obj_and_grad_fn(
    jax_alpha, jax_pred, jax_I_pred, jax_obs, bf.jax_lambda_r, bf.jax_sigma2
)
print(f"Objective value at predicted state: {float(obj_val)}")

# Test gradient with new parameter order
print(f"Gradient at predicted state:\n{np.array(grad_val)}")
# Perform update step
print("\nPerforming update step...")

updated_state, updated_cov, log_likelihood = bf.update(
    predicted_state, predicted_cov, observation
)


print("\nUpdate step completed")
print(f"Updated state:\n{updated_state}")
print(f"Log-likelihood: {log_likelihood}")

# Verify updated covariance
try:
    np.linalg.cholesky(updated_cov)
    print("\nUpdated covariance is positive definite ✓")
except np.linalg.LinAlgError:
    print("\nWARNING: Updated covariance is not positive definite!")
    print("Eigenvalues:", np.linalg.eigvals(updated_cov))
jnp.array(updated_state).block_until_ready()
jax.profiler.stop_trace()
with jax.profiler.trace("/tmp/jax-trace"):
    print("Running full filter...")
    filtered_states, filtered_covs, log_likelihood = bf.filter(y)
    print("\nFilter completed successfully!")
    print(f"Total log-likelihood: {log_likelihood}")
    print(f"\nFiltered states shape: {filtered_states.shape}")
    print(f"Filtered covs shape: {filtered_covs.shape}")

    # Compare with true states
    print("\nCorrelation with true states:")
    factor_corr = jnp.corrcoef(filtered_states[:, 0], factors[:, 0])[0, 1]
    vol_corr = jnp.corrcoef(filtered_states[:, 1], log_vols[:, 0])[0, 1]
    print(f"Factor correlation: {factor_corr:.4f}")
    print(f"Log-volatility correlation: {vol_corr:.4f}")
    vol_corr.block_until_ready()
