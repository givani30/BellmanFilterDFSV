import numpy as np
from functions.simulation import DFSV_params
from functions.filters import DFSVBellmanFilter

def test_bellman_single_step():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create test parameters
    N = 5  # Number of observed series
    K = 2  # Number of factors
    
    # Initialize model parameters
    params = DFSV_params(
        N=N,
        K=K,
        lambda_r=np.random.normal(0, 1, (N, K)),  # Random factor loadings
        Phi_f=0.9 * np.eye(K),  # AR(1) factor dynamics
        Phi_h=0.95 * np.eye(K),  # Persistent volatility
        mu=np.zeros(K),  # Mean log-volatility
        Q_h=0.1 * np.eye(K),  # Volatility innovation variance
        sigma2=0.1 * np.ones(N)  # Measurement noise variance
    )
    
    # Initialize filter
    bf = DFSVBellmanFilter(params)
    
    # Create some artificial data
    T = 1
    y = np.random.normal(0, 1, (T, N))
    
    # Initialize state
    initial_state, initial_cov = bf.initialize_state(y.T)
    
    print("Initial State:")
    print(initial_state)
    print("\nInitial Covariance:")
    print(initial_cov)
    
    # Prediction step
    predicted_state, predicted_cov = bf.predict(initial_state, initial_cov)
    
    print("\nPredicted State:")
    print(predicted_state)
    print("\nPredicted Covariance:")
    print(predicted_cov)
    
    # Update step
    updated_state, updated_cov, log_likelihood = bf.update(
        predicted_state, predicted_cov, y.T)
    
    print("\nUpdated State:")
    print(updated_state)
    print("\nUpdated Covariance:")
    print(updated_cov)
    print("\nLog-likelihood contribution:")
    print(log_likelihood)
    
    # Print the components separately
    print("\nFactors (first K elements):")
    print(updated_state[:K])
    print("\nLog-volatilities (last K elements):")
    print(updated_state[K:])

if __name__ == "__main__":
    test_bellman_single_step()
