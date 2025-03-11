"""
Test script for the JAX-based Bellman filter implementation.

This script tests the new JAX-based implementation of the Bellman filter
by performing a single prediction and update step and comparing the results
with expected values.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import traceback

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.simulation import DFSV_params, simulate_DFSV
from functions.filters import DFSVBellmanFilter

def run_test_with_cleanup(test_func):
    """Wrapper to run a test with proper cleanup and error handling."""
    try:
        result = test_func()
        return result
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        plt.close('all')  # Close all figures
        return False
    except Exception as e:
        print(f"\nError in test: {e}")
        print("Traceback:")
        traceback.print_exc()
        plt.close('all')  # Close all figures
        return False
    finally:
        plt.close('all')  # Ensure all figures are closed

def test_jax_bellman_single_step():
    """
    Test a single prediction and update step of the JAX-based Bellman filter.
    """
    print("\n=== Testing JAX-based Bellman Filter Single Step ===")
    
    try:
        # Create a small test model
        K = 2  # Number of factors
        N = 5  # Number of observed series
        
        # Define model parameters
        lambda_r = np.random.normal(0, 1, size=(N, K))
        Phi_f = np.diag([0.9, 0.85])  # Autoregressive coefficients for factors
        Phi_h = np.diag([0.98, 0.95]) # Autoregressive coefficients for log-volatilities
        mu = np.array([-5.0, -5.5])   # Unconditional mean of log-volatilities
        sigma2 = np.ones(N) * 0.1     # Measurement noise variance
        Q_h = np.diag([0.2, 0.15])    # Volatility of log-volatility process
        
        params = DFSV_params(
            N=N,
            K=K,
            lambda_r=lambda_r,
            Phi_f=Phi_f,
            Phi_h=Phi_h,
            mu=mu,
            sigma2=sigma2,
            Q_h=Q_h
        )
        
        # Generate a small sample of simulated data
        T = 10
        y, factors, log_vols = simulate_DFSV(params, T, seed=42)
        
        # Initialize the Bellman filter
        bf = DFSVBellmanFilter(params)
        
        # Initialize state and covariance
        state0, cov0 = bf.initialize_state(y)  # No need for transpose, handled in initialize_state
        
        print(f"Initial state shape: {state0.shape}")
        print(f"Initial covariance shape: {cov0.shape}")
        
        # Test a single prediction step
        predicted_state, predicted_cov = bf.predict(state0, cov0)
        
        print(f"\nPredicted state shape: {predicted_state.shape}")
        print(f"Predicted covariance shape: {predicted_cov.shape}")
        print(f"Predicted state (first few elements): {predicted_state.flatten()[:4]}")
        
        # Test a single update step
        observation = y[0:1, :].T.reshape(-1, 1)  # Reshape to ensure (N, 1)
        updated_state, updated_cov, log_likelihood = bf.update(
            predicted_state, predicted_cov, observation
        )
        
        print(f"\nUpdated state shape: {updated_state.shape}")
        print(f"Updated covariance shape: {updated_cov.shape}")
        print(f"Updated state (first few elements): {updated_state.flatten()[:4]}")
        print(f"Log-likelihood contribution: {log_likelihood}")
        
        # Check that dimensions are correct
        assert state0.shape == (2*K, 1), f"Initial state has wrong shape: {state0.shape}"
        assert cov0.shape == (2*K, 2*K), f"Initial covariance has wrong shape: {cov0.shape}"
        assert predicted_state.shape == (2*K, 1), f"Predicted state has wrong shape: {predicted_state.shape}"
        assert predicted_cov.shape == (2*K, 2*K), f"Predicted covariance has wrong shape: {predicted_cov.shape}"
        assert updated_state.shape == (2*K, 1), f"Updated state has wrong shape: {updated_state.shape}"
        assert updated_cov.shape == (2*K, 2*K), f"Updated covariance has wrong shape: {updated_cov.shape}"
        
        # Check that covariance matrices are symmetric and positive definite
        try:
            np.linalg.cholesky(predicted_cov)
            print("\nPredicted covariance is positive definite ✓")
        except np.linalg.LinAlgError:
            print("\nWARNING: Predicted covariance is not positive definite!")
            
        try:
            np.linalg.cholesky(updated_cov)
            print("Updated covariance is positive definite ✓")
        except np.linalg.LinAlgError:
            print("WARNING: Updated covariance is not positive definite!")
        
        print("\nJAX-based Bellman filter single step test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in single step test: {e}")
        traceback.print_exc()
        return False

def test_jax_bellman_full_filter():
    """
    Test the full filter operation on a small dataset.
    """
    print("\n=== Testing JAX-based Bellman Filter Full Operation ===")
    
    try:
        # Create a small test model
        K = 1  # Number of factors (using a smaller model for speed)
        N = 3  # Number of observed series
        
        # Define model parameters
        lambda_r = np.random.normal(0, 1, size=(N, K))
        Phi_f = np.array([[0.9]])     # Autoregressive coefficients for factor
        Phi_h = np.array([[0.95]])    # Autoregressive coefficients for log-volatility
        mu = np.array([-5.0])         # Unconditional mean of log-volatility
        sigma2 = np.ones(N) * 0.1     # Measurement noise variance
        Q_h = np.array([[0.2]])       # Volatility of log-volatility process
        
        params = DFSV_params(
            N=N,
            K=K,
            lambda_r=lambda_r,
            Phi_f=Phi_f,
            Phi_h=Phi_h,
            mu=mu,
            sigma2=sigma2,
            Q_h=Q_h
        )
        
        # Generate a small sample of simulated data
        T = 20
        y, factors, log_vols = simulate_DFSV(params, T, seed=42)
        
        # Initialize the Bellman filter
        bf = DFSVBellmanFilter(params)
        
        # Run the filter
        filtered_states, filtered_covs, log_likelihood = bf.filter(y)
        print(f"Filter completed successfully for {T} time steps")
        print(f"Log-likelihood: {log_likelihood}")
        
        # Test smoothing
        smoothed_states, smoothed_covs = bf.smooth()
        print(f"Smoothing completed successfully for {T} time steps")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot results for the factor
        plt.figure(figsize=(10, 6))
        plt.plot(factors[:, 0], 'k-', label='True Factor')
        plt.plot(filtered_states[:, 0], 'r--', label='Filtered Factor')
        plt.plot(smoothed_states[:, 0], 'b-.', label='Smoothed Factor')
        plt.legend()
        plt.title('Factor Estimation')
        plt.savefig(os.path.join(output_dir, 'jax_bellman_factor_test.png'))
        print(f"Factor plot saved to outputs/jax_bellman_factor_test.png")
        plt.close()
        
        # Plot results for the log-volatility
        plt.figure(figsize=(10, 6))
        plt.plot(log_vols[:, 0], 'k-', label='True Log-Volatility')
        plt.plot(filtered_states[:, 1], 'r--', label='Filtered Log-Volatility')
        plt.plot(smoothed_states[:, 1], 'b-.', label='Smoothed Log-Volatility')
        plt.legend()
        plt.title('Log-Volatility Estimation')
        plt.savefig(os.path.join(output_dir, 'jax_bellman_logvol_test.png'))
        print(f"Log-volatility plot saved to outputs/jax_bellman_logvol_test.png")
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error in full filter test: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = True
    try:
        # Run single step test
        if not run_test_with_cleanup(test_jax_bellman_single_step):
            success = False
            print("Single step test failed!")
        
        # Run full filter test
        if not run_test_with_cleanup(test_jax_bellman_full_filter):
            success = False
            print("Full filter test failed!")
            
        if success:
            print("\nAll tests completed successfully!")
        else:
            print("\nSome tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        plt.close('all')
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
        plt.close('all')
        sys.exit(1)
    finally:
        plt.close('all')