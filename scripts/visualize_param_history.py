#!/usr/bin/env python
"""
Visualize parameter history during optimization.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cloudpickle
import jax.numpy as jnp
from pathlib import Path

def load_param_history(filepath):
    """Load parameter history from a pickle file."""
    with open(filepath, 'rb') as f:
        return cloudpickle.load(f)

def load_true_params(filepath):
    """Load true parameters from a pickle file."""
    with open(filepath, 'rb') as f:
        return cloudpickle.load(f)

def extract_param_values(param_history):
    """Extract parameter values from parameter history."""
    # Initialize dictionaries to store parameter values
    param_values = {
        'lambda_r': [],
        'Phi_f': [],
        'Phi_h': [],
        'mu': [],
        'sigma2': [],
        'Q_h': []
    }
    
    # Extract parameter values from each step
    for params in param_history:
        param_values['lambda_r'].append(params.lambda_r)
        param_values['Phi_f'].append(params.Phi_f)
        param_values['Phi_h'].append(params.Phi_h)
        param_values['mu'].append(params.mu)
        param_values['sigma2'].append(params.sigma2)
        param_values['Q_h'].append(params.Q_h)
    
    return param_values

def plot_parameter_history(param_values, true_params=None, output_dir='outputs/param_history_plots'):
    """Plot parameter history."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot lambda_r
    lambda_r_values = np.array(param_values['lambda_r'])
    N, K = lambda_r_values[0].shape
    fig, axes = plt.subplots(N, K, figsize=(4*K, 3*N))
    for i in range(N):
        for j in range(K):
            ax = axes[i, j] if N > 1 else axes[j]
            ax.plot(lambda_r_values[:, i, j])
            if true_params is not None:
                ax.axhline(y=true_params.lambda_r[i, j], color='r', linestyle='--')
            ax.set_title(f'lambda_r[{i},{j}]')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lambda_r_history.png'))
    plt.close()
    
    # Plot Phi_f
    Phi_f_values = np.array(param_values['Phi_f'])
    K = Phi_f_values[0].shape[0]
    fig, axes = plt.subplots(K, K, figsize=(4*K, 3*K))
    for i in range(K):
        for j in range(K):
            ax = axes[i, j] if K > 1 else axes
            ax.plot(Phi_f_values[:, i, j])
            if true_params is not None:
                ax.axhline(y=true_params.Phi_f[i, j], color='r', linestyle='--')
            ax.set_title(f'Phi_f[{i},{j}]')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Phi_f_history.png'))
    plt.close()
    
    # Plot Phi_h
    Phi_h_values = np.array(param_values['Phi_h'])
    K = Phi_h_values[0].shape[0]
    fig, axes = plt.subplots(K, K, figsize=(4*K, 3*K))
    for i in range(K):
        for j in range(K):
            ax = axes[i, j] if K > 1 else axes
            ax.plot(Phi_h_values[:, i, j])
            if true_params is not None:
                ax.axhline(y=true_params.Phi_h[i, j], color='r', linestyle='--')
            ax.set_title(f'Phi_h[{i},{j}]')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Phi_h_history.png'))
    plt.close()
    
    # Plot mu
    mu_values = np.array(param_values['mu'])
    K = mu_values[0].shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(K):
        ax.plot(mu_values[:, i], label=f'mu[{i}]')
        if true_params is not None:
            ax.axhline(y=true_params.mu[i], color='r', linestyle='--')
    ax.set_title('mu')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mu_history.png'))
    plt.close()
    
    # Plot sigma2
    sigma2_values = np.array(param_values['sigma2'])
    N = sigma2_values[0].shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(N):
        ax.plot(sigma2_values[:, i], label=f'sigma2[{i}]')
        if true_params is not None:
            ax.axhline(y=true_params.sigma2[i], color='r', linestyle='--')
    ax.set_title('sigma2')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sigma2_history.png'))
    plt.close()
    
    # Plot Q_h
    Q_h_values = np.array(param_values['Q_h'])
    K = Q_h_values[0].shape[0]
    fig, axes = plt.subplots(K, K, figsize=(4*K, 3*K))
    for i in range(K):
        for j in range(K):
            ax = axes[i, j] if K > 1 else axes
            ax.plot(Q_h_values[:, i, j])
            if true_params is not None:
                ax.axhline(y=true_params.Q_h[i, j], color='r', linestyle='--')
            ax.set_title(f'Q_h[{i},{j}]')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Q_h_history.png'))
    plt.close()

def main():
    """Main function."""
    # Check if a parameter history file is provided
    if len(sys.argv) < 2:
        print("Usage: python visualize_param_history.py <param_history_file>")
        sys.exit(1)
    
    # Load parameter history
    param_history_file = sys.argv[1]
    print(f"Loading parameter history from {param_history_file}...")
    param_history = load_param_history(param_history_file)
    
    # Load true parameters if available
    true_params_file = os.path.join("outputs", "true_params.pkl")
    true_params = None
    if os.path.exists(true_params_file):
        print(f"Loading true parameters from {true_params_file}...")
        true_params = load_true_params(true_params_file)
    
    # Extract parameter values
    print("Extracting parameter values...")
    param_values = extract_param_values(param_history)
    
    # Plot parameter history
    print("Plotting parameter history...")
    output_dir = os.path.join("outputs", "param_history_plots", Path(param_history_file).stem)
    plot_parameter_history(param_values, true_params, output_dir)
    
    print(f"Parameter history plots saved to {output_dir}")

if __name__ == "__main__":
    main()
