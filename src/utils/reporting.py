"""
Reporting utilities for DFSV models.

This module provides standardized utilities for reporting and saving
optimization results for DFSV models.
"""

import os
import csv
import cloudpickle
import numpy as np
import jax.numpy as jnp
from typing import List, Dict, Any, Optional, Union
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.dfsv import DFSVParamsDataclass
from src.utils.optimization import OptimizerResult


def print_parameter_comparison(results: List[OptimizerResult], 
                              true_params: DFSVParamsDataclass):
    """Print comparison of estimated parameters with true parameters.
    
    Args:
        results: List of optimization results.
        true_params: True model parameters.
    """
    param_names = ["lambda_r", "Phi_f", "Phi_h", "mu", "sigma2", "Q_h"]
    
    print("\n--- Parameter Estimation Comparison ---")
    
    for res in sorted(results, key=lambda x: (x.filter_type.name, x.optimizer_name, x.uses_transformations, x.fix_mu)):
        if res.final_params is not None:
            print(f"\n-- Run: Filter='{res.filter_type.name}' | Optimizer='{res.optimizer_name}' | Fix_mu='{'Yes' if res.fix_mu else 'No'}' | Success='{'Yes' if res.success else 'No'}' --")
            print("-" * 80)
            print(f"{'Parameter':<10} | {'True Value':<35} | {'Estimated Value'}")
            print("-" * 80)
            
            for name in param_names:
                true_val = getattr(true_params, name)
                est_val = getattr(res.final_params, name)
                
                # Convert to numpy for consistent printing
                true_val_np = np.asarray(true_val)
                est_val_np = np.asarray(est_val)
                
                # Format for printing (handle multi-line arrays)
                true_str_lines = str(true_val_np).split('\n')
                est_str_lines = str(est_val_np).split('\n')
                
                # Print first line with parameter name
                print(f"{name:<10} | {true_str_lines[0]:<35} | {est_str_lines[0]}")
                
                # Print subsequent lines aligned
                max_lines = max(len(true_str_lines), len(est_str_lines))
                for i in range(1, max_lines):
                    true_line = true_str_lines[i] if i < len(true_str_lines) else ""
                    est_line = est_str_lines[i] if i < len(est_str_lines) else ""
                    print(f"{'':<10} | {true_line:<35} | {est_line}")
            
            print("-" * 80)
        else:
            print(f"\n-- Run: Filter='{res.filter_type.name}' | Optimizer='{res.optimizer_name}' --")
            print("  No final parameters available for comparison (likely failed early).")


def save_results_to_csv(results: List[OptimizerResult], 
                       filename: str = "filter_optimization_results.csv",
                       output_dir: str = "outputs"):
    """Save optimization results to a CSV file.
    
    Args:
        results: List of optimization results.
        filename: Name of the CSV file to save.
        output_dir: Directory to save the CSV file.
    """
    if not results:
        print("No results to save.")
        return
    
    # Get headers from the namedtuple fields, excluding complex objects
    excluded_fields = {'final_params', 'param_history', 'loss_history'}
    headers = [field for field in OptimizerResult._fields if field not in excluded_fields]
    
    try:
        # Ensure outputs directory exists
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(headers)
            # Write data rows
            for result in results:
                # Prepare row data, excluding complex objects
                row_data = {field: getattr(result, field) for field in headers}
                # Convert filter_type enum to string
                if 'filter_type' in row_data:
                    row_data['filter_type'] = row_data['filter_type'].name
                # Convert JAX arrays/scalars if necessary
                row = [float(item) if isinstance(item, (jnp.ndarray, jnp.generic)) else item for item in row_data.values()]
                writer.writerow(row)
        print(f"Results successfully saved to {filepath}")
    except IOError as e:
        print(f"Error saving results to CSV: {e}")


def save_estimated_params(results: List[OptimizerResult], 
                         true_params: Optional[DFSVParamsDataclass] = None,
                         output_dir: str = "outputs"):
    """Save estimated parameters, parameter history, and loss history to pickle files.
    
    Args:
        results: List of optimization results.
        true_params: True model parameters.
        output_dir: Directory to save the pickle files.
    """
    print("\nSaving optimization results...")
    
    # Ensure outputs directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save true parameters for reference
    if true_params is not None:
        true_params_path = os.path.join(output_dir, "true_params.pkl")
        try:
            with open(true_params_path, 'wb') as f:
                cloudpickle.dump(true_params, f)
            print(f"  Saved true parameters to {true_params_path}")
        except Exception as e:
            print(f"  ERROR saving true parameters: {e}")
    
    # Save estimated parameters and history for each run
    for res in results:
        if res.final_params is not None:
            status_str = 'Success' if res.success else 'Failure'
            filter_str = res.filter_type.name
            opt_name = res.optimizer_name.replace(' ', '_')
            transform_str = 'Transformed' if res.uses_transformations else 'Untransformed'
            fix_mu_str = 'FixedMu' if res.fix_mu else 'FreeMu'
            
            # Save final parameters
            filename = f"estimated_params_{filter_str}_{opt_name}_{transform_str}_{fix_mu_str}_{status_str}.pkl"
            filepath = os.path.join(output_dir, filename)
            
            try:
                with open(filepath, 'wb') as f:
                    cloudpickle.dump(res.final_params, f)
                print(f"  Saved parameters to {filepath}")
            except Exception as e:
                print(f"  ERROR saving parameters to {filepath}: {e}")
            
            # Save parameter history if available
            if res.param_history is not None:
                history_filename = f"param_history_{filter_str}_{opt_name}_{transform_str}_{fix_mu_str}_{status_str}.pkl"
                history_filepath = os.path.join(output_dir, history_filename)
                
                try:
                    with open(history_filepath, 'wb') as f:
                        cloudpickle.dump(res.param_history, f)
                    print(f"  Saved parameter history to {history_filepath}")
                except Exception as e:
                    print(f"  ERROR saving parameter history to {history_filepath}: {e}")
            
            # Save loss history if available
            if res.loss_history is not None:
                loss_history_filename = f"loss_history_{filter_str}_{opt_name}_{transform_str}_{fix_mu_str}_{status_str}.pkl"
                loss_history_filepath = os.path.join(output_dir, loss_history_filename)
                
                try:
                    with open(loss_history_filepath, 'wb') as f:
                        cloudpickle.dump(res.loss_history, f)
                    print(f"  Saved loss history to {loss_history_filepath}")
                except Exception as e:
                    print(f"  ERROR saving loss history to {loss_history_filepath}: {e}")
        else:
            print(f"  Skipping result saving for {res.filter_type.name}/{res.optimizer_name} (final_params is None)")


def visualize_parameter_history(param_history, true_params=None, output_dir=None):
    """Visualize parameter history.
    
    Args:
        param_history: List of parameter estimates during optimization.
        true_params: True model parameters.
        output_dir: Directory to save the plots.
    """
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = "outputs/param_history_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract parameter values
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


def visualize_loss_history(loss_history, output_dir=None):
    """Visualize loss history.
    
    Args:
        loss_history: List of loss values during optimization.
        output_dir: Directory to save the plots.
    """
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = "outputs/loss_history_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy array
    loss_values = np.array(loss_history)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot loss history
    ax.plot(loss_values, marker='o')
    ax.set_title('Loss History')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.grid(True)
    
    # Set y-axis to log scale if the loss values span multiple orders of magnitude
    if np.max(loss_values) / np.min(loss_values) > 100:
        ax.set_yscale('log')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_history.png'))
    plt.close()
    
    # Also create a zoomed-in version showing the last 80% of iterations
    if len(loss_values) > 5:  # Only create zoomed plot if we have enough points
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate the starting index for the last 80% of iterations
        start_idx = int(len(loss_values) * 0.2)
        
        # Plot the last 80% of loss values
        ax.plot(range(start_idx, len(loss_values)), loss_values[start_idx:], marker='o')
        ax.set_title('Loss History (Last 80% of Iterations)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_history_zoomed.png'))
        plt.close()
