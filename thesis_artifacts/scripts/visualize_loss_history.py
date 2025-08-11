#!/usr/bin/env python
"""
Visualize loss history during optimization.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cloudpickle
from pathlib import Path

def load_loss_history(filepath):
    """Load loss history from a pickle file."""
    with open(filepath, 'rb') as f:
        return cloudpickle.load(f)

def plot_loss_history(loss_history, output_dir='outputs/loss_history_plots'):
    """Plot loss history."""
    # Create output directory if it doesn't exist
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

def main():
    """Main function."""
    # Check if a loss history file is provided
    if len(sys.argv) < 2:
        print("Usage: python visualize_loss_history.py <loss_history_file>")
        sys.exit(1)
    
    # Load loss history
    loss_history_file = sys.argv[1]
    print(f"Loading loss history from {loss_history_file}...")
    loss_history = load_loss_history(loss_history_file)
    
    # Plot loss history
    print("Plotting loss history...")
    output_dir = os.path.join("outputs", "loss_history_plots", Path(loss_history_file).stem)
    plot_loss_history(loss_history, output_dir)
    
    print(f"Loss history plots saved to {output_dir}")

if __name__ == "__main__":
    main()
