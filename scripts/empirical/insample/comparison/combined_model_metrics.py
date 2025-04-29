#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combined Model Metrics Visualization Script

This script creates combined visualizations of generalized variance and average
conditional correlation for all four benchmark models (BIF-DFSV, PF-DFSV,
DCC-GARCH, Factor-CV) in a format suitable for academic publication.

The script:
1. Loads pre-calculated metrics for BIF and PF models
2. Calculates metrics for DCC-GARCH and Factor-CV models from covariance matrices
3. Aligns time series data across all models
4. Creates professional, publication-quality visualizations
5. Saves high-resolution outputs

Author: Givani Boekestijn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from datetime import datetime
import json
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# Set up seaborn for academic publication-quality plots
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.title_fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 600,  # Higher DPI for publication quality
    'lines.markersize': 6,
    'lines.linewidth': 1.5,
})

# Define paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs" / "empirical" / "insample"
SCRIPTS_DIR = BASE_DIR / "scripts" / "empirical" / "insample"
COMPARISON_OUTPUT_DIR = OUTPUTS_DIR / "comparison"

# Create output directory if it doesn't exist
os.makedirs(COMPARISON_OUTPUT_DIR, exist_ok=True)

# Model data paths
BIF_DATA_DIR = OUTPUTS_DIR / "bif" / "data"
PF_DATA_DIR = OUTPUTS_DIR / "pf" / "data"
FACTORCV_DATA_DIR = OUTPUTS_DIR / "factorcv" / "data"
DCC_DATA_DIR = SCRIPTS_DIR / "dcc" / "data"  # DCC data is in scripts directory

# Original returns data for date information
RETURNS_FILE = SCRIPTS_DIR.parent / "vw_returns_final_with_date.csv"

# Model names and colors for consistent plotting
MODEL_INFO = {
    'BIF-DFSV': {'color': '#1f77b4', 'linestyle': '-', 'zorder': 4, 'alpha': 0.8},     # Blue
    'PF-DFSV': {'color': '#ff7f0e', 'linestyle': '-', 'zorder': 3, 'alpha': 0.7},      # Orange (more transparent)
    'DCC-GARCH': {'color': '#2ca02c', 'linestyle': '-', 'zorder': 2, 'alpha': 0.8},    # Green
    'Factor-CV': {'color': '#d62728', 'linestyle': '-', 'zorder': 1, 'alpha': 0.8}     # Red
}

def load_returns_dates():
    """Load the original returns data to get dates."""
    df = pd.read_csv(RETURNS_FILE)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    return df[date_col]

def load_date_index(file_path):
    """Load date index from a text file."""
    with open(file_path, 'r') as f:
        dates = [line.strip() for line in f]
    return pd.to_datetime(dates)

def calculate_generalized_variance(covariance_matrices):
    """Calculate log-determinant from covariance matrices for numerical stability.

    Returns:
        log_generalized_variance: The log-determinant values (more stable)
    """
    T = covariance_matrices.shape[0]
    log_generalized_variance = np.zeros(T)

    for t in range(T):
        # Use log determinant for numerical stability with large matrices
        sign, logdet = np.linalg.slogdet(covariance_matrices[t])
        # Store the log-determinant directly (more stable)
        log_generalized_variance[t] = logdet

    return log_generalized_variance

def calculate_average_correlation(covariance_matrices):
    """Calculate average correlation from covariance matrices."""
    T, N, _ = covariance_matrices.shape
    avg_correlation = np.zeros(T)
    correlation_matrices = np.zeros((T, N, N))

    for t in range(T):
        # Get the covariance matrix at time t
        cov_t = covariance_matrices[t]

        # Calculate the correlation matrix
        # Correlation = Cov_ij / sqrt(Var_i * Var_j)
        std_devs = np.sqrt(np.diag(cov_t))
        std_outer = np.outer(std_devs, std_devs)
        corr_t = cov_t / std_outer
        np.fill_diagonal(corr_t, 1.0)  # Ensure diagonal is exactly 1

        # Store the correlation matrix
        correlation_matrices[t] = corr_t

        # Calculate average of off-diagonal elements
        # Use total number of off-diagonal elements (N * (N - 1)) since we're summing all of them
        n_off_diag = N * (N - 1)  # Total number of off-diagonal elements
        avg_correlation[t] = (np.sum(corr_t) - N) / n_off_diag  # Subtract diagonal elements (N ones)

    return avg_correlation, correlation_matrices

def load_model_data():
    """Load data for all models and calculate metrics where needed."""
    print("Loading model data...")

    # Dictionary to store data for each model
    model_data = {}

    # Load returns dates as reference
    returns_dates = load_returns_dates()

    # 1. Load BIF model data
    print("Loading BIF-DFSV data...")
    try:
        # Load generalized variance and convert to log-determinant
        bif_gv = np.load(BIF_DATA_DIR / "generalized_variance.npy")
        # Calculate log-determinant from saved generalized variance (approximation)
        bif_log_gv = np.log(np.maximum(bif_gv, 1e-320))

        bif_ac = np.load(BIF_DATA_DIR / "average_correlation.npy")

        # Load date index
        bif_dates = load_date_index(BIF_DATA_DIR / "date_index.txt")

        model_data['BIF-DFSV'] = {
            'log_determinant': bif_log_gv,
            'average_correlation': bif_ac,
            'dates': bif_dates
        }
        print(f"  Loaded BIF-DFSV data: {len(bif_dates)} time points")
        print(f"  Log-determinant range: {np.min(bif_log_gv)} to {np.max(bif_log_gv)}")
    except Exception as e:
        print(f"  Error loading BIF-DFSV data: {e}")

    # 2. Load PF model data
    print("Loading PF-DFSV data...")
    try:
        # Load generalized variance and convert to log-determinant
        pf_gv = np.load(PF_DATA_DIR / "generalized_variance.npy")
        # Calculate log-determinant from saved generalized variance (approximation)
        pf_log_gv = np.log(np.maximum(pf_gv, 1e-320))

        pf_ac = np.load(PF_DATA_DIR / "average_correlation.npy")

        # Load date index
        pf_dates = load_date_index(PF_DATA_DIR / "date_index.txt")

        model_data['PF-DFSV'] = {
            'log_determinant': pf_log_gv,
            'average_correlation': pf_ac,
            'dates': pf_dates
        }
        print(f"  Loaded PF-DFSV data: {len(pf_dates)} time points")
        print(f"  Log-determinant range: {np.min(pf_log_gv)} to {np.max(pf_log_gv)}")
    except Exception as e:
        print(f"  Error loading PF-DFSV data: {e}")

    # 3. Load and process DCC-GARCH model data
    print("Loading DCC-GARCH data...")
    try:
        # Load covariance matrices
        dcc_cov = np.load(DCC_DATA_DIR / "Ht.npy")

        # Calculate log-determinant directly
        dcc_log_gv = calculate_generalized_variance(dcc_cov)

        # Try to load correlation matrices directly for DCC
        try:
            dcc_corr = np.load(DCC_DATA_DIR / "Rt.npy")
            # Calculate average correlation from the correlation matrices
            T, N, _ = dcc_corr.shape
            dcc_ac = np.zeros(T)
            for t in range(T):
                # Calculate average of off-diagonal elements
                n_off_diag = N * (N - 1)  # Total number of off-diagonal elements
                dcc_ac[t] = (np.sum(dcc_corr[t]) - N) / n_off_diag
            print("  Using correlation matrices from Rt.npy")
        except Exception:
            # Fall back to calculating from covariance matrices
            dcc_ac, _ = calculate_average_correlation(dcc_cov)
            print("  Calculating correlations from covariance matrices")

        # Load date index
        dcc_dates = load_date_index(DCC_DATA_DIR / "date_index.txt")

        model_data['DCC-GARCH'] = {
            'log_determinant': dcc_log_gv,
            'average_correlation': dcc_ac,
            'dates': dcc_dates
        }
        print(f"  Loaded DCC-GARCH data: {len(dcc_dates)} time points")
        print(f"  Log-determinant range: {np.min(dcc_log_gv)} to {np.max(dcc_log_gv)}")
    except Exception as e:
        print(f"  Error loading DCC-GARCH data: {e}")

    # 4. Load and process Factor-CV model data
    print("Loading Factor-CV data...")
    try:
        # Load covariance matrices
        fcv_cov = np.load(FACTORCV_DATA_DIR / "Ht.npy")

        # Calculate log-determinant directly
        fcv_log_gv = calculate_generalized_variance(fcv_cov)
        fcv_ac, _ = calculate_average_correlation(fcv_cov)

        # Load date index
        fcv_dates = load_date_index(FACTORCV_DATA_DIR / "date_index.txt")

        model_data['Factor-CV'] = {
            'log_determinant': fcv_log_gv,
            'average_correlation': fcv_ac,
            'dates': fcv_dates
        }
        print(f"  Loaded Factor-CV data: {len(fcv_dates)} time points")
        print(f"  Log-determinant range: {np.min(fcv_log_gv)} to {np.max(fcv_log_gv)}")
    except Exception as e:
        print(f"  Error loading Factor-CV data: {e}")

    return model_data

def align_time_series(model_data):
    """Align time series data across all models to ensure consistent date ranges."""
    print("Aligning time series data...")

    # Find common date range
    all_dates = []
    for model_name, data in model_data.items():
        if 'dates' in data:
            all_dates.append(pd.Series(data['dates']))

    if not all_dates:
        print("Error: No valid date information found in any model")
        return model_data

    # Find common date range
    common_dates = all_dates[0]
    for dates in all_dates[1:]:
        common_dates = common_dates[common_dates.isin(dates)]

    print(f"Common date range: {common_dates.min()} to {common_dates.max()}, {len(common_dates)} points")

    # Align data for each model
    aligned_data = {}
    for model_name, data in model_data.items():
        if 'dates' not in data:
            print(f"  Skipping {model_name}: No date information")
            continue

        # Create DataFrame with dates as index
        model_df = pd.DataFrame({
            'log_determinant': data['log_determinant'],
            'average_correlation': data['average_correlation']
        }, index=data['dates'])

        # Filter to common dates
        aligned_df = model_df.loc[model_df.index.isin(common_dates)]

        # Sort by date
        aligned_df = aligned_df.sort_index()

        aligned_data[model_name] = {
            'log_determinant': aligned_df['log_determinant'].values,
            'average_correlation': aligned_df['average_correlation'].values,
            'dates': aligned_df.index
        }

        print(f"  Aligned {model_name}: {len(aligned_df)} time points")

    return aligned_data

def plot_generalized_variance(model_data):
    """Create publication-quality plots of log-determinant for all models."""
    print("Creating log-determinant plots...")

    # Number of initial observations to skip
    skip_n = 10

    # 1. Original log-determinant plot
    plt.figure(figsize=(10, 6))
    # Set seaborn style for this specific plot
    with sns.axes_style("whitegrid"):
        sns.set_context("paper", font_scale=1.1)
    ax = plt.gca()

    for model_name, data in model_data.items():
        if 'log_determinant' not in data or 'dates' not in data:
            print(f"  Skipping {model_name}: Missing required data")
            continue

        # Plot log-determinant (skipping initial observations) with model-specific transparency
        ax.plot(
            data['dates'][skip_n:],
            data['log_determinant'][skip_n:],
            label=model_name,
            color=MODEL_INFO[model_name]['color'],
            linestyle=MODEL_INFO[model_name]['linestyle'],
            zorder=MODEL_INFO[model_name]['zorder'],
            alpha=MODEL_INFO[model_name]['alpha']
        )

    # Format x-axis to show only selected years (every 5 years)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))  # Show every 5 years
    plt.xticks(rotation=45)

    # Add labels and title
    ax.set_title('Log-Determinant of Conditional Covariance Matrices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Log-Determinant')

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    ax.legend(loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = COMPARISON_OUTPUT_DIR / "log_determinant_comparison.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"  Saved to {output_path}")

    # Also save as PDF for publication
    pdf_path = COMPARISON_OUTPUT_DIR / "log_determinant_comparison.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"  Saved to {pdf_path}")

    plt.close()

    # 2. Demeaned log-determinant plot for better comparison
    plt.figure(figsize=(10, 6))
    # Set seaborn style for this specific plot
    with sns.axes_style("whitegrid"):
        sns.set_context("paper", font_scale=1.1)
    ax = plt.gca()

    for model_name, data in model_data.items():
        if 'log_determinant' not in data or 'dates' not in data:
            print(f"  Skipping {model_name}: Missing required data")
            continue

        # Demean the log-determinant series (skipping initial observations for mean calculation)
        log_det = data['log_determinant'][skip_n:]
        demeaned_log_det = log_det - np.mean(log_det)

        # Plot demeaned log-determinant with model-specific transparency
        ax.plot(
            data['dates'][skip_n:],
            demeaned_log_det,
            label=f"{model_name} (mean: {np.mean(log_det):.2f})",
            color=MODEL_INFO[model_name]['color'],
            linestyle=MODEL_INFO[model_name]['linestyle'],
            zorder=MODEL_INFO[model_name]['zorder'],
            alpha=MODEL_INFO[model_name]['alpha']
        )

    # Format x-axis to show only selected years (every 5 years)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))  # Show every 5 years
    plt.xticks(rotation=45)

    # Add labels and title
    ax.set_title('Demeaned Log-Determinant of Conditional Covariance Matrices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Demeaned Log-Determinant')

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    ax.legend(loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = COMPARISON_OUTPUT_DIR / "demeaned_log_determinant_comparison.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"  Saved to {output_path}")

    # Also save as PDF for publication
    pdf_path = COMPARISON_OUTPUT_DIR / "demeaned_log_determinant_comparison.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"  Saved to {pdf_path}")

    plt.close()

    # 3. Normalized log-determinant plot (scaled by standard deviation)
    plt.figure(figsize=(10, 6))
    # Set seaborn style for this specific plot
    with sns.axes_style("whitegrid"):
        sns.set_context("paper", font_scale=1.1)
    ax = plt.gca()

    for model_name, data in model_data.items():
        if 'log_determinant' not in data or 'dates' not in data:
            print(f"  Skipping {model_name}: Missing required data")
            continue

        # Normalize the log-determinant series (skipping initial observations)
        log_det = data['log_determinant'][skip_n:]
        normalized_log_det = (log_det - np.mean(log_det)) / np.std(log_det)

        # Plot normalized log-determinant with model-specific transparency
        ax.plot(
            data['dates'][skip_n:],
            normalized_log_det,
            label=model_name,
            color=MODEL_INFO[model_name]['color'],
            linestyle=MODEL_INFO[model_name]['linestyle'],
            zorder=MODEL_INFO[model_name]['zorder'],
            alpha=MODEL_INFO[model_name]['alpha']
        )

    # Format x-axis to show only selected years (every 5 years)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))  # Show every 5 years
    plt.xticks(rotation=45)

    # Add labels and title
    ax.set_title('Normalized Log-Determinant of Conditional Covariance Matrices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Log-Determinant (z-score)')

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    ax.legend(loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = COMPARISON_OUTPUT_DIR / "normalized_log_determinant_comparison.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"  Saved to {output_path}")

    # Also save as PDF for publication
    pdf_path = COMPARISON_OUTPUT_DIR / "normalized_log_determinant_comparison.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"  Saved to {pdf_path}")

    plt.close()

def plot_average_correlation(model_data):
    """Create a publication-quality plot of average correlation for all models."""
    print("Creating average correlation plot...")

    # Number of initial observations to skip
    skip_n = 10

    plt.figure(figsize=(10, 6))
    # Set seaborn style for this specific plot
    with sns.axes_style("whitegrid"):
        sns.set_context("paper", font_scale=1.1)
    ax = plt.gca()

    for model_name, data in model_data.items():
        if 'average_correlation' not in data or 'dates' not in data:
            print(f"  Skipping {model_name}: Missing required data")
            continue

        # Plot average correlation (skipping initial observations) with model-specific transparency
        ax.plot(
            data['dates'][skip_n:],
            data['average_correlation'][skip_n:],
            label=model_name,
            color=MODEL_INFO[model_name]['color'],
            linestyle=MODEL_INFO[model_name]['linestyle'],
            zorder=MODEL_INFO[model_name]['zorder'],
            alpha=MODEL_INFO[model_name]['alpha']
        )

    # Format x-axis to show only selected years (every 5 years)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))  # Show every 5 years
    plt.xticks(rotation=45)

    # Add labels and title
    ax.set_title('Average Conditional Correlation Comparison Across Models')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Correlation')

    # Set y-axis limits to focus on the relevant range
    ax.set_ylim(0, 1)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    ax.legend(loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = COMPARISON_OUTPUT_DIR / "average_correlation_comparison.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"  Saved to {output_path}")

    # Also save as PDF for publication
    pdf_path = COMPARISON_OUTPUT_DIR / "average_correlation_comparison.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"  Saved to {pdf_path}")

    plt.close()

def save_processed_data(model_data):
    """Save processed data for future use."""
    print("Saving processed data...")

    # Create a dictionary to store processed data
    processed_data = {}

    for model_name, data in model_data.items():
        if 'dates' not in data:
            continue

        # Convert dates to strings for JSON serialization
        dates_str = [d.strftime('%Y-%m-%d') for d in data['dates']]

        processed_data[model_name] = {
            'log_determinant': data['log_determinant'].tolist() if 'log_determinant' in data else None,
            'average_correlation': data['average_correlation'].tolist() if 'average_correlation' in data else None,
            'dates': dates_str
        }

    # Save as JSON
    output_path = COMPARISON_OUTPUT_DIR / "model_metrics_data.json"
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    print(f"  Saved to {output_path}")

    # Also save as NumPy arrays for easier loading
    output_path = COMPARISON_OUTPUT_DIR / "model_metrics_data.npz"
    np.savez(
        output_path,
        **{f"{model_name.replace('-', '_')}_log_det": data['log_determinant']
           for model_name, data in model_data.items() if 'log_determinant' in data},
        **{f"{model_name.replace('-', '_')}_ac": data['average_correlation']
           for model_name, data in model_data.items() if 'average_correlation' in data}
    )
    print(f"  Saved to {output_path}")

def main():
    """Main function to execute the script."""
    print("\n=== Combined Model Metrics Visualization ===\n")

    # 1. Load data from all models
    model_data = load_model_data()

    # 2. Align time series
    aligned_data = align_time_series(model_data)

    # 3. Create visualizations
    plot_generalized_variance(aligned_data)  # Now plots log-determinant
    plot_average_correlation(aligned_data)

    # 4. Save processed data
    save_processed_data(aligned_data)

    print("\nVisualization complete. Output saved to:", COMPARISON_OUTPUT_DIR)

if __name__ == "__main__":
    main()
