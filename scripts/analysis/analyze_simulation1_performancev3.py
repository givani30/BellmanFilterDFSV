# -*- coding: utf-8 -*-
"""
Analyzes simulation 1 results using pandas, focusing on BIF vs PF performance and stability.

Performs Stages:
I. Data Loading and Preprocessing
II. Computational Performance Analysis (Median Timing)
III. Accuracy Analysis (BIF vs Detailed PF)
IV. Stability Analysis (Mean vs Median) & Ratio Plots
V. Additional Visualizations (Trade-off, Heatmaps)
VI. LaTeX Table Generation

Generates plots and LaTeX summary tables suitable for academic papers.
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import warnings

# Suppress warnings if needed
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
INPUT_CSV = Path("outputs/simulation1/aggregated_simulation1_results.csv")
OUTPUT_DIR = Path("outputs/simulation1_analysis")
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
DATA_DIR = OUTPUT_DIR / "data" # Optional: For saving processed data

# Ensure output directories exist
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Columns for analysis
timing_cols = ["filter_time_mean", "filter_time_std", "filter_time_median"]
accuracy_cols_median = ['rmse_f_median', 'rmse_h_median', 'corr_f_median', 'corr_h_median']
accuracy_cols_mean = ['rmse_f_mean', 'rmse_h_mean', 'corr_f_mean', 'corr_h_mean']
all_metrics = timing_cols + accuracy_cols_median + accuracy_cols_mean

# Plotting settings
N_VAL_PLOT = 50 # Representative N for plotting vs K
K_VAL_PLOT = 5  # Representative K for plotting vs N
PF_PARTICLES_HEATMAP = 20000 # PF particle count for heatmap comparison
DPI_SETTING = 300 # Resolution for saved figures

# Aesthetics settings
sns.set_theme(style="whitegrid")
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.title_fontsize'] = 11
LINEWIDTH_PLOT = 2.0
MARKERSIZE_PLOT = 6

# --- Stage I: Data Loading and Preprocessing ---

print(f"--- Stage I: Data Loading & Preprocessing ---")
data_handling_notes = [] # Initialize the list here

print(f"Loading data from: {INPUT_CSV}")
if not INPUT_CSV.exists():
     print(f"Error: Input file not found at {INPUT_CSV}")
     exit(1)
data_handling_notes.append(f"Attempted to load data from {INPUT_CSV}.")

try:
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    data_handling_notes.append(f"Successfully loaded data. Initial shape: {df.shape}.")
except Exception as e:
    print(f"Error loading CSV: {e}")
    data_handling_notes.append(f"Failed to load data: {e}")
    exit(1)

# Validation: Check for nulls
key_cols_for_null_check = timing_cols + accuracy_cols_median + accuracy_cols_mean + ['N', 'K', 'filter_type']
existing_key_cols = [col for col in key_cols_for_null_check if col in df.columns]
null_counts = df[existing_key_cols].isnull().sum()
print("\n--- Data Validation ---")
print("Null counts summary (columns with nulls):")
print(null_counts[null_counts > 0])
data_handling_notes.append(f"Checked for nulls in key columns. Counts > 0:\n{null_counts[null_counts > 0].to_string()}")


# Basic outlier check/handling & NaN drop
initial_rows = len(df)
df = df.dropna(subset=['filter_time_median', 'rmse_f_median', 'rmse_h_median'])
df = df[
    (df["filter_time_median"] > 1e-9) & # Use small threshold instead of 0
    (df["rmse_f_median"] >= 0) &
    (df["rmse_h_median"] >= 0)
]
rows_after_filter = len(df)
removed_rows_step1 = initial_rows - rows_after_filter
if removed_rows_step1 > 0:
    print(f"Filtered out {removed_rows_step1} rows with non-positive or NaN median time/RMSE.")
    data_handling_notes.append(f"Filtered out rows where median time/RMSE <= 0 or NaN. Removed {removed_rows_step1} rows.")
else:
    print("No rows filtered based on non-positive or NaN median time/RMSE.")
    data_handling_notes.append("No rows filtered based on non-positive or NaN median time/RMSE.")


# Focus analysis on BIF and PF (remove BF)
initial_rows = len(df)
df_filtered = df[df['filter_type'] != 'BF'].copy()
rows_after_bf_filter = len(df_filtered)
removed_rows_step2 = initial_rows - rows_after_bf_filter
if removed_rows_step2 > 0:
     print(f"Filtered out {removed_rows_step2} rows for 'BF' filter type.")
     data_handling_notes.append(f"Filtered out 'BF' filter type. Removed {removed_rows_step2} rows.")
else:
     print("Filter type 'BF' not found or already excluded.")
     data_handling_notes.append("Filter type 'BF' not found or already excluded.")
df = df_filtered

# Document handling strategy (simplified)
print(f"Current shape for analysis (BIF/PF): {df.shape}.")
data_handling_notes.append(f"Current shape for analysis (BIF/PF): {df.shape}.")

# Preparation: Ensure correct types
try:
    df['N'] = df['N'].astype(np.int64)
    df['K'] = df['K'].astype(np.int64)
    df['num_particles'] = pd.to_numeric(df['num_particles'], errors='coerce').astype('Int64')
    df['filter_type'] = df['filter_type'].astype('category')
    for col in all_metrics:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['N', 'K'] + [col for col in all_metrics if col in df.columns]) # Drop if conversion failed
except Exception as e:
    print(f"Error during type conversion: {e}")
    data_handling_notes.append(f"Error during type conversion: {e}")
    exit(1)

print("\nData types checked/converted.")
data_handling_notes.append("Checked and converted data types.")

# --- Create Combined Filter/Particle Identifier ---
# Used for coloring/styling plots consistently
df['config_id'] = np.where(
    df['filter_type'] == 'BIF',
    'BIF',
    'PF ' + df['num_particles'].astype(str) # e.g., "PF 1000"
)
# Define a consistent order for legend/plotting
config_order = ['BIF'] + sorted([f"PF {p}" for p in df[df['filter_type']=='PF']['num_particles'].dropna().unique().astype(int)])


# --- Stage II: Computational Performance Analysis (Median Timing) ---

print(f"\n--- Stage II: Computational Performance Analysis (Median Timing) ---")

# Calculate summary stats using MEDIAN time
grouping_cols_timing = ["filter_type", "N", "K", "num_particles", "config_id"]
summary_stats_timing = df.groupby(grouping_cols_timing, observed=False, dropna=False).agg(
    median_time=('filter_time_median', 'median'),
    n_configs=('filter_time_median', 'count')
).reset_index().sort_values(by=["filter_type", "N", "K", "num_particles"])

print("Calculated summary statistics based on MEDIAN filter time.")

# --- Visualizations ---
# Define a custom color palette
palette = {"BIF": "blue"}
pf_configs = sorted([f"PF {p}" for p in df[df['filter_type']=='PF']['num_particles'].dropna().unique().astype(int)])
# Create a gradient for PF colors (e.g., light orange to dark red)
pf_cmap = sns.color_palette("OrRd", n_colors=len(pf_configs))
palette.update(dict(zip(pf_configs, pf_cmap)))


# --- 1. Median Timing vs. Dimensions (N & K) - BIF vs Detailed PF (Using COLOR) ---
print("Generating Median Timing vs. N/K line plots (using COLOR)...")

# Plot Median Time vs K (fixed N)
subset_k = summary_stats_timing[summary_stats_timing['N'] == N_VAL_PLOT]
if not subset_k.empty:
    plt.figure(figsize=(10, 6)) # Adjusted figsize slightly
    sns.lineplot(data=subset_k, x="K", y="median_time", hue="config_id", hue_order=config_order, palette=palette, style="filter_type", markers=True, dashes=False, linewidth=LINEWIDTH_PLOT, markersize=MARKERSIZE_PLOT)
    # plt.yscale('log')
    plt.title(f'Median Filter Time vs. Factors (K) for N={N_VAL_PLOT}')
    plt.xlabel('Number of Factors (K)')
    plt.ylabel('Median Filter Time (s)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Filter Config', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.) # Adjust legend position
    plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout rect
    plt.savefig(FIGURE_DIR / f"median_time_vs_K_N{N_VAL_PLOT}_color.png", dpi=DPI_SETTING)
    plt.close()
else:
    print(f"No data for N={N_VAL_PLOT} to plot Median Time vs K.")

# Plot Median Time vs N (fixed K)
subset_n = summary_stats_timing[summary_stats_timing['K'] == K_VAL_PLOT]
if not subset_n.empty:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=subset_n, x="N", y="median_time", hue="config_id", hue_order=config_order, palette=palette, style="filter_type", markers=True, dashes=False, linewidth=LINEWIDTH_PLOT, markersize=MARKERSIZE_PLOT)
    # plt.yscale('log')
    plt.title(f'Median Filter Time vs. Variables (N) for K={K_VAL_PLOT}')
    plt.xlabel('Number of Variables (N)')
    plt.ylabel('Median Filter Time (s)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Filter Config', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(FIGURE_DIR / f"median_time_vs_N_K{K_VAL_PLOT}_color.png", dpi=DPI_SETTING)
    plt.close()
else:
    print(f"No data for K={K_VAL_PLOT} to plot Median Time vs N.")


# --- Stage III: Accuracy Analysis (BIF vs Detailed PF) ---
print(f"\n--- Stage III: Accuracy Analysis ---")

# Calculate summary stats for accuracy metrics (median)
grouping_cols_acc = ["filter_type", "N", "K", "num_particles", "config_id"]
accuracy_summary_stats = df.groupby(grouping_cols_acc, observed=False, dropna=False).agg(
    rmse_f_median=('rmse_f_median', 'median'),
    rmse_h_median=('rmse_h_median', 'median'),
    corr_f_median=('corr_f_median', 'median'),
    corr_h_median=('corr_h_median', 'median')
).reset_index().sort_values(by=["filter_type", "N", "K", "num_particles"])

print("Calculated summary statistics for median accuracy metrics.")

# --- Accuracy Plots vs K for N=N_VAL_PLOT (Using COLOR) ---
subset_acc_k = accuracy_summary_stats[accuracy_summary_stats['N'] == N_VAL_PLOT]

if not subset_acc_k.empty:
    print(f"Generating Accuracy vs. K plots for N={N_VAL_PLOT} (using COLOR)...")
    # Factor RMSE vs K
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=subset_acc_k, x="K", y="rmse_f_median", hue="config_id", hue_order=config_order, palette=palette, style="filter_type", markers=True, dashes=False, linewidth=LINEWIDTH_PLOT, markersize=MARKERSIZE_PLOT)
    plt.title(f'Median Factor RMSE vs. K for N={N_VAL_PLOT}')
    plt.xlabel('Number of Factors (K)')
    plt.ylabel('Median Factor RMSE')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Filter Config', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(FIGURE_DIR / f"median_rmse_f_vs_K_N{N_VAL_PLOT}_color.png", dpi=DPI_SETTING)
    plt.close()

    # Volatility RMSE vs K
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=subset_acc_k, x="K", y="rmse_h_median", hue="config_id", hue_order=config_order, palette=palette, style="filter_type", markers=True, dashes=False, linewidth=LINEWIDTH_PLOT, markersize=MARKERSIZE_PLOT)
    plt.title(f'Median Volatility RMSE vs. K for N={N_VAL_PLOT}')
    plt.xlabel('Number of Factors (K)')
    plt.ylabel('Median Volatility RMSE')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Filter Config', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(FIGURE_DIR / f"median_rmse_h_vs_K_N{N_VAL_PLOT}_color.png", dpi=DPI_SETTING)
    plt.close()

    # Factor Correlation vs K
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=subset_acc_k, x="K", y="corr_f_median", hue="config_id", hue_order=config_order, palette=palette, style="filter_type", markers=True, dashes=False, linewidth=LINEWIDTH_PLOT, markersize=MARKERSIZE_PLOT)
    plt.title(f'Median Factor Correlation vs. K for N={N_VAL_PLOT}')
    plt.xlabel('Number of Factors (K)')
    plt.ylabel('Median Factor Correlation')
    plt.ylim(0.5, 1.0) # Set sensible limits for correlation
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Filter Config', loc='lower left', bbox_to_anchor=(1.02, 0), borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(FIGURE_DIR / f"median_corr_f_vs_K_N{N_VAL_PLOT}_color.png", dpi=DPI_SETTING)
    plt.close()

    # Volatility Correlation vs K
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=subset_acc_k, x="K", y="corr_h_median", hue="config_id", hue_order=config_order, palette=palette, style="filter_type", markers=True, dashes=False, linewidth=LINEWIDTH_PLOT, markersize=MARKERSIZE_PLOT)
    plt.title(f'Median Volatility Correlation vs. K for N={N_VAL_PLOT}')
    plt.xlabel('Number of Factors (K)')
    plt.ylabel('Median Volatility Correlation')
    plt.ylim(0.0, 1.0) # Set sensible limits
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Filter Config', loc='upper right', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(FIGURE_DIR / f"median_corr_h_vs_K_N{N_VAL_PLOT}_color.png", dpi=DPI_SETTING)
    plt.close()
else:
    print(f"No accuracy data for N={N_VAL_PLOT} to plot vs K.")


# --- Stage IV: Stability Analysis (Mean vs Median) & Ratio Plots ---
print(f"\n--- Stage IV: Stability Analysis & Ratio Plots ---")

# Function to calculate ratio safely
def calculate_ratio(mean_col, median_col):
    mean_num = pd.to_numeric(mean_col, errors='coerce')
    median_num = pd.to_numeric(median_col, errors='coerce')
    ratio = np.where(median_num > 1e-9, mean_num / median_num, np.inf)
    return ratio

# --- Calculate Stability Tables ---
# PF
pf_results_stab = df[df['filter_type'] == 'PF'].copy()
# **FIX**: Include filter_type in the list of columns to select
cols_stab_pf = ['N', 'K', 'num_particles', 'config_id', 'filter_type', 'rmse_f_mean', 'rmse_f_median', 'rmse_h_mean', 'rmse_h_median']
cols_stab_pf = [col for col in cols_stab_pf if col in pf_results_stab.columns] # Ensure cols exist
pf_table_stab = pf_results_stab[cols_stab_pf]
if not pf_table_stab.empty:
    pf_table_stab['rmse_f_ratio'] = calculate_ratio(pf_table_stab['rmse_f_mean'], pf_table_stab['rmse_f_median'])
    pf_table_stab['rmse_h_ratio'] = calculate_ratio(pf_table_stab['rmse_h_mean'], pf_table_stab['rmse_h_median'])
    pf_table_stab = pf_table_stab.sort_values(by='rmse_f_ratio', ascending=False)
    print("\nPF Stability Table (Top 20 by Factor Ratio):")
    print(pf_table_stab.head(20).round(3).to_markdown(index=False))
    pf_table_stab.to_csv(DATA_DIR / "pf_stability_analysis.csv", index=False)
else:
    print("No PF data for stability analysis.")
    pf_table_stab = pd.DataFrame()

# BIF
bif_results_stab = df[df['filter_type'] == 'BIF'].copy()
# **FIX**: Include filter_type in the list of columns to select
cols_stab_bif = ['N', 'K', 'config_id', 'filter_type', 'rmse_f_mean', 'rmse_f_median', 'rmse_h_mean', 'rmse_h_median']
cols_stab_bif = [col for col in cols_stab_bif if col in bif_results_stab.columns]
bif_table_stab = bif_results_stab[cols_stab_bif]
if not bif_table_stab.empty:
    bif_table_stab['rmse_f_ratio'] = calculate_ratio(bif_table_stab['rmse_f_mean'], bif_table_stab['rmse_f_median'])
    bif_table_stab['rmse_h_ratio'] = calculate_ratio(bif_table_stab['rmse_h_mean'], bif_table_stab['rmse_h_median'])
    bif_table_stab = bif_table_stab.sort_values(by='rmse_f_ratio', ascending=False)
    print("\nBIF Stability Table (Top 20 by Factor Ratio):")
    print(bif_table_stab.head(20).round(3).to_markdown(index=False))
    bif_table_stab.to_csv(DATA_DIR / "bif_stability_analysis.csv", index=False)
else:
    print("No BIF data for stability analysis.")
    bif_table_stab = pd.DataFrame()

# --- Stability Ratio Plots (Log Scale) ---
print("Generating Stability Ratio plots...")
# Combine stability data for plotting
# Now pf_table_stab and bif_table_stab should have filter_type
stability_plot_data = pd.concat([
    pf_table_stab, # Contains all needed cols including filter_type now
    bif_table_stab # Contains all needed cols including filter_type now
], ignore_index=True)

# Replace Inf with NaN for plotting
stability_plot_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Plot Factor Ratio vs K (fixed N)
subset_ratio_k = stability_plot_data[stability_plot_data['N'] == N_VAL_PLOT].dropna(subset=['rmse_f_ratio']) # Drop NaN ratios for plotting
if not subset_ratio_k.empty:
    y_max_plot = 1e4 # Set a reasonable upper limit for visualization

    plt.figure(figsize=(10, 6))
    plot = sns.lineplot(data=subset_ratio_k, x="K", y="rmse_f_ratio", hue="config_id", hue_order=config_order, palette=palette, style="filter_type", markers=True, dashes=False, linewidth=LINEWIDTH_PLOT, markersize=MARKERSIZE_PLOT)
    plot.set_yscale('log')
    current_ylim = plot.get_ylim()
    # Set limits, ensuring min is slightly above 0 for log scale and cap max
    plot.set_ylim(bottom=max(0.1, current_ylim[0]*0.9), top=min(y_max_plot, current_ylim[1]*1.5))

    plt.title(f'Factor RMSE Stability Ratio (Mean/Median) vs. K for N={N_VAL_PLOT} (Log Scale)')
    plt.xlabel('Number of Factors (K)')
    plt.ylabel('RMSE Ratio (Mean/Median) (Log Scale)')
    plt.grid(True, which='both', linestyle='--', alpha=0.6) # Grid for major and minor ticks on log scale
    plt.legend(title='Filter Config', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(FIGURE_DIR / f"stability_ratio_f_vs_K_N{N_VAL_PLOT}_log.png", dpi=DPI_SETTING)
    plt.close()
else:
    print(f"No stability data for N={N_VAL_PLOT} to plot ratio vs K.")


# --- Stage V: Additional Visualizations ---
print(f"\n--- Stage V: Additional Visualizations ---")

# --- 1. Accuracy vs. Time Trade-off Plot ---
print("Generating Accuracy vs. Time Trade-off plot...")
# Merge accuracy and timing summaries
tradeoff_data = pd.merge(
    accuracy_summary_stats,
    summary_stats_timing[['N', 'K', 'filter_type', 'num_particles', 'median_time']],
    on=['N', 'K', 'filter_type', 'num_particles'],
    how='left'
).dropna(subset=['median_time', 'rmse_h_median']) # Drop if merge failed or data missing

if not tradeoff_data.empty:
    plt.figure(figsize=(11, 7)) # Slightly wider for legend
    scatter = sns.scatterplot(
        data=tradeoff_data,
        x='median_time',
        y='rmse_h_median',
        hue='config_id',
        hue_order=config_order,
        palette=palette,
        style='K', # Use K for style/marker shape
        size='N', # Use N for size
        sizes=(40, 250), # Range of marker sizes
        alpha=0.75
    )
    scatter.set_xscale('log')
    plt.title('Accuracy (Volatility RMSE) vs. Time Trade-off')
    plt.xlabel('Median Filter Time (s) (Log Scale)')
    plt.ylabel('Median Volatility RMSE')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    handles, labels = scatter.get_legend_handles_labels()
    plt.legend(handles, labels, title='Filter / K / N', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(FIGURE_DIR / "tradeoff_rmse_h_vs_time.png", dpi=DPI_SETTING)
    plt.close()
else:
    print("Could not generate trade-off plot, data missing.")


# --- 2. Scalability Heatmaps (Median Performance) ---
print("Generating Scalability Heatmaps...")

def create_heatmap(data, value_col, title, filename, cmap="viridis", fmt=".2e", robust=False):
    """ Helper function to create heatmaps """
    if data.empty:
        print(f"Skipping heatmap '{title}': No data.")
        return
    if value_col not in data.columns:
        print(f"Skipping heatmap '{title}': Column '{value_col}' not found.")
        return
    data = data.dropna(subset=[value_col])
    if data.empty:
         print(f"Skipping heatmap '{title}': No non-NaN data for '{value_col}'.")
         return

    try:
        pivot_data = data.pivot_table(index='N', columns='K', values=value_col)
        pivot_data.columns = pivot_data.columns.astype(int)
        pivot_data = pivot_data.sort_index(axis=0).sort_index(axis=1)
    except Exception as e:
        print(f"Error pivoting data for heatmap '{title}': {e}")
        return

    if pivot_data.empty:
        print(f"Skipping heatmap '{title}': Pivot table empty.")
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        linewidths=.5,
        cbar_kws={'label': value_col},
        annot_kws={"size": 8},
        robust=robust
    )
    plt.title(title, fontsize=14)
    plt.xlabel('Number of Factors (K)')
    plt.ylabel('Number of Variables (N)')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=DPI_SETTING)
    plt.close()

# Merge accuracy and timing data for heatmaps
heatmap_base_data = pd.merge(
    accuracy_summary_stats,
    summary_stats_timing[['N', 'K', 'filter_type', 'num_particles', 'median_time']],
    on=['N', 'K', 'filter_type', 'num_particles'],
    how='left'
)

# Create heatmaps for BIF
bif_heatmap_data = heatmap_base_data[heatmap_base_data['filter_type'] == 'BIF']
create_heatmap(bif_heatmap_data, 'median_time', 'BIF: Median Filter Time (s)', 'heatmap_time_bif.png', cmap='viridis')
create_heatmap(bif_heatmap_data, 'rmse_h_median', 'BIF: Median Volatility RMSE', 'heatmap_rmse_h_bif.png', cmap='magma_r', robust=True)
create_heatmap(bif_heatmap_data, 'corr_f_median', 'BIF: Median Factor Correlation', 'heatmap_corr_f_bif.png', fmt=".3f", cmap='viridis', robust=True)

# Create heatmaps for PF (using specific particle count)
pf_heatmap_data = heatmap_base_data[(heatmap_base_data['filter_type'] == 'PF') & (heatmap_base_data['num_particles'] == PF_PARTICLES_HEATMAP)]
create_heatmap(pf_heatmap_data, 'median_time', f'PF ({PF_PARTICLES_HEATMAP} particles): Median Filter Time (s)', f'heatmap_time_pf{PF_PARTICLES_HEATMAP}.png', cmap='viridis')
create_heatmap(pf_heatmap_data, 'rmse_h_median', f'PF ({PF_PARTICLES_HEATMAP} particles): Median Volatility RMSE', f'heatmap_rmse_h_pf{PF_PARTICLES_HEATMAP}.png', cmap='magma_r', robust=True)
create_heatmap(pf_heatmap_data, 'corr_f_median', f'PF ({PF_PARTICLES_HEATMAP} particles): Median Factor Correlation', f'heatmap_corr_f_pf{PF_PARTICLES_HEATMAP}.png', fmt=".3f", cmap='viridis', robust=True)


# --- Stage VI: LaTeX Table Generation ---
print(f"\n--- Stage VI: LaTeX Table Generation ---")

# --- 1. Computational Performance Summary Table (Median Time) ---
timing_latex_summary = summary_stats_timing.groupby(['filter_type', 'N', 'K']).agg(
    median_time=('median_time', 'median')
).reset_index()
try:
    median_n = int(df['N'].median()) if not df.empty else 50
    median_k = int(df['K'].median()) if not df.empty else 5
    max_n = int(df['N'].max()) if not df.empty else 150
    max_k = int(df['K'].max()) if not df.empty else 15
except Exception:
    median_n, median_k, max_n, max_k = 50, 5, 150, 15

table_subset_timing = timing_latex_summary[
    ((timing_latex_summary["N"] == median_n) & (timing_latex_summary["K"] == median_k)) |
    ((timing_latex_summary["N"] == max_n) & (timing_latex_summary["K"] == max_k))
].sort_values(by=["N", "K", "filter_type"])

if not table_subset_timing.empty:
    table_subset_timing['median_time'] = table_subset_timing['median_time'].map('{:.3e}'.format)
    table_subset_timing = table_subset_timing.rename(columns={
        'filter_type': 'Filter Type', 'N': 'N', 'K': 'K', 'median_time': 'Median Time (s)'
    })
    latex_timing_string = table_subset_timing.to_latex(
        index=False, escape=False, column_format='llrc',
        caption='Summary of Median Computational Time for Selected Configurations (PF aggregated).',
        label='tab:comp_perf_summary_median', longtable=False, na_rep='-'
    )
    latex_timing_string = "% Requires \\usepackage{booktabs} in LaTeX preamble\n" + \
                          latex_timing_string.replace('\\toprule', '\\toprule').replace('\\midrule', '\\midrule').replace('\\bottomrule', '\\bottomrule')
    table_timing_path = TABLE_DIR / "median_timing_summary.tex"
    try:
        with open(table_timing_path, "w") as f: f.write(latex_timing_string)
        print(f"LaTeX median timing table saved to: {table_timing_path}")
    except Exception as e: print(f"Error writing LaTeX timing table: {e}")
else:
    print("No data for median timing LaTeX table.")

# --- 2. Stability Analysis Tables (Mean vs Median Ratio) ---
def generate_stability_latex(df_stab, filter_name, table_label):
    if df_stab.empty:
        print(f"Skipping LaTeX generation for {filter_name} stability: No data.")
        return None
    df_stab_latex = df_stab.head(10).copy() # Show top 10 most unstable
    num_cols = ['rmse_f_mean', 'rmse_f_median', 'rmse_h_mean', 'rmse_h_median']
    ratio_cols = ['rmse_f_ratio', 'rmse_h_ratio']
    for col in num_cols:
         if col in df_stab_latex.columns:
             df_stab_latex[col] = df_stab_latex[col].map('{:.2e}'.format)
    for col in ratio_cols:
         if col in df_stab_latex.columns:
             df_stab_latex[col] = df_stab_latex[col].replace([np.inf, -np.inf], np.nan)
             # Format large ratios scientifically, others fixed
             df_stab_latex[col] = df_stab_latex[col].apply(lambda x: '{:.2e}'.format(x) if pd.notnull(x) and abs(x) >= 1000 else '{:.2f}'.format(x) if pd.notnull(x) else 'NaN')
             df_stab_latex[col] = df_stab_latex[col].fillna('Inf')
    col_rename = {
        'N':'N', 'K':'K', 'num_particles':'Particles',
        'rmse_f_mean':'Mean RMSE (f)', 'rmse_f_median':'Median RMSE (f)', 'rmse_f_ratio':'Ratio (f)',
        'rmse_h_mean':'Mean RMSE (h)', 'rmse_h_median':'Median RMSE (h)', 'rmse_h_ratio':'Ratio (h)',
    }
    df_stab_latex = df_stab_latex.rename(columns=col_rename)
    if 'Particles' not in df_stab_latex.columns:
        # This case is for BIF table
        cols_final = ['N', 'K', 'Mean RMSE (f)', 'Median RMSE (f)', 'Ratio (f)', 'Mean RMSE (h)', 'Median RMSE (h)', 'Ratio (h)']
        col_format = 'llrrrrrr'
        # Check if needed columns exist before selection
        cols_final = [c for c in cols_final if c in df_stab_latex.columns]
        df_stab_latex = df_stab_latex[cols_final]
    else:
        # This case is for PF table
        df_stab_latex['Particles'] = df_stab_latex['Particles'].astype(int)
        cols_final = ['N', 'K', 'Particles', 'Mean RMSE (f)', 'Median RMSE (f)', 'Ratio (f)', 'Mean RMSE (h)', 'Median RMSE (h)', 'Ratio (h)']
        # Check if needed columns exist before selection
        cols_final = [c for c in cols_final if c in df_stab_latex.columns]
        df_stab_latex = df_stab_latex[cols_final]
        col_format = 'lllrrrrrr' # Adjust format string based on actual columns

    # Regenerate column format string based on actual final columns
    col_format = 'l' * len(cols_final) # Default to left-aligned

    latex_string = df_stab_latex.to_latex(
        index=False, escape=False, column_format=col_format, na_rep='-',
        caption=f'{filter_name} Stability Analysis: Mean vs Median RMSE (Top 10 by Factor Ratio).',
        label=table_label, longtable=False
    )
    latex_string = "% Requires \\usepackage{booktabs} in LaTeX preamble\n" + \
                   latex_string.replace('\\toprule', '\\toprule').replace('\\midrule', '\\midrule').replace('\\bottomrule', '\\bottomrule')
    return latex_string

# Generate PF Stability LaTeX
latex_pf_stab_string = generate_stability_latex(pf_table_stab, "PF", "tab:pf_stability")
if latex_pf_stab_string:
    table_pf_stab_path = TABLE_DIR / "pf_stability_summary.tex"
    try:
        with open(table_pf_stab_path, "w") as f: f.write(latex_pf_stab_string)
        print(f"LaTeX PF stability table saved to: {table_pf_stab_path}")
    except Exception as e: print(f"Error writing LaTeX PF stability table: {e}")

# Generate BIF Stability LaTeX
latex_bif_stab_string = generate_stability_latex(bif_table_stab, "BIF", "tab:bif_stability")
if latex_bif_stab_string:
    table_bif_stab_path = TABLE_DIR / "bif_stability_summary.tex"
    try:
        with open(table_bif_stab_path, "w") as f: f.write(latex_bif_stab_string)
        print(f"LaTeX BIF stability table saved to: {table_bif_stab_path}")
    except Exception as e: print(f"Error writing LaTeX BIF stability table: {e}")

print("\nAnalysis script finished.")

# Store data handling notes for final summary
notes_path = OUTPUT_DIR / "data_handling_notes.txt"
try:
    with open(notes_path, "w") as f:
        f.write(f"Analysis run on: {pd.Timestamp.now()}\n")
        # Ensure data_handling_notes is defined and populated
        if 'data_handling_notes' in locals() and isinstance(data_handling_notes, list):
             for note in data_handling_notes:
                 f.write(f"- {note}\n")
        else:
             f.write("- Data handling notes were not properly generated.\n")
    print(f"Data handling notes saved to: {notes_path}")
except Exception as e:
    print(f"Error writing data handling notes: {e}")

