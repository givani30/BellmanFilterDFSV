# -*- coding: utf-8 -*-
"""
Analyzes simulation 1 results, focusing on BIF vs PF performance and stability.

Performs Stages:
I. Data Loading and Preprocessing
II. Computational Performance Analysis (Median Timing)
III. Accuracy Analysis (BIF vs Detailed PF)
IV. Stability Analysis (Mean vs Median)
V. LaTeX Table Generation

Generates plots and LaTeX summary tables.
"""

import os
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings

# Suppress specific Polars warnings if they become noisy (optional)
warnings.filterwarnings("ignore", message=".*Converting.*to Series of dtype Categorical.*")

# --- Configuration ---
# --- Configuration ---
INPUT_CSV = Path("outputs/simulation1/aggregated_simulation1_results.csv")
OUTPUT_DIR = Path("outputs/simulation1_analysis")
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
DATA_DIR = OUTPUT_DIR / "data" # Optional: For saving processed data

# Ensure output directories exist
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
# DATA_DIR.mkdir(parents=True, exist_ok=True) # Uncomment if saving processed data

# Columns for analysis
timing_cols = ["filter_time_mean", "filter_time_std", "filter_time_median"]
accuracy_cols_median = ['rmse_f_median', 'rmse_h_median', 'corr_f_median', 'corr_h_median']
accuracy_cols_mean = ['rmse_f_mean', 'rmse_h_mean', 'corr_f_mean', 'corr_h_mean']
all_metrics = timing_cols + accuracy_cols_median + accuracy_cols_mean

# --- Stage I: Data Loading and Preprocessing ---

print(f"--- Stage I: Data Loading & Preprocessing ---")
print(f"Loading data from: {INPUT_CSV}")
if not INPUT_CSV.exists():
     print(f"Error: Input file not found at {INPUT_CSV}")
     exit(1)

try:
    df = pl.read_csv(INPUT_CSV)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

print("Initial data shape:", df.shape)
# print("Initial data columns:", df.columns)
# print("Initial data types:", df.dtypes)

# Validation: Check for nulls in key columns
key_cols_for_null_check = timing_cols + accuracy_cols_median + accuracy_cols_mean + ['N', 'K', 'filter_type']
null_counts = df.select(pl.col(key_cols_for_null_check).is_null().sum())
print("\n--- Data Validation ---")
print("Null counts summary (first row is counts):")
print(null_counts) # Shows counts per column

# Basic outlier check/handling (Example: Ensure times/RMSE > 0)
initial_rows = df.height
df = df.filter(
    (pl.col("filter_time_median") > 0) &
    (pl.col("rmse_f_median") >= 0) & # RMSE can be 0
    (pl.col("rmse_h_median") >= 0)
)
rows_after_filter = df.height
if initial_rows > rows_after_filter:
    print(f"Filtered out {initial_rows - rows_after_filter} rows with non-positive median time/RMSE.")
else:
    print("No rows filtered based on non-positive median time/RMSE.")

# Focus analysis on BIF and PF (remove BF)
initial_rows = df.height
df_filtered = df.filter(pl.col("filter_type") != 'BF')
rows_after_bf_filter = df_filtered.height
if initial_rows > rows_after_bf_filter:
     print(f"Filtered out {initial_rows - rows_after_bf_filter} rows for 'BF' filter type.")
     # Uncomment below line if you want to keep BF for some comparisons
     # df_bf_present = df.clone() # Keep a copy with BF if needed later
else:
     print("Filter type 'BF' not found or already excluded.")
     # df_bf_present = df.clone()

df = df_filtered # Use the filtered dataframe for subsequent analysis

# Document handling strategy
data_handling_notes = [
    f"Loaded data from {INPUT_CSV}.",
    f"Initial shape: {df.shape}.",
    f"Filtered out rows where median time/RMSE <= 0. Removed {initial_rows - rows_after_filter} rows.",
    f"Filtered out 'BF' filter type. Removed {initial_rows - rows_after_bf_filter} rows.",
    f"Current shape for analysis (BIF/PF): {df.shape}.",
]
print("\nData Handling Summary:")
for note in data_handling_notes:
    print(f"- {note}")

# Preparation: Ensure correct types
# Convert num_particles to Int64, handling potential nulls for non-PF types
# Cast relevant columns, handling potential errors during casting
df = df.with_columns([
    pl.col("N").cast(pl.Int64, strict=False),
    pl.col("K").cast(pl.Int64, strict=False),
    pl.col("num_particles").cast(pl.Float64, strict=False).cast(pl.Int64, strict=False), # Float first to handle potential decimals then Int
    pl.col("filter_type").cast(pl.Categorical),
    # Cast metric columns to Float64
    pl.col(all_metrics).cast(pl.Float64, strict=False)
])
print("\nFinal data types for analysis:")
print(df.dtypes)

# --- Stage II: Computational Performance Analysis (Median Timing) ---

print(f"\n--- Stage II: Computational Performance Analysis (Median Timing) ---")

# Calculate summary stats using MEDIAN time
# Grouping by num_particles needed for PF comparison plots later
summary_stats_timing = df.group_by(["filter_type", "N", "K", "num_particles"]).agg(
    # Keep mean/std for potential reporting, but focus plots on median
    pl.mean("filter_time_mean").alias("mean_filter_time_across_configs"), # Avg of means from CSV
    pl.std("filter_time_mean").alias("std_filter_time_across_configs"), # Std of means from CSV
    pl.median("filter_time_median").alias("median_time"), # Median of medians from CSV
    pl.count().alias("n_configs") # Number of config rows (usually 1 per N,K,filter,particles)
).sort(["filter_type", "N", "K", "num_particles"])

print("Calculated summary statistics based on MEDIAN filter time.")
# print("Timing Summary Head:\n", summary_stats_timing.head())


# --- Visualizations ---
sns.set_theme(style="whitegrid")

# --- 1. Median Timing vs. Dimensions (N & K) - BIF vs Detailed PF ---
print("Generating Median Timing vs. N/K line plots (BIF vs Detailed PF)...")

# Prepare data for plotting: BIF (needs placeholder for style) vs PF (style by particles)
plot_data_timing = summary_stats_timing.with_columns(
    pl.when(pl.col("filter_type") == 'BIF')
    .then(pl.lit("BIF")) # Use 'BIF' string for styling non-particle filters
    .otherwise(pl.col("num_particles").cast(pl.Int64).cast(pl.Utf8) + " particles") # Format PF particle counts
    .alias("style_group")
).sort("N", "K", "filter_type", "num_particles") # Ensure sorting

# Convert to pandas for potentially easier seaborn plotting with hue/style
plot_data_timing_pd = plot_data_timing.to_pandas()


# Plot Median Time vs K (fixed N)
n_val_plot = 50 # Example N value
subset_pd_k = plot_data_timing_pd[plot_data_timing_pd['N'] == n_val_plot]
if not subset_pd_k.empty:
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=subset_pd_k, x="K", y="median_time", hue="filter_type", style="style_group", marker="o", linewidth=2)
    # plt.yscale('log') # Use log scale if needed
    plt.title(f'Median Filter Time vs. Factors (K) for N={n_val_plot} (BIF vs PF by Particles)')
    plt.xlabel('Number of Factors (K)')
    plt.ylabel('Median Filter Time (s)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Filter Config', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(FIGURE_DIR / f"median_time_vs_K_N{n_val_plot}_detail.png", dpi=300)
    plt.close()
else:
    print(f"No data for N={n_val_plot} to plot Median Time vs K.")


# Plot Median Time vs N (fixed K)
k_val_plot = 5 # Example K value
subset_pd_n = plot_data_timing_pd[plot_data_timing_pd['K'] == k_val_plot]
if not subset_pd_n.empty:
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=subset_pd_n, x="N", y="median_time", hue="filter_type", style="style_group", marker="o", linewidth=2)
    # plt.yscale('log')
    plt.title(f'Median Filter Time vs. Variables (N) for K={k_val_plot} (BIF vs PF by Particles)')
    plt.xlabel('Number of Variables (N)')
    plt.ylabel('Median Filter Time (s)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Filter Config', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(FIGURE_DIR / f"median_time_vs_N_K{k_val_plot}_detail.png", dpi=300)
    plt.close()
else:
    print(f"No data for K={k_val_plot} to plot Median Time vs N.")

# --- (Optional: Keep original timing plots if desired, modify to use median) ---
# Code for box plots, bar charts, heatmaps using median_time could be adapted here
# from the original script if needed, ensuring filtering/grouping is correct.
# For brevity, focusing on the new detailed plots first.


# --- Stage III: Accuracy Analysis (BIF vs Detailed PF) ---
print(f"\n--- Stage III: Accuracy Analysis ---")

# Calculate summary stats for accuracy metrics (median)
# Grouping by num_particles needed for PF comparison plots
accuracy_summary_stats = df.group_by(["filter_type", "N", "K", "num_particles"]).agg(
    pl.median("rmse_f_median").alias("rmse_f_median"),
    pl.median("rmse_h_median").alias("rmse_h_median"),
    pl.median("corr_f_median").alias("corr_f_median"),
    pl.median("corr_h_median").alias("corr_h_median")
).sort(["filter_type", "N", "K", "num_particles"])

print("Calculated summary statistics for median accuracy metrics.")

# Prepare data for plotting (similar to timing)
plot_data_accuracy = accuracy_summary_stats.with_columns(
    pl.when(pl.col("filter_type") == 'BIF')
    .then(pl.lit("BIF"))
    .otherwise(pl.col("num_particles").cast(pl.Int64).cast(pl.Utf8) + " particles")
    .alias("style_group")
).sort("N", "K", "filter_type", "num_particles")

# Convert to pandas for plotting
plot_data_accuracy_pd = plot_data_accuracy.to_pandas()

# --- Accuracy Plots vs K for N=50 ---
subset_acc_pd_k = plot_data_accuracy_pd[plot_data_accuracy_pd['N'] == n_val_plot]

if not subset_acc_pd_k.empty:
    print(f"Generating Accuracy vs. K plots for N={n_val_plot}...")
    # Factor RMSE vs K
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=subset_acc_pd_k, x="K", y="rmse_f_median", hue="filter_type", style="style_group", marker="o", linewidth=2)
    plt.title(f'Median Factor RMSE vs. K (BIF vs PF by Particles) for N={n_val_plot}')
    plt.xlabel('Number of Factors (K)')
    plt.ylabel('Median Factor RMSE')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Filter Config', loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(FIGURE_DIR / f"median_rmse_f_vs_K_N{n_val_plot}_detail.png", dpi=300)
    plt.close()

    # Volatility RMSE vs K
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=subset_acc_pd_k, x="K", y="rmse_h_median", hue="filter_type", style="style_group", marker="o", linewidth=2)
    plt.title(f'Median Volatility RMSE vs. K (BIF vs PF by Particles) for N={n_val_plot}')
    plt.xlabel('Number of Factors (K)')
    plt.ylabel('Median Volatility RMSE')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Filter Config', loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(FIGURE_DIR / f"median_rmse_h_vs_K_N{n_val_plot}_detail.png", dpi=300)
    plt.close()

    # Factor Correlation vs K
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=subset_acc_pd_k, x="K", y="corr_f_median", hue="filter_type", style="style_group", marker="o", linewidth=2)
    plt.title(f'Median Factor Correlation vs. K (BIF vs PF by Particles) for N={n_val_plot}')
    plt.xlabel('Number of Factors (K)')
    plt.ylabel('Median Factor Correlation')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Filter Config', loc='lower left')
    # plt.ylim(0.6, 1.0)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(FIGURE_DIR / f"median_corr_f_vs_K_N{n_val_plot}_detail.png", dpi=300)
    plt.close()

    # Volatility Correlation vs K
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=subset_acc_pd_k, x="K", y="corr_h_median", hue="filter_type", style="style_group", marker="o", linewidth=2)
    plt.title(f'Median Volatility Correlation vs. K (BIF vs PF by Particles) for N={n_val_plot}')
    plt.xlabel('Number of Factors (K)')
    plt.ylabel('Median Volatility Correlation')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Filter Config', loc='upper right')
    # plt.ylim(0.1, 1.0)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(FIGURE_DIR / f"median_corr_h_vs_K_N{n_val_plot}_detail.png", dpi=300)
    plt.close()
else:
    print(f"No accuracy data for N={n_val_plot} to plot vs K.")


# --- Stage IV: Stability Analysis (Mean vs Median) ---
print(f"\n--- Stage IV: Stability Analysis (Mean vs Median RMSE) ---")

# --- PF Stability Table ---
pf_results_stab = df.filter(pl.col("filter_type") == 'PF')
cols_stab = ['N', 'K', 'num_particles', 'rmse_f_mean', 'rmse_f_median', 'rmse_h_mean', 'rmse_h_median']
pf_table_stab = pf_results_stab.select(cols_stab)

if pf_table_stab.height > 0:
    pf_table_stab = pf_table_stab.with_columns([
        (pl.when(pl.col('rmse_f_median') > 1e-9)
         .then(pl.col('rmse_f_mean') / pl.col('rmse_f_median'))
         .otherwise(np.inf)
         .alias('rmse_f_ratio')),
        (pl.when(pl.col('rmse_h_median') > 1e-9)
         .then(pl.col('rmse_h_mean') / pl.col('rmse_h_median'))
         .otherwise(np.inf)
         .alias('rmse_h_ratio'))
    ]).sort('rmse_f_ratio', descending=True)

    print("\nComparison of Mean vs Median RMSE for Particle Filter (PF)")
    print("Sorted by Factor RMSE Ratio (Mean/Median) - Higher ratio indicates more outlier impact")
    # Print table to console (Polars default)
    with pl.Config(tbl_rows=20): # Show more rows
         print(pf_table_stab)
    # Save to CSV
    try:
        pf_table_stab.write_csv(DATA_DIR / "pf_stability_analysis.csv")
        print(f"PF stability table saved to {DATA_DIR / 'pf_stability_analysis.csv'}")
    except Exception as e:
        print(f"Could not save PF stability table: {e}")

else:
    print("No PF data found for stability analysis.")
    pf_table_stab = pl.DataFrame() # Empty dataframe for LaTeX generation


# --- BIF Stability Table ---
bif_results_stab = df.filter(pl.col("filter_type") == 'BIF')
cols_stab_bif = ['N', 'K', 'rmse_f_mean', 'rmse_f_median', 'rmse_h_mean', 'rmse_h_median']
bif_table_stab = bif_results_stab.select(cols_stab_bif)

if bif_table_stab.height > 0:
    bif_table_stab = bif_table_stab.with_columns([
        (pl.when(pl.col('rmse_f_median') > 1e-9)
         .then(pl.col('rmse_f_mean') / pl.col('rmse_f_median'))
         .otherwise(np.inf)
         .alias('rmse_f_ratio')),
        (pl.when(pl.col('rmse_h_median') > 1e-9)
         .then(pl.col('rmse_h_mean') / pl.col('rmse_h_median'))
         .otherwise(np.inf)
         .alias('rmse_h_ratio'))
    ]).sort('rmse_f_ratio', descending=True)

    print("\nComparison of Mean vs Median RMSE for Bellman Importance Filter (BIF)")
    print("Sorted by Factor RMSE Ratio (Mean/Median)")
    with pl.Config(tbl_rows=20):
        print(bif_table_stab)
    # Save to CSV
    try:
        bif_table_stab.write_csv(DATA_DIR / "bif_stability_analysis.csv")
        print(f"BIF stability table saved to {DATA_DIR / 'bif_stability_analysis.csv'}")
    except Exception as e:
        print(f"Could not save BIF stability table: {e}")
else:
    print("No BIF data found for stability analysis.")
    bif_table_stab = pl.DataFrame() # Empty dataframe for LaTeX generation


# --- Stage V: LaTeX Table Generation ---
print(f"\n--- Stage V: LaTeX Table Generation ---")

# --- 1. Computational Performance Summary Table (Median Time) ---
# Use summary_stats_timing, focus on BIF/PF, maybe representative configs
# Aggregate PF across particles or show one representative particle count? Let's aggregate.
timing_latex_summary = summary_stats_timing.group_by(["filter_type", "N", "K"]).agg(
    pl.median("median_time").alias("median_time"), # Median of medians across particle counts for PF
    # Add other relevant stats if needed, like median of mean/std times?
).filter(pl.col("filter_type").is_in(['BIF', 'PF'])) # Ensure only BIF/PF included

# Select representative configs
median_n = int(df['N'].median()) if df.height > 0 else 50
median_k = int(df['K'].median()) if df.height > 0 else 5
max_n = df['N'].max() if df.height > 0 else 150
max_k = df['K'].max() if df.height > 0 else 15

table_subset_timing = timing_latex_summary.filter(
    ((pl.col("N") == median_n) & (pl.col("K") == median_k)) |
    ((pl.col("N") == max_n) & (pl.col("K") == max_k))
).sort(["N", "K", "filter_type"])

# Convert to pandas for easier LaTeX formatting
if table_subset_timing.height > 0:
    table_timing_pd = table_subset_timing.to_pandas()
    # Format numbers
    table_timing_pd['median_time'] = table_timing_pd['median_time'].map('{:.3e}'.format)
    # Rename columns
    table_timing_pd = table_timing_pd.rename(columns={
        'filter_type': 'Filter Type', 'N': 'N', 'K': 'K', 'median_time': 'Median Time (s)'
    })
    # Generate LaTeX
    latex_timing_string = table_timing_pd.to_latex(
        index=False, escape=False, column_format='llrc',
        caption='Summary of Median Computational Time for Selected Configurations (PF aggregated).',
        label='tab:comp_perf_summary_median', longtable=False
    )
    latex_timing_string = "% Requires \\usepackage{booktabs} in LaTeX preamble\n" + \
                          latex_timing_string.replace('\\toprule', '\\toprule').replace('\\midrule', '\\midrule').replace('\\bottomrule', '\\bottomrule')

    # Save LaTeX table
    table_timing_path = TABLE_DIR / "median_timing_summary.tex"
    try:
        with open(table_timing_path, "w") as f: f.write(latex_timing_string)
        print(f"LaTeX median timing table saved to: {table_timing_path}")
    except Exception as e: print(f"Error writing LaTeX timing table: {e}")
else:
    print("No data for median timing LaTeX table.")


# --- 2. Stability Analysis Tables (Mean vs Median Ratio) ---

def generate_stability_latex(df_stab, filter_name, table_label):
    """ Helper function to generate LaTeX for stability tables """
    if df_stab.is_empty():
        print(f"Skipping LaTeX generation for {filter_name} stability: No data.")
        return None

    # Select top N worst cases based on ratio? Or specific configs? Let's show top 10.
    df_stab_pd = df_stab.head(10).to_pandas()

    # Format numbers
    for col in ['rmse_f_mean', 'rmse_f_median', 'rmse_h_mean', 'rmse_h_median']:
         df_stab_pd[col] = df_stab_pd[col].map('{:.3e}'.format)
    for col in ['rmse_f_ratio', 'rmse_h_ratio']:
         df_stab_pd[col] = df_stab_pd[col].map('{:.2f}'.format) # Use fixed point for ratios

    # Rename and select/reorder columns
    col_rename = {
        'N':'N', 'K':'K', 'num_particles':'Particles',
        'rmse_f_mean':'Mean RMSE (f)', 'rmse_f_median':'Median RMSE (f)', 'rmse_f_ratio':'Ratio (f)',
        'rmse_h_mean':'Mean RMSE (h)', 'rmse_h_median':'Median RMSE (h)', 'rmse_h_ratio':'Ratio (h)',
    }
    # Include 'Particles' only for PF
    if 'num_particles' not in df_stab_pd.columns:
        col_rename.pop('Particles', None)
        cols_final = ['N', 'K', 'Mean RMSE (f)', 'Median RMSE (f)', 'Ratio (f)', 'Mean RMSE (h)', 'Median RMSE (h)', 'Ratio (h)']
        col_format = 'llrrrrrr'
    else:
        cols_final = ['N', 'K', 'Particles', 'Mean RMSE (f)', 'Median RMSE (f)', 'Ratio (f)', 'Mean RMSE (h)', 'Median RMSE (h)', 'Ratio (h)']
        col_format = 'lllrrrrrr'

    df_stab_pd = df_stab_pd.rename(columns=col_rename)[cols_final]


    latex_string = df_stab_pd.to_latex(
        index=False, escape=False, column_format=col_format,
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
        for note in data_handling_notes:
            f.write(f"- {note}\n")
    print(f"Data handling notes saved to: {notes_path}")
except Exception as e:
    print(f"Error writing data handling notes: {e}")