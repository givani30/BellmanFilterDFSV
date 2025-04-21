# -*- coding: utf-8 -*-
"""
Analyzes computational performance from aggregated simulation 1 results.

Performs Stages I & II of the analysis plan:
I. Data Loading and Preprocessing
II. Computational Performance Analysis (Timing vs. Dimensions, Comparative Timing)

Generates plots and a LaTeX summary table.
"""

import os
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

# --- Stage I: Data Loading and Preprocessing ---

print(f"Loading data from: {INPUT_CSV}")
try:
    df = pl.read_csv(INPUT_CSV)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

print("Initial data shape:", df.shape)
print("Initial data columns:", df.columns)
print("Initial data types:", df.dtypes)

# Validation: Check for nulls in key timing columns
timing_cols = ["filter_time_mean", "filter_time_std", "filter_time_median"]
null_counts = df.select(pl.col(timing_cols).is_null().sum()).row(0)
print("\n--- Data Validation ---")
print("Null counts in timing columns:")
for col, count in zip(timing_cols, null_counts):
    print(f"- {col}: {count}")

# Basic outlier check/handling (Example: Log times > 0)
# More sophisticated outlier detection could be added if needed.
initial_rows = df.height
df = df.filter(pl.col("filter_time_mean") > 0) # Assuming time must be positive
rows_after_filter = df.height
if initial_rows > rows_after_filter:
    print(f"Filtered out {initial_rows - rows_after_filter} rows with non-positive filter_time_mean.")
else:
    print("No rows filtered based on non-positive filter_time_mean.")

# Document handling strategy
data_handling_notes = [
    f"Loaded data from {INPUT_CSV}.",
    f"Initial shape: {df.shape}.",
    f"Checked for nulls in {timing_cols}. Found: {dict(zip(timing_cols, null_counts))}.",
    f"Filtered out rows where filter_time_mean <= 0. Removed {initial_rows - rows_after_filter} rows.",
    # Add more notes if other cleaning/handling is done
]
print("\nData Handling Summary:")
for note in data_handling_notes:
    print(f"- {note}")

# Preparation: Ensure correct types if needed (Polars usually infers well)
df = df.with_columns([
    pl.col("N").cast(pl.Int64),
    pl.col("K").cast(pl.Int64),
    pl.col("filter_type").cast(pl.Categorical) # Use Categorical for efficiency
])

# --- Stage II: Computational Performance Analysis ---

print("\n--- Computational Performance Analysis ---")

# Calculate summary stats
summary_stats = df.group_by(["filter_type", "N", "K"]).agg(
    pl.mean("filter_time_mean").alias("mean_time"),
    pl.std("filter_time_mean").alias("std_time"),
    pl.median("filter_time_mean").alias("median_time"),
    pl.count().alias("n_runs")
).sort(["filter_type", "N", "K"])

print("Calculated summary statistics.")

# --- Visualizations ---

sns.set_theme(style="whitegrid")

# 1. Timing vs. Dimensions (N & K) - Line Plots (Log Scale)
print("Generating Timing vs. N/K line plots...")
plt.figure(figsize=(10, 6))
sns.lineplot(data=summary_stats, x="N", y="mean_time", hue="filter_type", marker="o")
plt.yscale('log')
plt.title('Mean Filter Time vs. State Dimension (N) (Log Scale)')
plt.xlabel('State Dimension (N)')
plt.ylabel('Mean Filter Time (s) (Log Scale)')
plt.legend(title='Filter Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
plt.savefig(FIGURE_DIR / "mean_time_vs_N_log.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
sns.lineplot(data=summary_stats, x="K", y="mean_time", hue="filter_type", marker="o")
plt.yscale('log')
plt.title('Mean Filter Time vs. Observation Dimension (K) (Log Scale)')
plt.xlabel('Observation Dimension (K)')
plt.ylabel('Mean Filter Time (s) (Log Scale)')
plt.legend(title='Filter Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(FIGURE_DIR / "mean_time_vs_K_log.png", dpi=300)
plt.close()

# 2. Timing vs. Dimensions (N & K) - Box Plots
print("Generating Timing vs. N/K box plots...")
plt.figure(figsize=(12, 7))
sns.boxplot(data=df, x="N", y="filter_time_mean", hue="filter_type")
plt.yscale('log') # Often necessary due to scale differences
plt.title('Filter Time Distribution vs. State Dimension (N)')
plt.xlabel('State Dimension (N)')
plt.ylabel('Filter Time (s) (Log Scale)')
plt.legend(title='Filter Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(FIGURE_DIR / "box_time_vs_N.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 7))
sns.boxplot(data=df, x="K", y="filter_time_mean", hue="filter_type")
plt.yscale('log')
plt.title('Filter Time Distribution vs. Observation Dimension (K)')
plt.xlabel('Observation Dimension (K)')
plt.ylabel('Filter Time (s) (Log Scale)')
plt.legend(title='Filter Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(FIGURE_DIR / "box_time_vs_K.png", dpi=300)
plt.close()

# 3. Comparative Timing Analysis - Bar Charts
# Choose representative (N, K) pairs. Example: median N/K, max N/K
median_n = int(df['N'].median())
median_k = int(df['K'].median())
max_n = df['N'].max()
max_k = df['K'].max()
representative_configs = [(median_n, median_k), (max_n, max_k)]

print(f"Generating comparative bar charts for configs: {representative_configs}...")
for n_val, k_val in representative_configs:
    subset = summary_stats.filter((pl.col("N") == n_val) & (pl.col("K") == k_val))
    if subset.height == 0:
        print(f"Warning: No data found for N={n_val}, K={k_val}. Skipping bar chart.")
        continue

    plt.figure(figsize=(10, 6))
    # Convert to pandas for seaborn barplot error bars if needed, or plot manually
    subset_pd = subset.to_pandas()
    ax = sns.barplot(data=subset_pd, x="filter_type", y="mean_time", palette="viridis")

    # Add error bars manually using std_time
    # Get current positions and heights
    x_coords = [p.get_x() + p.get_width() / 2. for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    # Map filter types to their std_dev in the correct order
    filter_order = subset_pd['filter_type'].tolist()
    std_devs = subset_pd.set_index('filter_type').loc[filter_order]['std_time'].values
    
    # Check for NaN std_devs (can happen with n_runs=1)
    std_devs = np.nan_to_num(std_devs) 

    ax.errorbar(x=x_coords, y=y_coords, yerr=std_devs, fmt='none', c='black', capsize=5)

    plt.title(f'Mean Filter Time Comparison (N={n_val}, K={k_val})')
    plt.xlabel('Filter Type')
    plt.ylabel('Mean Filter Time (s)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"bar_compare_time_N{n_val}_K{k_val}.png", dpi=300)
    plt.close()


# 4. Comparative Timing Analysis - Heatmaps
print("Generating timing heatmaps...")
filter_types = df["filter_type"].unique().to_list()

for f_type in filter_types:
    subset = summary_stats.filter(pl.col("filter_type") == f_type)
    if subset.height == 0:
        print(f"Warning: No data for filter type {f_type}. Skipping heatmap.")
        continue

    # Pivot data for heatmap
    try:
        pivot_df = subset.pivot(
            index="N", columns="K", values="mean_time"
        ).sort("N")
        # Convert to pandas for seaborn heatmap (easier handling of index/columns)
        pivot_pd = pivot_df.to_pandas().set_index("N")
        pivot_pd.columns = pivot_pd.columns.astype(int) # Ensure K columns are int
        pivot_pd = pivot_pd.sort_index(axis=1) # Sort columns (K)

    except Exception as e:
         print(f"Error pivoting data for heatmap (filter: {f_type}): {e}")
         print("Pivot subset head:\n", subset.head())
         continue # Skip this heatmap if pivot fails


    if pivot_pd.empty:
        print(f"Warning: Pivot table empty for filter type {f_type}. Skipping heatmap.")
        continue

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_pd,
        annot=True,
        fmt=".2e", # Scientific notation, 2 decimal places
        cmap="viridis",
        linewidths=.5,
        cbar_kws={'label': 'Mean Filter Time (s)'}
    )
    plt.title(f'Mean Filter Time Heatmap for {f_type}')
    plt.xlabel('Observation Dimension (K)')
    plt.ylabel('State Dimension (N)')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"heatmap_time_{f_type}.png", dpi=300)
    plt.close()

# --- LaTeX Table Generation ---
print("Generating LaTeX summary table...")

# Select a subset of N, K for the table or show all? Let's show a few representative ones.
# Or maybe aggregate across one dimension? Let's show the configs used in bar charts.
table_subset = summary_stats.filter(
    ((pl.col("N") == median_n) & (pl.col("K") == median_k)) |
    ((pl.col("N") == max_n) & (pl.col("K") == max_k))
).sort(["N", "K", "filter_type"])

# Convert to pandas for easier LaTeX formatting
table_pd = table_subset.to_pandas()

# Format numbers for the table
table_pd['mean_time'] = table_pd['mean_time'].map('{:.3e}'.format)
table_pd['std_time'] = table_pd['std_time'].map('{:.3e}'.format)
table_pd['median_time'] = table_pd['median_time'].map('{:.3e}'.format)

# Rename columns for clarity
table_pd = table_pd.rename(columns={
    'filter_type': 'Filter Type',
    'N': 'N',
    'K': 'K',
    'mean_time': 'Mean Time (s)',
    'std_time': 'Std Dev Time (s)',
    'median_time': 'Median Time (s)',
    'n_runs': 'Runs'
})

# Generate LaTeX using pandas.to_latex with booktabs
latex_string = table_pd.to_latex(
    index=False,
    escape=False, # Allow LaTeX commands if needed (e.g., filter names)
    column_format='llrrrcr', # Adjust alignment (l=left, c=center, r=right)
    caption='Summary of Computational Performance for Selected Configurations.',
    label='tab:comp_perf_summary',
    longtable=False, # Use standard table environment
    # Note: Requires \usepackage{booktabs} in the LaTeX preamble
    # Using formatters might be needed for more complex formatting
)

# Add booktabs manually if needed or rely on pandas capability
# A simple way is to replace default rules with booktabs rules
latex_string = latex_string.replace('\\toprule', '\\toprule') # Ensure toprule is there
latex_string = latex_string.replace('\\midrule', '\\midrule') # Ensure midrule is there
latex_string = latex_string.replace('\\bottomrule', '\\bottomrule') # Ensure bottomrule is there

# Add booktabs package requirement note
latex_header = "% Requires \\usepackage{booktabs} in LaTeX preamble\n"
latex_string = latex_header + latex_string

# Save LaTeX table
table_path = TABLE_DIR / "computational_performance_summary.tex"
try:
    with open(table_path, "w") as f:
        f.write(latex_string)
    print(f"LaTeX table saved to: {table_path}")
except Exception as e:
    print(f"Error writing LaTeX table: {e}")

print("\nAnalysis script finished.")

# Store data handling notes for final summary
# This could be written to a file or passed back if needed
# For now, they are printed to stdout during execution.
# If needed, save them:
# with open(OUTPUT_DIR / "data_handling_notes.txt", "w") as f:
#     for note in data_handling_notes:
#         f.write(f"- {note}\n")