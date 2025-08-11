"""Plotting utility functions for simulation study 2 analysis.

This module provides functions for creating various visualization plots of the simulation
results, including scatter plots, heatmaps, eigenvalue distributions, and scaling analysis.
All plotting functions use a centralized plotting style configuration and return Figure
objects without performing file I/O operations.
"""

import polars as pl
import numpy as np
import jax.numpy as jnp
import logging
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from pathlib import Path

from analysis_plotting_config import apply_publication_style
from analysis_io_utils import save_analysis_output
from bellman_filter_dfsv.models.dfsv import DFSVParamsDataclass


def plot_scatter_comparison(df: pl.DataFrame) -> Dict[str, Figure]:
    """Create publication-ready scatter plots comparing Bias vs. RMSE for each parameter.

    Generates a separate figure for each parameter, with subplots faceted by (N, K) configurations.

    Args:
        df: DataFrame containing parameter errors and metrics. Must include columns
            like 'param_{param}_bias', 'param_{param}_rmse', 'N', 'K', 'config_T',
            'filter_config', and 'config_num_particles'.

    Returns:
        Dict[str, Figure]: Dictionary of matplotlib figures, keyed by parameter name.
    """
    apply_publication_style() # Apply centralized plotting style

    # Create mapping for filter config names (no changes)
    filter_config_map = {
        "BIF": "Bellman Filter",
        "PF-1000": "Particle Filter (1000)",
        "PF-5000": "Particle Filter (5000)",
        "PF-10000": "Particle Filter (10000)"
    }

    # Data preprocessing (no changes)
    df = df.with_columns([
        pl.col("N").cast(pl.Int64),
        pl.col("K").cast(pl.Int64),
        pl.col("config_T").cast(pl.Int64),
        pl.col("filter_config").cast(pl.Utf8).replace(filter_config_map).alias("filter_config_display"),
        pl.concat_str([
            pl.lit("T = "),
            pl.col("config_T").cast(pl.Utf8)
        ]).alias("time_length_display")
    ])

    figures = {}  # Dictionary to store figures

    params_to_plot = {
        "lambda_r": "Factor loading Parameters",
        "Phi_f": "Factor Transition Matrix",
        "Phi_h": "Log-Volatility Transition Matrix",
        "Q_h": "Volatility Covariance Matrix",
        "mu": "Long-run mean of Log-Volatilities",
        "sigma2": "Idiosyncratic Noise Variance"
    }

    for param, title in params_to_plot.items():
        bias_col = f"param_{param}_bias"
        rmse_col = f"param_{param}_rmse"

        # Filter data relevant to this parameter (only rows where error columns exist and are not NaN/Inf)
        # This filtering is crucial to avoid errors on parameters that might not have these metrics
        # Also filter out rows where bias or rmse are NaN/Inf
        param_df = df.filter(
            (bias_col in df.columns) & (rmse_col in df.columns) &
            pl.col(bias_col).is_finite() & pl.col(rmse_col).is_finite()
        )

        if param_df.height == 0:
            logging.info(f"Skipping {param} plot - no valid data found.")
            continue

        configs = param_df.select(["N", "K"]).unique().sort(["N", "K"])
        if configs.height == 0:
             # This case should be covered by param_df.height == 0, but double check
            logging.warning(f"No unique configurations found for {param} with valid data.")
            continue

        n_configs = configs.height
        n_cols = min(3, n_configs)
        n_rows = (n_configs + n_cols - 1) // n_cols

        # Create a NEW figure for this parameter
        # Adjust figure size based on the number of subplots for THIS parameter
        base_width = 2.3
        legend_space = 1.5
        fig_width = min(7.5, base_width * n_cols + legend_space)
        fig_height = base_width * n_rows
        fig = plt.figure(figsize=(fig_width, fig_height))

        scatter_plots = [] # To collect subplot objects for legend handles/labels

        first_subplot_handles = None
        first_subplot_labels = None

        for idx, (N, K) in enumerate(configs.iter_rows(), 1):
            ax = fig.add_subplot(n_rows, n_cols, idx) # Use fig.add_subplot

            plot_data = param_df.filter(
                (pl.col("N") == N) &
                (pl.col("K") == K)
            ).select([
                pl.col(bias_col).cast(pl.Float64),
                pl.col(rmse_col).cast(pl.Float64),
                "filter_config_display",
                "time_length_display"
            ]).to_pandas()

            # Add a larger offset to PF-1000 points to prevent overlap
            if "Particle Filter (1000)" in plot_data["filter_config_display"].values:
                pf1000_mask = plot_data["filter_config_display"] == "Particle Filter (1000)"
                plot_data.loc[pf1000_mask, bias_col] = plot_data.loc[pf1000_mask, bias_col] + 0.005

            scatter = sns.scatterplot(
                data=plot_data,
                x=bias_col,
                y=rmse_col,
                hue="filter_config_display",
                style="time_length_display",
                alpha=0.8,
                ax=ax,
                legend=True if idx == 1 else False,  # Only show legend on first subplot
                size="filter_config_display",
                sizes={
                    "Bellman Filter": 40,
                    "Particle Filter (1000)": 50,
                    "Particle Filter (5000)": 40,
                    "Particle Filter (10000)": 40
                },
            )

            if idx == 1: # Get handles/labels from the first subplot
                first_subplot_handles, first_subplot_labels = ax.get_legend_handles_labels()

            if ax.get_legend() is not None:
                 ax.get_legend().remove() # Remove legend after getting handles/labels


            # Professional subplot formatting
            ax.set_title(f"N = {N}, K = {K}", pad=10)
            ax.set_xlabel("Bias")
            ax.set_ylabel("RMSE")
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Add overall title
        fig.suptitle(f"{title} - Bias vs. RMSE Analysis", y=1.02, fontsize=13)

        # Create professional legend using handles/labels from the first subplot
        if first_subplot_handles is not None and first_subplot_labels is not None:
            # Split handles and labels (assuming hue comes first, then style)
            # Need to handle cases where not all filter configs or time lengths are present in the data
            unique_filter_displays = param_df.select("filter_config_display").unique().sort("filter_config_display").to_series().to_list()
            unique_time_lengths = param_df.select("time_length_display").unique().sort("time_length_display").to_series().to_list()

            legend_handles = []
            legend_labels = []

            # Add filter type handles/labels
            for cfg_display in unique_filter_displays:
                 try:
                    idx = first_subplot_labels.index(cfg_display)
                    legend_handles.append(first_subplot_handles[idx])
                    legend_labels.append(cfg_display)
                 except ValueError:
                    pass # Filter type not present in the first subplot's legend (shouldn't happen if data is present)

            # Add time length handles/labels
            for time_label in unique_time_lengths:
                 try:
                    idx = first_subplot_labels.index(time_label)
                    legend_handles.append(first_subplot_handles[idx])
                    legend_labels.append(time_label)
                 except ValueError:
                    pass # Time length not present in the first subplot's legend (shouldn't happen if data is present)


            if legend_handles: # Only add legend if there are handles
                legend = fig.legend(
                    legend_handles,
                    legend_labels,
                    title="Filter Type / Time Series Length",
                    bbox_to_anchor=(1.02, 0.5),
                    loc='center left',
                    ncol=1,
                    frameon=True,
                    framealpha=0.9,
                    edgecolor='lightgray'
                )
                legend.get_frame().set_alpha(0.9) # Ensure legend background is visible


        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to accommodate suptitle and potential legend outside

        figures[param] = fig  # Store the figure in the dictionary

    return figures  # Return the dictionary of figures


def plot_error_heatmaps(df: pl.DataFrame) -> Figure:
    """Create heatmaps showing average element-wise differences for matrix parameters.

    Args:
        df: DataFrame containing error metrics.

    Returns:
        Figure: The matplotlib figure containing the heatmaps.
    """
    apply_publication_style()

    matrix_params = ["Phi_f", "Phi_h"]
    param_titles = {
        "Phi_f": "State Transition Matrix",
        "Phi_h": "Volatility Transition Matrix"
    }

    df_casted = df.with_columns([
        pl.col("N").cast(pl.Int64),
        pl.col("K").cast(pl.Int64),
        pl.col("config_T").cast(pl.Int64),
        pl.col("filter_config").cast(pl.Categorical)
    ])
    configs = df_casted.select(["N", "K", "config_T"]).unique().sort(["N", "K", "config_T"])

    for param in matrix_params:
        for config_row in configs.iter_rows(named=True):
            N, K, T = config_row["N"], config_row["K"], config_row["config_T"]
            config_label = f"N{N}-K{K}-T{T}"

            filter_configs_to_plot = df_casted.filter(
                (pl.col("N") == N) & (pl.col("K") == K) & (pl.col("config_T") == T)
            ).select("filter_config").unique().sort("filter_config").to_series().to_list()

            if not filter_configs_to_plot:
                logging.warning(f"No filter configs found for heatmap: {param}, {config_label}. Skipping.")
                continue

            num_filters = len(filter_configs_to_plot)
            fig_width = min(16, max(8, num_filters * K * 1.5))
            fig_height = min(8, max(5, K * 1.5))
            fig, axes = plt.subplots(1, num_filters, figsize=(fig_width, fig_height), squeeze=False)

            for idx, filter_cfg in enumerate(filter_configs_to_plot):
                ax = axes[0, idx]
                fontsize = max(8, 12 - (K - 2))

                try:
                    filter_data = df_casted.filter(
                        (pl.col("filter_config") == filter_cfg) &
                        (pl.col("N") == N) &
                        (pl.col("K") == K) &
                        (pl.col("config_T") == T)
                    )

                    if filter_data.height == 0:
                        logging.warning(f"Data unexpectedly empty for heatmap: {param}, {filter_cfg}, {config_label}")
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=fontsize)
                        ax.set_title(f"{filter_cfg} - {config_label}\nMean Element-wise Bias",
                                  fontsize=fontsize + 2, pad=10)
                        continue

                    matrix_data = np.full((K, K), np.nan)
                    for i in range(K):
                        for j in range(K):
                            col_name = f"param_{param}_element_{i}_{j}_bias"
                            if col_name in filter_data.columns:
                                bias_val = filter_data.select(
                                    pl.col(col_name).cast(pl.Float64)
                                ).mean().item()
                                matrix_data[i, j] = bias_val if bias_val is not None else np.nan

                    sns.heatmap(
                        data=matrix_data,
                        cmap="RdBu_r",
                        center=0,
                        annot=True,
                        fmt=".3f",
                        annot_kws={'size': fontsize},
                        square=True,
                        cbar_kws={'label': 'Bias'},
                        ax=ax
                    )

                    ax.tick_params(axis='x', labelsize=fontsize)
                    ax.tick_params(axis='y', labelsize=fontsize)
                    ax.set_title(
                        f"{filter_cfg} - {config_label}\nMean Element-wise Bias (Est - True)",
                        fontsize=fontsize + 2,
                        pad=10
                    )

                except Exception as e:
                    logging.error(f"Error creating heatmap for {param} ({filter_cfg}, {config_label}): {str(e)}")
                    ax.text(0.5, 0.5, 'Error generating heatmap',
                          ha='center', va='center', fontsize=fontsize)
                    ax.set_title(f"{filter_cfg} - {config_label}\nError",
                              fontsize=fontsize + 2, pad=10)

            plt.suptitle(
                f"{param_titles[param]} Bias Analysis (K={K})",
                y=1.05,
                fontsize=fontsize + 4
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Close all other figures to prevent memory issues
            for i in plt.get_fignums():
                if i != fig.number:
                    plt.close(i)

    return fig


def plot_k2_eigenvalue_distributions(
    df: pl.DataFrame,
    true_params_dict: Dict[str, DFSVParamsDataclass],
    estimated_params_dict: Dict[str, DFSVParamsDataclass]
) -> Figure:
    """Create distribution plots and ellipse visualizations for K=2 configurations.

    Args:
        df: DataFrame containing parameter estimates.
        true_params_dict: Dictionary of true parameter objects.
        estimated_params_dict: Dictionary of estimated parameter objects.

    Returns:
        Figure: The matplotlib figure containing the eigenvalue plots.
    """
    apply_publication_style()

    # Cast columns and filter for K=2 configurations
    df_k2 = df.with_columns([
        pl.col("K").cast(pl.Int64),
        pl.col("N").cast(pl.Int64),
        pl.col("config_T").cast(pl.Int64),
        pl.col("filter_config").cast(pl.Categorical),
        pl.col("config_fix_mu").cast(pl.Boolean),
    ]).filter(pl.col("K") == 2)

    if df_k2.height == 0:
        logging.info("No K=2 configurations found for eigenvalue plots")
        return None

    df_k2 = df_k2.with_columns([
        pl.concat_str([
            pl.lit("N"), pl.col("N").cast(pl.Utf8),
            pl.lit("-K2-T"), pl.col("config_T").cast(pl.Utf8)
        ]).alias("config_label").cast(pl.Categorical)
    ])

    param_titles = {
        "Phi_f": "State Transition Matrix",
        "Phi_h": "Volatility Transition Matrix"
    }

    filter_configs = df_k2.select("filter_config").unique().sort("filter_config").to_series()
    color_palette = sns.color_palette("husl", n_colors=len(filter_configs))
    color_dict = dict(zip(filter_configs, color_palette))

    style_dict = {True: "-", False: "--"}
    hatch_dict = {True: "///", False: None}

    for param, title in param_titles.items():
        eig_col = f"param_{param}_eig_rmse"

        df_plot = df_k2.with_columns([
            pl.col(eig_col).cast(pl.Float64)
        ])

        fig = plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        sns.boxplot(
            data=df_plot.to_pandas(),
            x="config_label",
            y=eig_col,
            hue="filter_config",
            palette="Set2"
        )
        plt.title(f"Eigenvalue RMSE Distribution")
        plt.xticks(rotation=45)
        plt.ylabel("RMSE")
        plt.xlabel("Configuration")
        plt.gca().get_legend().set_title("Filter Configuration")

        plt.subplot(1, 2, 2)

        configs = df_plot.unique(["config_label", "filter_config"]).sort(["config_label", "filter_config"])

        for config_row in configs.iter_rows(named=True):
            filter_cfg = config_row["filter_config"]
            config = config_row["config_label"]

            mean_data = df_plot.filter(
                (pl.col("filter_config") == filter_cfg) &
                (pl.col("config_label") == config)
            )

            matrix = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    col_name = f"param_{param}_element_{i}_{j}_bias"
                    if col_name in mean_data.columns:
                        matrix[i, j] = mean_data.select(
                            pl.col(col_name).cast(pl.Float64)
                        ).mean().item()

            try:
                eigvals, eigvecs = np.linalg.eig(matrix)
                eigvecs_np = jnp.asarray(eigvecs).astype(float)
                angle = float(jnp.degrees(jnp.arctan2(eigvecs_np[1, 0], eigvecs_np[0, 0])))
                width = 2 * np.abs(eigvals[0])
                height = 2 * np.abs(eigvals[1])

                fix_mu = mean_data.select(pl.col("config_fix_mu")).unique()[0, 0]
                color = color_dict[filter_cfg]

                style_kwargs = {
                    "alpha": 0.3,
                    "color": color,
                    "label": f"{filter_cfg} ({config}{'†' if fix_mu else ''})",
                    "linestyle": style_dict[fix_mu],
                    "hatch": hatch_dict[fix_mu],
                    "fill": True
                }

                ellipse = plt.matplotlib.patches.Ellipse(
                    (0, 0), width, height, angle=angle, **style_kwargs
                )
                plt.gca().add_patch(ellipse)

            except np.linalg.LinAlgError as e:
                logging.warning(f"Failed to compute ellipse for {filter_cfg}-{config}: {e}")

        plt.title("Mean Matrix Eigenstructure Visualization")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

        legend = plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            title="Filter Config (N,K,T)\n† = Fix μ",
            frameon=True,
            framealpha=0.95,
            edgecolor='lightgray'
        )
        legend.get_frame().set_alpha(0.9)

        fig.suptitle(f"{title} Analysis (K=2)", y=1.05, fontsize=14, weight='bold')
        plt.tight_layout()
        
        # Close all other figures to prevent memory issues
        for i in plt.get_fignums():
            if i != fig.number:
                plt.close(i)

    return fig


def create_time_scaling_plots(
    df: pl.DataFrame,
    fixed_N: int = 10,
    fixed_T: int = 1000,
    fixed_K: int = 3
) -> Dict[str, plt.Figure]:
    """Create line plots showing how computation time scales with dimensions.

    Generates three separate figures: Time vs N (fixed T, K), Time vs T (fixed N, K),
    and Time vs K (fixed N, T).

    Args:
        df: DataFrame containing timing information. Must include columns
            like 'timing_total_script_duration_s', 'N', 'K', 'T', and 'filter_config'.
        fixed_N: The fixed value of N to use when plotting Time vs T and Time vs K.
        fixed_T: The fixed value of T to use when plotting Time vs N and Time vs K.
        fixed_K: The fixed value of K to use when plotting Time vs N and Time vs T.

    Returns:
        Dict[str, plt.Figure]: Dictionary of matplotlib figures, keyed by plot type
            ("scaling_N", "scaling_T", "scaling_K").
    """
    apply_publication_style() # Apply global style settings

    figures: Dict[str, plt.Figure] = {}

    # Professional color palette for filter types (from original code)
    palette = {
        'BIF': '#0173B2',
        'PF-1000': '#DE8F05',
        'PF-5000': '#029E73',
        'PF-10000': '#CC78BC'
    }

    # --- Figure 1: Scaling vs N (Fixed T, K) ---
    filtered_df_N = df.filter(
        (pl.col("config_T") == fixed_T) &
        (pl.col("K") == fixed_K)
    )
    if filtered_df_N.height > 0:
        fig_N, ax_N = plt.subplots(figsize=(8, 6))
        sns.lineplot(
            data=filtered_df_N,
            x="N",
            y="timing_total_script_duration_s_mean",
            hue="filter_config",
            marker="o",
            palette=palette, # Use the defined palette
            ax=ax_N
        )
        ax_N.set_yscale('log')
        ax_N.set_title(f"Computation Time Scaling vs. N (Fixed T={fixed_T}, K={fixed_K})")
        ax_N.set_xlabel("Number of Particles (N)")
        ax_N.set_ylabel("Mean Computation Time (s, log scale)")
        ax_N.grid(True, linestyle='--', alpha=0.7)
        apply_publication_style() # Apply style to the specific axes
        figures["scaling_N"] = fig_N
    else:
        logging.warning(f"No data for scaling vs N plot (T={fixed_T}, K={fixed_K}). Skipping.")


    # --- Figure 2: Scaling vs T (Fixed N, K) ---
    filtered_df_T = df.filter(
        (pl.col("N") == fixed_N) &
        (pl.col("K") == fixed_K)
    )
    if filtered_df_T.height > 0:
        fig_T, ax_T = plt.subplots(figsize=(8, 6))
        sns.lineplot(
            data=filtered_df_T,
            x="config_T",
            y="timing_total_script_duration_s_mean",
            hue="filter_config",
            marker="o",
            palette=palette, # Use the defined palette
            ax=ax_T
        )
        ax_T.set_yscale('log')
        ax_T.set_title(f"Computation Time Scaling vs. T (Fixed N={fixed_N}, K={fixed_K})")
        ax_T.set_xlabel("Time Series Length (T)")
        ax_T.set_ylabel("Mean Computation Time (s, log scale)")
        ax_T.grid(True, linestyle='--', alpha=0.7)
        apply_publication_style() # Apply style to the specific axes
        figures["scaling_T"] = fig_T
    else:
         logging.warning(f"No data for scaling vs T plot (N={fixed_N}, K={fixed_K}). Skipping.")


    # --- Figure 3: Scaling vs K (Fixed N, T) ---
    filtered_df_K = df.filter(
        (pl.col("N") == fixed_N) &
        (pl.col("config_T") == fixed_T)
    )
    if filtered_df_K.height > 0:
        fig_K, ax_K = plt.subplots(figsize=(8, 6))
        sns.lineplot(
            data=filtered_df_K,
            x="K",
            y="timing_total_script_duration_s_mean",
            hue="filter_config",
            marker="o",
            palette=palette, # Use the defined palette
            ax=ax_K
        )
        ax_K.set_yscale('log')
        ax_K.set_title(f"Computation Time Scaling vs. K (Fixed N={fixed_N}, T={fixed_T})")
        ax_K.set_xlabel("Number of Factors (K)")
        ax_K.set_ylabel("Mean Computation Time (s, log scale)")
        ax_K.grid(True, linestyle='--', alpha=0.7)
        apply_publication_style() # Apply style to the specific axes
        figures["scaling_K"] = fig_K
    else:
        logging.warning(f"No data for scaling vs K plot (N={fixed_N}, T={fixed_T}). Skipping.")

    # Close any figures that might have been implicitly created by seaborn if not explicitly managed
    # (though plt.subplots should handle this, it's good practice)
    # This loop is from the original code, keeping it for safety.
    # Need to handle cases where a figure might not have been created due to no data.
    created_fig_numbers = [fig.number for key, fig in figures.items()]
    for i in plt.get_fignums():
        if i not in created_fig_numbers:
            plt.close(i)

    return figures

# ---------- helper -----------------------------------------------------------
def _prepare_error_df(
    df: pl.DataFrame,
    metric_col: str,
) -> pl.DataFrame:
    """
    Ensure we have a tidy table with *one row per (filter,N,K,T,metric)*.

    If the requested metric already exists as a scalar column, we just rename
    it to `metric`. Otherwise, we compute the median by group.
    """
    if metric_col in df.columns and not df.schema[metric_col].starts_with("List"):
        return (
            df
            .group_by(["filter_config", "N", "K", "T"])
            .median()
            .select(["filter_config", "N", "K", "T", metric_col])
            .rename({metric_col: "metric"})
        )
    raise ValueError(f"Metric column '{metric_col}' not found or not scalar.")


# ---------- main plotting routine -------------------------------------------
def create_error_scaling_plots(
    df: pl.DataFrame,
    metric_col: str = "accuracy_state_estimation_factor_rmse_mean",
    fixed_N: int = 10,
    fixed_T: int = 1000,
    fixed_K: int = 3,
) -> plt.Figure:
    """
    Creates a single figure with three subplots, showing how the chosen error metric
    scales with N, K, and T respectively, holding the other two dimensions fixed.

    Returns:
        fig: matplotlib Figure containing the 1×3 grid of line plots.
    """
    # Prepare tidy data (one row per filter_config,N,K,T,metric)
    if metric_col not in df.columns:
        raise ValueError(f"Column '{metric_col}' not found in DataFrame")
    tidy = (
        df
        .group_by(["filter_config", "N", "K", "T"])
        .median()
        .select(["filter_config", "N", "K", "T", metric_col])
        .rename({metric_col: "metric"})
    )

    # Desired line order
    order= ["BIF", "PF-1000", "PF-5000", "PF-10000"]

    # Color palette
    palette = {
        "BIF": "#0173B2",
        "PF-1000": "#DE8F05",
        "PF-5000": "#029E73",
        "PF-10000": "#CC78BC",
    }

    apply_publication_style()  # your global styling

    # Create 1×3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)

    # Helper to plot one subplot
    def _plot_sub(ax, x_col: str, filter_mask: pl.Expr, title: str):
        data = (
            tidy
            .filter(filter_mask)
            .group_by(["filter_config", x_col])
            .median()
            .sort([pl.col("filter_config"), pl.col(x_col)])
            .to_pandas()
        )

        for cfg in order:
            df_cfg = data[data["filter_config"] == cfg]
            if not df_cfg.empty:
                ax.plot(
                    df_cfg[x_col],
                    df_cfg["metric"],
                    marker="o",
                    label=cfg,
                    color=palette[cfg],
                )

        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.grid(True, linestyle="--", alpha=0.6)

    # Plot vs N (K=fixed_K, T=fixed_T)
    _plot_sub(
        axes[0],
        x_col="N",
        filter_mask=(pl.col("K") == fixed_K) & (pl.col("T") == fixed_T),
        title=f"Error vs N (K={fixed_K}, T={fixed_T})",
    )
    axes[0].set_ylabel("Median " + metric_col.split("_")[-2].upper() + " RMSE")

    # Plot vs K (N=fixed_N, T=fixed_T)
    _plot_sub(
        axes[1],
        x_col="K",
        filter_mask=(pl.col("N") == fixed_N) & (pl.col("T") == fixed_T),
        title=f"Error vs K (N={fixed_N}, T={fixed_T})",
    )
    axes[1].set_ylabel("")  # share y‑axis

    # Plot vs T (N=fixed_N, K=fixed_K)
    _plot_sub(
        axes[2],
        x_col="T",
        filter_mask=(pl.col("N") == fixed_N) & (pl.col("K") == fixed_K),
        title=f"Error vs T (N={fixed_N}, K={fixed_K})",
    )
    axes[2].set_ylabel("")

    # Single legend on the rightmost subplot
    axes[2].legend(title="Filter", loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    return fig
