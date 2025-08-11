"""Configuration module for consistent plotting styles across simulation study 2 analysis.

This module provides centralized configuration for matplotlib and seaborn plotting settings
to ensure consistent visualization styles across all analysis scripts in simulation study 2.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def apply_publication_style() -> None:
    """Apply consistent publication-ready plotting style settings.

    This function configures matplotlib and seaborn settings to create publication-quality
    plots with consistent styling across all figures. It sets:
        - A clean, professional seaborn style with white grid
        - Serif font family with standardized sizes for different plot elements
        - High DPI for crisp rendering
        - Standardized line properties
        - A consistent color palette (blue, orange, green)

    Returns:
        None. The function modifies the global matplotlib and seaborn settings.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
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
        'lines.markersize': 6,
        'lines.linewidth': 1.5,
    })
    
    # Define consistent color palette (Blue, Orange, Green)
    colors = ['#0173B2', '#DE8F05', '#029E73']
    sns.set_palette(colors)