"""Utility functions for I/O operations in analysis scripts.

This module provides centralized I/O functionality for saving analysis outputs,
including matplotlib figures and polars DataFrames, with consistent naming and 
organization.
"""

from pathlib import Path
from datetime import datetime
import logging
from typing import Any
import matplotlib.figure
import matplotlib.pyplot as plt
import polars as pl


def create_output_directory() -> Path:
    """Create a timestamped output directory for analysis plots.

    Returns:
        Path object pointing to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/analysis_plots_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Created output directory: %s", output_dir)
    return output_dir


def save_analysis_output(output_object: Any, filename_base: str, output_dir: Path) -> None:
    """Save analysis output objects (figures or dataframes) to specified directory.

    The function handles different types of output objects:
    - matplotlib Figures: Saves as both PNG and PDF
    - polars DataFrames: Saves as CSV
    Other types will trigger a warning.

    Args:
        output_object: The object to save (matplotlib Figure or polars DataFrame)
        filename_base: Base filename without extension
        output_dir: Directory path where files should be saved

    Raises:
        TypeError: If output_dir is not a Path object
    """
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path object")
    
    if isinstance(output_object, matplotlib.figure.Figure):
        # Save as PNG with high DPI
        png_path = output_dir / f"{filename_base}.png"
        output_object.savefig(png_path, dpi=300, bbox_inches="tight")
        logging.info("Saved figure as PNG: %s", png_path)
        
        # Save as PDF for vector graphics
        pdf_path = output_dir / f"{filename_base}.pdf"
        output_object.savefig(pdf_path, bbox_inches="tight")
        logging.info("Saved figure as PDF: %s", pdf_path)
        
        # Close figure to free memory
        plt.close(output_object)
        
    elif isinstance(output_object, pl.DataFrame):
        # Save DataFrame as CSV
        csv_path = output_dir / f"{filename_base}.csv"
        output_object.write_csv(csv_path)
        logging.info("Saved DataFrame as CSV: %s", csv_path)
        
    else:
        logging.warning(
            "Unsupported output type: %s. No file was saved.", 
            type(output_object).__name__
        )