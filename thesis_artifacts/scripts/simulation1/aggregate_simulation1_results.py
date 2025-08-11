#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregates results from simulation 1 based on configuration parameters."""

import argparse
import ast
import logging
from pathlib import Path
from typing import List, Dict, Union, Any

import polars as pl


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_array_safely(array_str: str) -> Union[List[float], None]:
    """Safely parses a string representation of a Python list into float values."""
    try:
        result = ast.literal_eval(array_str)
        if not isinstance(result, list):
            return None
        return [float(x) for x in result]
    except (ValueError, SyntaxError, TypeError):
        return None


def create_array_parsing_expression(col_name: str) -> pl.Expr:
    """Creates a Polars expression for parsing array columns and calculating the mean."""
    return (
        pl.col(col_name)
        .map_elements(parse_array_safely, return_dtype=pl.List(pl.Float64), skip_nulls=False)
        .list.mean()
        .alias(f"{col_name}_mean")
    )


def create_generic_timing_column(
    generic_name: str,
    filter_map: Dict[str, str],
    available_cols: List[str]
) -> pl.Expr:
    """Creates a generic timing column expression based on filter type."""
    expr = pl.when(pl.col("filter_type").str.to_lowercase() == "bf")
    if filter_map["bf"] in available_cols:
        expr = expr.then(pl.col(filter_map["bf"]))
    else:
        expr = expr.then(None)

    if filter_map["pf"] in available_cols:
        expr = (expr
               .when(pl.col("filter_type").str.to_lowercase() == "pf")
               .then(pl.col(filter_map["pf"])))

    if filter_map["bif"] in available_cols:
        expr = (expr
               .when(pl.col("filter_type").str.to_lowercase() == "bif")
               .then(pl.col(filter_map["bif"])))

    return expr.otherwise(None).cast(pl.Float64).alias(generic_name)


def create_generic_array_column(
    generic_name: str,
    filter_map: Dict[str, str],
    available_cols: List[str]
) -> pl.Expr:
    """Creates a generic array metric column expression based on filter type."""
    expr = pl.when(pl.col("filter_type").str.to_lowercase() == "bf")
    # Check if the base column exists, not the derived _mean column
    if filter_map['bf'] in available_cols:
        expr = expr.then(pl.col(f"{filter_map['bf']}_mean"))
    else:
        expr = expr.then(None)

    if filter_map['pf'] in available_cols:
        expr = (expr
               .when(pl.col("filter_type").str.to_lowercase() == "pf")
               .then(pl.col(f"{filter_map['pf']}_mean")))

    if filter_map['bif'] in available_cols:
        expr = (expr
               .when(pl.col("filter_type").str.to_lowercase() == "bif")
               .then(pl.col(f"{filter_map['bif']}_mean")))

    return expr.otherwise(None).cast(pl.Float64).alias(generic_name)


def create_aggregation_expressions(cols_to_aggregate: List[str]) -> List[pl.Expr]:
    """Creates simple aggregation expressions for all columns."""
    aggs: List[pl.Expr] = []
    for col in cols_to_aggregate:
        aggs.extend([
            pl.col(col).mean().alias(f"{col}_mean"),
            pl.col(col).std().alias(f"{col}_std"),
            pl.col(col).median().alias(f"{col}_median"),
        ])
    return aggs


def aggregate_results(
    input_path: Union[str, Path], output_path: Union[str, Path]
) -> None:
    """Loads, aggregates, and saves simulation results using Polars."""
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.is_file():
        logging.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # --- Configuration based on the plan ---
    grouping_cols = ["N", "K", "T", "filter_type", "num_particles"]
    timing_cols_no_filter = ["param_gen_time", "data_sim_time"]
    array_cols = [
        "bf_rmse_f", "bf_rmse_h", "pf_rmse_f", "pf_rmse_h", "bif_rmse_f", "bif_rmse_h",
        "bf_corr_f", "bf_corr_h", "pf_corr_f", "pf_corr_h", "bif_corr_f", "bif_corr_h",
    ]

    generic_timing_map = {
        "filter_time": {
            "bf": "bf_filter_time",
            "pf": "pf_filter_time",
            "bif": "bif_filter_time",
        }
    }
    generic_array_map = {
        "rmse_f": {"bf": "bf_rmse_f", "pf": "pf_rmse_f", "bif": "bif_rmse_f"},
        "rmse_h": {"bf": "bf_rmse_h", "pf": "pf_rmse_h", "bif": "bif_rmse_h"},
        "corr_f": {"bf": "bf_corr_f", "pf": "pf_corr_f", "bif": "bif_corr_f"},
        "corr_h": {"bf": "bf_corr_h", "pf": "pf_corr_h", "bif": "bif_corr_h"},
    }
    # --- End Configuration ---

    logging.info(f"Starting aggregation for {input_file}")

    try:
        # 1. Load Data (Lazy)
        logging.info("Creating scan context...")
        lazy_df = pl.scan_csv(
            input_file,
            infer_schema_length=1000,
            try_parse_dates=False,
        )

        # Get schema to check column existence safely
        available_cols = list(lazy_df.collect_schema().keys())
        logging.info(f"Available columns: {available_cols}")

        # 2. Pre-processing (Array Columns)
        logging.info("Creating expressions for parsing array columns and calculating means...")
        array_cols_found = [col for col in array_cols if col in available_cols]
        logging.info(f"Found array columns: {array_cols_found}")

        array_exprs = [
            create_array_parsing_expression(col) for col in array_cols_found
        ]
        if array_exprs:
            lazy_df = lazy_df.with_columns(array_exprs)
            logging.info(f"Processed {len(array_exprs)} array columns")

            # For debugging, show a sample of the first few rows after processing arrays
            sample_df = lazy_df.limit(2).collect()
            for col in array_cols_found:
                mean_col = f"{col}_mean"
                if mean_col in sample_df.columns:
                    logging.info(f"Sample of {mean_col}: {sample_df[mean_col][0]}")
        else:
            logging.warning("No array columns found in the input file")

        # 3. Create generic columns before grouping
        logging.info("Creating generic columns for aggregation...")
        # 3a. Create generic timing columns
        generic_timing_exprs = [
            create_generic_timing_column(name, filter_map, available_cols)
            for name, filter_map in generic_timing_map.items()
        ]

        # 3b. Create generic array metric columns
        generic_array_exprs = [
            create_generic_array_column(name, filter_map, available_cols)
            for name, filter_map in generic_array_map.items()
        ]

        # Apply all generic column expressions
        logging.info(f"Creating {len(generic_timing_exprs)} generic timing columns and {len(generic_array_exprs)} generic array columns")

        # Log which generic array columns are being created
        for name, filter_map in generic_array_map.items():
            for filter_type, col_name in filter_map.items():
                if col_name in available_cols:
                    logging.info(f"Will create generic column {name} from {col_name} for filter type {filter_type}")
                else:
                    logging.info(f"Column {col_name} not found for filter type {filter_type}")

        lazy_df = lazy_df.with_columns(generic_timing_exprs + generic_array_exprs)

        # Check if generic columns were created
        sample_df = lazy_df.limit(2).collect()
        generic_cols_created = [col for col in sample_df.columns if col in generic_array_map.keys()]
        logging.info(f"Generic array columns created: {generic_cols_created}")

        # 4. Determine columns to aggregate
        cols_to_aggregate = (
            timing_cols_no_filter +
            list(generic_timing_map.keys()) +
            list(generic_array_map.keys())
        )

        # 5. Aggregation using generic columns
        logging.info(f"Grouping by {grouping_cols} and aggregating...")
        aggregation_exprs = create_aggregation_expressions(cols_to_aggregate)
        aggregated_lazy = lazy_df.group_by(grouping_cols).agg(aggregation_exprs)

        # 6. Collect & Save
        logging.info("Collecting results...")
        aggregated_df = aggregated_lazy.collect()

        logging.info(f"Saving aggregated results to {output_file}...")
        aggregated_df.write_csv(output_file)
        logging.info("Aggregation complete.")

    except pl.exceptions.ComputeError as e:
        logging.error(f"Polars computation error during aggregation: {e}")
        if "ColumnNotFoundError" in str(e):
            logging.error("This might be due to missing columns expected by the script.")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise


def main():
    """Main function to parse arguments and run the aggregation."""
    parser = argparse.ArgumentParser(
        description="Aggregate simulation results from a CSV file using Polars.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/simulation1/simulation1_merged_results.csv",
        help="Path to the input merged simulation results CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/simulation1/aggregated_simulation1_results.csv",
        help="Path to save the aggregated results CSV file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    args = parser.parse_args()

    # Update logging level based on argument
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    aggregate_results(args.input, args.output)


if __name__ == "__main__":
    main()