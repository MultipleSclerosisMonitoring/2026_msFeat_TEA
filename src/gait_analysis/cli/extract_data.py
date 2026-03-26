"""
Command-line interface for batch extraction of gait sensor data.

This module provides a CLI entry point to extract time-series data
from InfluxDB and store it locally (e.g., HDF5) for further processing.
"""

import argparse
from pathlib import Path

from gait_analysis.extractor import GaitDataExtractor


def build_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for the extract-data CLI.

    Returns:
        argparse.ArgumentParser: Configured parser with CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract gait sensor data from InfluxDB and store it locally."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config_db.yaml",
        help="Path to the InfluxDB configuration YAML file.",
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="mytests.csv",
        help="Path to the CSV file containing test windows.",
    )

    parser.add_argument(
        "--test",
        type=str,
        default="6MWT",
        help="Test type to extract (e.g. 6MWT, 2MWT).",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="es",
        choices=["es", "en"],
        help="Language for messages.",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Verbosity level: 0=errors, 1=info, 2=details, 3=debug.",
    )

    return parser


def main() -> None:
    """
    Entry point for the extract-data CLI command.

    Parses input arguments, initializes the data extractor,
    and executes the batch extraction process.
    """
    parser = build_parser()
    args = parser.parse_args()

    extractor = GaitDataExtractor(
        config_file=args.config,
        language=args.lang,
        verbose=args.verbose,
    )

    extractor.run_batch_extraction(
        csv_filepath=Path(args.csv),
        test_type=args.test,
    )


if __name__ == "__main__":
    main()