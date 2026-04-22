"""
Command-line interface for batch extraction of gait sensor data.

This module provides a CLI entry point to extract time-series data
from InfluxDB and store it locally (e.g., HDF5) for further processing.
"""

import argparse
import logging
from pathlib import Path
import os
import re

import yaml

from gait_analysis.extractor import GaitDataExtractor
from gait_analysis.utils.logging_config import setup_logging
from gait_analysis.utils.messages import get_message


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
        default="config/config.yaml",
        help="Path to the project configuration YAML file.",
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional override for the CSV file containing test windows.",
    )

    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Optional override for the clinical test type (e.g. 6MWT, 2MWT).",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        choices=["es", "en"],
        help="Optional override for message language.",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="Verbosity level: 0=errors, 1=info, 2=details, 3=debug.",
    )

    return parser


def _expand_env_vars(value):
    """
    Expand environment variables of the form ${VAR_NAME} inside strings.
    """
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([^}]+)\}")

        def replacer(match):
            var_name = match.group(1)
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ValueError(
                    f"Environment variable '{var_name}' is not defined."
                )
            return env_value

        return pattern.sub(replacer, value)

    return value


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file and expand environment variables.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    return _expand_env_vars(raw_config)


def main() -> None:
    """
    Entry point for the extract-data CLI command.

    Parses input arguments, loads project configuration,
    initializes the extractor, and executes batch extraction.
    """
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    project_cfg = config.get("project", {})
    paths_cfg = config.get("paths", {})
    batch_cfg = config.get("batch", {})
    influx_cfg = config.get("influxdb", {})

    verbose = args.verbose if args.verbose is not None else project_cfg.get("verbose", 1)
    setup_logging(verbose)

    language = args.lang if args.lang is not None else project_cfg.get("language", "es")
    csv_path = args.csv if args.csv is not None else paths_cfg.get("registry_csv", "tests.csv")
    test_type = args.test if args.test is not None else batch_cfg.get("test_type", "6MWT")
    output_h5 = paths_cfg.get("h5_path", "data/raw/gait_study_data.h5")

    logger = logging.getLogger(__name__)
    logger.info(get_message("config_loaded", language))
    logger.info(get_message("registry_csv_path", language, path=csv_path))
    logger.info(get_message("selected_test_type", language, test_type=test_type))
    logger.info(get_message("output_hdf5_path", language, path=output_h5))

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(
            f"Registry CSV not found: {csv_file}. "
            "Provide a valid file with --csv or configure paths.registry_csv."
        )

    output_h5_path = Path(output_h5)
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = GaitDataExtractor(
        influx_config=influx_cfg,
        output_h5=output_h5,
        lang=language,
        verbose=verbose,
    )

    try:
        extractor.run_batch_extraction(
            csv_filepath=csv_file,
            test_type=test_type,
        )
    finally:
        extractor.close()

    if not output_h5_path.exists():
        raise RuntimeError(
            f"Extraction completed but output HDF5 was not created: {output_h5_path}"
        )

    logger.info(f"HDF5 file generated successfully: {output_h5_path}")


if __name__ == "__main__":
    main()