"""
Command-line interface for gait signal processing and analysis.

This module provides a CLI entry point to process previously
extracted gait data and compute biomechanical metrics.
"""

import argparse
import logging
from logging import config
from pathlib import Path

import pandas as pd
import yaml

from gait_analysis.processor import GaitDataProcessor, ProcessConfig

def resolve_path(base_dir: Path, value: str) -> Path:
    """Resolve relative paths against the project root."""
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path

def build_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for the analyze-gait CLI.

    Returns:
        argparse.ArgumentParser: Configured parser with CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process extracted gait data and compute summary metrics."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the processing configuration YAML file.",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="Verbosity level: 0=errors, 1=info, 2=details, 3=debug.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation.",
    )

    return parser


def setup_logging(verbose: int) -> None:
    """
    Configure logging based on verbosity level.

    Args:
        verbose (int): Verbosity level from CLI.
    """
    level_map = {
        0: logging.ERROR,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.DEBUG,
    }

    logging.basicConfig(
        level=level_map.get(verbose, logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """
    Entry point for the analyze-gait CLI command.

    Loads configuration, sets up logging, and initializes
    the gait processing pipeline.
    """
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    base_dir = Path(__file__).resolve().parents[3]
    project_cfg = config.get("project", {})
    paths_cfg = config.get("paths", {})
    processing_cfg = config.get("processing", {})
    batch_cfg = config.get("batch", {})

    verbose = args.verbose if args.verbose is not None else project_cfg.get("verbose", 1)
    setup_logging(verbose)

    logger = logging.getLogger(__name__)
    logger.info("Configuration loaded successfully.")

    h5_path = resolve_path(base_dir, paths_cfg["h5_path"])
    output_metrics = resolve_path(base_dir, paths_cfg["output_metrics"])
    plots_dir = resolve_path(base_dir, paths_cfg.get("plots_dir", "reports/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    process_config = ProcessConfig(
        fs=processing_cfg.get("fs", 100.0),
        cutoff_pressure=processing_cfg.get("cutoff_pressure", 5.0),
        cutoff_gyro=processing_cfg.get("cutoff_gyro", 2.0),
        gyro_threshold=processing_cfg.get("gyro_threshold", 50.0),
        min_peak_distance=processing_cfg.get("min_peak_distance", 50),
        min_peak_height=processing_cfg.get("min_peak_height", 0.2),
    )

    processor = GaitDataProcessor(process_config)
    logger.info("Gait processor initialized and ready.")

    if args.no_plots:
        logger.info("Running without plot generation.")

    logger.info("Input HDF5 path: %s", h5_path)
    logger.info("Output metrics path: %s", output_metrics)
    logger.info("analyze-gait CLI is operational.")

if __name__ == "__main__":
    main()