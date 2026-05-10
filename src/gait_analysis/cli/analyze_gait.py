"""
Command-line interface for gait signal processing and analysis.

This module provides a CLI entry point to process previously
extracted gait data and compute biomechanical metrics.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from gait_analysis.processor import GaitDataProcessor, ProcessConfig
from gait_analysis.utils.config_loader import load_project_config
from gait_analysis.utils.logging_config import setup_logging
from gait_analysis.utils.messages import get_message


def resolve_path(base_dir: Path, value: str) -> Path:
    """Resolve relative paths against the project root."""
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def build_keyed_output_path(base_path: Path, h5_key: str) -> Path:
    """
    Append a sanitized HDF5 key to the configured output filename.

    Example:
    - reports/data/metrics_summary.csv
    - reports/data/metrics_summary_p_RHRHUG004-1_6MWT_start_2025-11-28T12-46-09Z_Right.csv
    """
    safe_key = h5_key.strip("/").replace("/", "_")
    safe_key = re.sub(r"[^A-Za-z0-9._-]+", "_", safe_key)
    return base_path.with_name(f"{base_path.stem}_{safe_key}{base_path.suffix}")

def parse_test_type_from_h5_key(h5_key: str) -> str | None:
    """
    Extract the clinical test type from a structured HDF5 key.

    HDF5 keys have the format:
        p_<codeid>/<test_type>/start_<iso_timestamp>/<foot>

    For example, ``p_RHRHUG004-1/TUG/start_2025-11-28T12-43-14Z/Right``
    yields ``"TUG"``.

    Returns
    -------
    str | None
        The test type string (e.g. "6MWT", "TUG", "T25FW"), or ``None``
        if the key does not follow the expected structure.
    """
    parts = h5_key.strip("/").split("/")
    if len(parts) >= 2:
        return parts[1]
    return None

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
        "--lang",
        type=str,
        default=None,
        choices=["es", "en"],
        help="Optional override for message language.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the processing configuration YAML file.",
    )

    parser.add_argument(
        "--h5-key",
        type=str,
        default=None,
        help="Optional override for the HDF5 key to analyze.",
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

def validate_config(config: dict) -> None:
    """Validate required config sections and keys."""
    required_sections = ["project", "paths", "processing", "analysis"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    required_path_keys = ["h5_path", "output_metrics"]
    for key in required_path_keys:
        if key not in config["paths"]:
            raise ValueError(f"Missing required config key: paths.{key}")

    if "h5_key" not in config["analysis"]:
        raise ValueError("Missing required config key: analysis.h5_key")


def save_plot(
    df_proc: pd.DataFrame,
    peaks,
    toe_offs,
    output_plot: Path,
    h5_key: str,
) -> None:
    """
    Generate and save the gait segmentation plot.

    Heel Strikes are marked with red squares, Toe-Offs with blue triangles.
    Turning intervals (gyro magnitude above threshold) are shaded in light
    blue.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df_proc["_time"], df_proc["S2_filt"], lw=1.2, label="S2 (heel)")
    ax.plot(
        df_proc["_time"].iloc[peaks],
        df_proc["S2_filt"].iloc[peaks],
        "rs",
        label="Heel Strike",
        markersize=7,
        markeredgewidth=1.5,
    )
    ax.plot(
        df_proc["_time"].iloc[toe_offs],
        df_proc["S2_filt"].iloc[toe_offs],
        "b^",
        label="Toe-Off",
        markersize=7,
        markeredgewidth=1.5,
    )
    ax.fill_between(
        df_proc["_time"],
        0,
        df_proc["S2_filt"].max(),
        where=df_proc["is_turning"],
        alpha=0.15,
        label="Turning Interval",
    )
    ax.set_title(f"Gait Segmentation - {h5_key}")
    ax.set_ylabel("Normalized Pressure (AU)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_fatigue_plot(
    per_minute_df: pd.DataFrame,
    output_plot: Path,
    h5_key: str,
    stride_time_slope: float,
    cadence_slope: float,
) -> None:
    """
    Generate and save a per-minute fatigue analysis plot.

    Two stacked panels share the X axis (block index):
    - Top: stride time per block, with std as a shaded band, plus a linear
      regression line that visualizes the trial-wide fatigue trend.
    - Bottom: stride cadence per block, also with a linear regression line.

    The slope annotations make the fatigue trend immediately readable.
    """
    if per_minute_df.empty:
        return
    # The internal block_index is 0-based for arithmetic, but for display
    # we shift to 1-based so the X axis reads 1..6 (= minute 1 ... minute 6
    # of the 6MWT) instead of 0..5, which is more intuitive for clinicians.
    blocks = per_minute_df["block_index"].to_numpy() + 1
    stride_mean = per_minute_df["stride_time_mean_s"].to_numpy()
    
    stride_std = per_minute_df["stride_time_std_s"].to_numpy()
    cadence = per_minute_df["stride_cadence_spm"].to_numpy()

    # Linear regression lines (only over blocks that actually have stride data,
    # to avoid letting empty blocks distort the visual).
    valid_mask = ~np.isnan(stride_mean)
    if valid_mask.sum() >= 2:
        x_valid = blocks[valid_mask].astype(float)
        st_fit = np.poly1d(np.polyfit(x_valid, stride_mean[valid_mask], 1))
        cad_fit = np.poly1d(np.polyfit(x_valid, cadence[valid_mask], 1))
    else:
        st_fit = None
        cad_fit = None

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ── Top: stride time ──
    ax_top = axes[0]
    ax_top.errorbar(
        blocks,
        stride_mean,
        yerr=stride_std,
        fmt="o-",
        color="#1f77b4",
        ecolor="#1f77b4",
        elinewidth=1.0,
        capsize=4,
        label="Stride time (mean ± std)",
    )
    if st_fit is not None:
        x_fit = np.linspace(blocks.min(), blocks.max(), 100)
        ax_top.plot(
            x_fit,
            st_fit(x_fit),
            "--",
            color="black",
            alpha=0.7,
            label=f"Linear fit (slope = {stride_time_slope:+.4f} s/block)",
        )
    ax_top.set_ylabel("Stride time (s)")
    ax_top.set_title(f"Fatigue Analysis - {h5_key}")
    ax_top.legend(loc="best")
    ax_top.grid(True, linestyle=":", alpha=0.6)

    # ── Bottom: stride cadence ──
    ax_bot = axes[1]
    ax_bot.plot(
        blocks,
        cadence,
        "o-",
        color="#d62728",
        label="Stride cadence",
    )
    if cad_fit is not None:
        x_fit = np.linspace(blocks.min(), blocks.max(), 100)
        ax_bot.plot(
            x_fit,
            cad_fit(x_fit),
            "--",
            color="black",
            alpha=0.7,
            label=f"Linear fit (slope = {cadence_slope:+.3f} spm/block)",
        )
    ax_bot.set_xlabel("Minute of trial")
    ax_bot.set_ylabel("Stride cadence (spm)")
    ax_bot.set_xticks(blocks)
    ax_bot.legend(loc="best")
    ax_bot.grid(True, linestyle=":", alpha=0.6)

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """
    Entry point for the analyze-gait CLI command.

    Loads configuration, sets up logging, executes the processing
    pipeline over a selected HDF5 key, and stores output metrics
    and optional plots.
    """
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = load_project_config(args.config)
        validate_config(config)

        base_dir = Path(__file__).resolve().parents[3]

        project_cfg = config.get("project", {})
        paths_cfg = config.get("paths", {})
        processing_cfg = config.get("processing", {})
        analysis_cfg = config.get("analysis", {})

        language = args.lang if args.lang is not None else project_cfg.get("language", "es")
        verbose = args.verbose if args.verbose is not None else project_cfg.get("verbose", 1)
        setup_logging(verbose)

        logger = logging.getLogger(__name__)
        logger.info(get_message("config_loaded", language))

        h5_path = resolve_path(base_dir, paths_cfg["h5_path"])
        output_metrics_base = resolve_path(base_dir, paths_cfg["output_metrics"])
        output_plot_base = resolve_path(
            base_dir,
            paths_cfg.get("output_plot", "reports/plots/gait_segmentation.png"),
        )
        h5_key = args.h5_key if args.h5_key is not None else analysis_cfg["h5_key"]
        output_metrics = build_keyed_output_path(output_metrics_base, h5_key)
        output_plot = build_keyed_output_path(output_plot_base, h5_key)

        logger.info(f"Experiment ID (HDF5 key): {h5_key}")
        logger.info(f"Execution timestamp: {pd.Timestamp.now()}")


        process_config = ProcessConfig(
            fs=processing_cfg.get("fs", 100.0),
            cutoff_pressure=processing_cfg.get("cutoff_pressure", 5.0),
            cutoff_gyro=processing_cfg.get("cutoff_gyro", 2.0),
            gyro_threshold=processing_cfg.get("gyro_threshold", 50.0),
            min_peak_distance_s=processing_cfg.get("min_peak_distance_s", 0.5),
            min_peak_height=processing_cfg.get("min_peak_height", 0.2),
        )

        logger.info("Processing configuration:")
        logger.info(f"  fs = {process_config.fs}")
        logger.info(f"  cutoff_pressure = {process_config.cutoff_pressure}")
        logger.info(f"  cutoff_gyro = {process_config.cutoff_gyro}")
        logger.info(f"  gyro_threshold = {process_config.gyro_threshold}")
        logger.info(f"  min_peak_distance_s = {process_config.min_peak_distance_s}")
        logger.info(f"  min_peak_height = {process_config.min_peak_height}")

        processor = GaitDataProcessor(process_config)
        logger.info(get_message("processor_ready", language))

        if args.no_plots:
            logger.info(get_message("running_no_plots", language))

        logger.info(get_message("input_hdf5_path", language, path=h5_path))
        logger.info(get_message("output_metrics_path", language, path=output_metrics))

        if not h5_path.exists():
            raise FileNotFoundError(
                f"Input HDF5 file not found: {h5_path}\n"
                "Generate it first with:\n"
                "  python -m gait_analysis.cli.extract_data --config config/config.yaml"
            )

        try:
            df_raw = pd.read_hdf(h5_path, key=h5_key)
        except (KeyError, FileNotFoundError, OSError) as e:
            raise RuntimeError(
                f"Unable to read HDF5 key '{h5_key}' from file '{h5_path}'."
            ) from e

        if df_raw.empty:
            raise RuntimeError(f"HDF5 key '{h5_key}' contains no rows.")

        logger.info(f"Loaded HDF5 key: {h5_key}")
        logger.info(f"Rows loaded: {len(df_raw)}")

        test_type = parse_test_type_from_h5_key(h5_key)
        if test_type:
            logger.info(f"Detected clinical test type: {test_type}")
        clinical_tests_cfg = config.get("clinical_tests", {})
        df_proc, metrics, peaks, toe_offs, per_minute_df = processor.process_signals(
            df_raw,
            test_type=test_type,
            clinical_tests_cfg=clinical_tests_cfg,
        )
        metrics["analysis_h5_key"] = h5_key
        metrics["analysis_timestamp"] = str(pd.Timestamp.now())
        metrics["processing_fs_hz"] = process_config.fs
        metrics["processing_cutoff_pressure_hz"] = process_config.cutoff_pressure
        metrics["processing_cutoff_gyro_hz"] = process_config.cutoff_gyro
        metrics["processing_gyro_threshold"] = process_config.gyro_threshold
        metrics["processing_min_peak_distance_s"] = process_config.min_peak_distance_s
        metrics["processing_min_peak_height"] = process_config.min_peak_height
        metrics["processing_minute_block_duration_s"] = process_config.minute_block_duration_s
        metrics["processing_edge_threshold"] = process_config.edge_threshold

        output_metrics.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(output_metrics, index=False)
        logger.info(f"Metrics saved to: {output_metrics}")

        # Save per-minute fatigue metrics next to the trial-wide summary.
        # Save the gait segmentation plot (S2 with HS, TO and turning).
        if not args.no_plots:
            save_plot(df_proc, peaks, toe_offs, output_plot, h5_key)
            logger.info(f"Plot saved to: {output_plot}")

        # Save per-minute fatigue metrics next to the trial-wide summary.
        if not per_minute_df.empty:
            per_minute_path = build_keyed_output_path(
                output_metrics.parent / "metrics_per_minute.csv",
                h5_key,
            )
            per_minute_df_out = per_minute_df.copy()
            per_minute_df_out.insert(0, "analysis_h5_key", h5_key)
            per_minute_df_out.to_csv(per_minute_path, index=False)
            logger.info(f"Per-minute metrics saved to: {per_minute_path}")

            if not args.no_plots:
                fatigue_plot_path = build_keyed_output_path(
                    output_plot.parent / "fatigue_analysis.png",
                    h5_key,
                )
                save_fatigue_plot(
                    per_minute_df=per_minute_df,
                    output_plot=fatigue_plot_path,
                    h5_key=h5_key,
                    stride_time_slope=metrics.get("stride_time_minute_slope", 0.0),
                    cadence_slope=metrics.get("stride_cadence_minute_slope", 0.0),
                )
                logger.info(f"Fatigue plot saved to: {fatigue_plot_path}")

    except Exception as e:
        logging.getLogger(__name__).error(f"Analysis pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
