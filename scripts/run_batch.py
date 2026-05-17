"""
Batch processing script for MS-Feat.

Iterates over all Left-foot HDF5 keys in the dataset, runs the full
gait analysis pipeline on each trial, and consolidates all results
into a single CSV file at reports/data/batch_metrics.csv.

Bilateral metrics are computed automatically for each trial where the
contralateral (Right) key is available.

Usage
-----
    python scripts/run_batch.py --config config/config.yaml

Optional flags
--------------
    --no-plots     Skip plot generation (faster for large datasets).
    --test-type    Filter by test type: 6MWT, TUG, T25FW (default: all).
    --output       Override output CSV path.
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gait_analysis.processor import (
    GaitDataProcessor,
    ProcessConfig,
    compute_bilateral_metrics,
)
from gait_analysis.utils.config_loader import load_project_config
from gait_analysis.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def get_contralateral_key(h5_key: str) -> str | None:
    """Return the Right key for a Left key, or None if not applicable."""
    parts = h5_key.strip("/").split("/")
    if len(parts) < 4:
        return None
    foot = parts[-1]
    if foot == "Left":
        parts[-1] = "Right"
    elif foot == "Right":
        parts[-1] = "Left"
    else:
        return None
    return "/".join(parts)


def parse_test_type(h5_key: str) -> str | None:
    parts = h5_key.strip("/").split("/")
    return parts[1] if len(parts) >= 2 else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch gait analysis over all HDF5 trials."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    parser.add_argument(
        "--test-type",
        default=None,
        choices=["6MWT", "TUG", "T25FW"],
        help="Filter by clinical test type (default: all).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output CSV path (default: reports/data/batch_metrics.csv).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = load_project_config(args.config)
    setup_logging(config.get("project", {}).get("verbose", 1))

    h5_path = PROJECT_ROOT / config["paths"]["h5_path"]
    if not h5_path.exists():
        logger.error(f"HDF5 file not found: {h5_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "reports" / "data" / "batch_metrics.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plots_dir = PROJECT_ROOT / "reports" / "plots" / "batch"
    if not args.no_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all keys, keep only Left ──────────────────────────────────────
    with pd.HDFStore(h5_path, mode="r") as store:
        all_keys = sorted(store.keys())

    left_keys = [k for k in all_keys if k.endswith("/Left")]
    if args.test_type:
        left_keys = [k for k in left_keys if f"/{args.test_type}/" in k]

    logger.info(f"Dataset: {len(all_keys)} total keys → {len(left_keys)} Left keys to process")
    if args.test_type:
        logger.info(f"Filter active: test_type={args.test_type}")

    # ── Build processor ─────────────────────────────────────────────────────
    processing_cfg = config.get("processing", {})
    process_config = ProcessConfig(**{
        k: v for k, v in processing_cfg.items()
        if k in ProcessConfig.model_fields
    })
    processor = GaitDataProcessor(process_config)

    clinical_tests_cfg = config.get("clinical_tests", {})
    gps_estimation_cfg = config.get("gps_estimation", {})
    spatial_models_cfg = config.get("spatial_models", {})

    # ── Batch loop ───────────────────────────────────────────────────────────
    results = []
    errors = []

    for i, h5_key in enumerate(left_keys, 1):
        trial_id = h5_key.strip("/")
        logger.info(f"[{i}/{len(left_keys)}] Processing: {trial_id}")

        try:
            df_raw = pd.read_hdf(h5_path, key=h5_key)
            if df_raw.empty:
                logger.warning(f"  Empty dataframe — skipping.")
                errors.append({"h5_key": trial_id, "error": "empty dataframe"})
                continue

            test_type = parse_test_type(h5_key)

            df_proc, metrics, peaks, toe_offs, per_minute_df = processor.process_signals(
                df_raw,
                test_type=test_type,
                clinical_tests_cfg=clinical_tests_cfg,
                gps_estimation_cfg=gps_estimation_cfg,
                spatial_models_cfg=spatial_models_cfg,
            )

            metrics["analysis_h5_key"] = trial_id
            metrics["patient_id"] = trial_id.split("/")[0]
            metrics["test_type"] = test_type or "unknown"
            metrics["foot"] = "Left"

            # ── Bilateral fusion ───────────────────────────────────────────
            contra_key = get_contralateral_key(h5_key)
            if contra_key:
                try:
                    df_contra = pd.read_hdf(h5_path, key=contra_key)
                    if not df_contra.empty:
                        df_contra_proc, metrics_contra, peaks_contra, toe_offs_contra, _ = (
                            processor.process_signals(
                                df_contra,
                                test_type=test_type,
                                clinical_tests_cfg=clinical_tests_cfg,
                                gps_estimation_cfg=gps_estimation_cfg,
                                spatial_models_cfg=spatial_models_cfg,
                            )
                        )
                        bil = compute_bilateral_metrics(
                            metrics, metrics_contra,
                            peaks, peaks_contra,
                            toe_offs, toe_offs_contra,
                            df_proc, df_contra_proc,
                        )
                        metrics.update(bil)
                        logger.info(
                            f"  Bilateral: asymmetry={bil['bilateral_stride_time_asymmetry_pct']:.1f}%, "
                            f"DS={bil['bilateral_double_support_mean_s']*1000:.0f} ms"
                        )
                except (KeyError, OSError):
                    logger.info(f"  Contralateral key not found — bilateral metrics skipped.")

            # ── Per-minute CSV ─────────────────────────────────────────────
            if not per_minute_df.empty:
                safe_key = trial_id.replace("/", "_")
                pm_path = output_path.parent / f"per_minute_{safe_key}.csv"
                per_minute_df.insert(0, "analysis_h5_key", trial_id)
                per_minute_df.to_csv(pm_path, index=False)

            # ── Optional plots ─────────────────────────────────────────────
            if not args.no_plots and len(peaks) > 1:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    safe_key = trial_id.replace("/", "_")
                    fig, ax = plt.subplots(figsize=(14, 5))
                    ax.plot(df_proc["_time"], df_proc["S2_filt"], lw=0.8, label="S2")
                    ax.plot(
                        df_proc["_time"].iloc[peaks],
                        df_proc["S2_filt"].iloc[peaks],
                        "rs", markersize=4, label="HS",
                    )
                    ax.plot(
                        df_proc["_time"].iloc[toe_offs],
                        df_proc["S2_filt"].iloc[toe_offs],
                        "b^", markersize=4, label="TO",
                    )
                    ax.fill_between(
                        df_proc["_time"], 0, df_proc["S2_filt"].max(),
                        where=df_proc["is_turning"], alpha=0.12, label="Turning",
                    )
                    ax.set_title(trial_id, fontsize=9)
                    ax.legend(fontsize=8)
                    ax.grid(True, linestyle=":", alpha=0.5)
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"{safe_key}.png", dpi=150)
                    plt.close(fig)
                except Exception as plot_err:
                    logger.warning(f"  Plot failed: {plot_err}")

            results.append(metrics)
            logger.info(
                f"  ✓ strides={len(peaks)-1}, "
                f"cadence={metrics.get('stride_cadence_spm', 0):.1f} spm, "
                f"spatial={metrics.get('spatial_method', 'none')}"
            )

        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            logger.debug(traceback.format_exc())
            errors.append({"h5_key": trial_id, "error": str(e)})

    # ── Consolidate results ──────────────────────────────────────────────────
    if results:
        df_out = pd.DataFrame(results)
        # Put identifying columns first
        id_cols = ["analysis_h5_key", "patient_id", "test_type", "foot"]
        other_cols = [c for c in df_out.columns if c not in id_cols]
        df_out = df_out[id_cols + other_cols]
        df_out.to_csv(output_path, index=False)
        logger.info(f"\nBatch complete: {len(results)} trials processed → {output_path}")
    else:
        logger.warning("No results produced.")

    if errors:
        errors_path = output_path.with_name(output_path.stem + "_errors.csv")
        pd.DataFrame(errors).to_csv(errors_path, index=False)
        logger.warning(f"{len(errors)} errors → {errors_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    if results:
        df_out = pd.read_csv(output_path)
        logger.info(f"\n{'='*50}")
        logger.info(f"BATCH SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total trials processed : {len(df_out)}")
        logger.info(f"By test type:")
        for tt, grp in df_out.groupby("test_type"):
            logger.info(f"  {tt}: {len(grp)} trials")
        if "spatial_method" in df_out.columns:
            logger.info(f"Spatial method distribution:")
            for method, cnt in df_out["spatial_method"].value_counts().items():
                logger.info(f"  {method}: {cnt}")
        if "stride_cadence_spm" in df_out.columns:
            logger.info(
                f"Cadence (mean ± std): "
                f"{df_out['stride_cadence_spm'].mean():.1f} ± "
                f"{df_out['stride_cadence_spm'].std():.1f} spm"
            )


if __name__ == "__main__":
    main()