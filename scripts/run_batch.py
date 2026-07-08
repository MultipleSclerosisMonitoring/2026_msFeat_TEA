"""Batch processing script for MS-Feat."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gait_analysis import __version__ as PIPELINE_VERSION
from gait_analysis.extractor import GaitDataExtractor
from gait_analysis.postgresql import EventSelector, HealthywearPostgresRepository, normalize_event_rows
from gait_analysis.processor import GaitDataProcessor, ProcessConfig, compute_bilateral_metrics
from gait_analysis.utils.config_loader import load_project_config
from gait_analysis.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def get_contralateral_key(h5_key: str) -> str | None:
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


def parse_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.to_datetime(value, utc=True)


def resolve_event_rows(config: dict, args: argparse.Namespace) -> pd.DataFrame | None:
    source = args.source or ("csv" if args.csv is not None else None)
    if source is None and not any([args.ids, args.codeid, args.test_type, args.from_date, args.to_date]):
        return None

    test_types = args.test_type if args.test_type else None
    date_from = parse_timestamp(args.from_date)
    date_to = parse_timestamp(args.to_date)

    if source == "csv":
        csv_path = Path(args.csv or config["paths"].get("registry_csv", "data/tests.csv"))
        rows = normalize_event_rows(pd.read_csv(csv_path))
        if args.ids:
            rows = rows[rows["id"].isin(args.ids)]
        if args.codeid:
            rows = rows[rows["codeid"].isin(args.codeid)]
        if test_types:
            rows = rows[rows["t_code"].isin(test_types)]
        if date_from is not None:
            rows = rows[rows["d_until"] >= date_from]
        if date_to is not None:
            rows = rows[rows["d_from"] <= date_to]
        return rows.sort_values(["d_from", "id"]).reset_index(drop=True)

    repo = HealthywearPostgresRepository(config.get("postgresql", {}))
    selector = EventSelector(
        ids=args.ids,
        codeids=args.codeid,
        test_types=test_types,
        date_from=date_from.to_pydatetime() if date_from is not None else None,
        date_to=date_to.to_pydatetime() if date_to is not None else None,
    )
    return repo.fetch_events(selector)


def resolve_h5_keys(h5_path: Path, event_rows: pd.DataFrame | None, test_types: list[str] | None) -> list[str]:
    with pd.HDFStore(h5_path, mode="r") as store:
        all_keys = sorted(store.keys())

    if event_rows is None:
        keys = [k for k in all_keys if k.endswith("/Left") or k.endswith("/Right")]
        if test_types:
            keys = [k for k in keys if any(f"/{tt}/" in k for tt in test_types)]
        return keys

    keys: list[str] = []
    for _, row in event_rows.iterrows():
        expected = GaitDataExtractor.build_expected_h5_keys_from_row(row)
        keys.extend([expected["Left"], expected["Right"]])

    available = set(all_keys)
    deduped = []
    seen = set()
    for key in keys:
        if key in available and key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def extract_event_id(df_raw: pd.DataFrame) -> int | None:
    if "healthywear_event_id" not in df_raw.columns or df_raw.empty:
        return None
    series = pd.to_numeric(df_raw["healthywear_event_id"], errors="coerce").dropna()
    if series.empty:
        return None
    return int(series.iloc[0])


def persist_results_if_configured(postgresql_cfg: dict, metrics_rows: list[dict]) -> None:
    if not postgresql_cfg or not metrics_rows:
        return
    repo = HealthywearPostgresRepository(postgresql_cfg)
    repo.upsert_test_results(metrics_rows)
    logger.info("Persisted %s result rows to PostgreSQL.", len(metrics_rows))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch gait analysis over HDF5 trials.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--test-type", nargs="+", default=None)
    parser.add_argument("--ids", type=int, nargs="+", default=None)
    parser.add_argument("--codeid", type=str, nargs="+", default=None)
    parser.add_argument("--from-date", type=str, default=None)
    parser.add_argument("--to-date", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--source", choices=["postgres", "csv"], default=None)
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = load_project_config(args.config)
    setup_logging(config.get("project", {}).get("verbose", 1))

    h5_path = PROJECT_ROOT / config["paths"]["h5_path"]
    if not h5_path.exists():
        logger.error(f"HDF5 file not found: {h5_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else (PROJECT_ROOT / "reports" / "data" / "batch_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plots_dir = PROJECT_ROOT / "reports" / "plots" / "batch"
    if not args.no_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    event_rows = resolve_event_rows(config, args)
    test_types = args.test_type if args.test_type else None
    trial_keys = resolve_h5_keys(h5_path, event_rows, test_types)

    logger.info(f"Dataset selection: {len(trial_keys)} HDF5 keys to process")

    processing_cfg = config.get("processing", {})
    process_config = ProcessConfig(**{k: v for k, v in processing_cfg.items() if k in ProcessConfig.model_fields})
    processor = GaitDataProcessor(process_config)

    clinical_tests_cfg = config.get("clinical_tests", {})
    gps_estimation_cfg = config.get("gps_estimation", {})
    spatial_models_cfg = config.get("spatial_models", {})
    postgresql_cfg = config.get("postgresql", {})

    results = []
    errors = []

    for i, h5_key in enumerate(trial_keys, 1):
        trial_id = h5_key.strip("/")
        logger.info(f"[{i}/{len(trial_keys)}] Processing: {trial_id}")

        try:
            df_raw = pd.read_hdf(h5_path, key=h5_key)
            if df_raw.empty:
                logger.warning("  Empty dataframe — skipping.")
                errors.append({"h5_key": trial_id, "error": "empty dataframe"})
                continue

            test_type = parse_test_type(h5_key)
            foot = trial_id.split("/")[-1]
            df_proc, metrics, peaks, toe_offs, per_minute_df = processor.process_signals(
                df_raw,
                test_type=test_type,
                clinical_tests_cfg=clinical_tests_cfg,
                gps_estimation_cfg=gps_estimation_cfg,
                spatial_models_cfg=spatial_models_cfg,
            )

            metrics["healthywear_event_id"] = extract_event_id(df_raw)
            metrics["analysis_h5_key"] = trial_id
            metrics["analysis_timestamp"] = str(pd.Timestamp.now())
            metrics["patient_id"] = trial_id.split("/")[0]
            metrics["test_type"] = test_type or "unknown"
            metrics["foot"] = foot
            metrics["pipeline_version"] = PIPELINE_VERSION

            contra_key = get_contralateral_key(h5_key)
            if contra_key:
                try:
                    df_contra = pd.read_hdf(h5_path, key=contra_key)
                    if not df_contra.empty:
                        df_contra_proc, metrics_contra, peaks_contra, toe_offs_contra, _ = processor.process_signals(
                            df_contra,
                            test_type=test_type,
                            clinical_tests_cfg=clinical_tests_cfg,
                            gps_estimation_cfg=gps_estimation_cfg,
                            spatial_models_cfg=spatial_models_cfg,
                        )
                        if foot == "Left":
                            bil = compute_bilateral_metrics(
                                metrics,
                                metrics_contra,
                                peaks,
                                peaks_contra,
                                toe_offs,
                                toe_offs_contra,
                                df_proc,
                                df_contra_proc,
                            )
                        else:
                            bil = compute_bilateral_metrics(
                                metrics_contra,
                                metrics,
                                peaks_contra,
                                peaks,
                                toe_offs_contra,
                                toe_offs,
                                df_contra_proc,
                                df_proc,
                            )
                        metrics.update(bil)
                except (KeyError, OSError):
                    logger.info("  Contralateral key not found — bilateral metrics skipped.")

            metrics["processing_fs_hz"] = process_config.fs
            metrics["processing_cutoff_pressure_hz"] = process_config.cutoff_pressure
            metrics["processing_cutoff_gyro_hz"] = process_config.cutoff_gyro
            metrics["processing_gyro_threshold"] = process_config.gyro_threshold
            metrics["processing_min_peak_distance_s"] = process_config.min_peak_distance_s
            metrics["processing_min_peak_height"] = process_config.min_peak_height
            metrics["processing_minute_block_duration_s"] = process_config.minute_block_duration_s
            metrics["processing_edge_threshold"] = process_config.edge_threshold

            if not per_minute_df.empty:
                safe_key = trial_id.replace("/", "_")
                pm_path = output_path.parent / f"per_minute_{safe_key}.csv"
                per_minute_df.insert(0, "analysis_h5_key", trial_id)
                per_minute_df.to_csv(pm_path, index=False)

            if not args.no_plots and len(peaks) > 1:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    safe_key = trial_id.replace("/", "_")
                    fig, ax = plt.subplots(figsize=(14, 5))
                    ax.plot(df_proc["_time"], df_proc["S2_filt"], lw=0.8, label="S2")
                    ax.plot(df_proc["_time"].iloc[peaks], df_proc["S2_filt"].iloc[peaks], "rs", markersize=4, label="HS")
                    ax.plot(df_proc["_time"].iloc[toe_offs], df_proc["S2_filt"].iloc[toe_offs], "b^", markersize=4, label="TO")
                    ax.fill_between(df_proc["_time"], 0, df_proc["S2_filt"].max(), where=df_proc["is_turning"], alpha=0.12, label="Turning")
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
                f"  ✓ foot={foot}, strides={len(peaks)-1}, cadence={metrics.get('stride_cadence_spm', 0):.1f} spm, spatial={metrics.get('spatial_method', 'none')}"
            )

        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            logger.debug(traceback.format_exc())
            errors.append({"h5_key": trial_id, "error": str(e)})

    if results:
        df_out = pd.DataFrame(results)
        id_cols = [
            "healthywear_event_id",
            "analysis_h5_key",
            "analysis_timestamp",
            "patient_id",
            "test_type",
            "foot",
            "pipeline_version",
        ]
        other_cols = [c for c in df_out.columns if c not in id_cols]
        df_out = df_out[id_cols + other_cols]
        df_out.to_csv(output_path, index=False)
        persist_results_if_configured(postgresql_cfg, results)
        logger.info(f"\nBatch complete: {len(results)} trials processed → {output_path}")
    else:
        logger.warning("No results produced.")

    if errors:
        errors_path = output_path.with_name(output_path.stem + "_errors.csv")
        pd.DataFrame(errors).to_csv(errors_path, index=False)
        logger.warning(f"{len(errors)} errors → {errors_path}")


if __name__ == "__main__":
    main()
