"""Direct batch CLI: PostgreSQL -> InfluxDB -> processing -> PostgreSQL/CSV."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd

from gait_analysis import __version__ as PIPELINE_VERSION
from gait_analysis.cli.analyze_gait import save_fatigue_plot, save_plot
from gait_analysis.extractor import GaitDataExtractor
from gait_analysis.postgresql import EventSelector, HealthywearPostgresRepository, normalize_event_rows
from gait_analysis.processor import GaitDataProcessor, ProcessConfig, compute_bilateral_metrics
from gait_analysis.utils.config_loader import load_project_config
from gait_analysis.utils.logging_config import setup_logging


logger = logging.getLogger(__name__)


def parse_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.to_datetime(value, utc=True)


def resolve_event_rows(config: dict, args: argparse.Namespace) -> pd.DataFrame:
    source = args.source or ("csv" if args.csv is not None else "postgres")
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


def persist_results_if_configured(postgresql_cfg: dict, metrics_rows: list[dict]) -> None:
    if not postgresql_cfg or not metrics_rows:
        return
    repo = HealthywearPostgresRepository(postgresql_cfg)
    rowcount = repo.upsert_test_results(metrics_rows)
    logger.info("Persisted %s result rows to PostgreSQL.", rowcount)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full gait batch directly from curated events without HDF5 intermediates."
    )
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
    parser.add_argument("--lang", choices=["es", "en"], default=None)
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2, 3], default=None)
    parser.add_argument(
        "--no-postgres-persist",
        action="store_true",
        help="Skip PostgreSQL upserts even when postgresql config is present.",
    )
    return parser


def _build_analysis_key(row: pd.Series, foot: str) -> str:
    return GaitDataExtractor.build_h5_key(
        codeid=str(row["codeid"]),
        test_type=str(row["t_code"]),
        start_dt=pd.to_datetime(row["d_from"], utc=True),
        foot=foot,
    )


def _attach_processing_metadata(metrics: dict, process_config: ProcessConfig) -> None:
    metrics["processing_fs_hz"] = process_config.fs
    metrics["processing_cutoff_pressure_hz"] = process_config.cutoff_pressure
    metrics["processing_cutoff_gyro_hz"] = process_config.cutoff_gyro
    metrics["processing_gyro_threshold"] = process_config.gyro_threshold
    metrics["processing_min_peak_distance_s"] = process_config.min_peak_distance_s
    metrics["processing_min_peak_height"] = process_config.min_peak_height
    metrics["processing_minute_block_duration_s"] = process_config.minute_block_duration_s
    metrics["processing_edge_threshold"] = process_config.edge_threshold


def _save_per_minute_metrics(output_dir: Path, analysis_key: str, per_minute_df: pd.DataFrame) -> None:
    if per_minute_df.empty:
        return
    safe_key = analysis_key.replace("/", "_")
    pm_path = output_dir / f"per_minute_{safe_key}.csv"
    per_minute_out = per_minute_df.copy()
    per_minute_out.insert(0, "analysis_h5_key", analysis_key)
    per_minute_out.to_csv(pm_path, index=False)


def main() -> None:
    args = build_parser().parse_args()

    try:
        config = load_project_config(args.config)
        project_cfg = config.get("project", {})
        paths_cfg = config.get("paths", {})
        processing_cfg = config.get("processing", {})
        postgresql_cfg = config.get("postgresql", {})

        language = args.lang if args.lang is not None else project_cfg.get("language", "es")
        verbose = args.verbose if args.verbose is not None else project_cfg.get("verbose", 1)
        setup_logging(verbose)

        output_path = Path(args.output) if args.output else Path(paths_cfg.get("output_metrics", "reports/data/metrics_summary.csv"))
        if not output_path.is_absolute():
            output_path = Path(__file__).resolve().parents[3] / output_path
        output_path = output_path.with_name(f"{output_path.stem}_direct_batch{output_path.suffix}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plots_dir = output_path.parent.parent / "plots" / "direct_batch"
        if not args.no_plots:
            plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Config loaded from %s", args.config)
        logger.info("Direct batch output: %s", output_path)

        rows = resolve_event_rows(config, args)
        if rows.empty:
            raise ValueError("No records found for the selected filters.")
        logger.info("Selected %s curated events for direct processing.", len(rows))

        process_config = ProcessConfig(
            **{k: v for k, v in processing_cfg.items() if k in ProcessConfig.model_fields}
        )
        processor = GaitDataProcessor(process_config)
        extractor = GaitDataExtractor(
            influx_config=config.get("influxdb", {}),
            output_h5=paths_cfg.get("h5_path", "data/raw/gait_study_data.h5"),
            lang=language,
            verbose=verbose,
        )

        clinical_tests_cfg = config.get("clinical_tests", {})
        gps_estimation_cfg = config.get("gps_estimation", {})
        spatial_models_cfg = config.get("spatial_models", {})

        results: list[dict] = []
        errors: list[dict] = []

        try:
            for index, row in rows.iterrows():
                event_id = int(row["id"])
                logger.info("[%s/%s] Event id=%s codeid=%s test=%s", index + 1, len(rows), event_id, row["codeid"], row["t_code"])
                foot_frames = extractor.extract_event_frames(row)
                if not foot_frames:
                    errors.append({"healthywear_event_id": event_id, "error": "no extracted data"})
                    continue

                event_results: dict[str, dict] = {}
                processed_feet: dict[str, tuple[pd.DataFrame, dict, object, object, pd.DataFrame]] = {}

                for foot in ("Left", "Right"):
                    df_raw = foot_frames.get(foot)
                    if df_raw is None or df_raw.empty:
                        continue

                    analysis_key = _build_analysis_key(row, foot)
                    try:
                        df_proc, metrics, peaks, toe_offs, per_minute_df = processor.process_signals(
                            df_raw,
                            test_type=str(row["t_code"]),
                            clinical_tests_cfg=clinical_tests_cfg,
                            gps_estimation_cfg=gps_estimation_cfg,
                            spatial_models_cfg=spatial_models_cfg,
                        )
                        metrics["healthywear_event_id"] = event_id
                        metrics["analysis_h5_key"] = analysis_key
                        metrics["analysis_timestamp"] = str(pd.Timestamp.now())
                        metrics["patient_id"] = f"p_{row['codeid']}"
                        metrics["test_type"] = str(row["t_code"])
                        metrics["foot"] = foot
                        metrics["pipeline_version"] = PIPELINE_VERSION
                        _attach_processing_metadata(metrics, process_config)

                        processed_feet[foot] = (df_proc, metrics, peaks, toe_offs, per_minute_df)
                        event_results[foot] = metrics
                        _save_per_minute_metrics(output_path.parent, analysis_key, per_minute_df)

                        if not args.no_plots:
                            plot_path = plots_dir / f"{analysis_key.replace('/', '_')}.png"
                            save_plot(df_proc, peaks, toe_offs, plot_path, analysis_key)
                            if not per_minute_df.empty:
                                fatigue_path = plots_dir / f"fatigue_{analysis_key.replace('/', '_')}.png"
                                save_fatigue_plot(
                                    per_minute_df=per_minute_df,
                                    output_plot=fatigue_path,
                                    h5_key=analysis_key,
                                    stride_time_slope=metrics.get("stride_time_minute_slope", 0.0),
                                    cadence_slope=metrics.get("stride_cadence_minute_slope", 0.0),
                                )

                        logger.info(
                            "  Processed foot=%s strides=%s cadence=%.1f spm spatial=%s",
                            foot,
                            max(len(peaks) - 1, 0),
                            metrics.get("stride_cadence_spm", 0.0),
                            metrics.get("spatial_method", "none"),
                        )
                    except Exception as exc:
                        logger.error("  Failed foot=%s: %s", foot, exc)
                        logger.debug(traceback.format_exc())
                        errors.append(
                            {
                                "healthywear_event_id": event_id,
                                "analysis_h5_key": analysis_key,
                                "foot": foot,
                                "error": str(exc),
                            }
                        )

                if "Left" in processed_feet and "Right" in processed_feet:
                    df_left, metrics_left, peaks_left, toe_offs_left, _ = processed_feet["Left"]
                    df_right, metrics_right, peaks_right, toe_offs_right, _ = processed_feet["Right"]
                    bilateral = compute_bilateral_metrics(
                        metrics_left,
                        metrics_right,
                        peaks_left,
                        peaks_right,
                        toe_offs_left,
                        toe_offs_right,
                        df_left,
                        df_right,
                    )
                    metrics_left.update(bilateral)
                    metrics_right.update(bilateral)

                results.extend(event_results.values())
        finally:
            extractor.close()

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
            if postgresql_cfg and not args.no_postgres_persist:
                persist_results_if_configured(postgresql_cfg, results)
            logger.info("Direct batch complete: %s trial-feet processed -> %s", len(results), output_path)
        else:
            logger.warning("No results produced.")

        if errors:
            errors_path = output_path.with_name(output_path.stem + "_errors.csv")
            pd.DataFrame(errors).to_csv(errors_path, index=False)
            logger.warning("%s errors -> %s", len(errors), errors_path)
    except Exception as exc:
        logger.error("Direct batch pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
