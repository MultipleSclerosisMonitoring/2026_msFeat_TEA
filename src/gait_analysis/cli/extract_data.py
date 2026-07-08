"""Command-line interface for batch extraction of gait sensor data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from gait_analysis.extractor import GaitDataExtractor
from gait_analysis.postgresql import EventSelector, HealthywearPostgresRepository, normalize_event_rows
from gait_analysis.utils.config_loader import load_project_config
from gait_analysis.utils.logging_config import setup_logging
from gait_analysis.utils.messages import get_message


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract gait sensor data from InfluxDB and store it locally."
    )
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument(
        "--source",
        type=str,
        choices=["postgres", "csv"],
        default=None,
        help="Registry source. Defaults to PostgreSQL unless --csv is provided.",
    )
    parser.add_argument("--test", type=str, nargs="+", default=None)
    parser.add_argument("--ids", type=int, nargs="+", default=None)
    parser.add_argument("--codeid", type=str, nargs="+", default=None)
    parser.add_argument("--from-date", type=str, default=None)
    parser.add_argument("--to-date", type=str, default=None)
    parser.add_argument("--missing-only", action="store_true")
    parser.add_argument("--list-hdf5-keys", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--lang", type=str, default=None, choices=["es", "en"])
    parser.add_argument("--verbose", type=int, default=None, choices=[0, 1, 2, 3])
    return parser


def _parse_optional_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.to_datetime(value, utc=True)


def _resolve_registry_subset_from_csv(
    csv_file: Path,
    ids: list[int] | None,
    codeids: list[str] | None,
    test_types: list[str] | None,
    date_from: pd.Timestamp | None,
    date_to: pd.Timestamp | None,
) -> pd.DataFrame:
    subset = normalize_event_rows(pd.read_csv(csv_file))
    if ids:
        subset = subset[subset["id"].isin(ids)]
    if codeids:
        subset = subset[subset["codeid"].isin(codeids)]
    if test_types:
        subset = subset[subset["t_code"].isin(test_types)]
    if date_from is not None:
        subset = subset[subset["d_until"] >= date_from]
    if date_to is not None:
        subset = subset[subset["d_from"] <= date_to]
    return subset.sort_values(["d_from", "id"]).reset_index(drop=True)


def _resolve_registry_subset_from_postgres(
    postgres_cfg: dict,
    ids: list[int] | None,
    codeids: list[str] | None,
    test_types: list[str] | None,
    date_from: pd.Timestamp | None,
    date_to: pd.Timestamp | None,
) -> pd.DataFrame:
    repo = HealthywearPostgresRepository(postgres_cfg)
    selector = EventSelector(
        ids=ids,
        codeids=codeids,
        test_types=test_types,
        date_from=date_from.to_pydatetime() if date_from is not None else None,
        date_to=date_to.to_pydatetime() if date_to is not None else None,
    )
    return repo.fetch_events(selector)


def _print_audit_results(results: list[dict]) -> None:
    if not results:
        print("No matching rows found in the selected registry source.")
        return
    for item in results:
        print(
            f"id={item['id']} codeid={item['codeid']} test={item['t_code']} "
            f"status={item['status']} left={item['left_present']} right={item['right_present']}"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_project_config(args.config)
    project_cfg = config.get("project", {})
    paths_cfg = config.get("paths", {})
    batch_cfg = config.get("batch", {})
    influx_cfg = config.get("influxdb", {})
    postgresql_cfg = config.get("postgresql", {})

    verbose = args.verbose if args.verbose is not None else project_cfg.get("verbose", 1)
    setup_logging(verbose)

    language = args.lang if args.lang is not None else project_cfg.get("language", "es")
    logger = logging.getLogger(__name__)

    source = args.source or ("csv" if args.csv is not None else "postgres")
    csv_path = args.csv if args.csv is not None else paths_cfg.get("registry_csv", "tests.csv")
    default_test_type = batch_cfg.get("test_type")
    test_types = args.test if args.test is not None else ([default_test_type] if default_test_type else None)
    date_from = _parse_optional_timestamp(args.from_date)
    date_to = _parse_optional_timestamp(args.to_date)
    output_h5 = paths_cfg.get("h5_path", "data/raw/gait_study_data.h5")

    logger.info(get_message("config_loaded", language))
    logger.info(f"Registry source: {source}")
    if source == "csv":
        logger.info(get_message("registry_csv_path", language, path=csv_path))
    if test_types:
        logger.info(
            get_message("selected_test_type", language, test_type=", ".join(test_types))
        )
    else:
        logger.info(get_message("selected_test_type", language, test_type="ALL"))
    logger.info(get_message("output_hdf5_path", language, path=output_h5))

    output_h5_path = Path(output_h5)
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = GaitDataExtractor(
        influx_config=influx_cfg,
        output_h5=output_h5,
        lang=language,
        verbose=verbose,
    )

    try:
        if args.list_hdf5_keys:
            keys = extractor.list_hdf5_keys()
            if not keys:
                print("No HDF5 keys found.")
            else:
                for key in keys:
                    print(key)
            return

        if source == "csv":
            csv_file = Path(csv_path)
            if not csv_file.exists():
                raise FileNotFoundError(
                    f"Registry CSV not found: {csv_file}. Provide a valid file with --csv or configure paths.registry_csv."
                )
            subset = _resolve_registry_subset_from_csv(
                csv_file=csv_file,
                ids=args.ids,
                codeids=args.codeid,
                test_types=test_types,
                date_from=date_from,
                date_to=date_to,
            )
        else:
            subset = _resolve_registry_subset_from_postgres(
                postgres_cfg=postgresql_cfg,
                ids=args.ids,
                codeids=args.codeid,
                test_types=test_types,
                date_from=date_from,
                date_to=date_to,
            )

        if args.check_only:
            _print_audit_results(extractor.audit_registry_rows(subset))
            return

        batch_label = ",".join(test_types) if test_types else "ALL"
        extractor.run_batch_extraction_from_rows(
            rows=subset,
            batch_label=batch_label,
            missing_only=args.missing_only,
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
