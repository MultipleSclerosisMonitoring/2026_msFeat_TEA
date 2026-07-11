"""Extraction layer for biomechanical gait data.

This module provides the bulk extractor that reads curated test windows,
queries InfluxDB for the corresponding wearable streams, validates the
minimum schema, and persists one HDF5 dataset per event and foot.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from influxdb_client import InfluxDBClient
from pydantic import BaseModel

from gait_analysis.postgresql import normalize_event_rows
from gait_analysis.utils.messages import get_message


class InfluxConfig(BaseModel):
    """Strongly typed InfluxDB connection settings."""

    url: str
    token: str
    org: str
    bucket: str


class GaitDataExtractor:
    """Bulk extractor from InfluxDB into per-foot HDF5 datasets.

    Args:
        influx_config: Mapping with the InfluxDB connection parameters.
        output_h5: Destination HDF5 path. Relative paths are resolved from
            the project root.
        lang: Message language used by the logging helpers.
        verbose: Logging verbosity level from 0 to 3.
    """

    def __init__(
        self,
        influx_config: dict,
        output_h5: str | Path = "data/raw/gait_study_data.h5",
        lang: str = "es",
        verbose: int = 1,
    ):
        self.lang = lang
        self.verbose = verbose
        self.logger = self._setup_logging()
        self.base_dir = Path(__file__).resolve().parents[2]

        output_h5_path = Path(output_h5)
        if not output_h5_path.is_absolute():
            output_h5_path = self.base_dir / output_h5_path
        output_h5_path.parent.mkdir(parents=True, exist_ok=True)
        self.h5_database = output_h5_path

        try:
            v_config = InfluxConfig(**influx_config)
        except Exception as e:
            raise ValueError(f"Invalid InfluxDB configuration: {e}") from e

        self._client = InfluxDBClient(
            url=v_config.url,
            token=v_config.token,
            org=v_config.org,
            timeout=60000,
            verify_ssl=False,
        )
        self._bucket = v_config.bucket

    def _setup_logging(self) -> logging.Logger:
        levels = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG, 3: logging.DEBUG}
        level = levels.get(self.verbose, logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s - [%(levelname)s] - %(message)s",
            datefmt="%H:%M:%S",
        )
        return logging.getLogger(__name__)

    def _validate_extracted_data(self, df: pd.DataFrame) -> bool:
        required = {"_time", "S2", "Foot"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            self.logger.error(f"Missing required columns: {missing}")
            return False
        return True

    @staticmethod
    def _coerce_query_result_to_dataframe(result: Any) -> pd.DataFrame:
        """Normalize InfluxDB query results to a single dataframe."""
        if isinstance(result, pd.DataFrame):
            return result
        if isinstance(result, list):
            frames = [item for item in result if isinstance(item, pd.DataFrame) and not item.empty]
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame()

    def _build_event_query(self, row: pd.Series) -> tuple[str, str, int, str, pd.Timestamp]:
        p_id = str(row["codeid"])
        event_id = int(row["id"])
        effective_test_type = str(row["t_code"])

        start_dt = pd.to_datetime(row["d_from"], utc=True)
        stop_dt = pd.to_datetime(row["d_until"], utc=True)
        start_iso = start_dt.isoformat()
        stop_iso = stop_dt.isoformat()

        query = f"""
        from(bucket: "{self._bucket}")
          |> range(start: {start_iso}, stop: {stop_iso})
          |> filter(fn: (r) => r["_measurement"] == "Gait")
          |> filter(fn: (r) => r["CodeID"] == "{p_id}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> drop(columns: ["_start", "_stop", "_measurement", "result", "table", "CodeID", "app", "mac"])
        """
        return query, p_id, event_id, effective_test_type, start_dt

    def extract_event_frames(self, row: pd.Series) -> dict[str, pd.DataFrame]:
        """Extract one curated event from InfluxDB and return per-foot dataframes."""
        query, p_id, event_id, _, start_dt = self._build_event_query(row)
        trial_label = self.build_trial_label_from_start(start_dt)

        try:
            result = self._client.query_api().query_data_frame(query)
            df = self._coerce_query_result_to_dataframe(result)
            if df.empty:
                self.logger.warning(get_message("no_data", self.lang, id=p_id))
                return {}

            if not self._validate_extracted_data(df):
                self.logger.error(get_message("err_schema", self.lang, id=p_id))
                return {}

            df["_time"] = pd.to_datetime(df["_time"], utc=True).dt.tz_localize(None)
            metadata_columns = ["Foot", "DeviceName", "type", "result", "table"]
            feet_present = df["Foot"].unique().tolist()
            expected_feet = {"Left", "Right"}
            missing_feet = expected_feet - set(feet_present)
            if missing_feet:
                self.logger.warning(
                    f"[{p_id}] {trial_label}: missing data for foot(s): {missing_feet}"
                )

            foot_frames: dict[str, pd.DataFrame] = {}
            for foot in feet_present:
                if foot not in expected_feet:
                    self.logger.warning(
                        f"[{p_id}] {trial_label}: ignoring unexpected foot label '{foot}'"
                    )
                    continue

                df_foot = df[df["Foot"] == foot].copy()
                if df_foot.empty:
                    continue

                cols_to_drop = [c for c in metadata_columns if c in df_foot.columns]
                df_foot = df_foot.drop(columns=cols_to_drop)
                df_foot["healthywear_event_id"] = event_id
                foot_frames[foot] = df_foot

            return foot_frames
        except Exception as e:
            self.logger.error(get_message("err_query", self.lang, id=p_id, e=str(e)))
            return {}

    @staticmethod
    def build_trial_label_from_start(start_dt: pd.Timestamp) -> str:
        start_utc = start_dt.tz_convert("UTC")
        return f"start_{start_utc.strftime('%Y-%m-%dT%H-%M-%SZ')}"

    @classmethod
    def build_h5_key(
        cls,
        codeid: str,
        test_type: str,
        start_dt: pd.Timestamp,
        foot: str,
        leading_slash: bool = False,
    ) -> str:
        prefix = "/" if leading_slash else ""
        trial_label = cls.build_trial_label_from_start(start_dt)
        return f"{prefix}p_{codeid}/{test_type}/{trial_label}/{foot}"

    @classmethod
    def build_expected_h5_keys_from_row(cls, row: pd.Series) -> dict[str, str]:
        start_dt = pd.to_datetime(row["d_from"], utc=True)
        codeid = str(row["codeid"])
        test_type = str(row["t_code"])
        return {
            foot: cls.build_h5_key(
                codeid=codeid,
                test_type=test_type,
                start_dt=start_dt,
                foot=foot,
                leading_slash=True,
            )
            for foot in ("Left", "Right")
        }

    def list_hdf5_keys(self) -> list[str]:
        if not self.h5_database.exists():
            return []
        with pd.HDFStore(self.h5_database, mode="r") as store:
            return list(store.keys())

    def audit_registry_rows(self, rows: pd.DataFrame) -> list[dict[str, Any]]:
        rows = normalize_event_rows(rows)
        existing_keys = set(self.list_hdf5_keys())
        results: list[dict[str, Any]] = []
        for _, row in rows.iterrows():
            expected_keys = self.build_expected_h5_keys_from_row(row)
            left_present = expected_keys["Left"] in existing_keys
            right_present = expected_keys["Right"] in existing_keys
            if left_present and right_present:
                status = "complete"
            elif left_present:
                status = "left_only"
            elif right_present:
                status = "right_only"
            else:
                status = "missing"

            results.append(
                {
                    "id": int(row["id"]),
                    "codeid": row["codeid"],
                    "t_code": row["t_code"],
                    "d_from": row["d_from"],
                    "left_present": left_present,
                    "right_present": right_present,
                    "status": status,
                    "left_key": expected_keys["Left"],
                    "right_key": expected_keys["Right"],
                }
            )
        return results

    def run_batch_extraction(
        self,
        csv_filepath: str | Path = "tests.csv",
        test_type: str | None = "6MWT",
        ids: list[int] | None = None,
        codeids: list[str] | None = None,
        apply_test_filter: bool = True,
        missing_only: bool = False,
    ) -> None:
        csv_path = Path(csv_filepath)
        if not csv_path.is_absolute():
            csv_path = self.base_dir / csv_path

        if not csv_path.exists():
            raise FileNotFoundError(get_message("csv_not_found", self.lang, path=csv_path))

        df_registry = normalize_event_rows(pd.read_csv(csv_path))
        subset = df_registry.copy()

        if ids:
            subset = subset[subset["id"].isin(ids)]
        if codeids:
            subset = subset[subset["codeid"].isin(codeids)]
        if apply_test_filter and test_type is not None:
            subset = subset[subset["t_code"] == test_type]

        self.run_batch_extraction_from_rows(
            rows=subset,
            batch_label=test_type if test_type is not None else "ALL",
            missing_only=missing_only,
        )

    def run_batch_extraction_from_rows(
        self,
        rows: pd.DataFrame,
        batch_label: str = "ALL",
        missing_only: bool = False,
    ) -> None:
        subset = normalize_event_rows(rows)
        if subset.empty:
            raise ValueError("No records found for the selected filters.")

        self.logger.info(get_message("start_batch", self.lang, test=batch_label))
        self.logger.info(get_message("records_found", self.lang, n=len(subset)))

        saved_count = 0
        skipped_count = 0
        existing_keys = set(self.list_hdf5_keys()) if missing_only else set()

        for idx, row in subset.iterrows():
            if missing_only:
                expected_keys = self.build_expected_h5_keys_from_row(row)
                if (
                    expected_keys["Left"] in existing_keys
                    and expected_keys["Right"] in existing_keys
                ):
                    skipped_count += 1
                    continue
            if self._extract_patient_data(row, idx):
                saved_count += 1

        if saved_count == 0:
            raise RuntimeError(
                f"Batch extraction finished but no trials were stored in {self.h5_database}."
            )

        self.logger.info(f"Stored {saved_count} trials in {self.h5_database}")
        if missing_only and skipped_count:
            self.logger.info(f"Skipped {skipped_count} trials already complete in HDF5.")

    def _extract_patient_data(self, row: pd.Series, idx: int) -> bool:
        p_id = str(row["codeid"])
        effective_test_type = str(row["t_code"])
        start_dt = pd.to_datetime(row["d_from"], utc=True)
        foot_frames = self.extract_event_frames(row)
        saved_any = False

        for foot, df_foot in foot_frames.items():
            h_key = self.build_h5_key(
                codeid=p_id,
                test_type=effective_test_type,
                start_dt=start_dt,
                foot=foot,
            )
            df_foot.to_hdf(
                self.h5_database,
                key=h_key,
                mode="a",
                format="table",
                data_columns=True,
            )
            self.logger.info(get_message("save_ok", self.lang, id=p_id, h5_key=h_key))
            saved_any = True

        return saved_any

    def close(self) -> None:
        self._client.close()
