"""PostgreSQL access layer for event selection and metric persistence.

This module isolates the SQL-facing logic used by the pipeline so the rest of
application code can work with typed selectors and metric dictionaries instead
of raw SQL strings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

EVENT_COLUMNS = ["id", "codeid", "t_code", "d_from", "d_until"]
RESULT_COLUMN_MAP = {
    "healthywear_event_id": "healthywear_event_id",
    "analysis_h5_key": "analysis_h5_key",
    "analysis_timestamp": "analysis_timestamp",
    "patient_id": "patient_id",
    "test_type": "test_type",
    "foot": "foot",
    "pipeline_version": "pipeline_version",
    "posicion_gps": "posicion_gps",
    "walking_duration_s": "walking_duration_s",
    "stride_time_mean_s": "stride_time_mean_s",
    "stride_time_std_s": "stride_time_std_s",
    "stride_time_cv": "stride_time_cv",
    "stride_time_slope": "stride_time_slope",
    "stride_cadence_spm": "stride_cadence_spm",
    "stride_cadence_first_half_spm": "stride_cadence_first_half_spm",
    "stride_cadence_second_half_spm": "stride_cadence_second_half_spm",
    "stride_cadence_change_spm": "stride_cadence_change_spm",
    "stance_time_mean_s": "stance_time_mean_s",
    "stance_time_std_s": "stance_time_std_s",
    "stance_time_cv": "stance_time_cv",
    "swing_time_mean_s": "swing_time_mean_s",
    "swing_time_std_s": "swing_time_std_s",
    "swing_time_cv": "swing_time_cv",
    "stance_swing_ratio": "stance_swing_ratio",
    "n_minute_blocks": "n_minute_blocks",
    "stride_time_minute_slope": "stride_time_minute_slope",
    "stride_cadence_minute_slope": "stride_cadence_minute_slope",
    "stance_time_minute_slope": "stance_time_minute_slope",
    "swing_time_minute_slope": "swing_time_minute_slope",
    "spatial_method": "spatial_method",
    "spatial_distance_m": "spatial_distance_m",
    "walking_speed_mean_m_s": "walking_speed_mean_m_s",
    "stride_length_mean_m": "stride_length_mean_m",
    "gps_n_unique_points": "gps_n_unique_points",
    "gps_span_m": "gps_span_m",
    "gps_total_path_m": "gps_total_path_m",
    "gyro_norm_stride_length_m": "gyro_norm_stride_length_m",
    "gyro_norm_walking_speed_m_s": "gyro_norm_walking_speed_m_s",
    "biometric_stride_length_m": "biometric_stride_length_m",
    "biometric_walking_speed_m_s": "biometric_walking_speed_m_s",
    "bilateral_stride_time_asymmetry_pct": "bilateral_stride_time_asymmetry_pct",
    "bilateral_cadence_asymmetry_pct": "bilateral_cadence_asymmetry_pct",
    "bilateral_stance_time_asymmetry_pct": "bilateral_stance_time_asymmetry_pct",
    "bilateral_step_time_LR_mean_s": "bilateral_step_time_lr_mean_s",
    "bilateral_step_time_RL_mean_s": "bilateral_step_time_rl_mean_s",
    "bilateral_step_time_asymmetry_pct": "bilateral_step_time_asymmetry_pct",
    "bilateral_double_support_mean_s": "bilateral_double_support_mean_s",
    "bilateral_double_support_pct": "bilateral_double_support_pct",
    "bilateral_available": "bilateral_available",
    "processing_fs_hz": "processing_fs_hz",
    "processing_cutoff_pressure_hz": "processing_cutoff_pressure_hz",
    "processing_cutoff_gyro_hz": "processing_cutoff_gyro_hz",
    "processing_gyro_threshold": "processing_gyro_threshold",
    "processing_min_peak_distance_s": "processing_min_peak_distance_s",
    "processing_min_peak_height": "processing_min_peak_height",
    "processing_minute_block_duration_s": "processing_minute_block_duration_s",
    "processing_edge_threshold": "processing_edge_threshold",
}
RESULT_IDENTITY_COLUMNS = [
    "healthywear_event_id",
    "analysis_h5_key",
    "analysis_timestamp",
    "patient_id",
    "test_type",
    "foot",
    "pipeline_version",
]
RESULT_METRIC_COLUMNS = [
    "posicion_gps",
    "walking_duration_s",
    "stride_time_mean_s",
    "stride_time_std_s",
    "stride_time_cv",
    "stride_time_slope",
    "stride_cadence_spm",
    "stride_cadence_first_half_spm",
    "stride_cadence_second_half_spm",
    "stride_cadence_change_spm",
    "stance_time_mean_s",
    "stance_time_std_s",
    "stance_time_cv",
    "swing_time_mean_s",
    "swing_time_std_s",
    "swing_time_cv",
    "stance_swing_ratio",
    "n_minute_blocks",
    "stride_time_minute_slope",
    "stride_cadence_minute_slope",
    "stance_time_minute_slope",
    "swing_time_minute_slope",
    "spatial_method",
    "spatial_distance_m",
    "walking_speed_mean_m_s",
    "stride_length_mean_m",
    "gps_n_unique_points",
    "gps_span_m",
    "gps_total_path_m",
    "gyro_norm_stride_length_m",
    "gyro_norm_walking_speed_m_s",
    "biometric_stride_length_m",
    "biometric_walking_speed_m_s",
    "bilateral_stride_time_asymmetry_pct",
    "bilateral_cadence_asymmetry_pct",
    "bilateral_stance_time_asymmetry_pct",
    "bilateral_step_time_LR_mean_s",
    "bilateral_step_time_RL_mean_s",
    "bilateral_step_time_asymmetry_pct",
    "bilateral_double_support_mean_s",
    "bilateral_double_support_pct",
    "bilateral_available",
    "processing_fs_hz",
    "processing_cutoff_pressure_hz",
    "processing_cutoff_gyro_hz",
    "processing_gyro_threshold",
    "processing_min_peak_distance_s",
    "processing_min_peak_height",
    "processing_minute_block_duration_s",
    "processing_edge_threshold",
]
RESULT_COLUMNS = RESULT_IDENTITY_COLUMNS + RESULT_METRIC_COLUMNS

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PostgresConfig(BaseModel):
    """Strongly typed PostgreSQL configuration."""

    model_config = ConfigDict(populate_by_name=True)

    host: str
    user: str
    password: str
    database: str
    port: int = Field(default=5432, ge=1)
    db_schema: str = Field(default="public", alias="schema")
    event_table: str = "healthywear_event"
    results_table: str = "healthywear_test_results"


@dataclass(slots=True)
class EventSelector:
    """Selection criteria used to query ``healthywear_event``."""

    ids: list[int] | None = None
    codeids: list[str] | None = None
    test_types: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None


def _validate_identifier(name: str, label: str) -> str:
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid {label}: {name!r}")
    return name


def normalize_event_rows(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in EVENT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required event columns: {missing}")

    out = df.copy()
    out["id"] = out["id"].astype(int)
    out["codeid"] = out["codeid"].astype(str)
    out["t_code"] = out["t_code"].astype(str)
    out["d_from"] = pd.to_datetime(out["d_from"], utc=True)
    out["d_until"] = pd.to_datetime(out["d_until"], utc=True)
    return out[EVENT_COLUMNS]


def build_event_query(selector: EventSelector, table_ref: str) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    if selector.ids:
        clauses.append("id = ANY(%s)")
        params.append(list(selector.ids))
    if selector.codeids:
        clauses.append("codeid = ANY(%s)")
        params.append(list(selector.codeids))
    if selector.test_types:
        clauses.append("t_code = ANY(%s)")
        params.append(list(selector.test_types))
    if selector.date_from is not None:
        clauses.append("d_until >= %s")
        params.append(selector.date_from)
    if selector.date_to is not None:
        clauses.append("d_from <= %s")
        params.append(selector.date_to)

    where_sql = ""
    if clauses:
        where_sql = " WHERE " + " AND ".join(clauses)

    query = (
        f"SELECT id, codeid, t_code, d_from, d_until "
        f"FROM {table_ref}{where_sql} "
        f"ORDER BY d_from ASC, id ASC"
    )
    return query, params


def extract_result_payload(metrics: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for metric_key in RESULT_COLUMNS:
        if metric_key not in metrics:
            continue
        column = RESULT_COLUMN_MAP[metric_key]
        value = metrics[metric_key]
        if isinstance(value, (dict, list)):
            payload[column] = json.dumps(value)
        elif value is None:
            payload[column] = None
        elif not isinstance(value, (str, bytes, bool)) and pd.isna(value):
            payload[column] = None
        else:
            payload[column] = value

    required = ["healthywear_event_id", "analysis_h5_key", "foot"]
    missing = [col for col in required if payload.get(col) in (None, "")]
    if missing:
        raise ValueError(f"Cannot persist result without required fields: {missing}")

    return payload


class HealthywearPostgresRepository:
    """Repository for curated event selection and result upsert operations."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = PostgresConfig(**config)
        schema = _validate_identifier(self.config.db_schema, "schema")
        event_table = _validate_identifier(self.config.event_table, "event_table")
        results_table = _validate_identifier(self.config.results_table, "results_table")
        self._event_ref = f'"{schema}"."{event_table}"'
        self._results_ref = f'"{schema}"."{results_table}"'

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "psycopg is required for PostgreSQL access. Install project dependencies first."
            ) from exc

        return psycopg.connect(
            host=self.config.host,
            user=self.config.user,
            password=self.config.password,
            dbname=self.config.database,
            port=self.config.port,
        )

    def fetch_events(self, selector: EventSelector) -> pd.DataFrame:
        query, params = build_event_query(selector, self._event_ref)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description]
        return normalize_event_rows(pd.DataFrame(rows, columns=columns))

    def upsert_test_result(self, metrics: Mapping[str, Any]) -> None:
        self.upsert_test_results([metrics])

    def upsert_test_results(self, metrics_list: Sequence[Mapping[str, Any]]) -> int:
        payloads = [extract_result_payload(metrics) for metrics in metrics_list]
        if not payloads:
            return 0

        columns = list(payloads[0].keys())
        values = [[payload.get(column) for column in columns] for payload in payloads]
        placeholders = ", ".join(["%s"] * len(columns))
        assignments = ", ".join(
            f'"{column}" = EXCLUDED."{column}"'
            for column in columns
            if column not in {"healthywear_event_id", "foot"}
        )
        column_sql = ", ".join(f'"{column}"' for column in columns)
        query = (
            f"INSERT INTO {self._results_ref} ({column_sql}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT (healthywear_event_id, foot) DO UPDATE SET {assignments}"
        )

        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(query, values)
            rowcount = cur.rowcount
            conn.commit()
        return rowcount
