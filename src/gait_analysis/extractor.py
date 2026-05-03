"""
Módulo de Extracción y Persistencia de Datos Biomecánicos (InfluxDB -> HDF5).

Este módulo implementa un motor de extracción masiva que garantiza la integridad
de las series temporales, gestiona la internacionalización de logs y centraliza
el almacenamiento en contenedores jerárquicos de alto rendimiento.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from influxdb_client import InfluxDBClient
from pydantic import BaseModel

from gait_analysis.utils.messages import get_message


class InfluxConfig(BaseModel):
    """Esquema de validación para parámetros de conexión a InfluxDB v2."""

    url: str
    token: str
    org: str
    bucket: str


class GaitDataExtractor:
    """
    Motor de extracción masiva con validación de esquema y gestión de persistencia.
    """

    def __init__(
        self,
        influx_config: dict,
        output_h5: str | Path = "data/raw/gait_study_data.h5",
        lang: str = "es",
        verbose: int = 1,
    ):
        """
        Inicializa el cliente de InfluxDB y valida el entorno de salida.
        """
        self.lang = lang
        self.verbose = verbose
        self.logger = self._setup_logging()

        # Resolución de rutas mediante Pathlib
        self.base_dir = Path(__file__).resolve().parents[2]

        # Ruta de salida HDF5 configurable
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
        """Configura el nivel de trazabilidad basado en el parámetro verbose."""
        levels = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG, 3: logging.DEBUG}
        level = levels.get(self.verbose, logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s - [%(levelname)s] - %(message)s",
            datefmt="%H:%M:%S",
        )
        return logging.getLogger(__name__)

    def _validate_extracted_data(self, df: pd.DataFrame) -> bool:
        """
        Verifica que el DataFrame contenga las señales mínimas antes de persistir.

        Required columns:
        - _time: timestamp.
        - S2: heel pressure sensor (used for heel-strike detection).
        - Foot: tag identifying which foot ('Left' or 'Right'),
          since each trial contains data from BOTH feet interleaved.
        """
        required = {"_time", "S2", "Foot"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            self.logger.error(f"Missing required columns: {missing}")
            return False
        return True

    @staticmethod
    def build_trial_label_from_start(start_dt: pd.Timestamp) -> str:
        """
        Build a stable, self-contained trial label from the test start time.

        The timestamp is normalized to UTC so that the HDF5 key does not depend
        on the original local timezone representation of the CSV.
        """
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
        start_dt = pd.to_datetime(row["d_from"])
        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize("UTC")
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
                    "id": row.get("id"),
                    "codeid": row.get("codeid"),
                    "t_code": row.get("t_code"),
                    "d_from": row.get("d_from"),
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
        """Ejecuta el pipeline de extracción masiva definido en el registro CSV."""
        csv_path = Path(csv_filepath)
        if not csv_path.is_absolute():
            csv_path = self.base_dir / csv_path

        if not csv_path.exists():
            raise FileNotFoundError(
                get_message("csv_not_found", self.lang, path=csv_path)
            )

        df_registry = pd.read_csv(csv_path)
        subset = df_registry.copy()

        if ids:
            subset = subset[subset["id"].isin(ids)]

        if codeids:
            subset = subset[subset["codeid"].isin(codeids)]

        if apply_test_filter and test_type is not None:
            subset = subset[subset["t_code"] == test_type]

        if subset.empty:
            raise ValueError(
                "No records found in registry CSV for the selected filters."
            )

        batch_label = test_type if test_type is not None else "ALL"
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
            if self._extract_patient_data(row, idx, test_type):
                saved_count += 1

        if saved_count == 0:
            raise RuntimeError(
                f"Batch extraction finished but no trials were stored in {self.h5_database}."
            )

        self.logger.info(f"Stored {saved_count} trials in {self.h5_database}")
        if missing_only and skipped_count:
            self.logger.info(f"Skipped {skipped_count} trials already complete in HDF5.")

    def _extract_patient_data(self, row: pd.Series, idx: int, test_type: str | None) -> bool:
        """Consulta Flux, normalización y almacenamiento HDF5."""
        p_id = str(row["codeid"])
        effective_test_type = str(test_type or row["t_code"])

        start_dt = pd.to_datetime(row["d_from"])
        stop_dt = pd.to_datetime(row["d_until"])

        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize("UTC")
            stop_dt = stop_dt.tz_localize("UTC")

        trial_label = self.build_trial_label_from_start(start_dt)
        start_iso = start_dt.isoformat()
        stop_iso = stop_dt.isoformat()

        query = f'''
        from(bucket: "{self._bucket}")
          |> range(start: {start_iso}, stop: {stop_iso})
          |> filter(fn: (r) => r["_measurement"] == "Gait")
          |> filter(fn: (r) => r["CodeID"] == "{p_id}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> drop(columns: ["_start", "_stop", "_measurement", "result", "table", "CodeID", "app", "mac"])
        '''

        try:
            df = self._client.query_api().query_data_frame(query)

            if isinstance(df, pd.DataFrame) and not df.empty:
                if not self._validate_extracted_data(df):
                    self.logger.error(get_message("err_schema", self.lang, id=p_id))
                    return False

                df["_time"] = df["_time"].dt.tz_localize(None)

                # Each trial contains interleaved samples from BOTH feet (one
                # sensor per foot). Persist them under separate HDF5 keys to
                # preserve per-foot timing and avoid mixing signals.
                metadata_columns = ["Foot", "DeviceName", "type", "result", "table"]
                feet_present = df["Foot"].unique().tolist()
                expected_feet = {"Left", "Right"}
                missing_feet = expected_feet - set(feet_present)
                if missing_feet:
                    self.logger.warning(
                        f"[{p_id}] {trial_label}: missing data for foot(s): {missing_feet}"
                    )

                saved_any = False
                for foot in feet_present:
                    if foot not in expected_feet:
                        self.logger.warning(
                            f"[{p_id}] {trial_label}: ignoring unexpected foot label '{foot}'"
                        )
                        continue

                    df_foot = df[df["Foot"] == foot].copy()
                    if df_foot.empty:
                        continue

                    # Drop tag columns and InfluxDB metadata that should not
                    # propagate downstream (lat, lng, Mx, My, Mz are kept).
                    cols_to_drop = [c for c in metadata_columns if c in df_foot.columns]
                    df_foot = df_foot.drop(columns=cols_to_drop)

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
                    self.logger.info(
                        get_message("save_ok", self.lang, id=p_id, h5_key=h_key)
                    )
                    saved_any = True

                return saved_any

            self.logger.warning(get_message("no_data", self.lang, id=p_id))
            return False

            self.logger.warning(get_message("no_data", self.lang, id=p_id))
            return False

        except Exception as e:
            self.logger.error(get_message("err_query", self.lang, id=p_id, e=str(e)))
            return False

    def close(self) -> None:
        """Finaliza la conexión con el servidor InfluxDB."""
        self._client.close()
